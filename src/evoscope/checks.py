"""Numerical checks of the theory's conditions for operators and assemblies.

These checks are meant for the stage at which an operator is still an idea: they run in
seconds on a handful of populations and say whether the operator behaves as the theory
requires, before anyone proves anything about it.

Operator level (the composition assumption of the paper, conditions A1-A4):

* :func:`check_first_order`  (A1) the change per unit step settles as the step shrinks;
* :func:`check_moment_stability`  (A2) one step does not blow populations up;
* :func:`check_near_identity`  (A3) a small step changes the population only a little;
* :func:`check_continuity`  (A4) the first-order effect changes continuously with the population;
* :func:`check_operator`  all four, as one report.

Assembly level:

* :func:`check_composition`  the composed step's change equals the sum of the operators' effects (composition theorem);
* :func:`check_cutoff`  the effect on an unbounded error measure is well defined (cutoff-admissible tests);
* :func:`check_assembly`  operator checks, composition, cutoff, and the drift coefficient with its attribution.

Every check works with operators given only as update rules: all quantities are
estimated from finite steps.  Random rules are averaged over replicates with common
random numbers; the reports state the noise level.
"""
import numpy as np

from . import attribution
from .metrics import bl_lower, w2
from .operators import Assembly, GaussianNoise
from .population import Population
from .report import Finding, Report
from .testfunctions import battery, truncated

__all__ = ["check_first_order", "check_moment_stability", "check_near_identity", "check_continuity",
           "check_operator", "check_composition", "check_cutoff", "check_assembly", "stress_populations"]


# ----------------------------------------------------------------------------- helpers
def _exact(rule, phi):
    if isinstance(rule, Assembly):
        return rule.exact or rule.exact_for(phi)
    if isinstance(rule, GaussianNoise):
        return hasattr(phi, "gaussian_mean")
    return rule.exact


COPIES = 8            # copies of every individual per random replicate (variance reduction)
RANDOM_TAUS = (.2, .1, .05, .025)


def _after(rule, pop, tau, phi, rng, copies=1):
    if isinstance(rule, Assembly) and rule.exact_for(phi) and not rule.exact:
        return rule.finite_step_mean(pop, tau, phi)
    if isinstance(rule, GaussianNoise) and hasattr(phi, "gaussian_mean"):
        return pop.expect(phi.gaussian_mean(pop.points, tau*rule.sigma*rule.sigma))
    law = rule.step(pop, tau, rng, copies)
    return law.expect(phi.value(law.points))


def _quotients(rule, pop, phi, taus, replicates, seed):
    """Per-replicate quotients (K, T), with common random numbers across taus and
    COPIES copies of every individual per replicate for random rules."""
    base = pop.expect(phi.value(pop.points))
    if _exact(rule, phi):
        return np.array([[(_after(rule, pop, t, phi, None)-base)/t for t in taus]])
    return np.array([[(_after(rule, pop, t, phi, np.random.default_rng([seed, k]), COPIES)-base)/t for t in taus]
                     for k in range(replicates)])


def _settling(taus, changes, noise, scale):
    """Verdict for a sequence that should go to zero as tau decreases (changes between
    successive steps, or remainders), given its noise level (zero for exact rules).
    Returns (verdict, statistic, slope)."""
    resolved = changes > noise
    if np.all(changes <= 1e-9*scale):
        return "consistent", "no remainder: the change per unit step does not depend on tau", np.nan
    if not resolved.any():
        verdict = "consistent" if noise.max() < scale else "not resolved"
        return verdict, f"stable within the noise level {noise.max():.2g} (effect size {scale:.2g})", np.nan
    if resolved.sum() == 1:
        if resolved[-1]:
            return "suspect", "only the change at the smallest step exceeds the noise level", np.nan
        return "consistent", f"only the largest steps differ beyond the noise level {noise.max():.2g}", np.nan
    slope = _slope(taus[resolved], changes[resolved])
    verdict = "consistent" if slope >= .5 else "suspect" if slope >= .15 else "violated"
    return verdict, f"~ tau^{slope:.2f}", slope


def _slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = (x > 0) & (y > 0) & np.isfinite(y)
    if keep.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)[0])


def _rules_name(rule):
    return getattr(rule, "label", None) or getattr(rule, "name", type(rule).__name__)


def stress_populations(populations, scales=(1., 4., 16.)):
    """The populations scaled about the origin by each factor: larger and farther away."""
    return {s: [Population(s*p.points, p.weights) for p in populations] for s in scales}


# ----------------------------------------------------------------------------- (A1)
def check_first_order(rule, populations, functions=None, taus=None, replicates=64, seed=0):
    """(A1): the quotient (<phi, T_tau mu> - <phi, mu>)/tau settles as tau decreases,
    uniformly over the populations and test functions examined."""
    functions = functions or battery(populations, 8)
    exact = all(_exact(rule, phi) for phi in functions)
    taus = np.asarray(taus if taus is not None else (.1/2.**np.arange(6) if exact else RANDOM_TAUS), float)
    Q, increments, noise = [], np.zeros(len(taus)-1), np.zeros(len(taus)-1)
    for pop in populations:
        for phi in functions:
            q = _quotients(rule, pop, phi, taus, replicates, seed)/phi.bl_norm
            Q.append(q.mean(axis=0))
            diff = np.abs(q[:, :-1]-q[:, 1:])
            increments = np.maximum(increments, np.abs(q[:, :-1].mean(axis=0)-q[:, 1:].mean(axis=0)))
            if len(q) > 1:
                se = (q[:, :-1]-q[:, 1:]).std(axis=0, ddof=1)/np.sqrt(len(q))
                noise = np.maximum(noise, 3*se)
            del diff
    Q = np.array(Q)
    scale = float(np.max(np.abs(Q)))+1e-12
    # a diverging quotient shows changes that grow as tau shrinks; stability within a noise level
    # smaller than the effect counts as consistent
    verdict, stat, slope = _settling(taus[:-1], increments, noise, scale)
    if stat.startswith("~"):
        stat = f"changes between step sizes {stat}; largest change per unit step {scale:.3g}"
    meaning = {
        "consistent": "The operator has a well-defined first-order effect: as the step size shrinks, its change "
                      "of population averages per unit step settles to a limit, at about the same rate on every "
                      "population examined. This is what allows its effect to be added to those of other operators.",
        "suspect": "The change per unit step settles only slowly. Check how the rule uses the step size: the "
                   "theory needs moves of size tau (drift) or sqrt(tau) (noise), reweighting by exp(-tau Phi), "
                   "or replacement of a fraction tau of the population.",
        "violated": "The change per unit step does not settle as the step shrinks, so the operator has no "
                    "first-order effect in the sense of the theory. Usually the rule does not become small when "
                    "tau is small (for example it replaces a fixed fraction of the population, or moves points "
                    "by a fixed amount); rescale it with tau as in the 'suspect' advice.",
        "not resolved": "The random variation of the rule hides the differences between step sizes. Increase "
                        "`replicates` or use larger populations."}[verdict]
    return Finding("first-order effect (A1)", "composition assumption (A1)", stat, verdict, meaning,
                   {"taus": taus, "increments": increments, "noise": noise, "slope": slope, "quotients": Q})


# ----------------------------------------------------------------------------- (A2)
def check_moment_stability(rule, populations, tau=.1, q=3, scales=(1., 4., 16.), replicates=8, seed=0):
    """(A2): the q-th moment int (1 + |x|^q) dmu after one step stays within a moderate factor
    of the moment before, also for populations scaled far out."""
    ratios = {}
    for s, pops in stress_populations(populations, scales).items():
        r = []
        for pop in pops:
            before = pop.expect(1+np.linalg.norm(pop.points, axis=1)**q)
            reps = 1 if getattr(rule, "exact", False) else replicates
            after = np.mean([(lambda law: law.expect(1+np.linalg.norm(law.points, axis=1)**q))(
                rule.step(pop, tau, np.random.default_rng([seed, k]))) for k in range(reps)])
            r.append(after/before)
        ratios[s] = np.array(r)
    worst = max(float(np.max(v)) for v in ratios.values())
    trend = [float(np.max(ratios[s])) for s in scales]
    if not np.isfinite(worst) or worst > 5:
        verdict = "violated"
    elif worst > 1.5:
        verdict = "suspect"
    else:
        verdict = "consistent"
    stat = f"largest moment ratio after one step (tau={tau}): " + ", ".join(
        f"{t:.3g} at scale {s:g}" for s, t in zip(scales, trend))
    meaning = {
        "consistent": "One step keeps the population's size (its q-th moment) within a moderate factor, also "
                      "for populations far from the origin. This keeps every intermediate population of a "
                      "composed step in a bounded class, which the composition theorem needs.",
        "suspect": "One step can enlarge the population noticeably; look at whether the factor grows with the "
                   "scale of the population. Growth that worsens with scale indicates a rule that pushes far-away "
                   "individuals farther out (superlinear drift, heavy-tailed perturbations).",
        "violated": "One step can blow the population up (the moment grows by a large factor or becomes "
                    "infinite). Bound the rule's moves (clip, normalize, or use light-tailed perturbations); "
                    "until then neither composition nor convergence results apply."}[verdict]
    return Finding("moment stability (A2)", "composition assumption (A2)", stat, verdict, meaning,
                   {"scales": np.array(scales), "worst_by_scale": np.array(trend), "ratios": ratios, "q": q})


# ----------------------------------------------------------------------------- (A3)
def check_near_identity(rule, populations, functions=None, taus=(.1, .05, .025, .0125), copies=8, seed=0):
    """(A3): d*(T_tau mu, mu) <= C tau^alpha for some alpha > 0, with the population metric
    d* = bounded-Lipschitz part (lower bound from the battery) + 2-Wasserstein part."""
    functions = functions or battery(populations, 8)
    taus = np.asarray(taus, float)
    dist = np.zeros((len(populations), len(taus)))
    thinned = False
    for i, pop in enumerate(populations):
        for j, t in enumerate(taus):
            law = rule.step(pop, t, np.random.default_rng([seed, i, j]), copies)
            d2, flag = w2(law, pop, np.random.default_rng([seed, i]), return_flag=True)
            dist[i, j] = bl_lower(law, pop, functions)+d2
            thinned |= flag
    worst = dist.max(axis=0)
    alpha = _slope(taus, worst)
    C = float(np.max(worst/taus**alpha)) if np.isfinite(alpha) else np.nan
    if np.all(worst < 1e-12):
        verdict, stat = "consistent", "the operator does not change these populations"
    else:
        verdict = "consistent" if alpha >= .35 else "suspect" if alpha >= .1 else "violated"
        stat = f"d*(T_tau mu, mu) ~ {C:.3g} tau^{alpha:.2f}"
    meaning = {
        "consistent": "A small step changes the population only a little: the distance between the population "
                      "before and after shrinks like a power of the step size (about tau^0.5 for random "
                      "perturbations and offspring replacement, tau^1 for deterministic moves). This lets the "
                      "effects of operators applied one after another be evaluated at the starting population.",
        "suspect": "The distance shrinks only slowly with the step size; the rule may contain a part that does "
                   "not scale with tau.",
        "violated": "A small step still changes the population by a fixed amount: the rule does not become the "
                    "identity as tau -> 0, and the composition theorem does not apply to it."}[verdict]
    notes = {"taus": taus, "distance": dist, "alpha": alpha, "C": C, "subsampled": thinned}
    return Finding("near identity (A3)", "composition assumption (A3)", stat, verdict, meaning, notes)


# ----------------------------------------------------------------------------- (A4)
def _path_generators(rule, pop, target, phi, steps, replicates, seed):
    """Generator estimates along the straight path from pop to target (same weights)."""
    exact = _exact(rule, phi)
    tau_pair = (2e-3, 1e-3) if exact else (RANDOM_TAUS[-1],)
    vals, ses = [], []
    for s in np.linspace(0, 1, steps+1):
        mid = Population(pop.points+s*(target-pop.points), pop.weights)
        q = _quotients(rule, mid, phi, tau_pair, replicates, seed)
        g = 2*q[:, 1]-q[:, 0] if exact else q[:, 0]
        vals.append(g)
    vals = np.array(vals)                       # (steps+1, K)
    diffs = np.abs(np.diff(vals.mean(axis=1)))
    se = np.zeros_like(diffs) if vals.shape[1] == 1 else np.diff(vals, axis=0).std(axis=1, ddof=1)/np.sqrt(vals.shape[1])
    return diffs, se


def check_continuity(rule, populations, functions=None, displacement=.5, levels=None, replicates=24, seed=0):
    """(A4): the first-order effect G[mu](phi) changes continuously (Lipschitz) with the
    population.  Atoms are moved along a straight path; the largest change of G per unit
    of distance is computed on finer and finer subdivisions of the path.  For a Lipschitz
    effect it stays bounded; a jump (e.g. hard truncation by rank) makes it grow in
    proportion to the refinement."""
    functions = functions or battery(populations, 3)
    exact = all(_exact(rule, phi) for phi in functions)
    levels = levels or ((16, 64, 256) if exact else (8, 32, 128))
    rng = np.random.default_rng(seed)
    lipschitz = np.zeros(len(levels))
    unresolved = np.zeros(len(levels), dtype=bool)
    for pop in populations:
        spread = float(np.sqrt(np.mean(np.var(pop.points, axis=0))))+1e-12
        target = pop.points+displacement*spread*rng.normal(size=pop.points.shape)
        length = float(np.sqrt(np.sum(pop.weights*np.sum((target-pop.points)**2, axis=1))))
        for phi in functions:
            for i, K in enumerate(levels):
                diffs, se = _path_generators(rule, pop, target, phi, K, replicates, seed)
                ok = diffs > 3*se
                if ok.any():
                    lipschitz[i] = max(lipschitz[i], float(np.max(diffs[ok]))/(length/K)/phi.bl_norm)
                if not ok.any():
                    unresolved[i] = True
    growth = lipschitz[-1]/lipschitz[0] if lipschitz[0] > 0 else (np.inf if lipschitz[-1] > 0 else 1.)
    last = lipschitz[-1]/lipschitz[-2] if lipschitz[-2] > 0 else (np.inf if lipschitz[-1] > 0 else 1.)
    factor = levels[-1]/levels[-2]
    if lipschitz.max() == 0:
        verdict, stat = ("not resolved" if unresolved.any() else "consistent"), "no change of the effect along the paths"
    else:
        # a jump makes the rate grow by the refinement factor at every refinement; a Lipschitz
        # effect stops growing once the subdivision resolves it
        verdict = "consistent" if last <= 1.5 else "suspect" if last <= .75*factor else "violated"
        stat = (f"largest rate of change {lipschitz[-1]:.3g}; last refinement x{factor:g} changes it x{last:.2f} "
                f"(x{growth:.2f} overall)")
        if verdict == "consistent" and growth > 3:
            stat += "; steep but continuous"
    meaning = {
        "consistent": "The operator's first-order effect changes in proportion to how much the population "
                      "changes (it is Lipschitz in the population). This is what lets the effects of operators "
                      "applied one after another be added.",
        "suspect": "The effect changes very steeply in places and the check cannot tell a steep but continuous "
                   "rule from a jump at this resolution. Look for thresholds in the rule (ranks, cut-offs, "
                   "best-of comparisons) and for smoothing parameters that are very small; a larger smoothing "
                   "width gives smaller constants in the theory.",
        "violated": "The effect jumps when the population changes slightly: the rule contains a discontinuity, "
                    "typically a hard threshold such as truncation by rank or an exact best-of comparison. "
                    "Replace it by a smoothed version (a soft rank or a sigmoid instead of a step).",
        "not resolved": "The random variation of the rule hides the changes along the path; increase `replicates`."}[verdict]
    return Finding("continuity in the population (A4)", "composition assumption (A4)", stat, verdict, meaning,
                   {"levels": np.array(levels), "lipschitz": lipschitz, "growth": growth})


# ----------------------------------------------------------------------------- operator report
def check_operator(rule, populations, functions=None, seed=0, fast=False):
    """All four conditions of the composition assumption for one operator, on the given populations.

    ``populations``: a few representative populations (for example broad and clustered
    ones of the size you intend to use).  ``fast`` uses fewer replicates and levels."""
    functions = functions or battery(populations, 8, np.random.default_rng(seed))
    reps = 16 if fast else 64
    findings = [check_first_order(rule, populations, functions, replicates=reps, seed=seed),
                check_moment_stability(rule, populations, seed=seed),
                check_near_identity(rule, populations, functions, seed=seed),
                check_continuity(rule, populations, functions[:3], replicates=8 if fast else 24, seed=seed,
                                 levels=None if not fast else ((16, 64, 256) if getattr(rule, "exact", False) else (4, 16, 64)))]
    report = Report(f"Operator check: {_rules_name(rule)} ({len(populations)} populations, {len(functions)} test functions)",
                    findings, plotter=_plot_operator)
    report.notes.append("Verdicts describe the populations examined; include populations of the kinds your "
                        "algorithm will meet (broad, clustered, far from the optimum, near it).")
    return report


def _plot_operator(report):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), layout="constrained")
    a1, a2, a3, a4 = (f.data for f in report.findings[:4])
    def log_ticks(ax, values):
        ax.set_xscale("log")
        ax.set_xticks(values, [f"{v:g}" for v in values])
        ax.minorticks_off()

    ax = axes[0]
    taus = a1["taus"][:-1]
    if np.any(a1["increments"] > 0):
        ax.loglog(taus, np.maximum(a1["increments"], 1e-300), "o-", label="change between steps")
        if np.any(a1["noise"] > 0):
            ax.loglog(taus, np.maximum(a1["noise"], 1e-300), ":", color="0.5", label="noise level")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(.5, .5, "no change between steps", ha="center", transform=ax.transAxes)
    log_ticks(ax, taus[::max(1, len(taus)//3)])
    ax.set(title="(A1) settles as tau shrinks", xlabel="tau")
    ax = axes[1]
    ax.plot(a2["scales"], a2["worst_by_scale"], "o-")
    log_ticks(ax, a2["scales"])
    ax.axhline(1, color="0.5", lw=.6)
    ax.set(title="(A2) moment ratio after a step", xlabel="population scale")
    ax = axes[2]
    ax.loglog(a3["taus"], a3["distance"].max(axis=0), "o-")
    log_ticks(ax, a3["taus"][::max(1, len(a3["taus"])//3)])
    ax.set(title=f"(A3) distance ~ tau^{a3['alpha']:.2f}", xlabel="tau")
    ax = axes[3]
    ax.plot(a4["levels"], np.maximum(a4["lipschitz"], 1e-300), "o-")
    log_ticks(ax, a4["levels"])
    ax.set(title="(A4) largest rate of change", xlabel="path subdivisions")
    for ax, f in zip(axes, report.findings):
        ax.text(.98, .97, f.verdict, transform=ax.transAxes, fontsize=9, ha="right", va="top", fontweight="bold",
                color={"consistent": "#347c63", "suspect": "#c16c23", "violated": "#c0392b"}.get(f.verdict, "0.4"))
    return fig


# ----------------------------------------------------------------------------- assembly level
def check_composition(assembly, populations, functions=None, taus=None, replicates=64, seed=0, alternatives=()):
    """Composition theorem: the composed step's change per unit step tends to the sum of the
    operators' first-order effects.  ``alternatives``: assemblies with other orders of the
    same operators, whose quotients should approach the same limit."""
    functions = functions or battery(populations, 8)
    exact = all(_exact(assembly, phi) for phi in functions)
    taus = np.asarray(taus if taus is not None else (.1/2.**np.arange(6) if exact else RANDOM_TAUS), float)
    remainder = np.zeros(len(taus))
    noise = np.zeros(len(taus))
    scale = 1e-12
    order_gap = np.zeros(len(taus))
    for p, pop in enumerate(populations):
        for phi in functions:
            G = sum(op.generator(pop, phi, rng=np.random.default_rng([seed, p])) for op in assembly.operators)
            q = _quotients(assembly, pop, phi, taus, replicates, seed)/phi.bl_norm
            remainder = np.maximum(remainder, np.abs(q.mean(axis=0)-G/phi.bl_norm))
            if len(q) > 1:
                noise = np.maximum(noise, 3*q.std(axis=0, ddof=1)/np.sqrt(len(q)))
            scale = max(scale, abs(G)/phi.bl_norm)
            for alt in alternatives:
                qa = _quotients(alt, pop, phi, taus, replicates, seed)/phi.bl_norm
                order_gap = np.maximum(order_gap, np.abs(qa.mean(axis=0)-q.mean(axis=0)))
    verdict, stat, slope = _settling(taus, remainder, noise, scale)
    if stat.startswith("~"):
        stat = f"remainder {stat} (at tau={taus[-1]:g}: {remainder[-1]:.3g})"
    if alternatives:
        stat += f"; order difference ~ tau^{_slope(taus, order_gap):.2f}"
    meaning = {
        "consistent": "To first order, the composed step does what its operators do separately, added up. The "
                      "first-order attribution of the assembly's effect to its operators is therefore meaningful, "
                      "and a convergence argument can be built operator by operator.",
        "suspect": "The composed step's change approaches the sum of the operators' effects only slowly; check "
                   "the operator reports (A1-A4) for the operator responsible.",
        "violated": "The composed step does not behave like the sum of its operators' effects as the step "
                    "shrinks: at least one operator fails a condition of the composition assumption, or the operators "
                    "interact at first order through a step-size scaling that is not of the required form."}[verdict]
    return Finding("composition (sum of effects)", "composition theorem", stat, verdict, meaning,
                   {"taus": taus, "remainder": remainder, "noise": noise, "order_gap": order_gap, "slope": slope})


def check_cutoff(rule, populations, observable, radii_factors=(1, 2, 4, 8, 16, 32), replicates=64, seed=0):
    """Cutoff-admissible tests: the effect on an unbounded quantity (objective gap, squared distance)
    is the limit of the effects on its smoothly cut-off versions chi(|x|/r) Upsilon as r grows,
    and these stay bounded."""
    values = []
    for p, pop in enumerate(populations):
        R0 = float(np.max(np.linalg.norm(pop.points, axis=1)))+1e-12
        row = []
        for f in radii_factors:
            obs = truncated(observable, R0*f)
            gens = [op.generator(pop, obs, rng=np.random.default_rng([seed, p]), replicates=replicates)
                    if not op.exact else op.generator(pop, obs) for op in (rule.operators if isinstance(rule, Assembly) else [rule])]
            row.append(sum(gens))
        values.append(row)
    values = np.array(values)
    scale = np.abs(values).max(axis=1)+1e-12
    change = (np.abs(np.diff(values, axis=1))/scale[:, None]).max(axis=0)
    if change[-1] < 1e-3:
        verdict = "consistent"
    elif change[-1] < change[0]*.5:
        verdict = "suspect"
    else:
        verdict = "violated"
    stat = f"relative change between the two largest cutoffs {change[-1]:.2g} (first {change[0]:.2g})"
    meaning = {
        "consistent": "The effect on the unbounded error measure is well defined: cutting the measure off at a "
                      "larger and larger radius changes the computed effect less and less. The convergence "
                      "theorem can then be applied with this error measure.",
        "suspect": "The effect still changes at large cut-off radii: the rule sends some mass very far out "
                   "(heavy tails). Check moment stability (A2) and consider light-tailed perturbations.",
        "violated": "The effect on the unbounded measure does not settle as the cut-off radius grows: the rule "
                    "produces too much mass at large distances, and the drift condition cannot be checked with "
                    "this error measure."}[verdict]
    return Finding("unbounded error measure (cutoff)", "cutoff-admissible tests", stat, verdict, meaning,
                   {"factors": np.array(radii_factors), "values": values, "change": change})


def check_assembly(assembly, populations, objective, offset=0., seed=0, operator_checks=True, fast=False,
                   calibration=None):
    """Assembly-level report: the operator conditions, composition, the unbounded error
    measure, and the drift coefficient with its attribution to the operators.

    ``objective``: the error measure Upsilon (an object with ``value``; a problem of
    :mod:`evoscope.problems` or an :class:`~evoscope.operators.Observable`).  ``offset`` is
    subtracted (a lower bound f_lb when f_* is unknown).  ``calibration``: populations for
    the drift-coefficient estimate (default: the given populations)."""
    findings = []
    functions = battery(populations, 8, np.random.default_rng(seed))
    if operator_checks:
        for op in assembly.operators:
            r = check_operator(op, populations, functions, seed=seed, fast=True)
            verdicts = ", ".join(f"{f.check.split('(')[-1].rstrip(')')}: {f.verdict}" for f in r.findings)
            findings.append(Finding(f"operator {op.name}", "composition assumption", verdicts, r.verdict,
                                    "Summary of the operator check (run check_operator for the details and advice).",
                                    {"report": r}))
    findings.append(check_composition(assembly, populations, functions, replicates=16 if fast else 64, seed=seed))
    findings.append(check_cutoff(assembly, populations[:3], objective, replicates=16 if fast else 64, seed=seed))
    result = attribution.coefficients(assembly, calibration or populations, objective, offset=offset,
                                      rng=np.random.default_rng(seed))
    lam = attribution.estimate_drift_coefficient(result["total"])
    minima = {n: float(np.nanmin(result["ratios"][:, j])) for j, n in enumerate(result["names"])}
    worst = min(minima, key=minima.get)
    stat = f"lambda_cal = {lam:.3g}; smallest component coefficients: " + ", ".join(f"{n} {v:.3g}" for n, v in minima.items())
    if lam > 0:
        verdict = "consistent"
        meaning = (f"On every population examined, the assembly reduces the error measure at first order, at "
                   f"relative rate at least {lam:.3g} per unit time. This is evidence for the drift hypothesis of "
                   f"the convergence theorem on populations like these (not a certificate). The component "
                   f"coefficients show what each operator contributes; negative ones are compensated by others.")
    else:
        verdict = "violated"
        meaning = (f"On some population the assembly increases the error measure at first order, so no positive "
                   f"drift coefficient exists on this class of populations. The operator with the most negative "
                   f"contribution is {worst}. Options: restrict attention to populations where the rate is "
                   f"positive (and check that the algorithm stays there), change the error measure, or use the "
                   f"residual form (evoscope.attribution.residual_constant).")
    findings.append(Finding("drift coefficient and attribution", "mean-field convergence theorem, operator-wise closure", stat, verdict,
                            meaning, {"coefficients": result, "lambda_cal": lam}))
    report = Report(f"Assembly check: {assembly.label}", findings, plotter=_plot_assembly)
    report.notes.append("The drift coefficient is estimated from the populations given; check it on fresh "
                        "populations with evoscope.attribution.validate before relying on it.")
    return report


def _plot_assembly(report):
    import matplotlib.pyplot as plt
    from .viz import plot_component_coefficients
    comp = report["composition (sum of effects)"].data
    cut = report["unbounded error measure (cutoff)"].data
    coef = report["drift coefficient and attribution"].data["coefficients"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
    axes[0].loglog(comp["taus"], np.maximum(comp["remainder"], 1e-300), "o-", label="remainder")
    if np.any(comp["noise"] > 0):
        axes[0].loglog(comp["taus"], np.maximum(comp["noise"], 1e-300), ":", color="0.5", label="noise level")
    axes[0].set(title="composition remainder", xlabel="tau")
    axes[0].legend(frameon=False, fontsize=8)
    if np.any(cut["change"] > 0):
        axes[1].loglog(cut["factors"][1:], np.maximum(cut["change"], 1e-300), "o-")
    else:
        axes[1].text(.5, .5, "no change: the cut-off\ndoes not affect the effect", ha="center", va="center",
                     transform=axes[1].transAxes)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    axes[1].set(title="cutoff: relative change", xlabel="radius / population radius")
    plot_component_coefficients(coef, ax=axes[2])
    axes[2].set_title("attribution")
    return fig
