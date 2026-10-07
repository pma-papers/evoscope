"""Algorithm-level checks: for an implemented algorithm, without splitting it into operators.

When an algorithm already runs, the questions of the theory can be asked of the whole
generation step:

* does one generation reduce the mean objective gap of the proposals, and by what factor
  per generation (:func:`check_contraction`; the finite-population recursion of
  recombinative evolution strategy in the paper, and its evaluation count);
* with what probability has the algorithm found a target point by a given generation
  (:func:`check_discovery`; the measured discovery bound);
* what does each switchable ingredient contribute, per generation and along runs
  (:func:`check_ablation`; the algorithm-level counterpart of the attribution);
* if the algorithm has a step-size parameter, does it behave like a small-step operator
  whose first-order effect can be analysed with the calculus (:func:`check_small_step`);
* do runs show the failure patterns the paper distinguishes: agreement at a point that is
  not optimal, stagnation, blow-up (:func:`diagnose_runs`).

The algorithm is used through an ask/tell interface (:class:`Algorithm`): ``ask`` draws
candidates from the current proposal law without changing the state, ``tell`` returns
the next state.  Wrappers exist for functions (:class:`FunctionalAlgorithm`), objects
with ``ask``/``tell`` methods (:class:`AskTellAlgorithm`), and assemblies of operators
(:class:`AssemblyAlgorithm`).
"""
import copy

import numpy as np

from . import contrasts, discovery
from .population import Population
from .report import Finding, Report
from .checks import _settling

__all__ = ["Algorithm", "FunctionalAlgorithm", "AskTellAlgorithm", "AssemblyAlgorithm", "run", "run_repetitions",
           "collect_states", "proposal_gap", "check_contraction", "check_discovery", "check_ablation",
           "ablation_runs", "check_small_step", "diagnose_runs", "check_algorithm"]


# ----------------------------------------------------------------------------- interface
class Algorithm:
    """Ask/tell interface used by the algorithm-level checks.

    ``batch_size``: candidates evaluated per generation.
    ``initialize(rng) -> state``;
    ``ask(state, size, rng, probe=False) -> array (size, d)``: draws from the current proposal
    law; must not change ``state`` (``probe=True`` marks draws that are only counted);
    ``tell(state, candidates, values, rng) -> state``;
    ``population(state) -> array`` (optional): the current population, for spread diagnostics.
    """
    batch_size = None

    def initialize(self, rng):
        raise NotImplementedError

    def ask(self, state, size, rng, probe=False):
        raise NotImplementedError

    def tell(self, state, candidates, values, rng):
        raise NotImplementedError

    def population(self, state):
        return None

    def copy_state(self, state):
        return copy.deepcopy(state)


class FunctionalAlgorithm(Algorithm):
    """An algorithm given by three functions (and optionally a population accessor)."""

    def __init__(self, initialize, ask, tell, batch_size, population=None, name="algorithm"):
        self._init, self._ask, self._tell, self._pop = initialize, ask, tell, population
        self.batch_size, self.name = batch_size, name

    def initialize(self, rng):
        return self._init(rng)

    def ask(self, state, size, rng, probe=False):
        return np.asarray(self._ask(state, size, rng))

    def tell(self, state, candidates, values, rng):
        return self._tell(state, candidates, values, rng)

    def population(self, state):
        return None if self._pop is None else self._pop(state)


class AskTellAlgorithm(Algorithm):
    """Wraps an optimizer object with methods ``ask(n)`` and ``tell(candidates, values)``.

    ``factory(rng)`` creates the optimizer.  Probes are drawn from a deep copy of the object,
    so the trajectory is unchanged; ``reseed(obj, rng)`` (optional) gives the copy fresh
    randomness, which keeps the probes independent of the next real batch when the object
    stores its own random generator."""

    def __init__(self, factory, batch_size, reseed=None, population=None, name="optimizer"):
        self.factory, self.batch_size, self.reseed, self._pop, self.name = factory, batch_size, reseed, population, name

    def initialize(self, rng):
        return self.factory(rng)

    def ask(self, state, size, rng, probe=False):
        target = self.copy_state(state) if probe else state
        if probe and self.reseed is not None:
            self.reseed(target, rng)
        return np.asarray(target.ask(size))

    def tell(self, state, candidates, values, rng):
        state.tell(candidates, values)
        return state

    def population(self, state):
        return None if self._pop is None else self._pop(state)


class AssemblyAlgorithm(Algorithm):
    """An assembly of operators run as an algorithm: each generation draws the new
    population of size N from T_tau mu."""

    def __init__(self, assembly, tau, initial, population_size, name=None):
        self.assembly, self.tau, self.initial, self.batch_size = assembly, tau, initial, population_size
        self.name = name or assembly.label

    def initialize(self, rng):
        out = self.initial(rng)
        return out if isinstance(out, Population) else Population.uniform(out)

    def ask(self, state, size, rng, probe=False):
        return self.assembly.sample(state, self.tau, size, rng)

    def tell(self, state, candidates, values, rng):
        return Population.uniform(candidates)

    def population(self, state):
        return state.points

    def copy_state(self, state):
        return state


# ----------------------------------------------------------------------------- running
def run(alg, objective, generations, rng, level=None, n_probe=0, f_star=0., keep_states=None):
    """One run.  Traces per generation (index 0 is before the first generation):
    ``best_so_far`` (gap of the best evaluated candidate), ``batch_mean_gap``,
    ``population_variance`` (if the algorithm exposes its population), ``offspring_hit``
    (a candidate with f <= level so far) and ``probe_counts`` (probes with f <= level)."""
    state = alg.initialize(rng)
    best, hit = np.inf, False
    out = {"best_so_far": np.full(generations+1, np.nan), "batch_mean_gap": np.full(generations+1, np.nan),
           "population_variance": np.full(generations+1, np.nan), "offspring_hit": np.zeros(generations+1, bool),
           "probe_counts": np.zeros(generations, int)}
    states = []

    def record(k, state):
        pop = alg.population(state)
        if pop is not None:
            pop = np.asarray(pop)
            out["population_variance"][k] = float(np.mean(np.sum((pop-pop.mean(axis=0))**2, axis=1)))
        if keep_states is not None and k in keep_states:
            states.append(alg.copy_state(state))

    record(0, state)
    for k in range(1, generations+1):
        if n_probe:
            out["probe_counts"][k-1] = int(np.sum(objective(alg.ask(state, n_probe, rng, probe=True)) <= level))
        X = alg.ask(state, alg.batch_size, rng)
        y = objective(X)
        state = alg.tell(state, X, y, rng)
        best = min(best, float(np.min(y)))
        hit = hit or (level is not None and bool(np.any(y <= level)))
        out["best_so_far"][k], out["batch_mean_gap"][k] = best-f_star, float(np.mean(y))-f_star
        out["offspring_hit"][k] = hit
        record(k, state)
    if keep_states is not None:
        out["states"] = states
    return out


def run_repetitions(alg, objective, generations, repetitions, rng, **kwargs):
    """Independent runs; arrays with a leading run axis (and ``states`` as a list of lists)."""
    runs = [run(alg, objective, generations, rng, **kwargs) for _ in range(repetitions)]
    out = {k: np.stack([r[k] for r in runs]) for k in runs[0] if k != "states"}
    if "states" in runs[0]:
        out["states"] = [r["states"] for r in runs]
    return out


def collect_states(alg, objective, generations, repetitions, rng, at=(0, 5, 10, 20, 40)):
    """States from pilot runs at the given generations: the population of states on which
    the per-generation behaviour is examined (include early and late generations)."""
    at = tuple(k for k in at if k <= generations)
    runs = run_repetitions(alg, objective, max(at), repetitions, rng, keep_states=set(at))
    return [s for states in runs["states"] for s in states]


def proposal_gap(alg, state, objective, rng, size=256, f_star=0.):
    """Mean objective gap of the current proposal law, estimated from `size` probes."""
    return float(np.mean(objective(alg.ask(state, size, rng, probe=True))))-f_star


def _next_gaps(alg, state, objective, rng, inner, size, f_star):
    vals = []
    for _ in range(inner):
        s = alg.copy_state(state)
        X = alg.ask(s, alg.batch_size, rng)
        s = alg.tell(s, X, objective(X), rng)
        vals.append(proposal_gap(alg, s, objective, rng, size, f_star))
    return np.array(vals)


# ----------------------------------------------------------------------------- checks
def check_contraction(alg, objective, states, rng, inner=16, size=256, f_star=0.):
    """Per-generation contraction of the mean objective gap V of the proposal law.

    For every state: V and the average of V after one generation over `inner` independent
    generations.  If E[V_next | state] <= rho V everywhere with rho < 1, the gap decays
    geometrically (the finite-generation recursion of the paper).  Otherwise the residual form
    E[V_next] <= rho V + c is reported, with the rho in (0, 1) that gives the lowest floor c/(1 - rho)."""
    V = np.array([proposal_gap(alg, s, objective, rng, size, f_star) for s in states])
    nxt = np.array([_next_gaps(alg, s, objective, rng, inner, size, f_star).mean() for s in states])
    keep = V > 1e-12
    ratio = nxt[keep]/V[keep]
    rho_max, median = float(np.max(ratio)), float(np.median(ratio))
    grid = np.linspace(.01, .99, 99)
    floors = [max(0., float(np.max(nxt[keep]-r*V[keep])))/(1-r) for r in grid]
    rho = float(grid[int(np.argmin(floors))])
    c = float(max(0., np.max(nxt[keep]-rho*V[keep])))
    if rho_max < 1:
        verdict = "consistent"
        stat = f"E[V_next]/V <= {rho_max:.3f} on all {keep.sum()} states (median {median:.3f})"
        meaning = (f"On every state examined, one generation reduces the mean objective gap of the proposals by "
                   f"at least the factor {rho_max:.3f} on average. If this holds along the run, the gap after k "
                   f"generations is at most {rho_max:.3f}^k times the initial gap, and the probability that no "
                   f"candidate is within eps of the optimum is at most that gap divided by eps (the "
                   f"finite-population recursion). This is evidence on the states sampled, not a certificate.")
    elif median < 1:
        verdict = "suspect"
        stat = (f"ratio up to {rho_max:.3f} (median {median:.3f}); residual form: E[V_next] <= {rho:.3f} V + {c:.3g}, "
                f"floor {c/(1-rho):.3g}")
        meaning = ("Most states contract, but some do not, typically states near a local minimum, with a "
                   "collapsed population, or where a fixed noise level keeps the gap from shrinking. The residual "
                   "form says the gap decreases geometrically until it reaches the floor c/(1-rho); below that "
                   "level no decrease is guaranteed. A floor below your target accuracy is harmless; a floor "
                   "above it means the mean gap cannot certify that accuracy.")
    else:
        verdict = "violated"
        stat = f"median ratio {median:.3f} >= 1"
        meaning = ("On most states one generation does not reduce the mean objective gap of the proposals. The "
                   "algorithm may still find good points (check discovery), but a convergence argument through "
                   "the mean gap does not apply.")
    return Finding("per-generation contraction", "finite-generation recursion, evaluation count", stat, verdict, meaning,
                   {"V": V, "next": nxt, "ratio": ratio, "rho_max": rho_max, "rho": rho, "c": c})


def check_discovery(results, batch, level, delta=.05, n_probe=None):
    """The measured discovery bound from runs with probes (see :func:`run_repetitions`
    with ``n_probe`` and ``level``)."""
    counts = results["probe_counts"]
    n_probe = n_probe or int(results.get("n_probe", 0)) or None
    if n_probe is None:
        raise ValueError("pass n_probe (the number of probes per generation used in the runs)")
    R, m = counts.shape
    bound = float(discovery.measured_bound(counts, n_probe, batch)["upper"])
    floor = discovery.floor(R, batch, m)
    observed = float(1-results["offspring_hit"][:, -1].mean())
    stat = (f"P(no candidate with f <= {level:g} in the batches up to generation {m}) <= {bound:.3f} "
            f"(95%, R = {R}); observed {observed:.3f}; floor {floor:.3f}")
    if bound <= delta:
        verdict, meaning = "consistent", (
            f"With 95% confidence over the {R} runs, a further run of this algorithm has a probability of at most "
            f"{bound:.3f} of not having proposed a target point by generation {m}. This is a guarantee about the "
            f"algorithm as run, obtained from measured success probabilities, with no approximation argument.")
    elif bound <= max(1.2*floor, floor+.02):
        verdict, meaning = "not resolved", (
            f"The design cannot certify the level {delta}: even with no failures the bound cannot go below "
            f"{floor:.3f}. Use more runs (the floor falls roughly like 1/R) or a larger batch.")
    else:
        verdict, meaning = "suspect", (
            "The bound is above the level: in the last generations the proposal law put less than a quarter of its "
            "mass on the target set in some runs, or this was not resolved by the probes. Either the algorithm "
            "stalls in some runs (compare with the observed frequency), or more probes are needed.")
    return Finding("discovery guarantee", "measured discovery bound", stat, verdict, meaning,
                   {"bound": bound, "floor": floor, "observed": observed})


def check_ablation(variants, reference, objective, states, rng, inner=16, size=256, f_star=0., seed=0):
    """One-generation effect of each switchable ingredient.

    ``variants``: dict name -> Algorithm, sharing the state format; ``reference``: the name
    of the full algorithm.  For every state and variant the mean gap after one generation is
    estimated with the same random numbers (common seeds), and the difference variant minus
    reference is averaged: positive means that removing or changing the ingredient makes the
    next generation worse in mean gap."""
    names = [n for n in variants if n != reference]
    effects = {n: [] for n in names}
    for i, s in enumerate(states):
        def gap_after(alg):
            g = np.random.default_rng([seed, i])
            return _next_gaps(alg, s, objective, g, inner, size, f_star).mean()
        base = gap_after(variants[reference])
        for n in names:
            effects[n].append(gap_after(variants[n])-base)
    summary = {n: (float(np.mean(v)), float(np.std(v, ddof=1)/np.sqrt(len(v))) if len(v) > 1 else 0.)
               for n, v in effects.items()}
    stat = "; ".join(f"{n}: {m:+.3g} (se {se:.2g})" for n, (m, se) in summary.items())
    meaning = ("Difference in the mean objective gap of the proposals after one generation, variant minus the "
               "full algorithm, averaged over the states examined (with common random numbers). Positive values "
               "mean the ingredient that the variant removes or changes helps the mean gap per generation; "
               "negative values mean it costs mean gap (it may still help discovery: compare runs with "
               "ablation_runs, which follows the effect along whole runs).")
    return Finding("ingredient effects per generation", "attribution of effects", stat, "info", meaning,
                   {"effects": effects, "summary": summary})


def ablation_runs(variants, reference, objective, generations, repetitions, seed=0, bootstrap=2000, **kwargs):
    """Paired contrasts along runs: every variant is started from the same initial state as
    the reference in each run (later random draws are independent).  Returns the output of
    :func:`evoscope.contrasts.paired_contrasts` for the traces best_so_far and batch_mean_gap,
    keyed by variant name."""
    traces = {}
    for v, (name, alg) in enumerate(variants.items()):
        rows = []
        for r in range(repetitions):
            init = alg.initialize(np.random.default_rng([seed, r]))
            rows.append(_run_from(alg, init, objective, generations, np.random.default_rng([seed, r, v+1]), **kwargs))
        traces[name] = np.stack(rows)                                     # runs, generations+1, metrics
    reference_traces = traces[reference]
    names = [n for n in variants if n != reference]
    out = contrasts.paired_contrasts(reference_traces, np.stack([traces[n] for n in names], axis=1),
                                     np.random.default_rng(seed), bootstrap)
    return {"names": names, "metrics": ("best_so_far", "batch_mean_gap"), **out}


def _run_from(alg, state, objective, generations, rng, f_star=0.):
    rows = []
    population = alg.population(state)
    # generation 0: the initial population itself (identical across variants with matched starts),
    # or a first batch of proposals when the algorithm does not expose its population
    X = np.asarray(population) if population is not None else alg.ask(state, alg.batch_size, rng, probe=True)
    y0 = objective(X)
    rows.append((float(np.min(y0))-f_star, float(np.mean(y0))-f_star))
    best = float(np.min(y0))
    for _ in range(generations):
        X = alg.ask(state, alg.batch_size, rng)
        y = objective(X)
        state = alg.tell(state, X, y, rng)
        best = min(best, float(np.min(y)))
        rows.append((best-f_star, float(np.mean(y))-f_star))
    return np.array(rows)


def check_small_step(make_alg, states, objective, taus=(.4, .2, .1, .05), size=4096, groups=8, f_star=0., seed=0):
    """For an algorithm with a step-size parameter: does its update behave like a small-step
    operator?

    The algorithm is treated as one operator that maps the current state to its proposal
    law; ``make_alg(tau)`` returns it with step size tau (sharing the state format), and
    ``make_alg(0.0)`` must be the identity-like proposal (e.g. resampling the population).
    The change of the mean objective gap of the proposals per unit step, (V_tau - V_0)/tau,
    is estimated in `groups` independent groups of `size` proposals, with common random
    numbers across tau within a group; it should settle as tau shrinks.  Its limit is the
    algorithm's first-order effect G, and -G/V its drift coefficient."""
    taus = np.asarray(taus, float)
    Q = np.zeros((len(states), groups, len(taus)))
    V0 = np.zeros((len(states), groups))
    for i, s in enumerate(states):
        for k in range(groups):
            def gap(t):
                return proposal_gap(make_alg(t), s, objective, np.random.default_rng([seed, i, k]), size, f_star)
            V0[i, k] = gap(0.)
            Q[i, k] = [(gap(t)-V0[i, k])/t for t in taus]
    mean = Q.mean(axis=1)                                            # states, taus
    diff = np.abs(np.diff(mean, axis=1))
    se = np.diff(Q, axis=2).std(axis=1, ddof=1)/np.sqrt(groups)
    increments, noise = diff.max(axis=0), 3*se.max(axis=0)
    scale = float(np.abs(mean).max())+1e-12
    verdict, stat, slope = _settling(taus[:-1], increments, noise, scale)
    if stat.startswith("~"):
        stat = f"changes between step sizes {stat}"
    lam = -mean[:, -1]/np.maximum(V0.mean(axis=1), 1e-12)
    stat += f"; drift coefficient -G/V from {lam.min():.3g} to {lam.max():.3g} over the states"
    meaning = {
        "consistent": "The algorithm behaves like a small-step operator: the change of the mean gap of its "
                      "proposals per unit step settles as the step size shrinks. Its first-order effect G and the "
                      "drift coefficient -G/V can then be used as for an assembly of operators; a positive "
                      "minimum drift coefficient is evidence for the drift hypothesis of the convergence theorem.",
        "suspect": "The change per unit step settles slowly or only partly; the step-size parameter may not scale "
                   "every part of the update.",
        "violated": "The change per unit step does not settle: the step-size parameter does not make the update "
                    "small, so the small-step theory does not describe the algorithm through this parameter.",
        "not resolved": "The random variation hides the changes between step sizes; increase `size` or `groups`."}[verdict]
    return Finding("small-step behaviour", "composition theorem applied to the whole update", stat, verdict, meaning,
                   {"taus": taus, "quotients": mean, "V0": V0.mean(axis=1), "drift_coefficients": lam,
                    "increments": increments, "noise": noise})


def diagnose_runs(results, level, collapse_variance=1e-10, stall_generations=50):
    """Failure patterns in runs: collapse (population variance below `collapse_variance`
    while the batch mean gap is above `level`: agreement at a point that is not good),
    stagnation (no improvement of the best-so-far for `stall_generations` while it is above
    `level`), and blow-up (non-finite values)."""
    best, gap, var = results["best_so_far"], results["batch_mean_gap"], results["population_variance"]
    R, T = best.shape
    blowup = int(np.sum(~np.all(np.isfinite(gap[:, 1:]), axis=1)))
    collapse = int(np.sum(np.any((var < collapse_variance) & (gap > level), axis=1))) if np.isfinite(var).any() else 0
    stall = 0
    for r in range(R):
        b = best[r, 1:]
        if len(b) > stall_generations and b[-1] > level and b[-stall_generations-1] - b[-1] <= 1e-12*(1+abs(b[-1])):
            stall += 1
    failed = float(np.mean(best[:, -1] > level))
    stat = f"runs above level at the end: {failed:.3f}; collapsed {collapse}/{R}, stalled {stall}/{R}, blown up {blowup}/{R}"
    verdict = "violated" if blowup else "suspect" if (collapse or stall) else "consistent"
    meaning = {
        "consistent": "No run shows the failure patterns: no collapse at a point that is not good, no stagnation "
                      "above the level, no blow-up.",
        "suspect": "Some runs agree on a point that is not good (the population has collapsed while the mean gap "
                   "stays above the level) or stop improving above the level. Agreement is not optimality: "
                   "these runs need exploration (noise, restarts) or a larger population; check discovery for "
                   "how often this happens.",
        "violated": "Some runs produce non-finite values: the update blows up. Bound the moves of the algorithm."}[verdict]
    return Finding("run diagnostics", "convergence targets", stat, verdict, meaning,
                   {"collapsed": collapse, "stalled": stall, "blown_up": blowup, "final_failure": failed})


def check_algorithm(alg, objective, level, rng, generations=100, repetitions=64, n_probe=128, f_star=0.,
                    variants=None, reference=None, make_alg=None, inner=16, delta=.05):
    """Algorithm-level report: runs with probes, discovery guarantee, run diagnostics,
    per-generation contraction, and (optionally) ingredient effects and small-step behaviour.

    ``level``: the target value c (target set {f <= c}; f_* + eps when f_* is known).
    ``variants`` (dict name -> algorithm) with ``reference`` adds the ablation; ``make_alg``
    (tau -> algorithm) adds the small-step check."""
    results = run_repetitions(alg, objective, generations, repetitions, rng, level=level, n_probe=n_probe,
                              f_star=f_star, keep_states={0, 5, 10, 20, 40})
    states = [s for row in results["states"][:8] for s in row]
    findings = [diagnose_runs(results, level-f_star),
                check_discovery(results, alg.batch_size, level, delta, n_probe),
                check_contraction(alg, objective, states, rng, inner, f_star=f_star)]
    if variants:
        findings.append(check_ablation(variants, reference, objective, states[:16], rng, inner, f_star=f_star))
    if make_alg is not None:
        findings.append(check_small_step(make_alg, states[:12], objective, f_star=f_star))
    report = Report(f"Algorithm check: {getattr(alg, 'name', type(alg).__name__)} "
                    f"({repetitions} runs x {generations} generations, batch {alg.batch_size})",
                    findings, plotter=_plot_algorithm)
    report.results = results
    report.notes.append("States for the per-generation checks are taken from the first runs at generations "
                        "0, 5, 10, 20 and 40.")
    return report


def _plot_algorithm(report):
    import matplotlib.pyplot as plt
    res = report.results
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
    g = np.arange(1, res["best_so_far"].shape[1])
    for key, ax, title in (("best_so_far", axes[0], "best-so-far gap"), ("batch_mean_gap", axes[1], "mean gap of the batch")):
        values = np.maximum(res[key][:, 1:], 1e-300)
        ax.semilogy(g, np.median(values, axis=0), color="#246f96")
        ax.fill_between(g, np.quantile(values, .1, axis=0), np.quantile(values, .9, axis=0), color="#246f96", alpha=.2, lw=0)
        ax.set(title=title+" (median, 10-90%)", xlabel="generation")
    contraction = next(f for f in report.findings if f.check == "per-generation contraction").data
    axes[2].loglog(contraction["V"][contraction["V"] > 1e-12], contraction["ratio"], "o", ms=3, color="#1b2f44")
    axes[2].axhline(1, color="#c0392b", lw=.8)
    axes[2].set(title="E[V_next]/V per state", xlabel="V (mean gap of proposals)")
    return fig
