# %% [markdown]
# # Tutorial 9: What kind of term does an operator contribute?
#
# Every operator adds one term to the evolution equation (Tutorial 7), and the terms come
# in four kinds:
#
# | kind | what it does | canonical example |
# |---|---|---|
# | **reaction** | changes the weights of the individuals where they are; creates no new points | selection |
# | **transport** | moves each individual continuously along a direction | drift |
# | **diffusion** | spreads each individual by small random steps | Gaussian noise |
# | **jump** | replaces individuals by new ones, possibly far away | recombination |
#
# The kind decides much of what an operator *can* do, whatever its details: a reaction
# term cannot discover a point that is not already in the population; a diffusion term
# always costs on a convex objective; a jump term is the only kind that can put mass
# far from the current population at once. This tutorial shows how to read the kind of
# term off an operator, also for operators that exist only as code, and what follows from
# the kind.
#
# 1. **Local characteristics**: what an operator does to each individual: growth rate,
#    drift, diffusion (`evolution.local_characteristics`).
# 2. **Fingerprints** of canonical and home-made operators (`evolution.fingerprint`).
# 3. **Sign rules** that follow from the kind of term, checked on many populations.
# 4. **Selection on Gaussian populations** is a smoothed, preconditioned gradient step.
# 5. **Closed moment equations**: when the terms of the spread close, the shape of the
#    population at equilibrium can be computed.

# %%
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next(p for p in (HERE, *HERE.parents) if (p/"src/evoscope").is_dir())
sys.path.insert(0, str(ROOT/"src"))
FIGURES = ROOT/"docs/figures"
FIGURES.mkdir(parents=True, exist_ok=True)

from evoscope import evolution, problems, viz
from evoscope.operators import (Assembly, Drift, GaussianNoise, Jump, Observable, Recombination, Reweighting,
                                Selection, StepOperator, Transport, normalized_fd_drift, objective_rate)
from evoscope.population import Population, broad_gaussian, two_cluster
from matplotlib.patches import Ellipse

viz.paper_style()
rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Local characteristics: what an operator does to each individual
#
# Near an individual at `x`, an operator's term acts on any smooth test function through
# its value, slope and curvature at `x`:
#
# ```
# term(psi) ~ w(x) [ c(x) psi(x)  +  b(x) . grad psi(x)  +  (1/2) a(x) : Hess psi(x) ]  +  mass arriving from elsewhere
#                    growth          drift                    diffusion                     jumps
# ```
#
# `c`, `b` and `a` are the **local characteristics** of the term at `x`. For the canonical
# operators they come from the formulas (`c = -(Phi(x) - mean Phi)` for selection, `b` the
# drift, `a = sigma^2 I` for the noise). For an operator given only as code that moves each
# individual, `local_characteristics` estimates them from repeated small steps: the mean
# displacement per unit step is the drift, the mean squared displacement the diffusion.
# Displacements longer than a radius are counted as jumps, and `innovation_rate` measures
# how much mass an operator places away from the whole population.
#
# Eight operators on a two-dimensional population of 60 individuals: the four canonical
# ones, and four written as plain code:
#
# * `rank S`: selection by rank (the rate depends on the population);
# * `pull`: a move of every individual towards a weighted average that favours good
#   individuals (the new operator of Tutorial 2);
# * `H (code)`: Gaussian mutation written as a function, without telling EvoScope what it is;
# * `DE`: differential variation, `x_i + 0.5 (x_j - x_k)` for random `i, j, k`, as offspring.

# %%
quartic = Observable(lambda x: np.sum((x*x-1)**2, axis=-1), lambda x: 4*x*(x*x-1),
                     lambda x: np.sum(12*x*x-4, axis=-1))
population = Population.uniform(rng.normal(0, .9, (60, 2)))


def soft_best_pull(x, pop, alpha=3.):
    values = quartic.value(pop.points)
    w = pop.weights*np.exp(-alpha*(values-values.min()))
    return w@pop.points/w.sum()-x


def differential_offspring(pop, size, rng):
    i, j, k = (pop.sample(size, rng) for _ in range(3))
    return pop.points[i]+.5*(pop.points[j]-pop.points[k])


operators = {
    "S": Selection(objective_rate(quartic.value)),
    "D": Drift(normalized_fd_drift(quartic.value)),
    "H": GaussianNoise(.3),
    "R": Recombination(),
    "rank S": Reweighting(lambda x, pop: np.argsort(np.argsort(quartic.value(x)))/(len(x)-1), name="rank S"),
    "pull": Transport(drift=soft_best_pull, name="pull"),
    "H (code)": StepOperator(lambda pop, tau, rng: pop.points+.3*np.sqrt(tau)*rng.normal(size=pop.points.shape),
                             name="H (code)"),
    "DE": Jump(offspring=differential_offspring, name="DE"),
}

extent = (-2.4, 2.4, -2.4, 2.4)
axis = np.linspace(*extent[:2], 61)
jump_view = evolution.density_view((axis, axis), bandwidth=.15)
fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.6), layout="constrained")
for ax, (name, op) in zip(axes.ravel(), operators.items()):
    if isinstance(op, Jump):
        viz.plot_term_map(jump_view, evolution.term_rates(op, population, jump_view)[op.name], ax=ax,
                          colorbar=False, objective=quartic.value)
        ax.plot(population.points[:, 0], population.points[:, 1], ".", ms=2.5, color="0.15")
        ax.set_title(f"{name}: where mass goes")
    else:
        chars = evolution.local_characteristics(op, population, rng=rng)
        viz.plot_characteristics(chars, ax=ax, objective=quartic.value, extent=extent)
        ax.set_title(f"{name} ({chars['method']})")
    ax.set(xlim=extent[:2], ylim=extent[2:], xticks=[], yticks=[])
fig.savefig(FIGURES/"t9_portraits.png", dpi=150)

# %% [markdown]
# ![Portraits of eight operators](../docs/figures/t9_portraits.png)
#
# Dots are individuals, coloured by their growth rate (green: their weight grows; purple:
# it shrinks); arrows show the drift and shaded discs the diffusion. For the two jump
# operators the panel shows their term as a map instead: red where offspring arrive,
# blue where individuals are replaced.
#
# * **Selection** and **rank selection** are pure reaction: colour only, no arrows, no
#   discs. Rank selection spreads the growth rates evenly over the population, whatever
#   the objective values.
# * **Drift** and **pull** are pure transport: arrows only. The drift points every
#   individual downhill towards its nearest minimum; the pull points all of them towards
#   one place, the soft best point of the population. The portraits show the difference
#   between a local and a population-level move at a glance.
# * **Noise** and **H (code)** are pure diffusion: equal discs everywhere. The operator
#   written as a function is recognized from its steps alone.
# * **Recombination** places its offspring inside the population (midpoints lie in the
#   convex hull of the parents); **differential variation** places them also outside it.
#
# ## 2. Fingerprints
#
# `fingerprint` condenses the local characteristics into the size of each kind of term
# (rates per unit time: the spread of the growth rates, the root-mean-square drift and the
# diffusion relative to the size of the population, and the innovation rate), adds the
# effect of the operator on the mean, the spread and the mean gap, and names the kinds of
# term it finds. A ninth operator is added: a **Levy flight**, a mutation with
# heavy-tailed (Cauchy) steps.

# %%
operators["flight"] = StepOperator(lambda pop, tau, rng: pop.points+.3*tau*rng.standard_cauchy(pop.points.shape),
                                   name="flight")
prints = evolution.fingerprint(list(operators.values()), population, objective=quartic, rng=rng)
print(evolution.format_fingerprint(prints))

# %% [markdown]
# Every canonical operator is recognized as the kind it is, and so is every operator
# written as code. The Levy flight is a mutation, but its term is a **jump**, not a
# diffusion: a Cauchy step is large with a probability proportional to the step size, so
# mass appears far from the population at a positive rate (its `jump` entry). It also has
# no finite spread rate: the large steps make the estimate of the spread rate huge and
# unstable. This is the kind of operator for which the moment-stability check (A2) of
# Tutorial 2 fails.
#
# The effects tell the roles: selection, drift and pull lower the mean gap and the noise
# raises it. Of the two jump operators, recombination lowers the mean gap on this broad
# population (midpoints of a broad cloud are better than their parents) and differential
# variation raises it (its offspring land outside the population, where `f` is large).
# Recombination and the noise leave the mean where it is (`mean shift` 0, and differential
# variation does so on average); selection, drift and pull move it.
#
# ## 3. Sign rules that follow from the kind of term
#
# Because a term's kind fixes its form, some of its effects have a definite sign on *every*
# population:
#
# * **reaction** with a rate that increases with `f` (selection by value or by rank) never
#   lowers the share of the population below any objective level, and so never raises
#   the mean gap: it moves weight towards the better individuals and nothing else;
# * **jump** to midpoints (or any blend `(1 - xi) x + xi y` with `0 <= xi <= 1`) keeps the
#   mean, and never raises the mean of a **convex** function: the child is never worse than
#   the average of its parents (Jensen's inequality). On a non-convex objective it can;
# * **diffusion** never lowers the mean of a convex function;
# * **transport** along a descent direction never raises the mean gap.
#
# The check below runs each rule on 120 populations of four kinds (broad, clustered,
# small, shifted) in four dimensions.

# %%
f4 = problems.get("quartic_d4")                     # non-convex, 16 minima
convex = Observable(lambda x: .5*np.sum(np.arange(1, 5)*x*x, axis=-1), lambda x: np.arange(1, 5)*x,
                    lambda x: np.full(np.shape(x)[:-1], 5.))
populations = []
for k in range(30):
    populations += [broad_gaussian(40, 4, rng), two_cluster(40, 4, rng),
                    Population.uniform(rng.normal(0, .2, (40, 4))), Population.uniform(rng.normal(1.5, .5, (40, 4)))]
levels = evolution.level_view(f4, np.geomspace(1e-2, 1e2, 25))
S4 = Selection(objective_rate(f4.value))
rank4 = Reweighting(lambda x, pop: np.argsort(np.argsort(f4.value(x)))/(len(x)-1), name="rank S")
R4, H4, D4 = Recombination(), GaussianNoise(.3), Drift(normalized_fd_drift(f4.value))
gap = evolution.View(lambda x: (f4.value(x))[:, None], ["mean gap"], lambda x: f4.gradient(x)[:, None, :],
                     lambda x: f4.laplacian(x)[:, None])
convex_view = evolution.View(lambda x: convex.value(x)[:, None], ["convex"], lambda x: convex.gradient(x)[:, None, :],
                             lambda x: convex.laplacian(x)[:, None])
mean = evolution.mean_view(4)
checks = {
    "S: share below every level never falls": lambda p: evolution.term_rates(S4, p, levels)["S"].min() >= -1e-12,
    "rank S: share below every level never falls": lambda p: evolution.term_rates(rank4, p, levels)["rank S"].min() >= -1e-12,
    "R: mean unchanged": lambda p: np.abs(evolution.term_rates(R4, p, mean)["R"]).max() < 1e-10,
    "R: mean of a convex function never rises": lambda p: evolution.term_rates(R4, p, convex_view)["R"][0] <= 1e-12,
    "R: mean gap of the quartic never rises": lambda p: evolution.term_rates(R4, p, gap)["R"][0] <= 1e-12,
    "H: mean of a convex function never falls": lambda p: evolution.term_rates(H4, p, convex_view)["H"][0] >= 0,
    "D: mean gap never rises": lambda p: evolution.term_rates(D4, p, gap)["D"][0] <= 0,
}
for claim, check in checks.items():
    holds = sum(bool(check(p)) for p in populations)
    print(f"{claim:48s} holds on {holds:3d} of {len(populations)} populations")

# %% [markdown]
# Every rule holds on all populations except the one that is *not* a rule: recombination
# raises the mean gap of the multimodal quartic on half of the populations. They are the
# clustered ones, whose midpoints fall between minima, and the small ones around the
# origin, where the quartic has a local maximum and is concave. Convexity is what makes the recombination rule
# work, and on a non-convex problem recombination has to earn its place population by
# population, which is what the attribution of Tutorial 1 and the ledgers of Tutorial 8
# measure.
#
# ## 4. Selection on a Gaussian population is a smoothed gradient step
#
# Selection creates no new points, but it moves the *mean* of the population, by
# reweighting. On a Gaussian population with mean `m` and covariance `Sigma`, Stein's
# lemma turns its terms into derivatives:
#
# ```
# selection's term for the mean        =  - omega Sigma E[grad Phi(X)]
# selection's term for the covariance  =  - omega Sigma E[Hess Phi(X)] Sigma
# ```
#
# The mean moves along the gradient of the selection rate, averaged over the population
# and **preconditioned by the covariance**, as in natural-gradient evolution strategies.
# The check below compares selection's term for the mean (computed from 20 000 individuals
# by `term_rates`) with the right-hand side, on Gaussian populations of different shapes
# on a rotated quadratic, and on a population that is not Gaussian.

# %%
Q = np.array([[np.cos(.5), -np.sin(.5)], [np.sin(.5), np.cos(.5)]])
A = Q@np.diag([1., 6.])@Q.T
quadratic = Observable(lambda x: .5*np.einsum("...i,ij,...j->...", x, A, x), lambda x: x@A,
                       lambda x: np.full(np.shape(x)[:-1], np.trace(A)))
S2 = Selection(objective_rate(quadratic.value))


def grad_phi(x):
    v = quadratic.value(x)
    return quadratic.gradient(x)/(1+v)[:, None]**2


cases = [((1.5, 1.), np.diag([.15, .15])), ((-1.2, 1.4), np.diag([.6, .05])), ((1.4, -1.2), np.array([[.1, .08], [.08, .3]])),
         ((-1.5, -1.), np.diag([.05, .5]))]
fig, ax = plt.subplots(figsize=(5, 4.6), layout="constrained")
scale = 18.                                          # arrow length per unit rate
gx = np.linspace(-2.6, 2.6, 160)
X, Y = np.meshgrid(gx, gx)
ax.contour(gx, gx, quadratic.value(np.stack([X, Y], -1)), levels=12, colors="0.8", linewidths=.6)
for centre, cov in cases:
    pop = Population.uniform(rng.multivariate_normal(centre, cov, 20_000))
    term = evolution.term_rates(S2, pop, evolution.mean_view(2))["S"]
    sigma = np.cov(pop.points.T, bias=True)
    stein = -sigma@(pop.weights@grad_phi(pop.points))
    plain = -(pop.weights@grad_phi(pop.points))
    print(f"mean {centre}: selection's term {np.round(term, 3)}  Stein  {np.round(stein, 3)}"
          f"  (plain gradient direction {np.round(plain/np.linalg.norm(plain), 2)})")
    values, vectors = np.linalg.eigh(cov)
    ax.add_patch(Ellipse(centre, 4*np.sqrt(values[1]), 4*np.sqrt(values[0]),
                         angle=np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1])), fc=viz.LIGHT, alpha=.35, ec="none"))
    ax.annotate("", xy=np.add(centre, scale*term), xytext=centre,
                arrowprops=dict(arrowstyle="-|>", color=viz.OPERATOR_COLORS["S"], lw=2))
    ax.annotate("", xy=np.add(centre, scale*stein), xytext=centre,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=.9, ls="--"))
    ax.annotate("", xy=np.add(centre, .45*plain/np.linalg.norm(plain)), xytext=centre,
                arrowprops=dict(arrowstyle="-", color=viz.ALERT, lw=1, ls=":"))
clusters = Population.uniform(np.concatenate([rng.normal([-1.8, .4], .05, (10_000, 2)), rng.normal([.4, -.6], .05, (10_000, 2))]))
term = evolution.term_rates(S2, clusters, evolution.mean_view(2))["S"]
stein = -np.cov(clusters.points.T, bias=True)@(clusters.weights@grad_phi(clusters.points))
print(f"two clusters: selection's term {np.round(term, 3)}  Stein formula {np.round(stein, 3)}")
ax.plot([], [], color=viz.OPERATOR_COLORS["S"], lw=2, label="selection's term (particles)")
ax.plot([], [], color="k", lw=.9, ls="--", label=r"$-\omega\Sigma\,E[\nabla\Phi]$ (Stein)")
ax.plot([], [], color=viz.ALERT, lw=1, ls=":", label=r"direction of $-E[\nabla\Phi]$")
ax.set(xlim=(-2.6, 2.6), ylim=(-2.6, 2.6), aspect="equal", xticks=[], yticks=[])
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.savefig(FIGURES/"t9_selection_gaussian.png", dpi=150)

# %% [markdown]
# ![Selection moves the mean of a Gaussian population along the preconditioned gradient](../docs/figures/t9_selection_gaussian.png)
#
# On every Gaussian population (shaded: two standard deviations) selection's term for the
# mean (green arrow) coincides with the Stein formula (dashed). It is *not* the plain
# averaged gradient direction (dotted): an elongated population moves mostly along its long
# axis, because selection can only shift weight among the individuals it has. On a
# population of two clusters (printed last) the formula fails, and even points the wrong
# way: selection moves the mean towards the better cluster, whereas the formula, which
# combines the covariance of the whole population with the local slopes inside the
# clusters, points elsewhere.
#
# The covariance term explains how selection shapes a population: it contracts the
# covariance by `Sigma E[Hess Phi] Sigma`, which is quadratic in the spread. A narrow
# population is hardly shaped by selection at all, as Tutorial 8 found.
#
# ## 5. Closed moment equations
#
# For some assemblies the terms of the covariance *close*: they depend on the population
# only through its covariance `Sigma`, which then obeys an ordinary differential equation.
# With a linear drift `b(x) = -kappa A x` on the quadratic `f = x'Ax/2`, Gaussian noise and
# midpoint recombination, the three terms are exactly
#
# ```
# drift: -kappa (A Sigma + Sigma A)     noise: + sigma^2 I     recombination: -(gamma/2) Sigma
# ```
#
# for every population, Gaussian or not. Along an eigen-direction of `A` with curvature
# `a`, the variance settles at `sigma^2/(2 kappa a + gamma/2)`: one source balanced by two
# sinks. Selection adds the term of Section 4, `-omega Sigma E[Hess Phi] Sigma`, which is
# exact for Gaussian populations; using it for every population (a *Gaussian closure*)
# gives an approximate equation.

# %%
kappa, noise, gamma, omega = 1., .6, 1., 4.
D2 = Drift(lambda x: -kappa*x@A, name="D")
base = Assembly([Recombination(rate=gamma), D2, GaussianNoise(noise)])
with_selection = Assembly([Selection(objective_rate(quadratic.value), strength=omega), Recombination(rate=gamma), D2,
                           GaussianNoise(noise)])
curvatures, directions = np.linalg.eigh(A)
normal = rng.normal(size=(4000, 2))


def mean_hessian_of_phi(cov):
    """E[Hess Phi(X)] for X ~ N(0, cov), Phi = f/(1 + f), by Monte Carlo."""
    x = normal@np.linalg.cholesky(cov).T
    v, g = quadratic.value(x), quadratic.gradient(x)
    return A*np.mean(1/(1+v)**2)-2*np.einsum("ni,nj,n->ij", g, g, 1/(1+v)**3)/len(x)


def closed_equation(cov, times, selection, substeps=20):
    out = [cov]
    for dt in np.diff(times):
        for _ in range(substeps):
            rate = -kappa*(A@cov+cov@A)+noise**2*np.eye(2)-gamma/2*cov
            if selection:
                rate = rate-selection*cov@mean_hessian_of_phi(cov)@cov
            cov = cov+dt/substeps*rate
        out.append(cov)
    return np.array(out)


start = Population.uniform(rng.normal(0, 1., (20_000, 2)))
times = np.linspace(0, 3., 61)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), layout="constrained", sharey=True)
for ax, (label, assembly, selection) in zip(axes, (("R, D, H", base, 0.), ("S, R, D, H", with_selection, omega))):
    fl = evolution.flow(assembly, start, .005, times[-1], rng, record=times)
    particles = fl.statistic(lambda p: [u@np.cov(p.points.T, aweights=p.weights, bias=True)@u for u in directions.T])
    closed = closed_equation(np.cov(start.points.T, bias=True), times, selection)
    for k, (a, color) in enumerate(zip(curvatures, (viz.PALETTE[0], viz.PALETTE[1]))):
        u = directions[:, k]
        prediction = np.einsum("i,tij,j->t", u, closed, u)
        ax.semilogy(fl.times, particles[:, k], "o", ms=2.5, color=color, mfc="white",
                    label=f"curvature {a:.0f}: particles")
        ax.semilogy(times, prediction, color=color, lw=1.4, label="closed equation")
        if not selection:
            ax.axhline(noise**2/(2*kappa*a+gamma/2), color=color, lw=.7, ls=":")
        print(f"{label}, curvature {a:.0f}: variance at t = 3 from the closed equation {prediction[-1]:.4f}, "
              f"from the particles {particles[-1, k]:.4f}")
    ax.set(title=label, xlabel="time t")
axes[0].set_ylabel("variance along eigen-direction of A")
axes[1].legend(frameon=False, fontsize=8)
fig.savefig(FIGURES/"t9_closure.png", dpi=150)

# %% [markdown]
# ![Variances along the eigen-directions: particles and the closed equation](../docs/figures/t9_closure.png)
#
# Without selection (left) the closed equation is exact, and the particles follow it to
# the equilibrium `sigma^2/(2 kappa a + gamma/2)` (dotted) along both directions. With
# selection (right) the Gaussian closure predicts that selection lowers the equilibrium
# variance by about a tenth along the flat direction and by a few percent along the
# steep one, and the particles agree with it to within about one percent. The
# equilibrium is where the terms balance, so it can be computed from the terms without
# running the algorithm; the closure is approximate only through the Gaussian form of the
# selection term.
#
# **Take-away.** The kind of term an operator contributes can be read off its local
# characteristics, also for operators given only as code, and the kind already decides
# much of what the operator can do: which quantities it keeps, which signs its effects
# have on every population, and how it enters the equations for the moments. The
# reasons behind these rules are collected in [the evolution equation, term by
# term](../docs/guide-evolution.md).
