# %% [markdown]
# # Tutorial 6: Gradient methods as special cases
#
# Gradient descent is not usually called a population method, but it fits the same
# framework: a population of one individual (or of many independent starts) and a single
# operator, a move along the negative gradient. Seeing it this way has two uses.
#
# * Familiar results about gradient descent reappear as special cases of what EvoScope
#   computes for any algorithm. This makes the output for population methods easier to
#   read: the **drift coefficient** of gradient descent is the constant of the
#   Polyak-Lojasiewicz inequality, the **moment-stability check** plays the role of the
#   step-size condition, and the **residual form** gives the noise floor of noisy gradient
#   descent.
# * Gradient steps can be combined with population operators (selection, noise,
#   recombination), and the combination analysed with the same tools.
#
# The tutorial has four parts: gradient descent as an operator (1), its drift coefficient
# (2), noisy gradient descent (3), and gradient steps inside a population method (4).

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

from evoscope import attribution, checks, problems, viz
from evoscope.operators import Assembly, Drift, GaussianNoise, Observable, Selection, objective_rate
from evoscope.population import Population, broad_gaussian, two_cluster

viz.paper_style()
rng = np.random.default_rng(0)
fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), layout="constrained")

# %% [markdown]
# ## 1. Gradient descent is a move operator
#
# Take the quadratic `f(x) = 0.5 * sum_j a_j x_j^2` with curvatures `a = (1, 2, 4, 8)`:
# smallest curvature `mu = 1`, largest `L = 8`, minimum value `f* = 0`. A user-defined
# objective is an `Observable` (value, gradient, and optionally the Laplacian and the
# Gaussian expectation, which make the computations below exact).
#
# Gradient descent is the drift `b(x) = -grad f(x)`. A `Drift` operator moves every
# individual by `tau * b(x)`, so with step size `tau` equal to the learning rate `eta`, a
# population of one individual *is* gradient descent:

# %%
a = np.array([1., 2., 4., 8.])
mu, L, d = a.min(), a.max(), len(a)
quadratic = Observable(value=lambda x: .5*np.sum(a*x*x, axis=-1),
                       gradient=lambda x: a*x,
                       laplacian=lambda x: np.full(np.shape(x)[:-1], a.sum()),
                       gaussian_mean=lambda m, v: .5*np.sum(a*m*m, axis=-1)+.5*v*a.sum())
GD = Drift(lambda x: -quadratic.gradient(x), name="gradient step")

eta = .05
x = np.array([[2., -1., .5, .3]])
population = Population.uniform(x.copy())
for k in range(10):
    x = x-eta*quadratic.gradient(x)                 # textbook gradient descent
    population = GD.step(population, eta)           # the operator, applied to a population of one
print("population of one individual = gradient descent:", np.allclose(population.points, x))

# %% [markdown]
# With `N` individuals the same operator runs `N` independent starts (multi-start gradient
# descent), each with its own weight.
#
# **The four checks.** On the quadratic, the gradient step passes all four. On the
# quartic `sum_j (x_j^2 - 1)^2` it fails **moment stability (A2)**: the gradient grows like
# `x^3`, so a population far from the origin is thrown out further. This is the population
# version of the classical step-size condition: a fixed learning rate is stable only where
# the curvature is below about `2/eta`, and the curvature of the quartic grows without
# bound. Clipping the gradient (dividing it by `sqrt(1 + |grad f|^2)`) fixes it.

# %%
test_populations = [broad_gaussian(16, d, rng), two_cluster(16, d, rng)]
print("gradient step on the quadratic:", checks.check_operator(GD, test_populations, fast=True).verdict)

quartic = problems.get("quartic_d4")
plain = Drift(lambda x: -quartic.gradient(x), name="plain")


def clipped_gradient(problem):
    def drift(x):
        g = problem.gradient(x)
        return -g/np.sqrt(1+np.sum(g*g, axis=-1, keepdims=True))
    return drift


clipped = Drift(clipped_gradient(quartic), name="clipped")
for rule in (plain, clipped):
    finding = checks.check_moment_stability(rule, test_populations)
    print(f"{rule.name:8s} gradient step on the quartic, (A2): {finding.verdict} - {finding.statistic}")

# %% [markdown]
# The moment ratio of the plain step reaches about `10^8` for populations scaled by 16;
# the clipped step keeps it below one. The downhill move `D` of Tutorials 1 and 3 is such
# a clipped gradient step, with the gradient replaced by finite differences
# (`normalized_fd_drift`): a derivative-free gradient method.
#
# ## 2. The drift coefficient is the Polyak-Lojasiewicz constant
#
# The effect of the gradient step on the mean gap `V = E[f(X) - f*]` is
# `G = E[grad f . b] = -E|grad f(X)|^2`, so its **rate** (effect divided by the mean
# gap) is
#
#     rate = E|grad f(X)|^2 / E[f(X) - f*].
#
# If `f` satisfies the **Polyak-Lojasiewicz (PL) inequality**
# `|grad f(x)|^2 >= 2 mu (f(x) - f*)` at every point, the rate is at least `2 mu` at
# every population. The smallest rate over populations is the drift coefficient
# `lambda`, so `lambda >= 2 mu`, and the convergence theorem gives
# `V(t) <= exp(-2 mu t) V(0)`: the classical linear convergence of gradient descent under
# the PL inequality, here for small steps. (For a finite learning rate `eta <= 1/L`, the
# classical per-step factor is `1 - 2 mu eta (1 - L eta / 2)`, which tends to
# `exp(-2 mu eta)` as `eta` shrinks.)
#
# EvoScope estimates the rate on populations without knowing `mu`. Broad and clustered
# populations give large rates; populations stretched along the flattest direction (the
# first coordinate, curvature 1) come close to `2 mu = 2`.

# %%
descent = Assembly([GD])
kinds = {"broad": [broad_gaussian(32, d, rng) for _ in range(64)],
         "two-cluster": [two_cluster(32, d, rng) for _ in range(64)],
         "stretched\nalong x1": [Population.uniform(rng.normal(size=(32, d))*np.array([2., .3, .2, .1]))
                                 for _ in range(64)]}
for j, (kind, pops) in enumerate(kinds.items()):
    rates = attribution.coefficients(descent, pops, quadratic)["total"]
    print(f"{kind.replace(chr(10), ' '):20s} smallest rate {rates.min():6.3f}   median {np.median(rates):6.3f}")
    jitter = np.random.default_rng(j).uniform(-.15, .15, len(rates))
    axes[0, 0].scatter(j+jitter, rates, s=7, color=viz.PALETTE[0], alpha=.8)
axes[0, 0].axhline(2*mu, color=viz.PALETTE[3], ls="--", lw=1)
axes[0, 0].text(-.4, 2*mu-.3, "PL bound 2 mu", ha="left", va="top", color=viz.PALETTE[3], fontsize=8)
axes[0, 0].set(ylim=(0, 14), xticks=range(3), xticklabels=list(kinds), ylabel="rate of the gradient step",
               title="(a) gradient descent on a quadratic")

# %% [markdown]
# The worst case is the population that sits along the flattest direction, where gradient
# descent is slowest. That is what the drift coefficient measures for any algorithm: the
# rate at the populations where it is slowest.
#
# The PL inequality does not require convexity. The separable function
# `sum_j (x_j^2 + 1.05 sin^2 x_j)` of the test panel is nonconvex but satisfies a global PL
# inequality, and its estimated drift coefficient is positive. On the multimodal quartic
# the rate tends to zero as the population concentrates at the local maximum (the origin)
# or at a saddle point such as `(1, 1, 1, 0)`, where the gradient vanishes but `f > f*`.
# There is no positive drift coefficient: gradient descent can stop at critical points
# that are not minimizers. In general, a drift coefficient that is not positive points
# to populations at which the algorithm makes no progress.

# %%
pl = problems.get("separable_pl_d4")
pl_populations = ([broad_gaussian(32, d, rng) for _ in range(64)]+[two_cluster(32, d, rng) for _ in range(64)]
                  +[Population.uniform(s*rng.normal(size=(32, d))) for s in (.05, .3, 3.) for _ in range(20)])
rates = attribution.coefficients(Assembly([Drift(lambda x: -pl.gradient(x))]), pl_populations, pl)["total"]
print("nonconvex PL function: estimated drift coefficient", round(float(np.nanmin(rates)), 3))

for name, center in (("local maximum", np.zeros(d)), ("saddle point", np.array([1., 1., 1., 0.]))):
    for spread in (.1, .03, .01):
        pops = [Population.uniform(center+spread*rng.normal(size=(32, d))) for _ in range(8)]
        rates = attribution.coefficients(Assembly([clipped]), pops, quartic)["total"]   # clipped: plain fails (A2)
        print(f"quartic, population of spread {spread:4.2f} at the {name}: largest rate {rates.max():.4f}")

# %% [markdown]
# ## 3. Noisy gradient descent: decay to a floor
#
# Adding Gaussian noise to every step, `x -> x - tau grad f(x) + sigma sqrt(tau) Z`, gives
# the Langevin algorithm (perturbed gradient descent). In EvoScope it is an assembly of the
# gradient step and a noise operator, and the effects add: the gradient step contributes
# `-E|grad f|^2`, the noise `+(sigma^2/2) E[Lap f]`, which is positive. Near the optimum
# the noise wins, so the rate is not positive everywhere. Instead the effect satisfies a
# **residual form** `G <= -lambda V + c`, which gives
#
#     V(t) <= exp(-lambda t) V(0) + (c/lambda) (1 - exp(-lambda t)):
#
# linear convergence down to a floor `c/lambda`. For the quadratic, `lambda = 2 mu` and
# `c = sigma^2 tr(A)/2` by hand; EvoScope estimates `c` from populations.

# %%
sigma = .5
langevin = Assembly([GD, GaussianNoise(sigma, name="noise")])
print("composition (effects add):", checks.check_composition(langevin, test_populations).verdict)
noisy_populations = (kinds["broad"]+kinds["two-cluster"]
                     +[Population.uniform(s*rng.normal(size=(32, d))) for s in (.02, .1, .3, 1., 3.) for _ in range(20)])
result = attribution.coefficients(langevin, noisy_populations, quadratic)
lam = 2*mu
c = attribution.residual_constant(result["generator"], result["V"], lam)
print(f"residual constant: estimated {c:.3f}, by hand sigma^2 tr(A)/2 = {sigma**2*a.sum()/2:.3f}; floor c/lambda = {c/lam:.3f}")

tau, population = .01, broad_gaussian(512, d, rng)
gaps = [population.expect(quadratic.value(population.points))]
for k in range(1500):
    population = langevin.step(population, tau, rng)      # 512 independent noisy gradient-descent chains
    gaps.append(population.expect(quadratic.value(population.points)))
gaps, t = np.array(gaps), tau*np.arange(len(gaps))
bound = np.exp(-lam*t)*gaps[0]+c/lam*(1-np.exp(-lam*t))
print(f"mean gap over the last 300 steps {gaps[-300:].mean():.3f}; stationary value sigma^2 d/4 = {sigma**2*d/4:.3f};"
      f" bound violated: {bool(np.any(gaps > bound))}")
axes[0, 1].plot(t, gaps, color=viz.PALETTE[0], lw=1.2, label="mean gap, 512 chains")
axes[0, 1].plot(t, bound, color=viz.PALETTE[1], lw=1.2, ls="--", label="residual-form bound")
axes[0, 1].axhline(sigma**2*d/4, color="0.5", lw=.8, ls=":", label="stationary value")
axes[0, 1].set(yscale="log", xlabel="time t = k tau", ylabel="mean gap", title="(b) noisy gradient descent")
axes[0, 1].legend(frameon=False, fontsize=8)

# %% [markdown]
# The bound holds along the whole run. The chains settle near the stationary value
# `sigma^2 d/4 = 0.25` of the Langevin dynamics, below the guaranteed floor of about 0.93:
# the residual form uses only the worst direction, so it is conservative when the
# curvatures differ (it is exact when all `a_j` are equal).
#
# **Stochastic gradient descent.** With learning rate `eta` and gradient noise of standard
# deviation `s` per coordinate, the step `x -> x - eta (grad f(x) + s xi)` is this
# operator with `tau = eta` and `sigma = s sqrt(eta)`, in the common Gaussian
# approximation of the gradient noise. The floor `c/lambda = eta s^2 tr(A)/(4 mu)` then
# shrinks in proportion to the learning rate: the usual reason for decreasing learning
# rates in SGD.
#
# ## 4. Gradient steps inside a population method
#
# On the unequal wells problem (a good well at `x1 = 1` with value 0, and a worse one at
# `x1 = -1`), start 32 individuals from a broad population and compare
#
# * **multi-start gradient descent**: clipped gradient steps only;
# * **gradient descent with selection**: the same steps after a reweighting by
#   `f/(1+f)`, which moves weight to individuals with lower objective values.
#
# Both rules are exact on populations (`Assembly.step` returns the exact new law), so the
# runs below follow the population law itself.

# %%
wells = problems.get("wells_d4")
G = Drift(clipped_gradient(wells), name="gradient step")
S = Selection(objective_rate(wells.value), name="selection")
variants = {"gradient steps only": Assembly([G]), "with selection": Assembly([S, G])}
tau, generations = .05, 400
for color, (name, assembly) in zip(viz.PALETTE, variants.items()):
    population = broad_gaussian(32, d, np.random.default_rng(1))
    gaps, rates = [], []
    for k in range(generations+1):
        gap = population.expect(wells.value(population.points))
        gaps.append(gap)
        rates.append({n: -v/gap for n, v in assembly.generator_actions(population, wells).items()})
        population = assembly.step(population, tau)
    worse = np.sum(population.weights[population.points[:, 0] < 0])
    print(f"{name:20s} mean gap at t = 2, 10, 20: {gaps[40]:.4f}, {gaps[200]:.4f}, {gaps[-1]:.2e};"
          f" weight in the worse well {worse:.3f}")
    t = tau*np.arange(generations+1)
    axes[1, 0].plot(t, gaps, color=color, lw=1.4, label=name)
    if name == "with selection":
        for n, line_color in (("gradient step", viz.PALETTE[2]), ("selection", viz.PALETTE[4])):
            axes[1, 1].plot(t, [r[n] for r in rates], color=line_color, lw=1.4, label=n)
axes[1, 0].set(yscale="log", ylim=(3e-5, 30), xlabel="time t = k tau", ylabel="mean gap",
               title="(c) multi-start gradient descent on the wells")
axes[1, 0].legend(frameon=False, fontsize=8)
axes[1, 1].set(xlabel="time t = k tau", ylabel="rate", ylim=(-.05, None),
               title="(d) with selection: rate of each operator")
axes[1, 1].legend(frameon=False, fontsize=8)
fig.savefig(FIGURES/"t6_gradient_methods.png", dpi=150)

# %% [markdown]
# ![Gradient methods](../docs/figures/t6_gradient_methods.png)
#
# (a) The rate of gradient descent on populations of three kinds; the smallest values
# approach the PL bound `2 mu`. (b) Noisy gradient descent decays to a floor, below the
# residual-form bound. (c) With gradient steps only, 7 of the 32 starts end in the worse
# well and the mean gap stops decreasing; with selection it goes to zero. (d) The rates of
# the two operators of the second variant: the gradient step does the work at first, then
# its rate falls to zero as the individuals reach the bottoms of their wells, and selection
# takes over, moving the weight from the worse well to the good one.
#
# **Take-away.** Gradient descent is the one-operator special case: its drift coefficient
# is the PL constant, moment stability plays the role of its step-size condition, and its noisy version
# decays to a floor described by the residual form. In a population method, gradient steps
# provide local descent and selection chooses between the basins they find. If no
# individual starts in the good basin, neither helps: exploration (noise, recombination)
# is needed, and the discovery guarantee of Tutorials 3 and 5 measures whether it happens.
#
# *References for the classical results:* B. T. Polyak, Gradient methods for minimizing
# functionals (1963); H. Karimi, J. Nutini and M. Schmidt, Linear convergence of gradient
# and proximal-gradient methods under the Polyak-Lojasiewicz condition (2016).
