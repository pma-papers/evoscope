# %% [markdown]
# # Tutorial 7: The evolution equation, term by term
#
# The earlier tutorials looked at *one step* of an algorithm: which operator changes the
# mean gap by how much. This tutorial looks at the **whole evolution** of the population
# that the operators drive, in a setting small enough to see everything: one and two
# dimensions.
#
# When the step size is small, an assembly of operators moves its population along the
# solution of an **evolution equation**. For the four canonical operators it reads
#
# ```
# d/dt rho  =  -(b rho)'  +  (sigma^2/2) rho''  -  omega (Phi - mean Phi) rho  +  gamma (r[rho] - rho)
#               drift D       noise H              selection S                  recombination R
# ```
#
# where `rho` is the density of the population (in one dimension), `b` the drift,
# `sigma` the noise level, `Phi` the selection rate, and `r[rho]` the density of the
# midpoints of two random parents. **Each operator contributes one term**, and the terms
# are of four different kinds:
#
# | operator | kind of term | what it does to the density |
# |---|---|---|
# | drift D | transport (first derivative) | shifts mass continuously along `b` |
# | noise H | diffusion (second derivative) | smooths: removes mass from peaks, adds it to the flanks |
# | selection S | reaction (no derivative) | multiplies the density by a rate: grows where `Phi` is below average, shrinks elsewhere; creates no new points |
# | recombination R | jump (non-local) | removes mass everywhere at rate `gamma` and puts it back at midpoints, possibly far away |
#
# The paper writes the equation in a weak form, `d/dt <phi, mu_t> = sum_j G_j[mu_t](phi)` for
# all test functions `phi`, which needs no density and holds in any dimension; the
# generator actions `G_j` of the previous tutorials are exactly the terms seen through a
# test function. Here we solve the equation, look at its terms, and check that the
# particle algorithm follows it.

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

from evoscope import evolution, viz
from evoscope.operators import (Assembly, Drift, GaussianNoise, Observable, Recombination, Selection,
                                normalized_fd_drift, objective_rate)
from evoscope.population import Population

viz.paper_style()
rng = np.random.default_rng(0)
colors = viz.operator_colors("SRDH")

# %% [markdown]
# ## 1. A one-dimensional problem with a good and a worse well
#
# `f(x) = (x^2 - 1)^2 + 0.2 (x - 1)^2` has its global minimum at `x = 1` (`f = 0`) and a
# worse local minimum near `x = -1` (`f` about 0.8), separated by a barrier near `x = 0`.
# The population starts in the worse well: a Gaussian with mean -0.9 and standard
# deviation 0.5.
#
# The four operators are those of Tutorial 1 with stronger selection (`omega = 2`) and
# recombination at half the rate (`gamma = 0.5`).

# %%
f = Observable(lambda x: (x[..., 0]**2-1)**2+.2*(x[..., 0]-1)**2,
               lambda x: 4*x*(x*x-1)+.4*(x-1),
               lambda x: 12*x[..., 0]**2-3.6)
S = Selection(objective_rate(f.value), strength=2.)
R = Recombination(rate=.5)
D = Drift(normalized_fd_drift(f.value))
H = GaussianNoise(.25)
assembly = Assembly([S, R, D, H])

grid = np.linspace(-2.5, 2.5, 501)
rho0 = np.exp(-(grid+.9)**2/(2*.5**2))

# %% [markdown]
# ## 2. Solving the equation
#
# `evolution.TRJ1D` solves the equation of an assembly on a grid in one dimension,
# keeping every term separate. It reads the drift, noise, selection rate and
# recombination rule from the operators themselves, so the equation solved is exactly the
# one the assembly defines.
#
# `evolution.flow` follows the same equation with particles instead: it applies the
# operators to 20 000 weighted particles with a small step `tau = 0.01`. Its cost grows
# linearly with the number of particles, and it works in any dimension.

# %%
solver = evolution.TRJ1D(assembly, grid)
times = np.linspace(0, 8, 81)
solution = solver.solve(rho0, 8., record=times)
particles = evolution.flow(assembly, Population.uniform(rng.normal(-.9, .5, (20_000, 1))), .01, 8., rng,
                           record=times)
good = grid > 0
mass_good = solution["densities"][:, good].sum(axis=1)*solver.h
mass_good_particles = particles.statistic(lambda p: p.weights@(p.points[:, 0] > 0))
for t in (0, 2, 4, 8):
    k = int(np.argmin(np.abs(times-t)))
    print(f"t = {t}: mass in the good well {mass_good[k]:.3f} (equation)  {mass_good_particles[k]:.3f} (particles)")

# %% [markdown]
# The population moves over the barrier slowly: the mass in the good well rises from 4% to
# 98% by `t = 8`. The particles agree with the grid solution to within the sampling error.
#
# A finite-population algorithm is a different matter. It keeps `N` individuals with
# equal weights and resamples them in every generation. `evolution.flow` does the same with
# `size=N` and `resample=1` (resample whenever the weights are unequal). With a small
# population the run is random: whether and when the population crosses the barrier
# depends on a few lucky individuals. The equation describes the limit of large
# populations, around which the runs scatter.

# %%
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), layout="constrained",
                         gridspec_kw={"width_ratios": [1.25, 1, 1]})
viz.plot_density_flow(solution["times"], grid, solution["densities"], ax=axes[0])
for y in (-1, 1):
    axes[0].axhline(y, color="w", lw=.6, ls=":")
axes[0].set_title("(a) solution of the equation")

ax = axes[1]
shades = plt.cm.Greys(np.linspace(.35, .95, 4))
for t, shade in zip((0, 2, 4, 8), shades):
    k = int(np.argmin(np.abs(times-t)))
    pop = particles.populations[k]
    ax.hist(pop.points[:, 0], bins=np.linspace(-2.5, 2.5, 81), weights=pop.weights, density=True,
            color=shade, alpha=.35)
    ax.plot(grid, solution["densities"][k], color=shade, lw=1.4, label=f"t = {t}")
twin = ax.twinx()
twin.plot(grid, f.value(grid[:, None]), color=viz.PALETTE[3], lw=.8, ls="--")
twin.set_ylim(0, 6)
twin.set_ylabel("f(x)", color=viz.PALETTE[3])
ax.set(xlabel="x", ylabel="density", title="(b) equation (lines) and particles")
ax.legend(frameon=False, fontsize=8)

ax = axes[2]
for N, shade in ((100, "#b9d3e6"), (1000, viz.LIGHT), (10_000, viz.PALETTE[0])):
    final = []
    for run in range(6):
        start = Population.uniform(rng.normal(-.9, .5, (N, 1)))
        run_flow = evolution.flow(assembly, start, .02, 8., rng, resample=1., record=times)
        trace = run_flow.statistic(lambda p: p.weights@(p.points[:, 0] > 0))
        final.append(trace[-1])
        ax.plot(run_flow.times, trace, color=shade, lw=.8, alpha=.8, label=f"N = {N:,}".replace(",", " ")
                if run == 0 else None)
    print(f"N = {N}: mass in the good well at t = 8 in six runs:", np.round(final, 2))
ax.plot(times, mass_good, color=viz.TOTAL, lw=2, label="equation")
ax.set(xlabel="time t", ylabel="mass in the good well (x > 0)", ylim=(-.02, 1.02),
       title="(c) finite populations scatter")
ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1, .05))
fig.savefig(FIGURES/"t7_flow_1d.png", dpi=150)

# %% [markdown]
# ![Solution of the equation and particle runs](../docs/figures/t7_flow_1d.png)
#
# (a) The density over time: the population first contracts in the worse well, then
# mass leaks over the barrier and the good well takes over. (b) Snapshots: the particle
# histograms lie on the grid solution. (c) The share of the population in the good well
# for the equation (dark) and for six runs each with `N = 100`, `1000` and `10 000`
# individuals, resampled in every step. Resampling loses individuals at random (genetic
# drift). With 100 individuals the few that start in the good well are easily all lost,
# and the run never crosses: here five of the six runs stay in the worse well, and so do
# two of the six with 1000 individuals. Runs that cross do so at random times. All six
# runs with 10 000 individuals follow the equation closely. The equation describes what large
# populations do; for the chance that a run of a given size finds the good well, the
# measured discovery bound (Tutorial 5) is the tool.
#
# ## 3. The terms, one operator at a time
#
# `solver.terms(rho)` returns each operator's term at a density: a signed curve, positive
# where the operator adds mass, negative where it removes it. Their sum is the rate of
# change of the density. Each term integrates to zero: every operator conserves the total
# mass and only moves it around. The figure shows the drift and noise terms (middle row)
# separately from the selection and recombination terms (bottom row), because they
# differ in size by more than a factor of ten.

# %%
fig, axes = plt.subplots(3, 3, figsize=(10.5, 6.4), layout="constrained", sharex=True)
for ax in axes[0, 1:]:
    ax.sharey(axes[0, 0])
for column, t in enumerate((0, 2, 5)):
    k = int(np.argmin(np.abs(times-t)))
    rho = solution["densities"][k]
    terms = solver.terms(rho)
    top, middle, bottom = axes[:, column]
    top.fill_between(grid, rho, color="0.82")
    top.plot(grid, rho, color="0.35", lw=1)
    top.set_title(f"t = {t}")
    viz.plot_density_terms(grid, {n: terms[n] for n in "DH"}, ax=middle, total=False, legend=False)
    middle.plot(grid, terms["D"]+terms["H"], color=viz.TOTAL, lw=1.2, ls="--", label="D + H")
    viz.plot_density_terms(grid, {n: terms[n] for n in "SR"}, ax=bottom, total=False, legend=False)
    bottom.plot(grid, sum(terms.values()), color=viz.TOTAL, lw=1.8, label="sum of all four")
    for ax in (middle, bottom):
        ax.set_ylabel("")
        if column == 2:
            ax.legend(frameon=False, fontsize=8)
axes[0, 0].set_ylabel("density")
axes[1, 0].set_ylabel("drift and noise")
axes[2, 0].set_ylabel("selection, recombination")
axes[2, 1].set_xlim(-2.2, 2.2)
fig.savefig(FIGURES/"t7_terms_1d.png", dpi=150)

for t in (0, 5):
    k = int(np.argmin(np.abs(times-t)))
    terms = solver.terms(solution["densities"][k])
    print(f"t = {t}: largest |term|", {name: round(float(np.abs(v).max()), 3) for name, v in terms.items()},
          " mass created by each term:", {name: f"{v.sum()*solver.h:.1e}" for name, v in terms.items()})

# %% [markdown]
# ![Terms of the equation at three times](../docs/figures/t7_terms_1d.png)
#
# Reading the curves against the density (top row):
#
# * **Drift D** (transport) is the largest term at `t = 0`: it moves mass down the slopes
#   into the bottom of the worse well (negative on the flanks, positive in the middle).
# * **Noise H** (diffusion) has the opposite shape wherever the density is peaked: it
#   takes mass from the peak and puts it on the flanks.
# * Once the population sits in the wells (`t = 2, 5`), drift and noise are by far the
#   largest terms, but they **nearly cancel** (dashed line): inside each well the drift
#   pushes mass to the bottom and the noise spreads it out, and the two balance. The
#   slow transfer between the wells is left to the small terms.
# * **Selection S** (reaction) is the density times a rate: negative where the objective is
#   above the population average, positive where it is below. It can only reweight mass
#   that is already there. With mass in both wells it moves mass from the worse well to
#   the better one, without anything passing over the barrier: it shrinks one peak and
#   grows the other.
# * **Recombination R** (jump) removes mass at rate `gamma` everywhere and returns it at
#   midpoints of pairs, closer to the centre of the population. With mass in both wells
#   (`t = 5`) it puts mass on the barrier between them, where `f` is large: the
#   interpolation penalty of recombination on a split population, now visible in space.
#   It also feeds the barrier region, from which the drift carries mass into either well.

# ## 4. The weak form: the same terms from particles
#
# The grid solver needs a density; particles do not have one. The weak form of the
# equation tests it with smooth functions instead. With Gaussian bumps of width `h`
# centred at grid points `y`, `<phi_y, mu>` is a smoothed density, and each operator's
# generator applied to the bumps is its term in the equation for the smoothed density.
# This is `evolution.density_view` with `evolution.term_rates`, which works for any
# operator, in any dimension, from particles alone.

# %%
view = evolution.density_view(np.linspace(-2.2, 2.2, 89), bandwidth=.08)
k = int(np.argmin(np.abs(times-5)))
from_grid = evolution.term_rates(assembly, solver.population(solution["densities"][k]), view)
from_particles = evolution.term_rates(assembly, particles.populations[k], view)

fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.4), layout="constrained", sharey=True)
for ax, name in zip(axes, "SRDH"):
    ax.plot(view.axis, from_grid[name], color=colors[name], lw=1.4, label="equation (grid)")
    ax.plot(view.axis, from_particles[name], "o", ms=2.2, color=colors[name], mfc="white", mew=.7,
            label="20 000 particles")
    ax.axhline(0, color="0.6", lw=.6)
    ax.set(title=f"{name}, t = 5", xlabel="y")
axes[0].set_ylabel("term (smoothed, h = 0.08)")
axes[0].legend(frameon=False, fontsize=7.5)
fig.savefig(FIGURES/"t7_weak_form.png", dpi=150)

# %% [markdown]
# ![Weak-form terms from the grid solution and from particles](../docs/figures/t7_weak_form.png)
#
# Each operator's term computed from the particles (circles) matches the term of the grid
# solution smoothed with the same bumps (lines). Smoothing is the price of having no
# density: the bandwidth `h` sets the resolution. In higher dimensions the same
# construction gives the term maps of marginal densities, and other families of test
# functions give the terms for the mean, the spread and the objective values (Tutorial 8).
#
# ## 5. Two dimensions: where each operator adds and removes mass
#
# On `f(x) = (x_1^2 - 1)^2 + (x_2^2 - 1)^2` (four equally good minima at `(+-1, +-1)`) the
# paper's assembly runs from a broad population centred at the origin. A two-dimensional
# `density_view` shows each term as a map: red where the operator adds mass, blue where it
# removes it. Each map has its own colour scale; the number in its corner is the largest
# value of the term, so the sizes of the terms can be compared.

# %%
quartic = Observable(lambda x: np.sum((x*x-1)**2, axis=-1), lambda x: 4*x*(x*x-1),
                     lambda x: np.sum(12*x*x-4, axis=-1))
assembly2 = Assembly([Selection(objective_rate(quartic.value)), Recombination(),
                      Drift(normalized_fd_drift(quartic.value)), GaussianNoise(.2)])
start = Population.uniform(rng.normal(0, 1.2, (40_000, 2)))
flow2 = evolution.flow(assembly2, start, .01, 3., rng, record=[0., .5, 3.])
axis = np.linspace(-2.2, 2.2, 49)
view2 = evolution.density_view((axis, axis), bandwidth=.15)
contours = np.log1p(quartic.value(np.stack(np.meshgrid(axis, axis), -1)))

fig, axes = plt.subplots(3, 5, figsize=(10.5, 6.6), layout="constrained")
for row, (t, pop) in enumerate(zip(flow2.times, flow2.populations)):
    rates = evolution.term_rates(assembly2, pop, view2, pairs=50_000)
    density = view2.tracked(pop).reshape(view2.shape)
    axes[row, 0].pcolormesh(axis, axis, density, cmap="Greys", shading="nearest")
    axes[row, 0].contour(axis, axis, contours, levels=8, colors="0.55", linewidths=.5)
    axes[row, 0].set_aspect("equal")
    axes[row, 0].set_ylabel(f"t = {t:.1f}")
    for column, name in enumerate("SRDH", start=1):
        ax = axes[row, column]
        viz.plot_term_map(view2, rates[name], ax=ax, colorbar=False, objective=quartic.value)
        ax.text(.03, .97, f"max {np.abs(rates[name]).max():.2g}", transform=ax.transAxes, va="top", fontsize=7.5,
                bbox=dict(fc="w", ec="none", alpha=.8, pad=1))
        if row == 0:
            ax.set_title({"S": "selection S", "R": "recombination R", "D": "drift D", "H": "noise H"}[name])
    if row == 0:
        axes[row, 0].set_title("density")
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
fig.savefig(FIGURES/"t7_maps_2d.png", dpi=150)

# %% [markdown]
# ![Term maps in two dimensions](../docs/figures/t7_maps_2d.png)
#
# * At the start (`t = 0`) the **drift** is the largest term: it moves mass from the
#   centre, where `f` has a local maximum, towards the four minima. **Recombination**
#   adds mass in the centre and removes it at the periphery: the midpoint of two members
#   of a broad cloud lies closer to its centre. **Selection** favours the regions around
#   the minima. The noise term is small and its map is mostly sampling noise: a term
#   built from second derivatives is the hardest to estimate from particles.
# * By `t = 0.5` the population sits in four clusters. Inside each cluster drift and noise
#   are large and opposite (contraction against spreading), as in one dimension.
# * At `t = 3` the map of **recombination** shows its kind: it removes mass from the
#   clusters and puts it on the midpoints between them, at the origin and at
#   `(0, +-1)` and `(+-1, 0)`, all far from any minimum. **Selection** removes that mass
#   again (blue between the clusters) and returns it to the clusters. The population is
#   close to a steady state, and the maps show what balances what: drift against noise
#   within each cluster, recombination against selection (and drift) between them.
#
# **Take-away.** The evolution equation is the sum of one term per operator. Transport,
# diffusion, reaction and jump terms act in visibly different ways: continuous motion,
# smoothing, reweighting in place, and relocation of mass. `evolution.TRJ1D` shows the
# terms exactly in one dimension; `evolution.density_view` with `evolution.term_rates`
# shows them for particles, for any operator and in projections of any dimension.
# Tutorial 8 turns to views that work in high dimension (moments and objective values), and
# Tutorial 9 to telling, for an operator given only as code, which kind of term it
# contributes.
