# %% [markdown]
# # Tutorial 8: Watching a population evolve in many dimensions
#
# In one or two dimensions the population can be drawn (Tutorial 7). In eight it cannot,
# but the evolution equation can still be watched through suitable **views**: families of
# test functions that show one aspect of the population at a time.
#
# | view | shows | in the toolbox |
# |---|---|---|
# | error measures | mean gap, spread, squared distance to a minimizer | `evolution.summary_view` |
# | objective levels | the share of the population within `c` of the optimal value, for every level `c` | `evolution.level_view` |
# | shape | the variances along the principal axes of the population | `evolution.spread_view` |
# | moments | the mean | `evolution.mean_view` |
#
# Each operator's **term** in the equation, seen through a view, is its generator applied
# to the view's functions (`evolution.term_rates`). Along a run, a **ledger**
# (`evolution.ledger`) records the terms and their time integrals: how much of the
# change of each quantity each operator is responsible for. By the definition of a
# solution of the equation, the integrals add up to the observed change.
#
# The example is a puzzle. On a problem with a good and a worse well, an assembly with
# recombination stays in the worse well, while the same assembly without recombination
# moves to the good one. The views show why.

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
from evoscope.operators import paper_assembly
from evoscope.population import Population

viz.paper_style()
rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Two assemblies on the unequal wells
#
# `wells_d4` is `f(x) = (x_1^2 - 1)^2 + 0.2 (x_1 - 1)^2 + 0.5 (x_2^2 + x_3^2 + x_4^2)` in
# four dimensions: the good minimum is at `x_1 = 1` (`f = 0`), a worse one near
# `x_1 = -1` (`f` about 0.8). The population starts in the worse well (a Gaussian around
# `(-1, 0, 0, 0)` with standard deviation 0.5).
#
# Both assemblies are the paper's: selection S, recombination R, drift D and noise H
# (`sigma = 0.2`), once with and once without recombination. `evolution.flow` follows
# each with 4000 weighted particles and step `tau = 0.02`.

# %%
f = problems.get("wells_d4")
assemblies = {"with R": paper_assembly(f, sigma=.2), "without R": paper_assembly(f, gamma=0, sigma=.2)}
start = Population.uniform(rng.normal(0, .5, (4000, 4))+np.array([-1., 0, 0, 0]))
times = np.concatenate([np.linspace(0, 1, 21), np.linspace(1.25, 12, 44)])
flows = {name: evolution.flow(a, start, .02, 12., rng, record=times) for name, a in assemblies.items()}
for name, fl in flows.items():
    gap = fl.statistic(lambda p: p.expect(f.value(p.points)))
    good = fl.statistic(lambda p: p.weights@(p.points[:, 0] > 0))
    at = [int(np.argmin(np.abs(fl.times-t))) for t in (0, 2, 6, 12)]
    print(f"{name:10s} mean gap at t = 0, 2, 6, 12: {np.round(gap[at], 3)}"
          f"   share in the good well: {np.round(good[at], 2)}")

# %% [markdown]
# Without recombination the mean gap keeps falling and most of the population ends in
# the good well. With recombination the mean gap stalls near the value of the worse
# well. In one step at a broad population (Tutorial 1), recombination *helped*. What
# changes along the run?
#
# ## 2. A ledger of the error measures
#
# `summary_view` holds the mean gap, the spread (the average squared distance to the
# population mean) and the squared distance to the good minimizer `(1, 0, 0, 0)`. The
# spread is centred at the current population, so the view is rebuilt at every time:
# pass a function of the population. The ledger's `contributions` are the time integrals
# of the terms; their sum is compared with the change actually observed along the flow.
# The flows are recorded densely at the start, where the population changes fast, so that
# the time integrals are accurate. Drift and noise are shown together (dashed): once the
# population sits at the bottom of a well, both are large and they nearly cancel.

# %%
target = np.array([1., 0, 0, 0])
ledgers = {name: evolution.ledger(assemblies[name], fl, lambda p: evolution.summary_view(f, p, target=target))
           for name, fl in flows.items()}
fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), layout="constrained", sharex=True)
for row, (name, led) in enumerate(ledgers.items()):
    for column in range(3):
        viz.plot_ledger(led, index=column, ax=axes[row, column], legend=(row, column) == (0, 2),
                        groups={"D + H": ("D", "H")})
        axes[row, column].set_title(f"{led['labels'][column]}, {name}")
        if row == 0:
            axes[row, column].set_xlabel("")
fig.savefig(FIGURES/"t8_ledger.png", dpi=150)
for name, led in ledgers.items():
    worst = np.max(np.abs(led["predicted"]-led["change"]), axis=0)/np.ptp(led["value"], axis=0)
    print(f"{name}: largest gap between the sum of the terms and the observed change, relative to the range:",
          np.round(worst, 3))

# %% [markdown]
# ![Ledgers of the error measures](../docs/figures/t8_ledger.png)
#
# In every panel the sum of the operators' contributions (dark line) follows the change
# observed along the flow (circles); the small remaining difference comes from the finite
# number of particles and the finite step. The ledger then says who did what.
#
# * **Mean gap.** In both assemblies drift and noise (dashed) do almost all the work at
#   first: the drift takes the population to the bottom of the worse well. Afterwards the
#   assemblies differ. Without recombination, selection's contribution keeps growing and
#   accounts for the further descent. With recombination, selection's contribution stays
#   small, and recombination's contribution, negative at first (midpoints of a broad cloud
#   are better than their parents), turns around after `t = 6`: from then on recombination
#   *raises* the mean gap.
# * **Spread.** Drift and noise together contract the population at first, and
#   recombination contracts it steadily (midpoints halve the spread). Selection
#   *increases* the spread while it moves mass from one well to the other: a population
#   split between two wells is more spread out than one in a single well.
# * **Squared distance to the good minimizer.** Without recombination, selection is the
#   operator that reduces it in the second half of the run: it moves mass to the good
#   well, which no other operator does here.
#
# ## 3. Objective values: the mass below each level
#
# The mean gap compresses the distribution of objective values into one number.
# `level_view` keeps all of it: for each level `c`, the share of the population with gap
# at most `c` (smoothed, so that every operator has a term). Its terms say which operator
# moves mass below which level. The figure shows the share over time for both
# assemblies (top) and the terms at `t = 3` (bottom).

# %%
levels = np.geomspace(1e-3, 10, 31)
level_view = evolution.level_view(f, levels, width=.5)
level_ledgers = {name: evolution.ledger(assemblies[name], fl, level_view) for name, fl in flows.items()}
fig, axes = plt.subplots(2, 2, figsize=(9.5, 6), layout="constrained")
for column, (name, led) in enumerate(level_ledgers.items()):
    viz.plot_levels(led, ax=axes[0, column])
    axes[0, column].axhline(.8, color="w", lw=.7, ls=":")
    axes[0, column].set_title(name)
    k = int(np.argmin(np.abs(led["times"]-3)))
    viz.plot_level_terms(levels, dict(zip(led["names"], led["rates"][k])), ax=axes[1, column],
                         legend=column == 1, groups={"D + H": ("D", "H")})
    axes[1, column].axvline(.8, color="0.5", lw=.7, ls=":")
    axes[1, column].set_title(f"terms at t = 3, {name}")
fig.savefig(FIGURES/"t8_levels.png", dpi=150)
for name, led in level_ledgers.items():
    k = int(np.argmin(np.abs(led["times"]-3)))
    below = levels < .8
    print(f"{name}: largest term of S and smallest term of R below the worse well's level at t = 3:",
          round(float(led["rates"][k, 0, below].max()), 4), round(float(led["rates"][k, 1, below].min()), 4))

# %% [markdown]
# ![Mass below each objective level](../docs/figures/t8_levels.png)
#
# Top: the colour at time `t` and level `c` is the share of the population with gap at
# most `c`; the dotted line marks the gap of the worse well, 0.8. Once the population
# sits in the worse well, almost all of it has a gap a little above 0.8, so the share jumps
# from near 0 to near 1 across that line. The mass below 0.8 is the mass in the good
# well. Without recombination it fills in over time (the region below the line turns
# yellow); with recombination it stays near zero.
#
# Bottom: the terms at `t = 3` (each panel has its own scale). Each curve is the rate at
# which an operator moves mass below level `c`. With recombination (left), selection adds
# mass at every level below the dotted line: it moves mass from the worse well to the good
# one without moving any point. This is a general property: a selection rate that
# increases with `f` can only raise the share of the population below every level.
# Recombination removes more mass than that at the levels between 0.01 and 0.8: the child
# of a parent in the good well and a parent in the worse well lands between them, at a
# high gap. Without recombination (right), selection's term is several times larger, and
# nothing works against it.
#
# ## 4. Who decides whether the good well grows?
#
# The share in the good well grows at a **per-capita rate**, the sum of four terms divided
# by the share itself. A one-level `level_view` (gap at most 0.3, well inside the good
# well) and its ledger give the four per-capita rates along the run.

# %%
good_view = evolution.level_view(f, [.3], width=.3)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), layout="constrained", sharey=True)
for ax, (name, fl) in zip(axes, flows.items()):
    led = evolution.ledger(assemblies[name], fl, good_view)
    per_capita = dict(led)
    per_capita["rates"] = led["rates"]/led["value"][:, None, :]
    per_capita["total"] = led["total"]/led["value"]
    viz.plot_ledger(per_capita, ax=ax, kind="rates", legend=False, groups={"D + H": ("D", "H")})
    ax.axhline(0, color="0.6", lw=.6)
    ax.set(title=f"good-well share, {name}", xlabel="time t", ylim=(-.6, 1.))
    k = int(np.argmin(np.abs(led["times"]-2)))
    print(f"{name}: per-capita rates of the good-well share at t = 2:",
          {op: round(float(per_capita["rates"][k, j, 0]), 2) for j, op in enumerate(led["names"])})
axes[0].set_ylabel("per-capita rate")
axes[1].set_ylabel("")
axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
fig.savefig(FIGURES/"t8_good_mass.png", dpi=150)

# %% [markdown]
# ![Per-capita rates of the good-well share](../docs/figures/t8_good_mass.png)
#
# The two panels separate a direct and an indirect effect of recombination.
#
# * **Direct:** recombination's own term is negative, and it grows in size along the run.
#   Every individual is replaced at rate `gamma = 1`; the child of a good-well individual
#   stays in the good well only when its partner is also there, which is rare while the
#   good well is small.
# * **Indirect:** selection's term is much smaller *with* recombination (0.06 against 0.21 at
#   `t = 2`). Selection grows the good well in proportion to how much better its members
#   are than the population average, and with recombination the good-well members are
#   fewer, more recent and less settled at the bottom. The operators interact through
#   the population: removing one changes the terms of the others.
# * **Drift and noise** together (dashed) are positive with recombination: they carry into
#   the good well part of the mass that recombination places near the barrier. Without
#   recombination nothing arrives from the barrier, and their sum vanishes once the good
#   well is in balance.
#
# With recombination the sum stays small, so the good well grows only slowly; without it
# the sum is clearly positive until the good well holds most of the population. In an
# idealized picture with two point-like clusters and no drift or noise, the good cluster
# grows exactly when `omega (Phi_worse - Phi_good) > gamma`: selection's advantage must
# beat the recombination rate. Drift and noise shift this threshold, which is why the
# per-capita terms are measured rather than computed.
#
# ## 5. The shape of the population in many dimensions
#
# `spread_view` follows the variances along the principal axes of the population. On the
# rotated anisotropic problem in eight dimensions (curvatures between 1 and 2 along
# rotated axes) the terms show how each operator shapes the population. The figure divides
# each term by the variance along its axis: a relative rate, comparable across axes.

# %%
g = problems.get("rotated_anisotropic_pl_d8")
assembly8 = paper_assembly(g, sigma=.2)
start8 = Population.uniform(rng.normal(0, 1., (4000, 8)))
flow8 = evolution.flow(assembly8, start8, .01, 5., rng, record=np.linspace(0, 5, 26))
shape = evolution.ledger(assembly8, flow8, evolution.spread_view)

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), layout="constrained")
axes[0].semilogy(shape["times"], shape["value"][:, 1:], color=viz.PALETTE[0], lw=1)
axes[0].semilogy(shape["times"], shape["value"][:, 0]/8, color=viz.TOTAL, lw=2, label="total spread / d")
axes[0].set(xlabel="time t", ylabel="variance along principal axis", title="(a) principal variances")
axes[0].legend(frameon=False, fontsize=8)
colors = viz.operator_colors(shape["names"])
for ax, t, title in ((axes[1], .4, "(b) relative rates at t = 0.4"), (axes[2], 5, "(c) relative rates at t = 5")):
    k = int(np.argmin(np.abs(shape["times"]-t)))
    variances = shape["value"][k, 1:]
    width = .2
    for j, op in enumerate(shape["names"]):
        ax.bar(np.arange(8)+(j-1.5)*width, shape["rates"][k, j, 1:]/variances, width, color=colors[op], label=op)
    ax.plot(np.arange(8), shape["total"][k, 1:]/variances, "k_", ms=12, mew=2, label="sum")
    ax.axhline(0, color="0.6", lw=.6)
    ax.set(xticks=range(8), xticklabels=[str(i+1) for i in range(8)], xlabel="principal axis (widest first)",
           title=title)
axes[1].set_ylabel("rate / variance")
axes[2].legend(frameon=False, fontsize=8, ncol=2)
fig.savefig(FIGURES/"t8_shape.png", dpi=150)
k = len(shape["times"])-1
print("relative rates at t = 5, axis 1 and axis 8:",
      {op: np.round(shape["rates"][k, j, [1, 8]]/shape["value"][k, [1, 8]], 2) for j, op in enumerate(shape["names"])})

# %% [markdown]
# ![Principal variances and their terms](../docs/figures/t8_shape.png)
#
# (a) The variances along the eight principal axes shrink and settle by `t = 4` at
# different levels: the population becomes an ellipsoid. (b, c) The relative rates per
# axis show different signatures:
#
# * **recombination** removes the same fraction of the variance on every axis (`-1/2`:
#   midpoints of independent parents have half the variance) and does not depend on the
#   problem;
# * **noise** adds the same *amount* `sigma^2` on every axis, so relative to the variance
#   it matters most on the thinnest axes;
# * **drift** contracts each axis according to the curvature of the objective along it,
#   most strongly on the thinnest axes, which are the steepest directions; it is the term
#   that carries the geometry of the problem;
# * **selection** acts through the differences of objective values within the population,
#   which become small when the population is narrow; its effect on the shape is small
#   at `t = 0.4` and negligible at `t = 5`.
#
# At `t = 5` the sum is close to zero on every axis: the shape has settled where the
# contraction by drift and recombination balances the noise, axis by axis.
#
# **Take-away.** Views make the evolution equation watchable in any dimension: error
# measures and their ledgers say which operator accounts for how much of the progress;
# the mass below each objective level shows the whole distribution of objective values
# and which operator moves mass across which level; principal variances show how each
# operator shapes the population. Ledgers of runs with and without an operator reveal
# interactions that a one-step analysis cannot: here recombination slows down the
# move to the good well both directly and by weakening selection.
