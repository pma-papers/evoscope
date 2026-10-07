# %% [markdown]
# # Tutorial 3: Adding a new operator to an existing algorithm
#
# A common situation: an algorithm works, and you want to know whether a new ingredient
# improves it, and why. Benchmarking the whole algorithm with and without the ingredient
# tells you *whether* results change, but not *how* the ingredient acts. The calculus
# adds the "how":
#
# 1. Does the extended algorithm still behave as the sum of its parts (composition)?
# 2. What does the new operator contribute, population by population (attribution)?
# 3. Does the algorithm reduce the mean objective gap at a positive rate on the
#    populations it actually meets (the **drift coefficient**), and does that estimate
#    carry over to fresh populations?
# 4. How reliably does it find a good point, with what guarantee?
# 5. What does the new operator change along whole runs?
#
# The existing algorithm is the paper's assembly of selection `S`, midpoint
# recombination `R`, derivative-free drift `D` and Gaussian noise `H`. The new operator
# is the pull towards a soft-best point from Tutorial 2, which passed the operator checks.

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

from evoscope import algorithm, attribution, checks, problems, viz
from evoscope.operators import (Assembly, Drift, GaussianNoise, Recombination, Selection, Transport,
                              normalized_fd_drift, objective_rate)
from evoscope.population import broad_gaussian, two_cluster

viz.paper_style()
f = problems.get("wells_d4")          # a good well at x1 = 1 (value 0) and a worse one at x1 = -1 (value 0.8)


def towards_soft_best(x, pop, alpha=5.):
    values = f.value(pop.points)
    weights = pop.weights*np.exp(-alpha*(values-values.min()))
    return weights@pop.points/weights.sum()-x


S, R = Selection(objective_rate(f.value), name="S"), Recombination(name="R")
D, H = Drift(normalized_fd_drift(f.value), name="D"), GaussianNoise(.2, name="H")
P = Transport(drift=towards_soft_best, name="P")
baseline = Assembly([S, R, D, H])
extended = Assembly([S, R, D, P, H])

# %% [markdown]
# ## 1. The assembly check
#
# `check_assembly` runs the operator checks for every component, checks the
# composition, checks that the mean objective gap can serve as the error measure (the
# cutoff check), and estimates the drift coefficient with its attribution.

# %%
rng = np.random.default_rng(2)
populations = [broad_gaussian(32, 4, rng) for _ in range(12)]+[two_cluster(32, 4, rng) for _ in range(12)]
report = checks.check_assembly(extended, populations, f, fast=True)
print(report.summary())
report.plot(FIGURES/"t3_assembly.png")

# %% [markdown]
# ![Assembly check](../docs/figures/t3_assembly.png)
#
# Left: the composed step's change minus the sum of the operators' effects shrinks in
# proportion to the step (composition holds). Right: the **attribution** of the mean-gap
# reduction to the operators on these populations; each dot is a population, and a
# positive value means the operator reduces the mean gap.
#
# ## 2-3. The populations the algorithm actually meets
#
# Initial populations are only the start. The drift coefficient matters along the whole
# run, so collect populations from runs (here at generations 0, 10, 40 and 80) and
# compare the two algorithms on them. Half of the runs calibrate the estimate, the
# other half validate it.

# %%
def run_states(assembly, seed, repetitions=64):
    alg = algorithm.AssemblyAlgorithm(assembly, .1, lambda g: broad_gaussian(32, 4, g), 32)
    runs = algorithm.run_repetitions(alg, f.value, 80, repetitions, np.random.default_rng(seed),
                                     level=.1, n_probe=128, keep_states={0, 10, 40, 80})
    return runs, runs["states"]

runs_ext, states_ext = run_states(extended, 3)
runs_base, states_base = run_states(baseline, 3)

fig, axes = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
for ax, (name, asm, states) in zip(axes, (("baseline S R D H", baseline, states_base),
                                          ("extended S R D P H", extended, states_ext))):
    calibration = [s for row in states[:32] for s in row]
    validation = [s for row in states[32:] for s in row]
    cal = attribution.coefficients(asm, calibration, f)
    val = attribution.coefficients(asm, validation, f)
    lam = attribution.estimate_drift_coefficient(cal["total"])
    check = attribution.validate(lam, val["total"], len(calibration))
    print(f"{name}: lambda_cal = {lam:.3f}; fresh run states below it: {check['violations']}/{check['populations']}"
          f" (exchangeability bound {check['exchangeability_bound']:.3f} per state)")
    print("   median rate of each operator:", {n: round(float(np.nanmedian(cal['ratios'][:, j])), 3)
                                               for j, n in enumerate(cal['names'])})
    viz.plot_component_coefficients(cal, ax=ax)
    ax.set_title(name)
fig.savefig(FIGURES/"t3_attribution.png", dpi=150)

# %% [markdown]
# ![Attribution on run states](../docs/figures/t3_attribution.png)
#
# On the populations met along runs, the new operator `P` reduces the mean gap on every
# population, and the smallest total rate (the drift coefficient `lambda_cal`) changes sign:
# negative for the baseline, slightly positive with `P`. The negative values come from late
# populations that sit close to the optimum, where the noise `H`, which keeps the population
# spread out, costs more than the other operators gain; `P` pulls these populations
# together faster than the noise spreads them. Validation on the run states of the other
# half of the runs gives about the number of violations that the exchangeability bound
# predicts, so the estimates describe the run states well.
#
# ## 4. Discovery and its guarantee
#
# During the runs, **probes** (extra candidates, evaluated and discarded) measured how much
# of each generation's proposal law lies in the target set `{f <= 0.1}`. The **measured
# discovery bound** turns these measurements into a guarantee.

# %%
fig, axes = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
for runs, name, color in ((runs_base, "baseline", viz.PALETTE[1]), (runs_ext, "extended", viz.PALETTE[0])):
    print(name, "-", algorithm.check_discovery(runs, batch=32, level=.1, n_probe=128).statistic)
    viz.plot_discovery(runs["probe_counts"], 128, 32, axes=[axes[0] if name == "extended" else None, axes[1], axes[2]],
                       observed_non_discovery=1-runs["offspring_hit"][:, 1:].mean(axis=0), label=name, color=color,
                       heatmap=name == "extended")
axes[1].legend(frameon=False)
fig.savefig(FIGURES/"t3_discovery.png", dpi=150)

# %% [markdown]
# ![Measured discovery bound](../docs/figures/t3_discovery.png)
#
# Left: the fraction of probes in the target set, run by run, for the extended algorithm
# (bright = high). Middle: the fraction of runs whose proposal law put less than a quarter
# of its mass on the target set. Right: the bound on the probability that no candidate in
# the target set has been proposed by each generation (solid), next to the observed
# frequency (dotted). The new operator lowers both the observed non-discovery frequency
# and the bound. With 64 runs the bound cannot go below about 0.05 (its *floor*);
# Tutorial 5 explains how to design the runs for a smaller level.
#
# ## 5. Along whole runs
#
# `ablation_runs` starts every variant from the same initial populations and compares the
# traces run by run, with bootstrap bands.

# %%
make = lambda asm: algorithm.AssemblyAlgorithm(asm, .1, lambda g: broad_gaussian(32, 4, g), 32)   # noqa: E731
variants = {"full": make(extended), "without P": make(baseline), "without R": make(Assembly([S, D, P, H]))}
out = algorithm.ablation_runs(variants, "full", f.value, generations=80, repetitions=64, bootstrap=1000)
fig, axes = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
g = np.arange(81)
for j, name in enumerate(out["names"]):
    for m, ax in enumerate(axes):
        viz.plot_contrast(g, out["mean"][j, :, m], out["low"][j, :, m], out["high"][j, :, m], ax=ax, label=name,
                          color=viz.PALETTE[j])
axes[0].set_title("best-so-far gap: variant - full")
axes[1].set_title("mean gap of the batch: variant - full")
axes[1].legend(frameon=False)
fig.savefig(FIGURES/"t3_ablation.png", dpi=150)

# %% [markdown]
# ![Ablation along runs](../docs/figures/t3_ablation.png)
#
# Positive differences mean the variant is worse than the full algorithm; bands are 95%
# bootstrap intervals over runs. Removing `P` slows the decrease of both the best-so-far
# gap and the mean gap.
#
# **Take-away.** The question "does the new operator help?" gets three complementary
# answers: what it does to each population (attribution), what it does to the
# algorithm's reliability (discovery and its guarantee), and what it does along whole runs
# (ablation contrasts).
