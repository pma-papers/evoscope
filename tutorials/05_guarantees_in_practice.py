# %% [markdown]
# # Tutorial 5: Discovery guarantees in practice
#
# The **measured discovery bound** is the most direct guarantee the toolbox offers: an
# upper bound, with stated confidence, on the probability that a run of your algorithm
# has not proposed any candidate in the target set `{f <= c}` by a given generation. It
# uses no approximation theory, only measurements. This tutorial covers what you need to
# use it well:
#
# 1. how it is computed, in plain words;
# 2. how many runs and probes you need (the *floor* and the *resolution*);
# 3. what to do when the optimum value is unknown;
# 4. how to read a bound that is much larger than the observed failure rate.

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

from evoscope import algorithm, discovery, problems, viz

viz.paper_style()

# %% [markdown]
# ## 1. How the bound works
#
# In every generation of every run, a few extra candidates (**probes**) are drawn from the
# algorithm's current proposal distribution, evaluated, counted (how many have
# `f <= c`?) and thrown away, so the run itself is not affected. From the count, a
# generation is **flagged** if the proposal distribution might put less than a quarter of
# its probability on the target set. If it does put at least a quarter there, a batch of
# `N` candidates misses the target with probability at most `(3/4)^N`, which is tiny.
# So a run can fail only through flagged generations, and the bound adds up the flag
# frequencies of the last generations (with exact confidence limits), discounted by how
# unlikely it is that all later batches missed as well.
#
# ## 2. The design: runs, probes and batch size
#
# Even if no generation is ever flagged, the bound cannot be smaller than its **floor**,
# which is set by the number of runs `R` and the batch size. And a generation is only
# unflagged if enough probes succeed, which is set by the number of probes.

# %%
print("floor of the bound at generation 200:")
for R in (32, 64, 128, 256):
    print(f"  R = {R:3d} runs:", "  ".join(f"batch {b:2d}: {discovery.floor(R, b, 200):.3f}" for b in (1, 4, 16, 32)))
print("\nsmallest fraction of successful probes that avoids a flag:")
for n in (32, 64, 128, 256, 1024):
    print(f"  {n:4d} probes per generation: {discovery.smallest_unflagged_fraction(n):.2f}")

# %% [markdown]
# Rules of thumb:
#
# * to certify a level `delta`, you need roughly `R >= 3/delta` runs (e.g. 64 runs for
#   0.05, 256 for 0.013) with batches of 16 or more;
# * with 128 probes a generation is unflagged only if at least 38% of the probes succeed;
#   if your algorithm's success fraction hovers between 25% and 40%, use more probes;
# * probes cost `R x generations x probes` extra evaluations, which is usually acceptable
#   in a benchmarking study but not in production.
#
# ## 3. When the optimum value is unknown
#
# Nothing in the algorithm needs the optimum: only the definition of the target set
# does. Two ways around it:
#
# * if a **lower bound** `f_lb <= f_*` is known (from a relaxation, or `0` for a
#   nonnegative objective), the target `{f <= f_lb + eps}` is contained in
#   `{f <= f_* + eps}`, so a guarantee for it is a guarantee for the true target;
# * with a **reference value** `f_ref` (the best value known so far, or a competitor's
#   result), the target `{f <= f_ref + eps}` certifies reaching within `eps` of that value.
#
# Below, an evolution strategy on the unequal-wells problem is certified for the target
# `{f <= 0.1}` (with the lower bound `f_lb = 0`), for three numbers of runs.

# %%
f = problems.get("wells_d4")


def make_es(mu=10, lam=40, sigma=.1):
    def ask(parents, size, rng):
        pairs = parents[rng.integers(0, len(parents), (size, 2))]
        return .5*(pairs[:, 0]+pairs[:, 1])+sigma*rng.normal(size=(size, 4))
    return algorithm.FunctionalAlgorithm(lambda rng: rng.normal(.6, 1.2, (lam, 4)), ask,
                                         lambda p, X, y, rng: X[np.argsort(y)[:mu]], batch_size=lam,
                                         population=lambda p: p)

fig, ax = plt.subplots(figsize=(4.5, 3), layout="constrained")
for R, color in ((32, viz.PALETTE[5]), (64, viz.PALETTE[1]), (256, viz.PALETTE[0])):
    runs = algorithm.run_repetitions(make_es(), f.value, 80, R, np.random.default_rng(R), level=.1, n_probe=128)
    finding = algorithm.check_discovery(runs, batch=40, level=.1, n_probe=128, delta=.05)
    print(f"R = {R:3d}: {finding.verdict:12s} {finding.statistic}")
    ax.plot(np.arange(1, 81), discovery.bound_by_endpoint(discovery.inadequate_mass_flags(runs["probe_counts"], 128).sum(axis=0), R, 40),
            color=color, label=f"R = {R}")
ax.axhline(.05, color="0.4", lw=.8, ls="--")
ax.set(xlabel="generation", ylabel="bound on non-discovery", ylim=(0, 1.02), title="more runs, lower floor")
ax.legend(frameon=False)
fig.savefig(FIGURES/"t5_runs.png", dpi=150)

# %% [markdown]
# ![Bound for different numbers of runs](../docs/figures/t5_runs.png)
#
# ## 4. Reading a bound that is much larger than the observed failure rate
#
# The bound is sufficient, not necessary. If it is far above the observed failure
# frequency, there are three possible reasons, and the report says which applies:
#
# * **the floor is too high**: more runs, or a larger batch (verdict *not resolved*);
# * **the success probability is adequate but not resolved by the probes**: the probe
#   success fractions sit between 25% and the threshold above; use more probes;
# * **the proposal distribution really puts little mass on the target** in the last
#   generations: for example, the algorithm keeps a large noise level, so most proposals
#   miss even though the best-so-far is good (verdict *suspect*). The bound then correctly
#   refuses to certify the *proposals*; a statement about the best point found so far
#   needs an earlier endpoint, or a smaller noise level late in the run.
#
# The next cell shows the third case: the same ES with a larger noise level finds good
# points, but its proposals stay spread out.

# %%
runs = algorithm.run_repetitions(make_es(sigma=.3), f.value, 80, 128, np.random.default_rng(5), level=.1, n_probe=128)
finding = algorithm.check_discovery(runs, batch=40, level=.1, n_probe=128)
print(finding.verdict, "-", finding.statistic)
print("median fraction of probes in the target set over the last 20 generations:",
      np.median(runs["probe_counts"][:, -20:]/128).round(3))

# %% [markdown]
# **Take-away.** Decide the target and the confidence level first, choose the number of
# runs from the floor, and check the probe resolution. A certified bound is a statement
# about your algorithm as run, valid without any assumption about the objective beyond
# what defines the target set.
