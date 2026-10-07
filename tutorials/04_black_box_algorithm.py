# %% [markdown]
# # Tutorial 4: Checking an algorithm you have already implemented
#
# You do not have to split an algorithm into operators to use the toolbox. If it runs,
# the questions of the theory can be asked of a whole generation:
#
# * **Diagnostics**: do runs collapse at a point that is not good, stall, or blow up?
# * **Discovery guarantee**: with what probability has it proposed a good point by a
#   given generation?
# * **Per-generation contraction**: does one generation reduce the mean objective gap of
#   the proposals, and by what factor?
# * **Ingredient effects**: if the algorithm has switches (crossover on/off, noise level),
#   what does each contribute per generation and along runs?
#
# All that is needed is an **ask/tell** interface: `ask` draws candidates from the
# current proposal law without changing the state; `tell` takes the evaluated
# candidates and returns the next state.

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

from evoscope import algorithm, problems, viz

viz.paper_style()
f = problems.get("wells_d4")          # a good well at x1 = 1 (value 0) and a worse one at x1 = -1 (value 0.8)

# %% [markdown]
# ## An algorithm written without the toolbox
#
# A plain (mu/2, lambda) evolution strategy: the state is the current parent set; `ask`
# picks two random parents, averages them (intermediate recombination, if switched on)
# and adds Gaussian noise; `tell` keeps the best `mu` candidates (truncation selection).
# Nothing in it refers to the theory.

# %%
def make_es(mu=10, lam=40, sigma=.1, recombine=True):
    def initialize(rng):
        return rng.normal(.6, 1.2, (lam, 4))

    def ask(parents, size, rng):
        pairs = parents[rng.integers(0, len(parents), (size, 2))]
        x = .5*(pairs[:, 0]+pairs[:, 1]) if recombine else pairs[:, 0]
        return x+sigma*rng.normal(size=x.shape)

    def tell(parents, candidates, values, rng):
        return candidates[np.argsort(values)[:mu]]

    return algorithm.FunctionalAlgorithm(initialize, ask, tell, batch_size=lam,
                                         population=lambda parents: parents,
                                         name=f"ES(sigma={sigma}, recombination={'on' if recombine else 'off'})")

# %% [markdown]
# An optimizer object from a library with `ask(n)` and `tell(X, f)` methods (as in many
# evolution-strategy packages) is wrapped with `AskTellAlgorithm(factory, batch_size)`
# instead; probes are then drawn from a copy of the object.
#
# ## The algorithm-level report

# %%
es = make_es()
report = algorithm.check_algorithm(
    es, f.value, level=.1, rng=np.random.default_rng(4), generations=100, repetitions=128,
    variants={"full": es, "no recombination": make_es(recombine=False), "sigma 0.3": make_es(sigma=.3)},
    reference="full")
print(report.summary())
report.plot(FIGURES/"t4_algorithm.png")

# %% [markdown]
# ![Algorithm check](../docs/figures/t4_algorithm.png)
#
# How to read the findings (the report prints the same explanations):
#
# * **Diagnostics** count runs that stall above the level or collapse at a point that is
#   not good. On this problem a population can settle in the worse well: agreement is not
#   optimality.
# * **Discovery guarantee**: the measured bound on the probability that no candidate with
#   `f <= 0.1` has been proposed by the last generation, with 95% confidence.
# * **Per-generation contraction**: for states taken from the runs, the average mean gap
#   of the proposals after one generation divided by the mean gap before (right panel;
#   below the red line = contraction). Where it is not below 1 everywhere, the report gives
#   the residual form `E[V_next] <= rho V + c` and its floor `c/(1 - rho)`: the mean gap
#   decreases geometrically until about that level. With a fixed noise level the
#   population cannot concentrate beyond what the noise allows, so a floor is expected.
# * **Ingredient effects per generation**: the change in the next generation's mean gap
#   when an ingredient is switched off or changed, with the same random numbers. Positive
#   means the ingredient helps the mean gap.
#
# ## Following the switches along runs

# %%
variants = {"full": es, "no recombination": make_es(recombine=False), "sigma 0.3": make_es(sigma=.3)}
out = algorithm.ablation_runs(variants, "full", f.value, generations=100, repetitions=128, bootstrap=1000)
fig, axes = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
g = np.arange(101)
for j, name in enumerate(out["names"]):
    for m, ax in enumerate(axes):
        viz.plot_contrast(g, out["mean"][j, :, m], out["low"][j, :, m], out["high"][j, :, m], ax=ax, label=name,
                          color=viz.PALETTE[j])
axes[0].set_title("best-so-far gap: variant - full")
axes[1].set_title("mean gap of the batch: variant - full")
axes[1].legend(frameon=False)
fig.savefig(FIGURES/"t4_ablation.png", dpi=150)

# %% [markdown]
# ![Ablation along runs](../docs/figures/t4_ablation.png)
#
# A larger noise level raises the mean gap of every batch (more spread), and its effect on
# the best-so-far gap is much smaller: the same ingredient can matter a lot for one target
# and little for another. Which matters depends on what you need from the algorithm: one
# good solution (best-so-far), reliably good candidates (mean gap), or agreement at an
# optimum. Without recombination, some runs settle in the worse well.
#
# ## When the algorithm has a step-size parameter
#
# If the update can be made smaller with a parameter `tau` (and `tau = 0` means "keep the
# population"), `check_small_step` tests whether the whole update behaves like one
# small-step operator. If it does, its first-order effect and drift coefficient can be
# estimated and the calculus applies to the algorithm as a single operator. The ES above
# has no such parameter (it replaces the whole population every generation), so this
# check does not apply to it as written; Tutorial 3's assemblies have one.
#
# **Take-away.** No decomposition is needed for a first assessment. When the diagnostics
# or the ingredient effects point at one component, it is worth writing that component as
# an operator and checking it as in Tutorial 2.
