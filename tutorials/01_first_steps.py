# %% [markdown]
# # Tutorial 1: First steps — what does each operator do?
#
# This tutorial needs no knowledge of the theory. It shows the three ideas everything
# else builds on:
#
# 1. a **population** is a cloud of candidate solutions (with weights);
# 2. an **operator** is one update rule (mutation, selection, crossover, ...) with a
#    **step size** `tau`: a small step changes the population only a little;
# 3. the **effect** of an operator is how fast it changes an average quantity of the
#    population, such as the mean objective gap, for a tiny step. Effects of different
#    operators **add up**, so the change produced by a whole algorithm step can be split
#    into the contributions of its operators.
#
# Run this file as a script (`python tutorials/01_first_steps.py`) or open it as a
# notebook (`tutorials/notebooks/01_first_steps.ipynb`).

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

from evoscope import attribution, problems, viz
from evoscope.operators import Assembly, GaussianNoise, Recombination, Selection, Drift, objective_rate, normalized_fd_drift
from evoscope.population import Population, broad_gaussian, two_cluster

viz.paper_style()
rng = np.random.default_rng(0)

# %% [markdown]
# ## A test problem and two populations
#
# We minimize the *multimodal quartic* `f(x) = sum_j (x_j^2 - 1)^2` in four dimensions.
# It has 16 global minimizers, at all points whose coordinates are +1 or -1. The value
# `f(x)` is the **gap** of `x` (the optimum value is 0).
#
# Two kinds of population: a *broad* cloud, and a *two-cluster* population whose members
# sit near two different minimizers (+1,+1,+1,+1) and (-1,-1,-1,-1).

# %%
f = problems.get("quartic_d4")
broad = broad_gaussian(32, 4, rng)
clusters = two_cluster(32, 4, rng)
print("mean gap of the broad population:      ", round(broad.expect(f.value(broad.points)), 3))
print("mean gap of the two-cluster population:", round(clusters.expect(f.value(clusters.points)), 3))

# %% [markdown]
# ## Four operators
#
# * `S` **selection**: reweights individuals, favouring low objective values;
# * `R` **recombination**: replaces a fraction `tau` of the population by midpoints of
#   random pairs of parents;
# * `D` **drift**: moves every individual a little downhill (here with a derivative-free
#   gradient estimate, normalized so that far-away points do not jump);
# * `H` **noise**: adds a Gaussian perturbation of size `0.2 * sqrt(tau)`.
#
# An **assembly** applies them in a given order: selection first, then recombination,
# then drift, then noise.

# %%
S = Selection(objective_rate(f.value))
R = Recombination()
D = Drift(normalized_fd_drift(f.value))
H = GaussianNoise(.2)
algorithm_step = Assembly([S, R, D, H])

# %% [markdown]
# ## The effect of each operator
#
# For a population, `generator_actions` returns each operator's **effect on the mean
# gap**: the rate of change of the mean gap per unit step, for a tiny step. A negative
# effect means the operator *decreases* the mean gap (good); a positive effect means it
# increases it.

# %%
for name, pop in (("broad", broad), ("two-cluster", clusters)):
    effects = algorithm_step.generator_actions(pop, f)
    print(f"{name:12s}", "  ".join(f"{k}: {v:+8.3f}" for k, v in effects.items()),
          f"  total: {sum(effects.values()):+8.3f}")

# %% [markdown]
# On the broad cloud every operator except the noise helps. On the two-cluster
# population, recombination is strongly harmful: the midpoint of a parent near
# (+1,+1,+1,+1) and one near (-1,-1,-1,-1) lies near the origin, where `f = 4`.
# The same operator helps or hurts depending on the **shape of the population**.
#
# ## Do the effects really add up?
#
# Take one actual step of size `tau` and compare the change of the mean gap per unit step
# with the sum of the four effects. As `tau` shrinks, the two agree.

# %%
taus = .1/2.**np.arange(6)
check = attribution.consistency(algorithm_step, clusters, f, taus)
for t, q in zip(taus, check["quotients"]):
    print(f"tau = {t:.5f}: change per unit step {q:+.4f}   sum of effects {check['generator']:+.4f}")

# %% [markdown]
# The difference shrinks in proportion to `tau`. This is the **composition theorem**
# of the paper in action: to first order, an algorithm step does what its
# operators do separately, added up. That is why the effects can be attributed to the
# operators.
#
# ## Relative effects over many populations
#
# Dividing an effect by the mean gap gives a **rate**: by what fraction per unit time
# the operator reduces the mean gap. Computing it over many populations shows how
# robust each operator's role is.

# %%
populations = {"broad": [broad_gaussian(32, 4, rng) for _ in range(48)],
               "two-cluster": [two_cluster(32, 4, rng) for _ in range(48)]}
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3), layout="constrained")
for ax, (name, pops) in zip(axes, populations.items()):
    result = attribution.coefficients(algorithm_step, pops, f)
    viz.plot_component_coefficients(result, ax=ax)
    ax.set_title(f"{name} populations")
    ax.set_ylabel("rate of mean-gap reduction")
fig.savefig(FIGURES/"t1_attribution.png", dpi=150)

# %% [markdown]
# ![Rates of mean-gap reduction](../docs/figures/t1_attribution.png)
#
# Each dot is one population; the black bars mark the smallest value. On broad clouds
# the total rate ("sum") is positive for every population: the algorithm step reduces
# the mean gap at a guaranteed relative rate on populations like these. On two-cluster
# populations the total is negative, mostly because of recombination, with the noise
# adding to it: near the minimizers every perturbation increases the gap.
#
# **Take-away.** The calculus answers "what does each ingredient contribute?" at the
# level of a single population, before any long runs. Tutorial 2 shows how to check that
# your own operator qualifies for this kind of analysis.
