# %% [markdown]
# # Tutorial 2: Checking a new operator idea
#
# You have an idea for a new operator. Before investing in proofs, or even in extensive
# benchmarking, you want to know: does the operator fit the theory, so that its effect
# can be combined with other operators and convergence arguments can be built on it?
#
# `check_operator` answers this numerically, from the update rule alone. It tests the
# four conditions that the paper's composition theorem asks of every operator, in plain
# words:
#
# | Condition | Plain-language question |
# |---|---|
# | (A1) first-order effect | Does a tiny step have a well-defined effect per unit step? |
# | (A2) moment stability | Does one step keep the population from blowing up? |
# | (A3) near identity | Does a small step change the population only a little? |
# | (A4) continuity | Does the effect change smoothly when the population changes slightly? |
#
# We follow one idea through three versions: **"pull every individual towards the best
# one."**

# %%
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next(p for p in (HERE, *HERE.parents) if (p/"src/evoscope").is_dir())
sys.path.insert(0, str(ROOT/"src"))
FIGURES = ROOT/"docs/figures"

from evoscope import checks, problems, viz
from evoscope.operators import StepOperator, Transport
from evoscope.population import Population, broad_gaussian, two_cluster

viz.paper_style()
f = problems.get("wells_d4")          # unequal wells: a good well at x1 = 1 and a worse one at x1 = -1
rng = np.random.default_rng(1)
populations = [broad_gaussian(16, 4, rng), two_cluster(16, 4, rng), broad_gaussian(16, 4, rng)]

# %% [markdown]
# Choose populations of the kinds your algorithm will meet: here two broad clouds and a
# two-cluster population. The verdicts describe these populations only.
#
# ## Version 1: as first written
#
# A typical first implementation moves every individual a fixed fraction (here 30%) of
# the way towards the current best individual, plus some noise. The step size `tau` is
# accepted but not used.

# %%
def pull_to_best_v1(pop, tau, rng):
    best = pop.points[np.argmin(f.value(pop.points))]
    return pop.points+.3*(best-pop.points)+.1*rng.normal(size=pop.points.shape)

v1 = StepOperator(pull_to_best_v1, name="pull v1")
report = checks.check_operator(v1, populations, fast=True)
print(report.summary())

# %% [markdown]
# Conditions (A1) and (A3) fail (and (A4) as well, for a reason we return to in version 2):
# a step of size `tau` should change the population
# **in proportion to `tau`**, and this rule moves it by a fixed amount whatever `tau` is.
# Most update rules can be put in small-step form by one of three scalings: move by
# `tau` times a drift, perturb by `sqrt(tau)` times noise, or replace a fraction `tau` of
# the population.
#
# ## Version 2: scaled with the step size
#
# Write the same idea as a **transport** (a move of each individual): drift towards the
# best, `b(x) = 0.3 (x_best - x)`, and noise of size 0.1. `Transport` applies
# `x -> x + tau b(x) + sqrt(tau) noise`. The drift depends on the whole population (who
# is best), so the drift function takes the population as a second argument.

# %%
def towards_best(x, pop):
    best = pop.points[np.argmin(f.value(pop.points))]
    return .3*(best-x)

v2 = Transport(drift=towards_best, sigma=.1, name="pull v2")
report = checks.check_operator(v2, populations, fast=True)
print(report.summary())
report.plot(FIGURES/"t2_operator_v2.png")

# %% [markdown]
# ![Operator check, version 2](../docs/figures/t2_operator_v2.png)
#
# (A1)-(A3) now hold, but **continuity (A4) fails**: when the population changes
# slightly, the identity of the best individual can switch, and the whole drift jumps
# to a different target. The right panel shows the symptom: the largest rate of change
# of the effect grows in proportion to the resolution of the check, as it does for a
# jump. Hard "best-of" comparisons, truncation by rank and thresholds all behave this
# way.
#
# ## Version 3: a soft "best"
#
# Replace the best individual by a **soft-best point**: the average of the population
# weighted by `exp(-alpha f(x))`. For large `alpha` it is close to the best individual,
# but it moves continuously when the population changes. (This is the *consensus point*
# of consensus-based optimization.)

# %%
def towards_soft_best(x, pop, alpha=5.):
    values = f.value(pop.points)
    weights = pop.weights*np.exp(-alpha*(values-values.min()))
    center = weights@pop.points/weights.sum()
    return .3*(center-x)

v3 = Transport(drift=towards_soft_best, sigma=.1, name="pull v3")
report = checks.check_operator(v3, populations, fast=True)
print(report.summary())
report.plot(FIGURES/"t2_operator_v3.png")

# %% [markdown]
# ![Operator check, version 3](../docs/figures/t2_operator_v3.png)
#
# All four conditions hold on these populations. The operator can now be combined with
# others in an assembly, and its contribution measured (Tutorial 3).
#
# ## A fourth lesson: plain gradient steps on steep objectives
#
# Moment stability (A2) catches rules that push far-away individuals even farther. A plain
# gradient step `x -> x - tau * 0.1 * grad f(x)` on the quartic `sum (x_j^2 - 1)^2` is an
# example: the gradient grows like `x^3`, so a population far from the origin is thrown
# out further.

# %%
quartic = problems.get("quartic_d4")
gradient_step = Transport(drift=lambda x: -.1*quartic.gradient(x), name="gradient")
finding = checks.check_moment_stability(gradient_step, populations)
print(finding.verdict, "-", finding.statistic)

normalized = Transport(drift=lambda x: -quartic.gradient(x)/np.sqrt(1+np.sum(quartic.gradient(x)**2, axis=-1, keepdims=True)),
                       name="normalized gradient")
finding = checks.check_moment_stability(normalized, populations)
print(finding.verdict, "-", finding.statistic)

# %% [markdown]
# Normalizing the step (dividing by `sqrt(1 + |grad f|^2)`) removes the problem; this is
# why the paper's derivative-free drift is normalized.
#
# ## Summary: common failures and their fixes
#
# | Verdict | Typical cause | Fix |
# |---|---|---|
# | (A1)/(A3) violated | the rule does not shrink with `tau` | move by `tau` x drift, perturb by `sqrt(tau)` x noise, or replace a fraction `tau` |
# | (A2) violated | superlinear moves, heavy-tailed noise | normalize or clip the move; light-tailed noise |
# | (A4) violated | hard best-of, rank threshold, truncation | soft-best (weighted mean), smoothed ranks, sigmoids |
#
# The checks are evidence on the populations you gave, not proofs. A "consistent" verdict
# means nothing contradicts the condition there; it is a good reason to go on, and the
# point at which a proof becomes worth the effort.
