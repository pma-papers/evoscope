# EvoScope documentation

EvoScope helps you develop population-based optimization algorithms (evolution
strategies, genetic algorithms, particle methods, consensus-based optimization, and your
own designs) with the help of a mathematical theory, **without having to do the
mathematics yourself**. You write update rules in ordinary Python; the toolbox checks
numerically whether they behave as the theory requires, tells you what each ingredient
contributes, and gives guarantees about how reliably your algorithm finds good solutions.

The theory is in the paper *Operator Calculus for Population-Based Optimization:
Modular Convergence and Finite-Population Guarantees*. You do not need to read it to use
the toolbox; every result is explained here in plain words, and the [theory
map](theory-map.md) points to the paper for those who want the details.

## Which situation are you in?

| You have | Start with | Guide |
|---|---|---|
| an idea for a new operator (a mutation, selection, crossover, or anything that updates a population) | `check_operator` | [Checking a new operator](guide-operator.md) |
| several operators you want to combine, or a new operator to add to an existing algorithm | `check_assembly`, attribution, discovery bound | [Combining operators](guide-assembly.md) |
| an algorithm that already runs, and no wish to split it into parts | `check_algorithm` | [Checking an implemented algorithm](guide-algorithm.md) |

The three entry points fit the way algorithms are usually developed: try an operator idea
cheaply, combine operators and see what each one contributes, and assess the finished
algorithm. You can enter at any of them.

## Five-minute start

```python
import numpy as np
from evoscope import checks, problems
from evoscope.operators import Transport
from evoscope.population import broad_gaussian, two_cluster

f = problems.get("wells_d4")                      # a test problem (or your own objective)
rng = np.random.default_rng(0)
populations = [broad_gaussian(16, 4, rng), two_cluster(16, 4, rng)]

# your operator idea: move every individual a little towards the population mean, plus noise
idea = Transport(drift=lambda x, pop: pop.mean()-x, sigma=.1)

report = checks.check_operator(idea, populations, fast=True)
print(report.summary())      # verdicts with plain-language explanations
report.plot("check.png")     # diagnostic panels
```

## Contents

* [Concepts in plain language](concepts.md): populations, operators, step sizes, effects,
  error measures, drift coefficients, discovery guarantees. **Read this first.**
* Guides: [new operator](guide-operator.md), [combining operators](guide-assembly.md),
  [implemented algorithm](guide-algorithm.md).
* [Reading the reports](reading-reports.md): every check, what it tests, what each verdict
  means, what to do about it.
* [Tutorials](../tutorials/): six runnable walk-throughs (scripts, and notebooks in
  `tutorials/notebooks/`), with the figures shown in these pages.
* [FAQ](faq.md) and [API overview](api.md).
* [Theory map](theory-map.md): which result of the paper each check relates to, and what
  a numerical check can and cannot establish.

## What the toolbox does not do

It does not prove anything. Every check examines the populations, runs, or states you
give it; a "consistent" verdict means that nothing contradicts the property there. This
is what you need while an idea is still evolving: it tells you whether an operator is
worth keeping and which property to fix, and it tells you when proving a result has become
worth the effort. The one exception is the measured discovery bound, which is a genuine
statistical guarantee about your algorithm as run (with a stated confidence level).
