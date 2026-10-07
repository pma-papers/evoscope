# EvoScope

**See what every operator of your population-based optimizer does, and check whether it
can be trusted.**

EvoScope is a Python toolbox for people who design evolution strategies, genetic
algorithms, particle-swarm and consensus methods, or their own population-based
optimizers. You write update rules in plain Python, and EvoScope:

* **checks a new operator in seconds**: does it behave well enough that its effect can be
  combined with other operators and analysed? If not, it says what is wrong and how to fix
  it, while the idea is still cheap to change;
* **shows what each ingredient contributes**: at any population, how much each operator
  improves (or worsens) the population, and what a new operator adds to an existing
  algorithm, with plots;
* **assesses a whole algorithm without taking it apart**: failure diagnostics, how much one
  generation improves the population, the effect of each switch along whole runs, and a
  **discovery guarantee**: a bound, with stated confidence, on the probability that a run
  has not yet found a good solution.

No mathematics is required to use it: every result comes with a plain-language
explanation of what it means for your algorithm.

![What each operator contributes](docs/figures/t3_attribution.png)

*Adding a new operator `P` to an algorithm made of selection `S`, recombination `R`, a
downhill move `D` and noise `H`: each dot is a population met during a run, and a positive
value means the operator improves that population. With `P` the smallest total rate
("sum") becomes positive.*

## Three ways in

| You have | Use | Guide | Tutorial |
|---|---|---|---|
| an idea for a new operator | `checks.check_operator` | [guide](docs/guide-operator.md) | [02](tutorials/02_checking_a_new_operator.py) |
| operators to combine, or a new operator for an existing algorithm | `checks.check_assembly`, `attribution`, `algorithm.ablation_runs` | [guide](docs/guide-assembly.md) | [03](tutorials/03_from_operators_to_an_algorithm.py) |
| an algorithm that already runs (ask/tell interface) | `algorithm.check_algorithm` | [guide](docs/guide-algorithm.md) | [04](tutorials/04_black_box_algorithm.py) |

New here? Read [the concepts in plain language](docs/concepts.md), then
[Tutorial 1](tutorials/01_first_steps.py).

## Installation

Python 3.10 or newer; depends on NumPy, SciPy and Matplotlib.

```sh
git clone https://github.com/pma-papers/evoscope.git
cd evoscope
pip install -e .
```

## A first check

```python
import numpy as np
from evoscope import checks, problems
from evoscope.operators import Transport
from evoscope.population import broad_gaussian, two_cluster

f = problems.get("wells_d4")                    # a test problem, or your own objective
rng = np.random.default_rng(0)
populations = [broad_gaussian(16, 4, rng), two_cluster(16, 4, rng)]

# an operator idea: move every individual towards the population mean, plus noise
idea = Transport(drift=lambda x, pop: pop.mean()-x, sigma=.1)
report = checks.check_operator(idea, populations, fast=True)
print(report.summary())                          # verdicts with plain-language explanations
report.plot("check.png")
```

```
Operator check: M (2 populations, 8 test functions)
first-order effect (A1)            consistent    ...
moment stability (A2)              consistent    ...
near identity (A3)                 consistent    ...
continuity in the population (A4)  consistent    ...
```

Operators are written in whatever form is closest to how you think of them: a move of each
individual, a reweighting (selection), a generator of offspring (crossover and variation),
or any update function. No derivatives or formulas are needed.

## Documentation

| | |
|---|---|
| [Start here](docs/index.md) | which situation you are in, and a five-minute start |
| [Concepts](docs/concepts.md) | populations, operators, step sizes, effects, error measures, drift coefficients, discovery guarantees, in plain language |
| Guides | [checking a new operator](docs/guide-operator.md), [combining operators](docs/guide-assembly.md), [checking an implemented algorithm](docs/guide-algorithm.md) |
| [Reading the reports](docs/reading-reports.md) | every check: what it tests, what each verdict means, what to do |
| [FAQ](docs/faq.md), [API](docs/api.md) | questions, and the API at a glance |
| [Tutorials](tutorials/) | five runnable walk-throughs, also as notebooks in `tutorials/notebooks/` |

The tutorials: (1) what each operator does; (2) taking an operator idea through three
versions until it passes every check; (3) adding a new operator to an existing algorithm
and measuring what it adds; (4) checking an algorithm written without EvoScope;
(5) discovery guarantees in practice.

## Background

EvoScope implements the operator calculus of

> P. Malo, L. Viitasaari, P. Nummi, A. Suominen, A. Sinha, O. Tahvonen.
> *Operator Calculus for Population-Based Optimization: Modular Convergence and
> Finite-Population Guarantees.* arXiv:2606.14289.

The paper shows that, under four conditions on each operator, the first-order effects of
separately specified operators add up, which makes convergence arguments modular and
performance attributable to operators; it also gives finite-population discovery bounds.
EvoScope's checks are the numerical counterparts of these conditions and results; the
[theory map](docs/theory-map.md) lists the correspondence and what a numerical check
cannot establish. The test suite verifies that EvoScope reproduces the numbers of the
paper's numerical study.

## Repository layout

```
src/evoscope/      the package
  operators.py       operator wrappers (Transport, Reweighting, Jump, StepOperator), ready-made operators, Assembly
  checks.py          operator- and assembly-level checks
  algorithm.py       ask/tell interface, runs with probes, algorithm-level checks, ablation
  attribution.py     effects of operators, drift coefficients, validation
  discovery.py       the discovery guarantee
  report.py          reports with verdicts and plain-language explanations
  population.py, problems.py, testfunctions.py, metrics.py, contrasts.py, viz.py
  criterion.py, band.py, modulus.py, runs.py   advanced: the paper's theorem-level evaluation criterion
docs/              documentation and figures
tutorials/         runnable tutorials and notebooks/
examples/          short examples
tests/             unit tests, including known pass/fail cases for every check
tools/             maintainer tools
```

## Tests

```sh
python -m unittest discover -s tests       # about 10 seconds; pytest also works
```

## Status

Version 0.2.0.dev0, under active development. Planned: ready-made wrappers for common
algorithms (CMA-ES-type, recombinative ES, consensus-based optimization) with their known
guarantees, examples with popular libraries through the ask/tell interface, a
documentation website, and a PyPI release. Questions, issues and contributions are welcome.

## Citation and license

If you use EvoScope, please cite the paper above (see `CITATION.cff`). EvoScope is
released under the [MIT License](LICENSE).
