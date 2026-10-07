# Concepts in plain language

This page explains the ideas the toolbox is built on, without formulas beyond what you
would write in code. Each section ends with the names used in the toolbox.

## Population

A population is the set of candidate solutions an algorithm currently holds, possibly
with weights (after selection, some individuals count more than others). The theory
treats it as a *distribution*: a cloud of points with weights adding up to one. What
matters is its shape: where the points are, how spread out they are, how many clusters
it has.

*In the toolbox:* `Population(points, weights)`, `Population.uniform(points)`;
`broad_gaussian` and `two_cluster` make the two kinds of test population used throughout.

## Operator and step size

An operator is one update rule: a mutation, a selection, a crossover, a move towards a
leader, a restart, anything that maps the current population to a new one. The theory
gives every operator a **step size** `tau`, chosen so that a small step changes the
population only a little:

* a **move** displaces individuals by `tau` times a direction (a drift) plus `sqrt(tau)`
  times random noise;
* a **reweighting** multiplies each individual's weight by `exp(-tau * rate)`, so that
  individuals with a high rate lose weight (selection);
* a **jump** replaces a fraction `tau` of the population by new individuals generated
  from it (crossover and any other offspring-generating mechanism).

Almost any update rule can be written in one of these forms. A rule that ignores the step
size (for example, one that always replaces half the population) can usually be
rescaled; the operator checks detect when it has not been.

*In the toolbox:* `Transport` (moves), `Reweighting` (selection), `Jump` (offspring),
`StepOperator` (any rule given as a function of population, step size and random
generator). `Drift`, `GaussianNoise`, `Selection` and `Recombination` are ready-made
special cases.

## Effect of an operator

Pick a quantity that measures how good a population is, for example its **mean objective
gap** (the average of `f(x) - f_*` over the population, where `f_*` is the optimum
value). The **effect** of an operator on that quantity is how fast it changes it per unit
step, for a tiny step. A negative effect on the mean gap means the operator improves the
population on average.

The theory calls the effect the *generator action*. You never need its formula: the
toolbox computes it from formulas where they are known and estimates it from small steps
otherwise.

*In the toolbox:* `operator.generator(population, quantity)`,
`assembly.generator_actions(population, quantity)`.

## Effects add up

The central result: under four conditions on each operator (below), applying several
operators one after another has, for small steps, an effect equal to the **sum** of their
separate effects. Two consequences matter in practice:

* the change produced by an algorithm step can be **attributed** to its operators, at any
  population: which operator helps, which costs, and how this depends on the shape of the
  population;
* a convergence argument can be assembled **operator by operator**: each operator
  contributes a term, and when you replace one operator you only need to redo its term.

*In the toolbox:* `Assembly([op1, op2, ...])` applies operators in order;
`attribution.coefficients` computes every operator's contribution over many populations;
`checks.check_composition` checks that the effects add up.

## The four conditions on an operator

| Condition | Question | Typical way to fail |
|---|---|---|
| (A1) first-order effect | Does a tiny step have a well-defined effect per unit step? | the rule does not shrink with `tau` |
| (A2) moment stability | Does one step keep the population from blowing up? | moves that grow faster than the distance (e.g. plain gradient steps on steep objectives), heavy-tailed noise |
| (A3) near identity | Does a small step change the population only a little? | the rule does not shrink with `tau` |
| (A4) continuity | Does the effect change smoothly when the population changes slightly? | hard thresholds: best-of comparisons, truncation by rank |

*In the toolbox:* `checks.check_operator` tests all four on populations you choose.

## Error measures and targets

"Converging" can mean three different things, and an algorithm can achieve one without
the others:

* **one good solution found**: the best point evaluated so far is good (measured by the
  *best-so-far gap*);
* **reliably good candidates**: the current population is good on average (the *mean
  gap*);
* **agreement at an optimum**: the population is concentrated near one optimal point
  (its *spread* and its *distance* to the optimum).

A population can agree on a point that is not optimal (small spread, large gap). An
exploratory operator can make the mean gap worse and still produce a better best-so-far.
The toolbox reports these quantities separately and the guides say which to look at.

## Drift coefficient

If, at every population the algorithm meets, its step reduces the mean gap at least at a
relative rate `lambda` (the mean gap shrinks by at least the fraction `lambda` per unit
time), then the mean gap decays exponentially and the probability that the population
contains no good point decays with it. That rate is the **drift coefficient**; a positive
value is the key hypothesis of the convergence theorem.

The toolbox estimates it as the smallest rate over sampled populations (`lambda_cal`),
and checks the estimate on fresh populations. A negative estimate tells you which
operator is responsible, at which populations. The estimate is evidence about populations
like the sampled ones, so sample populations from your runs, not only initial ones.

*In the toolbox:* `attribution.estimate_drift_coefficient`, `attribution.validate`,
`attribution.residual_constant` (when the rate is positive only above a floor).

## Finite populations and the discovery guarantee

Real algorithms use finitely many individuals, so their behaviour is random. The
**measured discovery bound** answers the practical question directly: *with what
probability has a run proposed at least one good candidate by generation `m`?* It is
computed from **probes**: in every generation of every run, a few extra candidates are
drawn from the algorithm's proposal distribution, evaluated, counted and discarded. If the
proposal distribution puts at least a quarter of its probability on good points, a batch
of `N` candidates misses them with probability at most `(3/4)^N`; the bound adds up how
often this was not the case, with exact confidence limits. It needs no approximation
argument and no knowledge of the optimum beyond the definition of "good".

*In the toolbox:* `algorithm.run_repetitions(..., level=c, n_probe=128)` records probes;
`algorithm.check_discovery` and `discovery.measured_bound` compute the bound;
`discovery.floor` says how small it can get with a given number of runs.

## Evidence, not proof

All checks are numerical and examine the populations, runs and states you provide. A
"consistent" verdict means nothing contradicts the property there. Choose populations of
the kinds your algorithm meets: broad and clustered, far from and near to the optimum,
and populations from late generations of real runs.
