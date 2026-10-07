# Frequently asked questions

### Does a "consistent" verdict mean my algorithm converges?

No. It means that the property checked holds on the populations examined. Convergence in
the sense of the paper needs, in addition, that the algorithm's populations stay where the
drift coefficient is positive. The checks make convergence *plausible* and point at what
to prove; the measured discovery bound gives a direct guarantee about your algorithm as
run.

### My operator replaces the whole population every generation. Can it be checked?

Write the part that generates new individuals as a `Jump` (it then replaces a fraction
`tau` of the population) or as a `Transport` (it then moves individuals by `tau` times a
drift). Running the algorithm with `tau = 1` recovers a full replacement; the theory
describes the small-`tau` behaviour, and the composition check shows how far that
description extends. If you prefer not to rewrite anything, use the algorithm-level
checks.

### Does this apply to gradient descent?

Yes. Gradient descent is a `Drift` with `b(x) = -grad f(x)` and step size equal to the
learning rate; a population of one individual reproduces it exactly, and several
individuals give multi-start gradient descent. Its drift coefficient is the constant of
the Polyak-Lojasiewicz inequality, the moment-stability check plays the role of its step-size condition,
and adding `GaussianNoise` gives noisy gradient descent with a floor described by the
residual form. Tutorial 6 works through these cases and combines gradient steps with
selection.

### Which populations should I use for the checks?

Several of each kind your algorithm meets: broad clouds at the start, clustered
populations, populations near a local and near the global optimum, and populations saved
from late generations of real runs (`algorithm.collect_states`). For the drift
coefficient, populations from runs matter most.

### How long do the checks take?

`check_operator(..., fast=True)` takes about a second for populations of 16-32 points;
without `fast`, a few seconds. `check_assembly` is the operator checks plus a few seconds.
`check_algorithm` is dominated by the runs: `repetitions x generations x (batch + probes)`
objective evaluations.

### What if I do not know the optimum?

Nothing in the algorithm needs it; only the definition of the target does. With a lower
bound `f_lb <= f_*` (from a relaxation, or 0 for a nonnegative objective), use
`level = f_lb + eps`: a guarantee for `{f <= f_lb + eps}` is a guarantee for the true
target, because that set is contained in `{f <= f_* + eps}`. With a reference value
(the best known value, or a competitor's result), use `level = f_ref + eps`. For the drift
coefficient, use `offset=f_lb` in `attribution.coefficients`: the rate is then relative to
the lower bound and cannot stay positive arbitrarily close to the optimum, so use the
residual form there.

### My objective has no gradient. Does that matter?

No. Operators are checked from their update rules alone. The ready-made `Drift` and
`GaussianNoise` use the gradient and Laplacian of the objective only to compute their
effect by formula; wrap your objective as `Observable(f)` and these are approximated by
finite differences, or write the operator as a `Transport` with a noise function and let
the toolbox estimate the effect.

### Why is the discovery bound much larger than the failure rate I observe?

See Tutorial 5. Either the number of runs is too small (the floor is too high), the probes
cannot resolve the success probability, or the proposals really put little mass on the
target in late generations (for example because of a large noise level) even though the
best point found is good.

### The continuity check says "violated" for my rank-based selection.

Ranks computed by counting are step functions of the objective values: when two
individuals swap order, the weights jump. Use smoothed ranks, e.g. replace the indicator
`f(y) <= f(x)` by `1/(1 + exp(-(f(x) - f(y))/eta))` with a width `eta` comparable to the
spread of objective values (Tutorial 2 shows the same fix for a best-of comparison).

### Can I use the toolbox with an existing library (pycma, nevergrad, DEAP, ...)?

Any optimizer with an ask/tell interface can be wrapped with `AskTellAlgorithm`. Probes
are drawn from a copy of the optimizer object; if the object keeps its own random
generator, give the copy fresh randomness with the `reseed` argument.
