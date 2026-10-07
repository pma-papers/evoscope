# Reading the reports

Every check returns a `Report`. `report.summary()` prints one line per finding (the check,
the verdict, the key numbers) followed by a plain-language explanation of each verdict;
`report.plot()` draws the diagnostic panels; `report.to_markdown()` gives a table for a
lab notebook; `report["check name"]` returns one finding with its data.

## Verdicts

| Verdict | Meaning | What to do |
|---|---|---|
| **consistent** | the numbers behave as the theory requires on the populations examined | continue; add other kinds of populations if you have not |
| **suspect** | partly: the property holds only slowly, or not on every population | look at the plot and at which populations fail |
| **violated** | clearly not; the corresponding result of the paper does not apply | fix the operator (see the advice in the report) |
| **not resolved** | random variation hides the answer | more replicates, more runs, larger populations |
| **info** | a measurement without a pass/fail criterion (ingredient effects) | interpret with the explanation |

`report.verdict` is the most severe verdict among the findings.

## Operator checks (`check_operator`)

**first-order effect (A1).** The change of population averages per unit step, for step
sizes halving from 0.1 (or from 0.2 for random rules), should settle: the differences
between successive step sizes should shrink. The panel shows these differences against
the step size; they should fall to the left. For random rules a dotted line shows the
noise level; differences below it are not distinguishable from zero.
*Fails when* the rule does not shrink with the step size.

**moment stability (A2).** The population's size after one step (its third moment,
roughly the average of the cubed distance from the origin) divided by its size before,
for populations scaled by 1, 4 and 16 to probe far-away populations. Values near or below
one are fine.
*Fails when* far-away individuals are pushed farther away, e.g. by a plain gradient step
on a steep objective, or by heavy-tailed perturbations.

**near identity (A3).** The distance between the population before and after one step,
against the step size. It should shrink like a power of the step size: about
`tau^0.5` for random perturbations, offspring replacement and reweighting, `tau^1` for
deterministic moves. The distance combines a Wasserstein distance (how far mass moves)
and a bounded-Lipschitz part (how much weight changes).
*Fails when* the rule does not shrink with the step size.

**continuity (A4).** Atoms of each population are moved along a straight path, and the
largest rate of change of the operator's effect is measured with finer and finer
subdivisions of the path. For a continuous (Lipschitz) effect it stops growing once the
subdivision is fine enough; at a jump it grows in proportion to the refinement. The
report gives the growth over the last refinement (about 4 means a jump).
*Fails when* the rule contains a hard threshold: a best-of comparison, a truncation by
rank, an if-statement on the objective value. Smoothed versions pass, with a "steep but
continuous" note when the smoothing width is small.

## Assembly checks (`check_assembly`)

**operator X.** A one-line summary of the operator check of each component.

**composition (sum of effects).** The composed step's change per unit step minus the sum
of the operators' separate effects, against the step size; it should shrink in proportion
to the step. If it does not, one of the operators fails a condition, or two operators
interact through the step size in a way that is not of the small-step form.

**unbounded error measure (cutoff).** The theory measures progress with an unbounded
quantity (the objective gap); it is handled by cutting the quantity off far away and
letting the cut-off radius grow. The effect computed with the cut-off should stop
changing as the radius grows. *Fails when* an operator sends mass extremely far (heavy
tails).

**drift coefficient and attribution.** `lambda_cal`, the smallest total rate of
mean-gap reduction over the populations given, and the smallest rate of each operator.
Positive `lambda_cal`: on every population examined the assembly reduces the mean gap at
least at this relative rate. Negative: on some population it increases it; the operator
with the most negative contribution is named. Validate on fresh populations
(`attribution.validate`) and sample populations from runs, not only initial ones.

## Algorithm checks (`check_algorithm`)

**run diagnostics.** Fractions of runs that end above the level, that collapse (spread
below `1e-10` while the batch mean gap is above the level: agreement at a point that is
not good), that stall (no improvement of the best-so-far over the last 50 generations
while above the level), and that blow up (non-finite values).

**discovery guarantee.** The measured bound on the probability that a run has proposed
no candidate with `f <= level` by the last generation (95% confidence), the observed
frequency, and the floor (the smallest value the bound can take with this many runs).
*consistent*: certified at the level `delta` (default 0.05). *not resolved*: the floor is
too high: more runs or a larger batch. *suspect*: in the last generations the proposal
distribution put less than a quarter of its mass on the target in some runs (or the
probes could not tell): either the algorithm stalls in some runs, or its proposals stay
spread out although its best-so-far is good, or more probes are needed.

**per-generation contraction.** For states from the runs: the mean objective gap `V` of
the proposals, and its average after one generation. If the ratio is below one on every
state, the mean gap decreases geometrically and so does the probability of having no good
point. Otherwise the report gives the residual form `E[V_next] <= rho V + c`: the gap
decreases geometrically down to about the floor `c/(1-rho)`. A floor below your target
accuracy is harmless; persistent noise always produces some floor.

**ingredient effects per generation** (with `variants`). The difference in the next
generation's mean gap between each variant and the full algorithm, with the same random
numbers, averaged over states. Positive: the ingredient that the variant removes or
changes helps the mean gap per generation.

**small-step behaviour** (with `make_alg`). Whether the change of the mean gap per unit
step settles as the step parameter shrinks, and the drift coefficients it implies.
