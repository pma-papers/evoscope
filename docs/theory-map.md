# Theory map: what each check relates to in the paper

For readers who want to connect the toolbox to the mathematics of the paper (see the
README for the reference). Results are named rather than numbered, so that the map does not
depend on a particular version of the paper.

| Toolbox | Result of the paper | What the theory needs | What the toolbox checks |
|---|---|---|---|
| `check_first_order` | composition assumption, condition (A1) | the quotient converges **uniformly** over mass-restricted moment sublevels, for every test function in `C_b^3` | convergence of the quotient over halving step sizes, on the populations and the battery of test functions given (Gaussian bumps and cosine waves adapted to the populations) |
| `check_moment_stability` | composition assumption, condition (A2) | all intermediate measures of a composed step stay in a common moment sublevel | ratio of `int (1 + |x|^3)` after and before one step, on the populations scaled by 1, 4, 16 |
| `check_near_identity` | composition assumption, condition (A3) | `d*(T_tau mu, mu) <= C tau^alpha` uniformly on sublevels | the same distance (Wasserstein part exact up to subsampling; bounded-Lipschitz part a lower bound from the battery) and the fitted exponent |
| `check_continuity` | composition assumption, condition (A4) | the generator is Lipschitz in `d*` for each test function | the largest rate of change of the generator along straight paths of atoms, at increasing resolution |
| `check_composition` | composition theorem (generator of the composition) | (A1)-(A4) for every component | the remainder of the composite quotient against the sum of component generators |
| `check_cutoff` | cutoff-admissible Lyapunov tests; their verification for the canonical generators | the extended action of an unbounded Lyapunov function is the limit of cut-off actions, uniformly bounded | the generator of the cut-off function as the radius grows |
| `attribution.coefficients`, `estimate_drift_coefficient` | drift hypothesis of the mean-field convergence theorem; operator-wise closure | `G[mu](Upsilon) <= -lambda V(mu)` on every law along the solution | the minimum of `-G/V` over sampled populations; `validate` estimates how representative the sample is |
| `attribution.residual_constant` | decay with a residual | `G <= -lambda V + c` on a class that persists | the smallest such `c` on the sample |
| `discovery.measured_bound`, `algorithm.check_discovery` | measured discovery bound; discounted discovery bound | fresh batches i.i.d. from the proposal law given the history | exactly the bound of the paper, from probe counts; a genuine confidence statement |
| `criterion`, `band`, `modulus` | discounted evaluation-complexity criterion and its practitioner's guide | approximation of the finite-particle state law by its mean-field limit | the ingredients of the criterion and the reference-run estimate of the modulus |
| `algorithm.check_contraction` | finite-generation recursion of the recombinative ES and its evaluation count | `E[V_{k+1} | history] <= rho V_k + c` for every state after entry | the ratio of the next-generation proposal gap to the current one on sampled states |
| `algorithm.check_small_step` | composition theorem, applied to the whole update | the update is a small-step operator | settling of the per-unit-step change of the proposal gap |
| `attribution.consistency` | composition theorem; consistency checks of the numerical study | | finite-step quotient against the generator at one population |
| `contrasts.paired_contrasts`, `algorithm.ablation_runs` | dynamic contrasts of the numerical study | | run-wise differences from matched starts, bootstrap bands |

## What a numerical check cannot establish

* **Uniformity.** The theorems require properties uniformly over classes of populations;
  the checks examine finitely many. Add populations of every kind the algorithm meets.
* **All test functions.** (A1), (A3) and (A4) concern all bounded smooth test functions;
  the checks use a battery. A rule that fails only on very sharp features can pass.
* **Invariance.** The drift hypothesis must hold along the whole trajectory. Sampling
  states from runs approximates this; it is not a proof that the algorithm stays where the
  rate is positive.
* **Mean field.** The operator and assembly checks concern the population-level update
  (the paper's mean-field object). Finite-population effects appear in the runs, the
  discovery bound and the algorithm-level checks.

The measured discovery bound is the exception: it is a statistical guarantee about the
algorithm as run, with the stated confidence, under the condition that the batch
candidates are drawn independently from the proposal law given the history.

## Validation against the paper

`tests/test_paper_results.py` checks that EvoScope reproduces the numbers of the paper's
numerical study (generator actions and exact finite steps of the study's assemblies,
drift-coefficient estimates and their validation counts, measured discovery bounds, the
worked example of the evaluation criterion, Wasserstein distances, bootstrap contrasts)
from small extracts of the study's frozen results. The study's own code is distributed with
the paper.
