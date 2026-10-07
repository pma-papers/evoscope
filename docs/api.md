# API overview

Every function has a docstring with the details (`help(evoscope.checks.check_operator)`).

## Building blocks

| Module | Main names | Purpose |
|---|---|---|
| `evoscope.population` | `Population(points, weights)`, `Population.uniform`, `broad_gaussian`, `two_cluster` | weighted populations; standard test populations |
| `evoscope.problems` | `panel()`, `get(id)`, `Problem` | the paper's ten test problems, with gradients, Laplacians and Gaussian expectations |
| `evoscope.operators` | `Transport`, `Reweighting`, `Jump`, `StepOperator` | wrappers for your own operators (move, selection, offspring, any rule) |
| | `Drift`, `GaussianNoise`, `Selection`, `Recombination` | ready-made canonical operators |
| | `Assembly(operators)` | operators applied in order; `generator_actions`, `step`, `sample`, `proposer` |
| | `Observable(value, gradient=None, laplacian=None)` | a test function or error measure from plain functions |
| | `estimate_generator` | the effect of any operator, estimated from finite steps |
| | `paper_assembly`, `normalized_fd_drift`, `objective_rate` | the operators of the paper's study |

## Checks and reports

| Function | Level | Returns |
|---|---|---|
| `checks.check_operator(op, populations, fast=False)` | operator | report with (A1)-(A4) |
| `checks.check_first_order`, `check_moment_stability`, `check_near_identity`, `check_continuity` | operator | one finding each |
| `checks.check_assembly(assembly, populations, objective)` | assembly | report: operators, composition, cutoff, drift coefficient and attribution |
| `checks.check_composition`, `check_cutoff` | assembly | one finding each |
| `algorithm.check_algorithm(alg, f, level, rng, ...)` | algorithm | report: diagnostics, discovery, contraction, (ablation, small step) |
| `algorithm.check_discovery`, `check_contraction`, `check_ablation`, `check_small_step`, `diagnose_runs` | algorithm | one finding each |

A `Report` has `summary()`, `plot(path)`, `to_markdown()`, `verdict`, and indexing by
check name; a `Finding` has `check`, `reference`, `statistic`, `verdict`, `meaning`, `data`.

## Algorithms and runs

| Name | Purpose |
|---|---|
| `algorithm.FunctionalAlgorithm(initialize, ask, tell, batch_size, population=None)` | an algorithm from three functions |
| `algorithm.AskTellAlgorithm(factory, batch_size, reseed=None)` | an optimizer object with `ask(n)` / `tell(X, f)` |
| `algorithm.AssemblyAlgorithm(assembly, tau, initial, population_size)` | an assembly run as an algorithm |
| `algorithm.run`, `run_repetitions(..., level=, n_probe=, keep_states=)` | runs with traces, probes and saved states |
| `algorithm.collect_states` | states from pilot runs at chosen generations |
| `algorithm.ablation_runs(variants, reference, f, generations, repetitions)` | paired contrasts along runs |

## Analysis

| Module | Main names |
|---|---|
| `evoscope.attribution` | `coefficients`, `estimate_drift_coefficient`, `validate`, `residual_constant`, `consistency`, `observed_rates` |
| `evoscope.discovery` | `measured_bound`, `bound_by_endpoint`, `floor`, `smallest_unflagged_fraction`, `inadequate_mass_flags` |
| `evoscope.contrasts` | `paired_contrasts` |
| `evoscope.criterion` | `check`, `boundary_strip_width`, `discount_sum`, `a_disc`, `required_modulus` (the evaluation-complexity criterion) |
| `evoscope.band` | `band_constants`, `gaussian_density_bound`, `coarea_density_bound`, `empirical_density_bound` |
| `evoscope.modulus` | `w2_squared`, `reference_run_estimate`, `plug_in` |
| `evoscope.metrics` | `w2`, `bl_lower`, `d_star` (distances between populations) |
| `evoscope.testfunctions` | `battery`, `gaussian_bump`, `cosine_wave`, `truncated` |
| `evoscope.viz` | `plot_component_coefficients`, `plot_drift_estimates`, `plot_residual_form`, `plot_consistency`, `plot_discovery`, `plot_contrast`, `paper_style` |
