"""A discovery guarantee from runs: the measured discovery bound.

Run the full assembly on the unequal-wells problem from broad starts for two population
sizes, record probe counts against the target set {f <= 0.1}, and compute the bound on
the probability that no batch up to generation m contains a target point.  Prints the
design table (floor of the bound and probe resolution) first.

    python examples/measured_discovery_bound.py

Writes measured_discovery_bound.pdf next to this script.  About half a minute.
"""
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from evoscope import discovery, problems, runs, viz                    # noqa: E402
from evoscope.operators import paper_assembly                          # noqa: E402
from evoscope.population import broad_gaussian                         # noqa: E402

REPETITIONS, GENERATIONS, N_PROBE, LEVEL, TAU = 64, 120, 128, .1, .1

print("floor of the bound at m = 200 (no generation flagged):")
for R in (64, 128, 256):
    print(f"  R = {R:3d}:", "  ".join(f"N_k={b}: {discovery.floor(R, b, 200):.3f}" for b in (1, 4, 16, 32)))
print("smallest unflagged probe success fraction:",
      {n: round(discovery.smallest_unflagged_fraction(n), 3) for n in (64, 128, 256, 1024)})

problem = problems.get("wells_d4")
assembly = paper_assembly(problem)
propose = assembly.proposer(TAU)
viz.paper_style()
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), layout="constrained")
for N, color in ((16, "#c9b27c"), (64, "#246f96")):
    rng = np.random.default_rng(N)
    initials = [broad_gaussian(N, problem.dimension, rng) for _ in range(REPETITIONS)]
    out = runs.run_repetitions(propose, initials, problem.value, GENERATIONS, LEVEL, rng, N_PROBE)
    bound = discovery.measured_bound(out["probe_counts"], N_PROBE, batch=N)
    observed = 1-out["offspring_hit"][:, 1:].mean(axis=0)
    print(f"N = {N:3d}: bound at m = {GENERATIONS}: {float(bound['upper']):.3f}"
          f"   observed non-discovery: {observed[-1]:.3f}   (R = {REPETITIONS}, pointwise 95%)")
    viz.plot_discovery(out["probe_counts"], N_PROBE, N, axes=[axes[0] if N == 16 else None, axes[1], axes[2]],
                       observed_non_discovery=observed, label=f"N = {N}", color=color, heatmap=N == 16)
axes[0].set_title("probes, N = 16")
axes[1].set_title("flagged fraction")
axes[2].set_title("bound by endpoint")
axes[1].legend(frameon=False)
out_path = Path(__file__).with_name("measured_discovery_bound.pdf")
fig.savefig(out_path)
print("wrote", out_path)
