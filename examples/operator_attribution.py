"""Which operator does what?  First-order attribution of the mean-gap change.

For one test problem, draw broad and two-cluster populations, and for the full assembly
(selection S, midpoint recombination R, derivative-free drift D, Gaussian noise H)
compute the component coefficients -G_j/V: the first-order rate at which each operator
reduces the mean objective gap at that population.  Then estimate the drift coefficient
lambda from calibration populations and check it on fresh ones.

    python examples/operator_attribution.py [problem_id]     (default: quartic_d4)

Writes operator_attribution.pdf next to this script.  Runs in a few seconds.
"""
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from evoscope import attribution, problems, viz                       # noqa: E402
from evoscope.operators import paper_assembly                          # noqa: E402
from evoscope.population import broad_gaussian, two_cluster            # noqa: E402

problem = problems.get(sys.argv[1] if len(sys.argv) > 1 else "quartic_d4")
assembly = paper_assembly(problem, omega=1, gamma=1, sigma=.2)
rng = np.random.default_rng(0)
d = problem.dimension
families = {"broad": lambda: broad_gaussian(32, d, rng), "two-cluster": lambda: two_cluster(32, d, rng)}

viz.paper_style()
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), layout="constrained")
groups = {}
for ax, (name, draw) in zip(axes[:2], families.items()):
    calibration = [draw() for _ in range(32)]
    validation = [draw() for _ in range(128)]
    cal = attribution.coefficients(assembly, calibration, problem)
    val = attribution.coefficients(assembly, validation, problem)
    viz.plot_component_coefficients(cal, ax=ax)
    ax.set_title(f"{problem.name}, {name} populations")
    lam = attribution.estimate_drift_coefficient(cal["total"])
    check = attribution.validate(lam, val["total"], len(calibration))
    groups[name] = (cal["total"], val["total"])
    print(f"{name:12s} lambda_cal = {lam:8.3f}   validation below: {check['violations']}/{check['populations']}"
          f"   (exchangeability bound per population {check['exchangeability_bound']:.3f})")
    print("   minimum component coefficients:",
          {n: round(float(np.nanmin(cal['ratios'][:, j])), 3) for j, n in enumerate(cal["names"])})
viz.plot_drift_estimates(groups, ax=axes[2])
axes[2].set_title("calibration (dark) and validation")
out = Path(__file__).with_name("operator_attribution.pdf")
fig.savefig(out)
print("wrote", out)
