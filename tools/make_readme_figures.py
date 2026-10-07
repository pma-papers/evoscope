"""Draw the two figures of the README (docs/figures/readme_*.png) and print the numbers quoted there.

    python tools/make_readme_figures.py

readme_effects.png        the operators of one algorithm step, their contributions, and how these depend on
                          the shape of the population (setting of Tutorial 1)
readme_new_operator.png   adding a new operator to an algorithm: its contribution on the populations met during
                          runs, and the measured discovery bound with and without it (setting of Tutorial 3)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
FIGURES = ROOT/"docs/figures"

from evoscope import algorithm, attribution, discovery, problems, viz
from evoscope.operators import (Assembly, Drift, GaussianNoise, Recombination, Selection, Transport,
                                normalized_fd_drift, objective_rate)
from evoscope.population import broad_gaussian, two_cluster

viz.paper_style()
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5})
GOOD, BAD, TOTAL = viz.PALETTE[2], viz.PALETTE[3], "#1d2b3a"


def contribution_bars(ax, result, labels):
    """Median rate of each operator and of the whole step, with the range over the populations."""
    columns = [result["ratios"][:, j] for j in range(len(result["names"]))]+[result["total"]]
    for k, values in enumerate(columns):
        values = values[np.isfinite(values)]
        median, low, high = np.median(values), values.min(), values.max()
        color = TOTAL if k == len(columns)-1 else (GOOD if median >= 0 else BAD)
        ax.bar(k, median, width=.62, color=color, alpha=.9)
        ax.plot([k, k], [low, high], color="0.15", lw=1.1)
        ax.plot([k-.12, k+.12], [low, low], color="0.15", lw=1.1)
        ax.plot([k-.12, k+.12], [high, high], color="0.15", lw=1.1)
    ax.axhline(0, color="0.3", lw=.8)
    ax.axvline(len(columns)-1.5, color="0.75", lw=.8, ls=":")
    ax.set_xticks(range(len(columns)), labels+["whole\nstep"])
    ax.set_ylabel("contribution to improvement\n(> 0 helps, < 0 hurts)")


def effects_figure():
    rng = np.random.default_rng(0)
    f = problems.get("quartic_d4")
    step = Assembly([Selection(objective_rate(f.value), name="S"), Recombination(name="R"),
                     Drift(normalized_fd_drift(f.value), name="D"), GaussianNoise(.2, name="H")])
    groups = {"one broad cloud": [broad_gaussian(32, 4, rng) for _ in range(48)],
              "two clusters at different optima": [two_cluster(32, 4, rng) for _ in range(48)]}
    labels = ["selection", "recom-\nbination", "downhill\nmove", "noise"]

    fig = plt.figure(figsize=(9.2, 5.6), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.35])
    u = np.linspace(-2.2, 2.2, 200)
    X1, X2 = np.meshgrid(u, u)
    slice_values = (X1**2-1)**2+(X2**2-1)**2
    for col, (name, pops) in enumerate(groups.items()):
        sketch = fig.add_subplot(grid[0, col])
        sketch.contour(X1, X2, np.log1p(slice_values), levels=9, colors="0.78", linewidths=.7)
        sketch.scatter(*np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]).T, marker="*", s=70, color="0.45",
                       zorder=2, label="optima")
        points = pops[0].points
        sketch.scatter(points[:, 0], points[:, 1], s=14, color=viz.PALETTE[0], zorder=3, label="population")
        if col == 1:
            a = points[np.argmax(points[:, 0]+points[:, 1])]
            b = points[np.argmin(points[:, 0]+points[:, 1])]
            middle = (a+b)/2
            sketch.plot([a[0], b[0]], [a[1], b[1]], ls="--", color=BAD, lw=1)
            sketch.scatter([middle[0]], [middle[1]], marker="X", s=80, color=BAD, zorder=4)
            sketch.annotate("offspring of parents from\ndifferent clusters: a poor point", middle[:2],
                            xytext=(.35, -1.75), fontsize=9, color=BAD,
                            arrowprops=dict(arrowstyle="->", color=BAD, lw=.9))
        sketch.set(xlim=(-2.2, 2.2), ylim=(-2.2, 2.2), aspect="equal", xticks=[], yticks=[],
                   title=f"population: {name}")
        sketch.set_xlabel("first two coordinates of a 4-dimensional problem", fontsize=8.5, color="0.35")
        for spine in sketch.spines.values():
            spine.set_visible(False)
        if col == 0:
            sketch.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9, handletextpad=.2)

        bars = fig.add_subplot(grid[1, col])
        result = attribution.coefficients(step, pops, f)
        contribution_bars(bars, result, labels)
        bars.set_title(f"what each operator does to {name.split(' at')[0]}", fontsize=10.5)
        print(name, "median rates:", {n: round(float(np.median(result["ratios"][:, j])), 2)
                                      for j, n in enumerate(result["names"])},
              "whole step: median", round(float(np.median(result["total"])), 2),
              "min", round(float(np.min(result["total"])), 2))
    fig.savefig(FIGURES/"readme_effects.png", dpi=150)
    plt.close(fig)


def new_operator_figure():
    f = problems.get("wells_d4")

    def towards_soft_best(x, pop, alpha=5.):
        values = f.value(pop.points)
        weights = pop.weights*np.exp(-alpha*(values-values.min()))
        return weights@pop.points/weights.sum()-x

    S, R = Selection(objective_rate(f.value), name="S"), Recombination(name="R")
    D, H = Drift(normalized_fd_drift(f.value), name="D"), GaussianNoise(.2, name="H")
    P = Transport(drift=towards_soft_best, name="P")
    variants = {"without the new operator": Assembly([S, R, D, H]), "with the new operator": Assembly([S, R, D, P, H])}
    runs, R_runs, batch, n_probe = {}, 64, 32, 128
    for name, assembly in variants.items():
        alg = algorithm.AssemblyAlgorithm(assembly, .1, lambda g: broad_gaussian(32, 4, g), batch)
        runs[name] = algorithm.run_repetitions(alg, f.value, 80, R_runs, np.random.default_rng(3), level=.1,
                                               n_probe=n_probe, keep_states={0, 10, 40, 80})

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.9), layout="constrained",
                                      gridspec_kw={"width_ratios": [1.15, 1]})
    with_P = "with the new operator"
    states = [s for row in runs[with_P]["states"] for s in row]
    result = attribution.coefficients(variants[with_P], states, f)
    contribution_bars(left, result, ["selec-\ntion", "recomb-\nination", "downhill\nmove", "NEW:\npull to\nbest", "noise"])
    left.get_xticklabels()[3].set_color(viz.PALETTE[0])
    left.get_xticklabels()[3].set_fontweight("bold")
    left.set_title("contributions on the populations met during runs")
    print(with_P, f"- {len(states)} populations; median contributions:",
          {n: round(float(np.nanmedian(r)), 3) for n, r in zip(result["names"], result["ratios"].T)},
          "smallest:", {n: round(float(np.nanmin(r)), 3) for n, r in zip(result["names"], result["ratios"].T)})
    for name in variants:
        states_v = [s for row in runs[name]["states"] for s in row]
        totals = result["total"] if name == with_P else attribution.coefficients(variants[name], states_v, f)["total"]
        print(name, "- smallest rate of the whole step (drift coefficient):", round(float(np.nanmin(totals)), 3))

    generations = np.arange(1, 81)
    for name, color in zip(variants, (viz.PALETTE[1], viz.PALETTE[0])):
        counts = runs[name]["probe_counts"]
        flags = discovery.inadequate_mass_flags(counts, n_probe).sum(axis=0)
        bound = discovery.bound_by_endpoint(flags, R_runs, batch)
        observed = 1-runs[name]["offspring_hit"][:, 1:].mean(axis=0)
        right.plot(generations, bound, color=color, lw=1.8, label=name.replace("the ", ""))
        right.plot(generations, observed, ":", color=color, lw=1.3)
        print(name, "- bound at generations 20, 40, 80:", [round(float(bound[g-1]), 3) for g in (20, 40, 80)],
              "- observed:", [round(float(observed[g-1]), 3) for g in (20, 40, 80)])
    right.plot([], [], "-", color="0.4", lw=1.8, label="guaranteed bound")
    right.plot([], [], ":", color="0.4", lw=1.3, label="observed in the runs")
    floor = discovery.floor(R_runs, batch, 80)
    right.set(xlabel="generation", ylim=(-.02, 1.02), xlim=(0, 81),
              ylabel="chance that a run has not yet\nproposed a good solution")
    right.set_title("discovery guarantee (95% confidence)")
    right.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, .62))
    print("floor:", round(floor, 3))
    fig.savefig(FIGURES/"readme_new_operator.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    effects_figure()
    new_operator_figure()
