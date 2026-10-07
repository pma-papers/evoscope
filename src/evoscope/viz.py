"""Visualizations of operator roles, drift-coefficient estimates, and discovery bounds.

Every function draws into given axes (or creates them) and returns the axes, so the
plots can be combined into larger figures.  ``paper_style()`` sets the lettering used
in the paper (9 pt at a 6 in figure width).
"""
import numpy as np
import matplotlib.pyplot as plt

from .discovery import bound_by_endpoint, inadequate_mass_flags

__all__ = ["paper_style", "plot_component_coefficients", "plot_drift_estimates", "plot_residual_form",
           "plot_consistency", "plot_discovery", "plot_contrast"]

DARK, LIGHT, ALERT = "#1b2f44", "#7fb3d5", "#c0392b"
PALETTE = ("#246f96", "#c16c23", "#347c63", "#8a2b2b", "#6a4c93", "#c9b27c")


def paper_style():
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 9,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})


def _axes(ax):
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3), layout="constrained")
    return ax


def _strip(ax, x, values, color, rng, width=.14, size=2.6, marker="o"):
    values = np.asarray(values)
    ax.plot(x+rng.uniform(-width, width, len(values)), values, marker, ms=size, color=color, alpha=.75, lw=0)


def plot_component_coefficients(result, ax=None, include_total=True, seed=0):
    """Component coefficients -G_j/V over populations (output of
    :func:`evoscope.attribution.coefficients`): one column per operator, bars at the minima.
    Positive values reduce the mean of the test function to first order."""
    ax, rng = _axes(ax), np.random.default_rng(seed)
    names, ratios = list(result["names"]), result["ratios"][result["valid"]]
    columns = [ratios[:, j] for j in range(len(names))]
    if include_total:
        names, columns = names+["sum"], columns+[result["total"][result["valid"]]]
    for j, column in enumerate(columns):
        _strip(ax, j, column, DARK if names[j] == "sum" else PALETTE[0], rng)
        ax.hlines(np.min(column), j-.32, j+.32, color="k", lw=1.2)
    ax.axhline(0, color="0.4", lw=.6)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel(r"$-G_j[\mu](\Upsilon)/V(\mu)$")
    return ax


def plot_drift_estimates(groups, ax=None, seed=0):
    """Calibration and validation total coefficients with lambda_cal per group.

    ``groups``: dict label -> (calibration ratios, validation ratios)."""
    ax, rng = _axes(ax), np.random.default_rng(seed)
    for x, (label, (cal, val)) in enumerate(groups.items()):
        cal, val = np.asarray(cal), np.asarray(val)
        lam = np.nanmin(cal)
        below = val < lam
        _strip(ax, x-.2, cal, DARK, rng, width=.09, size=3)
        _strip(ax, x+.2, val[~below], LIGHT, rng, width=.09, size=2.4)
        _strip(ax, x+.2, val[below], ALERT, rng, width=.09, size=5, marker="x")
        ax.hlines(lam, x-.42, x+.42, color="k", lw=1.2)
    ax.set_xticks(range(len(groups)), list(groups))
    ax.set_xlim(-.6, len(groups)-.4)
    ax.set_ylabel(r"$-G[\mu](\Upsilon)/V(\mu)$")
    return ax


def plot_residual_form(G_cal, V_cal, lam, c, G_val=None, V_val=None, ax=None):
    """Generator actions against V with the line G = -lam V + c of the residual form."""
    ax = _axes(ax)
    if G_val is not None:
        G_val, V_val = np.asarray(G_val), np.asarray(V_val)
        above = G_val+lam*V_val > c
        ax.plot(V_val[~above], G_val[~above], "o", ms=2.4, color=LIGHT, alpha=.7)
        ax.plot(V_val[above], G_val[above], "x", ms=5, mew=1.2, color=ALERT)
    ax.plot(V_cal, G_cal, "o", ms=3, color=DARK, alpha=.85)
    all_V = np.concatenate([np.ravel(V_cal)]+([np.ravel(V_val)] if V_val is not None else []))
    grid = np.linspace(all_V.min()*.9, all_V.max()*1.05, 50)
    ax.plot(grid, -lam*grid+c, color="k", lw=1.2)
    ax.set_xlabel(r"$V(\mu)$")
    ax.set_ylabel(r"$G[\mu](\Upsilon)$")
    return ax


def plot_consistency(taus, remainders, labels=None, ax=None):
    """Normalized remainders |D_tau - G|/(1 + V) against the step, with a slope-one guide."""
    ax = _axes(ax)
    remainders = np.atleast_2d(remainders)
    for i, r in enumerate(remainders):
        ax.loglog(taus, r, marker="o", ms=2.5, lw=1.1, color=PALETTE[i % len(PALETTE)],
                  label=None if labels is None else labels[i])
    anchor = np.max(remainders[:, 0])*1.6
    ax.loglog(taus, anchor*np.asarray(taus)/taus[0], ":", color="0.45", lw=.9, label="slope 1")
    ax.set_xlabel(r"step $\tau$")
    ax.set_ylabel(r"$|D_\tau-G[\mu]|/(1+V)$")
    ax.legend(frameon=False)
    return ax


def plot_discovery(probe_counts, n_probe, batch, axes=None, observed_non_discovery=None, level=.05,
                   label=None, color=None, heatmap=True):
    """The steps of the measured discovery bound for one configuration.

    ``probe_counts``: array (R, m).  Panels: (a) probe success fractions by repetition and
    generation, (b) fraction of flagged repetitions, (c) the bound at every endpoint, with
    the observed non-discovery frequency (optional, length m) dotted."""
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
    probe_counts = np.asarray(probe_counts)
    R, m = probe_counts.shape
    color = color or PALETTE[0]
    flags = inadequate_mass_flags(probe_counts, n_probe).sum(axis=0)
    generations = np.arange(1, m+1)
    if heatmap and axes[0] is not None:
        order = np.argsort(probe_counts[:, -min(50, m):].mean(axis=1))
        image = axes[0].imshow(probe_counts[order]/n_probe, aspect="auto", origin="lower", cmap="viridis",
                               vmin=0, vmax=1, extent=(.5, m+.5, .5, R+.5), interpolation="nearest")
        axes[0].figure.colorbar(image, ax=axes[0], location="bottom", pad=.03, shrink=.9,
                                label="probe success fraction")
        axes[0].set(xlabel="generation", ylabel="repetition (sorted)")
    axes[1].plot(generations, flags/R, color=color, lw=1.4, label=label)
    axes[1].set(xlabel="generation", ylabel="flagged fraction", ylim=(-.02, 1.02))
    axes[2].plot(generations, bound_by_endpoint(flags, R, batch), color=color, lw=1.4, label=label)
    if observed_non_discovery is not None:
        axes[2].plot(generations, observed_non_discovery, ":", color=color, lw=1.1)
    axes[2].axhline(level, color="0.4", lw=.8, ls="--")
    axes[2].set(xlabel="endpoint", ylabel="bound on non-discovery", ylim=(-.02, 1.02))
    return axes


def plot_contrast(generations, mean, low, high, ax=None, label=None, color=None, log_time=True):
    """A paired ablation contrast (variant minus reference) with its bootstrap band."""
    ax = _axes(ax)
    color = color or PALETTE[0]
    ax.fill_between(generations, low, high, color=color, alpha=.18, lw=0)
    ax.plot(generations, mean, color=color, lw=1.3, label=label)
    ax.axhline(0, color=".25", lw=.6, ls=":")
    if log_time:
        ax.set_xscale("function", functions=(np.log1p, np.expm1))
        top = float(np.max(generations))
        ax.set_xlim(0, top)
        ticks = [t for t in (0, 1, 5, 10, 20, 50, 100, 200, 500, 1000) if t <= top]
        ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlabel("generation")
    return ax
