"""Visualizations of operator roles, drift-coefficient estimates, discovery bounds, and the
terms of the evolution equation (ledgers, level masses, density terms, operator portraits).

Every function draws into given axes (or creates them) and returns the axes, so the
plots can be combined into larger figures.  ``paper_style()`` sets the lettering used
in the paper (9 pt at a 6 in figure width).
"""
import numpy as np
import matplotlib.pyplot as plt

from .discovery import bound_by_endpoint, inadequate_mass_flags

__all__ = ["paper_style", "plot_component_coefficients", "plot_drift_estimates", "plot_residual_form",
           "plot_consistency", "plot_discovery", "plot_contrast", "operator_colors", "plot_ledger", "plot_levels",
           "plot_level_terms", "plot_density_flow", "plot_density_terms", "plot_term_map", "plot_characteristics"]

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


# ----------------------------------------------------------------------------- the evolution equation
OPERATOR_COLORS = {"S": "#347c63", "R": "#c16c23", "D": "#246f96", "H": "#6a4c93"}
_EXTRA = ("#8a2b2b", "#c9b27c", "#3f8f9b", "#7a7a7a", "#b5527a", "#5d6d2f")
TOTAL = "#1b2f44"


def operator_colors(names):
    """A fixed colour for every operator: S, R, D, H as in all EvoScope figures, others in turn."""
    out, extra = {}, iter(_EXTRA*4)
    for name in names:
        out[name] = OPERATOR_COLORS.get(name) or next(extra)
    return out


def _grouped(names, series, groups):
    """(label, values, style) for every operator, with the operators of each group in
    ``groups`` (label -> names) replaced by one dashed line for their sum."""
    groups = groups or {}
    colors = operator_colors(names)
    member = {n: label for label, members in groups.items() for n in members}
    out, done = [], set()
    for j, name in enumerate(names):
        if name not in member:
            out.append((name, series[j], dict(color=colors[name], lw=1.4)))
        elif member[name] not in done:
            label = member[name]
            done.add(label)
            total = sum(series[names.index(n)] for n in groups[label] if n in names)
            out.append((label, total, dict(color="#5b6b7a", lw=1.3, ls="--")))
    return out


def plot_ledger(ledger, index=0, ax=None, kind="contributions", legend=True, groups=None):
    """The terms of the evolution equation along a flow for one quantity of the view
    (output of :func:`evoscope.evolution.ledger`).

    ``kind="contributions"``: the time integral of each operator's term (its share of the
    change since the start), their sum (dark line), and the change actually observed along
    the flow (dots); the evolution equation says the last two agree.  ``kind="rates"``: the
    terms themselves and their sum.  ``groups``: label -> operator names to show as one
    line, e.g. ``{"D + H": ("D", "H")}`` for two large terms that nearly cancel."""
    ax = _axes(ax)
    t = ledger["times"]
    values = ledger["contributions"] if kind == "contributions" else ledger["rates"]
    names = list(ledger["names"])
    for label, series, style in _grouped(names, [values[:, j, index] for j in range(len(names))], groups):
        ax.plot(t, series, label=label, **style)
    total = ledger["predicted"] if kind == "contributions" else ledger["total"]
    ax.plot(t, total[:, index], color=TOTAL, lw=2, label="sum of terms")
    if kind == "contributions":
        ax.plot(t, ledger["change"][:, index], "o", ms=2.6, color=TOTAL, mfc="white", mew=.9, label="observed change")
    ax.axhline(0, color="0.6", lw=.6)
    ax.set_xlabel("time t")
    ax.set_ylabel(("change of " if kind == "contributions" else "rate of ")+ledger["labels"][index])
    if legend:
        ax.legend(frameon=False, fontsize=8)
    return ax


def plot_levels(ledger, ax=None, log=True, cmap="viridis"):
    """The mass below each objective level along a flow (a ledger of
    :func:`evoscope.evolution.level_view`): time on the horizontal axis, level vertically."""
    ax = _axes(ax)
    levels = np.asarray(ledger["views"][0].axis)
    image = ax.pcolormesh(ledger["times"], levels, ledger["value"].T, cmap=cmap, vmin=0, vmax=1, shading="nearest",
                          rasterized=True)
    if log:
        ax.set_yscale("log")
    ax.figure.colorbar(image, ax=ax, label="mass below level", pad=.02)
    ax.set(xlabel="time t", ylabel=r"objective gap level $c$")
    return ax


def plot_level_terms(levels, rates, ax=None, log=True, total=True, legend=True, groups=None):
    """Each operator's term at every objective level: the rate at which it moves mass below
    the level (``rates``: dict name -> array over ``levels``, e.g. from term_rates).
    ``groups`` as in :func:`plot_ledger`."""
    ax = _axes(ax)
    for label, series, style in _grouped(list(rates), list(rates.values()), groups):
        ax.plot(levels, series, label=label, **style)
    if total:
        ax.plot(levels, sum(rates.values()), color=TOTAL, lw=2, label="sum")
    ax.axhline(0, color="0.6", lw=.6)
    if log:
        ax.set_xscale("log")
    ax.set(xlabel=r"objective gap level $c$", ylabel="rate of mass below level")
    if legend:
        ax.legend(frameon=False, fontsize=8)
    return ax


def plot_density_flow(times, grid, densities, ax=None, cmap="magma_r"):
    """A one-dimensional density over time (output of :meth:`evoscope.evolution.TRJ1D.solve`)."""
    ax = _axes(ax)
    image = ax.pcolormesh(times, grid, np.asarray(densities).T, cmap=cmap, shading="nearest", rasterized=True)
    ax.figure.colorbar(image, ax=ax, label="density", pad=.02)
    ax.set(xlabel="time t", ylabel="x")
    return ax


def plot_density_terms(grid, terms, ax=None, total=True, legend=True, groups=None):
    """The terms of the equation for a one-dimensional density: one signed curve per
    operator (``terms``: dict name -> array over ``grid``), positive where the operator adds
    mass and negative where it removes it, and their sum, the rate of change of the density.
    ``groups`` as in :func:`plot_ledger`."""
    ax = _axes(ax)
    for label, series, style in _grouped(list(terms), list(terms.values()), groups):
        ax.plot(grid, series, label=label, **style)
    if total:
        ax.plot(grid, sum(terms.values()), color=TOTAL, lw=1.9, label="sum")
    ax.axhline(0, color="0.6", lw=.6)
    ax.set(xlabel="x", ylabel="rate of change of density")
    if legend:
        ax.legend(frameon=False, fontsize=8)
    return ax


def plot_term_map(view, rates, ax=None, vmax=None, points=None, objective=None, cmap="RdBu_r", colorbar=True):
    """A term of the equation for a two-dimensional (projected) density: the rates of a
    :func:`evoscope.evolution.density_view` on a product grid, red where mass is added and
    blue where it is removed.  ``points``: individuals to overlay; ``objective``: a function
    of (n, 2) points whose contours are drawn."""
    ax = _axes(ax)
    gx, gy = view.axis
    values = np.asarray(rates).reshape(view.shape)
    vmax = float(np.max(np.abs(values))) if vmax is None else vmax
    image = ax.pcolormesh(gx, gy, values, cmap=cmap, vmin=-vmax, vmax=vmax, shading="nearest", rasterized=True)
    if objective is not None:
        X, Y = np.meshgrid(gx, gy)
        Z = objective(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
        ax.contour(gx, gy, np.log1p(Z), levels=8, colors="0.55", linewidths=.5)
    if points is not None:
        ax.plot(points[:, 0], points[:, 1], ".", ms=1.2, color="0.15", alpha=.35)
    if colorbar:
        ax.figure.colorbar(image, ax=ax, pad=.02, shrink=.85)
    ax.set_aspect("equal")
    return ax


def plot_characteristics(chars, ax=None, arrow_scale=None, growth_max=None, diffusion_scale=None,
                         objective=None, extent=None, cmap="PiYG"):
    """Portrait of an operator from :func:`evoscope.evolution.local_characteristics` in two
    dimensions: each individual is a dot coloured by its growth rate (reaction: green grows,
    purple shrinks), with an arrow for its drift (transport) and an ellipse for its
    diffusion matrix (axes proportional to the square roots of its eigenvalues).  Drifts
    estimated from steps are drawn only where they exceed three standard errors."""
    from matplotlib.patches import Ellipse
    ax = _axes(ax)
    x, growth, drift, diffusion = chars["points"], chars["growth"], chars["drift"], chars["diffusion"]
    if objective is not None and extent is not None:
        gx, gy = np.linspace(extent[0], extent[1], 120), np.linspace(extent[2], extent[3], 120)
        X, Y = np.meshgrid(gx, gy)
        Z = objective(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
        ax.contour(gx, gy, np.log1p(Z), levels=8, colors="0.75", linewidths=.5)
    span = np.ptp(x, axis=0).max() if len(x) > 1 else 1.
    eigen = np.linalg.eigvalsh(diffusion)
    if np.max(eigen, initial=0) > 1e-12:
        diffusion_scale = diffusion_scale or .06*span/np.sqrt(np.max(eigen))
        for point, matrix in zip(x, diffusion):
            values, vectors = np.linalg.eigh(matrix)
            values = np.clip(values, 0, None)
            if values.max() > 1e-12:
                angle = np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1]))
                ax.add_patch(Ellipse(point, 2*diffusion_scale*np.sqrt(values[-1]), 2*diffusion_scale*np.sqrt(values[0]),
                                     angle=angle, fc=OPERATOR_COLORS["H"], ec="none", alpha=.22))
    se = np.nan_to_num(chars.get("drift_se", np.zeros_like(drift)))
    shown = np.linalg.norm(drift, axis=1) > np.maximum(3*np.linalg.norm(se, axis=1), 1e-12)
    if shown.any():
        arrow_scale = arrow_scale or np.max(np.linalg.norm(drift[shown], axis=1))/(.12*span)
        ax.quiver(x[shown, 0], x[shown, 1], drift[shown, 0], drift[shown, 1], angles="xy", scale_units="xy",
                  scale=arrow_scale, color=OPERATOR_COLORS["D"], width=.004)
    growth_max = growth_max or max(1e-12, float(np.max(np.abs(growth))))
    ax.scatter(x[:, 0], x[:, 1], c=growth, cmap=cmap, vmin=-growth_max, vmax=growth_max, s=16,
               edgecolors="0.25", linewidths=.4, zorder=3)
    ax.set_aspect("equal")
    return ax
