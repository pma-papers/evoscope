"""The measured discovery bound of the paper.

Inputs are probe counts: in each generation j of each of R independent repetitions,
n_probe extra candidates are drawn from the current proposal law (and discarded) and
the number h_ij that fall in the target set {f <= c} is recorded.  A generation is
flagged when the exact lower confidence limit of h_ij / n_probe lies below the
adequate-mass threshold (1/4).  The bound combines the discounted upper confidence
limits of the flag frequencies over the last generations with the residual term rho^m,
rho = (1 - threshold)^batch, and bounds the probability that no batch up to the
endpoint contains a target point.  No mean-field approximation is used.

The computation is identical to the one of the paper's study (`discovery_inference.py`
of the reproduction package); the defaults are the study's.
"""
import math

import numpy as np
from scipy.stats import beta

__all__ = ["clopper_pearson_lower", "clopper_pearson_upper", "inadequate_mass_flags",
           "discovery_bound", "bound_by_endpoint", "measured_bound", "floor",
           "smallest_unflagged_fraction", "DEFAULTS"]

DEFAULTS = dict(alpha=.05, eta_diag=1e-3, threshold=.25, tail_cutoff=1e-3)


def clopper_pearson_lower(successes, trials, error):
    """Exact lower confidence limit for a binomial proportion (0 when successes = 0)."""
    counts = np.asarray(successes)
    if not np.all((0 <= counts) & (counts <= trials)):
        raise ValueError("counts must lie in [0, trials]")
    result = np.zeros(counts.shape, dtype=float)
    positive = counts > 0
    result[positive] = beta.ppf(error, counts[positive], trials-counts[positive]+1)
    return result


def clopper_pearson_upper(successes, trials, error):
    """Exact upper confidence limit for a binomial proportion (1 when successes = trials)."""
    counts = np.asarray(successes)
    if not np.all((0 <= counts) & (counts <= trials)):
        raise ValueError("counts must lie in [0, trials]")
    result = np.ones(counts.shape, dtype=float)
    below = counts < trials
    result[below] = beta.isf(error, counts[below]+1, trials-counts[below])
    return result


def inadequate_mass_flags(probe_counts, n_probe, eta_diag=DEFAULTS["eta_diag"], threshold=DEFAULTS["threshold"]):
    """True where the lower limit of the probe success fraction lies below the threshold.

    Whenever the success mass of the proposal law is below the threshold, the flag fires
    except with probability eta_diag."""
    lookup = clopper_pearson_lower(np.arange(n_probe+1), n_probe, eta_diag)
    return lookup[np.asarray(probe_counts)] < threshold


def discovery_bound(flag_counts, repetitions, batch, endpoint, alpha=DEFAULTS["alpha"],
                    eta_diag=DEFAULTS["eta_diag"], threshold=DEFAULTS["threshold"],
                    tail_cutoff=DEFAULTS["tail_cutoff"]):
    """The measured bound at one endpoint from flag counts over generations 1, ..., m.

    ``flag_counts[..., j-1]`` is the number of the R repetitions flagged in generation j
    (time is the last axis; leading axes are kept).  Returns a dict whose ``upper`` is the
    pointwise (1 - alpha) bound on P(no target point in the batches up to `endpoint`)."""
    flag_counts = np.asarray(flag_counts)
    if not 1 <= endpoint <= flag_counts.shape[-1]:
        raise ValueError("endpoint outside the recorded generations")
    rho = (1-threshold)**batch
    retained = min(endpoint, max(1, math.ceil(math.log(tail_cutoff)/math.log(rho))))
    weights = rho**np.arange(retained)
    last = flag_counts[..., endpoint-retained:endpoint][..., ::-1]
    lookup = clopper_pearson_upper(np.arange(repetitions+1), repetitions, alpha/retained)
    recent = np.sum(weights*np.minimum(1., lookup[last]+eta_diag), axis=-1)
    omitted = rho**retained*(-np.expm1((endpoint-retained)*np.log(rho)))/(1-rho)
    sampling = rho**endpoint
    frequency = flag_counts[..., :endpoint]/repetitions
    all_weights = rho**np.arange(endpoint-1, -1, -1)
    return {"upper": np.minimum(1., sampling+recent+omitted),
            "flag_discounted": np.sum(frequency*all_weights, axis=-1),
            "flag_undiscounted": frequency.sum(axis=-1),
            "sampling": sampling, "omitted_tail": omitted, "retained_generations": retained,
            "rho": rho, "recent_confidence_term": recent}


def bound_by_endpoint(flag_counts, repetitions, batch, **kwargs):
    """The bound at every endpoint m = 1, ..., M (each a pointwise statement)."""
    flag_counts = np.asarray(flag_counts)
    return np.array([float(discovery_bound(flag_counts, repetitions, batch, m, **kwargs)["upper"])
                     for m in range(1, flag_counts.shape[-1]+1)])


def measured_bound(probe_counts, n_probe, batch, endpoint=None, **kwargs):
    """Convenience: the bound from raw probe counts of shape (R, m)."""
    probe_counts = np.asarray(probe_counts)
    eta_diag = kwargs.get("eta_diag", DEFAULTS["eta_diag"])
    threshold = kwargs.get("threshold", DEFAULTS["threshold"])
    flags = inadequate_mass_flags(probe_counts, n_probe, eta_diag, threshold)
    counts = flags.sum(axis=0)
    endpoint = counts.shape[-1] if endpoint is None else endpoint
    result = discovery_bound(counts, probe_counts.shape[0], batch, endpoint, **kwargs)
    result["flag_counts"] = counts
    return result


def floor(repetitions, batch, endpoint, **kwargs):
    """The value of the bound when no generation is flagged: no design can certify below it."""
    return float(discovery_bound(np.zeros(endpoint, dtype=int), repetitions, batch, endpoint, **kwargs)["upper"])


def smallest_unflagged_fraction(n_probe, eta_diag=DEFAULTS["eta_diag"], threshold=DEFAULTS["threshold"]):
    """The smallest probe success fraction for which a generation is not flagged."""
    lower = clopper_pearson_lower(np.arange(n_probe+1), n_probe, eta_diag)
    return int(np.argmax(lower >= threshold))/n_probe
