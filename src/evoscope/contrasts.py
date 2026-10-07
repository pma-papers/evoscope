"""Ablation contrasts along runs: the effect of removing one operator.

Run the full assembly and the assembly with one component removed from the same
initial populations (matched starts), take the difference of a trace (mean gap,
best-so-far gap, variance, ...) run by run, and average.  The pointwise 95%
percentile-bootstrap interval resamples the runs, with the same resample for every
variant, generation and metric.  This is the dynamic counterpart of the first-order
attribution: along a run, removing an operator also changes the populations on which
the others act.
"""
import numpy as np

__all__ = ["paired_contrasts"]


def paired_contrasts(reference, variants, rng, bootstrap=2000):
    """Run-wise differences variant minus reference.

    ``reference``: array (runs, T, M); ``variants``: array (runs, V, T, M), with run r of
    every variant started from the same initial population as run r of the reference.
    Returns the mean difference (V, T, M) and the pointwise 2.5% and 97.5% bootstrap
    quantiles of the mean, and the run-wise differences themselves."""
    reference, variants = np.asarray(reference, dtype=float), np.asarray(variants, dtype=float)
    paired = variants-reference[:, None]
    n = len(paired)
    weights = rng.multinomial(n, np.full(n, 1/n), size=bootstrap)/n
    boot = (weights@paired.reshape(n, -1)).reshape(bootstrap, *paired.shape[1:])
    low, high = np.quantile(boot, [.025, .975], axis=0)
    return {"mean": paired.mean(axis=0), "low": low, "high": high, "paired": paired}
