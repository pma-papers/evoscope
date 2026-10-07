"""Estimating the state-law approximation modulus A_N(T) from a reference run.

A_N(T) is the worst expected squared 2-Wasserstein distance between the N-particle
state law and its mean-field limit after burn-in (evaluation-complexity criterion).  Without usable
theorem constants it is measured against a reference population much larger than N
(the reference-run procedure of the paper): the exact W_2^2 between the empirical laws of
the size-N population and the reference population of the same seed, averaged over
seeds, with a bootstrap interval, the i.i.d. floor, and the fitted decay exponent.
"""
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

__all__ = ["w2_squared", "reference_run_estimate", "plug_in"]


def w2_squared(points, reference):
    """Exact W_2^2 between the uniform empirical laws of two point sets.

    Both sets are replicated to their least common multiple of sizes, so that an optimal
    coupling is a permutation (assignment problem, cost O(L^3) for L = lcm(sizes))."""
    points, reference = np.atleast_2d(points), np.atleast_2d(reference)
    if len(reference) % len(points) == 0:
        a, b = np.repeat(points, len(reference)//len(points), axis=0), reference
    else:
        size = math.lcm(len(points), len(reference))
        a = np.repeat(points, size//len(points), axis=0)
        b = np.repeat(reference, size//len(reference), axis=0)
    cost = cdist(a, b, metric="sqeuclidean")
    rows, cols = linear_sum_assignment(cost)
    return cost[rows, cols].mean()


def reference_run_estimate(populations, reference, rng, bootstrap=2000):
    """Seed-averaged W_2^2 against a reference run, for several population sizes.

    ``populations``: dict N -> array (seeds, N, d) of states at one generation;
    ``reference``: array (seeds, N_ref, d) from the same seeds.  Returns, per N, the mean,
    the percentile-bootstrap 95% interval, the median, and the i.i.d. floor (N states drawn
    without replacement from the reference population); and the decay exponent b' of the
    least-squares fit of log mean on log N."""
    n_ref = reference.shape[1]
    out = {}
    for N, pop in populations.items():
        paired = np.array([w2_squared(pop[r], reference[r]) for r in range(len(pop))])
        iid = np.array([w2_squared(reference[r][rng.choice(n_ref, N, replace=False)], reference[r])
                        for r in range(len(pop))])
        boot = np.array([paired[rng.integers(0, len(paired), len(paired))].mean() for _ in range(bootstrap)])
        out[N] = {"mean": float(paired.mean()),
                  "interval_95": (float(np.quantile(boot, .025)), float(np.quantile(boot, .975))),
                  "median": float(np.median(paired)), "iid_floor": float(iid.mean()), "seeds": int(len(paired))}
    sizes = sorted(out)
    if len(sizes) >= 2:
        slope = np.polyfit(np.log(sizes), np.log([out[N]["mean"] for N in sizes]), 1)[0]
        out["exponent"] = float(-slope)
    return out


def plug_in(upper, N, N_ref, exponent):
    """Plug-in for A_N(T): 2 A^+ (1 + (N/N_ref)^b'), from A_N <= 2 E W_2^2(N, N_ref) + 2 A_{N_ref}
    with A_{N_ref} extrapolated by the fitted decay.  A diagnostic unless A_{N_ref} is bounded."""
    return 2*upper*(1+(N/N_ref)**exponent)
