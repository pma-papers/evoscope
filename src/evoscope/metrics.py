"""Distances between populations.

The theory compares population laws with the population metric
d*(mu, nu) = d_BL,q(mu, nu) + W_2(mu, nu) of the paper: a bounded-Lipschitz
part that sees changes of weight, and a Wasserstein part that sees motion of mass.

* ``w2`` computes the 2-Wasserstein distance between two finite weighted populations
  exactly (linear programming; large supports are subsampled with a warning flag).
* ``bl_lower`` gives a lower bound of the bounded-Lipschitz part from a battery of test
  functions (the supremum over the battery, each normalized by its norm bound).
* ``d_star`` is their sum: a lower bound of d* up to the subsampling of W_2.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import coo_matrix, vstack

__all__ = ["w2", "bl_lower", "d_star"]

MAX_ATOMS = 2500


def _thin(pop, rng, max_atoms):
    if pop.size <= max_atoms:
        return pop.points, pop.weights, False
    idx = rng.choice(pop.size, size=max_atoms, replace=True, p=pop.weights)
    return pop.points[idx], np.full(max_atoms, 1/max_atoms), True


def w2(a, b, rng=None, max_atoms=MAX_ATOMS, return_flag=False):
    """2-Wasserstein distance between two populations (exact up to subsampling of supports
    larger than `max_atoms`, which is reported when ``return_flag`` is set)."""
    rng = np.random.default_rng(0) if rng is None else rng
    pa, wa, ta = _thin(a, rng, max_atoms)
    pb, wb, tb = _thin(b, rng, max_atoms)
    cost = np.sum((pa[:, None, :]-pb[None, :, :])**2, axis=-1)
    if len(pa) == len(pb) and np.allclose(wa, wa[0]) and np.allclose(wb, wb[0]):
        rows, cols = linear_sum_assignment(cost)
        value = cost[rows, cols].mean()
    else:
        na, nb = len(pa), len(pb)
        cols = np.arange(na*nb)
        A = vstack([coo_matrix((np.ones(na*nb), (np.repeat(np.arange(na), nb), cols)), shape=(na, na*nb)),
                    coo_matrix((np.ones(na*nb), (np.tile(np.arange(nb), na), cols)), shape=(nb, na*nb))]).tocsr()
        b_eq = np.concatenate([wa, wb])
        result = linprog(cost.ravel(), A_eq=A[:-1], b_eq=b_eq[:-1], bounds=(0, None), method="highs")
        if not result.success:
            raise RuntimeError(f"transport problem failed: {result.message}")
        value = result.fun
    value = float(np.sqrt(max(value, 0.)))
    return (value, ta or tb) if return_flag else value


def bl_lower(a, b, functions):
    """max over the battery of |<phi, a - b>| / ||phi||_BL: a lower bound of the bounded-Lipschitz distance."""
    return max(abs(a.expect(phi.value(a.points))-b.expect(phi.value(b.points)))/phi.bl_norm for phi in functions)


def d_star(a, b, functions, rng=None):
    """bl_lower + w2: a lower bound of the population metric d*(a, b)."""
    return bl_lower(a, b, functions)+w2(a, b, rng)
