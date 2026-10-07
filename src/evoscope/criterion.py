"""The discounted evaluation-complexity criterion of the paper.

With burn-in index k_eps, endpoint m, batch sizes N_k, kernel constant L_K, band
constants (r_0, L_{T,eps}, M_{T,eps}) and the modulus A_N(T), the corollary gives
P(no eps-optimal candidate by t_m) <= delta whenever

    A_disc = (8 L_K / r_*)^2 A_N(T) sum_{j=k_eps}^m exp(-1/4 sum_{k=j+1}^m N_k) <= delta/2
    and     sum_{k=k_eps}^m N_k >= 4 log(2/delta),

with r_* = min(r_0, 1/(16 L_{T,eps} M_{T,eps})).  The paper's practitioner's guide
explains how to obtain each ingredient.
"""
import math

import numpy as np

__all__ = ["boundary_strip_width", "discount_sum", "uniform_discount_bound", "a_disc", "check",
           "required_modulus"]


def boundary_strip_width(r0, L, M):
    """r_* = min(r_0, 1/(16 L M))."""
    return min(r0, 1/(16*L*M))


def discount_sum(batch_sizes):
    """sum_{j} exp(-1/4 sum_{k>j} N_k) for the post-burn-in batch sizes N_{k_eps}, ..., N_m."""
    sizes = np.asarray(batch_sizes, dtype=float)
    later = np.concatenate([np.cumsum(sizes[::-1])[::-1][1:], [0.]])
    return float(np.sum(np.exp(-later/4)))


def uniform_discount_bound(N):
    """Upper bound of the discount sum for constant batches N_k = N: 1/(1 - e^{-N/4})."""
    return 1/(1-math.exp(-N/4))


def a_disc(L_K, r_star, A_N, discount):
    """The left-hand side of the criterion: (8 L_K / r_*)^2 A_N(T) times the discount sum."""
    return (8*L_K/r_star)**2*A_N*discount


def check(A_N, L_K, r0, L, M, batch_sizes, delta):
    """Evaluate both conditions of the criterion; ``batch_sizes`` covers generations k_eps..m."""
    r_star = boundary_strip_width(r0, L, M)
    discount = discount_sum(batch_sizes)
    lhs = a_disc(L_K, r_star, A_N, discount)
    budget = float(np.sum(batch_sizes))
    return {"r_star": r_star, "discount_sum": discount, "a_disc": lhs, "delta_over_2": delta/2,
            "modulus_condition": lhs <= delta/2, "post_burn_in_budget": budget,
            "budget_condition": budget >= 4*math.log(2/delta),
            "satisfied": lhs <= delta/2 and budget >= 4*math.log(2/delta),
            "modulus_needed": required_modulus(L_K, r_star, discount, delta)}


def required_modulus(L_K, r_star, discount, delta):
    """The largest A_N(T) for which the first condition holds."""
    return delta/2/((8*L_K/r_star)**2*discount)
