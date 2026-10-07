"""Attributing performance to operators, and estimating drift coefficients.

For an assembly with components G_1, ..., G_J and a Lyapunov function Upsilon
(for the mean objective gap, Upsilon = f - f_*), the component coefficients

    -G_j[mu](Upsilon) / V(mu),   V(mu) = int Upsilon dmu,

say how much each operator reduces V to first order at the population mu
(positive: reduces it).  Their sum is the quantity that the drift hypothesis of
the convergence theorem, G[mu](Upsilon) <= -lambda V(mu), bounds below by lambda.

This module computes the coefficients over many populations, the generator-based
estimate lambda_cal of the drift coefficient and its validation on fresh populations,
the residual form for families where no positive coefficient exists, the
finite-step consistency check of the composition theorem, and observed rates.
The documentation (docs/guide-assembly.md) describes the procedure.
"""
import numpy as np

__all__ = ["coefficients", "estimate_drift_coefficient", "validate", "residual_constant",
           "consistency", "observed_rates"]


def coefficients(assembly, populations, observable, offset=0., guard=1e-14, rng=None, **kwargs):
    """Component actions and coefficients over a list of populations.

    Returns a dict with ``names`` (operators), ``V`` (P,), ``actions`` (P, J),
    ``generator`` (P,), ``ratios`` (P, J) = -actions/V, ``total`` (P,) = -generator/V,
    and ``valid`` (P,), the populations with V > guard (ratios are NaN elsewhere).
    ``offset`` is subtracted from Upsilon (e.g. a lower bound f_lb when f_* is unknown).
    Operators without a formula for their effect are estimated from finite steps (``rng``
    and keyword arguments are passed to :func:`evoscope.operators.estimate_generator`).
    """
    names = assembly.names
    V = np.array([pop.expect(observable.value(pop.points))-offset for pop in populations])
    actions = np.array([[assembly.generator_actions(pop, observable, rng=rng, **kwargs)[n] for n in names]
                        for pop in populations])
    generator = actions.sum(axis=1)
    valid = V > guard
    ratios = np.full(actions.shape, np.nan)
    ratios[valid] = -actions[valid]/V[valid, None]
    total = np.full(len(V), np.nan)
    total[valid] = -generator[valid]/V[valid]
    return {"names": names, "V": V, "actions": actions, "generator": generator,
            "ratios": ratios, "total": total, "valid": valid}


def estimate_drift_coefficient(total_ratios):
    """lambda_cal: the smallest total coefficient over the sampled populations.

    A positive value is a plug-in drift coefficient for populations like the sampled ones;
    a value <= 0 means that no positive drift coefficient exists on that class.  The sample
    minimum overestimates the infimum over the class: evidence, not a certificate."""
    return float(np.nanmin(total_ratios))


def validate(lambda_cal, new_ratios, n_calibration):
    """Count fresh populations whose total coefficient lies below lambda_cal.

    If fresh and calibration populations are exchangeable, a fresh population falls below
    the minimum of n calibration populations with probability at most 1/(n + 1)."""
    new_ratios = np.asarray(new_ratios)
    finite = np.isfinite(new_ratios)
    violations = int(np.sum(new_ratios[finite] < lambda_cal))
    return {"violations": violations, "populations": int(finite.sum()),
            "frequency": violations/max(1, int(finite.sum())),
            "exchangeability_bound": 1/(n_calibration+1)}


def residual_constant(generator, V, lam):
    """The smallest c with G <= -lam V + c on the sample: max(0, max(G + lam V)).

    With G[mu](Upsilon) <= -lam V(mu) + c on a class that persists, V decays to a floor of
    order c/lam ("decay with a residual" in the paper)."""
    return float(max(0., np.max(np.asarray(generator)+lam*np.asarray(V))))


def consistency(assembly, population, observable, taus, **kwargs):
    """Finite-step check of the composition theorem at one population.

    Returns the finite-step quotients D_tau, the generator prediction G, and the normalized
    remainders |D_tau - G|/(1 + V), which should decrease linearly in tau."""
    taus = np.asarray(taus, dtype=float)
    V = population.expect(observable.value(population.points))
    G = assembly.generator(population, observable)
    quotients = np.array([assembly.finite_step_quotient(population, t, observable, **kwargs) for t in taus])
    return {"taus": taus, "quotients": quotients, "generator": G,
            "normalized_remainder": np.abs(quotients-G)/(1+V)}


def observed_rates(values, times):
    """Descriptive rates of a decaying sequence V(t_0), ..., V(t_m) (e.g. a run average):
    the end-to-end continuous-time rate -log(V_m/V_0)/(t_m - t_0) and the average factor
    per step (V_m/V_0)^(1/m).  Not a bound; compare with a certified or estimated lambda."""
    values, times = np.asarray(values, dtype=float), np.asarray(times, dtype=float)
    ratio = values[-1]/values[0]
    return {"rate": float(-np.log(ratio)/(times[-1]-times[0])),
            "factor_per_step": float(ratio**(1/(len(values)-1)))}
