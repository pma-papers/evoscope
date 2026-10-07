"""Running an assembly as an evolving population, with probes for the discovery bound.

Each generation draws N offspring from the current proposal law T_tau mu (the new
population, equally weighted) and, before that, `n_probe` probes from the same law,
which are only counted against the target set {f <= level} and then discarded, so the
trajectory is the one that would have run without them.  The probe counts are the
input of :func:`evoscope.discovery.measured_bound`.

Any proposal mechanism can be run this way: pass a function
``propose(population, size, rng) -> points`` instead of an assembly.
"""
import numpy as np

from .population import Population

__all__ = ["run", "run_repetitions"]


def run(propose, initial, objective, generations, level, rng, n_probe=128):
    """One run.  ``propose(population, size, rng) -> points`` draws candidates from the current
    proposal law: ``assembly.proposer(tau)`` for an :class:`~evoscope.operators.Assembly`, or any
    proposal function of the user's algorithm.  Returns per-generation traces (index 0 is the initial population):
    ``mean_gap``, ``best_so_far``, ``variance``, ``offspring_hit`` (a target point among the
    offspring so far), and ``probe_counts`` (generations 1..m)."""
    population = initial if isinstance(initial, Population) else Population.uniform(initial)
    N = population.size
    values = objective(population.points)
    best = float(values.min())
    traces = {k: np.empty(generations+1) for k in ("mean_gap", "best_so_far", "variance")}
    hit = np.zeros(generations+1, dtype=bool)
    probes = np.zeros(generations, dtype=int)
    traces["mean_gap"][0], traces["best_so_far"][0] = population.expect(values), best
    traces["variance"][0] = population.variance()
    for k in range(1, generations+1):
        probes[k-1] = int(np.sum(objective(propose(population, n_probe, rng)) <= level))
        points = propose(population, N, rng)
        values = objective(points)
        population = Population.uniform(points)
        best = min(best, float(values.min()))
        hit[k] = hit[k-1] or bool(np.any(values <= level))
        traces["mean_gap"][k], traces["best_so_far"][k] = values.mean(), best
        traces["variance"][k] = population.variance()
    return {**traces, "offspring_hit": hit, "probe_counts": probes}


def run_repetitions(propose, initials, objective, generations, level, rng, n_probe=128):
    """Independent runs from a list of initial populations; arrays with a leading run axis."""
    runs = [run(propose, x, objective, generations, level, rng, n_probe) for x in initials]
    return {key: np.stack([r[key] for r in runs]) for key in runs[0]}
