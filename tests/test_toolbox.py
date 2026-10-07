"""General behaviour of the toolbox beyond the paper's settings."""
from pathlib import Path
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())
import matplotlib  # noqa: E402
matplotlib.use("Agg")

from evoscope import attribution, discovery, problems, runs, viz  # noqa: E402
from evoscope.operators import (Assembly, Drift, GaussianNoise, Observable, Recombination,  # noqa: E402
                              Selection, objective_rate, paper_assembly)
from evoscope.population import Population, broad_gaussian, two_cluster  # noqa: E402


class Composition(unittest.TestCase):
    def test_first_order_consistency_decreases_linearly(self):
        problem = problems.get("quartic_d4")
        pop = two_cluster(32, 4, np.random.default_rng(1))
        out = attribution.consistency(paper_assembly(problem), pop, problem, .1/2.**np.arange(6))
        ratios = out["normalized_remainder"][:-1]/out["normalized_remainder"][1:]
        self.assertTrue(np.all((ratios > 1.6) & (ratios < 2.4)))

    def test_generator_is_order_independent_and_additive(self):
        problem = problems.get("wells_d4")
        pop = broad_gaussian(16, 4, np.random.default_rng(2))
        a, b = paper_assembly(problem, order="MRS"), paper_assembly(problem, order="MSR")
        self.assertAlmostEqual(a.generator(pop, problem), b.generator(pop, problem), places=12)
        self.assertAlmostEqual(a.generator(pop, problem), sum(a.generator_actions(pop, problem).values()), places=12)

    def test_monte_carlo_fallback_and_fd_observable(self):
        problem = problems.get("separable_pl_d4")
        pop = broad_gaussian(8, 4, np.random.default_rng(3))
        noise_first = Assembly([GaussianNoise(.3), Drift(lambda x: -x)])
        exact_last = Assembly([Drift(lambda x: -x), GaussianNoise(.3)])
        obs = Observable(problem.value)                                    # derivatives by finite differences
        self.assertAlmostEqual(exact_last.generator(pop, obs), exact_last.generator(pop, problem), places=4)
        mc = noise_first.finite_step_mean(pop, .05, problem, rng=np.random.default_rng(4), samples=400_000)
        exact = exact_last.finite_step_mean(pop, .05, problem)
        self.assertLess(abs(mc-exact), .05*abs(exact))

    def test_selection_reduces_objective_gap(self):
        problem = problems.get("rastrigin_d4")
        pop = broad_gaussian(32, 4, np.random.default_rng(5))
        S = Selection(objective_rate(problem.value))
        self.assertLessEqual(S.generator(pop, problem), 0.)

    def test_blend_recombination_weights(self):
        pop = Population.uniform(np.array([[0.], [1.]]))
        R = Recombination(coefficients=(0., 1.), probabilities=(.5, .5))
        stepped = R.finite_step(pop, .5)
        self.assertAlmostEqual(stepped.weights.sum(), 1.)
        self.assertEqual(stepped.size, 2+2*4)


class Runs(unittest.TestCase):
    def test_run_and_measured_bound_and_plots(self):
        problem = problems.get("separable_pl_d4")
        assembly = paper_assembly(problem)
        rng = np.random.default_rng(6)
        initials = [broad_gaussian(16, 4, rng) for _ in range(8)]
        out = runs.run_repetitions(assembly.proposer(.1), initials,
                                   problem.value, generations=30, level=.1, rng=rng, n_probe=32)
        self.assertEqual(out["probe_counts"].shape, (8, 30))
        bound = discovery.measured_bound(out["probe_counts"], 32, 16)
        self.assertTrue(0 < float(bound["upper"]) <= 1)
        axes = viz.plot_discovery(out["probe_counts"], 32, 16, observed_non_discovery=1-out["offspring_hit"][:, 1:].mean(axis=0))
        self.assertEqual(len(axes), 3)
        result = attribution.coefficients(assembly, initials, problem)
        viz.plot_component_coefficients(result)
        viz.plot_drift_estimates({"PL": (result["total"], result["total"])})
        viz.plot_residual_form(result["generator"], result["V"], 1., attribution.residual_constant(result["generator"], result["V"], 1.))
        viz.plot_consistency(.1/2.**np.arange(4), np.array([[1e-2, 5e-3, 2.5e-3, 1.25e-3]]))
        g = np.arange(31)
        viz.plot_contrast(g, np.zeros(31), -np.ones(31), np.ones(31))


class GradientMethods(unittest.TestCase):
    """Gradient descent as a one-operator special case (Tutorial 6)."""
    a = np.array([1., 2., 4., 8.])

    def quadratic(self):
        a = self.a
        return Observable(lambda x: .5*np.sum(a*x*x, axis=-1), lambda x: a*x,
                          lambda x: np.full(np.shape(x)[:-1], a.sum()),
                          lambda m, v: .5*np.sum(a*m*m, axis=-1)+.5*v*a.sum())

    def test_population_of_one_is_gradient_descent(self):
        f = self.quadratic()
        x = np.array([[2., -1., .5, .3]])
        pop = Population.uniform(x.copy())
        for _ in range(10):
            x = x-.05*f.gradient(x)
            pop = Drift(lambda y: -f.gradient(y)).step(pop, .05)
        np.testing.assert_allclose(pop.points, x)

    def test_drift_coefficient_is_at_least_the_pl_constant(self):
        f, rng = self.quadratic(), np.random.default_rng(0)
        pops = [Population.uniform(rng.normal(size=(32, 4))*s) for s in (np.array([2., .3, .2, .1]), 1.) for _ in range(16)]
        rates = attribution.coefficients(Assembly([Drift(lambda y: -f.gradient(y))]), pops, f)["total"]
        self.assertGreaterEqual(rates.min(), 2*self.a.min()-1e-12)
        self.assertLess(rates.min(), 2.5*self.a.min())       # stretched populations come close to 2 mu

    def test_noisy_gradient_descent_residual_constant(self):
        f, rng, sigma = self.quadratic(), np.random.default_rng(1), .5
        pops = [Population.uniform(rng.normal(size=(32, 4))*s) for s in (.01, .1, 1.) for _ in range(8)]
        result = attribution.coefficients(Assembly([Drift(lambda y: -f.gradient(y)), GaussianNoise(sigma)]), pops, f)
        c = attribution.residual_constant(result["generator"], result["V"], 2*self.a.min())
        self.assertLessEqual(c, sigma**2*self.a.sum()/2+1e-12)   # by hand: G + 2 mu V <= sigma^2 tr(A)/2
        self.assertGreater(c, .9*sigma**2*self.a.sum()/2)


if __name__ == "__main__":
    unittest.main()
