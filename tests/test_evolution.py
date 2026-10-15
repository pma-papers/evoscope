"""The evolution equation term by term: views, term rates, flows, ledgers, operator
fingerprints and the one-dimensional solver."""
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

from evoscope import evolution as ev, problems, viz  # noqa: E402
from evoscope.operators import (Assembly, Drift, GaussianNoise, Jump, Observable, Recombination,  # noqa: E402
                                Selection, StepOperator, objective_rate, paper_assembly)
from evoscope.population import Population, broad_gaussian, two_cluster  # noqa: E402

WELLS_1D = Observable(lambda x: (x[..., 0]**2-1)**2+.2*(x[..., 0]-1)**2,
                      lambda x: 4*x*(x*x-1)+.4*(x-1),
                      lambda x: 12*x[..., 0]**2-3.6)


class Views(unittest.TestCase):
    def setUp(self):
        self.f = problems.get("quartic_d4")
        self.pop = broad_gaussian(40, 4, np.random.default_rng(0))
        self.assembly = paper_assembly(self.f)

    def test_summary_view_reproduces_generator_actions(self):
        rates = ev.term_rates(self.assembly, self.pop, ev.summary_view(self.f, self.pop))
        for name, action in self.assembly.generator_actions(self.pop, self.f).items():
            self.assertAlmostEqual(rates[name][0], action, places=9)

    def test_moment_signatures_of_the_canonical_terms(self):
        d, spread = 4, self.pop.variance()
        mean = ev.term_rates(self.assembly, self.pop, ev.mean_view(d))
        self.assertLess(np.abs(mean["R"]).max(), 1e-12)          # recombination keeps the mean
        self.assertLess(np.abs(mean["H"]).max(), 1e-12)          # so does the noise
        rates = ev.term_rates(self.assembly, self.pop, ev.spread_view(self.pop))
        self.assertAlmostEqual(rates["H"][0], .2**2*d, places=12)            # constant source sigma^2 d
        self.assertAlmostEqual(rates["R"][0], -.5*spread, places=12)         # midpoints halve the spread
        self.assertAlmostEqual(rates["R"][0], rates["R"][1:].sum(), places=10)

    def test_selection_raises_the_mass_below_every_level(self):
        levels = np.geomspace(1e-2, 1e2, 25)
        view = ev.level_view(self.f, levels)
        for pop in (self.pop, two_cluster(40, 4, np.random.default_rng(1))):
            rates = ev.term_rates(Selection(objective_rate(self.f.value)), pop, view)["S"]
            self.assertTrue(np.all(rates >= -1e-12))

    def test_derivatives_of_views_match_finite_differences(self):
        x = self.pop.points[:5]
        for view in (ev.level_view(self.f, np.geomspace(.1, 10, 6)),
                     ev.density_view((np.linspace(-2, 2, 4), np.linspace(-1, 1, 3)), .5, [[1, 0, 0, 0], [0, 1, 0, 0]]),
                     ev._probe_view(self.pop.points[:3], np.full(3, .4), ev._directions(4, True))):
            plain = ev.View(view.value, view.labels)
            scale = 1+np.abs(view.laplacian(x)).max()
            self.assertLess(np.abs(view.gradient(x)-plain.gradient(x)).max(), 1e-5*scale)
            self.assertLess(np.abs(view.laplacian(x)-plain.laplacian(x)).max(), 1e-4*scale)

    def test_density_terms_conserve_mass(self):
        grid = np.linspace(-7, 7, 141)
        view = ev.density_view(grid, .3, directions=[1, 1, 0, 0])
        for name, rate in ev.term_rates(self.assembly, self.pop, view).items():
            self.assertLess(abs(rate.sum()*(grid[1]-grid[0])), 1e-6, name)


class Characteristics(unittest.TestCase):
    def setUp(self):
        self.pop = broad_gaussian(30, 2, np.random.default_rng(2))
        f = Observable(lambda x: np.sum(x*x, axis=-1))
        self.S, self.R = Selection(objective_rate(f.value)), Recombination()
        self.D, self.H = Drift(lambda x: -x), GaussianNoise(.3)

    def test_probes_reproduce_the_formulas(self):
        for op in (self.S, self.D, self.H):
            exact = ev.local_characteristics(op, self.pop)
            probed = ev.local_characteristics(op, self.pop, method="probes")
            for key in ("growth", "drift", "diffusion"):
                self.assertLess(np.abs(exact[key]-probed[key]).max(), 1e-3, (op.name, key))

    def test_tracking_estimates_drift_and_diffusion(self):
        rule = StepOperator(lambda p, tau, rng: p.points-tau*p.points+.3*np.sqrt(tau)*rng.normal(size=p.points.shape))
        chars = ev.local_characteristics(rule, self.pop, replicates=4000, rng=np.random.default_rng(3))
        self.assertLess(np.max(np.abs(chars["drift"]+chars["points"])/(4*chars["drift_se"]+1e-12)), 1.5)
        self.assertLess(np.abs(chars["diffusion"]-.09*np.eye(2)).max(), .015)

    def test_fingerprint_names_the_kind_of_each_term(self):
        flight = StepOperator(lambda p, tau, rng: p.points+.3*tau*rng.standard_cauchy(size=p.points.shape),
                              name="flight")
        result = ev.fingerprint([self.S, self.R, self.D, self.H, flight], self.pop, rng=np.random.default_rng(4))
        kinds = {name: entry["kinds"] for name, entry in result.items()}
        self.assertEqual(kinds, {"S": ["reaction"], "R": ["jump"], "D": ["transport"], "H": ["diffusion"],
                                 "flight": ["jump"]})
        self.assertIn("kinds of term", ev.format_fingerprint(result))


class Solver(unittest.TestCase):
    def setUp(self):
        self.assembly = Assembly([Selection(objective_rate(WELLS_1D.value), strength=2.), Recombination(rate=.5),
                                  Drift(lambda x: -WELLS_1D.gradient(x)/np.sqrt(1+WELLS_1D.gradient(x)**2)),
                                  GaussianNoise(.25)])
        self.grid = np.linspace(-3, 3, 301)
        self.solver = ev.TRJ1D(self.assembly, self.grid)
        self.rho = np.exp(-(self.grid+.9)**2/(2*.5**2))
        self.rho /= self.rho.sum()*self.solver.h

    def test_terms_conserve_mass_and_midpoints_halve_the_variance(self):
        h = self.solver.h
        for name, term in self.solver.terms(self.rho).items():
            self.assertLess(abs(term.sum()*h), 1e-10, name)
        r = self.solver._midpoint_density(self.rho)
        mean = np.sum(self.grid*self.rho)*h
        self.assertAlmostEqual(np.sum(self.grid*r)*h, mean, places=10)
        variance = np.sum((self.grid-mean)**2*self.rho)*h
        self.assertAlmostEqual(np.sum((self.grid-mean)**2*r)*h, variance/2+h*h/8, places=6)

    def test_strong_and_weak_forms_agree(self):
        view = ev.density_view(np.linspace(-2, 2, 21), .2)
        weak = ev.term_rates(self.assembly, self.solver.population(self.rho), view)
        h = self.solver.h
        kernel = np.exp(-(view.axis[:, None]-self.grid[None, :])**2/(2*.2**2))/np.sqrt(2*np.pi*.2**2)
        for name, term in self.solver.terms(self.rho).items():
            strong = kernel@term*h
            self.assertLess(np.abs(strong-weak[name]).max(), .02*np.abs(weak[name]).max()+1e-9, name)

    def test_particles_follow_the_solution(self):
        solution = self.solver.solve(self.rho, 1., record=[1.])
        rng = np.random.default_rng(5)
        flow = ev.flow(self.assembly, Population.uniform(rng.normal(-.9, .5, (8000, 1))), .01, 1., rng, record=[1.])
        mean = np.sum(self.grid*solution["densities"][-1])*self.solver.h
        self.assertLess(abs(flow.populations[-1].mean()[0]-mean), .03)


class Ledgers(unittest.TestCase):
    def test_contributions_add_up_to_the_change(self):
        rng = np.random.default_rng(6)
        f = problems.get("separable_pl_d4")
        assembly = Assembly([Selection(objective_rate(f.value)), Drift(lambda x: -.5*x, name="D"),
                             GaussianNoise(.3)])
        flow = ev.flow(assembly, broad_gaussian(4000, 4, rng), .01, 2., rng, record=np.linspace(0, 2, 21))
        result = ev.ledger(assembly, flow, lambda p: ev.summary_view(f, p, target=np.zeros(4)))
        self.assertEqual(result["rates"].shape, (21, 3, 3))
        scale = np.abs(result["change"]).max(axis=0)
        self.assertLess(np.max(np.abs(result["predicted"]-result["change"])/scale), .1)
        self.assertAlmostEqual(result["contributions"][-1, 2, 1], .3**2*4*2, places=10)   # noise: sigma^2 d t

    def test_plots(self):
        rng = np.random.default_rng(7)
        f = problems.get("quartic_d4")
        assembly = paper_assembly(f)
        flow = ev.flow(assembly, broad_gaussian(500, 4, rng), .05, .5, rng, record=[0, .25, .5])
        levels = ev.ledger(assembly, flow, ev.level_view(f, np.geomspace(.1, 10, 8)))
        viz.plot_ledger(levels, index=3)
        viz.plot_levels(levels)
        viz.plot_level_terms(levels["views"][0].axis, dict(zip(levels["names"], levels["rates"][-1])))
        view = ev.density_view((np.linspace(-2, 2, 9), np.linspace(-2, 2, 9)), .3)
        viz.plot_term_map(view, ev.term_rates(assembly, flow.populations[-1], view)["R"],
                          points=flow.populations[-1].points, objective=lambda x: np.sum((x*x-1)**2, axis=-1))
        viz.plot_characteristics(ev.local_characteristics(assembly.operators[2], broad_gaussian(20, 2, rng)),
                                 objective=lambda x: np.sum(x*x, axis=-1), extent=(-2, 2, -2, 2))


if __name__ == "__main__":
    unittest.main()
