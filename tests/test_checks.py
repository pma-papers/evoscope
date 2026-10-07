"""The checks give the expected verdicts on operators and algorithms whose properties are known."""
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

from evoscope import algorithm, checks, metrics, problems  # noqa: E402
from evoscope.operators import (Drift, GaussianNoise, Jump, Recombination, Reweighting, StepOperator,  # noqa: E402
                              Transport, estimate_generator, normalized_fd_drift, paper_assembly)
from evoscope.population import Population, broad_gaussian, two_cluster  # noqa: E402
from evoscope.testfunctions import battery  # noqa: E402

F = problems.get("quartic_d4")
RNG = np.random.default_rng(1)
POPS = [broad_gaussian(16, 4, RNG), two_cluster(16, 4, RNG), broad_gaussian(16, 4, RNG)]


def hard_rank(x, pop):
    values = F.value(pop.points)
    return np.mean(values[None, :] <= F.value(x)[:, None], axis=1)


def smooth_rank(x, pop, eta=.3):
    values = F.value(pop.points)
    return np.sum(pop.weights/(1+np.exp(-(F.value(x)[:, None]-values[None, :])/eta)), axis=1)


class OperatorChecks(unittest.TestCase):
    def verdicts(self, op):
        return {f.check.split("(")[-1].rstrip(")"): f.verdict for f in checks.check_operator(op, POPS, fast=True).findings}

    def test_well_behaved_operators_pass(self):
        for op in (GaussianNoise(.3), Transport(noise=lambda x, g: .3*g.normal(size=x.shape)), Recombination(),
                   Reweighting(lambda x, pop: 4*smooth_rank(x, pop)), Drift(normalized_fd_drift(F.value))):
            self.assertEqual(set(self.verdicts(op).values()), {"consistent"}, op)

    def test_hard_rank_threshold_breaks_continuity(self):
        self.assertEqual(self.verdicts(Reweighting(lambda x, pop: 8.*(hard_rank(x, pop) > .5)))["A4"], "violated")

    def test_superlinear_drift_breaks_moment_stability(self):
        self.assertEqual(self.verdicts(Drift(lambda x: x**3))["A2"], "violated")
        # plain gradient descent on the quartic (gradient ~ x^3) is flagged for the same reason
        self.assertEqual(self.verdicts(Drift(lambda x: -F.gradient(x)/10))["A2"], "violated")

    def test_rule_ignoring_the_step_size_fails_first_order_and_near_identity(self):
        def replace_half(pop, tau, rng):
            keep = pop.points[:8]
            new = pop.points[rng.integers(0, 16, 8)]+rng.normal(size=(8, 4))
            return Population.uniform(np.concatenate([keep, new]))
        v = self.verdicts(StepOperator(replace_half))
        self.assertEqual((v["A1"], v["A3"]), ("violated", "violated"))

    def test_same_rule_written_as_a_jump_passes(self):
        jump = Jump(offspring=lambda pop, size, rng: pop.points[pop.sample(size, rng)]+rng.normal(size=(size, 4)))
        v = self.verdicts(jump)
        self.assertEqual((v["A1"], v["A3"]), ("consistent", "consistent"))

    def test_numerical_generator_matches_formula(self):
        pop, phi = POPS[0], battery(POPS, 4)[0]
        op = Transport(noise=lambda x, g: .3*g.normal(size=x.shape))
        est = estimate_generator(op, pop, phi, replicates=400, taus=(.02,))
        exact = GaussianNoise(.3).generator(pop, phi)
        self.assertLess(abs(est["estimate"]-exact), 4*est["se"]+.02*abs(exact))


class AssemblyChecks(unittest.TestCase):
    def test_paper_assembly(self):
        report = checks.check_assembly(paper_assembly(F), POPS, F, operator_checks=False, fast=True)
        self.assertEqual(report["composition (sum of effects)"].verdict, "consistent")
        self.assertEqual(report["unbounded error measure (cutoff)"].verdict, "consistent")
        self.assertEqual(report["drift coefficient and attribution"].verdict, "violated")   # two-cluster quartic
        self.assertIn("lambda_cal", report.summary())
        report.plot()

    def test_heavy_tailed_jump_fails_cutoff(self):
        cauchy = Jump(offspring=lambda pop, size, rng: pop.points[pop.sample(size, rng)]+rng.standard_cauchy((size, 4)))
        finding = checks.check_cutoff(cauchy, POPS[:2], F, replicates=32)
        self.assertNotEqual(finding.verdict, "consistent")


class Metrics(unittest.TestCase):
    def test_w2_exact_cases(self):
        a = Population.uniform(np.array([[0.], [1.]]))
        b = Population(np.array([[0.], [1.], [3.]]), np.array([.5, .25, .25]))
        self.assertAlmostEqual(metrics.w2(a, a), 0., places=12)
        self.assertAlmostEqual(metrics.w2(a, Population.uniform(a.points+2)), 2., places=10)
        self.assertAlmostEqual(metrics.w2(a, b)**2, .25*4, places=10)


def es(mu=8, lam=32, sigma=.3, recombine=True):
    def ask(state, size, rng):
        parents = state[rng.integers(0, len(state), (size, 2))]
        x = .5*(parents[:, 0]+parents[:, 1]) if recombine else parents[:, 0]
        return x+sigma*rng.normal(size=x.shape)
    return algorithm.FunctionalAlgorithm(lambda rng: rng.normal(.6, 1.2, (lam, 4)), ask,
                                         lambda state, X, y, rng: X[np.argsort(y)[:mu]], lam, population=lambda s: s)


class AlgorithmChecks(unittest.TestCase):
    def test_black_box_es(self):
        f = problems.get("separable_pl_d4")
        report = algorithm.check_algorithm(es(), f.value, level=.1, rng=np.random.default_rng(0), generations=40,
                                           repetitions=16, n_probe=64,
                                           variants={"full": es(), "no recombination": es(recombine=False)},
                                           reference="full")
        self.assertEqual(report["run diagnostics"].verdict, "consistent")
        self.assertIn(report["per-generation contraction"].verdict, ("consistent", "suspect"))
        self.assertIn("no recombination", report["ingredient effects per generation"].statistic)
        report.plot()

    def test_small_step_detects_unscaled_noise(self):
        f = problems.get("wells_d4")
        states = [broad_gaussian(32, 4, np.random.default_rng(i)) for i in range(3)]

        def scaled(tau):
            return algorithm.AssemblyAlgorithm(paper_assembly(f), tau, None, 32)

        def unscaled(tau):     # mutation noise does not shrink with the step
            asm = paper_assembly(f, sigma=0.)
            noise = .3 if tau > 0 else 0.
            return algorithm.FunctionalAlgorithm(None, lambda s, n, g: asm.sample(s, tau, n, g)+noise*g.normal(size=(n, 4)),
                                                 None, 32)
        good = algorithm.check_small_step(scaled, states, f.value, size=2048, groups=6)
        bad = algorithm.check_small_step(unscaled, states, f.value, size=2048, groups=6)
        self.assertEqual(good.verdict, "consistent")
        self.assertEqual(bad.verdict, "violated")

    def test_ablation_runs_shapes(self):
        f = problems.get("separable_pl_d4")
        out = algorithm.ablation_runs({"full": es(), "no recombination": es(recombine=False)}, "full", f.value,
                                      generations=10, repetitions=8, bootstrap=200)
        self.assertEqual(out["mean"].shape, (1, 11, 2))


if __name__ == "__main__":
    unittest.main()
