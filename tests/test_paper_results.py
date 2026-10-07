"""The toolbox reproduces the paper's frozen results.

Fixtures (tests/fixtures) are extracts of the paper's frozen arrays and the outputs of the
frozen study code on them; tools/make_fixtures.py builds them in the paper repository.
Run with ``python -m unittest discover tests`` (or pytest).
"""
from pathlib import Path
import json
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from evoscope import attribution, band, contrasts, criterion, discovery, modulus, problems  # noqa: E402
from evoscope.operators import PAPER_SWITCHES, paper_assembly  # noqa: E402
from evoscope.population import Population  # noqa: E402

FIXTURES = Path(__file__).resolve().parent/"fixtures"
TAUS = .1/2.**np.arange(8)


class Problems(unittest.TestCase):
    def test_values_derivatives_gaussian_means(self):
        data = np.load(FIXTURES/"problems.npz")
        for pid, problem in problems.panel().items():
            x = data[f"{pid}__x"]
            np.testing.assert_allclose(problem.value(x), data[f"{pid}__value"], rtol=1e-14, atol=0)
            np.testing.assert_allclose(problem.gradient(x), data[f"{pid}__gradient"], rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(problem.laplacian(x), data[f"{pid}__laplacian"], rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(problem.gaussian_mean(x, .03), data[f"{pid}__gaussian_0.03"], rtol=1e-14, atol=0)


class OneStep(unittest.TestCase):
    """Component generator actions and exact finite steps of the eight assemblies of the study."""

    def test_component_actions_and_finite_steps(self):
        data = np.load(FIXTURES/"one_step.npz")
        keys = sorted({k.rsplit("__", 1)[0] for k in data.files})
        for key in keys:
            problem = problems.get(key.split("__")[0])
            for r, (points, weights) in enumerate(zip(data[f"{key}__input_points"], data[f"{key}__input_weights"])):
                pop = Population(points, weights)
                self.assertAlmostEqual(pop.expect(problem.value(points)), data[f"{key}__V"][r], places=12)
                for a, (omega, gamma, sigma) in enumerate(PAPER_SWITCHES):
                    for o, order in enumerate(("MRS", "MSR")):
                        assembly = paper_assembly(problem, omega, gamma, sigma, order)
                        if o == 0:
                            actions = assembly.generator_actions(pop, problem)
                            got = [actions[name] for name in ("D", "H", "S", "R")]
                            np.testing.assert_allclose(got, data[f"{key}__component_actions"][r, a],
                                                       rtol=1e-11, atol=1e-13, err_msg=f"{key} {r} {a}")
                        got = [assembly.finite_step_mean(pop, tau, problem) for tau in TAUS]
                        np.testing.assert_allclose(got, data[f"{key}__output_V"][r, a, o], rtol=1e-12, atol=1e-14,
                                                   err_msg=f"{key} {r} {a} {order}")


class DriftCoefficients(unittest.TestCase):
    """lambda_cal of the study and its validation counts."""

    def test_lambda_cal_and_validation(self):
        data = np.load(FIXTURES/"drift_coefficients.npz")
        for pid in ("wells_d4", "separable_pl_d4"):
            problem = problems.get(pid)
            cal = [Population.uniform(x) for fam in ("broad", "clustered")
                   for x in data[f"{pid}__{fam}__calibration_points"]]
            for a, switches in enumerate(PAPER_SWITCHES):
                assembly = paper_assembly(problem, *switches)
                result = attribution.coefficients(assembly, cal, problem)
                lam = attribution.estimate_drift_coefficient(result["total"])
                self.assertAlmostEqual(lam, data[f"{pid}__lambda_cal"][a], places=10)
                if pid == "wells_d4":
                    for fam in ("broad", "clustered"):
                        val = [Population.uniform(x) for x in data[f"{pid}__{fam}__validation_points"]]
                        check = attribution.validate(lam, attribution.coefficients(assembly, val, problem)["total"], 64)
                        self.assertEqual(check["violations"], data[f"{pid}__{fam}__validation_below"][a])


class Discovery(unittest.TestCase):
    """Measured discovery bounds of the study and their floors."""

    def test_bounds_and_floors(self):
        data = np.load(FIXTURES/"discovery.npz")
        for N in (16, 64):
            counts = data[f"N{N}__probe_counts"]
            got = [float(discovery.measured_bound(counts, 128, N, endpoint=int(m))["upper"]) for m in data["endpoints"]]
            np.testing.assert_allclose(got, data[f"N{N}__upper"], rtol=0, atol=1e-15)
        for batch in (1, 4, 16, 32):
            self.assertEqual(discovery.floor(128, batch, 200), float(data[f"floor_{batch}"]))
        self.assertEqual(discovery.smallest_unflagged_fraction(128), 49/128)


class Modulus(unittest.TestCase):
    def test_w2_against_reference_run(self):
        data = np.load(FIXTURES/"modulus.npz")
        for N in (16, 32):
            got = [modulus.w2_squared(data[f"N{N}__points"][r], data["N128__points"][r]) for r in range(12)]
            np.testing.assert_allclose(got, data[f"N{N}__w2"], rtol=1e-14)

    def test_lcm_replication_matches_divisible_case(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=(6, 2)), rng.normal(size=(12, 2))
        self.assertAlmostEqual(modulus.w2_squared(a, b), modulus.w2_squared(np.repeat(a, 2, axis=0), b), places=12)
        c = rng.normal(size=(4, 2))
        self.assertGreaterEqual(modulus.w2_squared(a, c), 0.)


class Contrasts(unittest.TestCase):
    def test_paired_contrasts(self):
        data = np.load(FIXTURES/"contrasts.npz")
        traces = np.random.default_rng(int(data["traces_seed"])).normal(size=(128, 4, 201, 3)).cumsum(axis=2)
        traces[:, :, 0] = traces[:, :1, 0]
        out = contrasts.paired_contrasts(traces[:, 0], traces[:, 1:], np.random.default_rng(int(data["bootstrap_seed"])))
        for key in ("mean", "low", "high"):
            np.testing.assert_allclose(out[key], data[key][..., :3], rtol=1e-13, atol=1e-13)


class WorkedExample(unittest.TestCase):
    """Criterion arithmetic and band constants of the worked example of the paper."""

    def setUp(self):
        self.example = json.loads((FIXTURES/"worked_example.json").read_text())

    def test_criterion_arithmetic(self):
        ex = self.example
        for N, rows in ex["criterion"].items():
            for row in rows.values():
                r_star = criterion.boundary_strip_width(ex["r0"], ex["L_band"], row["M"])
                self.assertAlmostEqual(r_star, row["r_star"], places=15)
                discount = criterion.uniform_discount_bound(int(N))
                lhs = criterion.a_disc(ex["L_K"], r_star, row["A_plug_in"], discount)
                self.assertTrue(np.isclose(lhs, row["lhs_of_(7)"], rtol=1e-13))
                self.assertTrue(np.isclose(criterion.required_modulus(ex["L_K"], r_star, discount, ex["delta"]),
                                           row["A_needed_for_delta"], rtol=1e-13))

    def test_band_constants(self):
        ex, problem = self.example, problems.get("separable_pl_d4")
        out = band.band_constants(problem.value, problem.gradient, ex["epsilon"], ex["eta_lev"], -.6, .6, 4,
                                  4_000_000, np.random.default_rng(4272600002))
        self.assertTrue(np.isclose(out["kappa"], ex["kappa_band"], rtol=1e-12))
        self.assertTrue(np.isclose(out["L"], ex["L_band"], rtol=1e-12))
        self.assertTrue(np.isclose(out["r0"], ex["r0"], rtol=1e-12))
        self.assertTrue(np.isclose(out["H"], ex["H_levelset_area_sup"], rtol=1e-12))
        rho = band.gaussian_density_bound(.2*np.sqrt(.1), 4)
        self.assertTrue(np.isclose(rho, ex["rho_worst_gaussian_mixture"], rtol=1e-12))
        self.assertTrue(np.isclose(band.coarea_density_bound(rho, out["H"], out["kappa"]), ex["M_worst_case"], rtol=1e-12))


if __name__ == "__main__":
    unittest.main()
