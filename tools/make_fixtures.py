"""Maintainer tool: build the test fixtures from the frozen results of the paper's numerical study.

    python tools/make_fixtures.py STUDY_DIR

STUDY_DIR is the directory that holds the study's frozen code (supplementary-code/) and
result folders (results-v8/, results-v11/, results-v14/, results-v15/).  The fixtures are
small extracts (inputs and the frozen outputs computed from them by the frozen code), so
that the tests check the toolbox against the paper without shipping the full arrays.
Nothing in STUDY_DIR is written.
"""
from pathlib import Path
import json
import sys

import numpy as np

sys.dont_write_bytecode = True
TOOLBOX = Path(__file__).resolve().parents[1]
if len(sys.argv) != 2:
    sys.exit(__doc__)
PAPER = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(PAPER/"supplementary-code"))

import common_panel_v7 as frozen_panel                     # noqa: E402
import canonical_assembly_v8 as frozen_one_step            # noqa: E402
import finite_discovery_inference_v11 as frozen_discovery  # noqa: E402
import reference_run_modulus_v14 as frozen_modulus         # noqa: E402
import canonical_dynamic_contrasts_v11 as frozen_contrasts  # noqa: E402

OUT = TOOLBOX/"tests/fixtures"
ASSEMBLIES = [a["id"] for a in frozen_one_step.ASSEMBLIES]


def problems():
    specs = {s["id"]: s for s in frozen_panel.instances()}
    rng = np.random.default_rng(11)
    out = {}
    for pid, spec in specs.items():
        x = rng.normal(.3, 1., (5, spec["dimension"]))
        g, lap = frozen_one_step.gradient_laplacian(x, spec)
        out.update({f"{pid}__x": x, f"{pid}__value": frozen_panel.objective(x, spec), f"{pid}__gradient": g,
                    f"{pid}__laplacian": lap, f"{pid}__gaussian_0.03": frozen_one_step.gaussian_objective(x, .03, spec)})
    np.savez_compressed(OUT/"problems.npz", **out)


def one_step():
    arrays = np.load(PAPER/"results-v8/canonical-validation/canonical_arrays.npz")
    out = {}
    for pid in ("separable_pl_d4", "rotated_anisotropic_pl_d8", "quartic_d4", "rosenbrock_d4", "wells_d4", "periodic_d1"):
        for family in ("broad", "clustered"):
            key = f"{pid}__{family}"
            for field in ("input_points", "input_weights", "V", "component_actions", "output_V"):
                out[f"{key}__{field}"] = arrays[f"{key}_{field}"][:3]
    np.savez_compressed(OUT/"one_step.npz", **out)


def drift_coefficients():
    cal = np.load(PAPER/"results-v8/canonical-calibration/canonical_arrays.npz")
    val = np.load(PAPER/"results-v8/canonical-validation/canonical_arrays.npz")
    audit = json.loads((PAPER/"results-v15/dissipation_estimates_audit.json").read_text())
    out = {}
    for pid in ("wells_d4", "separable_pl_d4"):
        for family in ("broad", "clustered"):
            out[f"{pid}__{family}__calibration_points"] = cal[f"{pid}__{family}_input_points"]
        lam = np.array(audit["per_instance"][pid]["lambda_cal"])
        out[f"{pid}__lambda_cal"] = lam
        if pid == "wells_d4":
            for family in ("broad", "clustered"):
                out[f"{pid}__{family}__validation_points"] = val[f"{pid}__{family}_input_points"]
                out[f"{pid}__{family}__validation_below"] = (val[f"{pid}__{family}_effective_decay_coefficient"] < lam).sum(axis=0)
    np.savez_compressed(OUT/"drift_coefficients.npz", **out)


def discovery():
    analysis = np.load(PAPER/"results-v11/finite_population_analysis.npz")
    case = list(analysis["cases"]).index("wells_d4__broad")
    out = {"endpoints": analysis["endpoints"]}
    for N in (16, 64):
        data = np.load(PAPER/f"results-v11/finite-population-full/wells_d4__broad__N{N}.npz")
        ai, oi = list(data["assemblies"]).index("S1_R1_H1"), list(data["orders"]).index("MRS")
        ei = list(data["epsilons"]).index(.1)
        counts = data["probe_success_counts"][:, ai, oi, :, ei]
        out[f"N{N}__probe_counts"] = counts
        out[f"N{N}__upper"] = analysis["upper"][case, list(analysis["populations"]).index(N), ai, oi, 0,
                                                list(analysis["epsilons"]).index(.1), :]
        flags = frozen_discovery.low_mass_flags(counts).sum(axis=0)
        out[f"N{N}__frozen_bound_200"] = frozen_discovery.discovery_bound(flags, 128, N, 200)["upper"]
    for batch in (1, 4, 16, 32):
        out[f"floor_{batch}"] = frozen_discovery.discovery_bound(np.zeros(200, dtype=int), 128, batch, 200)["upper"]
    np.savez_compressed(OUT/"discovery.npz", **out)


def modulus():
    folder = PAPER/"results-v11/finite-population-full"
    out = {}
    for N in (16, 32, 128):
        with np.load(folder/f"separable_pl_d4__broad__N{N}.npz") as d:
            ai, oi = list(d["assemblies"]).index("S1_R1_H1"), list(d["orders"]).index("MRS")
            out[f"N{N}__points"] = d["final_points"][:12, ai, oi]
    for N in (16, 32):
        out[f"N{N}__w2"] = np.array([frozen_modulus.w2_squared(out[f"N{N}__points"][r], out["N128__points"][r])
                                     for r in range(12)])
    np.savez_compressed(OUT/"modulus.npz", **out)


def contrasts():
    rng = np.random.default_rng(5)
    traces = rng.normal(size=(128, 4, 201, 3)).cumsum(axis=2)
    traces[:, :, 0] = traces[:, :1, 0]                 # matched starts
    names = [frozen_contrasts.FULL]+list(frozen_contrasts.OMISSIONS)
    full = np.zeros((128, 4, 201, len(frozen_contrasts.METRICS)+1))
    full[..., :3] = traces
    metrics = list(frozen_contrasts.METRICS)+["unused"]
    mean, low, high, _ = frozen_contrasts.paired_summary(full, names, metrics, 77)
    np.savez_compressed(OUT/"contrasts.npz", traces_seed=5, bootstrap_seed=77, mean=mean, low=low, high=high)


def worked_example():
    example = json.loads((PAPER/"results-v14/criterion_worked_example.json").read_text())
    keep = {k: example[k] for k in ("epsilon", "delta", "eta_lev", "L_K", "kappa_band", "L_band", "r0",
                                    "H_levelset_area_sup", "rho_worst_gaussian_mixture", "M_worst_case")}
    keep["criterion"] = {N: example["by_N"][N]["criterion"] for N in ("16", "32", "64")}
    (OUT/"worked_example.json").write_text(json.dumps(keep, indent=1)+"\n")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for build in (problems, one_step, drift_coefficients, discovery, modulus, contrasts, worked_example):
        build()
        print("built", build.__name__)
    for path in sorted(OUT.iterdir()):
        print(f"{path.name:28s} {path.stat().st_size/1024:8.1f} KB")
