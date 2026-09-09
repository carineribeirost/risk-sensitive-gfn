"""Tests for the analysis scripts (loaded by path; scripts/ is not a
package). Synthetic rows keep these exact and fast."""

import csv
import importlib.util
import pathlib

import numpy as np

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    """Import a scripts/<name>.py module by file path, since scripts/ is not a
    package."""
    spec = importlib.util.spec_from_file_location(name,
                                                  SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_o1b_separability_summary_endpoint():
    """separability_summary reports per-world max, median-of-max, and pass/fail
    per metric."""
    a = load_script("analyze_o1b")
    # two worlds; per-world max = 0.20 and 0.06 -> median 0.13 PASS;
    # second metric flat at 0.01 -> FAIL
    rows = [
        {"case": "B", "world_seed": "0", "tv_to_shared": "0.05",
         "tv_to_rho_split": "0.01"},
        {"case": "B", "world_seed": "0", "tv_to_shared": "0.20",
         "tv_to_rho_split": "0.01"},
        {"case": "B", "world_seed": "1", "tv_to_shared": "0.06",
         "tv_to_rho_split": "0.01"},
    ]
    s = a.separability_summary(rows, margin=0.05)
    assert set(s) == {"tv_to_shared", "tv_to_rho_split"}
    m = s["tv_to_shared"]
    assert m["n_worlds"] == 2
    assert m["per_world_max"] == {"0": 0.20, "1": 0.06}
    assert np.isclose(m["median_of_max"], 0.13)
    assert m["pass"]
    assert not s["tv_to_rho_split"]["pass"]
    flat = s["tv_to_rho_split"]
    assert flat["ci_lo"] <= 0.01 <= flat["ci_hi"]


def test_o3_asym_matched_pairs_and_confirmatory():
    """matched_pairs pairs rho values by kappa asymmetry, and asym_confirmatory
    produces valid confirmatory statistics."""
    from epgfn.o3 import run_o3
    from epgfn.worlds import WorldConfig
    a = load_script("analyze_o3_asym")
    rows, meta = run_o3("B", WorldConfig(H=8), [0, 1], n_stress=32,
                        kappa_range=(8, 60))
    pairs = a.matched_pairs(rows, meta)
    kappa = {m["world_seed"]: (m["kappa_p"], m["kappa_m"]) for m in meta}
    assert len(pairs) == 2
    for ws, prs in pairs.items():
        assert len(prs) == 6  # C(4,2) unordered distinct rho pairs
        kp, km = kappa[ws]
        for matched, mism in prs:
            # matched puts the larger rho on the smaller-kappa origin
            hi_on_p = matched["rho_p"] > matched["rho_m"]
            assert hi_on_p == (kp < km)
            assert {matched["rho_p"], matched["rho_m"]} == \
                   {mism["rho_p"], mism["rho_m"]}
    conf = a.asym_confirmatory(rows, meta, tost_margin=0.10)
    assert conf["n_worlds"] == 2
    assert 0.0 < conf["delta_stress_p05"]["p_value"] <= 1.0
    assert "delta_contam_worst" in conf  # contam on by default
    assert np.isfinite(conf["dose_response"]["spearman_rho"])


def test_o3_asym_classify_and_atlas():
    """classify labels regime rows correctly, and regime_atlas summarizes
    winning regions and corner dominance."""
    a = load_script("analyze_o3_asym")
    assert a.classify({"target": "risk", "beta_cvar": 1.0,
                       "rho": 0.0}) == "boltzmann"
    assert a.classify({"target": "risk", "beta_cvar": 0.3,
                       "rho": 0.0}) == "cvar_only"
    assert a.classify({"target": "risk", "beta_cvar": 1.0,
                       "rho": 0.5}) == "dro_only"
    assert a.classify({"target": "risk", "beta_cvar": 0.3,
                       "rho": 0.5}) == "interior"
    assert a.classify({"target": "worst"}) == "worst"

    def row(kind, b, r, val):
        return {"target": kind, "world_seed": "0", "t": 0.3,
                "t_star": 0.3, "beta_cvar": b, "rho": r,
                "sat_mass": val, "eff_candidates": val,
                "stress_mean": val, "stress_p05": val}
    rows = [row("boltzmann", "", "", 0.2), row("worst", "", "", 0.1),
            row("risk", 0.3, 0.5, 0.4), row("risk", 0.3, 0.0, 0.3)]
    atlas = a.regime_atlas(rows)
    m = atlas["sat_mass"]
    assert m["winning_region_mode"] == "interior"
    assert m["corner_dominance"]["cvar_only"][
        "best_interior_minus_best_corner"] > 0


def test_o3_asym_rho_boundaries():
    """rho_boundaries finds the smallest rho, per beta, at which a metric drops
    below the margin."""
    a = load_script("analyze_o3_asym")
    rows = [{"world_seed": "0", "beta_cvar": 0.5, "rho": r,
             "tv_vs_worst": v}
            for r, v in [(0.0, 0.5), (0.4, 0.2), (0.8, 0.04),
                         (1.2, 0.01)]]
    b = a.rho_boundaries(rows, margin=0.05)
    assert b["0"]["per_beta"]["0.5"] == 0.8
    assert b["0"]["min_rho_at_worst"] == 0.8


def test_wg_drowning_point():
    """drowning_point returns the last rho crossing above margin, resetting
    whenever a later value pops back above it."""
    p = load_script("probe_wg_drowning")
    maxes = {0.1: 0.30, 0.3: 0.20, 0.5: 0.04, 0.7: 0.06, 0.9: 0.01}
    # a pop back above the margin resets the point: 0.9, not 0.5
    assert p.drowning_point(maxes, margin=0.05) == 0.9
    assert p.drowning_point({0.1: 0.3, 0.5: 0.04, 0.9: 0.01},
                            margin=0.05) == 0.5
    assert p.drowning_point({0.1: 0.3, 0.9: 0.2}, margin=0.05) is None


def test_o1b_load_sep_rows_roundtrip(tmp_path):
    """load_sep_rows reads separability CSVs from disk and reproduces
    separability_summary's results."""
    a = load_script("analyze_o1b")
    rows = [{"case": "C", "world_seed": "3", "rho": "0.2",
             "delta": "0.05", "tv_to_shared": "0.11"}]
    with open(tmp_path / "o1_caseC_sep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by_case = a.load_sep_rows([tmp_path])
    assert list(by_case) == ["C"]
    s = a.separability_summary(by_case["C"])
    assert s["tv_to_shared"]["pass"]
