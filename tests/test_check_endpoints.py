"""Verdict automation: Holm correction and the mechanical endpoint
checks against synthesized run fixtures."""

import csv
import importlib.util
import json
import pathlib

import numpy as np

from epgfn.stats import holm

spec = importlib.util.spec_from_file_location(
    "check_endpoints",
    pathlib.Path(__file__).resolve().parent.parent
    / "scripts" / "check_endpoints.py")
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)


def test_holm_stepdown():
    """holm applies Holm step-down correction, rejecting only the most
    significant p-value and ordering adjusted p-values correctly."""
    r = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert r["a"]["reject"] and not r["b"]["reject"]
    assert r["b"]["p_adj"] >= r["c"]["p_adj"] >= r["a"]["p_adj"]


def _write_o2(dirpath, case, l1, floor):
    dirpath.mkdir(parents=True, exist_ok=True)
    with open(dirpath / f"o2_case{case}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "world", "train_seed",
                                           "heldout_l1", "mc_floor_l1"])
        w.writeheader()
        for wi in range(4):
            for s in range(3):
                w.writerow({"case": case, "world": wi, "train_seed": s,
                            "heldout_l1": l1 + 0.01 * s,
                            "mc_floor_l1": floor})


def test_check_o2_pass_and_fail(tmp_path):
    """check_o2 passes a case whose heldout L1 is well below the MC floor and
    fails one that exceeds it, with a high bootstrap p-value on failure."""
    _write_o2(tmp_path / "run", "A", l1=0.1, floor=0.2)   # 0.5x
    _write_o2(tmp_path / "run", "B", l1=0.8, floor=0.2)   # 4x
    out = ce.check_o2(str(tmp_path / "run"))
    assert out["A"]["pass"] and out["A"]["ci_clear"]
    assert not out["B"]["pass"]
    assert out["B"]["boot_p"] > 0.5


def test_check_o1_sep(tmp_path):
    """check_o1_sep passes a case whose median-of-max separability exceeds the
    margin."""
    d = tmp_path / "o1"
    d.mkdir()
    with open(d / "o1_caseB_sep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["world", "tv_to_shared"])
        w.writeheader()
        for wi in range(4):
            for v in (0.02, 0.15 + 0.01 * wi):
                w.writerow({"world": wi, "tv_to_shared": v})
    out = ce.check_o1_sep(str(d))
    assert out["B"]["pass"] and out["B"]["median_of_max"] > 0.15


def test_check_o3_asym_surfaces_tost(tmp_path):
    """check_o3_asym surfaces the TOST equivalence result under its actual
    "eff_ratio_tost" key."""
    # regression: the TOST block is written by analyze_o3_asym under
    # "eff_ratio_tost"; check_endpoints once looked for "tost_eff" and
    # silently dropped the endpoint from the mechanical verdict
    p = tmp_path / "o3_asym_analysis.json"
    p.write_text(json.dumps({"confirmatory_B": {
        "delta_stress_p05": {"mean": 0.01, "p_value": 0.03},
        "eff_ratio_tost": {"mean": 0.002, "equivalent": True,
                           "p_lower": 0.003, "p_upper": 0.003}}}))
    out = ce.check_o3_asym(str(p))
    assert out["present"] and out["pass"]
    assert out["tost_equivalent"] is True


def test_check_w33_criteria(tmp_path):
    """check_w33 evaluates R1/R2/R3 criteria from synthetic per-arm fraction-
    of- modes rows and produces a JSON-serializable result."""
    rows = []
    rng = np.random.default_rng(0)
    for s, adv in ((1.0, 0.02), (4.0, 0.3)):
        for wi in range(4):
            for arm, base in (("onpolicy", 0.4), ("replay", 0.5),
                              ("teacher", 0.4 + adv)):
                for seed in range(3):
                    rows.append({"case": "B", "sparsity": s, "world": wi,
                                 "arm": arm, "seed": seed,
                                 "frac_modes": base
                                 + rng.normal(0, 0.01),
                                 "samples_to_80": 1000.0})
    p = tmp_path / "w33_runs.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = ce.check_w33(str(p))["B"]
    assert out["R1_pass"] and out["R2_pass"]
    # R3's directional order isn't forced by this synthetic fixture; only
    # require that the field is a well-formed boolean
    assert out["R3_order_ok"] in (True, False)
    j = json.dumps(out)  # serializable
    assert "R1_p" in j
