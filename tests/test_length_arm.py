"""Length arm (TB vs SubTB on the depth axis): the pairing is by
(world, seed), the loss kind is read from args.json rather than the
CSV's same-named value column, and a mismatched pair of run dirs is
refused instead of silently compared."""

import csv
import json
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                       / "scripts"))

from analyze_length_arm import (analyse, compare_case,  # noqa: E402
                                _config_check, _read_arm, add_holm)

BASE = {"H": 16, "d": 3, "geometry": "grid", "worlds": 8,
        "world_seed0": 0, "seeds": 2, "steps": 12000, "cond_pool": 256,
        "logit_floor": -25.0, "weight_alpha": 2.0, "use_sigma": False,
        "use_rho_out": False, "k_guard": 0, "use_guard_delta": False}


def _make_dir(tmp_path, name, loss, l1_of, cases=("A", "B"),
              worlds=tuple(range(8)), seeds=(0, 1), floor=0.2):
    d = tmp_path / name
    (d / "frag").mkdir(parents=True)
    with open(d / "args.json", "w") as fh:
        json.dump(dict(BASE, loss=loss), fh)
    for case in cases:
        with open(d / "frag" / f"o2_case{case}.csv", "w",
                  newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "case", "world", "gate_attempts", "train_seed",
                "heldout_l1", "mc_floor_l1", "loss", "wall_s"])
            w.writeheader()
            for world in worlds:
                for seed in seeds:
                    w.writerow({"case": case, "world": world,
                                "gate_attempts": 1, "train_seed": seed,
                                "heldout_l1": l1_of(case, world, seed),
                                "mc_floor_l1": floor,
                                # the CSV column is the final loss
                                # VALUE, deliberately misleading here
                                "loss": 17.0, "wall_s": 1.0})
    return d


def test_reads_runs_and_pairs_by_world_and_seed(tmp_path):
    d = _make_dir(tmp_path, "tb", "tb",
                  lambda c, w, s: 0.5 + 0.1 * w + 0.01 * s)
    arm = _read_arm(str(d))
    assert set(arm) == {"A", "B"}
    assert arm["A"][("0", "1")][0] == pytest.approx(0.51)
    assert len(arm["B"]) == 16


def test_loss_kind_comes_from_args_not_the_csv_column(tmp_path):
    tb = _make_dir(tmp_path, "tb", "tb", lambda c, w, s: 0.5)
    sub = _make_dir(tmp_path, "sub", "subtb", lambda c, w, s: 0.4)
    assert _config_check(str(tb), str(sub))["ok"]
    # two TB dirs are not a length arm, however their CSVs look
    tb2 = _make_dir(tmp_path, "tb2", "tb", lambda c, w, s: 0.4)
    bad = _config_check(str(tb), str(tb2))
    assert not bad["ok"] and bad["diffs"]["loss"] == ["tb", "tb"]


def test_config_mismatch_is_refused_then_allowed(tmp_path):
    tb = _make_dir(tmp_path, "tb", "tb", lambda c, w, s: 0.5)
    sub = _make_dir(tmp_path, "sub", "subtb", lambda c, w, s: 0.4)
    with open(sub / "args.json", "w") as fh:
        json.dump(dict(BASE, loss="subtb", steps=36000), fh)
    cells = [("d3", str(tb), str(sub))]
    with pytest.raises(SystemExit):
        analyse(cells)
    rep = analyse(cells, strict=False)
    assert rep["cells"]["d3"]["config_check"]["diffs"]["steps"] == \
        [12000, 36000]


def test_paired_verdict_direction_and_endpoint_restoration(tmp_path):
    # TB above the 3x line (0.9 / 0.2 = 4.5x), SubTB below it
    tb = _make_dir(tmp_path, "tb", "tb",
                   lambda c, w, s: 0.90 + 0.02 * w + 0.01 * s)
    sub = _make_dir(tmp_path, "sub", "subtb",
                    lambda c, w, s: 0.40 + 0.02 * w + 0.01 * s)
    rep = add_holm(analyse([("d3", str(tb), str(sub))]))
    r = rep["cells"]["d3"]["cases"]["B"]
    assert r["verdict"] == "subtb better"
    assert r["paired_l1"]["mean"] == pytest.approx(-0.5)
    assert r["tb"]["ratio"] > 3.0 >= r["subtb"]["ratio"]
    assert r["restores_endpoint"]
    assert r["n_worlds"] == 8 and r["n_paired_runs"] == 16
    assert r["power"]["alpha_reachable"]
    assert set(rep["holm"]) == {"d3:A", "d3:B"}


def test_no_difference_when_arms_coincide(tmp_path):
    f = lambda c, w, s: 0.5 + 0.05 * w  # noqa: E731
    tb = _make_dir(tmp_path, "tb", "tb", f)
    sub = _make_dir(tmp_path, "sub", "subtb", f)
    r = analyse([("d3", str(tb), str(sub))])["cells"]["d3"]["cases"]["A"]
    assert r["verdict"] == "no difference"
    assert r["paired_l1"]["mean"] == pytest.approx(0.0)
    assert r["paired_l1"]["cohens_d"] == 0.0


def test_four_worlds_report_underpowered_not_no_difference(tmp_path):
    """The sign test bottoms out at p = 0.125 with 4 worlds, so a
    uniform improvement must not be reported as no difference."""
    tb = _make_dir(tmp_path, "tb", "tb", lambda c, w, s: 0.9,
                   worlds=(0, 1, 2, 3))
    sub = _make_dir(tmp_path, "sub", "subtb", lambda c, w, s: 0.4,
                    worlds=(0, 1, 2, 3))
    r = analyse([("d3", str(tb), str(sub))])["cells"]["d3"]["cases"]["A"]
    assert r["verdict"] == "subtb better in every world, underpowered"
    assert r["power"] == {"min_attainable_p": 0.125,
                          "alpha_reachable": False,
                          "worlds_needed_for_alpha": 6,
                          "consistent_direction": True}


def test_unshared_runs_are_dropped_and_reported():
    tb = {("0", "0"): (0.5, 0.2), ("0", "1"): (0.6, 0.2),
          ("1", "0"): (0.7, 0.2)}
    sub = {("0", "0"): (0.4, 0.2), ("1", "0"): (0.5, 0.2)}
    r = compare_case(tb, sub)
    assert r["n_paired_runs"] == 2
    assert r["dropped_runs"] == [["0", "1"]]
    assert r["per_world"]["worlds"] == ["0", "1"]
    assert np.allclose(r["per_world"]["subtb_l1"], [0.4, 0.5])


def test_disjoint_runs_error_rather_than_crash():
    r = compare_case({("0", "0"): (0.5, 0.2)},
                     {("9", "9"): (0.4, 0.2)})
    assert "error" in r
