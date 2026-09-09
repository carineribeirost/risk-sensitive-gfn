"""P2-primary recovery verdict: synthetic fixtures where the verdict
is known by construction."""

import csv
import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location(
    "check_endpoints",
    pathlib.Path(__file__).resolve().parent.parent
    / "scripts" / "check_endpoints.py")
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)


def _write(path, recover: bool):
    rows = []
    for w in range(8):
        for arm, sig, mass in (
                ("naive", 0.0, 0.40 + 0.01 * w),
                ("robust", 0.05, (0.60 if recover else 0.39)
                 + 0.01 * w),
                ("robust", 0.1, (0.65 if recover else 0.38)
                 + 0.01 * w),
                ("winsor", 0.0, 0.7)):
            rows.append({"case": "B", "world": w, "model": "crash",
                         "level": 0.05, "beta": 0.3, "rho": 0.2,
                         "arm": arm, "sigma": sig,
                         "sat_mass_true": mass, "tv_true": 0.1,
                         "n_skipped": 0})
    with open(path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)


def test_recovery_pass_and_fail(tmp_path):
    """check_recovery passes when a robust arm recovers saturation mass at some
    sigma and fails when none do."""
    good = tmp_path / "good.csv"
    _write(good, recover=True)
    out = ce.check_recovery(str(good))["B"]
    assert out["P2_primary_pass"] and out["passing_sigmas"]
    assert out["winsor_mean_delta"] > 0

    bad = tmp_path / "bad.csv"
    _write(bad, recover=False)
    out = ce.check_recovery(str(bad))["B"]
    assert not out["P2_primary_pass"]
