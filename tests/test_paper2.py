"""Paper-2 machinery: σ dial semantics (incl. the δ≡σ unification and
monotonicity), winsorization, corruption models, and the recovery
pipeline end-to-end on a tiny world. σ=0 bitwise identity is gated by
the existing golden suite (test_w31_regression)."""

import dataclasses

import numpy as np
import pytest

from epgfn.cases import psi_and_masks, target_for
from epgfn.conditions import Condition, ConditionRanges, RiskC, tied_risk
from epgfn.corrupt import MODELS, corrupt_world, crude_world
from epgfn.risk import winsorize_scores
from epgfn.train import TrainConfig, ranges_for, train_policy
from epgfn.worlds import WorldConfig, sample_world


@pytest.fixture(scope="module")
def worlds():
    """Sample one world per case (A-D) for reuse across the paper-2 tests."""
    return {c: sample_world(c, WorldConfig(H=8), seed=0)[0]
            for c in ("A", "B", "C", "D")}


def test_delta_sigma_unification_exact(worlds):
    """A box-robust sigma veto is bitwise identical to an equivalent margin
    shift via delta, on case C."""
    # box-robust veto == margin shift, bitwise (whiteboard §3)
    w = worlds["C"]
    a = psi_and_masks(w, RiskC(0.5, 0.3, delta=0.02, sigma=0.05))[2]
    b = psi_and_masks(w, RiskC(0.5, 0.3, delta=0.07, sigma=0.0))[2]
    assert np.array_equal(a, b)


def test_sigma_monotone_conservative(worlds):
    """Adding box-robust sigma never increases psi relative to the nominal
    (sigma=0) target, for every case."""
    # robust psi <= nominal psi everywhere, all four cases
    for case, w in worlds.items():
        r0 = tied_risk(case, 0.5, 0.5)
        r1 = dataclasses.replace(r0, sigma=0.1)
        psi0 = psi_and_masks(w, r0)[0]
        psi1 = psi_and_masks(w, r1)[0]
        assert (psi1 <= psi0 + 1e-12).all(), case


def test_sigma_zero_identical_targets(worlds):
    """sigma=0 produces a target identical to the tied-risk target without an
    explicit sigma override, for every case."""
    for case, w in worlds.items():
        c0 = Condition(2.0, 0.3, tied_risk(case, 0.5, 0.5))
        c1 = Condition(2.0, 0.3, dataclasses.replace(
            tied_risk(case, 0.5, 0.5), sigma=0.0))
        assert np.array_equal(target_for(w, c0), target_for(w, c1))


def test_sigma_ranges_axis(worlds):
    """Enabling use_sigma adds one feature axis, samples sigma within bounds,
    and the heldout grid spans three sigma values."""
    r = ranges_for(worlds["B"])
    n0 = r.n_features
    r.use_sigma = True
    assert r.n_features == n0 + 1
    rng = np.random.default_rng(0)
    c = r.sample(rng)
    assert 0.0 <= c.risk.sigma <= 0.2
    grid = r.heldout_grid()
    sigmas = {c.risk.sigma for c in grid}
    assert len(sigmas) == 3
    f = r.features(grid[:8])
    assert f.shape[1] == n0 + 1
    assert np.all(np.abs(f) <= 1.0 + 1e-9)


def test_sigma_training_smoke(worlds):
    """Training with the sigma axis enabled produces a finite held-out L1."""
    r = ranges_for(worlds["A"])
    r.use_sigma = True
    cfg = TrainConfig(steps=20, n_conds=2, n_points=16, eval_every=20,
                      seed=0, cond_pool=16,
                      net=dict(dim=32, depth=2, cond_dim=16, x1_dim=8))
    _, _, hist = train_policy(worlds["A"], cfg, r)
    assert np.isfinite(hist[-1]["heldout_l1"])


def test_winsorize_bounds_and_identity():
    """winsorize_scores clips each row to its second-smallest/largest value and
    rejects too few columns."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, size=(50, 5))
    a[3, 2] = 99.0  # crash-like outlier
    w = winsorize_scores(a, 1)
    srt = np.sort(a, axis=1)
    assert (w >= srt[:, 1:2] - 1e-12).all()
    assert (w <= srt[:, -2:-1] + 1e-12).all()
    assert w[3, 2] == pytest.approx(srt[3, -2])
    with pytest.raises(ValueError):
        winsorize_scores(a[:, :2], 1)


def test_corrupt_models_deterministic_and_scoped(worlds):
    """Each corruption model is deterministic for a fixed epsilon, leaves the
    true world's structure untouched, and affects only its intended scope."""
    w = worlds["B"]
    for model in MODELS:
        c1 = corrupt_world(w, model, 0.1)
        c2 = corrupt_world(w, model, 0.1)
        assert np.array_equal(c1.scores_plus, c2.scores_plus)
        # true world untouched
        assert c1 is not w and not np.shares_memory(
            c1.scores_plus, w.scores_plus)
        # g and world parameters never corrupted
        assert np.array_equal(c1.g, w.g)
        assert c1.floor_value == w.floor_value
    crash = corrupt_world(w, "crash", 0.2)
    frac = np.mean(crash.scores_plus != w.scores_plus)
    assert 0.1 < frac < 0.3  # ~eps_s per entry
    bias = corrupt_world(w, "bias", 0.0)
    changed_cols = sum(
        int(not np.array_equal(getattr(bias, a)[..., j],
                               getattr(w, a)[..., j]))
        for a in ("scores_plus", "scores_minus") if getattr(w, a) is not None
        for j in range(getattr(w, a).shape[-1]))
    assert changed_cols <= 1  # at most one field touched (may hit g? no)


def test_crude_world_levels(worlds):
    """crude_world discretizes scores_plus down to the coarse {0, 0.5, 1} level
    set."""
    cw = crude_world(worlds["C"])
    assert set(np.unique(cw.scores_plus)) <= {0.0, 0.5, 1.0}


def test_recovery_script_smoke(tmp_path):
    """The run_recovery.py script runs end-to-end and writes a recovery.csv
    with the expected arms and valid sat_mass_true values."""
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "scripts/run_recovery.py", "--cases", "A",
         "--worlds", "1", "--world-seed0", "0", "--H", "8",
         "--models", "crash", "--levels", "0.1",
         "--out", str(tmp_path / "rec")],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    import csv as csvmod
    import glob
    path = glob.glob(str(tmp_path / "rec" / "*" / "*" / "recovery.csv"))
    rows = list(csvmod.DictReader(open(path[0])))
    arms = {row["arm"] for row in rows}
    assert {"true", "crude", "naive", "robust", "winsor"} <= arms
    # the true-oracle arm is the ceiling: no other arm beats it on
    # its own metric by construction is NOT guaranteed per-condition,
    # but sat_mass must be finite and in [0,1]
    for row in rows:
        assert 0.0 <= float(row["sat_mass_true"]) <= 1.0 + 1e-9
