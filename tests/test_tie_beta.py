"""Tied-beta variant of case B (intrinsic dim 5, features 6-D)."""

import numpy as np
import pytest

from epgfn.conditions import ConditionRanges


def test_tied_sample_shares_beta_keeps_rho_free():
    """tie_beta=True forces beta_p == beta_m respecting both bounds while rho_p
    and rho_m stay independently drawn."""
    r = ConditionRanges(case="B", tie_beta=True,
                        beta_bounds={"p": 0.10, "m": 0.20})
    rng = np.random.default_rng(0)
    conds = [r.sample(rng) for _ in range(64)]
    for c in conds:
        assert c.risk.beta_p == c.risk.beta_m
        assert c.risk.beta_p >= 0.20  # shared beta respects BOTH bounds
    # rho axes stay independent draws
    assert any(c.risk.rho_p != c.risk.rho_m for c in conds)


def test_tied_heldout_grid_size_and_tying():
    """The tied held-out grid collapses to 81 points with beta_p == beta_m
    throughout, while the untied grid keeps 243."""
    r = ConditionRanges(case="B", tie_beta=True)
    grid = r.heldout_grid()
    assert len(grid) == 81  # 3 beta_t x 3 beta x 3 rho+ x 3 rho-
    assert all(c.risk.beta_p == c.risk.beta_m for c in grid)
    # untied grid unchanged
    assert len(ConditionRanges(case="B").heldout_grid()) == 243


def test_tied_features_stay_6d_with_duplicated_beta():
    """Tied conditions still yield 6-D feature vectors, with the two beta
    feature columns duplicated under matching bounds."""
    r = ConditionRanges(case="B", tie_beta=True)
    assert r.n_features == 6
    rng = np.random.default_rng(1)
    conds = [r.sample(rng) for _ in range(8)]
    f = r.features(conds)
    assert f.shape == (8, 6)
    # same bounds for both sides here -> duplicated normalized value
    assert np.allclose(f[:, 2], f[:, 4])


def test_tie_beta_rejected_outside_case_b():
    """tie_beta=True raises ValueError on both sample and heldout_grid for
    every case other than B."""
    rng = np.random.default_rng(0)
    for case in ("A", "C", "D"):
        r = ConditionRanges(case=case, tie_beta=True)
        with pytest.raises(ValueError):
            r.sample(rng)
        with pytest.raises(ValueError):
            r.heldout_grid()
