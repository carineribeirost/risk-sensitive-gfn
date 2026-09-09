"""Tests for worlds, cases, reward pipeline, and the exact target."""

import numpy as np
import pytest

from epgfn.cases import EPS_REWARD, log_reward, psi_and_masks, target_for
from epgfn.conditions import Condition, ConditionRanges, tied_risk
from epgfn.target import mc_floor, p_star, tv
from epgfn.worlds import (CASES, WorldConfig, grid_coords, is_hard,
                          make_world, sample_world, smooth_field)

CFG = WorldConfig(H=12)


def cond_on(case: str) -> Condition:
    """Build a Condition with risk aversion "on" (tight beta, positive rho) for
    the given case."""
    return Condition(4.0, 0.3, tied_risk(case, 0.3, 0.4))


def cond_off(case: str) -> Condition:
    """Build a Condition with risk aversion "off" (beta=1, rho=0) for the given
    case."""
    return Condition(4.0, 0.3, tied_risk(case, 1.0, 0.0))


def test_grid_canonical_order():
    """grid_coords returns points in canonical flat-index order."""
    xy = grid_coords(3)
    assert xy.shape == (9, 3 - 1)
    # canonical flat index is x1*H + x2
    assert (xy[:, 0] * 3 + xy[:, 1] == np.arange(9)).all()


def test_smooth_field_range():
    """smooth_field produces a flattened HxH field ranging over [0, 1]."""
    f = smooth_field(np.random.default_rng(0), 16)
    assert f.shape == (256,)
    assert f.min() == pytest.approx(0.0) and f.max() == pytest.approx(1.0)


@pytest.mark.parametrize("case", CASES)
def test_world_shapes_and_reward(case):
    """Each case's world yields finite, strictly positive log-rewards and a
    normalized target distribution."""
    w = make_world(case, CFG, seed=1)
    n = CFG.H ** 2
    assert w.g.shape == (n,)
    assert 0.0 < w.beta_min < 1.0
    for bound in w.beta_bounds.values():
        assert 0.0 < bound <= w.beta_min
    logr = log_reward(w, cond_on(case))
    assert logr.shape == (n,)
    assert np.all(np.isfinite(logr))          # strict positivity
    assert np.all(logr >= np.log(EPS_REWARD) - 1e-12)
    p = target_for(w, cond_on(case))
    assert p.shape == (n,)
    assert p.sum() == pytest.approx(1.0)
    assert np.all(p > 0)


def test_case_c_veto_hits_entire_reward():
    """Case C's veto mask forces log-reward down to the epsilon floor
    everywhere it is vetoed."""
    w = make_world("C", CFG, seed=2)
    risk = tied_risk("C", 0.3, 0.4)
    _, _, veto_ok = psi_and_masks(w, risk)
    assert 0 < (~veto_ok).sum() < w.n_points  # veto region is non-trivial
    logr = log_reward(w, cond_on("C"))
    np.testing.assert_allclose(logr[~veto_ok], np.log(EPS_REWARD))


def test_case_b_floor_drives_to_eps():
    """Case B's floor mask forces log-reward down to the epsilon floor wherever
    the floor condition fails."""
    w = make_world("B", CFG, seed=3)
    _, floor_ok, _ = psi_and_masks(w, tied_risk("B", 0.3, 0.4))
    logr = log_reward(w, cond_on("B"))
    if (~floor_ok).any():
        np.testing.assert_allclose(logr[~floor_ok], np.log(EPS_REWARD))


def test_log_reward_subset_matches_full():
    """log_reward computed on an index subset matches the corresponding entries
    of the full computation."""
    for case in CASES:
        w = make_world(case, CFG, seed=4)
        idx = np.array([0, 5, 77, 143])
        np.testing.assert_allclose(log_reward(w, cond_on(case), idx),
                                   log_reward(w, cond_on(case))[idx])


def test_beta_t_concentrates_target():
    """A higher temperature beta_t concentrates the target distribution's mass
    more sharply."""
    w = make_world("A", CFG, seed=5)
    p_cold = target_for(w, Condition(1.0, 0.3, tied_risk("A", 0.3, 0.4)))
    p_hot = target_for(w, Condition(8.0, 0.3, tied_risk("A", 0.3, 0.4)))
    assert p_hot.max() > p_cold.max()


def test_hardness_gate_accepts_moving_targets():
    """sample_world's hardness gate accepts worlds where risk-on/off targets
    differ, taking at least one attempt."""
    for case in CASES:
        world, attempts = sample_world(case, CFG, seed=0)
        assert is_hard(world)
        assert attempts >= 1
        assert tv(target_for(world, cond_on(case)),
                  target_for(world, cond_off(case))) >= 0.0


def test_mc_floor_shrinks_with_samples():
    """mc_floor's estimated floor shrinks as the number of Monte Carlo samples
    increases."""
    w = make_world("A", CFG, seed=6)
    p = target_for(w, cond_on("A"))
    rng = np.random.default_rng(0)
    f_small = mc_floor(p, 200, rng)
    f_big = mc_floor(p, 20_000, rng)
    assert f_big < f_small


def test_p_star_temperature_identity():
    """p_star is invariant to jointly rescaling log-rewards and inverse-scaling
    the temperature."""
    logr = np.random.default_rng(7).normal(size=100)
    p1 = p_star(logr, 2.0)
    p2 = p_star(2.0 * logr, 1.0)
    np.testing.assert_allclose(p1, p2, atol=1e-12)


_TEST_BOUNDS = {"A": {"n": 0.1}, "B": {"p": 0.1, "m": 0.2},
                "C": {"p": 0.1}, "D": {"in": 0.15, "out": 0.1}}


def _risk_betas(case, risk):
    if case == "A":
        return {"n": risk.beta}
    if case == "B":
        return {"p": risk.beta_p, "m": risk.beta_m}
    if case == "C":
        return {"p": risk.beta}
    return {"in": risk.beta_in, "out": risk.beta_out}


@pytest.mark.parametrize("case", CASES)
def test_condition_features_bounded(case):
    """ConditionRanges.features stays within [-1, 1], and sampled conditions
    respect the configured beta bounds."""
    r = ConditionRanges(case=case)
    r.beta_bounds = dict(_TEST_BOUNDS[case])
    rng = np.random.default_rng(8)
    conds = [r.sample(rng) for _ in range(64)] + r.heldout_grid()
    f = r.features(conds)
    assert f.shape[1] == r.n_features
    assert np.all(f >= -1 - 1e-9) and np.all(f <= 1 + 1e-9)
    for c in conds:  # per-set bounds respected
        for key, beta in _risk_betas(case, c.risk).items():
            assert beta >= r.beta_bounds[key] - 1e-12


def test_heldout_grid_sizes():
    """heldout_grid produces the expected number of held-out condition
    combinations per case."""
    # D: 3 β_t × 2 β_in × 2^O ρ × 2 β_out = 192 at O = 4
    sizes = {"A": 27, "B": 243, "C": 81, "D": 192}
    for case, n in sizes.items():
        r = ConditionRanges(case=case)
        assert len(r.heldout_grid()) == n
