"""Graded penalty on excluded points.

The plain reward gives every excluded point the same epsilon, so the
excluded region is exactly flat, and a flat region is what lets an
on-policy learner settle inside it. The ablation replaces that constant
with epsilon * exp(-kappa * violation depth): same boundary, no plateau.
"""
import numpy as np
import pytest

from epgfn.cases import (EPS_REWARD, log_reward, log_reward_graded,
                         psi_and_masks, violation_depth)
from epgfn.conditions import Condition, tied_risk
from epgfn.worlds import WorldConfig, make_world

CFG = WorldConfig(H=16, d=2, geometry="grid")
CASES = ("A", "B", "C", "D")


def _w(case):
    return make_world(case, CFG, seed=0)


def _c(case):
    return Condition(4.0, 0.3, tied_risk(case, 0.25, 0.5))


@pytest.mark.parametrize("case", CASES)
def test_kappa_zero_reproduces_the_plain_reward_bitwise(case):
    w, c = _w(case), _c(case)
    assert np.array_equal(log_reward(w, c), log_reward_graded(w, c, 0.0))


@pytest.mark.parametrize("case", CASES)
def test_depth_is_a_fraction(case):
    d = violation_depth(_w(case), _c(case).risk)
    assert d.shape == (_w(case).n_points,)
    assert np.all((d >= 0.0) & (d <= 1.0))
    assert np.all(np.isfinite(d))


@pytest.mark.parametrize("case", CASES)
def test_feasible_points_have_zero_depth_and_unchanged_reward(case):
    """The penalty must not touch anything inside the feasible set."""
    w, c = _w(case), _c(case)
    _, floor_ok, veto_ok = psi_and_masks(w, c.risk)
    ok = floor_ok & veto_ok
    assert np.all(violation_depth(w, c.risk)[ok] == 0.0)
    flat, graded = log_reward(w, c), log_reward_graded(w, c, 5.0)
    assert np.array_equal(flat[ok], graded[ok])


def test_the_plateau_disappears_on_the_floor_case():
    """The whole point: one distinct reward becomes many."""
    w, c = _w("B"), _c("B")
    flat, graded = log_reward(w, c), log_reward_graded(w, c, 5.0)
    dead = np.isclose(flat, np.log(EPS_REWARD))
    assert dead.mean() > 0.5, "expected the floor to exclude most of X"
    assert len(np.unique(flat[dead])) == 1
    assert len(np.unique(graded[dead])) > 10


def test_the_penalty_is_monotone_in_depth():
    """Deeper violations must be punished at least as hard, or the
    gradient the ablation adds would point the wrong way."""
    w, c = _w("B"), _c("B")
    d = violation_depth(w, c.risk)
    g = log_reward_graded(w, c, 5.0)
    _, f_ok, v_ok = psi_and_masks(w, c.risk)
    bad = ~(f_ok & v_ok)
    order = np.argsort(d[bad])
    vals = g[bad][order]
    assert np.all(np.diff(vals) <= 1e-12)


def test_the_boundary_still_costs_epsilon():
    """At depth 0 the graded reward equals the flat one, so the two
    agree exactly on the constraint surface and differ only inside."""
    w, c = _w("C"), _c("C")
    d = violation_depth(w, c.risk)
    g = log_reward_graded(w, c, 5.0)
    _, f_ok, v_ok = psi_and_masks(w, c.risk)
    on_edge = (~(f_ok & v_ok)) & (d == 0.0)
    if on_edge.any():
        assert np.allclose(g[on_edge], np.log(EPS_REWARD))


@pytest.mark.parametrize("case", CASES)
def test_graded_reward_stays_finite_and_below_the_clamp(case):
    w, c = _w(case), _c(case)
    g = log_reward_graded(w, c, 12.0)
    assert np.all(np.isfinite(g))
    _, f_ok, v_ok = psi_and_masks(w, c.risk)
    bad = ~(f_ok & v_ok)
    if bad.any():
        assert np.all(g[bad] <= np.log(EPS_REWARD) + 1e-12)


def test_larger_kappa_punishes_harder():
    w, c = _w("B"), _c("B")
    _, f_ok, v_ok = psi_and_masks(w, c.risk)
    bad = ~(f_ok & v_ok)
    g2 = log_reward_graded(w, c, 2.0)[bad]
    g8 = log_reward_graded(w, c, 8.0)[bad]
    assert np.all(g8 <= g2 + 1e-12)
    assert g8.sum() < g2.sum()
