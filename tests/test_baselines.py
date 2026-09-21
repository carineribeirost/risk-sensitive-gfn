"""Baseline aggregations: shape, bounds, and the identities that must hold."""
import numpy as np
import pytest

from epgfn.baselines import (KINDS, _desirability, _wgeo, psi_baseline,
                             target_baseline)
from epgfn.cases import psi_worst_case
from epgfn.conditions import Condition, tied_risk
from epgfn.worlds import WorldConfig, make_world

CFG = WorldConfig(H=8, d=2, geometry="grid")
CASES = ("A", "B", "C", "D")


def _world(case):
    return make_world(case, CFG, seed=0)


def _cond(case):
    return Condition(2.0, 0.3, tied_risk(case, 0.5, 0.2))


def test_weighted_geometric_mean_matches_closed_form():
    a = np.array([[0.2, 0.8, 0.5]])
    w = np.array([0.5, 0.25, 0.25])
    assert _wgeo(a, w)[0] == pytest.approx(
        0.2 ** 0.5 * 0.8 ** 0.25 * 0.5 ** 0.25)


def test_geometric_mean_is_bounded_by_the_extremes():
    """AM-GM's neighbour: the geometric mean lies between min and max."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0.01, 1.0, size=(200, 5))
    w = rng.dirichlet(np.ones(5))
    g = _wgeo(a, w)
    assert np.all(g >= a.min(axis=-1) - 1e-12)
    assert np.all(g <= a.max(axis=-1) + 1e-12)


def test_desirability_is_a_clamped_ramp():
    a = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    d = _desirability(a, np.float64(0.25), np.float64(0.75), 1.0)
    assert d[0] == 0.0 and d[1] == 0.0          # at or below lo
    assert d[2] == pytest.approx(0.5)           # midpoint, shape 1
    assert d[3] == 1.0 and d[4] == 1.0          # at or above hi
    assert np.all((d >= 0.0) & (d <= 1.0))


def test_hard_min_is_exactly_the_worst_pole():
    """The reviewer's 'hard min' baseline already exists; assert it."""
    for case in CASES:
        w = _world(case)
        a = psi_baseline(w, "hard_min")
        b = psi_worst_case(w)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_psi_shape_and_masks(case, kind):
    w = _world(case)
    psi, floor_ok, veto_ok = psi_baseline(w, kind)
    assert psi.shape == (w.n_points,)
    assert np.all(np.isfinite(psi))
    assert floor_ok.shape == veto_ok.shape == (w.n_points,)
    assert floor_ok.dtype == bool and veto_ok.dtype == bool


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_target_is_a_distribution(case, kind):
    p = target_baseline(_world(case), _cond(case), kind)
    assert p.shape == (CFG.H ** CFG.d,)
    assert np.all(p >= 0.0)
    assert p.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("case", CASES)
def test_case_geometry_is_preserved(case):
    """A baseline changes the pooling, never the exclusion structure:
    case C must still veto, case B must still floor."""
    w = _world(case)
    _, floor_b, veto_b = psi_baseline(w, "geometric")
    _, floor_w, veto_w = psi_worst_case(w)
    if case == "C":
        # the veto is a property of the scores, not of the aggregation
        assert np.array_equal(veto_b, veto_w)
    if case in ("A", "D"):
        assert floor_b.all()


def test_idx_subsetting_agrees_with_full_evaluation():
    w = _world("C")
    idx = np.array([3, 17, 42, 60])
    full, _, _ = psi_baseline(w, "geometric")
    sub, _, _ = psi_baseline(w, "geometric", idx=idx)
    assert np.allclose(sub, full[idx])


def test_desirability_bounds_are_declared_not_fixed():
    """Changing the declared quantiles must change the target, or the
    'declared choice' claim in the rebuttal would be empty."""
    w, c = _world("A"), _cond("A")
    a = target_baseline(w, c, "desirability", lo_q=0.50, hi_q=0.95)
    b = target_baseline(w, c, "desirability", lo_q=0.10, hi_q=0.99)
    assert not np.allclose(a, b)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        psi_baseline(_world("A"), "not-a-rule")
