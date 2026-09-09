"""O3 tests: worst-case pole correctness, satisfaction semantics, and
the compensation effect on a constructed geometry: the risk-on target
must put more mass on the jointly-satisfying point than Boltzmann,
which is fooled by a point whose strong state compensates a weak one."""

import numpy as np

from epgfn.cases import psi_worst_case, target_for, target_worst
from epgfn.conditions import Condition, tied_risk
from epgfn.o3 import (challenge_level, dist_metrics, draw_stress,
                      run_o3, satisfaction_mask, stress_metrics)
from epgfn.worlds import CASES, World, WorldConfig, make_world

CFG = WorldConfig(H=12)


def _hand_world_A():
    """4-point world, K=2, w_g=0 isolates Ψ. Point P=(0.80, 0.75) is
    jointly good; Q=(1.00, 0.60) has a higher MEAN (0.800 vs 0.775) via
    compensation but a worse joint profile; two filler points are bad."""
    scores = np.array([[0.80, 0.75],
                       [1.00, 0.60],
                       [0.05, 0.05],
                       [0.10, 0.05]])
    return World(case="A", cfg=WorldConfig(H=2), seed=0,
                 g=np.zeros(4), scores_neutral=scores,
                 w_neutral=np.array([0.5, 0.5]))


def test_worst_case_pole_is_min():
    """The worst-case pole psi_worst_case matches the per-point minimum over
    sub- scores, for cases A and D."""
    for case in CASES:
        w = make_world(case, CFG, seed=1)
        psi, _, _ = psi_worst_case(w)
        assert psi.shape == (w.n_points,)
    wa = make_world("A", CFG, seed=1)
    np.testing.assert_allclose(psi_worst_case(wa)[0],
                               wa.scores_neutral.min(axis=-1))
    wd = make_world("D", CFG, seed=1)
    np.testing.assert_allclose(
        psi_worst_case(wd)[0],
        wd.scores_nested.reshape(wd.n_points, -1).min(axis=-1))


def test_compensation_boltzmann_fooled_risk_on_not():
    """Boltzmann targeting is fooled by a compensating point, while risk-on and
    worst-case targeting are not."""
    w = _hand_world_A()
    t = 0.7
    sat = satisfaction_mask(w, t)
    assert sat.tolist() == [True, False, False, False]  # only P
    boltz = target_for(w, Condition(8.0, 0.0, tied_risk("A", 1.0, 0.0)))
    risk = target_for(w, Condition(8.0, 0.0, tied_risk("A", 0.5, 0.0)))
    worst = target_worst(w, Condition(8.0, 0.0, tied_risk("A", 1.0, 0.0)))
    m_b = dist_metrics(boltz, sat)["sat_mass"]
    m_r = dist_metrics(risk, sat)["sat_mass"]
    m_w = dist_metrics(worst, sat)["sat_mass"]
    assert m_b < 0.5            # Boltzmann prefers the compensator Q
    assert m_r > m_b + 0.2      # risk-on recovers joint satisfaction
    assert m_w > m_b            # worst-case also targets P here


def test_satisfaction_mask_cases():
    """satisfaction_mask matches manual threshold/veto logic
    for cases B and C."""
    wb = make_world("B", CFG, seed=2)
    sat = satisfaction_mask(wb, 0.4)
    manual = ((wb.scores_plus >= 0.4).all(-1)
              & (wb.scores_minus <= 0.6).all(-1))
    assert (sat == manual).all()
    wc = make_world("C", CFG, seed=2)
    vetoed = np.any(wc.scores_named >= wc.c_named, axis=-1)
    assert not satisfaction_mask(wc, 0.0)[vetoed].any()


def test_challenge_level_nonempty():
    """challenge_level picks a threshold that leaves a nonempty satisfying set
    for every case."""
    for case in CASES:
        w = make_world(case, CFG, seed=3)
        t_star = challenge_level(w)
        assert satisfaction_mask(w, t_star).mean() >= 0.01


def test_stress_metrics_finite_and_ordered():
    """Stress metrics are finite and the 5th-percentile stress does not exceed
    the mean stress."""
    rng = np.random.default_rng(0)
    for case in CASES:
        w = make_world(case, CFG, seed=4)
        stress = draw_stress(w, rng, n_draws=64, kappa=25.0)
        p = np.full(w.n_points, 1.0 / w.n_points)
        m = stress_metrics(w, p, stress)
        assert np.isfinite(m["stress_mean"])
        assert m["stress_p05"] <= m["stress_mean"] + 1e-12


def test_run_o3_shapes():
    """run_o3 produces rows for every target kind with valid sat_mass, and one
    probe_cell row per t level."""
    rows, meta = run_o3("A", WorldConfig(H=8), [0], n_stress=32)
    kinds = {r["target"] for r in rows}
    assert kinds == {"boltzmann", "worst", "probe_cell", "risk"}
    assert meta[0]["skipped_cells"] >= 0
    for r in rows:
        assert 0.0 <= r["sat_mass"] <= 1.0 + 1e-12
    # the pre-named cell: one row per t level, at the probe-on member,
    # respecting the β bound
    pc = [r for r in rows if r["target"] == "probe_cell"]
    assert len(pc) == len({r["t"] for r in rows})
    assert all(r["rho"] == 0.5 for r in pc)
    assert all(r["beta_cvar"] >= 0.25 - 1e-12 for r in pc)
