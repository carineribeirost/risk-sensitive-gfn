"""Oracle tests for per-origin risk: each origin's
(β, ρ) acts only on its own functional; C's δ is exactly a threshold
shift; D's per-origin radii match an explicit per-o loop; bounds are
per-set; the tied member reproduces the shared-pair behavior; and the
O1b separability metric is positive on a gated world."""

import numpy as np
import pytest

from epgfn.cases import psi_and_masks
from epgfn.conditions import RiskB, RiskC, RiskD, tied_risk
from epgfn.o1 import o1_separability
from epgfn.o3 import run_o3
from epgfn.risk import cvar_lower, cvar_upper, dro_cvar_lower
from epgfn.worlds import WorldConfig, make_world, sample_world

CFG = WorldConfig(H=12)


def test_b_rho_minus_only_hits_minus_side():
    """With rho_minus to infinity and rho_plus at zero, only the minus side
    collapses to its worst state."""
    # ρ⁻ → ∞ drives Φ(S⁻) to the weight-free max state while the plus
    # side stays the plain CVaR at nominal weights (per-origin limits)
    w = make_world("B", CFG, seed=1)
    psi, _, _ = psi_and_masks(w, RiskB(0.5, 0.0, 0.5, 50.0))
    want = (cvar_lower(w.scores_plus, w.p_plus, 0.5)
            - w.cfg.gamma * w.scores_minus.max(axis=-1))
    np.testing.assert_allclose(psi, want, atol=1e-5)


def test_b_rho_plus_only_hits_plus_side():
    """With rho_plus to infinity and rho_minus at zero, only the plus side
    collapses to its worst state."""
    w = make_world("B", CFG, seed=1)
    psi, _, _ = psi_and_masks(w, RiskB(0.5, 50.0, 0.5, 0.0))
    want = (w.scores_plus.min(axis=-1)
            - w.cfg.gamma * cvar_upper(w.scores_minus, w.q_minus, 0.5))
    np.testing.assert_allclose(psi, want, atol=1e-5)


def test_tied_member_equals_shared_pair():
    """tied_risk on case B reproduces the same psi as an explicit RiskB pair
    with matching (beta, rho) on both sides."""
    w = make_world("B", CFG, seed=6)
    psi_tied, _, _ = psi_and_masks(w, tied_risk("B", 0.4, 0.3))
    psi_split, _, _ = psi_and_masks(w, RiskB(0.4, 0.3, 0.4, 0.3))
    np.testing.assert_array_equal(psi_tied, psi_split)


def test_c_delta_is_threshold_shift():
    """RiskC's delta shifts the veto threshold exactly as a manual threshold
    comparison."""
    w = make_world("C", CFG, seed=2)
    delta = 0.07
    _, _, veto_ok = psi_and_masks(w, RiskC(0.4, 0.3, delta))
    manual = ~np.any(w.scores_named >= w.c_named - delta, axis=-1)
    assert (veto_ok == manual).all()


def test_c_delta_monotone():
    """Increasing delta in RiskC only shrinks the set of veto-satisfying
    points."""
    w = make_world("C", CFG, seed=2)
    prev_ok = None
    for d in (0.0, 0.03, 0.06, 0.1):
        _, _, ok = psi_and_masks(w, RiskC(0.4, 0.3, d))
        if prev_ok is not None:  # larger δ only shrinks the allowed set
            assert np.all(~ok | prev_ok)
        prev_ok = ok


def test_d_per_origin_rho_matches_loop():
    """dro_cvar_lower with a per-origin rho vector matches an explicit per-
    origin loop over the same computation."""
    w = make_world("D", CFG, seed=3)
    rho_vec = 0.8 * w.rel_profile
    vec = dro_cvar_lower(w.scores_nested, w.w_inner, 0.5, rho_vec)
    manual = np.stack(
        [dro_cvar_lower(w.scores_nested[:, o, :], w.w_inner[o], 0.5,
                        float(rho_vec[o]))
         for o in range(w.w_inner.shape[0])], axis=-1)
    np.testing.assert_allclose(vec, manual, atol=1e-7)


def test_d_shared_rho_monotone():
    """Increasing the shared outer rho in RiskD never increases psi."""
    w = make_world("D", CFG, seed=3)
    psi0, _, _ = psi_and_masks(w, RiskD(0.5, 0.0))
    assert np.all(np.isfinite(psi0))
    psi1, _, _ = psi_and_masks(w, RiskD(0.5, 1.0))
    assert np.all(psi1 <= psi0 + 1e-9)  # more robust → lower


def test_d_scalar_rho_equals_shared_tuple():
    """A scalar outer rho in RiskD is equivalent to a tuple with that value
    repeated for every outer origin."""
    w = make_world("D", CFG, seed=3)
    psi_s, _, _ = psi_and_masks(w, RiskD(0.5, 0.7, 0.5))
    psi_t, _, _ = psi_and_masks(w, RiskD(0.5, (0.7,) * CFG.n_outer, 0.5))
    np.testing.assert_array_equal(psi_s, psi_t)


def test_d_aimed_distrust_and_dial_move_target():
    """Per-origin distrust vectors and the compensatory/conjunctive outer dial
    each change psi, conjunctive never exceeding compensatory."""
    w = make_world("D", CFG, seed=3)
    n_o = CFG.n_outer
    v = np.full(n_o, 0.05)
    v[1] = 1.0
    psi_aim, _, _ = psi_and_masks(w, RiskD(0.5, tuple(v), 0.5))
    psi_shared, _, _ = psi_and_masks(w, RiskD(0.5, 0.05, 0.5))
    assert not np.allclose(psi_aim, psi_shared, atol=1e-4)
    # the joint-satisfaction dial: compensatory vs conjunctive
    psi_comp, _, _ = psi_and_masks(w, RiskD(0.5, 0.5, 1.0))
    psi_conj, _, _ = psi_and_masks(w, RiskD(0.5, 0.5,
                                            float(w.beta_bounds["out"])))
    assert not np.allclose(psi_comp, psi_conj, atol=1e-4)
    assert np.all(psi_conj <= psi_comp + 1e-9)  # deeper outer tail → lower


def test_d_profile_shape():
    """rel_profile has one entry per outer origin, peaks at 1.0, and stays
    within the configured profile range."""
    # profile persists as the tying-ablation arm
    w = make_world("D", CFG, seed=5)
    r = w.rel_profile
    assert r.shape == (CFG.n_outer,)
    assert r.max() == pytest.approx(1.0)
    assert r.min() >= CFG.profile_range[0] - 1e-12


def test_beta_bounds_per_set():
    """beta_bounds and beta_min for cases B and D match the minima of the
    corresponding weight arrays."""
    wb = make_world("B", CFG, seed=4)
    b = wb.beta_bounds
    assert b["p"] == pytest.approx(wb.p_plus.min())
    assert b["m"] == pytest.approx(wb.q_minus.min())
    assert wb.beta_min == pytest.approx(max(b.values()))
    wd = make_world("D", CFG, seed=4)
    assert wd.beta_bounds["in"] == pytest.approx(
        wd.w_inner.min(axis=1).max())
    assert wd.beta_bounds["out"] == pytest.approx(wd.pi_outer.min())


def test_separability_positive_for_b():
    """On a gated world, the asymmetric risk family diverges from the tied
    family with positive separability."""
    # smoke, not the confirmatory margin: on a gated world the
    # asymmetric family must leave the tied family's hull at all
    world, _ = sample_world("B", CFG, seed=0)
    rows = o1_separability(world,
                           beta_grid=np.linspace(0.2, 1.0, 4),
                           rho_grid=np.linspace(0.0, 1.2, 4))
    assert rows
    assert max(r["tv_to_shared"] for r in rows) > 1e-3
    # attribution columns present and consistent
    for r in rows:
        assert r["tv_to_rho_split"] >= 0.0


def test_run_o3_b_asym_cells_and_kappa():
    """run_o3 on case B includes risk_asym rows carrying rho_p/rho_m and
    reports per-side kappa values in its metadata."""
    rows, meta = run_o3("B", WorldConfig(H=8), [0], n_stress=16,
                        kappa_range=(8.0, 60.0))
    kinds = {r["target"] for r in rows}
    assert "risk_asym" in kinds
    assert {"boltzmann", "worst"} <= kinds
    assert "kappa_p" in meta[0] and "kappa_m" in meta[0]
    asym = [r for r in rows if r["target"] == "risk_asym"]
    assert all("rho_p" in r and "rho_m" in r for r in asym)
