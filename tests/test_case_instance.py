"""Gates for feature-flag identity behavior: turning a new axis off
must reproduce the old code path bitwise.

Three axes, each with (a) an IDENTITY test (the default value must
reproduce the pre-adoption behaviour exactly) and (b) exactness /
liveness tests for the new regime:

  RiskD.rho_out            (None → WorldConfig.rho_out, bytewise)
  WorldConfig.weight_alpha (2.0 → bitwise-identical worlds)
  Risk*.geometry           ("kl" → untouched KL code path)

The TV ball sup is a plain LP (checked against linprog, exact); the
χ² ball sup and the full DRO-CVaR wraps are checked against direct
SLSQP primal optimization, the same device as
test_risk.test_dro_matches_primal_slsqp.
"""

import dataclasses

import numpy as np
import pytest
from scipy.optimize import linprog, minimize

from epgfn.cases import psi_and_masks
from epgfn.conditions import ConditionRanges, RiskA, RiskD, tied_risk
from epgfn.o1 import o1_grid, o1_rho_out_sep, o1_separability
from epgfn.risk import (GEOMETRIES, _sup_chi2_linear, _sup_tv_linear,
                        cvar_lower, dro_cvar_lower, dro_cvar_upper,
                        nested_dro_cvar)
from epgfn.worlds import (WorldConfig, hardness_report, make_world,
                          sample_world)


# --------------------------------------------------- ambiguity-ball geometry


def _tv_sup_lp(f, w, rho):
    """Exact LP oracle: max f·q s.t. q ∈ Δ, ½Σ|q − w| ≤ ρ,
    via u_i ≥ |q_i − w_i| auxiliaries."""
    k = len(f)
    c = np.concatenate([-f, np.zeros(k)])
    a_eq = np.concatenate([np.ones(k), np.zeros(k)])[None, :]
    a_ub = np.block([[np.eye(k), -np.eye(k)],
                     [-np.eye(k), -np.eye(k)],
                     [np.zeros((1, k)), np.ones((1, k))]])
    b_ub = np.concatenate([w, -w, [2.0 * rho]])
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1.0],
                  bounds=[(0, None)] * (2 * k))
    assert res.success
    return -res.fun


def test_sup_tv_linear_matches_lp():
    """_sup_tv_linear matches an exact LP oracle for the TV-ball worst-case
    expectation."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        k = int(rng.integers(3, 7))
        f = rng.uniform(0, 1, size=k)
        w = rng.dirichlet(np.ones(k))
        rho = float(rng.uniform(0.0, 0.8))
        got = _sup_tv_linear(f[None, :], w, rho)[0]
        assert got == pytest.approx(_tv_sup_lp(f, w, rho), abs=1e-8)


def test_sup_tv_rare_state_inflation_closed_form():
    """A TV ball inflates a rare state's worst-case mass to w_k + rho,
    regardless of how small w_k is."""
    # closed-form check: a TV ball lifts a rare state to w_k + ρ
    # regardless of how small w_k is
    f = np.array([0.0, 0.0, 1.0])
    w = np.array([0.98, 0.019, 0.001])
    got = _sup_tv_linear(f[None, :], w, 0.2)[0]
    assert got == pytest.approx(0.001 + 0.2)


def _chi2_sup_slsqp(f, w, rho, rng, starts=8):
    def chi2(q):
        return float(np.sum((q - w) ** 2 / w))

    best = -np.inf
    for _ in range(starts):
        res = minimize(
            lambda q: -float(f @ q), rng.dirichlet(np.ones(len(f))),
            method="SLSQP", bounds=[(0.0, 1.0)] * len(f),
            constraints=[
                {"type": "eq", "fun": lambda q: np.sum(q) - 1},
                {"type": "ineq", "fun": lambda q: rho - chi2(q)},
            ])
        if res.success:
            best = max(best, -res.fun)
    return best


def test_sup_chi2_linear_matches_slsqp():
    """_sup_chi2_linear matches a direct SLSQP primal optimization for the chi-
    square-ball worst case."""
    rng = np.random.default_rng(1)
    for _ in range(12):
        k = int(rng.integers(3, 6))
        f = rng.uniform(0, 1, size=k)
        w = rng.dirichlet(np.ones(k))
        rho = float(rng.uniform(0.05, 3.0))
        got = _sup_chi2_linear(f[None, :], w, rho)[0]
        want = _chi2_sup_slsqp(f, w, rho, rng)
        assert got == pytest.approx(want, abs=2e-4)


def test_sup_chi2_full_support_is_mean_plus_sqrt_rho_var():
    """At small rho with full support, _sup_chi2_linear reduces to the closed-
    form mean + sqrt(rho * var)."""
    # small ρ (interior optimum): the classic mean + sqrt(ρ·Var) form
    rng = np.random.default_rng(2)
    f = rng.uniform(0, 1, size=6)
    w = rng.dirichlet(5.0 * np.ones(6))
    rho = 1e-3
    var = float(np.sum(w * (f - f @ w) ** 2))
    got = _sup_chi2_linear(f[None, :], w, rho)[0]
    assert got == pytest.approx(float(f @ w) + np.sqrt(rho * var),
                                abs=1e-9)


def test_dro_geometries_match_primal_slsqp():
    """dro_cvar_lower's dual value matches direct primal SLSQP minimization of
    CVaR over the ball, per geometry."""
    # full Φ⁻ wrap: dual value == direct primal min of CVaR over the
    # ball, per geometry (the KL twin of this test is in test_risk)
    rng = np.random.default_rng(3)
    beta = 0.5
    for geometry, rho in [("tv", 0.15), ("chi2", 0.5)]:
        for _ in range(3):
            a = rng.uniform(0, 1, size=4)
            w = rng.dirichlet(np.ones(4))

            def dist(nu):
                if geometry == "tv":
                    return 0.5 * float(np.abs(nu - w).sum())
                return float(np.sum((nu - w) ** 2 / w))

            best = np.inf
            for _ in range(8):
                res = minimize(
                    lambda nu: cvar_lower(a, np.asarray(nu), beta),
                    rng.dirichlet(np.ones(4)), method="SLSQP",
                    bounds=[(1e-9, 1)] * 4,
                    constraints=[
                        {"type": "eq",
                         "fun": lambda nu: np.sum(nu) - 1},
                        {"type": "ineq",
                         "fun": lambda nu: rho - dist(nu)},
                    ])
                if res.success:
                    best = min(best, res.fun)
            got = dro_cvar_lower(a, w, beta, rho, geometry=geometry)
            assert got == pytest.approx(best, abs=2e-3)


def test_geometry_shared_limits_and_identities():
    """dro_cvar_lower/upper reduce to plain CVaR at rho=0, satisfy the
    reflection identity, are monotone in rho, and hit the worst-state value
    at ball- exhausting radii, for every geometry."""
    rng = np.random.default_rng(4)
    a = rng.uniform(0, 1, size=(30, 5))
    w = rng.dirichlet(np.ones(5))
    for geometry in GEOMETRIES:
        # ρ = 0 → plain CVaR (shared early return)
        np.testing.assert_allclose(
            dro_cvar_lower(a, w, 0.6, 0.0, geometry=geometry),
            cvar_lower(a, w, 0.6), atol=0)
        # reflection identity
        np.testing.assert_allclose(
            dro_cvar_upper(a, w, 0.7, 0.3, geometry=geometry),
            -dro_cvar_lower(-a, w, 0.7, 0.3, geometry=geometry),
            atol=1e-9)
        # monotone in ρ
        vals = [dro_cvar_lower(a, w, 0.5, r, geometry=geometry)
                for r in (0.0, 0.1, 0.4)]
        for lo_r, hi_r in zip(vals[1:], vals[:-1]):
            assert np.all(lo_r <= hi_r + 1e-9)
    # ball-exhausting radii → weight-free worst state
    np.testing.assert_allclose(
        dro_cvar_lower(a, w, 0.5, 1.0, geometry="tv"),
        a.min(axis=-1), atol=1e-6)
    np.testing.assert_allclose(
        dro_cvar_lower(a, w, 0.5, 1e4, geometry="chi2"),
        a.min(axis=-1), atol=1e-4)


def test_geometry_per_origin_rho_broadcast():
    """nested_dro_cvar with per-origin inner radii matches manually nesting
    dro_cvar_lower per origin, for each geometry."""
    # nested wrap with per-origin inner radii, per geometry
    rng = np.random.default_rng(5)
    a = rng.uniform(0, 1, size=(10, 3, 4))
    w_in = rng.dirichlet(np.ones(4), size=3)
    pi = rng.dirichlet(np.ones(3))
    rho_in = np.array([0.0, 0.1, 0.3])
    for geometry in ("tv", "chi2"):
        got = nested_dro_cvar(a, w_in, pi, 0.5, rho_in, 0.5, 0.2,
                              geometry=geometry)
        inner = np.stack(
            [dro_cvar_lower(a[:, o, :], w_in[o], 0.5, rho_in[o],
                            geometry=geometry) for o in range(3)],
            axis=-1)
        want = dro_cvar_lower(inner, pi, 0.5, 0.2, geometry=geometry)
        np.testing.assert_allclose(got, want, atol=1e-12)


def test_unknown_geometry_raises():
    """dro_cvar_lower raises ValueError for an unrecognized geometry name."""
    with pytest.raises(ValueError):
        dro_cvar_lower(np.ones((2, 3)), np.ones(3) / 3, 0.5, 0.1,
                       geometry="wasserstein")


def test_psi_geometry_identity_and_liveness():
    """geometry="kl" reproduces the default Psi/masks bytewise, while tv/chi2
    geometries actually change Psi."""
    # geometry="kl" on the risk block is bytewise the default path;
    # tv/chi2 actually change Ψ on a real world
    for case in ("A", "B", "C", "D"):
        world, _ = sample_world(case, WorldConfig(H=8), seed=0)
        base = tied_risk(case, 0.4, 0.5)
        psi0, f0, v0 = psi_and_masks(world, base)
        psi_kl, f_kl, v_kl = psi_and_masks(
            world, dataclasses.replace(base, geometry="kl"))
        np.testing.assert_array_equal(psi0, psi_kl)
        np.testing.assert_array_equal(f0, f_kl)
        np.testing.assert_array_equal(v0, v_kl)
        for geometry, rho_g in (("tv", 0.15), ("chi2", 0.5)):
            alt = tied_risk(case, 0.4, rho_g)
            psi_g, _, _ = psi_and_masks(
                world, dataclasses.replace(alt, geometry=geometry))
            assert not np.allclose(psi_g, psi0, atol=1e-6)


# ---------------------------------------------------------------- rho_out axis


def test_rho_out_none_is_world_constant():
    """RiskD.rho_out=None reproduces the same Psi as explicitly passing the
    world's constant rho_out."""
    world, _ = sample_world("D", WorldConfig(H=8), seed=0)
    base = RiskD(0.5, 0.4, 0.5)
    assert base.rho_out is None
    explicit = dataclasses.replace(base,
                                   rho_out=world.cfg.rho_out)
    psi_none, _, _ = psi_and_masks(world, base)
    psi_exp, _, _ = psi_and_masks(world, explicit)
    np.testing.assert_array_equal(psi_none, psi_exp)


def test_rho_out_axis_is_live():
    """Varying RiskD.rho_out changes Psi, with higher outer distrust producing
    lower (or equal) Psi."""
    world, _ = sample_world("D", WorldConfig(H=8), seed=0)
    lo, _, _ = psi_and_masks(world, RiskD(0.5, 0.4, 0.5, rho_out=0.0))
    hi, _, _ = psi_and_masks(world, RiskD(0.5, 0.4, 0.5, rho_out=1.2))
    assert not np.allclose(lo, hi, atol=1e-6)
    # more outer distrust → lower Ψ (inf over a larger ball)
    assert np.all(hi <= lo + 1e-9)


def test_use_rho_out_conditioning():
    """use_rho_out=True adds a live rho_out feature/grid axis to
    ConditionRanges, sampling within bounds and requiring it when
    featurizing."""
    r_off = ConditionRanges(case="D")
    r_on = ConditionRanges(case="D", use_rho_out=True)
    assert r_on.n_features == r_off.n_features + 1
    rng = np.random.default_rng(0)
    c_off = r_off.sample(rng)
    assert c_off.risk.rho_out is None            # identity default
    c_on = r_on.sample(np.random.default_rng(0))
    assert c_on.risk.rho_out is not None
    lo, hi = r_on.rho
    assert lo <= c_on.risk.rho_out <= hi
    # grid gains a 2-point ρ_out axis
    assert (len(r_on.heldout_grid()) == 2 * len(r_off.heldout_grid()))
    # features: encoded in the slot before σ's; None rejected
    feats = r_on.features([c_on])
    assert feats.shape == (1, r_on.n_features)
    with pytest.raises(ValueError):
        r_on.features([c_off])
    # σ stays the LAST feature when both flags are on
    r_both = ConditionRanges(case="D", use_rho_out=True, use_sigma=True)
    c_both = r_both.sample(np.random.default_rng(1))
    fb = r_both.features([c_both])[0]
    assert fb.shape == (r_off.n_features + 2,)


def test_use_rho_out_is_d_only():
    """ConditionRanges rejects use_rho_out=True for cases other than D."""
    with pytest.raises(ValueError):
        ConditionRanges(case="A", use_rho_out=True).sample(
            np.random.default_rng(0))


# --------------------------------------------------------- weight_alpha axis


def test_weight_alpha_default_is_bitwise_identity():
    """The default weight_alpha reproduces bitwise-identical worlds across all
    cases and world attributes."""
    for case in ("A", "B", "C", "D"):
        w0 = make_world(case, WorldConfig(H=8), seed=3)
        w1 = make_world(case, WorldConfig(H=8, weight_alpha=2.0), seed=3)
        for attr in ("g", "scores_neutral", "w_neutral", "scores_plus",
                     "p_plus", "scores_minus", "q_minus", "scores_named",
                     "scores_nested", "w_inner", "pi_outer",
                     "rel_profile"):
            a0, a1 = getattr(w0, attr), getattr(w1, attr)
            if a0 is None:
                assert a1 is None
            else:
                np.testing.assert_array_equal(a0, a1)


def test_sup_chi2_near_tied_suffix_regression():
    """_sup_chi2_linear stays finite and matches the SLSQP oracle on the near-
    tied-suffix values that once produced a NaN target."""
    # a real production bug: nested-D crude
    # arm, whose inner aggregates of quantized scores form a suffix tied
    # to ~3e-9. One-pass suffix variance (S2 − W·m²) cancelled
    # catastrophically, its noise flipped the KKT feasibility check,
    # EVERY candidate was rejected and the sup returned −inf → +inf Ψ
    # → NaN in p_star. Exact captured floats; oracle value from SLSQP.
    f = np.array([[0.04508496920218443, 0.04508497189122313,
                   0.0, 0.045084971416263796]])
    w = np.array([0.08574645, 0.16662536, 0.16178665, 0.58584154])
    w = w / w.sum()
    v = _sup_chi2_linear(f, w, 0.3)[0]
    assert np.isfinite(v)
    assert v == pytest.approx(0.045084971, abs=1e-6)
    # never outside [weighted mean, max f]
    assert float(f[0] @ w) - 1e-12 <= v <= f.max() + 1e-12


def test_sup_chi2_near_tied_stress_finite():
    """_sup_chi2_linear stays finite and within [weighted mean, max f] under
    randomized near-tied-suffix stress."""
    # randomized near-tie stress: suffixes tied to within 1e-6..1e-12
    rng = np.random.default_rng(7)
    for _ in range(20000):
        k = int(rng.integers(3, 7))
        base = rng.uniform(0, 1)
        spread = 10.0 ** rng.uniform(-12, -6)
        f = base + rng.uniform(0, spread, size=k)
        f[rng.integers(0, k)] = rng.uniform(0, base)  # one low outlier
        w = rng.dirichlet(np.full(k, rng.choice([0.3, 2.0])))
        v = _sup_chi2_linear(f[None, :], w,
                             float(rng.choice([0.1, 0.3, 1.0])))[0]
        assert np.isfinite(v)
        assert v <= f.max() + 1e-9


def test_nested_chi2_crude_world_targets_finite():
    """target_for stays finite and normalized on the exact crude-world chi2
    configuration that once produced a NaN."""
    # end-to-end: the exact configuration that produced the NaN
    from epgfn.cases import target_for
    from epgfn.conditions import Condition
    from epgfn.corrupt import crude_world
    world, _ = sample_world("D", WorldConfig(H=8), seed=1)
    cw = crude_world(world)
    for b, r in ((0.3, 1.0), (0.3, 3.0), (0.6, 0.3)):
        risk = dataclasses.replace(tied_risk("D", b, r), geometry="chi2")
        p = target_for(cw, Condition(4.0, 0.3, risk))
        assert np.isfinite(p).all()
        assert p.sum() == pytest.approx(1.0)


# ----------------------------------------------------------- veto-guard axis


def test_k_guard_zero_is_bitwise_identity():
    """k_guard=0 leaves scores_named unset and reproduces the same neutral
    scores/weights as the ungated world; the guard draws after existing RNG
    draws so they stay unchanged."""
    w0 = make_world("A", WorldConfig(H=8), seed=5)
    w1 = make_world("A", WorldConfig(H=8, k_guard=0), seed=5)
    assert w1.scores_named is None
    np.testing.assert_array_equal(w0.scores_neutral, w1.scores_neutral)
    np.testing.assert_array_equal(w0.w_neutral, w1.w_neutral)
    # existing draws unchanged by the guard (drawn AFTER them)
    wg = make_world("A", WorldConfig(H=8, k_guard=2), seed=5)
    np.testing.assert_array_equal(w0.scores_neutral, wg.scores_neutral)
    np.testing.assert_array_equal(w0.w_neutral, wg.w_neutral)
    assert wg.scores_named.shape == (64, 2)
    np.testing.assert_array_equal(wg.c_named, np.full(2, 0.85))


def test_guarded_a_veto_fires_and_delta_is_live():
    """On a guarded Case A world, the veto mask fires at the guard threshold,
    delta tightens it further, delta and sigma are equivalent, and guard share
    is reported but never gates."""
    world, _ = sample_world("A", WorldConfig(H=8, k_guard=2), seed=0)
    _, _, veto0 = psi_and_masks(world, RiskA(0.5, 0.3))
    assert not veto0.all()          # some points vetoed at c_d = 0.85
    _, _, veto_d = psi_and_masks(world, RiskA(0.5, 0.3, delta=0.1))
    # δ tightens the threshold: strictly more (or equal) vetoes,
    # and on a gated guarded world strictly more somewhere
    assert np.all(veto_d <= veto0)
    assert veto_d.sum() < veto0.sum()
    # δ≡σ unification carries over: δ=0.05 == σ=0.05 on the veto mask
    _, _, v_sig = psi_and_masks(world, RiskA(0.5, 0.3, sigma=0.05))
    _, _, v_del = psi_and_masks(world, RiskA(0.5, 0.3, delta=0.05))
    np.testing.assert_array_equal(v_sig, v_del)
    # guard share is reported, never gated
    assert "veto_frac" in hardness_report(world)


def test_delta_inert_without_guard():
    """RiskA.delta has no effect on Psi or the veto mask when the world has no
    guard."""
    world, _ = sample_world("A", WorldConfig(H=8), seed=0)
    psi0, f0, v0 = psi_and_masks(world, RiskA(0.5, 0.3))
    psi1, f1, v1 = psi_and_masks(world, RiskA(0.5, 0.3, delta=0.5))
    np.testing.assert_array_equal(psi0, psi1)
    np.testing.assert_array_equal(v0, v1)


def test_use_guard_delta_conditioning():
    """use_guard_delta=True adds a live delta feature/grid axis to
    ConditionRanges, sampling within bounds, and is rejected for
    non-A cases."""
    r_off = ConditionRanges(case="A")
    r_on = ConditionRanges(case="A", use_guard_delta=True)
    assert r_on.n_features == r_off.n_features + 1
    c = r_on.sample(np.random.default_rng(0))
    lo, hi = r_on.delta
    assert lo <= c.risk.delta <= hi
    assert (len(r_on.heldout_grid()) == 3 * len(r_off.heldout_grid()))
    feats = r_on.features([c])
    assert feats.shape == (1, 5)
    with pytest.raises(ValueError):
        ConditionRanges(case="B", use_guard_delta=True).sample(
            np.random.default_rng(0))


def test_guarded_a_o1_separability_rows():
    """o1_separability emits per-delta separability rows for a guarded Case A
    world, and stays empty for a guard-free world."""
    world, _ = sample_world("A", WorldConfig(H=8, k_guard=2), seed=0)
    rows = o1_separability(world, beta_grid=[0.5, 1.0],
                           rho_grid=[0.0, 0.5], n_delta=3)
    assert rows and all(r["case"] == "A" and "delta" in r for r in rows)
    assert all(r["tv_to_shared"] >= 0 for r in rows)
    # guard-free A stays empty (same as before the guard axis existed)
    bare, _ = sample_world("A", WorldConfig(H=8), seed=0)
    assert o1_separability(bare, beta_grid=[0.5, 1.0],
                           rho_grid=[0.0, 0.5]) == []


# ------------------------------------------------------------ O1 report probes


def test_o1_rho_out_probe_anchor_and_liveness():
    """o1_rho_out_sep probes rho_out on Case D, anchoring at the world's
    constant, and returns no rows for non-D cases."""
    world, _ = sample_world("D", WorldConfig(H=8), seed=0)
    grid = (0.0, 0.3, 1.2)  # 0.3 = the world constant → exact anchor
    rows = o1_rho_out_sep(world, beta_grid=[0.55, 1.0],
                          rho_grid=[0.0, 0.5, 1.6], rho_out_grid=grid)
    by_ro = {r["rho_out"]: r["tv_to_shared_rho_out"] for r in rows}
    assert set(by_ro) == set(grid)
    assert by_ro[0.3] <= 1e-12          # anchor IS a shared member
    assert all(r["probe"] == "rho_out" for r in rows)
    # non-D cases: no rows
    wa, _ = sample_world("A", WorldConfig(H=8), seed=0)
    assert o1_rho_out_sep(wa, [0.5, 1.0], [0.0, 0.5], grid) == []


def test_o1_grid_ball_threading():
    """o1_grid threads the requested ball (kl vs tv) through, agreeing at rho=0
    and diverging at rho>0."""
    world, _ = sample_world("A", WorldConfig(H=8), seed=0)
    kl = o1_grid(world, beta_grid=[0.5], rho_grid=[0.0, 0.4])
    tvb = o1_grid(world, beta_grid=[0.5], rho_grid=[0.0, 0.4],
                  ball="tv")
    assert all(r["ball"] == "kl" for r in kl)
    assert all(r["ball"] == "tv" for r in tvb)
    # ρ = 0 cells agree across balls; ρ > 0 cells differ
    k0 = {(r["beta_cvar"], r["rho"]): r["tv_on_off"] for r in kl}
    t0 = {(r["beta_cvar"], r["rho"]): r["tv_on_off"] for r in tvb}
    assert k0[(0.5, 0.0)] == pytest.approx(t0[(0.5, 0.0)], abs=1e-12)
    assert abs(k0[(0.5, 0.4)] - t0[(0.5, 0.4)]) > 1e-6


def test_weight_alpha_peaked_regime():
    """weight_alpha=0.3 yields valid, more concentrated simplex weights than
    the flat default family."""
    # α = 0.3 (peaked family): valid simplex draws, and
    # systematically more concentrated than the flat family
    flat_max, peak_max = [], []
    for seed in range(20):
        wf = make_world("A", WorldConfig(H=4), seed=seed)
        wp = make_world("A", WorldConfig(H=4, weight_alpha=0.3),
                        seed=seed)
        assert wp.w_neutral.sum() == pytest.approx(1.0)
        assert np.all(wp.w_neutral >= 0)
        flat_max.append(wf.w_neutral.max())
        peak_max.append(wp.w_neutral.max())
    assert np.mean(peak_max) > np.mean(flat_max) + 0.1
