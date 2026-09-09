"""Oracle tests for the risk functionals, taken from the limits the
spec itself states (§2) plus a primal cross-check by direct constrained
optimization."""

import numpy as np
import pytest
from scipy.optimize import minimize

from epgfn.risk import (cvar_lower, cvar_upper, dro_cvar_lower,
                        dro_cvar_upper, flattened_dro_cvar,
                        nested_dro_cvar)

A = np.array([0.1, 0.5, 0.9])
W = np.array([0.2, 0.3, 0.5])


def test_cvar_lower_beta_one_is_mean():
    """cvar_lower at beta=1 reduces to the plain weighted mean."""
    assert cvar_lower(A, W, 1.0) == pytest.approx(float(A @ W))


def test_cvar_lower_small_beta_is_min():
    """cvar_lower at or below the worst state's weight equals that worst
    state's value alone."""
    # β at or below the worst state's weight → that state alone (§8.4)
    assert cvar_lower(A, W, 0.2) == pytest.approx(0.1)
    assert cvar_lower(A, W, 0.05) == pytest.approx(0.1)


def test_cvar_lower_splits_boundary_point_mass():
    """cvar_lower at a boundary beta splits the mass of the state straddling
    the tail cutoff."""
    # β = 0.35: all of state 1 (w=0.2) + 0.15 of state 2
    want = (0.2 * 0.1 + 0.15 * 0.5) / 0.35
    assert cvar_lower(A, W, 0.35) == pytest.approx(want)


def test_cvar_upper_limits():
    """cvar_upper reduces to the mean at beta=1 and to the best-state-weighted
    tail for smaller beta."""
    assert cvar_upper(A, W, 1.0) == pytest.approx(float(A @ W))
    assert cvar_upper(A, W, 0.5) == pytest.approx(0.9)  # best state w=0.5
    want = (0.5 * 0.9 + 0.1 * 0.5) / 0.6
    assert cvar_upper(A, W, 0.6) == pytest.approx(want)


def test_reflection_identity():
    """dro_cvar_upper on a and dro_cvar_lower on -a are negatives of each other
    for the same (beta, rho)."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, size=(50, 5))
    w = rng.dirichlet(np.ones(5))
    for beta, rho in [(0.3, 0.0), (0.7, 0.4), (1.0, 1.0)]:
        up = dro_cvar_upper(a, w, beta, rho)
        lo = dro_cvar_lower(-a, w, beta, rho)
        np.testing.assert_allclose(up, -lo, atol=1e-9)


def test_dro_rho_zero_is_plain_cvar():
    """dro_cvar_lower at rho=0 equals plain cvar_lower, and stays close to it
    at a tiny rho by continuity."""
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 1, size=(40, 6))
    w = rng.dirichlet(np.ones(6))
    for beta in (0.25, 0.6, 1.0):
        np.testing.assert_allclose(dro_cvar_lower(a, w, beta, 0.0),
                                   cvar_lower(a, w, beta), atol=0)
        # continuity: tiny rho ≈ plain CVaR
        np.testing.assert_allclose(dro_cvar_lower(a, w, beta, 1e-9),
                                   cvar_lower(a, w, beta), atol=1e-4)


def test_dro_rho_large_is_worst_state():
    """dro_cvar_lower/upper at a very large rho collapse to the per-row
    worst/best state."""
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 1, size=(40, 5))
    w = rng.dirichlet(np.ones(5))
    np.testing.assert_allclose(dro_cvar_lower(a, w, 0.5, 50.0),
                               a.min(axis=-1), atol=1e-6)
    np.testing.assert_allclose(dro_cvar_upper(a, w, 0.5, 50.0),
                               a.max(axis=-1), atol=1e-6)


def test_dro_monotone_in_rho_and_beta():
    """dro_cvar_lower is monotone non-increasing in rho, and cvar_lower is
    monotone non-decreasing in beta."""
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 1, size=(30, 5))
    w = rng.dirichlet(np.ones(5))
    rhos = [0.0, 0.1, 0.5, 2.0]
    vals = [dro_cvar_lower(a, w, 0.5, r) for r in rhos]
    for lo_r, hi_r in zip(vals[1:], vals[:-1]):
        assert np.all(lo_r <= hi_r + 1e-9)   # more robust → lower
    betas = [0.25, 0.5, 1.0]
    vals_b = [cvar_lower(a, w, b) for b in betas]
    for small_b, big_b in zip(vals_b[:-1], vals_b[1:]):
        assert np.all(small_b <= big_b + 1e-12)  # deeper tail → lower


def test_dro_beta_one_matches_entropic_mean_dual():
    """At beta=1, dro_cvar_lower matches the closed-form entropic-mean dual
    solved by scalar minimization."""
    # β = 1: Φ⁻ = inf over the KL ball of the mean, standard dual
    rng = np.random.default_rng(4)
    a = rng.uniform(0, 1, size=7)
    w = rng.dirichlet(np.ones(7))
    rho = 0.3

    def dual(loglam):
        lam = np.exp(loglam)
        # shifted for stability: exponents ≤ 0 for every λ
        return (lam * rho - a.min()
                + lam * np.log(np.sum(w * np.exp(-(a - a.min()) / lam))))

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(dual, bounds=(-20, 20), method="bounded")
    want = -res.fun
    got = dro_cvar_lower(a, w, 1.0, rho)
    assert got == pytest.approx(want, abs=1e-6)


def test_dro_matches_primal_slsqp():
    """dro_cvar_lower's dual value matches a direct primal SLSQP optimization
    over the KL ball, confirming strong duality."""
    # strong duality: our dual value == direct primal optimization over ν
    rng = np.random.default_rng(5)
    for trial in range(4):
        a = rng.uniform(0, 1, size=4)
        w = rng.dirichlet(np.ones(4))
        beta, rho = 0.5, 0.25

        def primal(nu):
            return cvar_lower(a, np.asarray(nu), beta)

        def kl(nu):
            nu = np.asarray(nu)
            return float(np.sum(nu * np.log(nu / w)))

        best = np.inf
        for _ in range(8):
            x0 = rng.dirichlet(np.ones(4))
            res = minimize(
                primal, x0, method="SLSQP",
                bounds=[(1e-9, 1)] * 4,
                constraints=[
                    {"type": "eq", "fun": lambda nu: np.sum(nu) - 1},
                    {"type": "ineq", "fun": lambda nu: rho - kl(nu)},
                ])
            if res.success:
                best = min(best, res.fun)
        got = dro_cvar_lower(a, w, beta, rho)
        assert got == pytest.approx(best, abs=2e-3)


def test_nested_means_collapse_to_flat_mean():
    """nested_dro_cvar with all (beta=1, rho=0) collapses to the flattened mean
    via the tower property."""
    # tower property: all-(β=1, ρ=0) nesting equals the flattened mean
    rng = np.random.default_rng(6)
    a = rng.uniform(0, 1, size=(20, 3, 4))
    w_in = rng.dirichlet(np.ones(4), size=3)
    pi = rng.dirichlet(np.ones(3))
    nested = nested_dro_cvar(a, w_in, pi, 1.0, 0.0, 1.0, 0.0)
    flat = flattened_dro_cvar(a, w_in, pi, 1.0, 0.0)
    np.testing.assert_allclose(nested, flat, atol=1e-12)


def test_nested_differs_from_flat_when_risk_on():
    """nested_dro_cvar diverges from flattened_dro_cvar once risk aversion is
    turned on at either level."""
    rng = np.random.default_rng(7)
    a = rng.uniform(0, 1, size=(20, 3, 4))
    w_in = rng.dirichlet(np.ones(4), size=3)
    pi = rng.dirichlet(np.ones(3))
    nested = nested_dro_cvar(a, w_in, pi, 0.4, 0.2, 0.5, 0.3)
    flat = flattened_dro_cvar(a, w_in, pi, 0.4, 0.2)
    assert not np.allclose(nested, flat, atol=1e-3)


def test_gss_matches_dense_grid_optimum():
    """dro_cvar_lower's golden-section-search dual matches a brute-force dense-
    grid optimization of the same objective."""
    # the golden-section duals must match a brute-force dense-grid
    # optimization of the same objective to fine tolerance
    rng = np.random.default_rng(9)
    a = rng.uniform(0, 1, size=(5, 6))
    w = rng.dirichlet(np.ones(6))
    beta, rho = 0.4, 0.35
    got = dro_cvar_lower(a, w, beta, rho)
    logw = np.log(w)
    taus = np.linspace(a.min() - 0.01, a.max() + 0.01, 4001)
    lams = np.exp(np.linspace(-30, 30, 4001))
    for i in range(a.shape[0]):
        # inner sup via dense λ grid, outer max via dense τ grid
        f = np.maximum(taus[:, None] - a[i][None, :], 0.0)  # (T, K)
        z = logw[None, None, :] + f[None, :, :] / lams[:, None, None]
        dual = (lams[:, None] * rho
                + lams[:, None] * np.log(np.sum(np.exp(z), axis=-1)))
        sup = dual.min(axis=0)                              # (T,)
        brute = np.max(taus - sup / beta)
        assert got[i] == pytest.approx(brute, abs=2e-4)


def test_beta_validation():
    """cvar_lower and dro_cvar_lower raise ValueError for an out-of-range beta
    or a negative rho."""
    with pytest.raises(ValueError):
        cvar_lower(A, W, 0.0)
    with pytest.raises(ValueError):
        dro_cvar_lower(A, W, 0.5, -0.1)
