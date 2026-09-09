"""ε-contamination oracle identities and the paired
sign-flip permutation test."""

import numpy as np

from epgfn.o3 import contamination_metrics, realized_utility, run_o3
from epgfn.stats import paired_sign_permutation
from epgfn.worlds import CASES, WorldConfig, make_world

CFG = WorldConfig(H=12)


def _uniform(world):
    return np.full(world.n_points, 1.0 / world.n_points)


def test_contam_eps0_equals_nominal_min_over_origins():
    """At eps=0, contamination_metrics' worst-case utility equals the plain
    nominal expected utility."""
    w = make_world("A", CFG, seed=1)
    p = _uniform(w)
    m = contamination_metrics(w, p, eps=0.0)
    nominal = float(p @ (w.scores_neutral @ w.w_neutral))
    assert np.isclose(m["contam_worst"], nominal)


def test_contam_eps1_is_worst_single_state():
    """At eps=1, contamination_metrics' worst case degenerates to the minimum
    per- state utility."""
    w = make_world("A", CFG, seed=1)
    p = _uniform(w)
    m = contamination_metrics(w, p, eps=1.0)
    per_state = p @ w.scores_neutral                      # (K,)
    assert np.isclose(m["contam_worst"], per_state.min())
    assert m["contam_state"] == int(np.argmin(per_state))


def test_contam_monotone_in_eps_and_all_cases_finite():
    """contam_worst is finite for all cases and non-increasing as eps grows."""
    for case in CASES:
        w = make_world(case, CFG, seed=4)
        p = _uniform(w)
        vals = [contamination_metrics(w, p, eps=e)["contam_worst"]
                for e in (0.0, 0.2, 0.5, 1.0)]
        assert all(np.isfinite(v) for v in vals)
        # U_ν is linear in ν: the vertex mix can only lower the min
        assert all(vals[i + 1] <= vals[i] + 1e-12
                   for i in range(len(vals) - 1))


def test_contam_matches_manual_stress_draw_case_b():
    """The contaminated ν, fed through the stress machinery by hand,
    must price identically (single-draw identity)."""
    w = make_world("B", CFG, seed=2)
    p = _uniform(w)
    m = contamination_metrics(w, p, eps=0.2)
    key, k = m["contam_origin"], m["contam_state"]
    nu_p, nu_q = w.p_plus.copy(), w.q_minus.copy()
    tgt = {"p": nu_p, "m": nu_q}[key]
    nu = 0.8 * tgt + 0.2 * np.eye(len(tgt))[k]
    stress = {"nu_p": nu[None, :] if key == "p" else nu_p[None, :],
              "nu_q": nu[None, :] if key == "m" else nu_q[None, :]}
    assert np.isclose(m["contam_worst"],
                      realized_utility(w, p, stress)[0])


def test_run_o3_contam_columns():
    """run_o3 includes contamination columns when contam_eps is set and omits
    them when it is None."""
    rows, _ = run_o3("A", WorldConfig(H=8), [0], n_stress=16,
                     contam_eps=0.2)
    assert all("contam_worst" in r for r in rows)
    rows_off, _ = run_o3("A", WorldConfig(H=8), [0], n_stress=16,
                         contam_eps=None)
    assert all("contam_worst" not in r for r in rows_off)


def test_paired_sign_permutation():
    """paired_sign_permutation detects a significant mean shift,
    reports non-significance for symmetric data, and is
    deterministic under a fixed seed."""
    strong = np.array([0.5, 0.6, 0.4, 0.55, 0.45, 0.5, 0.6, 0.5])
    r = paired_sign_permutation(strong, n_perm=2000, seed=0)
    assert r["p_value"] < 0.05 and r["mean"] > 0
    sym = np.array([0.3, -0.3, 0.2, -0.2, 0.1, -0.1])
    r2 = paired_sign_permutation(sym, n_perm=2000, seed=0)
    assert r2["p_value"] > 0.5
    assert np.isclose(r2["mean"], 0.0)
    # deterministic under the same seed
    r3 = paired_sign_permutation(strong, n_perm=2000, seed=0)
    assert r3 == r
