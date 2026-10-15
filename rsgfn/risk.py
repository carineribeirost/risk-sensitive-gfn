"""Risk functionals, exact and vectorized.

Notation resolution: functionals are labeled here by tail
side (Φ⁻ lower / Φ⁺ upper); elsewhere in this codebase they are labeled
by the *set* they act on (Φ⁺ on S⁺, Φ⁻ on S⁻, Φ⁰ on S⁰). We implement the
two tail-side primitives:

  dro_cvar_lower : inf over the KL ball of the lower-tail CVaR
                   (conservative view of a set you want to be good:
                    S⁺, S⁰, and both levels of Case D)
  dro_cvar_upper : sup over the KL ball of the upper-tail reverse-CVaR
                   (pessimistic view of a set you want to be small: S⁻)

Duals used (both from Sion's minimax + the entropic/log-sum-exp KL dual):

  Φ⁻(a;p,β,ρ) = max_τ [ τ − (1/β)·inf_{λ>0}( λρ
                                + λ log Σ_k p_k e^{(τ−a_k)⁺/λ} ) ]
  Φ⁺(a;q,β,ρ) = −Φ⁻(−a;q,β,ρ)          (exact reflection identity)

The objective is concave in τ and the inner dual convex in λ, so both 1-D
searches are golden-section searches (one new evaluation per iteration),
vectorized over leading (point) dimensions.

All functions accept `a` of shape (..., K) and weights of shape (K,) or
broadcastable to `a`, and return shape (...).
"""

from __future__ import annotations

import numpy as np

_GSS_ITERS = 40  # golden-section: invphi^39 ≈ 7e-9 interval shrink with
#                  ONE new evaluation per iteration (ternary needs two);
#                  value error is second-order at the optimum (both
#                  searches are on smooth concave/convex 1-D objectives)
_INVPHI = (np.sqrt(5.0) - 1.0) / 2.0   # 0.618...
_INVPHI2 = 1.0 - _INVPHI               # = invphi², golden identity


def _gss_min(f, lo, hi, iters: int):
    """Vectorized golden-section minimum of an elementwise-unimodal 1-D
    objective over arrays [lo, hi]. Reuses one interior evaluation per
    iteration (the golden identity makes the surviving interior point
    land exactly on the next probe). Returns f at the final midpoint."""
    h = hi - lo
    x1 = lo + _INVPHI2 * h
    x2 = lo + _INVPHI * h
    f1 = f(x1)
    f2 = f(x2)
    for _ in range(iters - 1):
        left = f1 < f2  # minimum is in [lo, x2]
        hi = np.where(left, x2, hi)
        lo = np.where(left, lo, x1)
        h = hi - lo
        x1n = lo + _INVPHI2 * h
        x2n = lo + _INVPHI * h
        fe = f(np.where(left, x1n, x2n))
        f1, f2 = (np.where(left, fe, f2), np.where(left, f1, fe))
        x1, x2 = x1n, x2n
    return f((lo + hi) / 2.0)


def cvar_lower(a: np.ndarray, w: np.ndarray, beta: float) -> np.ndarray:
    """Average of the worst β-fraction of a discrete weighted score set.

    Handles the boundary point mass exactly: the β-tail generally cuts
    through one state's weight, which contributes fractionally.
    """
    if not 0.0 < beta <= 1.0:
        raise ValueError(f"beta must be in (0, 1], got {beta}")
    a = np.asarray(a, dtype=float)
    w = np.broadcast_to(np.asarray(w, dtype=float), a.shape)
    order = np.argsort(a, axis=-1)
    a_s = np.take_along_axis(a, order, axis=-1)
    w_s = np.take_along_axis(w, order, axis=-1)
    cum_before = np.cumsum(w_s, axis=-1) - w_s
    take = np.clip(beta - cum_before, 0.0, w_s)
    return np.sum(a_s * take, axis=-1) / beta


def cvar_upper(a: np.ndarray, w: np.ndarray, beta: float) -> np.ndarray:
    """Average of the best β-fraction (reverse CVaR)."""
    return -cvar_lower(-np.asarray(a, dtype=float), w, beta)


GEOMETRIES = ("kl", "tv", "chi2")  # weight-ambiguity ball families

# Per-ball tied-grid radii: radii are geometry-specific and NOT
# comparable across balls. KL is the original grid; TV is in mass
# units; χ² in modified-χ² divergence units.
BALL_RHO_GRIDS = {"kl": (0.0, 0.2, 0.5, 1.2),
                  "tv": (0.0, 0.05, 0.15, 0.35),
                  "chi2": (0.0, 0.3, 1.0, 3.0)}


def _sup_kl_linear(f: np.ndarray, logw: np.ndarray, rho) -> np.ndarray:
    """sup_{KL(ν‖w) ≤ ρ} Σ_k ν_k f_k via the entropic dual
    inf_{λ>0} λρ + λ log Σ_k w_k e^{f_k/λ}, golden-section on log λ.

    f: (..., K), logw broadcastable to f; rho scalar or broadcastable to
    the leading dims (per-origin radii). Returns (...).
    """
    fmax = np.max(f, axis=-1)

    def dual(loglam: np.ndarray) -> np.ndarray:
        lam = np.exp(loglam)
        # λ·logΣ w e^{f/λ} = fmax + λ·logΣ e^{logw + (f−fmax)/λ};
        # exponents ≤ 0, so this is overflow-safe for any λ
        z = logw + (f - fmax[..., None]) / lam[..., None]
        return lam * rho + fmax + lam * np.log(np.sum(np.exp(z), axis=-1))

    lo = np.full(fmax.shape, -30.0)
    hi = np.full(fmax.shape, 30.0)
    return _gss_min(dual, lo, hi, _GSS_ITERS)


def _sup_tv_linear(f: np.ndarray, w: np.ndarray, rho) -> np.ndarray:
    """sup_{TV(q‖w) ≤ ρ, q ∈ Δ} Σ_k q_k f_k, exact greedy closed form:
    a linear objective moves
    mass δ = min(ρ, 1 − w_argmax) onto the argmax state, taken from the
    lowest-f states first, with the boundary state fractional (same
    point-mass-splitting device as `cvar_lower`). Unlike the KL ball,
    the TV ball can inflate a near-zero-weight state to w_k + ρ: the
    rare-tail-inflation direction KL under-covers.

    f: (..., K); w broadcastable to f; rho scalar or broadcastable to
    the leading dims. Returns (...).
    """
    order = np.argsort(f, axis=-1)
    f_s = np.take_along_axis(f, order, axis=-1)
    w_s = np.take_along_axis(np.broadcast_to(w, f.shape), order, axis=-1)
    cap = w_s.copy()
    cap[..., -1] = 0.0  # the receiving (argmax) state gives up nothing
    cum_before = np.cumsum(cap, axis=-1) - cap
    delta = np.broadcast_to(np.asarray(rho, dtype=float),
                            f.shape[:-1])[..., None]
    take = np.clip(delta - cum_before, 0.0, cap)
    moved = np.sum(take, axis=-1)
    return (np.sum(w_s * f_s, axis=-1) - np.sum(take * f_s, axis=-1)
            + moved * f_s[..., -1])


def _sup_chi2_linear(f: np.ndarray, w: np.ndarray, rho) -> np.ndarray:
    """sup_{χ²(q‖w) ≤ ρ, q ∈ Δ} Σ_k q_k f_k with the modified-χ²
    divergence Σ_k (q_k − w_k)²/w_k, exact KKT active-set enumeration.
    Sorting f ascending, the optimal support is a suffix A; on a
    fixed suffix the stationary point gives
        sup_A = m_A + s_A·sqrt(ρ − (1 − W_A)/W_A),
    W_A = Σ_A w, m_A the A-weighted mean, s_A² = Σ_A w (f − m_A)²
    (the full-support case W = 1 is the classic mean + sqrt(ρ·Var)).
    The objective is linear and the ball convex, so any candidate whose
    implied q is feasible (threshold t = m − 2λ/W between the last
    excluded and first included score) is the global optimum; we take
    the max over valid candidates. s_A = 0 suffixes (all-tied top
    block, incl. the single-vertex case) are valid iff the ball
    reaches them: ρ ≥ (1 − W_A)/W_A.
    """
    order = np.argsort(f, axis=-1)
    f_s = np.take_along_axis(f, order, axis=-1)
    w_s = np.take_along_axis(np.broadcast_to(w, f.shape), order, axis=-1)
    # suffix cumulants via reversed cumsums: index j ↔ A = {j, .., K−1}
    rev = (slice(None),) * (f.ndim - 1) + (slice(None, None, -1),)
    W = np.cumsum(w_s[rev], axis=-1)[rev]
    S1 = np.cumsum((w_s * f_s)[rev], axis=-1)[rev]
    m = S1 / W
    # suffix variance mass in TWO-PASS form: the one-pass S2 − W·m²
    # cancels catastrophically (absolute noise ~ eps·S2), which on a
    # nearly-tied suffix (spread ~1e-9, e.g. inner aggregates of
    # crude-quantized scores in nested D) perturbs the KKT threshold
    # t past its feasibility tolerance and rejects EVERY candidate
    # (the 2026-07-11 crude-arm NaN). O(K²) per point; K is small.
    k = f.shape[-1]
    diff = f_s[..., None, :] - m[..., :, None]          # (..., j, i)
    mask = np.triu(np.ones((k, k), dtype=bool))         # i ≥ j
    s2 = np.sum(np.where(mask, w_s[..., None, :] * diff * diff, 0.0),
                axis=-1)
    rho_b = np.broadcast_to(np.asarray(rho, dtype=float),
                            f.shape[:-1])[..., None]
    d = rho_b - (1.0 - W) / W
    tied = s2 <= 1e-30
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.sqrt(s2)
        lam = s / (2.0 * np.sqrt(np.maximum(d, 0.0)))
        t = m - 2.0 * lam / W          # support threshold η − 2λ
    val = m + s * np.sqrt(np.maximum(d, 0.0))
    # feasibility of the implied q: first included score ≥ t, last
    # excluded score ≤ t (no excluded state wants back in)
    ok_in = f_s >= t - 1e-12
    ok_out = np.ones_like(ok_in)
    ok_out[..., 1:] = f_s[..., :-1] <= t[..., 1:] + 1e-12
    valid = (d >= 0.0) & np.where(tied, True, ok_in & ok_out)
    best = np.max(np.where(valid, val, -np.inf), axis=-1)
    # completeness guard: exact arithmetic always yields one valid
    # KKT candidate, but float noise can in principle reject all of
    # them. Fall back to the tightest SAFE upper bound on the sup:
    # the best sign-relaxed candidate capped at max f (both are true
    # upper bounds). Overestimating the sup only makes the inf-side
    # wrap more conservative; it can never emit ±inf/NaN.
    if not np.isfinite(best).all():
        relaxed = np.minimum(
            np.max(np.where(d >= 0.0, val, -np.inf), axis=-1),
            f_s[..., -1])
        best = np.where(np.isfinite(best), best, relaxed)
    return best


def _sup_ball_linear(f: np.ndarray, w_arr: np.ndarray, rho,
                     geometry: str) -> np.ndarray:
    """Dispatch sup_{q ∈ B_ρ(w)} Σ_k q_k f_k over the ball geometry.
    Radii are geometry-specific (a KL, TV and χ² ρ are not comparable;
    see the per-ball grids in BALL_RHO_GRIDS)."""
    if geometry == "kl":
        return _sup_kl_linear(f, np.log(w_arr), rho)
    if geometry == "tv":
        return _sup_tv_linear(f, w_arr, rho)
    if geometry == "chi2":
        return _sup_chi2_linear(f, w_arr, rho)
    raise ValueError(f"unknown ball geometry {geometry!r}")


def dro_cvar_lower(a: np.ndarray, w: np.ndarray, beta: float, rho,
                   geometry: str = "kl") -> np.ndarray:
    """Φ⁻: DRO wrap of the lower-tail CVaR, exact dual form.

    `rho` may be a scalar or an array broadcastable to a.shape[:-1]
    (per-origin radii along leading dims). `geometry`
    picks the ambiguity ball: "kl" (default, entropic dual),
    "tv" or "chi2" (exact closed-form inner sup). The Sion swap
    behind the τ-search holds for any convex compact ball, so the
    outer golden-section is shared."""
    rho = np.asarray(rho, dtype=float)
    if np.any(rho < 0.0):
        raise ValueError(f"rho must be >= 0, got {rho}")
    a = np.asarray(a, dtype=float)
    if np.all(rho == 0.0):
        return cvar_lower(a, w, beta)
    w_arr = np.broadcast_to(np.asarray(w, dtype=float), a.shape)
    if geometry == "kl":
        logw = np.log(w_arr)

        def neg_objective(tau: np.ndarray) -> np.ndarray:
            f = np.maximum(tau[..., None] - a, 0.0)
            return _sup_kl_linear(f, logw, rho) / beta - tau
    else:

        def neg_objective(tau: np.ndarray) -> np.ndarray:
            f = np.maximum(tau[..., None] - a, 0.0)
            return _sup_ball_linear(f, w_arr, rho, geometry) / beta - tau

    lo = np.min(a, axis=-1)
    hi = np.max(a, axis=-1)
    return -_gss_min(neg_objective, lo, hi, _GSS_ITERS)


def dro_cvar_upper(a: np.ndarray, w: np.ndarray, beta: float, rho,
                   geometry: str = "kl") -> np.ndarray:
    """Φ⁺: DRO wrap of the upper-tail reverse-CVaR, via reflection."""
    return -dro_cvar_lower(-np.asarray(a, dtype=float), w, beta, rho,
                           geometry)


def nested_dro_cvar(a: np.ndarray, w_inner: np.ndarray, pi: np.ndarray,
                    beta_in: float, rho_in,
                    beta_out: float, rho_out: float,
                    geometry: str = "kl") -> np.ndarray:
    """Φ_nested: inner lower-tail DRO-CVaR over k within each o,
    outer lower-tail DRO-CVaR over o weighted by π.

    a: (..., O, K_in); w_inner: (O, K_in) rows summing to 1; pi: (O,).
    rho_in may be an (O,) array of per-origin radii.
    One `geometry` per condition applies to BOTH levels.
    """
    inner = dro_cvar_lower(a, w_inner, beta_in, rho_in,
                           geometry=geometry)  # (..., O)
    return dro_cvar_lower(inner, pi, beta_out, rho_out, geometry=geometry)


def flattened_dro_cvar(a: np.ndarray, w_inner: np.ndarray, pi: np.ndarray,
                       beta: float, rho: float,
                       geometry: str = "kl") -> np.ndarray:
    """Case D comparison target: single-level aggregation over all (o, k)
    states with product weights π_o · w_inner[o, k] (sums to 1)."""
    a = np.asarray(a, dtype=float)
    flat_w = (np.asarray(pi, dtype=float)[:, None]
              * np.asarray(w_inner, dtype=float))
    return dro_cvar_lower(a.reshape(*a.shape[:-2], -1),
                          flat_w.reshape(-1), beta, rho,
                          geometry=geometry)


def winsorize_scores(scores: np.ndarray, k: int = 1) -> np.ndarray:
    """Paper-2 crash-regime arm: per-point winsorization
    of a (N, K) score block to the [k-th lowest, k-th highest]
    coordinate values: clips unbounded per-origin corruption while
    preserving the weight–origin pairing (unlike trimming/dropping).
    Requires K > 2k. Applied BEFORE any aggregation Φ."""
    scores = np.asarray(scores)
    if scores.shape[-1] <= 2 * k:
        raise ValueError(f"need K > 2k (K={scores.shape[-1]}, k={k})")
    srt = np.sort(scores, axis=-1)
    lo = srt[..., k:k + 1]
    hi = srt[..., -k - 1:-k if k > 0 else None]
    return np.clip(scores, lo, hi)
