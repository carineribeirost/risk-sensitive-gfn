"""The four structural cases and the reward pipeline,
with per-origin risk parameters.

Set-superscript notation (see risk.py docstring): the functional on S⁺/S⁰
is the lower-tail DRO-CVaR (conservative on what should be good); the
functional on S⁻ is the upper-tail DRO-CVaR (pessimistic on what should
be small). Case B combines them as Ψ = Φ(S⁺) − γ·Φ(S⁻) with the floor on
Φ(S⁺); each side takes its own (β, ρ) from the condition's RiskB block.

Case C's thresholded origin carries the veto margin δ: the veto fires at
a_d ≥ c_d − δ. Case D conditions its per-origin inner
radii ρ_o directly (aimed distrust), the outer tail level β_out (the
joint-satisfaction dial), and the outer radius ρ_out (distrust of the
model-implied π); RiskD.rho_out=None resolves to the world constant
cfg.rho_out, keeping the case bytewise unchanged when that axis is
unused. Every weight ball takes the condition's ambiguity geometry
("kl" default = the plain KL ball).

ε policy: R = ε on floored and vetoed points, and as a
positivity clamp on Case B's signed objective when w_g·g + w_s·Ψ ≤ 0.
"""

from __future__ import annotations

import numpy as np

from .conditions import Condition
from .risk import (dro_cvar_lower, dro_cvar_upper, flattened_dro_cvar,
                   nested_dro_cvar)
from .target import mc_floor, p_star, tv
from .worlds import World

EPS_REWARD = 1e-4


def _sl(arr, idx):
    """Slice a per-point array to a subset of X (None = full grid)."""
    return arr if idx is None else arr[idx]


def psi_and_masks(world: World, risk, idx=None):
    """Aggregation Ψ(x) plus constraint masks, all shape (N,), or
    (len(idx),) when `idx` restricts to a subset of points (used by the
    trainer to price only sampled points; the math is per-point, so
    subsetting is exact).

    `risk` is the condition's per-case block (RiskA/B/C/D). Returns
    (psi, floor_ok, veto_ok); masks are all-True where the case has no
    such constraint.
    """
    n = world.n_points if idx is None else len(idx)
    floor_ok = np.ones(n, dtype=bool)
    veto_ok = np.ones(n, dtype=bool)
    cfg = world.cfg
    # Score-robustness margin σ (box ambiguity, closed form by
    # monotonicity): good scores −σ,
    # bad (S⁻) scores +σ, veto threshold tightened by σ. σ = 0
    # skips the shift entirely: bitwise no-op, gated by
    # the golden suite.
    sig = getattr(risk, "sigma", 0.0)
    # Ambiguity-ball geometry, one per condition, applied to every
    # weight ball in the case's aggregation; "kl" = the plain KL ball
    geom = getattr(risk, "geometry", "kl")

    def _lo(arr):  # scores that should be good: worst case is lower
        return arr - sig if sig else arr

    if world.case == "A":
        psi = dro_cvar_lower(_lo(_sl(world.scores_neutral, idx)),
                             world.w_neutral, risk.beta, risk.rho,
                             geometry=geom)
        if world.scores_named is not None:
            # guarded A: C's veto machinery verbatim (δ≡σ incl.);
            # k_guard = 0 worlds have no scores_named → unaffected
            thr = world.c_named - getattr(risk, "delta", 0.0) - sig
            veto_ok = ~np.any(_sl(world.scores_named, idx) >= thr,
                              axis=-1)
    elif world.case == "B":
        phi_plus = dro_cvar_lower(_lo(_sl(world.scores_plus, idx)),
                                  world.p_plus, risk.beta_p, risk.rho_p,
                                  geometry=geom)
        minus = _sl(world.scores_minus, idx)
        phi_minus = dro_cvar_upper(minus + sig if sig else minus,
                                   world.q_minus, risk.beta_m, risk.rho_m,
                                   geometry=geom)
        psi = phi_plus - cfg.gamma * phi_minus
        if cfg.use_floor:
            floor_ok = phi_plus >= world.floor_value
    elif world.case == "C":
        psi = dro_cvar_lower(_lo(_sl(world.scores_plus, idx)),
                             world.p_plus, risk.beta, risk.rho,
                             geometry=geom)
        # δ≡σ unification: box-robust veto == margin shift (exact)
        thr = world.c_named - risk.delta - sig
        veto_ok = ~np.any(_sl(world.scores_named, idx) >= thr, axis=-1)
    elif world.case == "D":
        # aimed distrust: per-origin radii from the condition;
        # a scalar means shared radii (the tied member)
        rho_in = np.broadcast_to(np.asarray(risk.rho, dtype=float),
                                 (world.w_inner.shape[0],))
        # Conditioned outer radius; the identity default None
        # resolves to the world constant unchanged
        rho_out = getattr(risk, "rho_out", None)
        if rho_out is None:
            rho_out = cfg.rho_out
        psi = nested_dro_cvar(_lo(_sl(world.scores_nested, idx)),
                              world.w_inner, world.pi_outer,
                              risk.beta_in, rho_in,
                              risk.beta_out, rho_out, geometry=geom)
    else:
        raise ValueError(f"unknown case {world.case!r}")
    return psi, floor_ok, veto_ok


def log_reward_from(world: World, cond: Condition, psi: np.ndarray,
                    floor_ok: np.ndarray, veto_ok: np.ndarray,
                    idx=None) -> np.ndarray:
    """Assemble log R from a precomputed (Ψ, masks) triple."""
    base = cond.w_g * _sl(world.g, idx) + cond.w_s * psi
    r = np.maximum(base, EPS_REWARD)
    r = np.where(floor_ok & veto_ok, r, EPS_REWARD)
    return np.log(r)


def violation_depth(world: World, risk, idx=None) -> np.ndarray:
    """How badly each point misses its case's constraint, in [0, 1].

    0 for feasible points and for points exactly on the boundary; 1
    for the worst violation the constraint admits. Cases with no
    exclusion return all zeros. This is the quantity the graded
    penalty rides on: the flat epsilon on every excluded point makes
    the excluded region exactly flat, and a flat region is what lets
    an on-policy learner settle inside it.
    """
    n = world.n_points if idx is None else len(idx)
    d = np.zeros(n, dtype=float)
    sig = getattr(risk, "sigma", 0.0)

    def _veto_depth(thr):
        a = _sl(world.scores_named, idx)
        over = np.clip((a - thr) / np.maximum(1.0 - thr, 1e-12), 0.0, 1.0)
        return over.max(axis=-1)

    if world.case == "A":
        if world.scores_named is not None:
            d = _veto_depth(world.c_named - getattr(risk, "delta", 0.0) - sig)
    elif world.case == "B":
        if world.cfg.use_floor:
            geom = getattr(risk, "geometry", "kl")
            sp = _sl(world.scores_plus, idx)
            phi_plus = dro_cvar_lower(sp - sig if sig else sp,
                                      world.p_plus, risk.beta_p,
                                      risk.rho_p, geometry=geom)
            f = world.floor_value
            d = np.clip((f - phi_plus) / max(abs(f), 1e-12), 0.0, 1.0)
    elif world.case == "C":
        d = _veto_depth(world.c_named - risk.delta - sig)
    return d


def log_reward_graded(world: World, cond: Condition, kappa: float,
                      idx=None) -> np.ndarray:
    """log R with the flat epsilon replaced by a graded penalty.

    Excluded points get `epsilon * exp(-kappa * depth)` rather than a
    constant, so the excluded region carries a gradient back toward
    the boundary instead of being a plateau. At depth 0 this equals
    the plain reward exactly, so the two agree on the boundary and
    differ only in how steeply the interior falls away. kappa = 0
    reproduces the flat reward bitwise.
    """
    psi, floor_ok, veto_ok = psi_and_masks(world, cond.risk, idx)
    base = cond.w_g * _sl(world.g, idx) + cond.w_s * psi
    r = np.maximum(base, EPS_REWARD)
    ok = floor_ok & veto_ok
    if kappa == 0.0:
        return np.log(np.where(ok, r, EPS_REWARD))
    d = violation_depth(world, cond.risk, idx)
    penal = EPS_REWARD * np.exp(-float(kappa) * d)
    return np.log(np.where(ok, r, np.maximum(penal, 1e-300)))


def log_reward(world: World, cond: Condition, idx=None) -> np.ndarray:
    """log R(x), strictly finite; over X or a subset `idx`."""
    psi, floor_ok, veto_ok = psi_and_masks(world, cond.risk, idx)
    return log_reward_from(world, cond, psi, floor_ok, veto_ok, idx)


def target_for(world: World, cond: Condition) -> np.ndarray:
    """Exact p*_c by enumeration."""
    return p_star(log_reward(world, cond), cond.beta_t)


def mean_mc_floor(world: World, heldout, n_floor: int) -> float:
    """Mean MC floor over a held-out condition grid, RNG seeded from
    the world (shared by run_o2/run_mix/run_oracle_gap/probe_convergence)."""
    floor_rng = np.random.default_rng(world.seed)
    return float(np.mean([mc_floor(target_for(world, c), n_floor, floor_rng)
                          for c in heldout]))


def psi_worst_case(world: World, idx=None):
    """Pure worst-case pole: every active functional replaced by its
    weight-free worst state (the ρ→∞ / β→0 limits). Floor
    and veto semantics are kept, applied to the worst-case quantities
    (veto at the nominal thresholds, δ = 0)."""
    n = world.n_points if idx is None else len(idx)
    floor_ok = np.ones(n, dtype=bool)
    veto_ok = np.ones(n, dtype=bool)
    cfg = world.cfg
    if world.case == "A":
        psi = _sl(world.scores_neutral, idx).min(axis=-1)
        if world.scores_named is not None:  # guarded A: veto at nominal c_d
            veto_ok = ~np.any(_sl(world.scores_named, idx)
                              >= world.c_named, axis=-1)
    elif world.case == "B":
        lo = _sl(world.scores_plus, idx).min(axis=-1)
        hi = _sl(world.scores_minus, idx).max(axis=-1)
        psi = lo - cfg.gamma * hi
        if cfg.use_floor:
            floor_ok = lo >= world.floor_value
    elif world.case == "C":
        psi = _sl(world.scores_plus, idx).min(axis=-1)
        veto_ok = ~np.any(_sl(world.scores_named, idx) >= world.c_named,
                          axis=-1)
    elif world.case == "D":
        psi = _sl(world.scores_nested, idx).reshape(n, -1).min(axis=-1)
    else:
        raise ValueError(f"unknown case {world.case!r}")
    return psi, floor_ok, veto_ok


def target_worst(world: World, cond: Condition) -> np.ndarray:
    """Exact worst-case target: softmax(β_t·log R) with Ψ replaced by
    the weight-free worst state. Ignores cond's risk block."""
    psi, floor_ok, veto_ok = psi_worst_case(world)
    log_r = log_reward_from(world, cond, psi, floor_ok, veto_ok)
    return p_star(log_r, cond.beta_t)


def nested_vs_flat_tv(world: World, cond: Condition) -> float:
    """Case D, O1: TV between the nested target and the flattened
    single-level aggregation over all (o, k) states. The
    flat comparison uses the condition's β_in and the mean of its
    per-origin radii as the single (β, ρ) pair."""
    assert world.case == "D"
    r = cond.risk
    rho_flat = float(np.mean(np.asarray(r.rho, dtype=float)))
    psi_flat = flattened_dro_cvar(world.scores_nested, world.w_inner,
                                  world.pi_outer, r.beta_in, rho_flat,
                                  geometry=getattr(r, "geometry", "kl"))
    base = cond.w_g * world.g + cond.w_s * psi_flat
    log_r_flat = np.log(np.maximum(base, EPS_REWARD))
    return tv(target_for(world, cond), p_star(log_r_flat, cond.beta_t))
