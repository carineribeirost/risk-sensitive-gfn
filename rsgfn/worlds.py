"""World family: sampled score geometries on the H×H grid,
with the faithful-hardness acceptance gate.

A *world* is one draw of: smooth per-state score fields a(x) mapped into
[0,1], nominal weight vectors (Dirichlet), the auxiliary objective g(x),
the case's fixed parameters, and (Case D) the reliability profile r
over the inner origins. Everything is invented (no
external data) and deterministic given the seed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

import numpy as np

CASES = ("A", "B", "C", "D")


def grid_coords(H: int, d: int = 2) -> np.ndarray:
    """All N = H^d points in canonical (mixed-radix, x₀-major) order,
    shape (N, d); at d=2 identical to the former meshgrid(ij) order."""
    return np.stack(np.unravel_index(np.arange(H ** d), (H,) * d), axis=1)


def smooth_field(rng: np.random.Generator, H: int, n_features: int = 8,
                 freq_scale: float = 2.0, d: int = 2) -> np.ndarray:
    """One random low-frequency field on the grid, min-max mapped to [0,1]."""
    u = grid_coords(H, d) / H  # (N, d) in [0,1)^d
    omega = rng.normal(0.0, freq_scale, size=(n_features, d))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    amp = rng.normal(0.0, 1.0, size=n_features)
    f = np.cos(2.0 * np.pi * u @ omega.T + phase) @ amp
    lo, hi = f.min(), f.max()
    return (f - lo) / (hi - lo)


def sequence_field(rng: np.random.Generator, H: int, d: int,
                   n_pairs: int | None = None) -> np.ndarray:
    """Sequence-native score on alphabet^d: position-weight
    matrix + sparse pairwise epistasis, min-max mapped to [0,1].
    raw(x) = Σ_t PWM[t, x_t] + Σ_{(i,j)∈P} E[i,j][x_i, x_j];
    PWM entries N(0,1); |P| = d pairs w/o replacement; E entries
    N(0, 0.5). The sequence analog of `smooth_field`: local structure
    plus limited interactions."""
    coords = grid_coords(H, d)  # (N, d)
    pwm = rng.normal(0.0, 1.0, size=(d, H))
    pairs = list(itertools.combinations(range(d), 2))
    k = min(d if n_pairs is None else n_pairs, len(pairs))
    sel = rng.choice(len(pairs), size=k, replace=False)
    eps_e = rng.normal(0.0, 0.5, size=(k, H, H))
    f = pwm[np.arange(d)[None, :], coords].sum(axis=1)
    for m, si in enumerate(sel):
        i, j = pairs[si]
        f = f + eps_e[m, coords[:, i], coords[:, j]]
    lo, hi = f.min(), f.max()
    return (f - lo) / (hi - lo)


def _score_field(rng, cfg: "WorldConfig",
                 constraint: bool = False) -> np.ndarray:
    """One score field per the world's geometry, with the sparsity
    exponent applied after min-max (a ← a^s; s=1 skips the op, so
    default worlds are bitwise untouched). s>1 thins every upper tail
    monotonically: the satisfying-set sparsity knob, an
    OBJECTIVE-field transform. Constraint fields (C's named vetoes,
    A's guard) pass constraint=True and are exempt: their
    thresholds are absolute (c_d, frozen), so sparsifying them
    would silently disable the veto at s>1. The exponent is post-draw,
    so the RNG stream is identical either way."""
    if cfg.geometry == "sequence":
        f = sequence_field(rng, cfg.H, cfg.d)
    elif cfg.geometry == "grid":
        f = smooth_field(rng, cfg.H, d=cfg.d)
    else:
        raise ValueError(f"unknown geometry {cfg.geometry!r}")
    if constraint or cfg.sparsity == 1.0:
        return f
    return f ** cfg.sparsity


def _fields_cfg(rng, cfg, k, constraint: bool = False) -> np.ndarray:
    return np.stack([_score_field(rng, cfg, constraint)
                     for _ in range(k)], axis=-1)


def _weights(rng, k, alpha: float = 2.0) -> np.ndarray:
    """Nominal weight draw. `alpha` is the scalar Dirichlet
    concentration: 2.0 = the flat family; a peaked family (e.g. 0.3,
    one dominant coordinate plus a low-mass tail) is the regime where
    CVaR at small β selects exactly the states whose weight estimate
    is least reliable."""
    return rng.dirichlet(alpha * np.ones(k))


@dataclass
class WorldConfig:
    """Fixed parameters of one world family: grid geometry, set sizes
    per case, case-specific constants, and the hardness-gate bounds.

    Passed to `make_world`/`sample_world` to draw a `World`; see the
    inline comments on each field for its per-case meaning and the
    identity value that keeps a given feature bitwise inert.
    """

    H: int = 32
    d: int = 2                  # coordinates per point; |X| = H^d
    geometry: str = "grid"      # "grid" (cosine fields) | "sequence"
    #                             (PWM + sparse epistasis)
    sparsity: float = 1.0       # satisfying-set sparsity knob:
    #                             every OBJECTIVE field and g ← field^s
    #                             after min-max; constraint fields (C
    #                             vetoes, A's guard) are exempt: their
    #                             thresholds are absolute;
    #                             1.0 = bitwise no-op
    # set sizes per case; only the case-relevant ones are used
    k_neutral: int = 6          # A: |S⁰|
    k_guard: int = 0            # guard states on template A: a
    #                             C-style veto attached to the single
    #                             desire group ("A with a guard").
    #                             0 = identity (no draws added, A
    #                             bitwise unchanged)
    k_plus: int = 5             # B, C: |S⁺|
    k_minus: int = 4            # B: |S⁻|
    k_named: int = 3            # C: |D|
    n_outer: int = 4            # D: |O|
    k_inner: int = 4            # D: inner states per o
    weight_alpha: float = 2.0   # Dirichlet concentration for ALL
    #                             nominal weight draws (w, p, q,
    #                             w_inner rows, π). 2.0 = flat family;
    #                             a peaked family uses e.g. 0.3
    # fixed case parameters (invented; swept separately in O1)
    gamma: float = 1.0          # B: trade-off on Φ(S⁻)
    # B: floor level f on Φ(S⁺). None → calibrated per world as the
    # floor_quantile of Φ(S⁺) at the fixed reference condition
    # floor_ref = (β_cvar, ρ); a universal constant fails because the
    # scale of Φ(S⁺) is world- and condition-dependent.
    floor: float | None = None
    floor_quantile: float = 0.35
    floor_ref: tuple = (0.5, 0.8)
    use_floor: bool = True
    c_named: float = 0.85       # C: veto threshold c_d (shared default)
    beta_out: float = 0.5       # D: reference outer tail level: β_out
    #                             is a conditioned axis; this value
    #                             defines the tied member
    #                             (see conditions.tied_risk) only
    rho_out: float = 0.3        # D: fixed outer KL radius
    profile_range: tuple = (0.05, 1.0)  # D: reliability profile r:
    #                             not part of the reward;
    #                             kept per world for the profile×scale
    #                             tying-ablation arm
    g_iid: bool = False         # g(x): smooth indep. field (default) or iid
    # hardness gate
    hardness_margin: float = 0.05   # min TV(risk-on, risk-off) at probe
    veto_frac_range: tuple = (0.02, 0.5)    # C: acceptable veto share of X
    floor_frac_range: tuple = (0.05, 0.7)   # B: acceptable floor-fail share
    delta_probe: float = 0.1    # C: veto-margin endpoint for the gate
    max_attempts: int = 200


@dataclass
class World:
    """One drawn world: score fields, nominal weights, and (for B) the
    resolved floor value, for a single case.

    Constructed by `make_world`; only the fields relevant to `case`
    are populated, the rest stay None.
    """

    case: str
    cfg: WorldConfig
    seed: int
    g: np.ndarray                                # (N,)
    scores_neutral: Optional[np.ndarray] = None  # A: (N, K⁰)
    w_neutral: Optional[np.ndarray] = None
    scores_plus: Optional[np.ndarray] = None     # B, C: (N, K⁺)
    p_plus: Optional[np.ndarray] = None
    scores_minus: Optional[np.ndarray] = None    # B: (N, K⁻)
    q_minus: Optional[np.ndarray] = None
    floor_value: Optional[float] = None          # B: resolved floor f
    scores_named: Optional[np.ndarray] = None    # C: (N, |D|)
    c_named: Optional[np.ndarray] = None         # C: per-state thresholds
    scores_nested: Optional[np.ndarray] = None   # D: (N, O, K_in)
    w_inner: Optional[np.ndarray] = None         # D: (O, K_in), rows sum to 1
    pi_outer: Optional[np.ndarray] = None        # D: (O,)
    rel_profile: Optional[np.ndarray] = None     # D: (O,), max = 1

    @property
    def n_points(self) -> int:
        """Size of X, i.e. H^d."""
        return self.g.shape[0]

    @property
    def beta_bounds(self) -> dict:
        """Per-origin lower bounds for the tail level:
        each β axis is bounded by its own set's smallest nominal weight.
        Case D's shared β_in uses max_o min_k w_inner (the level at which
        no inner functional is guaranteed-degenerate)."""
        if self.case == "A":
            return {"n": float(self.w_neutral.min())}
        if self.case == "B":
            return {"p": float(self.p_plus.min()),
                    "m": float(self.q_minus.min())}
        if self.case == "C":
            return {"p": float(self.p_plus.min())}
        if self.case == "D":
            # "out": the outer tail level β_out is a conditioned axis
            # (the joint-satisfaction dial) bounded by min π
            return {"in": float(self.w_inner.min(axis=1).max()),
                    "out": float(self.pi_outer.min())}
        raise ValueError(f"unknown case {self.case!r}")

    @property
    def beta_min(self) -> float:
        """Bound for a *shared* β applied to every active set at once
        (tied conditions, O1/O3 grids): the max of the per-set bounds.
        Supersedes a pooled global minimum, which under-protects
        sets whose smallest weight exceeds the pooled one."""
        return float(max(self.beta_bounds.values()))


def make_world(case: str, cfg: WorldConfig, seed: int) -> World:
    """One geometry draw (no hardness gate; see sample_world)."""
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}")
    rng = np.random.default_rng(seed)
    H, d = cfg.H, cfg.d
    n = H ** d
    if cfg.g_iid:
        g = rng.uniform(0.0, 1.0, size=n)
        g = g ** cfg.sparsity if cfg.sparsity != 1.0 else g
    else:
        g = _score_field(rng, cfg)
    w = World(case=case, cfg=cfg, seed=seed, g=g)
    if case == "A":
        w.scores_neutral = _fields_cfg(rng, cfg, cfg.k_neutral)
        w.w_neutral = _weights(rng, cfg.k_neutral, cfg.weight_alpha)
        if cfg.k_guard:  # guard draws AFTER the existing ones,
            #              so k_guard = 0 keeps the stream identical
            w.scores_named = _fields_cfg(rng, cfg, cfg.k_guard,
                                         constraint=True)
            w.c_named = np.full(cfg.k_guard, cfg.c_named)
    elif case == "B":
        w.scores_plus = _fields_cfg(rng, cfg, cfg.k_plus)
        w.p_plus = _weights(rng, cfg.k_plus, cfg.weight_alpha)
        w.scores_minus = _fields_cfg(rng, cfg, cfg.k_minus)
        w.q_minus = _weights(rng, cfg.k_minus, cfg.weight_alpha)
        if cfg.floor is not None:
            w.floor_value = float(cfg.floor)
        else:
            from .risk import dro_cvar_lower
            phi_ref = dro_cvar_lower(w.scores_plus, w.p_plus,
                                     *cfg.floor_ref)
            w.floor_value = float(np.quantile(phi_ref,
                                              cfg.floor_quantile))
    elif case == "C":
        w.scores_plus = _fields_cfg(rng, cfg, cfg.k_plus)
        w.p_plus = _weights(rng, cfg.k_plus, cfg.weight_alpha)
        w.scores_named = _fields_cfg(rng, cfg, cfg.k_named,
                                     constraint=True)
        w.c_named = np.full(cfg.k_named, cfg.c_named)
    elif case == "D":
        w.scores_nested = np.stack(
            [_fields_cfg(rng, cfg, cfg.k_inner)
             for _ in range(cfg.n_outer)], axis=1)
        rows = [_weights(rng, cfg.k_inner, cfg.weight_alpha)
                for _ in range(cfg.n_outer)]
        w.w_inner = np.stack(rows, axis=0)
        w.pi_outer = _weights(rng, cfg.n_outer, cfg.weight_alpha)
        lo, hi = cfg.profile_range
        r = np.exp(rng.uniform(np.log(lo), np.log(hi), size=cfg.n_outer))
        w.rel_profile = r / r.max()
    return w


def hardness_report(world: World) -> dict:
    """Faithful-hardness probes. Lazy imports of
    cases/target avoid a cycle."""
    from .cases import (log_reward_from, nested_vs_flat_tv, psi_and_masks,
                        target_for)
    from .conditions import Condition, RiskB, RiskD, tied_risk
    from .target import p_star, tv

    cfg = world.cfg
    bounds = world.beta_bounds

    def probe_beta(key=None):
        """Risk-on probe tail level, respecting the world's bound:
        the shared bound by default, one origin's own bound for the
        per-origin probes."""
        lo = world.beta_min if key is None else bounds[key]
        return 0.25 if 0.25 >= lo else min(0.99, lo * 1.05)

    probe_on = Condition(4.0, 0.3, tied_risk(world.case, probe_beta(), 0.5))
    probe_off = Condition(4.0, 0.3, tied_risk(world.case, 1.0, 0.0))
    psi, floor_ok, veto_ok = psi_and_masks(world, probe_on.risk)
    log_r_on = log_reward_from(world, probe_on, psi, floor_ok, veto_ok)
    p_on = p_star(log_r_on, probe_on.beta_t)
    p_off = target_for(world, probe_off)
    rep = {"tv_on_off": float(tv(p_on, p_off)), "probe": probe_on}
    if world.case == "A" and world.scores_named is not None:
        # guarded A: veto share REPORTED, not gated
        # (the gate stays untouched for this axis)
        rep["veto_frac"] = float(1.0 - veto_ok.mean())
    if world.case == "B":
        rep["floor_fail_frac"] = float(1.0 - floor_ok.mean())
        # the two origins must be distinguishable
        only_p = Condition(4.0, 0.3, RiskB(probe_beta("p"), 0.5, 1.0, 0.0))
        only_m = Condition(4.0, 0.3, RiskB(1.0, 0.0, probe_beta("m"), 0.5))
        rep["tv_origins"] = float(tv(target_for(world, only_p),
                                     target_for(world, only_m)))
    if world.case == "C":
        rep["veto_frac"] = float(1.0 - veto_ok.mean())
        # veto share at the margin endpoint δ = δ_max
        thr = world.c_named - cfg.delta_probe
        rep["veto_frac_delta_max"] = float(
            np.any(world.scores_named >= thr, axis=-1).mean())
    if world.case == "D":
        rep["tv_nested_flat"] = float(nested_vs_flat_tv(world, probe_on))
        # aimed distrust must matter: the one-hot per-origin
        # targets must be mutually distinguishable. The β_in probe sits
        # mid-family (≥ 0.5): at deep-tail levels the realized worst
        # state is invariant over the whole KL ball, ρ is pointwise
        # inert, and aiming cannot register.
        beta_pr = min(0.99, max(0.5, bounds["in"] * 1.05))
        n_o = world.w_inner.shape[0]
        one_hot = []
        for o in range(n_o):
            v = np.full(n_o, 0.05)
            v[o] = 0.8
            one_hot.append(target_for(world, Condition(
                4.0, 0.3, RiskD(beta_pr, tuple(v), 0.5))))
        rep["tv_origins"] = float(max(
            tv(one_hot[i], one_hot[j])
            for i in range(n_o) for j in range(i + 1, n_o)))
        # the joint-satisfaction dial must matter:
        # compensatory vs conjunctive outer at shared radii
        bo = probe_beta("out")
        p_comp = target_for(world, Condition(
            4.0, 0.3, RiskD(beta_pr, 0.5, 1.0)))
        p_conj = target_for(world, Condition(
            4.0, 0.3, RiskD(beta_pr, 0.5, bo)))
        rep["tv_beta_out"] = float(tv(p_comp, p_conj))
    return rep


def is_hard(world: World, rep: dict | None = None) -> bool:
    """Check a world against its case's faithful-hardness gate.

    Args:
        world: World to check.
        rep: Precomputed `hardness_report(world)`; computed if omitted.

    Returns:
        True iff every gate bound for `world.case` is satisfied.
    """
    cfg = world.cfg
    rep = hardness_report(world) if rep is None else rep
    if rep["tv_on_off"] < cfg.hardness_margin:
        return False
    if world.case == "B":
        lo, hi = cfg.floor_frac_range
        if not (lo <= rep["floor_fail_frac"] <= hi):
            return False
        if rep["tv_origins"] < cfg.hardness_margin:
            return False
    if world.case == "C":
        lo, hi = cfg.veto_frac_range
        # veto share is monotone in δ, so these two endpoint checks
        # bound the whole band
        if not (lo <= rep["veto_frac"]
                and rep["veto_frac_delta_max"] <= hi):
            return False
    if world.case == "D":
        if rep["tv_nested_flat"] < cfg.hardness_margin:
            return False
        if rep["tv_origins"] < cfg.hardness_margin:
            return False
        if rep["tv_beta_out"] < cfg.hardness_margin:
            return False
    return True


def sample_world(case: str, cfg: WorldConfig, seed: int,
                 gate: bool = True) -> tuple[World, int]:
    """Rejection-sample a world passing the hardness gate.

    Returns (world, attempts). Attempts are reported, never silently
    absorbed. Seeds are derived deterministically
    from (seed, attempt) so the family is reproducible.

    `gate=False` (ablation): return the FIRST draw unconditionally
    (attempts=1, same seed chain as the gated run); hardness probe
    values remain available via `hardness_report`/`is_hard`.
    """
    if not gate:
        return make_world(case, cfg, seed * 100_003), 1
    for attempt in range(cfg.max_attempts):
        world = make_world(case, cfg, seed * 100_003 + attempt)
        if is_hard(world):
            return world, attempt + 1
    raise RuntimeError(
        f"case {case}: no hard world in {cfg.max_attempts} attempts "
        f"(seed {seed}); geometry family or margins need revisiting")
