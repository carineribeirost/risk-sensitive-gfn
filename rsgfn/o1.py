"""O1 driver: map the region of condition space where the
exact target departs from its risk-off baseline (β_cvar = 1, ρ = 0),
and (Case D) from the flattened single-level aggregation. Extended by
O1b: the separability of the per-origin family from the
tied single-pair family.

Everything here is enumeration: deterministic per world, no training.
Output is a list of flat row dicts, ready for CSV.
"""

from __future__ import annotations

import numpy as np

from .cases import nested_vs_flat_tv, target_for, target_worst
from .conditions import (Condition, RiskA, RiskB, RiskC, RiskD,
                         tied_risk)
from .target import tv
from .worlds import World, WorldConfig, sample_world


def _with_ball(risk, ball: str):
    """Stamp the ambiguity-ball geometry on a risk block; "kl"
    returns the block untouched."""
    if ball == "kl":
        return risk
    import dataclasses
    return dataclasses.replace(risk, geometry=ball)


def _tied(world: World, beta_t, w_g, beta, rho,
          ball: str = "kl") -> Condition:
    return Condition(beta_t, w_g,
                     _with_ball(tied_risk(world.case, beta, rho), ball))


def o1_grid(world: World, beta_grid, rho_grid,
            beta_t: float = 4.0, w_g: float = 0.3,
            ball: str = "kl") -> list[dict]:
    """TV(p*_on, p*_off) over the tied (β, ρ) grid; plus nested-vs-flat
    for Case D. β values below the world's shared bound
    are skipped and reported as skipped, not silently dropped.
    `ball`: ambiguity geometry on every grid
    cell; pass the matching per-ball rho_grid: the
    (β, ρ)-map per ball is the atlas quantity."""
    base = _tied(world, beta_t, w_g, 1.0, 0.0)
    p_off = target_for(world, base)
    p_wst = target_worst(world, base)      # pure worst-case pole
    tv_off_worst = tv(p_off, p_wst)        # separation of the two poles
    rows = []
    for b in beta_grid:
        for r in rho_grid:
            row = {"case": world.case, "world_seed": world.seed,
                   "beta_cvar": float(b), "rho": float(r),
                   "beta_t": beta_t, "w_g": w_g, "ball": ball,
                   "tv_off_worst": tv_off_worst}
            if b < world.beta_min:
                row.update(tv_on_off=np.nan, tv_vs_worst=np.nan,
                           skipped="beta<beta_min")
                rows.append(row)
                continue
            cond = _tied(world, beta_t, w_g, float(b), float(r), ball)
            p_c = target_for(world, cond)
            row["tv_on_off"] = tv(p_c, p_off)
            row["tv_vs_worst"] = tv(p_c, p_wst)
            if world.case == "D":
                row["tv_nested_flat"] = nested_vs_flat_tv(world, cond)
            rows.append(row)
    return rows


def _shared_family(world: World, beta_grid, rho_grid, beta_t,
                   w_g, ball: str = "kl") -> list[np.ndarray]:
    """Exact targets of the tied family over the grid (admissible β
    only). For Case D the tied member has shared radii at the reference
    outer level: the original family."""
    return [target_for(world, _tied(world, beta_t, w_g,
                                    float(b), float(r), ball))
            for b in beta_grid if b >= world.beta_min
            for r in rho_grid]


def o1_separability(world: World, beta_grid=None, rho_grid=None,
                    beta_t: float = 4.0, w_g: float = 0.3,
                    n_delta: int = 5, ball: str = "kl") -> list[dict]:
    """O1b: for asymmetric per-origin conditions, the
    distance to the closest member of the tied single-pair family,
    tv_to_shared = min over the tied grid of TV(p*_asym, p*_tied).
    ≈ 0 everywhere → the per-origin split is decoration for this world.

    For Case B, rows carry the metric against both nested references
    (tied ⊂ ρ-split ⊂ full split), attributing enlargement to ρ vs β.
    For Case D the probes are one-hot aimed distrust per origin,
    at the reference outer level (`tv_to_shared`) and with the
    conjunctive outer (`tv_to_shared_full_split`), attributing
    enlargement to aiming vs the joint-satisfaction dial.
    Case A: no rows unless the world carries a guard
    (`WorldConfig.k_guard > 0`); then the probe is C's δ-sweep
    verbatim.
    """
    beta_grid = (np.linspace(0.1, 1.0, 7) if beta_grid is None
                 else np.asarray(beta_grid))
    rho_grid = (np.linspace(0.0, 1.6, 7) if rho_grid is None
                else np.asarray(rho_grid))
    rows: list[dict] = []
    if world.case == "A" and world.scores_named is None:
        return rows
    bounds = world.beta_bounds

    def mid_beta(key):
        lo = bounds[key]
        return float(np.clip(0.5, lo, 1.0))

    if world.case == "A":
        # guarded A: C's δ-sweep probe on the guard veto
        # (δ = 0 is the tied member; same margin, same verdict path)
        shared = _shared_family(world, beta_grid, rho_grid, beta_t,
                                w_g, ball)
        b = mid_beta("n")
        for r in rho_grid:
            for d in np.linspace(0.0, 0.1, n_delta)[1:]:
                p = target_for(world, Condition(
                    beta_t, w_g,
                    _with_ball(RiskA(b, float(r), delta=float(d)),
                               ball)))
                rows.append({"case": "A", "world_seed": world.seed,
                             "rho": float(r), "delta": float(d),
                             "tv_to_shared":
                                 min(tv(p, q) for q in shared)})
        return rows
    if world.case == "B":
        shared = _shared_family(world, beta_grid, rho_grid, beta_t,
                                w_g, ball)
        bp, bm = mid_beta("p"), mid_beta("m")
        b_lo = float(np.clip(0.25, bounds["p"], 1.0))
        cells = [(float(rp), float(rm))
                 for rp in rho_grid for rm in rho_grid]
        # ρ-split-only family (β tied at mid) over the (ρ⁺, ρ⁻) grid:
        # both the asymmetric probes and the projection reference for
        # the full-split targets
        p_rho = {c: target_for(world, Condition(
            beta_t, w_g,
            _with_ball(RiskB(bp, c[0], bm, c[1]), ball)))
            for c in cells}
        for c in cells:
            # full split: β asymmetric too (deep tail on +, mean on −)
            p_full = target_for(world, Condition(
                beta_t, w_g,
                _with_ball(RiskB(b_lo, c[0], 1.0, c[1]), ball)))
            rows.append({"case": "B", "world_seed": world.seed,
                         "rho_p": c[0], "rho_m": c[1],
                         "tv_to_shared":
                             min(tv(p_rho[c], q) for q in shared),
                         "tv_to_shared_full_split":
                             min(tv(p_full, q) for q in shared),
                         "tv_to_rho_split":
                             min(tv(p_full, q) for q in p_rho.values())})
    elif world.case == "C":
        shared = _shared_family(world, beta_grid, rho_grid, beta_t,
                                w_g, ball)
        b = mid_beta("p")
        for r in rho_grid:
            for d in np.linspace(0.0, 0.1, n_delta)[1:]:  # δ=0 is tied
                p = target_for(world, Condition(
                    beta_t, w_g,
                    _with_ball(RiskC(b, float(r), float(d)), ball)))
                rows.append({"case": "C", "world_seed": world.seed,
                             "rho": float(r), "delta": float(d),
                             "tv_to_shared": min(tv(p, q) for q in shared)})
    elif world.case == "D":
        # nested inclusions (mirroring B's attribution):
        # tied ⊂ per-origin ρ at the reference outer ⊂ + conditioned
        # outer. Probes are one-hot aimed distrust per origin.
        shared = _shared_family(world, beta_grid, rho_grid, beta_t,
                                w_g, ball)
        b = mid_beta("in")
        bo = float(np.clip(0.3, bounds["out"], 1.0))  # conjunctive probe
        n_o = world.w_inner.shape[0]
        for o in range(n_o):
            for lvl in rho_grid[1:]:
                v = np.full(n_o, 0.05)
                v[o] = float(lvl)
                p_aim = target_for(world, Condition(
                    beta_t, w_g,
                    _with_ball(RiskD(b, tuple(v), 0.5), ball)))
                p_full = target_for(world, Condition(
                    beta_t, w_g,
                    _with_ball(RiskD(b, tuple(v), bo), ball)))
                rows.append({"case": "D", "world_seed": world.seed,
                             "origin": o, "rho_hi": float(lvl),
                             "tv_to_shared":
                                 min(tv(p_aim, q) for q in shared),
                             "tv_to_shared_full_split":
                                 min(tv(p_full, q) for q in shared)})
    return rows


def o1_rho_out_sep(world: World, beta_grid, rho_grid, rho_out_grid,
                   beta_t: float = 4.0, w_g: float = 0.3,
                   ball: str = "kl") -> list[dict]:
    """Liveness probe (case D only): sweep the conditioned outer radius
    ρ_out at the reference outer level with tied mid inner radii;
    tv_to_shared_rho_out = min TV to the shared family, which holds
    ρ_out at the world constant, so the ρ_out = cfg.rho_out probe is
    an ≈0 sanity anchor. ≈ 0 across the whole grid → ρ_out is inert
    for this world (a limits-atlas row, reported not gated). Rows
    carry probe="rho_out" and are written to a SEPARATE CSV so the
    separability verdict is untouched."""
    rows: list[dict] = []
    if world.case != "D":
        return rows
    shared = _shared_family(world, beta_grid, rho_grid, beta_t, w_g,
                            ball)
    # β from the ADMISSIBLE grid nearest 0.5 (not clip(0.5): the probe
    # must be an exact member of the shared family's support so the
    # ρ_out = cfg.rho_out anchor lands at tv = 0 exactly)
    admissible = [float(x) for x in beta_grid if x >= world.beta_min]
    b = min(admissible, key=lambda x: abs(x - 0.5))
    # same exact-membership rule for the inner radius (an even-length
    # grid's median is not a grid point)
    med = float(np.median(np.asarray(rho_grid, dtype=float)))
    r_mid = min((float(r) for r in rho_grid), key=lambda x: abs(x - med))
    for ro in rho_out_grid:
        p = target_for(world, Condition(
            beta_t, w_g,
            _with_ball(RiskD(b, r_mid, 0.5, rho_out=float(ro)), ball)))
        rows.append({"case": "D", "world_seed": world.seed,
                     "probe": "rho_out", "rho_in_mid": r_mid,
                     "rho_out": float(ro),
                     "tv_to_shared_rho_out":
                         min(tv(p, q) for q in shared)})
    return rows


def o1_case_param_sweep(case: str, cfg: WorldConfig, world: World,
                        probe: Condition, values) -> list[dict]:
    """Sweep the case-specific parameter (γ for B, c_d for C) at a fixed
    risk-on probe, measuring TV against the world's own fixed-parameter
    target. Cases A and D have no scalar parameter here (D's nesting is
    covered by tv_nested_flat)."""
    rows = []
    p_ref = target_for(world, probe)
    for v in values:
        if case == "B":
            old = world.cfg.gamma
            world.cfg.gamma = float(v)
            p_v = target_for(world, probe)
            world.cfg.gamma = old
            name = "gamma"
        elif case == "C":
            old = world.c_named.copy()
            world.c_named = np.full_like(old, float(v))
            p_v = target_for(world, probe)
            world.c_named = old
            name = "c_named"
        else:
            return rows
        rows.append({"case": case, "world_seed": world.seed, name: float(v),
                     "tv_vs_fixed_param": tv(p_v, p_ref)})
    return rows


def run_o1(case: str, cfg: WorldConfig, world_seeds,
           beta_grid=None, rho_grid=None, gate: bool = True,
           ball: str = "kl", rho_out_grid=None
           ) -> tuple[list[dict], list[dict], list[dict]]:
    """Full O1 + O1b for one case over test worlds. Returns
    (grid_rows, sep_rows, meta): meta records hardness-gate attempts and
    probe values per world (no silent caps). `gate=False` takes
    the first world draw per seed; rows carry a `gated` column and meta
    an `is_hard` verdict either way.

    `ball`: ambiguity-ball geometry on every
    probe; pass the matching per-ball rho_grid. `rho_out_grid`
    (case D only): adds probe="rho_out" liveness rows to sep_rows:
    the caller writes them to a separate CSV. None = off
    (default: unchanged)."""
    from .worlds import hardness_report, is_hard

    beta_grid = (np.linspace(0.1, 1.0, 7) if beta_grid is None
                 else np.asarray(beta_grid))
    rho_grid = (np.linspace(0.0, 1.6, 7) if rho_grid is None
                else np.asarray(rho_grid))
    rows, sep_rows, meta = [], [], []
    for seed in world_seeds:
        world, attempts = sample_world(case, cfg, seed, gate=gate)
        rep = hardness_report(world)
        meta.append({"case": case, "seed": seed,
                     "world_seed": world.seed, "attempts": attempts,
                     "gated": gate, "is_hard": is_hard(world, rep),
                     **{f"probe_{k}": float(v) for k, v in rep.items()
                        if isinstance(v, (int, float))},
                     "beta_min": world.beta_min,
                     **{f"beta_bound_{k}": v
                        for k, v in world.beta_bounds.items()}})
        new_rows = o1_grid(world, beta_grid, rho_grid, ball=ball)
        new_sep = o1_separability(world, beta_grid, rho_grid, ball=ball)
        if rho_out_grid is not None and case == "D":
            new_sep = new_sep + o1_rho_out_sep(world, beta_grid,
                                               rho_grid, rho_out_grid,
                                               ball=ball)
        for r in new_rows + new_sep:
            r["gated"] = gate
        rows.extend(new_rows)
        sep_rows.extend(new_sep)
    return rows, sep_rows, meta
