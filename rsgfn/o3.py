"""O3 (extension beyond O1/O2): is the risk-sensitive family
*better* than its two poles, Boltzmann (risk-off) and pure worst-case,
at the declared utility: generating objects that jointly satisfy ALL
active states and the case's structural conditions?

Utility semantics (declared, invented, domain-translatable):
a point x is *satisfying at level t* iff every "good" state clears t,
every penalise state stays below 1 − t, and no hard veto fires:

  A: a_k(x) ≥ t  for all k ∈ S⁰
  B: a_k(x) ≥ t  for all k ∈ S⁺  and  a_k(x) ≤ 1 − t  for all k ∈ S⁻
  C: a_k(x) ≥ t  for all k ∈ S⁺  and  a_d(x) < c_d  for all d ∈ D
  D: a_{o,k}(x) ≥ t  for all (o, k)

(The Case-B floor is a shaping term of the reward, defined on Φ and
hence method-dependent, so it is deliberately NOT part of the utility.)

Per target distribution p (Boltzmann pole, worst-case pole, and the
risk-on grid) we report, exactly by enumeration:

- sat_mass: probability that one draw from p satisfies, P_{x~p}[sat].
- eff_candidates: exp(entropy) of p restricted to the satisfying set:
  the effective number of DISTINCT satisfying candidates the sampler
  yields (worst-case's known failure mode: it collapses this to ~1).
- stress_mean / stress_p05: mean and 5th percentile over invented
  weight perturbations ν ~ Dirichlet(κ·nominal) of the realized
  aggregate utility E_{x~p}[U_ν(x)] (Boltzmann's known failure mode:
  the tail). The same ν draws are used for every target in a world, so
  the comparison is paired.

The prediction, declared in advance: risk-on cells beat Boltzmann on
sat_mass and stress_p05 while beating worst-case on eff_candidates and
stress_mean, i.e. the (β_cvar, ρ) family traces a Pareto frontier
between the poles. If a pole dominates everywhere, the aggregation
machinery is decoration for this world family.
"""

from __future__ import annotations

import numpy as np

from .baselines import target_baseline
from .cases import target_for, target_worst
from .conditions import Condition, RiskB, tied_risk
from .worlds import (World, WorldConfig, hardness_report, is_hard,
                     sample_world)


def satisfaction_mask(world: World, t: float) -> np.ndarray:
    """Boolean (N,): joint satisfaction at level t (module docstring)."""
    if world.case == "A":
        return (world.scores_neutral >= t).all(axis=-1)
    if world.case == "B":
        return ((world.scores_plus >= t).all(axis=-1)
                & (world.scores_minus <= 1.0 - t).all(axis=-1))
    if world.case == "C":
        no_veto = ~np.any(world.scores_named >= world.c_named, axis=-1)
        return (world.scores_plus >= t).all(axis=-1) & no_veto
    if world.case == "D":
        n = world.n_points
        return (world.scores_nested >= t).reshape(n, -1).all(axis=-1)
    raise ValueError(f"unknown case {world.case!r}")


def challenge_level(world: World, min_frac: float = 0.01) -> float:
    """Highest t (on a fine grid) at which at least min_frac of X still
    satisfies: the hardest non-degenerate joint requirement this world
    supports. Reported per world; satisfaction metrics are evaluated
    here and at the fixed t_levels."""
    for t in np.linspace(0.95, 0.0, 96):
        if satisfaction_mask(world, float(t)).mean() >= min_frac:
            return float(round(t, 3))
    return 0.0


def dist_metrics(p: np.ndarray, sat: np.ndarray) -> dict:
    """Satisfaction mass and effective candidate count of a distribution.

    Args:
        p: Probability mass over X, shape (N,).
        sat: Boolean satisfaction mask over X, shape (N,).

    Returns:
        Dict with `sat_mass` (probability one draw satisfies), `n_sat`
        (number of satisfying states), and `eff_candidates` (the
        exponentiated entropy of p restricted and renormalized to the
        satisfying set; 0.0 if `sat_mass` is 0).
    """
    mass = float(p[sat].sum())
    out = {"sat_mass": mass, "n_sat": int(sat.sum())}
    if mass > 0.0:
        q = p[sat] / mass
        ent = float(-(q * np.log(np.maximum(q, 1e-300))).sum())
        out["eff_candidates"] = float(np.exp(ent))
    else:
        out["eff_candidates"] = 0.0
    return out


def _perturbed(rng, w, kappa, n_draws):
    alpha = np.maximum(kappa * np.asarray(w, dtype=float), 1e-3)
    return rng.dirichlet(alpha, size=n_draws)  # (S, K)


def draw_stress(world: World, rng, n_draws: int, kappa) -> dict:
    """One shared set of weight perturbations per world (paired across
    targets). `kappa` may be a scalar or a per-origin
    mapping: keys "p"/"m" (B), "p" (C), "n" (A), "in"/"out" (D), so
    worlds carry a *true* reliability asymmetry."""
    if not isinstance(kappa, dict):
        kappa = {k: float(kappa) for k in ("n", "p", "m", "in", "out")}
    d = {}
    if world.case == "A":
        d["nu"] = _perturbed(rng, world.w_neutral, kappa["n"], n_draws)
    elif world.case == "B":
        d["nu_p"] = _perturbed(rng, world.p_plus, kappa["p"], n_draws)
        d["nu_q"] = _perturbed(rng, world.q_minus, kappa["m"], n_draws)
    elif world.case == "C":
        d["nu_p"] = _perturbed(rng, world.p_plus, kappa["p"], n_draws)
    elif world.case == "D":
        d["pi"] = _perturbed(rng, world.pi_outer, kappa["out"], n_draws)
        d["nu_in"] = np.stack([_perturbed(rng, world.w_inner[o],
                                          kappa["in"], n_draws)
                               for o in range(world.w_inner.shape[0])],
                              axis=1)  # (S, O, K_in)
    return d


def realized_utility(world: World, p: np.ndarray, stress: dict) -> np.ndarray:
    """(S,): realized aggregate utility E_{x~p}[U_ν(x)] per perturbation
    ν in `stress`, exactly by enumeration."""
    cfg = world.cfg
    if world.case == "A":
        u = world.scores_neutral @ stress["nu"].T          # (N, S)
    elif world.case == "B":
        u = (world.scores_plus @ stress["nu_p"].T
             - cfg.gamma * (world.scores_minus @ stress["nu_q"].T))
    elif world.case == "C":
        ok = ~np.any(world.scores_named >= world.c_named, axis=-1)
        u = (world.scores_plus @ stress["nu_p"].T) * ok[:, None]
    elif world.case == "D":
        # (S,N): per draw s, Σ_o π'_so · (a[:,o,:] @ ν_so)
        inner = np.einsum("nok,sok->sno", world.scores_nested,
                          stress["nu_in"])
        u = np.einsum("sno,so->ns", inner, stress["pi"])
    else:
        raise ValueError(f"unknown case {world.case!r}")
    return p @ u                                            # (S,)


def stress_metrics(world: World, p: np.ndarray, stress: dict) -> dict:
    """Realized aggregate utility per perturbation ν, summarized by
    mean and 5th percentile over draws."""
    realized = realized_utility(world, p, stress)
    return {"stress_mean": float(realized.mean()),
            "stress_p05": float(np.quantile(realized, 0.05))}


def _nominal_weights(world: World) -> dict:
    """The weighted origins of a world, keyed by the stress-dict slot
    they occupy: origin key -> (slot, index-within-slot, weights)."""
    if world.case == "A":
        return {"n": ("nu", None, world.w_neutral)}
    if world.case == "B":
        return {"p": ("nu_p", None, world.p_plus),
                "m": ("nu_q", None, world.q_minus)}
    if world.case == "C":
        return {"p": ("nu_p", None, world.p_plus)}
    if world.case == "D":
        d = {"out": ("pi", None, world.pi_outer)}
        for o in range(world.w_inner.shape[0]):
            d[f"in{o}"] = ("nu_in", o, world.w_inner[o])
        return d
    raise ValueError(f"unknown case {world.case!r}")


def contamination_metrics(world: World, p: np.ndarray,
                          eps: float = 0.2) -> dict:
    """ε-contamination arm: out-of-Dirichlet perturbation. For each
    weighted origin and each of its states k, contaminate the nominal
    weights ν = (1−ε)·w + ε·e_k and price E_{x~p}[U_ν(x)] exactly via
    the stress machinery (each ν is one draw; all other origins stay
    nominal). Returns the min over origins × states (`contam_worst`)
    plus its argmin for the record."""
    origins = _nominal_weights(world)
    worst, arg_origin, arg_state = np.inf, None, None
    for key, (slot, idx, w) in origins.items():
        w = np.asarray(w, dtype=float)
        k = len(w)
        nus = (1.0 - eps) * w[None, :] + eps * np.eye(k)   # (K, K)
        stress = {}
        for key2, (slot2, idx2, w2) in origins.items():
            if slot2 == "nu_in":
                cur = stress.setdefault(
                    "nu_in", np.tile(world.w_inner, (k, 1, 1)))
                if key2 == key:
                    cur[:, idx2, :] = nus
            else:
                stress[slot2] = (nus if key2 == key
                                 else np.tile(np.asarray(w2, dtype=float),
                                              (k, 1)))
        realized = realized_utility(world, p, stress)      # (K,)
        j = int(np.argmin(realized))
        if realized[j] < worst:
            worst, arg_origin, arg_state = float(realized[j]), key, j
    return {"contam_worst": worst, "contam_origin": arg_origin,
            "contam_state": arg_state, "contam_eps": float(eps)}


def run_o3(case: str, cfg: WorldConfig, world_seeds,
           beta_grid=None, rho_grid=None,
           t_levels=(0.25, 0.35, 0.45), beta_t: float = 4.0,
           w_g: float = 0.3, n_stress: int = 200,
           kappa: float = 25.0, kappa_range=None, seed: int = 0,
           contam_eps: float | None = 0.2, gate: bool = True,
           ball: str = "kl", rho_out_grid=None,
           baselines=(), desirability_lo: float = 0.50,
           desirability_hi: float = 0.95,
           desirability_shape: float = 1.0, skip_grid: bool = False):
    """Returns (rows, meta). One row per (world, target, t-level);
    targets are the Boltzmann pole, the worst-case pole, every tied
    (β_cvar, ρ) grid cell above the world's β bound (below-bound cells
    are reported as skipped), and (Case B) the asymmetric per-origin
    (ρ⁺, ρ⁻) cells at mid tail levels (target kind "risk_asym").

    `kappa_range=(lo, hi)` draws per-origin κ log-uniformly per world,
    giving worlds a true reliability asymmetry; None
    keeps the single scalar `kappa` for every origin.

    `contam_eps`: every target row also carries `contam_worst`,
    the exact ε-contamination worst case over origins × states
    (`contamination_metrics`); None disables the columns.

    `ball`: ambiguity-ball geometry on every grid-cell risk block
    ("kl" default = the plain KL ball); pass the matching
    per-ball rho_grid (risk.BALL_RHO_GRIDS); radii are ball-specific.

    `rho_out_grid` (case D only): when given, adds target kind
    "risk_rho_out": the outer radius swept over the grid × the inner
    shared radius over `rho_grid`, both tail levels mid-family (the
    exact mirror of case B's "risk_asym" block). None = off (default:
    no new rows, unchanged).

    `baselines`: matched-conjunctivity aggregation baselines
    (`baselines.KINDS`) to add as extra target rows, kind
    "baseline_<name>"; `desirability_lo/hi/shape` are the declared
    Derringer-Suich ramp bounds passed through to the desirability
    baseline. `skip_grid`: poles, probe cell and baselines only, no
    (β_cvar, ρ) grid and no case-B/D asymmetric extensions — turns a
    baseline-only comparison into a much cheaper run."""
    # defaults include the comparator corners:
    # β=1 rows are DRO-only cells, ρ=0 columns CVaR-only, β=1∧ρ=0 is
    # priced separately as the Boltzmann pole; ρ=1.2 approaches the
    # worst-case pole
    beta_grid = (np.array([0.15, 0.3, 0.6, 1.0]) if beta_grid is None
                 else np.asarray(beta_grid))
    rho_grid = (np.array([0.0, 0.2, 0.5, 1.2]) if rho_grid is None
                else np.asarray(rho_grid))
    rows, meta = [], []
    for ws in world_seeds:
        world, attempts = sample_world(case, cfg, ws, gate=gate)
        rng = np.random.default_rng(10_000 + ws)
        if kappa_range is not None:
            lo, hi = kappa_range
            kap = {k: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                   for k in ("n", "p", "m", "in", "out")}
        else:
            kap = kappa
        stress = draw_stress(world, rng, n_stress, kap)
        t_star = challenge_level(world)
        ts = sorted(set([float(t) for t in t_levels] + [t_star]))
        base = Condition(beta_t, w_g, tied_risk(case, 1.0, 0.0))
        targets = [("boltzmann", {}, target_for(world, base)),
                   ("worst", {}, target_worst(world, base))]
        n_skipped = 0

        def with_ball(risk):
            # "kl" keeps the block bytewise identical
            if ball == "kl":
                return risk
            import dataclasses
            return dataclasses.replace(risk, geometry=ball)

        # pre-named operating point: the hardness probe-on
        # member (β = 0.25 ∨ bound·1.05, ρ = 0.5), fixed BEFORE the
        # fine grid was run. Adaptive
        # per world exactly like the gate probe, so the row exists
        # even where beta_min > 0.25 (the fixed sweep grid skips
        # below-bound β rows and need not contain 0.25 at all).
        bp = 0.25 if 0.25 >= world.beta_min else min(
            0.99, world.beta_min * 1.05)
        targets.append(("probe_cell",
                        {"beta_cvar": bp, "rho": 0.5},
                        target_for(world, Condition(
                            beta_t, w_g,
                            with_ball(tied_risk(case, bp, 0.5))))))
        for bl in baselines:
            targets.append(
                ("baseline_" + bl,
                 ({"lo_q": desirability_lo, "hi_q": desirability_hi,
                   "shape": desirability_shape}
                  if bl == "desirability" else {}),
                 target_baseline(world, base, bl,
                                 lo_q=desirability_lo,
                                 hi_q=desirability_hi,
                                 shape=desirability_shape)))
        for b in (() if skip_grid else beta_grid):
            if b < world.beta_min:
                n_skipped += len(rho_grid)
                continue
            for r in rho_grid:
                cond = Condition(beta_t, w_g,
                                 with_ball(tied_risk(case, float(b),
                                                     float(r))))
                targets.append(("risk",
                                {"beta_cvar": float(b), "rho": float(r)},
                                target_for(world, cond)))
        if case == "B" and not skip_grid:
            bounds = world.beta_bounds
            bp = float(np.clip(0.5, bounds["p"], 1.0))
            bm = float(np.clip(0.5, bounds["m"], 1.0))
            for rp in rho_grid:
                for rm in rho_grid:
                    cond = Condition(beta_t, w_g,
                                     with_ball(RiskB(bp, float(rp),
                                                     bm, float(rm))))
                    targets.append(("risk_asym",
                                    {"rho_p": float(rp),
                                     "rho_m": float(rm)},
                                    target_for(world, cond)))
        if case == "D" and rho_out_grid is not None and not skip_grid:
            from .conditions import RiskD
            bounds = world.beta_bounds
            bi = float(np.clip(0.5, bounds["in"], 1.0))
            bo = float(np.clip(0.5, bounds["out"], 1.0))
            for ri in rho_grid:
                for ro in rho_out_grid:
                    cond = Condition(beta_t, w_g,
                                     with_ball(RiskD(bi, float(ri), bo,
                                                     rho_out=float(ro))))
                    targets.append(("risk_rho_out",
                                    {"rho": float(ri),
                                     "rho_out": float(ro)},
                                    target_for(world, cond)))
        for kind, info, p in targets:
            s_m = stress_metrics(world, p, stress)
            if contam_eps is not None:
                s_m = {**s_m,
                       **contamination_metrics(world, p, contam_eps)}
            for t in ts:
                sat = satisfaction_mask(world, t)
                rows.append({"case": case, "world_seed": world.seed,
                             "target": kind, **info, "gated": gate,
                             "t": t, "t_star": t_star,
                             **dist_metrics(p, sat), **s_m})
        rep = hardness_report(world)
        meta_row = {"case": case, "seed": ws, "world_seed": world.seed,
                    "attempts": attempts, "gated": gate,
                    "is_hard": is_hard(world, rep),
                    **{f"probe_{k}": float(v) for k, v in rep.items()
                       if isinstance(v, (int, float))},
                    "beta_min": world.beta_min,
                    "t_star": t_star, "skipped_cells": n_skipped}
        if isinstance(kap, dict):
            meta_row.update({f"kappa_{k}": v for k, v in kap.items()})
        meta.append(meta_row)
    return rows, meta


def summarize(rows: list[dict]) -> dict:
    """Aggregate at each world's own challenge level t*: per-pole means
    and the best risk cell (by sat_mass) per world, averaged."""
    at_star = [r for r in rows if abs(r["t"] - r["t_star"]) < 1e-9]
    out = {}
    for kind in ("boltzmann", "worst", "probe_cell"):
        sub = [r for r in at_star if r["target"] == kind]
        out[kind] = {k: float(np.mean([r[k] for r in sub]))
                     for k in ("sat_mass", "eff_candidates",
                               "stress_mean", "stress_p05")}
    best = []
    for wsd in {r["world_seed"] for r in at_star}:
        cells = [r for r in at_star
                 if r["target"] == "risk" and r["world_seed"] == wsd]
        if cells:
            best.append(max(cells, key=lambda r: r["sat_mass"]))
    out["risk_best_cell"] = {k: float(np.mean([r[k] for r in best]))
                             for k in ("sat_mass", "eff_candidates",
                                       "stress_mean", "stress_p05")}
    return out
