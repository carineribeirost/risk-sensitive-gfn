"""Which knob absorbs which error? Exact, training-free, CPU-only.

The shipped O3 stress model perturbs only the WEIGHTS (populations), so
the score fields the CVaR tail is selected from are exact. Measured
consequence on the o3-pareto run: the radius that maximises stress_p05
sits at rho = 0 on A, C and D (median over 8 worlds) and at 0.05 on B,
and it does not track the size of the actual perturbation (Spearman
p >= 0.19 in every case, implied E[KL(nu||w)] spanning 0.037-0.216).
Reading under test: beta already absorbs population error, because tail
MEMBERSHIP is pinned by exact scores and only the tail WEIGHTING moves,
so the ambiguity ball insures against a dependence CVaR has flattened.

If that reading is right, the ball should start to pay once the scores
are themselves uncertain, which is the second error source in the
motivating setting (poorly minimised MD, independent runs for the two
states of a dimer). This script measures both error sources against all
three knobs on one grid.

Design
------
The designer sees nominal (w, a) and commits to a target p from a rule
cell (beta, rho, sigma). The truth is (nu, a'):

    nu ~ Dirichlet(kappa_o * w_o)          population error (as O3)
    a' = clip(a + N(0, s), 0, 1)           score error      (new)

and the yardstick is the realised PLAIN weighted mean under the truth,

    U = E_{x~p}[ sum_k nu_k a'_k(x) ],

summarised by its mean and 5th percentile over draws. The yardstick is
deliberately NOT the risk functional, so no knob is scored by its own
limiting case (unlike sat_mass, which is conjunctive and therefore
aligned with the beta -> 0 pole; it is reported alongside for reference).

Error models: "weights" (nu only), "scores" (a' only), "both". Draws are
shared across every rule cell within a world, so all comparisons are
paired; U depends only on (world, error model, s), never on the cell, so
it is built once and re-used across the whole grid.

Worlds, kappa draws and stress draws reproduce the o3-pareto run exactly
at its defaults (--world-seed0 0 --worlds 8 --H 32 --d 2), so the
sigma = 0, model = "weights" slice must reproduce that run's stress
columns cell for cell. That identity is the script's own control.

Constraint fields (C's vetoes, A's guard) are NOT perturbed: a veto is a
categorical filter, not a continuous score. Recorded as a choice.

Output is LONG format: one row per (case, world, beta, rho, sigma,
err_model, s_score). Skipped below-bound beta cells are counted in the
per-world meta CSV, never silently dropped.

Usage (20 cpus):
    python scripts/run_sigma_stress.py --jobs 20 --out results/sigma-stress
"""

import argparse
import dataclasses
import itertools
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from epgfn.cases import target_for
from epgfn.conditions import Condition, tied_risk
from epgfn.o3 import (challenge_level, contamination_metrics, draw_stress,
                      satisfaction_mask)
from epgfn.risk import BALL_RHO_GRIDS, GEOMETRIES
from epgfn.runio import Progress, unique_run_dir, write_csv
from epgfn.worlds import WorldConfig, hardness_report, is_hard, sample_world

ERR_MODELS = ("weights", "scores", "both")
_NOISY = {"weights": (True, False), "scores": (False, True),
          "both": (True, True)}


def score_blocks(world):
    """Objective score blocks, the ones carrying docking-score error."""
    if world.case == "A":
        return {"neutral": world.scores_neutral}
    if world.case == "B":
        return {"plus": world.scores_plus, "minus": world.scores_minus}
    if world.case == "C":
        return {"plus": world.scores_plus}
    return {"nested": world.scores_nested}


def lightest_weight(world):
    """Smallest nominal weight anywhere in the world: how light the
    lightest medoid actually is.

    NOT min(world.beta_bounds.values()). For case D that carries the
    shared-beta validity bound w_inner.min(axis=1).max(), a max over
    origins, which overstates the lightest inner weight by up to three
    orders of magnitude on peaked draws (alpha = 0.3, seed 3: 1.3e-2
    against a true 1.0e-5). A/B/C bounds happen to coincide with their
    set minima; D does not.
    """
    if world.case == "A":
        return float(world.w_neutral.min())
    if world.case == "B":
        return float(min(world.p_plus.min(), world.q_minus.min()))
    if world.case == "C":
        return float(world.p_plus.min())
    if world.case == "D":
        return float(min(world.w_inner.min(), world.pi_outer.min()))
    raise ValueError(f"unknown case {world.case!r}")


def utility_matrix(world, stress, model, s_score, rng, n_draws):
    """(N, S) realised plain-weighted-mean utility per stress draw."""
    w_on, a_on = _NOISY[model]
    blocks = score_blocks(world)
    pert = {}
    for name, arr in blocks.items():
        if a_on and s_score > 0.0:
            eps = rng.normal(0.0, s_score, size=(n_draws,) + arr.shape)
            pert[name] = np.clip(arr[None] + eps, 0.0, 1.0)
        else:
            pert[name] = arr[None]          # broadcasts over draws

    def wts(key, nominal):
        if w_on:
            return stress[key]
        return np.asarray(nominal, dtype=float)[None]

    if world.case == "A":
        return np.einsum("snk,sk->ns", pert["neutral"],
                         wts("nu", world.w_neutral))
    if world.case == "B":
        return (np.einsum("snk,sk->ns", pert["plus"],
                          wts("nu_p", world.p_plus))
                - world.cfg.gamma * np.einsum("snk,sk->ns", pert["minus"],
                                              wts("nu_q", world.q_minus)))
    if world.case == "C":
        # veto mask: a state failing ANY named threshold contributes
        # zero realised utility, regardless of its plus-score weight
        ok = ~np.any(world.scores_named >= world.c_named, axis=-1)
        return np.einsum("snk,sk->ns", pert["plus"],
                         wts("nu_p", world.p_plus)) * ok[:, None]
    if world.case == "D":
        nu_in = (stress["nu_in"] if w_on else world.w_inner[None])
        inner = np.einsum("snok,sok->sno", pert["nested"], nu_in)
        return np.einsum("sno,so->ns", inner, wts("pi", world.pi_outer))
    raise ValueError(f"unknown case {world.case!r}")


def run_unit(case, cfg, ws, betas, rhos, sigmas, s_scores, n_stress,
             kappa_range, beta_t, w_g, gate, ball, contam_eps):
    """One (case, world) unit. Fully determined by its seed, so the pool
    equals the sequential run byte-for-byte."""
    world, attempts = sample_world(case, cfg, ws, gate=gate)
    # identical stream to run_o3: kappa THEN the stress draws
    rng = np.random.default_rng(10_000 + ws)
    lo, hi = kappa_range
    kap = {k: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
           for k in ("n", "p", "m", "in", "out")}
    stress = draw_stress(world, rng, n_stress, kap)

    # U depends on (model, s_score) only, never on the rule cell
    # separate seed stream (offset 777) so score-error draws stay
    # independent of the weight-stress draws from `rng` above
    noise_rng = np.random.default_rng((777, ws))
    U = {}
    for s in s_scores:
        for m in ERR_MODELS:
            # fresh substream per (s_score, err_model) combo, still
            # fully determined by (ws, s, m) draw order
            sub = np.random.default_rng(noise_rng.integers(1 << 31))
            U[(m, s)] = utility_matrix(world, stress, m, s, sub, n_stress)

    t_star = challenge_level(world)
    sat = satisfaction_mask(world, t_star)
    rows, n_skipped = [], 0
    for b, r, sg in itertools.product(betas, rhos, sigmas):
        if b < world.beta_min:
            n_skipped += 1
            continue
        risk = dataclasses.replace(tied_risk(case, float(b), float(r)),
                                   sigma=float(sg), geometry=ball)
        p = target_for(world, Condition(beta_t, w_g, risk))
        mass = float(p[sat].sum())
        if mass > 0.0:
            q = p[sat] / mass
            eff = float(np.exp(-(q * np.log(np.maximum(q, 1e-300))).sum()))
        else:
            eff = 0.0
        # rare-state probe: put eps mass on ONE state, worst over states
        # and origins. Out-of-Dirichlet, and the direct analogue of "the
        # light medoid turned out to matter" (weights only, exact scores).
        contam = ({} if contam_eps is None
                  else {"contam_worst":
                        contamination_metrics(world, p,
                                              contam_eps)["contam_worst"]})
        for s in s_scores:
            for m in ERR_MODELS:
                realised = p @ U[(m, s)]
                rows.append({
                    "case": case, "world_seed": int(world.seed),
                    "beta_cvar": float(b), "rho": float(r),
                    "sigma": float(sg), "err_model": m,
                    "s_score": float(s), "t": t_star, "t_star": t_star,
                    "sat_mass": mass, "eff_candidates": eff,
                    "ball": ball,
                    "weight_alpha": world.cfg.weight_alpha,
                    "w_min": lightest_weight(world),
                    "stress_mean": float(realised.mean()),
                    "stress_p05": float(np.quantile(realised, 0.05)),
                    **contam})
    rep = hardness_report(world)
    meta = {"case": case, "seed": ws, "world_seed": int(world.seed),
            "attempts": attempts, "gated": gate,
            "is_hard": is_hard(world, rep),
            **{f"probe_{k}": float(v) for k, v in rep.items()
               if isinstance(v, (int, float))},
            "beta_min": world.beta_min, "t_star": t_star,
            "skipped_cells": n_skipped,
            **{f"kappa_{k}": v for k, v in kap.items()}}
    return rows, meta


def main() -> None:
    """Run the weight-vs-score error-absorption sweep for each case.

    Parses CLI args, runs `run_unit` over every (case, world seed)
    unit (in parallel when --jobs > 1, byte-identical to the
    sequential run), and writes per-case long-format rule-cell CSVs
    and world-metadata CSVs to --out, plus a summary JSON of the
    (beta, rho, sigma) cell maximising stress_p05 per (error model,
    score-error level).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--world-seed0", type=int, default=0,
                    help="0 reproduces the o3-pareto worlds exactly")
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--geometry", default="grid",
                    choices=["grid", "sequence"])
    ap.add_argument("--beta-grid", type=float, nargs="+",
                    default=[0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                             0.8, 0.9, 1.0])
    ap.add_argument("--ball", default="kl", choices=list(GEOMETRIES),
                    help="ambiguity-ball geometry. KL cannot inflate "
                    "a near-zero-weight state (the entropic cost "
                    "diverges); TV can, to w_k + rho, regardless of w_k; "
                    "modified-chi2 is worse than KL there, its 1/w "
                    "penalty explodes. Radii are NOT comparable across "
                    "balls, so without --rho-grid the fixed "
                    "per-ball grid is used")
    ap.add_argument("--rho-grid", type=float, nargs="+", default=None,
                    help="explicit radii; default = the fixed grid "
                    "for --ball (risk.BALL_RHO_GRIDS)")
    ap.add_argument("--contam-eps", type=float, default=0.2,
                    help="eps for the rare-state contamination probe "
                    "(mass eps onto ONE state, worst over states); "
                    "<= 0 disables the column")
    ap.add_argument("--sigma-grid", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.2],
                    help="the DESIGNER's score-robustness margin "
                    "(ConditionRanges.sigma declared range)")
    ap.add_argument("--s-score", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.2],
                    help="sd of the TRUE score error; fields live in "
                    "[0,1]. 0.0 is the no-score-error anchor")
    ap.add_argument("--n-stress", type=int, default=500)
    ap.add_argument("--kappa-range", type=float, nargs=2,
                    default=[8.0, 60.0])
    ap.add_argument("--beta-t", type=float, default=4.0)
    ap.add_argument("--w-g", type=float, default=0.3)
    ap.add_argument("--weight-alpha", type=float, default=2.0)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel (case, world) units; any value gives "
                    "byte-identical output")
    ap.add_argument("--out", default="results/sigma-stress")
    args = ap.parse_args()

    out = unique_run_dir(args.out, args)
    print(f"run dir: {out}")
    cfg = WorldConfig(H=args.H, d=args.d, geometry=args.geometry,
                      weight_alpha=args.weight_alpha)
    seeds = list(range(args.world_seed0, args.world_seed0 + args.worlds))
    # radii are ball-specific: an explicit grid wins, else the fixed
    # one for the chosen ball (a KL, TV and chi2 rho are not comparable)
    rho_grid = (args.rho_grid if args.rho_grid is not None
                else list(BALL_RHO_GRIDS[args.ball]))
    kw = dict(betas=args.beta_grid, rhos=rho_grid,
              sigmas=args.sigma_grid, s_scores=args.s_score,
              n_stress=args.n_stress,
              kappa_range=tuple(args.kappa_range), beta_t=args.beta_t,
              w_g=args.w_g, gate=not args.no_gate, ball=args.ball,
              contam_eps=(args.contam_eps if args.contam_eps > 0
                          else None))
    n_cells = (len(args.beta_grid) * len(rho_grid)
               * len(args.sigma_grid))
    print(f"ball={args.ball} rho grid={rho_grid} "
          f"weight_alpha={args.weight_alpha}")
    print(f"{n_cells} rule cells x {len(args.s_score)} score-error "
          f"levels x {len(ERR_MODELS)} error models per world")

    units = [(case, s) for case in args.cases for s in seeds]
    prog = Progress(len(units), "sigma-stress")
    res = {}
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_unit, c, cfg, s, **kw): (c, s)
                    for c, s in units}
            for f, u in futs.items():
                res[u] = f.result()
                prog.step(f"case {u[0]} world seed {u[1]}")
    else:
        for u in units:
            res[u] = run_unit(u[0], cfg, u[1], **kw)
            prog.step(f"case {u[0]} world seed {u[1]}")

    for case in args.cases:
        rows, meta = [], []
        for s in seeds:
            r, m = res[(case, s)]
            rows.extend(r)
            meta.append(m)
        write_csv(out / f"sigstress_case{case}.csv", rows)
        write_csv(out / f"sigstress_case{case}_worlds.csv", meta)
        # headline: per (err_model, s_score), the rule cell maximising
        # stress_p05, averaged over worlds within the cell
        summ = {}
        for m in ERR_MODELS:
            for s in args.s_score:
                sub = [r for r in rows
                       if r["err_model"] == m and r["s_score"] == s]
                by = {}
                for r in sub:
                    by.setdefault((r["beta_cvar"], r["rho"], r["sigma"]),
                                  []).append(r["stress_p05"])
                # only cells present in EVERY world are comparable
                full = {k: v for k, v in by.items() if len(v) == len(seeds)}
                if not full:
                    continue
                best = max(full, key=lambda k: float(np.mean(full[k])))
                summ[f"{m}|s={s}"] = {
                    "beta": best[0], "rho": best[1], "sigma": best[2],
                    "stress_p05": float(np.mean(full[best])),
                    "n_cells_full_coverage": len(full)}
        with open(out / f"sigstress_case{case}_summary.json", "w") as fh:
            json.dump(summ, fh, indent=2)
        print(f"\ncase {case}: best (beta, rho, sigma) for stress_p05")
        for k, v in summ.items():
            print(f"  {k:>18}: beta={v['beta']:<5} rho={v['rho']:<5} "
                  f"sigma={v['sigma']:<5} p05={v['stress_p05']:.4f}")


if __name__ == "__main__":
    main()
