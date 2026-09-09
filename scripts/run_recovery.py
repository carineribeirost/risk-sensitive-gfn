"""Paper-2 core experiment (EXACT, no training): corruption →
recovery → fallback.

For each (case, world, model, level, condition, arm):
- naive:      p* from CORRUPTED scores, σ = 0
- robust(σ):  p* from CORRUPTED scores with the σ pre-shift
- winsor:     p* from CORRUPTED scores winsorized (k=1), σ = 0
- crude:      p* from the quantized-TRUE-score fallback oracle, σ = 0
all scored on the TRUE world: primary = mass on truly-satisfying
states at the world's challenge level; secondary = TV to the true
target. The condition set is the o3-style tied (β, ρ) grid with
below-bound cells skipped and reported (no silent caps).

Usage (run machine):
    python scripts/run_recovery.py --cases A B C D --worlds 4 \
        --world-seed0 0 --H 32 --models crash bias heavy \
        --levels 0.01 0.05 0.1 0.2 --out results/p2-recovery
"""

import argparse
import copy
import csv
import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor

from epgfn.cases import target_for
from epgfn.conditions import Condition, tied_risk
from epgfn.corrupt import MODELS, corrupt_world, crude_world
from epgfn.o3 import challenge_level, satisfaction_mask
from epgfn.risk import BALL_RHO_GRIDS, GEOMETRIES, winsorize_scores
from epgfn.runio import Progress, unique_run_dir
from epgfn.target import tv
from epgfn.worlds import WorldConfig, sample_world

SIGMAS = (0.0, 0.05, 0.1, 0.15, 0.2)   # fixed σ grid
BETA_GRID = (0.15, 0.3, 0.6, 1.0)      # o3 defaults
# per-ball rho grids: risk.BALL_RHO_GRIDS (kl = the original grid)


def winsorized_copy(world, k: int = 1):
    """Return a deep copy of `world` with every score array winsorized.

    Args:
        world: world whose score arrays (`scores_neutral`,
            `scores_plus`, `scores_minus`, `scores_named`,
            `scores_nested`) are winsorized in place on the copy;
            arrays that are None are skipped, and Case D's 3-D nested
            array is winsorized per origin (axis 1) rather than
            flattened.
        k: number of extreme values clipped at each end
            (`winsorize_scores` parameter).

    Returns:
        The winsorized deep copy; `world` itself is untouched.
    """
    w = copy.deepcopy(world)
    for attr in ("scores_neutral", "scores_plus", "scores_minus",
                 "scores_named", "scores_nested"):
        arr = getattr(w, attr)
        if arr is None:
            continue
        if arr.ndim == 3:  # D nested: winsorize within each origin's set
            for o in range(arr.shape[1]):
                arr[:, o, :] = winsorize_scores(arr[:, o, :], k)
        else:
            arr[...] = winsorize_scores(arr, k)
    return w


def recovery_unit(case, wseed, cfg_w, ball, rho_grid, models, levels,
                  winsor_k):
    """All rows for one (case, world) unit, in the exact order the
    former sequential loop produced them: units are fully determined
    by (case, wseed), so pooling them is byte-identical to the
    sequential run."""
    world, _ = sample_world(case, cfg_w, wseed)
    t_star = challenge_level(world)
    sat = satisfaction_mask(world, t_star)
    conds, n_skip = [], 0
    for b in BETA_GRID:
        if b < world.beta_min:
            n_skip += len(rho_grid)
            continue
        for r in rho_grid:
            conds.append((b, r))

    def risk_for(b, r, sigma=0.0):
        risk = tied_risk(case, b, r)
        if ball != "kl":  # "kl" stays bytewise
            risk = dataclasses.replace(risk, geometry=ball)
        if sigma:
            risk = dataclasses.replace(risk, sigma=sigma)
        return risk

    truth = {(b, r): target_for(
        world, Condition(4.0, 0.3, risk_for(b, r)))
        for (b, r) in conds}
    rows = []

    def eval_arm(w_scores, arm, model, level, sigma=0.0):
        for (b, r) in conds:
            p = target_for(w_scores,
                           Condition(4.0, 0.3, risk_for(b, r, sigma)))
            rows.append({"case": case, "world": world.seed,
                         "model": model, "level": level,
                         "beta": b, "rho": r, "arm": arm,
                         "sigma": sigma, "ball": ball,
                         "weight_alpha": cfg_w.weight_alpha,
                         "sat_mass_true": float(p[sat].sum()),
                         "tv_true": float(tv(p, truth[(b, r)])),
                         "n_skipped": n_skip})

    # true-oracle reference + crude fallback (model-indep.)
    eval_arm(world, "true", "none", 0.0)
    eval_arm(crude_world(world), "crude", "none", 0.0)
    for model in models:
        # only "crash" sweeps --levels; bias/heavy use one fixed
        # severity and are recorded at level 0
        lvls = levels if model == "crash" else [0.0]
        for lv in lvls:
            cw = corrupt_world(world, model, lv)
            eval_arm(cw, "naive", model, lv, sigma=0.0)
            for s in SIGMAS[1:]:
                eval_arm(cw, "robust", model, lv, sigma=s)
            eval_arm(winsorized_copy(cw, winsor_k), "winsor", model, lv)
    msg = (f"{case} world {world.seed}: {len(conds)} conds "
           f"({n_skip} below-bound skipped), "
           f"t*={t_star:.2f}, |sat|={int(sat.sum())}")
    return rows, msg


def main() -> None:
    """Run the corruption/recovery/fallback exact experiment for each case.

    Parses CLI args, runs `recovery_unit` over every (case, world)
    unit (in parallel when --jobs > 1, in submission order so a
    killed run keeps completed units), and writes one long-format
    `recovery.csv` with one row per (case, world, model, level,
    condition, arm) scored against the true target.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--worlds", type=int, default=8,
                    help="8 (not 4): the primary sign permutation "
                    "needs >=8 clusters for p<0.05 to be reachable "
                    "after Holm over the sigma grid; exact-layer cost "
                    "makes 8 cheap")
    ap.add_argument("--world-seed0", type=int, default=0)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--geometry", default="grid",
                    choices=["grid", "sequence"])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--levels", type=float, nargs="+",
                    default=[0.01, 0.05, 0.1, 0.2],
                    help="crash eps_s grid; bias/heavy use fixed "
                    "constants and run once (level recorded as 0)")
    ap.add_argument("--winsor-k", type=int, default=1)
    ap.add_argument("--weight-alpha", type=float, default=2.0,
                    help="Dirichlet concentration for nominal weights; "
                    "2.0 = flat family (bitwise-identical to the old "
                    "default), chosen peaked family = 0.3")
    ap.add_argument("--ball", default="kl", choices=list(GEOMETRIES),
                    help="ambiguity-ball geometry for every risk "
                    "block (geometry ablation); without --rho-grid "
                    "the fixed per-ball grid is used")
    ap.add_argument("--rho-grid", type=float, nargs="+", default=None)
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel (case, world) units. Every unit is "
                    "fully determined by its seed, so any value gives "
                    "byte-identical output")
    ap.add_argument("--out", default="results/p2-recovery")
    args = ap.parse_args()

    out = unique_run_dir(args.out, args)
    print(f"run dir: {out}", flush=True)
    cfg_w = WorldConfig(H=args.H, d=args.d, geometry=args.geometry,
                        weight_alpha=args.weight_alpha)
    rho_grid = (args.rho_grid if args.rho_grid is not None
                else list(BALL_RHO_GRIDS[args.ball]))

    fields = ["case", "world", "model", "level", "beta", "rho", "arm",
              "sigma", "ball", "weight_alpha", "sat_mass_true",
              "tv_true", "n_skipped"]
    units = [(case, args.world_seed0 + wi)
             for case in args.cases for wi in range(args.worlds)]
    uargs = (cfg_w, args.ball, rho_grid, args.models, args.levels,
             args.winsor_k)
    with open(out / "recovery.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        fh.flush()

        prog = Progress(len(units), "recovery")

        def emit(rows, msg):
            # rows land in submission order (case-major, then world),
            # flushed per unit; a killed run keeps completed units
            writer.writerows(rows)
            fh.flush()
            prog.step(msg)

        if args.jobs > 1:
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futs = [ex.submit(recovery_unit, case, ws, *uargs)
                        for case, ws in units]
                for f in futs:
                    emit(*f.result())
        else:
            for case, ws in units:
                emit(*recovery_unit(case, ws, *uargs))
    print(json.dumps({"done": True, "out": str(out)}))


if __name__ == "__main__":
    main()
