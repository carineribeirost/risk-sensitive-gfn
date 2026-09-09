"""O1 experiment: exact-target sweeps over (β_cvar, ρ) per
case, on test worlds, with the hardness-gate metadata reported.

Usage:
    python scripts/run_o1.py --cases A B C D --worlds 8 --H 32 \
        --out results/o1
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from epgfn.o1 import run_o1
from epgfn.risk import BALL_RHO_GRIDS, GEOMETRIES
from epgfn.runio import Progress, unique_run_dir, write_csv
from epgfn.worlds import WorldConfig


def main() -> None:
    """Run the O1 exact-target (beta_cvar, rho) sweep for each case.

    Parses CLI args, runs `run_o1` over every (case, world seed) unit
    (in parallel when --jobs > 1, byte-identical to the sequential run
    since each unit is fully determined by its seed), and writes
    per-case grid, world-metadata, separability, and rho_out CSVs to
    --out, printing a per-case summary of gate attempts and
    grid/separability TV.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--worlds", type=int, default=8,
                    help="number of test worlds per case")
    ap.add_argument("--world-seed0", type=int, default=1000,
                    help="first world seed (dev/test split lives here: "
                    "use disjoint seed blocks for dev and test)")
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--d", type=int, default=2,
                    help="coordinates per point; |X| = H^d")
    ap.add_argument("--geometry", default="grid",
                    choices=["grid", "sequence"],
                    help="score-field family")
    ap.add_argument("--beta-grid", type=float, nargs="+", default=None)
    ap.add_argument("--rho-grid", type=float, nargs="+", default=None)
    ap.add_argument("--no-gate", action="store_true",
                    help="ablation: skip the hardness gate (first "
                    "world draw per seed; probes still recorded)")
    ap.add_argument("--weight-alpha", type=float, default=2.0,
                    help="Dirichlet concentration for nominal weights; "
                    "2.0 = flat family (bitwise-identical to the old "
                    "default), chosen peaked family = 0.3")
    ap.add_argument("--ball", default="kl", choices=list(GEOMETRIES),
                    help="ambiguity-ball geometry (atlas arm); "
                    "without --rho-grid the fixed per-ball grid is "
                    "used")
    ap.add_argument("--k-guard", type=int, default=0,
                    help="guarded template A: number of guard "
                    "states (0 = off, bitwise-identical to the old "
                    "default; chosen battery value 2)")
    ap.add_argument("--rho-out-grid", type=float, nargs="+",
                    default=None,
                    help="case D only: sweep the conditioned "
                    "outer radius; liveness rows go to "
                    "o1_caseD_rhoout.csv")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel (case, world) units. Every unit is "
                    "fully determined by its seed, so any value gives "
                    "byte-identical output")
    ap.add_argument("--out", default="results/o1",
                    help="base dir; each invocation gets a unique "
                    "timestamped subdirectory")
    args = ap.parse_args()

    out = unique_run_dir(args.out, args)
    print(f"run dir: {out}")
    cfg = WorldConfig(H=args.H, d=args.d, geometry=args.geometry,
                      weight_alpha=args.weight_alpha,
                      k_guard=args.k_guard)
    seeds = list(range(args.world_seed0, args.world_seed0 + args.worlds))
    # radii are ball-specific: an explicit --rho-grid wins, otherwise
    # the fixed per-ball grid, except at ball="kl" with no
    # override, where the o1 default (7-point linspace) is kept so
    # old invocations stay identical
    rho_grid = args.rho_grid
    if rho_grid is None and args.ball != "kl":
        rho_grid = list(BALL_RHO_GRIDS[args.ball])

    # (case, seed) units are independent and seed-determined, so the
    # pool result equals the sequential run byte-for-byte; results are
    # merged in submission order (case-major, then seed)
    units = [(case, s) for case in args.cases for s in seeds]
    kw = dict(beta_grid=args.beta_grid, rho_grid=rho_grid,
              gate=not args.no_gate, ball=args.ball,
              rho_out_grid=args.rho_out_grid)
    prog = Progress(len(units), "o1")
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(run_o1, case, cfg, [s], **kw)
                    for case, s in units]
            unit_res = {}
            for u, f in zip(units, futs):
                unit_res[u] = f.result()
                prog.step(f"case {u[0]} world seed {u[1]}")
    else:
        unit_res = {}
        for u in units:
            unit_res[u] = run_o1(u[0], cfg, [u[1]], **kw)
            prog.step(f"case {u[0]} world seed {u[1]}")

    for case in args.cases:
        rows, sep_rows, meta = [], [], []
        for s in seeds:
            r, sp, m = unit_res[(case, s)]
            rows.extend(r)
            sep_rows.extend(sp)
            meta.extend(m)
        ro_rows = [r for r in sep_rows if r.get("probe") == "rho_out"]
        sep_rows = [r for r in sep_rows if r.get("probe") != "rho_out"]
        write_csv(out / f"o1_case{case}.csv", rows)
        write_csv(out / f"o1_case{case}_worlds.csv", meta)
        if ro_rows:  # liveness rows, separate file so the
            #          separability verdict is untouched
            write_csv(out / f"o1_case{case}_rhoout.csv", ro_rows)
            ro = [r["tv_to_shared_rho_out"] for r in ro_rows]
            print(f"case {case}: rho_out liveness median "
                  f"{np.median(ro):.4f} max {np.max(ro):.4f}")
        ok = [r["tv_on_off"] for r in rows
              if not np.isnan(r.get("tv_on_off", np.nan))]
        att = [m["attempts"] for m in meta]
        msg = (f"case {case}: {len(meta)} worlds "
               f"(gate attempts mean {np.mean(att):.1f}), "
               f"grid TV median {np.median(ok):.4f} "
               f"max {np.max(ok):.4f}")
        if sep_rows:  # O1b; case A has one origin
            write_csv(out / f"o1_case{case}_sep.csv", sep_rows)
            sep = [r["tv_to_shared"] for r in sep_rows]
            msg += (f", sep median {np.median(sep):.4f} "
                    f"max {np.max(sep):.4f}")
        print(msg + f" -> {out}/o1_case{case}.csv")


if __name__ == "__main__":
    main()
