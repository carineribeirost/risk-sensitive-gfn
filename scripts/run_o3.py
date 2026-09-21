"""O3 experiment: joint-satisfaction utility + weight-perturbation
stress test, comparing the risk-on family against its two poles
(Boltzmann and pure worst-case). Exact enumeration; no training.

Usage:
    python scripts/run_o3.py --cases A B C D --worlds 3 --H 16 \
        --out results/o3
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

from epgfn.baselines import KINDS as BASELINE_KINDS
from epgfn.o3 import run_o3, summarize
from epgfn.risk import BALL_RHO_GRIDS, GEOMETRIES
from epgfn.runio import Progress, unique_run_dir, write_csv
from epgfn.worlds import WorldConfig


def main() -> None:
    """Run the O3 joint-satisfaction/stress exact experiment for each case.

    Parses CLI args, runs `run_o3` over every (case, world seed) unit
    (in parallel when --jobs > 1, byte-identical to the sequential
    run), and writes per-case grid and world-metadata CSVs plus a
    summary JSON to --out, printing sat_mass/eff_candidates/stress
    stats for the boltzmann, worst, probe_cell, and risk_best_cell
    rows of each case.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--worlds", type=int, default=3)
    ap.add_argument("--world-seed0", type=int, default=1000)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--beta-grid", type=float, nargs="+", default=None)
    ap.add_argument("--rho-grid", type=float, nargs="+", default=None)
    ap.add_argument("--n-stress", type=int, default=200)
    ap.add_argument("--kappa", type=float, default=25.0,
                    help="scalar Dirichlet concentration; used only if "
                    "--kappa-range is disabled with two equal values")
    ap.add_argument("--kappa-range", type=float, nargs=2,
                    default=[8.0, 60.0],
                    help="per-origin kappa drawn log-uniformly per world")
    ap.add_argument("--contam-eps", type=float, default=0.2,
                    help="ε for the exact contamination worst case per "
                    "target row; <= 0 disables")
    ap.add_argument("--no-gate", action="store_true",
                    help="ablation: skip the hardness gate (first "
                    "world draw per seed; probes still recorded)")
    ap.add_argument("--d", type=int, default=2,
                    help="coordinates per point; |X| = H^d")
    ap.add_argument("--geometry", default="grid",
                    choices=["grid", "sequence"],
                    help="score-field family")
    ap.add_argument("--weight-alpha", type=float, default=2.0,
                    help="Dirichlet concentration for nominal weights; "
                    "2.0 = flat family (bitwise-identical to the old "
                    "default), chosen peaked family = 0.3")
    ap.add_argument("--ball", default="kl", choices=list(GEOMETRIES),
                    help="ambiguity-ball geometry; without "
                    "--rho-grid the fixed per-ball grid is used")
    ap.add_argument("--rho-out-grid", type=float, nargs="+",
                    default=None,
                    help="case D only: sweep the conditioned "
                    "outer radius; adds target kind risk_rho_out")
    ap.add_argument("--baselines", nargs="+", default=[],
                    choices=list(BASELINE_KINDS),
                    help="matched-conjunctivity aggregation baselines. "
                    "hard_min is the worst pole and is already "
                    "reported; it is offered here only so a run can "
                    "assert the identity.")
    ap.add_argument("--desirability-lo", type=float, default=0.50,
                    help="declared lower quantile of the Derringer-Suich "
                    "desirability ramp, per score column")
    ap.add_argument("--desirability-hi", type=float, default=0.95,
                    help="declared upper quantile of the ramp")
    ap.add_argument("--desirability-shape", type=float, default=1.0,
                    help="ramp exponent; 1.0 is linear")
    ap.add_argument("--skip-grid", action="store_true",
                    help="poles, probe cell and baselines only; skip "
                    "the beta-by-rho risk grid. Turns a baseline "
                    "comparison into minutes.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel (case, world) units. Every unit is "
                    "fully determined by its seed, so any value gives "
                    "byte-identical output")
    ap.add_argument("--out", default="results/o3",
                    help="base dir; each invocation gets a unique "
                    "timestamped subdirectory")
    args = ap.parse_args()

    out = unique_run_dir(args.out, args)
    print(f"run dir: {out}")
    cfg = WorldConfig(H=args.H, d=args.d, geometry=args.geometry,
                      weight_alpha=args.weight_alpha)
    seeds = list(range(args.world_seed0, args.world_seed0 + args.worlds))
    # radii are ball-specific: an explicit --rho-grid wins, otherwise
    # the fixed grid for the chosen ball (identical to the old
    # default at ball="kl")
    rho_grid = (args.rho_grid if args.rho_grid is not None
                else list(BALL_RHO_GRIDS[args.ball]))

    kr = tuple(args.kappa_range)
    if kr[0] == kr[1]:  # degenerate range → scalar kappa for all origins
        kr = None
    kw = dict(beta_grid=args.beta_grid, rho_grid=rho_grid,
              ball=args.ball, rho_out_grid=args.rho_out_grid,
              n_stress=args.n_stress, kappa=args.kappa, kappa_range=kr,
              contam_eps=(args.contam_eps
                          if args.contam_eps > 0 else None),
              gate=not args.no_gate,
              baselines=tuple(args.baselines),
              desirability_lo=args.desirability_lo,
              desirability_hi=args.desirability_hi,
              desirability_shape=args.desirability_shape,
              skip_grid=args.skip_grid)
    # (case, seed) units are independent (stress rng is per-world
    # seeded), so the pool equals the sequential run byte-for-byte
    units = [(case, s) for case in args.cases for s in seeds]
    prog = Progress(len(units), "o3")
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(run_o3, case, cfg, [s], **kw)
                    for case, s in units]
            unit_res = {}
            for u, f in zip(units, futs):
                unit_res[u] = f.result()
                prog.step(f"case {u[0]} world seed {u[1]}")
    else:
        unit_res = {}
        for u in units:
            unit_res[u] = run_o3(u[0], cfg, [u[1]], **kw)
            prog.step(f"case {u[0]} world seed {u[1]}")
    for case in args.cases:
        rows, meta = [], []
        for s in seeds:
            r, m = unit_res[(case, s)]
            rows.extend(r)
            meta.extend(m)
        write_csv(out / f"o3_case{case}.csv", rows)
        write_csv(out / f"o3_case{case}_worlds.csv", meta)
        summ = summarize(rows)
        with open(out / f"o3_case{case}_summary.json", "w") as fh:
            json.dump(summ, fh, indent=2)
        print(f"case {case} (at per-world challenge level t*):")
        for kind in ("boltzmann", "worst", "probe_cell",
                     "risk_best_cell"):
            s = summ[kind]
            print(f"  {kind:>15}: sat_mass={s['sat_mass']:.4f} "
                  f"eff_cands={s['eff_candidates']:.1f} "
                  f"stress_mean={s['stress_mean']:.4f} "
                  f"stress_p05={s['stress_p05']:.4f}")


if __name__ == "__main__":
    main()
