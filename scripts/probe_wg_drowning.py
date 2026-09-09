"""w_g drowning-point marginal (operating manual §4/§5; exact, no
training). For each world, sweep w_g and price the standard O1 risk
grid at that composition: the *drowning point* is the smallest w_g at
which tv_on_off < the hardness margin for EVERY admissible risk cell;
beyond it the auxiliary objective drowns Ψ and risk-sensitive
aggregation is decoration for this world.

The family samples w_g in (0.1, 0.9); grid points above 0.9 are
out-of-family diagnostics and are marked as such.

Usage:
    python scripts/probe_wg_drowning.py --cases A B C D --worlds 4 \
        --world-seed0 0 --H 32 --out results/probes
"""

import argparse
import json

import numpy as np

from epgfn.o1 import o1_grid
from epgfn.runio import unique_run_dir, write_csv
from epgfn.worlds import WorldConfig, sample_world

WG_DEFAULT = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
WG_FAMILY_MAX = 0.9


def drowning_point(maxes: dict[float, float], margin: float):
    """Smallest probed w_g from which the whole-grid max tv_on_off
    stays below the margin at every larger probe too ("beyond it the
    signature is invisible"); None if the signature survives."""
    dp = None
    for wg in sorted(maxes):
        if maxes[wg] < margin:
            if dp is None:
                dp = wg
        else:
            dp = None
    return dp


def main() -> None:
    """Sweep w_g per world and locate the drowning point for each case.

    Parses CLI args, prices the O1 risk grid at each --wg-grid value
    for --worlds worlds per case, finds the smallest w_g beyond which
    the whole-grid max tv_on_off stays below --margin (the drowning
    point), and writes per-case CSVs plus a
    `wg_drowning_summary.json` to --out.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--world-seed0", type=int, default=0)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--wg-grid", type=float, nargs="+",
                    default=WG_DEFAULT)
    ap.add_argument("--beta-grid", type=float, nargs="+", default=None)
    ap.add_argument("--rho-grid", type=float, nargs="+", default=None)
    ap.add_argument("--margin", type=float, default=0.05,
                    help="hardness margin (fixed at 0.05)")
    ap.add_argument("--out", default="results/probes")
    args = ap.parse_args()

    out = unique_run_dir(args.out, args)
    print(f"run dir: {out}", flush=True)
    cfg = WorldConfig(H=args.H)
    beta_grid = (np.linspace(0.1, 1.0, 7) if args.beta_grid is None
                 else np.asarray(args.beta_grid))
    rho_grid = (np.linspace(0.0, 1.6, 7) if args.rho_grid is None
                else np.asarray(args.rho_grid))
    seeds = list(range(args.world_seed0, args.world_seed0 + args.worlds))

    summary = {"margin": args.margin, "wg_family_max": WG_FAMILY_MAX,
               "cases": {}}
    for case in args.cases:
        rows, per_world = [], {}
        for seed in seeds:
            world, _ = sample_world(case, cfg, seed)
            maxes = {}
            for wg in args.wg_grid:
                grid = o1_grid(world, beta_grid, rho_grid, w_g=float(wg))
                for r in grid:
                    r["in_family"] = wg <= WG_FAMILY_MAX
                rows.extend(grid)
                vals = [r["tv_on_off"] for r in grid
                        if np.isfinite(r.get("tv_on_off", np.nan))]
                maxes[float(wg)] = float(np.max(vals))
            dp = drowning_point(maxes, args.margin)
            per_world[str(world.seed)] = {
                "max_tv_on_off_by_wg": maxes, "drowning_point": dp,
                "drowns_in_family": (dp is not None
                                     and dp <= WG_FAMILY_MAX)}
            print(f"case {case} world {world.seed}: drowning point "
                  f"{dp if dp is not None else f'> {max(args.wg_grid)}'}",
                  flush=True)
        write_csv(out / f"wg_marginal_case{case}.csv", rows)
        dps = [v["drowning_point"] for v in per_world.values()]
        summary["cases"][case] = {
            "per_world": per_world,
            "n_worlds_drowning_in_family":
                sum(v["drowns_in_family"] for v in per_world.values()),
            "median_drowning_point":
                (float(np.median([d for d in dps if d is not None]))
                 if all(d is not None for d in dps) else None)}
    with open(out / "wg_drowning_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"summary -> {out}/wg_drowning_summary.json")


if __name__ == "__main__":
    main()
