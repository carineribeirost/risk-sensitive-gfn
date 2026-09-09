"""Off-policy mix sweep. Loop TrainConfig.uniform_mix over
{0, 0.25, 0.5, 1.0} at the canonical config; the claim "O2 measures
amortization, not exploration" requires mix=0.5 ~= mix=1.0 and both
>= mix=0. Heldout L1 per mix with cluster CIs; floors reused per
world across mixes.

Usage:
    python scripts/run_mix.py --case A --worlds 4 --world-seed0 0 \
        --seeds 3 --H 32 --steps 12000 --out results/mix
"""

import argparse
import csv
import json

import numpy as np

from epgfn.cases import mean_mc_floor
from epgfn.runio import unique_run_dir
from epgfn.stats import cluster_bootstrap_ci
from epgfn.train import TrainConfig, default_device, ranges_for, train_policy
from epgfn.worlds import WorldConfig, sample_world


def main() -> None:
    """Run the off-policy uniform_mix sweep and write per-mix results.

    Parses CLI args, and for each of --worlds worlds trains --seeds
    policies at each of --mixes uniform_mix values (canonical config
    otherwise), writing per-run heldout L1 rows to
    `mix_case{case}.csv` and a cluster-bootstrap summary of heldout L1
    by mix to `mix_case{case}_summary.json`.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="A")
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--world-seed0", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--mixes", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--floor-samples", type=int, default=None)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--cond-pool", type=int, default=256)
    ap.add_argument("--logit-floor", type=float, default=-25.0)
    ap.add_argument("--out", default="results/mix")
    args = ap.parse_args()

    out = unique_run_dir(args.out, args)
    print(f"run dir: {out}", flush=True)
    cfg_w = WorldConfig(H=args.H)
    n_floor = args.floor_samples or 10 * args.H * args.H

    fields = ["case", "world", "gate_attempts", "train_seed", "mix",
              "heldout_l1", "mc_floor_l1", "loss", "wall_s"]
    rows = []
    groups = {}  # mix -> {world: [l1, ...]}
    with open(out / f"mix_case{args.case}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        fh.flush()
        for wi in range(args.worlds):
            world, attempts = sample_world(args.case, cfg_w,
                                           args.world_seed0 + wi)
            ranges = ranges_for(world)
            heldout = ranges.heldout_grid()
            floors = mean_mc_floor(world, heldout, n_floor)
            for mix in args.mixes:
                for seed in range(args.seeds):
                    cfg = TrainConfig(steps=args.steps, seed=seed,
                                      device=args.device,
                                      uniform_mix=float(mix),
                                      cond_pool=args.cond_pool or None,
                                      logit_floor=args.logit_floor or None)
                    _, _, hist = train_policy(world, cfg, ranges)
                    final = hist[-1]
                    row = {"case": args.case, "world": world.seed,
                           "gate_attempts": attempts, "train_seed": seed,
                           "mix": float(mix),
                           "heldout_l1": final["heldout_l1"],
                           "mc_floor_l1": floors,
                           "loss": final["loss"],
                           "wall_s": final["wall_s"]}
                    rows.append(row)
                    writer.writerow(row)
                    fh.flush()
                    groups.setdefault(float(mix), {}).setdefault(
                        world.seed, []).append(final["heldout_l1"])
                    print(f"world {world.seed} mix {mix} seed {seed}: "
                          f"L1={final['heldout_l1']:.4f} "
                          f"(floor {floors:.4f}) {final['wall_s']:.0f}s",
                          flush=True)

    summary = {"case": args.case,
               "mc_floor_mean": float(np.mean([r["mc_floor_l1"]
                                               for r in rows])),
               "by_mix": {}}
    for mix, g in sorted(groups.items()):
        pt, lo, hi = cluster_bootstrap_ci(
            {k: np.array(v) for k, v in g.items()})
        summary["by_mix"][str(mix)] = {"mean": pt, "ci_lo": lo,
                                       "ci_hi": hi}
    with open(out / f"mix_case{args.case}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
