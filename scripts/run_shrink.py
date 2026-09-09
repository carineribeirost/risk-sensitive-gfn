"""Interpolation vs extrapolation. Train with SHRUNK condition
ranges (`ConditionRanges.shrink(factor)`), evaluate on the FULL-range
held-out grid, and report L1 separately for `interp` conditions (all
features inside the shrunk box, i.e. within [-1, 1] under the
training-ranges normalization) and `extrap` conditions (any feature
outside). Never pooled.

Usage:
    python scripts/run_shrink.py --case B --worlds 4 --world-seed0 0 \
        --seeds 3 --H 32 --steps 12000 --factor 0.8 --out results/shrink
"""

import argparse
import csv
import json

import numpy as np

from epgfn.cases import target_for
from epgfn.runio import unique_run_dir
from epgfn.stats import cluster_bootstrap_ci
from epgfn.target import mc_floor
from epgfn.train import (TrainConfig, default_device, evaluate, ranges_for,
                         train_policy)
from epgfn.worlds import WorldConfig, sample_world


def main() -> None:
    """Run the interpolation-vs-extrapolation shrunk-range experiment.

    Parses CLI args, and for each of --worlds worlds trains --seeds
    policies on the --factor-shrunk condition ranges, evaluates exact
    L1 on the full held-out grid split into `interp` (inside the
    shrunk box) and `extrap` (outside) subsets, and writes per-run
    rows to `shrink_case{case}.csv` plus a cluster-bootstrap summary
    (kept separate, never pooled) to `shrink_case{case}_summary.json`.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="B")
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--world-seed0", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--factor", type=float, default=0.8,
                    help="range-width shrink factor")
    ap.add_argument("--floor-samples", type=int, default=None)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--cond-pool", type=int, default=256)
    ap.add_argument("--logit-floor", type=float, default=-25.0)
    ap.add_argument("--out", default="results/shrink")
    args = ap.parse_args()

    out = unique_run_dir(args.out, args)
    print(f"run dir: {out}", flush=True)
    cfg_w = WorldConfig(H=args.H)
    n_floor = args.floor_samples or 10 * args.H * args.H

    fields = ["case", "world", "gate_attempts", "train_seed", "factor",
              "n_interp", "n_extrap", "l1_interp", "l1_extrap",
              "floor_interp", "floor_extrap", "loss", "wall_s"]
    rows = []
    gi, ge = {}, {}
    with open(out / f"shrink_case{args.case}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        fh.flush()
        for wi in range(args.worlds):
            world, attempts = sample_world(args.case, cfg_w,
                                           args.world_seed0 + wi)
            full = ranges_for(world)
            shr = full.shrink(args.factor)  # after bounds are set
            heldout = full.heldout_grid()
            feats = shr.features(heldout)   # training-ranges normalization
            extrap = np.abs(feats).max(axis=1) > 1.0 + 1e-9
            idx_i = [i for i, e in enumerate(extrap) if not e]
            idx_e = [i for i, e in enumerate(extrap) if e]
            targets = [target_for(world, c) for c in heldout]
            floor_rng = np.random.default_rng(world.seed)
            floors = np.array([mc_floor(t, n_floor, floor_rng)
                               for t in targets])
            print(f"world {world.seed}: interp {len(idx_i)} / "
                  f"extrap {len(idx_e)} of {len(heldout)}", flush=True)
            for seed in range(args.seeds):
                cfg = TrainConfig(steps=args.steps, seed=seed,
                                  device=args.device,
                                  cond_pool=args.cond_pool or None,
                                  logit_floor=args.logit_floor or None)
                policy, _, hist = train_policy(world, cfg, shr,
                                               eval_ranges=full)
                final = hist[-1]
                l1_i = (evaluate(policy, shr, [heldout[i] for i in idx_i],
                                 [targets[i] for i in idx_i], args.device)
                        if idx_i else float("nan"))
                l1_e = (evaluate(policy, shr, [heldout[i] for i in idx_e],
                                 [targets[i] for i in idx_e], args.device)
                        if idx_e else float("nan"))
                row = {"case": args.case, "world": world.seed,
                       "gate_attempts": attempts, "train_seed": seed,
                       "factor": args.factor,
                       "n_interp": len(idx_i), "n_extrap": len(idx_e),
                       "l1_interp": l1_i, "l1_extrap": l1_e,
                       "floor_interp": float(floors[idx_i].mean())
                       if idx_i else float("nan"),
                       "floor_extrap": float(floors[idx_e].mean())
                       if idx_e else float("nan"),
                       "loss": final["loss"], "wall_s": final["wall_s"]}
                rows.append(row)
                writer.writerow(row)
                fh.flush()
                gi.setdefault(world.seed, []).append(l1_i)
                ge.setdefault(world.seed, []).append(l1_e)
                print(f"world {world.seed} seed {seed}: "
                      f"interp L1={l1_i:.4f} extrap L1={l1_e:.4f} "
                      f"({final['wall_s']:.0f}s)", flush=True)

    def ci(groups):
        pt, lo, hi = cluster_bootstrap_ci(
            {k: np.array(v) for k, v in groups.items()})
        return {"mean": pt, "ci_lo": lo, "ci_hi": hi}

    summary = {"case": args.case, "factor": args.factor,
               "interp": ci(gi), "extrap": ci(ge),
               "n_interp": rows[0]["n_interp"],
               "n_extrap": rows[0]["n_extrap"],
               "floor_interp_mean": float(np.nanmean(
                   [r["floor_interp"] for r in rows])),
               "floor_extrap_mean": float(np.nanmean(
                   [r["floor_extrap"] for r in rows]))}
    with open(out / f"shrink_case{args.case}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
