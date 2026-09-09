"""O2 experiment: train the conditional policy per
(world, seed) pair, evaluate exact L1 on held-out conditions against
the Monte-Carlo finite-sample floor, and report cluster-bootstrap CIs
with worlds as the outer resampling unit.

Usage:
    python scripts/run_o2.py --case A --worlds 4 --seeds 3 --H 32 \
        --steps 4000 --out results/o2
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
    """Run the O2 train-and-evaluate experiment for one case.

    Parses CLI args, and for each of --worlds worlds trains --seeds
    policies (canonical config, with any requested feature-axis
    ablations applied to the condition ranges), evaluating exact
    heldout L1 against the Monte-Carlo floor; writes per-run rows to
    `o2_case{case}.csv` and a cluster-bootstrap summary to
    `o2_case{case}_summary.json`.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="A")
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--world-seed0", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=3,
                    help="training seeds nested per world")
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--d", type=int, default=2,
                    help="coordinates per point; |X| = H^d")
    ap.add_argument("--geometry", default="grid",
                    choices=["grid", "sequence"],
                    help="score-field family")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--floor-samples", type=int, default=None,
                    help="n for the MC floor; default 10*|X|")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--cond-pool", type=int, default=256,
                    help="training-condition pool size (full-grid reward "
                    "cache); 0 = stream fresh conditions (legacy)")
    ap.add_argument("--logit-floor", type=float, default=-25.0,
                    help="training-only clamp on target logits "
                    "beta_t*log R; 0 = off (ablation arm)")
    ap.add_argument("--loss", default="tb", choices=["tb", "subtb"],
                    help="training loss; subtb = SubTB(lambda=0.9) "
                    "credit-assignment arm (length axis)")
    ap.add_argument("--tie-beta", action="store_true",
                    help="case B only: one beta for both sides "
                    "(intrinsic dim 5; features stay 6-D)")
    ap.add_argument("--use-sigma", action="store_true",
                    help="paper 2: sigma score-robustness condition "
                    "axis (+1 feature, sigma in heldout grid)")
    ap.add_argument("--use-rho-out", action="store_true",
                    help="case D only: conditioned outer radius "
                    "rho_out (+1 feature, 2-point axis in the "
                    "heldout grid)")
    ap.add_argument("--weight-alpha", type=float, default=2.0,
                    help="Dirichlet concentration for nominal weights; "
                    "2.0 = flat family (bitwise-identical to the old "
                    "default), chosen peaked family = 0.3")
    ap.add_argument("--k-guard", type=int, default=0,
                    help="guarded template A: number of guard "
                    "states (0 = off, bitwise-identical to the old "
                    "default; chosen battery value 2)")
    ap.add_argument("--use-guard-delta", action="store_true",
                    help="case A with a guard: condition the "
                    "guard veto margin delta (+1 feature, 3-point "
                    "heldout axis)")
    ap.add_argument("--out", default="results/o2",
                    help="base dir; each invocation gets a unique "
                    "timestamped subdirectory")
    args = ap.parse_args()

    out = unique_run_dir(args.out, args)
    print(f"run dir: {out}", flush=True)
    cfg_w = WorldConfig(H=args.H, d=args.d, geometry=args.geometry,
                        weight_alpha=args.weight_alpha,
                        k_guard=args.k_guard)
    n_floor = args.floor_samples or 10 * args.H ** args.d

    # rows stream to the CSV as they complete (flushed), so a killed
    # run keeps its partial results; prints flush for the same reason
    fields = ["case", "world", "gate_attempts", "train_seed",
              "heldout_l1", "mc_floor_l1", "loss", "wall_s"]
    rows, groups = [], {}
    with open(out / f"o2_case{args.case}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        fh.flush()
        for wi in range(args.worlds):
            world, attempts = sample_world(args.case, cfg_w,
                                           args.world_seed0 + wi)
            ranges = ranges_for(world)
            ranges.tie_beta = args.tie_beta
            ranges.use_sigma = args.use_sigma
            ranges.use_rho_out = args.use_rho_out
            ranges.use_guard_delta = args.use_guard_delta
            heldout = ranges.heldout_grid()
            floor = mean_mc_floor(world, heldout, n_floor)
            for seed in range(args.seeds):
                cfg = TrainConfig(loss=args.loss, steps=args.steps,
                                  seed=seed, device=args.device,
                                  cond_pool=args.cond_pool or None,
                                  logit_floor=args.logit_floor or None)
                _, _, hist = train_policy(world, cfg, ranges)
                final = hist[-1]
                row = {"case": args.case, "world": world.seed,
                       "gate_attempts": attempts, "train_seed": seed,
                       "heldout_l1": final["heldout_l1"],
                       "mc_floor_l1": floor,
                       "loss": final["loss"],
                       "wall_s": final["wall_s"]}
                rows.append(row)
                writer.writerow(row)
                fh.flush()
                groups.setdefault(world.seed, []).append(
                    final["heldout_l1"])
                print(f"world {world.seed} seed {seed}: "
                      f"L1={final['heldout_l1']:.4f} "
                      f"(floor {floor:.4f}) "
                      f"loss={final['loss']:.4f} {final['wall_s']:.0f}s",
                      flush=True)
    point, lo, hi = cluster_bootstrap_ci(
        {k: np.array(v) for k, v in groups.items()})
    summary = {"case": args.case, "tie_beta": args.tie_beta,
               "heldout_l1_mean": point,
               "ci95": [lo, hi],
               "mc_floor_mean": float(np.mean([r["mc_floor_l1"]
                                               for r in rows]))}
    with open(out / f"o2_case{args.case}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
