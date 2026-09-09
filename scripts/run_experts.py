"""Conditional policy vs per-condition experts at matched compute.
The conditional trains once with S total
steps; each of the K experts gets S/K steps (K = 9 for A/B/C; 6 for D,
whose 2-point first axis collapses the {first, mid, last} rule; K is
reported, never silent). Experts are evaluated on their own condition;
the conditional on the same subset (all subset members are held-out
grid points, unseen in training by the exclusion balls).

Usage:
    python scripts/run_experts.py --case B --worlds 4 --seeds 3 \
        --steps 4500 --out results/experts
"""

import argparse
import csv
import json

import numpy as np
import torch

from epgfn.cases import target_for
from epgfn.conditions import expert_subset
from epgfn.runio import unique_run_dir
from epgfn.stats import cluster_bootstrap_ci, cluster_permutation_test
from epgfn.target import l1
from epgfn.train import (TrainConfig, default_device, ranges_for,
                         train_expert, train_policy)
from epgfn.worlds import WorldConfig, sample_world


def main() -> None:
    """Run the conditional-vs-experts matched-compute comparison.

    Parses CLI args, and for each of --worlds worlds and --seeds
    training seeds trains one conditional policy plus one expert per
    condition in the case's expert subset (each expert getting
    steps/K), writing per-(world, seed, condition) L1 rows to
    `experts_case{case}.csv` and a cluster-bootstrap/permutation
    summary (expert vs. conditional mean L1, and their difference) to
    `experts_case{case}_summary.json`.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="B")
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--world-seed0", type=int, default=0,
                    help="dev block: method diagnostic, never spends "
                    "test worlds")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4500,
                    help="conditional total; each expert gets steps/K")
    ap.add_argument("--cond-pool", type=int, default=256)
    ap.add_argument("--logit-floor", type=float, default=-25.0)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--out", default="results/experts")
    args = ap.parse_args()

    out = unique_run_dir(args.out, args)
    print(f"run dir: {out}", flush=True)
    cfg_w = WorldConfig(H=args.H)

    fields = ["case", "world", "train_seed", "cond_idx", "expert_steps",
              "l1_expert", "l1_conditional"]
    rows = []
    g_exp, g_cond = {}, {}
    with open(out / f"experts_case{args.case}.csv", "w",
              newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        fh.flush()
        for wi in range(args.worlds):
            world, _ = sample_world(args.case, cfg_w,
                                    args.world_seed0 + wi)
            ranges = ranges_for(world)
            subset = expert_subset(ranges)
            K = len(subset)
            expert_steps = int(round(args.steps / K))
            feats = ranges.features(subset)
            targets = [target_for(world, c) for c in subset]
            for seed in range(args.seeds):
                base = dict(seed=seed, device=args.device,
                            logit_floor=args.logit_floor or None)
                cfg_c = TrainConfig(steps=args.steps,
                                    cond_pool=args.cond_pool, **base)
                policy, _, _ = train_policy(world, cfg_c)
                l1_cond = []
                for i in range(K):
                    f = torch.as_tensor(feats[i], dtype=torch.float32,
                                        device=args.device)
                    lp = policy.log_pf_grid(f).cpu().numpy().reshape(-1)
                    l1_cond.append(l1(np.exp(lp), targets[i]))
                cfg_e = TrainConfig(steps=expert_steps,
                                    eval_every=expert_steps, **base)
                for i, cond in enumerate(subset):
                    _, hist = train_expert(world, cond, cfg_e)
                    row = {"case": args.case, "world": world.seed,
                           "train_seed": seed, "cond_idx": i,
                           "expert_steps": expert_steps,
                           "l1_expert": hist[-1]["heldout_l1"],
                           "l1_conditional": l1_cond[i]}
                    rows.append(row)
                    writer.writerow(row)
                    fh.flush()
                # rows[-K:]: the K expert rows just appended for this
                # (world, seed) pair
                g_exp.setdefault(world.seed, []).append(
                    float(np.mean([r["l1_expert"] for r in rows[-K:]])))
                g_cond.setdefault(world.seed, []).append(
                    float(np.mean(l1_cond)))
                print(f"world {world.seed} seed {seed}: "
                      f"expert mean L1={g_exp[world.seed][-1]:.4f} "
                      f"conditional mean L1={g_cond[world.seed][-1]:.4f} "
                      f"(K={K}, {expert_steps} steps/expert)",
                      flush=True)

    ge = {k: np.array(v) for k, v in g_exp.items()}
    gc = {k: np.array(v) for k, v in g_cond.items()}
    diff = {k: ge[k] - gc[k] for k in ge}  # >0 → conditional better
    summary = {"case": args.case,
               "expert": dict(zip(("mean", "ci_lo", "ci_hi"),
                                  cluster_bootstrap_ci(ge))),
               "conditional": dict(zip(("mean", "ci_lo", "ci_hi"),
                                       cluster_bootstrap_ci(gc))),
               "diff_expert_minus_conditional":
                   dict(zip(("mean", "ci_lo", "ci_hi"),
                            cluster_bootstrap_ci(diff))),
               "permutation": cluster_permutation_test(ge, gc)}
    with open(out / f"experts_case{args.case}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
