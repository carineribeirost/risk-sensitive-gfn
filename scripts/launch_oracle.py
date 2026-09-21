"""Parallel oracle-gap launcher: fan out run_oracle_gap.py across
(world, train-seed) units, then merge fragment CSVs and recompute the
paired cluster summary: same numbers as the sequential run (units
share no RNG state; the summary math is run_oracle_gap's, applied to
the merged rows).

Combine with --world-seeds (recorded gate-passing worlds, verified by
is_hard in each fragment) and --resume to turn a multi-day sequential
sequence-D run into ~the wall time of one exact-KL training.

Usage (finish the killed seq-oracle D run, ~30-40 min):
    python scripts/launch_oracle.py --case D --geometry sequence \
        --H 4 --d 8 --worlds 4 --seeds 3 \
        --world-seeds 59 100029 200013 300040 \
        --resume results/seq-oracle/2026-07-18/185021-5274 \
        --jobs 6 --out results/seq-oracle
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from epgfn.launch import fan_out, merge_fragments, one_fragment, \
    write_merged_csv
from epgfn.runio import unique_run_dir
from epgfn.stats import cluster_bootstrap_ci, cluster_permutation_test


def main() -> None:
    """Parse CLI arguments, fan run_oracle_gap.py out across (world,
    train-seed) units, merge the fragment CSVs, and recompute the
    paired TB-vs-exact-KL cluster summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="B")
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--world-seed0", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--geometry", default="grid",
                    choices=["grid", "sequence"])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--cond-pool", type=int, default=256)
    ap.add_argument("--logit-floor", type=float, default=-25.0)
    ap.add_argument("--use-sigma", action="store_true")
    ap.add_argument("--net-dim", type=int, default=None,
                    help="forwarded: trunk width")
    ap.add_argument("--net-depth", type=int, default=None,
                    help="forwarded: trunk depth")
    ap.add_argument("--floor-samples", type=int, default=None)
    ap.add_argument("--device", default=None,
                    help="forwarded to run_oracle_gap.py if set")
    ap.add_argument("--world-seeds", type=int, nargs="+", default=None,
                    help="known gate-passing world seeds, one per "
                    "world index; forwarded per fragment (each "
                    "fragment verifies is_hard before skipping the "
                    "attempt chain)")
    ap.add_argument("--resume", default=None,
                    help="forwarded to every fragment (each copies "
                    "only its own (world, seed, kind) cells)")
    ap.add_argument("--jobs", type=int,
                    default=min(6, os.cpu_count() or 1))
    ap.add_argument("--threads-per-job", type=int, default=2)
    ap.add_argument("--out", default="results/oracle-gap")
    args = ap.parse_args()

    if (args.world_seeds is not None
            and len(args.world_seeds) != args.worlds):
        raise SystemExit(f"--world-seeds: got {len(args.world_seeds)} "
                         f"seeds for {args.worlds} worlds")
    out = unique_run_dir(args.out, args)
    print(f"run dir: {out}", flush=True)
    units = [(args.world_seed0 + w, s) for w in range(args.worlds)
             for s in range(args.seeds)]
    env = dict(os.environ,
               OMP_NUM_THREADS=str(args.threads_per_job),
               MKL_NUM_THREADS=str(args.threads_per_job))

    def launch(ws, seed):
        frag_out = out / f"frag_w{ws}_s{seed}"
        cmd = [sys.executable, "scripts/run_oracle_gap.py",
               "--case", args.case, "--worlds", "1",
               "--world-seed0", str(ws), "--seeds", "1",
               "--seed0", str(seed), "--H", str(args.H),
               "--d", str(args.d), "--geometry", args.geometry,
               "--steps", str(args.steps),
               "--cond-pool", str(args.cond_pool),
               "--logit-floor", str(args.logit_floor),
               "--out", str(frag_out)]
        if args.world_seeds is not None:
            cmd += ["--world-seeds",
                    str(args.world_seeds[ws - args.world_seed0])]
        if args.floor_samples:
            cmd += ["--floor-samples", str(args.floor_samples)]
        if args.use_sigma:
            cmd += ["--use-sigma"]
        if args.net_dim is not None:
            cmd += ["--net-dim", str(args.net_dim)]
        if args.net_depth is not None:
            cmd += ["--net-depth", str(args.net_depth)]
        if args.device:
            cmd += ["--device", args.device]
        if args.resume:
            cmd += ["--resume", args.resume]
        log = open(out / f"frag_w{ws}_s{seed}.log", "w")
        return subprocess.Popen(cmd, env=env, stdout=log,
                                stderr=subprocess.STDOUT)

    fan_out(units, lambda u: launch(*u), args.jobs,
           f"oracle-{args.case}-launch", out,
           fmt=lambda u: f"world{u[0]}/seed{u[1]}")

    paths = [  # sequential run's row order
        one_fragment(str(out / f"frag_w{ws}_s{s}" / "**" /
                         f"oracle_gap_case{args.case}.csv"),
                    f"w{ws}/s{s}")
        for ws, s in units]
    expect = args.worlds * args.seeds * 2
    rows = merge_fragments(paths, expect)
    write_merged_csv(out / f"oracle_gap_case{args.case}.csv", rows)

    # paired cluster summary: run_oracle_gap's block on merged rows
    groups = {"tb": {}, "exact_kl": {}}
    for r in rows:
        groups[r["loss_kind"]].setdefault(r["world"], []).append(
            float(r["heldout_l1"]))
    g_tb = {k: np.array(v) for k, v in groups["tb"].items()}
    g_kl = {k: np.array(v) for k, v in groups["exact_kl"].items()}
    gaps = {k: g_tb[k] - g_kl[k] for k in g_tb}
    gap_mean, gap_lo, gap_hi = cluster_bootstrap_ci(gaps)
    summary = {"case": args.case,
               "tb": dict(zip(("mean", "ci_lo", "ci_hi"),
                              cluster_bootstrap_ci(g_tb))),
               "exact_kl": dict(zip(("mean", "ci_lo", "ci_hi"),
                                    cluster_bootstrap_ci(g_kl))),
               "gap_tb_minus_kl": {"mean": gap_mean, "ci_lo": gap_lo,
                                   "ci_hi": gap_hi},
               "permutation": cluster_permutation_test(g_tb, g_kl),
               "mc_floor_mean": float(np.mean(
                   [float(r["mc_floor_l1"]) for r in rows]))}
    with open(out / f"oracle_gap_case{args.case}_summary.json",
              "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
