"""Parallel O2 launcher: fan out run_o2.py across (case, world) units
at a worker budget, then merge fragment CSVs and recompute the
cluster-bootstrap summaries.

Bit-identical to the sequential runs: every unit is fully determined by
(case, world_seed0 + wi, train seed); no RNG state is shared across
units, so grouping cannot change any number. Thread pools are capped
per job to avoid oversubscription.

Usage:
    python scripts/launch_o2.py --cases A B C D --worlds 4 \
        --world-seed0 0 --seeds 3 --H 32 --steps 4000 \
        --jobs 16 --threads-per-job 2 --out results/o2-dev
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

from epgfn.launch import fan_out, merge_fragments, write_merged_csv
from epgfn.runio import unique_run_dir
from epgfn.stats import cluster_bootstrap_ci


def main() -> None:
    """Parse CLI arguments, fan run_o2.py out across (case, world) units
    at a worker budget, merge the fragment CSVs, and recompute the
    per-case cluster-bootstrap summaries."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--world-seed0", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--d", type=int, default=2,
                    help="forwarded to run_o2.py")
    ap.add_argument("--geometry", default="grid",
                    choices=["grid", "sequence"],
                    help="forwarded to run_o2.py")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--device", default=None,
                    help="forwarded to run_o2.py if set")
    ap.add_argument("--loss", default="tb", choices=["tb", "subtb"],
                    help="forwarded to run_o2.py")
    ap.add_argument("--use-sigma", action="store_true",
                    help="forwarded to run_o2.py (paper 2)")
    ap.add_argument("--use-rho-out", action="store_true",
                    help="forwarded to run_o2.py (case D only)")
    ap.add_argument("--weight-alpha", type=float, default=2.0,
                    help="forwarded to run_o2.py")
    ap.add_argument("--k-guard", type=int, default=0,
                    help="forwarded to run_o2.py (guarded A)")
    ap.add_argument("--use-guard-delta", action="store_true",
                    help="forwarded to run_o2.py (case A only)")
    ap.add_argument("--cond-pool", type=int, default=256,
                    help="forwarded to run_o2.py (0 = streaming legacy)")
    ap.add_argument("--logit-floor", type=float, default=-25.0,
                    help="forwarded to run_o2.py (0 = off)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(),
                    help="max concurrent (case, world) units")
    ap.add_argument("--threads-per-job", type=int, default=2)
    ap.add_argument("--out", default="results/o2")
    args = ap.parse_args()

    out = unique_run_dir(args.out, args)
    print(f"run dir: {out}", flush=True)
    units = [(c, args.world_seed0 + w)
             for c in args.cases for w in range(args.worlds)]
    env = dict(os.environ,
               OMP_NUM_THREADS=str(args.threads_per_job),
               MKL_NUM_THREADS=str(args.threads_per_job))

    def launch(case, ws):
        frag_out = out / f"frag_{case}_{ws}"
        cmd = [sys.executable, "scripts/run_o2.py", "--case", case,
               "--worlds", "1", "--world-seed0", str(ws),
               "--seeds", str(args.seeds), "--H", str(args.H),
               "--d", str(args.d), "--geometry", args.geometry,
               "--steps", str(args.steps), "--out", str(frag_out),
               "--cond-pool", str(args.cond_pool),
               "--logit-floor", str(args.logit_floor),
               "--loss", args.loss]
        if args.use_sigma:
            cmd += ["--use-sigma"]
        if args.use_rho_out:
            cmd += ["--use-rho-out"]
        if args.weight_alpha != 2.0:
            cmd += ["--weight-alpha", str(args.weight_alpha)]
        if args.k_guard:
            cmd += ["--k-guard", str(args.k_guard)]
        if args.use_guard_delta:
            cmd += ["--use-guard-delta"]
        if args.device:
            cmd += ["--device", args.device]
        log = open(out / f"frag_{case}_{ws}.log", "w")
        return subprocess.Popen(cmd, env=env, stdout=log,
                                stderr=subprocess.STDOUT)

    fan_out(units, lambda u: launch(*u), args.jobs, "o2", out,
           fmt=lambda u: f"{u[0]}/world{u[1]}")

    for case in args.cases:
        paths = sorted(glob.glob(
            str(out / f"frag_{case}_*" / "**" / f"o2_case{case}.csv"),
            recursive=True))
        expect = args.worlds * args.seeds
        rows = merge_fragments(paths, expect, context=f"case {case}: ")
        write_merged_csv(out / f"o2_case{case}.csv", rows)
        groups: dict = {}
        for r in rows:
            groups.setdefault(r["world"], []).append(
                float(r["heldout_l1"]))
        point, lo, hi = cluster_bootstrap_ci(
            {k: np.array(v) for k, v in groups.items()})
        summary = {"case": case, "heldout_l1_mean": point,
                   "ci95": [lo, hi],
                   "mc_floor_mean": float(np.mean(
                       [float(r["mc_floor_l1"]) for r in rows])),
                   "n_runs": len(rows)}
        with open(out / f"o2_case{case}_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
