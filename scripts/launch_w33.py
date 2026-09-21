"""Parallel launcher: fan out run_w33.py across (case, sparsity,
world) units at a worker budget, then merge fragment CSVs and
recompute the cluster-bootstrap summary.

Bit-identical to the sequential run: every unit is fully determined by
(case, sparsity, world_seed0 + wi, train seed); no RNG state crosses
units, so grouping cannot change any number. Trainings are tiny
nets, so many concurrent units share one GPU comfortably; thread
pools are capped per job to avoid CPU oversubscription.

--resume is forwarded to every fragment (each copies only its own
matching cells, and the run_w33 gate fast-path reconstructs recorded
worlds instead of re-sampling; sequence-D gates cost hours).

Progress: one `[i/N elapsed<ETA]` line per completed unit (runio
Progress); live per-unit detail streams to frag_*.log in the run dir
(`tail -f <run dir>/frag_*.log`).

Usage (run box):
    python scripts/launch_w33.py --cases A B C D --sparsity 1.0 \
        --arms onpolicy mix replay teacher contrastive \
        --worlds 4 --seeds 3 --H 4 --d 8 --steps 12000 \
        --resume results/w33/... --jobs 12 --out results/w33
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
from epgfn.stats import cluster_bootstrap_ci
from epgfn.w33 import ARMS


def main() -> None:
    """Parse CLI arguments, fan run_w33.py out across (case, sparsity,
    world) units at a worker budget, merge the fragment CSVs, and
    recompute the per-(case, sparsity, arm) cluster-bootstrap
    summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    choices=list(ARMS))
    ap.add_argument("--sparsity", type=float, nargs="+",
                    default=[1.0, 4.0])
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--world-seed0", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--geometry", default="sequence",
                    choices=["sequence", "grid"])
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--n-points", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--logit-floor", type=float, default=-25.0)
    ap.add_argument("--alpha-aux", type=float, default=1.0)
    ap.add_argument("--buf-cap", type=int, default=1024)
    ap.add_argument("--max-attempts", type=int, default=200,
                    help="forwarded to run_w33.py (gate attempt cap; "
                    "raising is identity-preserving, log it)")
    ap.add_argument("--device", default=None,
                    help="forwarded to run_w33.py if set")
    ap.add_argument("--resume", default=None,
                    help="forwarded to every fragment (path to a "
                    "stopped run's w33_runs.csv or its dir)")
    ap.add_argument("--world-seeds", type=int, nargs="+", default=None,
                    help="known gate-passing raw world seeds, one per "
                    "world index; each fragment receives its own seed "
                    "(run_w33 gate fast-path, is_hard verified). "
                    "Case-specific: single-case launches only")
    ap.add_argument("--trust-recorded-gates", action="store_true",
                    help="forwarded to every fragment: skip the "
                    "is_hard verification on fast-path worlds (only "
                    "for same-(case,sparsity,config) records, see "
                    "run_w33.py)")
    ap.add_argument("--teacher-c", type=float, default=19.0,
                    help="forwarded to run_w33.py")
    ap.add_argument("--teacher-alpha", type=float, default=0.0,
                    help="forwarded to run_w33.py")
    ap.add_argument("--graded-kappa", type=float, default=0.0,
                    help="forwarded to run_w33.py")
    ap.add_argument("--save-grids", action="store_true",
                    help="forwarded to run_w33.py: save each "
                    "fragment's exact policy grid.")
    ap.add_argument("--jobs", type=int,
                    default=min(12, os.cpu_count() or 1),
                    help="max concurrent (case, sparsity, world) "
                    "units; tiny nets share one GPU fine")
    ap.add_argument("--threads-per-job", type=int, default=2)
    ap.add_argument("--out", default="results/w33")
    args = ap.parse_args()

    if args.world_seeds is not None:
        if len(args.cases) != 1:
            raise SystemExit("--world-seeds: seeds are case-specific; "
                             "launch one case per invocation")
        if len(args.world_seeds) != args.worlds:
            raise SystemExit(f"--world-seeds: got "
                             f"{len(args.world_seeds)} seeds for "
                             f"{args.worlds} worlds")

    out = unique_run_dir(args.out, args)
    print(f"run dir: {out}", flush=True)
    units = [(c, s, args.world_seed0 + w) for c in args.cases
             for s in args.sparsity for w in range(args.worlds)]
    env = dict(os.environ,
               OMP_NUM_THREADS=str(args.threads_per_job),
               MKL_NUM_THREADS=str(args.threads_per_job))

    def launch(case, s, ws):
        frag_out = out / f"frag_{case}_s{s}_w{ws}"
        cmd = [sys.executable, "scripts/run_w33.py", "--cases", case,
               "--sparsity", str(s), "--arms", *args.arms,
               "--worlds", "1", "--world-seed0", str(ws),
               "--seeds", str(args.seeds), "--H", str(args.H),
               "--d", str(args.d), "--geometry", args.geometry,
               "--steps", str(args.steps),
               "--n-points", str(args.n_points),
               "--eval-every", str(args.eval_every),
               "--logit-floor", str(args.logit_floor),
               "--alpha-aux", str(args.alpha_aux),
               "--buf-cap", str(args.buf_cap),
               "--max-attempts", str(args.max_attempts),
               "--out", str(frag_out)]
        if args.graded_kappa:
            cmd += ["--graded-kappa", str(args.graded_kappa)]
        cmd += ["--teacher-c", str(args.teacher_c),
                "--teacher-alpha", str(args.teacher_alpha)]
        if args.save_grids:
            cmd += ["--save-grids"]
        if args.device:
            cmd += ["--device", args.device]
        if args.resume:
            cmd += ["--resume", args.resume]
        if args.world_seeds is not None:
            cmd += ["--world-seeds",
                    str(args.world_seeds[ws - args.world_seed0])]
        if args.trust_recorded_gates:
            cmd += ["--trust-recorded-gates"]
        log = open(out / f"frag_{case}_s{s}_w{ws}.log", "w")
        return subprocess.Popen(cmd, env=env, stdout=log,
                                stderr=subprocess.STDOUT)

    fan_out(units, lambda u: launch(*u), args.jobs, "w33-launch", out,
           fmt=lambda u: f"{u[0]}/s{u[1]}/world{u[2]}")

    # merge in the sequential run's row order (unit order = the
    # run_w33 loop order, rows within a fragment already ordered)
    paths = [
        one_fragment(str(out / f"frag_{case}_s{s}_w{ws}" / "**" /
                         "w33_runs.csv"), f"{case}/s{s}/w{ws}")
        for case, s, ws in units]
    expect = (len(args.cases) * len(args.sparsity) * args.worlds
              * len(args.arms) * args.seeds)
    rows = merge_fragments(paths, expect)
    write_merged_csv(out / "w33_runs.csv", rows)

    # per (case, sparsity, arm): cluster CI of frac_modes over worlds
    # (same block as run_w33.py, on the merged rows)
    summary = {}
    for case in args.cases:
        for s in args.sparsity:
            for arm in args.arms:
                sel = [r for r in rows if r["case"] == case
                       and float(r["sparsity"]) == s
                       and r["arm"] == arm]
                groups: dict = {}
                for r in sel:
                    groups.setdefault(r["world"], []).append(
                        float(r["frac_modes"]))
                point, lo, hi = cluster_bootstrap_ci(
                    {k: np.array(v) for k, v in groups.items()})
                summary[f"{case}/s{s}/{arm}"] = {
                    "frac_modes": point, "ci95": [lo, hi],
                    "samples_to_80_median": float(np.nanmedian(
                        [float(r["samples_to_80"]) if
                         r["samples_to_80"] not in ("", None)
                         else float("nan") for r in sel])),
                    "edge_share_mean": float(np.mean(
                        [float(r["edge_share"]) for r in sel]))}
    with open(out / "w33_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"merged {len(rows)} rows; hist/dens files remain in the "
          f"frag_*/ subdirs of {out}", flush=True)


if __name__ == "__main__":
    main()
