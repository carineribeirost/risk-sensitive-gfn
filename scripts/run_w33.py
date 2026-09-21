"""Reproduction runs: the off-policy arms on the (sequence,
sparse) world, native metrics streamed per (case, sparsity, world,
arm, seed).

Reproduction criteria are fixed in advance (user-confirmed before any
run): R1 teacher dominates on-policy mode discovery at high sparsity;
R2 the advantage shrinks at sparsity 1; R3 replay in between.

Usage (dev battery):
    python scripts/run_w33.py --cases A B C D --sparsity 1.0 4.0 \
        --arms onpolicy mix replay teacher --worlds 4 --seeds 3 \
        --H 4 --d 8 --steps 12000 --out results/w33
"""

import argparse
import csv
import json
import pathlib

import numpy as np
import torch

from epgfn.cases import log_reward
from epgfn.runio import Progress, unique_run_dir
from epgfn.stats import cluster_bootstrap_ci
from epgfn.target import p_star
from epgfn.train import TrainConfig, default_device
from epgfn.w33 import ARMS, fixed_condition, run_arm, samples_to_frac
from epgfn.worlds import WorldConfig, is_hard, make_world, sample_world


def main() -> None:
    """Run the off-policy-arm reproduction battery.

    Parses CLI args (including optional --resume of a prior run's
    completed trainings and --world-seeds/--trust-recorded-gates to
    skip re-gating known worlds), trains every (case, sparsity, world,
    arm, seed) combination, streaming per-training metrics to
    `w33_runs.csv` plus a history JSON and terminating-density `.npy`
    for each fresh training, and writes a cluster-bootstrap summary of
    frac_modes by (case, sparsity, arm) to `w33_summary.json`.
    """
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
    ap.add_argument("--alpha-aux", type=float, default=1.0,
                    help="contrastive arm only: α on the InfoNCE "
                    "auxiliary loss over the D⁺/D⁻ replay batch")
    ap.add_argument("--buf-cap", type=int, default=1024,
                    help="contrastive arm only: capacity of EACH "
                    "buffer (D⁺ lowest-R eviction, D⁻ FIFO)")
    ap.add_argument("--max-attempts", type=int, default=200,
                    help="hardness-gate attempt cap (WorldConfig "
                    "default 200). Raising it is identity-preserving: "
                    "attempt chains are deterministic prefixes, so "
                    "every world found under a lower cap is unchanged, "
                    "but log the expansion (probe_gate.py first: "
                    "futile if the failing criterion is structural)")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--resume", default=None,
                    help="path to a stopped run's w33_runs.csv (or the "
                    "dir containing it): completed (case, sparsity, "
                    "world, arm, seed) trainings are COPIED instead of "
                    "retrained; the new run dir ends complete. Args "
                    "must match the stopped run. Per-training history "
                    "JSONs are not re-copied; they remain valid in "
                    "the stopped run's dir. Density .npy files (saved "
                    "for FRESH trainings only) likewise are not "
                    "re-copied, and runs predating the density patch "
                    "have none: cells whose density a figure needs "
                    "must be retrained.")
    ap.add_argument("--world-seeds", type=int, nargs="+", default=None,
                    help="known gate-passing RAW world seeds, one per "
                    "world index (base seed world_seed0+wi): the gate "
                    "attempt chain is skipped after one is_hard "
                    "verification per (case, sparsity), as with "
                    "--resume (which this overrides for gates). Seeds "
                    "are case-specific: single-case runs only. "
                    "Pinning seeds recorded under another sparsity is "
                    "a protocol change (paired worlds across "
                    "sparsity): log it (probe_pinned_seeds.py first).")
    ap.add_argument("--trust-recorded-gates", action="store_true",
                    help="skip the one is_hard verification on "
                    "fast-path worlds. ONLY sound when the record "
                    "provably comes from a run with this exact "
                    "(case, sparsity, config) (e.g. resuming the "
                    "same battery) where the world was already "
                    "gated; one sequence-D attempt costs hours, and "
                    "this skips repaying it. Never combine with "
                    "cross-sparsity pinned seeds: there the "
                    "verification IS the hardness probe.")
    ap.add_argument("--teacher-c", type=float, default=19.0,
                    help="weight on undersampled states in the teacher "
                    "reward, eq. (5) of epgfn.w33's docstring; 19 is "
                    "the cited paper's value for every task, 0 "
                    "reduces to the bare eq. (4)")
    ap.add_argument("--teacher-alpha", type=float, default=0.0,
                    help="reward mixing, eq. (6); the cited paper uses "
                    "0.5 in general and 0.0 on exploration-intensive "
                    "tasks, which is the regime here")
    ap.add_argument("--graded-kappa", type=float, default=0.0,
                    help="replace the flat epsilon on excluded points "
                    "with epsilon*exp(-kappa*violation depth), "
                    "removing the plateau without moving the "
                    "boundary. 0 = the plain reward, bitwise")
    ap.add_argument("--save-grids", action="store_true",
                    help="write each run's exact terminating "
                    "log-density over X, with the target, to "
                    "<out>/grids/*.npz. Makes every coverage "
                    "quantity recomputable without retraining.")
    ap.add_argument("--out", default="results/w33")
    args = ap.parse_args()

    prior = {}
    prior_worlds = {}
    if args.resume:
        p = pathlib.Path(args.resume)
        if p.is_dir():
            hits = sorted(p.glob("**/w33_runs.csv"))
            if not hits:
                raise SystemExit(f"--resume: no w33_runs.csv under {p}")
            p = hits[-1]
        with open(p) as fh:
            for r in csv.DictReader(fh):
                prior[(r["case"], r["sparsity"], r["world"],
                       r["arm"], r["seed"])] = r
                # gate fast-path: world seeds are ws*100003+attempt-1
                # with attempt-1 < max_attempts << 100003, so the base
                # seed ws recovers by integer division
                prior_worlds[(r["case"], r["sparsity"],
                              int(r["world"]) // 100_003)] = (
                    int(r["world"]), int(r["gate_attempts"]))
        print(f"resume: {len(prior)} completed trainings from {p}",
              flush=True)
    if args.world_seeds is not None:
        if len(args.cases) != 1:
            raise SystemExit("--world-seeds: seeds are case-specific; "
                             "run one case per invocation")
        if len(args.world_seeds) != args.worlds:
            raise SystemExit(f"--world-seeds: got "
                             f"{len(args.world_seeds)} seeds for "
                             f"{args.worlds} worlds")
        for wi, wsd in enumerate(args.world_seeds):
            base = args.world_seed0 + wi
            if wsd // 100_003 != base:
                raise SystemExit(f"--world-seeds: seed {wsd} is not "
                                 f"on the chain of base seed {base}")
            # gate_attempts records the chain position (attempt+1),
            # well-defined for a pinned seed even though no scan ran
            for s in args.sparsity:
                prior_worlds[(args.cases[0], str(s), base)] = (
                    wsd, wsd % 100_003 + 1)

    out = unique_run_dir(args.out, args)
    print(f"run dir: {out}", flush=True)

    fields = ["case", "sparsity", "world", "gate_attempts", "arm",
              "seed", "n_modes", "modes_found", "frac_modes",
              "train_discovery", "policy_coverage", "target_coverage",
              "n_modes_live", "policy_coverage_live",
              "target_coverage_live",
              "policy_mass_modes", "reward_queries",
              "samples_to_50", "samples_to_80", "edge_share",
              "final_l1", "final_loss", "wall_s"]
    rows = []
    prog = Progress(len(args.cases) * len(args.sparsity) * args.worlds
                    * len(args.arms) * args.seeds, "w33")
    with open(out / "w33_runs.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        fh.flush()
        for case in args.cases:
            for s in args.sparsity:
                cfg_w = WorldConfig(H=args.H, d=args.d,
                                    geometry=args.geometry, sparsity=s,
                                    max_attempts=args.max_attempts)
                for wi in range(args.worlds):
                    pw = prior_worlds.get((case, str(s),
                                           args.world_seed0 + wi))
                    if pw is not None:
                        # gate fast-path: make_world(recorded seed) IS
                        # the world sample_world would return (the
                        # chain is deterministic), so the full attempt
                        # chain is skipped; one is_hard probe verifies
                        # the CSV matches this config (sequence-D
                        # gates cost hours: attempts up to 60/world)
                        world_seed, attempts = pw
                        world = make_world(case, cfg_w, world_seed)
                        if args.trust_recorded_gates:
                            print(f"{case} s={s} w{world_seed}: gate "
                                  f"TRUSTED from record "
                                  f"(attempts={attempts}), "
                                  f"verification skipped "
                                  f"(--trust-recorded-gates)",
                                  flush=True)
                        elif not is_hard(world):
                            raise SystemExit(
                                f"recorded world {world_seed} fails "
                                f"the hardness gate under this config "
                                f": the record (resume CSV or "
                                f"--world-seeds) does not match these "
                                f"args; probe_pinned_seeds.py prints "
                                f"the failing criterion")
                        else:
                            print(f"{case} s={s} w{world_seed}: gate "
                                  f"skipped (attempts={attempts} "
                                  f"recorded, is_hard verified)",
                                  flush=True)
                    else:
                        world, attempts = sample_world(
                            case, cfg_w, args.world_seed0 + wi)
                    cond = fixed_condition(world)
                    for arm in args.arms:
                        for seed in range(args.seeds):
                            done = prior.get((case, str(s),
                                              str(world.seed), arm,
                                              str(seed)))
                            if done is not None:
                                # CSV keeps the original strings
                                # verbatim; the summary math needs
                                # numeric copies in `rows`
                                writer.writerow(
                                    {k: done[k] for k in fields})
                                fh.flush()
                                num = dict(done)
                                for k in ("sparsity", "frac_modes",
                                          "samples_to_50",
                                          "samples_to_80",
                                          "edge_share", "final_l1",
                                          "final_loss", "wall_s"):
                                    num[k] = (float(done[k])
                                              if done[k] not in
                                              ("", None) else
                                              float("nan"))
                                rows.append(num)
                                prog.step(f"{case} s={s} "
                                          f"w{world.seed} {arm} "
                                          f"seed{seed}: resumed")
                                continue
                            cfg = TrainConfig(
                                steps=args.steps, seed=seed,
                                n_points=args.n_points,
                                eval_every=args.eval_every,
                                logit_floor=args.logit_floor or None,
                                device=args.device)
                            student, hist = run_arm(
                                world, cond, cfg, arm,
                                alpha_aux=args.alpha_aux,
                                buf_cap=args.buf_cap,
                                graded_kappa=args.graded_kappa,
                                teacher_c=args.teacher_c,
                                teacher_alpha=args.teacher_alpha)
                            f = hist[-1]
                            row = {"case": case, "sparsity": s,
                                   "world": world.seed,
                                   "gate_attempts": attempts,
                                   "arm": arm, "seed": seed,
                                   "n_modes": f["n_modes"],
                                   "modes_found": f["modes_found"],
                                   "frac_modes": f["frac_modes"],
                                   "train_discovery":
                                       f["train_discovery"],
                                   "policy_coverage":
                                       f["policy_coverage"],
                                   "target_coverage":
                                       f["target_coverage"],
                                   "n_modes_live": f["n_modes_live"],
                                   "policy_coverage_live":
                                       f["policy_coverage_live"],
                                   "target_coverage_live":
                                       f["target_coverage_live"],
                                   "policy_mass_modes":
                                       f["policy_mass_modes"],
                                   "reward_queries":
                                       f["reward_queries"],
                                   "samples_to_50":
                                       samples_to_frac(hist, 0.5),
                                   "samples_to_80":
                                       samples_to_frac(hist, 0.8),
                                   "edge_share": f["edge_share"],
                                   "final_l1": f["l1"],
                                   "final_loss": f["loss"],
                                   "wall_s": f["wall_s"]}
                            if args.save_grids:
                                gd = out / "grids"
                                gd.mkdir(exist_ok=True)
                                feats = torch.zeros(1, 1,
                                                    device=args.device)
                                lp = (student.log_pf_grid(feats[0])
                                      .cpu().numpy().reshape(-1))
                                np.savez_compressed(
                                    gd / (f"{case}_s{s}_w{world.seed}"
                                          f"_{arm}_seed{seed}.npz"),
                                    log_pf=lp,
                                    target=p_star(
                                        log_reward(world, cond),
                                        cond.beta_t))
                            rows.append(row)
                            writer.writerow(row)
                            fh.flush()
                            prog.step(
                                f"{case} s={s} w{world.seed} {arm} "
                                f"seed{seed}: modes "
                                f"{f['modes_found']}/{f['n_modes']} "
                                f"l1={f['l1']:.3f} "
                                f"edge={f['edge_share']:.3f} "
                                f"{f['wall_s']:.0f}s")
                            with open(out / f"hist_{case}_s{s}_w"
                                      f"{world.seed}_{arm}_{seed}.json",
                                      "w") as hf:
                                json.dump(hist, hf)
                            # final learned density (grid log P_F,
                            # float32, ~0.5 MB at 4^8): policies were
                            # previously discarded, so density panels
                            # were unrenderable without retraining
                            with torch.no_grad():
                                lp = student.log_pf_grid(torch.zeros(
                                    1, device=args.device))
                            np.save(out / f"dens_{case}_s{s}_w"
                                    f"{world.seed}_{arm}_{seed}.npy",
                                    lp.cpu().numpy().astype(
                                        np.float32).reshape(-1))

    # per (case, sparsity, arm): cluster CI of frac_modes over worlds
    summary = {}
    for case in args.cases:
        for s in args.sparsity:
            for arm in args.arms:
                sel = [r for r in rows if r["case"] == case
                       and r["sparsity"] == s and r["arm"] == arm]
                groups = {}
                for r in sel:
                    groups.setdefault(r["world"], []).append(
                        r["frac_modes"])
                point, lo, hi = cluster_bootstrap_ci(
                    {k: np.array(v) for k, v in groups.items()})
                summary[f"{case}/s{s}/{arm}"] = {
                    "frac_modes": point, "ci95": [lo, hi],
                    "samples_to_80_median": float(np.nanmedian(
                        [r["samples_to_80"] for r in sel])),
                    "edge_share_mean": float(np.mean(
                        [r["edge_share"] for r in sel]))}
    with open(out / "w33_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
