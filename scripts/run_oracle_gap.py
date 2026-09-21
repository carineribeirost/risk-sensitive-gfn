"""TB vs exact-KL oracle training at matched budget. Per (world, seed)
both losses train with the SAME condition pool and seed (identical
conditions, identical eval), so gap = L1_tb − L1_kl isolates the
optimization cost of TB.

Interpretation (fixed in advance): gap ≈ 0 → TB reaches the oracle.
Gap > 0 but L1_tb ≤ 3× floor → TB adequate, gap quantified. L1_kl far
from floor → capacity problem; revisit the net before blaming TB.

Usage:
    python scripts/run_oracle_gap.py --case B --worlds 4 --seeds 3 \
        --steps 12000 --cond-pool 256 --out results/oracle-gap
"""

import argparse
import csv
import json
import pathlib

import numpy as np

from epgfn.cases import mean_mc_floor
from epgfn.runio import Progress, unique_run_dir
from epgfn.stats import cluster_bootstrap_ci, cluster_permutation_test
from epgfn.train import TrainConfig, default_device, ranges_for, train_policy
from epgfn.worlds import WorldConfig, is_hard, make_world, sample_world


def main() -> None:
    """Run the TB-vs-exact-KL oracle gap experiment for one case.

    Parses CLI args (including optional --resume of a prior run's
    completed trainings and --world-seeds to skip re-gating known
    worlds), trains both the TB and exact-KL losses per (world, seed)
    at matched budget, and writes per-training rows to
    `oracle_gap_case{case}.csv` plus a cluster-bootstrap/permutation
    summary of the TB-minus-KL gap to
    `oracle_gap_case{case}_summary.json`.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="B")
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--world-seed0", type=int, default=0,
                    help="dev block: method diagnostic, never spends "
                    "test worlds")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0,
                    help="first train seed; seeds run [seed0, "
                    "seed0+seeds): lets a launcher fragment per "
                    "(world, seed) unit (default 0 = unchanged)")
    ap.add_argument("--world-seeds", type=int, nargs="+", default=None,
                    help="known gate-passing world seeds, one per "
                    "world index, from ANY prior run of the same "
                    "(case, cfg) chain: each is verified (is_hard + "
                    "seed//100003 == world_seed0+wi) and the attempt "
                    "chain is skipped; sequence-D gates cost ~6 h "
                    "per world")
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--d", type=int, default=2,
                    help="coordinates per point; |X| = H^d")
    ap.add_argument("--geometry", default="grid",
                    choices=["grid", "sequence"],
                    help="score-field family")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--cond-pool", type=int, default=256)
    ap.add_argument("--logit-floor", type=float, default=-25.0,
                    help="TB arm only; the KL targets are exact and "
                    "unclamped by construction")
    ap.add_argument("--use-sigma", action="store_true",
                    help="paper 2 / contingency C1: sigma condition "
                    "axis in both arms")
    ap.add_argument("--net-dim", type=int, default=None,
                    help="trunk width; default is the TrainConfig "
                    "value. Sweeping this is what separates a "
                    "capacity limit from an optimisation one.")
    ap.add_argument("--net-depth", type=int, default=None,
                    help="trunk depth; default is the TrainConfig value")
    ap.add_argument("--floor-samples", type=int, default=None)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--resume", default=None,
                    help="path to a stopped run's oracle_gap_case?.csv "
                    "(or the dir containing it): completed (world, "
                    "seed, arm) trainings are COPIED from it instead "
                    "of retrained. Rows are per-training and flushed, "
                    "so nothing is ever lost; the new run dir ends "
                    "complete, with a full summary. Args must match "
                    "the stopped run (worlds are seed-determined, so "
                    "copied rows are exactly what a retrain would "
                    "produce).")
    ap.add_argument("--out", default="results/oracle-gap")
    args = ap.parse_args()

    prior = {}
    prior_worlds = {}
    if args.resume:
        p = pathlib.Path(args.resume)
        if p.is_dir():
            hits = sorted(p.glob(f"**/oracle_gap_case{args.case}.csv"))
            if not hits:
                raise SystemExit(f"--resume: no oracle_gap_case"
                                 f"{args.case}.csv under {p}")
            p = hits[-1]
        with open(p) as fh:
            for r in csv.DictReader(fh):
                prior[(r["world"], r["train_seed"],
                       r["loss_kind"])] = r
                # gate fast-path source: world seeds are
                # ws*100003+attempt-1 with attempt << 100003
                prior_worlds[int(r["world"]) // 100_003] = int(r["world"])
        print(f"resume: {len(prior)} completed trainings from {p}",
              flush=True)
    if args.world_seeds is not None:
        if len(args.world_seeds) != args.worlds:
            raise SystemExit(f"--world-seeds: got "
                             f"{len(args.world_seeds)} seeds for "
                             f"{args.worlds} worlds")
        for wi, wsd in enumerate(args.world_seeds):
            prior_worlds[args.world_seed0 + wi] = wsd

    out = unique_run_dir(args.out, args)
    print(f"run dir: {out}", flush=True)
    cfg_w = WorldConfig(H=args.H, d=args.d, geometry=args.geometry)
    n_floor = args.floor_samples or 10 * args.H ** args.d

    fields = ["case", "world", "train_seed", "loss_kind", "heldout_l1",
              "mc_floor_l1", "final_loss", "wall_s"]
    rows = []
    groups = {"tb": {}, "exact_kl": {}}
    prog = Progress(args.worlds * args.seeds * 2,
                    f"oracle-{args.case}")
    with open(out / f"oracle_gap_case{args.case}.csv", "w",
              newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        fh.flush()
        for wi in range(args.worlds):
            known = prior_worlds.get(args.world_seed0 + wi)
            if known is not None:
                # gate fast-path (as in run_w33): make_world(recorded
                # seed) IS the world the deterministic chain returns;
                # one is_hard probe verifies the record matches this
                # config before hours of attempts are skipped
                if known // 100_003 != args.world_seed0 + wi:
                    raise SystemExit(
                        f"world seed {known} is not on the chain of "
                        f"base seed {args.world_seed0 + wi}")
                world = make_world(args.case, cfg_w, known)
                if not is_hard(world):
                    raise SystemExit(
                        f"recorded world {known} fails the hardness "
                        f"gate under this config: record does not "
                        f"match this (case, cfg)")
                print(f"world {known}: gate skipped (recorded seed, "
                      f"is_hard verified)", flush=True)
            else:
                world, _ = sample_world(args.case, cfg_w,
                                        args.world_seed0 + wi)
            ranges = ranges_for(world)
            ranges.use_sigma = args.use_sigma
            heldout = ranges.heldout_grid()
            floors = mean_mc_floor(world, heldout, n_floor)
            for seed in range(args.seed0, args.seed0 + args.seeds):
                for kind in ("tb", "exact_kl"):
                    done = prior.get((str(world.seed), str(seed), kind))
                    if done is not None:
                        row = {"case": args.case, "world": world.seed,
                               "train_seed": seed, "loss_kind": kind,
                               "heldout_l1": float(done["heldout_l1"]),
                               "mc_floor_l1": float(done["mc_floor_l1"]),
                               "final_loss": float(done["final_loss"]),
                               "wall_s": float(done["wall_s"])}
                        rows.append(row)
                        writer.writerow(row)
                        fh.flush()
                        groups[kind].setdefault(world.seed, []).append(
                            row["heldout_l1"])
                        prog.step(f"world {world.seed} seed {seed} "
                                  f"{kind}: resumed "
                                  f"(L1={row['heldout_l1']:.4f})")
                        continue
                    net = {}
                    if args.net_dim is not None:
                        net["dim"] = args.net_dim
                    if args.net_depth is not None:
                        net["depth"] = args.net_depth
                    cfg = TrainConfig(loss=kind, steps=args.steps,
                                      seed=seed, device=args.device,
                                      cond_pool=args.cond_pool,
                                      logit_floor=(args.logit_floor or
                                                   None),
                                      **({"net": net} if net else {}))
                    _, _, hist = train_policy(world, cfg, ranges)
                    final = hist[-1]
                    row = {"case": args.case, "world": world.seed,
                           "train_seed": seed, "loss_kind": kind,
                           "heldout_l1": final["heldout_l1"],
                           "mc_floor_l1": floors,
                           "final_loss": final["loss"],
                           "wall_s": final["wall_s"]}
                    rows.append(row)
                    writer.writerow(row)
                    fh.flush()
                    groups[kind].setdefault(world.seed, []).append(
                        final["heldout_l1"])
                    prog.step(f"world {world.seed} seed {seed} {kind}: "
                              f"L1={final['heldout_l1']:.4f} "
                              f"(floor {floors:.4f}) "
                              f"loss={final['loss']:.4f} "
                              f"{final['wall_s']:.0f}s")

    g_tb = {k: np.array(v) for k, v in groups["tb"].items()}
    g_kl = {k: np.array(v) for k, v in groups["exact_kl"].items()}
    gaps = {k: g_tb[k] - g_kl[k] for k in g_tb}   # paired by (world,seed)
    gap_mean, gap_lo, gap_hi = cluster_bootstrap_ci(gaps)
    summary = {"case": args.case,
               "tb": dict(zip(("mean", "ci_lo", "ci_hi"),
                              cluster_bootstrap_ci(g_tb))),
               "exact_kl": dict(zip(("mean", "ci_lo", "ci_hi"),
                                    cluster_bootstrap_ci(g_kl))),
               "gap_tb_minus_kl": {"mean": gap_mean, "ci_lo": gap_lo,
                                   "ci_hi": gap_hi},
               "permutation": cluster_permutation_test(g_tb, g_kl),
               "mc_floor_mean": float(np.mean([r["mc_floor_l1"]
                                               for r in rows]))}
    with open(out / f"oracle_gap_case{args.case}_summary.json",
              "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
