"""Convergence probe: train one (case, world) unit with the canonical
config and record the full evaluation trajectory: diagnoses
plateau-vs-still-descending for G0 stragglers (results log:
`g0-canonical`, case B at 3.02x floor).

Usage:
    python scripts/probe_convergence.py --case B --world-seed 0 \
        --steps 12000 --out results/probes
"""

import argparse
import json

from epgfn.runio import unique_run_dir
from epgfn.cases import mean_mc_floor
from epgfn.train import TrainConfig, default_device, ranges_for, \
    train_policy
from epgfn.worlds import WorldConfig, sample_world


def main() -> None:
    """Train one (case, world) unit and record the eval trajectory.

    Parses CLI args, samples the world, computes the Monte-Carlo floor
    on the held-out grid, trains the policy while printing heldout L1
    as a multiple of the floor at each eval, and dumps the floor plus
    full history to `probe_{case}_w{world_seed}.json` in --out.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="B")
    ap.add_argument("--world-seed", type=int, default=0)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--d", type=int, default=2,
                    help="coordinates per point; |X| = H^d")
    ap.add_argument("--geometry", default="grid",
                    choices=["grid", "sequence"],
                    help="score-field family")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--use-sigma", action="store_true",
                    help="paper 2 / contingency C1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cond-pool", type=int, default=256)
    ap.add_argument("--logit-floor", type=float, default=-25.0)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--out", default="results/probes")
    args = ap.parse_args()

    out = unique_run_dir(args.out, args)
    print(f"run dir: {out}", flush=True)
    world, attempts = sample_world(args.case,
                                   WorldConfig(H=args.H, d=args.d,
                                               geometry=args.geometry),
                                   args.world_seed)
    ranges = ranges_for(world)
    ranges.use_sigma = args.use_sigma
    n_floor = 10 * args.H ** args.d
    floor = mean_mc_floor(world, ranges.heldout_grid(), n_floor)
    print(f"world {world.seed} (attempts {attempts}), "
          f"mc floor {floor:.4f}, pass line {3 * floor:.4f}", flush=True)

    cfg = TrainConfig(steps=args.steps, eval_every=args.eval_every,
                      seed=args.seed, device=args.device,
                      cond_pool=args.cond_pool or None,
                      logit_floor=args.logit_floor or None)
    _, _, hist = train_policy(world, cfg, ranges)
    for h in hist:
        print(f"step {h['step']:>6}: loss {h['loss']:.3f} "
              f"heldout_l1 {h['heldout_l1']:.4f} "
              f"({h['heldout_l1'] / floor:.2f}x floor)", flush=True)
    with open(out / f"probe_{args.case}_w{args.world_seed}.json",
              "w") as fh:
        json.dump({"floor": floor, "history": hist}, fh, indent=1)


if __name__ == "__main__":
    main()
