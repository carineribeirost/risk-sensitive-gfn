"""Verify recorded gate-passing world seeds under a NEW config (the
cross-sparsity pinning check): rebuild each named raw seed's world
under (case, cfg) and print every hardness probe against its
threshold. A failing world's probe values ARE the measured reason if
it must be excluded (outline: excluded-with-measured-reason).

Unlike probe_gate.py, which scans a base seed's whole attempt chain,
this evaluates only the named raw seeds, one gate attempt each
(itself ~hours for sequence-D) instead of the 60+-attempt chain.

Exit code 0 iff every seed passes, so it works as a launch
precondition before run_w33.py/launch_w33.py --world-seeds.

Usage (D s=4.0, the four s=1.0-hard worlds; pinning them is a
protocol change: paired worlds across sparsity; log it):
    python scripts/probe_pinned_seeds.py --case D --sparsity 4.0 \
        --world-seeds 59 100029 200013 300040 --H 4 --d 8 \
        --geometry sequence
"""

import argparse
import sys

from probe_gate import criteria

from epgfn.worlds import WorldConfig, hardness_report, is_hard, make_world


def main() -> None:
    """Verify named world seeds pass the hardness gate under a given config.

    Parses CLI args, rebuilds each --world-seeds entry under (case,
    cfg), prints its hardness probes against threshold, and exits 0
    iff every seed passes (so this can gate run_w33.py/launch_w33.py
    invocations).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=list("ABCD"))
    ap.add_argument("--sparsity", type=float, default=1.0)
    ap.add_argument("--world-seeds", type=int, nargs="+",
                    required=True,
                    help="raw world seeds (base*100003+attempt), as "
                    "recorded in a w33_runs.csv 'world' column")
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--geometry", default="sequence",
                    choices=["sequence", "grid"])
    args = ap.parse_args()

    cfg = WorldConfig(H=args.H, d=args.d, geometry=args.geometry,
                      sparsity=args.sparsity)
    crit = criteria(args.case, cfg)
    n_pass = 0
    for wsd in args.world_seeds:
        world = make_world(args.case, cfg, wsd)
        rep = hardness_report(world)
        hard = is_hard(world, rep)
        n_pass += bool(hard)
        print(f"world {wsd} (base {wsd // 100_003}, "
              f"attempt {wsd % 100_003}): "
              f"{'HARD' if hard else 'NOT HARD'}")
        for k, op, t, label in crit:
            v = rep.get(k, float("nan"))
            ok = (v >= t) if op == ">=" else (v <= t)
            print(f"    {label:>24}: {v:.4f} vs {t:.3f} "
                  f"[{'ok' if ok else 'FAIL'}]")
    print(f"\n{args.case} s={args.sparsity} H={args.H} d={args.d} "
          f"{args.geometry}: {n_pass}/{len(args.world_seeds)} pinned "
          f"worlds pass")
    sys.exit(0 if n_pass == len(args.world_seeds) else 1)


if __name__ == "__main__":
    main()
