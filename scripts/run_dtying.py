"""Case D tying ablation (profile-scale vs per-origin), plus the
misspecification arm. Reserved for the TEST block; dev runs are
sanity checks only.

Three families, all priced exactly against the tied single-pair family
(`o1._shared_family`):
- profile-scale: RiskD(beta_in, tuple(s * rel_profile), 0.5) over an
  s-grid: one scalar dial aimed by the world's TRUE profile;
- misspec: same, but the profile rolled by 1 (np.roll), aiming the
  scalar dial at the WRONG origins; measures the cost of a wrong prior;
- one-hot: the per-origin probes from o1_separability (the full
  family's own separability rows), for comparison.

Usage:
    python scripts/run_dtying.py --worlds 4 --world-seed0 0 --H 32 \
        --out results/dtying
"""

import argparse
import csv
import json

import numpy as np

from epgfn.cases import target_for
from epgfn.conditions import Condition, RiskD
from epgfn.o1 import _shared_family, o1_separability
from epgfn.runio import unique_run_dir
from epgfn.stats import cluster_bootstrap_ci
from epgfn.target import tv
from epgfn.worlds import WorldConfig, sample_world


def main() -> None:
    """Run the Case D tying ablation and write per-world/per-family results.

    Parses CLI args, and for each of --worlds worlds prices the
    profile-scale, misspecified-profile, and one-hot families against
    the tied shared family, writing per-row results to
    `dtying_caseD.csv` and a cluster-bootstrap summary (median/mean of
    the per-world max TV, by tying family) to
    `dtying_caseD_summary.json`.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--world-seed0", type=int, default=0)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--beta-t", type=float, default=4.0)
    ap.add_argument("--w-g", type=float, default=0.3)
    ap.add_argument("--out", default="results/dtying")
    args = ap.parse_args()

    out = unique_run_dir(args.out, args)
    print(f"run dir: {out}", flush=True)
    cfg = WorldConfig(H=args.H)
    beta_grid = np.linspace(0.1, 1.0, 7)
    rho_grid = np.linspace(0.0, 1.6, 7)

    fields = ["world_seed", "tying", "s", "origin", "rho_hi",
              "tv_to_shared"]
    rows = []
    groups = {}  # tying -> {world: [tv, ...]}
    with open(out / "dtying_caseD.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        fh.flush()
        for wi in range(args.worlds):
            world, attempts = sample_world("D", cfg,
                                           args.world_seed0 + wi)
            shared = _shared_family(world, beta_grid, rho_grid,
                                    args.beta_t, args.w_g)
            # nominal beta_in=0.5, clamped into this world's valid range
            b = float(np.clip(0.5, world.beta_bounds["in"], 1.0))
            profiles = {"profile": world.rel_profile,
                        "misspec": np.roll(world.rel_profile, 1)}
            for tying, prof in profiles.items():
                for s in rho_grid[1:]:  # s=0 is the tied ρ=0 member
                    cond = Condition(args.beta_t, args.w_g,
                                     RiskD(b, tuple(float(s) * prof), 0.5))
                    d = min(tv(target_for(world, cond), q)
                            for q in shared)
                    row = {"world_seed": world.seed, "tying": tying,
                           "s": float(s), "origin": None,
                           "rho_hi": None, "tv_to_shared": d}
                    rows.append(row)
                    writer.writerow(row)
                    fh.flush()
                    groups.setdefault(tying, {}).setdefault(
                        world.seed, []).append(d)
            for r in o1_separability(world, beta_grid, rho_grid,
                                     args.beta_t, args.w_g):
                row = {"world_seed": world.seed, "tying": "one_hot",
                       "s": None, "origin": r["origin"],
                       "rho_hi": r["rho_hi"],
                       "tv_to_shared": r["tv_to_shared"]}
                rows.append(row)
                writer.writerow(row)
                fh.flush()
                groups.setdefault("one_hot", {}).setdefault(
                    world.seed, []).append(r["tv_to_shared"])
            print(f"world {world.seed} (attempts {attempts}): "
                  + "  ".join(f"{t}: max {max(v[world.seed]):.3f}"
                              for t, v in groups.items()
                              if world.seed in v), flush=True)

    summary = {"case": "D"}
    for tying, g in groups.items():
        maxes = {w: np.array([max(v)]) for w, v in g.items()}
        pt, lo, hi = cluster_bootstrap_ci(maxes)
        summary[tying] = {
            "median_of_max": float(np.median([max(v)
                                              for v in g.values()])),
            "mean_of_max": pt, "ci_lo": lo, "ci_hi": hi}
    with open(out / "dtying_caseD_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
