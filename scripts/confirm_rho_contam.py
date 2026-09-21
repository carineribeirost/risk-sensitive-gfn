"""Confirmatory test: does the ambiguity ball move the rare-state
contamination guarantee?

Motivation. On an earlier 8-world o3-pareto run the paired effect of
the ball on case B's contamination worst case (rho: 0 -> 0.5 at fixed
(world, beta), at each world's own t*) was +0.018 mean, positive in
5/8 worlds, cluster sign-flip p = 0.27, d = 0.43: directional, not
confirmed. At that effect size ~40 worlds give ~80% power at
alpha = 0.05 (two-sided), so the extension run is worlds 0..39 with
every other flag identical to the 8-world run (seeds are deterministic
per world, so worlds 0..7 reproduce byte-for-byte and the original run
embeds in the new one).

Launch:
    python scripts/run_o3.py --cases A B C D --worlds 40 \
        --beta-grid 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
        --rho-grid 0.0 0.05 0.1 0.15 0.2 0.3 0.5 0.8 1.2 \
        --n-stress 500 --kappa-range 8 60 --contam-eps 0.2 \
        --jobs 16 --out results/o3-pareto-w40

Pre-registered analysis (this script):
  PRIMARY (confirmatory): case B, per-world mean paired delta of
  contam_worst for rho 0 -> 0.5 at fixed beta, rows at t = t*,
  target kind "risk" only; one-sample sign-flip permutation on the
  per-world means (epgfn.stats.paired_sign_permutation, default
  n_perm/seed); confirmed iff p < 0.05 and mean > 0.
  SECONDARY (reported, not gated): the same delta for stress_p05;
  and cases A, C, D on both metrics (uniform case coverage).
  The rho pair (0, 0.5) and the (world, beta) pairing are fixed
  here, before the run; changing them afterwards is a deviation.

Usage after the run:
    python scripts/confirm_rho_contam.py results/o3-pareto-w40/DATE/RUN
"""

import csv
import pathlib
import sys
from collections import defaultdict

from epgfn.stats import paired_sign_permutation

RHO_LO, RHO_HI = 0.0, 0.5   # registered pair


def per_world_deltas(run_dir, case, key):
    rows = []
    with open(pathlib.Path(run_dir) / f"o3_case{case}.csv") as fh:
        rows = list(csv.DictReader(fh))
    tstar = {r["world_seed"]: r["t_star"] for r in rows}
    star = [r for r in rows if r["t"] == tstar[r["world_seed"]]
            and r["target"] == "risk"]
    cell = {(r["world_seed"], float(r["beta_cvar"]),
             float(r["rho"])): float(r[key]) for r in star}
    per_world = defaultdict(list)
    for (w, b, rho), v in cell.items():
        if rho == RHO_LO and (w, b, RHO_HI) in cell:
            per_world[w].append(cell[(w, b, RHO_HI)] - v)
    return {w: sum(v) / len(v) for w, v in sorted(per_world.items())}


def main() -> None:
    run_dir = sys.argv[1]
    print(f"run dir: {run_dir}  (rho pair {RHO_LO} -> {RHO_HI}, "
          "paired at fixed (world, beta), rows at t*)")
    for case in "ABCD":
        for key in ("contam_worst", "stress_p05"):
            d = per_world_deltas(run_dir, case, key)
            res = paired_sign_permutation(list(d.values()))
            tag = ("PRIMARY" if case == "B" and key == "contam_worst"
                   else "secondary")
            pos = sum(1 for x in d.values() if x > 0)
            print(f"case {case} {key:13s} [{tag:9s}] "
                  f"mean {res['mean']:+.4f}  {pos}/{res['n']} worlds "
                  f"positive  p = {res['p_value']:.4f}  "
                  f"d = {res['cohens_d']:.2f}")
    print("\nverdict rule (registered): PRIMARY confirmed iff "
          "p < 0.05 and mean > 0.")


if __name__ == "__main__":
    main()
