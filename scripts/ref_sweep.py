"""Sensitivity of the finite-sample reference to the sample count n.

The held-out L1 endpoint is normalised by a Monte-Carlo reference floor
whose sample count is a free choice, so it needs a sensitivity check.
This reproduces the stored estimator exactly (`epgfn.target.mc_floor`,
20 repetitions, averaged over the 243 held-out conditions, one RNG
threaded across conditions and seeded at the world seed) and sweeps n.

Calibration family only, H=32 d=2, |X|=1024, declared default n = 10|X| = 10240.
Targets are computed once per world and reused for every n, since the target
does not depend on n.
"""
import json
import pathlib
import time

import numpy as np

from epgfn.cases import target_for
from epgfn.target import mc_floor
from epgfn.train import ranges_for
from epgfn.worlds import WorldConfig, make_world

# The gate-accepted world seeds the oracle-gap batteries used, per case.
WORLDS = {"A": [0, 100003, 200006, 300009],
          "B": [0, 100003, 200006, 300009],
          "C": [0, 100004, 200006, 300009],
          "D": [1, 100011, 200009, 300009]}

# Stored student (TB) and oracle held-out L1 on this family, from
# results/oracle-gap/2026-07-08, so the ratios below are the published ones.
STORED = {"A": (0.0825, 0.0382), "B": (0.3985, 0.1739),
          "C": (0.3383, 0.0867), "D": (0.1334, 0.1004)}

OUT_DIR = "results/ref-sweep"

NX = 1024
NS = [NX, 2 * NX, 5 * NX, 10 * NX, 20 * NX, 50 * NX, 100 * NX]

cfg = WorldConfig(H=32, d=2, geometry="grid")
out = {"family": "calibration H=32 d=2", "n_points": NX,
       "default_n": 10 * NX, "n_grid": NS, "per_case": {}}

for case, seeds in WORLDS.items():
    per_world = []
    for ws in seeds:
        t0 = time.time()
        w = make_world(case, cfg, ws)
        hg = ranges_for(w).heldout_grid()
        tgts = [target_for(w, c) for c in hg]
        vals = []
        for n in NS:
            rng = np.random.default_rng(w.seed)
            vals.append(float(np.mean([mc_floor(p, n, rng) for p in tgts])))
        per_world.append(vals)
        print(f"case {case} world {ws}: {len(hg)} conditions, "
              f"{time.time() - t0:.0f}s, "
              + " ".join(f"{v:.4f}" for v in vals), flush=True)
    m = np.mean(per_world, axis=0)
    out["per_case"][case] = {"per_world": per_world, "mean": m.tolist()}
    tb, kl = STORED[case]
    print(f"== case {case} mean reference: "
          + " ".join(f"{v:.4f}" for v in m), flush=True)
    print(f"   student ratio:              "
          + " ".join(f"{tb / v:.2f}" for v in m), flush=True)
    print(f"   oracle ratio:               "
          + " ".join(f"{kl / v:.2f}" for v in m), flush=True)

print("\nn grid: " + " ".join(str(n) for n in NS))
print("\nstudent L1 / reference, by n")
print("case " + " ".join(f"{n:>8d}" for n in NS))
for case in WORLDS:
    m = np.array(out["per_case"][case]["mean"])
    tb = STORED[case][0]
    print(f"{case}    " + " ".join(f"{tb / v:8.2f}" for v in m))
print("\noracle L1 / reference, by n")
print("case " + " ".join(f"{n:>8d}" for n in NS))
for case in WORLDS:
    m = np.array(out["per_case"][case]["mean"])
    kl = STORED[case][1]
    print(f"{case}    " + " ".join(f"{kl / v:8.2f}" for v in m))

outdir = pathlib.Path(OUT_DIR)
outdir.mkdir(parents=True, exist_ok=True)
dest = outdir / "ref_sweep.json"
json.dump(out, open(dest, "w"), indent=1)
print(f"\nwrote {dest}")
