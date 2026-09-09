"""Analysis: nested-family separability endpoints from O1b sep
CSVs (`o1_case*_sep.csv` written by scripts/run_o1.py).

Per case and per metric column (`tv_to_*`): per-world max, then the
cluster bootstrap CI over worlds of those maxes; the endpoint is the
median-of-max against the fixed 0.05 separability margin
(PASS = the per-origin family separates from the reference family).

Usage:
    python scripts/analyze_o1b.py RUN_DIR [RUN_DIR ...] \
        [--margin 0.05] [--out summary.json]

Accepts several run dirs because dev runs are launched one case per
process (each with its own timestamped dir). The JSON is written to
--out, defaulting to <first RUN_DIR>/o1b_summary.json.
"""

import argparse
import csv
import json
import pathlib

import numpy as np

from epgfn.stats import cluster_bootstrap_ci

MARGIN = 0.05  # fixed separability margin; do not retune


def load_sep_rows(run_dirs) -> dict[str, list[dict]]:
    """All o1_case*_sep.csv rows under the given run dirs, keyed by case."""
    by_case: dict[str, list[dict]] = {}
    for d in run_dirs:
        for path in sorted(pathlib.Path(d).glob("o1_case*_sep.csv")):
            with open(path) as fh:
                for r in csv.DictReader(fh):
                    by_case.setdefault(r["case"], []).append(r)
    return by_case


def separability_summary(rows: list[dict], margin: float = MARGIN) -> dict:
    """Per tv_to_* metric: per-world max, median-of-max endpoint vs the
    margin, and the cluster bootstrap CI (worlds as clusters, one max
    per world) of the mean of maxes."""
    metrics = sorted({k for r in rows for k in r
                      if k.startswith("tv_to_") and r[k] not in ("", None)})
    out = {}
    for m in metrics:
        per_world: dict[str, list[float]] = {}
        for r in rows:
            v = r.get(m)
            if v in ("", None):
                continue
            per_world.setdefault(str(r["world_seed"]), []).append(float(v))
        groups = {w: np.array([max(vs)]) for w, vs in per_world.items()}
        maxes = np.array([g[0] for g in groups.values()])
        mean, lo, hi = cluster_bootstrap_ci(groups)
        med = float(np.median(maxes))
        out[m] = {"n_worlds": len(maxes),
                  "median_of_max": med,
                  "mean_of_max": mean, "ci_lo": lo, "ci_hi": hi,
                  "per_world_max": {w: float(g[0])
                                    for w, g in groups.items()},
                  "margin": margin,
                  "pass": bool(med >= margin)}
    return out


def main() -> None:
    """Parse CLI arguments, build the per-case separability summary from
    the given run dirs, write it to JSON, and print the endpoint
    verdict table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+",
                    help="run dir(s) from scripts/run_o1.py")
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--out", default=None,
                    help="JSON output path (default: "
                    "<first run dir>/o1b_summary.json)")
    args = ap.parse_args()

    by_case = load_sep_rows(args.run_dirs)
    if not by_case:
        raise SystemExit("no o1_case*_sep.csv found in the given dirs "
                         "(case A has one origin and emits none)")
    summary = {"run_dirs": [str(d) for d in args.run_dirs],
               "margin": args.margin,
               "cases": {case: separability_summary(rows, args.margin)
                         for case, rows in sorted(by_case.items())}}

    out = pathlib.Path(args.out) if args.out else (
        pathlib.Path(args.run_dirs[0]) / "o1b_summary.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"{'case':>4} {'metric':>28} {'median':>7} {'mean':>7} "
          f"{'95% CI':>16} {'n':>3}  verdict")
    for case, mets in sorted(summary["cases"].items()):
        for m, s in mets.items():
            verdict = "PASS" if s["pass"] else "FAIL"
            print(f"{case:>4} {m:>28} {s['median_of_max']:7.3f} "
                  f"{s['mean_of_max']:7.3f} "
                  f"[{s['ci_lo']:6.3f},{s['ci_hi']:6.3f}] "
                  f"{s['n_worlds']:>3}  {verdict} (margin {s['margin']})")
    print(f"summary -> {out}")


if __name__ == "__main__":
    main()
