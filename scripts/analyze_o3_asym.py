"""Analysis: O3 matched-vs-mismatched asymmetry (confirmatory) +
the regime atlas (exploratory).

Confirmatory (case B):
- For each world and each unordered ρ pair {a, b}, a ≠ b (equal total
  budget), the *matched* member puts the larger ρ on the smaller-κ
  (noisier) origin. Δ = stress_p05(matched) − stress_p05(mismatched),
  paired by construction (same stress draws).
- `paired_sign_permutation` on per-world mean Δ vs zero.
- TOST on the eff_candidates ratio (matched/mismatched − 1), margin
  10% relative, at each world's own challenge level t*.
- The same Δ test repeated on `contam_worst` (ε-contamination arm)
  when the column is present.
- Dose-response: per-world mean Δ vs |log(κ_p/κ_m)|, Spearman ρ.

Regime atlas (all cases in the run dir):
- winner maps: per metric, the argmax target per world, classified as
  a comparator corner (boltzmann / cvar_only / dro_only / worst) or
  interior; the winning region is the mode over worlds.
- corner dominance: per metric and corner, cluster bootstrap CI of
  (best interior − best corner) per world; a corner is competitive if
  the CI does not exclude zero from above.
- boundary extraction (needs --o1-dir): per world and β row, the
  smallest ρ at which tv_vs_worst < the hardness margin; beyond it
  the family has become the worst-case pole.

Usage:
    python scripts/analyze_o3_asym.py O3_RUN_DIR [--o1-dir O1_RUN_DIR]
        [--margin 0.05] [--tost-margin 0.10] [--out summary.json]
"""

import argparse
import csv
import json
import pathlib
from collections import Counter

import numpy as np

from epgfn.stats import (cluster_bootstrap_ci, paired_sign_permutation,
                         tost)

METRICS = ("sat_mass", "eff_candidates", "stress_mean", "stress_p05")


def _f(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else np.nan


def read_csv(path) -> list[dict]:
    """Load a CSV file's rows as a list of dicts.

    Args:
        path: Path to the CSV file.

    Returns:
        list[dict]: One dict per row, keyed by column header.
    """
    with open(path) as fh:
        return list(csv.DictReader(fh))


def at_t_star(rows: list[dict]) -> list[dict]:
    """Keep only the rows evaluated at each world's own challenge level t*.

    Args:
        rows: O3 CSV rows with numeric "t" and "t_star" columns.

    Returns:
        list[dict]: Rows where t == t_star (within floating-point tolerance).
    """
    return [r for r in rows if abs(_f(r, "t") - _f(r, "t_star")) < 1e-9]


# ---------------------------------------------------------------- part 1

def matched_pairs(rows: list[dict], meta: list[dict]) -> dict:
    """Per world: paired matched-vs-mismatched rows from the risk_asym
    grid. Returns {world_seed: list of (matched_row, mismatched_row)}."""
    kappa = {m["world_seed"]: (float(m["kappa_p"]), float(m["kappa_m"]))
             for m in meta}
    cells: dict[str, dict[tuple, dict]] = {}
    for r in at_t_star(rows):
        if r["target"] != "risk_asym":
            continue
        cells.setdefault(r["world_seed"], {})[
            (_f(r, "rho_p"), _f(r, "rho_m"))] = r
    out = {}
    for ws, grid in cells.items():
        kp, km = kappa[ws]
        noisier_p = kp < km   # smaller κ = noisier ⇒ deserves larger ρ
        pairs = []
        for (rp, rm), row in grid.items():
            if rp <= rm:      # visit each unordered pair once, a > b
                continue
            twin = grid.get((rm, rp))
            if twin is None:
                continue
            hi_on_p, hi_on_m = row, twin
            matched, mism = ((hi_on_p, hi_on_m) if noisier_p
                             else (hi_on_m, hi_on_p))
            pairs.append((matched, mism))
        out[ws] = pairs
    return out


def asym_confirmatory(rows, meta, tost_margin: float) -> dict:
    """Run the case B confirmatory test suite: paired matched-vs-mismatched
    Δstress_p05 (sign-permutation test + cluster bootstrap CI), TOST on
    the eff_candidates ratio, the contamination-arm Δ when present, and
    the dose-response check against |log κ ratio|.

    Args:
        rows: o3_caseB.csv rows (all targets, all worlds).
        meta: o3_caseB_worlds.csv rows giving kappa_p/kappa_m per world.
        tost_margin: Relative TOST equivalence margin for the
            eff_candidates ratio.

    Returns:
        dict: Per-world stats plus the pooled sign-permutation, CI,
        TOST, and dose-response results (see module docstring,
        "Confirmatory").
    """
    pairs = matched_pairs(rows, meta)
    kappa = {m["world_seed"]: (float(m["kappa_p"]), float(m["kappa_m"]))
             for m in meta}
    have_contam = any("contam_worst" in r and r["contam_worst"] != ""
                      for r in rows)
    per_world = {}
    n_eff_skipped = 0
    for ws, prs in sorted(pairs.items()):
        d_stress, d_contam, eff_ratio = [], [], []
        for matched, mism in prs:
            d_stress.append(_f(matched, "stress_p05")
                            - _f(mism, "stress_p05"))
            if have_contam:
                d_contam.append(_f(matched, "contam_worst")
                                - _f(mism, "contam_worst"))
            em, ex = _f(matched, "eff_candidates"), _f(mism,
                                                       "eff_candidates")
            if ex > 0:
                eff_ratio.append(em / ex - 1.0)
            else:
                n_eff_skipped += 1
        kp, km = kappa[ws]
        per_world[ws] = {
            "n_pairs": len(prs),
            "kappa_p": kp, "kappa_m": km,
            "abs_log_kappa_ratio": float(abs(np.log(kp / km))),
            "mean_delta_stress_p05": float(np.mean(d_stress)),
            "mean_delta_contam_worst": (float(np.mean(d_contam))
                                        if d_contam else None),
            "eff_ratio_minus_1": [float(x) for x in eff_ratio]}
    deltas = np.array([w["mean_delta_stress_p05"]
                       for w in per_world.values()])
    out = {"n_worlds": len(per_world),
           "n_eff_pairs_skipped_zero_mismatched": n_eff_skipped,
           "per_world": per_world,
           "delta_stress_p05": {
               **paired_sign_permutation(deltas),
               "ci": cluster_bootstrap_ci(
                   {w: np.array([v["mean_delta_stress_p05"]])
                    for w, v in per_world.items()})},
           "eff_ratio_tost": tost(
               {w: np.array(v["eff_ratio_minus_1"])
                for w, v in per_world.items()
                if v["eff_ratio_minus_1"]}, margin=tost_margin)}
    if have_contam:
        d_c = np.array([w["mean_delta_contam_worst"]
                        for w in per_world.values()])
        out["delta_contam_worst"] = paired_sign_permutation(d_c)
    # dose-response: Δ increases with the true asymmetry
    from scipy import stats as sps
    asym = np.array([w["abs_log_kappa_ratio"]
                     for w in per_world.values()])
    rho, p = sps.spearmanr(asym, deltas)
    out["dose_response"] = {"spearman_rho": float(rho),
                            "p_value": float(p),
                            "x_abs_log_kappa_ratio": asym.tolist(),
                            "y_mean_delta_stress_p05": deltas.tolist()}
    return out


# ---------------------------------------------------------------- part 2

def classify(row: dict) -> str:
    """Label a row's target as a comparator corner or an interior point.

    Args:
        row: One O3 CSV row (any case).

    Returns:
        str: One of "boltzmann", "worst", "cvar_only", "dro_only",
        "interior", or "interior_asym".
    """
    kind = row["target"]
    if kind == "boltzmann":
        return "boltzmann"
    if kind == "worst":
        return "worst"
    if kind == "risk_asym":
        rp, rm = _f(row, "rho_p"), _f(row, "rho_m")
        return "interior_asym" if rp != rm else "interior"
    b, r = _f(row, "beta_cvar"), _f(row, "rho")
    if b == 1.0 and r == 0.0:
        return "boltzmann"
    if r == 0.0:
        return "cvar_only"
    if b == 1.0:
        return "dro_only"
    return "interior"


def _label(row: dict) -> str:
    kind = row["target"]
    if kind == "risk":
        return f"risk(b={_f(row, 'beta_cvar'):g},rho={_f(row, 'rho'):g})"
    if kind == "risk_asym":
        return (f"asym(rho_p={_f(row, 'rho_p'):g},"
                f"rho_m={_f(row, 'rho_m'):g})")
    return kind


def regime_atlas(rows: list[dict]) -> dict:
    """Build the exploratory regime atlas: per-metric winner maps and
    corner-dominance CIs, evaluated at each world's own t*.

    Args:
        rows: O3 CSV rows for one case (all targets, all worlds).

    Returns:
        dict: Per metric, the winning-region mode, the region counts,
        the per-world winner labels, and the per-corner dominance CIs
        (see module docstring, "Regime atlas").
    """
    star = at_t_star(rows)
    by_world: dict[str, list[dict]] = {}
    for r in star:
        by_world.setdefault(r["world_seed"], []).append(r)
    out = {}
    corners = ("boltzmann", "cvar_only", "dro_only", "worst")
    for metric in METRICS:
        winners, dom = [], {c: {} for c in corners}
        for ws, rs in sorted(by_world.items()):
            vals = [(_f(r, metric), r) for r in rs
                    if np.isfinite(_f(r, metric))]
            if not vals:
                continue
            best_val, best_row = max(vals, key=lambda t: t[0])
            winners.append({"world_seed": ws,
                            "winner": _label(best_row),
                            "region": classify(best_row),
                            "value": best_val})
            interior = [v for v, r in vals
                        if classify(r).startswith("interior")]
            if not interior:
                continue
            for c in corners:
                cv = [v for v, r in vals if classify(r) == c]
                if cv:
                    dom[c][ws] = np.array([max(interior) - max(cv)])
        regions = Counter(w["region"] for w in winners)
        corner_dom = {}
        for c in corners:
            if not dom[c]:
                continue
            mean, lo, hi = cluster_bootstrap_ci(dom[c])
            corner_dom[c] = {"best_interior_minus_best_corner": mean,
                             "ci_lo": lo, "ci_hi": hi,
                             # corner within CI of the best interior
                             "corner_competitive": bool(lo <= 0.0)}
        out[metric] = {"winning_region_mode":
                       regions.most_common(1)[0][0] if regions else None,
                       "region_counts": dict(regions),
                       "per_world_winner": winners,
                       "corner_dominance": corner_dom}
    return out


# ---------------------------------------------------------------- part 3

def rho_boundaries(o1_rows: list[dict], margin: float) -> dict:
    """Per world: smallest ρ with tv_vs_worst < margin, per β row and
    the per-world min over β rows (NaN → never leaves the useful range
    on this grid)."""
    per: dict[str, dict[float, float]] = {}
    for r in o1_rows:
        v = _f(r, "tv_vs_worst")
        if not np.isfinite(v):
            continue
        b, rho = _f(r, "beta_cvar"), _f(r, "rho")
        if v < margin:
            cur = per.setdefault(r["world_seed"], {})
            cur[b] = min(cur.get(b, np.inf), rho)
    out = {}
    for ws, by_beta in sorted(per.items()):
        out[ws] = {"per_beta": {f"{b:g}": (None if not np.isfinite(x)
                                           else x)
                                for b, x in sorted(by_beta.items())},
                   "min_rho_at_worst": (min(by_beta.values())
                                        if by_beta else None)}
    return out


def main() -> None:
    """Parse CLI arguments, run the case B confirmatory test and the
    regime atlas for every case (plus the optional ρ-boundary
    extraction), write the JSON summary, and print a human-readable
    report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("o3_dir", help="run dir from scripts/run_o3.py")
    ap.add_argument("--o1-dir", default=None,
                    help="run dir from scripts/run_o1.py for the "
                    "ρ-boundary extraction (atlas §3.3)")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="hardness margin for the ρ boundary")
    ap.add_argument("--tost-margin", type=float, default=0.10,
                    help="TOST relative margin on the eff ratio")
    ap.add_argument("--out", default=None,
                    help="JSON output path (default: "
                    "<o3_dir>/o3_asym_analysis.json)")
    args = ap.parse_args()

    o3_dir = pathlib.Path(args.o3_dir)
    summary = {"o3_dir": str(o3_dir), "o1_dir": args.o1_dir,
               "atlas": {}, "boundary": {}}
    for path in sorted(o3_dir.glob("o3_case?.csv")):
        case = path.stem[-1]
        rows = read_csv(path)
        summary["atlas"][case] = regime_atlas(rows)
        if case == "B":
            meta = read_csv(o3_dir / "o3_caseB_worlds.csv")
            if meta and "kappa_p" in meta[0]:
                summary["confirmatory_B"] = asym_confirmatory(
                    rows, meta, args.tost_margin)
    if args.o1_dir:
        for path in sorted(pathlib.Path(args.o1_dir).glob("o1_case?.csv")):
            summary["boundary"][path.stem[-1]] = rho_boundaries(
                read_csv(path), args.margin)

    out = pathlib.Path(args.out) if args.out else (
        o3_dir / "o3_asym_analysis.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    conf = summary.get("confirmatory_B")
    if conf:
        d = conf["delta_stress_p05"]
        print(f"matched-vs-mismatched (B, {conf['n_worlds']} worlds): "
              f"mean Δstress_p05 {d['mean']:+.5f}, sign-perm "
              f"p={d['p_value']:.4f}, d={d['cohens_d']:.2f}")
        if "delta_contam_worst" in conf:
            c = conf["delta_contam_worst"]
            print(f"  contamination arm: mean Δ {c['mean']:+.5f}, "
                  f"p={c['p_value']:.4f}")
        t = conf["eff_ratio_tost"]
        print(f"  eff_candidates ratio TOST: mean {t['mean']:+.4f}, "
              f"equivalent={t['equivalent']} "
              f"(p_lo={t['p_lower']:.4f}, p_hi={t['p_upper']:.4f})")
        dr = conf["dose_response"]
        print(f"  dose-response: Spearman rho={dr['spearman_rho']:.3f} "
              f"(p={dr['p_value']:.4f})")
    for case, atlas in summary["atlas"].items():
        wins = {m: a["winning_region_mode"] for m, a in atlas.items()}
        print(f"atlas case {case}: winners {wins}")
    print(f"summary -> {out}")


if __name__ == "__main__":
    main()
