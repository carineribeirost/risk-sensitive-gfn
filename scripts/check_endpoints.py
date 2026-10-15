"""Mechanical endpoint verdicts: turn run-dir CSVs/summaries into
PASS/FAIL lines against the fixed margins, so no later session
has to exercise judgment. Margins are frozen:
separability/hardness 0.05; amortization amortization ≤ 3× MC floor; oracle
reference 1.2× floor; TOST 10% relative. Holm (test block only) runs
across the secondary tier: family separability separability per case + amortization floor-ratio
per case; the utility primary endpoint stands alone.

Operational p-value forms (documented here, not new protocol):
- family separability: cluster bootstrap of the median-of-max separability over
  worlds; p = share of resamples below the 0.05 margin.
- amortization: cluster bootstrap of mean L1 and mean floor over worlds;
  p = share of resamples with ratio > 3.

Usage:
    python scripts/check_endpoints.py --separability-dir results/test-separability/... \
        --amortization-dirs results/test-amortization/... [...] \
        --oracle-dirs results/oracle-gap-d3/... [...] \
        --utility-asym results/test-utility/.../utility_asym_analysis.json \
        --exploration-csv results/exploration/.../exploration_runs.csv \
        [--holm]   # apply Holm across the secondary tier (test block)
"""

import argparse
import csv
import glob
import json
import pathlib

import numpy as np

from rsgfn.stats import cluster_permutation_test, holm

SEP_MARGIN = 0.05
AMORTIZATION_LINE = 3.0
ORACLE_LINE = 1.2


def _median_of_max_boot(maxes: dict, margin: float, n_boot: int = 2000,
                        seed: int = 0) -> dict:
    """Cluster bootstrap of the median-of-max separability statistic
    over worlds, shared by check_separability_sep and check_rho_out_sep (which
    differ only in which CSV/column feeds `maxes`).

    Args:
        maxes: world -> that world's max statistic (a single scalar).
        margin: pass/fail threshold (median-of-max >= margin passes).
        n_boot: number of world-level bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        dict: median_of_max, margin, pass, boot_p (smoothed), n_worlds.
    """
    rng = np.random.default_rng(seed)
    keys = list(maxes)
    boots = [float(np.median([maxes[keys[i]] for i in
                              rng.integers(0, len(keys), len(keys))]))
             for _ in range(n_boot)]
    mom = float(np.median(list(maxes.values())))
    # +1/(N+1) smoothing: keeps the bootstrap p-value from landing
    # on an unachievable exact zero
    p = (int(sum(b < margin for b in boots)) + 1) / (n_boot + 1)
    return {"median_of_max": mom, "margin": margin,
            "pass": mom >= margin, "boot_p": p, "n_worlds": len(maxes)}


def check_separability_sep(separability_dir: str) -> dict:
    """Compute the family separability separability PASS/FAIL verdict per case from
    separability_case*_sep.csv files.

    Args:
        separability_dir: Directory tree to search for separability_case?_sep.csv files
            (searched recursively).

    Returns:
        dict: Per case, the median-of-max separability, the 0.05-margin
        verdict, the cluster-bootstrap p-value, and the world count.
    """
    out = {}
    for path in sorted(glob.glob(str(pathlib.Path(separability_dir)
                                     / "**" / "separability_case?_sep.csv"),
                                 recursive=True)):
        case = pathlib.Path(path).stem[len("separability_case")]
        per_world: dict = {}
        with open(path) as fh:
            for r in csv.DictReader(fh):
                w = r.get("world") or r.get("world_seed") or r.get("seed")
                per_world.setdefault(w, []).append(
                    float(r["tv_to_shared"]))
        maxes = {w: max(v) for w, v in per_world.items()}
        out[case] = _median_of_max_boot(maxes, SEP_MARGIN)
    return out


def check_rho_out_sep(separability_dir: str) -> dict:
    """ρ_out liveness: median-of-max tv_to_shared_rho_out over dev
    worlds ≥ the separability margin, same bootstrap form as
    check_separability_sep. Reads the separate separability_case?_rhoout.csv files (case D
    in practice); an empty dict means the probe was not run."""
    out = {}
    for path in sorted(glob.glob(str(pathlib.Path(separability_dir)
                                     / "**" / "separability_case?_rhoout.csv"),
                                 recursive=True)):
        case = pathlib.Path(path).stem[len("separability_case")]
        per_world: dict = {}
        with open(path) as fh:
            for r in csv.DictReader(fh):
                per_world.setdefault(r["world_seed"], []).append(
                    float(r["tv_to_shared_rho_out"]))
        maxes = {w: max(v) for w, v in per_world.items()}
        out[case] = _median_of_max_boot(maxes, SEP_MARGIN)
    return out


def check_amortization(amortization_dir: str) -> dict:
    """Compute the amortization amortization PASS/FAIL verdict per case from
    amortization_case?.csv files.

    Args:
        amortization_dir: Directory tree to search for amortization_case?.csv files
            (searched recursively).

    Returns:
        dict: Per case, the held-out-L1/MC-floor ratio, its bootstrap
        95% CI, the 3x-line verdict, and the bootstrap p-value.
    """
    out = {}
    for path in sorted(glob.glob(str(pathlib.Path(amortization_dir)
                                     / "**" / "amortization_case?.csv"),
                                 recursive=True)):
        case = pathlib.Path(path).stem[-1]
        l1s: dict = {}
        floors: dict = {}
        with open(path) as fh:
            for r in csv.DictReader(fh):
                l1s.setdefault(r["world"], []).append(
                    float(r["heldout_l1"]))
                floors.setdefault(r["world"], []).append(
                    float(r["mc_floor_l1"]))
        keys = list(l1s)
        rng = np.random.default_rng(0)
        fails = 0
        ratios = []
        for _ in range(2000):
            pick = [keys[i] for i in rng.integers(0, len(keys),
                                                  len(keys))]
            num = np.mean([np.mean(l1s[k]) for k in pick])
            den = np.mean([np.mean(floors[k]) for k in pick])
            ratios.append(num / den)
            fails += num / den > AMORTIZATION_LINE
        ratio = (np.mean([np.mean(v) for v in l1s.values()])
                 / np.mean([np.mean(v) for v in floors.values()]))
        lo, hi = np.quantile(ratios, [0.025, 0.975])
        out[case] = {"ratio": float(ratio), "ci95": [float(lo),
                                                     float(hi)],
                     "line": AMORTIZATION_LINE, "pass": ratio <= AMORTIZATION_LINE,
                     "ci_clear": hi <= AMORTIZATION_LINE,
                     # +1/(N+1) smoothing, same convention as check_separability_sep
                     "boot_p": (fails + 1) / 2001,
                     "n_worlds": len(keys)}
    return out


def check_oracle(oracle_dir: str) -> dict:
    """Compute the oracle-gap capacity verdict per case from
    oracle_gap_case?_summary.json files.

    Args:
        oracle_dir: Directory tree to search for
            oracle_gap_case?_summary.json files (searched recursively).

    Returns:
        dict: Per case, the exact-KL/floor ratio, the 1.2x capacity
        verdict, the TB/floor ratio, the TB-vs-exact-KL gap, and its
        permutation p-value.
    """
    out = {}
    for path in sorted(glob.glob(str(pathlib.Path(oracle_dir) / "**" /
                                     "oracle_gap_case?_summary.json"),
                                 recursive=True)):
        s = json.load(open(path))
        fl = s["mc_floor_mean"]
        ratio = s["exact_kl"]["mean"] / fl
        out[s["case"]] = {
            "kl_ratio": float(ratio), "line": ORACLE_LINE,
            "capacity_ok": ratio <= ORACLE_LINE,
            "tb_ratio": float(s["tb"]["mean"] / fl),
            "gap": s["gap_tb_minus_kl"]["mean"],
            "gap_p": s["permutation"]["p_value"]}
    return out


def check_utility_asym(path: str) -> dict:
    """Extract the utility primary endpoint verdict from an
    utility_asym_analysis.json summary.

    Args:
        path: Path to the JSON file written by analyze_utility_asym.py.

    Returns:
        dict: {"present": False} if case B's confirmatory block is
        absent; otherwise the Δstress_p05 mean/p-value verdict, plus
        the contamination-arm and TOST verdicts when present.
    """
    s = json.load(open(path))
    conf = s.get("confirmatory_B")
    if not conf:
        return {"present": False}
    d = conf["delta_stress_p05"]
    out = {"present": True, "delta_mean": d["mean"],
           "p": d["p_value"],
           "pass": d["mean"] > 0 and d["p_value"] < 0.05}
    if "delta_contam_worst" in conf:
        c = conf["delta_contam_worst"]
        out["contam_p"] = c["p_value"]
        out["contam_pass"] = c["mean"] > 0 and c["p_value"] < 0.05
    if "eff_ratio_tost" in conf:  # key as analyze_utility_asym writes it
        out["tost_equivalent"] = conf["eff_ratio_tost"].get("equivalent")
    return out


def check_exploration(csv_path: str) -> dict:
    """Fixed reproduction criteria R1–R3, evaluated mechanically."""
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        for k in ("sparsity", "frac_modes", "samples_to_80"):
            r[k] = float(r[k])
    sps = sorted({r["sparsity"] for r in rows})
    s_hi, s_lo = max(sps), min(sps)
    out = {}
    for case in sorted({r["case"] for r in rows}):
        def gr(arm, s):
            g: dict = {}
            for r in rows:
                if (r["case"] == case and r["arm"] == arm
                        and r["sparsity"] == s):
                    g.setdefault(r["world"], []).append(r["frac_modes"])
            return {k: np.array(v) for k, v in g.items()}

        t_hi, o_hi = gr("teacher", s_hi), gr("onpolicy", s_hi)
        if not t_hi or not o_hi:
            out[case] = {"present": False}
            continue
        r1 = cluster_permutation_test(t_hi, o_hi)
        adv_hi = r1["diff"]
        res = {"present": True, "s_hi": s_hi, "s_lo": s_lo,
               "R1_teacher_minus_onpolicy_frac_modes": adv_hi,
               "R1_p": r1["p_value"],
               "R1_pass": adv_hi > 0 and r1["p_value"] < 0.05}
        t_lo, o_lo = gr("teacher", s_lo), gr("onpolicy", s_lo)
        if t_lo and o_lo and s_lo != s_hi:
            adv_lo = cluster_permutation_test(t_lo, o_lo)["diff"]
            res["R2_advantage_at_s_lo"] = adv_lo
            res["R2_pass"] = adv_lo < adv_hi
        rep_hi = gr("replay", s_hi)
        if rep_hi:
            rep = float(np.mean([v.mean() for v in rep_hi.values()]))
            on = float(np.mean([v.mean() for v in o_hi.values()]))
            te = float(np.mean([v.mean() for v in t_hi.values()]))
            res["R3_order_ok"] = bool(min(on, te) <= rep <= max(on, te))
        out[case] = res
    return out


def check_recovery(csv_path: str, primary_model: str = "crash",
                   primary_level: float = 0.05) -> dict:
    """At crash eps_s=0.05 there exists sigma>0 on the fixed grid with
    robust > naive true-satisfying-mass, per case. Operational form:
    per-world paired delta (mean over conditions) per sigma; paired
    sign permutation; Holm across the sigma grid; pass if any sigma
    rejects with positive mean. Also reports the winsor and crude
    arms (fallback crossing is exploratory)."""
    from rsgfn.stats import paired_sign_permutation
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        for k in ("level", "sigma", "sat_mass_true", "tv_true"):
            r[k] = float(r[k])
    out = {}
    for case in sorted({r["case"] for r in rows}):
        sel = [r for r in rows if r["case"] == case
               and r["model"] == primary_model
               and r["level"] == primary_level]
        if not sel:
            out[case] = {"present": False}
            continue
        worlds = sorted({r["world"] for r in sel})

        def wmean(arm, sigma=None):
            return {w: np.mean([r["sat_mass_true"] for r in sel
                                if r["world"] == w and r["arm"] == arm
                                and (sigma is None
                                     or r["sigma"] == sigma)])
                    for w in worlds}

        naive = wmean("naive")
        sig_p = {}
        for s in sorted({r["sigma"] for r in sel
                         if r["arm"] == "robust"}):
            rob = wmean("robust", s)
            deltas = [rob[w] - naive[w] for w in worlds]
            t = paired_sign_permutation(deltas)
            sig_p[f"sigma_{s:g}"] = {"mean_delta": t["mean"],
                                     "p": t["p_value"]}
        hp = holm({k: v["p"] for k, v in sig_p.items()})
        passing = [k for k, v in sig_p.items()
                   if hp[k]["reject"] and v["mean_delta"] > 0]
        win = wmean("winsor")
        out[case] = {
            "present": True, "per_sigma": sig_p,
            "P2_primary_pass": bool(passing),
            "passing_sigmas": passing,
            "naive_mean": float(np.mean(list(naive.values()))),
            "winsor_mean_delta": float(np.mean(
                [win[w] - naive[w] for w in worlds])),
            "n_worlds": len(worlds)}
    return out


def main() -> None:
    """Parse CLI arguments, run the requested endpoint checks, optionally
    apply Holm correction across the secondary tier, print the JSON
    report, and print a human-readable summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--separability-dir", default=None)
    ap.add_argument("--amortization-dirs", nargs="+", default=[])
    ap.add_argument("--oracle-dirs", nargs="+", default=[])
    ap.add_argument("--utility-asym", default=None)
    ap.add_argument("--exploration-csv", default=None)
    ap.add_argument("--recovery-csv", default=None,
                    help="paper 2: recovery.csv from run_corruption_recovery.py")
    ap.add_argument("--holm", action="store_true",
                    help="Holm across the secondary tier (test block)")
    ap.add_argument("--out", default=None, help="JSON output path")
    args = ap.parse_args()

    report: dict = {}
    if args.separability_dir:
        report["family_separability"] = check_separability_sep(args.separability_dir)
        ro = check_rho_out_sep(args.separability_dir)  # auto-detected
        if ro:
            report["f1_rho_out_liveness"] = ro
    for d in args.amortization_dirs:
        report.setdefault("amortization", {})[d] = check_amortization(d)
    for d in args.oracle_dirs:
        report.setdefault("oracle", {})[d] = check_oracle(d)
    if args.utility_asym:
        report["utility_primary"] = check_utility_asym(args.utility_asym)
    if args.exploration_csv:
        report["exploration_reproduction"] = check_exploration(args.exploration_csv)
    if args.recovery_csv:
        report["p2_recovery"] = check_recovery(args.recovery_csv)

    if args.holm:
        fam = {}
        for case, r in report.get("family_separability", {}).items():
            fam[f"family_separability_{case}"] = r["boot_p"]
        for d, cases in report.get("amortization", {}).items():
            for case, r in cases.items():
                fam[f"amortization_{case}"] = r["boot_p"]
        if fam:
            report["holm_secondary_tier"] = holm(fam)

    print(json.dumps(report, indent=2, default=str))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
    # human summary
    for sec, body in report.items():
        if sec == "holm_secondary_tier":
            for k, v in body.items():
                print(f"HOLM {k}: p_adj={v['p_adj']:.4f} "
                      f"{'REJECT-H0(pass)' if v['reject'] else 'ns'}")


if __name__ == "__main__":
    main()
