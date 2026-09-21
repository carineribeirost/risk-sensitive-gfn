"""Length-arm verdict: TB vs SubTB(lambda) paired across the depth axis.

A second reproduction arm alongside the sparsity knob: training-
distribution methods (mix, teacher, replay, contrastive) reproduce
against sparsity, credit-assignment losses reproduce against the
trajectory-LENGTH axis instead. Its activation condition is met by the
existing depth cells (B's cost is monotone in d, C is depth-flat), so
the question is whether SubTB's published benefit appears where TB
strains.

Each cell pairs one TB run dir with one SubTB run dir at the same depth.
Runs are paired inside a cell by (world, train_seed): both arms train on
the SAME worlds with the same seeds, so the per-world difference is the
measurement and worlds stay the unit of replication (seeds nested).

Reported per case and cell:
  - each arm's exact held-out L1 and its ratio to the sampling floor,
    with cluster-bootstrap CIs (worlds as clusters), against the
    pre-registered 3x line;
  - the PAIRED per-world difference (SubTB - TB) in L1 and in
    floor-ratio, tested by sign-flip permutation over worlds;
  - an unpaired cluster-permutation test as the cross-check.

Power note, and the reason the runsheet asks for 8 worlds: the paired
sign test's smallest attainable two-sided p is 2^(1-n_worlds), so at
the 4 worlds of the existing depth cells it bottoms out at 0.125 and
can never clear alpha = 0.05. Such a cell is reported as "underpowered"
when every world moves the same way, never as "no difference". Six
worlds make alpha reachable; eight match the utility batteries.

Direction labels ("subtb better" / "tb better" / "no difference") are
descriptive at the given alpha. The reproduction criteria proper (the
length-axis analogue of the sparsity arm's R1-R3) still need explicit
confirmation before this output is read as a final verdict.

Usage:
    python scripts/analyze_length_arm.py \
        --cell d2 results/o2-dev/DIR results/o2-d2-subtb/DIR \
        --cell d3 results/o2-d3/DIR results/o2-d3-subtb/DIR \
        --cell d4 results/o2-d4/DIR results/o2-d4-subtb/DIR \
        --cell seq-d8 results/seq-o2/DIR results/seq-o2-subtb/DIR \
        --holm --out results/length-arm
"""

import argparse
import csv
import glob
import json
import pathlib

import numpy as np

from epgfn.runio import unique_run_dir
from epgfn.stats import (cluster_bootstrap_ci, cluster_permutation_test,
                         holm, paired_sign_permutation)

O2_LINE = 3.0
# args.json fields that must agree between the two arms of a cell; the
# loss itself is the only difference the comparison is allowed to carry.
CONFIG_KEYS = ("H", "d", "geometry", "worlds", "world_seed0", "seeds",
               "steps", "cond_pool", "logit_floor", "weight_alpha",
               "use_sigma", "use_rho_out", "k_guard", "use_guard_delta")


def _run_args(run_dir: str) -> dict:
    """Top-level args.json of a run dir. Dirs written before --loss
    existed carry no such key and are TB by construction."""
    path = pathlib.Path(run_dir) / "args.json"
    if not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)


def _config_check(tb_dir: str, subtb_dir: str) -> dict:
    a, b = _run_args(tb_dir), _run_args(subtb_dir)
    diffs = {}
    for k in CONFIG_KEYS:
        va, vb = a.get(k), b.get(k)
        if k in a and k in b and va != vb:
            diffs[k] = [va, vb]
    losses = [a.get("loss", "tb"), b.get("loss", "tb")]
    if losses != ["tb", "subtb"]:
        diffs["loss"] = losses
    return {"ok": not diffs, "diffs": diffs,
            "config": {k: a.get(k) for k in CONFIG_KEYS if k in a},
            "loss_kinds": losses}


def _read_arm(run_dir: str) -> dict:
    """{case: {(world, train_seed): (heldout_l1, mc_floor_l1)}}.

    The CSV's own `loss` column is the final loss VALUE, not the loss
    kind; the kind lives in args.json, which is why _config_check reads
    it there and never off the rows.
    """
    out: dict = {}
    pattern = str(pathlib.Path(run_dir) / "**" / "o2_case?.csv")
    for path in sorted(glob.glob(pattern, recursive=True)):
        case = pathlib.Path(path).stem[-1]
        with open(path) as fh:
            for r in csv.DictReader(fh):
                key = (r["world"], r["train_seed"])
                out.setdefault(case, {})[key] = (
                    float(r["heldout_l1"]), float(r["mc_floor_l1"]))
    return out


def _arm_summary(pairs: dict) -> dict:
    """Ratio of means over worlds (the endpoint convention of
    check_endpoints.check_o2) plus a cluster-bootstrap CI on L1."""
    l1: dict = {}
    floor: dict = {}
    for (world, _seed), (v, f) in pairs.items():
        l1.setdefault(world, []).append(v)
        floor.setdefault(world, []).append(f)
    groups = {k: np.array(v) for k, v in l1.items()}
    point, lo, hi = cluster_bootstrap_ci(groups)
    num = float(np.mean([np.mean(v) for v in l1.values()]))
    den = float(np.mean([np.mean(v) for v in floor.values()]))
    return {"l1": num, "l1_ci95": [lo, hi], "floor": den,
            "ratio": num / den, "pass_3x": bool(num / den <= O2_LINE),
            "n_worlds": len(l1), "n_runs": len(pairs)}


def _per_world(pairs: dict, keys) -> dict:
    """world -> array of per-seed values, restricted to `keys`."""
    out: dict = {}
    for key in keys:
        world, _seed = key
        out.setdefault(world, []).append(pairs[key])
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def compare_case(tb_pairs: dict, subtb_pairs: dict,
                 alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired TB vs SubTB on the (world, seed) runs both arms share."""
    keys = sorted(set(tb_pairs) & set(subtb_pairs))
    dropped = sorted(set(tb_pairs) ^ set(subtb_pairs))
    if not keys:
        return {"error": "no (world, train_seed) run is shared by the "
                         "two arms", "dropped": dropped}

    tb_l1 = _per_world({k: tb_pairs[k][0] for k in keys}, keys)
    sub_l1 = _per_world({k: subtb_pairs[k][0] for k in keys}, keys)
    # each run's ratio against its own floor: the paired quantity that
    # survives floors differing slightly between the two arms' draws.
    tb_r = _per_world({k: tb_pairs[k][0] / tb_pairs[k][1]
                       for k in keys}, keys)
    sub_r = _per_world({k: subtb_pairs[k][0] / subtb_pairs[k][1]
                        for k in keys}, keys)

    worlds = sorted(tb_l1)
    d_l1 = [float(sub_l1[w].mean() - tb_l1[w].mean()) for w in worlds]
    d_ratio = [float(sub_r[w].mean() - tb_r[w].mean()) for w in worlds]

    paired_l1 = paired_sign_permutation(d_l1, seed=seed)
    paired_ratio = paired_sign_permutation(d_ratio, seed=seed)
    unpaired = cluster_permutation_test(sub_l1, tb_l1, seed=seed)

    # The sign test cannot return a two-sided p below 2^(1-n_worlds):
    # with 4 worlds its floor is 0.125, so alpha = 0.05 is unreachable
    # however large the effect. Report that as underpowered rather than
    # as an absence of difference.
    n_w = len(worlds)
    min_p = 2.0 ** (1 - n_w)
    same_sign = bool(n_w and (all(v < 0 for v in d_l1)
                              or all(v > 0 for v in d_l1)))
    direction = "subtb better" if paired_l1["mean"] < 0 else "tb better"
    if paired_l1["p_value"] < alpha:
        verdict = direction
    elif min_p >= alpha and same_sign:
        verdict = f"{direction} in every world, underpowered"
    else:
        verdict = "no difference"

    tb_sum = _arm_summary({k: tb_pairs[k] for k in keys})
    sub_sum = _arm_summary({k: subtb_pairs[k] for k in keys})
    return {
        "tb": tb_sum, "subtb": sub_sum,
        "n_worlds": len(worlds), "n_paired_runs": len(keys),
        "dropped_runs": [list(k) for k in dropped],
        "paired_l1": paired_l1, "paired_ratio": paired_ratio,
        "cluster_perm_l1": unpaired,
        "alpha": alpha, "verdict": verdict,
        "power": {"min_attainable_p": min_p,
                  "alpha_reachable": bool(min_p < alpha),
                  "worlds_needed_for_alpha":
                      int(np.ceil(1 - np.log2(alpha))),
                  "consistent_direction": same_sign},
        # the 3x line is the pre-registered endpoint: does the arm that
        # breached it come back under the line?
        "restores_endpoint": bool(not tb_sum["pass_3x"]
                                  and sub_sum["pass_3x"]),
        "per_world": {
            "worlds": worlds,
            "tb_ratio": [float(tb_r[w].mean()) for w in worlds],
            "subtb_ratio": [float(sub_r[w].mean()) for w in worlds],
            "tb_l1": [float(tb_l1[w].mean()) for w in worlds],
            "subtb_l1": [float(sub_l1[w].mean()) for w in worlds],
        },
    }


def analyse(cells, alpha: float = 0.05, seed: int = 0,
            strict: bool = True) -> dict:
    report: dict = {"cells": {}, "alpha": alpha}
    for label, tb_dir, subtb_dir in cells:
        cfg = _config_check(tb_dir, subtb_dir)
        if strict and not cfg["ok"]:
            raise SystemExit(
                f"cell {label}: the two run dirs disagree on "
                f"{sorted(cfg['diffs'])} ({cfg['diffs']}). Fix the "
                f"pairing, or pass --allow-config-mismatch to compare "
                f"them anyway.")
        tb, sub = _read_arm(tb_dir), _read_arm(subtb_dir)
        cases = sorted(set(tb) & set(sub))
        entry = {"tb_dir": tb_dir, "subtb_dir": subtb_dir,
                 "config_check": cfg, "cases": {}}
        for case in cases:
            entry["cases"][case] = compare_case(tb[case], sub[case],
                                                alpha=alpha, seed=seed)
        missing = sorted(set(tb) ^ set(sub))
        if missing:
            entry["cases_in_one_arm_only"] = missing
        report["cells"][label] = entry
    return report


def add_holm(report: dict, alpha: float = 0.05, cases=None) -> dict:
    """Holm over the family of paired tests, the same correction the
    secondary tier uses elsewhere. DECLARE THE FAMILY BEFORE RUNNING:
    all four cases x four cells is a 16-member family, which costs an
    order of magnitude in adjusted p; if the registered question is the
    binding case only, pass cases=["B"]."""
    pvals = {f"{label}:{case}": r["paired_l1"]["p_value"]
             for label, cell in report["cells"].items()
             for case, r in cell["cases"].items()
             if "paired_l1" in r
             and (cases is None or case in cases)}
    if pvals:
        report["holm"] = holm(pvals, alpha=alpha)
        report["holm_family"] = {"size": len(pvals),
                                 "cases": sorted(cases) if cases
                                 else "all"}
    return report


def print_report(report: dict) -> None:
    for label, cell in report["cells"].items():
        cfg = cell["config_check"]
        note = "" if cfg["ok"] else f"  [CONFIG MISMATCH {cfg['diffs']}]"
        print(f"\n== cell {label}{note}")
        print(f"   {cfg['config']}")
        head = (f"{'case':<5}{'TB L1':>9}{'TB/floor':>10}"
                f"{'SubTB L1':>10}{'SubTB/floor':>13}"
                f"{'dL1':>9}{'p':>8}{'d':>8}  verdict")
        print(head)
        for case, r in cell["cases"].items():
            if "error" in r:
                print(f"{case:<5}  {r['error']}")
                continue
            flag = "  <- restores 3x endpoint" if r["restores_endpoint"] \
                else ""
            print(f"{case:<5}{r['tb']['l1']:>9.3f}"
                  f"{r['tb']['ratio']:>10.2f}"
                  f"{r['subtb']['l1']:>10.3f}"
                  f"{r['subtb']['ratio']:>13.2f}"
                  f"{r['paired_l1']['mean']:>9.3f}"
                  f"{r['paired_l1']['p_value']:>8.3f}"
                  f"{r['paired_l1']['cohens_d']:>8.2f}"
                  f"  {r['verdict']}{flag}")
    if "holm" in report:
        fam = report.get("holm_family", {})
        print(f"\n== Holm over the paired family "
              f"(size {fam.get('size')}, cases {fam.get('cases')})")
        for name, h in report["holm"].items():
            print(f"   {name:<14} p={h['p']:.4f} p_adj={h['p_adj']:.4f} "
                  f"reject={h['reject']}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", nargs=3, action="append", required=True,
                    metavar=("LABEL", "TB_DIR", "SUBTB_DIR"),
                    help="one depth cell: a label, the TB run dir and "
                         "the SubTB run dir (repeatable)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0,
                    help="permutation-test seed")
    ap.add_argument("--holm", action="store_true",
                    help="Holm-correct the family of paired tests")
    ap.add_argument("--holm-cases", nargs="+", default=None,
                    metavar="CASE",
                    help="restrict the Holm family to these cases "
                         "(declare it before running; default: all)")
    ap.add_argument("--allow-config-mismatch", action="store_true",
                    help="compare dirs whose args.json disagree "
                         "(recorded in the report either way)")
    ap.add_argument("--out", default="results/length-arm",
                    help="base dir for the report (a unique run dir is "
                         "created inside it)")
    args = ap.parse_args()

    report = analyse([tuple(c) for c in args.cell], alpha=args.alpha,
                     seed=args.seed,
                     strict=not args.allow_config_mismatch)
    if args.holm:
        add_holm(report, alpha=args.alpha, cases=args.holm_cases)
    print_report(report)

    out = unique_run_dir(args.out, args)
    path = out / "length_arm.json"
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
