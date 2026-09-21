"""Extract the measured behaviour of the hardness gate from the results tree.

The gate criteria themselves live in `epgfn.worlds` (`hardness_report`,
`is_hard`) and are read from there, so the thresholds reported here cannot
drift from the ones that ran. Everything else is measured:

  accepted worlds   every `*_worlds.csv` records one row per world that
                    passed, with its attempt count and the probe values at
                    acceptance
  attempt counts    every training CSV carrying a `gate_attempts` column
  gate-off draws    rows with `gated=False` (the ungated ablation) record
                    `is_hard` on the FIRST draw — an unbiased estimate of
                    the gate's acceptance probability, unlike the accepted
                    rows, which are conditioned on passing
  refusals          `probe_gate.py` logs, which enumerate every rejected
                    draw and so are the only place the binding criterion
                    is directly observable

Rejected draws are not recorded by the run scripts (only their count), so
outside the probe logs the binding criterion is an inference from attempt
counts. This script says which is which rather than blurring them.

Writes report/gate_stats.json and report/gate_stats.md.

Usage:
    python scripts/gate_stats.py [--results results] [--out report]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from epgfn.worlds import WorldConfig  # noqa: E402

CFG = WorldConfig()

# criterion -> (probe column, comparison, threshold, prose)
CRITERIA = {
    "tv_on_off": (
        "probe_tv_on_off", ">=", CFG.hardness_margin,
        "risk dials must move the target: TV between the risk-on and "
        "risk-off targets at the probe condition"),
    "floor_fail_frac_lo": (
        "probe_floor_fail_frac", ">=", CFG.floor_frac_range[0],
        "case B: the floor must exclude a non-trivial share of X"),
    "floor_fail_frac_hi": (
        "probe_floor_fail_frac", "<=", CFG.floor_frac_range[1],
        "case B: the floor must not exclude almost all of X"),
    "tv_origins": (
        "probe_tv_origins", ">=", CFG.hardness_margin,
        "cases B, D: the origins must be separately addressable"),
    "veto_frac": (
        "probe_veto_frac", ">=", CFG.veto_frac_range[0],
        "case C: the veto must bind somewhere"),
    "veto_frac_delta_max": (
        "probe_veto_frac_delta_max", "<=", CFG.veto_frac_range[1],
        "case C: at the margin endpoint delta_max the veto must not "
        "swallow X (monotone in delta, so the two endpoints bound the "
        "whole band)"),
    "tv_nested_flat": (
        "probe_tv_nested_flat", ">=", CFG.hardness_margin,
        "case D: the nested aggregation must not collapse to its flat "
        "form"),
    "tv_beta_out": (
        "probe_tv_beta_out", ">=", CFG.hardness_margin,
        "case D: the joint-satisfaction dial must matter (compensatory "
        "vs conjunctive outer level)"),
}

# which criteria apply to which case (mirrors worlds.is_hard)
PER_CASE = {
    "A": ["tv_on_off"],
    "B": ["tv_on_off", "floor_fail_frac_lo", "floor_fail_frac_hi",
          "tv_origins"],
    "C": ["tv_on_off", "veto_frac", "veto_frac_delta_max"],
    "D": ["tv_on_off", "tv_nested_flat", "tv_origins", "tv_beta_out"],
}


def family_of(path: Path, results: Path) -> str:
    rel = path.relative_to(results).parts
    return rel[1] if rel[0] == "results" else rel[0]


def q(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
    return xs[i]


def summarize(xs: list[float]) -> dict:
    return {"n": len(xs), "min": min(xs), "q25": q(xs, 0.25),
            "median": st.median(xs), "q75": q(xs, 0.75), "max": max(xs),
            "mean": st.fmean(xs)}


def read_worlds(results: Path) -> tuple[dict, dict]:
    """Accepted-world records, grouped by (family, case) and by case."""
    by_key = defaultdict(list)
    for path in sorted(results.rglob("*_worlds.csv")):
        fam = family_of(path, results)
        for row in csv.DictReader(open(path)):
            row["_family"] = fam
            row["_file"] = path.relative_to(ROOT).as_posix()
            by_key[(fam, row["case"])].append(row)
    by_case = defaultdict(list)
    for (_fam, case), rows in by_key.items():
        by_case[case].extend(rows)
    return dict(by_key), dict(by_case)


def gate_block(rows: list[dict]) -> dict:
    """Attempt statistics and probe distributions for one group."""
    gated = [r for r in rows if r.get("gated", "True") == "True"]
    ungated = [r for r in rows if r.get("gated", "True") == "False"]
    out = {"n_worlds": len(gated), "n_ungated_draws": len(ungated)}
    if gated:
        att = [int(r["attempts"]) for r in gated]
        out["attempts"] = summarize([float(a) for a in att])
        # rejection sampling: attempts ~ Geometric(p), so the MLE of the
        # acceptance probability is (number of accepted) / (total draws)
        out["acceptance_rate_mle"] = len(att) / sum(att)
        out["max_attempts_cap"] = CFG.max_attempts
    if ungated:
        # unconditional: is_hard evaluated on the first draw, no rejection
        hard = sum(1 for r in ungated if r["is_hard"] == "True")
        out["ungated_pass_rate"] = hard / len(ungated)
    case = rows[0]["case"]
    probes = {}
    for name in PER_CASE.get(case, []):
        col, cmp_, thr, prose = CRITERIA[name]
        vals = [float(r[col]) for r in gated if r.get(col) not in (None, "")]
        if not vals:
            continue
        s = summarize(vals)
        # headroom: how far the accepted worlds sit from the threshold,
        # in units of the threshold. Near 1 means the criterion binds.
        s["threshold"] = thr
        s["comparison"] = cmp_
        s["headroom_median"] = (s["median"] / thr if cmp_ == ">="
                                else thr / s["median"] if s["median"] else
                                float("inf"))
        s["share_within_20pct_of_threshold"] = sum(
            1 for v in vals
            if (cmp_ == ">=" and v < 1.2 * thr)
            or (cmp_ == "<=" and v > 0.8 * thr)) / len(vals)
        s["criterion"] = prose
        probes[name] = s
    out["probes"] = probes
    return out


def read_attempt_columns(results: Path) -> dict:
    """`gate_attempts` from the training CSVs — attempt counts only, but
    they cover the batteries that write no worlds file."""
    by_key = defaultdict(list)
    for path in sorted(results.rglob("*.csv")):
        if path.name.endswith("_worlds.csv"):
            continue
        try:
            with open(path) as fh:
                rdr = csv.DictReader(fh)
                if not rdr.fieldnames or "gate_attempts" not in rdr.fieldnames:
                    continue
                fam = family_of(path, results)
                seen = set()
                for row in rdr:
                    # one world appears on many rows (arms x seeds)
                    key = (row.get("case"), row.get("sparsity"),
                           row.get("world"))
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        a = int(row["gate_attempts"])
                    except (ValueError, TypeError):
                        continue
                    by_key[(fam, row.get("case"),
                            row.get("sparsity"))].append(a)
        except (OSError, csv.Error):
            continue
    out = {}
    for (fam, case, sp), att in sorted(by_key.items()):
        k = f"{fam}/{case}" + (f"/s{sp}" if sp else "")
        out[k] = {"n_worlds": len(att),
                  "attempts": summarize([float(a) for a in att]),
                  "acceptance_rate_mle": len(att) / sum(att)}
    return out


LOG_HEAD = re.compile(r"^(\w)\s+s=([\d.]+)\s+seed\s+(\d+):\s+(\d+)/(\d+)"
                      r"\s+attempts pass")
LOG_ROW = re.compile(r"^\s*(\w+)>=?<?=?margin.*threshold\s+([\d.]+)\s*\|"
                     r"\s*median\s+([\d.]+)\s+max\s+([\d.]+)\s+min\s+"
                     r"([\d.]+)\s+fail_share\s+([\d.]+)")


def read_probe_logs(results: Path) -> list[dict]:
    """probe_gate.py refusal logs: the only record of rejected draws."""
    out = []
    for path in sorted(results.glob("*.log")):
        text = path.read_text(errors="replace")
        if "attempts pass" not in text:
            continue
        cur = None
        for line in text.splitlines():
            m = LOG_HEAD.match(line)
            if m:
                case, sp, seed, npass, ntot = m.groups()
                cur = {"file": path.relative_to(ROOT).as_posix(),
                       "case": case, "sparsity": float(sp),
                       "seed": int(seed), "n_pass": int(npass),
                       "n_draws": int(ntot), "criteria": {}}
                out.append(cur)
                continue
            m = LOG_ROW.match(line)
            if m and cur is not None:
                name, thr, med, mx, mn, fail = m.groups()
                cur["criteria"][name] = {
                    "threshold": float(thr), "median": float(med),
                    "max": float(mx), "min": float(mn),
                    "fail_share": float(fail)}
    for rec in out:
        # a criterion that fails on EVERY draw, with its best draw still
        # short of the threshold, is structural: more attempts cannot help
        rec["binding"] = sorted(
            (n for n, c in rec["criteria"].items()
             if c["fail_share"] >= 1.0),
            key=lambda n: rec["criteria"][n]["max"])
        rec["verdict"] = "structural" if rec["binding"] else "thin-tail"
    return out


def md_tables(by_case: dict, by_family: dict, attempts: dict,
              refusals: list) -> str:
    L = ["# Hardness gate: measured behaviour",
         "",
         "Generated by `scripts/gate_stats.py`. Thresholds are read from "
         "`epgfn.worlds.WorldConfig`; everything else is measured from "
         "`results/`.",
         "",
         "## Criteria (from `epgfn.worlds.is_hard`)",
         "",
         "| Case | Criterion | Test | Threshold | What it enforces |",
         "|---|---|---|---|---|"]
    for case, names in PER_CASE.items():
        for name in names:
            col, cmp_, thr, prose = CRITERIA[name]
            L.append(f"| {case} | `{col.removeprefix('probe_')}` | "
                     f"`{cmp_}` | {thr:g} | {prose} |")
    L += ["",
          f"Worlds are rejection-sampled, at most "
          f"`max_attempts = {CFG.max_attempts}` draws per world; the "
          f"attempt count is recorded per world and never silently "
          f"absorbed. Seeds are derived as "
          f"`seed * 100_003 + attempt`, so the draw sequence is "
          f"reproducible.",
          "",
          "## Acceptance, pooled per case",
          "",
          "| Case | Worlds | Attempts mean | median | max | Acceptance "
          "rate (MLE) | Ungated first-draw pass |",
          "|---|---|---|---|---|---|---|"]
    for case in sorted(by_case):
        b = by_case[case]
        a = b.get("attempts")
        if not a:
            continue
        ung = (f"{b['ungated_pass_rate']:.2f} "
               f"(n={b['n_ungated_draws']})"
               if "ungated_pass_rate" in b else "---")
        L.append(f"| {case} | {b['n_worlds']} | {a['mean']:.2f} | "
                 f"{a['median']:.0f} | {a['max']:.0f} | "
                 f"{b['acceptance_rate_mle']:.3f} | {ung} |")
    L += ["",
          "The acceptance rate is the maximum-likelihood estimate for "
          "geometric attempt counts, accepted / total draws. The last "
          "column is the independent ungated estimate: the ablation that "
          "skips rejection records `is_hard` on the first draw.",
          "",
          "## Probe values at acceptance, pooled per case",
          "",
          "These are conditioned on passing, so they describe the worlds "
          "the study ran on, not the draw distribution. `headroom` is the "
          "median in units of its threshold; `near` is the share of "
          "accepted worlds within 20% of the threshold — a criterion with "
          "high `near` is the one doing the work.",
          "",
          "| Case | Probe | Test | Threshold | min | median | max | "
          "headroom | near |",
          "|---|---|---|---|---|---|---|---|---|"]
    for case in sorted(by_case):
        for name, s in by_case[case].get("probes", {}).items():
            L.append(
                f"| {case} | `{name}` | `{s['comparison']}` | "
                f"{s['threshold']:g} | {s['min']:.4f} | "
                f"{s['median']:.4f} | {s['max']:.4f} | "
                f"{s['headroom_median']:.1f}x | "
                f"{s['share_within_20pct_of_threshold']:.2f} |")
    L += ["", "## Acceptance per family", "",
          "| Family / case | Worlds | Attempts mean | max | Acceptance |",
          "|---|---|---|---|---|"]
    for key in sorted(by_family):
        b = by_family[key]
        a = b.get("attempts")
        if not a:
            continue
        L.append(f"| {key} | {b['n_worlds']} | {a['mean']:.2f} | "
                 f"{a['max']:.0f} | {b['acceptance_rate_mle']:.3f} |")
    if attempts:
        L += ["", "### From training CSVs (`gate_attempts` column)", "",
              "Batteries that write no worlds file still record the "
              "attempt count per world.", "",
              "| Family / case / sparsity | Worlds | Attempts mean | max "
              "| Acceptance |", "|---|---|---|---|---|"]
        for key, b in sorted(attempts.items()):
            a = b["attempts"]
            L.append(f"| {key} | {b['n_worlds']} | {a['mean']:.2f} | "
                     f"{a['max']:.0f} | {b['acceptance_rate_mle']:.3f} |")
    if refusals:
        L += ["", "## Refusals: where the gate cannot be satisfied", "",
              "`probe_gate.py` enumerates every rejected draw, so these "
              "are the only records in which the binding criterion is "
              "observed rather than inferred. A criterion failing on "
              "every draw whose best draw is still short of the "
              "threshold is structural — raising `max_attempts` cannot "
              "help.", ""]
        for rec in refusals:
            L.append(f"**Case {rec['case']}, sparsity {rec['sparsity']:g}, "
                     f"seed {rec['seed']}: {rec['n_pass']}/"
                     f"{rec['n_draws']} draws pass — {rec['verdict']}"
                     + (f", bound by `{'`, `'.join(rec['binding'])}`"
                        if rec["binding"] else "")
                     + f"** (`{rec['file']}`)")
            L += ["", "| Criterion | Threshold | min | median | max | "
                  "fail share |", "|---|---|---|---|---|---|"]
            for name, c in rec["criteria"].items():
                L.append(f"| `{name}` | {c['threshold']:g} | "
                         f"{c['min']:.4f} | {c['median']:.4f} | "
                         f"{c['max']:.4f} | {c['fail_share']:.2f} |")
            L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "report"))
    args = ap.parse_args()
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    by_key, by_case_rows = read_worlds(results)
    by_case = {c: gate_block(rows) for c, rows in by_case_rows.items()}
    by_family = {f"{fam}/{case}": gate_block(rows)
                 for (fam, case), rows in by_key.items()}
    attempts = read_attempt_columns(results)
    refusals = read_probe_logs(results)

    payload = {
        "thresholds": {
            "hardness_margin": CFG.hardness_margin,
            "floor_frac_range": list(CFG.floor_frac_range),
            "veto_frac_range": list(CFG.veto_frac_range),
            "delta_probe": CFG.delta_probe,
            "max_attempts": CFG.max_attempts,
        },
        "criteria": {c: {n: {"column": CRITERIA[n][0],
                             "comparison": CRITERIA[n][1],
                             "threshold": CRITERIA[n][2],
                             "enforces": CRITERIA[n][3]}
                         for n in names}
                     for c, names in PER_CASE.items()},
        "per_case": by_case,
        "per_family": by_family,
        "attempt_columns": attempts,
        "refusals": refusals,
    }
    (out / "gate_stats.json").write_text(json.dumps(payload, indent=2))
    (out / "gate_stats.md").write_text(
        md_tables(by_case, by_family, attempts, refusals))

    n = sum(b.get("n_worlds", 0) for b in by_case.values())
    print(f"{n} accepted worlds across {len(by_family)} family/case "
          f"groups; {len(attempts)} attempt-column groups; "
          f"{len(refusals)} refusal record(s)")
    for case in sorted(by_case):
        b = by_case[case]
        if "attempts" in b:
            print(f"  case {case}: {b['n_worlds']} worlds, "
                  f"acceptance {b['acceptance_rate_mle']:.3f}, "
                  f"attempts max {b['attempts']['max']:.0f}")
    print(f"-> {out/'gate_stats.json'}, {out/'gate_stats.md'}")


if __name__ == "__main__":
    main()
