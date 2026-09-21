"""Paper figures F1–F5 and LaTeX tables T1/T2/T4. One function per
figure; matplotlib only, PDF output, fontsize 9, single-column
figsize (3.3, 2.5); panels scale that unit. Error bars are
cluster-bootstrap 95% CIs with worlds as clusters.

Confirmatory figures are built from TEST-block run dirs only;
pointing these at dev dirs is for drafting layouts.

Usage examples:
    python scripts/make_figures.py f1 --o1-dir results/test-o1/DIR
    python scripts/make_figures.py f2 --o2-dirs results/test-o2/DIR \
        --tied-dir results/o2-tied/DIR --oracle-dir results/oracle-gap/DIR
    python scripts/make_figures.py f3 \
        --asym results/test-o3/DIR/o3_asym_analysis.json
    python scripts/make_figures.py f4 --grid-o2 DIR --seq-o2 DIR \
        --grid-o1 DIR --seq-o1 DIR [--grid-asym J --seq-asym J]
    python scripts/make_figures.py f5 \
        --w33-csv results/w33/DIR/w33_runs.csv
    python scripts/make_figures.py f6|f7 \
        --recovery-csv results/p2-recovery/DIR/recovery.csv
    python scripts/make_figures.py tables --o1-dir ... --o2-dirs ... \
        --oracle-dir ... --experts-dir ...
"""

import argparse
import csv
import glob
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: E402,F401

from epgfn.stats import cluster_bootstrap_ci  # noqa: E402

# 'bright' = Paul Tol's colorblind-safe palette (user standard);
# 'no-latex' keeps rendering machine-independent (no TeX needed).
plt.style.use(["science", "bright", "no-latex"])
plt.rcParams.update({"font.size": 9,
                     "figure.constrained_layout.use": True})
UNIT = (3.3, 2.5)
OUT = pathlib.Path("results/figures")


def _rows(pattern):
    out = []
    for p in sorted(glob.glob(pattern, recursive=True)):
        with open(p) as fh:
            out += list(csv.DictReader(fh))
    return out


def _ci_over_worlds(rows, key, world_key="world"):
    groups: dict = {}
    for r in rows:
        groups.setdefault(r[world_key], []).append(float(r[key]))
    return cluster_bootstrap_ci(
        {k: np.array(v) for k, v in groups.items()})


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight")
    print(f"wrote {OUT / name}")
    plt.close(fig)


# ---------------------------------------------------------------- F1
def fig_separability(o1_dir):
    """Render F1: the four-panel O1b separability figure (one panel per
    case A-D).

    Args:
        o1_dir: Run dir from scripts/run_o1.py containing
            o1_case?_sep.csv files.
    """
    fig, axes = plt.subplots(2, 2, figsize=(2 * UNIT[0], 2 * UNIT[1]))
    ax = axes[0, 0]
    ax.text(0.5, 0.5, "pooled evidence (A):\nsingle source, N/A",
            ha="center", va="center")
    ax.set_axis_off()

    ax = axes[0, 1]  # B heatmap rho+ x rho- of world-median tv_to_shared
    rows = _rows(str(pathlib.Path(o1_dir) / "**" / "o1_caseB_sep.csv"))
    rows = [r for r in rows if r.get("rho_p") not in (None, "")]
    if rows:
        rps = sorted({float(r["rho_p"]) for r in rows})
        rms = sorted({float(r["rho_m"]) for r in rows})
        grid = np.full((len(rms), len(rps)), np.nan)
        for i, rm in enumerate(rms):
            for j, rp in enumerate(rps):
                vals = [float(r["tv_to_shared"]) for r in rows
                        if float(r["rho_p"]) == rp
                        and float(r["rho_m"]) == rm]
                if vals:
                    grid[i, j] = np.median(vals)
        im = ax.imshow(grid, origin="lower", aspect="auto",
                       extent=[min(rps), max(rps), min(rms), max(rms)])
        fig.colorbar(im, ax=ax, shrink=0.8)
        if np.isfinite(grid).all():
            ax.contour(np.linspace(min(rps), max(rps), grid.shape[1]),
                       np.linspace(min(rms), max(rms), grid.shape[0]),
                       grid, levels=[0.05], colors="w")
    ax.set_xlabel(r"$\rho^+$")
    ax.set_ylabel(r"$\rho^-$")
    ax.set_title("conflicting evidence (B)")

    ax = axes[1, 0]  # C: delta vs tv per rho
    rows = _rows(str(pathlib.Path(o1_dir) / "**" / "o1_caseC_sep.csv"))
    if rows:
        rhos = sorted({float(r["rho"]) for r in rows})
        for rho in rhos:
            ds = sorted({float(r["delta"]) for r in rows
                         if float(r["rho"]) == rho})
            med = [np.median([float(r["tv_to_shared"]) for r in rows
                              if float(r["rho"]) == rho
                              and float(r["delta"]) == d]) for d in ds]
            ax.plot(ds, med, marker="o", ms=2.5,
                    label=rf"$\rho$={rho:g}")
        ax.axhline(0.05, ls=":", lw=0.8, color="gray",
                   label="enlargement margin")
        ax.legend(fontsize=7)
    ax.set_xlabel(r"veto margin $\delta$")
    ax.set_ylabel("distance to tied-dial family (TV)")
    ax.set_title("veto criteria (C)")

    ax = axes[1, 1]  # D: grouped bars per origin
    rows = _rows(str(pathlib.Path(o1_dir) / "**" / "o1_caseD_sep.csv"))
    if rows:
        origins = sorted({r["origin"] for r in rows})
        med = [np.median([float(r["tv_to_shared"]) for r in rows
                          if r["origin"] == o]) for o in origins]
        full = [np.median([float(r["tv_to_shared_full_split"])
                           for r in rows if r["origin"] == o
                           and r.get("tv_to_shared_full_split")])
                for o in origins]
        x = np.arange(len(origins))
        ax.bar(x - 0.2, med, width=0.4, label="one-hot")
        ax.bar(x + 0.2, full, width=0.4, label="full split")
        ax.axhline(0.05, ls=":", lw=0.8, color="gray",
                   label="enlargement margin")
        ax.set_xticks(x, [f"source {o}" for o in origins])
        ax.legend(fontsize=7)
    ax.set_ylabel("distance to tied-dial family (TV)")
    ax.set_title("nested panels (D)")
    _save(fig, "fig_separability.pdf")


# ---------------------------------------------------------------- F2
DIM_AXIS = [("A", 4, "A"), ("C", 5, "C"), ("B-tied", 5.35, "5t"),
            ("B", 6, "B"), ("D", 8, "D")]


def fig_amortization(o2_dirs, tied_dir=None, oracle_dir=None):
    """Render F2: held-out L1 vs number of dials (left) and the training
    gap to the oracle (right).

    Args:
        o2_dirs: Run dirs from scripts/run_o2.py, one point per case.
        tied_dir: Optional run dir for the B-tied comparison point.
        oracle_dir: Optional run dir from scripts/launch_oracle.py for
            the oracle panel.
    """
    fig, axes = plt.subplots(1, 2, figsize=(2 * UNIT[0], UNIT[1]))
    ax = axes[0]
    for label, xpos, case in DIM_AXIS:
        if label == "B-tied":
            if not tied_dir:
                continue
            rows = _rows(str(pathlib.Path(tied_dir)
                             / "**" / "o2_caseB.csv"))
        else:
            rows = []
            for d in o2_dirs:
                rows += _rows(str(pathlib.Path(d)
                                  / "**" / f"o2_case{case}.csv"))
        if not rows:
            continue
        pt, lo, hi = _ci_over_worlds(rows, "heldout_l1")
        fl = np.mean([float(r["mc_floor_l1"]) for r in rows])
        ax.errorbar([xpos], [pt], yerr=[[pt - lo], [hi - pt]],
                    fmt="o", ms=3, capsize=2, color="C0")
        ax.plot([xpos - 0.15, xpos + 0.15], [fl, fl], color="0.35",
                lw=1.4,
                label="sampling floor" if label == "A" else None)
        ax.annotate(label, (xpos, pt), textcoords="offset points",
                    xytext=(4, 3), fontsize=7)
    ax.set_xlabel("number of dials (condition dimension)")
    ax.set_ylabel("exact error (L1), unseen dial settings")
    ax.set_title("One policy serves the whole dial family")
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    if oracle_dir:
        for path in sorted(glob.glob(str(
                pathlib.Path(oracle_dir) / "**"
                / "oracle_gap_case?_summary.json"), recursive=True)):
            s = json.load(open(path))
            fl = s["mc_floor_mean"]
            ax.plot([0, 1], [s["tb"]["mean"] / fl,
                             s["exact_kl"]["mean"] / fl],
                    marker="o", ms=3, label=s["case"])
        ax.axhline(1.0, ls=":", lw=0.8, color="k")
        ax.axhline(3.0, ls="--", lw=0.8, color="gray")
        ax.annotate("pass line ($3\\times$)",
                    (0.02, 3.0), textcoords="offset points",
                    xytext=(2, 3), fontsize=7, color="gray")
        ax.annotate("sampling floor", (0.02, 1.0),
                    textcoords="offset points", xytext=(2, 3),
                    fontsize=7, color="k")
        ax.set_xticks([0, 1], ["trajectory balance", "oracle"])
        ax.set_ylabel("exact error / sampling floor")
        ax.legend(fontsize=7)
        ax.set_title("Training gap to the oracle")
    _save(fig, "fig_amortization.pdf")


# ---------------------------------------------------------------- F3
def fig_utility(asym_json):
    """Render F3: case B's aimed-distrust advantage vs the true
    reliability asymmetry, for the stress and contamination arms.

    Args:
        asym_json: Path to the o3_asym_analysis.json summary from
            analyze_o3_asym.py.
    """
    s = json.load(open(asym_json))
    conf = s["confirmatory_B"]
    per = conf["per_world"]
    fig, axes = plt.subplots(1, 2, figsize=(2 * UNIT[0], UNIT[1]))
    for ax, key, title in ((axes[0], "mean_delta_stress_p05",
                            "correctly aimed − wrongly aimed"),
                           (axes[1], "mean_delta_contam_worst",
                            "contamination arm")):
        xs, ys = [], []
        for w in per.values():
            if key in w:
                xs.append(w["abs_log_kappa_ratio"])
                ys.append(w[key])
        if xs:
            ax.scatter(xs, ys, s=12)
            ax.axhline(0, ls=":", lw=0.8, color="gray")
        if key == "mean_delta_stress_p05" and "dose_response" in conf:
            dr = conf["dose_response"]
            ax.set_title(f"{title}\nSpearman ρ={dr['spearman_rho']:.2f}"
                         f" (p={dr['p_value']:.2f})", fontsize=8)
        else:
            ax.set_title(title, fontsize=8)
        ax.set_xlabel("true reliability asymmetry "
                      r"$|\log \kappa^+/\kappa^-|$")
        ax.set_ylabel("advantage of correctly-aimed distrust\n"
                      r"($\Delta$ tail-stress)")
    _save(fig, "fig_utility.pdf")


# ---------------------------------------------------------------- F4
def _sep_mom(o1_dir, case):
    rows = _rows(str(pathlib.Path(o1_dir) / "**"
                     / f"o1_case{case}_sep.csv"))
    per: dict = {}
    for r in rows:
        w = r.get("world") or r.get("world_seed")
        per.setdefault(w, []).append(float(r["tv_to_shared"]))
    return (float(np.median([max(v) for v in per.values()]))
            if per else np.nan)


def fig_scale(grid_o2, seq_o2, grid_o1, seq_o1,
              grid_asym=None, seq_asym=None):
    """Render F4: separability, amortization, and the case B matched-Δ,
    grid world vs sequence world side by side.

    Args:
        grid_o2: Grid-world run dir from scripts/run_o2.py.
        seq_o2: Sequence-world run dir from scripts/run_o2.py.
        grid_o1: Grid-world run dir from scripts/run_o1.py.
        seq_o1: Sequence-world run dir from scripts/run_o1.py.
        grid_asym: Optional grid-world o3_asym_analysis.json path.
        seq_asym: Optional sequence-world o3_asym_analysis.json path.
    """
    cases = ["A", "B", "C", "D"]
    fig, axes = plt.subplots(1, 3, figsize=(3 * UNIT[0], UNIT[1]))

    ax = axes[0]  # separability median-of-max per case
    for k, (d, lab) in enumerate(((grid_o1, "grid world"),
                                  (seq_o1, "sequence world"))):
        vals = [_sep_mom(d, c) for c in cases]
        ax.bar(np.arange(4) + (k - 0.5) * 0.35, vals, width=0.35,
               label=lab)
    ax.axhline(0.05, ls=":", lw=0.8, color="gray")
    ax.set_xticks(range(4), cases)
    ax.set_ylabel("distance to tied-dial family (TV)")
    ax.set_title("family enlargement", fontsize=8)
    ax.legend(fontsize=7)

    ax = axes[1]  # L1/floor per case
    for k, (d, lab) in enumerate(((grid_o2, "grid world"),
                                  (seq_o2, "sequence world"))):
        vals = []
        for c in cases:
            rows = _rows(str(pathlib.Path(d) / "**"
                             / f"o2_case{c}.csv"))
            if rows:
                l1 = np.mean([float(r["heldout_l1"]) for r in rows])
                fl = np.mean([float(r["mc_floor_l1"]) for r in rows])
                vals.append(l1 / fl)
            else:
                vals.append(np.nan)
        ax.bar(np.arange(4) + (k - 0.5) * 0.35, vals, width=0.35,
               label=lab)
    ax.axhline(3.0, ls=":", lw=0.8, color="gray")
    ax.set_xticks(range(4), cases)
    ax.set_ylabel("exact error / sampling floor")
    ax.set_title("error vs sampling floor", fontsize=8)
    ax.legend(fontsize=7)

    ax = axes[2]  # matched-Δ (B primary)
    vals, labs = [], []
    for j, lab in ((grid_asym, "grid world"),
                   (seq_asym, "sequence world")):
        if j:
            conf = json.load(open(j))["confirmatory_B"]
            vals.append(conf["delta_stress_p05"]["mean"])
            labs.append(lab)
    if vals:
        ax.bar(labs, vals, width=0.5)
        ax.axhline(0, ls=":", lw=0.8, color="gray")
    ax.set_ylabel(r"$\Delta$ tail-stress (B)")
    ax.set_title("aimed-distrust advantage", fontsize=8)
    _save(fig, "fig_scale.pdf")


# ---------------------------------------------------------------- F5
def fig_reproduction(w33_csv):
    """Render F5: per-case fraction of satisfying candidates discovered
    vs target sparsity, one line per exploration arm, plus a bonus
    exact-error panel.

    Args:
        w33_csv: Path to w33_runs.csv from scripts/run_w33.py or
            scripts/launch_w33.py.
    """
    rows = list(csv.DictReader(open(w33_csv)))
    for r in rows:
        r["sparsity"] = float(r["sparsity"])
        r["frac_modes"] = float(r["frac_modes"])
        r["final_l1"] = float(r["final_l1"])
    cases = sorted({r["case"] for r in rows})
    ARM_LABEL = {"onpolicy": "on-policy", "mix": "uniform mix",
                 "replay": "reward replay", "teacher": "adaptive teacher"}
    arms = [a for a in ("onpolicy", "mix", "replay", "teacher")
            if any(r["arm"] == a for r in rows)]
    fig, axes = plt.subplots(1, len(cases) + 1,
                             figsize=((len(cases) + 1) * UNIT[0] * 0.85,
                                      UNIT[1]))
    axes = np.atleast_1d(axes)
    for ci, case in enumerate(cases):
        ax = axes[ci]
        sps = sorted({r["sparsity"] for r in rows if r["case"] == case})
        for arm in arms:
            pts, los, his = [], [], []
            for s in sps:
                sel = [r for r in rows if r["case"] == case
                       and r["arm"] == arm and r["sparsity"] == s]
                pt, lo, hi = _ci_over_worlds(sel, "frac_modes")
                pts.append(pt), los.append(pt - lo), his.append(hi - pt)
            ax.errorbar(sps, pts, yerr=[los, his], marker="o", ms=3,
                        capsize=2, label=ARM_LABEL[arm])
        ax.set_xlabel("target sparsity")
        ax.set_ylabel("satisfying candidates\ndiscovered (fraction)")
        ax.set_title(f"case {case}", fontsize=8)
        if ci == 0:
            ax.legend(fontsize=6)
    ax = axes[-1]  # bonus panel: exact error (subordinate)
    for arm in arms:
        sel = [r for r in rows if r["arm"] == arm]
        sps = sorted({r["sparsity"] for r in sel})
        med = [np.median([r["final_l1"] for r in sel
                          if r["sparsity"] == s]) for s in sps]
        ax.plot(sps, med, marker="o", ms=3, label=ARM_LABEL[arm])
    ax.set_xlabel("target sparsity")
    ax.set_ylabel("exact error (L1)")
    ax.set_title("exact error", fontsize=8)
    _save(fig, "fig_reproduction.pdf")


# ------------------------------------------- F6/F7 (paper 2)
def _rec_rows(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        for k in ("level", "sigma", "sat_mass_true", "tv_true"):
            r[k] = float(r[k])
    return rows


def fig_recovery(recovery_csv, model="crash"):
    """F6: true satisfying mass vs sigma, one line per corruption
    level, naive/winsor/true as reference lines; one panel per case."""
    rows = [r for r in _rec_rows(recovery_csv) if r["model"] in
            (model, "none")]
    cases = sorted({r["case"] for r in rows})
    fig, axes = plt.subplots(1, len(cases),
                             figsize=(len(cases) * UNIT[0] * 0.85,
                                      UNIT[1]))
    axes = np.atleast_1d(axes)
    for ci, case in enumerate(cases):
        ax = axes[ci]
        cr = [r for r in rows if r["case"] == case]
        levels = sorted({r["level"] for r in cr if r["model"] == model})
        for i, lv in enumerate(levels):
            lr = [r for r in cr if r["level"] == lv]
            sigs = sorted({r["sigma"] for r in lr
                           if r["arm"] in ("naive", "robust")})
            med = [np.median([r["sat_mass_true"] for r in lr
                              if r["sigma"] == s
                              and r["arm"] in ("naive", "robust")])
                   for s in sigs]
            ax.plot(sigs, med, marker="o", ms=2.5,
                    label=rf"$\epsilon_s$={lv:g}")
            wm = [r["sat_mass_true"] for r in lr if r["arm"] == "winsor"]
            if wm:
                ax.axhline(np.median(wm), ls="--", lw=0.7, color="gray",
                           label="winsorized" if i == 0 else None)
        tr = [r["sat_mass_true"] for r in cr if r["arm"] == "true"]
        if tr:
            ax.axhline(np.median(tr), ls=":", lw=0.9, color="k",
                       label="true oracle")
        ax.set_xlabel(r"score margin $\sigma$")
        ax.set_ylabel("true satisfying mass")
        ax.set_title(f"case {case}", fontsize=8)
        if ci == 0:
            ax.legend(fontsize=6, title="corruption rate",
                      title_fontsize=6)
    _save(fig, "fig_recovery.pdf")


def fig_fallback(recovery_csv, model="crash"):
    """F7: fallback decision, best-arm true mass vs corruption level:
    crashy-accurate oracle (naive / best-sigma / winsor) vs the
    crude-stable oracle line; the crossing IS the deliverable."""
    rows = _rec_rows(recovery_csv)
    cases = sorted({r["case"] for r in rows})
    fig, axes = plt.subplots(1, len(cases),
                             figsize=(len(cases) * UNIT[0] * 0.85,
                                      UNIT[1]))
    axes = np.atleast_1d(axes)
    for ci, case in enumerate(cases):
        ax = axes[ci]
        cr = [r for r in rows if r["case"] == case]
        levels = sorted({r["level"] for r in cr if r["model"] == model})

        def med(arm, lv=None, best_sigma=False):
            sel = [r for r in cr if r["arm"] == arm
                   and (lv is None or (r["model"] == model
                                       and r["level"] == lv))]
            if not sel:
                return np.nan
            if best_sigma:
                return max(np.median([r["sat_mass_true"] for r in sel
                                      if r["sigma"] == s])
                           for s in {r["sigma"] for r in sel})
            return np.median([r["sat_mass_true"] for r in sel])

        ax.plot(levels, [med("naive", lv) for lv in levels],
                marker="o", ms=2.5, label="naive")
        ax.plot(levels, [med("robust", lv, best_sigma=True)
                         for lv in levels],
                marker="s", ms=2.5, label=r"best $\sigma$")
        ax.plot(levels, [med("winsor", lv) for lv in levels],
                marker="^", ms=2.5, label="winsorized")
        ax.axhline(med("crude"), ls="--", lw=0.9, color="firebrick",
                   label="stable-but-crude oracle")
        ax.set_xlabel(r"oracle failure rate $\epsilon_s$")
        ax.set_ylabel("true satisfying mass")
        ax.set_title(f"case {case}", fontsize=8)
        if ci == 0:
            ax.legend(fontsize=6)
    fig.suptitle("Keep the accurate-but-failing oracle, or fall back?",
                 fontsize=9)
    _save(fig, "fig_fallback.pdf")


# ------------------------------------------------------------ tables
def _tex_table(path, header, rows, caption, label):
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{" + "l" * len(header) + "}",
             r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}",
              r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def tables(o1_dir=None, o2_dirs=(), oracle_dir=None, experts_dir=None):
    """Write LaTeX tables T1 (separability) and T2 (amortization +
    oracle) to results/figures.

    Args:
        o1_dir: Optional run dir from scripts/run_o1.py, for T1.
        o2_dirs: Run dirs from scripts/run_o2.py, for T2.
        oracle_dir: Optional run dir from scripts/launch_oracle.py;
            adds the exact-KL ratio column to T2.
        experts_dir: Currently unused (reserved parameter).
    """
    OUT.mkdir(parents=True, exist_ok=True)
    if o1_dir:  # T1 separability
        rows = [(c, f"{_sep_mom(o1_dir, c):.3f}")
                for c in "BCD"]
        _tex_table(OUT / "tab1_separability.tex",
                   ["case", "sep (median-of-max)"],
                   [("A", "N/A (single origin)")] + rows,
                   "Separability endpoints (margin 0.05).",
                   "tab:sep")
    if o2_dirs:  # T2 amortization + oracle
        body = []
        for c in "ABCD":
            rows = []
            for d in o2_dirs:
                rows += _rows(str(pathlib.Path(d) / "**"
                                  / f"o2_case{c}.csv"))
            if not rows:
                continue
            pt, lo, hi = _ci_over_worlds(rows, "heldout_l1")
            fl = np.mean([float(r["mc_floor_l1"]) for r in rows])
            cells = [c, f"{pt:.3f} [{lo:.3f}, {hi:.3f}]", f"{fl:.3f}",
                     f"{pt / fl:.2f}$\\times$"]
            if oracle_dir:
                js = glob.glob(str(pathlib.Path(oracle_dir) / "**" /
                                   f"oracle_gap_case{c}_summary.json"),
                               recursive=True)
                if js:
                    s = json.load(open(js[0]))
                    cells.append(
                        f"{s['exact_kl']['mean'] / s['mc_floor_mean']:.2f}"
                        "$\\times$")
            body.append(cells)
        hdr = ["case", "held-out L1 [CI]", "floor", "ratio"]
        if oracle_dir:
            hdr.append("exact-KL ratio")
        _tex_table(OUT / "tab2_amortization.tex", hdr, body,
                   "Amortization vs the MC floor (pass rule "
                   "$\\leq 3\\times$).", "tab:amort")


# ---------------------------------------------------------------- F8
def fig_rho_out(o3_dir, o1_dir):
    """ρ_out liveness panel: left: o1 liveness curve,
    tv_to_shared_rho_out vs ρ_out (per-world light
    lines + median, 0.05 margin in neutral gray); right: o3 utility,
    sat_mass vs ρ_out per inner ρ at each world's own t*."""
    fig, axes = plt.subplots(1, 2, figsize=(2 * UNIT[0], UNIT[1]))

    ax = axes[0]
    rows = _rows(str(pathlib.Path(o1_dir) / "**" / "o1_caseD_rhoout.csv"))
    if rows:
        worlds = sorted({r["world_seed"] for r in rows})
        ros = sorted({float(r["rho_out"]) for r in rows})
        med = []
        for w in worlds:
            ys = [float(r["tv_to_shared_rho_out"]) for r in rows
                  if r["world_seed"] == w]
            xs = [float(r["rho_out"]) for r in rows
                  if r["world_seed"] == w]
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order],
                    color="gray", alpha=0.3, lw=0.7)
        for ro in ros:
            med.append(np.median(
                [float(r["tv_to_shared_rho_out"]) for r in rows
                 if float(r["rho_out"]) == ro]))
        ax.plot(ros, med, marker="o", ms=3)
        ax.axhline(0.05, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\rho_{\mathrm{out}}$")
    ax.set_ylabel("TV to shared family")
    ax.set_title(r"D: $\rho_{\mathrm{out}}$ liveness (O1b)")

    ax = axes[1]
    rows = _rows(str(pathlib.Path(o3_dir) / "**" / "o3_caseD.csv"))
    rows = [r for r in rows if r.get("target") == "risk_rho_out"
            and abs(float(r["t"]) - float(r["t_star"])) < 1e-9]
    if rows:
        for ri in sorted({float(r["rho"]) for r in rows}):
            ros = sorted({float(r["rho_out"]) for r in rows
                          if float(r["rho"]) == ri})
            pts, los, his = [], [], []
            for ro in ros:
                sel = [r for r in rows if float(r["rho"]) == ri
                       and float(r["rho_out"]) == ro]
                pt, lo, hi = _ci_over_worlds(sel, "sat_mass",
                                             world_key="world_seed")
                pts.append(pt), los.append(lo), his.append(hi)
            ax.plot(ros, pts, marker="o", ms=2.5,
                    label=rf"$\rho_{{\mathrm{{in}}}}$={ri:g}")
            ax.fill_between(ros, los, his, alpha=0.2, lw=0)
        ax.legend(fontsize=7)
    ax.set_xlabel(r"$\rho_{\mathrm{out}}$")
    ax.set_ylabel(r"satisfying mass at $t^*$")
    ax.set_title(r"D: utility vs $\rho_{\mathrm{out}}$ (O3)")
    _save(fig, "f8_rho_out.pdf")


# ---------------------------------------------------------------- F9
def fig_weight_regime(flat_o1, peaked_o1):
    """Weight-regime atlas panel: per case, median tv_on_off vs ρ,
    one line per β: flat worlds solid, peaked (α = 0.3) dashed, the
    0.05 margin in neutral gray. Balls other than KL in the peaked dir
    are drawn as separate dashed lines labeled by ball."""
    fig, axes = plt.subplots(2, 2, figsize=(2 * UNIT[0], 2 * UNIT[1]))
    for ax, case in zip(axes.flat, "ABCD"):
        for src, ls, tag in ((flat_o1, "-", "flat"),
                             (peaked_o1, "--", "peaked")):
            rows = _rows(str(pathlib.Path(src) / "**"
                             / f"o1_case{case}.csv"))
            rows = [r for r in rows if r.get("tv_on_off")
                    not in ("", "nan", None)]
            for ball in sorted({r.get("ball", "kl") for r in rows}):
                sub = [r for r in rows if r.get("ball", "kl") == ball]
                betas = sorted({float(r["beta_cvar"]) for r in sub})
                b = betas[0]  # deepest admissible tail: the atlas edge
                rhos = sorted({float(r["rho"]) for r in sub
                               if float(r["beta_cvar"]) == b})
                med = [np.median([float(r["tv_on_off"]) for r in sub
                                  if float(r["beta_cvar"]) == b
                                  and float(r["rho"]) == rh])
                       for rh in rhos]
                lbl = (f"{tag} {ball}" if ball != "kl" else tag)
                ax.plot(rhos, med, ls=ls, marker="o", ms=2,
                        label=rf"{lbl} ($\beta$={b:g})")
        ax.axhline(0.05, color="gray", ls=":", lw=0.8)
        ax.set_title(f"Case {case}")
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel("TV(on, off)")
        ax.legend(fontsize=6)
    _save(fig, "f9_weight_regime.pdf")


def fig_length_arm(length_json):
    """Overlay of the two losses on the depth axis: exact error as a
    multiple of the sampling floor, per case, TB against SubTB. Built
    from analyze_length_arm.py's report so the figure and the verdict
    read the same paired runs."""
    rep = json.load(open(length_json))
    cells = list(rep["cells"])
    cases = sorted({c for cell in rep["cells"].values()
                    for c in cell["cases"]})
    fig, axes = plt.subplots(1, len(cases),
                             figsize=(len(cases) * UNIT[0] * 0.62,
                                      UNIT[1]),
                             sharey=True)
    axes = np.atleast_1d(axes)
    xs = np.arange(len(cells))
    ARMS = (("tb_ratio", "TB", "-", "o"),
            ("subtb_ratio", r"SubTB ($\lambda$=0.9)", "--", "s"))

    for ax, case in zip(axes, cases):
        for key, label, ls, marker in ARMS:
            ys, los, his = [], [], []
            for name in cells:
                r = rep["cells"][name]["cases"].get(case)
                if not r or "per_world" not in r:
                    ys.append(np.nan)
                    los.append(np.nan)
                    his.append(np.nan)
                    continue
                pw = r["per_world"]
                groups = {w: np.array([v]) for w, v in
                          zip(pw["worlds"], pw[key])}
                point, lo, hi = cluster_bootstrap_ci(groups)
                ys.append(point)
                los.append(point - lo)
                his.append(hi - point)
            ax.errorbar(xs, ys, yerr=[los, his], ls=ls, marker=marker,
                        ms=3.5, lw=1.0, capsize=2, label=label)
        ax.axhline(3.0, ls=":", lw=0.8, color="gray")
        ax.set_xticks(xs, cells, fontsize=7)
        ax.set_title(f"case {case}", fontsize=8)
        ax.set_xlabel("depth cell", fontsize=8)
    axes[0].set_ylabel("exact error / sampling floor")
    axes[0].legend(fontsize=7)
    _save(fig, "fig_length_arm.pdf")


def main() -> None:
    """Parse CLI arguments and dispatch to the requested figure- or
    table-generating function."""
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("f1")
    p.add_argument("--o1-dir", required=True)
    p = sub.add_parser("f2")
    p.add_argument("--o2-dirs", nargs="+", required=True)
    p.add_argument("--tied-dir", default=None)
    p.add_argument("--oracle-dir", default=None)
    p = sub.add_parser("f3")
    p.add_argument("--asym", required=True)
    p = sub.add_parser("f4")
    for f in ("--grid-o2", "--seq-o2", "--grid-o1", "--seq-o1"):
        p.add_argument(f, required=True)
    p.add_argument("--grid-asym", default=None)
    p.add_argument("--seq-asym", default=None)
    p = sub.add_parser("f5")
    p.add_argument("--w33-csv", required=True)
    for name in ("f6", "f7"):
        p = sub.add_parser(name)
        p.add_argument("--recovery-csv", required=True)
        p.add_argument("--model", default="crash")
    p = sub.add_parser("f8")
    p.add_argument("--o3-dir", required=True)
    p.add_argument("--o1-dir", required=True)
    p = sub.add_parser("f9")
    p.add_argument("--flat-o1", required=True)
    p.add_argument("--peaked-o1", required=True)
    p = sub.add_parser("f10")
    p.add_argument("--length-json", required=True)
    p = sub.add_parser("tables")
    p.add_argument("--o1-dir", default=None)
    p.add_argument("--o2-dirs", nargs="+", default=[])
    p.add_argument("--oracle-dir", default=None)
    p.add_argument("--experts-dir", default=None)
    a = ap.parse_args()
    if a.cmd == "f1":
        fig_separability(a.o1_dir)
    elif a.cmd == "f2":
        fig_amortization(a.o2_dirs, a.tied_dir, a.oracle_dir)
    elif a.cmd == "f3":
        fig_utility(a.asym)
    elif a.cmd == "f4":
        fig_scale(a.grid_o2, a.seq_o2, a.grid_o1, a.seq_o1,
                  a.grid_asym, a.seq_asym)
    elif a.cmd == "f5":
        fig_reproduction(a.w33_csv)
    elif a.cmd == "f6":
        fig_recovery(a.recovery_csv, a.model)
    elif a.cmd == "f7":
        fig_fallback(a.recovery_csv, a.model)
    elif a.cmd == "f8":
        fig_rho_out(a.o3_dir, a.o1_dir)
    elif a.cmd == "f9":
        fig_weight_regime(a.flat_o1, a.peaked_o1)
    elif a.cmd == "f10":
        fig_length_arm(a.length_json)
    elif a.cmd == "tables":
        tables(a.o1_dir, a.o2_dirs, a.oracle_dir, a.experts_dir)


if __name__ == "__main__":
    main()
