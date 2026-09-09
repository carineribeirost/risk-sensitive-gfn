"""o3 regime maps on the (rho, beta) plane, NLDL-sized.

The plane IS the Section-2 hierarchy: bottom-left cell = Boltzmann
(level 0), left column = CVaR only (1), bottom row = KL-DRO only (2),
interior = DRO-CVaR (3), top-right = worst-case limit (4). Axes are
oriented so risk INCREASES up (beta 1 -> 0.1) and right (rho 0 -> 1.2):
Boltzmann sits bottom-left, worst-case top-right.

Cells are classified, not shaded continuously: each (beta, rho) cell is
binned by the fraction of the Boltzmann->worst-case gap it closes
(sat_mass / stress_p05), or by the fraction of Boltzmann diversity it
retains (eff_candidates). One pastel single-hue ramp per figure
(sequential job -> one hue, light->dark; colorblind-safe by
construction), white gaps between cells, a shared legend naming every
band, and the pre-named probe cell (beta 0.25, rho 0.5) as a star.

Sized for the NLDL template (a4, 2.5cm margins -> 16cm text width,
10pt): full-width figure* at 6.3in, 7-8pt type.

Usage (--run defaults to the newest run under results/o3-pareto):
    python scripts/plot_o3_surfaces.py --out results/figures
"""
import argparse
import csv
import pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: F401, E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

CASES = [("A", "smooth"), ("B", "floor"), ("C", "veto"), ("D", "nested")]
PROBE = (0.25, 0.5)  # (beta, rho), chosen before the sweep

GAP_BINS = [-np.inf, 0.25, 0.75, 0.95, 1.05, np.inf]
GAP_LABELS = ["Boltzmann-like ($<$25\\% of gap)",
              "transition (25--75\\%)",
              "near worst-case (75--95\\%)",
              "matches worst-case (95--105\\%)",
              "exceeds worst-case ($>$105\\%)"]
DIV_BINS = [-np.inf, 0.70, 0.90, 1.00, np.inf]
DIV_LABELS = ["$<$70\\% of Boltzmann diversity",
              "70--90\\%", "90--100\\%",
              "$\\geq$100\\% (full diversity)"]

BLUES = ["#EDF4FA", "#C6DDF0", "#93BFE0", "#5A9BC9", "#2E6FA3"]
GREENS = ["#EAF6EE", "#BFE3CC", "#83C79B", "#3D9960"]
PURPLES = ["#F2EFF8", "#D6CDED", "#AF9DDA", "#8168C1", "#5646A0"]

FIGS = [
    ("sat_mass", "gap", BLUES,
     "Joint satisfaction: where each case reaches worst-case"),
    ("eff_candidates", "div", GREENS,
     "Diversity retained (effective candidates vs.\\ Boltzmann)"),
    ("stress_p05", "gap", PURPLES,
     "Tail robustness (stress p05): gap to worst-case closed"),
]


def num(x):
    """Parse `x` as a float, tolerating non-numeric CSV cells.

    Args:
        x: raw CSV field value.

    Returns:
        float(x), or None if the value is missing or not a number.
    """
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_case(run, case):
    """Load one case's o3 CSV and grid its risk-family metrics by (beta, rho).

    Args:
        run: directory containing `o3_case{case}.csv`.
        case: case letter ("A"-"D").

    Returns:
        Tuple `(grid, betas, rhos, pole)`: `grid[metric][(beta, rho)]` is
        the list of raw values at that cell (target == "risk" rows
        only); `betas` is sorted descending (row 0 = beta 1 = the
        Boltzmann end); `rhos` is sorted ascending; `pole(target,
        metric)` averages `metric` over rows with the given `target`
        ("boltzmann" or "worst").
    """
    rows = [r for r in csv.DictReader(open(run / f"o3_case{case}.csv"))
            if r["t"] == r["t_star"]]

    def pole(target, metric):
        v = [num(r[metric]) for r in rows if r["target"] == target]
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else float("nan")

    grid = defaultdict(lambda: defaultdict(list))
    betas, rhos = set(), set()
    for r in rows:
        if r["target"] != "risk":
            continue
        b, rho = num(r["beta_cvar"]), num(r["rho"])
        if b is None or rho is None:
            continue
        betas.add(b)
        rhos.add(rho)
        for m, _, _, _ in FIGS:
            v = num(r[m])
            if v is not None:
                grid[m][(b, rho)].append(v)
    # beta DESCENDING bottom->top: row 0 (bottom) = beta 1 = Boltzmann end
    return grid, sorted(betas, reverse=True), sorted(rhos), pole


def band_surface(grid_m, betas, rhos, mode, b_pole, w_pole):
    """Bin one metric's cell means into the regime bands for a single case.

    Args:
        grid_m: `grid[metric]` mapping (beta, rho) to a list of raw
            values, as returned by `load_case`.
        betas: beta values, descending (row axis).
        rhos: rho values, ascending (column axis).
        mode: "gap" bins the fraction of the Boltzmann-to-worst-case
            gap closed at each cell; anything else bins the fraction
            of Boltzmann diversity retained.
        b_pole: metric value at the Boltzmann pole.
        w_pole: metric value at the worst-case pole.

    Returns:
        `(len(betas), len(rhos))` array of band indices (float, NaN
        where the cell has no data), suitable for `pcolormesh`.
    """
    Z = np.full((len(betas), len(rhos)), np.nan)
    for i, b in enumerate(betas):
        for j, rho in enumerate(rhos):
            c = grid_m.get((b, rho))
            if not c:
                continue
            v = sum(c) / len(c)
            if mode == "gap":
                span = w_pole - b_pole
                f = (v - b_pole) / span if abs(span) > 1e-9 else np.nan
                bins = GAP_BINS
            else:
                f = v / b_pole if abs(b_pole) > 1e-9 else np.nan
                bins = DIV_BINS
            if not np.isnan(f):
                # bins[1:-1] drops the -inf/inf sentinels: digitize
                # already treats the values below/above them as the
                # outermost bins, so including them would double up
                Z[i, j] = np.digitize(f, bins[1:-1])
    return Z


def frac_pos(vals, x):
    """Fractional index of x along the grid values (star placement).

    Works for ascending (rhos) and descending (betas) axes; linear
    interpolation between the two nearest grid positions.
    """
    if x in vals:
        return float(vals.index(x))
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    seq = [vals[i] for i in order]
    j = np.clip(np.searchsorted(seq, x), 1, len(seq) - 1)
    a, b = seq[j - 1], seq[j]
    ia, ib = order[j - 1], order[j]
    return ia + (x - a) / (b - a) * (ib - ia)


def latest_run(base="results/o3-pareto"):
    """Find the most recently created o3-pareto run directory.

    Args:
        base: root directory to search for `*/*/o3_caseA.csv`.

    Returns:
        Path to the run directory containing the newest `o3_caseA.csv`.

    Raises:
        SystemExit: if no o3-pareto run is found under `base`.
    """
    runs = sorted(pathlib.Path(base).glob("*/*/o3_caseA.csv"))
    if not runs:
        raise SystemExit(f"no o3-pareto runs under {base}")
    return runs[-1].parent


def main():
    """Render the four regime-map figures and save them.

    Parses CLI args, loads all four cases' o3 CSVs (defaulting to the
    newest run under results/o3-pareto), and writes one PNG+PDF pair
    per metric in FIGS to --out.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help="o3-pareto run dir (default: newest under "
                    "results/o3-pareto)")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    run = pathlib.Path(args.run) if args.run else latest_run()
    out = pathlib.Path(args.out)
    print(f"run dir: {run}")
    out.mkdir(parents=True, exist_ok=True)

    data = {c: load_case(run, c) for c, _ in CASES}

    with plt.style.context(["science", "no-latex"]):
        plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                             "xtick.labelsize": 6.5,
                             "ytick.labelsize": 6.5,
                             "legend.fontsize": 7,
                             "axes.labelsize": 8})
        for metric, mode, ramp, title in FIGS:
            labels = GAP_LABELS if mode == "gap" else DIV_LABELS
            cmap = matplotlib.colors.ListedColormap(ramp[:len(labels)])
            fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.9))
            for (case, motif), ax in zip(CASES, axes.ravel()):
                grid, betas, rhos, pole = data[case]
                b_pole = pole("boltzmann", metric)
                w_pole = pole("worst", metric)
                Z = band_surface(grid[metric], betas, rhos, mode,
                                 b_pole, w_pole)
                X = np.arange(len(rhos) + 1)
                Y = np.arange(len(betas) + 1)
                ax.pcolormesh(X, Y, Z, cmap=cmap, vmin=-0.5,
                              vmax=len(labels) - 0.5,
                              edgecolors="white", linewidth=1.2)
                ax.set_xticks(np.arange(len(rhos)) + 0.5)
                ax.set_xticklabels([f"{r:g}" for r in rhos])
                ax.set_yticks(np.arange(len(betas)) + 0.5)
                ax.set_yticklabels([f"{b:g}" for b in betas])
                ax.tick_params(length=0)
                ax.set_xlabel(r"$\rho$ (weight distrust $\rightarrow$)")
                ax.set_ylabel(r"$\beta$ (tail focus $\rightarrow$)")
                ax.set_title(f"Case {case} ({motif})")
                # hierarchy anchors, neutral ink
                ax.annotate("Boltzmann", (0.5, 0.5),
                            textcoords="offset points", xytext=(2, 2),
                            fontsize=5.5, color="0.35")
                ax.annotate("worst-case", (len(rhos) - 0.5,
                                           len(betas) - 0.5),
                            textcoords="offset points",
                            xytext=(-40, -8), fontsize=5.5, color="0.35")
                # pre-named probe cell
                px = frac_pos(rhos, PROBE[1]) + 0.5
                py = frac_pos(betas, PROBE[0]) + 0.5
                ax.plot(px, py, marker="*", markersize=11,
                        markerfacecolor="white",
                        markeredgecolor="0.15", markeredgewidth=0.9,
                        linestyle="none", zorder=5)
            # labels carry LaTeX escaping for other renderers; strip it
            # since the "no-latex" style context here won't parse it
            handles = [Patch(facecolor=ramp[i], edgecolor="0.8",
                             label=labels[i].replace("\\%", "%")
                             .replace("$<$", "<").replace("$\\geq$", ">=")
                             .replace("--", "-"))
                       for i in range(len(labels))]
            handles.append(Line2D([], [], marker="*", markersize=10,
                                  markerfacecolor="white",
                                  markeredgecolor="0.15",
                                  linestyle="none",
                                  label="probe cell (pre-named)"))
            fig.suptitle(title.replace("\\ ", " "), y=0.995, fontsize=10)
            fig.legend(handles=handles, loc="lower center",
                       ncol=3, frameon=False,
                       bbox_to_anchor=(0.5, -0.005))
            fig.tight_layout(rect=[0, 0.07, 1, 0.97])
            p = out / f"o3_regime_{metric}.png"
            fig.savefig(p, dpi=300)
            fig.savefig(p.with_suffix(".pdf"))
            plt.close(fig)
            print(f"wrote {p} (+.pdf)")


if __name__ == "__main__":
    main()
