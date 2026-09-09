"""Per-case figure: the exact target vs what each training arm reaches.

Six panels: the analytic target p* at the w33 fixed condition (the
condition every arm was trained on) and the five arms' exact
terminating densities (dens_*.npy saved by the density patch; these
store grid log P_F: exp before use). The 4^8 sequence space renders
as 256x256 (first four coordinates x last four, canonical order), log
color. Each arm panel annotates its exact L1 to the unclamped target
(matches the recorded final_l1 to the digit).

Default figure-cell worlds: A/B/C -> w0, D -> w59 (the gated D world
with all five arms' densities on disk).

Usage:
    python scripts/plot_target_vs_arms.py --case B [--world-seed 0] \
        [--root results] [--out results/figures]
"""
import argparse
import glob
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: F401, E402
from matplotlib.colors import LogNorm  # noqa: E402

from epgfn.cases import log_reward  # noqa: E402
from epgfn.target import l1, p_star  # noqa: E402
from epgfn.w33 import ARMS, fixed_condition, mode_indices  # noqa: E402
from epgfn.worlds import WorldConfig, make_world  # noqa: E402

CASE_TITLE = {"A": "smooth", "B": "floor", "C": "veto", "D": "nested"}
DEFAULT_WORLD = {"A": 0, "B": 0, "C": 0, "D": 59}


def main():
    """Render the six-panel target-vs-arms density comparison figure.

    Parses CLI args, builds the analytic target at the case's fixed
    condition, loads each arm's saved terminating density
    (dens_*.npy), and writes one PNG+PDF figure with the target panel
    plus one panel per arm annotated with its exact L1 to the target.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="B", choices=list("ABCD"))
    ap.add_argument("--world-seed", type=int, default=None,
                    help="raw world seed (default: figure-cell world)")
    ap.add_argument("--sparsity", type=float, default=1.0)
    ap.add_argument("--root", default="results",
                    help="tree searched recursively for dens_*.npy")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    case = args.case
    ws = args.world_seed if args.world_seed is not None \
        else DEFAULT_WORLD[case]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = WorldConfig(H=4, d=8, geometry="sequence",
                      sparsity=args.sparsity)
    world = make_world(case, cfg, ws)
    cond = fixed_condition(world)
    target = p_star(log_reward(world, cond), cond.beta_t)
    modes = mode_indices(world)

    def grid(v):
        return v.reshape(256, 256)

    panels = [("exact target $p^*$", target, None)]
    for arm in ARMS:
        hits = []
        # any training seed will do; use the first one with a saved density
        for seed in (0, 1, 2):
            hits = glob.glob(
                f"{args.root}/**/dens_{case}_s{args.sparsity}"
                f"_w{ws}_{arm}_{seed}.npy",
                recursive=True)
            if hits:
                break
        if not hits:
            print(f"missing density for {arm}; skipped")
            continue
        q = np.exp(np.load(hits[0]).astype(np.float64)).reshape(-1)
        panels.append((arm, q, l1(q, target)))

    vmax = max(p[1].max() for p in panels)
    vmin = vmax * 1e-6
    with plt.style.context(["science", "no-latex"]):
        plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})
        fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.4),
                                 constrained_layout=True)
        for (title, v, dist), ax in zip(panels, axes.ravel()):
            im = ax.imshow(np.maximum(grid(v), vmin), norm=LogNorm(
                vmin=vmin, vmax=vmax), cmap="viridis",
                interpolation="nearest")
            ax.set_title(title if dist is None
                         else f"{title}  (L1 = {dist:.3f})")
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        fig.colorbar(im, ax=axes, shrink=0.8,
                     label="probability (log scale)")
        stag = "" if args.sparsity == 1.0 else f", s={args.sparsity:g}"
        fig.suptitle(f"Case {case} ({CASE_TITLE[case]}): target vs "
                     f"what each arm reaches (w{ws}{stag}, fixed condition)",
                     fontsize=10)
        suffix = "" if args.sparsity == 1.0 else f"_s{args.sparsity:g}"
        p = out / f"{case.lower()}_target_vs_arms{suffix}.png"
        fig.savefig(p, dpi=300)
        fig.savefig(p.with_suffix(".pdf"))
        print(f"wrote {p} (+.pdf)")
        print(f"live states: {(~np.isclose(target, 0)).sum()} of "
              f"{target.size}; modes at t*: {len(modes)}")


if __name__ == "__main__":
    main()
