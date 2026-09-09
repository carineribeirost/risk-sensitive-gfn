"""Object-level paper figures: exact targets (fig_targets, all four
cases with an R(z) panel) and learned-policy-vs-target (fig_policy,
cases B and C), one gated dev world each (seed 0).

The policy figure needs training, so its compute runs on a run box
and its DATA is saved (drafts/figures/fig_policy_data.npz) so any
machine can re-render styling without retraining.

Usage (run box; device auto-selects cuda via default_device):
    .venv/bin/python scripts/make_fig_objects.py --which policy
Then copy drafts/figures/fig_policy_data.npz (and/or the PDF) back
and re-render locally if styling changes:
    .venv/bin/python scripts/make_fig_objects.py --which render
Exact-layer targets figure (no training, any machine):
    .venv/bin/python scripts/make_fig_objects.py --which targets
"""

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: E402,F401
from matplotlib.colors import ListedColormap, LogNorm  # noqa: E402

from epgfn.cases import EPS_REWARD, log_reward, target_for  # noqa: E402
from epgfn.conditions import Condition, tied_risk  # noqa: E402
from epgfn.target import l1  # noqa: E402
from epgfn.train import (TrainConfig, default_device, ranges_for,  # noqa: E402
                         train_policy)
from epgfn.worlds import WorldConfig, sample_world  # noqa: E402

plt.style.use(["science", "bright", "no-latex"])
plt.rcParams.update({"font.size": 9})

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = [("B", "conflicting evidence (B)"), ("C", "veto criteria (C)")]
# fig_targets covers the full battery (uniform case coverage); the
# policy figure keeps the two dead-zone cases B/C (training budget)
ALL_CASES = [("A", "pooled evidence (A)")] + CASES + [
    ("D", "nested panels (D)")]
H = 32


def _display_condition(case, ranges):
    """Deterministic held-out member nearest the display setting
    (beta_t = 2, radii 0.8, C's delta mid), machine-independent."""
    def key(c):
        r = c.risk
        if case == "B":
            return (abs(c.beta_t - 2.0), abs(r.rho_p - 0.8),
                    abs(r.rho_m - 0.8), abs(r.beta_p - 0.5))
        return (abs(c.beta_t - 2.0), abs(r.rho - 0.8),
                abs(r.delta - 0.05), abs(r.beta - 0.5))
    return min(ranges.heldout_grid(), key=key)


def fig_targets(out_dir):
    """4 cases × [R(z), p* off, p* on, dead zones]. R spans
    [ε=1e-4, ~1], so the reward panel shares the targets' LogNorm
    treatment; a linear colormap would show nothing but the peak."""
    fig, axes = plt.subplots(4, 4, figsize=(8.6, 8.6),
                             constrained_layout=True)
    for row, (case, label) in enumerate(ALL_CASES):
        world, _ = sample_world(case, WorldConfig(H=H), seed=0)
        lo = world.beta_min
        beta_on = 0.25 if 0.25 >= lo else min(0.99, lo * 1.05)
        p_off = target_for(world, Condition(4.0, 0.3,
                                            tied_risk(case, 1.0, 0.0)))
        cond_on = Condition(4.0, 0.3, tied_risk(case, beta_on, 0.5))
        p_on = target_for(world, cond_on)
        p_off, p_on = p_off.reshape(H, H), p_on.reshape(H, H)
        log_r = log_reward(world, cond_on)
        r_on = np.exp(log_r).reshape(H, H)
        dead = np.isclose(log_r, np.log(EPS_REWARD)).reshape(H, H)

        ax = axes[row, 0]
        imr = ax.imshow(r_on, origin="lower",
                        norm=LogNorm(vmin=EPS_REWARD, vmax=1.0),
                        cmap="viridis", interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_ylabel(label, fontsize=8)
        if row == 0:
            ax.set_title(r"reward $R(z)$ (risk-on)", fontsize=8)

        vmax = max(p_off.max(), p_on.max())
        norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
        for col, (p, title) in enumerate(
                [(p_off, "compensatory pole "
                  r"($\beta{=}1$, $\rho{=}0$)"),
                 (p_on, "risk-on "
                  rf"($\beta{{=}}{beta_on:.2f}$, $\rho{{=}}0.5$)")],
                start=1):
            ax = axes[row, col]
            im = ax.imshow(np.maximum(p, vmax * 1e-4), origin="lower",
                           norm=norm, cmap="viridis",
                           interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=8)
        fig.colorbar(imr, ax=axes[row, 0], shrink=0.85, pad=0.02)
        fig.colorbar(im, ax=axes[row, 2], shrink=0.85, pad=0.02)
        ax = axes[row, 3]
        ax.imshow(dead, origin="lower", interpolation="nearest",
                  cmap=ListedColormap(["0.85", "#994455"]),
                  vmin=0, vmax=1)
        ax.set_xticks([]), ax.set_yticks([])
        if row == 0:
            ax.set_title(r"dead zones ($R{=}\varepsilon$)", fontsize=8)
        ax.text(0.03, 0.95, f"{100 * dead.mean():.0f}% dead",
                transform=ax.transAxes, fontsize=7.5, va="top",
                bbox=dict(fc="white", alpha=0.85, ec="none", pad=1.5))
    out = out_dir / "fig_targets.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def compute_policy_data(out_dir, steps, seed, device):
    """Train the policy for cases B and C and save the target/policy
    densities and L1 metrics to fig_policy_data.npz.

    Args:
        out_dir: Directory to write fig_policy_data.npz into.
        steps: Number of training steps per case.
        seed: Training seed.
        device: Torch device string for training.
    """
    data = {}
    for case, _ in CASES:
        world, _ = sample_world(case, WorldConfig(H=H), seed=0)
        ranges = ranges_for(world)
        cfg = TrainConfig(steps=steps, seed=seed, device=device)
        policy, ranges, hist = train_policy(world, cfg, ranges)
        cond = _display_condition(case, ranges)
        p_star = target_for(world, cond)
        # deferred: only the policy path needs torch
        import torch
        feats = torch.as_tensor(ranges.features([cond]),
                                dtype=torch.float32, device=device)
        lp = policy.log_pf_grid(feats[0]).cpu().numpy().reshape(-1)
        p_theta = np.exp(lp)
        data[f"{case}_pstar"] = p_star.reshape(H, H)
        data[f"{case}_pi"] = p_theta.reshape(H, H)
        data[f"{case}_l1"] = np.array(l1(p_theta, p_star))
        data[f"{case}_heldout_l1"] = np.array(hist[-1]["heldout_l1"])
        print(f"{case}: display-cond L1 {data[f'{case}_l1']:.3f}, "
              f"held-out mean L1 {data[f'{case}_heldout_l1']:.3f} "
              f"({steps} steps, seed {seed}, {device})", flush=True)
    np.savez(out_dir / "fig_policy_data.npz", **data)
    print(f"wrote {out_dir / 'fig_policy_data.npz'}")


def fig_policy(out_dir):
    """Render fig_policy.pdf from the saved policy-vs-target densities
    (cases B and C).

    Args:
        out_dir: Directory containing fig_policy_data.npz; the PDF is
            written here too.
    """
    d = np.load(out_dir / "fig_policy_data.npz")
    fig, axes = plt.subplots(2, 2, figsize=(4.9, 4.6),
                             constrained_layout=True)
    for row, (case, label) in enumerate(CASES):
        p_s, p_t = d[f"{case}_pstar"], d[f"{case}_pi"]
        vmax = max(p_s.max(), p_t.max())
        norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
        for col, (p, title) in enumerate(
                [(p_s, "exact target $p^{*}_{c}$"),
                 (p_t, r"learned $\pi_{\theta}(\cdot \mid c)$")]):
            ax = axes[row, col]
            im = ax.imshow(np.maximum(p, vmax * 1e-4), origin="lower",
                           norm=norm, cmap="viridis",
                           interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=8)
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
        axes[row, 1].text(0.03, 0.95,
                          f"L1 = {float(d[f'{case}_l1']):.2f}",
                          transform=axes[row, 1].transAxes,
                          fontsize=7.5, va="top", color="white")
        fig.colorbar(im, ax=axes[row, 1], shrink=0.9, pad=0.02)
    out = out_dir / "fig_policy.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    """Parse CLI arguments and dispatch to the targets figure, the
    train+save+render policy path, or a render-only rebuild from saved
    data."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="targets",
                    choices=["targets", "policy", "render"],
                    help="targets: exact-layer figure (any machine); "
                    "policy: train + save data + render (run box); "
                    "render: rebuild fig_policy.pdf from saved data")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--out", default=None,
                    help="output dir (default drafts/figures)")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out) if args.out else (
        ROOT / "drafts" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.which == "targets":
        fig_targets(out_dir)
    elif args.which == "policy":
        compute_policy_data(out_dir, args.steps, args.seed, args.device)
        fig_policy(out_dir)
    else:
        fig_policy(out_dir)


if __name__ == "__main__":
    main()
