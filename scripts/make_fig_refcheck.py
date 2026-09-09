"""Three-column reference-check figure (fig_refcheck): exact target vs
rejection reference vs learned GFlowNet density, one gated dev world
per case (seed 0, H=32 grid), at the hardness probe-on condition.

The rejection column is the FINITE-SAMPLE REFERENCE, not independent
ground truth: it consumes the same log_reward array as p* and bypasses
only the softmax normalization, so it can never catch a
reward-computation bug. What it adds is (a) a density-space picture of
what N exact draws look like (the mc_floor made visible) and (b)
triangulation that is independent of the POLICY: GFlowNet matches
rejection → residual vs p* is sampling noise; GFlowNet disagrees with
rejection → the defect is in the GFlowNet.

The GFlowNet column needs training (O2 runs save no policies), so its
compute runs on a run box and the DATA is saved
(drafts/figures/fig_refcheck_data.npz) for local re-rendering, same
pattern as make_fig_objects.

Usage (run box; device auto-selects cuda via default_device):
    .venv/bin/python scripts/make_fig_refcheck.py --which data
Re-render locally after copying the npz back:
    .venv/bin/python scripts/make_fig_refcheck.py --which render
"""

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: E402,F401
from matplotlib.colors import LogNorm  # noqa: E402

from epgfn.cases import log_reward, target_for  # noqa: E402
from epgfn.target import l1, mc_floor  # noqa: E402
from epgfn.train import TrainConfig, default_device  # noqa: E402
from epgfn.w33 import fixed_condition, run_arm  # noqa: E402
from epgfn.worlds import WorldConfig, sample_world  # noqa: E402

plt.style.use(["science", "bright", "no-latex"])
plt.rcParams.update({"font.size": 9})

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASES = [("A", "pooled evidence (A)"),
         ("B", "conflicting evidence (B)"),
         ("C", "veto criteria (C)"),
         ("D", "nested panels (D)")]
H = 32


def rejection_density(world, cond, n_samples: int, seed: int,
                      chunk: int = 1_000_000):
    """Exact rejection sampling from R^{β_t}: propose uniform over the
    enumerated X, accept w.p. exp(β_t·log R − max): the unnormalized
    exponentiated reward, softmax never touched. Returns (density,
    acceptance_rate). Deterministic given the seed; case B's probe-on
    acceptance is ~0.5%, so chunked proposals keep it seconds."""
    rng = np.random.default_rng(seed)
    lw = cond.beta_t * log_reward(world, cond)
    acc_p = np.exp(lw - lw.max())
    n = world.n_points
    counts = np.zeros(n, dtype=np.int64)
    accepted, proposed = 0, 0
    while accepted < n_samples:
        idx = rng.integers(0, n, size=chunk)
        keep_pos = np.flatnonzero(rng.random(chunk) < acc_p[idx])
        need = n_samples - accepted
        # count only proposals up to the last USED accept, else the
        # final partial chunk biases the reported rate low
        proposed += (int(keep_pos[need - 1]) + 1
                     if len(keep_pos) >= need else chunk)
        take = idx[keep_pos[:need]]
        np.add.at(counts, take, 1)
        accepted += len(take)
    return counts / n_samples, accepted / proposed


def compute_data(out_dir, n_samples, steps, seed, device):
    """For each case, build the exact target, the rejection-sampling
    reference density, and a trained "mix"-arm policy density, then
    save all three plus L1 metrics to fig_refcheck_data.npz.

    Args:
        out_dir: Directory to write fig_refcheck_data.npz into.
        n_samples: Number of accepted rejection-sampling draws per case.
        steps: Number of training steps for the "mix" arm.
        seed: Seed shared by rejection sampling, the MC floor (as
            seed + 1), and training.
        device: Torch device string for training.
    """
    data = {}
    for ci, (case, _) in enumerate(CASES, 1):
        print(f"[refcheck {ci}/{len(CASES)}] case {case}: world + "
              f"rejection reference ({n_samples:,} samples)...",
              flush=True)
        world, _ = sample_world(case, WorldConfig(H=H), seed=0)
        cond = fixed_condition(world)  # the hardness probe-on member
        target = target_for(world, cond)
        q_rej, acc = rejection_density(world, cond, n_samples, seed)
        # seed + 1: independent RNG stream from the rejection sampler above
        floor = mc_floor(target, n_samples,
                         np.random.default_rng(seed + 1))
        print(f"[refcheck {ci}/{len(CASES)}] case {case}: training "
              f"mix arm ({steps} steps, {device})...", flush=True)
        cfg = TrainConfig(steps=steps, seed=seed, device=device)
        # "mix" arm = uniform off-policy half, the O2-like full-support
        # regime, the fairest single-condition convergence reference
        student, hist = run_arm(world, cond, cfg, "mix")
        # deferred: only the "data" path needs torch
        import torch
        with torch.no_grad():
            lp = student.log_pf_grid(
                torch.zeros(1, device=device)).cpu().numpy().reshape(-1)
        p_theta = np.exp(lp)
        data[f"{case}_pstar"] = target.reshape(H, H)
        data[f"{case}_rej"] = q_rej.reshape(H, H)
        data[f"{case}_pi"] = p_theta.reshape(H, H)
        data[f"{case}_l1_rej"] = np.array(l1(q_rej, target))
        data[f"{case}_l1_pi"] = np.array(l1(p_theta, target))
        data[f"{case}_l1_pi_rej"] = np.array(l1(p_theta, q_rej))
        data[f"{case}_mc_floor"] = np.array(floor)
        data[f"{case}_acc_rate"] = np.array(acc)
        print(f"{case}: acc={acc:.4f} L1(rej,p*)={data[f'{case}_l1_rej']:.3f} "
              f"mc_floor={floor:.3f} L1(pi,p*)={data[f'{case}_l1_pi']:.3f} "
              f"L1(pi,rej)={data[f'{case}_l1_pi_rej']:.3f} "
              f"({steps} steps, seed {seed}, {device})", flush=True)
    np.savez(out_dir / "fig_refcheck_data.npz",
             n_samples=n_samples, **data)
    print(f"wrote {out_dir / 'fig_refcheck_data.npz'}")


def fig_refcheck(out_dir):
    """Render fig_refcheck.pdf: exact target vs rejection reference vs
    learned policy, one row per case.

    Args:
        out_dir: Directory containing fig_refcheck_data.npz; the PDF is
            written here too.
    """
    d = np.load(out_dir / "fig_refcheck_data.npz")
    n_samples = int(d["n_samples"])
    fig, axes = plt.subplots(4, 3, figsize=(6.4, 8.2),
                             constrained_layout=True)
    exp10 = int(np.log10(n_samples))
    n_lab = (rf"10^{{{exp10}}}" if n_samples == 10 ** exp10
             else f"{n_samples:,}")
    titles = ["exact target $p^{*}_{c}$",
              rf"rejection reference ($N{{=}}{n_lab}$)",
              r"learned $\pi_{\theta}$"]
    for row, (case, label) in enumerate(CASES):
        panels = [d[f"{case}_pstar"], d[f"{case}_rej"], d[f"{case}_pi"]]
        vmax = max(p.max() for p in panels)
        norm = LogNorm(vmin=vmax * 1e-4, vmax=vmax)
        for col, p in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(np.maximum(p, vmax * 1e-4), origin="lower",
                           norm=norm, cmap="viridis",
                           interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=8)
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
        axes[row, 1].text(
            0.03, 0.95,
            f"L1 = {float(d[f'{case}_l1_rej']):.2f}\n"
            f"floor {float(d[f'{case}_mc_floor']):.2f}",
            transform=axes[row, 1].transAxes, fontsize=7,
            va="top", color="white")
        axes[row, 2].text(
            0.03, 0.95,
            f"L1 = {float(d[f'{case}_l1_pi']):.2f}\n"
            f"vs rej {float(d[f'{case}_l1_pi_rej']):.2f}",
            transform=axes[row, 2].transAxes, fontsize=7,
            va="top", color="white")
        fig.colorbar(im, ax=axes[row, 2], shrink=0.9, pad=0.02)
    out = out_dir / "fig_refcheck.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    """Parse CLI arguments and either compute+save+render the
    reference-check data, or render-only from previously saved data."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="data",
                    choices=["data", "render"],
                    help="data: rejection + training + save + render "
                    "(run box); render: rebuild the PDF from saved data")
    ap.add_argument("--n-samples", type=int, default=100_000)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--out", default=None,
                    help="output dir (default drafts/figures)")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out) if args.out else (
        ROOT / "drafts" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.which == "data":
        compute_data(out_dir, args.n_samples, args.steps, args.seed,
                     args.device)
    fig_refcheck(out_dir)


if __name__ == "__main__":
    main()
