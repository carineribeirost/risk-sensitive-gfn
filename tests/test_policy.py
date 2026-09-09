"""Policy and TB training tests: exact enumerability of the terminating
density, and a smoke check that TB training moves the policy toward the
exact target family on a tiny world."""

import numpy as np
import pytest
import torch

from epgfn.policy import ConditionalPolicy
from epgfn.train import (TrainConfig, default_device, evaluate,
                         ranges_for, train_policy)
from epgfn.cases import target_for
from epgfn.worlds import WorldConfig, sample_world


def test_terminating_density_sums_to_one():
    """log_pf_grid produces a valid density over the H x H grid that sums to
    one."""
    torch.manual_seed(0)
    pol = ConditionalPolicy(H=8)
    feats = torch.zeros(4)
    lp = pol.log_pf_grid(feats)
    assert lp.shape == (8, 8)
    assert torch.exp(lp).sum().item() == pytest.approx(1.0, abs=1e-5)


def test_log_pf_points_matches_grid():
    """log_pf_points evaluated at specific grid coordinates matches the
    corresponding entries of log_pf_grid."""
    torch.manual_seed(1)
    pol = ConditionalPolicy(H=6)
    feats = torch.randn(1, 4)
    grid = pol.log_pf_grid(feats[0])
    pts = torch.tensor([[0, 0], [3, 5], [5, 2]])
    with torch.no_grad():
        lp = pol.log_pf_points(pts, feats.expand(3, -1))
    for i, (a, b) in enumerate(pts):
        assert lp[i].item() == pytest.approx(grid[a, b].item(), abs=1e-5)


def test_sample_respects_support():
    """Sampled points from the policy stay within the [0, H) grid support."""
    torch.manual_seed(2)
    pol = ConditionalPolicy(H=5)
    gen = torch.Generator().manual_seed(0)
    pts = pol.sample(torch.zeros(64, 4), gen)
    assert pts.shape == (64, 2)
    assert pts.min() >= 0 and pts.max() < 5


@pytest.mark.skipif(default_device() != "cuda", reason="no usable CUDA")
def test_cuda_sample_grid_and_train_step():
    """Sampling, log_pf_grid, and one training step run correctly on a CUDA
    device when available."""
    # device-correctness guard: sample/log_pf_grid/one training step on GPU
    cfg_w = WorldConfig(H=8)
    world, _ = sample_world("A", cfg_w, seed=0)
    cfg = TrainConfig(steps=2, n_conds=2, n_points=16, eval_every=2,
                      seed=0, device="cuda",
                      net=dict(dim=64, depth=2, cond_dim=32))
    _, _, hist = train_policy(world, cfg)
    assert np.isfinite(hist[-1]["heldout_l1"])


def test_streaming_trainer_smoke():
    """The legacy per-step streaming trainer (cond_pool=None) still runs and
    yields a finite held-out L1."""
    # legacy per-step pricing path (cond_pool=None) must keep working:
    # it is the pooled-vs-streaming ablation arm
    cfg_w = WorldConfig(H=8)
    world, _ = sample_world("A", cfg_w, seed=0)
    cfg = TrainConfig(steps=20, n_conds=2, n_points=16, eval_every=20,
                      seed=0, cond_pool=None,
                      net=dict(dim=64, depth=2, cond_dim=32))
    _, _, hist = train_policy(world, cfg)
    assert np.isfinite(hist[-1]["heldout_l1"])


def test_tb_training_improves_heldout_l1():
    """Short TB training on a tiny world drops held-out L1 well below the
    untrained policy's baseline."""
    # tiny world + short run (pooled trainer): L1 must drop clearly
    # below the untrained policy's. Not a convergence claim, just a
    # pipeline-sanity check.
    cfg_w = WorldConfig(H=8)
    world, _ = sample_world("A", cfg_w, seed=0)
    cfg = TrainConfig(steps=300, n_conds=4, n_points=32, eval_every=300,
                      seed=0, cond_pool=128,
                      net=dict(dim=128, depth=2, cond_dim=64))
    torch.manual_seed(0)
    ranges = ranges_for(world)
    heldout = ranges.heldout_grid()
    ho_targets = [target_for(world, c) for c in heldout]
    untrained = ConditionalPolicy(8, **cfg.net)
    l1_before = evaluate(untrained, ranges, heldout, ho_targets)
    _, _, hist = train_policy(world, cfg)
    l1_after = hist[-1]["heldout_l1"]
    assert l1_after < 0.7 * l1_before
