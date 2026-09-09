"""d > 2 correctness of the dimensional generalization: exact
enumerability, point/grid agreement, sampler support, world
construction, and an end-to-end TB smoke run on a d=3 world."""

import numpy as np
import pytest
import torch

from epgfn.cases import target_for
from epgfn.policy import ConditionalPolicy
from epgfn.train import TrainConfig, train_policy
from epgfn.worlds import WorldConfig, grid_coords, make_world, sample_world


def test_grid_coords_canonical_order_d3():
    """grid_coords returns d=3 points
    in mixed-radix, x0-major canonical order."""
    c = grid_coords(3, d=3)
    assert c.shape == (27, 3)
    # mixed radix, x0-major: flat index == x0*9 + x1*3 + x2
    flat = c[:, 0] * 9 + c[:, 1] * 3 + c[:, 2]
    assert np.array_equal(flat, np.arange(27))


def test_terminating_density_sums_to_one_d3():
    """log_pf_grid's exponentiated grid densities sum to one at d=3."""
    torch.manual_seed(0)
    pol = ConditionalPolicy(H=5, d=3, cond_dim=32, dim=64, depth=2,
                            x1_dim=16)
    lp = pol.log_pf_grid(torch.zeros(4))
    assert lp.shape == (5, 5, 5)
    assert torch.exp(lp).sum().item() == pytest.approx(1.0, abs=1e-5)


def test_log_pf_points_and_train_grid_match_d3():
    """log_pf_grid_train matches per-condition log_pf_grid calls, and
    log_pf_points matches the corresponding grid entries, at d=3."""
    torch.manual_seed(1)
    pol = ConditionalPolicy(H=4, d=3, cond_dim=32, dim=64, depth=2,
                            x1_dim=16)
    feats = torch.randn(2, 4)
    grids = torch.stack([pol.log_pf_grid(feats[i]) for i in range(2)])
    with torch.no_grad():
        gt = pol.log_pf_grid_train(feats)
    assert torch.allclose(gt, grids, atol=1e-5)
    pts = torch.tensor([[0, 0, 0], [3, 1, 2], [2, 3, 3]])
    with torch.no_grad():
        lp = pol.log_pf_points(pts, feats[0].unsqueeze(0).expand(3, -1))
    for i, (a, b, c) in enumerate(pts):
        assert lp[i].item() == pytest.approx(grids[0, a, b, c].item(),
                                             abs=1e-5)


def test_sample_respects_support_d3():
    """Sampled points stay within the [0, H) support on every axis at d=3."""
    torch.manual_seed(2)
    pol = ConditionalPolicy(H=5, d=3, cond_dim=32, dim=64, depth=2,
                            x1_dim=16)
    gen = torch.Generator().manual_seed(0)
    pts = pol.sample(torch.zeros(64, 4), gen)
    assert pts.shape == (64, 3)
    assert pts.min() >= 0 and pts.max() < 5


def test_world_shapes_d3():
    """Each case's d=3 world has n_points = H**3 and an all-finite g array."""
    cfg = WorldConfig(H=6, d=3)
    for case in ("A", "B", "C", "D"):
        w = make_world(case, cfg, seed=0)
        assert w.n_points == 6 ** 3
        assert np.isfinite(w.g).all()


def test_tb_training_smoke_d3():
    """End-to-end pooled-TB training produces a finite held-out L1 at d=3."""
    # gate + exact targets + pooled TB + evaluate, end to end at d=3
    world, _ = sample_world("A", WorldConfig(H=6, d=3), seed=0)
    cfg = TrainConfig(steps=20, n_conds=2, n_points=16, eval_every=20,
                      seed=0, cond_pool=16,
                      net=dict(dim=64, depth=2, cond_dim=32, x1_dim=16))
    _, _, hist = train_policy(world, cfg)
    assert np.isfinite(hist[-1]["heldout_l1"])


def test_exact_kl_smoke_d3():
    """End-to-end exact-KL training produces a finite held-out L1 at d=3."""
    world, _ = sample_world("A", WorldConfig(H=6, d=3), seed=0)
    cfg = TrainConfig(loss="exact_kl", steps=10, n_conds=2, n_points=16,
                      eval_every=10, seed=0, cond_pool=16,
                      net=dict(dim=64, depth=2, cond_dim=32, x1_dim=16))
    _, _, hist = train_policy(world, cfg)
    assert np.isfinite(hist[-1]["heldout_l1"])


def test_target_normalized_d3():
    """target_for produces a normalized probability distribution over the full
    d=3 grid."""
    from epgfn.conditions import Condition, tied_risk
    world, _ = sample_world("B", WorldConfig(H=5, d=3), seed=0)
    p = target_for(world, Condition(2.0, 0.3, tied_risk("B", 0.5, 0.5)))
    assert p.shape == (125,)
    assert p.sum() == pytest.approx(1.0, abs=1e-8)
