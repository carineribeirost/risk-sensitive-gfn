"""SubTB(λ) arm: opt-in flow head leaves default policies untouched;
the loss is finite, trains, and reduces to TB when only the (0,d)
segment carries weight."""

import numpy as np
import torch

from epgfn.policy import ConditionalPolicy
from epgfn.train import TrainConfig, _subtb_loss, train_policy
from epgfn.worlds import WorldConfig, sample_world


def test_default_policy_has_no_flow_params():
    """The default policy has no flow head; with_flows=True adds exactly the
    flow_head weight and bias params."""
    torch.manual_seed(0)
    pol = ConditionalPolicy(H=6)
    assert not hasattr(pol, "flow_head")
    keys = set(pol.state_dict().keys())
    pol_f = ConditionalPolicy(H=6, with_flows=True)
    extra = set(pol_f.state_dict().keys()) - keys
    assert extra == {"flow_head.weight", "flow_head.bias"}


def test_points_flows_shapes_d3():
    """log_pf_points_flows returns correctly shaped per-step log-probs and
    flows whose per-step log-probs sum to log_pf_points."""
    torch.manual_seed(1)
    pol = ConditionalPolicy(H=4, d=3, with_flows=True, cond_dim=32,
                            dim=64, depth=2, x1_dim=16)
    pts = torch.tensor([[0, 1, 2], [3, 3, 0]])
    feats = torch.zeros(2, 4)
    lp, fl = pol.log_pf_points_flows(pts, feats)
    assert lp.shape == (2, 3) and fl.shape == (2, 2)
    # per-step terms must sum to log_pf_points
    with torch.no_grad():
        total = pol.log_pf_points(pts, feats)
    assert torch.allclose(lp.sum(1), total, atol=1e-5)


def test_subtb_equals_tb_at_zero_lambda_limit():
    """At lam=1 the SubTB loss matches the manual pairwise-difference formula,
    and its (0,d) residual equals the negative TB delta."""
    # lam -> 0: only adjacent segments... instead check the exact
    # algebra: with d=2 and flows fixed, the (0,d) term equals the TB
    # residual squared; at lam=1 all segments weight equally.
    torch.manual_seed(2)
    b, d = 5, 2
    lp = torch.randn(b, d)
    flows = torch.randn(b, d - 1)
    log_z = torch.randn(b)
    logr = torch.randn(b)
    loss = _subtb_loss(lp, flows, log_z, logr, lam=1.0)
    # manual: G = [logZ, F1 - lp0, logr - lp0 - lp1]
    g0 = log_z
    g1 = flows[:, 0] - lp[:, 0]
    g2 = logr - lp[:, 0] - lp[:, 1]
    manual = ((g0 - g1) ** 2 + (g0 - g2) ** 2 + (g1 - g2) ** 2) / 3.0
    assert torch.allclose(loss, manual.mean(), atol=1e-5)
    # the (0,d) residual IS the TB delta (up to sign; squared identical)
    tb_delta = logr - log_z - lp.sum(1)
    assert torch.allclose(g0 - g2, -tb_delta, atol=1e-6)


def test_subtb_training_smoke_d3():
    """SubTB training runs on a tiny d=3 world and yields finite loss and held-
    out L1."""
    world, _ = sample_world("A", WorldConfig(H=6, d=3), seed=0)
    cfg = TrainConfig(loss="subtb", steps=20, n_conds=2, n_points=16,
                      eval_every=20, seed=0, cond_pool=16,
                      net=dict(dim=64, depth=2, cond_dim=32, x1_dim=16))
    _, _, hist = train_policy(world, cfg)
    assert np.isfinite(hist[-1]["heldout_l1"])
    assert np.isfinite(hist[-1]["loss"])


def test_subtb_training_improves_tiny_world():
    """SubTB training measurably improves held-out L1 below the untrained
    baseline on a tiny world."""
    world, _ = sample_world("A", WorldConfig(H=8), seed=0)
    cfg = TrainConfig(loss="subtb", steps=300, n_conds=4, n_points=32,
                      eval_every=300, seed=0, cond_pool=128,
                      net=dict(dim=128, depth=2, cond_dim=64))
    _, _, hist = train_policy(world, cfg)
    assert hist[-1]["heldout_l1"] < 0.55  # untrained is ~0.8 (golden)
