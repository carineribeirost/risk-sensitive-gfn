"""Exact-KL oracle loss. Grid-density oracle identity between the
grad-capable batched path and the @no_grad evaluation path, gradient
flow, and the trainer branch."""

import numpy as np
import pytest
import torch

from epgfn.policy import ConditionalPolicy
from epgfn.train import TrainConfig, train_policy
from epgfn.worlds import WorldConfig, make_world

H = 8


def test_log_pf_grid_train_matches_eval_path():
    """The batched grad-capable log_pf_grid_train path matches the per-sample
    log_pf_grid eval path and stays normalized."""
    torch.manual_seed(0)
    policy = ConditionalPolicy(H, n_cond_features=3, dim=32, depth=2,
                               cond_dim=16, x1_dim=8)
    feats = torch.randn(5, 3)
    batched = policy.log_pf_grid_train(feats)
    assert batched.shape == (5, H, H)
    for i in range(5):
        ref = policy.log_pf_grid(feats[i])
        torch.testing.assert_close(batched[i], ref)
    # each row is a normalized density
    total = batched.reshape(5, -1).logsumexp(-1)
    torch.testing.assert_close(total, torch.zeros(5), atol=1e-5,
                               rtol=0.0)


def test_log_pf_grid_train_gradients_flow():
    """Gradients flow through log_pf_grid_train into the policy head but not
    into the unused logz head."""
    torch.manual_seed(0)
    policy = ConditionalPolicy(H, n_cond_features=3, dim=32, depth=2,
                               cond_dim=16, x1_dim=8)
    feats = torch.randn(2, 3)
    tgt = torch.softmax(torch.randn(2, H * H), dim=-1)
    logp = policy.log_pf_grid_train(feats).reshape(2, -1)
    loss = -(tgt * logp).sum(-1).mean()
    loss.backward()
    assert policy.head.weight.grad is not None
    assert policy.head.weight.grad.abs().sum() > 0
    # the logz head is unused by the KL loss
    assert policy.logz_head[0].weight.grad is None


def test_exact_kl_trainer_runs_and_descends():
    """Training with the exact_kl loss runs to completion with finite losses
    and held-out L1, and the loss descends."""
    world = make_world("A", WorldConfig(H=H), seed=1)
    cfg = TrainConfig(loss="exact_kl", steps=60, n_conds=4,
                      cond_pool=16, eval_every=20, seed=0,
                      net=dict(dim=32, depth=2, cond_dim=16, x1_dim=8))
    _, _, hist = train_policy(world, cfg)
    assert hist[-1]["step"] == 60
    losses = [h["loss"] for h in hist]
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]  # cross-entropy descends
    assert np.isfinite(hist[-1]["heldout_l1"])


def test_exact_kl_requires_pool_and_valid_loss_name():
    """train_policy raises ValueError for a missing cond_pool under exact_kl
    loss and for an unknown loss name."""
    world = make_world("A", WorldConfig(H=H), seed=1)
    with pytest.raises(ValueError, match="cond_pool"):
        train_policy(world, TrainConfig(loss="exact_kl", cond_pool=None))
    with pytest.raises(ValueError, match="unknown loss"):
        train_policy(world, TrainConfig(loss="kl"))
