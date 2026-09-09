"""Hard regression gate: the d-generalized policy must reproduce the
2-step implementation bit-for-bit at d=2.

Two independent guards:
1. Golden-file test: `tests/data/w31_golden.npz` was captured from the
   old 2-step-hardcoded code by `make_w31_golden.py` (committed). Init
   weights, exact densities, sampler draws, and a short pooled-TB
   training trace must match EXACTLY (same machine class; if a different
   BLAS ever breaks bitwise equality on the forward/training arrays,
   loosen those asserts to atol=1e-6 and record why, with a date; never
   silently).
2. In-code reference: the old two-softmax composition, re-implemented
   here from the module's own submodules, must equal the generalized
   stepwise-expansion path on the same weights (machine-independent).

Key translation: the old `x1_embed` table (H+1 rows: H values + null)
IS the new position-tagged `coord_embed` table at d=2 ((d-1)·H value
rows + 1 start row): same shape, same row order, same RNG.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from epgfn.policy import ConditionalPolicy
from epgfn.train import TrainConfig, train_policy
from epgfn.worlds import WorldConfig, sample_world

GOLDEN = "tests/data/w31_golden.npz"
KEYMAP = {"x1_embed.weight": "coord_embed.weight"}


@pytest.fixture(scope="module")
def golden():
    """Load the frozen golden reference arrays captured from the old
    2-step-hardcoded implementation."""
    return np.load(GOLDEN)


@pytest.fixture(scope="module")
def policy():
    """Build the deterministic ConditionalPolicy shared by the golden-
    comparison tests."""
    torch.manual_seed(1234)
    return ConditionalPolicy(H=8, n_cond_features=4, cond_dim=32, dim=64,
                             depth=2, x1_dim=16)


def test_init_weights_bitwise(golden, policy):
    """The policy's initial weights match the golden state dict bit-for-bit,
    accounting for the x1_embed/coord_embed rename."""
    sd = policy.state_dict()
    gkeys = {k[4:] for k in golden.files if k.startswith("sd::")}
    translated = {(KEYMAP[k] if k in KEYMAP and k not in sd else k): k
                  for k in gkeys}
    assert set(translated) == set(sd.keys())
    for nk, gk in translated.items():
        assert np.array_equal(golden[f"sd::{gk}"], sd[nk].numpy()), \
            f"weight mismatch: {gk}"


def test_exact_densities_bitwise(golden, policy):
    """Exact grid, point, and training-grid log-densities match the golden
    arrays bit-for-bit."""
    torch.manual_seed(99)
    feats = torch.randn(3, 4)
    grid = np.stack([policy.log_pf_grid(feats[i]).numpy()
                     for i in range(3)])
    assert np.array_equal(grid, golden["grid"])
    pts = torch.tensor([[0, 0], [3, 5], [7, 2], [5, 7]])
    with torch.no_grad():
        lp = policy.log_pf_points(
            pts, feats[0].unsqueeze(0).expand(4, -1)).numpy()
        gt = policy.log_pf_grid_train(feats).numpy()
    assert np.array_equal(lp, golden["pts"])
    assert np.array_equal(gt, golden["grid_train"])


def test_sampler_rng_bitwise(golden, policy):
    """Sampling with a fixed RNG generator reproduces the golden sample draws
    bit- for-bit."""
    torch.manual_seed(99)
    feats = torch.randn(3, 4)
    gen = torch.Generator().manual_seed(7)
    smp = policy.sample(feats[1].unsqueeze(0).expand(32, -1), gen).numpy()
    assert np.array_equal(smp, golden["samples"])


def test_training_trace_bitwise(golden):
    """A short training run on the golden world reproduces the golden loss and
    held-out L1 traces bit-for-bit."""
    world, _ = sample_world("A", WorldConfig(H=8), seed=0)
    assert np.array_equal(world.g, golden["world_g"])
    cfg = TrainConfig(steps=30, n_conds=2, n_points=16, eval_every=10,
                      seed=0, cond_pool=32,
                      net=dict(dim=64, depth=2, cond_dim=32))
    _, _, hist = train_policy(world, cfg)
    assert np.array_equal(np.array([h["loss"] for h in hist]),
                          golden["hist_loss"])
    assert np.array_equal(np.array([h["heldout_l1"] for h in hist]),
                          golden["hist_l1"])


def _old_two_softmax_grid(pol, cond_feats_one):
    """The old 2-step log_pf_grid, verbatim modulo the embedding-table
    attribute name: the frozen reference for old-vs-new."""
    dev = cond_feats_one.device
    table = pol.coord_embed if hasattr(pol, "coord_embed") else pol.x1_embed
    null_id = pol.null_id

    def step_logits(ids, e):
        h = pol.inp(table(ids))
        for blk in pol.blocks:
            h = blk(h, e)
        return pol.head(h)

    e1 = pol.embed_cond(cond_feats_one.unsqueeze(0))
    null = torch.tensor([null_id], device=dev)
    lp1 = F.log_softmax(step_logits(null, e1), dim=-1)[0]
    eH = e1.expand(pol.H, -1)
    x1 = torch.arange(pol.H, device=dev)
    lp2 = F.log_softmax(step_logits(x1, eH), dim=-1)
    return lp1[:, None] + lp2


def test_old_vs_new_grid_same_weights(policy):
    """The generalized stepwise-expansion log_pf_grid matches the old two-
    softmax composition exactly on the same weights."""
    torch.manual_seed(5)
    for _ in range(3):
        feats = torch.randn(4)
        with torch.no_grad():
            ref = _old_two_softmax_grid(policy, feats)
        new = policy.log_pf_grid(feats)
        assert torch.equal(ref, new)
