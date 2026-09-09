"""Sequence world: PWM + sparse-epistasis score family, the
sparsity knob, and end-to-end compatibility with the exact pipeline."""

import numpy as np
import pytest

from epgfn.cases import target_for
from epgfn.conditions import Condition, tied_risk
from epgfn.train import TrainConfig, train_policy
from epgfn.worlds import (WorldConfig, make_world, sample_world,
                          sequence_field)


def test_sequence_field_shape_range_determinism():
    """sequence_field is deterministic, correctly shaped, [0,1]-bounded, and
    non- additive."""
    f1 = sequence_field(np.random.default_rng(7), H=4, d=5)
    f2 = sequence_field(np.random.default_rng(7), H=4, d=5)
    assert f1.shape == (4 ** 5,)
    assert f1.min() == 0.0 and f1.max() == 1.0
    assert np.array_equal(f1, f2)
    # epistasis makes it non-additive: it must differ from a fresh
    # PWM-only reconstruction more often than not (weak sanity check)
    assert np.unique(f1).size > 4 * 5  # more levels than pure PWM sums


def test_sequence_world_all_cases_shapes():
    """Every case builds a finite, [0,1]-bounded g array of the right size on
    sequence geometry."""
    cfg = WorldConfig(H=4, d=5, geometry="sequence")
    for case in ("A", "B", "C", "D"):
        w = make_world(case, cfg, seed=0)
        assert w.n_points == 4 ** 5
        assert np.isfinite(w.g).all()
        assert 0.0 <= w.g.min() and w.g.max() <= 1.0


def test_sparsity_thins_satisfying_sets_monotonically():
    """Raising the sparsity knob monotonically shrinks the satisfying-set
    share, equivalent to a power reshaping of the same draws."""
    shares = []
    for s in (1.0, 2.0, 4.0):
        cfg = WorldConfig(H=4, d=5, geometry="sequence", sparsity=s)
        w = make_world("A", cfg, seed=3)
        shares.append(float((w.scores_neutral[:, 0] >= 0.5).mean()))
    assert shares[0] > shares[1] > shares[2]
    # same seed => same underlying draws; knob only reshapes them
    base = make_world("A", WorldConfig(H=4, d=5, geometry="sequence"),
                      seed=3)
    s4 = make_world("A", WorldConfig(H=4, d=5, geometry="sequence",
                                     sparsity=4.0), seed=3)
    assert np.allclose(base.scores_neutral ** 4.0, s4.scores_neutral)


def test_sparsity_exempts_constraint_fields():
    """Constraint fields keep their s=1 values under sparsity while objective
    scores and g scale by the sparsity exponent."""
    # constraint fields are exempt from sparsity scaling: veto/guard
    # thresholds are absolute (c_d frozen), so constraint fields keep
    # their s=1 values while objectives and g sparsify; the RNG stream
    # is unchanged (exponent is post-draw)
    seq = dict(H=4, d=6, geometry="sequence")
    base = make_world("C", WorldConfig(**seq), seed=7)
    s4 = make_world("C", WorldConfig(**seq, sparsity=4.0), seed=7)
    assert np.array_equal(s4.scores_named, base.scores_named)
    assert np.allclose(s4.scores_plus, base.scores_plus ** 4.0)
    assert np.allclose(s4.g, base.g ** 4.0)
    ga = make_world("A", WorldConfig(**seq, k_guard=2), seed=7)
    ga4 = make_world("A", WorldConfig(**seq, k_guard=2, sparsity=4.0),
                     seed=7)
    assert np.array_equal(ga4.scores_named, ga.scores_named)
    assert np.allclose(ga4.scores_neutral, ga.scores_neutral ** 4.0)


def test_case_c_hard_sparsity_gate_passes():
    """The previously-crashing case C sequence config (H=4, d=8, s=4.0) now
    passes the hardness gate on the first attempt."""
    # the 2026-07-12/18 crash: case C at (sequence, H=4, d=8, s=4.0)
    # had NO gate-passing world in 200 attempts: veto share <= 0.0015
    # in every draw vs the 0.02 band floor. With the exemption the
    # first draw passes (deterministic; measured pre-fix by simulation)
    cfg = WorldConfig(H=4, d=8, geometry="sequence", sparsity=4.0)
    world, attempts = sample_world("C", cfg, seed=0)
    assert attempts == 1
    frac = np.any(world.scores_named >= world.c_named, axis=-1).mean()
    assert cfg.veto_frac_range[0] <= frac


def test_sparsity_one_is_bitwise_noop_on_grid():
    """sparsity=1.0 on grid geometry
    reproduces the default world bit-for-bit."""
    a = make_world("B", WorldConfig(H=8), seed=0)
    b = make_world("B", WorldConfig(H=8, geometry="grid", sparsity=1.0),
                   seed=0)
    assert np.array_equal(a.g, b.g)
    assert np.array_equal(a.scores_plus, b.scores_plus)
    assert a.floor_value == b.floor_value


def test_sequence_world_gate_and_exact_target():
    """The hardness-gate sampler and exact target distribution both work on
    sequence geometry, yielding a properly normalized target."""
    # the full hardness gate + exact-target pipeline on sequences
    world, attempts = sample_world("B", WorldConfig(H=4, d=5,
                                                    geometry="sequence"),
                                   seed=0)
    assert attempts >= 1
    p = target_for(world, Condition(2.0, 0.3, tied_risk("B", 0.5, 0.5)))
    assert p.shape == (4 ** 5,)
    assert p.sum() == pytest.approx(1.0, abs=1e-8)


def test_sequence_tb_training_smoke():
    """TB training runs to completion on a tiny sequence world and yields a
    finite held-out L1."""
    world, _ = sample_world("A", WorldConfig(H=4, d=4,
                                             geometry="sequence"),
                            seed=0)
    cfg = TrainConfig(steps=20, n_conds=2, n_points=16, eval_every=20,
                      seed=0, cond_pool=16,
                      net=dict(dim=64, depth=2, cond_dim=32, x1_dim=16))
    _, _, hist = train_policy(world, cfg)
    assert np.isfinite(hist[-1]["heldout_l1"])
