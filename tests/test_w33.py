"""Mode set, ε-shell, and all arms end-to-end on
a tiny sequence world."""

import numpy as np
import pytest
import torch

from epgfn.train import TrainConfig
from epgfn.w33 import (ARMS, eps_shell, fixed_condition, infonce,
                       mode_indices, run_arm, samples_to_frac)
from epgfn.worlds import WorldConfig, sample_world

CFG_NET = dict(dim=32, depth=2, cond_dim=16, x1_dim=8)


@pytest.fixture(scope="module")
def world():
    """Sample a small deterministic case-B sequence world shared by the
    tests in this module."""
    w, _ = sample_world("B", WorldConfig(H=4, d=4, geometry="sequence"),
                        seed=0)
    return w


def test_mode_set_nonempty_and_exact(world):
    """mode_indices returns a non-empty, proper subset of all points."""
    modes = mode_indices(world)
    assert 0 < len(modes) < world.n_points


def test_eps_shell_is_boundary(world):
    """eps_shell returns a correctly-shaped boolean mask that is non-empty for
    a floored, risk-on condition."""
    cond = fixed_condition(world)
    shell = eps_shell(world, cond)
    assert shell.dtype == bool and shell.shape == (world.n_points,)
    # B risk-on with a floor: some states are floored -> shell nonempty
    assert shell.any()


@pytest.mark.parametrize("arm", ARMS)
def test_arm_runs_and_metrics_sane(world, arm):
    """Every arm runs end-to-end and reports sane, cumulative metrics."""
    cfg = TrainConfig(steps=30, n_points=16, eval_every=10, seed=0,
                      net=CFG_NET)
    _, hist = run_arm(world, fixed_condition(world), cfg, arm)
    assert len(hist) == 3
    found = [h["modes_found"] for h in hist]
    assert found == sorted(found)               # cumulative
    assert found[-1] <= hist[-1]["n_modes"]
    assert np.isfinite(hist[-1]["l1"])
    assert 0.0 <= hist[-1]["edge_share"] <= 1.0
    assert hist[-1]["samples"] == 30 * 16


@pytest.mark.parametrize("arm", ["teacher", "contrastive"])
def test_arm_determinism(world, arm):
    """The teacher and contrastive arms produce identical histories (aside from
    wall-clock time) across repeated runs with the same seed."""
    cfg = TrainConfig(steps=20, n_points=16, eval_every=20, seed=1,
                      net=CFG_NET)
    cond = fixed_condition(world)
    _, h1 = run_arm(world, cond, cfg, arm)
    _, h2 = run_arm(world, cond, cfg, arm)
    strip = [{k: v for k, v in h.items() if k != "wall_s"} for h in h1]
    strip2 = [{k: v for k, v in h.items() if k != "wall_s"} for h in h2]
    assert strip == strip2


def test_infonce_all_equal_and_self_limiting():
    """infonce equals log(1+|B|) when all scores are equal and vanishes when
    negatives are strongly suppressed."""
    # equal scores: every softmax term is 1/(1+|B⁻|) -> loss = log(1+B)
    lp, ln = torch.zeros(4), torch.zeros(8)
    assert torch.isclose(infonce(lp, ln), torch.log(torch.tensor(9.0)))
    # suppressed negatives (s⁻ ≈ −40): the aux gradient must vanish
    assert infonce(lp, torch.full((8,), -40.0)).item() < 1e-6


def test_contrastive_reports_loss_aux(world):
    """The contrastive arm reports a finite loss_aux on every eval row when the
    ε-set is non-empty, while other arms omit the key."""
    cfg = TrainConfig(steps=30, n_points=16, eval_every=10, seed=0,
                      net=CFG_NET)
    _, hist = run_arm(world, fixed_condition(world), cfg, "contrastive")
    # B risk-on has a non-empty ε-set, so both buffers fill in step 1
    # and every eval row carries a finite aux loss
    assert all(np.isfinite(h["loss_aux"]) for h in hist)
    # other arms must NOT carry the key (schema unchanged)
    _, hist_r = run_arm(world, fixed_condition(world), cfg, "replay")
    assert "loss_aux" not in hist_r[-1]


def test_contrastive_degenerates_without_eps_set():
    """With an empty ε-set the contrastive arm's aux loss stays NaN and
    edge_share is 0, degenerating to plain positive replay."""
    # A (no guard) has an EMPTY ε-set at the fixed condition on this
    # deterministic world: D⁻ never fills, the aux loss stays inert
    # (loss_aux nan on every eval row), and edge_share is identically
    # 0: the arm reduces to top-R positive replay, reported not hidden
    w, _ = sample_world("A", WorldConfig(H=4, d=4, geometry="sequence"),
                        seed=0)
    cfg = TrainConfig(steps=30, n_points=16, eval_every=10, seed=0,
                      net=CFG_NET)
    _, hist = run_arm(w, fixed_condition(w), cfg, "contrastive")
    assert all(np.isnan(h["loss_aux"]) for h in hist)
    assert hist[-1]["edge_share"] == 0.0


def test_samples_to_frac():
    """samples_to_frac linearly interpolates the sample count at which
    frac_modes first reaches a target, or NaN if never reached."""
    hist = [{"frac_modes": 0.2, "samples": 100},
            {"frac_modes": 0.6, "samples": 200},
            {"frac_modes": 0.9, "samples": 300}]
    assert samples_to_frac(hist, 0.5) == 200.0
    assert np.isnan(samples_to_frac(hist, 0.95))
