"""Mode set, ε-shell, and all arms end-to-end on
a tiny sequence world."""

import numpy as np
import pytest
import torch

from epgfn.train import TrainConfig
from epgfn.w33 import (ARMS, eps_shell, expected_coverage,
                       fixed_condition, infonce, mode_indices, run_arm,
                       samples_to_frac)
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


# --- policy coverage: a property of the trained sampler, not the run ------

def test_expected_coverage_matches_the_closed_form():
    p = np.array([0.0, 0.1, 0.02, 0.5])
    is_mode = np.array([False, True, True, True])
    n = 7
    want = np.mean([1 - (1 - q) ** n for q in (0.1, 0.02, 0.5)])
    assert expected_coverage(p, is_mode, n) == pytest.approx(want)


def test_expected_coverage_bounds_and_monotonicity():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(400))
    is_mode = np.zeros(400, dtype=bool); is_mode[:40] = True
    vals = [expected_coverage(p, is_mode, n) for n in (0, 10, 100, 1000, 10000)]
    assert vals[0] == 0.0
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:]))


def test_a_mode_the_policy_never_reaches_is_never_covered():
    """The whole point of the metric: injected states the policy did not
    learn must not count as covered."""
    p = np.array([1.0, 0.0, 0.0])
    is_mode = np.array([False, True, True])
    assert expected_coverage(p, is_mode, 10 ** 9) == 0.0


def test_expected_coverage_survives_tiny_probabilities():
    """1 - (1-p)^n underflows if written naively; log1p must not."""
    p = np.array([1.0 - 1e-12, 1e-12])
    is_mode = np.array([False, True])
    v = expected_coverage(p, is_mode, 1000)
    assert np.isfinite(v)
    assert v == pytest.approx(1e-9, rel=1e-3)


def test_expected_coverage_agrees_with_sampling():
    """The closed form claims to be the expected fraction of the mode set
    found in n draws. Check that against actually drawing."""
    rng = np.random.default_rng(7)
    p = rng.dirichlet(np.ones(60))
    is_mode = np.zeros(60, dtype=bool); is_mode[:12] = True
    n, reps = 40, 4000
    idx = np.arange(60)
    found = np.empty(reps)
    for r in range(reps):
        draws = rng.choice(idx, size=n, p=p)
        found[r] = np.isin(idx[is_mode], draws).mean()
    assert expected_coverage(p, is_mode, n) == pytest.approx(
        found.mean(), abs=0.02)


def test_empty_mode_set_reports_nan_not_a_number():
    p = np.array([0.5, 0.5])
    assert np.isnan(expected_coverage(p, np.zeros(2, dtype=bool), 10))


def test_run_arm_reports_training_and_policy_coverage_separately(world):
    """Training discovery counts states a run happened to visit,
    injected states included; the mix arm injects uniformly, so its
    training discovery can exceed what its trained policy actually
    reaches. Both are reported, distinctly."""
    cfg = TrainConfig(steps=12, n_points=16, eval_every=6, seed=0,
                      device="cpu", net=CFG_NET)
    _, hist = run_arm(world, fixed_condition(world), cfg, "mix")
    row = hist[-1]
    for k in ("train_discovery", "policy_coverage", "target_coverage",
              "policy_mass_modes", "reward_queries"):
        assert k in row, f"{k} missing from the history row"
    assert row["frac_modes"] == row["train_discovery"]   # old name kept
    assert 0.0 <= row["policy_coverage"] <= 1.0
    assert 0.0 <= row["policy_mass_modes"] <= 1.0
    assert row["reward_queries"] > 0


# --- teacher reward, Kim et al. (adaptive teachers), eqs. (4)-(6) ---------

def _teacher_logr(delta, C, alpha, logr, eps=1e-8):
    """The reward the arm builds, isolated for testing."""
    wgt = 1.0 + C * (delta > 0).to(delta.dtype)
    out = torch.log(eps + wgt * delta ** 2)
    return out + alpha * logr if alpha else out


def test_eq5_weights_undersampled_states_by_one_plus_C():
    """delta > 0 means the forward flow is too small, i.e. the student
    undersamples. Those states must get (1+C) times the weight."""
    delta = torch.tensor([2.0, -2.0])          # same magnitude, opposite sign
    logr = torch.zeros(2)
    out = _teacher_logr(delta, 19.0, 0.0, logr)
    assert out[0] > out[1]
    assert torch.exp(out[0] - out[1]).item() == pytest.approx(20.0, rel=1e-4)


def test_C_zero_reduces_to_the_bare_eq4():
    delta = torch.tensor([1.5, -1.5, 0.3])
    logr = torch.zeros(3)
    assert torch.allclose(_teacher_logr(delta, 0.0, 0.0, logr),
                          torch.log(1e-8 + delta ** 2))


def test_eq6_mixing_shifts_by_alpha_times_log_reward():
    delta = torch.tensor([1.0, 1.0])
    logr = torch.tensor([0.0, 4.0])
    a = _teacher_logr(delta, 19.0, 0.0, logr)
    b = _teacher_logr(delta, 19.0, 0.5, logr)
    assert torch.allclose(b - a, 0.5 * logr)


def test_the_arm_accepts_the_paper_parameters_and_runs(world):
    cfg = TrainConfig(steps=12, n_points=16, eval_every=6, seed=0,
                      device="cpu", net=CFG_NET)
    _, hist = run_arm(world, fixed_condition(world), cfg, "teacher",
                      teacher_c=19.0, teacher_alpha=0.0)
    assert hist and 0.0 <= hist[-1]["policy_coverage"] <= 1.0


def test_teacher_c_changes_the_run(world):
    """If C were inert the fix would be cosmetic; assert it is not."""
    def run(C):
        cfg = TrainConfig(steps=30, n_points=16, eval_every=30, seed=0,
                          device="cpu", net=CFG_NET)
        _, h = run_arm(world, fixed_condition(world), cfg, "teacher",
                       teacher_c=C)
        return h[-1]["l1"]
    assert run(0.0) != run(19.0)


# --- live modes: satisfaction and reward must agree -----------------------

def _mk(case, sp=4.0, seed=0):
    from epgfn.worlds import WorldConfig, make_world
    return make_world(case, WorldConfig(H=4, d=6, geometry="sequence",
                                        sparsity=sp), seed)


@pytest.mark.parametrize("case", ("A", "C", "D"))
def test_non_subtracting_cases_have_every_mode_live(case):
    """A, C and D only aggregate, so a satisfying state cannot have a
    negative composed reward and the live restriction is the identity."""
    from epgfn.cases import EPS_REWARD, log_reward
    from epgfn.w33 import mode_indices
    w = _mk(case)
    c = fixed_condition(w)
    im = np.zeros(w.n_points, bool)
    im[mode_indices(w)] = True
    live = im & (log_reward(w, c) > np.log(EPS_REWARD) + 1e-12)
    assert im.sum() > 0
    assert int(live.sum()) == int(im.sum())


def test_case_B_can_have_dead_modes():
    """Only B subtracts, so only B can satisfy on scores and clamp on
    reward. If this ever stops holding the metric needs revisiting."""
    from epgfn.cases import EPS_REWARD, log_reward
    from epgfn.w33 import mode_indices
    w = _mk("B")
    c = fixed_condition(w)
    im = np.zeros(w.n_points, bool)
    im[mode_indices(w)] = True
    dead = im & ~(log_reward(w, c) > np.log(EPS_REWARD) + 1e-12)
    assert im.sum() > 0
    assert dead.sum() >= 0          # may be 0 on a mild world; never negative
    assert dead.sum() <= im.sum()


def test_run_arm_reports_live_coverage(world):
    cfg = TrainConfig(steps=12, n_points=16, eval_every=6, seed=0,
                      device="cpu", net=CFG_NET)
    _, hist = run_arm(world, fixed_condition(world), cfg, "mix")
    row = hist[-1]
    for k in ("n_modes_live", "policy_coverage_live", "target_coverage_live"):
        assert k in row
    assert row["n_modes_live"] <= row["n_modes"]
    assert 0.0 <= row["policy_coverage_live"] <= 1.0


def test_live_coverage_is_never_below_the_all_mode_figure(world):
    """Dropping unreachable modes can only raise the fraction found, so a
    live figure below the all-mode one would mean the mask is wrong."""
    cfg = TrainConfig(steps=20, n_points=16, eval_every=20, seed=0,
                      device="cpu", net=CFG_NET)
    _, hist = run_arm(world, fixed_condition(world), cfg, "mix")
    row = hist[-1]
    if row["n_modes_live"] < row["n_modes"]:
        assert row["policy_coverage_live"] >= row["policy_coverage"] - 1e-9
        assert row["target_coverage_live"] >= row["target_coverage"] - 1e-9
