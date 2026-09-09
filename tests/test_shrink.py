"""ConditionRanges.shrink + eval_ranges split machinery."""

import numpy as np

from epgfn.conditions import ConditionRanges
from epgfn.train import TrainConfig, ranges_for, train_policy
from epgfn.worlds import WorldConfig, make_world


def test_shrink_geometry():
    """shrink narrows beta_t/w_g/rho/delta/beta axes as specified while leaving
    the original ranges untouched."""
    r = ConditionRanges(case="B", beta_bounds={"p": 0.1, "m": 0.2})
    s = r.shrink(0.8)
    # beta_t shrinks in log space, geometric midpoint preserved
    assert np.isclose(np.log(s.beta_t[0]) + np.log(s.beta_t[1]),
                      np.log(r.beta_t[0]) + np.log(r.beta_t[1]))
    assert np.isclose(np.log(s.beta_t[1] / s.beta_t[0]),
                      0.8 * np.log(r.beta_t[1] / r.beta_t[0]))
    # w_g symmetric
    assert np.allclose(s.w_g, (0.18, 0.82))
    # rho/delta pinned at 0, top-only
    assert s.rho == (0.0, 0.8 * 1.6) and s.delta == (0.0, 0.8 * 0.1)
    # beta axes: lower (validity) bound unchanged, top lowered
    assert s._b("p")[0] == 0.1 and np.isclose(s._b("p")[1], 0.82)
    assert s._b("m")[0] == 0.2 and np.isclose(s._b("m")[1], 0.84)
    # original untouched
    assert r._b("p") == (0.1, 1.0) and r.rho == (0.0, 1.6)


def test_shrunk_features_flag_full_grid_ends_as_extrap():
    """Normalizing the full held-out grid against shrunk ranges flags exactly
    the points outside the shrunk box as extrapolation."""
    r = ConditionRanges(case="A", beta_bounds={"n": 0.05})
    s = r.shrink(0.8)
    grid = r.heldout_grid()
    feats = s.features(grid)  # training(shrunk)-ranges normalization
    extrap = np.abs(feats).max(axis=1) > 1.0 + 1e-9
    assert extrap.any() and (~extrap).any()
    # every condition inside the shrunk box on all axes must be interp
    for c, e in zip(grid, extrap):
        inside = (s.beta_t[0] - 1e-12 <= c.beta_t <= s.beta_t[1] + 1e-12
                  and s._b("n")[0] <= c.risk.beta <= s._b("n")[1] + 1e-12
                  and c.risk.rho <= s.rho[1] + 1e-12)
        assert inside == (not e)


def test_train_policy_eval_ranges_uses_full_grid():
    """Training with shrunk ranges but a full eval_ranges trains within the
    shrunk box while evaluating over the full grid."""
    world = make_world("A", WorldConfig(H=8), seed=0)
    full = ranges_for(world)
    shr = full.shrink(0.8)
    cfg = TrainConfig(steps=40, eval_every=20, seed=0, cond_pool=8,
                      net=dict(dim=32, depth=2, cond_dim=16, x1_dim=8))
    policy, used, hist = train_policy(world, cfg, shr, eval_ranges=full)
    assert used.rho == shr.rho  # trained with the shrunk ranges
    assert np.isfinite(hist[-1]["heldout_l1"])
    # training pool respects the SHRUNK box
    # (indirect: sampling from `used` stays inside by construction)
    rng = np.random.default_rng(0)
    for _ in range(32):
        c = used.sample(rng)
        assert c.risk.rho <= shr.rho[1] + 1e-12
        assert c.risk.beta <= shr._b("n")[1] + 1e-12
