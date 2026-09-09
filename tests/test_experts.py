"""Deterministic expert subset rule + the unconditional expert
trainer."""

import numpy as np

from epgfn.conditions import ConditionRanges, expert_subset
from epgfn.train import TrainConfig, train_expert
from epgfn.worlds import WorldConfig, make_world


def _ranges(case, **kw):
    r = ConditionRanges(case=case, **kw)
    return r


def test_expert_subset_sizes_and_membership():
    """Expert subset sizes and grid membership match the deterministic rule for
    each case."""
    for case, k in (("A", 9), ("B", 9), ("C", 9), ("D", 6)):
        r = _ranges(case)
        sub = expert_subset(r)
        assert len(sub) == k, case
        grid = r.heldout_grid()
        for c in sub:  # frozen dataclasses: exact-equality membership
            assert c in grid, (case, c)
        assert sub == expert_subset(r)  # deterministic


def test_expert_subset_varies_only_beta_t_and_first_axis():
    """Expert subset for case B only varies beta_t and the first risk axis,
    holding the rest fixed."""
    sub = expert_subset(_ranges("B"))
    assert len({c.beta_t for c in sub}) == 3
    assert len({c.risk.beta_p for c in sub}) == 3
    assert len({c.risk.rho_p for c in sub}) == 1  # other axes at mid
    assert len({c.risk.beta_m for c in sub}) == 1
    assert len({c.risk.rho_m for c in sub}) == 1
    assert len({c.w_g for c in sub}) == 1


def test_train_expert_descends_to_its_target():
    """Training a single expert on its target condition reduces held-out L1
    below the initial value."""
    world = make_world("A", WorldConfig(H=8), seed=1)
    from epgfn.train import ranges_for
    cond = expert_subset(ranges_for(world))[4]  # mid condition
    cfg = TrainConfig(steps=80, eval_every=40, seed=0,
                      net=dict(dim=32, depth=2, cond_dim=16, x1_dim=8))
    _, hist = train_expert(world, cond, cfg)
    assert hist[-1]["step"] == 80
    l1s = [h["heldout_l1"] for h in hist]
    assert all(np.isfinite(l1s)) and 0.0 <= l1s[-1] <= 2.0
    assert l1s[-1] < l1s[0]
