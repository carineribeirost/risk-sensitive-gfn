"""Hardness gate on/off. Ungated runs take the first world draw
per seed (same seed chain), record the probe values, and mark rows."""

import numpy as np

from epgfn.o1 import run_o1
from epgfn.o3 import run_o3
from epgfn.worlds import WorldConfig, make_world, sample_world

CFG = WorldConfig(H=12)


def test_sample_world_no_gate_is_first_draw():
    """With the hardness gate disabled, sample_world returns the same first
    draw as make_world for the seed."""
    for case in ("A", "B"):
        w, attempts = sample_world(case, CFG, seed=5, gate=False)
        assert attempts == 1
        # sample_world's internal seed scrambling for attempt 0
        ref = make_world(case, CFG, seed=5 * 100_003)
        assert w.seed == ref.seed
        np.testing.assert_array_equal(w.g, ref.g)


def test_run_o1_gate_columns():
    """run_o1 reports the gated/attempts/is_hard/probe_tv_on_off columns
    correctly whether gating is on or off."""
    grids = dict(beta_grid=[0.5, 1.0], rho_grid=[0.0, 0.5])
    rows, sep, meta = run_o1("C", WorldConfig(H=8), [0], gate=False,
                             **grids)
    assert all(r["gated"] is False for r in rows + sep)
    m = meta[0]
    assert m["gated"] is False and m["attempts"] == 1
    assert isinstance(m["is_hard"], (bool, np.bool_))
    assert np.isfinite(m["probe_tv_on_off"])
    # gated run on the same seed reports its verdict too
    _, _, meta_g = run_o1("C", WorldConfig(H=8), [0], gate=True, **grids)
    assert meta_g[0]["gated"] is True and meta_g[0]["is_hard"]


def test_run_o3_gate_columns():
    """run_o3 reports the gated/attempts/probe_tv_on_off columns correctly when
    the hardness gate is disabled."""
    rows, meta = run_o3("A", WorldConfig(H=8), [0], n_stress=8,
                        contam_eps=None, gate=False)
    assert all(r["gated"] is False for r in rows)
    assert meta[0]["attempts"] == 1
    assert np.isfinite(meta[0]["probe_tv_on_off"])
