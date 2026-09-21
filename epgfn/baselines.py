"""Matched-conjunctivity aggregation baselines.

A conjunctive rule that concentrates mass on conjunctively good
candidates is close to definitional, so beating a weighted mean alone
is weak evidence that *this* conjunctive rule is the right one. The
answer is a comparison at matched conjunctivity: same worlds, same
auxiliary objective, same exclusions, same temperature, with only the
aggregation Ψ replaced.

Three rules:

  hard minimum        `cases.psi_worst_case` IS the hard min. Case A
                      takes min over the neutral set, case C over the
                      promoted set, case D over all nested scores, and
                      case B takes min(plus) - gamma*max(minus). It is
                      already reported as the "worst pole" column, so
                      it needs no new code and no new run.

  geometric mean      implemented here, weighted by the same nominal
                      weights the DRO-CVaR functional distrusts. This
                      is the natural matched comparator to the
                      weighted ARITHMETIC mean, which is the Boltzmann
                      pole.

  desirability        Derringer-Suich: map each score through a
                      one-sided desirability d in [0,1], then take
                      their weighted geometric mean. Implemented here.

Every function returns the same (psi, floor_ok, veto_ok) triple as
`cases.psi_and_masks`, so the reward assembly, the exclusions and the
temperature are shared with the risk targets and only the aggregation
differs.

The desirability bounds are a declared choice, not a tuned one: each
score is referenced to per-world quantiles of its own distribution,
`lo_q` and `hi_q`. They are parameters here so the declaration is
visible in the run's args.json rather than buried.
"""

from __future__ import annotations

import numpy as np

from .cases import EPS_REWARD, log_reward_from, psi_worst_case
from .conditions import Condition
from .target import p_star
from .worlds import World

KINDS = ("geometric", "desirability", "hard_min")


def _sl(arr, idx):
    return arr if idx is None else arr[idx]


def _wgeo(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted geometric mean over the last axis, weights summing to 1.

    Computed in log space and floored at EPS_REWARD before the log, so
    a single zero score does not send the aggregate to -inf. That
    floor is the same constant the reward already clamps at, so the
    two agree.
    """
    aw = np.maximum(np.asarray(a, dtype=float), EPS_REWARD)
    return np.exp((np.log(aw) * np.asarray(w, dtype=float)).sum(axis=-1))


def _desirability(a: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                  shape: float) -> np.ndarray:
    """One-sided larger-is-better Derringer-Suich transform.

    d = 0 below `lo`, 1 above `hi`, ((a-lo)/(hi-lo))**shape between.
    """
    span = np.maximum(hi - lo, 1e-12)
    d = np.clip((np.asarray(a, dtype=float) - lo) / span, 0.0, 1.0)
    return d ** float(shape)


def _quantile_bounds(a: np.ndarray, lo_q: float, hi_q: float):
    """Per-score-column bounds from the world's own score distribution."""
    return (np.quantile(a, lo_q, axis=0), np.quantile(a, hi_q, axis=0))


def psi_baseline(world: World, kind: str, idx=None, *,
                 lo_q: float = 0.50, hi_q: float = 0.95,
                 shape: float = 1.0):
    """(psi, floor_ok, veto_ok) under a non-DRO aggregation.

    Floor and veto semantics are preserved exactly as in
    `cases.psi_worst_case`: the floor is applied to the promoted
    aggregate, the veto at the nominal thresholds with delta = 0, so
    the geometry of each case is untouched and only the pooling
    changes.
    """
    if kind == "hard_min":
        return psi_worst_case(world, idx)
    if kind not in ("geometric", "desirability"):
        raise ValueError(f"unknown baseline kind {kind!r}")

    n = world.n_points if idx is None else len(idx)
    floor_ok = np.ones(n, dtype=bool)
    veto_ok = np.ones(n, dtype=bool)
    cfg = world.cfg

    def agg(scores_full, scores, w):
        if kind == "geometric":
            return _wgeo(scores, w)
        lo, hi = _quantile_bounds(scores_full, lo_q, hi_q)
        return _wgeo(_desirability(scores, lo, hi, shape), w)

    if world.case == "A":
        psi = agg(world.scores_neutral, _sl(world.scores_neutral, idx),
                  world.w_neutral)
        if world.scores_named is not None:      # guarded A variant
            veto_ok = ~np.any(_sl(world.scores_named, idx)
                              >= world.c_named, axis=-1)
    elif world.case == "B":
        lo_agg = agg(world.scores_plus, _sl(world.scores_plus, idx),
                     world.p_plus)
        hi_agg = agg(world.scores_minus, _sl(world.scores_minus, idx),
                     world.q_minus)
        psi = lo_agg - cfg.gamma * hi_agg
        if cfg.use_floor:
            floor_ok = lo_agg >= world.floor_value
    elif world.case == "C":
        psi = agg(world.scores_plus, _sl(world.scores_plus, idx),
                  world.p_plus)
        veto_ok = ~np.any(_sl(world.scores_named, idx) >= world.c_named,
                          axis=-1)
    elif world.case == "D":
        sc = _sl(world.scores_nested, idx)       # (n, O, K_in)
        full = world.scores_nested
        inner = np.stack(
            [agg(full[:, o, :], sc[:, o, :], world.w_inner[o])
             for o in range(sc.shape[1])], axis=-1)   # (n, O)
        psi = _wgeo(inner, world.pi_outer)
    else:
        raise ValueError(f"unknown case {world.case!r}")
    return psi, floor_ok, veto_ok


def target_baseline(world: World, cond: Condition, kind: str, *,
                    lo_q: float = 0.50, hi_q: float = 0.95,
                    shape: float = 1.0) -> np.ndarray:
    """Exact target under a baseline aggregation. Ignores cond's risk
    block; everything else about the reward is shared with the risk
    targets, which is what makes the comparison matched."""
    psi, floor_ok, veto_ok = psi_baseline(
        world, kind, lo_q=lo_q, hi_q=hi_q, shape=shape)
    log_r = log_reward_from(world, cond, psi, floor_ok, veto_ok)
    return p_star(log_r, cond.beta_t)
