"""Paper-2 corruption models. Oracle SCORES only: g is the
auxiliary objective, not an oracle output; world parameters
(weights, floor_value, thresholds) define the task and are never
corrupted. The TRUE world is untouched (deep copy); the exact layer
prices both, which is the whole point of the instrument.

Models:
- "crash":  per score field (each column of each block), a fraction
            `level` of states gets its entry replaced by U[0,1]
            garbage: unbounded corruption.
- "bias":   ONE rng-chosen field gets +0.2 wherever it is above its
            own 0.70-quantile (systematic regional over-reporting),
            clipped to [0,1]: bounded corruption.
- "heavy":  additive t(df=2)·0.05 noise on every entry, clipped to
            [0,1]: heavy-tailed but mostly bounded corruption.
- crude_world: the fallback comparator, TRUE scores quantized to 3
            uniform levels {0, 0.5, 1}: large but bounded distortion,
            no crashes (the "stable-but-crude oracle").
"""

from __future__ import annotations

import copy

import numpy as np

from .worlds import World

MODELS = ("crash", "bias", "heavy")
BIAS_DELTA = 0.2
BIAS_QUANTILE = 0.70
HEAVY_SCALE = 0.05

_SCORE_ATTRS = ("scores_neutral", "scores_plus", "scores_minus",
                "scores_named", "scores_nested")


def _blocks(world: World):
    for attr in _SCORE_ATTRS:
        arr = getattr(world, attr)
        if arr is not None:
            yield attr, arr


def _columns(arr: np.ndarray):
    """Iterate the per-origin score fields (columns) of a block as
    (index, (N,) view); handles D's (N, O, K) nested block."""
    flat = arr.reshape(arr.shape[0], -1)
    for j in range(flat.shape[1]):
        yield j, flat[:, j]


def corrupt_world(world: World, model: str, level: float,
                  seed: int = 0) -> World:
    """Deterministic corrupted copy of `world` (true world untouched).
    `level` is ε_s for "crash"; ignored for "bias"/"heavy" (their
    parameters are module constants)."""
    if model not in MODELS:
        raise ValueError(f"unknown corruption model {model!r}")
    rng = np.random.default_rng((world.seed, MODELS.index(model),
                                 int(level * 1000), seed))
    w = copy.deepcopy(world)
    if model == "crash":
        for _, arr in _blocks(w):
            for _, col in _columns(arr):
                n = col.shape[0]
                hit = rng.random(n) < level
                col[hit] = rng.uniform(0.0, 1.0, size=int(hit.sum()))
    elif model == "bias":
        cols = [(attr, j) for attr, arr in _blocks(w)
                for j, _ in _columns(arr)]
        attr, j = cols[rng.integers(0, len(cols))]
        flat = getattr(w, attr).reshape(getattr(w, attr).shape[0], -1)
        col = flat[:, j]
        thr = np.quantile(col, BIAS_QUANTILE)
        col[col >= thr] = np.clip(col[col >= thr] + BIAS_DELTA, 0.0, 1.0)
    elif model == "heavy":
        for _, arr in _blocks(w):
            noise = rng.standard_t(2, size=arr.shape) * HEAVY_SCALE
            arr[...] = np.clip(arr + noise, 0.0, 1.0)
    return w


def crude_world(world: World) -> World:
    """The fallback comparator: true scores quantized to
    3 uniform levels (deterministic, no rng)."""
    w = copy.deepcopy(world)
    for _, arr in _blocks(w):
        arr[...] = np.round(arr * 2.0) / 2.0
    return w
