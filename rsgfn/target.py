"""Exact target p*, distances, and the Monte-Carlo floor.

p*_c is computed by full enumeration of X and is independent of any
sampler.
"""

from __future__ import annotations

import numpy as np


def p_star(log_reward: np.ndarray, beta_t: float) -> np.ndarray:
    """softmax(β_t · log R) over the enumerated X."""
    z = beta_t * log_reward
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def l1(p: np.ndarray, q: np.ndarray) -> float:
    """L1 distance Σ|p − q| (O2 metric). Equals 2·TV."""
    return float(np.abs(p - q).sum())


def tv(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation distance (O1 metric)."""
    return 0.5 * l1(p, q)


def mc_floor(p: np.ndarray, n_samples: int, rng: np.random.Generator,
             n_reps: int = 20) -> float:
    """Finite-sample floor: mean L1 of an exact n-sample empirical
    estimate of p against p (O2 reference)."""
    dists = []
    for _ in range(n_reps):
        counts = rng.multinomial(n_samples, p)
        dists.append(l1(counts / n_samples, p))
    return float(np.mean(dists))
