"""Statistical protocol: hierarchical cluster bootstrap,
cluster-level permutation test with effect size, and TOST equivalence.

Data enter as `groups`: a mapping cluster_id → 1-D array of measurements
(worlds are clusters; seed replicates live inside a cluster's array), so
correlated within-world measurements are never treated as independent.
"""

from __future__ import annotations

import numpy as np


def _cluster_means(groups: dict) -> np.ndarray:
    return np.array([np.mean(v) for v in groups.values()], dtype=float)


def _cohens_d(obs: float, sd: float) -> float:
    """Effect size, with the two degenerate cases kept apart. A zero
    spread around a non-zero mean is a genuinely unbounded effect
    (inf); a zero spread around a zero mean is the strongest possible
    null and must report 0.0, not inf."""
    if sd > 0:
        return float(obs / sd)
    return 0.0 if obs == 0.0 else np.inf


def cluster_bootstrap_ci(groups: dict, n_boot: int = 2000,
                         alpha: float = 0.05, seed: int = 0) -> tuple:
    """Percentile CI for the grand mean; outer resampling over clusters,
    nested resampling of within-cluster replicates."""
    rng = np.random.default_rng(seed)
    vals = [np.asarray(v, dtype=float) for v in groups.values()]
    n_c = len(vals)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_c, size=n_c)
        means = []
        for i in pick:
            v = vals[i]
            if len(v) > 1:
                v = v[rng.integers(0, len(v), size=len(v))]
            means.append(v.mean())
        stats[b] = np.mean(means)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    point = float(np.mean([v.mean() for v in vals]))
    return point, float(lo), float(hi)


def cluster_permutation_test(groups_a: dict, groups_b: dict,
                             n_perm: int = 5000, seed: int = 0) -> dict:
    """Two-sided cluster-level permutation test of equal means, with
    Cohen's d on cluster means as the effect size."""
    rng = np.random.default_rng(seed)
    ma, mb = _cluster_means(groups_a), _cluster_means(groups_b)
    obs = ma.mean() - mb.mean()
    pooled = np.concatenate([ma, mb])
    na = len(ma)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(pooled[:na].mean() - pooled[na:].mean()) >= abs(obs):
            count += 1
    sd = np.sqrt(((na - 1) * ma.var(ddof=1)
                  + (len(mb) - 1) * mb.var(ddof=1))
                 / (na + len(mb) - 2))
    return {"diff": float(obs),
            "p_value": (count + 1) / (n_perm + 1),
            "cohens_d": _cohens_d(obs, sd)}


def paired_sign_permutation(deltas, n_perm: int = 5000,
                            seed: int = 0) -> dict:
    """One-sample sign-flip permutation test on per-cluster paired
    differences (per-world mean Δ vs zero). Two-sided
    p on the mean under random sign flips; Cohen's d on the deltas."""
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, dtype=float)
    obs = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm = (signs * d).mean(axis=1)
    count = int((np.abs(perm) >= abs(obs)).sum())
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    return {"mean": float(obs),
            "p_value": (count + 1) / (n_perm + 1),
            "cohens_d": _cohens_d(obs, sd),
            "n": int(len(d))}


def tost(groups_diff: dict, margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided tests on cluster-mean differences against a
    pre-declared margin. Equivalent iff both p < alpha."""
    from scipy import stats as sps
    d = _cluster_means(groups_diff)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t_lo = (d.mean() + margin) / se   # H0: mean <= -margin
    t_hi = (d.mean() - margin) / se   # H0: mean >= +margin
    p_lo = float(sps.t.sf(t_lo, df=n - 1))
    p_hi = float(sps.t.cdf(t_hi, df=n - 1))
    return {"mean": float(d.mean()), "p_lower": p_lo, "p_upper": p_hi,
            "equivalent": bool(max(p_lo, p_hi) < alpha)}


def holm(pvals: dict, alpha: float = 0.05) -> dict:
    """Holm–Bonferroni step-down over a named family of p-values.
    Returns {name: {"p": raw, "p_adj": adjusted, "reject": bool}};
    adjusted p is the standard monotone max-rank form."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    rejecting = True
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        rejecting = rejecting and (running < alpha)
        out[name] = {"p": float(p), "p_adj": float(running),
                     "reject": bool(rejecting)}
    return out
