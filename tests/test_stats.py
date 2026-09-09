"""Statistical protocol sanity tests."""

import numpy as np

from epgfn.stats import (cluster_bootstrap_ci, cluster_permutation_test,
                         tost)


def _groups(rng, n_clusters, mu, cluster_sd=0.5, noise_sd=0.2, n_rep=4):
    out = {}
    for c in range(n_clusters):
        center = mu + rng.normal(0, cluster_sd)
        out[c] = center + rng.normal(0, noise_sd, size=n_rep)
    return out


def test_bootstrap_ci_covers_mean():
    """The cluster bootstrap CI contains both the point estimate and the true
    mean."""
    rng = np.random.default_rng(0)
    g = _groups(rng, 30, mu=2.0)
    point, lo, hi = cluster_bootstrap_ci(g, n_boot=500)
    assert lo < point < hi
    assert lo < 2.0 < hi


def test_permutation_detects_shift_and_not_null():
    """The cluster permutation test detects a real mean shift with large effect
    size but stays non-significant under the null."""
    rng = np.random.default_rng(1)
    a = _groups(rng, 20, mu=0.0)
    b = _groups(rng, 20, mu=2.0)
    res = cluster_permutation_test(a, b, n_perm=1000)
    assert res["p_value"] < 0.01 and abs(res["cohens_d"]) > 1
    a2 = _groups(rng, 20, mu=0.0)
    res_null = cluster_permutation_test(a, a2, n_perm=1000)
    assert res_null["p_value"] > 0.05


def test_tost_equivalence():
    """TOST declares near-identical cluster means equivalent within margin but
    rejects equivalence once the means are shifted apart."""
    rng = np.random.default_rng(2)
    near_zero = _groups(rng, 25, mu=0.0, cluster_sd=0.05, noise_sd=0.02)
    res = tost(near_zero, margin=0.2)
    assert res["equivalent"]
    shifted = _groups(rng, 25, mu=1.0, cluster_sd=0.05, noise_sd=0.02)
    assert not tost(shifted, margin=0.2)["equivalent"]
