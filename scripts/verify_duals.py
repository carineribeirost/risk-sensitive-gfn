"""Numerical verification of the DRO-CVaR duals in `epgfn.risk`.

The duals are used as ground truth everywhere in the study — the exact
targets, the oracle comparisons, the hardness probes — so the appendix
should be able to say they were checked rather than asserted. Three
families of check:

  identities   the ones that must hold to machine precision: rho = 0
               reduces to the plain CVaR, the reflection
               Phi+(a) = -Phi-(-a), and DRO <= nominal
  inner sup    each ball's inner maximization against brute force over a
               discretized simplex. The code value must be >= the grid
               maximum (the grid cannot reach the true optimum) and the
               excess must be on the order of the grid spacing
  Sion swap    the implementation computes max_tau [ tau - (1/beta) *
               sup_q E_q (tau-a)^+ ], i.e. the swapped form. This is
               compared against inf_q CVaR_beta(a; q) evaluated directly
               over the discretized ball — the swap itself, not just the
               inner dual. Reported at several grid resolutions, since
               only convergence under refinement distinguishes a genuine
               gap from grid coarseness.

Usage:
    python scripts/verify_duals.py            # ~1 min
    python scripts/verify_duals.py --quick
Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from epgfn.risk import (  # noqa: E402
    BALL_RHO_GRIDS, GEOMETRIES, _sup_chi2_linear, _sup_kl_linear,
    _sup_tv_linear, cvar_lower, dro_cvar_lower, dro_cvar_upper)

# radii at which the checks run: the largest registered grid value per
# ball, i.e. the most stressed case the study actually prices
RHO = {g: max(BALL_RHO_GRIDS[g]) for g in GEOMETRIES}

INNER = {"kl": lambda f, w, r: _sup_kl_linear(f, np.log(w), r),
         "tv": _sup_tv_linear,
         "chi2": _sup_chi2_linear}


def simplex_grid(k: int, step: float) -> np.ndarray:
    """All lattice points of the k-simplex with the given spacing."""
    n = int(round(1 / step))
    rows = []
    for c in itertools.combinations(range(n + k - 1), k - 1):
        prev, parts = -1, []
        for x in c:
            parts.append(x - prev - 1)
            prev = x
        parts.append(n + k - 2 - prev)
        rows.append(parts)
    return np.array(rows, dtype=float) / n


def divergence(q: np.ndarray, w: np.ndarray, geometry: str) -> np.ndarray:
    """Row-wise divergence of q from w, matching each ball's convention.

    TV is the half-L1 (mass-moved) convention: `_sup_tv_linear` moves a
    mass of rho, which is TV only under this convention. chi2 is the
    modified chi-square of the docstring, sum (q-w)^2 / w.
    """
    if geometry == "kl":
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(q > 0, q * np.log(np.maximum(q, 1e-300) / w), 0.0)
        return t.sum(-1)
    if geometry == "tv":
        return 0.5 * np.abs(q - w).sum(-1)
    if geometry == "chi2":
        return ((q - w) ** 2 / w).sum(-1)
    raise ValueError(geometry)


def check_identities(rng, n: int = 200) -> list[tuple]:
    """Exact algebraic identities — these must hold to machine zero."""
    out = []
    zero_err = refl_err = 0.0
    mono_ok = rho_mono_ok = True
    for _ in range(n):
        k = int(rng.integers(3, 8))
        a = rng.normal(size=(4, k))
        w = rng.dirichlet(np.ones(k) * 2)
        beta = float(rng.uniform(0.15, 1.0))
        nominal = cvar_lower(a, w, beta)
        for g in GEOMETRIES:
            zero_err = max(zero_err, float(np.abs(
                dro_cvar_lower(a, w, beta, 0.0, geometry=g) - nominal).max()))
            lo = dro_cvar_lower(a, w, beta, RHO[g], geometry=g)
            up = dro_cvar_upper(a, w, beta, RHO[g], geometry=g)
            refl_err = max(refl_err, float(np.abs(
                up + dro_cvar_lower(-a, w, beta, RHO[g],
                                    geometry=g)).max()))
            mono_ok &= bool(np.all(lo <= nominal + 1e-9))
            # a larger ball can only be more pessimistic
            half = dro_cvar_lower(a, w, beta, RHO[g] / 2, geometry=g)
            rho_mono_ok &= bool(np.all(lo <= half + 1e-9))
    out.append(("rho = 0 reduces to the plain CVaR",
                f"max abs err {zero_err:.1e}", zero_err == 0.0))
    out.append(("reflection  Phi+(a) = -Phi-(-a)",
                f"max abs err {refl_err:.1e}", refl_err == 0.0))
    out.append(("DRO-CVaR <= nominal CVaR", "all instances", mono_ok))
    out.append(("monotone non-increasing in rho", "all instances",
                rho_mono_ok))
    return out


def check_inner_sup(rng, steps, n: int = 25) -> list[tuple]:
    """Each ball's inner maximization against a brute-force simplex grid."""
    out = []
    for k, step in steps:
        grid = simplex_grid(k, step)
        gaps = {g: 0.0 for g in GEOMETRIES}
        below = {g: 0.0 for g in GEOMETRIES}
        for _ in range(n):
            f = np.abs(rng.normal(size=k))
            w = rng.dirichlet(np.ones(k) * 2)
            for g in GEOMETRIES:
                feas = grid[divergence(grid, w, g) <= RHO[g] + 1e-12]
                best = float((feas @ f).max())
                got = float(INNER[g](f, w, RHO[g]))
                gaps[g] = max(gaps[g], got - best)
                below[g] = min(below[g], got - best)
        for g in GEOMETRIES:
            # the code must never fall BELOW a feasible point's value
            ok = below[g] > -1e-9 and gaps[g] < 30 * step
            out.append((f"inner sup, {g}, K={k}, grid step {step}",
                        f"code - grid max in [{below[g]:+.1e}, "
                        f"{gaps[g]:+.1e}]", ok))
    return out


def check_sion(rng, steps, n: int = 12) -> list[tuple]:
    """The swap itself: the dual's value against inf_q CVaR over the ball.

    The grid infimum is an upper bound on the true one, so the code must
    sit at or below it, and the gap must shrink as the grid refines.
    """
    out = []
    # every registered radius, not just the largest: at the widest ball
    # the minimizer often sits on a simplex vertex the grid represents
    # exactly, which makes the check pass trivially. Interior optima at
    # the smaller radii are where the swap is actually exercised.
    cases = [(g, r) for g in GEOMETRIES for r in BALL_RHO_GRIDS[g] if r > 0]
    trace = {c: [] for c in cases}
    # One set of draws, reused at every grid step. Drawing inside the step
    # loop would compare a different random problem at each refinement, and
    # the trace would then be noise rather than convergence.
    draws = [(rng.normal(size=3), rng.dirichlet(np.ones(3) * 2))
             for _ in range(n)]
    for step in steps:
        grid = simplex_grid(3, step)
        worst = {c: 0.0 for c in cases}
        above = {c: 0.0 for c in cases}
        for a, w in draws:
            beta = 0.25
            for g, rho in cases:
                feas = grid[divergence(grid, w, g) <= rho + 1e-12]
                direct = float(cvar_lower(np.broadcast_to(a, feas.shape),
                                          feas, beta).min())
                code = float(dro_cvar_lower(a, w, beta, rho, geometry=g))
                worst[(g, rho)] = max(worst[(g, rho)], abs(code - direct))
                above[(g, rho)] = max(above[(g, rho)], code - direct)
        for c in cases:
            trace[c].append((step, worst[c], above[c]))
    for g, rho in cases:
        t = trace[(g, rho)]
        line = ", ".join(f"step {s}: {w:.1e}" for s, w, _ in t)
        # the code must never exceed the grid infimum, and refining the
        # grid must not push the gap up (it converges down, or is already
        # at floating-point tolerance)
        never_above = all(a < 1e-6 for _, _, a in t)
        converging = t[-1][1] <= t[0][1] + 1e-6
        out.append((f"Sion swap, {g} rho={rho:g}: max |dual - inf_q CVaR|",
                    line, never_above and converging))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    inner_steps = [(3, 0.004), (4, 0.02)] if args.quick else \
        [(3, 0.002), (4, 0.01)]
    sion_steps = [0.008, 0.002] if args.quick else [0.008, 0.004, 0.002]

    print("radii under test (largest registered per ball): "
          + ", ".join(f"{g}={RHO[g]:g}" for g in GEOMETRIES) + "\n")
    rows = (check_identities(rng, 60 if args.quick else 200)
            + check_inner_sup(rng, inner_steps, 12 if args.quick else 25)
            + check_sion(rng, sion_steps, 8 if args.quick else 12))
    width = max(len(r[0]) for r in rows)
    failed = 0
    for name, detail, ok in rows:
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print(f"\n{len(rows) - failed}/{len(rows)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
