# Risk-Sensitive Reward Composition for Conditional GFlowNets

Risk-sensitive reward aggregation (CVaR + KL-DRO) on an exactly
enumerable 2-D grid, with a conditional GFlowNet sampler validated
against exact targets.

## Layout

| module | contents |
|---|---|
| `epgfn/risk.py` | CVaR⁻/CVaR⁺ (exact split-mass), KL-DRO duals, nested & flattened composition |
| `epgfn/worlds.py` | grid, smooth score geometries, weights, `g(x)`, faithful-hardness gate |
| `epgfn/cases.py` | cases A–D, floor/veto/ε, `log R`, exact `p*` per condition |
| `epgfn/conditions.py` | condition vector `c`, ranges, held-out grid, network features |
| `epgfn/target.py` | `p*`, TV/L1, Monte-Carlo finite-sample floor |
| `epgfn/policy.py` | conditional policy (FiLM MLP), `log Z(c)` head, exact density enumeration |
| `epgfn/train.py` | trajectory-balance training, held-out exclusion, exact evaluation |
| `epgfn/o1.py`, `scripts/run_o1.py` | O1 sweep driver (vs Boltzmann and worst-case poles) |
| `epgfn/o3.py`, `scripts/run_o3.py` | O3 extension: joint-satisfaction utility + stress test vs both poles |
| `scripts/run_o2.py` | O2 driver: (world, seed) grid, floor-referenced L1, bootstrap CI |
| `epgfn/stats.py` | cluster bootstrap, cluster permutation test, TOST |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest        # oracle test suite
```

Device-agnostic; CPU works for tests and the O1/O3 exact sweeps. For
GPU training, install a CUDA build of PyTorch >= 2.9 before the
editable install.

## Running

```bash
# O1: exact sweeps, no training (minutes, CPU)
.venv/bin/python scripts/run_o1.py --cases A B C D --worlds 8 --H 32

# O2: conditional-policy amortization per case
.venv/bin/python scripts/run_o2.py --case A --worlds 4 --seeds 3 \
    --H 32 --steps 4000
```

```bash
# O3: is risk-on BETTER than the Boltzmann / worst-case poles at the
# declared utility (joint satisfaction of all states + constraints)?
.venv/bin/python scripts/run_o3.py --cases A B C D --worlds 3 --H 16
```

Each invocation writes into its own timestamped subdirectory of
`--out` (with an `args.json` manifest), so concurrent launches never
overwrite each other.

## O3: joint-satisfaction utility

O1/O2 prove the targets are *distinct* and *amortizable*; they cannot
say "better" because `p*_c` itself defines the goal. O3 adds a
declared utility: **generate objects that jointly satisfy all active
states and the case's structural conditions**: every good state ≥ t,
every penalise state ≤ 1 − t, no veto (the Case-B floor is
reward-shaping on Φ, hence method-dependent, and deliberately excluded
from the utility). Per target (Boltzmann pole, worst-case pole, risk-on
grid) it reports, exactly: `sat_mass` (probability one draw satisfies),
`eff_candidates` (effective number of distinct satisfying candidates,
worst-case's failure mode), and `stress_mean`/`stress_p05` (realized
utility under paired Dirichlet weight perturbations; Boltzmann's
failure mode is the tail). Prediction, declared in advance: risk-on beats
Boltzmann on `sat_mass`/`stress_p05` and beats worst-case on
`eff_candidates`/`stress_mean`, tracing a Pareto frontier between the
poles. Each world also reports its *challenge level* `t*`, the
hardest level at which ≥1% of X still satisfies.

## Design decisions

1. **Φ notation.** Functionals are labeled by tail side in `risk.py`,
   and by the set they act on elsewhere in this codebase. Resolved:
   S⁺/S⁰ (and both Case-D levels) get the
   *lower-tail* DRO-CVaR (conservative on what should be good); S⁻ gets
   the *upper-tail* DRO-CVaR (pessimistic on what should be small);
   Case B is `Ψ = Φ(S⁺) − γ·Φ(S⁻)` with the floor on `Φ(S⁺)`.
   The floor level `f` is calibrated per world as a quantile (default
   0.35) of `Φ(S⁺)` at a fixed mid-sweep reference condition, since a
   universal constant fails because the scale of `Φ(S⁺)` is world- and
   condition-dependent (measured: fail fractions of 0.90–0.99 with
   `f = 0.5`, i.e. the floor was inert-by-saturation).
2. **DRO evaluation.** Exact dual forms (Sion + entropic KL dual),
   nested vectorized ternary searches; `Φ⁺` derived from `Φ⁻` via the
   reflection identity. Verified in tests against all closed-form limits, a
   direct SLSQP primal solve, and the β=1 entropic-mean dual.
3. **Case D conditioning.** The condition's `(β_cvar, ρ)` drives the
   *inner* functional; the outer level uses fixed world parameters
   (`beta_out=0.5, rho_out=0.3`), keeping the two levels distinct.
4. **ε policy.** `ε = 1e-4` (`cases.EPS_REWARD`): applied on vetoed and
   floored points and as the positivity clamp for Case B's signed
   objective. Chosen so the worst log-reward gap (`β_t = 8`)
   stays within float32 softmax range while keeping vetoed mass
   genuinely negligible.
5. **Hardness gate.** Rejection sampling on worlds with probes:
   TV(risk-on, risk-off) ≥ 0.05, veto share in [2%, 50%] (C), floor-fail
   share in [5%, 70%] (B), TV(nested, flat) ≥ 0.05 (D). Attempt counts
   are always reported (no silent caps).
6. **`g(x)`.** Independent random smooth field by default (independent
   draw, same family as the scores, nothing engineered);
   `WorldConfig(g_iid=True)` switches to i.i.d. uniform.
7. **TB off-policy mix.** Each batch is 50% on-policy / 50% uniform
   over X, so O2 measures amortization capacity, not exploration.
   `log Z` is a head on the condition embedding (never a scalar) with
   10× the trunk learning rate.
8. **Held-out conditions.** Deterministic grid per world; training
   draws inside an L∞ ball (radius 0.05 in feature space) of any
   held-out point are rejected.
9. **Dev/test worlds.** Split by disjoint seed blocks
   (`--world-seed0`); fix any free parameters on the dev block, report
   on the test block only.

## License

CC BY 4.0 — see `LICENSE`.
