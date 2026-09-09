"""Gate diagnostic: scan the deterministic attempt chain of one
(case, config, seed) and print every hardness probe against its
threshold: distinguishes a STRUCTURAL gate failure (some probe
collapsed far below its margin in every draw, as in the C s=4.0
veto-share postmortem) from a THIN-TAIL one (probes hover near the
margin; raising max_attempts suffices, identity-preserving since the
chain is a deterministic prefix).

Attempts are independent draws, so they are evaluated in parallel and
reported in chain order: byte-identical values to the sequential
gate, which only ever reads attempt i's own world.

Usage (D s=4.0 seed 1, the 2026-07-18 exhaustion):
    python scripts/probe_gate.py --case D --sparsity 4.0 --seed 1 \
        --H 4 --d 8 --geometry sequence --attempts 200 --jobs 24
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from epgfn.worlds import WorldConfig, hardness_report, is_hard, make_world

# probe -> (threshold attr or value, direction) resolved per case in
# `criteria`; kept in one place so the printout names every check the
# gate actually applies (worlds.is_hard)


def criteria(case: str, cfg: WorldConfig) -> list:
    """[(probe key, op, threshold, label)] matching worlds.is_hard."""
    m = cfg.hardness_margin
    crit = [("tv_on_off", ">=", m, "tv_on_off>=margin")]
    if case == "B":
        lo, hi = cfg.floor_frac_range
        crit += [("floor_fail_frac", ">=", lo, "floor_fail>=lo"),
                 ("floor_fail_frac", "<=", hi, "floor_fail<=hi"),
                 ("tv_origins", ">=", m, "tv_origins>=margin")]
    if case == "C":
        lo, hi = cfg.veto_frac_range
        crit += [("veto_frac", ">=", lo, "veto_frac>=lo"),
                 ("veto_frac_delta_max", "<=", hi, "veto@dmax<=hi")]
    if case == "D":
        crit += [("tv_nested_flat", ">=", m, "tv_nested_flat>=margin"),
                 ("tv_origins", ">=", m, "tv_origins>=margin"),
                 ("tv_beta_out", ">=", m, "tv_beta_out>=margin")]
    return crit


def probe_one(args_tuple):
    """Build one attempt's world and evaluate its hardness probes.

    Args:
        args_tuple: `(case, cfg, seed, attempt)`; the world is built
            at raw seed `seed * 100_003 + attempt`, matching the
            gate's deterministic attempt chain.

    Returns:
        `(attempt, probe_values, hard)`: `probe_values` is the
        numeric subset of `hardness_report(world)`; `hard` is
        `is_hard(world, rep)`.
    """
    case, cfg, seed, attempt = args_tuple
    # raw seed for this attempt of the base seed; 100_003 is large enough
    # that attempt never collides with the next base seed's own chain
    world = make_world(case, cfg, seed * 100_003 + attempt)
    rep = hardness_report(world)
    return attempt, {k: v for k, v in rep.items()
                     if isinstance(v, (int, float))}, is_hard(world, rep)


def main() -> None:
    """Scan one (case, config, seed)'s attempt chain against the hardness gate.

    Parses CLI args, evaluates --attempts worlds (in parallel when
    --jobs > 1) against the criteria for --case, prints a per-attempt
    table of probe values with pass/fail flags, then a per-criterion
    summary (median/max/min/fail share) that distinguishes a
    structural gate failure from a thin-tail one.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=list("ABCD"))
    ap.add_argument("--sparsity", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--geometry", default="sequence",
                    choices=["sequence", "grid"])
    ap.add_argument("--attempts", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    cfg = WorldConfig(H=args.H, d=args.d, geometry=args.geometry,
                      sparsity=args.sparsity)
    crit = criteria(args.case, cfg)
    keys = list(dict.fromkeys(k for k, _, _, _ in crit))
    units = [(args.case, cfg, args.seed, a)
             for a in range(args.attempts)]
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(probe_one, units, chunksize=1))
    else:
        results = [probe_one(u) for u in units]

    vals = {k: [] for k in keys}
    n_pass, first_pass = 0, None
    print("attempt  " + "  ".join(f"{k:>18}" for k in keys) + "  hard")
    for attempt, rep, hard in results:
        for k in keys:
            vals[k].append(rep.get(k, float("nan")))
        flags = " ".join(
            "ok" if (rep.get(k, np.nan) >= t if op == ">=" else
                     rep.get(k, np.nan) <= t) else "FAIL"
            for k, op, t, _ in crit)
        print(f"{attempt:>7}  "
              + "  ".join(f"{rep.get(k, float('nan')):>18.4f}"
                          for k in keys)
              + f"  {'HARD' if hard else '-':>4}  [{flags}]")
        if hard:
            n_pass += 1
            first_pass = attempt if first_pass is None else first_pass

    print(f"\n{args.case} s={args.sparsity} seed {args.seed}: "
          f"{n_pass}/{args.attempts} attempts pass"
          + (f", first at attempt {first_pass} "
             f"(gate_attempts={first_pass + 1})"
             if first_pass is not None else ""))
    for k, op, t, label in crit:
        v = np.array(vals[k], dtype=float)
        print(f"  {label:>24}: threshold {t:.3f} | median "
              f"{np.nanmedian(v):.4f}  max {np.nanmax(v):.4f}  "
              f"min {np.nanmin(v):.4f}  fail_share "
              f"{float(np.mean(~(v >= t) if op == '>=' else ~(v <= t))):.2f}")
    print("verdict hint: a criterion whose max sits far below (>=) or "
          "above (<=) its threshold in EVERY draw is STRUCTURAL "
          "(cf. C-s4 veto share, 14x below); values hovering at the "
          "threshold with occasional passes are THIN-TAIL: raise "
          "max_attempts (identity-preserving deterministic prefix).")


if __name__ == "__main__":
    main()
