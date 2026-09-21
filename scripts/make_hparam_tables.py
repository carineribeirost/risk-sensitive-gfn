"""Assemble the appendix hyperparameter tables from the results tree.

Each run directory records only its CLI surface in `args.json`; everything
else (optimizer, architecture, condition ranges, world family constants)
lives as dataclass defaults in `epgfn/`. Neither half is a table on its
own. This script joins them — defaults introspected from the dataclasses
at run time, per-run overrides read from `args.json` — and writes:

  hparams.json  the full resolved configuration of every run directory.
                This is the record `results/` is missing: without it a run
                can only be reconstructed by reading the working tree that
                produced it, and nothing pins that tree.
  hparams.tex   the appendix tables — architecture, optimization defaults,
                world/condition defaults, and one row per result family
                listing only what that family changed.

No silent drops (spec's "no silent caps" applied to provenance): every key
in every args.json is either wired to a config field, declared a
script-level knob, or reported as UNMAPPED on stderr and in the JSON.

Usage:
    python scripts/make_hparam_tables.py                 # -> report/
    python scripts/make_hparam_tables.py --out /tmp/x
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from epgfn.conditions import ConditionRanges          # noqa: E402
from epgfn.policy import ConditionalPolicy            # noqa: E402
from epgfn.train import TrainConfig                   # noqa: E402
from epgfn.worlds import WorldConfig                  # noqa: E402
from epgfn.risk import BALL_RHO_GRIDS, GEOMETRIES     # noqa: E402


# ---------------------------------------------------------------- wiring
# How each producing script maps its CLI args onto the config objects,
# transcribed from the WorldConfig(...) / TrainConfig(...) call sites.
# ("world"|"train"|"ranges", field) — or one of the two sentinels:
#   SWEEP  the arg is a sweep axis (its values become table rows/columns
#          elsewhere in the paper, not a fixed hyperparameter)
#   KNOB   script-level plumbing with no effect on the model or the
#          target (paths, job counts, resume flags, device)
SWEEP = ("sweep", None)
KNOB = ("knob", None)

# `--cond-pool 0` / `--logit-floor 0` mean "off" at most call sites
# (`args.cond_pool or None`); run_oracle_gap passes cond_pool raw.
OR_NONE = "or_none"

_COMMON = {
    "out": KNOB, "device": KNOB, "jobs": KNOB, "threads_per_job": KNOB,
    "resume": KNOB, "trust_recorded_gates": KNOB, "cases_note": KNOB,
    "case": SWEEP, "cases": SWEEP, "worlds": SWEEP, "world_seed0": SWEEP,
    "world_seeds": SWEEP, "seeds": SWEEP, "seed0": SWEEP, "seed": SWEEP,
    "world_seed": SWEEP,
    "beta_grid": SWEEP, "rho_grid": SWEEP, "rho_out_grid": SWEEP,
    "ball": SWEEP, "levels": SWEEP, "models": SWEEP, "arms": SWEEP,
    "mixes": SWEEP, "factor": SWEEP, "sparsity": SWEEP, "wg_grid": SWEEP,
    "kappa_range": SWEEP,
    # floor_samples=None resolves to 10*H^d inside the run scripts; it is
    # a Monte-Carlo reference-floor sample count, not a model knob
    "floor_samples": KNOB,
}

_WORLD = {k: ("world", k) for k in
          ("H", "d", "geometry", "weight_alpha", "k_guard", "sparsity",
           "max_attempts")}
_RANGE_FLAGS = {k: ("ranges", k) for k in
                ("tie_beta", "use_sigma", "use_rho_out", "use_guard_delta")}

WIRING = {
    # analytic (enumeration only — no policy is trained)
    "run_o1.py": {**_COMMON, **{k: _WORLD[k] for k in
                                ("H", "d", "geometry", "weight_alpha",
                                 "k_guard")},
                  "no_gate": ("world", "_gate_off"), "margin": SWEEP},
    "run_o3.py": {**_COMMON, **{k: _WORLD[k] for k in
                                ("H", "d", "geometry", "weight_alpha")},
                  "no_gate": ("world", "_gate_off"), "contam_eps": SWEEP,
                  "kappa": SWEEP, "n_stress": SWEEP, "k_guard": SWEEP},
    "run_recovery.py": {**_COMMON, **{k: _WORLD[k] for k in
                                      ("H", "d", "geometry",
                                       "weight_alpha")},
                        "winsor_k": SWEEP},
    # trained
    "run_o2.py": {**_COMMON, **{k: _WORLD[k] for k in
                                ("H", "d", "geometry", "weight_alpha",
                                 "k_guard")},
                  **_RANGE_FLAGS,
                  "steps": ("train", "steps"), "loss": ("train", "loss"),
                  "cond_pool": ("train", "cond_pool", OR_NONE),
                  "logit_floor": ("train", "logit_floor", OR_NONE)},
    "run_w33.py": {**_COMMON, "H": _WORLD["H"], "d": _WORLD["d"],
                   "geometry": _WORLD["geometry"],
                   "max_attempts": _WORLD["max_attempts"],
                   "steps": ("train", "steps"),
                   "n_points": ("train", "n_points"),
                   "eval_every": ("train", "eval_every"),
                   "logit_floor": ("train", "logit_floor", OR_NONE),
                   "alpha_aux": SWEEP, "buf_cap": SWEEP},
    "run_oracle_gap.py": {**_COMMON, **{k: _WORLD[k] for k in
                                        ("H", "d", "geometry")},
                          "use_sigma": _RANGE_FLAGS["use_sigma"],
                          "steps": ("train", "steps"),
                          # raw, not `or None`
                          "cond_pool": ("train", "cond_pool"),
                          "logit_floor": ("train", "logit_floor", OR_NONE)},
    "run_shrink.py": {**_COMMON, "H": _WORLD["H"],
                      "steps": ("train", "steps"),
                      "cond_pool": ("train", "cond_pool", OR_NONE),
                      "logit_floor": ("train", "logit_floor", OR_NONE)},
    "run_mix.py": {**_COMMON, "H": _WORLD["H"],
                   "steps": ("train", "steps"),
                   "cond_pool": ("train", "cond_pool", OR_NONE),
                   "logit_floor": ("train", "logit_floor", OR_NONE)},
    "run_experts.py": {**_COMMON, "H": _WORLD["H"],
                       "steps": ("train", "steps"),
                       "cond_pool": ("train", "cond_pool"),
                       "logit_floor": ("train", "logit_floor", OR_NONE)},
    "run_dtying.py": {**_COMMON, "H": _WORLD["H"],
                      "steps": ("train", "steps"),
                      "cond_pool": ("train", "cond_pool", OR_NONE),
                      "logit_floor": ("train", "logit_floor", OR_NONE)},
    "probe_convergence.py": {**_COMMON, **{k: _WORLD[k] for k in
                                           ("H", "d", "geometry")},
                             "steps": ("train", "steps"),
                             "eval_every": ("train", "eval_every"),
                             "use_sigma": _RANGE_FLAGS["use_sigma"],
                             "cond_pool": ("train", "cond_pool", OR_NONE),
                             "logit_floor": ("train", "logit_floor",
                                             OR_NONE)},
    "probe_gate.py": {**_COMMON, **{k: _WORLD[k] for k in
                                    ("H", "d", "geometry", "sparsity",
                                     "weight_alpha", "max_attempts")}},
    "probe_pinned_seeds.py": {**_COMMON, **{k: _WORLD[k] for k in
                                            ("H", "d", "geometry",
                                             "sparsity")}},
    "probe_wg_drowning.py": {**_COMMON, "H": _WORLD["H"], "margin": SWEEP},
}

# The launch_*.py wrappers fan out over (case, world) units and forward
# every model-relevant flag verbatim to the run script beneath them, so
# they share its wiring (the parallelism knobs are already in _COMMON).
WIRING["launch_o2.py"] = WIRING["run_o2.py"]
WIRING["launch_oracle.py"] = WIRING["run_oracle_gap.py"]
WIRING["launch_w33.py"] = WIRING["run_w33.py"]

# Flags whose default is a documented identity (adding them left every
# earlier path bitwise unchanged), so resolving a run that predates the
# flag to today's default is exact. Any OTHER unrecorded flag is an
# assumption — the run predates the feature and its true value may differ
# from the current default (e.g. the logit clamp did not exist, so those
# runs trained unclamped rather than at -25).
NO_OP_DEFAULTS = {"geometry", "weight_alpha", "use_sigma", "k_guard",
                  "loss", "use_guard_delta", "use_rho_out", "d",
                  "tie_beta", "no_gate", "max_attempts", "sparsity"}

# scripts that never construct a TrainConfig — their rows carry no
# optimization column (a trained-model hyperparameter table would be
# misleading for an exact-enumeration run)
ANALYTIC = {"run_o1.py", "run_o3.py", "run_recovery.py", "probe_gate.py",
            "probe_pinned_seeds.py", "probe_wg_drowning.py"}


def script_options() -> dict[str, set[str]]:
    """Harvest each script's argparse option names (dest form)."""
    pat = re.compile(r"""add_argument\(\s*["']--([A-Za-z0-9\-_]+)""")
    opts = {}
    for path in sorted((ROOT / "scripts").glob("*.py")):
        names = {m.group(1).replace("-", "_")
                 for m in pat.finditer(path.read_text())}
        if names:
            opts[path.name] = names
    return opts


def identify(keys: set[str], opts: dict[str, set[str]],
             out_hint: str) -> tuple[str | None, set[str]]:
    """Best-matching producing script for one args.json key set.

    Returns (script name or None, keys the script does not declare).
    Scored by coverage of the recorded keys; ties broken by the smallest
    option set (the most specific parser), then by the `out` path hint.
    """
    best, best_score = None, None
    for name, names in opts.items():
        missing = keys - names
        hint = 0
        default_stem = re.sub(r"^(run_|probe_)|\.py$", "", name)
        if default_stem and default_stem.split("_")[0] in out_hint:
            hint = 1
        score = (-len(missing), hint, -len(names))
        if best_score is None or score > best_score:
            best, best_score = name, score
    return best, keys - opts.get(best, set())


def defaults_of(cls) -> dict:
    """Dataclass field defaults (default_factory resolved)."""
    out = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            out[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore
            out[f.name] = f.default_factory()               # type: ignore
    return out


def net_defaults() -> dict:
    sig = inspect.signature(ConditionalPolicy.__init__)
    return {n: p.default for n, p in sig.parameters.items()
            if p.default is not inspect.Parameter.empty}


def resolve(args: dict, script: str) -> dict:
    """Resolve one run's args.json against the dataclass defaults."""
    wiring = WIRING.get(script, {})
    world = defaults_of(WorldConfig)
    train = defaults_of(TrainConfig)
    ranges = defaults_of(ConditionRanges)
    sweeps, knobs, unmapped = {}, {}, {}
    for key, val in args.items():
        rule = wiring.get(key)
        if rule is None:
            unmapped[key] = val
            continue
        target, field = rule[0], rule[1]
        if target == "sweep":
            sweeps[key] = val
            continue
        if target == "knob":
            knobs[key] = val
            continue
        if len(rule) > 2 and rule[2] == OR_NONE:
            val = val or None
        if field == "_gate_off":       # `--no-gate` ablation
            world["hardness_gate"] = not val
            continue
        {"world": world, "train": train, "ranges": ranges}[target][field] = val
    world.setdefault("hardness_gate", True)
    rec = {"script": script, "world": world, "ranges": ranges,
           "sweeps": sweeps, "knobs": knobs, "unmapped": unmapped}
    if script in ANALYTIC:
        rec["train"] = None
        rec["net"] = None
    else:
        rec["train"] = train
        rec["net"] = net_defaults()
    return rec


def family_of(run_dir: Path, root: Path) -> str:
    return run_dir.relative_to(root).parts[0]


def collect(results: Path) -> tuple[dict, list]:
    """Walk every args.json under results/ (both the top-level tree and
    the nested results/results/ tree, which are complementary)."""
    opts = script_options()
    per_family = defaultdict(list)
    problems = []
    seen = {}
    for args_path in sorted(results.rglob("args.json")):
        try:
            args = json.loads(args_path.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"{args_path}: unreadable ({e})")
            continue
        run_dir = args_path.parent
        nested = "results/results" in run_dir.as_posix()
        root = results / "results" if nested else results
        try:
            fam = family_of(run_dir, root)
        except ValueError:
            problems.append(f"{args_path}: outside a family directory")
            continue
        script, missing = identify(set(args), opts,
                                   str(args.get("out", "")))
        # a key the parser does not declare but the wiring resolves is a
        # launcher-written field, not a provenance hole
        missing -= set(WIRING.get(script, {}))
        if missing:
            problems.append(f"{args_path}: {script} does not declare "
                            f"{sorted(missing)}")
        rec = resolve(args, script)
        rec["run_dir"] = run_dir.relative_to(ROOT).as_posix()
        rec["nested_tree"] = nested
        # a fan-out fragment written by a launcher inside its parent run
        rec["fragment"] = "frag_" in run_dir.as_posix()
        # Flags the current parser declares that this run did not record:
        # the script predates them, so their values here come from today's
        # defaults rather than from the run itself. Worth knowing before
        # quoting a number as recorded.
        rec["unrecorded_flags"] = sorted(
            k for k in opts.get(script, set()) - set(args)
            if WIRING.get(script, {}).get(k, KNOB)[0] in ("world", "train",
                                                          "ranges"))
        # the two trees overlap; identical (family, run, args) counts once
        key = (fam, run_dir.name, json.dumps(args, sort_keys=True))
        if key in seen:
            seen[key]["duplicate_of"] = seen[key].get("duplicate_of", 0) + 1
            continue
        seen[key] = rec
        per_family[fam].append(rec)
        if rec["unmapped"]:
            problems.append(f"{args_path}: UNMAPPED keys "
                            f"{sorted(rec['unmapped'])}")
    return dict(per_family), problems


# ------------------------------------------------------------ rendering
def tex_escape(s) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def fmt(v) -> str:
    if isinstance(v, bool):
        return r"\texttt{True}" if v else r"\texttt{False}"
    if v is None:
        return "---"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(fmt(x) for x in v)
    if isinstance(v, dict):
        return "---" if not v else tex_escape(json.dumps(v))
    return tex_escape(v)


def deltas(rec: dict) -> dict:
    """What this run changed relative to the dataclass defaults."""
    out = {}
    for name, cls in (("world", WorldConfig), ("train", TrainConfig),
                      ("ranges", ConditionRanges)):
        cur = rec.get(name)
        if cur is None:
            continue
        base = defaults_of(cls)
        base["hardness_gate"] = True
        for k, v in cur.items():
            if k in base and base[k] != v:
                out[k] = v
    return out


# The appendix carries one consolidated table rather than four. These are the
# fields a reader needs to reconstruct a run; everything omitted is either set
# per run, inert in every reported battery, or a documented identity. Pass
# --full to emit the four per-dataclass tables and the per-battery table
# instead.
ESSENTIAL = {
    "architecture": ["cond_dim", "dim", "depth", "x1_dim"],
    "optimisation": ["loss", "subtb_lambda", "steps", "n_conds", "n_points",
                     "uniform_mix", "lr", "lr_logz", "cond_pool",
                     "heldout_radius", "logit_floor", "eval_every"],
    "world": ["H", "d", "geometry", "sparsity", "k_neutral", "k_plus",
              "k_minus", "k_named", "n_outer", "k_inner", "weight_alpha",
              "gamma", "floor_quantile", "c_named", "beta_out", "rho_out",
              "hardness_margin", "veto_frac_range", "floor_frac_range",
              "delta_probe", "max_attempts"],
    "condition ranges": ["beta_t", "w_g", "rho", "delta"],
}


def essential_table(arch: dict) -> str:
    groups = [
        ("architecture", arch, ARCH_NOTES),
        ("optimisation", defaults_of(TrainConfig), TRAIN_NOTES),
        ("world", defaults_of(WorldConfig), WORLD_NOTES),
        ("condition ranges", defaults_of(ConditionRanges), RANGE_NOTES),
    ]
    lines = []
    for i, (name, src, notes) in enumerate(groups):
        if i:
            lines.append(r"    \addlinespace[3pt]")
        first = True
        for k in ESSENTIAL[name]:
            if k not in src:
                continue
            head = rf"\emph{{{name}}}" if first else ""
            first = False
            lines.append(rf"    {head} & \texttt{{{tex_escape(k)}}} & "
                         rf"{fmt(src[k])} & {notes.get(k, '')} \\")
    body = "\n".join(lines)
    return (
        "\\begin{table*}[htbp]\n  \\centering\n  \\footnotesize\n"
        "  \\caption{Configuration. Values are read from the dataclasses that "
        "ran (\\texttt{epgfn.policy.ConditionalPolicy}, "
        "\\texttt{epgfn.train.TrainConfig}, \\texttt{epgfn.worlds.WorldConfig}, "
        "\\texttt{epgfn.conditions.ConditionRanges}), so they cannot drift "
        "from the configuration they describe. Per-battery overrides and the "
        "flags that are inert in every reported result are recorded in the "
        "released results tree.}\n  \\label{tab:config}\n"
        "  \\begin{tabular}{@{}lllp{0.46\\textwidth}@{}}\n    \\toprule\n"
        "    Group & Field & Value & Meaning \\\\\n    \\midrule\n"
        f"{body}\n    \\bottomrule\n  \\end{{tabular}}\n"
        "\\end{table*}\n")


def defaults_table(title: str, label: str, rows: list, note: str) -> str:
    body = "\n".join(rf"    \texttt{{{tex_escape(k)}}} & {fmt(v)} & {d} \\"
                     for k, v, d in rows)
    # the appendix is set \onecolumn, so \linewidth is the full text width and
    # the meaning column can take more of it; [htbp] keeps the table inside the
    # appendix section that \input's it rather than deferring it to the end
    return (
        "\\begin{table}[htbp]\n  \\centering\n  \\footnotesize\n"
        f"  \\caption{{{title}}}\n  \\label{{tab:{label}}}\n"
        "  \\begin{tabular}{@{}llp{0.55\\linewidth}@{}}\n    \\toprule\n"
        "    Field & Value & Meaning \\\\\n    \\midrule\n"
        f"{body}\n    \\bottomrule\n  \\end{{tabular}}\n"
        # the note is a box the width of the table, not a full-width
        # paragraph: \centering would otherwise stretch it across the page
        f"  \\vspace{{2pt}}\n"
        f"  \\parbox{{0.8\\linewidth}}{{\\footnotesize {note}}}\n"
        "\\end{table}\n")


ARCH_NOTES = {
    "cond_dim": "condition-embedding width (FiLM conditioning input)",
    "dim": "trunk width",
    "depth": "number of FiLM residual blocks",
    "x1_dim": "coordinate-embedding width",
    "n_cond_features": "condition features; set per run from the active "
                       "condition axes",
    "with_flows": "SubTB state-flow head (credit-assignment arm only)",
    "d": "coordinates per point; taken from the world",
}
TRAIN_NOTES = {
    "loss": "trajectory balance (\\texttt{subtb}/\\texttt{exact\\_kl} in "
            "the named ablations)",
    "subtb_lambda": "geometric segment weight, SubTB arm",
    "steps": "gradient steps (overridden per battery)",
    "n_conds": "conditions per batch",
    "n_points": "points per condition",
    "uniform_mix": "share of uniform-over-$\\mathcal{X}$ points "
                   "(off-policy mixture)",
    "lr": "Adam learning rate, trunk",
    "lr_logz": "Adam learning rate, $\\log Z$ head",
    "eval_every": "evaluation interval (steps)",
    "heldout_radius": "$L_\\infty$ exclusion around held-out conditions",
    "cond_pool": "pre-priced condition pool ($\\texttt{None}$ = stream "
                 "fresh conditions per step)",
    "logit_floor": "training-only clamp on $\\beta_t \\log R$; evaluation "
                   "targets are never clamped",
    "seed": "training seed (swept)",
    "device": "compute device",
    "net": "architecture overrides (none used; see "
           "Table~\\ref{tab:arch})",
}
WORLD_NOTES = {
    "H": "grid resolution per coordinate", "d": "coordinates; $|X| = H^d$",
    "geometry": "score-field family",
    "sparsity": "satisfying-set sparsity exponent",
    "k_neutral": "$|S^0|$ (case A)", "k_guard": "guard states (guarded A)",
    "k_plus": "$|S^+|$ (cases B, C)", "k_minus": "$|S^-|$ (case B)",
    "k_named": "$|D|$ (case C)", "n_outer": "$|O|$ (case D)",
    "k_inner": "inner states per origin (case D)",
    "weight_alpha": "Dirichlet concentration for all nominal weights",
    "gamma": "trade-off on $\\Phi(S^-)$ (case B)",
    "floor": "floor level (\\texttt{None} = calibrated per world)",
    "floor_quantile": "quantile defining the calibrated floor",
    "floor_ref": "reference $(\\beta, \\rho)$ for floor calibration",
    "use_floor": "floor active (case B)",
    "c_named": "veto threshold $c_d$ (case C)",
    "beta_out": "reference outer tail level (case D)",
    "rho_out": "fixed outer KL radius (case D)",
    "profile_range": "reliability profile range (case D ablation)",
    "g_iid": "i.i.d.\\ score field instead of the smooth field",
    "hardness_margin": "gate: minimum TV(risk-on, risk-off)",
    "veto_frac_range": "gate: admissible veto share (case C)",
    "floor_frac_range": "gate: admissible floor-fail share (case B)",
    "delta_probe": "gate: veto-margin endpoint $\\delta_{\\max}$",
    "max_attempts": "gate: rejection-sampling cap",
}
RANGE_NOTES = {
    "beta_t": "inverse-temperature range (log-uniform)",
    "w_g": "auxiliary-objective weight range",
    "rho": "ambiguity radius range, per axis",
    "delta": "veto margin range (case C)",
    "n_outer": "per-origin $\\rho$ axes (case D)",
    "tie_beta": "share one $\\beta$ across both origins",
    "sigma": "score-robustness margin range (paper 2)",
    "use_sigma": "$\\sigma$ axis active",
    "use_rho_out": "conditioned outer radius active (case D)",
    "use_guard_delta": "guard margin axis active (guarded A)",
    "beta_tops": "per-axis upper ends (set by the shrink ablation)",
    "beta_bounds": "per-set lower bounds, filled from each world",
    "case": "case template",
}


def render_tex(per_family: dict, full: bool = False) -> str:
    arch = net_defaults()
    header = ("% Generated by scripts/make_hparam_tables.py — do not edit by "
              "hand.\n% Values are read from the epgfn dataclasses and the "
              "per-run args.json\n% files in results/; regenerate after any "
              "config change.\n")
    if not full:
        return header + "\n" + essential_table(arch)
    parts = [
        header,
        defaults_table(
            "Policy architecture (\\texttt{epgfn.policy.ConditionalPolicy}). "
            "No run overrides these.",
            "arch",
            [(k, v, ARCH_NOTES.get(k, "")) for k, v in arch.items()],
            "FiLM-conditioned residual trunk; $P_B = 1$, so only the "
            "forward policy and the $\\log Z$ head are learned."),
        defaults_table(
            "Optimization defaults (\\texttt{epgfn.train.TrainConfig}). "
            "Per-battery overrides in Table~\\ref{tab:fam}.",
            "train",
            [(k, v, TRAIN_NOTES.get(k, ""))
             for k, v in defaults_of(TrainConfig).items()],
            "Adam, two parameter groups. Every deviation from these "
            "values is recorded in the corresponding "
            "\\texttt{args.json}."),
        defaults_table(
            "World family and hardness gate "
            "(\\texttt{epgfn.worlds.WorldConfig}).",
            "world",
            [(k, v, WORLD_NOTES.get(k, ""))
             for k, v in defaults_of(WorldConfig).items()],
            # self-contained: no \\ref into the body, so the file can be
            # \\input anywhere without creating a dangling reference
            "The last five rows are the acceptance gate: worlds are "
            "rejection-sampled until every criterion holds, and the "
            "attempt count is reported per world."),
        defaults_table(
            "Condition ranges "
            "(\\texttt{epgfn.conditions.ConditionRanges}).",
            "ranges",
            [(k, v, RANGE_NOTES.get(k, ""))
             for k, v in defaults_of(ConditionRanges).items()],
            "Training conditions are sampled from these ranges outside "
            "the held-out exclusion balls; held-out conditions come from "
            "the declared grid."),
        # ball_table() is deliberately not emitted: the same grids are stated
        # as Table~\ref{tab:radii} beside the three ball geometries in the
        # duals appendix, which is where a reader needs them. Keep the
        # function as the check that the two agree.
        family_table(per_family),
        provenance_note(per_family),
    ]
    return "\n".join(parts)


def provenance_note(per_family: dict) -> str:
    """State where a table entry is a reconstruction rather than a record."""
    assumed = defaultdict(set)
    for fam, recs in per_family.items():
        for r in recs:
            for k in r["unrecorded_flags"]:
                if k not in NO_OP_DEFAULTS:
                    assumed[k].add(fam)
    if not assumed:
        return ""
    items = "; ".join(
        rf"\texttt{{{tex_escape(k)}}} in "
        + ", ".join(rf"\texttt{{{tex_escape(f)}}}" for f in sorted(fams))
        for k, fams in sorted(assumed.items()))
    touched = {f for fams in assumed.values() for f in fams}
    # only claim this when it is true of every affected family
    tail = (" Every affected family is a development battery that backs "
            "no reported result."
            if all("dev" in f or "pilot" in f for f in touched) else "")
    return (
        "\\paragraph{Provenance.}\n"
        "Each run directory records its command line in "
        "\\texttt{args.json}; the tables above join those records to the "
        "configuration defaults in \\texttt{epgfn}. Where a flag was "
        "added after a run, that run's entry falls back to the current "
        "default. For flags whose default is a documented identity "
        "(geometry, weight concentration, $\\sigma$, guard states, loss, "
        "$d$, gate cap) the fallback is exact. It is an assumption only "
        f"for: {items}.{tail}\n")


def ball_table() -> str:
    rows = "\n".join(
        rf"    {tex_escape(g)} & {fmt(BALL_RHO_GRIDS[g])} \\"
        for g in GEOMETRIES)
    return (
        "\\begin{table}[t]\n  \\centering\n  \\small\n"
        "  \\caption{Registered ambiguity-radius grids, per ball "
        "geometry. Radii are geometry-specific and are \\emph{not} "
        "comparable across balls (KL in nats, TV in mass, $\\chi^2$ in "
        "modified-$\\chi^2$ units).}\n  \\label{tab:balls}\n"
        "  \\begin{tabular}{ll}\n    \\toprule\n"
        "    Ball & $\\rho$ grid \\\\\n    \\midrule\n"
        f"{rows}\n    \\bottomrule\n  \\end{{tabular}}\n\\end{{table}}\n")


def family_table(per_family: dict) -> str:
    lines = []
    for fam in sorted(per_family):
        # launcher fragments inherit their parent's configuration; count
        # the parent runs so the column reads as "configurations launched"
        recs = [r for r in per_family[fam] if not r["fragment"]] \
            or per_family[fam]
        variants = defaultdict(int)
        for r in recs:
            variants[json.dumps(deltas(r), sort_keys=True)] += 1
        # the .py suffix is dead weight in a page-width table
        scripts = sorted({r["script"].removesuffix(".py") for r in recs})
        for i, (blob, n) in enumerate(sorted(variants.items(),
                                             key=lambda kv: -kv[1])):
            d = json.loads(blob)
            shown = ", ".join(rf"\texttt{{{tex_escape(k)}}}={fmt(v)}"
                              for k, v in sorted(d.items())) or "defaults"
            name = tex_escape(fam) if i == 0 else ""
            src = tex_escape(", ".join(scripts)) if i == 0 else ""
            lines.append(rf"    {name} & {src} & {n} & {shown} \\")
    body = "\n".join(lines)
    return (
        "\\begin{table}[htbp]\n  \\centering\n  \\footnotesize\n"
        "  \\caption{Per-battery configuration: every result family in "
        "\\texttt{results/}, the script that produced it, the number of "
        "run directories, and the settings that differ from the defaults "
        "in Tables~\\ref{tab:train}--\\ref{tab:ranges}. Anything not "
        "listed took its default value. Fan-out fragments inherit their "
        "parent run's configuration and are not counted separately; a "
        "family with more than one row was launched at more than one "
        "setting.}\n  \\label{tab:fam}\n"
        "  \\begin{tabular}{@{}lp{0.17\\linewidth}rp{0.5\\linewidth}@{}}\n"
        "    \\toprule\n"
        "    Family & Script & Runs & Overrides \\\\\n    \\midrule\n"
        f"{body}\n    \\bottomrule\n  \\end{{tabular}}\n\\end{{table}}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "report"))
    ap.add_argument("--full", action="store_true",
                    help="emit the four per-dataclass tables and the "
                         "per-battery table instead of the consolidated one")
    args = ap.parse_args()

    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    per_family, problems = collect(results)
    n_runs = sum(len(v) for v in per_family.values())

    payload = {
        "generated_from": str(results),
        "n_families": len(per_family),
        "n_runs": n_runs,
        "defaults": {
            "WorldConfig": defaults_of(WorldConfig),
            "TrainConfig": defaults_of(TrainConfig),
            "ConditionRanges": defaults_of(ConditionRanges),
            "ConditionalPolicy": net_defaults(),
            "BALL_RHO_GRIDS": {k: list(v)
                               for k, v in BALL_RHO_GRIDS.items()},
        },
        "families": {fam: [{**r, "deltas": deltas(r)} for r in recs]
                     for fam, recs in sorted(per_family.items())},
        "no_op_defaults": sorted(NO_OP_DEFAULTS),
        "problems": problems,
    }
    (out / "hparams.json").write_text(json.dumps(payload, indent=2,
                                                 default=str))
    (out / "hparams.tex").write_text(render_tex(per_family, args.full))

    print(f"{n_runs} run directories across {len(per_family)} families "
          f"-> {out/'hparams.json'}, {out/'hparams.tex'}")
    trained = sum(1 for recs in per_family.values() for r in recs
                  if r["train"] is not None)
    print(f"  trained runs: {trained}   analytic runs: {n_runs - trained}")
    exact, assumed = defaultdict(set), defaultdict(set)
    for fam, recs in per_family.items():
        for r in recs:
            for k in r["unrecorded_flags"]:
                (exact if k in NO_OP_DEFAULTS else assumed)[k].add(fam)
    if exact:
        print("  unrecorded but exact (default is a documented identity): "
              + ", ".join(sorted(exact)))
    if assumed:
        print("  ASSUMED — run predates the flag and its default is not an "
              "identity:")
        for k, fams in sorted(assumed.items()):
            print(f"    {k}: {', '.join(sorted(fams))}")
    if problems:
        print(f"\n{len(problems)} provenance problem(s):", file=sys.stderr)
        for p in problems[:40]:
            print(f"  {p}", file=sys.stderr)
        if len(problems) > 40:
            print(f"  ... {len(problems) - 40} more (see hparams.json)",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
