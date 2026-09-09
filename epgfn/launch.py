"""Shared fan-out/merge harness for the launch_*.py parallel launchers:
subprocess pool at a worker budget, then fragment-CSV merge. Every
unit is fully determined by its own CLI arguments, so grouping units
into a pool can't change any number (no RNG state crosses units).
"""

from __future__ import annotations

import csv
import glob
import time

from .runio import Progress


def fan_out(units: list, launch, jobs: int, label: str, out,
           fmt=str) -> None:
    """Run one subprocess per unit at a `jobs` concurrency cap.

    Args:
        units: hashable unit keys, one subprocess launched per unit.
        launch: launch(unit) -> subprocess.Popen.
        jobs: max concurrent subprocesses.
        label: runio.Progress label.
        out: run dir, referenced in the failure message.
        fmt: unit -> str, for the launched/done/FAILED log lines.

    Raises:
        SystemExit: naming every failed unit, if any subprocess exited
            non-zero. Callers should not merge fragment output past
            that point.
    """
    pending = list(units)
    running: dict = {}
    failed = []
    prog = Progress(len(units), label)
    while pending or running:
        while pending and len(running) < jobs:
            unit = pending.pop(0)
            running[unit] = launch(unit)
            print(f"launched {fmt(unit)} "
                  f"({len(running)} running, {len(pending)} queued)",
                  flush=True)
        time.sleep(5)
        for unit, proc in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[unit]
            if rc != 0:
                failed.append(unit)
                prog.step(f"FAILED {fmt(unit)} (rc={rc}): see frag log")
            else:
                prog.step(f"done {fmt(unit)}")
    if failed:  # no silent caps: a partial merge must be explicit
        raise SystemExit(f"{len(failed)} unit(s) failed: {failed}; "
                         f"not merging. Logs under {out}")


def one_fragment(frag_glob: str, label: str) -> str:
    """Resolve exactly one fragment CSV matching `frag_glob`
    (recursive glob); a launcher fragment always writes exactly one."""
    hits = sorted(glob.glob(frag_glob, recursive=True))
    if len(hits) != 1:
        raise SystemExit(f"unit {label}: expected one fragment CSV, "
                         f"found {len(hits)}")
    return hits[0]


def merge_fragments(paths: list, expect: int, context: str = "") -> list[dict]:
    """Read fragment CSVs into one row list, in `paths` order;
    refuses (SystemExit) to continue if the total isn't exactly
    `expect` rows (no silent partial merge). `context` prefixes the
    failure message (e.g. "case A: ")."""
    rows = []
    for path in paths:
        with open(path) as fh:
            rows.extend(csv.DictReader(fh))
    if len(rows) != expect:
        raise SystemExit(f"{context}merged {len(rows)} rows, expected "
                         f"{expect}; refusing to summarize")
    return rows


def write_merged_csv(out_path, rows: list[dict]) -> None:
    """Write a merged fragment row list to `out_path`, columns taken
    from the first row (every fragment shares the source script's
    field order)."""
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
