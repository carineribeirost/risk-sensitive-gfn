"""Run-output isolation: every script invocation writes into its own
unique subdirectory of the requested --out base, so concurrent launches
can never silently overwrite each other's results.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import time


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    """Write `rows` to a CSV at `path`, columns = sorted union of all row keys.

    Args:
        path: output CSV path.
        rows: dicts to write; not all rows need share every key.
    """
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


class Progress:
    """Log-friendly progress: one `[i/N elapsed<ETA]` line per
    completed unit (no carriage returns: reads cleanly in tmux AND
    in `> file.log` redirections, unlike an animated bar). Purely
    stdout; never touches result files, so outputs stay byte-
    identical. ETA is the running mean of completed units."""

    def __init__(self, total: int, label: str = ""):
        """Args:
            total: Total number of units this progress tracker covers.
            label: Optional prefix printed before each progress line.
        """
        self.total = int(total)
        self.done = 0
        self.label = f"{label} " if label else ""
        self.t0 = time.monotonic()

    @staticmethod
    def _fmt(seconds: float) -> str:
        s = int(seconds)
        if s < 3600:
            return f"{s // 60}m{s % 60:02d}s"
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"

    def step(self, msg: str = "") -> None:
        """Record one completed unit and print its progress line.

        Args:
            msg: Optional message appended to the printed line.
        """
        self.done += 1
        el = time.monotonic() - self.t0
        eta = el / self.done * (self.total - self.done)
        print(f"[{self.label}{self.done}/{self.total} "
              f"{self._fmt(el)}<{self._fmt(eta)}] {msg}", flush=True)


def unique_run_dir(base, args=None) -> pathlib.Path:
    """Create and return <base>/<YYYY-MM-DD>/<HHMMSS>-<pid>[-n>,
    guaranteed fresh via exclusive mkdir; runs group by day. If `args`
    (an argparse Namespace or dict) is given, its values are recorded
    in args.json inside the run dir."""
    base = pathlib.Path(base) / time.strftime("%Y-%m-%d")
    stamp = time.strftime("%H%M%S")
    for n in range(1000):
        name = f"{stamp}-{os.getpid()}" + (f"-{n}" if n else "")
        run_dir = base / name
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        if args is not None:
            payload = args if isinstance(args, dict) else vars(args)
            with open(run_dir / "args.json", "w") as fh:
                json.dump(payload, fh, indent=2, default=str)
        return run_dir
    raise RuntimeError(f"could not create a unique run dir under {base}")
