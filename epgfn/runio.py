"""Run-output isolation: every script invocation writes into its own
unique subdirectory of the requested --out base, so concurrent launches
can never silently overwrite each other's results.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import sys
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


class _NewlineWriter:
    """Turn tqdm's carriage returns into newlines.

    tqdm redraws by writing "\r<bar>", which is right on a terminal and
    wrong in a file: the whole run collapses onto one line, `tail -f`
    shows a smear and `grep` finds nothing. Wrapping the stream gives
    one line per update in a log while leaving the terminal path
    untouched.
    """

    def __init__(self, stream):
        self._s = stream

    def write(self, text):
        self._s.write(text.replace("\r", "\n"))

    def flush(self):
        self._s.flush()

    def isatty(self):
        return False


class Progress:
    """tqdm progress bar over units of work.

    Falls back to a plain one-line-per-unit print if tqdm is missing,
    so the package still runs in an environment that only has the core
    dependencies. Purely stdout; never touches result files, so
    outputs stay byte-identical either way.

    When stdout is redirected to a log, tqdm is put in
    one-line-per-update mode: an animated bar written to a file fills
    it with carriage returns and makes `tail -f` unreadable. Set
    EPGFN_BAR=0 to force the plain form, EPGFN_BAR=1 to force the
    animated one.
    """

    def __init__(self, total: int, label: str = ""):
        """Args:
            total: Total number of units this progress tracker covers.
            label: Optional prefix printed before each progress line.
        """
        self.total = max(int(total), 1)
        self.done = 0
        self.label = label or "run"
        self.t0 = time.monotonic()
        env = os.environ.get("EPGFN_BAR")
        live = sys.stdout.isatty() if env not in ("0", "1") else env == "1"
        self._bar = None
        try:
            from tqdm.auto import tqdm
        except ImportError:
            self._live = live
            return
        self._live = live
        self._bar = tqdm(
            total=self.total, desc=self.label, unit="unit",
            dynamic_ncols=True, leave=True, mininterval=0.0,
            # in a log, emit one line per update instead of redrawing
            file=(sys.stdout if live else _NewlineWriter(sys.stdout)),
            ascii=not live,
            bar_format=("{l_bar}{bar}| {n_fmt}/{total_fmt} "
                        "[{elapsed}<{remaining}, {rate_fmt}]{postfix}"))
        if not live:
            self._bar.miniters = 1

    @staticmethod
    def _fmt(seconds: float) -> str:
        s = max(int(seconds), 0)
        if s < 3600:
            return f"{s // 60}m{s % 60:02d}s"
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"

    def step(self, msg: str = "") -> None:
        """Record one completed unit and print its progress line.

        Args:
            msg: Optional message appended to the printed line.
        """
        self.done += 1
        if self._bar is not None:
            if msg:
                self._bar.set_postfix_str(msg, refresh=False)
            self._bar.update(1)
            if self.done >= self.total:
                self._bar.close()
            return
        el = time.monotonic() - self.t0
        eta = el / self.done * (self.total - self.done)
        print(f"[{self.label} {self.done}/{self.total} "
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
