"""Run-directory isolation: concurrent invocations must never share an
output directory."""

import json

from epgfn.runio import unique_run_dir


def test_unique_run_dirs_never_collide(tmp_path):
    """Concurrent calls under the same base never share a run directory."""
    dirs = [unique_run_dir(tmp_path) for _ in range(5)]
    assert len({d.name for d in dirs}) == 5
    for d in dirs:
        assert d.is_dir()
        # runs group by day: <base>/<YYYY-MM-DD>/<HHMMSS>-<pid>[-n]
        assert len(d.parent.name.split("-")) == 3


def test_args_manifest_written(tmp_path):
    """Args passed to unique_run_dir are persisted verbatim as args.json."""
    d = unique_run_dir(tmp_path, {"case": "A", "steps": 42})
    manifest = json.loads((d / "args.json").read_text())
    assert manifest == {"case": "A", "steps": 42}
