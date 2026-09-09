"""Profile-scale vs misspec vs one-hot rows, exact."""

import subprocess
import sys
import csv
import json
import pathlib


def test_run_dtying_smoke(tmp_path):
    """run_dtying.py's CLI produces profile/misspec/one_hot tying rows with
    valid TV values and a summary containing median_of_max for each tying."""
    root = pathlib.Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_dtying.py"),
         "--worlds", "1", "--world-seed0", "0", "--H", "8",
         "--out", str(tmp_path)],
        capture_output=True, text=True, check=True)
    # unique_run_dir nests output two levels deep (e.g. a config-hash dir
    # containing a timestamped run dir)
    run_dir = next(tmp_path.glob("*/*"))
    rows = list(csv.DictReader(open(run_dir / "dtying_caseD.csv")))
    tyings = {row["tying"] for row in rows}
    assert tyings == {"profile", "misspec", "one_hot"}
    # 6 s-values per scale arm; one-hot rows = O * 6 levels
    assert sum(r_["tying"] == "profile" for r_ in rows) == 6
    assert sum(r_["tying"] == "misspec" for r_ in rows) == 6
    for row in rows:
        assert 0.0 <= float(row["tv_to_shared"]) <= 1.0
    s = json.load(open(run_dir / "dtying_caseD_summary.json"))
    for k in ("profile", "misspec", "one_hot"):
        assert "median_of_max" in s[k]
