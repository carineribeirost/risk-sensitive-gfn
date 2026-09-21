#!/usr/bin/env bash
# Contrastive fill orchestrator — the ONLY remaining s=4.0 cells.
# All gate decisions are settled (D bases 1-2 structural, w59 the
# recorded exception), so this does zero gate work: every world
# is pinned from a same-sparsity record and launched with
# --trust-recorded-gates straight into training.
#
# Idempotent: counts existing contrastive rows per case first and
# launches only what is missing (safe to rerun after a partial run —
# --resume copies finished cells). Maximum concurrency: every
# (case, world) unit is its own fragment, all on cuda simultaneously
# (A 4 + B 4 + C 4 + D 1 = 13 units x 2 threads on a 32-thread box;
# tiny nets share the GPU comfortably). Wall-clock floor = the 3 seeds
# trained sequentially inside each unit.
#
# Usage (run box, repo root):
#   nohup bash scripts/run_s4_fill.sh > s4fill.log 2>&1 &
#   tail -f s4fill.log
set -uo pipefail
cd "$(dirname "$0")/.."

# the box has no bare `python`; everything runs through uv. Override
# with PY="python3" (or a venv python) on hosts without uv.
PY=${PY:-uv run python}

RESUME_AB=${RESUME_AB:-results/w33/2026-07-12/090236-16060}
RESUME_C=${RESUME_C:-results/w33/2026-07-21/054830-46161}
RESUME_D=${RESUME_D:-results/w33/2026-07-18/185114-5541}

LOG=results/s4fill-$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOG"
note() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/decisions.log"; }
note "log dir: $LOG"

if pgrep -f "scripts/(run|launch)_w33.py" >/dev/null 2>&1; then
    note "WARNING: a w33 process is already running — its unfinished cells are invisible to the row count below"
fi

# how many contrastive s=4.0 rows already exist per case
counts=$(python3 - <<'EOF'
import csv, glob
seen = {}
for p in glob.glob("results/w33/**/w33_runs.csv", recursive=True):
    try:
        for r in csv.DictReader(open(p)):
            if r.get("sparsity") == "4.0" and r.get("arm") == "contrastive" \
               and r.get("frac_modes") not in (None, ""):
                seen.setdefault(r["case"], set()).add((r["world"], r["seed"]))
    except OSError:
        pass
print(" ".join(f"{c}={len(seen.get(c, set()))}" for c in "ABCD"))
EOF
)
note "existing contrastive s=4.0 cells: $counts"
get() { echo "$counts" | tr ' ' '\n' | sed -n "s/^$1=//p"; }

PIDS=() NAMES=()
launch() {  # name, needed, have, launcher args...
    local name=$1 needed=$2 have=$3; shift 3
    if [ "$have" -ge "$needed" ]; then
        note "$name: already complete ($have/$needed cells) — skipped"
        return
    fi
    note "$name: $have/$needed cells — launching"
    $PY scripts/launch_w33.py "$@" \
        --arms contrastive --sparsity 4.0 \
        --trust-recorded-gates --threads-per-job 2 --device cuda \
        > "$LOG/launch_$name.log" 2>&1 &
    PIDS+=($!); NAMES+=("$name")
}

launch AB 24 $(( $(get A) + $(get B) )) \
    --cases A B --resume "$RESUME_AB" --jobs 8
launch C 12 "$(get C)" \
    --cases C --resume "$RESUME_C" --jobs 4
launch D_w59 3 "$(get D)" \
    --cases D --worlds 1 --world-seed0 0 --world-seeds 59 \
    --resume "$RESUME_D" --jobs 1

rc=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        note "${NAMES[$i]}: OK"
    else
        note "${NAMES[$i]}: FAILED — see $LOG/launch_${NAMES[$i]}.log"
        rc=1
    fi
done

note "---- summaries ----"
for l in "$LOG"/launch_*.log; do
    [ -e "$l" ] || continue
    dir=$(grep -m1 "^run dir: " "$l" | cut -d' ' -f3-)
    note "$(basename "$l" .log): run dir $dir"
    [ -n "$dir" ] && [ -f "$dir/w33_summary.json" ] \
        && cat "$dir/w33_summary.json" | tee -a "$LOG/decisions.log" \
        || note "  (no merged summary — see fragment CSVs under $dir)"
done
note "done (rc=$rc)."
exit "$rc"
