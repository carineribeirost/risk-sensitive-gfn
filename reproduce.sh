#!/usr/bin/env bash
# The cheap tiers, in order, with what each one costs.
#
# Every stage here is CPU-only except the last, and the whole script is under
# half an hour on a laptop. The training batteries are separate: see
# scripts/run_rebuttal_queue.sh and scripts/run_paper_battery.sh.
#
# Usage:  ./reproduce.sh            all stages
#         ./reproduce.sh 3          one stage
#         PY=.venv/bin/python ./reproduce.sh

set -u
cd "$(dirname "$0")"

PY="${PY:-python3}"
export PYTHONPATH="${PYTHONPATH:-}:$PWD"
ONLY="${1:-all}"

stage() {          # stage <n> <cost> <description>
    [ "$ONLY" = "all" ] || [ "$ONLY" = "$1" ] && {
        printf '\n=== stage %s (%s) %s\n' "$1" "$2" "$3"
        return 0
    }
    return 1
}

stage 1 "minutes, CPU" "the test suite" && {
    "$PY" -m pytest -q
}

stage 2 "seconds, CPU" "the DRO-CVaR duals against brute force" && {
    "$PY" scripts/verify_duals.py
}

stage 3 "about a minute, CPU" \
      "our composition against the geometric mean, hard minimum, desirability" && {
    "$PY" scripts/run_o3.py --cases A B C D --worlds 8 --world-seed0 0 \
        --H 32 --d 2 --geometry grid --skip-grid \
        --baselines geometric desirability hard_min \
        --out results/o3-baselines
}

stage 4 "about 20 minutes, CPU" \
      "sensitivity of the finite-sample reference to its sample count" && {
    # Its own control: at the declared n it must reproduce 0.2227, 0.1470,
    # 0.1981 and 0.2210, and so the published ratios 0.37, 2.71, 1.71, 0.60.
    "$PY" scripts/ref_sweep.py
}

stage 5 "tens of minutes, GPU preferred" "one short conditional-policy run" && {
    "$PY" scripts/run_o2.py --case A --worlds 4 --seeds 3 \
        --H 32 --steps 4000 --out results/o2
}

printf '\ndone. outputs are under results/, each in its own timestamped\n'
printf 'directory with an args.json manifest.\n'
