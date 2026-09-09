#!/usr/bin/env bash
# One-shot orchestrator for the unified-paper battery (2026-07-20).
# Covers EVERYTHING outstanding: the five-arm w33 tables, the O3
# fine grid + registered probe cell, refcheck, C s=4.0 fill, D-gate
# probes, figure-cell densities, AND the oracle track (seq-D finish
# + grid A/D for Table 1's oracle column). Kills any leftover
# sequential run_oracle_gap first (superseded; user-approved).
#
# Dependency-aware scheduling on a single box: wave 1 (steps 1-4,
# plus the D figure-cell unit — its ~6 h single-core serial gate IS
# the battery's critical path, so it starts at t=0) launches
# immediately; steps 5+6 start the moment step 1 finishes (its 12
# job slots free up); the A/B/C figure cells start when step 2
# finishes. Correctness never depends on ordering — every unit is
# seed-determined and writes its own run dir — this script only
# manages box load; total wall time ~= the D gate chain itself.
#
# Launch once, watch anywhere:
#   nohup bash scripts/run_paper_battery.sh > run_all.log 2>&1 &
#   tail -f run_all.log            # wave transitions + final status
#   tail -f w33_abc.log            # per-step [i/N elapsed<ETA] detail
# Dry-run (print the schedule without executing anything):
#   DRY=1 bash scripts/run_paper_battery.sh

set -u
cd "$(dirname "$0")/.."

# ---- battery constants (edit here if run dirs move) ----------------
RESUME_ABC=results/w33/2026-07-12/090236-16060
RESUME_D=results/w33/2026-07-18/185114-5541
RESUME_SEQD=results/seq-oracle/2026-07-18/185021-5274
W33="--worlds 4 --seeds 3 --H 4 --d 8 --steps 12000"
ARMS5="onpolicy mix replay teacher contrastive"
DRY="${DRY:-0}"

say() { echo "[$(date +'%H:%M:%S')] $*"; }

run_bg() {  # run_bg <logfile> <cmd...> -> pid in $PID
    local log=$1; shift
    if [ "$DRY" = "1" ]; then
        say "DRY: $* > $log"; PID=""
        return
    fi
    "$@" > "$log" 2>&1 &
    PID=$!
    say "launched (pid $PID, log $log): $*"
}

finish() {  # finish <name> <pid> -> records failures
    [ -z "$2" ] && return
    if wait "$2"; then
        say "$1: DONE"
    else
        say "$1: FAILED (rc=$?) — check its log"
        FAILED="$FAILED $1"
    fi
}

FAILED=""

# the old sequential seq-oracle D run is superseded by the resumed
# parallel chain below (user-approved kill 2026-07-20); its CSV rows
# are flushed per training, so at most the in-flight unit is lost
if [ "$DRY" != "1" ] && pgrep -f run_oracle_gap > /dev/null; then
    say "killing the old sequential run_oracle_gap (superseded; "\
"completed rows are safe and will be resumed)"
    pkill -f run_oracle_gap
    sleep 2
fi

say "WAVE 1: steps 1-4"

run_bg w33_abc.log uv run python scripts/launch_w33.py \
    --cases A B C --sparsity 1.0 --arms $ARMS5 $W33 \
    --resume "$RESUME_ABC" --jobs 12 --out results/w33
P1=$PID

run_bg w33_d.log uv run python scripts/launch_w33.py \
    --cases D --sparsity 1.0 --arms $ARMS5 $W33 \
    --resume "$RESUME_D" --jobs 4 --out results/w33
P2=$PID

run_bg o3_pareto.log uv run python scripts/run_o3.py \
    --cases A B C D --worlds 8 --world-seed0 0 --H 32 \
    --beta-grid 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --rho-grid 0.0 0.05 0.1 0.15 0.2 0.3 0.5 0.8 1.2 \
    --kappa-range 8 60 --n-stress 500 --contam-eps 0.2 \
    --jobs 16 --out results/o3-pareto
P3=$PID

run_bg refcheck.log uv run python scripts/make_fig_refcheck.py \
    --which data
P4=$PID

# step 7a at t=0: the D figure-cell unit — ~6 h SERIAL gate on one
# core (the battery's critical path), then ~20 min of training
run_bg w33_figcells_D.log uv run python scripts/launch_w33.py \
    --cases D --sparsity 1.0 --arms $ARMS5 \
    --worlds 1 --seeds 1 --H 4 --d 8 --steps 12000 \
    --jobs 1 --out results/w33-figcells
P7D=$PID

# step 8 at t=0, its own dependency track: finish seq-oracle D
# (resume + recorded world seeds, ~30-40 min), THEN the two grid
# oracles that complete Table 1's oracle column (~15-20 min each)
if [ "$DRY" = "1" ]; then
    say "DRY: oracle chain: seq-D finish -> grid A + grid D" \
        "(seq_oracle_d.log, oracle_a.log, oracle_d.log)"
    P8=""
else
    (
        set -e
        uv run python scripts/launch_oracle.py --case D \
            --geometry sequence --H 4 --d 8 --worlds 4 --seeds 3 \
            --world-seeds 59 100029 200013 300040 \
            --resume "$RESUME_SEQD" --jobs 6 \
            --out results/seq-oracle > seq_oracle_d.log 2>&1
        uv run python scripts/launch_oracle.py --case A --jobs 6 \
            --out results/oracle-ad > oracle_a.log 2>&1 &
        GA=$!
        uv run python scripts/launch_oracle.py --case D --jobs 6 \
            --out results/oracle-ad > oracle_d.log 2>&1
        wait "$GA"
    ) &
    P8=$!
    say "launched (pid $P8): oracle chain -> seq_oracle_d.log," \
        "then oracle_a.log + oracle_d.log"
fi

finish "step1-w33-ABC" "$P1"
say "WAVE 2: steps 5+6 take step 1's freed slots"

run_bg w33_c_s4.log uv run python scripts/launch_w33.py \
    --cases C --sparsity 1.0 4.0 \
    --arms onpolicy mix replay teacher $W33 \
    --resume "$RESUME_ABC" --jobs 8 --out results/w33
P5=$PID

if [ "$DRY" = "1" ]; then
    say "DRY: d-gate probes seeds 1-3 (sequential, jobs 16)"
    P6=""
else
    (   # D s=4.0 gate diagnostics, seeds sequential to bound load
        for s in 1 2 3; do
            uv run python scripts/probe_gate.py --case D \
                --sparsity 4.0 --seed "$s" --H 4 --d 8 \
                --geometry sequence --attempts 200 --jobs 16 \
                > "d_probe_s$s.log" 2>&1
        done
    ) &
    P6=$!
    say "launched (pid $P6, logs d_probe_s{1,2,3}.log): gate probes"
fi

finish "step2-w33-D" "$P2"
say "WAVE 3: A/B/C figure cells take step 2's freed slots"

run_bg w33_figcells.log uv run python scripts/launch_w33.py \
    --cases A B C --sparsity 1.0 --arms $ARMS5 \
    --worlds 1 --seeds 1 --H 4 --d 8 --steps 12000 \
    --jobs 3 --out results/w33-figcells
P7=$PID

finish "step3-o3-pareto"    "$P3"
finish "step4-refcheck"     "$P4"
finish "step5-w33-C-s4"     "$P5"
finish "step6-d-probes"     "$P6"
finish "step7-figcells-ABC" "$P7"
finish "step7a-figcells-D"  "$P7D"
finish "step8-oracles"      "$P8"

if [ -n "$FAILED" ]; then
    say "BATTERY FINISHED WITH FAILURES:$FAILED"
    exit 1
fi
say "BATTERY COMPLETE — all eight steps done."
say "collect: results/o3-pareto/* results/w33/* results/w33-figcells/*"
say "         results/seq-oracle/* results/oracle-ad/*"
say "         drafts/figures/fig_refcheck_data.npz d_probe_s*.log"
