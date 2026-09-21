#!/usr/bin/env bash
#
# One-shot resumable runner for the whole rebuttal battery.
#
#   bash scripts/run_rebuttal_queue.sh
#
# Detached under tmux (preferred):
#   tmux new-session -d -s queue \
#     "PY=.venv/bin/python bash scripts/run_rebuttal_queue.sh 2>&1 | tee queue.log"
#
# ---------------------------------------------------------------- tuning
#
# The w33 launcher fans out over WORLDS, so a single fragment never uses
# more than 4 workers. On a big box the win comes from running several
# FRAGMENTS at once, which FRAG_PAR controls.
#
#   PY           python to use                       (default: python)
#   FRAG_PAR     coverage fragments run concurrently (default: 1)
#   JOBS_SEQ     workers per sequence fragment       (default: 4)
#   JOBS_GRID    workers per grid-family run         (default: 8)
#   THREADS      --threads-per-job                   (default: 2)
#
# Presets:
#   20 cores / 10 GB VRAM
#     FRAG_PAR=1 JOBS_SEQ=4  JOBS_GRID=8  THREADS=2
#   i9-13900K, 32 cores / 48 GB VRAM
#     FRAG_PAR=4 JOBS_SEQ=4  JOBS_GRID=16 THREADS=2
#
# Other switches: ONLY="R5 R1" runs named steps; DRYRUN=1 prints only;
# AUTOCOMMIT=0 skips committing; AUTOPUSH=1 pushes each commit.
#
# RESUMABLE. Every success writes a marker under .queue-state/; rerunning
# skips it. Delete a marker to force that step to run again.
#
# NOT `set -e`: a failed step logs its tail and the queue continues.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-python}"

# The project has no [build-system], so `uv sync` installs dependencies
# without installing `epgfn` itself, and every driver imports it by name.
# Putting the repo root on the path makes the queue work whether or not
# the package was installed.
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$REPO"
FRAG_PAR="${FRAG_PAR:-1}"
JOBS_SEQ="${JOBS_SEQ:-4}"
JOBS_GRID="${JOBS_GRID:-8}"
THREADS="${THREADS:-2}"
ONLY="${ONLY:-}"
DRYRUN="${DRYRUN:-0}"
AUTOCOMMIT="${AUTOCOMMIT:-1}"
AUTOPUSH="${AUTOPUSH:-0}"

STATE="$REPO/.queue-state"
LOGS="$REPO/queue-logs"
mkdir -p "$STATE" "$LOGS"

# Queue-level progress. run_one may execute in a subshell when fragments
# run in parallel, so the counter lives in a file rather than a variable.
QT0=$(date +%s)
COUNTER="$STATE/.counter"
echo 0 > "$COUNTER"
# Counting source lines undercounts: the loops expand R6 into eight
# steps, R1 into four and R9 into two. With ONLY the user names the steps
# explicitly, so the word count is exact.
if [ -n "$ONLY" ]; then
  QUEUE_TOTAL="$(printf '%s\n' $ONLY | wc -l)"
else
  #              R5 R6 R2 R1 R3 R4 R7 R8 R9
  QUEUE_TOTAL=$(( 1 + 8 + 4 + 4 + 1 + 1 + 1 + 1 + 2 ))
fi

qbar() {   # qbar <done> <total>
  local d="$1" t="$2" w=24 f
  f=$(( w * d / (t > 0 ? t : 1) ))
  printf '%*s' "$f" '' | tr ' ' '#'
  printf '%*s' "$(( w - f ))" '' | tr ' ' '.'
}

qfmt() { local s="$1"; if [ "$s" -lt 3600 ]; then printf '%dm%02ds' $((s/60)) $((s%60));
         else printf '%dh%02dm' $((s/3600)) $(((s%3600)/60)); fi; }

qprogress() {  # called after each step completes
  # No estimate of time remaining: steps range from two minutes to
  # hours, so a mean over completed steps would predict nonsense.
  # Position and elapsed are facts; the rest would be invention.
  local d el
  ( flock 8; d=$(( $(cat "$COUNTER") + 1 )); echo "$d" > "$COUNTER" ) 8>"$STATE/.clock"
  d=$(cat "$COUNTER")
  el=$(( $(date +%s) - QT0 ))
  printf '\033[1m[queue %d/%d] %s %d%%  %s elapsed\033[0m\n' \
    "$d" "$QUEUE_TOTAL" "$(qbar "$d" "$QUEUE_TOTAL")" \
    "$(( 100 * d / QUEUE_TOTAL ))" "$(qfmt "$el")"
}

ALL_STEPS=()

say()  { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\n\033[33m[%s] WARNING: %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }

wanted() {
  [ -z "$ONLY" ] && return 0
  [[ " $ONLY " == *" $1 "* ]]
}

# ---------------------------------------------------------------- preflight

preflight() {
  say "preflight"
  local bad=0
  "$PY" -c "import torch" 2>/dev/null || {
    warn "'$PY' cannot import torch. Try PY=.venv/bin/python"; bad=1; }
  # Without these the battery reproduces exactly the number the reviewer
  # objected to, at a cost of tens of GPU-hours.
  grep -q policy_coverage epgfn/w33.py || {
    warn "epgfn/w33.py has no policy_coverage; the tree is behind. git pull."; bad=1; }
  grep -q target_coverage scripts/run_w33.py || {
    warn "scripts/run_w33.py does not write target_coverage. git pull."; bad=1; }
  [ -f epgfn/baselines.py ] || { warn "epgfn/baselines.py missing. git pull."; bad=1; }

  "$PY" - <<'EOF' 2>/dev/null || warn "GPU query failed; continuing"
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU: {p.name}, {p.total_memory/2**30:.1f} GiB")
else:
    print("  no CUDA device visible; everything falls back to CPU and will crawl")
EOF
  printf '  cores: %s   commit: %s\n' "$(nproc 2>/dev/null || echo '?')" \
         "$(git log --oneline -1)"
  printf '  free disk: %s\n' "$(df -h . | awk 'NR==2{print $4}')"
  printf '  tuning: FRAG_PAR=%s JOBS_SEQ=%s JOBS_GRID=%s THREADS=%s\n' \
         "$FRAG_PAR" "$JOBS_SEQ" "$JOBS_GRID" "$THREADS"
  [ "$bad" -eq 0 ] || { warn "preflight failed"; exit 1; }
  printf '  queue: %s steps\n' "$QUEUE_TOTAL"
  echo "  preflight ok"
}

# ------------------------------------------------------------------ runner

# run_one <name> <outdir|-> <cmd...>   (safe to call in a subshell)
run_one() {
  local name="$1"; shift
  local outdir="$1"; shift
  local t0 rc log
  t0=$(date +%s); log="$LOGS/$name.log"
  printf '[%s] %s starting\n' "$(date +%H:%M:%S)" "$name"
  "$@" > "$log" 2>&1
  rc=$?
  local mins=$(( ($(date +%s) - t0) / 60 ))
  if [ $rc -ne 0 ]; then
    printf '\033[33m[%s] %s FAILED (exit %s) after %s min\033[0m\n' \
           "$(date +%H:%M:%S)" "$name" "$rc" "$mins"
    tail -12 "$log" | sed 's/^/      /'
    echo "$rc" > "$STATE/$name.failed"
    qprogress
    return 0
  fi
  printf '[%s] %s done in %s min\n' "$(date +%H:%M:%S)" "$name" "$mins"
  echo "$mins" > "$STATE/$name.done"
  if [ "$AUTOCOMMIT" = "1" ] && [ "$outdir" != "-" ] && [ -d "$outdir" ]; then
    # serialise index access; parallel fragments would race otherwise
    (
      flock 9
      git add -f "$outdir" 2>/dev/null
      git diff --cached --quiet || {
        git commit -q -m "rebuttal runs: $name"
        [ "$AUTOPUSH" = "1" ] && git push -q origin main 2>/dev/null
        echo "      committed $outdir"
      }
    ) 9>"$STATE/.gitlock"
  fi
  qprogress
}

# step <name> <outdir|-> <cmd...>  — sequential
step() {
  local name="$1"
  ALL_STEPS+=("$name")
  wanted "$name" || return 0
  [ -f "$STATE/$name.done" ] && return 0
  if [ "$DRYRUN" = "1" ]; then
    printf '  would run %-16s %s\n' "$name" "${*:3}"; return 0
  fi
  rm -f "$STATE/$name.failed"
  run_one "$@"
}

# slot <name> <outdir|-> <cmd...>  — background, respecting FRAG_PAR
slot() {
  local name="$1"
  ALL_STEPS+=("$name")
  wanted "$name" || return 0
  [ -f "$STATE/$name.done" ] && return 0
  if [ "$DRYRUN" = "1" ]; then
    printf '  would run %-16s (parallel) %s\n' "$name" "${*:3}"; return 0
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$FRAG_PAR" ]; do sleep 15; done
  rm -f "$STATE/$name.failed"
  run_one "$@" &
}

# -------------------------------------------------------------------- queue

preflight

# R5: CPU only, minutes. Answers Reviewer 2k4Z's one demanded experiment.
# Starts immediately so it never queues behind the GPU work.
R5_PID=""
if wanted R5 && [ ! -f "$STATE/R5.done" ] && [ "$DRYRUN" != "1" ]; then
  say "R5 (baselines, CPU) in background"
  ( run_one R5 results/o3-baselines \
      "$PY" scripts/run_o3.py --cases A B C D --worlds 8 --world-seed0 0 \
        --H 32 --d 2 --geometry grid --ball kl \
        --n-stress 500 --kappa 25.0 --kappa-range 8.0 60.0 --contam-eps 0.2 \
        --baselines geometric desirability hard_min --skip-grid \
        --jobs 6 --out results/o3-baselines ) > "$LOGS/R5.outer" 2>&1 &
  R5_PID=$!
fi
wanted R5 && ALL_STEPS+=("R5")

# R6: the corrected-coverage relaunch. Longest and most important.
# One fragment per (case, sparsity); FRAG_PAR of them at a time.
say "R6 coverage battery (FRAG_PAR=$FRAG_PAR, JOBS_SEQ=$JOBS_SEQ)"
for S in 4.0 1.0; do
  for C in A B C D; do
    slot "R6_${C}_s${S}" results/w33-cov \
      "$PY" scripts/launch_w33.py --cases "$C" --sparsity "$S" \
        --arms onpolicy mix replay teacher contrastive \
        --worlds 4 --seeds 3 --world-seeds 0 100003 200006 300009 \
        --trust-recorded-gates --H 4 --d 8 --geometry sequence \
        --steps 12000 --n-points 64 --eval-every 200 \
        --save-grids --jobs "$JOBS_SEQ" --threads-per-job "$THREADS" \
        --out results/w33-cov
  done
done
[ "$DRYRUN" = "1" ] || wait

# R2: unclamped TB control. Matched worlds, so the recorded seeds; a floor
# of -1e9 never binds. Answers the objective-mismatch objection.
say "R2 unclamped control"
step R2_A results/oracle-gap-noclamp \
  "$PY" scripts/launch_oracle.py --case A --H 32 --d 2 --geometry grid \
    --worlds 4 --seeds 3 --world-seeds 0 100003 200006 300009 \
    --steps 12000 --logit-floor -1e9 --jobs "$JOBS_GRID" \
    --threads-per-job "$THREADS" --out results/oracle-gap-noclamp
step R2_B results/oracle-gap-noclamp \
  "$PY" scripts/launch_oracle.py --case B --H 32 --d 2 --geometry grid \
    --worlds 4 --seeds 3 --world-seeds 0 100003 200006 300009 \
    --steps 12000 --logit-floor -1e9 --jobs "$JOBS_GRID" \
    --threads-per-job "$THREADS" --out results/oracle-gap-noclamp
step R2_C results/oracle-gap-noclamp \
  "$PY" scripts/launch_oracle.py --case C --H 32 --d 2 --geometry grid \
    --worlds 4 --seeds 3 --world-seeds 0 100004 200006 300009 \
    --steps 12000 --logit-floor -1e9 --jobs "$JOBS_GRID" \
    --threads-per-job "$THREADS" --out results/oracle-gap-noclamp
step R2_D results/oracle-gap-noclamp \
  "$PY" scripts/launch_oracle.py --case D --H 32 --d 2 --geometry grid \
    --worlds 4 --seeds 3 --world-seeds 1 100011 200009 300009 \
    --steps 12000 --logit-floor -1e9 --jobs "$JOBS_GRID" \
    --threads-per-job "$THREADS" --out results/oracle-gap-noclamp

# R1: four more calibration worlds. Takes the paired sign-flip test from a
# floor of p = 0.125 at four worlds to 0.008 at eight.
say "R1 eight-world oracle gap"
for C in A B C D; do
  step "R1_${C}" results/oracle-gap-w8 \
    "$PY" scripts/launch_oracle.py --case "$C" --H 32 --d 2 --geometry grid \
      --worlds 4 --world-seed0 4 --seeds 3 --steps 12000 \
      --jobs "$JOBS_GRID" --threads-per-job "$THREADS" \
      --out results/oracle-gap-w8
done

# R3: case D at deployment scale, for parity in the learnability table.
# Recorded gate-passing seeds plus --resume, or gating costs ~6 h/world.
say "R3 case D at deployment scale"
# --resume only skips trainings already on disk; the expensive part, the
# gate, is skipped by --world-seeds either way. So on a machine without
# the earlier run (a fresh clone or the worker zip) we simply drop it and
# retrain the twelve units.
R3_RESUME="results/seq-oracle/2026-07-21/042636-43847"
R3_RESUME_ARGS=()
if [ -d "$R3_RESUME" ]; then
  R3_RESUME_ARGS=(--resume "$R3_RESUME")
else
  say "R3: no earlier run here, training all units from scratch"
fi
step R3_D_seq results/seq-oracle \
  "$PY" scripts/launch_oracle.py --case D --geometry sequence \
    --H 4 --d 8 --worlds 4 --seeds 3 \
    --world-seeds 59 100029 200013 300040 \
    "${R3_RESUME_ARGS[@]}" \
    --steps 12000 --jobs "$JOBS_SEQ" --threads-per-job "$THREADS" \
    --out results/seq-oracle

# R4: oracle restarts on the binding case. Lowest value, since case B's
# oracle already passes at deployment scale. Seeds continue from 3.
say "R4 oracle restarts"
step R4_B_restarts results/oracle-gap-restarts \
  "$PY" scripts/launch_oracle.py --case B --H 32 --d 2 --geometry grid \
    --worlds 4 --seeds 5 --seed0 3 --world-seeds 0 100003 200006 300009 \
    --steps 12000 --jobs "$JOBS_GRID" --threads-per-job "$THREADS" \
    --out results/oracle-gap-restarts

# R7: matched comparison of the two error sources against all three
# knobs, on one grid with one yardstick. Exact, training-free, CPU only.
# Answers "score uncertainty is better spent as rho than conditioned on".
say "R7 matched score-noise versus weight-distrust"
step R7_sigma results/sigma-stress \
  "$PY" scripts/run_sigma_stress.py --cases A B C D --worlds 8 \
    --H 32 --d 2 --geometry grid --ball kl \
    --n-stress 500 --kappa-range 8.0 60.0 --contam-eps 0.2 \
    --jobs 6 --out results/sigma-stress-matched

# R8: graded-penalty ablation. Same worlds and budget as the coverage
# battery, with the flat epsilon replaced by epsilon*exp(-kappa*depth),
# so the boundary is unmoved and only the plateau goes. The floor case at
# high sparsity is where the collapse was, so that is what it tests.
say "R8 graded-penalty ablation"
step R8_B_s4_graded results/w33-graded \
  "$PY" scripts/launch_w33.py --cases B --sparsity 4.0 \
    --arms onpolicy mix replay teacher contrastive \
    --worlds 4 --seeds 3 --world-seeds 0 100003 200006 300009 \
    --trust-recorded-gates --H 4 --d 8 --geometry sequence \
    --steps 12000 --n-points 64 --eval-every 200 \
    --graded-kappa 5.0 --save-grids \
    --jobs "$JOBS_SEQ" --threads-per-job "$THREADS" \
    --out results/w33-graded

# R9: width sweep on the binding case, the other half of R4. Two widths
# either side of the default settle whether the oracle's shortfall is
# capacity or optimisation.
say "R9 oracle width sweep"
for W in 128 512; do
  step "R9_B_dim${W}" results/oracle-gap-width \
    "$PY" scripts/launch_oracle.py --case B --H 32 --d 2 --geometry grid \
      --worlds 4 --seeds 3 --world-seeds 0 100003 200006 300009 \
      --steps 12000 --net-dim "$W" \
      --jobs "$JOBS_GRID" --threads-per-job "$THREADS" \
      --out results/oracle-gap-width
done

[ -n "$R5_PID" ] && { say "waiting for R5"; wait "$R5_PID"; }

# ------------------------------------------------------------------ summary

say "queue finished"
printf '\n%-18s %s\n%s\n' "STEP" "RESULT" \
  "------------------------------------------------------------"
for n in "${ALL_STEPS[@]}"; do
  if   [ -f "$STATE/$n.done" ];   then printf '%-18s ok (%s min)\n' "$n" "$(cat "$STATE/$n.done")"
  elif [ -f "$STATE/$n.failed" ]; then printf '%-18s FAILED, see %s/%s.log\n' "$n" "$LOGS" "$n"
  else                                 printf '%-18s not run\n' "$n"
  fi
done
printf '\nlogs:    %s\n' "$LOGS"
printf 'markers: %s  (delete one to force a rerun)\n' "$STATE"
[ "$AUTOPUSH" = "1" ] || printf '\nCommitted locally. Push with:\n  git push origin main\n'
