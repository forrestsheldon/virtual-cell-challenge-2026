#!/bin/zsh
set -euo pipefail

root="${0:A:h:h:h}"
log="$root/reports/linear-response-ladder/logs/ladder.log"

if [[ "${LR_LADDER_WORKER:-0}" != 1 ]]; then
  mkdir -p "${log:h}"
  screen -dmS lr-ladder zsh -lc \
    "cd '$root' && exec env LR_LADDER_WORKER=1 caffeinate -ims '$0' > '$log' 2>&1"
  print "Started detached screen session lr-ladder; log: $log"
  exit
fi

cd "$root"
status="reports/linear-response-ladder/stage_status.csv"
print 'stage,status,time_utc' > "$status"

run_stage() {
  local stage="$1"
  print "$stage,started,$(date -u +%FT%TZ)" >> "$status"
  shift
  "$@"
  print "$stage,completed,$(date -u +%FT%TZ)" >> "$status"
}

[[ -s data/derived/linear_response/ladder/empirical.npz ]] || \
  run_stage empirical pixi run python -m scripts.linear_response.ladder_empirical
[[ -s data/derived/linear_response/ladder/sparse.npz ]] || \
  run_stage sparse pixi run python -m scripts.linear_response.ladder_sparse
[[ -s data/derived/linear_response/ladder/factor.npz ]] || \
  run_stage factor pixi run python -m scripts.linear_response.ladder_factor

run_stage forcing_ceiling pixi run python -m scripts.linear_response.forcing_ceiling
run_stage evaluate pixi run python -m scripts.linear_response.evaluate_ladder
run_stage blog_artifacts pixi run python -m scripts.linear_response.plot_ladder
