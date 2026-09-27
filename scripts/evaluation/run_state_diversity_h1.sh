#!/bin/zsh
set -euo pipefail

root="${0:A:h:h:h}"
report="$root/reports/linear-response-state-diversity"
log="$report/run.log"
stage_status="$report/status.csv"
benchmark="/Users/forrestsheldon/Library/Caches/vcc2026-h1-benchmark"
scorer="/Users/forrestsheldon/Projects/vcc2026-h1-benchmark/.venv/bin/vcc-h1"
prediction="$root/data/derived/linear_response/state_diversity/prediction_tmp.h5ad"

if [[ "${LR_STATE_DIVERSITY_WORKER:-0}" != 1 ]]; then
  mkdir -p "$report"
  screen -dmS lr-state-diversity zsh -lc \
    "cd '$root' && exec env LR_STATE_DIVERSITY_WORKER=1 caffeinate -ims zsh '$0' > '$log' 2>&1"
  print "Started detached screen session lr-state-diversity; log: $log"
  exit
fi

cd "$root"
mkdir -p "$report/cell_eval"
print 'stage,status,time_utc' > "$stage_status"

run_stage() {
  local stage="$1"
  print "$stage,started,$(date -u +%FT%TZ)" >> "$stage_status"
  shift
  "$@"
  print "$stage,completed,$(date -u +%FT%TZ)" >> "$stage_status"
}

artifact="data/derived/linear_response/state_diversity/state_balanced_covariance.npz"
effects="data/derived/linear_response/state_diversity/effects.npz"
[[ -s "$artifact" ]] || run_stage fit pixi run python -m \
  scripts.linear_response.state_balanced_covariance
[[ -s "$effects" ]] || run_stage prepare pixi run python -m \
  scripts.evaluation.prepare_state_diversity_h1

for arm in control all_within all_between all_combined \
  crossfit_within crossfit_between crossfit_combined; do
  output="$report/cell_eval/$arm"
  if [[ ! -s "$output/scores.csv" ]]; then
    run_stage "generate_$arm" pixi run python -m \
      scripts.evaluation.generate_state_diversity_h1 "$arm" "$prediction" \
      --manifest "$output/generation_manifest.json"
    run_stage "validate_$arm" "$scorer" validate "$prediction" \
      --data-dir "$benchmark"
    run_stage "score_$arm" "$scorer" score "$prediction" \
      --output "$output" --de-threads 8 --data-dir "$benchmark"
    unlink "$prediction"
  fi
done

run_stage summarize pixi run python -m \
  scripts.evaluation.summarize_state_diversity_h1
print "complete,completed,$(date -u +%FT%TZ)" >> "$stage_status"
