#!/bin/zsh
set -euo pipefail

root="${0:A:h:h:h}"
report="$root/reports/linear-response-ladder"
log="$report/logs/paired-full-h1.log"
stage_status="$report/paired_h1_status.csv"
benchmark="/Users/forrestsheldon/Library/Caches/vcc2026-h1-benchmark"
scorer="/Users/forrestsheldon/Projects/vcc2026-h1-benchmark/.venv/bin/vcc-h1"
prediction="$root/data/derived/linear_response/ladder/paired_full_cell_tmp.h5ad"

if [[ "${LR_PAIRED_H1_WORKER:-0}" != 1 ]]; then
  mkdir -p "${log:h}"
  screen -dmS lr-paired-h1 zsh -lc \
    "cd '$root' && exec env LR_PAIRED_H1_WORKER=1 caffeinate -ims zsh '$0' > '$log' 2>&1"
  print "Started detached screen session lr-paired-h1; log: $log"
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

run_stage prepare pixi run python -m scripts.evaluation.prepare_paired_linear_response_h1

for arm in global lr lr_global; do
  output="$report/cell_eval/paired_$arm"
  if [[ ! -s "$output/scores.csv" ]]; then
    run_stage "generate_$arm" pixi run python -m \
      scripts.evaluation.generate_paired_linear_response_h1 "$arm" "$prediction" \
      --manifest "$output/generation_manifest.json"
    run_stage "validate_$arm" "$scorer" validate "$prediction" --data-dir "$benchmark"
    run_stage "score_$arm" "$scorer" score "$prediction" \
      --output "$output" --de-threads 8 --data-dir "$benchmark"
    unlink "$prediction"
  fi
done

run_stage plot "$scorer" plot \
  "null=$root/reports/linear-response-three-models/cell_eval/standalone_control_baseline" \
  "global=$report/cell_eval/paired_global" \
  "LR=$report/cell_eval/paired_lr" \
  "LR+global=$report/cell_eval/paired_lr_global" \
  --output "$report/figures/paired_h1_per_target.png" --data-dir "$benchmark"
run_stage summarize pixi run python -m scripts.evaluation.summarize_paired_linear_response_h1
print "complete,completed,$(date -u +%FT%TZ)" >> "$stage_status"
