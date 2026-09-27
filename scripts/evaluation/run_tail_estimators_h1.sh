#!/bin/zsh
set -euo pipefail

root="${0:A:h:h:h}"
log="$root/reports/tail-estimator-h1/full-score.log"
stage_status="$root/reports/tail-estimator-h1/full-score-status.csv"
benchmark="/Users/forrestsheldon/Library/Caches/vcc2026-h1-benchmark"
scorer="/Users/forrestsheldon/Projects/vcc2026-h1-benchmark/.venv/bin/vcc-h1"
prediction="$root/data/derived/linear_response/tail_estimator_tmp.h5ad"

if [[ "${TAIL_H1_WORKER:-0}" != 1 ]]; then
  mkdir -p "${log:h}"
  screen -dmS tail-h1-score zsh -lc \
    "cd '$root' && exec env TAIL_H1_WORKER=1 caffeinate -ims zsh '$0' > '$log' 2>&1"
  print "Started detached screen session tail-h1-score; log: $log"
  exit
fi

cd "$root"
print 'stage,status,time_utc' > "$stage_status"

run_stage() {
  local stage="$1"
  print "$stage,started,$(date -u +%FT%TZ)" >> "$stage_status"
  shift
  "$@"
  print "$stage,completed,$(date -u +%FT%TZ)" >> "$stage_status"
}

for arm in transductive control_only; do
  if [[ "$arm" == transductive ]]; then
    report="$root/reports/tail-estimator-h1"
    selected="$report/selected_source_rows.npy"
    kind="transductive bottom-tail prediction"
  else
    report="$root/reports/control-tail-estimator-h1"
    selected="$report/selected_source_rows.npy"
    kind="control-only bottom-tail prediction"
  fi
  output="$report/cell_eval"
  if [[ ! -s "$output/scores.csv" ]]; then
    run_stage "generate_$arm" pixi run python -m \
      scripts.evaluation.write_tail_estimator_h1 "$prediction" \
      --selected "$selected" --kind "$kind" \
      --manifest "$output/generation_manifest.json"
    run_stage "validate_$arm" "$scorer" validate "$prediction" \
      --data-dir "$benchmark"
    run_stage "score_$arm" "$scorer" score "$prediction" \
      --output "$output" --de-threads 8 --data-dir "$benchmark"
    unlink "$prediction"
  fi
done

print "complete,completed,$(date -u +%FT%TZ)" >> "$stage_status"
