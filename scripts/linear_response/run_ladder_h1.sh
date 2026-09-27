#!/bin/zsh
set -euo pipefail

root="${0:A:h:h:h}"
report="$root/reports/linear-response-ladder"
log="$report/logs/full-h1.log"

if [[ "${LR_FULL_H1_WORKER:-0}" != 1 ]]; then
  mkdir -p "${log:h}"
  screen -dmS lr-full-h1 zsh -lc \
    "cd '$root' && exec env LR_FULL_H1_WORKER=1 caffeinate -ims zsh '$0' > '$log' 2>&1"
  print "Started detached screen session lr-full-h1; log: $log"
  exit
fi

cd "$root"
scorer="/Users/forrestsheldon/Projects/vcc2026-h1-benchmark/.venv/bin/vcc-h1"
cell="$report/cell_eval"
prediction="data/derived/linear_response/ladder/full_cell_tmp.h5ad"
stage_status_path="$report/full_cell_status.csv"
mkdir -p "$cell"
print 'stage,status,time_utc' > "$stage_status_path"

run_stage() {
  local stage="$1"
  print "$stage,started,$(date -u +%FT%TZ)" >> "$stage_status_path"
  shift
  "$@"
  print "$stage,completed,$(date -u +%FT%TZ)" >> "$stage_status_path"
}

for arm in null empirical; do
  output="$cell/$arm"
  if [[ ! -s "$output/scores.csv" ]]; then
    run_stage "generate_$arm" pixi run python -m \
      scripts.linear_response.generate_ladder_h1 "$arm" "$prediction" \
      --manifest "$output/generation_manifest.json"
    run_stage "validate_$arm" "$scorer" validate "$prediction"
    run_stage "score_$arm" "$scorer" score "$prediction" \
      --output "$output" --de-threads 8
    unlink "$prediction"
  fi
done

run_stage plot_per_target "$scorer" plot \
  "control=$root/reports/linear-response-three-models/cell_eval/standalone_control_baseline" \
  "null=$cell/null" "empirical=$cell/empirical" \
  --output "$report/figures/full_cell_per_target.png"
run_stage summarize pixi run python -m scripts.linear_response.summarize_ladder_h1
run_stage blog_artifacts pixi run python -m scripts.linear_response.plot_ladder
print "complete,completed,$(date -u +%FT%TZ)" >> "$stage_status_path"
