#!/bin/zsh
# Generate and fully score the frozen transfer candidates on the H1 harness.
set -euo pipefail
root="${0:A:h:h:h}"
report="$root/reports/h1-transfer-regression/cell_eval"
benchmark="/Users/forrestsheldon/Library/Caches/vcc2026-h1-benchmark"
scorer="/Users/forrestsheldon/Projects/vcc2026-h1-benchmark/.venv/bin/vcc-h1"
prediction="$root/data/derived/h1_transfer_candidates/candidate_tmp.h5ad"
cd "$root"
mkdir -p "$report"
for arm in avg_scaled avg_direct study_avg_scaled multi_nnls h1_mean avg_h1_mean; do
  output="$report/$arm"
  [[ -s "$output/scores.csv" ]] && continue
  mkdir -p "$output"
  start=$(date +%s)
  pixi run python -m scripts.evaluation.generate_h1_transfer_candidates "$arm" "$prediction" \
    | tee "$output/generation.log"
  "$scorer" score "$prediction" --output "$output" --data-dir "$benchmark"
  rm -f "$prediction"
  echo "$arm finished in $(( $(date +%s) - start ))s"
done
