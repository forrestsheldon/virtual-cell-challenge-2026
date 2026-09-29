#!/bin/zsh
# kNN-neighbourhood Poisson null tests (k = 50, 30), scored one at a time (run under caffeinate -i).
set -euo pipefail
root="${0:A:h:h:h}"; cd "$root"
report="reports/h1-harness-review/generators"
scorer="/Users/forrestsheldon/Projects/vcc2026-h1-benchmark/.venv/bin/vcc-h1"
bench="/Users/forrestsheldon/Library/Caches/vcc2026-h1-benchmark"
tmp="data/derived/h1_transfer_candidates/generator_knn.h5ad"
for k in 50 30; do
  label="knn_poisson_k$k"
  [[ -s "$report/$label/full/scores.csv" ]] && continue
  start=$(date +%s)
  pixi run python -m scripts.evaluation.generate_h1_null_generators knn_poisson --k "$k" "$tmp"
  "$scorer" score "$tmp" --output "$report/$label/full" --data-dir "$bench" --calibration vcc2026-val-1 > "$report/$label/full.log" 2>&1
  rm -f "$tmp"; echo "$label null-scored in $(( $(date +%s) - start ))s"
done
echo "all knn nulls done"
