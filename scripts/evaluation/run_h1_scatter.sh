#!/bin/zsh
# Scattered-count generator for sparse increases: the a = 1 available-source transfer, then its null (run under caffeinate -i).
set -euo pipefail
root="${0:A:h:h:h}"; cd "$root"
base="reports/h1-transfer-regression/cell_eval_v3"
scorer="/Users/forrestsheldon/Projects/vcc2026-h1-benchmark/.venv/bin/vcc-h1"
bench="/Users/forrestsheldon/Library/Caches/vcc2026-h1-benchmark"
tmp="data/derived/h1_transfer_candidates/scatter.h5ad"
for label in available_avg_direct_c20_scatter scatter_null; do
  [[ -s "$base/$label/full/scores.csv" ]] && continue
  start=$(date +%s)
  flag=(); [[ $label == scatter_null ]] && flag=(--null)
  pixi run python -m scripts.evaluation.build_scatter_candidate "$tmp" "${flag[@]}"
  "$scorer" score "$tmp" --output "$base/$label/full" --data-dir "$bench" --calibration vcc2026-val-1 > "$base/$label/full.log" 2>&1
  rm -f "$tmp"; echo "$label scored in $(( $(date +%s) - start ))s"
done
echo "scatter runs done"
