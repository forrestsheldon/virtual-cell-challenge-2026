#!/bin/zsh
# Full H1 harness scores at the frozen c = 20 (run under caffeinate -i). Paired null: v3 unchanged.
set -euo pipefail
root="${0:A:h:h:h}"
report="$root/reports/h1-transfer-regression/cell_eval_v3"
scorer="/Users/forrestsheldon/Projects/vcc2026-h1-benchmark/.venv/bin/vcc-h1"
bench="/Users/forrestsheldon/Library/Caches/vcc2026-h1-benchmark"
tmp="$root/data/derived/h1_transfer_candidates"
cd "$root"
for spec in "available_avg_scaled_c20:" "available_avg_direct_c20:--direct" "available_avg_direct_wrong_target_c20:--direct --wrong-target"; do
  label=${spec%%:*}; flags=(${=spec#*:})
  [[ -s "$report/$label/full/scores.csv" ]] && continue
  start=$(date +%s)
  pixi run python -m scripts.evaluation.build_available_average_candidate "$tmp/$label.h5ad" --pseudocount 20 $flags
  mkdir -p "$report/$label"
  "$scorer" score "$tmp/$label.h5ad" --output "$report/$label/full" --data-dir "$bench" > "$report/$label/full.log" 2>&1
  rm -f "$tmp/$label.h5ad"; echo "$label full-scored in $(( $(date +%s) - start ))s"
done
echo "all c20 finalists done"
