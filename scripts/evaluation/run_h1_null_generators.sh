#!/bin/zsh
# Generate and fully score the generator null tests one at a time (run under caffeinate -i).
set -euo pipefail
root="${0:A:h:h:h}"; cd "$root"
report="reports/h1-harness-review/generators"
scorer="/Users/forrestsheldon/Projects/vcc2026-h1-benchmark/.venv/bin/vcc-h1"
bench="/Users/forrestsheldon/Library/Caches/vcc2026-h1-benchmark"
tmp="data/derived/h1_transfer_candidates/generator_null.h5ad"
for kind in pooled_nb metacell_poisson metacell_nb; do
  [[ -s "$report/$kind/full/scores.csv" ]] && continue
  start=$(date +%s)
  pixi run python -m scripts.evaluation.generate_h1_null_generators "$kind" "$tmp"
  "$scorer" score "$tmp" --output "$report/$kind/full" --data-dir "$bench" --calibration vcc2026-val-1 > "$report/$kind/full.log" 2>&1
  rm -f "$tmp"; echo "$kind null-scored in $(( $(date +%s) - start ))s"
done
echo "all generator nulls done"
