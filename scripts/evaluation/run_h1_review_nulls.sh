#!/bin/zsh
# Full-score the harness-review predictions one at a time (run under caffeinate -i).
set -euo pipefail
root="${0:A:h:h:h}"
report="$root/reports/h1-harness-review"
scorer="/Users/forrestsheldon/Projects/vcc2026-h1-benchmark/.venv/bin/vcc-h1"
bench="/Users/forrestsheldon/Library/Caches/vcc2026-h1-benchmark"
tmp="$root/data/derived/h1_transfer_candidates/review_tmp.h5ad"
cd "$root"
run() {  # $1 label, rest: generator args
  local label=$1; shift
  [[ -s "$report/$label/full/scores.csv" ]] && return
  local start=$(date +%s)
  pixi run python -m scripts.evaluation.generate_h1_review_nulls "$@"
  "$scorer" score "$tmp" --output "$report/$label/full" --data-dir "$bench" > "$report/$label/full.log" 2>&1
  rm -f "$tmp"; echo "$label full-scored in $(( $(date +%s) - start ))s"
}
run global_shift_stochastic shift "$tmp"
run depth_null_x1 depth "$tmp" --k 1
run depth_null_x2 depth "$tmp" --k 2
run depth_null_x4 depth "$tmp" --k 4
echo "all review nulls done"
