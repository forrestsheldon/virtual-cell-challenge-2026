#!/bin/zsh
# Full H1 harness scores (with DE) for the finalists, one at a time (run under caffeinate -i).
# Generator v3: sample-matched factors, gene-wise rounding; the v3 unchanged control is the paired null.
set -euo pipefail
root="${0:A:h:h:h}"
report="$root/reports/h1-transfer-regression/cell_eval_v3"
scorer="/Users/forrestsheldon/Projects/vcc2026-h1-benchmark/.venv/bin/vcc-h1"
bench="/Users/forrestsheldon/Library/Caches/vcc2026-h1-benchmark"
tmp="$root/data/derived/h1_transfer_candidates"
cd "$root"
full() {  # $1 label, $2 prediction
  local start=$(date +%s)
  "$scorer" score "$2" --output "$report/$1/full" --data-dir "$bench" > "$report/$1/full.log" 2>&1
  rm -f "$2"; echo "$1 full-scored in $(( $(date +%s) - start ))s"
}
if [[ ! -s "$report/unchanged/full/scores.csv" ]]; then
  pixi run python -c "import numpy as np; from scripts.evaluation.h1_generation import canonical_axes, generate; t, g, _ = canonical_axes(); generate('unchanged', np.zeros((len(t), len(g))), '$tmp/unchanged.h5ad', '$report/unchanged')"
  full unchanged "$tmp/unchanged.h5ad"
fi
if [[ ! -s "$report/available_avg_scaled/full/scores.csv" ]]; then
  pixi run python -m scripts.evaluation.build_available_average_candidate "$tmp/available_avg_scaled.h5ad"
  full available_avg_scaled "$tmp/available_avg_scaled.h5ad"
fi
if [[ ! -s "$report/available_avg_direct/full/scores.csv" ]]; then
  pixi run python -m scripts.evaluation.build_available_average_candidate "$tmp/available_avg_direct.h5ad" --direct
  full available_avg_direct "$tmp/available_avg_direct.h5ad"
fi
if [[ ! -s "$report/avg_h1_mean/full/scores.csv" ]]; then
  pixi run python -m scripts.evaluation.generate_h1_transfer_candidates avg_h1_mean "$tmp/avg_h1_mean.h5ad"
  full avg_h1_mean "$tmp/avg_h1_mean.h5ad"
fi
echo "all finalists done"
