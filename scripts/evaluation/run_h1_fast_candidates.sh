#!/bin/zsh
# Regenerate candidates with dependent rounding and fast-score them (PDS + MSE only).
set -euo pipefail
root="${0:A:h:h:h}"
report="$root/reports/h1-transfer-regression/cell_eval_v2"
scorer="/Users/forrestsheldon/Projects/vcc2026-h1-benchmark/.venv/bin/vcc-h1"
bench="/Users/forrestsheldon/Library/Caches/vcc2026-h1-benchmark"
tmp="$root/data/derived/h1_transfer_candidates"
cd "$root"
fast() {  # $1 label, $2 prediction
  "$scorer" score-fast "$2" --output "$report/$1/fast" --data-dir "$bench" > "$report/$1/fast.log" 2>&1
  rm -f "$2"; echo "$1 fast-scored"
}
for arm in avg_scaled avg_direct study_avg_scaled multi_nnls h1_mean avg_h1_mean; do
  for tag in "" "_s2" "_s3"; do
    [[ -n "$tag" && "$arm" != "avg_scaled" ]] && continue
    [[ -s "$report/$arm$tag/fast/scores.csv" ]] && continue
    pixi run python -m scripts.evaluation.generate_h1_transfer_candidates "$arm" "$tmp/$arm$tag.h5ad" --tag "$tag"
    fast "$arm$tag" "$tmp/$arm$tag.h5ad"
  done
done
if [[ ! -s "$report/available_avg_scaled/fast/scores.csv" ]]; then
  pixi run python -m scripts.evaluation.build_available_average_candidate "$tmp/available_avg_scaled.h5ad"
  fast available_avg_scaled "$tmp/available_avg_scaled.h5ad"
fi
if [[ ! -s "$report/shared_plus_specific/fast/scores.csv" ]]; then
  pixi run python -m scripts.evaluation.build_shared_response_candidates "$tmp"
  fast shared_only "$tmp/shared_only.h5ad"
  fast shared_plus_specific "$tmp/shared_plus_specific.h5ad"
fi
echo "all fast candidates done"
