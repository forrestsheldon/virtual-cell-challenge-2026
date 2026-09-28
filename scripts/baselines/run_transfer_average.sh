#!/bin/zsh
# Build and validate (dry run) the two 2026 transfer-average submissions. Does not submit.
set -euo pipefail
root="${0:A:h:h:h}"; cd "$root"
for scale in 1 0.1323; do
  out="submissions/transfer-average-c20-a$scale"
  mkdir -p "$out"
  if [[ ! -s "$out/prediction.h5ad" ]]; then
    pixi run python -m scripts.baselines.transfer_average "$out/prediction.h5ad" --scale "$scale"
  fi
  pixi run vcc prep "$out/prediction.h5ad" -g data/controls/gene_names.csv \
    --perts data/controls/pert_counts.csv -o "$out/prediction.vcc" --dry-run 2>&1 | tail -20
  echo "a=$scale built and dry-run validated"
done
echo "all transfer-average builds done"
