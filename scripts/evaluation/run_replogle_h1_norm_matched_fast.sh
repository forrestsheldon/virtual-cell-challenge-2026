#!/bin/zsh
set -euo pipefail

root="${0:A:h:h:h}"
report="$root/reports/replogle-h1-transfer/norm-matched"
effects="$root/data/derived/replogle_h1_transfer/norm_matched_effects.npz"
benchmark="/Users/forrestsheldon/Library/Caches/vcc2026-h1-benchmark"
scorer="/Users/forrestsheldon/Projects/vcc2026-h1-benchmark/.venv/bin/vcc-h1"
prediction="$root/data/derived/replogle_h1_transfer/norm_matched_candidate_tmp.h5ad"

cd "$root"
mkdir -p "$report/cell_eval"

for arm in \
  continuous_natural \
  dense_at_continuous_norm \
  continuous_at_dense_norm \
  dense_at_hard_2of3_norm \
  hard_2of3_at_dense_norm; do
  output="$report/cell_eval/$arm"
  if [[ -s "$output/scores.csv" ]]; then
    continue
  fi
  pixi run python -m scripts.evaluation.generate_replogle_h1_phase1 \
    "$arm" "$prediction" --effects "$effects" \
    --manifest "$output/generation_manifest.json"
  "$scorer" score-fast "$prediction" --output "$output" --data-dir "$benchmark"
  pixi run python - "$output" <<'PY'
import hashlib
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
generation = json.loads((output / "generation_manifest.json").read_text())
score = json.loads((output / "manifest.json").read_text())
if generation["prediction"]["sha256"] != score["prediction"]["sha256"]:
    raise SystemExit("generation/scoring prediction hashes differ")
for record in score["outputs"].values():
    path = output / record["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise SystemExit(f"compact output checksum differs: {path}")
PY
  unlink "$prediction"
done
