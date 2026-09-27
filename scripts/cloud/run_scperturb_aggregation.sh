#!/usr/bin/env bash
set -euo pipefail

work=/mnt/vcc/work
base_url=https://zenodo.org/records/13350497/files

run_one() {
  local context=$1
  local dataset=$2
  local filename=$3
  local md5=$4
  local sha256=$5
  local output="$work/final/$context"

  "$work/venv/bin/python" "$work/aggregate_scperturb_context.py" \
    "$work/$filename" \
    --context "$context" \
    --dataset "$dataset" \
    --source-url "$base_url/$filename" \
    --source-md5 "$md5" \
    --source-sha256 "$sha256" \
    --output "$output"
  "$work/venv/bin/python" "$work/audit_context_pseudobulks.py" "$output"
  gcloud storage cp "$output"/* \
    "gs://bold-bastion-509200-f9-vcc2026/derived/context_atlas/final/$context/"
}

run_one \
  HepG2 \
  "Nadig and O'Connor 2024 HepG2 CRISPRi" \
  NadigOConner2024_hepg2.h5ad \
  af2be47f7477cf32fa6e4bec1c6a4868 \
  1af2f7b3e692ad3d077e6027d68a1f800619e29aa3f7a0efa70146fb7223a4bf &
hep_pid=$!
run_one \
  Jurkat \
  "Nadig and O'Connor 2024 Jurkat CRISPRi" \
  NadigOConner2024_jurkat.h5ad \
  d8b05d00bfbd686d37ffdd4293bc6c8c \
  ade3d83150fd6944212363cd11da77c0d2143fa6df1febb3f15a66d4dab0677f &
jurkat_pid=$!
wait "$hep_pid"
wait "$jurkat_pid"

run_one \
  K562_essential \
  "Replogle and Weissman 2022 K562 essential CRISPRi" \
  ReplogleWeissman2022_K562_essential.h5ad \
  d8cba17576d1a8afc0f7d71b79cad0f7 \
  412fd0df8c4ccea9f4db91cd88033c49200838b29d40945e48574be588b48789 &
k562_pid=$!
run_one \
  RPE1 \
  "Replogle and Weissman 2022 RPE1 essential CRISPRi" \
  ReplogleWeissman2022_rpe1.h5ad \
  cc7f1ec50aeb3a3e1b4a6cfa713d80fa \
  12d6a0cf9378c4f09411e27f0ce07b59f1d97731bd576d6fadf112f6f088d1a4 &
rpe1_pid=$!
wait "$k562_pid"
wait "$rpe1_pid"

sudo shutdown -h now
