#!/usr/bin/env bash
set -euo pipefail

cd /mnt/vcc/work

export VCC_ARROW_THREADS=2
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

common=(
    --targets reference_cells.csv
    --targets pert_counts.csv
    --scratch /mnt/vcc/work/xatlas
    --workers 4
)

./venv/bin/python aggregate_xatlas_shards.py \
    --context HCT116 \
    --destination gs://bold-bastion-509200-f9-vcc2026/derived/xatlas_orion/shards/HCT116 \
    "${common[@]}"

./venv/bin/python aggregate_xatlas_shards.py \
    --context HEK293T \
    --destination gs://bold-bastion-509200-f9-vcc2026/derived/xatlas_orion/shards/HEK293T \
    "${common[@]}"
