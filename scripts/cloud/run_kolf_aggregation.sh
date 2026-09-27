#!/usr/bin/env bash
set -euo pipefail

work=/mnt/vcc/work
panel="$work/kolf_h1_vcc_union.csv"
output="$work/staging/KOLF2.1J-h1-vcc"

"$work/venv/bin/python" "$work/aggregate_kolf_h5ad.py" \
  "$work/KOLF_Pan_Genome_QC_Filtered.h5ad" \
  --targets "$panel" \
  --source-md5 afd30fde1e6ad32969c29868394385d1 \
  --source-sha256 3e7b0eaae92cc4aacc1d85f6a416998d44445dd296aa44dab4f39d16de0dec79 \
  --output "$output" \
  --gene-block 64

"$work/venv/bin/python" "$work/audit_context_pseudobulks.py" "$output"
