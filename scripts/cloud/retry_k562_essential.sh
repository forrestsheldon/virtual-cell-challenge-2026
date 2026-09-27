#!/usr/bin/env bash
set -euo pipefail

work=/mnt/vcc/work
output="$work/final/K562_essential"
filename=ReplogleWeissman2022_K562_essential.h5ad

"$work/venv/bin/python" "$work/aggregate_scperturb_context.py" \
  "$work/$filename" \
  --context K562_essential \
  --dataset "Replogle and Weissman 2022 K562 essential CRISPRi" \
  --source-url "https://zenodo.org/records/13350497/files/$filename" \
  --source-md5 d8cba17576d1a8afc0f7d71b79cad0f7 \
  --source-sha256 412fd0df8c4ccea9f4db91cd88033c49200838b29d40945e48574be588b48789 \
  --output "$output"

"$work/venv/bin/python" "$work/audit_context_pseudobulks.py" "$output"
gcloud storage cp "$output"/* \
  gs://bold-bastion-509200-f9-vcc2026/derived/context_atlas/final/K562_essential/
sudo shutdown -h now
