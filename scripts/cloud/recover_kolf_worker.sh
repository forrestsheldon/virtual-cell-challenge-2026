#!/usr/bin/env bash
set -euo pipefail

work=/mnt/vcc/work
source_h5ad="$work/KOLF_Pan_Genome_QC_Filtered.h5ad"
source_url=https://ndownloader.figshare.com/files/64650261
expected_size=189393177972
expected_md5=afd30fde1e6ad32969c29868394385d1
expected_sha256=3e7b0eaae92cc4aacc1d85f6a416998d44445dd296aa44dab4f39d16de0dec79

mkdir -p "$work"
python3 -m venv "$work/venv"
"$work/venv/bin/pip" install --upgrade pip
"$work/venv/bin/pip" install anndata h5py numpy pandas scipy

wget --continue --output-document="$source_h5ad" "$source_url"

test "$(stat --format=%s "$source_h5ad")" = "$expected_size"
echo "$expected_md5  $source_h5ad" | md5sum --check --status
echo "$expected_sha256  $source_h5ad" | sha256sum --check --status

bash "$work/run_kolf_aggregation.sh"
date --utc --iso-8601=seconds > "$work/KOLF_RECOVERY_COMPLETE"
