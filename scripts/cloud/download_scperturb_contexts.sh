#!/usr/bin/env bash
set -euo pipefail

base=https://zenodo.org/api/records/13350497/files
files=(
    NadigOConner2024_hepg2.h5ad
    NadigOConner2024_jurkat.h5ad
    ReplogleWeissman2022_K562_essential.h5ad
    ReplogleWeissman2022_rpe1.h5ad
)
md5s=(
    af2be47f7477cf32fa6e4bec1c6a4868
    d8b05d00bfbd686d37ffdd4293bc6c8c
    d8cba17576d1a8afc0f7d71b79cad0f7
    cc7f1ec50aeb3a3e1b4a6cfa713d80fa
)

pids=()
for i in "${!files[@]}"; do
    file=${files[$i]}
    (
        curl --fail --location --silent --show-error \
            --retry 8 --retry-all-errors --continue-at - \
            "$base/$file/content" --output "$file"
        actual=$(md5sum "$file" | cut -d' ' -f1)
        test "$actual" = "${md5s[$i]}"
        echo "$actual  $file"
        sha256sum "$file"
    ) >"$file.download.log" 2>&1 &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid"
done
