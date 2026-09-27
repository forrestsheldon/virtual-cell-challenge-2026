#!/usr/bin/env bash
set -euo pipefail

url=$1
output=$2
total_size=$3
expected_md5=$4
workers=${5:-8}

prefix_size=$(stat -c %s "$output")
remaining=$((total_size - prefix_size))
chunk_size=$(((remaining + workers - 1) / workers))
pids=()

for ((i = 0; i < workers; i++)); do
    start=$((prefix_size + i * chunk_size))
    end=$((start + chunk_size - 1))
    ((end >= total_size)) && end=$((total_size - 1))
    ((start >= total_size)) && continue
    (
        curl --fail --location --silent --show-error \
            --retry 8 --retry-all-errors \
            --header "Range: bytes=$start-$end" \
            --output "$output.part.$i" "$url"
        test "$(stat -c %s "$output.part.$i")" -eq "$((end - start + 1))"
    ) &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

for ((i = 0; i < workers; i++)); do
    part="$output.part.$i"
    [[ -e "$part" ]] || continue
    dd if="$part" of="$output" oflag=append conv=notrunc status=none
    rm "$part"
done

test "$(stat -c %s "$output")" -eq "$total_size"
actual_md5=$(md5sum "$output" | cut -d' ' -f1)
test "$actual_md5" = "$expected_md5"
echo "$actual_md5  $output"
sha256sum "$output"
