#!/usr/bin/env bash
set -euo pipefail

device=/dev/disk/by-id/google-vcc-xatlas-scratch
mount_point=/mnt/vcc

if ! blkid "$device" >/dev/null 2>&1; then
    mkfs.ext4 -F -m 0 -L vcc-scratch "$device"
fi

mkdir -p "$mount_point"
if ! mountpoint -q "$mount_point"; then
    mount "$device" "$mount_point"
fi
