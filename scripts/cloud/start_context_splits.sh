#!/usr/bin/env bash
set -euo pipefail
exec > >(tee -a /var/log/vcc-splits.log /dev/console) 2>&1
finish() {
    result=$?
    trap - EXIT
    set +e
    if command -v gcloud >/dev/null && [[ "$(gcloud config configurations list --filter=is_active:true --format='value(name)')" == vcc-2026 ]] && [[ "$(gcloud config get-value project)" == bold-bastion-509200-f9 ]]; then
        gcloud storage cp /var/log/vcc-splits.log gs://bold-bastion-509200-f9-vcc2026/derived/context_transfer_splits/staging/split-v1-20260924/worker.log
    fi
    sync
    shutdown -h now
    exit "$result"
}
trap finish EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y python3-venv curl ca-certificates gnupg
if ! command -v gcloud >/dev/null; then
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
    echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main' > /etc/apt/sources.list.d/google-cloud-sdk.list
    apt-get update -q
    apt-get install -y google-cloud-cli
fi
gcloud config configurations create vcc-2026 --activate
gcloud config set project bold-bastion-509200-f9
[[ "$(gcloud config configurations list --filter=is_active:true --format='value(name)')" == vcc-2026 ]]
[[ "$(gcloud config get-value project)" == bold-bastion-509200-f9 ]]
mkdir -p /mnt/vcc
# Only format the new, explicitly named release disk; never discover other disks.
if ! blkid /dev/disk/by-id/google-vcc-splits-scratch; then
    mkfs.ext4 /dev/disk/by-id/google-vcc-splits-scratch
fi
mount /dev/disk/by-id/google-vcc-splits-scratch /mnt/vcc
mkdir -p /mnt/vcc/split-release
cd /mnt/vcc/split-release
gcloud storage cp gs://bold-bastion-509200-f9-vcc2026/derived/context_transfer_splits/staging/split-v1-20260924/launch/bundle.tar.gz bundle.tar.gz
expected=$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/attributes/bundle-sha256)
echo "$expected  bundle.tar.gz" | sha256sum --check
tar -xzf bundle.tar.gz
python3 -m venv venv
venv/bin/pip install --disable-pip-version-check -r requirements.txt
venv/bin/pip freeze > environment.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VCC_ARROW_THREADS=2
venv/bin/python scripts/cloud/run_context_splits.py
