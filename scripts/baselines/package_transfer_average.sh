#!/bin/zsh
# Build, dry-run, package, and record one 2026 transfer-average variant; then drop the
# 16 GB intermediate .h5ad (reproducible from transfer_average.py; its hash is in the manifest).
set -euo pipefail
root="${0:A:h:h:h}"; cd "$root"
scale=$1; shift; flags=("$@")  # --global-shift, --scatter
suffix=""
for f in "${flags[@]}"; do case $f in --global-shift) suffix+="-global";; --scatter) suffix+="-scatter";; esac; done
out="submissions/transfer-average-c20-a$scale$suffix"
mkdir -p "$out"
[[ -s "$out/prediction.h5ad" || -s "$out/prediction.vcc" ]] || \
  pixi run python -m scripts.baselines.transfer_average "$out/prediction.h5ad" --scale "$scale" "${flags[@]}"
if [[ ! -s "$out/prediction.vcc" ]]; then
  pixi run vcc prep "$out/prediction.h5ad" -g data/controls/gene_names.csv \
    --perts data/controls/pert_counts.csv -o "$out/prediction.vcc" --dry-run
  pixi run vcc prep "$out/prediction.h5ad" -g data/controls/gene_names.csv \
    --perts data/controls/pert_counts.csv -o "$out/prediction.vcc"
fi
pixi run python - "$out" <<'PY'
import hashlib, json, pathlib, sys
out = pathlib.Path(sys.argv[1]); m = out / "prediction.manifest.json"
d = hashlib.sha256()
with (out / "prediction.vcc").open("rb") as f:
    for b in iter(lambda: f.read(8 << 20), b""): d.update(b)
info = json.loads(m.read_text()); info["vcc_sha256"] = d.hexdigest()
info["vcc_bytes"] = (out / "prediction.vcc").stat().st_size
m.write_text(json.dumps(info, indent=1)); print("vcc", info["vcc_sha256"], info["vcc_bytes"])
PY
rm -f "$out/prediction.h5ad"
echo "a=$scale packaged"
