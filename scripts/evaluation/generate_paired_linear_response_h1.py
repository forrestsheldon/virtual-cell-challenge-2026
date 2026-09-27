"""Generate one deterministic paired H1 linear-response arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
from platformdirs import user_cache_path

from scripts.linear_response.kernel import (
    decode_lfc_largest_remainder,
    write_prediction,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = Path(user_cache_path("vcc2026-h1-benchmark"))
CONTROLS = BENCHMARK / "h1_controls.h5ad"
CONTROL_MANIFEST = BENCHMARK / "h1_controls_manifest.json"
EFFECTS = ROOT / "data/derived/linear_response/ladder/paired_h1_effects.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=["global", "lr", "lr_global"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    with np.load(EFFECTS, allow_pickle=False) as saved:
        arms = saved["arm"].astype(str).tolist()
        targets = saved["target_gene"].astype(str).tolist()
        genes = saved["gene"].astype(str).tolist()
        source_rows = saved["source_rows"].astype(np.int64)
        effects = saved["lfc_effect"][arms.index(args.arm)].astype(np.float64)

    controls = ad.read_h5ad(CONTROLS, backed="r")
    try:
        if controls.var_names.astype(str).tolist() != genes:
            raise ValueError("effect and benchmark gene axes differ")
        blocks = {}
        for index, target in enumerate(targets):
            raw = controls.X[source_rows[index]].tocsr()
            blocks[target] = decode_lfc_largest_remainder(raw, effects[index])
            print(f"{args.arm} {index + 1}/{len(targets)}: {target}", flush=True)
    finally:
        controls.file.close()

    write_prediction(blocks, genes, args.output)
    control_artifact = json.loads(CONTROL_MANIFEST.read_text())["artifact"]
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "arm": args.arm,
        "generator": "multiplicative LFC closure with deterministic largest-remainder counts",
        "source_rows": "canonical all-control H1 baseline rows, identical across arms",
        "cells_per_target": 400,
        "truth_cells_read": False,
        "inputs": {
            str(EFFECTS.relative_to(ROOT)): sha256(EFFECTS),
            str(CONTROLS): control_artifact["sha256"],
        },
        "prediction": {
            "path": str(args.output.resolve()),
            "sha256": sha256(args.output),
            "bytes": args.output.stat().st_size,
        },
        "software": {
            "python": platform.python_version(),
            **{name: version(name) for name in ["anndata", "numpy", "scipy"]},
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
