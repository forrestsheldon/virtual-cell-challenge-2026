"""Generate one deterministic H1 candidate from a pooled Replogle Phase 1 arm."""

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
EFFECTS = ROOT / "data/derived/replogle_h1_transfer/phase1_effects.npz"
CONTROLS = Path(user_cache_path("vcc2026-h1-benchmark")) / "h1_controls.h5ad"
CONTROL_MANIFEST = Path(user_cache_path("vcc2026-h1-benchmark")) / "h1_controls_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm")
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--effects", type=Path, default=EFFECTS)
    args = parser.parse_args()

    with np.load(args.effects, allow_pickle=False) as saved:
        arms = saved["arm"].astype(str).tolist()
        if args.arm not in arms:
            raise ValueError(f"unknown arm {args.arm!r}; choose from {arms}")
        targets = saved["target_gene"].astype(str).tolist()
        genes = saved["output_gene"].astype(str).tolist()
        indices = saved["full_gene_index"].astype(int)
        source_rows = saved["source_rows"].astype(int)
        shared_effects = saved["intended_lfc"][arms.index(args.arm)].astype(np.float64)

    controls = ad.read_h5ad(CONTROLS, backed="r")
    try:
        if controls.var_names.astype(str).tolist() != genes:
            raise ValueError("effect and H1 control gene axes differ")
        blocks = {}
        for target_index, target in enumerate(targets):
            effect = np.zeros(len(genes), dtype=np.float64)
            effect[indices] = shared_effects[target_index]
            raw = controls.X[source_rows[target_index]].tocsr()
            blocks[target] = decode_lfc_largest_remainder(raw, effect)
            print(f"{args.arm} {target_index + 1}/{len(targets)}: {target}", flush=True)
    finally:
        controls.file.close()

    write_prediction(blocks, genes, args.output)
    control_artifact = json.loads(CONTROL_MANIFEST.read_text())["artifact"]
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "arm": args.arm,
        "generator": "multiplicative LFC closure with deterministic largest-remainder counts",
        "source_rows": "canonical all-control H1 rows, identical across arms",
        "cells_per_target": 400,
        "truth_cells_read": False,
        "inputs": {
            str(args.effects.resolve()): sha256(args.effects),
            str(CONTROLS): control_artifact["sha256"],
        },
        "prediction": {
            "path": str(args.output.resolve()),
            "sha256": sha256(args.output),
            "bytes": args.output.stat().st_size,
        },
        "software": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("anndata", "numpy", "scipy")},
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
