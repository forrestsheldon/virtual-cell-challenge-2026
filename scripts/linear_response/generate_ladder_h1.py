"""Generate one cell-level H1 arm from the calibrated empirical ladder model."""

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
import pandas as pd

from scripts.linear_response.kernel import decode_multinomial, write_prediction

ROOT = Path(__file__).resolve().parents[2]
H1 = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
MODEL = ROOT / "data/derived/linear_response/ladder/empirical.npz"
SCALES = ROOT / "reports/linear-response-ladder/downstream_amplitude_distribution.csv"
H1_SHA256 = "a09977104fefb622368ca74b50c9d3c1e891733e6c83db07acfca49b0219c02b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed(arm: str, target_index: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([0, 91, 0, target_index, arm == "null"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=["null", "empirical"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    with np.load(MODEL, allow_pickle=False) as model:
        targets = model["target_gene"].astype(str)
        fit_indices = model["full_gene_index"].astype(int)
        source_rows = model["source_rows"].astype(np.int64)
        responses = model["response"].astype(np.float64)
    scales = (
        pd.read_csv(SCALES)
        .query("model == 'empirical'")
        .set_index("target_gene")
        .loc[targets, "loo_median_gamma"]
        .to_numpy()
    )
    amplitudes = np.zeros(len(targets)) if args.arm == "null" else scales

    data = ad.read_h5ad(H1, backed="r")
    try:
        genes = data.var_names.astype(str).tolist()
        blocks = {}
        for index, target in enumerate(targets):
            rows = np.sort(source_rows[index])
            if not (
                data.obs.iloc[rows]["target_gene"]
                .astype(str)
                .eq("non-targeting")
                .all()
            ):
                raise ValueError("saved source rows include truth cells")
            direction = np.zeros(data.n_vars)
            direction[fit_indices] = responses[index]
            blocks[target] = decode_multinomial(
                data.X[rows].tocsr(),
                direction,
                amplitudes[index],
                np.random.default_rng(seed(args.arm, index)),
            )
            print(f"{args.arm} {index + 1}/{len(targets)}: {target}", flush=True)
    finally:
        data.file.close()

    write_prediction(blocks, genes, args.output)
    payload = {
        "created_utc": datetime.now(UTC).isoformat(),
        "arm": args.arm,
        "model": "CPM>5 empirical covariance, on-target coefficient -1",
        "amplitude": "leave-one-target-out median downstream L1 gamma" if args.arm == "empirical" else 0.0,
        "truth_cells_read": False,
        "targets": len(targets),
        "cells_per_target": 400,
        "genes": len(genes),
        "source_rows": "identical saved balanced controls in both arms",
        "decode_stream": f"SeedSequence([0,91,0,target_index,{int(args.arm == 'null')}])",
        "inputs": {
            str(H1.relative_to(ROOT)): H1_SHA256,
            str(MODEL.relative_to(ROOT)): sha256(MODEL),
            str(SCALES.relative_to(ROOT)): sha256(SCALES),
        },
        "prediction": {
            "path": str(args.output.resolve()),
            "sha256": sha256(args.output),
            "bytes": args.output.stat().st_size,
        },
        "software": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ["anndata", "numpy", "pandas", "scipy"]
            },
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
