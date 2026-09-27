"""Generate one H1 arm from a state-balanced covariance response."""

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
from platformdirs import user_cache_path

from scripts.linear_response.kernel import (
    decode_lfc_dependent_round,
    expected_lfc_decoded_sum,
    write_prediction,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = Path(user_cache_path("vcc2026-h1-benchmark"))
CONTROLS = BENCHMARK / "h1_controls.h5ad"
CONTROL_MANIFEST = BENCHMARK / "h1_controls_manifest.json"
EFFECTS = ROOT / "data/derived/linear_response/state_diversity/effects.npz"
ARMS = [
    "control",
    "all_within",
    "all_between",
    "all_combined",
    "crossfit_within",
    "crossfit_between",
    "crossfit_combined",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log2_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.log2((numerator + 1e-9) / (denominator + 1e-9))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    with np.load(EFFECTS, allow_pickle=False) as saved:
        models = saved["model"].astype(str).tolist()
        targets = saved["target_gene"].astype(str).tolist()
        genes = saved["gene"].astype(str).tolist()
        output_genes = saved["output_gene"].astype(str).tolist()
        source_rows = saved["source_rows"].astype(np.int64)
        effects = saved["lfc_effect"][models.index(args.arm)].astype(np.float64)
    output_indices = np.asarray([genes.index(gene) for gene in output_genes])

    controls = ad.read_h5ad(CONTROLS, backed="r")
    try:
        if controls.var_names.astype(str).tolist() != genes:
            raise ValueError("state-diversity effects and control gene axes differ")
        blocks = {}
        audit = []
        for target_index, target in enumerate(targets):
            raw = controls.X[source_rows[target_index]].tocsr()
            rng = np.random.default_rng(
                np.random.SeedSequence([0, 114, target_index])
            )
            block = decode_lfc_dependent_round(raw, effects[target_index], rng)
            blocks[target] = block
            baseline = np.asarray(raw.sum(axis=0)).ravel()
            fractional = expected_lfc_decoded_sum(raw, effects[target_index])
            realized = np.asarray(block.sum(axis=0)).ravel()
            expected_lfc = log2_ratio(fractional, baseline)[output_indices]
            realized_lfc = log2_ratio(realized, baseline)[output_indices]
            lfc_error = realized_lfc - expected_lfc
            audit.append(
                {
                    "target_gene": target,
                    "mean_abs_lfc_error": np.mean(np.abs(lfc_error)),
                    "median_abs_lfc_error": np.median(np.abs(lfc_error)),
                    "max_abs_lfc_error": np.max(np.abs(lfc_error)),
                    "pooled_count_relative_l1": np.abs(realized - fractional).sum()
                    / fractional.sum(),
                }
            )
            print(
                f"{args.arm} {target_index + 1}/{len(targets)}: {target}",
                flush=True,
            )
    finally:
        controls.file.close()

    write_prediction(blocks, genes, args.output)
    audit_path = args.manifest.parent / "rounding_audit.csv"
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit).to_csv(audit_path, index=False)
    control_artifact = json.loads(CONTROL_MANIFEST.read_text())["artifact"]
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "arm": args.arm,
        "generator": "multiplicative LFC closure with unbiased dependent rounding",
        "source_rows": "canonical all-control H1 rows, shared across every arm",
        "rounding_stream": "SeedSequence([0,114,target_index]), shared across arms",
        "cells_per_target": 400,
        "inputs": {
            str(EFFECTS.relative_to(ROOT)): sha256(EFFECTS),
            str(CONTROLS): control_artifact["sha256"],
        },
        "prediction": {
            "path": str(args.output.resolve()),
            "sha256": sha256(args.output),
            "bytes": args.output.stat().st_size,
        },
        "rounding_audit": {
            "path": str(audit_path.relative_to(ROOT)),
            "sha256": sha256(audit_path),
            "mean_abs_lfc_error": float(
                pd.DataFrame(audit)["mean_abs_lfc_error"].mean()
            ),
            "median_pooled_count_relative_l1": float(
                pd.DataFrame(audit)["pooled_count_relative_l1"].median()
            ),
        },
        "software": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ["anndata", "numpy", "pandas", "scipy"]
            },
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
