"""Replay one validated linear-response arm as a canonical H1 count H5AD."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import sparse

from scripts.linear_response.kernel import decode_multinomial, write_prediction
from scripts.linear_response.model3_poisson_lognormal import (
    BATCH_SIZE,
    INFER,
    dense_batch,
)

ROOT = Path(__file__).resolve().parents[2]
H1_PATH = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
DERIVED = ROOT / "data/derived/linear_response"
REPORT = ROOT / "reports/linear-response-three-models"
H1_SHA256 = "a09977104fefb622368ca74b50c9d3c1e891733e6c83db07acfca49b0219c02b"

ARMS = {
    "model1": ("model1_expected_profiles.npz", "model1_on_target_closure.csv"),
    "model2": ("model2_expected_profiles.npz", "model2_on_target_closure.csv"),
    "model3": ("model3_expected_profiles.npz", "model3_on_target_closure.csv"),
    "normalized_null": ("model1_expected_profiles.npz", None),
    "model3_posterior_null": ("model3_expected_profiles.npz", None),
}
MODEL_INDEX = {"model1": 0, "model2": 1, "model3": 2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_for(arm: str, source_order: int) -> np.random.SeedSequence:
    if arm == "normalized_null":
        coordinates = [0, 8, MODEL_INDEX["model1"], source_order, 0, 1]
    elif arm == "model3_posterior_null":
        coordinates = [0, 8, MODEL_INDEX["model3"], source_order, 0, 1]
    else:
        coordinates = [0, 8, MODEL_INDEX[arm], source_order, 0, 0]
    return np.random.SeedSequence(coordinates)


def closure_amplitudes(path: Path, targets: list[str]) -> dict[str, float]:
    table = pd.read_csv(path)
    if table["target_gene"].astype(str).tolist() != targets:
        raise ValueError(f"closure target order differs from artifact: {path}")
    return dict(
        zip(
            table["target_gene"].astype(str),
            table["amplitude"].astype(float),
            strict=True,
        )
    )


def validate_source_rows(
    data: ad.AnnData, rows: np.ndarray, targets: list[str]
) -> None:
    flattened = np.unique(rows.reshape(-1))
    labels = data.obs.iloc[flattened]["target_gene"].astype(str)
    if not labels.eq("non-targeting").all():
        raise ValueError("saved source rows contain perturbation truth cells")
    if rows.ndim == 1 and len(rows) != 400:
        raise ValueError("expected one fixed 400-cell source panel")
    if rows.ndim == 2 and rows.shape != (len(targets), 400):
        raise ValueError("expected one 400-cell source panel per target")


def multinomial_blocks(
    data: ad.AnnData,
    arm: str,
    targets: list[str],
    source_order: np.ndarray,
    source_rows: np.ndarray,
    directions: np.ndarray,
    amplitudes: dict[str, float] | None,
) -> dict[str, sparse.csr_matrix]:
    blocks = {}
    for index, (target, column) in enumerate(zip(targets, source_order, strict=True)):
        rows = source_rows[index]
        raw = data.X[np.sort(rows)].tocsr()
        amplitude = 0.0 if amplitudes is None else amplitudes[target]
        rng = np.random.default_rng(seed_for(arm, int(column)))
        blocks[target] = decode_multinomial(
            raw, directions[:, int(column)], amplitude, rng
        )
        print(f"{arm} cells {index + 1}/{len(targets)}: {target}", flush=True)
    return blocks


def posterior_rates(
    data: ad.AnnData, rows: np.ndarray, m: np.ndarray, loadings: np.ndarray
) -> np.ndarray:
    batches = []
    for start in range(0, len(rows), BATCH_SIZE):
        counts, totals = dense_batch(data, rows[start : start + BATCH_SIZE])
        _, rates, _ = INFER(
            jnp.asarray(m),
            jnp.asarray(loadings),
            jnp.asarray(counts),
            jnp.asarray(totals),
        )
        batches.append(np.asarray(rates, dtype=np.float64))
    result = np.vstack(batches)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("Model 3 posterior rates are invalid")
    return result


def poisson_blocks(
    data: ad.AnnData,
    arm: str,
    targets: list[str],
    source_order: np.ndarray,
    source_rows: np.ndarray,
    directions: np.ndarray,
    amplitudes: dict[str, float] | None,
) -> dict[str, sparse.csr_matrix]:
    with np.load(DERIVED / "model3_fit.npz", allow_pickle=False) as fit:
        m = fit["m"].astype(np.float64)
        loadings = fit["loadings"].astype(np.float64)
    baseline = posterior_rates(data, source_rows, m, loadings)
    blocks = {}
    for index, (target, column) in enumerate(zip(targets, source_order, strict=True)):
        rates = baseline
        if amplitudes is not None:
            exponent = np.clip(amplitudes[target] * directions[:, int(column)], -80, 80)
            rates = baseline * np.exp(exponent)[None, :]
        rng = np.random.default_rng(seed_for(arm, int(column)))
        generated = rng.poisson(rates).astype(np.int32, copy=False)
        block = sparse.csr_matrix(generated)
        block.eliminate_zeros()
        blocks[target] = block
        print(
            f"{arm} cells {index + 1}/{len(targets)}: {target}",
            flush=True,
        )
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--shared-amplitude",
        type=float,
        help="replace Model 1 target-specific amplitudes with one fixed scalar",
    )
    args = parser.parse_args()

    if args.shared_amplitude is not None:
        if args.arm != "model1":
            raise ValueError("--shared-amplitude is only defined for model1")
        if not np.isfinite(args.shared_amplitude):
            raise ValueError("--shared-amplitude must be finite")

    artifact_name, closure_name = ARMS[args.arm]
    artifact_path = DERIVED / artifact_name
    closure_path = REPORT / closure_name if closure_name else None
    with np.load(artifact_path, allow_pickle=False) as artifact:
        targets = artifact["target_gene"].astype(str).tolist()
        source_order = artifact["source_order"].astype(np.int64)
        genes = artifact["gene_names"].astype(str).tolist()
        source_rows = artifact["source_rows"].astype(np.int64)
        directions = artifact["directions"].astype(np.float64)
    if len(targets) != 126 or len(set(targets)) != 126:
        raise ValueError("artifact must contain 126 unique H1 targets")
    if directions.shape[1] <= int(source_order.max()):
        raise ValueError("direction matrix does not cover source_order")
    if args.shared_amplitude is not None:
        amplitudes = dict.fromkeys(targets, args.shared_amplitude)
        closure_path = None
        amplitude_provenance = {
            "mode": "shared_scalar_scan",
            "value": args.shared_amplitude,
        }
    else:
        amplitudes = closure_amplitudes(closure_path, targets) if closure_path else None
        amplitude_provenance = {
            "mode": "target_specific_closure" if closure_path else "null",
            "value": None,
        }

    data = ad.read_h5ad(H1_PATH, backed="r")
    try:
        if data.var_names.astype(str).tolist() != genes:
            raise ValueError("artifact gene axis differs from H1")
        validate_source_rows(data, source_rows, targets)
        if args.arm in {"model3", "model3_posterior_null"}:
            blocks = poisson_blocks(
                data,
                args.arm,
                targets,
                source_order,
                source_rows,
                directions,
                amplitudes,
            )
            generator = "Poisson draws from converged source posterior rates"
        else:
            blocks = multinomial_blocks(
                data,
                args.arm,
                targets,
                source_order,
                source_rows,
                directions,
                amplitudes,
            )
            generator = "multinomial decoder preserving each source-cell total"
    finally:
        data.file.close()

    totals = np.concatenate(
        [np.asarray(block.sum(axis=1)).ravel() for block in blocks.values()]
    )
    invalid = (totals <= 0) | (totals > 1_000_000)
    if invalid.any():
        per_target = []
        for target, block in blocks.items():
            target_totals = np.asarray(block.sum(axis=1)).ravel()
            per_target.append(
                {
                    "target_gene": target,
                    "cell_total_min": int(target_totals.min()),
                    "cell_total_median": float(np.median(target_totals)),
                    "cell_total_max": int(target_totals.max()),
                    "zero_cells": int((target_totals <= 0).sum()),
                    "over_one_million_cells": int((target_totals > 1_000_000).sum()),
                }
            )
        failed = {
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "validation_failed_no_h5ad_written",
            "arm": args.arm,
            "truth_cells_read": False,
            "failure": "generated cell totals fall outside the unchanged H1 validator range",
            "invalid_cells": int(invalid.sum()),
            "zero_cells": int((totals <= 0).sum()),
            "over_one_million_cells": int((totals > 1_000_000).sum()),
            "cell_total_min": int(totals.min()),
            "cell_total_median": float(np.median(totals)),
            "cell_total_max": int(totals.max()),
            "violating_targets": [
                row
                for row in per_target
                if row["zero_cells"] or row["over_one_million_cells"]
            ],
            "inputs": {
                "model_artifact": {
                    "path": artifact_path.relative_to(ROOT).as_posix(),
                    "sha256": sha256(artifact_path),
                },
                "closure": {
                    "path": closure_path.relative_to(ROOT).as_posix(),
                    "sha256": sha256(closure_path),
                }
                if closure_path
                else None,
            },
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(failed, indent=2, sort_keys=True) + "\n")
        raise ValueError("generated cell totals fall outside the H1 validator range")
    write_prediction(blocks, genes, args.output)
    prediction_hash = sha256(args.output)
    inputs = {
        "model_artifact": {
            "path": artifact_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(artifact_path),
        },
        "h1": {
            "path": H1_PATH.relative_to(ROOT).as_posix(),
            "sha256": H1_SHA256,
            "hash_inherited_from_validated_model_manifest": True,
        },
    }
    if closure_path:
        inputs["closure"] = {
            "path": closure_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(closure_path),
        }
    if args.arm in {"model3", "model3_posterior_null"}:
        fit = DERIVED / "model3_fit.npz"
        inputs["model3_fit"] = {
            "path": fit.relative_to(ROOT).as_posix(),
            "sha256": sha256(fit),
        }
    payload = {
        "created_utc": datetime.now(UTC).isoformat(),
        "arm": args.arm,
        "truth_cells_read": False,
        "generator": generator,
        "amplitude": amplitude_provenance,
        "targets": len(targets),
        "cells_per_target": 400,
        "genes": len(genes),
        "source_rows": "saved fixed balanced control draw",
        "seed_coordinates": (
            "[0, 8, 0, source_order, 0, 1]"
            if args.arm == "normalized_null"
            else "[0, 8, 2, source_order, 0, 1]"
            if args.arm == "model3_posterior_null"
            else f"[0, 8, {MODEL_INDEX[args.arm]}, source_order, 0, 0]"
        ),
        "cell_total_min": int(totals.min()),
        "cell_total_median": float(np.median(totals)),
        "cell_total_max": int(totals.max()),
        "stored_nonzeros": int(sum(block.nnz for block in blocks.values())),
        "inputs": inputs,
        "prediction": {
            "path": args.output.resolve().relative_to(ROOT).as_posix(),
            "sha256": prediction_hash,
            "bytes": args.output.stat().st_size,
        },
        "software": {
            "python": platform.python_version(),
            "anndata": version("anndata"),
            "jax": version("jax"),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
