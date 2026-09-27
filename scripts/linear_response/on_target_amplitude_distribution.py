"""Compare simple on-target Model 1 amplitudes in matched CP10K coordinates."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scripts.linear_response.kernel import CELL_TARGET_SUM

ROOT = Path(__file__).resolve().parents[2]
H1 = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
MODEL = ROOT / "data/derived/linear_response/model1_expected_profiles.npz"
SCALE_AUDIT = ROOT / "reports/linear-response-three-models/calibration_scale_audit.csv"
TARGET_TABLE = ROOT / "reports/linear-response-three-models/h1_target_counts.csv"
REPORT = ROOT / "reports/linear-response-three-models"
TABLE = REPORT / "model1_individual_amplitude_distribution.csv"
SUMMARY = REPORT / "model1_individual_amplitude_distribution.json"
CHUNK_SIZE = 1_024
H1_SHA256 = "a09977104fefb622368ca74b50c9d3c1e891733e6c83db07acfca49b0219c02b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: pd.Series) -> dict[str, float]:
    return {
        "minimum": float(values.min()),
        "q10": float(values.quantile(0.1)),
        "median": float(values.median()),
        "q90": float(values.quantile(0.9)),
        "maximum": float(values.max()),
    }


def main() -> None:
    with np.load(MODEL, allow_pickle=False) as saved:
        genes = saved["gene_names"].astype(str)
        control_mean = saved["control_mean"].astype(np.float64)
        directions = saved["directions"].astype(np.float64)
    targets = pd.read_csv(TARGET_TABLE).sort_values("source_order")
    target_order = targets["target_gene"].astype(str).to_numpy()
    source_order = targets["source_order"].astype(int).to_numpy()
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    target_indices = np.asarray([gene_lookup[target] for target in target_order])
    target_lookup = {target: index for index, target in enumerate(target_order)}
    diagonal = directions[target_indices, source_order]

    data = ad.read_h5ad(H1, backed="r")
    labels = data.obs["target_gene"].astype(str).to_numpy()
    selected_rows = np.flatnonzero(np.isin(labels, target_order))
    log_sum = np.zeros(len(target_order), dtype=np.float64)
    n_cells = np.zeros(len(target_order), dtype=np.int64)
    try:
        for start in range(0, len(selected_rows), CHUNK_SIZE):
            rows = selected_rows[start : start + CHUNK_SIZE]
            raw = data.X[rows].tocsr().astype(np.float64)
            totals = np.asarray(raw.sum(axis=1)).ravel()
            chunk_labels = labels[rows]
            for target in np.unique(chunk_labels):
                target_number = target_lookup[target]
                mask = chunk_labels == target
                counts = raw[mask, target_indices[target_number]].toarray().ravel()
                log_sum[target_number] += np.log1p(
                    CELL_TARGET_SUM * counts / totals[mask]
                ).sum()
                n_cells[target_number] += mask.sum()
    finally:
        data.file.close()

    perturbed_mean = log_sum / n_cells
    control_target_mean = control_mean[target_indices]
    delta = perturbed_mean - control_target_mean
    table = pd.DataFrame(
        {
            "target_gene": target_order,
            "source_order": source_order,
            "n_perturbed_cells": n_cells,
            "control_mean_log1cp10k": control_target_mean,
            "perturbed_mean_log1cp10k": perturbed_mean,
            "delta_mean_log1cp10k": delta,
            "covariance_diagonal": diagonal,
            "individual_amplitude_mean_log1cp10k": delta / diagonal,
        }
    )
    pseudobulk = pd.read_csv(SCALE_AUDIT)[["target_gene", "observed_target_shift_10k"]]
    table = table.merge(pseudobulk, on="target_gene", validate="one_to_one")
    table["individual_amplitude_pseudobulk_log1cp10k"] = (
        table["observed_target_shift_10k"] / table["covariance_diagonal"]
    )
    ridge = 0.1 * table["covariance_diagonal"].median()
    table["ridge_0p1_median_covariance"] = ridge
    table["individual_amplitude_mean_log1cp10k_ridge"] = table[
        "delta_mean_log1cp10k"
    ] / (table["covariance_diagonal"] + ridge)
    table["ridge_amplitude_over_unregularized"] = (
        table["individual_amplitude_mean_log1cp10k_ridge"]
        / table["individual_amplitude_mean_log1cp10k"]
    )
    table.to_csv(TABLE, index=False)

    mean_log = table["individual_amplitude_mean_log1cp10k"]
    regularized = table["individual_amplitude_mean_log1cp10k_ridge"]
    pooled = table["individual_amplitude_pseudobulk_log1cp10k"]
    result = {
        "created_utc": datetime.now(UTC).isoformat(),
        "definition": "(perturbed mean per-cell log1CP10K target expression - strict-control mean per-cell log1CP10K target expression) / control covariance diagonal",
        "not_downstream_optimized": True,
        "n_targets": len(table),
        "mean_log1cp10k_amplitude": distribution(mean_log),
        "mean_log1cp10k_positive_sign_failures": int((mean_log >= 0).sum()),
        "ridge": {
            "definition": "delta / (C_gg + 0.1 * median_g(C_gg))",
            "median_covariance_diagonal": float(table["covariance_diagonal"].median()),
            "additive_ridge": float(ridge),
            "amplitude_distribution": distribution(regularized),
            "median_amplitude_ratio_to_unregularized": float(
                table["ridge_amplitude_over_unregularized"].median()
            ),
            "minimum_amplitude_ratio_to_unregularized": float(
                table["ridge_amplitude_over_unregularized"].min()
            ),
        },
        "pseudobulk_log1cp10k_sensitivity": distribution(pooled),
        "pearson_between_definitions": float(mean_log.corr(pooled, method="pearson")),
        "spearman_between_definitions": float(mean_log.corr(pooled, method="spearman")),
        "spearman_absolute_amplitude_vs_covariance_diagonal": float(
            mean_log.abs().corr(table["covariance_diagonal"], method="spearman")
        ),
        "inputs": {
            "h1": {"path": H1.relative_to(ROOT).as_posix(), "sha256": H1_SHA256},
            "model1": {
                "path": MODEL.relative_to(ROOT).as_posix(),
                "sha256": sha256(MODEL),
            },
        },
        "output": TABLE.relative_to(ROOT).as_posix(),
    }
    SUMMARY.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
