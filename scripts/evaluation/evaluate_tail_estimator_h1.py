"""Evaluate a blinded bottom-tail estimator on a balanced H1 cell mixture."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scripts.evaluation.crossfit_empirical_scale import (
    BINS,
    DE,
    MIN_DE,
    score_grid,
    truth_and_masks,
)
from scripts.evaluation.scan_empirical_scale import sha256
from scripts.linear_response.kernel import BULK_TARGET_SUM

ROOT = Path(__file__).resolve().parents[2]
H1 = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
REFERENCE = ROOT / "reports/vcc2026-h1/reference_cells.csv"
MODEL = ROOT / "data/derived/linear_response/ladder/empirical.npz"
OUTPUT = ROOT / "reports/tail-estimator-h1"
CONTROL = "non-targeting"
POOL_SIZE = 80_000
TAIL_CELLS = 400
CONTROL_CELLS = 29_600
SEED = 20260917
CHUNK_SIZE = 2_000


def log_bulk(counts: np.ndarray) -> np.ndarray:
    return np.log1p(BULK_TARGET_SUM * counts / counts.sum())


def cpm(counts: np.ndarray) -> np.ndarray:
    return 1_000_000 * counts / counts.sum()


def bottom_k(
    values: np.ndarray, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, float, int]:
    threshold = float(np.partition(values, k - 1)[k - 1])
    below = np.flatnonzero(values < threshold)
    tied = np.flatnonzero(values == threshold)
    chosen = np.r_[below, rng.choice(tied, k - len(below), replace=False)]
    return np.sort(chosen), threshold, len(tied)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(REFERENCE).sort_values(["source_order", "source_row"])
    targets = (
        reference.drop_duplicates("source_order")["target_gene"].astype(str).tolist()
    )
    target_rows = reference["source_row"].to_numpy(dtype=int)

    data = ad.read_h5ad(H1, backed="r")
    try:
        labels = data.obs["target_gene"].astype(str).to_numpy()
        genes = data.var_names.astype(str).tolist()
        lookup = {gene: index for index, gene in enumerate(genes)}
        target_indices = np.asarray([lookup[target] for target in targets])
        control_rows_all = np.flatnonzero(labels == CONTROL)
        rng = np.random.default_rng(SEED)
        control_rows = np.sort(
            rng.choice(control_rows_all, CONTROL_CELLS, replace=False)
        )
        pool_rows = np.sort(np.r_[target_rows, control_rows])
        if len(pool_rows) != POOL_SIZE or len(np.unique(pool_rows)) != POOL_SIZE:
            raise AssertionError("balanced blinded pool is not 80,000 unique cells")
        pool_labels = labels[pool_rows]
        if any(
            np.count_nonzero(pool_labels == target) != TAIL_CELLS for target in targets
        ):
            raise AssertionError(
                "each hidden perturbation must occupy exactly 0.5% of the pool"
            )

        scan_path = OUTPUT / "pool_scan.npz"
        if scan_path.exists():
            with np.load(scan_path) as saved:
                if not np.array_equal(saved["pool_rows"], pool_rows):
                    raise AssertionError("cached pool rows do not match")
                scores = saved["target_expression"].astype(np.float32)
                global_counts = saved["global_counts"].astype(np.float64)
        else:
            scores = np.empty((POOL_SIZE, len(targets)), dtype=np.float32)
            global_counts = np.zeros(data.n_vars, dtype=np.float64)
            for start in range(0, POOL_SIZE, CHUNK_SIZE):
                stop = min(start + CHUNK_SIZE, POOL_SIZE)
                block = data.X[pool_rows[start:stop]].tocsr()
                totals = np.asarray(block.sum(axis=1)).ravel()
                scores[start:stop] = (
                    block[:, target_indices].toarray() / totals[:, None]
                )
                global_counts += np.asarray(block.sum(axis=0)).ravel()
                print(f"pool scan {stop:,}/{POOL_SIZE:,}", flush=True)
            np.savez_compressed(
                scan_path,
                pool_rows=pool_rows,
                target_expression=scores,
                global_counts=global_counts,
            )

        selected_rows = np.empty((len(targets), TAIL_CELLS), dtype=np.int64)
        enrichment_rows = []
        for index, target in enumerate(targets):
            chosen, threshold, ties = bottom_k(
                scores[:, index], TAIL_CELLS, np.random.default_rng(SEED + index + 1)
            )
            selected_rows[index] = pool_rows[chosen]
            selected_labels = pool_labels[chosen]
            recovered = int(np.count_nonzero(selected_labels == target))
            enrichment_rows.append(
                {
                    "target_gene": target,
                    "threshold_count_fraction": threshold,
                    "pool_zero_fraction": float(np.mean(scores[:, index] == 0)),
                    "cells_tied_at_threshold": ties,
                    "true_target_cells_selected": recovered,
                    "precision": recovered / TAIL_CELLS,
                    "recall": recovered / TAIL_CELLS,
                    "enrichment_over_0p5_percent": (recovered / TAIL_CELLS) / 0.005,
                }
            )
        enrichment = pd.DataFrame(enrichment_rows)
        enrichment.to_csv(OUTPUT / "tail_enrichment.csv", index=False)
        np.save(OUTPUT / "selected_source_rows.npy", selected_rows)

        counts_path = OUTPUT / "tail_counts.npz"
        if counts_path.exists():
            with np.load(counts_path) as saved:
                tail_counts = saved["tail_counts"].astype(np.float64)
                control_counts = saved["control_counts"].astype(np.float64)
        else:
            tail_counts = np.empty((len(targets), data.n_vars), dtype=np.float64)
            for index, target in enumerate(targets):
                tail_counts[index] = np.asarray(
                    data.X[np.sort(selected_rows[index])].sum(axis=0)
                ).ravel()
                print(f"tail profile {index + 1}/{len(targets)}: {target}", flush=True)
            control_counts = np.asarray(data.X[control_rows_all].sum(axis=0)).ravel()
            np.savez_compressed(
                counts_path, tail_counts=tail_counts, control_counts=control_counts
            )
    finally:
        data.file.close()

    tail_log_bulk = np.vstack([log_bulk(row) for row in tail_counts])
    global_log_bulk = log_bulk(global_counts)
    artifact = OUTPUT / "expected_profiles.npz"
    np.savez_compressed(
        artifact,
        target_gene=np.asarray(targets),
        gene_names=np.asarray(genes),
        expected_log_bulk=tail_log_bulk,
        null_log_bulk=np.repeat(global_log_bulk[None, :], len(targets), axis=0),
        source_rows=selected_rows,
        pool_rows=pool_rows,
    )

    with np.load(MODEL) as model:
        output_genes = model["output_gene"].astype(str).tolist()
        output_indices = model["full_gene_index"].astype(int)[: len(output_genes)]
    truth, masks = truth_and_masks(targets, output_genes)
    control_cpm = cpm(control_counts)[output_indices]
    tail_lfc = np.log2(
        (np.vstack([cpm(row)[output_indices] for row in tail_counts]) + 1e-9)
        / (control_cpm[None, :] + 1e-9)
    )
    global_lfc = np.log2(
        (cpm(global_counts)[output_indices] + 1e-9) / (control_cpm + 1e-9)
    )
    per_target_rows = []
    for index, target in enumerate(targets):
        row: dict[str, object] = {"target_gene": target}
        for name in BINS:
            row[f"{name}_de_genes"] = int(masks[name][index].sum())
            for arm, prediction in [
                ("tail", tail_lfc[index]),
                ("global", global_lfc),
                ("control", np.zeros(len(output_genes))),
            ]:
                scores_ = score_grid(
                    prediction[None, :], truth[index], masks[name][index]
                )
                for metric, value in scores_.items():
                    row[f"{name}_{arm}_{metric}"] = value[0]
        per_target_rows.append(row)
    per_target = pd.DataFrame(per_target_rows).merge(enrichment, on="target_gene")
    per_target.to_csv(OUTPUT / "de_per_target.csv", index=False)

    summary_rows = []
    for name in BINS:
        keep = per_target[f"{name}_de_genes"] >= MIN_DE
        for arm in ["control", "global", "tail"]:
            summary_rows.append(
                {
                    "de_bin": name,
                    "arm": arm,
                    "eligible_targets": int(keep.sum()),
                    "mean_nmae": per_target.loc[keep, f"{name}_{arm}_nmae"].mean(),
                    "median_nmae": per_target.loc[keep, f"{name}_{arm}_nmae"].median(),
                    "mean_direction_agreement": per_target.loc[
                        keep, f"{name}_{arm}_sign_agreement"
                    ].mean(),
                    "targets_improved_over_global": int(
                        (
                            per_target.loc[keep, f"{name}_{arm}_nmae"]
                            < per_target.loc[keep, f"{name}_global_nmae"]
                        ).sum()
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "de_summary.csv", index=False)
    pd.DataFrame(
        [
            {"statistic": "median_precision", "value": enrichment.precision.median()},
            {"statistic": "mean_precision", "value": enrichment.precision.mean()},
            {
                "statistic": "targets_above_random_precision",
                "value": int((enrichment.precision > 0.005).sum()),
            },
            {
                "statistic": "targets_with_zero_threshold",
                "value": int((enrichment.threshold_count_fraction == 0).sum()),
            },
        ]
    ).to_csv(OUTPUT / "enrichment_summary.csv", index=False)

    manifest = {
        "analysis": "transductive bottom-0.5%-target-expression tail estimator",
        "pool": {
            "cells": POOL_SIZE,
            "cells_per_hidden_perturbation": TAIL_CELLS,
            "control_cells": CONTROL_CELLS,
            "target_prior": TAIL_CELLS / POOL_SIZE,
            "labels_used_by_estimator": False,
            "labels_used_to_construct_balanced_pool_and_evaluate": True,
            "same_cells_used_for_transductive_estimation_and_H1_reference": True,
        },
        "selection": "exactly 400 cells with lowest observed target count fraction; seeded random tie-breaking",
        "null": "pseudobulk profile of the entire 80,000-cell blinded pool",
        "inputs": {
            str(H1.relative_to(ROOT)): sha256(H1),
            str(REFERENCE.relative_to(ROOT)): sha256(REFERENCE),
            str(DE.relative_to(ROOT)): sha256(DE),
        },
        "seed": SEED,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\nEnrichment")
    print(pd.read_csv(OUTPUT / "enrichment_summary.csv").to_string(index=False))
    print("\nDE metrics")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
