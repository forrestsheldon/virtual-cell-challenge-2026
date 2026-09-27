"""Evaluate target-expression tails selected from H1 control cells alone."""

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
from scripts.evaluation.evaluate_tail_estimator_h1 import bottom_k, cpm, log_bulk
from scripts.evaluation.scan_empirical_scale import sha256

ROOT = Path(__file__).resolve().parents[2]
H1 = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
REFERENCE = ROOT / "reports/vcc2026-h1/reference_cells.csv"
MODEL = ROOT / "data/derived/linear_response/ladder/empirical.npz"
OUTPUT = ROOT / "reports/control-tail-estimator-h1"
CONTROL = "non-targeting"
TAIL_FRACTION = 0.005
MIN_TAIL_CELLS = 400
SEED = 20260917
CHUNK_SIZE = 2_000


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(REFERENCE).sort_values(["source_order", "source_row"])
    targets = reference.drop_duplicates("source_order")["target_gene"].astype(str).tolist()
    data = ad.read_h5ad(H1, backed="r")
    try:
        labels = data.obs["target_gene"].astype(str).to_numpy()
        genes = data.var_names.astype(str).tolist()
        lookup = {gene: index for index, gene in enumerate(genes)}
        target_indices = np.asarray([lookup[target] for target in targets])
        control_rows = np.flatnonzero(labels == CONTROL)
        tail_cells = max(int(np.ceil(TAIL_FRACTION * len(control_rows))), MIN_TAIL_CELLS)

        scan_path = OUTPUT / "control_scan.npz"
        if scan_path.exists():
            with np.load(scan_path) as saved:
                if not np.array_equal(saved["control_rows"], control_rows):
                    raise AssertionError("cached control rows do not match")
                scores = saved["target_expression"].astype(np.float32)
                control_counts = saved["control_counts"].astype(np.float64)
        else:
            scores = np.empty((len(control_rows), len(targets)), dtype=np.float32)
            control_counts = np.zeros(data.n_vars, dtype=np.float64)
            for start in range(0, len(control_rows), CHUNK_SIZE):
                stop = min(start + CHUNK_SIZE, len(control_rows))
                block = data.X[control_rows[start:stop]].tocsr()
                totals = np.asarray(block.sum(axis=1)).ravel()
                scores[start:stop] = block[:, target_indices].toarray() / totals[:, None]
                control_counts += np.asarray(block.sum(axis=0)).ravel()
                print(f"control scan {stop:,}/{len(control_rows):,}", flush=True)
            np.savez_compressed(
                scan_path,
                control_rows=control_rows,
                target_expression=scores,
                control_counts=control_counts,
            )

        selected_rows = np.empty((len(targets), tail_cells), dtype=np.int64)
        selection_rows = []
        for index, target in enumerate(targets):
            chosen, threshold, ties = bottom_k(
                scores[:, index], tail_cells, np.random.default_rng(SEED + index + 1)
            )
            selected_rows[index] = control_rows[chosen]
            selection_rows.append(
                {
                    "target_gene": target,
                    "tail_cells": tail_cells,
                    "effective_tail_fraction": tail_cells / len(control_rows),
                    "threshold_count_fraction": threshold,
                    "control_zero_fraction": float(np.mean(scores[:, index] == 0)),
                    "cells_tied_at_threshold": ties,
                }
            )
        selection = pd.DataFrame(selection_rows)
        selection.to_csv(OUTPUT / "tail_selection.csv", index=False)
        np.save(OUTPUT / "selected_source_rows.npy", selected_rows)

        counts_path = OUTPUT / "tail_counts.npz"
        if counts_path.exists():
            with np.load(counts_path) as saved:
                tail_counts = saved["tail_counts"].astype(np.float64)
        else:
            tail_counts = np.empty((len(targets), data.n_vars), dtype=np.float64)
            for index, target in enumerate(targets):
                tail_counts[index] = np.asarray(
                    data.X[np.sort(selected_rows[index])].sum(axis=0)
                ).ravel()
                print(f"control tail {index + 1}/{len(targets)}: {target}", flush=True)
            np.savez_compressed(counts_path, tail_counts=tail_counts)
    finally:
        data.file.close()

    control_log_bulk = log_bulk(control_counts)
    tail_log_bulk = np.vstack([log_bulk(row) for row in tail_counts])
    artifact = OUTPUT / "expected_profiles.npz"
    np.savez_compressed(
        artifact,
        target_gene=np.asarray(targets),
        gene_names=np.asarray(genes),
        expected_log_bulk=tail_log_bulk,
        null_log_bulk=np.repeat(control_log_bulk[None, :], len(targets), axis=0),
        source_rows=selected_rows,
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
    rows = []
    for index, target in enumerate(targets):
        row: dict[str, object] = {"target_gene": target}
        for name in BINS:
            row[f"{name}_de_genes"] = int(masks[name][index].sum())
            for arm, prediction in [
                ("tail", tail_lfc[index]),
                ("control", np.zeros(len(output_genes))),
            ]:
                values = score_grid(prediction[None, :], truth[index], masks[name][index])
                for metric, value in values.items():
                    row[f"{name}_{arm}_{metric}"] = value[0]
        rows.append(row)
    per_target = pd.DataFrame(rows)
    per_target.to_csv(OUTPUT / "de_per_target.csv", index=False)

    summary_rows = []
    for name in BINS:
        keep = per_target[f"{name}_de_genes"] >= MIN_DE
        for arm in ["control", "tail"]:
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
                    "targets_improved_over_control": int(
                        (per_target.loc[keep, f"{name}_{arm}_nmae"] < 1).sum()
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "de_summary.csv", index=False)
    manifest = {
        "analysis": "bottom target-expression tails selected from H1 controls only",
        "control_cells": len(control_rows),
        "requested_tail_fraction": TAIL_FRACTION,
        "minimum_tail_cells": MIN_TAIL_CELLS,
        "selected_cells_per_target": tail_cells,
        "effective_tail_fraction": tail_cells / len(control_rows),
        "selection": "observed target count fraction with seeded random tie-breaking",
        "inputs": {
            str(H1.relative_to(ROOT)): sha256(H1),
            str(REFERENCE.relative_to(ROOT)): sha256(REFERENCE),
            str(DE.relative_to(ROOT)): sha256(DE),
        },
        "seed": SEED,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
