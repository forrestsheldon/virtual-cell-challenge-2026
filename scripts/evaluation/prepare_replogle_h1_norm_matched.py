"""Separate support selection from amplitude in Replogle-to-H1 transfer."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scripts.evaluation.prepare_replogle_h1_phase1 import (
    evaluate,
    expected_realized_effects,
    truth_arrays,
)
from scripts.evaluation.prepare_replogle_h1_reliability import (
    dirichlet_lfc,
    expand_effects,
    stable_sign_mask,
)

ROOT = Path(__file__).resolve().parents[2]
BULK = ROOT / "data/external/replogle2022/K562_gwps_raw_bulk_01.h5ad"
AGGREGATION = ROOT / "data/derived/replogle_h1_transfer/effects.npz"
PHASE1 = ROOT / "data/derived/replogle_h1_transfer/phase1_effects.npz"
SPLITS = ROOT / "data/derived/replogle_h1_transfer/source_cell_splits.npz"
DERIVED = ROOT / "data/derived/replogle_h1_transfer/norm_matched_effects.npz"
REPORT = ROOT / "reports/replogle-h1-transfer/norm-matched"
PRIOR_UMIS = 100_000.0

ARMS = (
    "moderate_shrink",
    "continuous_natural",
    "dense_at_continuous_norm",
    "continuous_at_dense_norm",
    "hard_2of3_natural",
    "dense_at_hard_2of3_norm",
    "hard_2of3_at_dense_norm",
)


def continuous_reliability(full: np.ndarray, halves: np.ndarray) -> np.ndarray:
    """Wiener-style weight from effect magnitude and split-half sampling noise."""
    noise_variance = np.mean((halves[:, 0] - halves[:, 1]) ** 2, axis=0) / 4
    signal_energy = np.asarray(full, dtype=np.float64) ** 2
    denominator = signal_energy + noise_variance
    return np.divide(
        signal_energy,
        denominator,
        out=np.zeros_like(signal_energy),
        where=denominator > 0,
    )


def match_row_norm(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Scale each row of values to the corresponding reference L2 norm."""
    value_norm = np.linalg.norm(values, axis=1)
    reference_norm = np.linalg.norm(reference, axis=1)
    scale = np.divide(
        reference_norm,
        value_norm,
        out=np.zeros_like(reference_norm),
        where=value_norm > 0,
    )
    return values * scale[:, None]


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    with np.load(AGGREGATION, allow_pickle=False) as saved:
        targets = saved["target_gene"].astype(str).tolist()
        matched = saved["matched_target_gene"].astype(str).tolist()
        shared_genes = saved["shared_gene"].astype(str).tolist()
        target_counts = saved["replogle_count_sum"].astype(np.int64)
        control_counts = saved["replogle_control_count_sum"].astype(np.int64)
        h1_effect = saved["h1_lfc_native"].astype(np.float64)
    with np.load(SPLITS, allow_pickle=False) as saved:
        split_target_counts = saved["target_count_sum"].astype(np.int64)
        split_target_totals = saved["target_library_size"].astype(np.int64)
        split_control_counts = saved["control_count_sum"].astype(np.int64)
        split_control_totals = saved["control_library_size"].astype(np.int64)

    bulk = ad.read_h5ad(BULK, backed="r")
    try:
        symbols = bulk.var["gene_name"].astype(str).to_numpy()
    finally:
        bulk.file.close()
    unique_symbols = pd.unique(symbols).tolist()
    lookup = {gene: index for index, gene in enumerate(unique_symbols)}
    shared_indices = np.asarray([lookup[gene] for gene in shared_genes])
    target_shared = target_counts[:, shared_indices]
    control_shared = control_counts[shared_indices]

    dense = dirichlet_lfc(
        target_shared,
        target_counts.sum(axis=1),
        control_shared,
        control_counts.sum(),
        PRIOR_UMIS,
    )
    split_effects = np.empty_like(split_target_counts, dtype=np.float64)
    for repeat in range(split_target_counts.shape[0]):
        for half in (0, 1):
            split_effects[repeat, half] = dirichlet_lfc(
                split_target_counts[repeat, half],
                split_target_totals[repeat, half],
                split_control_counts[repeat, half],
                split_control_totals[repeat, half],
                PRIOR_UMIS / 2,
            )

    weights = continuous_reliability(dense, split_effects)
    continuous = dense * weights
    hard = dense * stable_sign_mask(dense, split_effects, minimum_repeats=2)
    effects = np.stack(
        [
            dense,
            continuous,
            match_row_norm(dense, continuous),
            match_row_norm(continuous, dense),
            hard,
            match_row_norm(dense, hard),
            match_row_norm(hard, dense),
        ]
    )
    intended = np.stack(
        [expand_effects(targets, matched, effect) for effect in effects]
    )

    with np.load(PHASE1, allow_pickle=False) as saved:
        output_genes = saved["output_gene"].astype(str).tolist()
    truth, stable, de_valid = truth_arrays(targets, output_genes)
    realized, source_rows, output_shared_indices = expected_realized_effects(
        intended, targets, output_genes, shared_genes, arm_names=ARMS
    )
    per_target, summary = evaluate(
        intended,
        realized,
        targets,
        output_genes,
        shared_genes,
        output_shared_indices,
        set(matched),
        h1_effect,
        truth,
        stable,
        de_valid,
        arm_names=ARMS,
    )
    per_target.to_csv(REPORT / "expected_per_target.csv", index=False)
    summary.to_csv(REPORT / "expected_summary.csv", index=False)

    dense_norm = np.linalg.norm(dense, axis=1)
    diagnostics = {
        "reliability_weight_quantiles": {
            name: float(value)
            for name, value in zip(
                ("min", "q10", "q25", "median", "q75", "q90", "max"),
                np.quantile(weights, (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)),
            )
        },
        "fraction_weight_below": {
            str(threshold): float((weights < threshold).mean())
            for threshold in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "median_norm_ratio_to_moderate_shrink": {
            arm: float(np.median(np.linalg.norm(effect, axis=1) / dense_norm))
            for arm, effect in zip(ARMS, effects)
        },
        "continuous_weight_definition": "w = full_effect^2 / (full_effect^2 + mean(split_half_0 - split_half_1)^2 / 4)",
        "noise_interpretation": "For equal halves, squared half-difference divided by four estimates full-pseudobulk sampling variance. Repeated splits reuse cells and are a reliability diagnostic, not biological replication.",
        "norm_matching": "performed independently for each of the 115 source targets before the 11 missing-target mean fallback",
    }
    (REPORT / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(
        DERIVED,
        arm=np.asarray(ARMS),
        target_gene=np.asarray(targets),
        matched_target_gene=np.asarray(matched),
        shared_gene=np.asarray(shared_genes),
        output_gene=np.asarray(output_genes),
        full_gene_index=output_shared_indices,
        source_rows=source_rows,
        reliability_weight=weights.astype(np.float32),
        matched_effect=effects.astype(np.float32),
        intended_lfc=intended.astype(np.float32),
        expected_realized_lfc=realized.astype(np.float32),
    )
    print(summary.query("population == 'all_126'").to_string(index=False))
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
