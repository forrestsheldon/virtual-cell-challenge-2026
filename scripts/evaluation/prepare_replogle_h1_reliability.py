"""Test source-only shrinkage and cell-split sparsification for K562-to-H1 transfer."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scripts.evaluation.prepare_replogle_h1_phase1 import (
    evaluate,
    expected_realized_effects,
    truth_arrays,
)
from scripts.evaluation.prepare_replogle_h1_transfer import target_from_population

ROOT = Path(__file__).resolve().parents[2]
SINGLE_CELL = ROOT / "data/external/replogle2022/ReplogleWeissman2022_K562_gwps.h5ad"
BULK = ROOT / "data/external/replogle2022/K562_gwps_raw_bulk_01.h5ad"
AGGREGATION = ROOT / "data/derived/replogle_h1_transfer/effects.npz"
PHASE1 = ROOT / "data/derived/replogle_h1_transfer/phase1_effects.npz"
SPLITS = ROOT / "data/derived/replogle_h1_transfer/source_cell_splits.npz"
DERIVED = ROOT / "data/derived/replogle_h1_transfer/reliability_effects.npz"
REPORT = ROOT / "reports/replogle-h1-transfer/reliability"
REPEATS = 3
PRIOR_UMIS = (10_000.0, 100_000.0, 1_000_000.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dirichlet_lfc(
    counts: np.ndarray,
    totals: np.ndarray,
    control_counts: np.ndarray,
    control_total: float,
    prior_umis: float,
) -> np.ndarray:
    """Log ratio after shrinking proportions toward the pooled control profile."""
    counts = np.asarray(counts, dtype=np.float64)
    totals = np.asarray(totals, dtype=np.float64)
    control_probability = np.asarray(control_counts, dtype=np.float64) / control_total
    result = np.zeros_like(counts, dtype=np.float64)
    measured = control_probability > 0
    posterior = (
        counts[..., measured] + prior_umis * control_probability[measured]
    ) / (totals[..., None] + prior_umis)
    result[..., measured] = np.log2(
        posterior / control_probability[measured]
    )
    return result


def stable_sign_mask(full: np.ndarray, halves: np.ndarray, minimum_repeats: int) -> np.ndarray:
    """Keep effects whose two cell halves agree with the full sign repeatedly."""
    full_sign = np.sign(full)[None, None, :, :]
    agreement = np.all(np.sign(halves) == full_sign, axis=1).sum(axis=0)
    return (full != 0) & (agreement >= minimum_repeats)


def expand_effects(
    targets: list[str], matched: list[str], effects: np.ndarray
) -> np.ndarray:
    """Use each matched source effect and the source mean for missing targets."""
    result = np.repeat(effects.mean(axis=0, keepdims=True), len(targets), axis=0)
    lookup = {target: index for index, target in enumerate(matched)}
    for row, target in enumerate(targets):
        if target in lookup:
            result[row] = effects[lookup[target]]
    return result


def split_assignments(
    population_labels: np.ndarray, selected_populations: list[str]
) -> np.ndarray:
    """Make reproducible, population-stratified cell halves."""
    result = np.full((REPEATS, len(population_labels)), -1, dtype=np.int8)
    selected = np.flatnonzero(np.isin(population_labels, selected_populations))
    frame = pd.DataFrame(
        {"population": population_labels[selected], "row": selected}
    )
    groups = frame.groupby("population", sort=True)["row"]
    for population_index, (_, population_rows) in enumerate(groups):
        rows = population_rows.to_numpy(dtype=np.int64)
        for repeat in range(REPEATS):
            rng = np.random.default_rng(
                np.random.SeedSequence([2026, 9, 18, repeat, population_index])
            )
            shuffled = rng.permutation(rows)
            result[repeat, shuffled[::2]] = 0
            result[repeat, shuffled[1::2]] = 1
    return result


def aggregate_cell_splits(
    matched: list[str], shared_genes: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Scan the source cells once and aggregate three construct-stratified splits."""
    bulk = ad.read_h5ad(BULK, backed="r")
    try:
        bulk_populations = bulk.obs_names.astype(str).to_numpy()
        bulk_targets = np.asarray(
            [target_from_population(value) for value in bulk_populations]
        )
        core_populations = bulk_populations[
            bulk.obs["core_control"].to_numpy(dtype=bool)
            & (bulk_targets == "non-targeting")
        ].tolist()
        target_populations = bulk_populations[np.isin(bulk_targets, matched)].tolist()
        bulk_symbols = bulk.var["gene_name"].astype(str).to_numpy()
    finally:
        bulk.file.close()

    shared_columns = {
        gene: np.flatnonzero(bulk_symbols == gene) for gene in shared_genes
    }
    first_columns = np.asarray([columns[0] for columns in shared_columns.values()])
    duplicate_columns = {
        index: columns[1:]
        for index, columns in enumerate(shared_columns.values())
        if len(columns) > 1
    }
    target_lookup = {target: index for index, target in enumerate(matched)}

    cells = ad.read_h5ad(SINGLE_CELL, backed="r")
    try:
        populations = cells.obs["gene_transcript"].astype(str).to_numpy()
        selected_populations = core_populations + target_populations
        halves = split_assignments(populations, selected_populations)
        population_group = {population: -1 for population in core_populations}
        population_group.update(
            {
                population: target_lookup[target_from_population(population)]
                for population in target_populations
            }
        )
        group = (
            pd.Series(populations, copy=False)
            .map(population_group)
            .fillna(-2)
            .to_numpy(dtype=np.int16)
        )
        if np.any((halves[0] >= 0) != (group >= -1)):
            raise ValueError("cell-split assignment and selected populations differ")

        target_counts = np.zeros(
            (REPEATS, 2, len(matched), len(shared_genes)), dtype=np.int64
        )
        target_totals = np.zeros((REPEATS, 2, len(matched)), dtype=np.int64)
        control_counts = np.zeros((REPEATS, 2, len(shared_genes)), dtype=np.int64)
        control_totals = np.zeros((REPEATS, 2), dtype=np.int64)
        chunk_size = cells.X.chunks[0]
        for start in range(0, cells.n_obs, chunk_size):
            stop = min(start + chunk_size, cells.n_obs)
            selected = group[start:stop] >= -1
            if not selected.any():
                continue
            raw = np.asarray(cells.X[start:stop])[selected]
            values = np.rint(raw[:, first_columns]).astype(np.int64)
            for output_column, extra_columns in duplicate_columns.items():
                values[:, output_column] += np.rint(
                    raw[:, extra_columns].sum(axis=1)
                ).astype(np.int64)
            totals = np.rint(raw.sum(axis=1)).astype(np.int64)
            groups = group[start:stop][selected]
            for repeat in range(REPEATS):
                split = halves[repeat, start:stop][selected]
                for half in (0, 1):
                    control = (groups == -1) & (split == half)
                    control_counts[repeat, half] += values[control].sum(axis=0)
                    control_totals[repeat, half] += totals[control].sum()
                    for target_index in np.unique(groups[(groups >= 0) & (split == half)]):
                        chosen = (groups == target_index) & (split == half)
                        target_counts[repeat, half, target_index] += values[chosen].sum(
                            axis=0
                        )
                        target_totals[repeat, half, target_index] += totals[chosen].sum()
            if start % (20 * chunk_size) == 0:
                print(f"Source split scan: {stop:,}/{cells.n_obs:,} cells", flush=True)
    finally:
        cells.file.close()

    audit = {
        "repeats": REPEATS,
        "split_unit": "cells, stratified independently within each author target-transcript dual-guide population",
        "core_control_populations": len(core_populations),
        "target_populations": len(target_populations),
        "selected_control_cells": int((group == -1).sum()),
        "selected_target_cells": int((group >= 0).sum()),
    }
    return target_counts, target_totals, control_counts, control_totals, audit


def load_or_create_splits(
    matched: list[str], shared_genes: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    if not SPLITS.exists():
        SPLITS.parent.mkdir(parents=True, exist_ok=True)
        values = aggregate_cell_splits(matched, shared_genes)
        target_counts, target_totals, control_counts, control_totals, audit = values
        np.savez_compressed(
            SPLITS,
            matched_target_gene=np.asarray(matched),
            shared_gene=np.asarray(shared_genes),
            target_count_sum=target_counts,
            target_library_size=target_totals,
            control_count_sum=control_counts,
            control_library_size=control_totals,
            audit_json=np.asarray(json.dumps(audit, sort_keys=True)),
        )
    with np.load(SPLITS, allow_pickle=False) as saved:
        if saved["matched_target_gene"].astype(str).tolist() != matched:
            raise ValueError("cached split target axis differs")
        if saved["shared_gene"].astype(str).tolist() != shared_genes:
            raise ValueError("cached split gene axis differs")
        return (
            saved["target_count_sum"],
            saved["target_library_size"],
            saved["control_count_sum"],
            saved["control_library_size"],
            json.loads(str(saved["audit_json"])),
        )


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    with np.load(AGGREGATION, allow_pickle=False) as saved:
        targets = saved["target_gene"].astype(str).tolist()
        matched = saved["matched_target_gene"].astype(str).tolist()
        shared_genes = saved["shared_gene"].astype(str).tolist()
        target_counts_full = saved["replogle_count_sum"].astype(np.int64)
        control_counts_full = saved["replogle_control_count_sum"].astype(np.int64)
        h1_effect = saved["h1_lfc_native"].astype(np.float64)

    bulk = ad.read_h5ad(BULK, backed="r")
    try:
        symbols = bulk.var["gene_name"].astype(str).to_numpy()
    finally:
        bulk.file.close()
    unique_symbols = pd.unique(symbols).tolist()
    unique_lookup = {gene: index for index, gene in enumerate(unique_symbols)}
    shared_indices = np.asarray([unique_lookup[gene] for gene in shared_genes])
    full_target_shared = target_counts_full[:, shared_indices]
    full_control_shared = control_counts_full[shared_indices]
    full_target_totals = target_counts_full.sum(axis=1)
    full_control_total = int(control_counts_full.sum())

    split = load_or_create_splits(matched, shared_genes)
    target_counts, target_totals, control_counts, control_totals, split_audit = split
    reconstructed_target = target_counts.sum(axis=1)
    reconstructed_control = control_counts.sum(axis=1)
    max_target_difference = int(
        np.abs(reconstructed_target - full_target_shared[None, :, :]).max()
    )
    max_control_difference = int(
        np.abs(reconstructed_control - full_control_shared[None, :]).max()
    )
    if max_target_difference or max_control_difference:
        raise ValueError(
            "cell halves do not reproduce author count sums: "
            f"target={max_target_difference}, control={max_control_difference}"
        )

    arm_names = ["dense_unshrunken"]
    with np.load(AGGREGATION, allow_pickle=False) as saved:
        dense_effect = saved["replogle_lfc_native"].astype(np.float64)
    intended = [expand_effects(targets, matched, dense_effect)]
    support_rows = []
    for prior in PRIOR_UMIS:
        full = dirichlet_lfc(
            full_target_shared,
            full_target_totals,
            full_control_shared,
            full_control_total,
            prior,
        )
        half_effects = np.empty_like(target_counts, dtype=np.float64)
        for repeat in range(REPEATS):
            for half in (0, 1):
                half_effects[repeat, half] = dirichlet_lfc(
                    target_counts[repeat, half],
                    target_totals[repeat, half],
                    control_counts[repeat, half],
                    control_totals[repeat, half],
                    prior / 2,
                )
        label = f"shrink_{int(prior):d}"
        arm_names.append(label)
        intended.append(expand_effects(targets, matched, full))
        support_rows.append(
            {
                "arm": label,
                "prior_umis": prior,
                "minimum_stable_repeats": 0,
                "fraction_effects_retained": 1.0,
                "median_effects_retained_per_target": len(shared_genes),
            }
        )
        for minimum in (2, 3):
            mask = stable_sign_mask(full, half_effects, minimum)
            sparse_effect = full * mask
            sparse_label = f"{label}_stable_{minimum}of{REPEATS}"
            arm_names.append(sparse_label)
            intended.append(expand_effects(targets, matched, sparse_effect))
            support_rows.append(
                {
                    "arm": sparse_label,
                    "prior_umis": prior,
                    "minimum_stable_repeats": minimum,
                    "fraction_effects_retained": float(mask.mean()),
                    "median_effects_retained_per_target": float(
                        np.median(mask.sum(axis=1))
                    ),
                }
            )
    intended = np.asarray(intended, dtype=np.float64)

    with np.load(PHASE1, allow_pickle=False) as saved:
        output_genes = saved["output_gene"].astype(str).tolist()
    truth, stable, de_valid = truth_arrays(targets, output_genes)
    realized, source_rows, output_shared_indices = expected_realized_effects(
        intended, targets, output_genes, shared_genes, arm_names=arm_names
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
        arm_names=arm_names,
    )
    per_target.to_csv(REPORT / "expected_per_target.csv", index=False)
    summary.to_csv(REPORT / "expected_summary.csv", index=False)
    pd.DataFrame(support_rows).to_csv(REPORT / "support_summary.csv", index=False)
    np.savez_compressed(
        DERIVED,
        arm=np.asarray(arm_names),
        target_gene=np.asarray(targets),
        matched_target_gene=np.asarray(matched),
        shared_gene=np.asarray(shared_genes),
        output_gene=np.asarray(output_genes),
        full_gene_index=output_shared_indices,
        source_rows=source_rows,
        intended_lfc=intended.astype(np.float32),
        expected_realized_lfc=realized.astype(np.float32),
    )
    audit = {
        **split_audit,
        "max_abs_target_count_difference_after_recombining_halves": max_target_difference,
        "max_abs_control_count_difference_after_recombining_halves": max_control_difference,
        "shrinkage": "Dirichlet posterior proportions centered on the pooled source control profile; prior_umis is the prior concentration in equivalent UMIs",
        "stability": "retain a target-gene effect when both cell halves have the same sign as the full effect in at least the stated number of three independently stratified splits",
        "inference_warning": "cell-split stability measures source sampling reliability, not biological replication or DE significance",
    }
    (REPORT / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "stage": "source-only shrinkage and cell-split sparsification",
        "inputs": {
            str(SINGLE_CELL.relative_to(ROOT)): sha256(SINGLE_CELL),
            str(BULK.relative_to(ROOT)): sha256(BULK),
            str(AGGREGATION.relative_to(ROOT)): sha256(AGGREGATION),
        },
        "outputs": {
            str(SPLITS.relative_to(ROOT)): sha256(SPLITS),
            str(DERIVED.relative_to(ROOT)): sha256(DERIVED),
            **{
                str(path.relative_to(ROOT)): sha256(path)
                for path in sorted(REPORT.glob("*.csv"))
            },
            str((REPORT / "audit.json").relative_to(ROOT)): sha256(
                REPORT / "audit.json"
            ),
        },
    }
    (REPORT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(summary.query("population == 'all_126'").to_string(index=False))


if __name__ == "__main__":
    main()
