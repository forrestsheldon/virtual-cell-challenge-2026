"""Construct and audit count-derived Replogle-to-H1 pseudobulk effects."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPLOGLE = ROOT / "data/external/replogle2022/K562_gwps_raw_bulk_01.h5ad"
H1 = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
REFERENCE_CELLS = ROOT / "reports/vcc2026-h1/reference_cells.csv"
DERIVED = ROOT / "data/derived/replogle_h1_transfer/effects.npz"
REPORT = ROOT / "reports/replogle-h1-transfer"
CONTROL = "non-targeting"
TARGET_SUM = 1_000_000.0
EPSILON = 1e-9
ROW_CHUNK = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_integer_counts(
    raw_mean: np.ndarray, num_cells: np.ndarray
) -> tuple[np.ndarray, float]:
    """Invert author means and report the largest distance from an integer."""
    scaled = np.asarray(raw_mean, dtype=np.float64) * np.asarray(
        num_cells, dtype=np.float64
    )[:, None]
    rounded = np.rint(scaled)
    return rounded.astype(np.int64), float(np.abs(scaled - rounded).max(initial=0))


def collapse_duplicate_symbols(
    matrix: np.ndarray, symbols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Sum columns with the same symbol, retaining first-appearance order."""
    codes, unique = pd.factorize(np.asarray(symbols, dtype=str), sort=False)
    collapsed = np.zeros((len(matrix), len(unique)), dtype=np.int64)
    for column, code in enumerate(codes):
        collapsed[:, code] += matrix[:, column]
    return collapsed, unique.astype(str)


def loo_components(effects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return equal-target leave-one-out global effects and residuals."""
    effects = np.asarray(effects, dtype=np.float64)
    if len(effects) < 2:
        raise ValueError("leave-one-out effects require at least two targets")
    global_loo = (effects.sum(axis=0) - effects) / (len(effects) - 1)
    return global_loo, effects - global_loo


def expand_target_rows(
    source_targets: list[str], source_values: np.ndarray, output_targets: list[str]
) -> np.ndarray:
    """Expand target rows in requested order, using zero for absent targets."""
    lookup = {target: index for index, target in enumerate(source_targets)}
    result = np.zeros((len(output_targets), source_values.shape[1]), dtype=source_values.dtype)
    for row, target in enumerate(output_targets):
        if target in lookup:
            result[row] = source_values[lookup[target]]
    return result


def effect(counts: np.ndarray, control: np.ndarray, columns=None) -> np.ndarray:
    """Library-normalize count sums within the selected gene universe, then log-ratio."""
    selected = np.arange(counts.shape[1]) if columns is None else np.asarray(columns)
    target = counts[:, selected].astype(np.float64)
    baseline = control[selected].astype(np.float64)
    target *= TARGET_SUM / target.sum(axis=1, keepdims=True)
    baseline *= TARGET_SUM / baseline.sum()
    return np.log2((target + EPSILON) / (baseline[None, :] + EPSILON))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else np.nan


def target_from_population(value: str) -> str:
    parts = value.split("_", 3)
    if len(parts) != 4:
        raise ValueError(f"unexpected Replogle population name: {value}")
    return parts[1]


def reconstruct_replogle_rows(
    data: ad.AnnData, finite_rows: np.ndarray, retained_rows: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Audit every finite row while retaining only controls and matched guides."""
    retained_lookup = {int(row): index for index, row in enumerate(retained_rows)}
    retained = np.empty((len(retained_rows), data.n_vars), dtype=np.int64)
    max_residual = 0.0
    total_counts = 0
    library_sizes = np.empty(data.n_obs, dtype=np.float64)
    library_sizes.fill(np.nan)
    n_cells = data.obs["num_cells_filtered"].to_numpy(dtype=float)
    for start in range(0, len(finite_rows), ROW_CHUNK):
        rows = finite_rows[start : start + ROW_CHUNK]
        counts, residual = reconstruct_integer_counts(data.X[rows], n_cells[rows])
        max_residual = max(max_residual, residual)
        total_counts += int(counts.sum())
        library_sizes[rows] = counts.sum(axis=1)
        for local, source_row in enumerate(rows):
            if int(source_row) in retained_lookup:
                retained[retained_lookup[int(source_row)]] = counts[local]
    return retained, {
        "max_distance_from_integer": max_residual,
        "total_reconstructed_umi": total_counts,
        "library_sizes": library_sizes,
    }


def aggregate_h1_counts(
    data: ad.AnnData, reference: pd.DataFrame, targets: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    target_counts = np.empty((len(targets), data.n_vars), dtype=np.int64)
    for index, target in enumerate(targets):
        rows = np.sort(
            reference.loc[reference.target_gene == target, "source_row"].to_numpy(dtype=int)
        )
        target_counts[index] = np.asarray(
            data.X[rows].astype(np.float64).sum(axis=0)
        ).ravel().astype(np.int64)
    labels = data.obs["target_gene"].astype(str).to_numpy()
    control_rows = np.flatnonzero(labels == CONTROL)
    control_counts = np.zeros(data.n_vars, dtype=np.int64)
    for start in range(0, len(control_rows), 4096):
        rows = control_rows[start : start + 4096]
        control_counts += np.asarray(
            data.X[rows].astype(np.float64).sum(axis=0)
        ).ravel().astype(np.int64)
    return target_counts, control_counts


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        name: float(value)
        for name, value in zip(
            ("min", "q25", "median", "q75", "max"),
            np.quantile(values, (0, 0.25, 0.5, 0.75, 1)),
        )
    }


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    DERIVED.parent.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(REFERENCE_CELLS)
    targets = reference.drop_duplicates("target_gene")["target_gene"].astype(str).tolist()

    replogle = ad.read_h5ad(REPLOGLE, backed="r")
    try:
        populations = replogle.obs_names.astype(str).tolist()
        row_targets = np.asarray([target_from_population(value) for value in populations])
        n_cells = replogle.obs["num_cells_filtered"].to_numpy(dtype=float)
        finite_rows = np.flatnonzero(np.isfinite(n_cells))
        core = replogle.obs["core_control"].to_numpy(dtype=bool)
        is_control = row_targets == CONTROL
        core_control_rows = np.flatnonzero(is_control & core & np.isfinite(n_cells))
        noncore_control_rows = np.flatnonzero(is_control & ~core)
        matched_targets = [target for target in targets if target in set(row_targets)]
        matched_guide_rows = np.flatnonzero(np.isin(row_targets, matched_targets))
        retained_rows = np.concatenate([core_control_rows, matched_guide_rows])
        retained_counts, reconstruction = reconstruct_replogle_rows(
            replogle, finite_rows, retained_rows
        )
        symbols = replogle.var["gene_name"].astype(str).tolist()
    finally:
        replogle.file.close()

    collapsed, k_genes = collapse_duplicate_symbols(retained_counts, symbols)
    n_controls = len(core_control_rows)
    control_guide_counts = collapsed[:n_controls]
    guide_counts = collapsed[n_controls:]
    guide_targets = row_targets[matched_guide_rows]
    k_control = control_guide_counts.sum(axis=0)
    k_target_counts = np.vstack(
        [guide_counts[guide_targets == target].sum(axis=0) for target in matched_targets]
    )

    h1 = ad.read_h5ad(H1, backed="r")
    try:
        h_genes = h1.var_names.astype(str).to_numpy()
        h_target_counts, h_control = aggregate_h1_counts(h1, reference, targets)
    finally:
        h1.file.close()

    k_gene_set = set(k_genes)
    common = [gene for gene in h_genes if gene in k_gene_set]
    k_lookup = {gene: index for index, gene in enumerate(k_genes)}
    h_lookup = {gene: index for index, gene in enumerate(h_genes)}
    k_common = np.asarray([k_lookup[gene] for gene in common])
    h_common = np.asarray([h_lookup[gene] for gene in common])

    k_native = effect(k_target_counts, k_control)[:, k_common]
    k_shared = effect(k_target_counts, k_control, k_common)
    h_native = effect(h_target_counts, h_control)[:, h_common]
    h_shared = effect(h_target_counts, h_control, h_common)
    k_native_expanded = expand_target_rows(matched_targets, k_native, targets)
    k_shared_expanded = expand_target_rows(matched_targets, k_shared, targets)

    sensitivity_rows = []
    common_lookup = {gene: index for index, gene in enumerate(common)}
    matched_lookup = {target: index for index, target in enumerate(matched_targets)}
    target_lookup = {target: index for index, target in enumerate(targets)}
    for target in targets:
        excluded = np.ones(len(common), dtype=bool)
        if target in common_lookup:
            excluded[common_lookup[target]] = False
        h_index = target_lookup[target]
        sensitivity_rows.append(
            {
                "dataset": "H1",
                "target_gene": target,
                "target_excluded_cosine": cosine(
                    h_native[h_index, excluded], h_shared[h_index, excluded]
                ),
                "median_abs_lfc_change": float(
                    np.median(np.abs(h_native[h_index] - h_shared[h_index]))
                ),
            }
        )
        if target in matched_lookup:
            k_index = matched_lookup[target]
            sensitivity_rows.append(
                {
                    "dataset": "K562",
                    "target_gene": target,
                    "target_excluded_cosine": cosine(
                        k_native[k_index, excluded], k_shared[k_index, excluded]
                    ),
                    "median_abs_lfc_change": float(
                        np.median(np.abs(k_native[k_index] - k_shared[k_index]))
                    ),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(REPORT / "normalization_sensitivity.csv", index=False)

    guide_native = effect(guide_counts, k_control)[:, k_common]
    concordance_rows = []
    for target in matched_targets:
        rows = np.flatnonzero(guide_targets == target)
        if len(rows) < 2:
            continue
        excluded = np.ones(len(common), dtype=bool)
        if target in common_lookup:
            excluded[common_lookup[target]] = False
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                a, b = guide_native[rows[left], excluded], guide_native[rows[right], excluded]
                concordance_rows.append(
                    {
                        "target_gene": target,
                        "guide_pair_population_a": populations[
                            matched_guide_rows[rows[left]]
                        ],
                        "guide_pair_population_b": populations[
                            matched_guide_rows[rows[right]]
                        ],
                        "target_excluded_cosine": cosine(a, b),
                        "norm_ratio_a_over_b": float(np.linalg.norm(a) / np.linalg.norm(b)),
                        "cells_a": int(n_cells[matched_guide_rows[rows[left]]]),
                        "cells_b": int(n_cells[matched_guide_rows[rows[right]]]),
                        "on_target_log2fc_a": (
                            guide_native[rows[left], common_lookup[target]]
                            if target in common_lookup
                            else np.nan
                        ),
                        "on_target_log2fc_b": (
                            guide_native[rows[right], common_lookup[target]]
                            if target in common_lookup
                            else np.nan
                        ),
                    }
                )
    concordance = pd.DataFrame(concordance_rows)
    concordance.to_csv(REPORT / "guide_concordance.csv", index=False)

    target_rows = []
    row_libraries = reconstruction.pop("library_sizes")
    for target in targets:
        rows = np.flatnonzero(row_targets == target)
        finite = rows[np.isfinite(n_cells[rows])]
        matched = target in matched_lookup
        on_target = np.nan
        if matched and target in common_lookup:
            on_target = k_native[matched_lookup[target], common_lookup[target]]
        target_rows.append(
            {
                "target_gene": target,
                "replogle_overlap": matched,
                "n_guide_pair_populations": len(finite),
                "n_cells": int(n_cells[finite].sum()) if len(finite) else 0,
                "pooled_library_size": int(row_libraries[finite].sum()) if len(finite) else 0,
                "on_target_log2fc": on_target,
            }
        )
    target_table = pd.DataFrame(target_rows)
    target_table.to_csv(REPORT / "target_overlap_and_knockdown.csv", index=False)

    core_libraries = row_libraries[core_control_rows]
    matched_libraries = row_libraries[matched_guide_rows]
    library_table = pd.DataFrame(
        {
            "population_type": np.repeat(
                ["core_control", "matched_target_guide"],
                [len(core_libraries), len(matched_libraries)],
            ),
            "library_size": np.concatenate([core_libraries, matched_libraries]),
            "num_cells_filtered": np.concatenate(
                [n_cells[core_control_rows], n_cells[matched_guide_rows]]
            ).astype(int),
        }
    )
    library_table["umi_per_cell"] = (
        library_table.library_size / library_table.num_cells_filtered
    )
    library_table.to_csv(REPORT / "library_sizes.csv", index=False)

    measurable_pairs = concordance[
        concordance.on_target_log2fc_a.notna()
        & concordance.on_target_log2fc_b.notna()
    ]
    available_on_target = target_table.on_target_log2fc.notna()
    summary = {
        "reconstruction": {
            **reconstruction,
            "finite_populations": len(finite_rows),
            "represented_cells": int(n_cells[finite_rows].sum()),
        },
        "controls": {
            "all_non_targeting_rows": int(is_control.sum()),
            "core_rows_used": len(core_control_rows),
            "core_cells_used": int(n_cells[core_control_rows].sum()),
            "noncore_rows_excluded": len(noncore_control_rows),
            "noncore_rows_without_filtered_counts": int(
                np.isnan(n_cells[noncore_control_rows]).sum()
            ),
        },
        "matching": {
            "h1_targets": len(targets),
            "shared_targets": len(matched_targets),
            "shared_target_populations": len(matched_guide_rows),
            "shared_target_cells": int(n_cells[matched_guide_rows].sum()),
            "missing_targets": [target for target in targets if target not in matched_targets],
            "replogle_symbol_rows": len(symbols),
            "replogle_unique_symbols": len(k_genes),
            "duplicate_symbols_summed": sorted(
                pd.Series(symbols)[pd.Series(symbols).duplicated(keep=False)].unique().tolist()
            ),
            "h1_genes": len(h_genes),
            "shared_genes": len(common),
        },
        "libraries": {
            group: {
                "total": quantiles(frame.library_size.to_numpy()),
                "per_cell": quantiles(frame.umi_per_cell.to_numpy()),
            }
            for group, frame in library_table.groupby("population_type", sort=False)
        },
        "on_target_knockdown": {
            "available_targets": int(available_on_target.sum()),
            "unavailable_target_symbols": target_table.loc[
                target_table.replogle_overlap & ~available_on_target, "target_gene"
            ].tolist(),
            **quantiles(target_table.loc[available_on_target, "on_target_log2fc"]),
        },
        "guide_pair_population_concordance": {
            "unit": "author target-transcript pseudobulk; each is one dual-guide construct, not an individual guide",
            "multi_population_targets": int(concordance.target_gene.nunique()),
            "pairs": len(concordance),
            "cosine": quantiles(concordance.target_excluded_cosine),
            "on_target_measurable_pairs": len(measurable_pairs),
            "on_target_same_sign_pairs": int(
                (
                    np.sign(measurable_pairs.on_target_log2fc_a)
                    == np.sign(measurable_pairs.on_target_log2fc_b)
                ).sum()
            ),
        },
        "normalization_sensitivity": {
            dataset: {
                "cosine": quantiles(frame.target_excluded_cosine),
                "median_abs_lfc_change": quantiles(frame.median_abs_lfc_change),
            }
            for dataset, frame in sensitivity.groupby("dataset", sort=False)
        },
    }
    (REPORT / "checkpoint1_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    np.savez_compressed(
        DERIVED,
        target_gene=np.asarray(targets),
        matched_target_gene=np.asarray(matched_targets),
        shared_gene=np.asarray(common),
        replogle_count_sum=k_target_counts,
        replogle_control_count_sum=k_control,
        replogle_lfc_native=k_native,
        replogle_lfc_shared=k_shared,
        replogle_lfc_native_126=k_native_expanded,
        replogle_lfc_shared_126=k_shared_expanded,
        h1_count_sum=h_target_counts,
        h1_control_count_sum=h_control,
        h1_lfc_native=h_native,
        h1_lfc_shared=h_shared,
    )
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "stage": "checkpoint 1 aggregation audit",
        "producer": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "normalization": {
            "primary": "library-size normalize each count-sum pseudobulk over its native measured gene universe to 1e6, then log2 ratio with epsilon 1e-9; align symbols afterward",
            "sensitivity": "restrict to 7,583 shared symbols before library-size normalization",
            "target_weighting_for_future_global_means": "equal target weight",
        },
        "replogle_population_unit": "The author pseudobulk rows are target-transcript populations. Cell-level metadata shows each population corresponds to one dual-guide construct; therefore this audit compares construct populations, not individual component guides.",
        "inputs": {
            "replogle_author_raw_bulk": {
                "path": str(REPLOGLE.relative_to(ROOT)),
                "sha256": sha256(REPLOGLE),
                "source_url": "https://figshare.com/ndownloader/files/35774443",
                "accessed": "2026-09-18",
            },
            "h1_training_counts": {
                "path": str(H1.relative_to(ROOT)),
                "sha256": sha256(H1),
                "source_url": "gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/",
                "accessed": "2026-08-30",
            },
            "h1_reference_cells": {
                "path": str(REFERENCE_CELLS.relative_to(ROOT)),
                "sha256": sha256(REFERENCE_CELLS),
            },
        },
        "outputs": {
            str(DERIVED.relative_to(ROOT)): sha256(DERIVED),
            **{
                str(path.relative_to(ROOT)): sha256(path)
                for path in sorted(REPORT.glob("*.csv"))
            },
            str((REPORT / "checkpoint1_summary.json").relative_to(ROOT)): sha256(
                REPORT / "checkpoint1_summary.json"
            ),
        },
        "software": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("anndata", "numpy", "pandas")},
        },
    }
    (REPORT / "checkpoint1_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
