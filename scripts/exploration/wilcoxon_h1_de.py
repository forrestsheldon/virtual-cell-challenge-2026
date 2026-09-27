"""Cell-level DE for each H1 CRISPRi target against non-targeting controls.

Raw counts are normalized to 10,000 counts per cell and log1p transformed.
Tests are two-sided Mann-Whitney U (Wilcoxon rank-sum), with BH correction
across all genes separately for each perturbation.
"""

import argparse
import gc
import json
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import numba
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pdex import mwu, pseudobulk, sparse_column_index
from statsmodels.stats.multitest import multipletests


@numba.njit(parallel=True)
def normalize_log1p(data, indptr, target_sum=10_000.0):
    for row in numba.prange(len(indptr) - 1):
        left, right = indptr[row], indptr[row + 1]
        total = data[left:right].sum()
        if total:
            scale = target_sum / total
            for i in range(left, right):
                data[i] = np.log1p(data[i] * scale)


def adjust_pvalues(pvalues):
    pvalues = np.nan_to_num(pvalues, nan=1.0)
    return multipletests(pvalues, method="fdr_bh")[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/external/vcc2025_h1/adata_Training.h5ad"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/crispri-h1-exploration/generated")
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.input, backed="r")
    labels = adata.obs["target_gene"].astype(str).to_numpy()
    targets = sorted(set(labels) - {"non-targeting"})
    if args.limit:
        targets = targets[: args.limit]

    control_rows = np.flatnonzero(labels == "non-targeting")
    control = adata.X[control_rows].tocsr()
    normalize_log1p(control.data, control.indptr)
    control_mean = pseudobulk(control, geometric_mean=True, is_log1p=True)
    control_pct = np.asarray(control.getnnz(axis=0)).ravel() / control.shape[0]
    control_index = sparse_column_index(control)
    del control
    gc.collect()

    output_path = args.output / "wilcoxon_de.parquet"
    writer = None
    summaries = []

    for number, target in enumerate(targets, start=1):
        rows = np.flatnonzero(labels == target)
        matrix = adata.X[rows].tocsr()
        normalize_log1p(matrix.data, matrix.indptr)

        target_mean = pseudobulk(matrix, geometric_mean=True, is_log1p=True)
        target_pct = np.asarray(matrix.getnnz(axis=0)).ravel() / matrix.shape[0]
        result = mwu(matrix, control_index)
        pvalue = np.asarray(result.pvalue).clip(0, 1)
        fdr = adjust_pvalues(pvalue)
        log2fc = np.log2((target_mean + 1e-9) / (control_mean + 1e-9))

        frame = pd.DataFrame(
            {
                "target": target,
                "gene": adata.var_names,
                "target_mean": target_mean,
                "control_mean": control_mean,
                "target_pct": target_pct,
                "control_pct": control_pct,
                "log2_fold_change": log2fc,
                "p_value": pvalue,
                "statistic": result.statistic,
                "fdr": fdr,
            }
        )
        table = pa.Table.from_pandas(frame, preserve_index=False)
        writer = writer or pq.ParquetWriter(
            output_path, table.schema, compression="zstd"
        )
        writer.write_table(table)

        target_gene = adata.var_names.get_loc(target)
        significant = fdr <= 0.05
        summaries.append(
            {
                "target": target,
                "n_cells": len(rows),
                "n_de": int(significant.sum()),
                "n_up": int((significant & (log2fc > 0)).sum()),
                "n_down": int((significant & (log2fc < 0)).sum()),
                "target_log2_fold_change": log2fc[target_gene],
                "target_fdr": fdr[target_gene],
                "target_pct": target_pct[target_gene],
                "control_pct": control_pct[target_gene],
            }
        )
        print(f"{number:3}/{len(targets)} {target:12} {significant.sum():5} DE genes")
        del matrix, frame, table

    writer.close()
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / "wilcoxon_summary.csv", index=False)

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "input": str(args.input),
        "comparison": "each target_gene versus non-targeting cells",
        "normalization": "per-cell total-count normalization to 10000, then log1p",
        "test": "two-sided Mann-Whitney U / Wilcoxon rank-sum",
        "implementation": "pdex 0.3.0",
        "multiple_testing": "Benjamini-Hochberg within each perturbation across 18080 genes",
        "fdr_threshold": 0.05,
        "n_controls": len(control_rows),
        "n_targets": len(targets),
    }
    (args.output / "wilcoxon_run.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
