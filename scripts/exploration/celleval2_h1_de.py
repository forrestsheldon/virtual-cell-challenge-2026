"""Cell-level DE for each H1 CRISPRi target in the cell-eval2 scoring universe.

Mirrors ArcInstitute/cell-eval2 0.16.0 under configs/vcc2026.yaml: raw counts
are normalized per cell to 1e6 (CPM), the gene universe is the genes whose mean
CPM in the *reference* group exceeds 5 (one gene set for every target, #351),
the test is a two-sided Mann-Whitney U on cells, log2 fold change comes from
arithmetic CPM means with a 1e-9 pseudocount, and Benjamini-Hochberg runs
within each comparison over the surviving genes only.

This differs from wilcoxon_h1_de.py, which normalized to 10,000 counts, used
geometric pseudobulk means, and corrected across all 18,080 genes.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pdex import mwu, sparse_column_index
from statsmodels.stats.multitest import multipletests

CPM_GATE = 5.0
EPSILON = 1e-9


def normalize_cpm(matrix):
    """Scale every cell to 1e6 total counts, reproducing scanpy's arithmetic exactly.

    Neither the float64 cast nor the division is incidental. cell-eval2 casts to
    float64 and scanpy divides by counts/target_sum; computing this in float32, or
    multiplying by target_sum/counts, moves values by one ulp, and one ulp makes and
    breaks ties in the rank test. Both cost about 1e-4 in p_adj (check_celleval2_parity.py).
    """
    matrix = matrix.astype(np.float64)
    counts = np.asarray(matrix.sum(axis=1)).ravel()
    assert counts.all(), "empty cells cannot be CPM-normalized"
    matrix.data /= np.repeat(counts / 1e6, np.diff(matrix.indptr))
    return matrix


def log1p(matrix):
    """In place. The rank test runs on log1p(CPM), as cell-eval2 does.

    log1p is monotone but not injective in float64: it merges a few near-tied CPM
    values, so ranks computed here are not quite the ranks of the linear matrix.
    """
    np.log1p(matrix.data, out=matrix.data)
    return matrix


def column_mean(matrix):
    return np.asarray(matrix.mean(axis=0)).ravel()


def bh(pvalues):
    return multipletests(pvalues, method="fdr_bh")[1]


def de_table(target_log, control_index, target_mean, control_mean, n_control):
    """One comparison. Every input is already restricted to the gated universe; the
    matrix and index are in log space, the means are linear CPM."""
    result = mwu(target_log, control_index)
    pvalue = np.asarray(result.pvalue).clip(0, 1)
    return pd.DataFrame(
        {
            "target_mean": target_mean,
            "control_mean": control_mean,
            "log2_fold_change": np.log2((target_mean + EPSILON) / (control_mean + EPSILON)),
            "p_value": pvalue,
            "p_adj": bh(pvalue),
            "auc": np.asarray(result.statistic) / (target_log.shape[0] * n_control),
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("data/external/vcc2025_h1/adata_Training.h5ad")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/crispri-h1-exploration/generated")
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input, backed="r")
    labels = adata.obs["target_gene"].astype(str).to_numpy()
    targets = sorted(set(labels) - {"non-targeting"})
    if args.limit:
        targets = targets[: args.limit]

    control_rows = np.flatnonzero(labels == "non-targeting")
    control = normalize_cpm(adata.X[control_rows].tocsr())
    keep = column_mean(control) > CPM_GATE
    genes = adata.var_names.to_numpy()[keep]
    control = control[:, keep]
    control_mean = column_mean(control)
    control_index = sparse_column_index(log1p(control))
    print(f"{len(control_rows)} controls; {keep.sum()} of {len(keep)} genes clear CPM > {CPM_GATE}")

    writer = None
    summaries = []
    for number, target in enumerate(targets, start=1):
        rows = np.flatnonzero(labels == target)
        matrix = normalize_cpm(adata.X[rows].tocsr())[:, keep]
        target_mean = column_mean(matrix)
        frame = de_table(log1p(matrix), control_index, target_mean, control_mean, len(control_rows))
        frame.insert(0, "gene", genes)
        frame.insert(0, "target", target)

        table = pa.Table.from_pandas(frame, preserve_index=False)
        writer = writer or pq.ParquetWriter(
            args.output / "celleval2_de.parquet", table.schema, compression="zstd"
        )
        writer.write_table(table)

        significant = frame.p_adj < 0.05
        on_target = np.flatnonzero(genes == target)
        summaries.append(
            {
                "target": target,
                "n_cells": len(rows),
                "n_de": int(significant.sum()),
                "n_up": int((significant & (frame.log2_fold_change > 0)).sum()),
                "n_down": int((significant & (frame.log2_fold_change < 0)).sum()),
                "target_gene_gated": len(on_target) == 0,
                "target_log2_fold_change": frame.log2_fold_change[on_target[0]] if len(on_target) else np.nan,
                "target_p_adj": frame.p_adj[on_target[0]] if len(on_target) else np.nan,
            }
        )
        print(f"{number:3}/{len(targets)} {target:12} {significant.sum():5} DE genes", flush=True)

    writer.close()
    pd.DataFrame(summaries).to_csv(args.output / "celleval2_summary.csv", index=False)
    (args.output / "celleval2_run.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "input": str(args.input),
                "comparison": "each target_gene versus all non-targeting cells",
                "universe": "cell-eval2 0.16.0 configs/vcc2026.yaml",
                "normalization": "per-cell total-count normalization to 1e6 (CPM)",
                "gene_filter": f"reference mean CPM > {CPM_GATE}",
                "n_genes": int(keep.sum()),
                "n_genes_total": len(keep),
                "test": "two-sided Mann-Whitney U / Wilcoxon rank-sum on log1p(CPM)",
                "log2_fold_change": "arithmetic CPM means, pseudocount 1e-9",
                "implementation": "pdex 0.3.0",
                "multiple_testing": "Benjamini-Hochberg within each perturbation over surviving genes",
                "n_controls": len(control_rows),
                "n_targets": len(targets),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
