"""Wilcoxon DE for each non-targeting guide against the other NT guides."""

import argparse
import gc
import json
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pdex import mwu, pseudobulk, sparse_column_index
from wilcoxon_h1_de import adjust_pvalues, normalize_log1p


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
    parser.add_argument("--exclude-guides", nargs="*", default=[])
    parser.add_argument("--prefix", default="wilcoxon_null")
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input, backed="r")
    nt_rows = np.flatnonzero(adata.obs.target_gene.to_numpy() == "non-targeting")
    guides = adata.obs.guide_id.iloc[nt_rows].astype(str).to_numpy()
    keep = ~np.isin(guides, args.exclude_guides)
    nt_rows = nt_rows[keep]
    guides = guides[keep]
    pseudo_guides = sorted(set(guides))
    if args.limit:
        pseudo_guides = pseudo_guides[: args.limit]

    matrix = adata.X[nt_rows].tocsr()
    normalize_log1p(matrix.data, matrix.indptr)
    output_path = args.output / f"{args.prefix}_de.parquet"
    writer = None
    summaries = []

    for number, guide in enumerate(pseudo_guides, 1):
        target_mask = guides == guide
        target = matrix[target_mask]
        control = matrix[~target_mask]
        target_mean = pseudobulk(target, geometric_mean=True, is_log1p=True)
        control_mean = pseudobulk(control, geometric_mean=True, is_log1p=True)
        result = mwu(target, sparse_column_index(control))
        pvalue = np.asarray(result.pvalue).clip(0, 1)
        fdr = adjust_pvalues(pvalue)
        log2fc = np.log2((target_mean + 1e-9) / (control_mean + 1e-9))
        auc_delta = np.abs(
            result.statistic / (target.shape[0] * control.shape[0]) - 0.5
        )
        frame = pd.DataFrame(
            {
                "pseudo_guide": guide,
                "gene": adata.var_names,
                "log2_fold_change": log2fc,
                "p_value": pvalue,
                "fdr": fdr,
                "auc_delta": auc_delta,
            }
        )
        table = pa.Table.from_pandas(frame, preserve_index=False)
        writer = writer or pq.ParquetWriter(
            output_path, table.schema, compression="zstd"
        )
        writer.write_table(table)
        summaries.append(
            {
                "pseudo_guide": guide,
                "n_cells": target.shape[0],
                "n_controls": control.shape[0],
                "n_fdr_05": int((fdr <= 0.05).sum()),
            }
        )
        print(
            f"[{number}/{len(pseudo_guides)}] {guide}: {(fdr <= 0.05).sum()} DE genes",
            flush=True,
        )
        del target, control, result, frame, table
        gc.collect()

    writer.close()
    pd.DataFrame(summaries).to_csv(
        args.output / f"{args.prefix}_summary.csv", index=False
    )
    (args.output / f"{args.prefix}_run.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "input": str(args.input),
                "comparison": "each NT guide versus all other NT guides",
                "normalization": "per-cell total-count normalization to 10000, then log1p",
                "test": "two-sided Mann-Whitney U / Wilcoxon rank-sum",
                "implementation": "pdex 0.3.0",
                "multiple_testing": "Benjamini-Hochberg within each guide across 18080 genes",
                "pseudo_guides": len(pseudo_guides),
                "excluded_guides": args.exclude_guides,
                "seed": 0,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
