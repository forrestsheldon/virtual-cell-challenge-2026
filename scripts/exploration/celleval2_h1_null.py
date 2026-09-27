"""Held-out control null: what does a perturbation with no effect score in the
cell-eval2 universe?

Non-targeting cells are drawn at random into pseudo-perturbation groups spanning
the range of real target cell counts (33 to 4,760) and tested against every other
non-targeting cell with exactly the pipeline in celleval2_h1_de.py. A group is
always held out of its own reference, so any DE call here is a false positive.

Groups within one replicate are disjoint, which lets a single one-vs-rest pass
rank each gene once for all of them. The reference for a group of n cells is the
remaining 38,176 - n controls rather than the full pool a real target is scored
against; at the largest group size that is a 13% smaller reference.
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
from celleval2_h1_de import CPM_GATE, EPSILON, bh, log1p, normalize_cpm
from pdex import mwu_one_vs_rest

SIZES = [25, 50, 100, 200, 400, 800, 1600, 3200, 4800]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("data/external/vcc2025_h1/adata_Training.h5ad")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/crispri-h1-exploration/generated")
    )
    parser.add_argument("--replicates", type=int, default=10)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input, backed="r")
    labels = adata.obs["target_gene"].astype(str).to_numpy()
    control_rows = np.flatnonzero(labels == "non-targeting")
    matrix = normalize_cpm(adata.X[control_rows].tocsr())
    genes = adata.var_names.to_numpy()
    n_total = matrix.shape[0]
    total_sum = np.asarray(matrix.sum(axis=0)).ravel()
    print(f"{n_total} non-targeting cells; {sum(SIZES)} drawn per replicate")

    # Draw every replicate's disjoint groups, and their linear CPM sums, before the
    # matrix is overwritten by log1p.
    draws = []
    for replicate in range(args.replicates):
        order = np.random.default_rng(replicate).permutation(n_total)
        start = 0
        for size in SIZES:
            cells = order[start : start + size]
            start += size
            draws.append(
                {
                    "replicate": replicate,
                    "n_cells": size,
                    "cells": cells,
                    "group_sum": np.asarray(matrix[cells].sum(axis=0)).ravel(),
                }
            )
    log1p(matrix)

    writer = None
    summaries = []
    for replicate in range(args.replicates):
        batch = [d for d in draws if d["replicate"] == replicate]
        codes = np.full(n_total, len(batch), dtype=np.int64)  # everything else is the rest
        for group, draw in enumerate(batch):
            codes[draw["cells"]] = group
        result = mwu_one_vs_rest(matrix, codes, len(batch) + 1)

        for group, draw in enumerate(batch):
            n_cells = draw["n_cells"]
            n_reference = n_total - n_cells
            target_mean = draw["group_sum"] / n_cells
            reference_mean = (total_sum - draw["group_sum"]) / n_reference
            keep = reference_mean > CPM_GATE
            pvalue = np.asarray(result.pvalue[group])[keep].clip(0, 1)
            frame = pd.DataFrame(
                {
                    "replicate": replicate,
                    "n_cells": n_cells,
                    "gene": genes[keep],
                    "log2_fold_change": np.log2(
                        (target_mean[keep] + EPSILON) / (reference_mean[keep] + EPSILON)
                    ),
                    "p_value": pvalue,
                    "p_adj": bh(pvalue),
                    "auc": np.asarray(result.statistic[group])[keep] / (n_cells * n_reference),
                }
            )
            table = pa.Table.from_pandas(frame, preserve_index=False)
            writer = writer or pq.ParquetWriter(
                args.output / "celleval2_null_de.parquet", table.schema, compression="zstd"
            )
            writer.write_table(table)
            summaries.append(
                {
                    "replicate": replicate,
                    "n_cells": n_cells,
                    "n_reference": n_reference,
                    "n_genes": int(keep.sum()),
                    "n_de": int((frame.p_adj < 0.05).sum()),
                    "n_up": int(((frame.p_adj < 0.05) & (frame.log2_fold_change > 0)).sum()),
                    "n_down": int(((frame.p_adj < 0.05) & (frame.log2_fold_change < 0)).sum()),
                }
            )
        counts = [s["n_de"] for s in summaries if s["replicate"] == replicate]
        print(f"replicate {replicate}: " + " ".join(f"n={s:<5}{c:>6}" for s, c in zip(SIZES, counts)), flush=True)

    writer.close()
    pd.DataFrame(summaries).to_csv(args.output / "celleval2_null_summary.csv", index=False)
    (args.output / "celleval2_null_run.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "input": str(args.input),
                "comparison": "random non-targeting pseudo-groups versus all remaining non-targeting cells",
                "universe": "cell-eval2 0.16.0 configs/vcc2026.yaml, matching celleval2_h1_de.py",
                "gene_filter": f"reference mean CPM > {CPM_GATE}, recomputed per comparison",
                "test": "two-sided Mann-Whitney U / Wilcoxon rank-sum on log1p(CPM), one-vs-rest",
                "multiple_testing": "Benjamini-Hochberg within each comparison over surviving genes",
                "n_controls": n_total,
                "sizes": SIZES,
                "replicates": args.replicates,
                "seeds": "numpy default_rng(replicate)",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
