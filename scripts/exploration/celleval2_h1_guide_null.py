"""Guide-identity null: each non-targeting guide pair against the other NT cells.

The random null in celleval2_h1_null.py destroys guide identity and batch
structure by construction. This one keeps both: the groups are the real
non-targeting guide pairs, tested with the same universe, kernel and correction.
Comparing the two separates a mis-calibrated test from real guide-associated
structure, which the size-matched random draws cannot do on their own.

Guide groups are disjoint, so this is a single one-vs-rest pass.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from celleval2_h1_de import CPM_GATE, EPSILON, bh, log1p, normalize_cpm
from pdex import mwu_one_vs_rest

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, default=Path("data/external/vcc2025_h1/adata_Training.h5ad"))
parser.add_argument("--output", type=Path, default=Path("reports/crispri-h1-exploration/generated"))
args = parser.parse_args()

adata = ad.read_h5ad(args.input, backed="r")
labels = adata.obs["target_gene"].astype(str).to_numpy()
control_rows = np.flatnonzero(labels == "non-targeting")
guides = adata.obs["guide_id"].iloc[control_rows].astype(str).to_numpy()
names = sorted(set(guides))
codes = np.searchsorted(names, guides)
genes = adata.var_names.to_numpy()

matrix = normalize_cpm(adata.X[control_rows].tocsr())
n_total = matrix.shape[0]
total_sum = np.asarray(matrix.sum(axis=0)).ravel()
group_sums = np.stack([np.asarray(matrix[codes == g].sum(axis=0)).ravel() for g in range(len(names))])
print(f"{n_total} non-targeting cells in {len(names)} guide groups", flush=True)

result = mwu_one_vs_rest(log1p(matrix), codes, len(names))

frames, summaries = [], []
for group, name in enumerate(names):
    n_cells = int((codes == group).sum())
    n_reference = n_total - n_cells
    target_mean = group_sums[group] / n_cells
    reference_mean = (total_sum - group_sums[group]) / n_reference
    keep = reference_mean > CPM_GATE
    pvalue = np.asarray(result.pvalue[group])[keep].clip(0, 1)
    frame = pd.DataFrame(
        {
            "guide": name,
            "n_cells": n_cells,
            "gene": genes[keep],
            "log2_fold_change": np.log2((target_mean[keep] + EPSILON) / (reference_mean[keep] + EPSILON)),
            "p_value": pvalue,
            "p_adj": bh(pvalue),
            "auc": np.asarray(result.statistic[group])[keep] / (n_cells * n_reference),
        }
    )
    frames.append(frame)
    summaries.append(
        {
            "guide": name,
            "n_cells": n_cells,
            "n_genes": int(keep.sum()),
            "n_de": int((frame.p_adj < 0.05).sum()),
            "n_up": int(((frame.p_adj < 0.05) & (frame.log2_fold_change > 0)).sum()),
            "n_down": int(((frame.p_adj < 0.05) & (frame.log2_fold_change < 0)).sum()),
        }
    )
    print(f"{name:45} {n_cells:6} cells {summaries[-1]['n_de']:6} DE genes", flush=True)

pd.concat(frames).to_parquet(args.output / "celleval2_guide_null_de.parquet", compression="zstd", index=False)
pd.DataFrame(summaries).to_csv(args.output / "celleval2_guide_null_summary.csv", index=False)
(args.output / "celleval2_guide_null_run.json").write_text(
    json.dumps(
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "input": str(args.input),
            "comparison": "each non-targeting guide pair versus all other non-targeting cells",
            "universe": "cell-eval2 0.16.0 configs/vcc2026.yaml, matching celleval2_h1_de.py",
            "gene_filter": f"reference mean CPM > {CPM_GATE}, recomputed per comparison",
            "test": "two-sided Mann-Whitney U / Wilcoxon rank-sum on log1p(CPM), one-vs-rest",
            "multiple_testing": "Benjamini-Hochberg within each guide over surviving genes",
            "n_controls": n_total,
            "n_guides": len(names),
        },
        indent=2,
    )
    + "\n"
)
