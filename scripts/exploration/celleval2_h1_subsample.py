"""DE at the 2026 evaluation's 400 cells per target.

Every count in celleval2_h1_de.py comes from a median of 1,045 cells per target,
but the 2026 task supplies exactly 400 cells per target, and cell-eval2's
`de_wilcoxon_lfc_nmae` needs at least 10 reference-significant genes before a
target can be scored at all. This asks how much of the panel stays scoreable when
the perturbed group shrinks to the size the competition actually uses.

The 24 targets that already have fewer than 400 cells are run whole, once, and
flagged; they cannot be subsampled up.
"""

import argparse
import json
import zlib
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from celleval2_h1_de import CPM_GATE, column_mean, de_table, log1p, normalize_cpm
from pdex import sparse_column_index

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, default=Path("data/external/vcc2025_h1/adata_Training.h5ad"))
parser.add_argument("--output", type=Path, default=Path("reports/crispri-h1-exploration/generated"))
parser.add_argument("--cells", type=int, default=400)
parser.add_argument("--replicates", type=int, default=5)
parser.add_argument("--n-controls", type=int, help="subsample controls too (2026 gives 18,400 per context)")
parser.add_argument("--prefix", default="celleval2_subsample")
parser.add_argument("--limit", type=int)
args = parser.parse_args()

adata = ad.read_h5ad(args.input, backed="r")
labels = adata.obs["target_gene"].astype(str).to_numpy()
targets = sorted(set(labels) - {"non-targeting"})
if args.limit:
    targets = targets[: args.limit]

control_rows = np.flatnonzero(labels == "non-targeting")
if args.n_controls:
    control_rows = np.sort(np.random.default_rng(0).choice(control_rows, args.n_controls, replace=False))
control = normalize_cpm(adata.X[control_rows].tocsr())
keep = column_mean(control) > CPM_GATE
genes = adata.var_names.to_numpy()[keep]
control = control[:, keep]
control_mean = column_mean(control)
control_index = sparse_column_index(log1p(control))
print(f"{len(control_rows)} controls; {keep.sum()} genes; {args.cells} cells x {args.replicates} reps per target", flush=True)

writer = None
summaries = []
for number, target in enumerate(targets, start=1):
    rows = np.flatnonzero(labels == target)
    matrix = normalize_cpm(adata.X[rows].tocsr())[:, keep]
    undersized = len(rows) <= args.cells
    replicates = 1 if undersized else args.replicates
    # crc32, not hash(): Python's string hash is salted per process, so hash() would
    # make replicate 0 a different draw on every run and the frozen table below
    # would not reproduce.
    rng = np.random.default_rng(zlib.crc32(target.encode()))

    for replicate in range(replicates):
        sub = matrix if undersized else matrix[rng.choice(len(rows), args.cells, replace=False)]
        target_mean = column_mean(sub)
        frame = de_table(log1p(sub.copy()), control_index, target_mean, control_mean, control.shape[0])
        frame.insert(0, "gene", genes)
        frame.insert(0, "replicate", replicate)
        frame.insert(0, "target", target)

        table = pa.Table.from_pandas(frame, preserve_index=False)
        writer = writer or pq.ParquetWriter(args.output / f"{args.prefix}_de.parquet", table.schema, compression="zstd")
        writer.write_table(table)

        significant = frame.p_adj < 0.05
        on_target = np.flatnonzero(genes == target)
        summaries.append(
            {
                "target": target,
                "replicate": replicate,
                "n_cells": sub.shape[0],
                "undersized": undersized,
                "n_de": int(significant.sum()),
                "n_up": int((significant & (frame.log2_fold_change > 0)).sum()),
                "n_down": int((significant & (frame.log2_fold_change < 0)).sum()),
                "target_log2_fold_change": frame.log2_fold_change[on_target[0]] if len(on_target) else np.nan,
                "target_p_adj": frame.p_adj[on_target[0]] if len(on_target) else np.nan,
            }
        )
    counts = [s["n_de"] for s in summaries if s["target"] == target]
    print(f"{number:3}/{len(targets)} {target:12} n={sub.shape[0]:5}{' *' if undersized else '  '} DE {counts}", flush=True)

writer.close()
summary = pd.DataFrame(summaries)
summary.to_csv(args.output / f"{args.prefix}_summary.csv", index=False)

# The frozen reference: one deterministic 400-cell draw per target that has 400 cells,
# against all controls. Replicate 0 is that draw, so the reference is a slice of the
# resampling study rather than a second, separately-seeded run.
frozen = summary[(~summary.undersized) & (summary.replicate == 0)]
frozen.drop(columns=["replicate", "undersized"]).to_csv(
    args.output / f"{args.prefix}_frozen_summary.csv", index=False
)
table = pq.read_table(args.output / f"{args.prefix}_de.parquet")
mask = pa.compute.and_(
    pa.compute.equal(table["replicate"], 0),
    pa.compute.is_in(table["target"], value_set=pa.array(frozen.target.tolist())),
)
pq.write_table(table.filter(mask).drop_columns(["replicate"]),
               args.output / f"{args.prefix}_frozen_de.parquet", compression="zstd")
print(f"frozen reference: {len(frozen)} targets x {args.cells} cells")
(args.output / f"{args.prefix}_run.json").write_text(
    json.dumps(
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "input": str(args.input),
            "question": "how much of the panel stays scoreable at the 2026 evaluation's 400 cells per target",
            "universe": "cell-eval2 0.16.0 configs/vcc2026.yaml, matching celleval2_h1_de.py",
            "cells_per_target": args.cells,
            "replicates": args.replicates,
            "undersized_targets_run_whole": int(sum(s["undersized"] for s in summaries)),
            "n_controls": len(control_rows),
            "n_genes": int(keep.sum()),
        },
        indent=2,
    )
    + "\n"
)
