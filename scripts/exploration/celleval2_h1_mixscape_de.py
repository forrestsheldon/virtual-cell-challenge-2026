"""Does Mixscape responder filtering change the DE set, in the cell-eval2 universe?

The earlier pre/post comparison ran at 10,000-count normalization over all 18,080
genes, where three quarters of the null's calls came from genes the scorer's CPM
gate removes. This repeats it inside the scoring universe.

"Pre" is not recomputed: with all 31 non-targeting guides as the reference, the
pre-Mixscape test is exactly celleval2_h1_de.py, so its table is read back and only
the KO-like subset is tested. The comparison then isolates responder filtering --
the two arms differ in which perturbed cells are included and in nothing else.

A third arm keeps the count comparison honest. Post-Mixscape uses fewer cells, so a
change in DE-set size mixes "filtering kept the responding cells" with "filtering
removed cells". The matched arm draws the same number of cells at random from the
same target, so the KO arm is read against a loss of power alone rather than
against the full-cell test.

Mixscape labels are inputs, carried over unchanged from the pertpy run; only the
differential expression moves to the new universe.
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
parser.add_argument("--calls", type=Path, default=Path("reports/crispri-h1-exploration/generated/mixscape_cells.parquet"))
parser.add_argument("--output", type=Path, default=Path("reports/crispri-h1-exploration/generated"))
parser.add_argument("--limit", type=int)
args = parser.parse_args()

adata = ad.read_h5ad(args.input, backed="r")
labels = adata.obs["target_gene"].astype(str).to_numpy()
calls = pd.read_parquet(args.calls)
ko_ids = set(calls.loc[(calls.target_gene != "non-targeting") & (calls.mixscape_class_global == "KO"), "cell_id"])

pre = pd.read_parquet(args.output / "celleval2_de.parquet", columns=["target", "gene", "p_adj"])
pre_sets = {t: set(g.gene[g.p_adj < 0.05]) for t, g in pre.groupby("target")}
targets = sorted(pre_sets)
if args.limit:
    targets = targets[: args.limit]

control_rows = np.flatnonzero(labels == "non-targeting")
control = normalize_cpm(adata.X[control_rows].tocsr())
keep = column_mean(control) > CPM_GATE
genes = adata.var_names.to_numpy()[keep]
control = control[:, keep]
control_mean = column_mean(control)
control_index = sparse_column_index(log1p(control))
print(f"{len(control_rows)} controls; {keep.sum()} genes; {len(targets)} targets", flush=True)

writer = None
rows_out = []
for number, target in enumerate(targets, start=1):
    rows = np.flatnonzero(labels == target)
    ko = np.fromiter((c in ko_ids for c in adata.obs_names[rows]), bool, len(rows))
    pre_genes = pre_sets[target]

    post_genes, matched_genes = set(), set()
    if ko.any():
        cells = normalize_cpm(adata.X[rows].tocsr())[:, keep]

        def test(selection):
            sub = cells[selection]
            return de_table(log1p(sub.copy()), control_index, column_mean(sub),
                            control_mean, control.shape[0])

        frame = test(ko)
        frame.insert(0, "gene", genes)
        frame.insert(0, "target", target)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        writer = writer or pq.ParquetWriter(args.output / "celleval2_post_mixscape_de.parquet",
                                            table.schema, compression="zstd")
        writer.write_table(table)
        post_genes = set(frame.gene[frame.p_adj < 0.05])

        # Size-matched random arm: same cell count, no responder selection.
        rng = np.random.default_rng(zlib.crc32(target.encode()))
        matched = np.zeros(len(rows), bool)
        matched[rng.choice(len(rows), int(ko.sum()), replace=False)] = True
        matched_genes = set(genes[test(matched).p_adj.to_numpy() < 0.05])

    union = pre_genes | post_genes
    rows_out.append(
        {
            "target": target,
            "n_pre": len(rows),
            "n_post": int(ko.sum()),
            "responder_fraction": float(ko.mean()),
            "pre_de": len(pre_genes),
            "post_de": len(post_genes),
            "matched_de": len(matched_genes),
            "shared": len(pre_genes & post_genes),
            "gained": len(post_genes - pre_genes),
            "lost": len(pre_genes - post_genes),
            "jaccard": len(pre_genes & post_genes) / len(union) if union else np.nan,
            "jaccard_matched": (len(pre_genes & matched_genes) / len(pre_genes | matched_genes)
                                if (pre_genes | matched_genes) else np.nan),
        }
    )
    r = rows_out[-1]
    print(f"{number:3}/{len(targets)} {target:12} {len(rows):5}->{int(ko.sum()):5} cells  "
          f"DE {r['pre_de']:5} -> KO {r['post_de']:5} / matched {r['matched_de']:5}  "
          f"jaccard {r['jaccard']:.3f}", flush=True)

writer.close()
summary = pd.DataFrame(rows_out)
summary.to_csv(args.output / "celleval2_mixscape_de_summary.csv", index=False)
(args.output / "celleval2_mixscape_de_run.json").write_text(
    json.dumps(
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "input": str(args.input),
            "calls": str(args.calls),
            "comparison": "all labelled cells versus Mixscape KO-like cells versus a size-matched random subsample, all against all 31 NT guide pairs",
            "matched_arm": "same cell count as the KO arm, drawn at random from the same target, seed crc32(target)",
            "pre_arm": "read back from celleval2_de.parquet; identical reference and universe",
            "mixscape_labels": "carried over unchanged from the pertpy run (old normalization)",
            "universe": "cell-eval2 0.16.0 configs/vcc2026.yaml, matching celleval2_h1_de.py",
            "criterion": "p_adj < 0.05, no effect-size threshold",
            "n_controls": len(control_rows),
            "n_genes": int(keep.sum()),
            "n_targets": len(targets),
            "n_targets_with_ko_cells": int((summary.n_post > 0).sum()),
        },
        indent=2,
    )
    + "\n"
)
print(f"targets with KO-like cells: {(summary.n_post > 0).sum()} of {len(summary)}")
