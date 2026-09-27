"""Positive control: does celleval2_h1_de reproduce cell_eval2.compute_de exactly?

Both are run on the same small in-memory slice of the H1 training data. Agreement
on the kept gene set, p-values, BH-adjusted p-values and log2 fold changes is the
evidence that the reimplementation is the scorer's DE and not merely similar to it.
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import polars as pl
from cell_eval2.de_compute import compute_de
from celleval2_h1_de import (
    CPM_GATE,
    EPSILON,
    column_mean,
    de_table,
    log1p,
    normalize_cpm,
)
from pdex import sparse_column_index

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, default=Path("data/external/vcc2025_h1/adata_Training.h5ad"))
parser.add_argument("--n-controls", type=int, default=6000)
parser.add_argument("--targets", nargs="*", default=["LAD1", "KLF10", "SMARCA4"])
args = parser.parse_args()

adata = ad.read_h5ad(args.input, backed="r")
labels = adata.obs["target_gene"].astype(str).to_numpy()
rng = np.random.default_rng(0)
control_rows = rng.choice(np.flatnonzero(labels == "non-targeting"), args.n_controls, replace=False)
rows = np.sort(np.concatenate([control_rows] + [np.flatnonzero(labels == t) for t in args.targets]))
slice_ = adata[rows].to_memory()
print(f"slice: {slice_.n_obs} cells x {slice_.n_vars} genes")

reference = compute_de(
    slice_, backend="pdex", groupby="target_gene", reference="non-targeting",
    mean_calc="arithmetic", epsilon=EPSILON, input_type="counts", target_sum=1e6,
    filter_gene_min_cpm_cell=CPM_GATE, fdr_scope="per_pert",
).sort(["target", "feature"])

control = normalize_cpm(slice_[labels[rows] == "non-targeting"].X.tocsr())
keep = column_mean(control) > CPM_GATE
genes = slice_.var_names.to_numpy()[keep]
control = control[:, keep]
control_mean, control_index = column_mean(control), sparse_column_index(log1p(control))

mine = []
for target in args.targets:
    matrix = normalize_cpm(slice_[labels[rows] == target].X.tocsr())[:, keep]
    target_mean = column_mean(matrix)
    frame = de_table(log1p(matrix), control_index, target_mean, control_mean, control.shape[0])
    frame.insert(0, "feature", genes)
    frame.insert(0, "target", target)
    mine.append(pl.from_pandas(frame))
mine = pl.concat(mine).sort(["target", "feature"])

print(f"\ngenes kept: mine {keep.sum()}, cell_eval2 {reference['feature'].n_unique()}")
print(f"rows: mine {mine.height}, cell_eval2 {reference.height}")
assert mine["target"].to_list() == reference["target"].to_list()
assert mine["feature"].to_list() == reference["feature"].to_list()

for column in ["p_value", "p_adj", "log2_fold_change"]:
    a, b = mine[column].to_numpy(), reference[column].to_numpy()
    finite = np.isfinite(a) & np.isfinite(b)
    print(
        f"{column:18} max |diff| {np.abs(a[finite] - b[finite]).max():.3e}"
        f"   allclose {np.allclose(a[finite], b[finite], rtol=1e-9, atol=1e-12)}"
        f"   non-finite {(~finite).sum()}"
    )
def significant(frame):
    return set(map(tuple, frame.filter(pl.col("p_adj") < 0.05)[["target", "feature"]].to_numpy()))


print(f"significant sets identical: {significant(mine) == significant(reference)}")
