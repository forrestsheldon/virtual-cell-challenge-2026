"""Positive control: the null's one-vs-rest kernel must agree with the pairwise kernel.

celleval2_h1_null.py ranks each gene once for all pseudo-groups with
mwu_one_vs_rest, while celleval2_h1_de.py tests one target at a time with mwu.
If the two kernels disagree the null is not on the same scale as the targets and
nothing can be read across them.
"""

import anndata as ad
import numpy as np
from celleval2_h1_de import log1p, normalize_cpm
from pdex import mwu, mwu_one_vs_rest, sparse_column_index

adata = ad.read_h5ad("data/external/vcc2025_h1/adata_Training.h5ad", backed="r")
labels = adata.obs["target_gene"].astype(str).to_numpy()
rng = np.random.default_rng(0)
rows = np.sort(rng.choice(np.flatnonzero(labels == "non-targeting"), 3000, replace=False))
matrix = log1p(normalize_cpm(adata[rows].to_memory().X.tocsr()))

codes = np.ones(matrix.shape[0], dtype=np.int64)
group = rng.permutation(matrix.shape[0])[:200]
codes[group] = 0
combined = mwu_one_vs_rest(matrix, codes, 2)

mask = codes == 0
pairwise = mwu(matrix[mask], sparse_column_index(matrix[~mask]))
for name, a, b in [
    ("p_value", np.asarray(combined.pvalue[0]), np.asarray(pairwise.pvalue)),
    ("statistic", np.asarray(combined.statistic[0]), np.asarray(pairwise.statistic)),
]:
    print(f"{name:10} max |diff| {np.abs(a - b).max():.3e}   allclose {np.allclose(a, b, rtol=1e-12, atol=0)}")
