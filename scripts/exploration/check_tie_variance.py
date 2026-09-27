"""Why the >5 CPM gate exists: the rank test's normal approximation fails on sparse genes.

For a gene whose values are 0/1, the Mann-Whitney null is exactly hypergeometric, so
Fisher's exact test gives the true p-value and the asymptotic rank test can be scored
against it. Ultra-sparse genes are where the old, ungated null made 76% of its calls.
"""

import anndata as ad
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

adata = ad.read_h5ad("data/external/vcc2025_h1/adata_Training.h5ad", backed="r")
labels = adata.obs["target_gene"].astype(str).to_numpy()
control = adata.X[np.flatnonzero(labels == "non-targeting")].tocsr()
N = control.shape[0]
detected = np.asarray(control.getnnz(axis=0)).ravel()
counts = np.asarray(control.sum(axis=0)).ravel()
library = np.asarray(control.sum(axis=1)).ravel().sum()
cpm = counts / library * 1e6

print(f"{N} control cells. Detection versus the gate:")
for low, high, name in [(0, 5, "below gate"), (5, np.inf, "above gate")]:
    sel = (cpm > low) & (cpm <= high)
    print(f"  {name:11} {sel.sum():6} genes, detected in median "
          f"{np.median(detected[sel]):8.0f} cells ({100*np.median(detected[sel])/N:6.3f}%)")

# A guide-sized group, and the sparsity range the old null actually called in.
n1 = 1287
print(f"\nGroup of {n1} cells ({100*n1/N:.1f}% of controls), gene present in k cells, x of them in the group.")
print(f"{'k':>6} {'x':>4} {'asymptotic p':>14} {'exact p':>12} {'ratio':>10}")
# Search the ENRICHMENT tail only: below the expected count the approximation is
# conservative instead, which is a different regime and not what the old null hit.
for k in [4, 10, 20, 50, 200, 1000]:
    for x in range(int(np.ceil(k * n1 / N)) + 1, k + 1):
        group = np.zeros(n1); group[:x] = 1
        rest = np.zeros(N - n1); rest[: k - x] = 1
        p_asymptotic = mannwhitneyu(group, rest, alternative="two-sided", method="asymptotic").pvalue
        if p_asymptotic < 0.05 / 10780:  # first x the gated BH universe would call
            p_exact = fisher_exact([[x, n1 - x], [k - x, N - n1 - (k - x)]])[1]
            print(f"{k:>6} {x:>4} {p_asymptotic:>14.3e} {p_exact:>12.3e} {p_exact/p_asymptotic:>10.1f}x")
            break
