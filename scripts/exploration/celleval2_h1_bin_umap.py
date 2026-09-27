"""Independent HVG/PCA/UMAP embeddings, one per response bin, restricted to the
CPM>5-gated gene universe used throughout the DE analysis.

The shared UMAP (explore_vcc2025_h1.py, embedding.parquet) is fit once on all
221,273 cells with 2,000 HVGs selected from the full 18,080-gene panel. Within
that shared PC space, strong-bin cells have 55% more variance across PC1-20 than
negligible-bin cells, and negligible cells are themselves barely more variable
than pure controls (237.0 vs 236.6 total PC1-20 variance) -- so the shared layout
has little reason to resolve structure within the weaker bins even where it exists.

This reruns the identical recipe -- normalize to 10,000, log1p, 2,000 Seurat HVGs
from a seeded <=30,000-cell sample, z-score and clip at 10, sklearn
PCA(50, svd_solver="covariance_eigh"), 15-neighbor graph on PC1-30, default scanpy
UMAP, seed 0 throughout -- separately for each bin, with two changes: candidate
genes for HVG selection are restricted to the 10,780 genes clearing the reference
CPM>5 gate (celleval2_shift_genes.csv) before dispersion is computed, and the cell
set is that bin's perturbed cells (full cell counts, celleval2_response_bins.csv)
plus all 38,176 non-targeting controls, rather than 221k cells dominated by the
other two bins.

Gating before HVG selection matters most for the negligible bin: with little
biological variance to rank genes on, unrestricted HVG selection tends to surface
the sparsest, most dropout-driven genes as "highly variable" -- exactly the genes
the gate exists to remove (see check_tie_variance.py). Gating first means a
near-null bin can only select genes already known to be reliably measured.

The two normalize+log1p passes in the original script (scanpy's normalize_total
for the HVG sample, a manual chunked version for the full pass) implement
identical math -- divide each cell to a 10,000 total, then log1p -- so both are
replaced here by one chunked function used for both.
"""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA

HVG_SAMPLE_CAP = 30_000


def build_matrix(adata, rows, gene_mask, chunk=5_000):
    """Chunked normalize-to-10,000 + log1p over the gated genes. `rows` must be
    sorted ascending -- backed h5ad requires increasing indices."""
    matrix = np.empty((len(rows), gene_mask.sum()), dtype=np.float32)
    for start in range(0, len(rows), chunk):
        stop = min(start + chunk, len(rows))
        block = adata.X[rows[start:stop]].tocsr()
        totals = np.asarray(block.sum(axis=1)).ravel()
        dense = block[:, gene_mask].toarray()
        dense *= (10_000 / totals)[:, None]
        matrix[start:stop] = np.log1p(dense)
    return matrix


def embed(adata, rows, gene_mask, genes):
    rng = np.random.default_rng(0)
    sample_rows = np.sort(rows[rng.choice(len(rows), min(HVG_SAMPLE_CAP, len(rows)), replace=False)])
    sample = ad.AnnData(build_matrix(adata, sample_rows, gene_mask), var=pd.DataFrame(index=genes))
    sc.pp.highly_variable_genes(sample, n_top_genes=2_000, flavor="seurat")
    hvg = sample.var.highly_variable.to_numpy()

    full = build_matrix(adata, rows, gene_mask)[:, hvg]
    full -= full.mean(axis=0)
    full /= full.std(axis=0, ddof=1)
    np.clip(full, -10, 10, out=full)

    pca = PCA(n_components=50, svd_solver="covariance_eigh")
    pcs = pca.fit_transform(full)
    reduced = ad.AnnData(obs=pd.DataFrame(index=np.arange(len(rows)).astype(str)))
    reduced.obsm["X_pca"] = pcs
    sc.pp.neighbors(reduced, n_neighbors=15, n_pcs=30, use_rep="X_pca", random_state=0)
    sc.tl.umap(reduced, random_state=0)
    return pcs, reduced.obsm["X_umap"], int(hvg.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/external/vcc2025_h1/adata_Training.h5ad"))
    parser.add_argument("--output", type=Path, default=Path("reports/crispri-h1-exploration/generated"))
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input, backed="r")
    labels = adata.obs["target_gene"].astype(str).to_numpy()
    genes = pd.read_csv(args.output / "celleval2_shift_genes.csv").gene.to_numpy()
    gene_mask = np.isin(adata.var_names.to_numpy(), genes)
    assert gene_mask.sum() == len(genes), "gene name mismatch between the gate list and this h5ad's var_names"

    bins = pd.read_csv(args.output / "celleval2_response_bins.csv")
    control_rows = np.flatnonzero(labels == "non-targeting")
    hvg_counts = {}

    for name in ["negligible", "subtle", "strong"]:
        targets = bins.loc[bins["bin"] == name, "target"].tolist()
        rows = np.sort(np.concatenate([control_rows, np.flatnonzero(np.isin(labels, targets))]))
        print(f"{name}: {len(rows):,} cells ({len(control_rows):,} control + "
              f"{len(rows) - len(control_rows):,} perturbed across {len(targets)} targets)", flush=True)

        pcs, umap, n_hvg = embed(adata, rows, gene_mask, genes)
        hvg_counts[name] = n_hvg
        frame = adata.obs.iloc[rows].reset_index(names="cell_id")
        frame["target_gene"] = frame["target_gene"].astype(str)
        for c in range(pcs.shape[1]):
            frame[f"PC{c + 1}"] = pcs[:, c]
        frame["UMAP1"], frame["UMAP2"] = umap[:, 0], umap[:, 1]
        frame.to_parquet(args.output / f"celleval2_bin_umap_{name}.parquet", index=False)
        print(f"  {n_hvg} HVGs selected; wrote celleval2_bin_umap_{name}.parquet", flush=True)

    (args.output / "celleval2_bin_umap_run.json").write_text(json.dumps({
        "recipe": "explore_vcc2025_h1.py, unchanged parameters, restricted gene universe and cell set",
        "gene_universe": "reference CPM > 5 gate, 10780 genes (celleval2_shift_genes.csv)",
        "hvg_sample_cap": HVG_SAMPLE_CAP,
        "n_hvg_selected": hvg_counts,
        "n_pca": 50, "n_neighbors": 15, "n_pcs_for_neighbors": 30, "seed": 0,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
