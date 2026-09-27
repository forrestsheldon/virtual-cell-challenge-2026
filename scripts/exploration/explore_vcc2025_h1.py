"""Default PCA/UMAP view of the official VCC 2025 H1 training split."""

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA


def plot_umaps(cells, output):
    extent = [
        cells.UMAP1.min(),
        cells.UMAP1.max(),
        cells.UMAP2.min(),
        cells.UMAP2.max(),
    ]
    views = {
        "umap_all.png": (np.ones(len(cells), dtype=bool), "All cells", "Greys"),
        "umap_control.png": (
            cells.target_gene == "non-targeting",
            "Non-targeting controls",
            "Blues",
        ),
        "umap_perturbed.png": (
            cells.target_gene != "non-targeting",
            "Perturbed cells",
            "Oranges",
        ),
    }
    for filename, (mask, title, cmap) in views.items():
        fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
        points = cells.loc[mask]
        density = ax.hexbin(
            points.UMAP1,
            points.UMAP2,
            gridsize=180,
            bins="log",
            mincnt=1,
            cmap=cmap,
            extent=extent,
        )
        fig.colorbar(density, ax=ax, label="Cells per bin (log scale)")
        ax.set(title=title, xlabel="UMAP1", ylabel="UMAP2", xticks=[], yticks=[])
        ax.spines[:].set_visible(False)
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)


def compute_embedding(input_path, output):
    adata = ad.read_h5ad(input_path, backed="r")
    rng = np.random.default_rng(0)
    sample = adata[rng.choice(adata.n_obs, 30_000, replace=False)].to_memory()
    sc.pp.normalize_total(sample, target_sum=10_000)
    sc.pp.log1p(sample)
    sc.pp.highly_variable_genes(sample, n_top_genes=2_000, flavor="seurat")

    hvg = sample.var.highly_variable.to_numpy()
    matrix = np.empty((adata.n_obs, hvg.sum()), dtype=np.float32)
    totals = np.empty(adata.n_obs, dtype=np.float32)
    for start in range(0, adata.n_obs, 5_000):
        stop = min(start + 5_000, adata.n_obs)
        counts = adata.X[start:stop].tocsr()
        totals[start:stop] = np.asarray(counts.sum(axis=1)).ravel()
        block = counts[:, hvg].toarray()
        block *= (10_000 / totals[start:stop])[:, None]
        matrix[start:stop] = np.log1p(block)

    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)
    matrix -= mean
    matrix /= std
    np.clip(matrix, -10, 10, out=matrix)

    pca = PCA(n_components=50, svd_solver="covariance_eigh")
    pcs = pca.fit_transform(matrix)
    reduced = ad.AnnData(obs=adata.obs.copy())
    reduced.obsm["X_pca"] = pcs
    sc.pp.neighbors(reduced, n_neighbors=15, n_pcs=30, use_rep="X_pca", random_state=0)
    sc.tl.umap(reduced, random_state=0)

    cells = adata.obs.reset_index(names="cell_id")
    cells["total_counts"] = totals
    for component in range(50):
        cells[f"PC{component + 1}"] = pcs[:, component]
    cells[["UMAP1", "UMAP2"]] = reduced.obsm["X_umap"]
    cells.to_parquet(output / "embedding.parquet", index=False)

    hvg_table = sample.var[
        ["highly_variable", "means", "dispersions", "dispersions_norm"]
    ].copy()
    hvg_table.to_csv(output / "highly_variable_genes.csv")
    pd.DataFrame(
        {
            "component": np.arange(1, 51),
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    ).to_csv(output / "pca_variance.csv", index=False)
    (output / "run.json").write_text(
        json.dumps(
            {
                "input": str(input_path),
                "cells": adata.n_obs,
                "genes": adata.n_vars,
                "hvg_sample_cells": 30_000,
                "highly_variable_genes": 2_000,
                "normalization_target": 10_000,
                "pca_components": 50,
                "neighbor_pcs": 30,
                "neighbors": 15,
                "umap_min_dist": 0.5,
                "seed": 0,
            },
            indent=2,
        )
        + "\n"
    )
    return cells


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
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cells = (
        pd.read_parquet(args.output / "embedding.parquet")
        if args.plots_only
        else compute_embedding(args.input, args.output)
    )
    plot_umaps(cells, args.output)


if __name__ == "__main__":
    main()
