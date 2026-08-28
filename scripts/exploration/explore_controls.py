"""Make a compact QC panel and UMAP from the VCC 2026 control cells."""

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

DEFAULT_OUTPUT = Path("reports/control-data-exploration/generated")
PLOTS = {
    "total_counts": "UMIs per cell",
    "n_genes_by_counts": "Genes detected",
    "pct_counts_mt": "Mitochondrial counts (%)",
    "good_turing_unseen_pct": "Good–Turing unseen mass (%)",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controls", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cells-per-context", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    paths = sorted(args.controls.glob("context_*.h5ad"))
    if not paths:
        raise FileNotFoundError(f"no context_*.h5ad files found in {args.controls}")
    genes = (
        pd.read_csv(args.controls / "gene_names.csv").iloc[:, 0].astype(str).tolist()
    )
    rng = np.random.default_rng(args.seed)
    qc_tables, samples = [], []
    for path in paths:
        data = ad.read_h5ad(path)
        if data.var_names.tolist() != genes:
            raise ValueError(f"gene names or order do not match gene_names.csv: {path}")

        data.var["mt"] = data.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(data, qc_vars=["mt"], percent_top=[20], inplace=True)
        singletons = np.asarray((data.X == 1).sum(axis=1)).ravel()
        data.obs["good_turing_unseen_pct"] = 100 * singletons / data.obs["total_counts"]
        columns = ["context", "ntc_id", *PLOTS]
        table = data.obs[columns].copy()
        table.insert(0, "cell_id", data.obs_names)
        qc_tables.append(table.reset_index(drop=True))

        count = min(args.cells_per_context, data.n_obs)
        rows = np.sort(rng.choice(data.n_obs, count, replace=False))
        sample = data[rows].copy()
        sample.obs_names = sample.obs["context"].iloc[0] + ":" + sample.obs_names
        samples.append(sample)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    qc = pd.concat(qc_tables, ignore_index=True)
    qc.to_csv(args.output_dir / "qc_metrics.csv", index=False)

    contexts = sorted(qc["context"].unique())
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), constrained_layout=True)
    for ax, (metric, title) in zip(axes, PLOTS.items(), strict=True):
        values = [qc.loc[qc["context"] == context, metric] for context in contexts]
        ax.boxplot(values, tick_labels=contexts, showfliers=False)
        ax.set(title=title, xlabel="Context")
    fig.savefig(args.output_dir / "qc_panel.png", dpi=180)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for context in contexts:
        values = qc.loc[qc["context"] == context, "good_turing_unseen_pct"]
        ax.hist(values, bins=50, density=True, histtype="step", label=context)
    ax.set(xlabel="Estimated unseen UMI mass (%)", ylabel="Density")
    ax.legend(title="Context")
    fig.savefig(args.output_dir / "good_turing_distribution.png", dpi=180)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for context in contexts:
        values = qc.loc[qc["context"] == context, "total_counts"]
        ax.hist(values, bins=100, density=True, histtype="step", label=context)
    ax.set(xlabel="UMIs per cell", ylabel="Density")
    ax.legend(title="Context")
    fig.savefig(args.output_dir / "UMI_distribution.png", dpi=180)

    embedding = ad.concat(samples, join="inner")
    sc.pp.normalize_total(embedding, target_sum=10_000)
    sc.pp.log1p(embedding)
    sc.pp.highly_variable_genes(embedding, n_top_genes=2_000, flavor="seurat")
    embedding = embedding[:, embedding.var["highly_variable"]].copy()
    sc.pp.scale(embedding, max_value=10)
    sc.tl.pca(embedding, n_comps=50, random_state=args.seed)
    sc.pp.neighbors(embedding, n_neighbors=15, n_pcs=30, random_state=args.seed)
    sc.tl.umap(embedding, random_state=args.seed)
    colors = ["context",  "total_counts"]
    sc.pl.umap(embedding, color=colors, frameon=False, show=False)
    figure = plt.gcf()
    # figure.axes[1].set_title("Good–Turing unseen mass (%)")
    figure.axes[1].set_title("UMIs per cell")
    figure.savefig(args.output_dir / "umap.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
