"""Rank robust context markers as a first step toward guessing cell-line identity."""

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import false_discovery_control, ttest_ind

DEFAULT_OUTPUT = Path("reports/cell-line-identity/generated")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controls", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-genes", type=int, default=10)
    return parser.parse_args()


def pseudobulk(data: ad.AnnData) -> tuple[np.ndarray, np.ndarray]:
    """Return log2(CPM + 1) by NTC construct and cell detection fractions."""
    groups = pd.Categorical(data.obs["ntc_id"])
    rows = groups.codes
    selector = sparse.csr_matrix(
        (np.ones(data.n_obs), (rows, np.arange(data.n_obs))),
        shape=(len(groups.categories), data.n_obs),
    )
    counts = (selector @ data.X).toarray()
    cpm = counts / counts.sum(axis=1, keepdims=True) * 1_000_000
    detected = np.asarray((data.X > 0).mean(axis=0)).ravel()
    return np.log2(cpm + 1), detected


def main() -> None:
    args = arguments()
    paths = sorted(args.controls.glob("context_*.h5ad"))
    if not paths:
        raise FileNotFoundError(f"no context_*.h5ad files found in {args.controls}")

    genes = pd.read_csv(args.controls / "gene_names.csv").iloc[:, 0].astype(str)
    profiles, detected = {}, {}
    for path in paths:
        data = ad.read_h5ad(path)
        if not data.var_names.equals(pd.Index(genes)):
            raise ValueError(f"gene names or order do not match gene_names.csv: {path}")
        context = str(data.obs["context"].iloc[0])
        profiles[context], detected[context] = pseudobulk(data)

    contexts = sorted(profiles)
    means = {context: profiles[context].mean(axis=0) for context in contexts}
    tables = []
    for context in contexts:
        others = [other for other in contexts if other != context]
        own, rest = profiles[context], np.vstack([profiles[other] for other in others])
        pairwise = np.vstack([means[context] - means[other] for other in others])
        reference = np.maximum.reduce([means[other] for other in others])
        pvalue = ttest_ind(own, rest, axis=0, equal_var=False).pvalue
        pvalue = np.nan_to_num(pvalue, nan=1.0)
        tables.append(
            pd.DataFrame(
                {
                    "context": context,
                    "gene": genes,
                    "mean_log2_cpm": means[context],
                    "log2fc_vs_rest": means[context] - rest.mean(axis=0),
                    "min_pairwise_log2fc": pairwise.min(axis=0),
                    "cell_fraction": detected[context],
                    "construct_consistency": (own > reference).mean(axis=0),
                    "pvalue": pvalue,
                    "fdr": false_discovery_control(pvalue),
                }
            )
        )

    markers = pd.concat(tables, ignore_index=True)
    markers = markers.sort_values(
        ["context", "min_pairwise_log2fc"], ascending=[True, False]
    )
    eligible = markers.query(
        "fdr < 0.05 and cell_fraction >= 0.05 and construct_consistency >= 0.8"
    )
    top = eligible.groupby("context", group_keys=False).head(args.top_genes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    markers.to_csv(args.output_dir / "context_markers.csv", index=False)
    top.to_csv(args.output_dir / "top_markers.csv", index=False)

    selected = top.drop_duplicates("gene")
    indices = pd.Series(np.arange(len(genes)), index=genes).loc[selected["gene"]]
    matrix = np.vstack([means[context][indices] for context in contexts]).T
    scale = matrix.std(axis=1, keepdims=True)
    zscore = (matrix - matrix.mean(axis=1, keepdims=True)) / np.maximum(scale, 1e-8)
    fig, ax = plt.subplots(figsize=(5, max(5, len(selected) * 0.28)))
    image = ax.imshow(zscore, aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5)
    ax.set(xticks=range(len(contexts)), xticklabels=contexts, xlabel="Context")
    ax.set(yticks=range(len(selected)), yticklabels=selected["gene"])
    fig.colorbar(image, ax=ax, label="Relative pseudobulk expression (z-score)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "marker_heatmap.png", dpi=180)


if __name__ == "__main__":
    main()
