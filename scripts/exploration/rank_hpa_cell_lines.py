"""Rank HPA cell lines against three VCC control pseudobulks."""

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

DEFAULT_OUTPUT = Path("reports/cell-line-identity/hpa-rank/generated")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controls", type=Path)
    parser.add_argument("hpa", type=Path, help="HPA rna_celline.tsv.zip")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variable-genes", type=int, default=3_000)
    return parser.parse_args()


def plot_top_three(scores: pd.DataFrame, path: Path) -> None:
    top = scores.query("rank <= 3")
    xmin, xmax = top["spearman_rho"].min() - 0.03, top["spearman_rho"].max() + 0.04
    fig, axes = plt.subplots(3, 1, figsize=(7, 5.5), sharex=True, layout="constrained")
    for ax, (context, group) in zip(axes, top.groupby("context"), strict=True):
        group = group.sort_values("rank").copy()
        group["cell_line"] = group["cell_line"].str.replace(
            r" \(clone.*\)$", "", regex=True
        )
        y = np.arange(len(group))
        ax.hlines(y, xmin, group["spearman_rho"], color="#d9dee3", linewidth=2)
        colors = np.where(group["rank"] == 1, "#2780e3", "#d17a2b")
        ax.scatter(group["spearman_rho"], y, c=colors, s=[55, 35, 35], zorder=3)
        for yi, row in enumerate(group.itertuples()):
            ax.text(
                row.spearman_rho + 0.008, yi, f"{row.spearman_rho:.3f}", va="center"
            )
        gap = group.iloc[0]["spearman_rho"] - group.iloc[1]["spearman_rho"]
        ax.set(
            title=f"Context {context}  ·  top-to-second gap Δρ = {gap:.3f}",
            yticks=y,
            yticklabels=group["cell_line"],
            xlim=(xmin, xmax),
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.15)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[-1].set_xlabel("Spearman rank correlation, ρ")
    fig.supylabel("HPA candidate")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = arguments()
    genes = pd.read_csv(args.controls / "gene_names.csv").iloc[:, 0].astype(str)
    technical = genes.str.startswith(("MT-", "RPL", "RPS"))
    technical &= ~genes.str.startswith("RPS6K")
    genes = genes[~technical].reset_index(drop=True)
    gene_index = pd.Series(np.arange(len(genes)), index=genes)

    profiles = {}
    for path in sorted(args.controls.glob("context_*.h5ad")):
        data = ad.read_h5ad(path)
        context = str(data.obs["context"].iloc[0])
        counts = pd.Series(np.asarray(data.X.sum(axis=0)).ravel(), index=data.var_names)
        profiles[context] = counts.loc[genes].to_numpy()
    if not profiles:
        raise FileNotFoundError(f"no context_*.h5ad files found in {args.controls}")

    columns = ["Gene name", "Cell line", "nTPM"]
    head = pd.read_csv(args.hpa, sep="\t", usecols=columns, nrows=5_000)
    first_gene = head["Gene name"].iloc[0]
    lines = pd.Index(head.loc[head["Gene name"] == first_gene, "Cell line"].unique())
    line_index = pd.Series(np.arange(len(lines)), index=lines)
    reference = np.full((len(lines), len(genes)), np.nan, dtype=np.float32)

    for chunk in pd.read_csv(args.hpa, sep="\t", usecols=columns, chunksize=1_000_000):
        gene_ids = chunk["Gene name"].map(gene_index)
        line_ids = chunk["Cell line"].map(line_index)
        keep = gene_ids.notna() & line_ids.notna()
        reference[
            line_ids[keep].astype(int).to_numpy(), gene_ids[keep].astype(int).to_numpy()
        ] = chunk.loc[keep, "nTPM"].to_numpy()

    present = np.isfinite(reference).any(axis=0)
    reference = np.nan_to_num(reference[:, present])
    query = np.vstack([profiles[context][present] for context in sorted(profiles)])
    if args.variable_genes:
        logged = np.log2(reference + 1)
        median = np.median(logged, axis=0)
        variability = np.median(np.abs(logged - median), axis=0)
        selected = np.argsort(variability)[-args.variable_genes :]
        reference, query = reference[:, selected], query[:, selected]
    print(f"Compared {reference.shape[1]:,} genes")
    ref_ranks = rankdata(reference, axis=1)
    query_ranks = rankdata(query, axis=1)
    ref_ranks -= ref_ranks.mean(axis=1, keepdims=True)
    query_ranks -= query_ranks.mean(axis=1, keepdims=True)
    correlation = ref_ranks @ query_ranks.T
    correlation /= np.linalg.norm(ref_ranks, axis=1, keepdims=True)
    correlation /= np.linalg.norm(query_ranks, axis=1)

    scores = pd.DataFrame(correlation, index=lines, columns=sorted(profiles))
    scores = (
        scores.rename_axis("cell_line")
        .reset_index()
        .melt("cell_line", var_name="context", value_name="spearman_rho")
    )
    scores = scores.sort_values(["context", "spearman_rho"], ascending=[True, False])
    scores["rank"] = scores.groupby("context").cumcount() + 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"_{args.variable_genes}_variable_genes"
        if args.variable_genes
        else "_all_genes"
    )
    scores.to_csv(args.output_dir / f"hpa_rank_correlations{suffix}.csv", index=False)
    plot_top_three(scores, args.output_dir / f"hpa_rank_correlations{suffix}_top3.png")
    print(scores.query("rank <= 5").to_string(index=False))


if __name__ == "__main__":
    main()
