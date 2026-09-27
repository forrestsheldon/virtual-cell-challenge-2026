"""Matched pre/post-Mixscape Wilcoxon DE against the strict NT ensemble."""

import argparse
import gc
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pdex import mwu, pseudobulk, sparse_column_index
from wilcoxon_h1_de import adjust_pvalues, normalize_log1p

BLUE = "#4C78A8"
ORANGE = "#F58518"
GRAY = "#B8B8B8"
CRITERIA = ("fdr", "log2fc", "auc", "any", "all")


def test_group(matrix, control_index, control_mean, control_pct, n_controls, genes):
    target_mean = pseudobulk(matrix, geometric_mean=True, is_log1p=True)
    target_pct = np.asarray(matrix.getnnz(axis=0)).ravel() / matrix.shape[0]
    result = mwu(matrix, control_index)
    pvalue = np.asarray(result.pvalue).clip(0, 1)
    fdr = adjust_pvalues(pvalue)
    log2fc = np.log2((target_mean + 1e-9) / (control_mean + 1e-9))
    auc = result.statistic / (matrix.shape[0] * n_controls)
    auc_delta = np.abs(auc - 0.5)
    passes_fdr = fdr <= 0.05
    passes_log2fc = np.abs(log2fc) >= 0.5
    passes_auc = auc_delta >= 0.15
    return pd.DataFrame(
        {
            "gene": genes,
            "target_mean": target_mean,
            "control_mean": control_mean,
            "target_pct": target_pct,
            "control_pct": control_pct,
            "log2_fold_change": log2fc,
            "p_value": pvalue,
            "statistic": result.statistic,
            "fdr": fdr,
            "auc": auc,
            "auc_delta": auc_delta,
            "passes_fdr": passes_fdr,
            "passes_log2fc": passes_log2fc,
            "passes_auc": passes_auc,
            "passes_any": passes_fdr | passes_log2fc | passes_auc,
            "passes_all": passes_fdr & passes_log2fc & passes_auc,
        }
    )


def counts(frame):
    return {criterion: int(frame[f"passes_{criterion}"].sum()) for criterion in CRITERIA}


def compare_sets(pre, post, criterion):
    pre_genes = set(pre.loc[pre[f"passes_{criterion}"], "gene"])
    post_genes = set(post.loc[post[f"passes_{criterion}"], "gene"])
    union = pre_genes | post_genes
    return {
        "shared": len(pre_genes & post_genes),
        "gained": len(post_genes - pre_genes),
        "lost": len(pre_genes - post_genes),
        "jaccard": len(pre_genes & post_genes) / len(union) if union else np.nan,
    }


def plot_comparison(summary, universes, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    points = None
    for ax, criterion, title in zip(
        axes[0], ("fdr", "all"), ("FDR ≤ 0.05", "FDR + log2FC + AUC"), strict=True
    ):
        x = summary[f"pre_{criterion}"] + 1
        y = summary[f"post_{criterion}"] + 1
        limit = max(x.max(), y.max()) * 1.2
        ax.plot([1, limit], [1, limit], color=GRAY, linewidth=1)
        points = ax.scatter(
            x, y, c=summary.responder_fraction, cmap="viridis", s=25, vmin=0, vmax=1
        )
        ax.set(
            xscale="log",
            yscale="log",
            xlim=(0.8, limit),
            ylim=(0.8, limit),
            xlabel="Pre-Mixscape genes + 1",
            ylabel="Post-Mixscape genes + 1",
            title=title,
        )
    fig.colorbar(points, ax=axes[0].tolist(), label="Mixscape KO-like fraction")

    for criterion, label, color in (
        ("fdr", "FDR ≤ 0.05", BLUE),
        ("all", "FDR + log2FC + AUC", ORANGE),
    ):
        change = np.log2(
            (summary[f"post_{criterion}"] + 1)
            / (summary[f"pre_{criterion}"] + 1)
        )
        axes[1, 0].scatter(
            summary.responder_fraction, change, s=20, alpha=0.7, color=color, label=label
        )
    axes[1, 0].axhline(0, color=GRAY, linewidth=1)
    axes[1, 0].set(
        xlabel="Mixscape KO-like fraction",
        ylabel="log2((post + 1) / (pre + 1))",
        title="Responder filtering changes DE-set size",
    )
    axes[1, 0].legend(frameon=False)

    positions = np.arange(2)
    width = 0.36
    pre_values = [universes[f"pre_{criterion}"] for criterion in ("fdr", "all")]
    post_values = [universes[f"post_{criterion}"] for criterion in ("fdr", "all")]
    axes[1, 1].bar(positions - width / 2, pre_values, width, color=BLUE, label="Pre")
    axes[1, 1].bar(
        positions + width / 2, post_values, width, color=ORANGE, label="Post"
    )
    for position, values in zip((positions - width / 2, positions + width / 2), (pre_values, post_values), strict=True):
        for x, value in zip(position, values, strict=True):
            axes[1, 1].text(x, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    axes[1, 1].set(
        xticks=positions,
        xticklabels=("FDR", "All three"),
        ylabel="Unique genes across targets",
        title="DE gene universe",
    )
    axes[1, 1].legend(frameon=False)

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Differential expression before and after Mixscape")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/external/vcc2025_h1/adata_Training.h5ad"),
    )
    parser.add_argument(
        "--calls",
        type=Path,
        default=Path(
            "reports/crispri-h1-exploration/generated/mixscape_cells.parquet"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/crispri-h1-exploration/generated")
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.input, backed="r")
    calls = pd.read_parquet(args.calls)
    targeting = calls.target_gene != "non-targeting"
    targets = sorted(calls.loc[targeting, "target_gene"].unique())
    if args.limit:
        targets = targets[: args.limit]
    ko_ids = set(calls.loc[targeting & (calls.mixscape_class_global == "KO"), "cell_id"])
    control_ids = calls.loc[calls.mixscape_class_global == "NT", "cell_id"]
    control_rows = adata.obs_names.get_indexer(control_ids)

    control = adata.X[control_rows].tocsr()
    normalize_log1p(control.data, control.indptr)
    control_mean = pseudobulk(control, geometric_mean=True, is_log1p=True)
    control_pct = np.asarray(control.getnnz(axis=0)).ravel() / control.shape[0]
    control_index = sparse_column_index(control)
    del control
    gc.collect()

    paths = {
        "pre": args.output / "wilcoxon_pre_mixscape_de.parquet",
        "post": args.output / "wilcoxon_post_mixscape_de.parquet",
    }
    writers = {"pre": None, "post": None}
    summaries = []
    universes = {
        f"{stage}_{criterion}": set()
        for stage in ("pre", "post")
        for criterion in CRITERIA
    }
    labels = adata.obs.target_gene.astype(str).to_numpy()
    started = time.monotonic()

    for number, target in enumerate(targets, 1):
        rows = np.flatnonzero(labels == target)
        matrix = adata.X[rows].tocsr()
        normalize_log1p(matrix.data, matrix.indptr)
        pre = test_group(
            matrix,
            control_index,
            control_mean,
            control_pct,
            len(control_rows),
            adata.var_names,
        )
        pre.insert(0, "target", target)
        post_mask = adata.obs_names[rows].isin(ko_ids)
        if post_mask.any():
            post = test_group(
                matrix[post_mask],
                control_index,
                control_mean,
                control_pct,
                len(control_rows),
                adata.var_names,
            )
            post.insert(0, "target", target)
        else:
            post = pre.iloc[0:0].copy()

        for stage, frame in (("pre", pre), ("post", post)):
            if len(frame):
                table = pa.Table.from_pandas(frame, preserve_index=False)
                writers[stage] = writers[stage] or pq.ParquetWriter(
                    paths[stage], table.schema, compression="zstd"
                )
                writers[stage].write_table(table)
            for criterion in CRITERIA:
                universes[f"{stage}_{criterion}"].update(
                    frame.loc[frame[f"passes_{criterion}"], "gene"]
                )

        pre_counts = counts(pre)
        post_counts = counts(post) if len(post) else dict.fromkeys(CRITERIA, 0)
        summary = {
            "target": target,
            "n_pre": len(rows),
            "n_post": int(post_mask.sum()),
            "responder_fraction": post_mask.mean(),
        }
        for criterion in CRITERIA:
            summary[f"pre_{criterion}"] = pre_counts[criterion]
            summary[f"post_{criterion}"] = post_counts[criterion]
            comparison = compare_sets(pre, post, criterion)
            summary.update(
                {f"{criterion}_{key}": value for key, value in comparison.items()}
            )
        summaries.append(summary)
        elapsed = time.monotonic() - started
        eta = elapsed / number * (len(targets) - number) / 60
        print(
            f"[{number}/{len(targets)}] {target}: {len(rows)} → {post_mask.sum()} cells; "
            f"ETA {eta:.1f} min",
            flush=True,
        )
        del matrix, pre, post
        gc.collect()

    for writer in writers.values():
        writer.close()
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / "wilcoxon_mixscape_de_summary.csv", index=False)
    universe_counts = {key: len(value) for key, value in universes.items()}
    plot_comparison(
        summary,
        universe_counts,
        args.output / "wilcoxon_mixscape_de_comparison.png",
    )
    (args.output / "wilcoxon_mixscape_de_run.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "input": str(args.input),
                "calls": str(args.calls),
                "comparison": "all labelled cells versus Mixscape KO-like cells, both against strict NT controls",
                "normalization": "per-cell total-count normalization to 10000, then log1p",
                "test": "two-sided Mann-Whitney U / Wilcoxon rank-sum",
                "implementation": "pdex 0.3.0",
                "multiple_testing": "Benjamini-Hochberg within each target across 18080 genes",
                "thresholds": {
                    "fdr": 0.05,
                    "absolute_log2_fold_change": 0.5,
                    "auc_delta": 0.15,
                },
                "n_controls": len(control_rows),
                "n_targets_pre": len(targets),
                "n_targets_post": int((summary.n_post > 0).sum()),
                "gene_universes": universe_counts,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
