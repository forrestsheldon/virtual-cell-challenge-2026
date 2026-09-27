"""DE for KO-like cells from the five non-targeting Mixscape null failures."""

import argparse
import gc
import json
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from adjustText import adjust_text
from pdex import mwu, pseudobulk, sparse_column_index
from wilcoxon_h1_de import adjust_pvalues, normalize_log1p

BLUE = "#4C78A8"
ORANGE = "#F58518"
GRAY = "#B8B8B8"


def short_guide(guide):
    return guide.replace("non-targeting_", "")


def plot_results(results, summary, output):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, row in zip(axes.flat, summary.itertuples(), strict=False):
        data = results[results.pseudo_guide == row.pseudo_guide]
        significant = data.fdr <= 0.05
        material = (
            significant
            & (data.log2_fold_change.abs() >= 0.5)
            & (data.auc_delta >= 0.15)
        )
        x = np.log10(data.control_mean + 1e-3)
        y = data.log2_fold_change.clip(-6, 6)
        ax.scatter(x[~significant], y[~significant], color=GRAY, s=4, alpha=0.2)
        ax.scatter(x[significant], y[significant], color=BLUE, s=5, alpha=0.3)
        ax.scatter(x[material], y[material], color=ORANGE, s=8, alpha=0.75)
        labels = []
        for gene in data.loc[material].nlargest(5, "auc_delta").gene:
            point = data.gene == gene
            labels.append(
                ax.text(x[point].iloc[0], y[point].iloc[0], gene, fontsize=8)
            )
        adjust_text(
            labels,
            ax=ax,
            arrowprops={"arrowstyle": "-", "color": "0.4", "linewidth": 0.5},
        )
        ax.axhline(0, color="0.75", linewidth=1)
        ax.set_title(
            f"{short_guide(row.pseudo_guide)}\n"
            f"{row.n_ko:,}/{row.n_guide_cells:,} KO-like; "
            f"{row.n_material:,} material DE"
        )
        ax.set(
            xlabel="log10(strict-control mean + 0.001)",
            ylabel="log2 fold change (clipped at ±6)",
        )
        ax.spines[["top", "right"]].set_visible(False)

    axes.flat[-1].axis("off")
    axes.flat[-1].scatter([], [], color=GRAY, label="Not significant")
    axes.flat[-1].scatter([], [], color=BLUE, label="FDR ≤ 0.05")
    axes.flat[-1].scatter([], [], color=ORANGE, label="+ FC and AUC thresholds")
    axes.flat[-1].legend(frameon=False, loc="center")
    fig.suptitle("DE among KO-like cells from non-targeting Mixscape failures")
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
            "reports/crispri-h1-exploration/generated/mixscape_null_cells.parquet"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/crispri-h1-exploration/generated")
    )
    parser.add_argument("--prefix", default="wilcoxon_null_ko")
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input, backed="r")
    calls = pd.read_parquet(args.calls)
    ko_calls = calls[calls.mixscape_class_global == "KO"]
    guides = sorted(ko_calls.pseudo_guide.unique())
    control_ids = calls.loc[~calls.direction_available, "cell_id"]
    control_rows = adata.obs_names.get_indexer(control_ids)

    control = adata.X[control_rows].tocsr()
    normalize_log1p(control.data, control.indptr)
    control_mean = pseudobulk(control, geometric_mean=True, is_log1p=True)
    control_pct = np.asarray(control.getnnz(axis=0)).ravel() / control.shape[0]
    control_index = sparse_column_index(control)
    del control
    gc.collect()

    output_path = args.output / f"{args.prefix}_de.parquet"
    writer = None
    frames = []
    summaries = []
    for number, guide in enumerate(guides, 1):
        guide_calls = calls[calls.pseudo_guide == guide]
        cell_ids = ko_calls.loc[ko_calls.pseudo_guide == guide, "cell_id"]
        rows = adata.obs_names.get_indexer(cell_ids)
        matrix = adata.X[rows].tocsr()
        normalize_log1p(matrix.data, matrix.indptr)

        target_mean = pseudobulk(matrix, geometric_mean=True, is_log1p=True)
        target_pct = np.asarray(matrix.getnnz(axis=0)).ravel() / matrix.shape[0]
        result = mwu(matrix, control_index)
        pvalue = np.asarray(result.pvalue).clip(0, 1)
        fdr = adjust_pvalues(pvalue)
        log2fc = np.log2((target_mean + 1e-9) / (control_mean + 1e-9))
        auc_delta = np.abs(
            result.statistic / (matrix.shape[0] * len(control_rows)) - 0.5
        )
        frame = pd.DataFrame(
            {
                "pseudo_guide": guide,
                "gene": adata.var_names,
                "target_mean": target_mean,
                "control_mean": control_mean,
                "target_pct": target_pct,
                "control_pct": control_pct,
                "log2_fold_change": log2fc,
                "p_value": pvalue,
                "statistic": result.statistic,
                "fdr": fdr,
                "auc_delta": auc_delta,
            }
        )
        material = (
            (fdr <= 0.05) & (np.abs(log2fc) >= 0.5) & (auc_delta >= 0.15)
        )
        summaries.append(
            {
                "pseudo_guide": guide,
                "n_guide_cells": len(guide_calls),
                "n_ko": len(rows),
                "n_controls": len(control_rows),
                "n_fdr_05": int((fdr <= 0.05).sum()),
                "n_material": int(material.sum()),
            }
        )
        table = pa.Table.from_pandas(frame, preserve_index=False)
        writer = writer or pq.ParquetWriter(
            output_path, table.schema, compression="zstd"
        )
        writer.write_table(table)
        frames.append(frame)
        print(
            f"[{number}/{len(guides)}] {short_guide(guide)}: "
            f"{material.sum()} material DE genes",
            flush=True,
        )
        del matrix, result, table
        gc.collect()

    writer.close()
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / f"{args.prefix}_summary.csv", index=False)
    plot_results(
        pd.concat(frames, ignore_index=True),
        summary,
        args.output / f"{args.prefix}_de.png",
    )
    (args.output / f"{args.prefix}_run.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "input": str(args.input),
                "calls": str(args.calls),
                "comparison": "KO-like cells per failed NT guide versus the 26-guide strict NT ensemble",
                "normalization": "per-cell total-count normalization to 10000, then log1p",
                "test": "two-sided Mann-Whitney U / Wilcoxon rank-sum",
                "implementation": "pdex 0.3.0",
                "multiple_testing": "Benjamini-Hochberg within each guide across 18080 genes",
                "material_threshold": {
                    "fdr": 0.05,
                    "absolute_log2_fold_change": 0.5,
                    "auc_delta": 0.15,
                },
                "n_controls": len(control_rows),
                "pseudo_guides": guides,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
