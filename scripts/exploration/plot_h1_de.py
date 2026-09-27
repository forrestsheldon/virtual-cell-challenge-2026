"""Plot complementary views of the H1 cell-level DE results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr

BLUE = "#4C78A8"
ORANGE = "#F58518"
GREEN = "#54A24B"
GRAY = "#B8B8B8"


def load_results(output):
    summary = pd.read_csv(output / "wilcoxon_summary.csv")
    controls = json.loads((output / "wilcoxon_run.json").read_text())["n_controls"]
    cells = pl.DataFrame(summary[["target", "n_cells"]])
    results = (
        pl.scan_parquet(output / "wilcoxon_de.parquet")
        .join(cells.lazy(), on="target")
        .with_columns(
            (pl.col("statistic") / (pl.col("n_cells") * controls) - 0.5)
            .abs()
            .alias("auc_delta")
        )
    )
    response = (
        results.group_by("target")
        .agg(
            (pl.col("fdr") <= 0.05).sum().alias("fdr_only"),
            ((pl.col("fdr") <= 0.05) & (pl.col("log2_fold_change").abs() >= 0.5))
            .sum()
            .alias("fold_change"),
            (
                (pl.col("fdr") <= 0.05)
                & (pl.col("log2_fold_change").abs() >= 0.5)
                & (pl.col("auc_delta") >= 0.15)
            )
            .sum()
            .alias("fold_change_auc"),
        )
        .collect()
        .to_pandas()
    )
    return summary.merge(response, on="target"), results


def finish(ax):
    ax.spines[["top", "right"]].set_visible(False)


def plot_spectrum(summary, output):
    ordered = summary.sort_values("fdr_only").reset_index(drop=True)
    rank = np.arange(1, len(ordered) + 1)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for column, label, color in [
        ("fdr_only", "FDR ≤ 0.05", BLUE),
        ("fold_change", "+ |log2FC| ≥ 0.5", ORANGE),
        ("fold_change_auc", "+ |AUC − 0.5| ≥ 0.15", GREEN),
    ]:
        ax.plot(rank, ordered[column], color=color, linewidth=1.5, label=label)
        ax.scatter(rank, ordered[column], color=color, s=10)
    ticks = [0, 24, 74, 124, 149]
    ax.set_xticks(np.array(ticks) + 1, ordered.loc[ticks, "target"])
    ax.set(
        yscale="log",
        xlabel="Perturbations ordered by FDR-only DE count",
        ylabel="Genes passing criterion",
        title="Effect-size thresholds reshape the response spectrum",
    )
    ax.legend(frameon=False)
    finish(ax)
    fig.savefig(output / "response_spectrum.png", dpi=180)
    plt.close(fig)


def plot_diagnostics(summary, output):
    low = summary.n_de <= 100
    colors = np.where(low, ORANGE, BLUE)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    axes[0].scatter(summary.n_cells, summary.n_de, c=colors, s=26, alpha=0.85)
    rho = spearmanr(summary.n_cells, summary.n_de).statistic
    axes[0].text(
        0.04, 0.94, f"Spearman ρ = {rho:.2f}", transform=axes[0].transAxes, va="top"
    )
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Labelled cells",
        ylabel="DE genes",
        title="Cell count is not driving DE count",
    )

    axes[1].plot([0, 1], [0, 1], color=GRAY, linewidth=1)
    axes[1].scatter(summary.control_pct, summary.target_pct, c=colors, s=26, alpha=0.85)
    axes[1].set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Target transcript detected: controls",
        ylabel="Target transcript detected: labelled cells",
        title="On-target transcript detection falls",
    )

    limit = max(summary.n_up.max(), summary.n_down.max()) * 1.15
    axes[2].plot([1, limit], [1, limit], color=GRAY, linewidth=1)
    axes[2].scatter(summary.n_up, summary.n_down, c=colors, s=26, alpha=0.85)
    axes[2].set(
        xscale="log",
        yscale="log",
        xlim=(1, limit),
        ylim=(1, limit),
        xlabel="Upregulated DE genes",
        ylabel="Downregulated DE genes",
        title="Up- and downregulation scale together",
    )

    for ax in axes:
        finish(ax)
    axes[0].scatter([], [], color=ORANGE, label="≤100 DE genes")
    axes[0].scatter([], [], color=BLUE, label=">100 DE genes")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Response diagnostics before Mixscape")
    fig.savefig(output / "response_diagnostics.png", dpi=180)
    plt.close(fig)


def plot_representative_effects(summary, results, output):
    median = summary.iloc[(summary.n_de - summary.n_de.median()).abs().argsort()[:1]]
    representatives = [
        summary.loc[summary.n_de.idxmin(), "target"],
        median.iloc[0].target,
        summary.loc[summary.n_de.idxmax(), "target"],
    ]
    selected = (
        results.filter(pl.col("target").is_in(representatives))
        .select(
            "target", "gene", "control_mean", "log2_fold_change", "fdr", "auc_delta"
        )
        .collect()
        .to_pandas()
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 4.2), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, target in zip(axes, representatives, strict=True):
        data = selected[selected.target == target]
        significant = data.fdr <= 0.05
        material = (
            significant
            & (data.log2_fold_change.abs() >= 0.5)
            & (data.auc_delta >= 0.15)
        )
        x = np.log10(data.control_mean + 1e-3)
        y = data.log2_fold_change.clip(-6, 6)
        ax.scatter(
            x[~significant],
            y[~significant],
            color=GRAY,
            s=4,
            alpha=0.25,
            rasterized=True,
        )
        ax.scatter(
            x[significant], y[significant], color=BLUE, s=5, alpha=0.35, rasterized=True
        )
        ax.scatter(
            x[material], y[material], color=ORANGE, s=7, alpha=0.75, rasterized=True
        )
        target_row = data.gene == target
        ax.scatter(
            x[target_row], y[target_row], color="black", marker="*", s=90, zorder=5
        )
        ax.annotate(
            target,
            (x[target_row].iloc[0], y[target_row].iloc[0]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=9,
        )
        count = summary.loc[summary.target == target, "n_de"].iloc[0]
        ax.set_title(f"{target}: {count:,} DE genes")
        ax.axhline(0, color="0.75", linewidth=1)
        finish(ax)
    axes[0].set_ylabel("log2 fold change (clipped at ±6)")
    for ax in axes:
        ax.set_xlabel("log10(control mean + 0.001)")
    axes[0].scatter([], [], color=GRAY, label="Not significant")
    axes[0].scatter([], [], color=BLUE, label="FDR only")
    axes[0].scatter([], [], color=ORANGE, label="FC + AUC threshold")
    axes[0].scatter([], [], color="black", marker="*", label="Target gene")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Gene-level effects for low, median, and high responders")
    fig.savefig(output / "representative_effects.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("reports/crispri-h1-exploration/generated")
    )
    args = parser.parse_args()
    summary, results = load_results(args.output)
    plot_spectrum(summary, args.output)
    plot_diagnostics(summary, args.output)
    plot_representative_effects(summary, results, args.output)


if __name__ == "__main__":
    main()
