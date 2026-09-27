"""Relate Mixscape responder fractions to baseline target expression."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

BLUE = "#4C78A8"
ORANGE = "#F58518"


def correlations(data, column, transform="identity", subset="all targets"):
    x = data[column]
    if transform == "log10_plus_0.001":
        x = np.log10(x + 1e-3)
    spearman = spearmanr(x, data.responder_fraction)
    pearson = pearsonr(x, data.responder_fraction)
    return {
        "subset": subset,
        "measure": column,
        "transform": transform,
        "n_targets": len(data),
        "spearman_rho": spearman.statistic,
        "spearman_p": spearman.pvalue,
        "pearson_r": pearson.statistic,
        "pearson_p": pearson.pvalue,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--de",
        type=Path,
        default=Path(
            "reports/crispri-h1-exploration/generated/wilcoxon_pre_mixscape_de.parquet"
        ),
    )
    parser.add_argument(
        "--mixscape",
        type=Path,
        default=Path(
            "reports/crispri-h1-exploration/generated/wilcoxon_mixscape_de_summary.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/crispri-h1-exploration/generated"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    columns = ["target", "gene", "control_mean", "control_pct"]
    de = pd.read_parquet(args.de, columns=columns)
    on_target = de.loc[de.target == de.gene, columns].drop(columns="gene")
    mixscape = pd.read_csv(args.mixscape)[
        ["target", "n_pre", "n_post", "responder_fraction"]
    ]
    data = mixscape.merge(on_target, on="target", validate="one_to_one")
    data.to_csv(args.output / "mixscape_responder_expression.csv", index=False)

    subsets = {
        "all targets": data,
        "excluding zero-responder targets": data.loc[data.responder_fraction > 0],
        "excluding TMSB4X": data.loc[data.target != "TMSB4X"],
    }
    results = [
        correlations(frame, column, transform, subset)
        for subset, frame in subsets.items()
        for column, transform in (
            ("control_mean", "log10_plus_0.001"),
            ("control_pct", "identity"),
        )
    ]
    pd.DataFrame(results).to_csv(
        args.output / "mixscape_responder_expression_correlations.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    specifications = (
        (
            "control_mean",
            "Target expression in strict controls\n(normalized pseudobulk mean)",
            results[0],
        ),
        ("control_pct", "Target detected in strict controls", results[1]),
    )
    for ax, (column, label, result) in zip(axes, specifications, strict=True):
        ax.scatter(data[column], data.responder_fraction, s=28, alpha=0.75, color=BLUE)
        ax.text(
            0.03,
            0.97,
            f"Spearman ρ = {result['spearman_rho']:.2f}\np = {result['spearman_p']:.3g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
        ax.set(xlabel=label, ylim=(-0.03, 1.03))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_xscale("log")
    axes[0].set_ylabel("Mixscape KO-like fraction")
    axes[1].set_xlim(0, 1.02)

    quartiles = data.assign(
        expression_quartile=pd.qcut(data.control_mean, 4, labels=False)
    ).groupby("expression_quartile", observed=True)
    quartile_x = quartiles.control_mean.median()
    quartile_y = quartiles.responder_fraction.median()
    axes[0].plot(quartile_x, quartile_y, color=ORANGE, linewidth=1.5)
    axes[0].scatter(
        quartile_x,
        quartile_y,
        s=48,
        marker="D",
        color=ORANGE,
        label="Expression-quartile medians",
    )
    axes[0].legend(frameon=False, loc="lower left")

    outlier = data.loc[data.target == "TMSB4X"].iloc[0]
    axes[0].annotate(
        "TMSB4X",
        (outlier.control_mean, outlier.responder_fraction),
        xytext=(-7, 12),
        textcoords="offset points",
        ha="right",
    )
    fig.suptitle("Baseline target expression weakly tracks Mixscape response fraction")
    fig.savefig(args.output / "mixscape_responder_expression.png", dpi=180)
    plt.close(fig)

    (args.output / "mixscape_responder_expression_run.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "de_input": str(args.de),
                "mixscape_input": str(args.mixscape),
                "control_population": "32,616 cells from 26 strict NT guide pairs",
                "control_expression": (
                    "target-gene pseudobulk mean after per-cell normalization to "
                    "10,000 and log1p, reconstructed on the count scale"
                ),
                "control_detection": "fraction of strict NT cells with nonzero target counts",
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
