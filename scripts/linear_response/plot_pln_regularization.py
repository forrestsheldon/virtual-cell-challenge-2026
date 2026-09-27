"""Plot the compact factorial-PLN regularization diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/linear-response-sequencing-model"
SUMMARY = REPORT / "regularization_summary.csv"
BY_CPM = REPORT / "regularization_by_cpm.csv"
H1 = REPORT / "expected_profile_summary.csv"
FIGURE = REPORT / "pln_regularization.png"
H1_TABLE = REPORT / "regularization_h1_summary.csv"
MANIFEST = REPORT / "regularization_figure_manifest.json"
COLORS = {
    "raw_factorial_pln": "#5B6770",
    "mean_shrink_5cpm": "#4C78A8",
    "adaptive_eb": "#E45756",
}
LABELS = {
    "raw_factorial_pln": "Raw PLN",
    "mean_shrink_5cpm": "Mean shrinkage",
    "adaptive_eb": "Adaptive EB",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    summary = pd.read_csv(SUMMARY)
    by_cpm = pd.read_csv(BY_CPM)
    h1 = pd.read_csv(H1)
    selected_lambda = summary.loc[
        summary["model"].str.startswith("mean_shrink")
        & summary["lambda_cpm"].eq(5),
        "lambda_cpm",
    ].item()
    mean_name = f"mean_shrink_{selected_lambda:g}cpm"
    names = ["raw_factorial_pln", mean_name, "adaptive_eb"]

    h1_names = {
        "control_factorial_pln": "Raw PLN",
        "control_factorial_pln_mean_shrunk": "Mean shrinkage",
        "control_factorial_pln_adaptive_eb": "Adaptive EB",
    }
    h1_table = h1.loc[h1["model"].isin(h1_names)].copy()
    h1_table["label"] = h1_table["model"].map(h1_names)
    h1_table = h1_table[
        [
            "model",
            "label",
            "mean_signed_recovery",
            "wrong_target_mean",
            "wrong_target_p",
            "mean_exact_crossfit_nmae",
        ]
    ]
    h1_table.to_csv(H1_TABLE, index=False)

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    raw_cpm = by_cpm.loc[by_cpm["model"] == "raw_factorial_pln"]
    positions = np.arange(len(raw_cpm))
    cpm_labels = [
        f"{lower:g}–{upper:g}" if np.isfinite(upper) else f">{lower:g}"
        for lower, upper in zip(
            raw_cpm["cpm_lower"], raw_cpm["cpm_upper"], strict=True
        )
    ]
    axes[0, 0].plot(
        positions,
        raw_cpm["median_standard_error"],
        marker="o",
        label="Sampling SE",
    )
    axes[0, 0].plot(
        positions,
        raw_cpm["median_abs_covariance"],
        marker="o",
        label=r"Median $|\hat\Sigma_{gt}|$",
    )
    axes[0, 0].set_xticks(positions, cpm_labels, rotation=30, ha="right")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Control CPM")
    axes[0, 0].set_title("A  Low-count coefficients are less precise")
    axes[0, 0].legend(frameon=False)

    for name in names:
        frame = by_cpm.loc[by_cpm["model"] == name]
        axes[0, 1].plot(
            positions,
            frame["median_split_half_cosine"],
            marker="o",
            color=COLORS[name],
            label=LABELS[name],
        )
    axes[0, 1].set_xticks(positions, cpm_labels, rotation=30, ha="right")
    axes[0, 1].set_ylim(0.7, 1)
    axes[0, 1].set_xlabel("Control CPM")
    axes[0, 1].set_ylabel("Median split-half cosine")
    axes[0, 1].set_title("B  Adaptive shrinkage helps where counts are low")
    axes[0, 1].legend(frameon=False)

    mean_curve = summary.loc[summary["model"].str.startswith("mean_shrink")]
    axes[1, 0].plot(
        mean_curve["retained_covariance_energy"],
        mean_curve["median_split_half_cosine"],
        color=COLORS[mean_name],
        marker="o",
        label="Mean-shrinkage curve",
    )
    for row in mean_curve.itertuples(index=False):
        axes[1, 0].annotate(
            f"{row.lambda_cpm:g}",
            (row.retained_covariance_energy, row.median_split_half_cosine),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
        )
    for name in ["raw_factorial_pln", "adaptive_eb"]:
        row = summary.loc[summary["model"] == name].iloc[0]
        axes[1, 0].scatter(
            row["retained_covariance_energy"],
            row["median_split_half_cosine"],
            color=COLORS[name],
            s=55,
            label=LABELS[name],
            zorder=3,
        )
    axes[1, 0].set_xlabel("Covariance energy retained")
    axes[1, 0].set_ylabel("Median split-half cosine")
    axes[1, 0].set_title("C  Cosine can improve by deleting covariance energy")
    axes[1, 0].legend(frameon=False, fontsize=8)

    x = np.arange(len(h1_table))
    width = 0.34
    axes[1, 1].bar(
        x - width / 2,
        h1_table["mean_signed_recovery"],
        width,
        color="#4C78A8",
        label="Matched target",
    )
    axes[1, 1].bar(
        x + width / 2,
        h1_table["wrong_target_mean"],
        width,
        color="#B9C0C5",
        label="Wrong target",
    )
    axes[1, 1].set_xticks(x, h1_table["label"])
    axes[1, 1].set_ylabel("Signed top-K recovery")
    axes[1, 1].set_ylim(0, 0.05)
    axes[1, 1].set_title("D  Reproducibility does not rescue prediction")
    axes[1, 1].legend(
        frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0)
    )

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)

    manifest = {
        "kind": "factorial PLN regularization figure",
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [SUMMARY, BY_CPM, H1]
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [FIGURE, H1_TABLE]
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
