"""Plot target-level H1 expected-profile performance across model variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/linear-response-three-models"
PROFILE_SCORES = REPORT / "profile_scores"
CEILING = REPORT / "h1_profile_ceiling_per_target.csv"
OUTPUT = REPORT / "figures/h1_perturbation_performance.png"

SCORE_SPECS = [
    ("model1", "Model 1: normalized covariance"),
    ("model1_raw_count", "Raw-count covariance ablation"),
    ("model2", "Model 2: rank-50 covariance"),
    ("model3", "Model 3: pure-factor PLN (legacy fit)"),
    ("model3_full", "Model 3: full PLN ($LL^T+D$)"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_performance() -> tuple[pd.DataFrame, list[Path]]:
    tables = []
    inputs = []
    for directory, label in SCORE_SPECS:
        path = PROFILE_SCORES / directory / "per_target.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        table.insert(1, "model", label)
        tables.append(table)
        inputs.append(path)
    if not tables:
        raise FileNotFoundError("no expected-profile per-target tables found")
    return pd.concat(tables, ignore_index=True), inputs


def prepare_tables(performance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ceiling = (
        pd.read_csv(CEILING)
        .groupby("target_gene", as_index=False)
        .agg(
            split_half_cosine_median=("cosine", "median"),
            split_half_spearman_median=("spearman", "median"),
            split_half_pds_median=("pds", "median"),
        )
    )
    long = performance.merge(ceiling, on="target_gene", validate="many_to_one")
    long["expected_error_ratio"] = (
        long["expected_squared_error"] / long["control_squared_error"]
    )
    long["descriptive_candidate"] = (
        long["expected_cosine"].ge(0.10)
        & long["expected_pds"].ge(0.90)
        & long["expected_error_ratio"].lt(1.0)
        & long["split_half_cosine_median"].ge(0.20)
    )
    best = (
        long.sort_values(
            ["target_gene", "expected_cosine", "expected_pds"],
            ascending=[True, False, False],
        )
        .drop_duplicates("target_gene")
        .rename(
            columns={
                "model": "best_model_by_cosine",
                "expected_cosine": "best_expected_cosine",
                "expected_spearman": "best_expected_spearman",
                "expected_pds": "best_expected_pds",
                "expected_error_ratio": "best_expected_error_ratio",
            }
        )
    )
    columns = [
        "target_gene",
        "best_model_by_cosine",
        "best_expected_cosine",
        "best_expected_spearman",
        "best_expected_pds",
        "best_expected_error_ratio",
        "split_half_cosine_median",
        "split_half_spearman_median",
        "split_half_pds_median",
        "truth_strength",
        "response_stratum",
        "descriptive_candidate",
    ]
    return long, best[columns].sort_values("best_expected_cosine", ascending=False)


def heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    cmap: str,
    norm=None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    image = axis.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(image, ax=axis, fraction=0.025, pad=0.02)


def plot(long: pd.DataFrame, best: pd.DataFrame, path: Path) -> None:
    models = [label for _, label in SCORE_SPECS if label in set(long["model"])]
    order = best["target_gene"].tolist()
    cosine = (
        long.pivot(index="target_gene", columns="model", values="expected_cosine")
        .reindex(index=order, columns=models)
        .to_numpy()
    )
    pds = (
        long.pivot(index="target_gene", columns="model", values="expected_pds")
        .reindex(index=order, columns=models)
        .to_numpy()
    )
    error = (
        long.pivot(index="target_gene", columns="model", values="expected_error_ratio")
        .reindex(index=order, columns=models)
        .to_numpy()
    )
    reliability = (
        best.set_index("target_gene")
        .loc[order, "split_half_cosine_median"]
        .to_numpy()[:, None]
    )
    limit = max(0.2, float(np.nanquantile(np.abs(cosine), 0.98)))
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(15, 27),
        gridspec_kw={"width_ratios": [len(models), len(models), len(models), 1]},
        constrained_layout=True,
    )
    heatmap(
        axes[0],
        cosine,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
    )
    heatmap(axes[1], pds, cmap="viridis", vmin=0, vmax=1)
    heatmap(
        axes[2],
        np.log10(np.maximum(error, 1e-2)),
        cmap="magma",
        vmin=-1,
        vmax=max(1, float(np.nanquantile(np.log10(np.maximum(error, 1e-2)), 0.98))),
    )
    heatmap(
        axes[3],
        reliability,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
    )
    titles = [
        "Target-excluded cosine",
        "Target-specific PDS",
        "log10(error / unchanged control)",
        "Truth split-half\ncosine",
    ]
    for axis, title in zip(axes, titles, strict=True):
        axis.set_title(title, fontsize=11)
        axis.set_yticks(np.arange(len(order)))
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticklabels(order, fontsize=5)
    for axis in axes[1:]:
        axis.set_yticklabels([])
    for axis in axes[:3]:
        axis.set_xticks(np.arange(len(models)))
        axis.set_xticklabels(models, rotation=55, ha="right", fontsize=8)
    axes[3].set_xticks([0])
    axes[3].set_xticklabels(["reliability"], rotation=55, ha="right", fontsize=8)
    candidate_targets = set(best.loc[best["descriptive_candidate"], "target_gene"])
    for row, target in enumerate(order):
        if target in candidate_targets:
            axes[0].text(
                -0.65,
                row,
                "★",
                ha="center",
                va="center",
                fontsize=5,
                color="#111111",
                clip_on=False,
            )
    figure.suptitle(
        "H1 perturbation-level linear-response diagnostics\n"
        "Rows sorted by the best model cosine; ★ is a descriptive multi-metric candidate",
        fontsize=15,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    performance, score_inputs = load_performance()
    long, best = prepare_tables(performance)
    plot(long, best, args.output)
    outputs = {
        "long_table": REPORT / "h1_perturbation_performance.csv",
        "best_by_target": REPORT / "h1_perturbation_best_models.csv",
        "candidate_hits": REPORT / "h1_perturbation_candidate_hits.csv",
        "png": args.output,
        "pdf": args.output.with_suffix(".pdf"),
    }
    long.to_csv(outputs["long_table"], index=False)
    best.to_csv(outputs["best_by_target"], index=False)
    best[best["descriptive_candidate"]].to_csv(outputs["candidate_hits"], index=False)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "kind": "descriptive expected-profile perturbation comparison; not an official H1 score",
        "sort": "descending best target-excluded cosine across available models",
        "candidate_definition": {
            "expected_cosine": ">= 0.10",
            "expected_pds": ">= 0.90",
            "expected_error_ratio": "< 1.0",
            "median_truth_split_half_cosine": ">= 0.20",
            "interpretation": "descriptive screen only; not a multiple-testing-adjusted significance claim",
        },
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [*score_inputs, CEILING]
        },
        "outputs": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for name, path in outputs.items()
        },
        "software": {
            "python": platform.python_version(),
            **{name: version(name) for name in ["matplotlib", "numpy", "pandas"]},
        },
    }
    (REPORT / "h1_perturbation_performance_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
