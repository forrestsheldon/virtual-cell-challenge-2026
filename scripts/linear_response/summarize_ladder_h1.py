"""Summarize the current H1 cell scores for the empirical ladder model."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/linear-response-ladder"
CELL = REPORT / "cell_eval"
CONTROL = ROOT / "reports/linear-response-three-models/cell_eval/standalone_control_baseline"
BLOG_FIGURES = (
    ROOT.parent
    / "forrestsheldon.github.io/virtual-cell/posts/draft6-baselines I: Linear Response/figures"
)
ARMS = {
    "Unchanged controls": CONTROL,
    "Matched-source generator null": CELL / "null",
    "Calibrated empirical response": CELL / "empirical",
}
METRICS = [
    "pds_cosine",
    "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_lfc_nmae",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw",
    "de_wilcoxon_sig_jaccard",
    "avg_score",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for arm, directory in ARMS.items():
        scores = pd.read_csv(directory / "scores.csv").set_index("metric")
        rows.append(
            {"arm": arm, **scores.loc[METRICS, "from_replicate"].to_dict()}
        )
    table = pd.DataFrame(rows)
    null = table.iloc[1]
    model = table.iloc[2]
    difference = {"arm": "Empirical minus matched null"}
    difference.update({metric: model[metric] - null[metric] for metric in METRICS})
    table = pd.concat([table, pd.DataFrame([difference])], ignore_index=True)
    score_path = REPORT / "full_cell_scores.csv"
    table.to_csv(score_path, index=False)

    aggregates = []
    for arm, directory in ARMS.items():
        frame = pd.read_csv(directory / "aggregates.csv")
        frame.insert(0, "arm", arm)
        aggregates.append(frame)
    aggregate_path = REPORT / "full_cell_aggregates.csv"
    pd.concat(aggregates, ignore_index=True).to_csv(aggregate_path, index=False)

    plot = table.iloc[:3].set_index("arm")[METRICS[:-1]].T
    figure, axis = plt.subplots(figsize=(10, 4.2))
    plot.plot.bar(ax=axis, width=0.78)
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_ylabel("H1 reference-scaled score")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=28)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure_path = REPORT / "figures/full_cell_score_comparison.png"
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    BLOG_FIGURES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figure_path, BLOG_FIGURES / figure_path.name)
    per_target_figure = REPORT / "figures/full_cell_per_target.png"
    shutil.copy2(per_target_figure, BLOG_FIGURES / per_target_figure.name)

    inputs = [
        directory / name
        for directory in ARMS.values()
        for name in ["scores.csv", "aggregates.csv", "manifest.json"]
    ]
    manifest = {
        "kind": "current H1 cell-level empirical ladder comparison",
        "arms": list(ARMS),
        "model_minus_null": "difference of reference-scaled metric scores; no ratio debiasing",
        "inputs": {str(path): sha256(path) for path in inputs},
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                score_path,
                aggregate_path,
                figure_path,
                per_target_figure,
            ]
        },
    }
    (REPORT / "full_cell_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
