"""Summarize the clean paired H1 global-axis and linear-response scores."""

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
    "Generator null": CONTROL,
    "Global axis": CELL / "paired_global",
    "Vanilla LR": CELL / "paired_lr",
    "LR + global axis": CELL / "paired_lr_global",
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
    aggregates = []
    for arm, directory in ARMS.items():
        scores = pd.read_csv(directory / "scores.csv").set_index("metric")
        rows.append({"arm": arm, **scores.loc[METRICS, "from_replicate"].to_dict()})
        aggregate = pd.read_csv(directory / "aggregates.csv")
        aggregate.insert(0, "arm", arm)
        aggregates.append(aggregate)
    scores = pd.DataFrame(rows)
    null = scores.iloc[0]
    differences = scores.iloc[1:].copy()
    differences["arm"] += " minus null"
    differences[METRICS] = differences[METRICS] - null[METRICS]
    scores = pd.concat([scores, differences], ignore_index=True)
    score_path = REPORT / "paired_h1_full_scores.csv"
    aggregate_path = REPORT / "paired_h1_full_aggregates.csv"
    scores.to_csv(score_path, index=False)
    pd.concat(aggregates, ignore_index=True).to_csv(aggregate_path, index=False)

    plot = scores.iloc[:4].set_index("arm")[METRICS[:-1]].T
    figure, axis = plt.subplots(figsize=(10, 4.2))
    plot.plot.bar(ax=axis, width=0.8)
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_ylabel("H1 reference-scaled score")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=28)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure_path = REPORT / "figures/paired_h1_full_scores.png"
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    BLOG_FIGURES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figure_path, BLOG_FIGURES / figure_path.name)

    inputs = [
        directory / name
        for directory in ARMS.values()
        for name in ["scores.csv", "aggregates.csv", "manifest.json"]
    ]
    manifest = {
        "kind": "clean paired H1 global-axis and linear-response comparison",
        "arms": list(ARMS),
        "null": "existing canonical all-control identity baseline",
        "differences": "additive differences of reference-scaled scores",
        "inputs": {str(path): sha256(path) for path in inputs},
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [score_path, aggregate_path, figure_path]
        },
    }
    (REPORT / "paired_h1_full_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
