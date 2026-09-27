"""Summarize full H1 scores for the state-diversity covariance experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/linear-response-state-diversity"
CONTROL = ROOT / "reports/linear-response-three-models/cell_eval/standalone_control_baseline"
LABELS = {
    "control": "Control-only LR",
    "all_within": "All states: within",
    "all_between": "All states: between",
    "all_combined": "All states: combined",
    "crossfit_within": "Cross-fit: within",
    "crossfit_between": "Cross-fit: between",
    "crossfit_combined": "Cross-fit: combined",
}
METRICS = [
    "pds_cosine",
    "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_lfc_nmae",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw",
    "de_wilcoxon_sig_jaccard",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_row(label: str, path: Path) -> dict[str, float | str]:
    scores = pd.read_csv(path / "scores.csv").set_index("metric")
    values = {metric: scores.loc[metric, "from_replicate"] for metric in METRICS}
    return {"model": label, **values, "avg_score": np.mean(list(values.values()))}


def main() -> None:
    paths = {"Generator null": CONTROL}
    paths.update({label: REPORT / "cell_eval" / arm for arm, label in LABELS.items()})
    scores = pd.DataFrame([score_row(label, path) for label, path in paths.items()])
    null = scores.iloc[0]
    deltas = scores.iloc[1:].copy()
    deltas[METRICS + ["avg_score"]] = (
        deltas[METRICS + ["avg_score"]] - null[METRICS + ["avg_score"]]
    )
    deltas["model"] += " minus null"
    scores_path = REPORT / "full_scores.csv"
    deltas_path = REPORT / "full_score_deltas.csv"
    scores.to_csv(scores_path, index=False)
    deltas.to_csv(deltas_path, index=False)

    aggregates = []
    for label, path in paths.items():
        frame = pd.read_csv(path / "aggregates.csv")
        frame.insert(0, "model", label)
        aggregates.append(frame)
    aggregates_path = REPORT / "full_aggregates.csv"
    pd.concat(aggregates, ignore_index=True).to_csv(aggregates_path, index=False)

    audits = []
    for arm, label in LABELS.items():
        frame = pd.read_csv(REPORT / "cell_eval" / arm / "rounding_audit.csv")
        audits.append(
            {
                "model": label,
                "mean_abs_lfc_error": frame["mean_abs_lfc_error"].mean(),
                "median_abs_lfc_error": frame["median_abs_lfc_error"].median(),
                "max_abs_lfc_error": frame["max_abs_lfc_error"].max(),
                "median_pooled_count_relative_l1": frame[
                    "pooled_count_relative_l1"
                ].median(),
            }
        )
    audit_path = REPORT / "rounding_audit_summary.csv"
    pd.DataFrame(audits).to_csv(audit_path, index=False)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    shown = scores.set_index("model")
    axes[0].barh(shown.index, shown["avg_score"])
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Full H1 average score")
    image = axes[1].imshow(shown[METRICS], aspect="auto", cmap="coolwarm")
    axes[1].set_xticks(range(len(METRICS)), [m.replace("de_wilcoxon_", "") for m in METRICS])
    axes[1].tick_params(axis="x", rotation=55)
    axes[1].set_yticks(range(len(shown)), shown.index)
    figure.colorbar(image, ax=axes[1], label="Scaled score")
    figure_path = REPORT / "full_scores.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    inputs = [
        file
        for path in paths.values()
        for file in [path / "scores.csv", path / "aggregates.csv"]
    ]
    inputs.extend(
        REPORT / "cell_eval" / arm / "rounding_audit.csv" for arm in LABELS
    )
    manifest = {
        "models": list(paths),
        "score": "unweighted mean of six scaled H1 metric scores",
        "inputs": {str(path): sha256(path) for path in inputs},
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                scores_path,
                deltas_path,
                aggregates_path,
                audit_path,
                figure_path,
            ]
        },
    }
    (REPORT / "full_score_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(scores.to_string(index=False))
    print("\nRounding audit")
    print(pd.DataFrame(audits).to_string(index=False))


if __name__ == "__main__":
    main()
