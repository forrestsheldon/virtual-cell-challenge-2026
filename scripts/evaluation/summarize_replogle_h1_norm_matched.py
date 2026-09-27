"""Summarize amplitude-matched reliability and sparsity comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/replogle-h1-transfer/norm-matched"
RELIABILITY = ROOT / "reports/replogle-h1-transfer/reliability/cell_eval"

EXISTING = {
    "moderate_shrink": RELIABILITY / "shrink_100000",
    "hard_2of3_natural": RELIABILITY / "shrink_100000_stable_2of3",
}


def fast_values(directory: Path) -> dict[str, float]:
    aggregate = pd.read_csv(directory / "aggregates.csv").set_index("metric")
    scores = pd.read_csv(directory / "scores.csv").set_index("metric")
    return {
        "fast_raw_pds": aggregate.loc["pds_cosine", "raw_value"],
        "fast_raw_expression_mse": aggregate.loc[
            "expr_mse_unbiased", "raw_value"
        ],
        "fast_normalized_expression_mse": aggregate.loc[
            "expr_mse_unbiased_capped_norm", "raw_value"
        ],
        "fast_pds_from_baseline": scores.loc["pds_cosine", "from_baseline"],
    }


def main() -> None:
    expected = pd.read_csv(REPORT / "expected_summary.csv").query(
        "population == 'all_126'"
    )
    rows = []
    for row in expected.itertuples(index=False):
        directory = EXISTING.get(row.arm, REPORT / "cell_eval" / row.arm)
        if not (directory / "scores.csv").exists():
            continue
        rows.append({**row._asdict(), **fast_values(directory)})
    result = pd.DataFrame(rows)
    result.to_csv(REPORT / "checkpoint_summary.csv", index=False)

    indexed = result.set_index("arm")
    comparisons = {}
    for selected, dense in (
        ("continuous_natural", "dense_at_continuous_norm"),
        ("hard_2of3_natural", "dense_at_hard_2of3_norm"),
        ("continuous_at_dense_norm", "moderate_shrink"),
        ("hard_2of3_at_dense_norm", "moderate_shrink"),
    ):
        comparisons[f"{selected}_minus_{dense}"] = {
            metric: float(indexed.loc[selected, metric] - indexed.loc[dense, metric])
            for metric in (
                "mean_signed_recovery",
                "mean_strong_de_lfc_nmae",
                "fast_pds_from_baseline",
                "fast_normalized_expression_mse",
            )
        }
    review = {
        "paired_arm_differences": comparisons,
        "direction": "positive is better for signed recovery and PDS; negative is better for NMAE and expression MSE",
        "decision_rule": "Support selection must improve perturbation identification at matched norm; an MSE improvement alone is not evidence for better support.",
    }
    (REPORT / "checkpoint_review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n"
    )
    print(result.to_string(index=False))
    print(json.dumps(review, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
