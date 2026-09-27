"""Assemble Checkpoint 3 for perturbation-scale cross-fitting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/replogle-h1-transfer/crossfit-scale"
PHASE1 = ROOT / "reports/replogle-h1-transfer/phase1/cell_eval"

EXISTING = {
    "h1_global_only": PHASE1 / "h1_global",
    "h1_global_raw_unscaled": PHASE1 / "h1_global_source_residual",
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
        rows.append({**row._asdict(), **fast_values(directory)})
    result = pd.DataFrame(rows)
    result.to_csv(REPORT / "checkpoint3_summary.csv", index=False)

    scales = pd.read_csv(REPORT / "per_target_scales.csv")
    scale_prediction = pd.read_csv(REPORT / "scale_prediction_summary.csv")
    comparisons = {}
    for variant in ("raw", "count_shrunk_100000"):
        frame = scales.query("source_effect == @variant and strong_oracle == strong_oracle")
        absolute_global = np.abs(frame.global_strong - frame.strong_oracle)
        absolute_ridge = np.abs(frame.ridge - frame.strong_oracle)
        rng = np.random.default_rng(20260919)
        bootstrap = np.empty(10_000)
        differences = (absolute_ridge - absolute_global).to_numpy()
        for index in range(len(bootstrap)):
            bootstrap[index] = rng.choice(differences, len(differences)).mean()
        comparisons[variant] = {
            "eligible_targets": len(frame),
            "ridge_minus_global_strong_mae": float(differences.mean()),
            "bootstrap_95pct_interval": np.quantile(
                bootstrap, (0.025, 0.975)
            ).tolist(),
            "ridge_better_targets": int((absolute_ridge < absolute_global).sum()),
            "ridge_worse_targets": int((absolute_ridge > absolute_global).sum()),
        }
    review = {
        "scale_prediction": comparisons,
        "interpretation_rule": "A useful feature model must improve held-out oracle-scale error and perturbation-specific PDS/strong-DE metrics over the cross-fitted constant scale. MSE improvement alone is insufficient.",
        "eligibility": "All arms are H1-informed diagnostics and ineligible for final zero-shot selection.",
        "scale_prediction_table": scale_prediction.astype(object)
        .where(scale_prediction.notna(), None)
        .to_dict(orient="records"),
    }
    (REPORT / "checkpoint3_review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(result.to_string(index=False))
    print(json.dumps(review, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
