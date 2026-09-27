"""Combine expected-effect and fast cell scores for the reliability experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/replogle-h1-transfer/reliability"
PHASE1 = ROOT / "reports/replogle-h1-transfer/phase1"


def fast_values(directory: Path) -> dict[str, float]:
    aggregate = pd.read_csv(directory / "aggregates.csv").set_index("metric")
    scores = pd.read_csv(directory / "scores.csv").set_index("metric")
    return {
        "fast_raw_pds": aggregate.loc["pds_cosine", "raw_value"],
        "fast_raw_expression_mse": aggregate.loc[
            "expr_mse_unbiased", "raw_value"
        ],
        "fast_raw_normalized_expression_mse": aggregate.loc[
            "expr_mse_unbiased_capped_norm", "raw_value"
        ],
        "fast_pds_from_baseline": scores.loc["pds_cosine", "from_baseline"],
    }


def main() -> None:
    expected = pd.read_csv(REPORT / "expected_summary.csv").query(
        "population == 'all_126'"
    )
    support = pd.read_csv(REPORT / "support_summary.csv")
    rows = []
    dense_fast = fast_values(PHASE1 / "cell_eval/source_effect")
    for row in expected.itertuples(index=False):
        if row.arm == "dense_unshrunken":
            fast = dense_fast
        elif (REPORT / "cell_eval" / row.arm / "scores.csv").exists():
            fast = fast_values(REPORT / "cell_eval" / row.arm)
        else:
            continue
        arm_support = support.loc[support.arm == row.arm]
        rows.append(
            {
                **row._asdict(),
                "fraction_effects_retained": (
                    float(arm_support.iloc[0].fraction_effects_retained)
                    if len(arm_support)
                    else 1.0
                ),
                **fast,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(REPORT / "checkpoint_summary.csv", index=False)
    expected_per_target = pd.read_csv(REPORT / "expected_per_target.csv")
    eligible = expected_per_target.query("n_strong >= 10")
    without_sox2 = eligible.query("target_gene != 'SOX2'")
    dense_mse = pd.read_csv(PHASE1 / "cell_eval/source_effect/per_target.csv")
    dense_mse = dense_mse.query("metric == 'expr_mse_unbiased_capped'").set_index(
        "target_gene"
    ).value
    unchanged_mse = pd.read_csv(PHASE1 / "cell_eval/unchanged/per_target.csv")
    unchanged_mse = unchanged_mse.query(
        "metric == 'expr_mse_unbiased_capped'"
    ).set_index("target_gene").value
    comparisons = {}
    for arm in result.arm:
        if arm == "dense_unshrunken":
            mse = dense_mse
        else:
            frame = pd.read_csv(REPORT / "cell_eval" / arm / "per_target.csv")
            mse = frame.query("metric == 'expr_mse_unbiased_capped'").set_index(
                "target_gene"
            ).value
        comparisons[arm] = {
            "targets_with_mse_below_dense": int((mse < dense_mse).sum()),
            "targets_with_mse_below_unchanged": int((mse < unchanged_mse).sum()),
            "mean_strong_de_nmae_excluding_sox2": float(
                without_sox2.loc[
                    without_sox2.arm == arm, "strong_de_lfc_nmae"
                ].mean()
            ),
        }
    review = {
        "interpretation": "Shrinkage provides a broad amplitude/MSE tradeoff. Cell-split sparsification removes many effects but does not improve strong-DE fidelity or NMAE at fixed shrinkage. The aggregate NMAE benefit of shrinkage is dominated by SOX2, the seven-cell source pseudobulk.",
        "comparisons": comparisons,
        "next_identifiability_check": "Compare sparse and dense effects at matched per-target norms or along matched amplitude sweeps; the present masks also shrink vector norms, so their MSE gain alone does not identify a sparsity benefit.",
    }
    (REPORT / "checkpoint_review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
