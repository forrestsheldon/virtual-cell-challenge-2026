"""Assemble Checkpoint 4 for response-gene scaling and context features."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/replogle-h1-transfer/crossfit-gene-scale"
PREVIOUS = ROOT / "reports/replogle-h1-transfer/crossfit-scale"


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
        directory = REPORT / "cell_eval" / row.arm
        rows.append({**row._asdict(), **fast_values(directory)})
    checkpoint = pd.DataFrame(rows)
    checkpoint.to_csv(REPORT / "checkpoint4_summary.csv", index=False)

    genes = pd.read_csv(REPORT / "per_gene_coefficients.csv")
    associations = []
    top = {}
    for variant in ("raw", "count_shrunk_100000"):
        frame = genes.query("source_effect == @variant").copy()
        average_scale = (
            frame.scale_trained_for_fold0 + frame.scale_trained_for_fold1
        ) / 2
        for feature in (
            "mean_control_log2_cpm1p",
            "h1_minus_k562_control_log2_cpm1p",
        ):
            association = spearmanr(average_scale, frame[feature])
            associations.append(
                {
                    "source_effect": variant,
                    "feature": feature,
                    "spearman": association.statistic,
                    "p_value_descriptive_only": association.pvalue,
                }
            )
        stable = frame.loc[frame.stable_same_side_of_global].copy()
        stable["average_deviation"] = (
            (
                stable.scale_trained_for_fold0
                - stable.global_trained_for_fold0
            )
            + (
                stable.scale_trained_for_fold1
                - stable.global_trained_for_fold1
            )
        ) / 2
        stable = stable.reindex(
            stable.average_deviation.abs().sort_values(ascending=False).index
        )
        top[variant] = stable.head(20)[
            [
                "gene",
                "average_deviation",
                "mean_control_log2_cpm1p",
                "h1_minus_k562_control_log2_cpm1p",
            ]
        ].to_dict(orient="records")
    pd.DataFrame(associations).to_csv(
        REPORT / "gene_coefficient_associations.csv", index=False
    )

    design = json.loads((REPORT / "design.json").read_text())
    permutation = pd.read_csv(REPORT / "context_difference_permutation.csv")
    shared_summary = pd.read_csv(REPORT / "shared_normalization_summary.csv")
    shared_permutation = pd.read_csv(
        REPORT / "shared_normalization_context_permutation.csv"
    )
    previous_expected = pd.read_csv(PREVIOUS / "expected_per_target.csv").query(
        "arm == 'h1_global_shrunk_unscaled'"
    ).set_index("target_gene")
    breadth = {}
    for arm in (
        "h1_average_plus_shrunk_gene_scales",
        "h1_average_plus_shrunk_expression_context",
    ):
        current = pd.read_csv(REPORT / "expected_per_target.csv").query(
            "arm == @arm"
        ).set_index("target_gene")
        eligible = (
            previous_expected.strong_de_lfc_nmae.notna()
            & current.strong_de_lfc_nmae.notna()
        )
        nmae_change = (
            current.loc[eligible, "strong_de_lfc_nmae"]
            - previous_expected.loc[eligible, "strong_de_lfc_nmae"]
        )
        current_pds = pd.read_csv(
            REPORT / "cell_eval" / arm / "per_target.csv"
        ).query("metric == 'pds_cosine'").set_index("target_gene").value
        previous_pds = pd.read_csv(
            PREVIOUS
            / "cell_eval/h1_global_shrunk_unscaled/per_target.csv"
        ).query("metric == 'pds_cosine'").set_index("target_gene").value
        pds_change = current_pds - previous_pds
        breadth[arm] = {
            "strong_de_nmae_improved_targets": int((nmae_change < 0).sum()),
            "strong_de_eligible_targets": len(nmae_change),
            "mean_strong_de_nmae_change": float(nmae_change.mean()),
            "median_strong_de_nmae_change": float(nmae_change.median()),
            "pds_improved_targets": int((pds_change > 0).sum()),
            "pds_targets": len(pds_change),
            "mean_pds_change": float(pds_change.mean()),
            "median_pds_change": float(pds_change.median()),
        }
    review = {
        "eligibility": "All candidate arms use H1 perturbation truth and are diagnostic only.",
        "decision_rule": "A gene-level rule must improve held-out strong-DE fidelity/NMAE and PDS, remain stable across target folds, and survive the expression-matched difference permutation. MSE alone is not sufficient.",
        "gene_scale_stability": design["gene_scale_stability"],
        "context_difference_permutation": permutation.to_dict(orient="records"),
        "shared_gene_normalization_summary": shared_summary.to_dict(orient="records"),
        "shared_gene_normalization_context_permutation": shared_permutation.to_dict(
            orient="records"
        ),
        "breadth_relative_to_unscaled_count_shrunk_residual": breadth,
        "top_bootstrap_stable_gene_scale_deviations": top,
    }
    (REPORT / "checkpoint4_review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(checkpoint.to_string(index=False))
    print(pd.DataFrame(associations).to_string(index=False))
    print(json.dumps(review, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
