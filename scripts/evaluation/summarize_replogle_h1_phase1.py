"""Assemble the expected-effect and fast-score Phase 1 checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/replogle-h1-transfer/phase1"
CELL_EVAL = REPORT / "cell_eval"
AGGREGATION = ROOT / "data/derived/replogle_h1_transfer/effects.npz"


def main() -> None:
    expected = pd.read_csv(REPORT / "expected_summary.csv").query(
        "population == 'all_126'"
    )
    per_target = pd.read_csv(REPORT / "expected_per_target.csv")
    rows = []
    for row in expected.itertuples(index=False):
        aggregate = pd.read_csv(CELL_EVAL / row.arm / "aggregates.csv").set_index(
            "metric"
        )
        scores = pd.read_csv(CELL_EVAL / row.arm / "scores.csv").set_index("metric")
        target = per_target.query("arm == @row.arm and n_strong >= 10")
        rows.append(
            {
                **row._asdict(),
                "eligibility": (
                    "diagnostic_transductive"
                    if row.arm.startswith("h1_global")
                    else "zero_shot"
                ),
                "targets_with_nmae_below_unchanged": int(
                    (target.strong_de_lfc_nmae < 1).sum()
                ),
                "targets_with_nonzero_signed_recovery": int(
                    (target.signed_recovery > 0).sum()
                ),
                "fast_raw_pds": aggregate.loc["pds_cosine", "raw_value"],
                "fast_raw_expression_mse": aggregate.loc[
                    "expr_mse_unbiased", "raw_value"
                ],
                "fast_raw_normalized_expression_mse": aggregate.loc[
                    "expr_mse_unbiased_capped_norm", "raw_value"
                ],
                "fast_pds_from_baseline": scores.loc[
                    "pds_cosine", "from_baseline"
                ],
                "fast_pds_from_replicate": scores.loc[
                    "pds_cosine", "from_replicate"
                ],
                "fast_mse_from_baseline": scores.loc[
                    "expr_mse_unbiased_capped_norm", "from_baseline"
                ],
                "fast_mse_from_replicate": scores.loc[
                    "expr_mse_unbiased_capped_norm", "from_replicate"
                ],
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(REPORT / "checkpoint2_summary.csv", index=False)

    wrong = pd.read_csv(REPORT / "wrong_target_reference.csv").set_index(
        "target_gene"
    )
    residual = per_target.query(
        "arm == 'source_residual' and source_overlap and n_strong >= 10"
    ).set_index("target_gene")
    wrong = wrong.loc[residual.index]
    with np.load(AGGREGATION, allow_pickle=False) as saved:
        source_global = saved["replogle_lfc_native"].astype(np.float64).mean(axis=0)
        h1_global = saved["h1_lfc_native"].astype(np.float64).mean(axis=0)
    checkpoint = {
        "source_h1_global_alignment": {
            "cosine": float(
                source_global @ h1_global
                / (np.linalg.norm(source_global) * np.linalg.norm(h1_global))
            ),
            "source_to_h1_norm_ratio": float(
                np.linalg.norm(source_global) / np.linalg.norm(h1_global)
            ),
            "h1_on_source_projection": float(
                source_global @ h1_global / (source_global @ source_global)
            ),
        },
        "source_residual_wrong_target_check": {
            "eligible_shared_targets": len(residual),
            "fraction_cosine_above_own_wrong_target_median": float(
                (residual.effect_cosine > wrong.median_wrong_target_cosine).mean()
            ),
            "fraction_signed_recovery_above_own_wrong_target_median": float(
                (
                    residual.signed_recovery
                    > wrong.median_wrong_target_signed_recovery
                ).mean()
            ),
        },
        "selection_status": "checkpoint only; no full VCC scorer run",
        "eligible_zero_shot_arms": [
            "unchanged",
            "source_global",
            "source_residual",
            "source_effect",
        ],
        "fast_score_avg_score_emitted": False,
    }
    (REPORT / "checkpoint2_review.json").write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n"
    )
    print(summary.to_string(index=False))
    print(json.dumps(checkpoint, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
