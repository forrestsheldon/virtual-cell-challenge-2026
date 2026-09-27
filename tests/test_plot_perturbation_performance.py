from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.linear_response.plot_perturbation_performance as plot_module


def test_candidate_screen_requires_accuracy_specificity_and_reliability(
    tmp_path: Path,
) -> None:
    ceiling = pd.DataFrame(
        {
            "target_gene": ["a", "a", "b", "b"],
            "cosine": [0.3, 0.5, 0.1, 0.1],
            "spearman": [0.2, 0.4, 0.1, 0.1],
            "pds": [1.0, 1.0, 0.8, 0.8],
        }
    )
    ceiling_path = tmp_path / "ceiling.csv"
    ceiling.to_csv(ceiling_path, index=False)
    original = plot_module.CEILING
    plot_module.CEILING = ceiling_path
    try:
        performance = pd.DataFrame(
            {
                "target_gene": ["a", "b"],
                "model": ["model", "model"],
                "expected_cosine": [0.2, 0.3],
                "expected_spearman": [0.1, 0.2],
                "expected_pds": [0.95, 0.95],
                "expected_squared_error": [0.5, 0.5],
                "control_squared_error": [1.0, 1.0],
                "truth_strength": [1.0, 1.0],
                "response_stratum": ["Q1", "Q2"],
            }
        )
        long, best = plot_module.prepare_tables(performance)
    finally:
        plot_module.CEILING = original
    assert long.loc[long["target_gene"].eq("a"), "descriptive_candidate"].item()
    assert not long.loc[long["target_gene"].eq("b"), "descriptive_candidate"].item()
    assert best.iloc[0]["target_gene"] == "b"
