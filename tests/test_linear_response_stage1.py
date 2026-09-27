import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.linear_response.build_stage1_manifest import (
    H1_GENES,
    h1_target_counts,
    read_h1_genes,
    strict_control_counts,
)


def test_read_h1_genes_treats_first_row_as_data(tmp_path: Path) -> None:
    path = tmp_path / "genes.csv"
    genes = [f"gene_{index}" for index in range(H1_GENES)]
    pd.Series(genes).to_csv(path, index=False, header=False)

    assert read_h1_genes(path) == genes


def test_h1_target_counts_preserves_source_order() -> None:
    obs = pd.DataFrame(
        {
            "target_gene": [
                "non-targeting",
                "B",
                "A",
                "B",
                "A",
                "A",
            ]
        }
    )

    result = h1_target_counts(obs, ["A", "B"])

    assert result["target_gene"].tolist() == ["A", "B"]
    assert result["source_order"].tolist() == [0, 1]
    assert result["n_cells"].tolist() == [3, 2]
    assert result["calibration_candidate"].tolist() == [True, True]
    assert result["benchmark_candidate"].tolist() == [False, False]


def test_h1_target_counts_drops_unused_categories() -> None:
    obs = pd.DataFrame(
        {
            "target_gene": pd.Categorical(
                ["non-targeting", "A", "A"],
                categories=["non-targeting", "A", "unused"],
            )
        }
    )

    result = h1_target_counts(obs, ["A"])

    assert result["n_cells"].tolist() == [2]


def test_strict_control_counts_preserves_manifest_order() -> None:
    obs = pd.DataFrame(
        {
            "target_gene": ["non-targeting"] * 5 + ["A"],
            "guide_id": ["g2", "g1", "g2", "excluded", "g1", "targeting"],
            "batch": ["b1", "b1", "b2", "b1", "b1", "b1"],
        }
    )

    result = strict_control_counts(obs, ["g1", "g2"])

    assert result.to_dict("records") == [
        {"guide_id": "g1", "n_cells": 2, "n_batches": 1},
        {"guide_id": "g2", "n_cells": 2, "n_batches": 2},
    ]
