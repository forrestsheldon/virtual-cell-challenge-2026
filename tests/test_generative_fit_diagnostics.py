from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.linear_response.generative_fit_diagnostics import (
    column_generalization,
    covariance_columns,
)


def test_covariance_columns_matches_dense_covariance() -> None:
    values = np.array(
        [[1.0, 2.0, 4.0], [2.0, 1.0, 3.0], [4.0, 3.0, 2.0], [3.0, 5.0, 1.0]]
    )
    observed = covariance_columns(values, np.array([0, 1]), np.array([1, 2]))
    expected = np.cov(values, rowvar=False, ddof=1)[np.ix_([0, 1], [1, 2])]
    assert np.allclose(observed, expected)


def test_matched_covariance_beats_random_column_pairing() -> None:
    rng = np.random.default_rng(4)
    observed = rng.normal(size=(200, 12))
    predicted = observed + rng.normal(scale=0.05, size=observed.shape)
    result = column_generalization(predicted, observed, rng, n_random=1_000)
    assert result["median_matched_cosine"] > result["random_pairing_q95"]
    assert result["random_pairing_p"] < 0.01
    assert result["squared_error_over_zero_covariance"] < 0.01
