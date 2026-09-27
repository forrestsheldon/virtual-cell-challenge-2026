from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.linear_response.kernel import CELL_TARGET_SUM
from scripts.linear_response.state_balanced_covariance import (
    balanced_folds,
    between_columns,
    state_moments,
)


def test_state_moments_match_dense_covariance_columns() -> None:
    raw = np.array(
        [[3, 1, 0, 2], [1, 2, 4, 1], [5, 0, 1, 2], [2, 3, 2, 1]],
        dtype=np.float64,
    )
    data = ad.AnnData(sparse.csr_matrix(raw))
    fit_indices = np.array([0, 1, 2])
    target_local = np.array([0, 2])
    mean, covariance = state_moments(
        data, np.arange(len(raw)), fit_indices, target_local
    )
    values = np.log1p(
        raw[:, fit_indices] * (CELL_TARGET_SUM / raw.sum(axis=1))[:, None]
    )
    expected = np.cov(values, rowvar=False, ddof=1)
    assert np.allclose(mean, values.mean(axis=0))
    assert np.allclose(covariance, expected[:, target_local])


def test_between_columns_equal_weights_states() -> None:
    means = np.array([[1.0, 5.0, 2.0], [3.0, 4.0, 8.0], [7.0, 0.0, 5.0]])
    expected = np.cov(means, rowvar=False, ddof=1)
    assert np.allclose(between_columns(means, np.array([0, 2])), expected[:, [0, 2]])


def test_balanced_folds_split_scored_auxiliary_and_are_reproducible() -> None:
    states = ["a", "b", "c", "d", "e", "f"]
    scored = {"a", "b", "c", "d"}
    labels = np.repeat(states, [8, 7, 6, 5, 4, 3])
    batches = np.concatenate(
        [np.resize(np.array(["x", "y", "z"]), count) for count in [8, 7, 6, 5, 4, 3]]
    )
    first, _ = balanced_folds(labels, batches, states, scored)
    second, _ = balanced_folds(labels, batches, states, scored)
    assert first == second
    assert sum(first[state] == 0 for state in scored) == 2
    assert sum(first[state] == 0 for state in set(states) - scored) == 1
