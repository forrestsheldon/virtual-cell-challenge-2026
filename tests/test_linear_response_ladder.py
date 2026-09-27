from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluation.prepare_paired_linear_response_h1 import canonical_source_rows
from scripts.linear_response.evaluate_ladder import holm, scales
from scripts.linear_response.generate_ladder_h1 import seed
from scripts.linear_response.ladder_factor import factor_response
from scripts.linear_response.ladder_sparse import (
    heldout_error,
    regression_coefficients,
)


def test_sparse_zero_penalty_is_simple_regression() -> None:
    covariance = np.array([[2.0, 0.2], [0.3, 3.0], [0.4, -0.6]])
    variance = np.array([2.0, 3.0, 4.0])
    targets = np.array([0, 1])
    observed = regression_coefficients(covariance, variance, targets, 100, 0)
    expected = covariance / variance[targets]
    expected[targets, np.arange(2)] = 1
    assert np.allclose(observed, expected)


def test_sparse_large_penalty_removes_downstream_coefficients() -> None:
    covariance = np.array([[2.0], [0.2], [-0.1]])
    variance = np.array([2.0, 3.0, 4.0])
    observed = regression_coefficients(covariance, variance, np.array([0]), 20, 100)
    assert np.array_equal(observed[:, 0], [1.0, 0.0, 0.0])


def test_heldout_error_rewards_correct_regression() -> None:
    variance = np.array([2.0, 3.0, 4.0])
    covariance = np.array([[2.0], [1.5], [-1.0]])
    target = np.array([0])
    zero = np.array([[1.0], [0.0], [0.0]])
    fitted = covariance / variance[target]
    assert heldout_error(fitted, covariance, variance, target) < heldout_error(
        zero, covariance, variance, target
    )


def test_factor_response_matches_dense_factor_covariance() -> None:
    components = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.8, 0.6]], dtype=np.float64
    )
    singular = np.array([4.0, 2.0])
    variance = np.array([2.0, 1.0, 1.5])
    targets = np.array([0, 2])
    response, covariance, residual = factor_response(
        components, singular, 10, variance, targets, 2
    )
    dense = components.T @ np.diag(np.square(singular) / 9) @ components
    assert np.allclose(covariance, dense[:, targets])
    assert np.all(residual >= 0)
    assert np.array_equal(response[targets, np.arange(2)], [-1.0, -1.0])


def test_amplitude_transfer_excludes_held_target() -> None:
    predictions = {"model": np.eye(3)}
    truth = np.diag([1.0, 2.0, 9.0])
    stable = np.eye(3, dtype=bool)
    table, transferred, oracle = scales(predictions, truth, stable)
    assert np.isnan(table["oracle_gamma"]).all()
    assert np.isnan(transferred["model"]).all()
    assert np.isnan(oracle["model"]).all()

    predictions = {"model": np.ones((3, 12))}
    truth = np.asarray([[1.0] * 12, [2.0] * 12, [9.0] * 12])
    stable = np.ones_like(truth, dtype=bool)
    table, transferred, oracle = scales(predictions, truth, stable)
    assert table["eligible"].all()
    assert np.allclose(transferred["model"], [5.5, 5.0, 1.5])
    assert np.allclose(oracle["model"], [1.0, 2.0, 9.0])


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = holm(np.array([0.04, 0.001, 0.02]))
    assert np.allclose(adjusted, [0.04, 0.003, 0.04])


def test_model_scripts_do_not_reference_truth_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ["ladder_empirical.py", "ladder_sparse.py", "ladder_factor.py"]:
        source = (root / "scripts/linear_response" / name).read_text()
        assert "reference_cells" not in source
        assert "eval_cache" not in source
        assert "strong_de_truth" not in source


def test_full_cell_model_and_null_streams_are_reproducible_and_independent() -> None:
    model = np.random.default_rng(seed("empirical", 7)).integers(0, 100, 10)
    repeated = np.random.default_rng(seed("empirical", 7)).integers(0, 100, 10)
    null = np.random.default_rng(seed("null", 7)).integers(0, 100, 10)
    assert np.array_equal(model, repeated)
    assert not np.array_equal(model, null)


def test_paired_h1_sources_are_canonical_deterministic_control_draws() -> None:
    first = canonical_source_rows(2_000, ["A", "B"])
    repeated = canonical_source_rows(2_000, ["A", "B"])
    assert first.shape == (2, 400)
    assert np.array_equal(first, repeated)
    assert all(len(np.unique(rows)) == 400 for rows in first)
    assert not np.array_equal(first[0], first[1])
