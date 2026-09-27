import numpy as np

from scripts.evaluation.prepare_replogle_h1_norm_matched import (
    continuous_reliability,
    match_row_norm,
)


def test_continuous_reliability_rewards_large_stable_effects() -> None:
    full = np.array([[2.0, 0.1, 0.0]])
    halves = np.array(
        [
            [[[2.1, 1.0, 0.0]], [[1.9, -0.8, 0.0]]],
            [[[2.0, -0.9, 0.0]], [[2.0, 1.1, 0.0]]],
        ]
    )
    weights = continuous_reliability(full, halves)
    assert weights[0, 0] > 0.99
    assert weights[0, 1] < 0.02
    assert weights[0, 2] == 0


def test_match_row_norm_preserves_direction_and_matches_nonzero_rows() -> None:
    values = np.array([[3.0, 4.0], [0.0, 0.0]])
    reference = np.array([[0.0, 10.0], [1.0, 2.0]])
    matched = match_row_norm(values, reference)
    assert np.allclose(matched[0], [6.0, 8.0])
    assert np.isclose(np.linalg.norm(matched[0]), np.linalg.norm(reference[0]))
    assert np.array_equal(matched[1], [0.0, 0.0])
