import numpy as np

from scripts.evaluation.oracle_scale_comparison import linear_scale, weighted_median


def test_weighted_median_matches_brute_force_l1_minimum():
    values = np.array([0.2, 0.7, 1.5, 3.0])
    weights = np.array([1.0, 2.0, 5.0, 1.0])
    selected = weighted_median(values, weights)
    grid = np.linspace(0, 3, 30_001)
    objective = np.sum(weights * np.abs(grid[:, None] - values), axis=1)
    assert abs(selected - grid[np.argmin(objective)]) < 1e-4


def test_linear_scale_uses_equal_target_nmae_weighting():
    reference = np.array([[1.0, 1.0], [2.0, 0.0]])
    truth = np.array([[1.0, 1.0], [4.0, 100.0]])
    mask = np.array([[True, True], [True, False]])
    selected = linear_scale(reference, truth, mask, np.array([True, True]))
    grid = np.linspace(0, 3, 30_001)
    objective = np.mean(
        [
            np.abs(grid[:, None] * reference[target, keep] - truth[target, keep]).sum(axis=1)
            / np.abs(truth[target, keep]).sum()
            for target, keep in enumerate(mask)
        ],
        axis=0,
    )
    assert abs(selected - grid[np.argmin(objective)]) < 1e-4
