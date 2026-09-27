from __future__ import annotations

import numpy as np

from scripts.evaluation.crossfit_empirical_scale import choose_scale, target_folds
from scripts.evaluation.plot_oracle_de_nmae import full_de_oracle_indices


def test_target_folds_are_balanced_reproducible_and_order_only() -> None:
    targets = [f"g{index}" for index in range(9)]
    first = target_folds(targets)
    repeated = target_folds(targets)
    assert np.array_equal(first, repeated)
    assert sorted(np.bincount(first).tolist()) == [4, 5]


def test_scale_selection_uses_only_training_targets() -> None:
    errors = np.array([[0.0, 10.0], [0.0, 10.0], [100.0, 0.0]])
    assert choose_scale(errors, np.array([True, True, False])) == 0
    assert choose_scale(errors, np.array([False, False, True])) == 1


def test_per_target_oracle_is_selected_from_full_de_error() -> None:
    errors = np.array([[1.0, 0.5, 0.8], [1.0, 0.9, 0.7], [1.0, 0.2, 0.3]])
    counts = np.array([10, 20, 9])
    assert np.array_equal(full_de_oracle_indices(errors, counts), [1, 2, -1])
