import numpy as np

from scripts.evaluation.prepare_replogle_h1_reliability import (
    dirichlet_lfc,
    expand_effects,
    stable_sign_mask,
)


def test_dirichlet_lfc_shrinks_toward_control_and_handles_unmeasured_genes() -> None:
    counts = np.array([[40, 0, 9]])
    control = np.array([20, 20, 0])
    weak = dirichlet_lfc(counts, np.array([49]), control, 40, 10)
    strong = dirichlet_lfc(counts, np.array([49]), control, 40, 1_000)

    assert abs(strong[0, 0]) < abs(weak[0, 0])
    assert abs(strong[0, 1]) < abs(weak[0, 1])
    assert weak[0, 2] == 0
    assert strong[0, 2] == 0


def test_stable_sign_requires_both_halves_in_enough_repeats() -> None:
    full = np.array([[1.0, -1.0, 1.0]])
    halves = np.array(
        [
            [[[1, -1, 1]], [[2, -2, -1]]],
            [[[1, -1, 1]], [[2, 2, 1]]],
            [[[1, -1, -1]], [[2, -2, -1]]],
        ]
    )
    assert stable_sign_mask(full, halves, 2).tolist() == [[True, True, False]]
    assert stable_sign_mask(full, halves, 3).tolist() == [[True, False, False]]


def test_expand_effects_uses_mean_only_for_missing_targets() -> None:
    effects = np.array([[2.0, 4.0], [6.0, 10.0]])
    result = expand_effects(["A", "B", "C"], ["A", "B"], effects)
    assert np.array_equal(result, [[2, 4], [6, 10], [4, 7]])
