import numpy as np

from scripts.evaluation.prepare_replogle_h1_transfer import (
    collapse_duplicate_symbols,
    expand_target_rows,
    loo_components,
    reconstruct_integer_counts,
)


def test_count_reconstruction_rounds_float_means_back_to_sums() -> None:
    counts = np.array([[0, 3, 11], [7, 0, 5]])
    cells = np.array([7, 13])
    raw_mean = (counts / cells[:, None]).astype(np.float32)

    reconstructed, residual = reconstruct_integer_counts(raw_mean, cells)

    assert np.array_equal(reconstructed, counts)
    assert residual < 1e-5


def test_duplicate_symbols_are_summed_in_first_appearance_order() -> None:
    matrix = np.array([[1, 2, 4, 8], [3, 5, 7, 11]])

    collapsed, symbols = collapse_duplicate_symbols(matrix, ["A", "B", "A", "C"])

    assert symbols.tolist() == ["A", "B", "C"]
    assert np.array_equal(collapsed, [[5, 2, 8], [10, 5, 11]])


def test_leave_one_out_global_and_residual_reconstruct_every_effect() -> None:
    effects = np.array([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]])

    global_loo, residual = loo_components(effects)

    assert np.allclose(global_loo, [[7, 9], [5, 7], [3, 5]])
    assert np.allclose(global_loo + residual, effects)


def test_missing_targets_receive_zero_specific_effect() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]])

    expanded = expand_target_rows(["A", "C"], values, ["A", "B", "C"])

    assert np.array_equal(expanded, [[1, 2], [0, 0], [3, 4]])
