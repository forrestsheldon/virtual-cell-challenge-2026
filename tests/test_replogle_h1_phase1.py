import numpy as np

from scripts.evaluation.prepare_replogle_h1_phase1 import ARMS, build_arms


def test_phase1_arms_use_pooled_leave_one_out_components_and_missing_fallback() -> None:
    targets = ["A", "B", "C"]
    matched = ["A", "B"]
    source = np.array([[2.0, 4.0], [6.0, 10.0]])
    h1 = np.array([[1.0, 2.0], [3.0, 6.0], [8.0, 12.0]])

    arms = build_arms(targets, matched, source, h1)

    source_global = arms[ARMS.index("source_global")]
    residual = arms[ARMS.index("source_residual")]
    complete = arms[ARMS.index("source_effect")]
    assert np.array_equal(source_global, [[6, 10], [2, 4], [4, 7]])
    assert np.array_equal(residual, [[-4, -6], [4, 6], [0, 0]])
    assert np.array_equal(complete, [[2, 4], [6, 10], [4, 7]])
    assert np.array_equal(
        arms[ARMS.index("h1_global")], [[5.5, 9], [4.5, 7], [2, 4]]
    )
