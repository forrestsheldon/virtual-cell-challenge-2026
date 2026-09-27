"""Check common-support construction and count-axis alignment."""

import numpy as np

from scripts.evaluation.all_source_h1_transfer import Counts, axes, extract


def test_matched_axes_use_intersection_and_align_counts():
    h = Counts(
        "H1",
        ["g2", "g1", "g3"],
        ["t2", "t1"],
        np.array([[2, 1, 3], [5, 4, 6]]),
        np.array([20, 10, 30]),
        np.array([400, 400]),
        {},
    )
    s = Counts(
        "source",
        ["g3", "g1"],
        ["t1", "t3"],
        np.array([[60, 40], [90, 70]]),
        np.array([300, 100]),
        np.array([20, 30]),
        {},
    )
    targets, genes = axes([s], h)
    assert targets == ["t1"] and genes == ["g1", "g3"]
    a, c, cells = extract(s, targets, genes)
    np.testing.assert_array_equal(a, [[40, 60]])
    np.testing.assert_array_equal(c, [100, 300])
    np.testing.assert_array_equal(cells, [20])
    truth, _, _ = extract(h, targets, genes)
    np.testing.assert_array_equal(truth, [[4, 6]])


def test_source_order_does_not_change_matched_panel():
    h = Counts("H1", ["a", "b", "c"], ["x", "y", "z"], None, None, None, {})
    s1 = Counts("s1", ["c", "b"], ["z", "x"], None, None, None, {})
    s2 = Counts("s2", ["b", "a"], ["y", "x"], None, None, None, {})
    assert axes([s1, s2], h) == axes([s2, s1], h) == (["x"], ["b"])
