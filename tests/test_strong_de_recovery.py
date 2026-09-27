from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.linear_response.kernel import expected_decoded_sum, log_pseudobulk
from scripts.linear_response.strong_de_recovery import (
    benjamini_hochberg,
    decoder_tangent,
    signed_topk,
    stable_strong_mask,
)


def test_benjamini_hochberg_restores_original_order() -> None:
    observed = benjamini_hochberg(np.array([0.04, 0.001, 0.03, 0.8]))
    assert np.allclose(observed, [0.053333333333, 0.004, 0.053333333333, 0.8])


def test_decoder_tangent_matches_one_sided_finite_difference() -> None:
    raw = np.array([[4, 0, 2, 0], [0, 5, 1, 3], [2, 1, 0, 2]])
    direction = np.array([-0.3, 0.2, -0.1, 0.4])
    step = 1e-6
    baseline = log_pseudobulk(expected_decoded_sum(raw, direction, 0.0))
    finite = (
        log_pseudobulk(expected_decoded_sum(raw, direction, step)) - baseline
    ) / step
    observed = decoder_tangent(raw, direction)
    assert np.allclose(observed, finite, atol=2e-5, rtol=2e-5)


def test_signed_topk_penalizes_wrong_genes_and_signs() -> None:
    prediction = np.array([9.0, -8.0, -7.0, 6.0, 1.0])
    strong = np.array([True, True, True, False, False])
    truth_sign = np.array([1, -1, 1, 0, 0])
    result = signed_topk(prediction, strong, truth_sign, None)
    assert result["n_strong"] == 3
    assert result["unsigned_recall"] == 1
    assert result["signed_recovery"] == 2 / 3
    assert result["sign_given_recovered"] == 2 / 3


def test_signed_topk_excludes_perturbed_target() -> None:
    prediction = np.array([100.0, 5.0, -4.0, 1.0])
    strong = np.array([False, True, True, False])
    truth_sign = np.array([0, 1, -1, 0])
    result = signed_topk(prediction, strong, truth_sign, 0)
    assert result["signed_recovery"] == 1


def test_stability_requires_four_splits_matching_full_sign() -> None:
    full_lfc = np.array([[0.8, -0.7]])
    p_adj = np.array([[0.001, 0.001]])
    group_counts = np.array([400, 400])
    split_counts = np.full((5, 2), 200)
    full_sums = np.array([[400.0, 400.0], [800.0, 200.0]])
    split_a = np.empty((5, 2, 2))
    split_a[:, 0] = [200.0, 200.0]
    split_a[:, 1] = [400.0, 100.0]
    # Break the second gene's sign agreement in two splits by changing the
    # independently split control denominator.
    split_a[0, 0, 1] = 50.0
    split_a[1, 0, 1] = 50.0
    stable, count = stable_strong_mask(
        full_lfc, p_adj, full_sums, split_a, group_counts, split_counts
    )
    assert count.tolist() == [[5, 3]]
    assert stable.tolist() == [[True, False]]
