from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.linear_response.forcing_ceiling import (
    anchored_omp_factorized,
    deterministic_gene_folds,
    fold_score,
    heldout_lfc_metrics,
)


def test_gene_folds_are_deterministic_and_name_based() -> None:
    genes = [f"g{index}" for index in range(100)]
    first = deterministic_gene_folds(genes)
    second = deterministic_gene_folds(list(reversed(genes)))[::-1]
    assert np.array_equal(first, second)
    assert set(first) == {0, 1}


def test_anchored_omp_recovers_secondary_force_out_of_sample() -> None:
    rng = np.random.default_rng(2)
    left = rng.normal(size=(80, 3))
    codes = np.array(
        [
            [1.0, 0.0, 0.0, 0.3],
            [0.0, 1.0, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.1],
        ]
    )
    truth = left @ (codes[:, [0, 2]] @ np.array([-2.0, 1.5]))
    predictions, trace = anchored_omp_factorized(
        left[:40], left[40:], codes, truth[:40], 0, support_sizes=(1, 2)
    )
    assert trace[-1]["support"] == [0, 2]
    assert np.allclose(predictions[2], truth[40:], atol=1e-9)
    assert trace[-1]["coefficients"][0] < 0


def test_fold_score_ranks_within_each_heldout_fold() -> None:
    prediction = np.array([10.0, -9.0, 8.0, -7.0, 0.1, 0.2])
    stable = np.array([True, True, False, True, False, False])
    signs = np.array([1, -1, 0, -1, 0, 0])
    folds = np.array([0, 1, 0, 1, 0, 1])
    result = fold_score(prediction, stable, signs, folds, None)
    assert result["n_strong"] == 3
    assert result["signed_recovery"] == 1


def test_fold_score_excludes_target_gene() -> None:
    prediction = np.array([100.0, 3.0, -2.0, 0.0])
    stable = np.array([False, True, True, False])
    signs = np.array([0, 1, -1, 0])
    folds = np.array([0, 0, 1, 1])
    result = fold_score(prediction, stable, signs, folds, 0)
    assert result["signed_recovery"] == 1


def test_heldout_lfc_metrics_exclude_target_and_reward_truth() -> None:
    truth = np.array([100.0, 2.0, -1.0, 0.5])
    prediction = np.array([-100.0, 2.0, -1.0, 0.5])
    result = heldout_lfc_metrics(prediction, truth, 0)
    assert np.isclose(result["heldout_lfc_cosine"], 1)
    assert np.isclose(result["heldout_lfc_spearman"], 1)
    assert np.isclose(result["heldout_lfc_nmae"], 0)
    assert np.isclose(result["heldout_lfc_explained_energy"], 1)
