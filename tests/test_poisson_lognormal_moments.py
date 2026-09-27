from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.linear_response.poisson_lognormal_moments import (
    FactorialMoments,
    lognormal_covariance_columns,
    state_halves,
)
from scripts.linear_response.regularize_factorial_pln import (
    mean_shrinkage,
    normal_mixture_posterior_mean,
    sampling_variance,
)


def test_factorial_moments_match_dense_formula() -> None:
    raw = np.array(
        [[2, 1, 3], [4, 2, 1], [1, 5, 2], [3, 0, 4]], dtype=np.float64
    )
    totals = np.array([8, 9, 10, 11], dtype=np.float64)
    targets = np.array([0, 2])
    moments = FactorialMoments.zeros(3, 2)
    moments.update(raw, totals, targets)
    mean, second = moments.finish()

    rates = raw / totals[:, None]
    expected = rates.T @ rates[:, targets]
    expected[targets, np.arange(2)] -= (
        raw[:, targets] / np.square(totals[:, None])
    ).sum(axis=0)
    assert np.allclose(mean, rates.mean(axis=0))
    assert np.allclose(second, expected / len(raw))


def test_lognormal_moment_identity_recovers_covariance() -> None:
    covariance = np.array(
        [[0.2, 0.04, -0.03], [0.04, 0.1, 0.02], [-0.03, 0.02, 0.15]]
    )
    mean = np.array([0.1, 0.2, 0.3])
    targets = np.array([0, 2])
    second = np.outer(mean, mean[targets]) * np.exp(covariance[:, targets])
    observed, ratio = lognormal_covariance_columns(mean, second, targets)
    assert np.allclose(observed, covariance[:, targets])
    assert np.allclose(ratio, np.exp(covariance[:, targets]))


def test_factorial_estimator_recovers_simulated_poisson_lognormal_column() -> None:
    rng = np.random.default_rng(4)
    n = 50_000
    covariance = np.array(
        [[0.16, 0.04, -0.03], [0.04, 0.12, 0.02], [-0.03, 0.02, 0.10]]
    )
    mean = np.array([0.08, 0.12, 0.18])
    latent = rng.multivariate_normal(
        np.log(mean) - np.diag(covariance) / 2, covariance, size=n
    )
    totals = rng.integers(500, 2_000, size=n).astype(np.float64)
    counts = rng.poisson(totals[:, None] * np.exp(latent)).astype(np.float64)
    moments = FactorialMoments.zeros(3, 1)
    moments.update(counts, totals, np.array([0]))
    observed_mean, observed_second = moments.finish()
    observed, _ = lognormal_covariance_columns(
        observed_mean, observed_second, np.array([0])
    )
    assert np.allclose(observed[:, 0], covariance[:, 0], atol=0.01)


def test_state_halves_are_disjoint_complete_and_batch_balanced() -> None:
    batches = np.array(["a"] * 7 + ["b"] * 8 + ["c"] * 9)
    rows = np.arange(len(batches))
    halves = state_halves(batches, rows, 3)
    assert set(halves) == {0, 1}
    for batch in set(batches):
        counts = np.bincount(halves[batches == batch], minlength=2)
        assert abs(counts[0] - counts[1]) <= 1


def test_mean_shrinkage_preserves_independence_and_targets_low_rates() -> None:
    covariance = np.array([[0.2], [0.2], [0.0]])
    mean = np.array([5e-6, 500e-6, 100e-6])
    observed = mean_shrinkage(covariance, mean, np.array([2]), 20.0)
    assert observed[2, 0] == 0
    assert 0 < observed[0, 0] < observed[1, 0] < covariance[1, 0]


def test_normal_mixture_shrinks_uncertain_coefficient_more() -> None:
    observed = normal_mixture_posterior_mean(
        np.array([0.2, 0.2]),
        np.array([0.01, 0.5]),
        np.array([0.0, 0.2]),
        np.array([0.2, 0.8]),
    )
    assert 0 < observed[1] < observed[0] < 0.2


def test_repeated_half_differences_recover_sampling_variance() -> None:
    split = np.array([[[[1.0]], [[-1.0]]]] * 5)
    cells = np.array([[100, 100]] * 5)
    full, process = sampling_variance(split, cells, 200)
    assert np.allclose(process, 200)
    assert np.allclose(full, 1)
