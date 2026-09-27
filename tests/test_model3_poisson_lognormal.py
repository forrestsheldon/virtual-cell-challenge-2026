from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.linear_response.kernel import log_pseudobulk
from scripts.linear_response.model3_poisson_lognormal import (
    inference_latent_modes,
    laplace_mode_numpy,
    normalize_population,
    solve_amplitude,
    tiny_laplace_validation,
)


def test_tiny_laplace_moments_match_quadrature() -> None:
    result = tiny_laplace_validation().iloc[0]
    assert result.absolute_mean_error < 0.01
    assert result.relative_variance_error < 0.05


def test_laplace_mode_has_zero_gradient_and_positive_covariance() -> None:
    m = np.log(np.array([0.2, 0.3, 0.5]))
    loadings = np.array([[0.1, -0.2], [0.2, 0.05], [-0.1, 0.15]])
    counts = np.array([20.0, 31.0, 49.0])
    mode, covariance = laplace_mode_numpy(counts, counts.sum(), m, loadings)
    rates = counts.sum() * np.exp(m + loadings @ mode)
    gradient = mode - loadings.T @ (counts - rates)
    assert np.linalg.norm(gradient) < 1e-8
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_final_inference_reaches_the_map_on_a_tiny_batch() -> None:
    m = jnp.log(jnp.asarray([0.2, 0.3, 0.5]))
    loadings = jnp.asarray([[0.1, -0.2], [0.2, 0.05], [-0.1, 0.15]])
    counts = jnp.asarray([[20.0, 31.0, 49.0], [18.0, 35.0, 47.0]])
    totals = counts.sum(1)
    modes, rates, _ = inference_latent_modes(m, loadings, counts, totals)
    gradient = modes - (counts - rates) @ loadings
    assert np.linalg.norm(np.asarray(gradient), axis=1).max() < 1e-3


def test_population_normalization_sums_marginal_rates_to_one() -> None:
    m = jnp.asarray([-2.0, -1.0, -3.0])
    loadings = jnp.asarray([[0.2, 0.1], [-0.1, 0.3], [0.0, -0.2]])
    normalized = np.asarray(normalize_population(m, loadings))
    q = np.exp(normalized + 0.5 * np.square(np.asarray(loadings)).sum(1))
    assert np.isclose(q.sum(), 1.0)


def test_model3_amplitude_recovers_exact_bulk_target_shift() -> None:
    baseline = np.array([100.0, 200.0, 300.0])
    direction = np.array([0.2, 0.1, -0.05])
    amplitude, realized, shifted = solve_amplitude(baseline, direction, 0, -0.4)
    observed = log_pseudobulk(shifted)[0] - log_pseudobulk(baseline)[0]
    assert amplitude < 0
    assert np.isclose(realized, -0.4)
    assert np.isclose(observed, -0.4)
