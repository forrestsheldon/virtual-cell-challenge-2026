from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.linear_response.model3_full_poisson_lognormal as model3_full
from scripts.linear_response.model3_full_poisson_lognormal import (
    batch_moment_sums,
    covariance_columns,
    distribution_change,
    exact_sign_flip_p,
    fit_m_step,
    fit_m_step_fixed_d,
    joint_laplace_mode,
    make_parameters,
    negative_joint,
    negative_joint_change,
    population_parameters,
    posterior_blocks,
    posterior_mean_rates,
    sample_full_prior,
    solve_lograte_amplitude,
)


def example():
    mean_rates = np.array([0.20, 0.30, 0.50])
    loadings = np.array([[0.10, -0.05], [-0.08, 0.12], [0.03, 0.06]])
    d = np.array([0.04, 0.03, 0.02])
    parameters = make_parameters(mean_rates, loadings, d)
    mu, loadings_jax, d_jax = population_parameters(parameters)
    counts = jnp.asarray([[20.0, 31.0, 49.0]])
    totals = counts.sum(1)
    return parameters, mu, loadings_jax, d_jax, counts, totals


def test_population_normalization_includes_residual_variance() -> None:
    _, mu, loadings, d, _, _ = example()
    rates = np.exp(
        np.asarray(mu) + 0.5 * (np.square(np.asarray(loadings)).sum(1) + np.asarray(d))
    )
    assert np.isclose(rates.sum(), 1.0)


def test_distribution_change_is_invariant_to_loading_rotation() -> None:
    parameters, _, _, _, _, _ = example()
    rotation = jnp.asarray([[0.0, -1.0], [1.0, 0.0]])
    rotated = {**parameters, "loadings": parameters["loadings"] @ rotation}
    mean_change, covariance_change = distribution_change(parameters, rotated)
    assert mean_change < 1e-12
    assert covariance_change < 1e-8


def test_block_mode_and_covariance_match_dense_calculation() -> None:
    _, mu, loadings, d, counts, totals = example()
    mode = joint_laplace_mode(mu, loadings, d, counts, totals)
    assert mode.converged.item()
    assert not mode.line_search_failed.item()

    z = np.asarray(mode.z[0])
    x = np.asarray(mode.x[0])
    loadings = np.asarray(loadings)
    d = np.asarray(d)
    rates = totals.item() * np.exp(x)
    u = x - np.asarray(mu) - loadings @ z
    gradient_z = z - loadings.T @ (u / d)
    gradient_x = u / d + rates - np.asarray(counts[0])
    assert np.max(np.abs(np.r_[gradient_z, gradient_x])) < 2e-4

    d_inv = 1.0 / d
    dense_hessian = np.block(
        [
            [
                np.eye(loadings.shape[1]) + loadings.T @ (d_inv[:, None] * loadings),
                -loadings.T * d_inv[None, :],
            ],
            [
                -d_inv[:, None] * loadings,
                np.diag(d_inv + rates),
            ],
        ]
    )
    dense_covariance = np.linalg.inv(dense_hessian)
    v_zz, v_xz, v_xx_diagonal = posterior_blocks(mode, jnp.asarray(loadings))
    rank = loadings.shape[1]
    assert np.allclose(np.asarray(v_zz[0]), dense_covariance[:rank, :rank], atol=1e-6)
    assert np.allclose(np.asarray(v_xz[0]), dense_covariance[rank:, :rank], atol=1e-6)
    assert np.allclose(
        np.asarray(v_xx_diagonal[0]), np.diag(dense_covariance)[rank:], atol=1e-6
    )


def test_joint_newton_mode_decreases_the_objective() -> None:
    _, mu, loadings, d, counts, totals = example()
    initial_z = jnp.zeros((1, loadings.shape[1]))
    initial_x = jnp.broadcast_to(mu, counts.shape)
    initial = negative_joint(initial_z, initial_x, counts, totals, mu, loadings, d)
    mode = joint_laplace_mode(mu, loadings, d, counts, totals)
    final = negative_joint(mode.z, mode.x, counts, totals, mu, loadings, d)
    assert final.item() < initial.item()


def test_stable_objective_change_matches_direct_evaluation() -> None:
    _, mu, loadings, d, counts, totals = example()
    z = jnp.asarray([[0.04, -0.03]])
    x = jnp.broadcast_to(mu, counts.shape) + jnp.asarray([[0.02, -0.01, 0.03]])
    delta_z = jnp.asarray([[-0.007, 0.004]])
    delta_x = jnp.asarray([[0.006, -0.003, -0.004]])
    direct = negative_joint(
        z + delta_z, x + delta_x, counts, totals, mu, loadings, d
    ) - negative_joint(z, x, counts, totals, mu, loadings, d)
    stable = negative_joint_change(
        z,
        x,
        delta_z,
        delta_x,
        counts,
        totals,
        mu,
        loadings,
        d,
    )
    assert np.allclose(np.asarray(stable), np.asarray(direct), atol=1e-12)


def test_e_step_resumes_exactly_from_a_partial_atomic_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parameters, _, _, _, _, _ = example()
    rows = np.arange(128, dtype=np.int64)
    counts = np.tile(np.array([[20, 31, 49]], dtype=np.int32), (len(rows), 1))
    data = ad.AnnData(csr_matrix(counts))
    checkpoint = tmp_path / "e_step.npz"
    original_infer = model3_full.INFER_BATCH
    calls = 0

    def interrupt_second_batch(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original_infer(*args)

    monkeypatch.setattr(model3_full, "INFER_BATCH", interrupt_second_batch)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        model3_full.run_e_step(
            data,
            rows,
            parameters,
            collect_moments=True,
            checkpoint_path=checkpoint,
            checkpoint_every=1,
        )
    with np.load(checkpoint, allow_pickle=False) as saved:
        assert int(saved["completed_batches"]) == 1

    monkeypatch.setattr(model3_full, "INFER_BATCH", original_infer)
    resumed_sums, resumed_diagnostics, resumed_values = model3_full.run_e_step(
        data,
        rows,
        parameters,
        collect_moments=True,
        checkpoint_path=checkpoint,
        resume_checkpoint=True,
        checkpoint_every=1,
    )
    fresh_sums, fresh_diagnostics, fresh_values = model3_full.run_e_step(
        data, rows, parameters, collect_moments=True
    )
    assert resumed_sums is not None and fresh_sums is not None
    assert resumed_sums.n_cells == fresh_sums.n_cells == 128
    for name in ["x", "z", "zz", "xz", "x2"]:
        assert np.array_equal(getattr(resumed_sums, name), getattr(fresh_sums, name))
    assert np.array_equal(resumed_values, fresh_values)
    assert resumed_diagnostics == fresh_diagnostics


def test_posterior_mean_rate_retains_laplace_uncertainty() -> None:
    _, mu, loadings, d, counts, totals = example()
    mode = joint_laplace_mode(mu, loadings, d, counts, totals)
    mean_rates = np.asarray(posterior_mean_rates(mode, loadings))
    assert np.all(mean_rates > np.asarray(mode.rates))


def test_m_step_recovers_complete_data_parameters() -> None:
    rng = np.random.default_rng(7)
    cells = 20_000
    mean_rates = np.array([0.15, 0.20, 0.25, 0.40])
    loadings = np.array([[0.18, -0.04], [-0.10, 0.15], [0.08, 0.12], [-0.06, -0.08]])
    d = np.array([0.035, 0.025, 0.045, 0.030])
    truth = make_parameters(mean_rates, loadings, d)
    mu, _, _ = population_parameters(truth)
    z = rng.normal(size=(cells, 2))
    x = (
        np.asarray(mu)[None, :]
        + z @ loadings.T
        + rng.normal(scale=np.sqrt(d), size=(cells, 4))
    )
    moments = {
        "x": x.mean(0),
        "z": z.mean(0),
        "zz": z.T @ z / cells,
        "xz": x.T @ z / cells,
        "x2": np.square(x).mean(0),
    }
    initial = make_parameters(
        np.array([0.18, 0.18, 0.27, 0.37]), loadings * 0.8, d * 1.3
    )
    fitted, result = fit_m_step(initial, moments, max_iterations=100, tolerance=1e-6)
    fitted_mu, fitted_loadings, fitted_d = population_parameters(fitted)
    assert float(result.state.error) < 3e-5
    assert np.allclose(np.asarray(fitted_mu), np.asarray(mu), atol=0.025)
    assert np.allclose(np.asarray(fitted_loadings), loadings, atol=0.025)
    assert np.allclose(np.asarray(fitted_d), d, atol=0.01)


def test_moment_accumulation_and_full_covariance_column() -> None:
    _, mu, loadings, d, counts, totals = example()
    mode = joint_laplace_mode(mu, loadings, d, counts, totals)
    moments = batch_moment_sums(mode, loadings)
    assert moments["n_cells"] == 1
    assert moments["xz"].shape == (3, 2)
    assert np.all(moments["x2"] >= np.square(np.asarray(mode.x[0])))

    columns = covariance_columns(np.asarray(loadings), np.asarray(d), np.array([1]))
    shared = np.asarray(loadings) @ np.asarray(loadings)[1]
    assert np.allclose(np.delete(columns[:, 0], 1), np.delete(shared, 1))
    assert np.isclose(columns[1, 0], shared[1] + np.asarray(d)[1])


def test_fixed_d_m_step_and_rank_zero_model() -> None:
    rng = np.random.default_rng(9)
    mean_rates = np.array([0.2, 0.3, 0.5])
    residual_variance = np.array([0.02, 0.03, 0.04])
    rank_zero = make_parameters(mean_rates, np.empty((3, 0)), residual_variance)
    mu, loadings, d = population_parameters(rank_zero)
    counts = jnp.asarray([[20.0, 31.0, 49.0]])
    mode = joint_laplace_mode(mu, loadings, d, counts, counts.sum(1))
    assert mode.converged.item()
    assert mode.z.shape == (1, 0)

    cells = 10_000
    z = rng.normal(size=(cells, 1))
    true_loadings = np.array([[0.12], [-0.08], [0.03]])
    truth = make_parameters(mean_rates, true_loadings, residual_variance)
    true_mu, _, _ = population_parameters(truth)
    x = (
        np.asarray(true_mu)[None, :]
        + z @ true_loadings.T
        + rng.normal(scale=np.sqrt(residual_variance), size=(cells, 3))
    )
    moments = {
        "x": x.mean(0),
        "z": z.mean(0),
        "zz": z.T @ z / cells,
        "xz": x.T @ z / cells,
        "x2": np.square(x).mean(0),
    }
    initial = make_parameters(mean_rates, true_loadings * 0.7, residual_variance)
    initial_d = np.asarray(population_parameters(initial)[2])
    fitted, result = fit_m_step_fixed_d(
        initial, moments, max_iterations=100, tolerance=1e-6
    )
    _, fitted_loadings, fitted_d = population_parameters(fitted)
    assert float(result.state.error) < 3e-5
    assert np.allclose(np.asarray(fitted_d), initial_d)
    assert np.allclose(np.asarray(fitted_loadings), true_loadings, atol=0.025)


def test_prior_sampling_and_unclipped_output_matching() -> None:
    parameters, _, _, _, _, _ = example()
    rng = np.random.default_rng(31)
    totals = np.full(20_000, 200)
    sampled = sample_full_prior(parameters, totals, rng)
    sampled_fraction = sampled.sum(0) / sampled.sum()
    assert np.allclose(sampled_fraction, [0.2, 0.3, 0.5], atol=0.01)

    baseline = np.array([100.0, 200.0, 700.0])
    direction = np.array([0.05, 0.20, 0.03])
    amplitude, realized, shifted = solve_lograte_amplitude(baseline, direction, 1, -1.5)
    assert amplitude < 0
    assert np.isclose(realized, -1.5, atol=1e-8)
    assert np.isclose(shifted.sum(), baseline.sum())


def test_exact_guide_sign_flip_has_declared_four_guide_resolution() -> None:
    assert np.isclose(exact_sign_flip_p(np.ones(4)), 1 / 16)
    assert np.isclose(exact_sign_flip_p(np.array([1.0, -1.0])), 0.75)
