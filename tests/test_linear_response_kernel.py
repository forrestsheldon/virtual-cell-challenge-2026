from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from cell_eval2.prep import bulk_lognorm_means
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.linear_response.kernel import (
    BULK_TARGET_SUM,
    decode_largest_remainder,
    decode_lfc_dependent_round,
    decode_lfc_largest_remainder,
    decode_multinomial,
    decoder_lfc_tangent,
    decoder_lfc_tangents,
    expected_decoded_cpm_mean,
    expected_decoded_lfc,
    expected_decoded_sum,
    expected_lfc_decoded_sum,
    expected_log_rate_cpm,
    expected_log_rate_cpms,
    expected_restricted_cpm,
    fit_covariance_columns,
    intended_target_shift,
    knockdown_depths,
    log_pseudobulk,
    log_rate_lfc_tangents,
    nonnegative_l1_scale,
    normalize_response_columns,
    pca_covariance_columns,
    sample_balanced_controls,
    solve_output_matched_amplitude,
    write_prediction,
)


def test_bulk_scale_matches_cell_eval2_and_not_10k() -> None:
    counts = np.array([[4.0, 12.0], [3.0, 1.0]])
    pooled = counts.sum(axis=0)
    observed = log_pseudobulk(pooled)
    expected = bulk_lognorm_means(pooled[None, :], BULK_TARGET_SUM)[0]
    assert np.allclose(observed, expected)
    assert not np.allclose(observed, log_pseudobulk(pooled, 10_000))


def test_knockdown_depth_excludes_held_target_and_sign_failures() -> None:
    control = np.array([100.0, 100.0, 100.0])
    perturbed = np.array(
        [[10.0, 100.0, 100.0], [100.0, 20.0, 100.0], [100.0, 100.0, 200.0]]
    )
    table = knockdown_depths(["a", "b", "c"], [0, 1, 2], control, perturbed)
    assert table["eligible_knockdown_donor"].tolist() == [True, True, False]
    assert np.isclose(
        table.loc[0, "leave_one_out_knockdown_depth"],
        table.loc[1, "observed_knockdown_depth"],
    )
    assert np.isclose(
        table.loc[1, "leave_one_out_knockdown_depth"],
        table.loc[0, "observed_knockdown_depth"],
    )


def test_output_matching_recovers_known_amplitude() -> None:
    raw = np.array([[8, 2, 1], [4, 5, 1], [7, 2, 2]], dtype=float)
    direction = np.array([0.4, -0.1, 0.05])
    known = -1.25
    baseline = log_pseudobulk(raw.sum(axis=0))[0]
    shifted = log_pseudobulk(expected_decoded_sum(raw, direction, known))[0]
    recovered, realized = solve_output_matched_amplitude(
        raw, direction, 0, shifted - baseline
    )
    assert np.isclose(recovered, known)
    assert np.isclose(realized, shifted - baseline)
    assert intended_target_shift(0.1, -1.0) < 0


def test_expected_decoder_is_mean_of_sampled_decoders() -> None:
    raw = np.array([[30, 10, 5], [10, 20, 15]], dtype=float)
    direction = np.array([0.2, -0.1, 0.05])
    expected = expected_decoded_sum(raw, direction, -0.7)
    draws = np.mean(
        [
            np.asarray(
                decode_multinomial(raw, direction, -0.7, np.random.default_rng(seed)).sum(axis=0)
            ).ravel()
            for seed in range(2_000)
        ],
        axis=0,
    )
    assert np.allclose(draws, expected, atol=0.2)


def test_expected_lfc_tangent_matches_finite_difference() -> None:
    raw = np.array([[30, 10, 5], [10, 20, 15]], dtype=float)
    direction = np.array([-1.0, 0.3, -0.2])
    step = 1e-5
    assert np.allclose(
        decoder_lfc_tangent(raw, direction),
        expected_decoded_lfc(raw, direction, step) / step,
        atol=2e-5,
        rtol=2e-5,
    )


def test_restricted_decoder_matches_full_decoder() -> None:
    raw = np.array(
        [[30, 10, 5, 17, 0], [10, 20, 15, 2, 8]], dtype=float
    )
    panel = np.array([0, 2, 4])
    restricted_direction = np.array([-1.0, 0.3, -0.2])
    full_direction = np.zeros(raw.shape[1])
    full_direction[panel] = restricted_direction
    amplitude = 0.7
    expected = expected_decoded_cpm_mean(raw, full_direction, amplitude)[panel]
    observed = expected_restricted_cpm(
        raw[:, panel], raw.sum(axis=1), restricted_direction, amplitude
    )
    assert np.allclose(observed, expected)


def test_restricted_tangents_match_full_tangents() -> None:
    raw = np.array(
        [[30, 10, 5, 17, 0], [10, 20, 15, 2, 8]], dtype=float
    )
    panel = np.array([0, 2, 4])
    directions = np.array([[-1.0, 0.3, -0.2], [0.1, -0.4, 0.2]])
    full_directions = np.zeros((len(directions), raw.shape[1]))
    full_directions[:, panel] = directions
    expected = decoder_lfc_tangents(raw, full_directions)[:, panel]
    observed = decoder_lfc_tangents(
        raw[:, panel], directions, full_totals=raw.sum(axis=1)
    )
    assert np.allclose(observed, expected)


def test_log_rate_tangent_matches_exact_restricted_decoder() -> None:
    raw = np.array(
        [[30, 10, 5, 17, 0], [10, 20, 15, 2, 8]], dtype=float
    )
    panel = np.array([0, 2, 4])
    direction = np.array([-1.0, 0.3, -0.2])
    totals = raw.sum(axis=1)
    baseline = expected_log_rate_cpm(raw[:, panel], totals, direction, 0)
    step = 1e-6
    shifted = expected_log_rate_cpm(raw[:, panel], totals, direction, step)
    finite = np.log2(shifted / baseline) / step
    observed = log_rate_lfc_tangents(
        raw[:, panel], totals, direction[None, :]
    )[0]
    assert np.allclose(observed, finite, atol=2e-6, rtol=2e-6)


def test_vectorized_log_rate_decoder_matches_scalar_decoder() -> None:
    raw = np.array(
        [[30, 10, 5, 17, 0], [10, 20, 15, 2, 8]], dtype=float
    )
    panel = np.array([0, 2, 4])
    directions = np.array([[-1.0, 0.3, -0.2], [0.2, -0.4, 0.1]])
    amplitudes = np.array([0.7, 1.2])
    totals = raw.sum(axis=1)
    expected = np.vstack(
        [
            expected_log_rate_cpm(
                raw[:, panel], totals, direction, amplitude
            )
            for direction, amplitude in zip(directions, amplitudes, strict=True)
        ]
    )
    observed = expected_log_rate_cpms(
        raw[:, panel], totals, directions, amplitudes, batch_size=1
    )
    assert np.allclose(observed, expected)


def test_largest_remainder_zero_is_identity_and_shift_preserves_totals() -> None:
    raw = sparse.csr_matrix([[30, 10, 5], [10, 20, 15]], dtype=np.int32)
    direction = np.array([-1.0, 0.3, -0.2])
    identity = decode_largest_remainder(raw, direction, 0.0)
    shifted = decode_largest_remainder(raw, direction, 0.7)
    assert (identity != raw).nnz == 0
    assert np.array_equal(
        np.asarray(shifted.sum(axis=1)).ravel(), np.asarray(raw.sum(axis=1)).ravel()
    )
    assert not (shifted.data == 0).any()


def test_lfc_decoder_is_identity_at_zero_and_preserves_support_and_totals() -> None:
    raw = sparse.csr_matrix([[30, 10, 0], [0, 20, 15]], dtype=np.int32)
    identity = decode_lfc_largest_remainder(raw, np.zeros(3))
    shifted = decode_lfc_largest_remainder(raw, np.array([-1.0, 0.3, 0.8]))
    assert (identity != raw).nnz == 0
    assert np.array_equal(
        np.asarray(shifted.sum(axis=1)).ravel(), np.asarray(raw.sum(axis=1)).ravel()
    )
    assert np.all(shifted.toarray()[raw.toarray() == 0] == 0)
    assert not (shifted.data == 0).any()


def test_dependent_rounding_preserves_totals_support_and_expected_mean() -> None:
    raw = sparse.csr_matrix([[30, 10, 5], [10, 20, 15]], dtype=np.int32)
    lfc = np.array([-1.0, 0.3, 0.8])
    expected = expected_lfc_decoded_sum(raw, lfc)
    draws = np.asarray(
        [
            decode_lfc_dependent_round(
                raw, lfc, np.random.default_rng(seed)
            ).sum(axis=0)
            for seed in range(2_000)
        ]
    ).reshape(2_000, 3)
    one = decode_lfc_dependent_round(raw, lfc, np.random.default_rng(0))
    assert np.array_equal(
        np.asarray(one.sum(axis=1)).ravel(), np.asarray(raw.sum(axis=1)).ravel()
    )
    assert np.all(one.toarray()[raw.toarray() == 0] == 0)
    assert np.allclose(draws.mean(axis=0), expected, atol=0.04)


def test_expected_lfc_decoder_composes_additive_effects() -> None:
    raw = sparse.csr_matrix([[30, 10, 5], [10, 20, 15]], dtype=np.int32)
    first = np.array([-1.0, 0.3, -0.2])
    second = np.array([0.2, -0.1, 0.4])
    observed = expected_lfc_decoded_sum(raw, first + second)
    factors = np.exp2(first + second)
    dense = raw.toarray() * factors
    totals = np.asarray(raw.sum(axis=1))
    expected = np.sum(dense / dense.sum(axis=1, keepdims=True) * totals, axis=0)
    assert np.allclose(observed, expected)


def test_response_normalization_sets_target_to_minus_one() -> None:
    covariance = np.array([[2.0, 0.3], [0.3, 3.0], [0.4, -0.2]])
    response, diagonal = normalize_response_columns(covariance, [0, 1])
    assert np.array_equal(diagonal, [2.0, 3.0])
    assert np.array_equal(response[[0, 1], [0, 1]], [-1.0, -1.0])


def test_nonnegative_l1_scale_matches_brute_force() -> None:
    direction = np.array([1.0, -2.0, 0.5, 4.0])
    truth = 0.7 * direction + np.array([0.05, 0.0, -0.05, 0.1])
    mask = np.array([True, True, True, True])
    observed = nonnegative_l1_scale(direction, truth, mask)
    grid = np.linspace(0, 1.5, 30_001)
    error = np.abs(grid[:, None] * direction - truth).sum(axis=1)
    assert np.isclose(np.abs(observed * direction - truth).sum(), error.min(), atol=1e-4)


def test_balanced_source_draw_is_seeded_and_seed_sensitive() -> None:
    obs = pd.DataFrame(
        {
            "target_gene": ["non-targeting"] * 60,
            "guide_id": ["g1"] * 30 + ["g2"] * 30,
        }
    )
    first = sample_balanced_controls(
        obs, ["g1", "g2"], np.random.default_rng(3), cells_per_target=20
    )
    repeated = sample_balanced_controls(
        obs, ["g1", "g2"], np.random.default_rng(3), cells_per_target=20
    )
    changed = sample_balanced_controls(
        obs, ["g1", "g2"], np.random.default_rng(4), cells_per_target=20
    )
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, changed)
    assert obs.iloc[first]["guide_id"].value_counts().to_dict() == {"g1": 10, "g2": 10}


def test_model_and_null_decodes_use_independent_streams_on_shared_rows() -> None:
    raw = np.array([[30, 10, 5], [10, 20, 15]], dtype=float)
    direction = np.array([0.2, -0.1, 0.05])
    model = decode_multinomial(raw, direction, -0.7, np.random.default_rng(10))
    null = decode_multinomial(raw, direction, 0.0, np.random.default_rng(11))
    assert np.array_equal(np.asarray(model.sum(axis=1)).ravel(), raw.sum(axis=1))
    assert np.array_equal(np.asarray(null.sum(axis=1)).ravel(), raw.sum(axis=1))
    assert not np.array_equal(model.toarray(), null.toarray())


def test_pca_columns_match_dense_covariance() -> None:
    rng = np.random.default_rng(4)
    components, _ = np.linalg.qr(rng.normal(size=(6, 3)))
    components = components.T
    singular = np.array([4.0, 2.0, 1.0])
    observed = pca_covariance_columns(components, singular, 11, [1, 4])
    eigenvalues = singular**2 / 10
    covariance = components.T @ np.diag(eigenvalues) @ components
    assert np.allclose(observed, covariance[:, [1, 4]])


def test_covariance_columns_match_dense_covariance() -> None:
    values = np.array(
        [[1, 3, 2], [2, 1, 4], [5, 2, 1], [3, 6, 2]], dtype=np.float64
    )
    raw = np.expm1(values)
    raw *= 10_000 / raw.sum(axis=1, keepdims=True)
    data = ad.AnnData(sparse.csr_matrix(raw))
    observed, half0, half1, mean = fit_covariance_columns(
        data, np.arange(4), [0, 2]
    )
    transformed = np.log1p(raw)
    expected = np.cov(transformed, rowvar=False, ddof=1)[:, [0, 2]]
    assert np.allclose(observed, expected)
    assert np.allclose(mean, transformed.mean(axis=0))
    assert half0 is None and half1 is None


def test_writer_eliminates_stored_zeros(tmp_path: Path) -> None:
    block = sparse.csr_matrix(np.ones((400, 2), dtype=np.int32))
    block.data[0] = 0
    path = tmp_path / "prediction.h5ad"
    write_prediction({"a": block}, ["a", "b"], path)
    result = ad.read_h5ad(path)
    assert result.obs.columns.tolist() == ["target_gene"]
    assert not (result.X.data == 0).any()
