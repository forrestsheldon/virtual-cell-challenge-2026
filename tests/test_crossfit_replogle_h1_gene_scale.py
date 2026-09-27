import numpy as np

from scripts.evaluation.crossfit_replogle_h1_gene_scale import (
    build_intended,
    context_features,
    fit_context_model,
    gene_scales,
    permute_within_deciles,
)


def test_gene_scales_recover_independent_slopes_without_shrinkage() -> None:
    source = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]])
    truth = np.array([0.5, 1.5, -2.0])
    scales, _, prior = gene_scales(source, source * truth, 0.0)
    assert prior == 0
    assert np.allclose(scales, truth)


def test_infinite_gene_prior_equals_pooled_scale() -> None:
    source = np.arange(1, 13, dtype=float).reshape(4, 3)
    target = source * np.array([0.5, 1.0, 2.0])
    scales, global_scale, prior = gene_scales(source, target, np.inf)
    assert np.isinf(prior)
    assert np.all(scales == global_scale)


def test_context_model_recovers_feature_mediated_scales() -> None:
    source = np.arange(1, 21, dtype=float).reshape(4, 5)
    features = np.column_stack(
        [np.ones(5), np.linspace(-1, 1, 5), np.array([-1, 1, -1, 1, 0])]
    )
    beta = np.array([0.8, 0.2, -0.3])
    target = source * (features @ beta)
    fitted, penalty = fit_context_model(source, target, features, 0.0)
    assert penalty == 0
    assert np.allclose(fitted, beta)


def test_context_features_have_intercept_and_standardized_covariates() -> None:
    features, mean, difference = context_features(
        np.array([10, 20, 40, 80]), np.array([80, 30, 20, 10])
    )
    assert np.all(features[:, 0] == 1)
    assert np.allclose(features[:, 1:].mean(axis=0), 0)
    assert np.allclose(features[:, 1:].std(axis=0), 1)
    assert mean.shape == difference.shape == (4,)


def test_permutation_stays_within_mean_expression_deciles() -> None:
    values = np.arange(100)
    means = np.arange(100)
    permuted = permute_within_deciles(values, means, np.random.default_rng(3))
    for block in range(10):
        rows = slice(10 * block, 10 * (block + 1))
        assert set(permuted[rows]) == set(values[rows])


def test_build_intended_uses_h1_average_and_zero_residual_for_missing_target() -> None:
    targets = ["A", "B", "C"]
    matched = ["A", "B"]
    h1_average = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    raw = {
        "one_scale": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "gene_scales": np.array([[2.0, 4.0], [6.0, 8.0]]),
        "expression_context": np.array([[3.0, 6.0], [9.0, 12.0]]),
    }
    shrunk = {name: value / 2 for name, value in raw.items()}
    intended = build_intended(targets, matched, h1_average, raw, shrunk)
    assert np.array_equal(intended[0, 0], [2.0, 4.0])
    assert np.array_equal(intended[2, 1], [12.0, 16.0])
    assert np.array_equal(intended[4, 0], [2.0, 4.0])
    assert np.array_equal(
        intended[:, 2], np.repeat([[5.0, 6.0]], len(intended), axis=0)
    )
