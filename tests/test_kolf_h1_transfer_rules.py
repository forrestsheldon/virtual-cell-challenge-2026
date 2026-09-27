"""Algebra, boundary, and retrieval controls for the fixed transfer comparison."""

import numpy as np
from cell_eval2.metrics.discrimination import discrimination_score

from scripts.evaluation.kolf_h1_transfer_rules import (
    close_prediction,
    cpm,
    evaluate,
    log_profile,
    retrieval,
    transfer,
)


def test_identity_transfer_reconstructs_donor_and_null():
    control = np.array([20.0, 30.0, 50.0])
    perturbed = np.array([[10.0, 40.0, 50.0], [40.0, 40.0, 20.0]])
    for rule in ["additive", "ratio", "log1p"]:
        np.testing.assert_allclose(
            transfer(perturbed, control, control, rule), perturbed
        )
        np.testing.assert_allclose(
            transfer(control[None], control, 2 * control, rule), 2 * control[None]
        )


def test_log1p_is_exact_shifted_ratio():
    k0 = np.array([0.01, 5.0, 200.0])
    kp = np.array([[0.2, 10.0, 50.0]])
    h0 = np.array([0.1, 20.0, 100.0])
    expected = np.exp(np.log1p(h0) + np.log1p(kp) - np.log1p(k0)) - 1
    np.testing.assert_allclose(transfer(kp, k0, h0, "log1p"), expected)


def test_cpm_and_boundary_mass_reconstruction():
    np.testing.assert_allclose(
        cpm(np.array([[2, 3], [20, 30]])), [[4e5, 6e5], [4e5, 6e5]]
    )
    raw = np.array([[-10.0, 40.0, 60.0]])
    pred, audit = close_prediction(raw)
    np.testing.assert_allclose(pred, [[0.0, 4e5, 6e5]])
    assert audit.negative_genes.iloc[0] == 1
    assert audit.negative_mass_cpm.iloc[0] == 10
    assert audit.raw_total_cpm.iloc[0] == 90
    assert audit.clipped_total_cpm.iloc[0] == 100


def test_retrieval_positive_negative_and_shared_controls():
    truth = np.eye(4)
    np.testing.assert_allclose(retrieval(truth, truth), 1)
    np.testing.assert_allclose(retrieval(np.zeros_like(truth), truth), 0.5)
    np.testing.assert_allclose(retrieval(np.roll(truth, 1, axis=0), truth), 1 / 3)
    assert np.isclose(
        retrieval(np.broadcast_to(np.arange(4), truth.shape), truth).mean(), 0.5
    )


def test_retrieval_matches_pinned_official_kernel():
    rng = np.random.default_rng(12)
    p, y = rng.normal(size=(2, 8, 20))
    targets = np.array([f"t{i}" for i in range(8)])
    scores = discrimination_score(
        pred_bulk=(targets, p),
        real_bulk=(
            np.concatenate([["control"], targets]),
            np.vstack([np.zeros(20), y]),
        ),
        control="control",
        control_source="real",
        distance="cosine",
        rank_denominator="n-1",
        tie_policy="midrank",
        exclude_target_gene=False,
    )
    np.testing.assert_allclose(retrieval(p, y), [scores[t] for t in targets])


def test_metric_exclusion_removes_target_spike():
    rng = np.random.default_rng(5)
    truth = cpm(rng.uniform(1, 20, size=(4, 150)))
    baseline = cpm(np.ones(150))
    keep = np.ones(150, dtype=bool)
    keep[:4] = False
    perturbed = truth.copy()
    perturbed[:, :4] += 1e5
    result = evaluate(perturbed, truth, baseline, keep, np.ones_like(truth, dtype=bool))
    np.testing.assert_allclose(result.cosine, 1)
    np.testing.assert_allclose(result.squared_error, 0)
    np.testing.assert_allclose(result.signed_top100, 1)
    assert np.isfinite(log_profile(np.zeros(150))).all()


def test_ratio_zero_fallback_and_zero_destination():
    from scripts.evaluation.kolf_h1_transfer_rules import transfer

    k0 = np.array([0.0, 0.0, 2.0, 3.0])
    kp = np.array([[0.0, 1.0, 4.0, 6.0]])
    h0 = np.array([5.0, 8.0, 0.0, 4.0])
    np.testing.assert_allclose(transfer(kp, k0, h0, "ratio"), [[5.0, 8.0, 0.0, 8.0]])


def test_stochastic_rounding_preserves_expectation_and_integer_support():
    from scripts.evaluation.kolf_h1_transfer_rules import stochastic_pseudobulk

    rates = np.tile([0.0, 0.02, 0.5, 0.9, 1.0, 1.37, 50.25], (10000, 1))
    counts = stochastic_pseudobulk(rates, np.random.default_rng(3))
    assert np.issubdtype(counts.dtype, np.integer)
    assert np.all(counts >= 400 * np.floor(rates))
    assert np.all(counts <= 400 * np.ceil(rates))
    np.testing.assert_allclose(counts.mean(axis=0) / 400, rates[0], atol=0.001)
    assert np.abs(np.rint(rates) - rates).mean() > np.abs(counts / 400 - rates).mean()
