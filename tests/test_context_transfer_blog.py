import numpy as np
import pandas as pd
from scipy import sparse

from scripts.evaluation.context_transfer_blog import (
    collapse_columns,
    crossfit_predictions,
    deterministic_folds,
    normalized_effects,
    projection_coefficients,
    retrieval_scores,
    single_metrics,
    target_gene_indices,
)
from scripts.evaluation.context_transfer_calibration_v2 import (
    GuideData,
    cross_context_records,
    evaluate_records,
    promoter_class,
)
from scripts.evaluation.context_transfer_h1 import expected_lfc_decoded_sum
from scripts.linear_response.kernel import (
    expected_lfc_decoded_sum as reference_expected_lfc_decoded_sum,
)


def test_duplicate_symbols_are_summed_in_count_space():
    matrix = sparse.csr_matrix([[2, 3, 5], [7, 11, 13]])
    collapsed, genes = collapse_columns(matrix, ["A", "B", "A"])

    assert genes == ["A", "B"]
    np.testing.assert_array_equal(collapsed.toarray(), [[7, 3], [20, 11]])


def test_normalization_uses_complete_native_gene_universe():
    counts = np.asarray([[10, 10, 80], [20, 20, 60]])
    control = np.asarray([10, 10, 80])
    effect, _, _ = normalized_effects(counts, control)

    np.testing.assert_allclose(effect[0], 0)
    np.testing.assert_allclose(effect[1, :2], 1)
    assert effect[1, 2] < 0


def test_target_gene_is_excluded_from_projection_and_metrics():
    genes = ["T", "A", "B"]
    targets = ["T"]
    source = np.asarray([[1.0, 2.0, -3.0]])
    truth = np.asarray([[1_000.0, 2.0, -3.0]])
    excluded = target_gene_indices(targets, genes)
    coefficients = projection_coefficients(source, truth, np.ones(3), excluded)
    scores = single_metrics(source[0], truth[0], excluded[0], 2)

    assert coefficients.oracle_projection.iloc[0] == 1
    assert np.isclose(scores["cosine"], 1)
    assert scores["nae"] == 0


def test_heldout_truth_cannot_change_its_fold_prediction():
    rng = np.random.default_rng(7)
    targets = [f"T{i}" for i in range(10)]
    genes = [*targets, "A", "B", "C"]
    source = rng.normal(size=(10, len(genes)))
    truth = source * np.linspace(0.5, 1.4, 10)[:, None]
    difference = rng.normal(size=len(genes))
    folds = deterministic_folds(targets)
    first, _, _ = crossfit_predictions(targets, genes, source, truth, difference, folds)
    changed_truth = truth.copy()
    changed_truth[0] *= 100
    second, _, _ = crossfit_predictions(
        targets, genes, source, changed_truth, difference, folds
    )

    np.testing.assert_allclose(first["global_median"][0], second["global_median"][0])
    np.testing.assert_allclose(
        first["context_interaction"][0], second["context_interaction"][0]
    )


def test_fold_coefficients_use_median_per_perturbation_projection():
    targets = [f"T{i}" for i in range(10)]
    genes = [*targets, "A", "B"]
    source = np.ones((10, len(genes)))
    slopes = np.arange(1, 11, dtype=float)
    truth = source * slopes[:, None]
    folds = deterministic_folds(targets)
    _, per_target, fold_coefficients = crossfit_predictions(
        targets, genes, source, truth, np.linspace(-1, 1, len(genes)), folds
    )

    for row in fold_coefficients.itertuples(index=False):
        training = per_target.fold != row.heldout_fold
        expected = np.median(per_target.loc[training, "oracle_projection"])
        assert row.global_beta_median == expected


def test_rank_deficient_interaction_is_omitted():
    source = np.asarray([[1.0, 2.0, 3.0]])
    truth = source.copy()
    coefficients = projection_coefficients(source, truth, np.zeros(3), np.asarray([-1]))

    assert coefficients.interaction_rank.iloc[0] == 1
    assert np.isnan(coefficients.interaction_beta0.iloc[0])
    assert np.isnan(coefficients.interaction_beta_d.iloc[0])


def test_retrieval_ranks_the_correct_target_first():
    truth = np.eye(3)
    scores, ranks, top1 = retrieval_scores(truth.copy(), truth, np.full(3, -1))

    np.testing.assert_array_equal(ranks, np.ones(3))
    np.testing.assert_array_equal(scores, np.ones(3))
    assert top1.all()


def test_signed_recovery_requires_overlap_and_correct_sign():
    truth = np.asarray([3.0, -2.0, 0.1, 0.0])
    correct = np.asarray([4.0, -1.0, 0.0, 0.0])
    wrong_sign = np.asarray([4.0, 1.0, 0.0, 0.0])

    assert single_metrics(correct, truth, -1, 2)["signed_recovery"] == 1
    assert single_metrics(wrong_sign, truth, -1, 2)["signed_recovery"] == 0.5


def test_self_and_unchanged_controls_have_expected_endpoints():
    truth = np.asarray([2.0, -1.0, 0.5])
    self_scores = single_metrics(truth, truth, -1, 3)
    unchanged = single_metrics(np.zeros_like(truth), truth, -1, 3)
    coefficients = projection_coefficients(
        truth[None, :], truth[None, :], np.ones(3), np.asarray([-1])
    )

    assert self_scores["cosine"] == 1
    assert self_scores["nae"] == 0
    assert unchanged["nae"] == 1
    assert coefficients.oracle_projection.iloc[0] == 1


def test_vectorized_h1_expected_decoder_matches_frozen_kernel():
    raw = sparse.csr_matrix([[2, 0, 3, 1], [0, 5, 2, 1], [4, 1, 0, 2]])
    lfc = np.asarray([-1.2, 0.3, 1.1, -0.4])

    observed = expected_lfc_decoded_sum(raw, lfc)
    expected = reference_expected_lfc_decoded_sum(raw, lfc)

    np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-13)


def test_promoter_class_distinguishes_promoter_designs():
    assert promoter_class("GENE_a-P1|GENE_b-P1") == "P1"
    assert promoter_class("GENE_a-P1P2|GENE_b-P1P2") == "P1P2"
    assert promoter_class("GENE_P1-1|GENE_P1-2") == "P1"
    assert promoter_class("GENE_without_annotation") is None


def test_cross_context_records_separate_exact_same_and_cross_promoter():
    source = GuideData(
        "source",
        ["A", "B"],
        np.asarray(["A_1-P1", "A_2-P1", "A_3-P2"]),
        np.asarray(["A", "A", "A"]),
        np.asarray(["P1", "P1", "P2"], dtype=object),
        np.ones(3, dtype=int),
        np.zeros((3, 2)),
        np.zeros((3, 2)),
    )
    target = GuideData(
        "target",
        source.genes,
        source.guide_ids.copy(),
        source.targets.copy(),
        source.promoter_classes.copy(),
        source.cells.copy(),
        source.epsilon_lfc.copy(),
        source.log2_cpm1p.copy(),
    )

    records = cross_context_records(source, target)
    counts = pd.Series(record["calibration"] for record in records).value_counts()

    assert counts["cross_context_exact_construct"] == 3
    assert counts["cross_context_same_promoter_different_construct"] == 2
    assert counts["cross_context_cross_promoter_different_construct"] == 4


def test_construct_retrieval_excludes_other_constructs_for_same_target():
    effects = np.asarray(
        [
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
        ]
    )
    data = GuideData(
        "context",
        ["X", "Y", "Z"],
        np.asarray(["X_1-P1", "X_2-P1", "Y_1-P1"]),
        np.asarray(["X", "X", "Y"]),
        np.asarray(["P1", "P1", "P1"], dtype=object),
        np.ones(3, dtype=int),
        effects,
        effects,
    )
    records = [
        {
            "calibration": "test",
            "prediction_index": 0,
            "correct_index": 1,
            "target_gene": "X",
            "prediction_guide": "X_1-P1",
            "truth_guide": "X_2-P1",
            "promoter_relation": "same",
            "promoter_class": "P1",
        }
    ]

    result = evaluate_records(data, data, records, "epsilon_lfc")

    assert result.candidate_wrong_constructs.iloc[0] == 1
    assert result.retrieval_score.iloc[0] == 0.5
