import numpy as np

from scripts.evaluation.crossfit_replogle_h1_scale import (
    build_intended,
    choose_ridge_alpha,
    crossfit_scales,
    fit_ridge,
    predict_ridge,
)


def test_ridge_prediction_uses_training_imputation_and_scaling() -> None:
    train = np.array([[0.0, 1.0], [1.0, np.nan], [2.0, 3.0], [3.0, 4.0]])
    outcome = np.array([0.0, 1.0, 2.0, 3.0])
    model = fit_ridge(train, outcome, alpha=1.0)
    prediction = predict_ridge(model, np.array([[4.0, np.nan]]))
    assert prediction.shape == (1,)
    assert np.isfinite(prediction[0])


def test_inner_alpha_choice_is_deterministic() -> None:
    features = np.arange(24, dtype=float).reshape(12, 2)
    outcome = np.linspace(-1, 1, 12)
    names = [f"T{i}" for i in range(12)]
    assert choose_ridge_alpha(features, outcome, names, 3) == choose_ridge_alpha(
        features, outcome, names, 3
    )


def test_outer_heldout_truth_cannot_change_its_scale_predictions() -> None:
    names = ["A", "B", "C", "D"]
    source = np.arange(48, dtype=float).reshape(4, 12) / 10 + 1
    target = source * np.array([0.5, 1.0, 1.5, 2.0])[:, None]
    strong = np.ones_like(source, dtype=bool)
    features = np.arange(24, dtype=float).reshape(4, 6)
    folds = np.array([0, 0, 1, 1])
    first, _ = crossfit_scales(
        names, source, target, strong, features, folds, seed=7
    )
    changed = target.copy()
    changed[folds == 0] *= 100
    second, _ = crossfit_scales(
        names, source, changed, strong, features, folds, seed=7
    )
    for model in ("global_all", "global_strong", "ridge"):
        assert np.allclose(first[model][folds == 0], second[model][folds == 0])


def test_build_intended_uses_h1_global_and_zero_residual_for_missing_target() -> None:
    targets = ["A", "B", "C"]
    matched = ["A", "B"]
    global_effect = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    raw = np.array([[2.0, 4.0], [6.0, 8.0]])
    shrunk = raw / 2
    scales = {
        "unscaled": np.ones(2),
        "global_all": np.array([2.0, 2.0]),
        "global_strong": np.array([3.0, 3.0]),
        "ridge": np.array([4.0, 5.0]),
    }
    intended = build_intended(
        targets, matched, global_effect, raw, shrunk, scales, scales
    )
    assert np.array_equal(intended[0], global_effect)
    assert np.array_equal(intended[1, 0], [3.0, 6.0])
    assert np.array_equal(intended[4, 1], [33.0, 44.0])
    assert np.array_equal(intended[:, 2], np.repeat([[5.0, 6.0]], 9, axis=0))
