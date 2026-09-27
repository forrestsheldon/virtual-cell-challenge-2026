"""Estimate control-only linear and nonlinear natural-expression responses."""

from __future__ import annotations

import hashlib
import json
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scripts.linear_response.forcing_ceiling import DE_TABLE, STRONG_TRUTH, load_truth
from scripts.linear_response.kernel import log1cp10k, split_control_halves
from scripts.linear_response.strong_de_recovery import (
    benjamini_hochberg,
    derangement_null,
    pairwise_wrong_target_scores,
    randomized_gene_null,
    ranked_genes,
    signed_topk,
    summarize_null,
)

ROOT = Path(__file__).resolve().parents[2]
H1_PATH = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
DERIVED = ROOT / "data/derived/linear_response"
REPORT = ROOT / "reports/linear-response-three-models"
MODEL1 = DERIVED / "model1_expected_profiles.npz"
MODEL2 = DERIVED / "model2_expected_profiles.npz"
STRICT_CONTROLS = REPORT / "h1_strict_controls.csv"

N_STATE_AXES = 3
MIN_STRONG = 10
LOW_QUANTILE = 0.10
CHUNK_SIZE = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nuisance_design(
    obs: pd.DataFrame,
    state_scores: np.ndarray,
    *,
    include_state: bool,
) -> np.ndarray:
    """Return intercept, batch, guide and optional state-score nuisance columns."""
    batch = pd.get_dummies(obs["batch"].astype(str), drop_first=True, dtype=float)
    guide = pd.get_dummies(obs["guide_id"].astype(str), drop_first=True, dtype=float)
    columns = [np.ones((len(obs), 1)), batch.to_numpy(), guide.to_numpy()]
    if include_state:
        centered = state_scores - state_scores.mean(axis=0)
        scale = centered.std(axis=0, ddof=1)
        columns.append(centered / scale)
    return np.column_stack(columns).astype(np.float64)


def residualize(orthonormal_nuisance: np.ndarray, values: np.ndarray) -> np.ndarray:
    return values - orthonormal_nuisance @ (orthonormal_nuisance.T @ values)


def solve_quadratic_coefficients(
    residual: np.ndarray,
    squared_residual: np.ndarray,
    cross_linear: np.ndarray,
    cross_squared: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve independent two-predictor regressions for every target."""
    s11 = np.square(residual).sum(axis=0)
    s12 = (residual * squared_residual).sum(axis=0)
    s22 = np.square(squared_residual).sum(axis=0)
    determinant = s11 * s22 - np.square(s12)
    floor = np.finfo(np.float64).eps * np.maximum(s11 * s22, 1.0)
    if np.any(determinant <= floor):
        bad = np.flatnonzero(determinant <= floor)
        raise RuntimeError(f"quadratic dose design is singular for targets {bad.tolist()}")
    beta_linear = (
        s22[:, None] * cross_linear - s12[:, None] * cross_squared
    ) / determinant[:, None]
    beta_squared = (
        s11[:, None] * cross_squared - s12[:, None] * cross_linear
    ) / determinant[:, None]
    return beta_linear, beta_squared


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(left), np.nan),
        where=denominator > 0,
    )


def make_log_matrix(
    data: ad.AnnData,
    strict_rows: np.ndarray,
    path: Path,
    control_mean: np.ndarray,
    state_components: np.ndarray,
) -> tuple[np.memmap, np.ndarray]:
    """Materialize one temporary normalized matrix and its three state scores."""
    matrix = np.memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(len(strict_rows), data.n_vars),
    )
    scores = np.empty((len(strict_rows), len(state_components)), dtype=np.float64)
    for start in range(0, len(strict_rows), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(strict_rows))
        values = log1cp10k(data.X[strict_rows[start:stop]].tocsr())
        matrix[start:stop] = values
        scores[start:stop] = (values - control_mean) @ state_components.T
        if (start // CHUNK_SIZE) % 8 == 0 or stop == len(strict_rows):
            print(f"normalized controls {stop}/{len(strict_rows)}", flush=True)
    matrix.flush()
    return matrix, scores


def feature_outcome_cross(
    matrix: np.memmap,
    positions: np.ndarray,
    features: np.ndarray,
    output_indices: np.ndarray,
    label: str,
) -> np.ndarray:
    cross = np.zeros((features.shape[1], len(output_indices)), dtype=np.float64)
    for start in range(0, len(positions), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(positions))
        block = np.asarray(matrix[positions[start:stop]], dtype=np.float64)
        cross += features[start:stop].T @ block[:, output_indices]
        if (start // CHUNK_SIZE) % 8 == 0 or stop == len(positions):
            print(f"{label}: outcome cross {stop}/{len(positions)}", flush=True)
    return cross


def fit_half(
    matrix: np.memmap,
    positions: np.ndarray,
    target_indices: np.ndarray,
    output_indices: np.ndarray,
    nuisance: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Fit linear and quadratic low-expression contrasts on one control half."""
    orthonormal = np.linalg.qr(nuisance[positions], mode="reduced")[0]
    target_values = np.asarray(
        matrix[positions][:, target_indices], dtype=np.float64
    )
    residual = residualize(orthonormal, target_values)
    squared = residualize(orthonormal, np.square(residual))
    features = np.column_stack([residual, squared])
    cross = feature_outcome_cross(
        matrix, positions, features, output_indices, label
    )
    n_targets = len(target_indices)
    cross_linear = cross[:n_targets]
    cross_squared = cross[n_targets:]
    linear_coefficients = cross_linear / np.square(residual).sum(axis=0)[:, None]
    beta_linear, beta_squared = solve_quadratic_coefficients(
        residual, squared, cross_linear, cross_squared
    )
    median = np.median(residual, axis=0)
    low = np.quantile(residual, LOW_QUANTILE, axis=0)
    delta = low - median
    delta_squared = np.square(low) - np.square(median)
    linear_prediction = delta[:, None] * linear_coefficients
    quadratic_prediction = (
        delta[:, None] * beta_linear + delta_squared[:, None] * beta_squared
    )
    diagnostics = pd.DataFrame(
        {
            "target_index": np.arange(n_targets),
            "cells": len(positions),
            "residual_sd": residual.std(axis=0, ddof=1),
            "residual_median": median,
            "residual_q10": low,
            "natural_low_shift": delta,
            "linear_quadratic_cosine": cosine_rows(
                linear_prediction, quadratic_prediction
            ),
        }
    )
    return linear_prediction, quadratic_prediction, diagnostics


def evaluate_predictions(
    predictions: dict[str, np.ndarray],
    half_predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    targets: list[str],
    de_genes: list[str],
    lfc: np.ndarray,
    stable: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    de_lookup = {gene: index for index, gene in enumerate(de_genes)}
    excluded = [de_lookup.get(target) for target in targets]
    signs = np.sign(lfc).astype(np.int8)
    per_target_rows = []
    summary_rows = []
    for model_index, (name, prediction) in enumerate(predictions.items()):
        ranking = ranked_genes(prediction)
        observed = np.full(len(targets), np.nan)
        model_rows = []
        stability = cosine_rows(*half_predictions[name])
        for target_index, target in enumerate(targets):
            result = signed_topk(
                prediction[target_index],
                stable[target_index],
                signs[target_index],
                excluded[target_index],
            )
            eligible = int(result["n_strong"]) >= MIN_STRONG
            if eligible:
                observed[target_index] = float(result["signed_recovery"])
            model_rows.append(
                {
                    "model": name,
                    "target_gene": target,
                    "eligible": eligible,
                    "split_half_direction_cosine": stability[target_index],
                    **result,
                }
            )
        pairwise = pairwise_wrong_target_scores(
            prediction, ranking, stable, signs, excluded
        )
        if not np.allclose(
            observed[np.isfinite(observed)],
            np.diag(pairwise)[np.isfinite(observed)],
        ):
            raise AssertionError(f"{name}: pairwise diagonal differs from direct score")
        for target_index, row in enumerate(model_rows):
            if not row["eligible"]:
                row.update(
                    wrong_target_mean=np.nan,
                    wrong_target_p_upper=np.nan,
                )
                continue
            null = np.delete(pairwise[target_index], target_index)
            null = null[np.isfinite(null)]
            row["wrong_target_mean"] = float(null.mean())
            row["wrong_target_p_upper"] = float(
                (1 + np.sum(null >= observed[target_index])) / (1 + len(null))
            )
        eligible_rows = [row for row in model_rows if row["eligible"]]
        adjusted = benjamini_hochberg(
            np.asarray([row["wrong_target_p_upper"] for row in eligible_rows])
        )
        for row, q_value in zip(eligible_rows, adjusted, strict=True):
            row["wrong_target_q_bh"] = float(q_value)
        for row in model_rows:
            if not row["eligible"]:
                row["wrong_target_q_bh"] = np.nan
        per_target_rows.extend(model_rows)

        wrong_target = derangement_null(
            pairwise, np.random.default_rng(np.random.SeedSequence([0, 91, model_index]))
        )
        random_gene = randomized_gene_null(
            prediction,
            ranking,
            stable,
            signs,
            excluded,
            np.random.default_rng(np.random.SeedSequence([0, 92, model_index])),
        )
        summary = summarize_null(
            name,
            observed,
            wrong_target,
            random_gene,
            np.random.default_rng(np.random.SeedSequence([0, 93, model_index])),
        )
        eligible_stability = stability[np.isfinite(observed)]
        summary["median_split_half_direction_cosine"] = float(
            np.nanmedian(eligible_stability)
        )
        summary["q10_split_half_direction_cosine"] = float(
            np.nanquantile(eligible_stability, 0.1)
        )
        summary_rows.append(summary)
    return pd.DataFrame(per_target_rows), pd.DataFrame(summary_rows)


def main() -> None:
    with np.load(MODEL1, allow_pickle=False) as model1, np.load(
        MODEL2, allow_pickle=False
    ) as model2:
        targets = model1["target_gene"].astype(str).tolist()
        genes = model1["gene_names"].astype(str).tolist()
        source_order = model1["source_order"].astype(int)
        control_mean = model1["control_mean"].astype(np.float64)
        components = model2["components"].astype(np.float64)[:N_STATE_AXES]
    de_genes, de_indices, lfc, stable = load_truth(targets, genes)
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    target_indices = np.asarray(
        [gene_lookup[target] for target in targets], dtype=int
    )
    strict_guides = pd.read_csv(STRICT_CONTROLS)["guide_id"].astype(str).tolist()

    data = ad.read_h5ad(H1_PATH, backed="r")
    with tempfile.NamedTemporaryFile(
        prefix="dose_response_", suffix=".float32", dir=DERIVED, delete=False
    ) as temporary:
        work_path = Path(temporary.name)
    try:
        control_mask = data.obs["target_gene"].astype(str).eq("non-targeting") & data.obs[
            "guide_id"
        ].astype(str).isin(strict_guides)
        strict_rows = np.flatnonzero(control_mask.to_numpy())
        halves = split_control_halves(data.obs, strict_rows, strict_guides)
        matrix, state_scores = make_log_matrix(
            data, strict_rows, work_path, control_mean, components
        )
        control_obs = data.obs.iloc[strict_rows]
        detection = np.asarray(
            (data.X[strict_rows][:, target_indices].tocsr() > 0).mean(axis=0)
        ).ravel()

        predictions: dict[str, np.ndarray] = {}
        half_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        diagnostic_rows = []
        for nuisance_name, include_state in [
            ("batch_guide", False),
            ("batch_guide_top3", True),
        ]:
            nuisance = nuisance_design(
                control_obs, state_scores, include_state=include_state
            )
            half_results = []
            for half in (0, 1):
                positions = np.flatnonzero(halves == half)
                linear, quadratic, diagnostics = fit_half(
                    matrix,
                    positions,
                    target_indices,
                    de_indices,
                    nuisance,
                    f"{nuisance_name} half {half}",
                )
                half_results.append((linear, quadratic))
                diagnostics.insert(0, "half", half)
                diagnostics.insert(0, "nuisance", nuisance_name)
                diagnostics["target_gene"] = targets
                diagnostics["control_detection_fraction"] = detection
                diagnostic_rows.append(diagnostics)
            linear_name = f"dose_linear_{nuisance_name}"
            quadratic_name = f"dose_quadratic_{nuisance_name}"
            linear_halves = (half_results[0][0], half_results[1][0])
            quadratic_halves = (half_results[0][1], half_results[1][1])
            predictions[linear_name] = np.mean(linear_halves, axis=0)
            predictions[quadratic_name] = np.mean(quadratic_halves, axis=0)
            half_predictions[linear_name] = linear_halves
            half_predictions[quadratic_name] = quadratic_halves
        del matrix
    finally:
        data.file.close()
        work_path.unlink(missing_ok=True)

    predictions = {name: values[source_order] for name, values in predictions.items()}
    half_predictions = {
        name: (left[source_order], right[source_order])
        for name, (left, right) in half_predictions.items()
    }
    per_target, summary = evaluate_predictions(
        predictions, half_predictions, targets, de_genes, lfc, stable
    )
    summary["wrong_target_p_bonferroni_models"] = np.minimum(
        1.0, len(summary) * summary["wrong_target_p_upper"]
    )
    diagnostics = pd.concat(diagnostic_rows, ignore_index=True)

    outputs = {
        "summary": REPORT / "control_dose_response_summary.csv",
        "per_target": REPORT / "control_dose_response_per_target.csv",
        "fit": REPORT / "control_dose_response_fit.csv",
    }
    summary.to_csv(outputs["summary"], index=False)
    per_target.to_csv(outputs["per_target"], index=False)
    diagnostics.to_csv(outputs["fit"], index=False)
    manifest = {
        "kind": "control-only natural-expression dose-response diagnostic",
        "control_cells": 32_616,
        "control_split": "deterministic guide-by-batch halves",
        "nuisance_models": ["batch + guide", "batch + guide + top 3 control PCs"],
        "dose": {
            "coordinate": "target log1CP10k residual",
            "contrast": f"residual median to residual q{int(100 * LOW_QUANTILE)}",
            "curves": ["linear", "quadratic"],
        },
        "truth_access": "evaluation only; no perturbed cells enter curve fitting",
        "score": "signed top-K stable strong-DE recovery",
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                H1_PATH,
                MODEL1,
                MODEL2,
                STRICT_CONTROLS,
                DE_TABLE,
                STRONG_TRUTH,
            ]
        },
        "outputs": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for name, path in outputs.items()
        },
        "software": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ["anndata", "numpy", "pandas", "scipy"]
            },
        },
    }
    (REPORT / "control_dose_response_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(summary.sort_values("mean_signed_recovery", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
