"""Evaluate fast Poisson-lognormal LR directions against normalized LR."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from platformdirs import user_cache_path

from scripts.linear_response.evaluate_ladder import (
    load_truth,
    scales,
    score_directions,
)
from scripts.linear_response.kernel import (
    decoder_lfc_tangents,
    expected_log_rate_cpm,
    expected_log_rate_cpms,
    expected_restricted_cpm,
    log_rate_lfc_tangents,
    sample_balanced_controls,
)

ROOT = Path(__file__).resolve().parents[2]
CONTROLS = Path(user_cache_path("vcc2026-h1-benchmark")) / "h1_controls.h5ad"
STRICT_GUIDES = ROOT / "reports/linear-response-three-models/h1_strict_controls.csv"
EMPIRICAL = ROOT / "data/derived/linear_response/ladder/empirical.npz"
STATE_MODEL = (
    ROOT
    / "data/derived/linear_response/state_diversity/state_balanced_covariance.npz"
)
FACTORIAL = (
    ROOT
    / "data/derived/linear_response/sequencing_model/factorial_moment_response.npz"
)
REGULARIZED = (
    ROOT
    / "data/derived/linear_response/sequencing_model/regularized_factorial_response.npz"
)
LAPLACE = ROOT / "data/derived/linear_response/model3_fit.npz"
DERIVED = ROOT / "data/derived/linear_response/sequencing_model"
REPORT = ROOT / "reports/linear-response-sequencing-model"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load_models() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], set[str]]:
    with np.load(EMPIRICAL, allow_pickle=False) as empirical, np.load(
        STATE_MODEL, allow_pickle=False
    ) as state, np.load(FACTORIAL, allow_pickle=False) as factorial, np.load(
        REGULARIZED, allow_pickle=False
    ) as regularized, np.load(LAPLACE, allow_pickle=False) as laplace:
        metadata = {
            key: empirical[key]
            for key in [
                "target_gene",
                "output_gene",
                "fit_gene",
                "full_gene_index",
                "target_fit_index",
                "source_rows",
            ]
        }
        state_names = state["model"].astype(str).tolist()
        factorial_names = factorial["model"].astype(str).tolist()
        regularized_names = regularized["model"].astype(str).tolist()
        fit_indices = metadata["full_gene_index"].astype(int)
        target_indices = metadata["target_fit_index"].astype(int)
        loadings = laplace["loadings"][fit_indices].astype(np.float64)
        laplace_response = []
        for target in target_indices:
            column = loadings @ loadings[target]
            laplace_response.append(-column / column[target])
        models = {
            "control_log1cp10k": empirical["response"].astype(np.float64),
            "control_factorial_pln": factorial["response"][
                factorial_names.index("control_factorial_pln")
            ].astype(np.float64),
            **{
                name: regularized["response"][index].astype(np.float64)
                for index, name in enumerate(regularized_names)
            },
            "control_laplace_rank50": np.asarray(laplace_response),
            "all_combined_log1cp10k": state["response"][
                state_names.index("all_combined")
            ].astype(np.float64),
            "all_combined_factorial_pln": factorial["response"][
                factorial_names.index("all_combined_factorial_pln")
            ].astype(np.float64),
        }
    latent = {
        "control_factorial_pln",
        *regularized_names,
        "control_laplace_rank50",
        "all_combined_factorial_pln",
    }
    return models, metadata, latent


def decoded_predictions(
    models: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    latent_models: set[str],
    truth: np.ndarray,
    stable: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame, np.ndarray]:
    checkpoint = DERIVED / "expected_profile_diagnostics.npz"
    scale_path = REPORT / "downstream_scale.csv"
    inputs = [
        EMPIRICAL,
        STATE_MODEL,
        FACTORIAL,
        REGULARIZED,
        LAPLACE,
        STRICT_GUIDES,
    ]
    if (
        checkpoint.exists()
        and scale_path.exists()
        and checkpoint.stat().st_mtime >= max(path.stat().st_mtime for path in inputs)
    ):
        with np.load(checkpoint, allow_pickle=False) as saved:
            if saved["model"].astype(str).tolist() == list(models):
                tangent = {
                    name: saved["tangent"][index].astype(np.float64)
                    for index, name in enumerate(models)
                }
                exact = {
                    name: saved["exact_lfc"][index].astype(np.float64)
                    for index, name in enumerate(models)
                }
                return (
                    tangent,
                    exact,
                    pd.read_csv(scale_path),
                    saved["source_rows"].astype(int),
                )
    output_count = len(metadata["output_gene"])
    fit_indices = metadata["full_gene_index"].astype(int)
    tangent = {
        name: np.empty((len(metadata["target_gene"]), output_count))
        for name in models
    }
    controls = ad.read_h5ad(CONTROLS, backed="r")
    try:
        guides = pd.read_csv(STRICT_GUIDES)["guide_id"].astype(str).tolist()
        source_rows = np.sort(
            sample_balanced_controls(
                controls.obs,
                guides,
                np.random.default_rng(np.random.SeedSequence([0, 221])),
            )
        )
        raw_full = controls.X[source_rows].tocsr()
        totals = np.asarray(raw_full.sum(axis=1)).ravel()
        raw = raw_full[:, fit_indices]
        normalized_names = [name for name in models if name not in latent_models]
        latent_names = [name for name in models if name in latent_models]
        normalized = decoder_lfc_tangents(
            raw,
            np.vstack(
                [models[name][target] for name in normalized_names for target in range(len(truth))]
            ),
            full_totals=totals,
        ).reshape(len(normalized_names), len(truth), -1)
        latent = log_rate_lfc_tangents(
            raw,
            totals,
            np.vstack(
                [models[name][target] for name in latent_names for target in range(len(truth))]
            ),
        ).reshape(len(latent_names), len(truth), -1)
        for index, name in enumerate(normalized_names):
            tangent[name] = normalized[index, :, :output_count]
        for index, name in enumerate(latent_names):
            tangent[name] = latent[index, :, :output_count]
    finally:
        controls.file.close()

    scale_table, transferred, _ = scales(tangent, truth, stable)
    exact = {name: np.empty_like(truth) for name in models}
    baseline = expected_log_rate_cpm(
        raw, totals, np.zeros(len(fit_indices)), 0
    )[:output_count]
    latent_names = [name for name in models if name in latent_models]
    latent_pairs = [
        (name, target) for name in latent_names for target in range(len(truth))
    ]
    latent_exact = expected_log_rate_cpms(
        raw,
        totals,
        np.vstack([models[name][target] for name, target in latent_pairs]),
        np.asarray([transferred[name][target] for name, target in latent_pairs]),
    )
    for prediction, (name, target) in zip(latent_exact, latent_pairs, strict=True):
        exact[name][target] = np.log2(
            (prediction[:output_count] + 1e-9) / (baseline + 1e-9)
        )
    raw_dense = raw.toarray()
    for target in range(len(truth)):
        for name, response in models.items():
            if name in latent_models:
                continue
            else:
                shifted = expected_restricted_cpm(
                    raw_dense, totals, response[target], transferred[name][target]
                )
            exact[name][target] = np.log2(
                (shifted[:output_count] + 1e-9) / (baseline + 1e-9)
            )
    return tangent, exact, scale_table, source_rows


def main() -> None:
    models, metadata, latent_models = load_models()
    cached_predictions = DERIVED / "expected_profile_diagnostics.npz"
    cached_tables = [
        REPORT / "expected_profile_per_target.csv",
        REPORT / "expected_profile_summary.csv",
    ]
    cache_inputs = [
        EMPIRICAL,
        STATE_MODEL,
        FACTORIAL,
        REGULARIZED,
        LAPLACE,
        STRICT_GUIDES,
    ]
    use_cached_tables = (
        cached_predictions.exists()
        and all(path.exists() for path in cached_tables)
        and cached_predictions.stat().st_mtime
        >= max(path.stat().st_mtime for path in cache_inputs)
    )
    targets = metadata["target_gene"].astype(str).tolist()
    genes = metadata["output_gene"].astype(str).tolist()
    truth, stable = load_truth(targets, genes)
    tangent, exact, scale_table, source_rows = decoded_predictions(
        models, metadata, latent_models, truth, stable
    )
    _, transferred, oracle = scales(tangent, truth, stable)
    REPORT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "scale": REPORT / "downstream_scale.csv",
        "per_target": REPORT / "expected_profile_per_target.csv",
        "summary": REPORT / "expected_profile_summary.csv",
        "geometry": REPORT / "response_geometry_per_target.csv",
        "geometry_summary": REPORT / "response_geometry_summary.csv",
        "predictions": DERIVED / "expected_profile_diagnostics.npz",
    }
    if use_cached_tables:
        per_target = pd.read_csv(outputs["per_target"])
        summary = pd.read_csv(outputs["summary"])
    else:
        per_target, summary = score_directions(
            tangent,
            exact,
            {},
            transferred,
            oracle,
            truth,
            stable,
            targets,
            genes,
        )
        summary["coordinate"] = [
            "latent log rate" if name in latent_models else "log1p CP10K"
            for name in summary["model"]
        ]
        summary["transductive"] = summary["model"].str.startswith("all_combined")

    with np.load(FACTORIAL, allow_pickle=False) as factorial, np.load(
        REGULARIZED, allow_pickle=False
    ) as regularized, np.load(LAPLACE, allow_pickle=False) as laplace:
        factorial_names = factorial["model"].astype(str).tolist()
        q_by_model = {
            name: factorial["mean_rate"][factorial_names.index(name)].astype(
                np.float64
            )
            for name in factorial_names
        }
        control_q = q_by_model["control_factorial_pln"]
        q_by_model.update(
            {name: control_q for name in regularized["model"].astype(str)}
        )
        full_q = np.exp(
            laplace["m"].astype(np.float64)
            + 0.5 * np.square(laplace["loadings"].astype(np.float64)).sum(1)
        )
        q_by_model["control_laplace_rank50"] = full_q[
            metadata["full_gene_index"].astype(int)
        ]
    target_indices = metadata["target_fit_index"].astype(int)
    geometry_rows = []
    for name, response in models.items():
        for target, direction in enumerate(response):
            downstream = np.arange(len(direction)) != target_indices[target]
            values = np.abs(direction[downstream])
            row = {
                "model": name,
                "target_gene": targets[target],
                "median_abs_downstream_coefficient": np.median(values),
                "q90_abs_downstream_coefficient": np.quantile(values, 0.9),
                "q99_abs_downstream_coefficient": np.quantile(values, 0.99),
                "fraction_abs_downstream_above_0p1": np.mean(values >= 0.1),
                "downstream_norm": np.linalg.norm(values),
            }
            if name in q_by_model:
                q = q_by_model[name]
                gamma = transferred[name][target]
                row["panel_mean_rate_sum"] = q.sum()
                row["mean_rate_weighted_direction"] = q @ direction
                row["unnormalized_total_rate_ratio_at_gamma"] = (
                    1 - q.sum() + q @ np.exp(np.clip(gamma * direction, -80, 80))
                )
            geometry_rows.append(row)
    geometry = pd.DataFrame(geometry_rows)
    geometry_summary = (
        geometry.groupby("model", sort=False)
        .agg(
            median_abs_downstream_coefficient=(
                "median_abs_downstream_coefficient",
                "median",
            ),
            median_q90_abs_downstream_coefficient=(
                "q90_abs_downstream_coefficient",
                "median",
            ),
            median_q99_abs_downstream_coefficient=(
                "q99_abs_downstream_coefficient",
                "median",
            ),
            median_fraction_abs_downstream_above_0p1=(
                "fraction_abs_downstream_above_0p1",
                "median",
            ),
            median_downstream_norm=("downstream_norm", "median"),
            median_unnormalized_total_rate_ratio_at_gamma=(
                "unnormalized_total_rate_ratio_at_gamma",
                "median",
            ),
            q05_unnormalized_total_rate_ratio_at_gamma=(
                "unnormalized_total_rate_ratio_at_gamma",
                lambda values: values.quantile(0.05),
            ),
            q95_unnormalized_total_rate_ratio_at_gamma=(
                "unnormalized_total_rate_ratio_at_gamma",
                lambda values: values.quantile(0.95),
            ),
        )
        .reset_index()
    )
    scale_table["target_gene"] = np.asarray(targets)[
        scale_table["target_index"].to_numpy(dtype=int)
    ]
    scale_table.to_csv(outputs["scale"], index=False)
    per_target.to_csv(outputs["per_target"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    geometry.to_csv(outputs["geometry"], index=False)
    geometry_summary.to_csv(outputs["geometry_summary"], index=False)
    np.savez_compressed(
        outputs["predictions"],
        model=np.asarray(list(models)),
        target_gene=np.asarray(targets),
        output_gene=np.asarray(genes),
        tangent=np.asarray(list(tangent.values()), dtype=np.float32),
        exact_lfc=np.asarray(list(exact.values()), dtype=np.float32),
        gamma=np.asarray([transferred[name] for name in models]),
        source_rows=source_rows,
    )
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "kind": "truth-side expected-profile diagnostic",
        "comparison": "normalized covariance versus factorial-moment and iterative Poisson-lognormal covariance",
        "magnitude": "model-specific leave-one-target-out median downstream L1 scale",
        "decoder": {
            "normalized": "exact log1p-CP10K decoder",
            "poisson_lognormal": "multiplicative latent log-rate shift followed by per-cell rate normalization",
        },
        "inputs": {
            display(path): sha256(path)
            for path in [
                CONTROLS,
                STRICT_GUIDES,
                EMPIRICAL,
                STATE_MODEL,
                FACTORIAL,
                REGULARIZED,
                LAPLACE,
            ]
        },
        "outputs": {
            display(path): sha256(path) for path in outputs.values()
        },
        "software": {
            "python": platform.python_version(),
            **{name: version(name) for name in ["anndata", "numpy", "pandas"]},
        },
    }
    (REPORT / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    columns = [
        "model",
        "coordinate",
        "transductive",
        "median_crossfit_gamma",
        "mean_signed_recovery",
        "wrong_target_mean",
        "wrong_target_p",
        "mean_exact_crossfit_nmae",
        "mean_oracle_linear_nmae",
    ]
    print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
