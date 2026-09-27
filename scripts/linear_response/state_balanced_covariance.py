"""Fit equal-state H1 covariance columns and target-cross-fitted variants."""

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

from scripts.linear_response.kernel import CELL_TARGET_SUM, normalize_response_columns

ROOT = Path(__file__).resolve().parents[2]
H1 = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
EMPIRICAL = ROOT / "data/derived/linear_response/ladder/empirical.npz"
STRICT_GUIDES = ROOT / "reports/linear-response-three-models/h1_strict_controls.csv"
DERIVED = ROOT / "data/derived/linear_response/state_diversity"
REPORT = ROOT / "reports/linear-response-state-diversity"
ARTIFACT = DERIVED / "state_balanced_covariance.npz"
CHECKPOINTS = DERIVED / "state_moments"
H1_SHA256 = "a09977104fefb622368ca74b50c9d3c1e891733e6c83db07acfca49b0219c02b"
MODEL_NAMES = [
    "control",
    "all_within",
    "all_between",
    "all_combined",
    "crossfit_within",
    "crossfit_between",
    "crossfit_combined",
]
CHUNK_SIZE = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def balanced_folds(
    labels: np.ndarray,
    batches: np.ndarray,
    states: list[str],
    scored: set[str],
) -> tuple[dict[str, int], dict[str, float]]:
    """Choose equal target folds with balanced cell counts in every batch."""
    table = pd.crosstab(labels, batches).reindex(index=states, fill_value=0)
    counts = table.to_numpy(dtype=np.float64)
    total = counts.sum(axis=0)
    scored_indices = np.asarray([i for i, state in enumerate(states) if state in scored])
    auxiliary_indices = np.asarray(
        [i for i, state in enumerate(states) if state not in scored]
    )
    if len(scored_indices) % 2 or len(auxiliary_indices) % 2:
        raise ValueError("scored and auxiliary target counts must both be even")
    rng = np.random.default_rng(np.random.SeedSequence([0, 111]))
    best_score = np.inf
    best = None
    for _ in range(50_000):
        fold_zero = np.concatenate(
            [
                rng.choice(scored_indices, len(scored_indices) // 2, replace=False),
                rng.choice(
                    auxiliary_indices, len(auxiliary_indices) // 2, replace=False
                ),
            ]
        )
        imbalance = np.abs(2 * counts[fold_zero].sum(axis=0) - total) / total
        score = float(np.sqrt(np.mean(np.square(imbalance))) + imbalance.max())
        if score < best_score:
            best_score = score
            best = fold_zero.copy()
    fold = np.ones(len(states), dtype=np.int8)
    fold[best] = 0
    achieved = np.abs(2 * counts[best].sum(axis=0) - total) / total
    return (
        dict(zip(states, fold.tolist(), strict=True)),
        {
            "search_candidates": 50_000,
            "objective": best_score,
            "batch_relative_imbalance_rms": float(
                np.sqrt(np.mean(np.square(achieved)))
            ),
            "batch_relative_imbalance_max": float(achieved.max()),
        },
    )


def state_moments(
    data: ad.AnnData,
    rows: np.ndarray,
    fit_indices: np.ndarray,
    target_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(len(fit_indices), dtype=np.float64)
    cross = np.zeros((len(fit_indices), len(target_local)), dtype=np.float64)
    for start in range(0, len(rows), CHUNK_SIZE):
        raw = data.X[rows[start : start + CHUNK_SIZE]].tocsr().astype(np.float64)
        cell_totals = np.asarray(raw.sum(axis=1)).ravel()
        values = raw[:, fit_indices].toarray()
        values *= (CELL_TARGET_SUM / cell_totals)[:, None]
        np.log1p(values, out=values)
        total += values.sum(axis=0)
        cross += values.T @ values[:, target_local]
    mean = total / len(rows)
    covariance = (
        cross - len(rows) * np.outer(mean, mean[target_local])
    ) / (len(rows) - 1)
    return mean, covariance


def between_columns(means: np.ndarray, target_local: np.ndarray) -> np.ndarray:
    centered = means - means.mean(axis=0)
    return centered.T @ centered[:, target_local] / (len(means) - 1)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else 0.0


def cached_state_moments(
    data: ad.AnnData,
    state: str,
    rows: np.ndarray,
    fit_indices: np.ndarray,
    target_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    path = CHECKPOINTS / f"{state}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            return saved["mean"].astype(np.float64), saved["covariance"].astype(
                np.float64
            )
    mean, covariance = state_moments(data, rows, fit_indices, target_local)
    mean = mean.astype(np.float32)
    covariance = covariance.astype(np.float32)
    np.savez_compressed(path, mean=mean, covariance=covariance)
    return mean.astype(np.float64), covariance.astype(np.float64)


def main() -> None:
    with np.load(EMPIRICAL, allow_pickle=False) as empirical:
        targets = empirical["target_gene"].astype(str).tolist()
        output_genes = empirical["output_gene"].astype(str).tolist()
        fit_genes = empirical["fit_gene"].astype(str).tolist()
        fit_indices = empirical["full_gene_index"].astype(int)
        target_local = empirical["target_fit_index"].astype(int)
        control_response = empirical["response"].astype(np.float64)
        control_diagonal = empirical["variance"].astype(np.float64)[target_local]
    strict_guides = set(pd.read_csv(STRICT_GUIDES)["guide_id"].astype(str))

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    data = ad.read_h5ad(H1, backed="r")
    try:
        labels = data.obs["target_gene"].astype(str).to_numpy()
        guides = data.obs["guide_id"].astype(str).to_numpy()
        batches = data.obs["batch"].astype(str).to_numpy()
        states = sorted(set(labels) - {"non-targeting"})
        fold_by_state, balance = balanced_folds(
            labels[labels != "non-targeting"],
            batches[labels != "non-targeting"],
            states,
            set(targets),
        )
        state_rows = {
            state: np.flatnonzero(labels == state).astype(np.int64) for state in states
        }
        control_rows = np.flatnonzero(
            (labels == "non-targeting") & np.isin(guides, list(strict_guides))
        ).astype(np.int64)

        means = {}
        within_sums = {
            "all": np.zeros((len(fit_genes), len(targets))),
            0: np.zeros((len(fit_genes), len(targets))),
            1: np.zeros((len(fit_genes), len(targets))),
        }
        means["non-targeting"], covariance = cached_state_moments(
            data, "non-targeting", control_rows, fit_indices, target_local
        )
        for key in within_sums:
            within_sums[key] += covariance
        print(f"state 1/{len(states) + 1}: non-targeting", flush=True)
        for index, state in enumerate(states, start=2):
            means[state], covariance = cached_state_moments(
                data, state, state_rows[state], fit_indices, target_local
            )
            within_sums["all"] += covariance
            within_sums[fold_by_state[state]] += covariance
            print(f"state {index}/{len(states) + 1}: {state}", flush=True)
    finally:
        data.file.close()

    all_states = ["non-targeting", *states]
    fold_states = {
        fold: ["non-targeting", *[s for s in states if fold_by_state[s] == fold]]
        for fold in (0, 1)
    }
    operators = {}
    selected_sets = {"all": all_states, 0: fold_states[0], 1: fold_states[1]}
    for name, selected in selected_sets.items():
        within = within_sums[name] / len(selected)
        between = between_columns(
            np.vstack([means[state] for state in selected]), target_local
        )
        operators[name] = {
            "within": within,
            "between": between,
            "combined": within + between,
        }

    responses = [control_response]
    diagonals = [control_diagonal]
    for component in ("within", "between", "combined"):
        response, diagonal = normalize_response_columns(
            operators["all"][component], target_local
        )
        responses.append(response.T)
        diagonals.append(diagonal)
    for component in ("within", "between", "combined"):
        fold_responses = {
            fold: normalize_response_columns(
                operators[fold][component], target_local
            )
            for fold in (0, 1)
        }
        response = np.empty_like(control_response)
        diagonal = np.empty(len(targets))
        for target_index, target in enumerate(targets):
            training_fold = 1 - fold_by_state[target]
            response[target_index] = fold_responses[training_fold][0][
                :, target_index
            ]
            diagonal[target_index] = fold_responses[training_fold][1][target_index]
        responses.append(response)
        diagonals.append(diagonal)
    responses = np.asarray(responses)
    diagonals = np.asarray(diagonals)

    diagnostics = []
    for target_index, target in enumerate(targets):
        downstream = np.arange(len(fit_genes)) != target_local[target_index]
        row = {
            "target_gene": target,
            "fold": fold_by_state[target],
            "n_cells": len(state_rows[target]),
        }
        for model_index, model in enumerate(MODEL_NAMES):
            row[f"{model}_diagonal"] = diagonals[model_index, target_index]
            row[f"{model}_downstream_norm"] = np.linalg.norm(
                responses[model_index, target_index, downstream]
            )
            row[f"{model}_cosine_control"] = cosine(
                responses[model_index, target_index, downstream],
                control_response[target_index, downstream],
            )
        diagnostics.append(row)

    DERIVED.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ARTIFACT,
        model=np.asarray(MODEL_NAMES),
        response=responses.astype(np.float32),
        diagonal=diagonals.astype(np.float32),
        target_gene=np.asarray(targets),
        output_gene=np.asarray(output_genes),
        fit_gene=np.asarray(fit_genes),
        full_gene_index=fit_indices,
        target_fit_index=target_local,
        target_fold=np.asarray([fold_by_state[target] for target in targets]),
    )
    assignments = pd.DataFrame(
        [
            {
                "target_gene": state,
                "fold": fold_by_state[state],
                "scored_target": state in set(targets),
                "n_cells": len(state_rows[state]),
                "n_batches": len(np.unique(batches[state_rows[state]])),
            }
            for state in states
        ]
    )
    assignments.to_csv(REPORT / "state_fold_assignments.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(
        REPORT / "covariance_direction_diagnostics.csv", index=False
    )
    summary = pd.DataFrame(
        [
            {
                "model": model,
                "training_states": 1
                + (len(states) if model.startswith("all_") else len(states) // 2)
                if model != "control"
                else 1,
                "median_diagonal": np.median(diagonals[index]),
                "median_downstream_norm": np.median(
                    [row[f"{model}_downstream_norm"] for row in diagnostics]
                ),
                "median_cosine_control": np.median(
                    [row[f"{model}_cosine_control"] for row in diagnostics]
                ),
            }
            for index, model in enumerate(MODEL_NAMES)
        ]
    )
    summary.to_csv(REPORT / "covariance_component_summary.csv", index=False)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "design": "equal-target-state covariance with one balanced two-fold target split",
        "states": {
            "perturbations": len(states),
            "scored": len(targets),
            "auxiliary": len(states) - len(targets),
            "control": "one strict-26-guide state",
            "weighting": "equal state weight; all cells retained within each state",
        },
        "fold_balance": balance,
        "components": {
            "within": "equal-state mean of within-state sample covariance",
            "between": "sample covariance of equally weighted state means",
            "combined": "within + between",
        },
        "leakage": {
            "all": "transductive: the scored target state is included",
            "crossfit": "target state and its cells are excluded from its training fold",
        },
        "normalization": "per-cell log1p CP10K",
        "response": "minus covariance column divided by its target diagonal",
        "inputs": {
            str(H1.relative_to(ROOT)): H1_SHA256,
            str(EMPIRICAL.relative_to(ROOT)): sha256(EMPIRICAL),
            str(STRICT_GUIDES.relative_to(ROOT)): sha256(STRICT_GUIDES),
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                ARTIFACT,
                REPORT / "state_fold_assignments.csv",
                REPORT / "covariance_direction_diagnostics.csv",
                REPORT / "covariance_component_summary.csv",
            ]
        },
        "software": {
            "python": platform.python_version(),
            **{name: version(name) for name in ["anndata", "numpy", "pandas"]},
        },
    }
    manifest_path = REPORT / "fit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"Wrote {ARTIFACT} and {manifest_path}")


if __name__ == "__main__":
    main()
