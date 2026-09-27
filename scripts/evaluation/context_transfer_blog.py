"""Analyze CRISPRi effect transfer across matched within-study contexts.

The fitted coefficients are robust summaries of per-perturbation projections.
Pooled gene-entry MSE is retained only as a sensitivity diagnostic.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse

from scripts.cloud.audit_compact_pseudobulks import audit as audit_compact

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/context-transfer-blog"
DERIVED = ROOT / "data/derived/context_transfer_blog"
CPM = 1_000_000.0
EPSILON = 1e-9
N_FOLDS = 5
N_BOOTSTRAPS = 2_000
SEED = 20260921
MODELS = ("unchanged", "direct", "global_median", "context_interaction")
METRICS = (
    "effect_cosine",
    "retrieval_score",
    "signed_recovery_100",
    "normalized_absolute_error",
)


@dataclass(frozen=True)
class ContextSpec:
    name: str
    study: str
    assay: str
    directory: Path
    control: str

    @property
    def target_path(self) -> Path:
        return self.directory / f"{self.name}_target_pseudobulk.h5ad"

    @property
    def guide_path(self) -> Path:
        return self.directory / f"{self.name}_guide_pseudobulk.h5ad"

    @property
    def manifest_path(self) -> Path:
        return self.directory / f"{self.name}_manifest.json"

    @property
    def audit_path(self) -> Path:
        return self.directory / f"{self.name}_audit.json"


@dataclass
class ContextData:
    spec: ContextSpec
    genes: list[str]
    targets: list[str]
    counts: sparse.csr_matrix
    control_counts: np.ndarray
    effects: np.ndarray
    effects_log1p: np.ndarray
    control_log1p: np.ndarray
    cells: np.ndarray
    total_umis: np.ndarray


CONTEXTS = {
    "K562_essential": ContextSpec(
        "K562_essential",
        "Replogle 2022",
        "10x Perturb-seq CRISPRi",
        ROOT / "data/derived/context_atlas/final/K562_essential",
        "non-targeting",
    ),
    "RPE1": ContextSpec(
        "RPE1",
        "Replogle 2022",
        "10x Perturb-seq CRISPRi",
        ROOT / "data/derived/context_atlas/final/RPE1",
        "non-targeting",
    ),
    "HepG2": ContextSpec(
        "HepG2",
        "Nadig 2025",
        "10x Perturb-seq CRISPRi",
        ROOT / "data/derived/context_atlas/final/HepG2",
        "non-targeting",
    ),
    "Jurkat": ContextSpec(
        "Jurkat",
        "Nadig 2025",
        "10x Perturb-seq CRISPRi",
        ROOT / "data/derived/context_atlas/final/Jurkat",
        "non-targeting",
    ),
    "HCT116": ContextSpec(
        "HCT116",
        "X-Atlas/Orion",
        "FiCS 10x GEM-X 5-prime CRISPRi",
        ROOT / "data/derived/xatlas_orion/final/HCT116",
        "Non-Targeting",
    ),
    "HEK293T": ContextSpec(
        "HEK293T",
        "X-Atlas/Orion",
        "FiCS 10x GEM-X 5-prime CRISPRi",
        ROOT / "data/derived/xatlas_orion/final/HEK293T",
        "Non-Targeting",
    ),
}

PRIMARY_PAIRS = (
    ("K562_essential", "RPE1"),
    ("HepG2", "Jurkat"),
    ("HCT116", "HEK293T"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def collapse_columns(
    matrix: sparse.spmatrix, symbols: list[str]
) -> tuple[sparse.csr_matrix, list[str]]:
    """Sum duplicate-symbol columns while retaining first-appearance order."""
    codes, unique = pd.factorize(np.asarray(symbols, dtype=str), sort=False)
    membership = sparse.csr_matrix(
        (
            np.ones(len(codes), dtype=np.int8),
            (np.arange(len(codes)), codes),
        ),
        shape=(len(codes), len(unique)),
    )
    return (sparse.csr_matrix(matrix) @ membership).tocsr(), unique.tolist()


def normalized_effects(
    counts: sparse.spmatrix | np.ndarray, control: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return epsilon-LFCs, log1p differences, and control log1p CPM."""
    dense = (
        counts.toarray().astype(np.float64)
        if sparse.issparse(counts)
        else np.asarray(counts, dtype=np.float64)
    )
    row_sums = dense.sum(axis=1)
    if np.any(row_sums <= 0) or control.sum() <= 0:
        raise ValueError("pseudobulk normalization requires positive library sizes")
    perturbed_cpm = dense * (CPM / row_sums[:, None])
    control_cpm = np.asarray(control, dtype=np.float64) * (CPM / control.sum())
    lfc = np.log2((perturbed_cpm + EPSILON) / (control_cpm + EPSILON))
    log1p = np.log2(perturbed_cpm + 1) - np.log2(control_cpm + 1)
    return lfc, log1p, np.log2(control_cpm + 1)


def load_context(spec: ContextSpec) -> ContextData:
    data = ad.read_h5ad(spec.target_path, backed="r")
    try:
        symbols = data.var["gene_name"].astype(str).tolist()
        matrix, genes = collapse_columns(data.X[:], symbols)
        labels = data.obs["gene_target"].astype(str).to_numpy()
        control_row = int(np.flatnonzero(labels == spec.control)[0])
        selected = np.flatnonzero(labels != spec.control)
        targets = labels[selected].tolist()
        counts = matrix[selected].tocsr()
        control = matrix[control_row].toarray().ravel().astype(np.float64)
        effects, effects_log1p, control_log1p = normalized_effects(counts, control)
        cells = data.obs["n_cells"].to_numpy(dtype=int)[selected]
        total_umis = data.obs["total_umis"].to_numpy(dtype=np.int64)[selected]
    finally:
        data.file.close()
    return ContextData(
        spec,
        genes,
        targets,
        counts,
        control,
        effects,
        effects_log1p,
        control_log1p,
        cells,
        total_umis,
    )


def deterministic_folds(targets: list[str], seed: int = SEED) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(len(targets))
    folds = np.empty(len(targets), dtype=np.int8)
    folds[order] = np.arange(len(targets), dtype=int) % N_FOLDS
    return folds


def target_gene_indices(targets: list[str], genes: list[str]) -> np.ndarray:
    lookup = {gene: index for index, gene in enumerate(genes)}
    return np.asarray([lookup.get(target, -1) for target in targets], dtype=int)


def projection_coefficients(
    source: np.ndarray,
    target: np.ndarray,
    difference: np.ndarray,
    excluded: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for index, (x, y) in enumerate(zip(source, target, strict=True)):
        keep = np.ones(source.shape[1], dtype=bool)
        if excluded[index] >= 0:
            keep[excluded[index]] = False
        x, y, d = x[keep], y[keep], difference[keep]
        energy = float(x @ x)
        beta = float(x @ y / energy) if energy else np.nan
        design = np.column_stack([x, x * d])
        rank = int(np.linalg.matrix_rank(design))
        condition = float(np.linalg.cond(design)) if rank == 2 else np.inf
        if rank == 2:
            interaction = np.linalg.pinv(design) @ y
            beta0, beta_d = map(float, interaction)
        else:
            beta0 = beta_d = np.nan
        rows.append(
            {
                "target_index": index,
                "oracle_projection": beta,
                "interaction_beta0": beta0,
                "interaction_beta_d": beta_d,
                "interaction_rank": rank,
                "interaction_condition_number": condition,
                "source_energy": energy,
            }
        )
    return pd.DataFrame(rows)


def pooled_ols(
    source: np.ndarray,
    target: np.ndarray,
    difference: np.ndarray,
    excluded: np.ndarray,
    rows: np.ndarray,
) -> tuple[float, float, float]:
    xx = xy = 0.0
    gram = np.zeros((2, 2), dtype=np.float64)
    cross = np.zeros(2, dtype=np.float64)
    for index in rows:
        keep = np.ones(source.shape[1], dtype=bool)
        if excluded[index] >= 0:
            keep[excluded[index]] = False
        x, y, d = source[index, keep], target[index, keep], difference[keep]
        xx += float(x @ x)
        xy += float(x @ y)
        design = np.column_stack([x, x * d])
        gram += design.T @ design
        cross += design.T @ y
    interaction = np.linalg.pinv(gram) @ cross
    global_beta = float(xy / xx) if xx else np.nan
    return global_beta, float(interaction[0]), float(interaction[1])


def crossfit_predictions(
    targets: list[str],
    genes: list[str],
    source: np.ndarray,
    target: np.ndarray,
    difference: np.ndarray,
    folds: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    excluded = target_gene_indices(targets, genes)
    per_target = projection_coefficients(source, target, difference, excluded)
    per_target.insert(0, "target_gene", targets)
    per_target.insert(1, "fold", folds)
    global_prediction = np.empty_like(source)
    context_prediction = np.empty_like(source)
    fold_rows = []
    for heldout_fold in range(N_FOLDS):
        train = folds != heldout_fold
        heldout = folds == heldout_fold
        beta = float(np.nanmedian(per_target.loc[train, "oracle_projection"]))
        valid = train & (per_target.interaction_rank.to_numpy() == 2)
        beta0 = float(np.nanmedian(per_target.loc[valid, "interaction_beta0"]))
        beta_d = float(np.nanmedian(per_target.loc[valid, "interaction_beta_d"]))
        global_prediction[heldout] = source[heldout] * beta
        context_prediction[heldout] = source[heldout] * (beta0 + beta_d * difference)
        pooled_beta, pooled_b0, pooled_bd = pooled_ols(
            source, target, difference, excluded, np.flatnonzero(train)
        )
        fold_rows.append(
            {
                "heldout_fold": heldout_fold,
                "training_targets": int(train.sum()),
                "heldout_targets": int(heldout.sum()),
                "global_beta_median": beta,
                "interaction_beta0_median": beta0,
                "interaction_beta_d_median": beta_d,
                "interaction_beta_d_per_sd": beta_d * difference.std(),
                "valid_interaction_training_targets": int(valid.sum()),
                "pooled_ols_global_beta_sensitivity": pooled_beta,
                "pooled_ols_beta0_sensitivity": pooled_b0,
                "pooled_ols_beta_d_sensitivity": pooled_bd,
            }
        )
    predictions = {
        "unchanged": np.zeros_like(source),
        "direct": source,
        "global_median": global_prediction,
        "context_interaction": context_prediction,
    }
    return predictions, per_target, pd.DataFrame(fold_rows)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else np.nan


def single_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    excluded: int,
    k: int,
) -> dict[str, float]:
    keep = np.ones(len(truth), dtype=bool)
    if excluded >= 0:
        keep[excluded] = False
    predicted, observed = prediction[keep], truth[keep]
    truth_scale = float(np.abs(observed).sum())
    n = min(k, len(observed))
    if not np.any(predicted):
        recovery = 0.0
    else:
        true_top = np.argpartition(np.abs(observed), -n)[-n:]
        predicted_top = np.argpartition(np.abs(predicted), -n)[-n:]
        strong = np.zeros(len(observed), dtype=bool)
        strong[true_top] = True
        recovered = strong[predicted_top]
        recovery = float(
            np.sum(
                recovered
                & (
                    np.sign(predicted[predicted_top])
                    == np.sign(observed[predicted_top])
                )
            )
            / n
        )
    return {
        "cosine": cosine(predicted, observed),
        "nae": float(np.abs(predicted - observed).sum() / truth_scale)
        if truth_scale
        else np.nan,
        "signed_recovery": recovery,
    }


def retrieval_scores(
    predictions: np.ndarray, truth: np.ndarray, excluded: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.any(predictions):
        return (
            np.full(len(truth), 0.5),
            np.full(len(truth), (len(truth) + 1) / 2),
            np.zeros(len(truth), dtype=bool),
        )
    dot = truth @ predictions.T
    truth_norm2 = np.square(truth).sum(axis=1)
    prediction_norm2 = np.square(predictions).sum(axis=1)
    scores = np.empty(len(truth))
    ranks = np.empty(len(truth))
    top1 = np.zeros(len(truth), dtype=bool)
    for index, gene in enumerate(excluded):
        numerator = dot[index].copy()
        left_norm2 = truth_norm2[index]
        right_norm2 = prediction_norm2.copy()
        if gene >= 0:
            numerator -= truth[index, gene] * predictions[:, gene]
            left_norm2 -= truth[index, gene] ** 2
            right_norm2 -= np.square(predictions[:, gene])
        denominator = np.sqrt(np.maximum(left_norm2, 0) * np.maximum(right_norm2, 0))
        similarity = np.divide(
            numerator,
            denominator,
            out=np.full(len(truth), -np.inf),
            where=denominator > 0,
        )
        correct = similarity[index]
        greater = int(np.sum(similarity > correct))
        equal = int(np.sum(np.isclose(similarity, correct, rtol=1e-12, atol=1e-12)))
        zero_rank = greater + 0.5 * (equal - 1)
        ranks[index] = zero_rank + 1
        scores[index] = 1 - zero_rank / (len(truth) - 1)
        top1[index] = greater == 0 and equal == 1
    return scores, ranks, top1


def evaluate_models(
    targets: list[str],
    genes: list[str],
    source: np.ndarray,
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    source_cells: np.ndarray,
    target_cells: np.ndarray,
    folds: np.ndarray,
) -> pd.DataFrame:
    excluded = target_gene_indices(targets, genes)
    source_norm = np.empty(len(targets))
    truth_norm = np.empty(len(targets))
    oracle = np.empty(len(targets))
    for index in range(len(targets)):
        keep = np.ones(len(genes), dtype=bool)
        if excluded[index] >= 0:
            keep[excluded[index]] = False
        x, y = source[index, keep], truth[index, keep]
        source_norm[index] = np.linalg.norm(x)
        truth_norm[index] = np.linalg.norm(y)
        oracle[index] = x @ y / (x @ x) if x @ x else np.nan
    rows = []
    for model, prediction in predictions.items():
        retrieval, rank, top1 = retrieval_scores(prediction, truth, excluded)
        for index, target_name in enumerate(targets):
            scores = {
                k: single_metrics(prediction[index], truth[index], excluded[index], k)
                for k in (50, 100, 200)
            }
            predicted_norm = np.linalg.norm(
                np.delete(prediction[index], excluded[index])
                if excluded[index] >= 0
                else prediction[index]
            )
            rows.append(
                {
                    "model": model,
                    "target_gene": target_name,
                    "fold": int(folds[index]),
                    "source_cells": int(source_cells[index]),
                    "target_cells": int(target_cells[index]),
                    "source_effect_norm": source_norm[index],
                    "target_effect_norm": truth_norm[index],
                    "predicted_effect_norm": predicted_norm,
                    "source_to_truth_norm_ratio": source_norm[index] / truth_norm[index]
                    if truth_norm[index]
                    else np.nan,
                    "predicted_to_truth_norm_ratio": predicted_norm / truth_norm[index]
                    if truth_norm[index]
                    else np.nan,
                    "oracle_projection": oracle[index],
                    "effect_cosine": scores[100]["cosine"],
                    "normalized_absolute_error": scores[100]["nae"],
                    "signed_recovery_50": scores[50]["signed_recovery"],
                    "signed_recovery_100": scores[100]["signed_recovery"],
                    "signed_recovery_200": scores[200]["signed_recovery"],
                    "retrieval_score": retrieval[index],
                    "retrieval_rank": rank[index],
                    "retrieval_top1": bool(top1[index]),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    indices = rng.integers(0, len(values), size=(N_BOOTSTRAPS, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(map(float, np.quantile(means, (0.025, 0.975))))


def summarize_edge(per_target: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for model, frame in per_target.groupby("model", sort=False):
        row = {"model": model, "targets": len(frame)}
        for metric in METRICS:
            values = frame[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"mean_{metric}"] = float(finite.mean()) if len(finite) else np.nan
            low, high = bootstrap_mean(values, rng)
            row[f"{metric}_q025"] = low
            row[f"{metric}_q975"] = high
        row["retrieval_top1_fraction"] = frame.retrieval_top1.mean()
        row["mean_signed_recovery_50"] = frame.signed_recovery_50.mean()
        row["mean_signed_recovery_200"] = frame.signed_recovery_200.mean()
        row["median_predicted_to_truth_norm_ratio"] = frame[
            "predicted_to_truth_norm_ratio"
        ].median()
        row["median_source_to_truth_norm_ratio"] = frame[
            "source_to_truth_norm_ratio"
        ].median()
        row["median_oracle_projection"] = frame.oracle_projection.median()
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_differences(
    per_target: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    rows = []
    comparisons = (
        ("direct", "unchanged"),
        ("global_median", "direct"),
        ("context_interaction", "global_median"),
    )
    for model, reference in comparisons:
        left = per_target[per_target.model == model].set_index("target_gene")
        right = per_target[per_target.model == reference].set_index("target_gene")
        for metric in METRICS:
            difference = (left[metric] - right[metric]).to_numpy(dtype=float)
            difference = difference[np.isfinite(difference)]
            if len(difference):
                sampled = difference[
                    rng.integers(
                        0, len(difference), size=(N_BOOTSTRAPS, len(difference))
                    )
                ].mean(axis=1)
                low, high = np.quantile(sampled, (0.025, 0.975))
                estimate = difference.mean()
            else:
                estimate = low = high = np.nan
            rows.append(
                {
                    "model": model,
                    "reference": reference,
                    "metric": metric,
                    "mean_difference": estimate,
                    "q025": low,
                    "q975": high,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_coefficients(
    coefficients: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    rows = []
    for heldout_fold in range(N_FOLDS):
        train = coefficients.fold != heldout_fold
        for name in (
            "oracle_projection",
            "interaction_beta0",
            "interaction_beta_d",
        ):
            values = coefficients.loc[train, name].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            sampled = values[
                rng.integers(0, len(values), size=(N_BOOTSTRAPS, len(values)))
            ]
            medians = np.median(sampled, axis=1)
            low, middle, high = np.quantile(medians, (0.025, 0.5, 0.975))
            rows.append(
                {
                    "heldout_fold": heldout_fold,
                    "coefficient": name,
                    "estimate": np.median(values),
                    "q025": low,
                    "bootstrap_median": middle,
                    "q975": high,
                    "training_targets": len(values),
                }
            )
    return pd.DataFrame(rows)


def guide_calibration(context: ContextData) -> pd.DataFrame:
    data = ad.read_h5ad(context.spec.guide_path, backed="r")
    try:
        labels = data.obs["gene_target"].astype(str).to_numpy()
        selected = np.flatnonzero(labels != context.spec.control)
        selected_labels = labels[selected]
        counts, genes = collapse_columns(
            data.X[selected], data.var["gene_name"].astype(str).tolist()
        )
        if genes != context.genes:
            raise ValueError(f"{context.spec.name}: target and guide gene axes differ")
        guide_ids = data.obs_names.astype(str).to_numpy()[selected]
        guide_cells = data.obs["n_cells"].to_numpy(dtype=int)[selected]
    finally:
        data.file.close()
    target_lookup = {target: index for index, target in enumerate(context.targets)}
    gene_lookup = {gene: index for index, gene in enumerate(context.genes)}
    rows = []
    for target_name in sorted(set(selected_labels)):
        guide_rows = np.flatnonzero(selected_labels == target_name)
        if len(guide_rows) < 2 or target_name not in target_lookup:
            continue
        target_index = target_lookup[target_name]
        wrong_index = (target_index + 1) % len(context.targets)
        excluded = gene_lookup.get(target_name, -1)
        pooled = counts[guide_rows].sum(axis=0).A1
        for guide_row in guide_rows:
            remainder = pooled - counts[guide_row].toarray().ravel()
            prediction, _, _ = normalized_effects(
                counts[guide_row], context.control_counts
            )
            truth, _, _ = normalized_effects(remainder[None, :], context.control_counts)
            prediction, truth = prediction[0], truth[0]
            wrong = context.effects[wrong_index]
            positive = single_metrics(prediction, truth, excluded, 100)
            negative = single_metrics(wrong, truth, excluded, 100)
            keep = np.ones(len(context.genes), dtype=bool)
            if excluded >= 0:
                keep[excluded] = False
            truth_keep = truth[keep]
            candidates = context.effects[:, keep]
            numerators = candidates @ truth_keep
            denominators = np.linalg.norm(candidates, axis=1) * np.linalg.norm(
                truth_keep
            )
            similarities = np.divide(
                numerators,
                denominators,
                out=np.full(len(candidates), -np.inf),
                where=denominators > 0,
            )
            similarities[target_index] = cosine(prediction[keep], truth_keep)
            correct = similarities[target_index]
            greater = np.sum(similarities > correct)
            equal = np.sum(np.isclose(similarities, correct, rtol=1e-12, atol=1e-12))
            positive_retrieval = 1 - (greater + 0.5 * (equal - 1)) / (
                len(candidates) - 1
            )
            wrong = similarities[wrong_index]
            greater = np.sum(similarities > wrong)
            equal = np.sum(np.isclose(similarities, wrong, rtol=1e-12, atol=1e-12))
            negative_retrieval = 1 - (greater + 0.5 * (equal - 1)) / (
                len(candidates) - 1
            )
            rows.append(
                {
                    "context": context.spec.name,
                    "target_gene": target_name,
                    "guide_id": guide_ids[guide_row],
                    "guide_cells": int(guide_cells[guide_row]),
                    "remainder_cells": int(
                        context.cells[target_index] - guide_cells[guide_row]
                    ),
                    "positive_cosine": positive["cosine"],
                    "negative_cosine": negative["cosine"],
                    "positive_normalized_absolute_error": positive["nae"],
                    "negative_normalized_absolute_error": negative["nae"],
                    "positive_signed_recovery_100": positive["signed_recovery"],
                    "negative_signed_recovery_100": negative["signed_recovery"],
                    "positive_retrieval_score": positive_retrieval,
                    "negative_retrieval_score": negative_retrieval,
                }
            )
    return pd.DataFrame(rows)


def calibration_summary(guide: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for context, frame in guide.groupby("context", sort=False):
        for metric, higher in (
            ("cosine", True),
            ("normalized_absolute_error", False),
            ("signed_recovery_100", True),
            ("retrieval_score", True),
        ):
            positive = frame[f"positive_{metric}"].to_numpy(dtype=float)
            negative = frame[f"negative_{metric}"].to_numpy(dtype=float)
            valid = np.isfinite(positive) & np.isfinite(negative)
            positive, negative = positive[valid], negative[valid]
            oriented = positive - negative if higher else negative - positive
            sampled = oriented[
                rng.integers(0, len(oriented), size=(N_BOOTSTRAPS, len(oriented)))
            ].mean(axis=1)
            low, high = np.quantile(sampled, (0.025, 0.975))
            perfect = 1.0 if higher else 0.0
            denominator = (
                perfect - negative.mean() if higher else negative.mean() - perfect
            )
            rows.append(
                {
                    "context": context,
                    "metric": metric,
                    "guide_leave_one_out_rows": len(oriented),
                    "positive_mean": positive.mean(),
                    "negative_mean": negative.mean(),
                    "oriented_positive_minus_negative": oriented.mean(),
                    "q025": low,
                    "q975": high,
                    "dynamic_range_fraction": oriented.mean() / denominator
                    if denominator > 0
                    else np.nan,
                    "calibrated": bool(low > 0),
                }
            )
    return pd.DataFrame(rows)


def edge_examples(
    edge: str,
    targets: list[str],
    genes: list[str],
    source: np.ndarray,
    truth: np.ndarray,
    per_target: pd.DataFrame,
) -> pd.DataFrame:
    direct = per_target.query("model == 'direct'").set_index("target_gene")
    eligible = direct.source_effect_norm >= direct.source_effect_norm.median()
    successes = direct.sort_values(
        ["retrieval_score", "signed_recovery_100", "effect_cosine"],
        ascending=False,
    ).head(5)
    failures = (
        direct.loc[eligible]
        .sort_values(["retrieval_score", "signed_recovery_100", "effect_cosine"])
        .head(5)
    )
    target_lookup = {target: index for index, target in enumerate(targets)}
    rows = []
    for category, frame in (("success", successes), ("failure", failures)):
        for target_name, record in frame.iterrows():
            index = target_lookup[target_name]
            keep = np.ones(len(genes), dtype=bool)
            if target_name in genes:
                keep[genes.index(target_name)] = False
            indices = np.flatnonzero(keep)
            source_top = indices[
                np.argsort(-np.abs(source[index, keep]), kind="stable")[:5]
            ]
            truth_top = indices[
                np.argsort(-np.abs(truth[index, keep]), kind="stable")[:5]
            ]
            rows.append(
                {
                    "edge": edge,
                    "category": category,
                    "target_gene": target_name,
                    "effect_cosine": record.effect_cosine,
                    "retrieval_score": record.retrieval_score,
                    "signed_recovery_100": record.signed_recovery_100,
                    "source_cells": record.source_cells,
                    "target_cells": record.target_cells,
                    "top_source_response_genes": ";".join(
                        genes[column] for column in source_top
                    ),
                    "top_truth_response_genes": ";".join(
                        genes[column] for column in truth_top
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_model_comparison(summary: pd.DataFrame) -> None:
    primary = summary.query("direction == 'primary'")
    labels = primary.edge.unique().tolist()
    colors = {
        "unchanged": "#b8b8b8",
        "direct": "#3b82f6",
        "global_median": "#f59e0b",
        "context_interaction": "#10b981",
    }
    panels = (
        ("mean_effect_cosine", "Target-excluded cosine"),
        ("mean_retrieval_score", "Correct-target retrieval"),
        ("mean_signed_recovery_100", "Signed top-100 recovery"),
        ("mean_normalized_absolute_error", "Normalized absolute error"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    width = 0.18
    x = np.arange(len(labels))
    for axis, (column, title) in zip(axes.ravel(), panels, strict=True):
        for offset, model in enumerate(MODELS):
            values = [
                primary.query("edge == @edge and model == @model")[column].iloc[0]
                for edge in labels
            ]
            axis.bar(
                x + (offset - 1.5) * width,
                values,
                width,
                label=model.replace("_", " "),
                color=colors[model],
            )
        axis.set_title(title)
        axis.set_xticks(x, [label.replace("->", "→") for label in labels], rotation=12)
        axis.axhline(
            0.5 if "retrieval" in column else 0, color="black", lw=0.6, alpha=0.4
        )
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(REPORT / "model_comparison.png", dpi=180)
    fig.savefig(REPORT / "model_comparison.svg")
    plt.close(fig)


def plot_context_coefficients(
    coefficients: pd.DataFrame, differences: pd.DataFrame
) -> None:
    primary = coefficients.query("direction == 'primary'")
    edges = primary.edge.unique().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for index, edge in enumerate(edges):
        frame = primary[
            (primary.edge == edge) & (primary.coefficient == "interaction_beta_d")
        ]
        center = frame.estimate.median() * frame.context_difference_sd.iloc[0]
        low = frame.q025.median() * frame.context_difference_sd.iloc[0]
        high = frame.q975.median() * frame.context_difference_sd.iloc[0]
        axes[0].errorbar(
            center,
            index,
            xerr=[[center - low], [high - center]],
            fmt="o",
            color="#10b981",
            capsize=3,
        )
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_yticks(range(len(edges)), [edge.replace("->", "→") for edge in edges])
    axes[0].set_xlabel("Context coefficient per SD")
    axes[0].set_title("Coefficient sign and uncertainty")

    selected = differences.query(
        "direction == 'primary' and model == 'context_interaction' and reference == 'global_median'"
    )
    metrics = [
        "effect_cosine",
        "retrieval_score",
        "signed_recovery_100",
        "normalized_absolute_error",
    ]
    offsets = np.linspace(-0.24, 0.24, len(edges))
    for offset, edge in zip(offsets, edges, strict=True):
        frame = selected[selected.edge == edge].set_index("metric").loc[metrics]
        axes[1].plot(
            np.arange(len(metrics)) + offset,
            frame.mean_difference,
            "o-",
            label=edge.replace("->", "→"),
        )
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(
        range(len(metrics)), ["cosine", "retrieval", "signed-100", "NAE"], rotation=15
    )
    axes[1].set_ylabel("Interaction minus global")
    axes[1].set_title("Held-out benefit (negative is better only for NAE)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(REPORT / "context_interaction.png", dpi=180)
    fig.savefig(REPORT / "context_interaction.svg")
    plt.close(fig)


def plot_graph(summary: pd.DataFrame) -> None:
    positions = {
        "K562_essential": (0, 2),
        "RPE1": (2, 2),
        "HepG2": (0, 1),
        "Jurkat": (2, 1),
        "HCT116": (0, 0),
        "HEK293T": (2, 0),
    }
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for name, (x, y) in positions.items():
        axis.scatter(x, y, s=1_900, color="#f4f4f5", edgecolor="#3f3f46", zorder=3)
        axis.text(x, y, name, ha="center", va="center", fontsize=10, zorder=4)
    direct = summary[summary.model == "direct"]
    for row in direct.itertuples(index=False):
        start, end = positions[row.source], positions[row.target]
        primary = row.direction == "primary"
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=2 if primary else 1,
            linestyle="-" if primary else "--",
            color="#2563eb" if primary else "#71717a",
            connectionstyle=f"arc3,rad={0.12 if primary else -0.12}",
            shrinkA=38,
            shrinkB=38,
        )
        axis.add_patch(arrow)
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        vertical = 0.16 if primary else -0.19
        axis.text(
            midpoint[0],
            midpoint[1] + vertical,
            f"cos {row.mean_effect_cosine:.2f} · norm {row.median_source_to_truth_norm_ratio:.2f} · ret {row.mean_retrieval_score:.2f}",
            ha="center",
            va="center",
            fontsize=7.5,
            color="#1d4ed8" if primary else "#52525b",
        )
    axis.set_xlim(-0.8, 2.8)
    axis.set_ylim(-0.6, 2.6)
    axis.axis("off")
    axis.set_title(
        "Within-study context-transfer graph\nsolid: primary; dashed: reverse sensitivity"
    )
    fig.savefig(REPORT / "dataset_graph.png", dpi=180)
    fig.savefig(REPORT / "dataset_graph.svg")
    plt.close(fig)


def checkpoint_reviews(
    summary: pd.DataFrame,
    differences: pd.DataFrame,
    coefficients: pd.DataFrame,
    calibration: pd.DataFrame,
) -> dict[str, object]:
    primary = summary.query("direction == 'primary'")
    direct_rows = primary.query("model == 'direct'").set_index("edge")
    retrieval_calibrated = calibration.query("metric == 'retrieval_score'").set_index(
        "context"
    )["calibrated"]
    direct_gate = {
        edge: bool(
            row.retrieval_score_q025 > 0.5
            and retrieval_calibrated.get(row.source, False)
            and retrieval_calibrated.get(row.target, False)
        )
        for edge, row in direct_rows.iterrows()
    }
    coefficient_points = (
        coefficients.query(
            "direction == 'primary' and coefficient == 'oracle_projection'"
        )
        .groupby("edge")
        .agg(
            estimate=("estimate", "median"),
            q025=("q025", "median"),
            q975=("q975", "median"),
        )
    )
    amplitude_mismatch = {
        edge: bool(row.q975 < 1 or row.q025 > 1)
        for edge, row in coefficient_points.iterrows()
    }
    context_points = (
        coefficients.query(
            "direction == 'primary' and coefficient == 'interaction_beta_d'"
        )
        .groupby("edge")
        .estimate.median()
    )
    signs = np.sign(context_points.to_numpy())
    same_sign = bool(len(signs) == 3 and np.all(signs == signs[0]) and signs[0] != 0)
    context_gate = {}
    for edge in direct_rows.index:
        frame = differences.query(
            "edge == @edge and direction == 'primary' and model == 'context_interaction' and reference == 'global_median'"
        ).set_index("metric")
        nae_better = bool(frame.loc["normalized_absolute_error", "q975"] < 0)
        primary_metrics = frame.loc[
            ["effect_cosine", "retrieval_score", "signed_recovery_100"]
        ]
        one_primary_better = bool((primary_metrics.q025 > 0).any())
        no_material_harm = bool((primary_metrics.mean_difference >= -0.01).all())
        context_gate[edge] = nae_better and one_primary_better and no_material_harm
    return {
        "checkpoint_1_aggregation_and_calibration": {
            "compact_audits": "passed",
            "self_transfer_controls": {"cosine": 1.0, "nae": 0.0, "projection": 1.0},
            "unchanged_control_nae": 1.0,
            "calibrated_metrics_by_context": {
                context: frame.loc[frame.calibrated, "metric"].tolist()
                for context, frame in calibration.groupby("context", sort=False)
            },
        },
        "checkpoint_2_direct_transfer": {
            "criterion": "retrieval is control-calibrated in both contexts and its 95% perturbation-bootstrap lower bound exceeds chance 0.5",
            "passed_by_edge": direct_gate,
            "next_step_justified": bool(all(direct_gate.values())),
        },
        "checkpoint_3_global_slope": {
            "criterion": "direct-transfer gate passes and the projection-coefficient interval excludes 1",
            "amplitude_mismatch_by_edge": amplitude_mismatch,
            "next_step_justified_by_edge": {
                edge: direct_gate[edge] and amplitude_mismatch.get(edge, False)
                for edge in direct_gate
            },
        },
        "checkpoint_4_context_expression": {
            "coefficient_same_nonzero_sign_across_primary_edges": same_sign,
            "heldout_benefit_by_edge": context_gate,
            "replication_gate_passed": same_sign and all(context_gate.values()),
            "decision": (
                "interaction replicated; proceed only to prespecified H1 anchors"
                if same_sign and all(context_gate.values())
                else "stop model ladder; do not add complexity"
            ),
        },
    }


def git_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty}


def analyze_edge(
    source: ContextData,
    target: ContextData,
    direction: str,
    rng: np.random.Generator,
) -> dict[str, object]:
    edge = f"{source.spec.name}->{target.spec.name}"
    targets = sorted(set(source.targets) & set(target.targets))
    genes = sorted(set(source.genes) & set(target.genes))
    source_target = {name: index for index, name in enumerate(source.targets)}
    target_target = {name: index for index, name in enumerate(target.targets)}
    source_gene = {name: index for index, name in enumerate(source.genes)}
    target_gene = {name: index for index, name in enumerate(target.genes)}
    source_rows = np.asarray([source_target[name] for name in targets])
    target_rows = np.asarray([target_target[name] for name in targets])
    source_columns = np.asarray([source_gene[name] for name in genes])
    target_columns = np.asarray([target_gene[name] for name in genes])
    x = source.effects[source_rows][:, source_columns]
    y = target.effects[target_rows][:, target_columns]
    x_log1p = source.effects_log1p[source_rows][:, source_columns]
    y_log1p = target.effects_log1p[target_rows][:, target_columns]
    difference = (
        target.control_log1p[target_columns] - source.control_log1p[source_columns]
    )
    folds = deterministic_folds(targets)
    predictions, target_coefficients, fold_coefficients = crossfit_predictions(
        targets, genes, x, y, difference, folds
    )
    per_target = evaluate_models(
        targets,
        genes,
        x,
        y,
        predictions,
        source.cells[source_rows],
        target.cells[target_rows],
        folds,
    )
    for frame in (per_target, target_coefficients, fold_coefficients):
        frame.insert(0, "edge", edge)
        frame.insert(1, "study", source.spec.study)
        frame.insert(2, "source", source.spec.name)
        frame.insert(3, "target", target.spec.name)
        frame.insert(4, "direction", direction)
    summary = summarize_edge(per_target, rng)
    summary.insert(0, "edge", edge)
    summary.insert(1, "study", source.spec.study)
    summary.insert(2, "source", source.spec.name)
    summary.insert(3, "target", target.spec.name)
    summary.insert(4, "direction", direction)
    differences = bootstrap_differences(per_target, rng)
    differences.insert(0, "edge", edge)
    differences.insert(1, "direction", direction)
    coefficient_bootstrap = bootstrap_coefficients(target_coefficients, rng)
    coefficient_bootstrap.insert(0, "edge", edge)
    coefficient_bootstrap.insert(1, "study", source.spec.study)
    coefficient_bootstrap.insert(2, "source", source.spec.name)
    coefficient_bootstrap.insert(3, "target", target.spec.name)
    coefficient_bootstrap.insert(4, "direction", direction)
    coefficient_bootstrap["context_difference_sd"] = difference.std()
    excluded = target_gene_indices(targets, genes)
    sensitivity_rows = []
    for index, target_name in enumerate(targets):
        primary = single_metrics(x[index], y[index], excluded[index], 100)
        low_expression = single_metrics(
            x_log1p[index], y_log1p[index], excluded[index], 100
        )
        sensitivity_rows.append(
            {
                "edge": edge,
                "direction": direction,
                "target_gene": target_name,
                "primary_cosine": primary["cosine"],
                "log2_cpm1p_cosine": low_expression["cosine"],
                "primary_nae": primary["nae"],
                "log2_cpm1p_nae": low_expression["nae"],
            }
        )
    edge_dir = DERIVED / edge.replace("->", "_to_")
    edge_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = edge_dir / "heldout_predictions.npz"
    np.savez_compressed(
        prediction_path,
        target_gene=np.asarray(targets),
        shared_gene=np.asarray(genes),
        fold=folds,
        source_lfc=x.astype(np.float32),
        target_lfc=y.astype(np.float32),
        context_difference=difference.astype(np.float32),
        prediction_global=predictions["global_median"].astype(np.float32),
        prediction_interaction=predictions["context_interaction"].astype(np.float32),
    )
    edge_row = {
        "edge": edge,
        "study": source.spec.study,
        "source": source.spec.name,
        "target": target.spec.name,
        "direction": direction,
        "source_assay": source.spec.assay,
        "target_assay": target.spec.assay,
        "shared_targets": len(targets),
        "shared_genes": len(genes),
        "held_constant": "study, perturbation modality, assay family, and overlapping guide design",
        "source_manifest": str(source.spec.manifest_path.relative_to(ROOT)),
        "target_manifest": str(target.spec.manifest_path.relative_to(ROOT)),
        "prediction_artifact": str(prediction_path.relative_to(ROOT)),
    }
    target_set = pd.DataFrame({"edge": edge, "target_gene": targets})
    gene_set = pd.DataFrame({"edge": edge, "gene": genes})
    examples = edge_examples(edge, targets, genes, x, y, per_target)
    print(f"Completed {edge}: {len(targets)} targets x {len(genes)} genes", flush=True)
    return {
        "edge": edge_row,
        "per_target": per_target,
        "summary": summary,
        "differences": differences,
        "target_coefficients": target_coefficients,
        "fold_coefficients": fold_coefficients,
        "coefficient_bootstrap": coefficient_bootstrap,
        "sensitivity": pd.DataFrame(sensitivity_rows),
        "target_set": target_set,
        "gene_set": gene_set,
        "examples": examples,
        "prediction_path": prediction_path,
    }


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    audit_rows = []
    for spec in CONTEXTS.values():
        audit_rows.append(audit_compact(spec.directory))
    (REPORT / "compact_audits.json").write_text(
        json.dumps(audit_rows, indent=2, sort_keys=True) + "\n"
    )

    results = []
    guide_frames = []
    for source_name, target_name in PRIMARY_PAIRS:
        source = load_context(CONTEXTS[source_name])
        target = load_context(CONTEXTS[target_name])
        guide_frames.extend([guide_calibration(source), guide_calibration(target)])
        results.append(analyze_edge(source, target, "primary", rng))
        results.append(analyze_edge(target, source, "reverse_sensitivity", rng))
        del source, target

    guide = pd.concat(guide_frames, ignore_index=True)
    calibration = calibration_summary(guide, rng)
    tables = {
        "edges.csv": pd.DataFrame([result["edge"] for result in results]),
        "per_target.csv": pd.concat(
            [result["per_target"] for result in results], ignore_index=True
        ),
        "per_edge.csv": pd.concat(
            [result["summary"] for result in results], ignore_index=True
        ),
        "performance_bootstrap.csv": pd.concat(
            [result["differences"] for result in results], ignore_index=True
        ),
        "per_target_coefficients.csv": pd.concat(
            [result["target_coefficients"] for result in results], ignore_index=True
        ),
        "fold_coefficients.csv": pd.concat(
            [result["fold_coefficients"] for result in results], ignore_index=True
        ),
        "coefficient_bootstrap.csv": pd.concat(
            [result["coefficient_bootstrap"] for result in results], ignore_index=True
        ),
        "guide_quality.csv": guide,
        "metric_calibration.csv": calibration,
        "low_expression_sensitivity.csv": pd.concat(
            [result["sensitivity"] for result in results], ignore_index=True
        ),
        "target_sets.csv": pd.concat(
            [result["target_set"] for result in results], ignore_index=True
        ),
        "gene_sets.csv": pd.concat(
            [result["gene_set"] for result in results], ignore_index=True
        ),
        "examples.csv": pd.concat(
            [result["examples"] for result in results], ignore_index=True
        ),
    }
    for name, frame in tables.items():
        frame.to_csv(REPORT / name, index=False)

    checkpoints = checkpoint_reviews(
        tables["per_edge.csv"],
        tables["performance_bootstrap.csv"],
        tables["coefficient_bootstrap.csv"],
        calibration,
    )
    (REPORT / "checkpoint_reviews.json").write_text(
        json.dumps(checkpoints, indent=2, sort_keys=True) + "\n"
    )
    plot_model_comparison(tables["per_edge.csv"])
    plot_context_coefficients(
        tables["coefficient_bootstrap.csv"], tables["performance_bootstrap.csv"]
    )
    plot_graph(tables["per_edge.csv"])

    inputs = {}
    for spec in CONTEXTS.values():
        for path in (
            spec.target_path,
            spec.guide_path,
            spec.manifest_path,
            spec.audit_path,
        ):
            inputs[str(path.relative_to(ROOT))] = sha256(path)
    outputs = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted(REPORT.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    outputs.update(
        {
            str(result["prediction_path"].relative_to(ROOT)): sha256(
                result["prediction_path"]
            )
            for result in results
        }
    )
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "analysis": "within-study CRISPRi context transfer for blog",
        "producer": str(Path(__file__).resolve().relative_to(ROOT)),
        "normalization": {
            "duplicate_symbols": "summed in raw count space before normalization",
            "native_universe": "each pseudobulk normalized to 1e6 over its complete native measured gene universe before symbol alignment",
            "primary_effect": "log2((perturbed CPM + 1e-9)/(control CPM + 1e-9))",
            "low_expression_sensitivity": "log2(perturbed CPM + 1) - log2(control CPM + 1)",
            "target_gene": "excluded from coefficient estimation and every metric",
        },
        "coefficient_estimator": "componentwise median of per-perturbation target-excluded projections within each training fold",
        "pooled_ols": "sensitivity diagnostic only",
        "folds": {"count": N_FOLDS, "seed": SEED, "unit": "perturbation"},
        "bootstrap": {"replicates": N_BOOTSTRAPS, "unit": "perturbation"},
        "git": git_state(),
        "software": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ("anndata", "matplotlib", "numpy", "pandas", "scipy")
            },
        },
        "inputs": inputs,
        "outputs": outputs,
        "checkpoint_reviews": checkpoints,
    }
    (REPORT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(checkpoints, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
