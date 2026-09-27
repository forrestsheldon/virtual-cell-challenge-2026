"""Measure scale-free recovery of stable, strong H1 truth-DE genes."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse

from scripts.linear_response.kernel import (
    BULK_TARGET_SUM,
    CELL_TARGET_SUM,
    pca_covariance_columns,
)

ROOT = Path(__file__).resolve().parents[2]
H1_PATH = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
REFERENCE = ROOT / "reports/vcc2026-h1/reference_cells.csv"
DE_TABLE = (
    ROOT
    / "data/derived/linear_response/eval_cache/de_wilcoxon_table-12f179b9e966af64.parquet"
)
DERIVED = ROOT / "data/derived/linear_response"
REPORT = ROOT / "reports/linear-response-three-models"
MODEL1 = DERIVED / "model1_expected_profiles.npz"
MODEL2 = DERIVED / "model2_expected_profiles.npz"

DE_TARGET_SUM = 1_000_000.0
P_ADJ_THRESHOLD = 0.05
MIN_ABS_LOG2FC = 0.5
N_SPLITS = 5
MIN_STABLE_SPLITS = 4
MIN_GENES = 10
N_NULL_REPLICATES = 10_000
EPSILON = 1e-9
CHUNK_SIZE = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_halves(
    rows: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    permuted = rng.permutation(np.asarray(rows, dtype=np.int64))
    half = len(permuted) // 2
    return permuted[:half], permuted[half : 2 * half]


def split_membership(
    n_obs: int,
    control_rows: np.ndarray,
    target_rows: list[np.ndarray],
) -> np.ndarray:
    """Return five deterministic half-A membership masks."""
    membership = np.zeros((N_SPLITS, n_obs), dtype=bool)
    for split, seed in enumerate(np.random.SeedSequence(0).generate_state(N_SPLITS)):
        rng = np.random.default_rng(int(seed))
        half_a, _ = deterministic_halves(control_rows, rng)
        membership[split, half_a] = True
        for rows in target_rows:
            half_a, _ = deterministic_halves(rows, rng)
            membership[split, half_a] = True
    return membership


def accumulate_normalized_sums(
    data: ad.AnnData,
    selected_rows: np.ndarray,
    row_groups: np.ndarray,
    half_a: np.ndarray,
    gene_indices: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate full and half-A CPM sums in one sparse pass over selected cells."""
    full = np.zeros((n_groups, len(gene_indices)), dtype=np.float64)
    split_a = np.zeros((N_SPLITS, n_groups, len(gene_indices)), dtype=np.float64)
    for start in range(0, len(selected_rows), CHUNK_SIZE):
        rows = selected_rows[start : start + CHUNK_SIZE]
        groups = row_groups[rows]
        raw = data.X[rows].tocsr().astype(np.float64)
        totals = np.asarray(raw.sum(axis=1)).ravel()
        if np.any(totals <= 0):
            raise ValueError("DE normalization encountered a zero-count cell")
        normalized = raw.multiply((DE_TARGET_SUM / totals)[:, None]).tocsr()[
            :, gene_indices
        ]
        for group in np.unique(groups):
            in_group = groups == group
            full[group] += np.asarray(normalized[in_group].sum(axis=0)).ravel()
            for split in range(N_SPLITS):
                take = in_group & half_a[split, rows]
                if take.any():
                    split_a[split, group] += np.asarray(
                        normalized[take].sum(axis=0)
                    ).ravel()
        if (start // CHUNK_SIZE) % 25 == 0 or start + CHUNK_SIZE >= len(selected_rows):
            print(f"Truth stability normalization {min(start + CHUNK_SIZE, len(selected_rows))}/{len(selected_rows)}")
    return full, split_a


def stable_strong_mask(
    full_lfc: np.ndarray,
    p_adj: np.ndarray,
    full_sums: np.ndarray,
    split_a_sums: np.ndarray,
    group_counts: np.ndarray,
    split_a_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Require both half signs to match the full-panel sign in at least four splits."""
    full_sign = np.sign(full_lfc).astype(np.int8)
    strong = (p_adj < P_ADJ_THRESHOLD) & (np.abs(full_lfc) >= MIN_ABS_LOG2FC)
    stable_count = np.zeros_like(full_lfc, dtype=np.int8)
    for split in range(N_SPLITS):
        a_means = split_a_sums[split] / split_a_counts[split, :, None]
        b_counts = group_counts - split_a_counts[split]
        b_means = (full_sums - split_a_sums[split]) / b_counts[:, None]
        control_a, control_b = a_means[0], b_means[0]
        lfc_a = np.log2((a_means[1:] + EPSILON) / (control_a + EPSILON))
        lfc_b = np.log2((b_means[1:] + EPSILON) / (control_b + EPSILON))
        stable_count += (
            (np.sign(lfc_a) == full_sign)
            & (np.sign(lfc_b) == full_sign)
            & (full_sign != 0)
        )
    return strong & (stable_count >= MIN_STABLE_SPLITS), stable_count


def decoder_tangent(
    raw_controls: sparse.spmatrix | np.ndarray,
    effect_direction: np.ndarray,
) -> np.ndarray:
    """One-sided derivative of the exact normalized-count decoder at zero effect."""
    raw = sparse.csr_matrix(raw_controls, dtype=np.float64)
    totals = np.asarray(raw.sum(axis=1)).ravel()
    if np.any(totals <= 0):
        raise ValueError("decoder tangent requires positive source-cell totals")
    direction = np.asarray(effect_direction, dtype=np.float64)
    positive = np.maximum(direction, 0)
    negative = np.minimum(direction, 0)
    presence = raw.copy()
    presence.data.fill(1.0)
    pooled = np.asarray(raw.sum(axis=0)).ravel()
    weighted_presence = np.asarray(presence.T @ totals).ravel()
    d_weight_total = (
        positive.sum()
        + np.asarray(presence @ negative).ravel()
        + CELL_TARGET_SUM / totals * np.asarray(raw @ direction).ravel()
    )
    first = (
        positive * totals.sum() / CELL_TARGET_SUM
        + negative * weighted_presence / CELL_TARGET_SUM
        + direction * pooled
    )
    derivative_counts = first - np.asarray(raw.T @ d_weight_total).ravel() / CELL_TARGET_SUM
    pooled_total = pooled.sum()
    factor = (BULK_TARGET_SUM / pooled_total) / (
        1 + BULK_TARGET_SUM * pooled / pooled_total
    )
    return factor * derivative_counts


def signed_topk(
    prediction: np.ndarray,
    strong: np.ndarray,
    truth_sign: np.ndarray,
    target_gene_index: int | None,
) -> dict[str, float | int]:
    """Score strong-gene recovery after ranking by absolute predicted response."""
    valid = np.ones(len(prediction), dtype=bool)
    if target_gene_index is not None:
        valid[target_gene_index] = False
    k = int(strong[valid].sum())
    if k == 0:
        return {
            "n_strong": 0,
            "signed_recovery": np.nan,
            "unsigned_recall": np.nan,
            "sign_given_recovered": np.nan,
            "non_strong_energy_fraction": np.nan,
        }
    indices = np.flatnonzero(valid)
    top = indices[np.argpartition(np.abs(prediction[valid]), -k)[-k:]]
    recovered = strong[top]
    correct = recovered & (np.sign(prediction[top]) == truth_sign[top])
    total_energy = float(np.square(prediction[valid]).sum())
    strong_energy = float(np.square(prediction[strong & valid]).sum())
    overlap = int(recovered.sum())
    return {
        "n_strong": k,
        "signed_recovery": float(correct.sum() / k),
        "unsigned_recall": float(overlap / k),
        "sign_given_recovered": float(correct.sum() / overlap) if overlap else np.nan,
        "non_strong_energy_fraction": (
            float(1 - strong_energy / total_energy) if total_energy > 0 else np.nan
        ),
    }


def ranked_genes(prediction: np.ndarray) -> np.ndarray:
    return np.argsort(-np.abs(prediction), axis=1)


def pairwise_wrong_target_scores(
    prediction: np.ndarray,
    ranking: np.ndarray,
    strong: np.ndarray,
    truth_sign: np.ndarray,
    target_gene_indices: list[int | None],
) -> np.ndarray:
    """Score every truth target against every predicted target direction."""
    n_targets = len(prediction)
    scores = np.full((n_targets, n_targets), np.nan)
    for truth_index in range(n_targets):
        k = int(strong[truth_index].sum())
        if k < MIN_GENES:
            continue
        excluded = target_gene_indices[truth_index]
        for predicted_index in range(n_targets):
            order = ranking[predicted_index]
            if excluded is not None:
                order = order[order != excluded]
            top = order[:k]
            correct = strong[truth_index, top] & (
                np.sign(prediction[predicted_index, top])
                == truth_sign[truth_index, top]
            )
            scores[truth_index, predicted_index] = correct.sum() / k
    return scores


def derangement_null(
    pairwise: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    eligible = np.flatnonzero(np.isfinite(np.diag(pairwise)))
    target_indices = np.arange(len(pairwise))
    values = np.empty(N_NULL_REPLICATES)
    for replicate in range(N_NULL_REPLICATES):
        while True:
            permutation = rng.permutation(len(pairwise))
            if np.all(permutation != target_indices):
                break
        values[replicate] = np.nanmean(pairwise[eligible, permutation[eligible]])
    return values


def randomized_gene_null(
    prediction: np.ndarray,
    ranking: np.ndarray,
    strong: np.ndarray,
    truth_sign: np.ndarray,
    target_gene_indices: list[int | None],
    rng: np.random.Generator,
) -> np.ndarray:
    """Monte Carlo null for random gene labels while preserving top-K signs."""
    aggregate = np.zeros(N_NULL_REPLICATES)
    eligible = np.flatnonzero(strong.sum(axis=1) >= MIN_GENES)
    for target in eligible:
        excluded = target_gene_indices[target]
        valid = np.ones(prediction.shape[1], dtype=bool)
        if excluded is not None:
            valid[excluded] = False
        order = ranking[target]
        if excluded is not None:
            order = order[order != excluded]
        k = int(strong[target].sum())
        top_sign = np.sign(prediction[target, order[:k]])
        predicted_up = int((top_sign > 0).sum())
        predicted_down = int((top_sign < 0).sum())
        truth_up = int((strong[target] & (truth_sign[target] > 0)).sum())
        truth_down = int((strong[target] & (truth_sign[target] < 0)).sum())
        universe = int(valid.sum())
        up_in_up = rng.hypergeometric(
            truth_up, universe - truth_up, predicted_up, N_NULL_REPLICATES
        )
        down_in_up = rng.hypergeometric(
            truth_down,
            universe - truth_up - truth_down,
            predicted_up - up_in_up,
        )
        remaining_down = truth_down - down_in_up
        remaining_universe = universe - predicted_up
        down_in_down = rng.hypergeometric(
            remaining_down,
            remaining_universe - remaining_down,
            predicted_down,
        )
        aggregate += (up_in_up + down_in_down) / k
    return aggregate / len(eligible)


def summarize_null(
    model: str,
    observed: np.ndarray,
    wrong_target: np.ndarray,
    random_gene: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    finite = observed[np.isfinite(observed)]
    bootstrap = rng.choice(finite, size=(N_NULL_REPLICATES, len(finite)), replace=True).mean(1)
    observed_mean = float(finite.mean())
    return {
        "model": model,
        "eligible_targets": len(finite),
        "mean_signed_recovery": observed_mean,
        "median_signed_recovery": float(np.median(finite)),
        "bootstrap_q025": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_q975": float(np.quantile(bootstrap, 0.975)),
        "wrong_target_mean": float(wrong_target.mean()),
        "wrong_target_q95": float(np.quantile(wrong_target, 0.95)),
        "wrong_target_p_upper": float(
            (1 + np.sum(wrong_target >= observed_mean)) / (1 + len(wrong_target))
        ),
        "random_gene_mean": float(random_gene.mean()),
        "random_gene_q95": float(np.quantile(random_gene, 0.95)),
        "random_gene_p_upper": float(
            (1 + np.sum(random_gene >= observed_mean)) / (1 + len(random_gene))
        ),
    }


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values in original order."""
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.arange(1, len(values) + 1)
    adjusted_sorted = values[order] * len(values) / ranks
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1)
    return adjusted


def main() -> None:
    reference = pd.read_csv(REFERENCE)
    with np.load(MODEL1, allow_pickle=False) as model1, np.load(
        MODEL2, allow_pickle=False
    ) as model2:
        targets = model1["target_gene"].astype(str).tolist()
        genes = model1["gene_names"].astype(str).tolist()
        if targets != model2["target_gene"].astype(str).tolist():
            raise ValueError("Model target orders differ")
        if genes != model2["gene_names"].astype(str).tolist():
            raise ValueError("Model gene axes differ")
        source_order = model1["source_order"].astype(int)
        source_rows = model1["source_rows"].astype(np.int64)
        full_directions = model1["directions"].astype(np.float64)
        rank50_directions = model2["directions"].astype(np.float64)
        components = model2["components"].astype(np.float64)
        singular_values = model2["singular_values"].astype(np.float64)
        model1_finite = (
            model1["expected_log_bulk"] - model1["null_log_bulk"]
        ).astype(np.float64)
        model2_finite = (
            model2["expected_log_bulk"] - model2["null_log_bulk"]
        ).astype(np.float64)

    de = pl.read_parquet(DE_TABLE).to_pandas()
    de_genes = sorted(de["feature"].astype(str).unique())
    if len(de_genes) != 10_780:
        raise ValueError(f"expected 10,780 CPM-filtered genes, found {len(de_genes)}")
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    de_lookup = {gene: index for index, gene in enumerate(de_genes)}
    de_gene_indices = np.asarray([gene_lookup[gene] for gene in de_genes], dtype=int)
    full_lfc = np.empty((len(targets), len(de_genes)), dtype=np.float64)
    p_adj = np.empty_like(full_lfc)
    for target_index, target in enumerate(targets):
        group = de.loc[de["target"] == target].set_index("feature").reindex(de_genes)
        if group["log2_fold_change"].isna().any() or group["p_adj"].isna().any():
            raise ValueError(f"incomplete truth DE table for {target}")
        full_lfc[target_index] = group["log2_fold_change"].to_numpy()
        p_adj[target_index] = group["p_adj"].to_numpy()

    data = ad.read_h5ad(H1_PATH, backed="r")
    try:
        labels = data.obs["target_gene"].astype(str).to_numpy()
        control_rows = np.flatnonzero(labels == "non-targeting")
        target_rows = [
            reference.loc[reference["target_gene"] == target, "source_row"].to_numpy(
                dtype=np.int64
            )
            for target in targets
        ]
        if any(len(rows) != 400 or len(np.unique(rows)) != 400 for rows in target_rows):
            raise ValueError("every canonical target panel must contain 400 unique cells")
        row_groups = np.full(data.n_obs, -1, dtype=np.int16)
        row_groups[control_rows] = 0
        for target_index, rows in enumerate(target_rows, start=1):
            row_groups[rows] = target_index
        selected_rows = np.flatnonzero(row_groups >= 0)
        half_a = split_membership(data.n_obs, control_rows, target_rows)
        group_counts = np.bincount(
            row_groups[selected_rows], minlength=len(targets) + 1
        ).astype(np.int64)
        split_a_counts = np.stack(
            [
                np.bincount(
                    row_groups[selected_rows[half_a[split, selected_rows]]],
                    minlength=len(targets) + 1,
                )
                for split in range(N_SPLITS)
            ]
        ).astype(np.int64)
        if not np.all(split_a_counts[:, 1:] == 200):
            raise AssertionError("target truth halves must contain exactly 200 cells")
        full_sums, split_a_sums = accumulate_normalized_sums(
            data,
            selected_rows,
            row_groups,
            half_a,
            de_gene_indices,
            len(targets) + 1,
        )
        full_means = full_sums / group_counts[:, None]
        recomputed_lfc = np.log2(
            (full_means[1:] + EPSILON) / (full_means[0] + EPSILON)
        )
        lfc_difference = np.abs(recomputed_lfc - full_lfc)
        max_lfc_difference = float(lfc_difference.max())
        if max_lfc_difference > 1e-10:
            raise AssertionError(
                f"recomputed truth LFC differs from frozen cache by {max_lfc_difference}"
            )
        stable, stable_count = stable_strong_mask(
            full_lfc,
            p_adj,
            full_sums,
            split_a_sums,
            group_counts,
            split_a_counts,
        )

        target_gene_indices = [de_lookup.get(target) for target in targets]
        for target_index, gene_index in enumerate(target_gene_indices):
            if gene_index is not None:
                stable[target_index, gene_index] = False

        n_control_cells = json.loads((REPORT / "model1_manifest.json").read_text())[
            "control_cells"
        ]
        calibration_targets = pd.read_csv(REPORT / "h1_target_counts.csv")[
            "target_gene"
        ].astype(str).tolist()
        calibration_indices = [gene_lookup[target] for target in calibration_targets]
        top3_all = pca_covariance_columns(
            components[:3], singular_values[:3], n_control_cells, calibration_indices
        )
        top3 = top3_all[:, source_order]
        full = full_directions[:, source_order]
        rank50 = rank50_directions[:, source_order]

        model1_tangent = np.empty((len(targets), len(de_genes)), dtype=np.float64)
        model2_tangent = np.empty_like(model1_tangent)
        for target_index, rows in enumerate(source_rows):
            raw = data.X[np.sort(rows)].tocsr()
            model1_tangent[target_index] = decoder_tangent(
                raw, -full[:, target_index]
            )[de_gene_indices]
            model2_tangent[target_index] = decoder_tangent(
                raw, -rank50[:, target_index]
            )[de_gene_indices]
            if target_index % 20 == 0 or target_index + 1 == len(targets):
                print(f"Decoder tangents {target_index + 1}/{len(targets)}")
    finally:
        data.file.close()

    predictions = {
        "model1_covariance_direction": -full[de_gene_indices].T,
        "model1_decoder_tangent": model1_tangent,
        "model1_decoder_finite": model1_finite[:, de_gene_indices],
        "model2_rank50_direction": -rank50[de_gene_indices].T,
        "model2_decoder_tangent": model2_tangent,
        "model2_decoder_finite": model2_finite[:, de_gene_indices],
        "shared_top3_direction": -top3[de_gene_indices].T,
        "model1_residual_after_top3": -(full - top3)[de_gene_indices].T,
    }
    truth_sign = np.sign(full_lfc).astype(np.int8)
    per_target_rows: list[dict] = []
    summary_rows: list[dict] = []
    for model_index, (name, prediction) in enumerate(predictions.items()):
        if not np.isfinite(prediction).all():
            raise ValueError(f"{name} contains non-finite predictions")
        ranking = ranked_genes(prediction)
        observed = np.full(len(targets), np.nan)
        model_rows: list[dict] = []
        for target_index, target in enumerate(targets):
            result = signed_topk(
                prediction[target_index],
                stable[target_index],
                truth_sign[target_index],
                target_gene_indices[target_index],
            )
            eligible = int(result["n_strong"]) >= MIN_GENES
            if eligible:
                observed[target_index] = float(result["signed_recovery"])
            model_rows.append(
                {
                    "model": name,
                    "target_gene": target,
                    "eligible": eligible,
                    **result,
                }
            )
        pairwise = pairwise_wrong_target_scores(
            prediction, ranking, stable, truth_sign, target_gene_indices
        )
        if not np.allclose(
            observed[np.isfinite(observed)],
            np.diag(pairwise)[np.isfinite(observed)],
        ):
            raise AssertionError(f"{name}: direct and pairwise diagonal scores differ")
        for target_index, row in enumerate(model_rows):
            if not row["eligible"]:
                row.update(
                    wrong_target_mean=np.nan,
                    wrong_target_q95=np.nan,
                    wrong_target_p_upper=np.nan,
                    signed_recovery_excess_wrong_target=np.nan,
                )
                continue
            null = np.delete(pairwise[target_index], target_index)
            null = null[np.isfinite(null)]
            row.update(
                wrong_target_mean=float(null.mean()),
                wrong_target_q95=float(np.quantile(null, 0.95)),
                wrong_target_p_upper=float(
                    (1 + np.sum(null >= observed[target_index])) / (1 + len(null))
                ),
                signed_recovery_excess_wrong_target=float(
                    observed[target_index] - null.mean()
                ),
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
            pairwise, np.random.default_rng(np.random.SeedSequence([0, 41, model_index]))
        )
        random_gene = randomized_gene_null(
            prediction,
            ranking,
            stable,
            truth_sign,
            target_gene_indices,
            np.random.default_rng(np.random.SeedSequence([0, 42, model_index])),
        )
        summary_rows.append(
            summarize_null(
                name,
                observed,
                wrong_target,
                random_gene,
                np.random.default_rng(np.random.SeedSequence([0, 43, model_index])),
            )
        )
        summary_rows[-1]["wrong_target_p_bonferroni_models"] = min(
            1.0, float(summary_rows[-1]["wrong_target_p_upper"]) * len(predictions)
        )
        summary_rows[-1]["random_gene_p_bonferroni_models"] = min(
            1.0, float(summary_rows[-1]["random_gene_p_upper"]) * len(predictions)
        )
        print(f"Scored {name}")

    candidate = (p_adj < P_ADJ_THRESHOLD) & (np.abs(full_lfc) >= MIN_ABS_LOG2FC)
    for target_index, gene_index in enumerate(target_gene_indices):
        if gene_index is not None:
            candidate[target_index, gene_index] = False
    truth_rows = []
    for target_index, target in enumerate(targets):
        for gene_index in np.flatnonzero(candidate[target_index]):
            truth_rows.append(
                {
                    "target_gene": target,
                    "feature": de_genes[gene_index],
                    "log2_fold_change": full_lfc[target_index, gene_index],
                    "p_adj": p_adj[target_index, gene_index],
                    "stable_full_sign_splits": int(stable_count[target_index, gene_index]),
                    "stable_strong": bool(stable[target_index, gene_index]),
                }
            )

    REPORT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "truth": REPORT / "strong_de_truth_genes.csv",
        "per_target": REPORT / "strong_de_recovery_per_target.csv",
        "summary": REPORT / "strong_de_recovery_summary.csv",
    }
    pd.DataFrame(truth_rows).to_csv(outputs["truth"], index=False)
    pd.DataFrame(per_target_rows).to_csv(outputs["per_target"], index=False)
    pd.DataFrame(summary_rows).to_csv(outputs["summary"], index=False)
    manifest = {
        "kind": "scale-free recovery of stable strong truth-DE genes",
        "truth_definition": {
            "control_cpm": ">5, inherited from frozen DE table",
            "p_adj": f"<{P_ADJ_THRESHOLD}",
            "abs_log2_fold_change": f">={MIN_ABS_LOG2FC}",
            "target_gene_excluded": True,
            "stability": f"both 200-cell half signs match full sign in >={MIN_STABLE_SPLITS}/{N_SPLITS} deterministic splits",
            "minimum_genes_per_target": MIN_GENES,
        },
        "score": "correct-sign truth genes among top K absolute predictions, K=stable strong truth count",
        "aggregation": "equal-weight mean over eligible perturbations",
        "full_lfc_positive_control": {
            "maximum_absolute_difference_from_frozen_cache": max_lfc_difference,
            "tolerance": 1e-10,
        },
        "null_replicates": N_NULL_REPLICATES,
        "seeds": {
            "truth_splits": "SeedSequence(0).generate_state(5)",
            "wrong_target": "SeedSequence([0,41,model_index])",
            "random_gene": "SeedSequence([0,42,model_index])",
            "bootstrap": "SeedSequence([0,43,model_index])",
        },
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [H1_PATH, REFERENCE, DE_TABLE, MODEL1, MODEL2]
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
                package: version(package)
                for package in ["anndata", "numpy", "pandas", "polars", "scipy"]
            },
        },
    }
    (REPORT / "strong_de_recovery_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
