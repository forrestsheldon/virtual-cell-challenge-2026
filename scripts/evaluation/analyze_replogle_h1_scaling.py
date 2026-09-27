"""Diagnose amplitude versus direction in pooled Replogle-to-H1 residuals."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.evaluation.prepare_replogle_h1_transfer import loo_components

ROOT = Path(__file__).resolve().parents[2]
EFFECTS = ROOT / "data/derived/replogle_h1_transfer/effects.npz"
REPLOGLE = ROOT / "data/external/replogle2022/K562_gwps_raw_bulk_01.h5ad"
H1 = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
STRONG = ROOT / "reports/linear-response-three-models/strong_de_truth_genes.csv"
TARGET_AUDIT = ROOT / "reports/replogle-h1-transfer/target_overlap_and_knockdown.csv"
PHASE1 = ROOT / "reports/replogle-h1-transfer/phase1"
REPORT = ROOT / "reports/replogle-h1-transfer/phase2"
MIN_STRONG = 10


def projection_diagnostics(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    source_energy = float(source @ source)
    target_energy = float(target @ target)
    if source_energy == 0 or target_energy == 0:
        return {
            "oracle_scale": np.nan,
            "cosine": np.nan,
            "source_to_target_norm_ratio": np.nan,
            "orthogonal_norm": np.nan,
            "orthogonal_energy_fraction": np.nan,
        }
    scale = float(source @ target / source_energy)
    orthogonal = target - scale * source
    return {
        "oracle_scale": scale,
        "cosine": float(source @ target / np.sqrt(source_energy * target_energy)),
        "source_to_target_norm_ratio": float(np.sqrt(source_energy / target_energy)),
        "orthogonal_norm": float(np.linalg.norm(orthogonal)),
        "orthogonal_energy_fraction": float(orthogonal @ orthogonal / target_energy),
    }


def quantiles(values: pd.Series | np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        name: float(value)
        for name, value in zip(
            ("min", "q10", "q25", "median", "q75", "q90", "max"),
            np.quantile(values, (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)),
        )
    }


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def fast_metric(arm: str, metric: str) -> pd.Series:
    frame = pd.read_csv(PHASE1 / "cell_eval" / arm / "per_target.csv")
    return frame.loc[frame.metric == metric].set_index("target_gene").value


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    with np.load(EFFECTS, allow_pickle=False) as saved:
        targets = saved["target_gene"].astype(str).tolist()
        matched = saved["matched_target_gene"].astype(str).tolist()
        genes = saved["shared_gene"].astype(str).tolist()
        source = saved["replogle_lfc_native"].astype(np.float64)
        h1 = saved["h1_lfc_native"].astype(np.float64)
        k_control_counts = saved["replogle_control_count_sum"].astype(np.float64)
        h_control_counts = saved["h1_control_count_sum"].astype(np.float64)

    source_global, source_residual = loo_components(source)
    h1_global, h1_residual = loo_components(h1)
    del source_global, h1_global
    target_lookup = {target: index for index, target in enumerate(targets)}
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    stable = pd.read_csv(STRONG).query("stable_strong")
    strong_by_target = {
        target: set(frame.feature) for target, frame in stable.groupby("target_gene")
    }

    target_audit = pd.read_csv(TARGET_AUDIT).set_index("target_gene")
    replogle = ad.read_h5ad(REPLOGLE, backed="r")
    try:
        _, k_genes = pd.factorize(
            replogle.var["gene_name"].astype(str).to_numpy(), sort=False
        )
        k_genes = k_genes.astype(str).tolist()
    finally:
        replogle.file.close()
    h1_data = ad.read_h5ad(H1, backed="r")
    try:
        h_genes = h1_data.var_names.astype(str).tolist()
    finally:
        h1_data.file.close()
    k_gene_lookup = {gene: index for index, gene in enumerate(k_genes)}
    h_gene_lookup = {gene: index for index, gene in enumerate(h_genes)}
    k_control_cpm = 1_000_000 * k_control_counts / k_control_counts.sum()
    h_control_cpm = 1_000_000 * h_control_counts / h_control_counts.sum()

    rows = []
    for source_index, target in enumerate(matched):
        h1_index = target_lookup[target]
        valid = np.ones(len(genes), dtype=bool)
        if target in gene_lookup:
            valid[gene_lookup[target]] = False
        source_vector = source_residual[source_index, valid]
        target_vector = h1_residual[h1_index, valid]
        all_gene = projection_diagnostics(source_vector, target_vector)

        strong_indices = np.asarray(
            [
                index
                for index, gene in enumerate(genes)
                if gene != target and gene in strong_by_target.get(target, set())
            ]
        )
        strong_gene = (
            projection_diagnostics(
                source_residual[source_index, strong_indices],
                h1_residual[h1_index, strong_indices],
            )
            if len(strong_indices)
            else projection_diagnostics(np.zeros(1), np.zeros(1))
        )

        k_expression = (
            np.log2(1 + k_control_cpm[k_gene_lookup[target]])
            if target in k_gene_lookup
            else np.nan
        )
        h_expression = (
            np.log2(1 + h_control_cpm[h_gene_lookup[target]])
            if target in h_gene_lookup
            else np.nan
        )
        rows.append(
            {
                "target_gene": target,
                **{f"all_gene_{key}": value for key, value in all_gene.items()},
                "source_residual_norm": float(np.linalg.norm(source_vector)),
                "source_residual_energy": float(source_vector @ source_vector),
                "n_stable_strong_shared_genes": len(strong_indices),
                **{f"strong_gene_{key}": value for key, value in strong_gene.items()},
                "k562_target_control_log2_cpm1p": k_expression,
                "h1_target_control_log2_cpm1p": h_expression,
                "h1_minus_k562_target_control_log2_cpm1p": h_expression
                - k_expression,
                "k562_on_target_log2fc": target_audit.loc[
                    target, "on_target_log2fc"
                ],
                "source_cells": target_audit.loc[target, "n_cells"],
            }
        )
    per_target = pd.DataFrame(rows).set_index("target_gene")

    unchanged_mse = fast_metric("unchanged", "expr_mse_unbiased_capped")
    residual_mse = fast_metric("source_residual", "expr_mse_unbiased_capped")
    full_mse = fast_metric("source_effect", "expr_mse_unbiased_capped")
    real_distance = fast_metric("source_effect", "expr_distance_unbiased")
    per_target["unchanged_expression_mse"] = unchanged_mse.reindex(per_target.index)
    per_target["residual_expression_mse"] = residual_mse.reindex(per_target.index)
    per_target["full_expression_mse"] = full_mse.reindex(per_target.index)
    per_target["reference_expression_distance"] = real_distance.reindex(
        per_target.index
    )
    per_target["residual_mse_excess_over_unchanged"] = (
        per_target.residual_expression_mse - per_target.unchanged_expression_mse
    )
    per_target["full_mse_excess_over_unchanged"] = (
        per_target.full_expression_mse - per_target.unchanged_expression_mse
    )
    per_target.reset_index().to_csv(REPORT / "oracle_scaling_per_target.csv", index=False)

    features = [
        "k562_target_control_log2_cpm1p",
        "h1_target_control_log2_cpm1p",
        "h1_minus_k562_target_control_log2_cpm1p",
        "k562_on_target_log2fc",
        "source_cells",
        "source_residual_norm",
    ]
    outcomes = [
        "all_gene_oracle_scale",
        "all_gene_cosine",
        "strong_gene_oracle_scale",
        "strong_gene_cosine",
        "strong_gene_orthogonal_energy_fraction",
        "full_mse_excess_over_unchanged",
    ]
    association_rows = []
    for outcome in outcomes:
        for feature in features:
            frame = per_target[[feature, outcome]].dropna()
            if outcome.startswith("strong_gene"):
                frame = frame.loc[
                    per_target.loc[frame.index, "n_stable_strong_shared_genes"]
                    >= MIN_STRONG
                ]
            result = spearmanr(frame[feature], frame[outcome])
            association_rows.append(
                {
                    "outcome": outcome,
                    "feature": feature,
                    "n_targets": len(frame),
                    "spearman_rho": result.statistic,
                    "p_value": result.pvalue,
                }
            )
    associations = pd.DataFrame(association_rows)
    associations["p_adj_bh_all_tests"] = bh_adjust(associations.p_value.to_numpy())
    associations.to_csv(REPORT / "feature_associations.csv", index=False)

    eligible = per_target.n_stable_strong_shared_genes >= MIN_STRONG
    positive_excess = per_target.full_mse_excess_over_unchanged.clip(lower=0).sort_values(
        ascending=False
    )
    total_positive = positive_excess.sum()
    summary = {
        "all_shared_genes": {
            "targets": len(per_target),
            "oracle_scale": quantiles(per_target.all_gene_oracle_scale),
            "cosine": quantiles(per_target.all_gene_cosine),
            "orthogonal_energy_fraction": quantiles(
                per_target.all_gene_orthogonal_energy_fraction
            ),
            "fraction_positive_scale": float((per_target.all_gene_oracle_scale > 0).mean()),
        },
        "stable_strong_shared_genes": {
            "eligible_targets": int(eligible.sum()),
            "oracle_scale": quantiles(per_target.loc[eligible, "strong_gene_oracle_scale"]),
            "cosine": quantiles(per_target.loc[eligible, "strong_gene_cosine"]),
            "orthogonal_energy_fraction": quantiles(
                per_target.loc[eligible, "strong_gene_orthogonal_energy_fraction"]
            ),
            "fraction_positive_scale": float(
                (per_target.loc[eligible, "strong_gene_oracle_scale"] > 0).mean()
            ),
        },
        "expression_mse_concentration_shared_targets": {
            "targets_worse_than_unchanged": int(
                (per_target.full_mse_excess_over_unchanged > 0).sum()
            ),
            "targets_better_than_unchanged": int(
                (per_target.full_mse_excess_over_unchanged < 0).sum()
            ),
            "top_5_fraction_of_positive_excess": float(
                positive_excess.iloc[:5].sum() / total_positive
            ),
            "top_10_fraction_of_positive_excess": float(
                positive_excess.iloc[:10].sum() / total_positive
            ),
            "largest_positive_contributors": positive_excess.head(10).index.tolist(),
        },
        "multiple_testing": "Benjamini-Hochberg across all predefined outcome-feature associations",
    }
    (REPORT / "oracle_scaling_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nAssociations with BH-adjusted p < 0.05")
    print(
        associations.loc[associations.p_adj_bh_all_tests < 0.05].to_string(index=False)
    )


if __name__ == "__main__":
    main()
