"""Lightweight expected-profile evaluation for H1 linear-response models.

This is deliberately not an official scorer. It owns the canonical H1 truth cells,
uses cell_eval2's PDS kernel, and emits simple paired profile diagnostics without DE
or score scaling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from cell_eval2.ceiling import _spearman_brown
from cell_eval2.metrics.discrimination import discrimination_score
from scipy.stats import spearmanr

from scripts.linear_response.kernel import log_pseudobulk

ROOT = Path(__file__).resolve().parents[2]
H1_PATH = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
REFERENCE_CELLS = ROOT / "reports/vcc2026-h1/reference_cells.csv"
REPORT_DIR = ROOT / "reports/linear-response-three-models"
CONTROL = "non-targeting"
N_SPLITS = 5
H1_SHA256 = "a09977104fefb622368ca74b50c9d3c1e891733e6c83db07acfca49b0219c02b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else 0.0


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return 0.0
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else 0.0


def pds(
    predicted_profiles: np.ndarray,
    truth_profiles: np.ndarray,
    control_profile: np.ndarray,
    targets: list[str],
    genes: list[str],
) -> dict[str, float]:
    real_labels = np.asarray([CONTROL, *targets])
    real_profiles = np.vstack([control_profile, truth_profiles])
    return discrimination_score(
        pred_bulk=(np.asarray(targets), predicted_profiles),
        real_bulk=(real_labels, real_profiles),
        pert_col="target_gene",
        control=CONTROL,
        distance="cosine",
        rank_denominator="n-1",
        tie_policy="midrank",
        exclude_target_gene=True,
        exclusion_scope="panel",
        control_source="real",
        genes=np.asarray(genes),
    )


def paired_profile_metrics(
    predicted_profiles: np.ndarray,
    null_profiles: np.ndarray,
    truth_profiles: np.ndarray,
    control_profile: np.ndarray,
    targets: list[str],
    genes: list[str],
) -> pd.DataFrame:
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    predicted_effects = predicted_profiles - control_profile
    null_effects = null_profiles - control_profile
    truth_effects = truth_profiles - control_profile
    predicted_pds = pds(predicted_profiles, truth_profiles, control_profile, targets, genes)
    null_pds = pds(null_profiles, truth_profiles, control_profile, targets, genes)
    rows = []
    for index, target in enumerate(targets):
        mask = np.ones(len(genes), dtype=bool)
        mask[gene_lookup[target]] = False
        observed = truth_effects[index, mask]
        predicted = predicted_effects[index, mask]
        null = null_effects[index, mask]
        rows.append(
            {
                "target_gene": target,
                "expected_cosine": cosine(predicted, observed),
                "null_cosine": cosine(null, observed),
                "expected_spearman": spearman(predicted, observed),
                "null_spearman": spearman(null, observed),
                "expected_pds": predicted_pds[target],
                "null_pds": null_pds[target],
                "expected_squared_error": float(np.square(predicted - observed).sum()),
                "null_squared_error": float(np.square(null - observed).sum()),
                "control_squared_error": float(np.square(observed).sum()),
                "truth_strength": float(np.linalg.norm(observed)),
            }
        )
    result = pd.DataFrame(rows)
    result["response_stratum"] = pd.qcut(
        result["truth_strength"].rank(method="first"),
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    ).astype(str)
    return result


def summarize_profiles(
    per_target: pd.DataFrame,
    sampled: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for scope in ["overall", "Q1", "Q2", "Q3", "Q4"]:
        group = (
            per_target
            if scope == "overall"
            else per_target[per_target["response_stratum"] == scope]
        )
        baseline = group["control_squared_error"].sum()
        for arm in ["expected", "null"]:
            arm_label = "expected" if arm == "expected" else "normalized_decoder_null"
            rows.extend(
                [
                    {
                        "scope": scope,
                        "arm": arm_label,
                        "metric": "cosine",
                        "aggregation": "median",
                        "value": group[f"{arm}_cosine"].median(),
                    },
                    {
                        "scope": scope,
                        "arm": arm_label,
                        "metric": "spearman",
                        "aggregation": "median",
                        "value": group[f"{arm}_spearman"].median(),
                    },
                    {
                        "scope": scope,
                        "arm": arm_label,
                        "metric": "pds",
                        "aggregation": "mean",
                        "value": group[f"{arm}_pds"].mean(),
                    },
                    {
                        "scope": scope,
                        "arm": arm_label,
                        "metric": "squared_error_ratio",
                        "aggregation": "ratio_of_sums",
                        "value": group[f"{arm}_squared_error"].sum() / baseline,
                    },
                ]
            )
    if sampled is not None:
        base = sampled["control_squared_error"].sum()
        model_ratio = sampled["sampled_squared_error"].sum() / base
        null_ratio = sampled["sampled_null_squared_error"].sum() / base
        expected_ratio = per_target["expected_squared_error"].sum() / base
        rows.extend(
            [
                {
                    "scope": "overall",
                    "arm": "sampled",
                    "metric": "squared_error_ratio",
                    "aggregation": "ratio_of_sums",
                    "value": model_ratio,
                },
                {
                    "scope": "overall",
                    "arm": "sampled_null",
                    "metric": "squared_error_ratio",
                    "aggregation": "ratio_of_sums",
                    "value": null_ratio,
                },
                {
                    "scope": "overall",
                    "arm": "sampled_additive_audit",
                    "metric": "squared_error_ratio",
                    "aggregation": "model/base - (null/base - 1)",
                    "value": model_ratio - (null_ratio - 1),
                },
                {
                    "scope": "overall",
                    "arm": "expected_audit_target",
                    "metric": "squared_error_ratio",
                    "aggregation": "ratio_of_sums",
                    "value": expected_ratio,
                },
            ]
        )
    return pd.DataFrame(rows)


def collect_reference_profiles(
    data: ad.AnnData, reference: pd.DataFrame, targets: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.empty((len(targets), data.n_vars), dtype=np.float64)
    for index, target in enumerate(targets):
        rows = reference.loc[reference["target_gene"] == target, "source_row"].to_numpy(
            dtype=int
        )
        if len(rows) != 400 or len(np.unique(rows)) != 400:
            raise ValueError(f"{target}: canonical reference must contain 400 unique cells")
        truth[index] = log_pseudobulk(data.X[np.sort(rows)].tocsr())
    labels = data.obs["target_gene"].astype(str).to_numpy()
    control_rows = np.flatnonzero(labels == CONTROL)
    return truth, log_pseudobulk(data.X[control_rows].tocsr())


def deterministic_halves(rows: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    permuted = rng.permutation(np.asarray(rows, dtype=int))
    half = len(permuted) // 2
    return permuted[:half], permuted[half : 2 * half]


def ceiling_diagnostics(
    data: ad.AnnData,
    reference: pd.DataFrame,
    targets: list[str],
    genes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = data.obs["target_gene"].astype(str).to_numpy()
    control_rows = np.flatnonzero(labels == CONTROL)
    target_rows = {
        target: reference.loc[
            reference["target_gene"] == target, "source_row"
        ].to_numpy(dtype=int)
        for target in targets
    }
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    seeds = np.random.SeedSequence(0).generate_state(N_SPLITS)
    records = []
    pds_means = []
    for split_index, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        control_a_rows, control_b_rows = deterministic_halves(control_rows, rng)
        control_a = log_pseudobulk(data.X[np.sort(control_a_rows)].tocsr())
        control_b = log_pseudobulk(data.X[np.sort(control_b_rows)].tocsr())
        profiles_a = np.empty((len(targets), data.n_vars), dtype=np.float64)
        profiles_b = np.empty_like(profiles_a)
        split_rows: list[tuple[np.ndarray, np.ndarray]] = []
        for index, target in enumerate(targets):
            rows_a, rows_b = deterministic_halves(target_rows[target], rng)
            if len(rows_a) != 200 or len(rows_b) != 200:
                raise AssertionError("canonical target split must be 200 vs 200")
            if np.intersect1d(rows_a, rows_b).size:
                raise AssertionError("ceiling halves are not disjoint")
            split_rows.append((rows_a, rows_b))
            profiles_a[index] = log_pseudobulk(data.X[np.sort(rows_a)].tocsr())
            profiles_b[index] = log_pseudobulk(data.X[np.sort(rows_b)].tocsr())
        pds_values = pds(profiles_b, profiles_a, control_a, targets, genes)
        pds_means.append(np.mean(list(pds_values.values())))
        effects_a = profiles_a - control_a
        effects_b = profiles_b - control_b
        for index, target in enumerate(targets):
            mask = np.ones(len(genes), dtype=bool)
            mask[gene_lookup[target]] = False
            first = effects_a[index, mask]
            second = effects_b[index, mask]
            records.append(
                {
                    "split_index": split_index,
                    "seed": int(seed),
                    "target_gene": target,
                    "cells_per_half": len(split_rows[index][0]),
                    "cosine": cosine(first, second),
                    "spearman": spearman(first, second),
                    "disagreement_energy_fraction": float(
                        np.square(first - second).sum()
                        / (np.square(first).sum() + np.square(second).sum())
                    ),
                    "pds": pds_values[target],
                }
            )
    raw_pds = float(np.mean(pds_means))
    corrected = _spearman_brown(
        pl.DataFrame({"metric": ["pds_cosine"], "mean": [raw_pds]}),
        ["pds_cosine"],
    )["ceiling"].item()
    per_target = pd.DataFrame(records)
    summary = pd.DataFrame(
        [
            {"metric": "split_half_cosine", "value": per_target["cosine"].median()},
            {"metric": "split_half_spearman", "value": per_target["spearman"].median()},
            {
                "metric": "disagreement_energy_fraction",
                "value": per_target["disagreement_energy_fraction"].median(),
            },
            {"metric": "pds_cosine_raw_200v200", "value": raw_pds},
            {"metric": "pds_cosine_spearman_brown_400", "value": corrected},
        ]
    )
    return per_target, summary


def truth_geometry(
    truth_profiles: np.ndarray,
    control_profile: np.ndarray,
    targets: list[str],
    genes: list[str],
) -> pd.DataFrame:
    effects = truth_profiles - control_profile
    lookup = {gene: index for index, gene in enumerate(genes)}
    for index, target in enumerate(targets):
        effects[index, lookup[target]] = 0
    norms = np.linalg.norm(effects, axis=1)
    unit = np.divide(effects, norms[:, None], out=np.zeros_like(effects), where=norms[:, None] > 0)
    similarities = unit @ unit.T
    upper = similarities[np.triu_indices(len(targets), k=1)]
    singular = np.linalg.svd(unit, compute_uv=False)
    energy = singular**2
    probabilities = energy / energy.sum()
    effective_rank = np.exp(-(probabilities * np.log(probabilities)).sum())
    np.fill_diagonal(similarities, -np.inf)
    return pd.DataFrame(
        [
            {"statistic": "pairwise_cosine_q10", "value": np.quantile(upper, 0.1)},
            {"statistic": "pairwise_cosine_median", "value": np.median(upper)},
            {"statistic": "pairwise_cosine_q90", "value": np.quantile(upper, 0.9)},
            {"statistic": "nearest_cosine_median", "value": np.median(similarities.max(axis=1))},
            {"statistic": "top3_axis_energy_fraction", "value": probabilities[:3].sum()},
            {"statistic": "effective_rank", "value": effective_rank},
        ]
    )


def evaluate(args: argparse.Namespace) -> None:
    artifact = np.load(args.artifact)
    targets = artifact["target_gene"].astype(str).tolist()
    genes = artifact["gene_names"].astype(str).tolist()
    reference = pd.read_csv(args.reference_cells)
    if reference.drop_duplicates("target_gene")["target_gene"].astype(str).tolist() != targets:
        raise ValueError("model target order does not match the canonical H1 reference")
    data = ad.read_h5ad(args.h1, backed="r")
    try:
        if data.var_names.astype(str).tolist() != genes:
            raise ValueError("model gene axis does not match H1")
        truth_profiles, control_profile = collect_reference_profiles(data, reference, targets)
        per_target = paired_profile_metrics(
            artifact["expected_log_bulk"],
            artifact["null_log_bulk"],
            truth_profiles,
            control_profile,
            targets,
            genes,
        )
        sampled = None
        if "sampled_log_bulk" in artifact and "sampled_null_log_bulk" in artifact:
            sampled = paired_profile_metrics(
                artifact["sampled_log_bulk"],
                artifact["sampled_null_log_bulk"],
                truth_profiles,
                control_profile,
                targets,
                genes,
            ).rename(
                columns={
                    "expected_squared_error": "sampled_squared_error",
                    "null_squared_error": "sampled_null_squared_error",
                }
            )
        summary = summarize_profiles(per_target, sampled)
        if "posterior_null_log_bulk" in artifact:
            posterior_null = paired_profile_metrics(
                artifact["posterior_null_log_bulk"],
                artifact["null_log_bulk"],
                truth_profiles,
                control_profile,
                targets,
                genes,
            )
            posterior_columns = {
                column: column.replace("expected_", "posterior_null_")
                for column in posterior_null.columns
                if column.startswith("expected_")
            }
            per_target = per_target.merge(
                posterior_null[["target_gene", *posterior_columns]].rename(
                    columns=posterior_columns
                ),
                on="target_gene",
                validate="one_to_one",
            )
            posterior_summary = summarize_profiles(posterior_null)
            posterior_summary = posterior_summary[
                posterior_summary["arm"] == "expected"
            ].copy()
            posterior_summary["arm"] = "model3_posterior_null"
            summary = pd.concat([summary, posterior_summary], ignore_index=True)
        geometry = truth_geometry(truth_profiles, control_profile, targets, genes)
        if args.skip_ceiling:
            ceiling_per_target = ceiling_summary = None
        else:
            ceiling_per_target, ceiling_summary = ceiling_diagnostics(
                data, reference, targets, genes
            )
    finally:
        data.file.close()

    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_target": args.output / "per_target.csv",
        "summary": args.output / "summary.csv",
        "truth_geometry": REPORT_DIR / "h1_truth_geometry.csv",
    }
    per_target.to_csv(paths["per_target"], index=False)
    summary.to_csv(paths["summary"], index=False)
    geometry.to_csv(paths["truth_geometry"], index=False)
    if ceiling_per_target is not None and ceiling_summary is not None:
        paths["ceiling_per_target"] = REPORT_DIR / "h1_profile_ceiling_per_target.csv"
        paths["ceiling_summary"] = REPORT_DIR / "h1_profile_ceiling_summary.csv"
        ceiling_per_target.to_csv(paths["ceiling_per_target"], index=False)
        ceiling_summary.to_csv(paths["ceiling_summary"], index=False)
        ceiling_manifest = {
            "kind": "H1 expected-profile reliability ceiling",
            "h1": {"path": str(args.h1), "sha256": H1_SHA256},
            "reference_cells": {
                "path": str(args.reference_cells),
                "sha256": sha256(args.reference_cells),
            },
            "splits": "five deterministic disjoint 200-vs-200 target splits with independently split controls",
            "seeds": np.random.SeedSequence(0).generate_state(N_SPLITS).tolist(),
            "pds_implementation": "cell_eval2 ceiling and discrimination kernels",
            "outputs": {
                name: {"path": str(path), "sha256": sha256(path)}
                for name, path in paths.items()
                if name.startswith("ceiling_")
            },
            "software": {
                "python": platform.python_version(),
                "cell-eval2": version("cell-eval2"),
                "numpy": version("numpy"),
                "pandas": version("pandas"),
                "polars": version("polars"),
            },
        }
        (REPORT_DIR / "h1_profile_ceiling_manifest.json").write_text(
            json.dumps(ceiling_manifest, indent=2) + "\n"
        )
    manifest = {
        "kind": "expected-profile diagnostics; not an official score",
        "truth_cells_read": True,
        "model_artifact": {"path": str(args.artifact), "sha256": sha256(args.artifact)},
        "h1": {"path": str(args.h1), "sha256": H1_SHA256},
        "reference_cells": {
            "path": str(args.reference_cells),
            "sha256": sha256(args.reference_cells),
        },
        "pds_implementation": "cell_eval2.metrics.discrimination.discrimination_score",
        "ceiling": "five deterministic disjoint 200-vs-200 target splits",
        "ceiling_seeds": np.random.SeedSequence(0).generate_state(N_SPLITS).tolist(),
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "software": {
            "python": platform.python_version(),
            "anndata": version("anndata"),
            "cell-eval2": version("cell-eval2"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "polars": version("polars"),
            "scipy": version("scipy"),
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--h1", type=Path, default=H1_PATH)
    parser.add_argument("--reference-cells", type=Path, default=REFERENCE_CELLS)
    parser.add_argument("--skip-ceiling", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
