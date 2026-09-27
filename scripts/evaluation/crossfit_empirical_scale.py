"""Cross-fit one native empirical-LR scale using full-DE NMAE."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from scripts.evaluation.scan_empirical_scale import exact_lfc_grid, sha256

ROOT = Path(__file__).resolve().parents[2]
H1 = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
MODEL = ROOT / "data/derived/linear_response/ladder/empirical.npz"
DE = ROOT / "data/derived/linear_response/eval_cache/de_wilcoxon_table-12f179b9e966af64.parquet"
OUTPUT = ROOT / "reports/linear-response-ladder/empirical-scale-crossfit"
GRID = np.unique(np.r_[0.0, np.geomspace(0.01, 500.0, 80), 9.72, 19.6])
FOLD_SEED = 20260916
MIN_DE = 10
BINS = ("all", "weak", "strong")
METRICS = ("nmae", "sign_agreement", "lfc_l2_ratio")


def target_folds(targets: list[str], seed: int = FOLD_SEED) -> np.ndarray:
    """Make an exact balanced target split without consulting expression or truth."""
    order = np.random.default_rng(seed).permutation(len(targets))
    folds = np.empty(len(targets), dtype=np.int8)
    folds[order[::2]] = 0
    folds[order[1::2]] = 1
    return folds


def truth_and_masks(
    targets: list[str], genes: list[str]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frame = pl.read_parquet(DE).to_pandas()
    truth = np.empty((len(targets), len(genes)))
    significant = np.zeros_like(truth, dtype=bool)
    lookup = {gene: index for index, gene in enumerate(genes)}
    for index, target in enumerate(targets):
        group = frame.loc[frame["target"] == target].set_index("feature").loc[genes]
        truth[index] = group["log2_fold_change"].to_numpy()
        significant[index] = group["p_adj"].to_numpy() < 0.05
        if target in lookup:
            significant[index, lookup[target]] = False
    absolute = np.abs(truth)
    return truth, {
        "all": significant,
        "weak": significant & (absolute < 0.5),
        "strong": significant & (absolute >= 0.5),
    }


def score_grid(
    predictions: np.ndarray, truth: np.ndarray, mask: np.ndarray
) -> dict[str, np.ndarray]:
    selected = predictions[:, mask]
    observed = truth[mask]
    if not len(observed):
        return {metric: np.full(len(predictions), np.nan) for metric in METRICS}
    return {
        "nmae": np.abs(selected - observed).sum(axis=1)
        / np.abs(observed).sum(),
        "sign_agreement": np.mean(
            np.sign(selected) == np.sign(observed), axis=1
        ),
        "lfc_l2_ratio": np.linalg.norm(selected, axis=1)
        / np.linalg.norm(observed),
    }


def choose_scale(errors: np.ndarray, training: np.ndarray) -> int:
    return int(np.nanargmin(np.nanmean(errors[training], axis=0)))


def mean_or_nan(values: np.ndarray) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else np.nan


def checkpoint_path() -> Path:
    return OUTPUT / "exact_native_grid_checkpoint.npz"


def save_checkpoint(
    targets: list[str],
    completed: np.ndarray,
    counts: dict[str, np.ndarray],
    results: dict[str, dict[str, np.ndarray]],
    signature: str,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "target": np.asarray(targets),
        "grid": GRID,
        "completed": completed,
        "signature": np.asarray(signature),
    }
    for name in BINS:
        arrays[f"{name}_count"] = counts[name]
        for metric in METRICS:
            arrays[f"{name}_{metric}"] = results[name][metric]
    temporary = checkpoint_path().with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, checkpoint_path())


def load_checkpoint(
    targets: list[str], counts: dict[str, np.ndarray], signature: str
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    path = checkpoint_path()
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            valid = (
                str(saved["signature"]) == signature
                and saved["target"].astype(str).tolist() == targets
                and np.array_equal(saved["grid"], GRID)
                and all(
                    np.array_equal(saved[f"{name}_count"], counts[name])
                    for name in BINS
                )
            )
            if valid:
                return saved["completed"].astype(bool), {
                    name: {
                        metric: saved[f"{name}_{metric}"].astype(np.float64)
                        for metric in METRICS
                    }
                    for name in BINS
                }
    return np.zeros(len(targets), dtype=bool), {
        name: {
            metric: np.full((len(targets), len(GRID)), np.nan)
            for metric in METRICS
        }
        for name in BINS
    }


def summarize(
    targets: list[str],
    folds: np.ndarray,
    counts: dict[str, np.ndarray],
    results: dict[str, dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = counts["all"] >= MIN_DE
    selected_by_fold: dict[int, int] = {}
    fold_rows = []
    curve_rows = []
    for heldout_fold in (0, 1):
        training = eligible & (folds != heldout_fold)
        heldout = eligible & (folds == heldout_fold)
        selected = choose_scale(results["all"]["nmae"], training)
        selected_by_fold[heldout_fold] = selected
        for index, magnitude in enumerate(GRID):
            curve_rows.extend(
                [
                    {
                        "heldout_fold": heldout_fold,
                        "role": role,
                        "a": -magnitude,
                        "mean_full_de_nmae": np.nanmean(
                            results["all"]["nmae"][mask, index]
                        ),
                    }
                    for role, mask in [("training", training), ("heldout", heldout)]
                ]
            )
        fold_rows.append(
            {
                "heldout_fold": heldout_fold,
                "training_targets": int(training.sum()),
                "heldout_targets": int(heldout.sum()),
                "selected_a": -GRID[selected],
                "training_full_de_nmae": np.nanmean(
                    results["all"]["nmae"][training, selected]
                ),
                "heldout_full_de_nmae": np.nanmean(
                    results["all"]["nmae"][heldout, selected]
                ),
                "heldout_full_de_sign_agreement": mean_or_nan(
                    results["all"]["sign_agreement"][heldout, selected]
                ),
            }
        )

    per_target_rows = []
    for target, name in enumerate(targets):
        selected = selected_by_fold[int(folds[target])]
        row: dict[str, object] = {
            "target_gene": name,
            "fold": int(folds[target]),
            "selected_a": -GRID[selected],
        }
        for bin_name in BINS:
            row[f"{bin_name}_de_genes"] = int(counts[bin_name][target])
            for metric in METRICS:
                row[f"{bin_name}_{metric}"] = results[bin_name][metric][
                    target, selected
                ]
        per_target_rows.append(row)
    per_target = pd.DataFrame(per_target_rows)

    summary_rows = []
    fold_bin_rows = []
    for bin_name in BINS:
        keep = per_target[f"{bin_name}_de_genes"] >= MIN_DE
        nmae = per_target.loc[keep, f"{bin_name}_nmae"]
        summary_rows.append(
            {
                "de_bin": bin_name,
                "eligible_targets": int(keep.sum()),
                "mean_crossfit_nmae": nmae.mean(),
                "median_crossfit_nmae": nmae.median(),
                "targets_improved_over_zero": int((nmae < 1 - 1e-12).sum()),
                "mean_sign_agreement": per_target.loc[
                    keep, f"{bin_name}_sign_agreement"
                ].mean(),
                "median_lfc_l2_ratio": per_target.loc[
                    keep, f"{bin_name}_lfc_l2_ratio"
                ].median(),
            }
        )
        for fold in (0, 1):
            fold_keep = keep & per_target["fold"].eq(fold)
            fold_nmae = per_target.loc[fold_keep, f"{bin_name}_nmae"]
            fold_bin_rows.append(
                {
                    "heldout_fold": fold,
                    "selected_a": per_target.loc[
                        per_target["fold"].eq(fold), "selected_a"
                    ].iloc[0],
                    "de_bin": bin_name,
                    "eligible_targets": int(fold_keep.sum()),
                    "mean_crossfit_nmae": fold_nmae.mean(),
                    "median_crossfit_nmae": fold_nmae.median(),
                    "targets_improved_over_zero": int(
                        (fold_nmae < 1 - 1e-12).sum()
                    ),
                    "mean_sign_agreement": per_target.loc[
                        fold_keep, f"{bin_name}_sign_agreement"
                    ].mean(),
                    "median_lfc_l2_ratio": per_target.loc[
                        fold_keep, f"{bin_name}_lfc_l2_ratio"
                    ].median(),
                }
            )
    all_selected = choose_scale(results["all"]["nmae"], eligible)
    refit = pd.DataFrame(
        [
            {
                "calibration_targets": int(eligible.sum()),
                "selected_a": -GRID[all_selected],
                "in_sample_full_de_nmae": np.nanmean(
                    results["all"]["nmae"][eligible, all_selected]
                ),
            }
        ]
    )
    return (
        pd.DataFrame(curve_rows),
        pd.DataFrame(fold_rows),
        per_target,
        pd.DataFrame(summary_rows),
        pd.DataFrame(fold_bin_rows),
        refit,
    )


def plot(
    curves: pd.DataFrame, folds: pd.DataFrame, summary: pd.DataFrame
) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for heldout_fold in (0, 1):
        for role, style in [("training", "-"), ("heldout", "--")]:
            table = curves[
                (curves["heldout_fold"] == heldout_fold)
                & (curves["role"] == role)
                & (curves["a"] < 0)
            ]
            axes[0].plot(
                -table["a"],
                table["mean_full_de_nmae"],
                style,
                label=f"fold {heldout_fold} {role}",
            )
        selected = -folds.loc[
            folds["heldout_fold"] == heldout_fold, "selected_a"
        ].iloc[0]
        if selected > 0:
            axes[0].axvline(selected, color=f"C{heldout_fold}", alpha=0.4)
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.97, 1.25)
    axes[0].set_xlabel(r"native $|a|$")
    axes[0].set_ylabel("mean full-DE NMAE")
    axes[0].legend(fontsize=8)

    axes[1].bar(summary["de_bin"], summary["mean_crossfit_nmae"])
    axes[1].axhline(1, color="0.5", linestyle="--")
    axes[1].set_ylabel("held-out mean NMAE")
    axes[2].bar(summary["de_bin"], summary["mean_sign_agreement"])
    axes[2].axhline(0.5, color="0.5", linestyle="--")
    axes[2].set_ylabel("held-out sign agreement")
    path = OUTPUT / "crossfit_full_de_scale.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with np.load(MODEL, allow_pickle=False) as model:
        targets = model["target_gene"].astype(str).tolist()
        genes = model["output_gene"].astype(str).tolist()
        fit_genes = model["fit_gene"].astype(str).tolist()
        fit_indices = model["full_gene_index"].astype(int)
        source_rows = model["source_rows"].astype(int)
        covariance = model["covariance"].T.astype(np.float64)
    if fit_genes[: len(genes)] != genes:
        raise AssertionError("output genes are not the leading fit genes")
    truth, masks = truth_and_masks(targets, genes)
    counts = {name: mask.sum(axis=1) for name, mask in masks.items()}
    folds = target_folds(targets)
    signature = (
        sha256(MODEL) + sha256(DE) + hashlib.sha256(GRID.tobytes()).hexdigest()
    )
    completed, results = load_checkpoint(targets, counts, signature)

    data = ad.read_h5ad(H1, backed="r")
    try:
        for target in np.flatnonzero(~completed):
            rows = np.sort(source_rows[target])
            raw_full = data.X[rows].tocsr()
            totals = np.asarray(raw_full.sum(axis=1)).ravel()
            raw = raw_full[:, fit_indices].toarray().astype(np.float64)
            predictions = exact_lfc_grid(
                raw,
                totals,
                covariance[target],
                -GRID,
                len(genes),
            )
            for bin_name in BINS:
                scores = score_grid(
                    predictions, truth[target], masks[bin_name][target]
                )
                for metric in METRICS:
                    results[bin_name][metric][target] = scores[metric]
            completed[target] = True
            save_checkpoint(targets, completed, counts, results, signature)
            print(
                f"Full-DE native scale {completed.sum()}/{len(targets)}: "
                f"{targets[target]}",
                flush=True,
            )
    finally:
        data.file.close()

    for bin_name in BINS:
        results[bin_name]["sign_agreement"][:, 0] = np.nan
    save_checkpoint(targets, completed, counts, results, signature)

    curves, fold_table, per_target, summary, fold_bins, refit = summarize(
        targets, folds, counts, results
    )
    paths = {
        "curves": OUTPUT / "fold_curves.csv",
        "folds": OUTPUT / "fold_selection.csv",
        "per_target": OUTPUT / "per_target.csv",
        "summary": OUTPUT / "summary.csv",
        "fold_bins": OUTPUT / "fold_bin_summary.csv",
        "refit": OUTPUT / "all_target_refit.csv",
    }
    curves.to_csv(paths["curves"], index=False)
    fold_table.to_csv(paths["folds"], index=False)
    per_target.to_csv(paths["per_target"], index=False)
    summary.to_csv(paths["summary"], index=False)
    fold_bins.to_csv(paths["fold_bins"], index=False)
    refit.to_csv(paths["refit"], index=False)
    figure = plot(curves, fold_table, summary)
    outputs = [*paths.values(), figure, checkpoint_path()]
    manifest = {
        "analysis": "two-fold target-cross-fitted native empirical-LR scale",
        "calibration": "mean per-target NMAE over all reference-significant DE genes",
        "heldout_diagnostics": "all, |LFC| < 0.5 weak, and |LFC| >= 0.5 strong reference-significant genes",
        "gene_panel": "10,780 control CPM > 5 genes; perturbed target excluded",
        "eligibility": f"at least {MIN_DE} genes in the reported DE bin",
        "target_split": {
            "seed": FOLD_SEED,
            "independent_of_expression_and_truth": True,
            "fold_sizes": np.bincount(folds).tolist(),
        },
        "grid": {
            "points": len(GRID),
            "minimum_magnitude": float(GRID.min()),
            "maximum_magnitude": float(GRID.max()),
        },
        "decoder": "exact expected restricted log1p-CP10K decoder on fixed balanced 400-cell sources",
        "inputs": {
            str(MODEL.relative_to(ROOT)): sha256(MODEL),
            str(DE.relative_to(ROOT)): sha256(DE),
            str(H1.relative_to(ROOT)): "a09977104fefb622368ca74b50c9d3c1e891733e6c83db07acfca49b0219c02b",
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path) for path in outputs
        },
        "software": {
            "python": platform.python_version(),
            **{
                package: version(package)
                for package in [
                    "anndata",
                    "matplotlib",
                    "numpy",
                    "pandas",
                    "polars",
                ]
            },
        },
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(fold_table.to_string(index=False))
    print(summary.to_string(index=False))
    print(fold_bins.to_string(index=False))
    print(refit.to_string(index=False))


if __name__ == "__main__":
    main()
