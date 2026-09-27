"""Line-search the empirical linear-response scale on stable strong-DE genes."""

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

from scripts.linear_response.evaluate_ladder import load_truth
from scripts.linear_response.kernel import CELL_TARGET_SUM
from scripts.linear_response.strong_de_recovery import signed_topk

ROOT = Path(__file__).resolve().parents[2]
H1 = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
EMPIRICAL = ROOT / "data/derived/linear_response/ladder/empirical.npz"
DE = ROOT / "data/derived/linear_response/eval_cache/de_wilcoxon_table-12f179b9e966af64.parquet"
STRONG = ROOT / "reports/linear-response-three-models/strong_de_truth_genes.csv"
OUTPUT = ROOT / "reports/linear-response-ladder/empirical-scale-scan"

GRIDS = {
    "target_normalized_gamma": np.r_[0.0, np.geomspace(0.01, 12.0, 64)],
    "native_cipher_a": np.r_[0.0, np.geomspace(0.1, 500.0, 72)],
}
METRICS = (
    "strong_de_nmae",
    "signed_recovery",
    "unsigned_recovery",
    "conditional_sign_accuracy",
    "strong_lfc_l2_ratio",
    "strong_energy_fraction",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_lfc_grid(
    raw: np.ndarray,
    totals: np.ndarray,
    direction: np.ndarray,
    amplitudes: np.ndarray,
    output_count: int,
) -> np.ndarray:
    """Decode a scale grid exactly while reusing the expensive cell representation."""
    weights = raw * (CELL_TARGET_SUM / totals)[:, None]
    log_weights = np.log1p(weights)
    outside = CELL_TARGET_SUM - weights.sum(axis=1)
    baseline = 1_000_000 * np.mean(weights / CELL_TARGET_SUM, axis=0)
    result = np.empty((len(amplitudes), output_count))
    for index, amplitude in enumerate(amplitudes):
        if amplitude == 0:
            result[index] = 0
            continue
        shifted = np.clip(
            np.expm1(log_weights + amplitude * direction), 0.0, None
        )
        denominator = outside + shifted.sum(axis=1)
        cpm = 1_000_000 * np.mean(shifted / denominator[:, None], axis=0)
        result[index] = np.log2(
            (cpm[:output_count] + 1e-9) / (baseline[:output_count] + 1e-9)
        )
    return result


def profile_metrics(
    predictions: np.ndarray,
    truth: np.ndarray,
    stable: np.ndarray,
    truth_sign: np.ndarray,
    target_gene_index: int | None,
) -> dict[str, np.ndarray]:
    denominator = np.abs(truth[stable]).sum()
    truth_l2 = np.linalg.norm(truth[stable])
    values = {name: np.empty(len(predictions)) for name in METRICS}
    for index, prediction in enumerate(predictions):
        if not np.any(prediction):
            values["strong_de_nmae"][index] = 1
            values["signed_recovery"][index] = 0
            values["unsigned_recovery"][index] = np.nan
            values["conditional_sign_accuracy"][index] = np.nan
            values["strong_lfc_l2_ratio"][index] = 0
            values["strong_energy_fraction"][index] = np.nan
            continue
        score = signed_topk(prediction, stable, truth_sign, target_gene_index)
        values["strong_de_nmae"][index] = (
            np.abs(prediction[stable] - truth[stable]).sum() / denominator
        )
        values["signed_recovery"][index] = score["signed_recovery"]
        values["unsigned_recovery"][index] = score["unsigned_recall"]
        values["conditional_sign_accuracy"][index] = score[
            "sign_given_recovered"
        ]
        values["strong_lfc_l2_ratio"][index] = (
            np.linalg.norm(prediction[stable]) / truth_l2
        )
        values["strong_energy_fraction"][index] = (
            1 - score["non_strong_energy_fraction"]
            if np.isfinite(score["non_strong_energy_fraction"])
            else np.nan
        )
    return values


def leave_one_out_choices(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Choose a common scale on every target except the one being evaluated."""
    choices = np.empty(len(errors), dtype=int)
    heldout = np.empty(len(errors))
    for target in range(len(errors)):
        training = np.delete(errors, target, axis=0)
        choices[target] = int(np.nanargmin(np.nanmean(training, axis=0)))
        heldout[target] = errors[target, choices[target]]
    return choices, heldout


def mean_or_nan(values: np.ndarray) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else np.nan


def checkpoint_path() -> Path:
    return OUTPUT / "exact_grid_checkpoint.npz"


def save_checkpoint(
    grids: dict[str, np.ndarray],
    targets: list[str],
    eligible: np.ndarray,
    completed: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
    signature: str,
) -> None:
    path = checkpoint_path()
    temporary = path.with_suffix(".tmp.npz")
    arrays = {
        "target": np.asarray(targets),
        "eligible": eligible,
        "completed": completed,
        "signature": np.asarray(signature),
    }
    for convention, grid in grids.items():
        arrays[f"{convention}_grid"] = grid
        for metric, values in results[convention].items():
            arrays[f"{convention}_{metric}"] = values
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_or_create_checkpoint(
    grids: dict[str, np.ndarray],
    targets: list[str],
    eligible: np.ndarray,
    signature: str,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    path = checkpoint_path()
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            valid = (
                str(saved["signature"]) == signature
                and saved["target"].astype(str).tolist() == targets
                and np.array_equal(saved["eligible"], eligible)
                and all(
                    np.array_equal(saved[f"{name}_grid"], grid)
                    for name, grid in grids.items()
                )
            )
            if valid:
                completed = saved["completed"].astype(bool)
                results = {
                    name: {
                        metric: saved[f"{name}_{metric}"].astype(np.float64)
                        for metric in METRICS
                    }
                    for name in grids
                }
                return completed, results
    completed = np.zeros(len(targets), dtype=bool)
    results = {
        name: {
            metric: np.full((len(targets), len(grid)), np.nan)
            for metric in METRICS
        }
        for name, grid in grids.items()
    }
    return completed, results


def summarize(
    grids: dict[str, np.ndarray],
    targets: list[str],
    eligible: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curve_rows = []
    selection_rows = []
    summary_rows = []
    eligible_targets = np.flatnonzero(eligible)
    for convention, magnitudes in grids.items():
        errors = results[convention]["strong_de_nmae"][eligible]
        global_index = int(np.nanargmin(np.nanmean(errors, axis=0)))
        nonzero_index = 1 + int(np.nanargmin(np.nanmean(errors[:, 1:], axis=0)))
        median_index = int(np.nanargmin(np.nanmedian(errors, axis=0)))
        fidelity = np.nanmean(
            results[convention]["signed_recovery"][eligible], axis=0
        )
        fidelity_index = int(np.nanargmax(fidelity))
        loo_indices, loo_errors = leave_one_out_choices(errors)
        oracle_indices = np.nanargmin(errors, axis=1)
        for grid_index, magnitude in enumerate(magnitudes):
            selected_errors = errors[:, grid_index]
            curve_rows.append(
                {
                    "convention": convention,
                    "scale": -magnitude
                    if convention == "native_cipher_a"
                    else magnitude,
                    "scale_magnitude": magnitude,
                    "mean_strong_de_nmae": np.nanmean(selected_errors),
                    "median_strong_de_nmae": np.nanmedian(selected_errors),
                    "mean_signed_recovery": np.nanmean(
                        results[convention]["signed_recovery"][eligible, grid_index]
                    ),
                    "mean_unsigned_recovery": mean_or_nan(
                        results[convention]["unsigned_recovery"][eligible, grid_index]
                    ),
                    "mean_conditional_sign_accuracy": mean_or_nan(
                        results[convention]["conditional_sign_accuracy"][
                            eligible, grid_index
                        ]
                    ),
                    "median_strong_lfc_l2_ratio": np.nanmedian(
                        results[convention]["strong_lfc_l2_ratio"][
                            eligible, grid_index
                        ]
                    ),
                    "mean_strong_energy_fraction": mean_or_nan(
                        results[convention]["strong_energy_fraction"][
                            eligible, grid_index
                        ]
                    ),
                    "targets_improved_over_zero": int(
                        np.sum(selected_errors < errors[:, 0])
                    ),
                }
            )
        for local_index, target_index in enumerate(eligible_targets):
            oracle_index = int(oracle_indices[local_index])
            loo_index = int(loo_indices[local_index])
            selection_rows.append(
                {
                    "convention": convention,
                    "target_gene": targets[target_index],
                    "oracle_scale": (-1 if convention == "native_cipher_a" else 1)
                    * magnitudes[oracle_index],
                    "oracle_strong_de_nmae": errors[local_index, oracle_index],
                    "loo_scale": (-1 if convention == "native_cipher_a" else 1)
                    * magnitudes[loo_index],
                    "loo_strong_de_nmae": loo_errors[local_index],
                    "global_scale": (-1 if convention == "native_cipher_a" else 1)
                    * magnitudes[global_index],
                    "global_strong_de_nmae": errors[local_index, global_index],
                    "zero_strong_de_nmae": errors[local_index, 0],
                }
            )
        summary_rows.append(
            {
                "convention": convention,
                "eligible_targets": int(eligible.sum()),
                "global_scale": (-1 if convention == "native_cipher_a" else 1)
                * magnitudes[global_index],
                "global_mean_strong_de_nmae": np.nanmean(errors[:, global_index]),
                "loo_mean_strong_de_nmae": np.nanmean(loo_errors),
                "oracle_mean_strong_de_nmae": np.nanmean(
                    errors[np.arange(len(errors)), oracle_indices]
                ),
                "oracle_zero_fraction": np.mean(oracle_indices == 0),
                "oracle_upper_boundary_fraction": np.mean(
                    oracle_indices == len(magnitudes) - 1
                ),
                "median_oracle_scale": (-1 if convention == "native_cipher_a" else 1)
                * np.median(magnitudes[oracle_indices]),
                "global_mean_signed_recovery": np.nanmean(
                    results[convention]["signed_recovery"][eligible, global_index]
                ),
                "global_median_strong_lfc_l2_ratio": np.nanmedian(
                    results[convention]["strong_lfc_l2_ratio"][
                        eligible, global_index
                    ]
                ),
                "targets_improved_global_over_zero": int(
                    np.sum(errors[:, global_index] < errors[:, 0])
                ),
                "targets_improved_loo_over_zero": int(
                    np.sum(loo_errors < errors[:, 0])
                ),
                "best_nonzero_scale": (
                    -1 if convention == "native_cipher_a" else 1
                )
                * magnitudes[nonzero_index],
                "best_nonzero_mean_strong_de_nmae": np.nanmean(
                    errors[:, nonzero_index]
                ),
                "targets_improved_best_nonzero_over_zero": int(
                    np.sum(errors[:, nonzero_index] < errors[:, 0])
                ),
                "best_fidelity_scale": (
                    -1 if convention == "native_cipher_a" else 1
                )
                * magnitudes[fidelity_index],
                "best_mean_signed_recovery": fidelity[fidelity_index],
                "mean_nmae_at_best_fidelity": np.nanmean(
                    errors[:, fidelity_index]
                ),
                "median_l2_ratio_at_best_fidelity": np.nanmedian(
                    results[convention]["strong_lfc_l2_ratio"][
                        eligible, fidelity_index
                    ]
                ),
                "best_median_nmae_scale": (
                    -1 if convention == "native_cipher_a" else 1
                )
                * magnitudes[median_index],
                "best_median_strong_de_nmae": np.nanmedian(
                    errors[:, median_index]
                ),
                "mean_nmae_at_best_median": np.nanmean(
                    errors[:, median_index]
                ),
            }
        )
    return (
        pd.DataFrame(curve_rows),
        pd.DataFrame(selection_rows),
        pd.DataFrame(summary_rows),
    )


def plot(curves: pd.DataFrame, selections: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    labels = {
        "target_normalized_gamma": r"target-normalized $\gamma$",
        "native_cipher_a": r"native $|a|$",
    }
    for column, convention in enumerate(labels):
        table = curves[curves["convention"] == convention]
        nonzero = table[table["scale_magnitude"] > 0]
        best = table.loc[table["mean_strong_de_nmae"].idxmin()]
        axes[0, column].plot(
            nonzero["scale_magnitude"], nonzero["mean_strong_de_nmae"]
        )
        axes[0, column].axhline(
            table.iloc[0]["mean_strong_de_nmae"], color="0.6", linestyle="--"
        )
        axes[0, column].axvline(
            best["scale_magnitude"], color="tab:red", linestyle=":"
        )
        axes[0, column].set_xscale("log")
        axes[0, column].set_xlabel(labels[convention])
        axes[0, column].set_ylabel("mean strong-DE LFC NMAE")
        axes[1, column].plot(
            nonzero["scale_magnitude"], nonzero["mean_signed_recovery"]
        )
        axes[1, column].set_xscale("log")
        axes[1, column].set_xlabel(labels[convention])
        axes[1, column].set_ylabel("mean signed top-K recovery")
    path = OUTPUT / "empirical_scale_scan.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)

    comparison = selections.pivot(
        index="target_gene", columns="convention", values="oracle_strong_de_nmae"
    )
    comparison.to_csv(OUTPUT / "oracle_convention_comparison.csv")
    return path


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with np.load(EMPIRICAL, allow_pickle=False) as model:
        targets = model["target_gene"].astype(str).tolist()
        genes = model["output_gene"].astype(str).tolist()
        fit_genes = model["fit_gene"].astype(str)
        full_indices = model["full_gene_index"].astype(int)
        source_rows = model["source_rows"].astype(int)
        target_indices = model["target_fit_index"].astype(int)
        response = model["response"].astype(np.float64)
        covariance = model["covariance"].T.astype(np.float64)
    if fit_genes[: len(genes)].tolist() != genes:
        raise AssertionError("output genes are not the leading fit genes")
    diagonal = covariance[np.arange(len(targets)), target_indices]
    if not np.allclose(response, -covariance / diagonal[:, None], atol=2e-6):
        raise AssertionError("stored empirical response is not -C[:,t]/C[t,t]")

    truth, stable = load_truth(targets, genes)
    eligible = stable.sum(axis=1) >= 10
    target_gene_indices = [genes.index(target) if target in genes else None for target in targets]
    signature = sha256(EMPIRICAL) + sha256(DE) + sha256(STRONG)
    completed, results = load_or_create_checkpoint(
        GRIDS, targets, eligible, signature
    )

    data = ad.read_h5ad(H1, backed="r")
    try:
        for target in np.flatnonzero(eligible & ~completed):
            rows = np.sort(source_rows[target])
            raw_full = data.X[rows].tocsr()
            totals = np.asarray(raw_full.sum(axis=1)).ravel()
            raw = raw_full[:, full_indices].toarray().astype(np.float64)
            profiles = {
                "target_normalized_gamma": exact_lfc_grid(
                    raw,
                    totals,
                    response[target],
                    GRIDS["target_normalized_gamma"],
                    len(genes),
                ),
                "native_cipher_a": exact_lfc_grid(
                    raw,
                    totals,
                    covariance[target],
                    -GRIDS["native_cipher_a"],
                    len(genes),
                ),
            }
            for convention, prediction in profiles.items():
                scores = profile_metrics(
                    prediction,
                    truth[target],
                    stable[target],
                    np.sign(truth[target]).astype(np.int8),
                    target_gene_indices[target],
                )
                for metric, values in scores.items():
                    results[convention][metric][target] = values
            completed[target] = True
            save_checkpoint(
                GRIDS, targets, eligible, completed, results, signature
            )
            print(
                f"Exact scale scan {completed[eligible].sum()}/{eligible.sum()}: "
                f"{targets[target]}",
                flush=True,
            )
    finally:
        data.file.close()

    for convention in GRIDS:
        results[convention]["strong_de_nmae"][eligible, 0] = 1
        results[convention]["signed_recovery"][eligible, 0] = 0
        results[convention]["unsigned_recovery"][eligible, 0] = np.nan
        results[convention]["conditional_sign_accuracy"][eligible, 0] = np.nan
        results[convention]["strong_lfc_l2_ratio"][eligible, 0] = 0
        results[convention]["strong_energy_fraction"][eligible, 0] = np.nan
    save_checkpoint(GRIDS, targets, eligible, completed, results, signature)

    curves, selections, summary = summarize(
        GRIDS, targets, eligible, results
    )
    curves.to_csv(OUTPUT / "scale_scan.csv", index=False)
    selections.to_csv(OUTPUT / "scale_selection_per_target.csv", index=False)
    summary.to_csv(OUTPUT / "scale_summary.csv", index=False)
    figure = plot(curves, selections)
    outputs = [
        OUTPUT / "scale_scan.csv",
        OUTPUT / "scale_selection_per_target.csv",
        OUTPUT / "scale_summary.csv",
        OUTPUT / "oracle_convention_comparison.csv",
        figure,
        checkpoint_path(),
    ]
    manifest = {
        "analysis": "exact expected-profile line search for empirical LR scale",
        "truth_use": "retrospective H1 diagnostic; not a zero-shot predictor",
        "decoder": "exact expected restricted log1p-CP10K decoder on fixed balanced 400-cell sources",
        "selection": {
            "global": "single grid point minimizing mean target NMAE",
            "leave_one_target_out": "grid point minimizing mean NMAE on all other eligible targets",
            "oracle": "separate best grid point for each target",
        },
        "strong_de": "CPM > 5, padj < 0.05, |LFC| >= 0.5, stable sign in >=4/5 splits, target gene excluded, >=10 genes",
        "grids": {
            name: {"points": len(grid), "minimum": float(grid.min()), "maximum": float(grid.max())}
            for name, grid in GRIDS.items()
        },
        "inputs": {
            str(EMPIRICAL.relative_to(ROOT)): sha256(EMPIRICAL),
            str(DE.relative_to(ROOT)): sha256(DE),
            str(STRONG.relative_to(ROOT)): sha256(STRONG),
            str(H1.relative_to(ROOT)): "a09977104fefb622368ca74b50c9d3c1e891733e6c83db07acfca49b0219c02b",
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path) for path in outputs
        },
        "software": {
            "python": platform.python_version(),
            **{
                package: version(package)
                for package in ["anndata", "matplotlib", "numpy", "pandas"]
            },
        },
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
