"""Choose global oracle scales for control and balanced all-state LR."""

from __future__ import annotations

import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.evaluation.crossfit_empirical_scale import (
    BINS,
    DE,
    H1,
    METRICS,
    MIN_DE,
    MODEL,
    score_grid,
    truth_and_masks,
)
from scripts.evaluation.scan_empirical_scale import exact_lfc_grid, sha256

ROOT = Path(__file__).resolve().parents[2]
STATE_MODEL = (
    ROOT
    / "data/derived/linear_response/state_diversity/state_balanced_covariance.npz"
)
CONTROL_GRID = (
    ROOT
    / "reports/linear-response-ladder/empirical-scale-crossfit/exact_native_grid_checkpoint.npz"
)
OUTPUT = ROOT / "reports/linear-response-state-diversity/oracle-scale-comparison"
REFERENCE_MAGNITUDE = 10.0


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum() / 2)])


def linear_scale(
    reference: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    eligible: np.ndarray,
) -> float:
    ratios, weights = [], []
    for target in np.flatnonzero(eligible):
        keep = mask[target] & (np.abs(reference[target]) > 1e-12)
        ratios.append(truth[target, keep] / reference[target, keep])
        weights.append(
            np.abs(reference[target, keep])
            / np.abs(truth[target, mask[target]]).sum()
        )
    return max(0.0, weighted_median(np.concatenate(ratios), np.concatenate(weights)))


def save_reference(
    path: Path,
    targets: list[str],
    completed: np.ndarray,
    predictions: np.ndarray,
    signature: str,
) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        target=np.asarray(targets),
        completed=completed,
        prediction=predictions.astype(np.float32),
        signature=np.asarray(signature),
    )
    os.replace(temporary, path)


def reference_predictions(
    data: ad.AnnData,
    targets: list[str],
    genes: list[str],
    fit_indices: np.ndarray,
    source_rows: np.ndarray,
    covariance: np.ndarray,
    signature: str,
) -> np.ndarray:
    path = OUTPUT / "reference_lfc_checkpoint.npz"
    completed = np.zeros(len(targets), dtype=bool)
    predictions = np.full((len(targets), len(genes)), np.nan, dtype=np.float32)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if (
                str(saved["signature"]) == signature
                and saved["target"].astype(str).tolist() == targets
            ):
                completed = saved["completed"].astype(bool)
                predictions = saved["prediction"].astype(np.float32)
    for target in np.flatnonzero(~completed):
        raw_full = data.X[np.sort(source_rows[target])].tocsr()
        totals = np.asarray(raw_full.sum(axis=1)).ravel()
        raw = raw_full[:, fit_indices].toarray().astype(np.float64)
        predictions[target] = exact_lfc_grid(
            raw,
            totals,
            covariance[target],
            np.asarray([-REFERENCE_MAGNITUDE]),
            len(genes),
        )[0]
        completed[target] = True
        save_reference(path, targets, completed, predictions, signature)
        print(f"reference {completed.sum()}/{len(targets)}: {targets[target]}", flush=True)
    return predictions.astype(np.float64)


def save_grid(
    path: Path,
    targets: list[str],
    magnitudes: np.ndarray,
    completed: np.ndarray,
    counts: dict[str, np.ndarray],
    results: dict[str, dict[str, np.ndarray]],
    signature: str,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "target": np.asarray(targets),
        "magnitude": magnitudes,
        "completed": completed,
        "signature": np.asarray(signature),
    }
    for name in BINS:
        arrays[f"{name}_count"] = counts[name]
        for metric in METRICS:
            arrays[f"{name}_{metric}"] = results[name][metric]
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def exact_grid(
    path: Path,
    data: ad.AnnData,
    targets: list[str],
    genes: list[str],
    fit_indices: np.ndarray,
    source_rows: np.ndarray,
    covariance: np.ndarray,
    truth: np.ndarray,
    masks: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
    magnitudes: np.ndarray,
    signature: str,
) -> dict[str, dict[str, np.ndarray]]:
    completed = np.zeros(len(targets), dtype=bool)
    results = {
        name: {
            metric: np.full((len(targets), len(magnitudes)), np.nan)
            for metric in METRICS
        }
        for name in BINS
    }
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if (
                str(saved["signature"]) == signature
                and saved["target"].astype(str).tolist() == targets
                and np.array_equal(saved["magnitude"], magnitudes)
            ):
                completed = saved["completed"].astype(bool)
                results = {
                    name: {
                        metric: saved[f"{name}_{metric}"].astype(np.float64)
                        for metric in METRICS
                    }
                    for name in BINS
                }
    for target in np.flatnonzero(~completed):
        raw_full = data.X[np.sort(source_rows[target])].tocsr()
        totals = np.asarray(raw_full.sum(axis=1)).ravel()
        raw = raw_full[:, fit_indices].toarray().astype(np.float64)
        predictions = exact_lfc_grid(
            raw, totals, covariance[target], -magnitudes, len(genes)
        )
        for name in BINS:
            scores = score_grid(predictions, truth[target], masks[name][target])
            for metric in METRICS:
                results[name][metric][target] = scores[metric]
        completed[target] = True
        save_grid(
            path,
            targets,
            magnitudes,
            completed,
            counts,
            results,
            signature,
        )
        print(f"{path.stem} {completed.sum()}/{len(targets)}: {targets[target]}", flush=True)
    return results


def curve(magnitudes: np.ndarray, nmae: np.ndarray, eligible: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "magnitude": magnitudes,
            "mean_all_de_nmae": np.nanmean(nmae[eligible], axis=0),
        }
    )


def plot_operator(table: pd.DataFrame, operator: str, selected_a: float) -> Path:
    titles = {
        "control": "Control-state",
        "all_combined": "Balanced all-state",
    }
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    labels = {
        "weak": r"Significant DE, $|\log_2\mathrm{FC}|<0.5$",
        "strong": r"Significant DE, $|\log_2\mathrm{FC}|\geq0.5$",
    }
    for name, color in [("weak", "#2f6f9f"), ("strong", "#c45a28")]:
        keep = table[f"{name}_de_genes"] >= MIN_DE
        for axis, metric in zip(axes, ["nmae", "sign_agreement"], strict=True):
            axis.plot(
                table.loc[keep, "de_count_rank"],
                table.loc[keep, f"{name}_{metric}"],
                color=color,
                linewidth=1,
                marker="o",
                markersize=2.5,
                label=labels[name],
            )
    axes[0].axhline(1, color="0.45", linestyle="--", linewidth=1)
    axes[0].set_ylabel("DE LFC NMAE")
    axes[1].axhline(0.5, color="0.45", linestyle="--", linewidth=1)
    axes[1].set_ylabel("DE direction agreement")
    axes[1].set_xlabel("Perturbations ordered by total significant DE genes")
    figure.suptitle(f"{titles[operator]} linear response at oracle $a={selected_a:.2f}$")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    figure.tight_layout()
    path = OUTPUT / f"{operator}_oracle_scale.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with np.load(MODEL, allow_pickle=False) as empirical:
        targets = empirical["target_gene"].astype(str).tolist()
        genes = empirical["output_gene"].astype(str).tolist()
        fit_indices = empirical["full_gene_index"].astype(int)
        source_rows = empirical["source_rows"].astype(int)
    with np.load(STATE_MODEL, allow_pickle=False) as state:
        assert state["target_gene"].astype(str).tolist() == targets
        index = state["model"].astype(str).tolist().index("all_combined")
        covariance = -state["response"][index].astype(np.float64)
        covariance *= state["diagonal"][index].astype(np.float64)[:, None]

    truth, masks = truth_and_masks(targets, genes)
    counts = {name: mask.sum(axis=1) for name, mask in masks.items()}
    eligible = counts["all"] >= MIN_DE
    signature = sha256(STATE_MODEL) + sha256(MODEL) + sha256(DE)

    data = ad.read_h5ad(H1, backed="r")
    try:
        reference = reference_predictions(
            data,
            targets,
            genes,
            fit_indices,
            source_rows,
            covariance,
            signature,
        )
        estimate = REFERENCE_MAGNITUDE * linear_scale(
            reference, truth, masks["all"], eligible
        )
        coarse_magnitudes = np.unique(
            np.r_[0.0, REFERENCE_MAGNITUDE, estimate * np.asarray([0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3])]
        )
        coarse = exact_grid(
            OUTPUT / "coarse_grid_checkpoint.npz",
            data,
            targets,
            genes,
            fit_indices,
            source_rows,
            covariance,
            truth,
            masks,
            counts,
            coarse_magnitudes,
            signature,
        )
        coarse_curve = curve(coarse_magnitudes, coarse["all"]["nmae"], eligible)
        selected = int(coarse_curve["mean_all_de_nmae"].argmin())
        lower = coarse_magnitudes[max(0, selected - 1)]
        upper = coarse_magnitudes[min(len(coarse_magnitudes) - 1, selected + 1)]
        if selected == len(coarse_magnitudes) - 1:
            upper *= 1.5
        refine_magnitudes = np.linspace(lower, upper, 9)
        refine = exact_grid(
            OUTPUT / "refine_grid_checkpoint.npz",
            data,
            targets,
            genes,
            fit_indices,
            source_rows,
            covariance,
            truth,
            masks,
            counts,
            refine_magnitudes,
            signature,
        )
    finally:
        data.file.close()

    with np.load(CONTROL_GRID, allow_pickle=False) as control:
        control_magnitudes = control["grid"].astype(np.float64)
        control_results = {
            name: {
                metric: control[f"{name}_{metric}"].astype(np.float64)
                for metric in METRICS
            }
            for name in BINS
        }
    all_magnitudes = np.r_[coarse_magnitudes, refine_magnitudes]
    all_results = {
        name: {
            metric: np.c_[coarse[name][metric], refine[name][metric]]
            for metric in METRICS
        }
        for name in BINS
    }
    operators = {
        "control": (control_magnitudes, control_results),
        "all_combined": (all_magnitudes, all_results),
    }

    summary_rows, target_rows, curve_rows, figures = [], [], [], []
    for operator, (magnitudes, results) in operators.items():
        mean_nmae = np.nanmean(results["all"]["nmae"][eligible], axis=0)
        selected = int(np.argmin(mean_nmae))
        selected_a = -magnitudes[selected]
        for magnitude, value in zip(magnitudes, mean_nmae, strict=True):
            curve_rows.append(
                {
                    "operator": operator,
                    "a": -magnitude,
                    "mean_all_de_nmae": value,
                }
            )
        for name in BINS:
            keep = counts[name] >= MIN_DE
            summary_rows.append(
                {
                    "operator": operator,
                    "selected_a": selected_a,
                    "de_bin": name,
                    "eligible_targets": int(keep.sum()),
                    "mean_nmae": np.nanmean(results[name]["nmae"][keep, selected]),
                    "median_nmae": np.nanmedian(results[name]["nmae"][keep, selected]),
                    "mean_direction_agreement": np.nanmean(
                        results[name]["sign_agreement"][keep, selected]
                    ),
                }
            )
        for target, target_name in enumerate(targets):
            row: dict[str, object] = {
                "operator": operator,
                "target_gene": target_name,
                "selected_a": selected_a,
                "all_de_genes": counts["all"][target],
            }
            for name in BINS:
                row[f"{name}_de_genes"] = counts[name][target]
                for metric in METRICS:
                    row[f"{name}_{metric}"] = results[name][metric][target, selected]
            target_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    per_target = pd.DataFrame(target_rows)
    for operator in operators:
        table = per_target[
            (per_target["operator"] == operator) & (per_target["all_de_genes"] >= MIN_DE)
        ].sort_values(["all_de_genes", "target_gene"], kind="stable")
        table = table.copy()
        table["de_count_rank"] = np.arange(1, len(table) + 1)
        selected_a = float(table["selected_a"].iloc[0])
        figures.append(plot_operator(table, operator, selected_a))

    paths = {
        "summary": OUTPUT / "summary.csv",
        "per_target": OUTPUT / "per_target.csv",
        "curves": OUTPUT / "scale_curves.csv",
    }
    summary.to_csv(paths["summary"], index=False)
    per_target.to_csv(paths["per_target"], index=False)
    pd.DataFrame(curve_rows).to_csv(paths["curves"], index=False)
    outputs = [*paths.values(), *figures]
    manifest = {
        "analysis": "global oracle native-scale comparison",
        "selection": "one operator-specific a minimizing mean per-target NMAE over all significant DE genes",
        "diagnostics": "weak and strong DE bins are evaluated after scale selection",
        "all_state_operator": "equal-state within plus between covariance across strict controls and 150 perturbation states",
        "transductive": True,
        "decoder": "exact expected restricted log1p-CP10K decoder on fixed balanced 400-cell sources",
        "inputs": {
            str(MODEL.relative_to(ROOT)): sha256(MODEL),
            str(STATE_MODEL.relative_to(ROOT)): sha256(STATE_MODEL),
            str(CONTROL_GRID.relative_to(ROOT)): sha256(CONTROL_GRID),
            str(DE.relative_to(ROOT)): sha256(DE),
        },
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in outputs},
        "software": {
            "python": platform.python_version(),
            **{
                package: version(package)
                for package in ["anndata", "matplotlib", "numpy", "pandas"]
            },
        },
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"linear estimate for all_combined: a={-estimate:.6g}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
