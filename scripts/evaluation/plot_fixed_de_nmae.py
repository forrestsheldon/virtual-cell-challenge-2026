"""Plot weak and strong DE NMAE for native LR at a = -10."""

from __future__ import annotations

import json
import platform
from argparse import ArgumentParser
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
AMPLITUDE = -10.0


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--operator", choices=["control", "all_combined"], default="control"
    )
    return parser.parse_args()


def load_operator(name: str):
    with np.load(MODEL, allow_pickle=False) as empirical:
        targets = empirical["target_gene"].astype(str).tolist()
        genes = empirical["output_gene"].astype(str).tolist()
        fit_indices = empirical["full_gene_index"].astype(int)
        source_rows = empirical["source_rows"].astype(int)
        if name == "control":
            covariance = empirical["covariance"].T.astype(np.float64)
            return targets, genes, fit_indices, source_rows, covariance

    with np.load(STATE_MODEL, allow_pickle=False) as state:
        assert state["target_gene"].astype(str).tolist() == targets
        assert state["output_gene"].astype(str).tolist() == genes
        assert np.array_equal(state["full_gene_index"], fit_indices)
        index = state["model"].astype(str).tolist().index(name)
        covariance = -state["response"][index].astype(np.float64)
        covariance *= state["diagonal"][index].astype(np.float64)[:, None]
    return targets, genes, fit_indices, source_rows, covariance


def main() -> None:
    operator = parse_args().operator
    output = (
        ROOT / "reports/linear-response-ladder/fixed-a-minus10-de-nmae"
        if operator == "control"
        else ROOT
        / "reports/linear-response-state-diversity/fixed-a-minus10-all-combined-de-nmae"
    )
    output.mkdir(parents=True, exist_ok=True)
    targets, genes, fit_indices, source_rows, covariance = load_operator(operator)
    truth, masks = truth_and_masks(targets, genes)
    counts = {name: mask.sum(axis=1) for name, mask in masks.items()}
    results = {
        name: {metric: np.full(len(targets), np.nan) for metric in ["nmae", "sign_agreement", "lfc_l2_ratio"]}
        for name in BINS
    }

    data = ad.read_h5ad(H1, backed="r")
    try:
        for target, rows in enumerate(source_rows):
            raw_full = data.X[np.sort(rows)].tocsr()
            totals = np.asarray(raw_full.sum(axis=1)).ravel()
            raw = raw_full[:, fit_indices].toarray().astype(np.float64)
            prediction = exact_lfc_grid(
                raw,
                totals,
                covariance[target],
                np.asarray([AMPLITUDE]),
                len(genes),
            )
            for name in BINS:
                scores = score_grid(prediction, truth[target], masks[name][target])
                for metric in results[name]:
                    results[name][metric][target] = scores[metric][0]
            print(
                f"{operator} a=-10 {target + 1}/{len(targets)}: "
                f"{targets[target]}"
            )
    finally:
        data.file.close()

    rows = []
    for target, name in enumerate(targets):
        if counts["all"][target] < MIN_DE:
            continue
        row: dict[str, object] = {
            "target_gene": name,
            "all_de_genes": counts["all"][target],
        }
        for bin_name in BINS:
            row[f"{bin_name}_de_genes"] = counts[bin_name][target]
            for metric in results[bin_name]:
                row[f"{bin_name}_{metric}"] = results[bin_name][metric][target]
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(
        ["all_de_genes", "target_gene"], kind="stable"
    )
    table.insert(0, "de_count_rank", np.arange(1, len(table) + 1))
    table.to_csv(output / "per_target.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "de_bin": name,
                "eligible_targets": int(
                    (table[f"{name}_de_genes"] >= MIN_DE).sum()
                ),
                "mean_nmae": table.loc[
                    table[f"{name}_de_genes"] >= MIN_DE, f"{name}_nmae"
                ].mean(),
                "median_nmae": table.loc[
                    table[f"{name}_de_genes"] >= MIN_DE, f"{name}_nmae"
                ].median(),
                "mean_sign_agreement": table.loc[
                    table[f"{name}_de_genes"] >= MIN_DE,
                    f"{name}_sign_agreement",
                ].mean(),
            }
            for name in BINS
        ]
    )
    summary.to_csv(output / "summary.csv", index=False)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    labels = {
        "weak": r"Significant DE, $|\log_2\mathrm{FC}|<0.5$",
        "strong": r"Significant DE, $|\log_2\mathrm{FC}|\geq0.5$",
    }
    for name, color in [("weak", "#2f6f9f"), ("strong", "#c45a28")]:
        keep = table[f"{name}_de_genes"] >= MIN_DE
        axes[0].plot(
            table.loc[keep, "de_count_rank"],
            table.loc[keep, f"{name}_nmae"],
            color=color,
            linewidth=1,
            marker="o",
            markersize=2.5,
            label=labels[name],
        )
        axes[1].plot(
            table.loc[keep, "de_count_rank"],
            table.loc[keep, f"{name}_sign_agreement"],
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
    title = {
        "control": r"Control-state linear response at $a=-10$",
        "all_combined": r"Balanced all-state linear response at $a=-10$",
    }
    figure.suptitle(title[operator])
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure_path = output / "weak_strong_nmae_by_de_count.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    outputs = [output / "per_target.csv", output / "summary.csv", figure_path]
    model_inputs = {str(MODEL.relative_to(ROOT)): sha256(MODEL)}
    if operator == "all_combined":
        model_inputs[str(STATE_MODEL.relative_to(ROOT))] = sha256(STATE_MODEL)
    manifest = {
        "analysis": f"weak and strong DE NMAE for native {operator} LR at fixed a=-10",
        "operator": operator,
        "ordering": "ascending total significant DE gene count, then target name",
        "bins": {
            "weak": "reference-significant and |LFC| < 0.5",
            "strong": "reference-significant and |LFC| >= 0.5",
            "minimum_genes_for_curve": MIN_DE,
        },
        "inputs": {
            **model_inputs,
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
                for package in ["anndata", "matplotlib", "numpy", "pandas"]
            },
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
