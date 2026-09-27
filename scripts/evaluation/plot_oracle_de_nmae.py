"""Plot weak and strong DE NMAE at each target's full-DE oracle scale."""

from __future__ import annotations

import json
import platform
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.evaluation.scan_empirical_scale import sha256

ROOT = Path(__file__).resolve().parents[2]
INPUT = (
    ROOT
    / "reports/linear-response-ladder/empirical-scale-crossfit/exact_native_grid_checkpoint.npz"
)
OUTPUT = ROOT / "reports/linear-response-ladder/oracle-de-nmae"
MIN_DE = 10
BINS = ("all", "weak", "strong")
FIXED_A = -9.72


def full_de_oracle_indices(errors: np.ndarray, counts: np.ndarray) -> np.ndarray:
    indices = np.full(len(errors), -1, dtype=int)
    eligible = counts >= MIN_DE
    indices[eligible] = np.nanargmin(errors[eligible], axis=1)
    return indices


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with np.load(INPUT, allow_pickle=False) as saved:
        targets = saved["target"].astype(str)
        grid = saved["grid"].astype(np.float64)
        counts = {name: saved[f"{name}_count"].astype(int) for name in BINS}
        metrics = {
            name: {
                metric: saved[f"{name}_{metric}"].astype(np.float64)
                for metric in ["nmae", "sign_agreement", "lfc_l2_ratio"]
            }
            for name in BINS
        }

    oracle = full_de_oracle_indices(metrics["all"]["nmae"], counts["all"])
    fixed_index = int(np.flatnonzero(np.isclose(grid, abs(FIXED_A)))[0])
    rows = []
    for target in np.flatnonzero(oracle >= 0):
        selected = oracle[target]
        row: dict[str, object] = {
            "target_gene": targets[target],
            "all_de_genes": counts["all"][target],
            "oracle_a": -grid[selected],
            "oracle_at_zero": selected == 0,
        }
        for name in BINS:
            row[f"{name}_de_genes"] = counts[name][target]
            for metric in ["nmae", "sign_agreement", "lfc_l2_ratio"]:
                row[f"{name}_{metric}"] = metrics[name][metric][target, selected]
                row[f"fixed_{name}_{metric}"] = metrics[name][metric][
                    target, fixed_index
                ]
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(
        ["all_de_genes", "target_gene"], kind="stable"
    )
    table.insert(0, "de_count_rank", np.arange(1, len(table) + 1))
    table.to_csv(OUTPUT / "per_target.csv", index=False)

    summary_rows = []
    for method, prefix in [("oracle_with_zero", ""), ("fixed_a_minus9p72", "fixed_")]:
        for name in BINS:
            keep = table[f"{name}_de_genes"] >= MIN_DE
            direction_keep = (
                keep & ~table["oracle_at_zero"]
                if method == "oracle_with_zero"
                else keep
            )
            values = table.loc[keep, f"{prefix}{name}_nmae"]
            summary_rows.append(
                {
                    "method": method,
                    "de_bin": name,
                    "eligible_targets": int(keep.sum()),
                    "mean_nmae": values.mean(),
                    "median_nmae": values.median(),
                    "targets_improved_over_zero": int(
                        (values < 1 - 1e-12).sum()
                    ),
                    "mean_sign_agreement": table.loc[
                        direction_keep, f"{prefix}{name}_sign_agreement"
                    ].mean(),
                    "median_lfc_l2_ratio": table.loc[
                        keep, f"{prefix}{name}_lfc_l2_ratio"
                    ].median(),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "summary.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    colors = {"weak": "#2f6f9f", "strong": "#c45a28"}
    panels = [
        (axes[0], "", "Per-target oracle, zero allowed"),
        (axes[1], "fixed_", r"Fixed $a=-9.72\approx-10$"),
    ]
    for axis, prefix, title in panels:
        for name in ["weak", "strong"]:
            keep = table[f"{name}_de_genes"] >= MIN_DE
            axis.plot(
                table.loc[keep, "de_count_rank"],
                table.loc[keep, f"{prefix}{name}_nmae"],
                color=colors[name],
                linewidth=1,
                marker="o",
                markersize=2.5,
                label=f"{name.capitalize()} DE",
            )
        axis.axhline(1, color="0.45", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Perturbations ordered by total significant DE genes")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    axes[0].set_ylabel("DE LFC NMAE")
    axes[0].set_ylim(0.5, 1.75)
    figure.tight_layout()
    figure_path = OUTPUT / "weak_strong_nmae_by_de_count.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    outputs = [OUTPUT / "per_target.csv", OUTPUT / "summary.csv", figure_path]
    manifest = {
        "analysis": "weak and strong DE NMAE under per-target oracle and fixed native scaling",
        "scale_selection": {
            "oracle": "one native a per target minimizing NMAE over all significant DE genes, including zero; the same a is used for weak and strong subsets",
            "fixed": FIXED_A,
        },
        "ordering": "ascending total significant DE gene count, then target name",
        "bins": {
            "weak": "reference-significant and |LFC| < 0.5",
            "strong": "reference-significant and |LFC| >= 0.5",
            "minimum_genes_for_curve": MIN_DE,
        },
        "input": {
            str(INPUT.relative_to(ROOT)): sha256(INPUT),
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path) for path in outputs
        },
        "software": {
            "python": platform.python_version(),
            **{
                package: version(package)
                for package in ["matplotlib", "numpy", "pandas"]
            },
        },
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(
        "oracle zero fraction:",
        f"{table['oracle_at_zero'].mean():.6f}",
        "median nonzero a:",
        f"{table.loc[~table['oracle_at_zero'], 'oracle_a'].median():.6f}",
    )


if __name__ == "__main__":
    main()
