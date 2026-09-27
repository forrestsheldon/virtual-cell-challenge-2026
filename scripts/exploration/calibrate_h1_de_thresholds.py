"""Calibrate H1 DE thresholds against non-targeting guide comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

FDR_THRESHOLDS = [0.05, 0.01, 0.001, 0.0001]
FOLD_CHANGE_THRESHOLDS = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]
AUC_THRESHOLDS = [0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.16, 0.2, 0.25, 0.3, 0.35, 0.4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("reports/crispri-h1-exploration/generated")
    )
    args = parser.parse_args()

    null = pl.read_parquet(
        args.output / "wilcoxon_null_de.parquet",
        columns=["pseudo_guide", "fdr", "log2_fold_change", "auc_delta"],
    )
    mixscape_null = pl.read_csv(args.output / "mixscape_null_guides.csv")
    non_inert_guides = mixscape_null.filter(pl.col("direction_available"))[
        "pseudo_guide"
    ].to_list()
    target_cells = pl.read_csv(args.output / "wilcoxon_summary.csv").select(
        "target", "n_cells"
    )
    controls = json.loads((args.output / "wilcoxon_run.json").read_text())["n_controls"]
    target = (
        pl.read_parquet(
            args.output / "wilcoxon_de.parquet",
            columns=["target", "fdr", "log2_fold_change", "statistic"],
        )
        .join(target_cells, on="target")
        .with_columns(
            (pl.col("statistic") / (pl.col("n_cells") * controls) - 0.5)
            .abs()
            .alias("auc_delta")
        )
    )
    target_names = target_cells["target"].to_list()
    rows = []

    control_sets = {
        "all_31_nt_guides": null,
        "26_mixscape_null_passes": null.filter(
            ~pl.col("pseudo_guide").is_in(non_inert_guides)
        ),
    }
    for control_set, null_frame in control_sets.items():
        for fdr in FDR_THRESHOLDS:
            for fold_change in FOLD_CHANGE_THRESHOLDS:
                for auc in AUC_THRESHOLDS:
                    null_hits = null_frame.filter(
                        (pl.col("fdr") <= fdr)
                        & (pl.col("log2_fold_change").abs() >= fold_change)
                        & (pl.col("auc_delta") >= auc)
                    )
                    target_hits = target.filter(
                        (pl.col("fdr") <= fdr)
                        & (pl.col("log2_fold_change").abs() >= fold_change)
                        & (pl.col("auc_delta") >= auc)
                    )
                    counts = dict(target_hits.group_by("target").len().iter_rows())
                    target_counts = np.array(
                        [counts.get(name, 0) for name in target_names]
                    )
                    rows.append(
                        {
                            "control_set": control_set,
                            "fdr": fdr,
                            "absolute_log2_fold_change": fold_change,
                            "auc_delta": auc,
                            "null_genes": null_hits.height,
                            "null_guides": null_hits["pseudo_guide"].n_unique(),
                            "null_guide_fpr": null_hits["pseudo_guide"].n_unique()
                            / null_frame["pseudo_guide"].n_unique(),
                            "target_genes": target_hits.height,
                            "target_guides": int((target_counts > 0).sum()),
                            "median_target_genes": float(np.median(target_counts)),
                        }
                    )

    sweep = pd.DataFrame(rows)
    sweep.to_csv(args.output / "wilcoxon_null_threshold_sweep.csv", index=False)

    def record(frame):
        return frame.iloc[0].to_dict() if len(frame) else None

    recommendation = {
        "recommended_descriptive_threshold": {
            "fdr": 0.05,
            "absolute_log2_fold_change": 0.5,
            "auc_delta": 0.15,
            "basis": "zero calls among the 26 NT guides that passed the independent Mixscape null",
        },
        "non_inert_nt_guides": non_inert_guides,
        "control_sets": {},
        "grid": {
            "fdr": FDR_THRESHOLDS,
            "absolute_log2_fold_change": FOLD_CHANGE_THRESHOLDS,
            "auc_delta": AUC_THRESHOLDS,
        },
    }
    for control_set in control_sets:
        subset = sweep[sweep.control_set == control_set]
        zero_null = subset[subset.null_genes == 0].sort_values(
            ["target_genes", "fdr"], ascending=[False, False]
        )
        guide_fpr_05 = subset[subset.null_guide_fpr <= 0.05].sort_values(
            ["target_genes", "null_genes"], ascending=[False, True]
        )
        recommendation["control_sets"][control_set] = {
            "zero_null_genes_maximum_target_retention": record(zero_null),
            "null_guide_fpr_at_most_0.05_maximum_target_retention": record(
                guide_fpr_05
            ),
        }
    (args.output / "wilcoxon_null_threshold_recommendation.json").write_text(
        json.dumps(recommendation, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
