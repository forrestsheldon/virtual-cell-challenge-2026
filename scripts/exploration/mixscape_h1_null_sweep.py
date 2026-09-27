"""Sweep Mixscape neighbor matching over the failing non-targeting guides."""

import argparse
import itertools
import json
import time
from pathlib import Path

import pandas as pd
from mixscape_h1_null_controls import load_control_shards, run_guide

NEIGHBORS = [10, 20, 50]
DIMENSIONS = [10, 15, 30]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--signatures",
        type=Path,
        default=Path("data/derived/vcc2025_h1_mixscape/signatures"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/crispri-h1-exploration/generated")
    )
    parser.add_argument("--control-cells", type=int, default=5_000)
    args = parser.parse_args()

    baseline = pd.read_csv(args.output / "mixscape_null_guides.csv")
    guides = baseline.loc[baseline.n_ko > 0, "pseudo_guide"].tolist()
    total_nt_cells = int(baseline.n_cells.sum())
    paths = sorted(args.signatures.glob("*.h5ad"))
    shards = load_control_shards(paths)
    rows = []
    settings = list(itertools.product(NEIGHBORS, DIMENSIONS))
    started = time.monotonic()

    for setting_number, (neighbors, dimensions) in enumerate(settings, 1):
        for guide_number, guide in enumerate(guides, 1):
            result = run_guide(shards, guide, args.control_cells, neighbors, dimensions)
            ko = result.mixscape_class_global == "KO"
            rows.append(
                {
                    "neighbors": neighbors,
                    "pca_dimensions": dimensions,
                    "pseudo_guide": guide,
                    "n_cells": len(result),
                    "n_ko": int(ko.sum()),
                    "ko_fraction": ko.mean(),
                    "maximum_posterior": result.mixscape_class_p_ko.max(),
                }
            )
            complete = (setting_number - 1) * len(guides) + guide_number
            total = len(settings) * len(guides)
            eta = (time.monotonic() - started) / complete * (total - complete) / 60
            print(
                f"[{complete}/{total}] k={neighbors}, PCs={dimensions}, "
                f"{guide}: {ko.sum()} KO, ETA {eta:.1f} min",
                flush=True,
            )

    guide_results = pd.DataFrame(rows)
    summary = (
        guide_results.groupby(["neighbors", "pca_dimensions"])
        .agg(
            failing_guides=("n_ko", lambda x: int((x > 0).sum())),
            n_ko=("n_ko", "sum"),
            maximum_guide_fraction=("ko_fraction", "max"),
        )
        .reset_index()
    )
    summary["null_ko_fraction"] = summary.n_ko / total_nt_cells
    guide_results.to_csv(
        args.output / "mixscape_null_parameter_sweep_guides.csv", index=False
    )
    summary.to_csv(args.output / "mixscape_null_parameter_sweep.csv", index=False)
    (args.output / "mixscape_null_parameter_sweep_run.json").write_text(
        json.dumps(
            {
                "neighbors": NEIGHBORS,
                "pca_dimensions": DIMENSIONS,
                "pseudo_guides": guides,
                "invariant_clean_guides": int((baseline.n_ko == 0).sum()),
                "control_cells_per_guide": args.control_cells,
                "test_method": "wilcoxon",
                "seed": 0,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
