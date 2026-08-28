#!/usr/bin/env python3
"""Regenerate experiment-level VCC target coverage from curated target lists."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


def read_column(path: Path, column: str) -> list[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            row[column].strip() for row in csv.DictReader(stream) if row[column].strip()
        ]


def read_targets(path: Path) -> set[str]:
    return {
        value.strip().upper()
        for value in path.read_text(encoding="utf-8").splitlines()
        if value.strip() and not value.startswith("#")
    }


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("metadata/datasets/coverage_sources.yaml")
    )
    parser.add_argument(
        "--vcc-targets", type=Path, default=Path("data/controls/pert_counts.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("metadata/datasets"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    vcc_targets = read_column(args.vcc_targets, "target_gene")
    sources = config["sources"]
    source_targets = {
        source["coverage_id"]: read_targets(args.config.parent / source["target_file"])
        for source in sources
    }

    long_rows: list[dict[str, object]] = []
    for target in vcc_targets:
        for source in sources:
            present = target.upper() in source_targets[source["coverage_id"]]
            long_rows.append(
                {
                    "target_gene": target,
                    "dataset_id": source["dataset_id"],
                    "cell_context": source["cell_context"],
                    "variant_id": source["variant_id"],
                    "perturbation_modality": source["perturbation_modality"],
                    "present": int(present),
                    "source_target_name": target if present else "",
                    "n_cells": "",
                    "n_guides": "",
                    "knockdown_efficiency": "",
                    "source_url": source["source_url"],
                }
            )

    long_fields = list(long_rows[0])
    write_csv(args.output_dir / "target_coverage_long.csv", long_fields, long_rows)

    matrix_fields = ["target_gene"] + [source["coverage_id"] for source in sources]
    matrix_rows = [
        {"target_gene": target}
        | {
            source["coverage_id"]: int(
                target.upper() in source_targets[source["coverage_id"]]
            )
            for source in sources
        }
        for target in vcc_targets
    ]
    write_csv(
        args.output_dir / "target_coverage_matrix.csv", matrix_fields, matrix_rows
    )

    summary_rows = []
    for target in vcc_targets:
        hits = [
            source
            for source in sources
            if target.upper() in source_targets[source["coverage_id"]]
        ]
        crispri = [
            source for source in hits if source["perturbation_modality"] == "CRISPRi"
        ]
        summary_rows.append(
            {
                "target_gene": target,
                "n_biological_datasets": len({source["dataset_id"] for source in hits}),
                "n_crispri_datasets": len({source["dataset_id"] for source in crispri}),
                "n_cell_contexts": len({source["coverage_id"] for source in hits}),
                "n_crispri_cell_contexts": len(
                    {source["coverage_id"] for source in crispri}
                ),
                "contexts": ";".join(
                    sorted({source["cell_context"] for source in hits})
                ),
                "datasets": ";".join(sorted({source["dataset_id"] for source in hits})),
            }
        )
    write_csv(
        args.output_dir / "target_summary.csv", list(summary_rows[0]), summary_rows
    )


if __name__ == "__main__":
    main()
