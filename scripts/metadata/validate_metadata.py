#!/usr/bin/env python3
"""Validate curated YAML/CSV structure, atlas, and coverage invariants."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "metadata" / "datasets"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return next(csv.reader(stream))


def validate_atlas() -> tuple[int, int, int, int]:
    config = yaml.safe_load(
        (DATASETS / "perturbation_atlas_sources.yaml").read_text(encoding="utf-8")
    )
    context_ids = [row["dataset_context_id"] for row in config["contexts"]]
    assert len(context_ids) == len(set(context_ids))

    registry_rows = rows(DATASETS / "dataset_contexts.csv")
    registry = {row["dataset_context_id"]: row for row in registry_rows}
    assert len(registry) == len(registry_rows)
    assert set(context_ids) <= set(registry)

    observations = rows(DATASETS / "perturbation_observations.csv")
    required_fields = {
        "target_gene",
        "source_target_gene",
        "dataset_id",
        "dataset_context_id",
        "canonical_context_id",
        "study_id",
        "perturbation_modality",
        "crispri",
        "n_cells",
        "n_cells_status",
        "n_cells_source",
        "n_cells_scope",
        "n_guides",
        "n_control_cells",
        "guide_level_metadata_available",
        "raw_counts_available",
        "knockdown_efficiency_available",
        "knockdown_efficiency",
        "source_variant_id",
        "source_url",
        "target_list_source",
        "confidence",
        "present_in_vcc2026_panel",
        "notes",
    }
    assert required_fields <= set(observations[0])
    observation_keys = [
        (row["target_gene"], row["dataset_context_id"]) for row in observations
    ]
    assert len(observations) == 20073
    assert len(observation_keys) == len(set(observation_keys))
    assert {row["dataset_context_id"] for row in observations} == set(context_ids)
    assert {row["n_cells_status"] for row in observations} == {"exact"}

    excluded = {label.upper() for label in config["excluded_control_labels"]}
    for row in observations:
        assert row["target_gene"] == row["target_gene"].strip().upper()
        assert row["target_gene"] not in excluded
        assert row["crispri"] == "True"
        registry_row = registry[row["dataset_context_id"]]
        assert row["dataset_id"] == registry_row["dataset_id"]
        assert row["canonical_context_id"] == registry_row["canonical_context_id"]
        assert row["study_id"] == registry_row["study_id"]
        assert row["n_cells_status"] in {"exact", "partial", "unresolved"}
        if row["n_cells_status"] == "exact":
            assert row["n_cells"] != ""
            assert int(row["n_cells"]) >= 0
            assert row["n_cells_source"] != ""
        elif row["n_cells_status"] == "unresolved":
            assert row["n_cells"] == ""
        assert row["n_cells_scope"] != ""

    parquet = pq.read_table(DATASETS / "perturbation_observations.parquet")
    assert parquet.num_rows == len(observations)
    assert set(parquet.column_names) == set(observations[0])
    parquet_keys = set(
        zip(
            parquet.column("target_gene").to_pylist(),
            parquet.column("dataset_context_id").to_pylist(),
            strict=True,
        )
    )
    assert parquet_keys == set(observation_keys)

    context_sets: dict[str, set[str]] = defaultdict(set)
    canonical_sets: dict[str, set[str]] = defaultdict(set)
    for row in observations:
        context_sets[row["dataset_context_id"]].add(row["target_gene"])
        canonical_sets[row["canonical_context_id"]].add(row["target_gene"])
    genes = sorted({row["target_gene"] for row in observations})

    gene_context = rows(DATASETS / "gene_x_dataset_context.csv")
    assert header(DATASETS / "gene_x_dataset_context.csv") == [
        "target_gene",
        *context_ids,
    ]
    assert [row["target_gene"] for row in gene_context] == genes
    for row in gene_context:
        for context_id in context_ids:
            assert row[context_id] == str(
                int(row["target_gene"] in context_sets[context_id])
            )

    canonical_ids = list(
        dict.fromkeys(
            registry[context_id]["canonical_context_id"] for context_id in context_ids
        )
    )
    gene_canonical = rows(DATASETS / "gene_x_canonical_context.csv")
    assert header(DATASETS / "gene_x_canonical_context.csv") == [
        "target_gene",
        *canonical_ids,
    ]
    assert [row["target_gene"] for row in gene_canonical] == genes
    for row in gene_canonical:
        for canonical_id in canonical_ids:
            assert row[canonical_id] == str(
                int(row["target_gene"] in canonical_sets[canonical_id])
            )

    overlap = rows(DATASETS / "dataset_context_overlap.csv")
    assert header(DATASETS / "dataset_context_overlap.csv") == [
        "dataset_context_id",
        *context_ids,
    ]
    assert [row["dataset_context_id"] for row in overlap] == context_ids
    by_source = {row["dataset_context_id"]: row for row in overlap}
    for source in context_ids:
        assert int(by_source[source][source]) == len(context_sets[source])
        for target in context_ids:
            expected = len(context_sets[source] & context_sets[target])
            assert int(by_source[source][target]) == expected
            assert by_source[source][target] == by_source[target][source]

    edges = rows(DATASETS / "bridge_edges.csv")
    edge_keys = [
        (row["source_dataset_context"], row["target_dataset_context"]) for row in edges
    ]
    assert len(edge_keys) == len(set(edge_keys))
    for row in edges:
        source = row["source_dataset_context"]
        target = row["target_dataset_context"]
        assert int(row["n_shared_perturbations"]) == len(
            context_sets[source] & context_sets[target]
        )

    exposure = rows(DATASETS / "benchmark_gene_exposure.csv")
    assert len(exposure) == 2 * len(observations)
    assert {row["benchmark_variant"] for row in exposure} == {
        "E5_canonical_context_holdout",
        "E5_ablation_experimental_context_holdout",
    }
    assert {row["exposure_label"] for row in exposure} <= {
        "Seen-1",
        "Seen-multi",
        "Unseen",
    }
    assert all(int(row["k_g"]) >= 0 for row in exposure)

    for dataset, prefix in (("jiang2025", "jiang"), ("mcfaline2024", "mcfaline")):
        audit = rows(DATASETS / "source_targets" / f"{prefix}_cell_count_audit.csv")
        assert {row["dataset"] for row in audit} == {dataset}
        by_context: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in audit:
            by_context[row["context"]].append(row)
        for context, context_rows in by_context.items():
            total = sum(int(row["n_cells_in_representation"]) for row in context_rows)
            controls = sum(int(row["n_control_or_missing"]) for row in context_rows)
            multiples = sum(
                int(row["n_multiple_assignment_labels"]) for row in context_rows
            )
            single_targets = sum(
                int(row["n_single_target_cells"]) for row in context_rows
            )
            assert total == controls + multiples + single_targets
            count_rows = rows(
                DATASETS / "source_targets" / f"{prefix}_{context.lower()}_counts.csv"
            )
            assert sum(int(row["n_cells"]) for row in count_rows) == single_targets
            assert all(int(row["n_cells"]) > 0 for row in count_rows)

    report_dir = ROOT / "reports" / "benchmark-capacity"
    capacity = rows(report_dir / "capacity_summary.csv")
    assert [row["benchmark_id"] for row in capacity] == [
        "E0",
        "E1",
        "E2",
        "E3a",
        "E3b",
        "E4",
        "E5",
        "E5_ablation",
    ]
    assert len(rows(report_dir / "E0_sampling_capacity.csv")) == len(context_ids)
    assert len(rows(report_dir / "E1_distribution_capacity.csv")) == len(context_ids)
    assert len(rows(report_dir / "E3a_pairwise_within_study.csv")) == 40
    assert len(rows(report_dir / "E3b_multisource_within_study_holdouts.csv")) == 13
    assert len(rows(report_dir / "E5_multisource_holdouts.csv")) == 2 * len(context_ids)

    cell_summary = rows(report_dir / "cell_count_attrition_summary.csv")
    assert len(cell_summary) == 4 * 5
    assert {row["benchmark"] for row in cell_summary} == {
        "E3a",
        "E3b",
        "E4",
        "E5",
    }
    expected_thresholds = [50, 100, 200, 400, 800]
    for benchmark in ("E3a", "E3b", "E4", "E5"):
        benchmark_rows = [row for row in cell_summary if row["benchmark"] == benchmark]
        assert [int(row["n_min"]) for row in benchmark_rows] == expected_thresholds
        for metric in (
            "n_target_contexts",
            "n_unique_genes",
            "n_gene_context_evaluations",
        ):
            values = [int(row[metric]) for row in benchmark_rows]
            assert values == sorted(values, reverse=True)

    cell_by_target = rows(report_dir / "cell_count_attrition_by_target_context.csv")
    target_keys = [
        (row["benchmark"], row["n_min"], row["target_dataset_context"])
        for row in cell_by_target
    ]
    assert len(target_keys) == len(set(target_keys))
    for row in cell_by_target:
        assert row["target_dataset_context"] in registry
        if row["benchmark"] == "E4":
            assert (
                registry[row["target_dataset_context"]]["canonical_context_id"]
                != "K562"
            )

    pair_thresholds = rows(report_dir / "E3a_pairwise_cell_thresholds.csv")
    pair_keys = [
        (
            row["n_min"],
            row["source_dataset_context"],
            row["target_dataset_context"],
        )
        for row in pair_thresholds
    ]
    assert len(pair_keys) == len(set(pair_keys))
    assert all(
        row["source_dataset_context"] != row["target_dataset_context"]
        for row in pair_thresholds
    )

    qualified_exposure = rows(report_dir / "cell_count_qualified_exposure.csv")
    exposure_keys = [
        (
            row["benchmark"],
            row["n_min"],
            row["target_dataset_context"],
            row["target_gene"],
        )
        for row in qualified_exposure
    ]
    assert len(exposure_keys) == len(set(exposure_keys))
    assert {row["benchmark"] for row in qualified_exposure} == {"E3b", "E5"}
    for row in qualified_exposure:
        assert int(row["k_g_qualified"]) >= 0
        expected_usable = (
            row["target_qualified"] == "True" and int(row["k_g_qualified"]) >= 1
        )
        assert (row["usable"] == "True") == expected_usable

    return (
        len(observations),
        len(genes),
        len(context_ids),
        len(canonical_ids),
    )


def main() -> None:
    for path in sorted((ROOT / "context" / "vcc2026").glob("*.yaml")) + sorted(
        DATASETS.glob("*.yaml")
    ):
        yaml.safe_load(path.read_text(encoding="utf-8"))

    targets = rows(ROOT / "data" / "controls" / "pert_counts.csv")
    matrix = rows(DATASETS / "target_coverage_matrix.csv")
    summary = rows(DATASETS / "target_summary.csv")
    long = rows(DATASETS / "target_coverage_long.csv")
    assert len(targets) == 300
    assert len(matrix) == len(summary) == 300
    assert [row["target_gene"] for row in targets] == [
        row["target_gene"] for row in matrix
    ]
    assert len(long) == 300 * (len(matrix[0]) - 1)

    variants = rows(DATASETS / "dataset_variants.csv")
    variant_ids = [row["variant_id"] for row in variants]
    assert len(variant_ids) == len(set(variant_ids))
    valid_variants = set(variant_ids)
    for row in long:
        assert row["variant_id"] in valid_variants

    atlas_observations, atlas_genes, atlas_contexts, canonical_contexts = (
        validate_atlas()
    )

    print(
        f"validated: 300 targets, {len(variants)} variants, "
        f"{len(long) // 300} VCC coverage columns; "
        f"{atlas_observations} atlas observations, {atlas_genes} genes, "
        f"{atlas_contexts} dataset-contexts, {canonical_contexts} canonical contexts"
    )


if __name__ == "__main__":
    main()
