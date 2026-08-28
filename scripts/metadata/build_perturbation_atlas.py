#!/usr/bin/env python3
"""Build the nominal public CRISPRi atlas and benchmark-capacity tables."""

from __future__ import annotations

import argparse
import csv
import itertools
import statistics
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

OBSERVATION_FIELDS = [
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
    "assay_family",
    "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def as_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def normalize_gene(value: str) -> str:
    return value.strip().upper()


def resolve_contexts(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["dataset_context_id"]: row for row in raw}
    resolved: dict[str, dict[str, Any]] = {}

    def resolve(context_id: str, stack: tuple[str, ...] = ()) -> dict[str, Any]:
        if context_id in resolved:
            return resolved[context_id]
        if context_id in stack:
            raise ValueError(f"Cyclic context inheritance: {stack + (context_id,)}")
        row = deepcopy(by_id[context_id])
        parent = row.pop("inherit", None)
        if parent:
            merged = deepcopy(resolve(parent, stack + (context_id,)))
            merged.update(row)
            row = merged
        resolved[context_id] = row
        return row

    return [resolve(row["dataset_context_id"]) for row in raw]


def load_guide_counts(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    return {
        normalize_gene(row["source_target_gene"]): int(row["n_guides"])
        for row in read_csv(path)
    }


def load_target_rows(
    path: Path,
    file_format: str,
    excluded: set[str],
    source_variant_id: str,
    n_control_cells: int | None,
    panel: str | None = None,
) -> list[dict[str, Any]]:
    if file_format == "counts_csv":
        raw = read_csv(path)
        values = [
            {
                "source_target_gene": row["source_target_gene"].strip(),
                "n_cells": as_int(row["n_cells"]),
            }
            for row in raw
        ]
    elif file_format == "lines":
        values = [
            {"source_target_gene": line.strip(), "n_cells": None}
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        raise ValueError(f"Unknown target file format: {file_format}")

    result = []
    for row in values:
        target = normalize_gene(row["source_target_gene"])
        if target.lower() in excluded:
            continue
        result.append(
            row
            | {
                "target_gene": target,
                "source_variant_id": source_variant_id,
                "n_control_cells": n_control_cells,
                "panel": panel,
            }
        )
    return result


def prefer_target(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    """Prefer the panel with the larger observed cell group; never sum panels."""
    candidate_cells = candidate["n_cells"]
    current_cells = current["n_cells"]
    if current_cells is None:
        return candidate_cells is not None
    if candidate_cells is None:
        return False
    return candidate_cells > current_cells


def build_observations(
    contexts: list[dict[str, Any]],
    datasets_dir: Path,
    vcc_targets: set[str],
    excluded: set[str],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for context in contexts:
        guide_path = context.get("guide_counts_file")
        guide_counts = load_guide_counts(
            datasets_dir / guide_path if guide_path else None
        )
        target_rows: list[dict[str, Any]] = []
        if "target_sources" in context:
            for source in context["target_sources"]:
                target_rows.extend(
                    load_target_rows(
                        datasets_dir / source["target_file"],
                        context["target_file_format"],
                        excluded,
                        source["source_variant_id"],
                        as_int(source.get("n_control_cells")),
                        source.get("panel"),
                    )
                )
        else:
            target_rows = load_target_rows(
                datasets_dir / context["target_file"],
                context["target_file_format"],
                excluded,
                context["source_variant_id"],
                as_int(context.get("n_control_cells")),
            )

        selected: dict[str, dict[str, Any]] = {}
        for row in target_rows:
            target = row["target_gene"]
            if target not in selected or prefer_target(row, selected[target]):
                selected[target] = row

        for target in sorted(selected):
            row = selected[target]
            n_guides = guide_counts.get(
                target, as_int(context.get("constant_n_guides"))
            )
            notes = context.get("notes")
            if row.get("panel"):
                notes = f"{notes or ''} Selected panel: {row['panel']}.".strip()
            observations.append(
                {
                    "target_gene": target,
                    "source_target_gene": row["source_target_gene"],
                    "dataset_id": context["dataset_id"],
                    "dataset_context_id": context["dataset_context_id"],
                    "canonical_context_id": context["canonical_context_id"],
                    "study_id": context["study_id"],
                    "perturbation_modality": context["perturbation_modality"],
                    "crispri": as_bool(context["crispri"]),
                    "n_cells": row["n_cells"],
                    "n_cells_status": context["cell_count_status"],
                    "n_cells_source": context.get("cell_count_source"),
                    "n_cells_scope": context["cell_count_scope"],
                    "n_guides": n_guides,
                    "n_control_cells": row["n_control_cells"],
                    "guide_level_metadata_available": as_bool(
                        context["guide_level_metadata_available"]
                    ),
                    "raw_counts_available": as_bool(context["raw_counts_available"]),
                    "knockdown_efficiency_available": as_bool(
                        context["knockdown_efficiency_available"]
                    ),
                    "knockdown_efficiency": None,
                    "source_variant_id": row["source_variant_id"],
                    "source_url": context["source_url"],
                    "target_list_source": context["target_list_source"],
                    "confidence": context["confidence"],
                    "present_in_vcc2026_panel": target in vcc_targets,
                    "assay_family": context["assay_family"],
                    "notes": notes,
                }
            )
    keys = [(row["target_gene"], row["dataset_context_id"]) for row in observations]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate gene × dataset-context observations")
    return observations


def make_sets(observations: list[dict[str, Any]], key: str) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in observations:
        result[row[key]].add(row["target_gene"])
    return dict(result)


def matrix_rows(
    genes: list[str], columns: list[str], sets: dict[str, set[str]]
) -> list[dict[str, Any]]:
    return [
        {"target_gene": gene}
        | {column: int(gene in sets.get(column, set())) for column in columns}
        for gene in genes
    ]


def build_edges(
    context_ids: list[str],
    context_sets: dict[str, set[str]],
    context_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    edges = []
    for source, target in itertools.combinations(context_ids, 2):
        shared = context_sets[source] & context_sets[target]
        if not shared:
            continue
        left, right = context_meta[source], context_meta[target]
        same_context = left["canonical_context_id"] == right["canonical_context_id"]
        same_study = left["study_id"] == right["study_id"]
        assay_left = left.get("assay_family", "unknown")
        assay_right = right.get("assay_family", "unknown")
        same_assay = assay_left == assay_right and assay_left != "unknown"
        if same_context:
            note = "Same named biological context; state and protocol equivalence not guaranteed."
        elif same_study:
            note = "Within-study biological-context bridge."
        else:
            note = "Cross-study and cross-context perturbation bridge."
        edges.append(
            {
                "source_dataset_context": source,
                "target_dataset_context": target,
                "source_canonical_context": left["canonical_context_id"],
                "target_canonical_context": right["canonical_context_id"],
                "same_canonical_context": same_context,
                "same_study": same_study,
                "n_shared_perturbations": len(shared),
                "shared_fraction_source": len(shared) / len(context_sets[source]),
                "shared_fraction_target": len(shared) / len(context_sets[target]),
                "same_modality": left["perturbation_modality"]
                == right["perturbation_modality"],
                "same_or_similar_assay": same_assay,
                "source_dataset": left["dataset_id"],
                "target_dataset": right["dataset_id"],
                "source_assay_family": assay_left,
                "target_assay_family": assay_right,
                "notes": note,
            }
        )
    return edges


def summarize_genes(
    genes: set[str],
    observations: list[dict[str, Any]],
    vcc_targets: set[str],
) -> list[dict[str, Any]]:
    by_gene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_gene[row["target_gene"]].append(row)
    result = []
    for gene in sorted(genes | vcc_targets):
        rows = by_gene.get(gene, [])
        contexts = sorted({row["dataset_context_id"] for row in rows})
        canonical = sorted({row["canonical_context_id"] for row in rows})
        studies = sorted({row["study_id"] for row in rows})
        if len(contexts) == 0:
            exposure = "unseen_as_perturbation_in_atlas"
        elif len(contexts) == 1:
            exposure = "seen_one_dataset_context"
        else:
            exposure = "seen_multiple_dataset_contexts"
        result.append(
            {
                "target_gene": gene,
                "n_dataset_contexts": len(contexts),
                "n_canonical_contexts": len(canonical),
                "n_studies": len(studies),
                "n_crispri_dataset_contexts": sum(
                    as_bool(row["crispri"]) for row in rows
                ),
                "n_crispri_canonical_contexts": len(
                    {
                        row["canonical_context_id"]
                        for row in rows
                        if as_bool(row["crispri"])
                    }
                ),
                "dataset_contexts": ";".join(contexts),
                "canonical_contexts": ";".join(canonical),
                "studies": ";".join(studies),
                "present_in_replogle_k562": any(
                    row["dataset_context_id"] == "replogle2022::K562" for row in rows
                ),
                "present_in_vcc2025_h1": any(
                    row["dataset_context_id"] == "vcc2025_h1::H1" for row in rows
                ),
                "present_in_vcc2026_panel": gene in vcc_targets,
                "nominal_exposure_class": exposure,
            }
        )
    return result


def build_e0(
    context_ids: list[str],
    observations: list[dict[str, Any]],
    context_meta: dict[str, dict[str, Any]],
    registry: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[row["dataset_context_id"]].append(row)
    result = []
    for context_id in context_ids:
        rows = grouped[context_id]
        known = [row["n_cells"] for row in rows if row["n_cells"] is not None]
        result.append(
            {
                "dataset_context_id": context_id,
                "canonical_context_id": context_meta[context_id][
                    "canonical_context_id"
                ],
                "study_id": context_meta[context_id]["study_id"],
                "n_perturbations": len(rows),
                "n_with_known_cell_count": len(known),
                "minimum_cells": min(known) if known else None,
                "median_cells": statistics.median(known) if known else None,
                "maximum_cells": max(known) if known else None,
                "n_supporting_400_vs_400": sum(value >= 800 for value in known),
                "n_supporting_200_vs_200": sum(value >= 400 for value in known),
                "n_supporting_100_vs_100": sum(value >= 200 for value in known),
                "n_supporting_50_vs_50": sum(value >= 100 for value in known),
                "fraction_supporting_400_vs_400_among_known": (
                    sum(value >= 800 for value in known) / len(known) if known else None
                ),
                "raw_counts_available": context_meta[context_id][
                    "raw_counts_available"
                ],
                "controls_available": as_bool(
                    registry[context_id]["controls_available"]
                ),
                "notes": (
                    "Split sizes are descriptive capacity counts, not quality filters."
                    if known
                    else "Per-perturbation cell counts require a larger source data object."
                ),
            }
        )
    return result


def build_e1(
    context_ids: list[str],
    context_sets: dict[str, set[str]],
    context_meta: dict[str, dict[str, Any]],
    registry: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    return [
        {
            "dataset_context_id": context_id,
            "canonical_context_id": context_meta[context_id]["canonical_context_id"],
            "study_id": context_meta[context_id]["study_id"],
            "n_nominal_perturbations": len(context_sets[context_id]),
            "perturbation_cells_available": True,
            "controls_available": as_bool(registry[context_id]["controls_available"]),
            "raw_counts_available": context_meta[context_id]["raw_counts_available"],
            "guide_level_metadata_available": context_meta[context_id][
                "guide_level_metadata_available"
            ],
            "nominally_eligible": as_bool(registry[context_id]["controls_available"])
            and as_bool(context_meta[context_id]["raw_counts_available"]),
            "notes": "No cell-count, response-strength, or knockdown-quality filter applied.",
        }
        for context_id in context_ids
    ]


def build_e2(
    context_ids: list[str],
    context_sets: dict[str, set[str]],
    context_meta: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    result = []
    usable: dict[str, set[str]] = {}
    for source, target in itertools.combinations(context_ids, 2):
        left, right = context_meta[source], context_meta[target]
        if left["canonical_context_id"] != right["canonical_context_id"]:
            continue
        if left["study_id"] == right["study_id"]:
            continue
        shared = context_sets[source] & context_sets[target]
        if not shared:
            continue
        pair_id = f"{source} -> {target}"
        usable[pair_id] = shared
        result.append(
            {
                "source_dataset_context": source,
                "target_dataset_context": target,
                "canonical_context_id": left["canonical_context_id"],
                "source_study": left["study_id"],
                "target_study": right["study_id"],
                "source_dataset": left["dataset_id"],
                "target_dataset": right["dataset_id"],
                "n_shared_perturbations": len(shared),
                "shared_fraction_source": len(shared) / len(context_sets[source]),
                "shared_fraction_target": len(shared) / len(context_sets[target]),
                "same_modality": left["perturbation_modality"]
                == right["perturbation_modality"],
                "source_assay_family": left["assay_family"],
                "target_assay_family": right["assay_family"],
                "same_or_similar_assay": left["assay_family"] == right["assay_family"]
                and left["assay_family"] != "unknown",
                "notes": "Named cell line matches; study, state, and protocol remain distinct.",
            }
        )
    return result, usable


def build_e3a(
    context_ids: list[str],
    context_sets: dict[str, set[str]],
    context_meta: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    by_study: dict[str, list[str]] = defaultdict(list)
    for context_id in context_ids:
        by_study[context_meta[context_id]["study_id"]].append(context_id)
    rows = []
    usable: dict[str, set[str]] = {}
    for study, study_contexts in by_study.items():
        for source, target in itertools.permutations(study_contexts, 2):
            shared = context_sets[source] & context_sets[target]
            if not shared:
                continue
            pair_id = f"{source} -> {target}"
            usable[pair_id] = shared
            rows.append(
                {
                    "study": study,
                    "source_dataset_context": source,
                    "source_canonical_context": context_meta[source][
                        "canonical_context_id"
                    ],
                    "target_dataset_context": target,
                    "target_canonical_context": context_meta[target][
                        "canonical_context_id"
                    ],
                    "n_source_genes": len(context_sets[source]),
                    "n_target_genes": len(context_sets[target]),
                    "n_shared_genes": len(shared),
                    "notes": "Eligibility is target ∩ source; ordered source-target pairs are distinct evaluation units.",
                }
            )
    return rows, usable


def build_e3b(
    context_ids: list[str],
    context_sets: dict[str, set[str]],
    context_meta: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    by_study: dict[str, list[str]] = defaultdict(list)
    for context_id in context_ids:
        by_study[context_meta[context_id]["study_id"]].append(context_id)
    rows = []
    usable: dict[str, set[str]] = {}
    for study, study_contexts in by_study.items():
        if len(study_contexts) < 2:
            continue
        for target in study_contexts:
            sources = [context for context in study_contexts if context != target]
            source_union = set().union(*(context_sets[source] for source in sources))
            source_intersection = set.intersection(
                *(context_sets[source] for source in sources)
            )
            broad = context_sets[target] & source_union
            strict = context_sets[target] & source_intersection
            usable[target] = broad
            rows.append(
                {
                    "study": study,
                    "target_dataset_context": target,
                    "target_canonical_context": context_meta[target][
                        "canonical_context_id"
                    ],
                    "source_dataset_contexts": ";".join(sources),
                    "n_source_contexts": len(sources),
                    "n_target_genes": len(context_sets[target]),
                    "n_genes_present_in_at_least_one_source": len(source_union),
                    "n_genes_present_in_all_source_contexts": len(source_intersection),
                    "n_genes_also_present_in_target": len(broad),
                    "n_genes_usable_for_zero_shot_evaluation": len(broad),
                    "n_genes_strict_all_sources_plus_target": len(strict),
                    "pairwise_source_target_overlaps": ";".join(
                        f"{source}:{len(context_sets[source] & context_sets[target])}"
                        for source in sources
                    ),
                    "notes": "Main rule uses target ∩ union(source contexts); stricter subset uses target ∩ intersection(all source contexts).",
                }
            )
    return rows, usable


def build_e4(
    anchor: str,
    context_ids: list[str],
    context_sets: dict[str, set[str]],
    context_meta: dict[str, dict[str, Any]],
    vcc_targets: set[str],
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    rows = []
    usable = {}
    anchor_set = context_sets[anchor]
    for target in context_ids:
        if target == anchor:
            continue
        if context_meta[target]["study_id"] == context_meta[anchor]["study_id"]:
            continue
        shared = anchor_set & context_sets[target]
        if not shared:
            continue
        usable[target] = shared
        rows.append(
            {
                "target_dataset_context": target,
                "canonical_context": context_meta[target]["canonical_context_id"],
                "target_study": context_meta[target]["study_id"],
                "n_target_genes": len(context_sets[target]),
                "n_shared_with_replogle": len(shared),
                "shared_gene_fraction": len(shared) / len(context_sets[target]),
                "shared_fraction_replogle": len(shared) / len(anchor_set),
                "same_context_as_replogle": context_meta[target]["canonical_context_id"]
                == context_meta[anchor]["canonical_context_id"],
                "same_study": context_meta[target]["study_id"]
                == context_meta[anchor]["study_id"],
                "n_vcc300_shared": len(shared & vcc_targets),
                "n_non_vcc_shared": len(shared - vcc_targets),
                "assay_modality_notes": (
                    f"CRISPRi; {context_meta[anchor]['assay_family']} -> "
                    f"{context_meta[target]['assay_family']}"
                ),
                "notes": "Target perturbations are hidden; target controls remain available in the planned benchmark.",
            }
        )
    return rows, usable


def build_e5(
    variant: str,
    context_ids: list[str],
    context_sets: dict[str, set[str]],
    context_meta: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, set[str]], list[dict[str, Any]]]:
    rows = []
    usable = {}
    exposures = []
    for target in context_ids:
        target_canonical = context_meta[target]["canonical_context_id"]
        if variant == "E5_ablation_experimental_context_holdout":
            training = [context for context in context_ids if context != target]
        elif variant == "E5_canonical_context_holdout":
            training = [
                context
                for context in context_ids
                if context_meta[context]["canonical_context_id"] != target_canonical
            ]
        else:
            raise ValueError(variant)
        counts = {
            gene: sum(gene in context_sets[context] for context in training)
            for gene in context_sets[target]
        }
        eligible = {gene for gene, count in counts.items() if count >= 1}
        usable[target] = eligible
        rows.append(
            {
                "benchmark_variant": variant,
                "target_dataset_context": target,
                "canonical_context": target_canonical,
                "target_study": context_meta[target]["study_id"],
                "n_training_dataset_contexts": len(training),
                "n_training_canonical_contexts": len(
                    {
                        context_meta[context]["canonical_context_id"]
                        for context in training
                    }
                ),
                "n_target_genes": len(context_sets[target]),
                "n_genes_seen_1": sum(count == 1 for count in counts.values()),
                "n_genes_seen_multi": sum(count >= 2 for count in counts.values()),
                "n_genes_k_g_ge_3": sum(count >= 3 for count in counts.values()),
                "n_genes_unseen": sum(count == 0 for count in counts.values()),
                "notes": "Capacity only; no quality or minimum-cell filtering.",
            }
        )
        for gene, count in sorted(counts.items()):
            if count == 0:
                exposure = "Unseen"
            elif count == 1:
                exposure = "Seen-1"
            else:
                exposure = "Seen-multi"
            exposures.append(
                {
                    "benchmark_variant": variant,
                    "target_dataset_context": target,
                    "canonical_context": target_canonical,
                    "target_gene": gene,
                    "k_g": count,
                    "n_training_canonical_contexts_with_gene": len(
                        {
                            context_meta[context]["canonical_context_id"]
                            for context in training
                            if gene in context_sets[context]
                        }
                    ),
                    "exposure_label": exposure,
                }
            )
    return rows, usable, exposures


def capacity_row(
    benchmark_id: str,
    target_keys: list[str],
    usable: dict[str, set[str]],
    context_meta: dict[str, dict[str, Any]],
    notes: str,
) -> dict[str, Any]:
    counts = [len(usable[key]) for key in target_keys]
    unique = (
        set().union(*(usable[key] for key in target_keys)) if target_keys else set()
    )
    context_targets = [key for key in target_keys if key in context_meta]
    return {
        "benchmark_id": benchmark_id,
        "n_studies": len({context_meta[key]["study_id"] for key in context_targets}),
        "n_target_dataset_contexts": len(target_keys),
        "n_target_canonical_contexts": len(
            {context_meta[key]["canonical_context_id"] for key in context_targets}
        ),
        "n_unique_genes": len(unique),
        "n_gene_evaluation_units": sum(counts),
        "median_genes_per_evaluation_target": statistics.median(counts)
        if counts
        else 0,
        "min_genes_per_evaluation_target": min(counts) if counts else 0,
        "max_genes_per_evaluation_target": max(counts) if counts else 0,
        "notes": notes,
    }


def pair_capacity_row(
    benchmark_id: str,
    rows: list[dict[str, Any]],
    usable: dict[str, set[str]],
    context_meta: dict[str, dict[str, Any]],
    notes: str,
) -> dict[str, Any]:
    counts = [len(genes) for genes in usable.values()]
    unique = set().union(*usable.values()) if usable else set()
    targets = {row["target_dataset_context"] for row in rows}
    return {
        "benchmark_id": benchmark_id,
        "n_studies": len({row["study"] for row in rows}),
        "n_target_dataset_contexts": len(targets),
        "n_target_canonical_contexts": len(
            {context_meta[target]["canonical_context_id"] for target in targets}
        ),
        "n_unique_genes": len(unique),
        "n_gene_evaluation_units": sum(counts),
        "median_genes_per_evaluation_target": statistics.median(counts)
        if counts
        else 0,
        "min_genes_per_evaluation_target": min(counts) if counts else 0,
        "max_genes_per_evaluation_target": max(counts) if counts else 0,
        "notes": notes,
    }


CELL_THRESHOLDS = (50, 100, 200, 400, 800)


def build_cell_threshold_capacity(
    context_ids: list[str],
    observations: list[dict[str, Any]],
    context_meta: dict[str, dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Build E3--E5 capacity using only observations with verified cell counts."""
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    target_genes: dict[str, set[str]] = defaultdict(set)
    for row in observations:
        context = row["dataset_context_id"]
        gene = row["target_gene"]
        target_genes[context].add(gene)
        if row["n_cells"] is not None:
            counts[context][gene] = int(row["n_cells"])

    by_study: dict[str, list[str]] = defaultdict(list)
    for context in context_ids:
        by_study[context_meta[context]["study_id"]].append(context)

    summary_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []

    def add_summary(
        benchmark: str, threshold: int, usable: dict[str, set[str]]
    ) -> None:
        nonempty = {target: genes for target, genes in usable.items() if genes}
        unique = set().union(*nonempty.values()) if nonempty else set()
        summary_rows.append(
            {
                "benchmark": benchmark,
                "n_min": threshold,
                "n_target_contexts": len(nonempty),
                "n_unique_genes": len(unique),
                "n_gene_context_evaluations": sum(
                    len(genes) for genes in nonempty.values()
                ),
            }
        )

    for threshold in CELL_THRESHOLDS:
        qualified = {
            context: {
                gene
                for gene, n_cells in counts.get(context, {}).items()
                if n_cells >= threshold
            }
            for context in context_ids
        }

        # E3a: ordered within-study source -> target pairs.
        e3a_by_target: dict[str, set[str]] = defaultdict(set)
        e3a_evaluations: dict[str, int] = defaultdict(int)
        e3a_sources: dict[str, list[str]] = defaultdict(list)
        for study, study_contexts in by_study.items():
            for source, target in itertools.permutations(study_contexts, 2):
                usable = qualified[source] & qualified[target]
                pair_rows.append(
                    {
                        "n_min": threshold,
                        "study": study,
                        "source_dataset_context": source,
                        "target_dataset_context": target,
                        "n_usable_genes": len(usable),
                    }
                )
                e3a_by_target[target].update(usable)
                e3a_evaluations[target] += len(usable)
                e3a_sources[target].append(source)
        unique_e3a = set().union(*e3a_by_target.values()) if e3a_by_target else set()
        summary_rows.append(
            {
                "benchmark": "E3a",
                "n_min": threshold,
                "n_target_contexts": sum(
                    e3a_evaluations[target] > 0 for target in context_ids
                ),
                "n_unique_genes": len(unique_e3a),
                "n_gene_context_evaluations": sum(e3a_evaluations.values()),
            }
        )
        for target in context_ids:
            if len(by_study[context_meta[target]["study_id"]]) < 2:
                continue
            target_rows.append(
                {
                    "benchmark": "E3a",
                    "n_min": threshold,
                    "target_dataset_context": target,
                    "permitted_source_contexts": ";".join(e3a_sources[target]),
                    "n_source_contexts_with_qualified_observations": sum(
                        bool(qualified[source]) for source in e3a_sources[target]
                    ),
                    "n_unique_genes": len(e3a_by_target[target]),
                    "n_gene_context_evaluations": e3a_evaluations[target],
                    "notes": "Evaluation count sums ordered source-target pairs; unique genes are deduplicated within the target.",
                }
            )

        # E3b: target qualified and present in at least one qualified same-study source.
        e3b_usable: dict[str, set[str]] = {}
        for study, study_contexts in by_study.items():
            if len(study_contexts) < 2:
                continue
            for target in study_contexts:
                sources = [context for context in study_contexts if context != target]
                source_union = set().union(*(qualified[source] for source in sources))
                usable = qualified[target] & source_union
                e3b_usable[target] = usable
                target_rows.append(
                    {
                        "benchmark": "E3b",
                        "n_min": threshold,
                        "target_dataset_context": target,
                        "permitted_source_contexts": ";".join(sources),
                        "n_source_contexts_with_qualified_observations": sum(
                            bool(qualified[source]) for source in sources
                        ),
                        "n_unique_genes": len(usable),
                        "n_gene_context_evaluations": len(usable),
                        "notes": "Target qualified and present in at least one qualified same-study source.",
                    }
                )
                for gene in sorted(target_genes[target]):
                    k_g = sum(gene in qualified[source] for source in sources)
                    exposure_rows.append(
                        {
                            "benchmark": "E3b",
                            "n_min": threshold,
                            "target_dataset_context": target,
                            "target_gene": gene,
                            "target_n_cells": counts.get(target, {}).get(gene),
                            "target_qualified": gene in qualified[target],
                            "k_g_qualified": k_g,
                            "usable": gene in qualified[target] and k_g >= 1,
                        }
                    )
        add_summary("E3b", threshold, e3b_usable)

        # E4: Replogle K562 anchor; canonical K562 targets are excluded.
        anchor = "replogle2022::K562"
        e4_usable: dict[str, set[str]] = {}
        for target in context_ids:
            if target == anchor:
                continue
            if context_meta[target]["study_id"] == context_meta[anchor]["study_id"]:
                continue
            if context_meta[target]["canonical_context_id"] == "K562":
                continue
            usable = qualified[anchor] & qualified[target]
            e4_usable[target] = usable
            target_rows.append(
                {
                    "benchmark": "E4",
                    "n_min": threshold,
                    "target_dataset_context": target,
                    "permitted_source_contexts": anchor,
                    "n_source_contexts_with_qualified_observations": int(
                        bool(qualified[anchor])
                    ),
                    "n_unique_genes": len(usable),
                    "n_gene_context_evaluations": len(usable),
                    "notes": "Canonical K562 targets excluded from the primary context-transfer result.",
                }
            )
        add_summary("E4", threshold, e4_usable)

        # E5: target qualified and present in at least one qualified source outside
        # the held-out canonical context.
        e5_usable: dict[str, set[str]] = {}
        for target in context_ids:
            target_canonical = context_meta[target]["canonical_context_id"]
            sources = [
                context
                for context in context_ids
                if context_meta[context]["canonical_context_id"] != target_canonical
            ]
            source_union = set().union(*(qualified[source] for source in sources))
            usable = qualified[target] & source_union
            e5_usable[target] = usable
            target_rows.append(
                {
                    "benchmark": "E5",
                    "n_min": threshold,
                    "target_dataset_context": target,
                    "permitted_source_contexts": ";".join(sources),
                    "n_source_contexts_with_qualified_observations": sum(
                        bool(qualified[source]) for source in sources
                    ),
                    "n_unique_genes": len(usable),
                    "n_gene_context_evaluations": len(usable),
                    "notes": "Sources sharing the held-out canonical context are excluded.",
                }
            )
            for gene in sorted(target_genes[target]):
                k_g = sum(gene in qualified[source] for source in sources)
                exposure_rows.append(
                    {
                        "benchmark": "E5",
                        "n_min": threshold,
                        "target_dataset_context": target,
                        "target_gene": gene,
                        "target_n_cells": counts.get(target, {}).get(gene),
                        "target_qualified": gene in qualified[target],
                        "k_g_qualified": k_g,
                        "usable": gene in qualified[target] and k_g >= 1,
                    }
                )
        add_summary("E5", threshold, e5_usable)

    return summary_rows, target_rows, pair_rows, exposure_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("metadata/datasets/perturbation_atlas_sources.yaml"),
    )
    parser.add_argument(
        "--contexts",
        type=Path,
        default=Path("metadata/datasets/dataset_contexts.csv"),
    )
    parser.add_argument(
        "--vcc-targets", type=Path, default=Path("data/controls/pert_counts.csv")
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("reports/benchmark-capacity")
    )
    args = parser.parse_args()

    datasets_dir = args.config.parent
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    contexts = resolve_contexts(config["contexts"])
    context_ids = [context["dataset_context_id"] for context in contexts]
    context_meta = {context["dataset_context_id"]: context for context in contexts}
    registry_rows = read_csv(args.contexts)
    registry = {row["dataset_context_id"]: row for row in registry_rows}
    if not set(context_ids) <= set(registry):
        raise ValueError("Atlas context is missing from dataset_contexts.csv")
    vcc_targets = {
        normalize_gene(row["target_gene"]) for row in read_csv(args.vcc_targets)
    }
    excluded = {label.lower() for label in config["excluded_control_labels"]}
    observations = build_observations(contexts, datasets_dir, vcc_targets, excluded)

    observation_csv = datasets_dir / "perturbation_observations.csv"
    write_csv(observation_csv, OBSERVATION_FIELDS, observations)
    table = pa.Table.from_pylist(observations)
    pq.write_table(
        table,
        datasets_dir / "perturbation_observations.parquet",
        compression="zstd",
    )

    genes = sorted({row["target_gene"] for row in observations})
    context_sets = make_sets(observations, "dataset_context_id")
    canonical_sets = make_sets(observations, "canonical_context_id")
    canonical_ids = list(
        dict.fromkeys(context["canonical_context_id"] for context in contexts)
    )
    write_csv(
        datasets_dir / "gene_x_dataset_context.csv",
        ["target_gene", *context_ids],
        matrix_rows(genes, context_ids, context_sets),
    )
    write_csv(
        datasets_dir / "gene_x_canonical_context.csv",
        ["target_gene", *canonical_ids],
        matrix_rows(genes, canonical_ids, canonical_sets),
    )

    registry_datasets = list(dict.fromkeys(row["dataset_id"] for row in registry_rows))
    registry_canonical = list(
        dict.fromkeys(row["canonical_context_id"] for row in registry_rows)
    )
    registry_pairs = {
        (row["canonical_context_id"], row["dataset_id"]) for row in registry_rows
    }
    context_dataset_rows = [
        {"canonical_context_id": canonical}
        | {
            dataset: int((canonical, dataset) in registry_pairs)
            for dataset in registry_datasets
        }
        for canonical in registry_canonical
    ]
    write_csv(
        datasets_dir / "context_x_dataset.csv",
        ["canonical_context_id", *registry_datasets],
        context_dataset_rows,
    )

    overlap_rows = [
        {"dataset_context_id": source}
        | {
            target: len(context_sets[source] & context_sets[target])
            for target in context_ids
        }
        for source in context_ids
    ]
    write_csv(
        datasets_dir / "dataset_context_overlap.csv",
        ["dataset_context_id", *context_ids],
        overlap_rows,
    )
    edges = build_edges(context_ids, context_sets, context_meta)
    write_csv(datasets_dir / "bridge_edges.csv", list(edges[0]), edges)

    summary = summarize_genes(set(genes), observations, vcc_targets)
    write_csv(
        datasets_dir / "perturbation_context_summary.csv", list(summary[0]), summary
    )
    threshold_rows = []
    for field, label, thresholds in [
        ("n_canonical_contexts", "canonical_contexts", [1, 2, 3, 4]),
        ("n_dataset_contexts", "dataset_contexts", [2, 3, 4]),
        ("n_studies", "independent_studies", [2, 3]),
    ]:
        for threshold in thresholds:
            threshold_rows.append(
                {
                    "measure": label,
                    "threshold": threshold,
                    "n_genes": sum(
                        int(row[field]) >= threshold
                        for row in summary
                        if int(row["n_dataset_contexts"]) > 0
                    ),
                    "notes": "Nominal exact-symbol coverage; no perturbation-quality filters.",
                }
            )
    write_csv(
        datasets_dir / "perturbation_context_thresholds.csv",
        list(threshold_rows[0]),
        threshold_rows,
    )

    e0 = build_e0(context_ids, observations, context_meta, registry)
    e1 = build_e1(context_ids, context_sets, context_meta, registry)
    e2, e2_usable = build_e2(context_ids, context_sets, context_meta)
    e3a, e3a_usable = build_e3a(context_ids, context_sets, context_meta)
    e3b, e3b_usable = build_e3b(context_ids, context_sets, context_meta)
    e4, e4_usable = build_e4(
        "replogle2022::K562", context_ids, context_sets, context_meta, vcc_targets
    )
    e5, e5_usable, e5_exposure = build_e5(
        "E5_canonical_context_holdout", context_ids, context_sets, context_meta
    )
    e5_ablation, e5_ablation_usable, e5_ablation_exposure = build_e5(
        "E5_ablation_experimental_context_holdout",
        context_ids,
        context_sets,
        context_meta,
    )

    write_csv(args.report_dir / "E0_sampling_capacity.csv", list(e0[0]), e0)
    write_csv(args.report_dir / "E1_distribution_capacity.csv", list(e1[0]), e1)
    write_csv(args.report_dir / "E2_same_context_pairs.csv", list(e2[0]), e2)
    write_csv(args.report_dir / "E3a_pairwise_within_study.csv", list(e3a[0]), e3a)
    write_csv(
        args.report_dir / "E3b_multisource_within_study_holdouts.csv",
        list(e3b[0]),
        e3b,
    )
    write_csv(args.report_dir / "E4_replogle_targets.csv", list(e4[0]), e4)
    write_csv(
        args.report_dir / "E5_multisource_holdouts.csv",
        list(e5[0]),
        e5 + e5_ablation,
    )
    exposure = e5_exposure + e5_ablation_exposure
    write_csv(datasets_dir / "benchmark_gene_exposure.csv", list(exposure[0]), exposure)

    e0_usable = {
        row["dataset_context_id"]: {
            obs["target_gene"]
            for obs in observations
            if obs["dataset_context_id"] == row["dataset_context_id"]
            and obs["n_cells"] is not None
            and obs["n_cells"] >= 2
        }
        for row in e0
        if row["n_with_known_cell_count"] > 0
    }
    e1_usable = {
        row["dataset_context_id"]: context_sets[row["dataset_context_id"]]
        for row in e1
        if row["nominally_eligible"]
    }
    e2_context_usable = {
        row["target_dataset_context"]: e2_usable[
            f"{row['source_dataset_context']} -> {row['target_dataset_context']}"
        ]
        for row in e2
    }
    e2_capacity = capacity_row(
        "E2",
        list(e2_context_usable),
        e2_context_usable,
        context_meta,
        f"{len(e2)} same-canonical cross-study pair(s); each pair counted once.",
    )
    e2_capacity["n_studies"] = len(
        {study for row in e2 for study in (row["source_study"], row["target_study"])}
    )
    capacity = [
        capacity_row(
            "E0",
            list(e0_usable),
            e0_usable,
            context_meta,
            "Quantified only where per-target cell counts are available; >=2 cells is the minimal split, not a quality threshold.",
        ),
        capacity_row(
            "E1",
            list(e1_usable),
            e1_usable,
            context_meta,
            "Nominal raw-count + controls capacity; no minimum-cell or response-quality rule.",
        ),
        e2_capacity,
        pair_capacity_row(
            "E3a",
            e3a,
            e3a_usable,
            context_meta,
            "Each ordered within-study source-target pair is a distinct evaluation unit; eligibility is target ∩ source.",
        ),
        capacity_row(
            "E3b",
            list(e3b_usable),
            e3b_usable,
            context_meta,
            "Main target ∩ union(source contexts) eligibility; all-source intersection retained as a stricter subset.",
        ),
        capacity_row(
            "E4",
            list(e4_usable),
            e4_usable,
            context_meta,
            "Replogle K562 anchor; cross-study targets only.",
        ),
        capacity_row(
            "E5",
            list(e5_usable),
            e5_usable,
            context_meta,
            "Hold out every dataset-context sharing the target canonical context.",
        ),
        capacity_row(
            "E5_ablation",
            list(e5_ablation_usable),
            e5_ablation_usable,
            context_meta,
            "Experimental-context-only holdout; not a separate main ladder rung.",
        ),
    ]
    next(row for row in capacity if row["benchmark_id"] == "E4")["n_studies"] += 1
    write_csv(args.report_dir / "capacity_summary.csv", list(capacity[0]), capacity)

    cell_summary, cell_by_target, cell_pairs, cell_exposure = (
        build_cell_threshold_capacity(context_ids, observations, context_meta)
    )
    write_csv(
        args.report_dir / "cell_count_attrition_summary.csv",
        list(cell_summary[0]),
        cell_summary,
    )
    write_csv(
        args.report_dir / "cell_count_attrition_by_target_context.csv",
        list(cell_by_target[0]),
        cell_by_target,
    )
    write_csv(
        args.report_dir / "E3a_pairwise_cell_thresholds.csv",
        list(cell_pairs[0]),
        cell_pairs,
    )
    write_csv(
        args.report_dir / "cell_count_qualified_exposure.csv",
        list(cell_exposure[0]),
        cell_exposure,
    )

    anchor = "replogle2022::K562"
    other_contexts = [context for context in context_ids if context != anchor]
    anchor_counts = {
        gene: sum(gene in context_sets[context] for context in other_contexts)
        for gene in context_sets[anchor]
    }
    anchor_studies = {
        gene: {
            context_meta[context]["study_id"]
            for context in other_contexts
            if gene in context_sets[context]
        }
        for gene in context_sets[anchor]
    }
    anchor_summary = [
        {
            "metric": "total_replogle_k562_perturbations",
            "value": len(context_sets[anchor]),
            "notes": "Union of genome-wide and essential panels; duplicate genes counted once.",
        },
        {
            "metric": "shared_with_at_least_one_other_dataset_context",
            "value": sum(count >= 1 for count in anchor_counts.values()),
            "notes": "Includes Replogle RPE1.",
        },
        {
            "metric": "shared_with_at_least_two_other_dataset_contexts",
            "value": sum(count >= 2 for count in anchor_counts.values()),
            "notes": "Nominal target-list overlap.",
        },
        {
            "metric": "shared_with_at_least_three_other_dataset_contexts",
            "value": sum(count >= 3 for count in anchor_counts.values()),
            "notes": "Nominal target-list overlap.",
        },
        {
            "metric": "shared_with_at_least_one_independent_study",
            "value": sum(
                bool(studies - {"replogle2022"}) for studies in anchor_studies.values()
            ),
            "notes": "Excludes same-study-only RPE1 evidence.",
        },
        {
            "metric": "shared_with_multiple_independent_studies",
            "value": sum(
                len(studies - {"replogle2022"}) >= 2
                for studies in anchor_studies.values()
            ),
            "notes": "Observed outside Replogle in at least two study IDs.",
        },
        {
            "metric": "shared_with_vcc2025_h1",
            "value": len(context_sets[anchor] & context_sets["vcc2025_h1::H1"]),
            "notes": "Direct perturbation-gene overlap.",
        },
        {
            "metric": "in_vcc2026_300_panel",
            "value": len(context_sets[anchor] & vcc_targets),
            "notes": "Annotation only; atlas is not restricted to the VCC panel.",
        },
        {
            "metric": "outside_vcc2026_300_panel",
            "value": len(context_sets[anchor] - vcc_targets),
            "notes": "Full public perturbation universe.",
        },
    ]
    write_csv(
        args.report_dir / "replogle_anchor_summary.csv",
        list(anchor_summary[0]),
        anchor_summary,
    )

    print(
        f"built atlas: {len(observations)} observations, {len(genes)} genes, "
        f"{len(context_ids)} dataset-contexts, {len(canonical_ids)} canonical contexts"
    )


if __name__ == "__main__":
    main()
