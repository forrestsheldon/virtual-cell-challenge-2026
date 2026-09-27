"""Promoter-aware calibration for the context-transfer analysis.

The first calibration treated every construct assigned to the same gene as a
replicate.  This revision separates exact-construct reproducibility from
same-promoter and cross-promoter construct generalization.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

from scripts.evaluation.context_transfer_blog import (
    CONTEXTS,
    N_BOOTSTRAPS,
    PRIMARY_PAIRS,
    REPORT,
    ROOT,
    SEED,
    collapse_columns,
    load_context,
    normalized_effects,
    sha256,
    single_metrics,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt

PROMOTER_PATTERN = re.compile(r"(?:-|_)(P1P2|P1|P2)(?:-|\||$)")
REPRESENTATIONS = ("epsilon_lfc", "log2_cpm1p")
BATCH_SIZE = 128


@dataclass
class GuideData:
    context: str
    genes: list[str]
    guide_ids: np.ndarray
    targets: np.ndarray
    promoter_classes: np.ndarray
    cells: np.ndarray
    epsilon_lfc: np.ndarray
    log2_cpm1p: np.ndarray


def promoter_class(guide_id: str) -> str | None:
    """Return a promoter-design class, or None when the label is uninformative."""
    matches = sorted(set(PROMOTER_PATTERN.findall(str(guide_id))))
    return "+".join(matches) if matches else None


def load_guides(context_name: str) -> GuideData:
    spec = CONTEXTS[context_name]
    context = load_context(spec)
    data = ad.read_h5ad(spec.guide_path, backed="r")
    try:
        labels = data.obs["gene_target"].astype(str).to_numpy()
        selected = np.flatnonzero(labels != spec.control)
        counts, genes = collapse_columns(
            data.X[selected], data.var["gene_name"].astype(str).tolist()
        )
        if genes != context.genes:
            raise ValueError(f"{context_name}: guide and target gene axes differ")
        guide_ids = data.obs["guide_target"].astype(str).to_numpy()[selected]
        if len(np.unique(guide_ids)) != len(guide_ids):
            raise ValueError(f"{context_name}: guide_target labels are not unique")
        cells = data.obs["n_cells"].to_numpy(dtype=int)[selected]
        targets = labels[selected]
    finally:
        data.file.close()
    epsilon_lfc, log2_cpm1p, _ = normalized_effects(counts, context.control_counts)
    return GuideData(
        context=context_name,
        genes=genes,
        guide_ids=guide_ids,
        targets=targets,
        promoter_classes=np.asarray([promoter_class(value) for value in guide_ids]),
        cells=cells,
        epsilon_lfc=epsilon_lfc.astype(np.float32),
        log2_cpm1p=log2_cpm1p.astype(np.float32),
    )


def same_promoter_records(data: GuideData) -> list[dict[str, object]]:
    records = []
    frame = pd.DataFrame(
        {
            "row": np.arange(len(data.guide_ids)),
            "target_gene": data.targets,
            "promoter_class": data.promoter_classes,
        }
    ).dropna(subset=["promoter_class"])
    for (target, promoter), group in frame.groupby(
        ["target_gene", "promoter_class"], sort=True
    ):
        rows = group.row.to_numpy(dtype=int)
        for prediction in rows:
            for truth in rows:
                if prediction == truth:
                    continue
                records.append(
                    {
                        "calibration": "within_context_same_promoter_different_construct",
                        "prediction_index": prediction,
                        "correct_index": truth,
                        "target_gene": target,
                        "prediction_guide": data.guide_ids[prediction],
                        "truth_guide": data.guide_ids[truth],
                        "promoter_relation": "same",
                        "promoter_class": promoter,
                    }
                )
    return records


def cross_context_records(
    source: GuideData, target: GuideData
) -> list[dict[str, object]]:
    source_lookup = {guide: index for index, guide in enumerate(source.guide_ids)}
    target_lookup = {guide: index for index, guide in enumerate(target.guide_ids)}
    shared_guides = sorted(set(source_lookup) & set(target_lookup))
    records = []
    by_target: dict[str, list[str]] = {}
    for guide in shared_guides:
        source_index = source_lookup[guide]
        target_index = target_lookup[guide]
        if source.targets[source_index] != target.targets[target_index]:
            raise ValueError(f"{guide}: target annotation differs between contexts")
        gene = source.targets[source_index]
        by_target.setdefault(gene, []).append(guide)
        records.append(
            {
                "calibration": "cross_context_exact_construct",
                "prediction_index": source_index,
                "correct_index": target_index,
                "target_gene": gene,
                "prediction_guide": guide,
                "truth_guide": guide,
                "promoter_relation": "exact",
                "promoter_class": source.promoter_classes[source_index],
            }
        )
    for gene, guides in sorted(by_target.items()):
        for prediction_guide in guides:
            for truth_guide in guides:
                if prediction_guide == truth_guide:
                    continue
                source_index = source_lookup[prediction_guide]
                target_index = target_lookup[truth_guide]
                source_promoter = source.promoter_classes[source_index]
                target_promoter = target.promoter_classes[target_index]
                if source_promoter is None or target_promoter is None:
                    continue
                relation = "same" if source_promoter == target_promoter else "different"
                records.append(
                    {
                        "calibration": (
                            "cross_context_same_promoter_different_construct"
                            if relation == "same"
                            else "cross_context_cross_promoter_different_construct"
                        ),
                        "prediction_index": source_index,
                        "correct_index": target_index,
                        "target_gene": gene,
                        "prediction_guide": prediction_guide,
                        "truth_guide": truth_guide,
                        "promoter_relation": relation,
                        "promoter_class": (
                            source_promoter
                            if relation == "same"
                            else f"{source_promoter}->{target_promoter}"
                        ),
                    }
                )
    return records


def evaluate_records(
    source: GuideData,
    target: GuideData,
    records: list[dict[str, object]],
    representation: str,
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    shared_genes = sorted(set(source.genes) & set(target.genes))
    source_lookup = {gene: index for index, gene in enumerate(source.genes)}
    target_lookup = {gene: index for index, gene in enumerate(target.genes)}
    source_columns = np.asarray([source_lookup[gene] for gene in shared_genes])
    target_columns = np.asarray([target_lookup[gene] for gene in shared_genes])
    gene_lookup = {gene: index for index, gene in enumerate(shared_genes)}
    source_effects = getattr(source, representation)[:, source_columns].astype(
        np.float64
    )
    target_effects = getattr(target, representation)[:, target_columns].astype(
        np.float64
    )
    target_norm2 = np.square(target_effects).sum(axis=1)
    rows = []
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        prediction_indices = np.asarray(
            [record["prediction_index"] for record in batch], dtype=int
        )
        predictions = source_effects[prediction_indices]
        dots = predictions @ target_effects.T
        prediction_norm2 = np.square(predictions).sum(axis=1)
        for local, record in enumerate(batch):
            target_gene = str(record["target_gene"])
            excluded = gene_lookup.get(target_gene, -1)
            numerator = dots[local].copy()
            left_norm2 = prediction_norm2[local]
            right_norm2 = target_norm2.copy()
            if excluded >= 0:
                numerator -= predictions[local, excluded] * target_effects[:, excluded]
                left_norm2 -= predictions[local, excluded] ** 2
                right_norm2 -= np.square(target_effects[:, excluded])
            denominator = np.sqrt(
                np.maximum(left_norm2, 0) * np.maximum(right_norm2, 0)
            )
            similarities = np.divide(
                numerator,
                denominator,
                out=np.full(len(target_effects), -np.inf),
                where=denominator > 0,
            )
            correct_index = int(record["correct_index"])
            candidates = (target.targets != target_gene) | (
                np.arange(len(target.targets)) == correct_index
            )
            candidate_scores = similarities[candidates]
            candidate_indices = np.flatnonzero(candidates)
            correct_position = int(
                np.flatnonzero(candidate_indices == correct_index)[0]
            )
            correct = candidate_scores[correct_position]
            greater = int(np.sum(candidate_scores > correct))
            equal = int(
                np.sum(np.isclose(candidate_scores, correct, rtol=1e-12, atol=1e-12))
            )
            zero_rank = greater + 0.5 * (equal - 1)
            retrieval = (
                1 - zero_rank / (len(candidate_scores) - 1)
                if len(candidate_scores) > 1 and np.isfinite(correct)
                else 0.5
            )
            wrong = candidate_scores[
                np.arange(len(candidate_scores)) != correct_position
            ]
            wrong = wrong[np.isfinite(wrong)]
            metrics = single_metrics(
                predictions[local],
                target_effects[correct_index],
                excluded,
                100,
            )
            rows.append(
                {
                    **record,
                    "source_context": source.context,
                    "target_context": target.context,
                    "representation": representation,
                    "shared_genes": len(shared_genes),
                    "candidate_wrong_constructs": len(candidate_scores) - 1,
                    "prediction_cells": int(source.cells[prediction_indices[local]]),
                    "truth_cells": int(target.cells[correct_index]),
                    "correct_cosine": correct,
                    "median_wrong_cosine": np.median(wrong) if len(wrong) else np.nan,
                    "cosine_margin_over_median_wrong": (
                        correct - np.median(wrong) if len(wrong) else np.nan
                    ),
                    "retrieval_score": retrieval,
                    "signed_recovery_100": metrics["signed_recovery"],
                    "normalized_absolute_error": metrics["nae"],
                }
            )
    return pd.DataFrame(rows)


def bootstrap_summary(
    per_comparison: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    metrics = (
        "correct_cosine",
        "cosine_margin_over_median_wrong",
        "retrieval_score",
        "signed_recovery_100",
        "normalized_absolute_error",
    )
    groups = [
        "calibration",
        "source_context",
        "target_context",
        "representation",
    ]
    rows = []
    for keys, frame in per_comparison.groupby(groups, sort=False):
        target_means = frame.groupby("target_gene")[list(metrics)].mean()
        for metric in metrics:
            values = target_means[metric].to_numpy(dtype=float)
            valid = np.isfinite(values)
            values = values[valid]
            if len(values):
                indices = rng.integers(0, len(values), size=(N_BOOTSTRAPS, len(values)))
                sampled = values[indices].mean(axis=1)
                q025, q975 = np.quantile(sampled, (0.025, 0.975))
                estimate = values.mean()
            else:
                estimate = q025 = q975 = np.nan
            rows.append(
                {
                    **dict(zip(groups, keys, strict=True)),
                    "metric": metric,
                    "comparisons": len(frame),
                    "targets": len(target_means),
                    "estimate": estimate,
                    "q025": q025,
                    "q975": q975,
                }
            )
    return pd.DataFrame(rows)


def availability_table(
    within: list[pd.DataFrame], cross: list[pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    within_frame = pd.concat(within, ignore_index=True) if within else pd.DataFrame()
    cross_frame = pd.concat(cross, ignore_index=True) if cross else pd.DataFrame()
    for context in CONTEXTS:
        selected = (
            within_frame[within_frame.source_context == context]
            if len(within_frame)
            else pd.DataFrame()
        )
        rows.append(
            {
                "calibration": "within_context_same_promoter_different_construct",
                "source_context": context,
                "target_context": context,
                "status": "available" if len(selected) else "unavailable",
                "targets": selected.target_gene.nunique() if len(selected) else 0,
                "reason": (
                    "distinct constructs share a promoter-design class"
                    if len(selected)
                    else "multi-construct targets do not contain labeled same-promoter designs"
                ),
            }
        )
        rows.append(
            {
                "calibration": "within_context_same_construct_split",
                "source_context": context,
                "target_context": context,
                "status": "unavailable",
                "targets": 0,
                "reason": "audited compact artifacts pool all cells for each construct and contain no guide-by-batch split",
            }
        )
    for source_name, target_name in PRIMARY_PAIRS:
        for calibration in (
            "cross_context_exact_construct",
            "cross_context_same_promoter_different_construct",
            "cross_context_cross_promoter_different_construct",
        ):
            selected = cross_frame[
                (cross_frame.source_context == source_name)
                & (cross_frame.target_context == target_name)
                & (cross_frame.calibration == calibration)
            ]
            rows.append(
                {
                    "calibration": calibration,
                    "source_context": source_name,
                    "target_context": target_name,
                    "status": "available" if len(selected) else "unavailable",
                    "targets": selected.target_gene.nunique() if len(selected) else 0,
                    "reason": (
                        "audited guide constructs overlap"
                        if len(selected)
                        else "no qualifying shared construct pairs"
                    ),
                }
            )
    return pd.DataFrame(rows)


def checkpoint_review(summary: pd.DataFrame, availability: pd.DataFrame) -> dict:
    primary = summary[
        summary.apply(
            lambda row: (row.source_context, row.target_context) in PRIMARY_PAIRS,
            axis=1,
        )
    ]

    def retrieval_rows(calibration: str, representation: str) -> pd.DataFrame:
        return primary[
            (primary.calibration == calibration)
            & (primary.representation == representation)
            & (primary.metric == "retrieval_score")
        ]

    exact = retrieval_rows("cross_context_exact_construct", "log2_cpm1p")
    gene_level = retrieval_rows(
        "cross_context_same_promoter_different_construct", "log2_cpm1p"
    )
    exact_status = {
        f"{row.source_context}->{row.target_context}": bool(row.q025 > 0.5)
        for row in exact.itertuples(index=False)
    }
    gene_status = {
        f"{row.source_context}->{row.target_context}": bool(row.q025 > 0.5)
        for row in gene_level.itertuples(index=False)
    }
    unavailable_gene_edges = []
    for source_name, target_name in PRIMARY_PAIRS:
        edge = f"{source_name}->{target_name}"
        if edge not in gene_status:
            unavailable_gene_edges.append(edge)
    return {
        "revision": "promoter-aware calibration v2",
        "supersedes_interpretation": (
            "The original mixed-promoter leave-one-guide-out result cannot be used "
            "as a replicate-positive calibration."
        ),
        "same_construct_split_ceiling": {
            "status": "unavailable",
            "reason": "no guide-by-batch or guide-by-cell split in compact artifacts",
        },
        "exact_construct_transfer": {
            "interpretation": "construct reproducibility, including possible guide-specific effects",
            "log2_cpm1p_retrieval_lower_ci_above_chance": exact_status,
        },
        "same_promoter_cross_construct_transfer": {
            "interpretation": "best available gene-level calibration independent of exact construct identity",
            "log2_cpm1p_retrieval_lower_ci_above_chance": gene_status,
            "unavailable_edges": unavailable_gene_edges,
        },
        "direct_transfer_checkpoint": {
            "status": "partially_identified",
            "decision": (
                "Separate construct identity from gene-level transfer. Treat exact-construct "
                "retrieval as construct reproducibility; make gene-level claims only on edges "
                "with calibrated same-promoter cross-construct evidence."
            ),
        },
        "availability_rows": len(availability),
    }


def plot_calibration(summary: pd.DataFrame) -> list[Path]:
    retrieval = summary[
        (summary.metric == "retrieval_score") & (summary.representation == "log2_cpm1p")
    ]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    within = retrieval[
        retrieval.calibration == "within_context_same_promoter_different_construct"
    ]
    for index, row in enumerate(within.itertuples(index=False)):
        axes[0].errorbar(
            row.estimate,
            index,
            xerr=[[row.estimate - row.q025], [row.q975 - row.estimate]],
            fmt="o",
            capsize=3,
            color="#2563eb",
        )
    axes[0].set_yticks(
        range(len(within)),
        [row.source_context for row in within.itertuples(index=False)],
    )
    axes[0].set_title("Within-context, same promoter\ndifferent constructs")
    axes[0].set_xlabel("Correct-target retrieval")

    labels = {
        "cross_context_exact_construct": "exact",
        "cross_context_same_promoter_different_construct": "same promoter",
        "cross_context_cross_promoter_different_construct": "cross promoter",
    }
    primary = retrieval[
        retrieval.apply(
            lambda row: (row.source_context, row.target_context) in PRIMARY_PAIRS,
            axis=1,
        )
    ]
    primary = primary[primary.calibration.isin(labels)]
    primary = primary.sort_values(
        ["source_context", "calibration"], kind="stable"
    ).reset_index(drop=True)
    colors = {
        "cross_context_exact_construct": "#7c3aed",
        "cross_context_same_promoter_different_construct": "#059669",
        "cross_context_cross_promoter_different_construct": "#dc2626",
    }
    for index, row in enumerate(primary.itertuples(index=False)):
        axes[1].errorbar(
            row.estimate,
            index,
            xerr=[[row.estimate - row.q025], [row.q975 - row.estimate]],
            fmt="o",
            capsize=3,
            color=colors[row.calibration],
        )
    axes[1].set_yticks(
        range(len(primary)),
        [
            f"{row.source_context}→{row.target_context} · {labels[row.calibration]}"
            for row in primary.itertuples(index=False)
        ],
    )
    axes[1].set_title("Cross-context construct tests")
    axes[1].set_xlabel("Correct-target retrieval")
    for axis in axes:
        axis.axvline(0.5, color="black", lw=0.8, alpha=0.6)
        axis.set_xlim(0.35, 0.85)
    png = REPORT / "metric_calibration_v2.png"
    svg = REPORT / "metric_calibration_v2.svg"
    figure.savefig(png, dpi=180)
    figure.savefig(svg)
    plt.close(figure)
    return [png, svg]


def update_manifest(outputs: list[Path], checkpoint: dict) -> None:
    path = REPORT / "manifest.json"
    manifest = json.loads(path.read_text())
    producer = Path(__file__).resolve()
    manifest["calibration_v2"] = {
        "producer": str(producer.relative_to(ROOT)),
        "producer_sha256": sha256(producer),
        "design": "promoter-aware construct calibration with target-level bootstrap",
        "representations": list(REPRESENTATIONS),
        "checkpoint": checkpoint,
    }
    manifest["current_checkpoint_review"] = str(
        (REPORT / "checkpoint_reviews_v2.json").relative_to(ROOT)
    )
    for output in outputs:
        manifest["outputs"][str(output.relative_to(ROOT))] = sha256(output)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    rng = np.random.default_rng(SEED + 2)
    within_frames = []
    cross_frames = []
    completed_within = set()
    for source_name, target_name in PRIMARY_PAIRS:
        source = load_guides(source_name)
        target = load_guides(target_name)
        for data in (source, target):
            if data.context in completed_within:
                continue
            records = same_promoter_records(data)
            for representation in REPRESENTATIONS:
                frame = evaluate_records(data, data, records, representation)
                if len(frame):
                    within_frames.append(frame)
            completed_within.add(data.context)
        for first, second, direction in (
            (source, target, "primary"),
            (target, source, "reverse_sensitivity"),
        ):
            records = cross_context_records(first, second)
            for representation in REPRESENTATIONS:
                frame = evaluate_records(first, second, records, representation)
                if len(frame):
                    frame.insert(0, "direction", direction)
                    cross_frames.append(frame)
        print(f"Calibrated {source_name}<->{target_name}", flush=True)

    within = pd.concat(within_frames, ignore_index=True)
    within.insert(0, "direction", "within_context")
    cross = pd.concat(cross_frames, ignore_index=True)
    per_comparison = pd.concat([within, cross], ignore_index=True)
    summary = bootstrap_summary(per_comparison, rng)
    availability = availability_table(within_frames, cross_frames)
    checkpoint = checkpoint_review(summary, availability)
    tables = {
        "metric_calibration_v2_per_comparison.csv": per_comparison,
        "metric_calibration_v2_summary.csv": summary,
        "metric_calibration_v2_availability.csv": availability,
    }
    outputs = []
    for name, frame in tables.items():
        path = REPORT / name
        frame.to_csv(path, index=False)
        outputs.append(path)
    checkpoint_path = REPORT / "checkpoint_reviews_v2.json"
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")
    outputs.append(checkpoint_path)
    outputs.extend(plot_calibration(summary))
    update_manifest(outputs, checkpoint)
    print(json.dumps(checkpoint, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
