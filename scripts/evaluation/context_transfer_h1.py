"""Evaluate prespecified cross-study source-to-H1 context-transfer anchors.

These anchors are descriptive and are not treated as additional context replicates.
The fast profile calculation reuses the frozen canonical-control prediction contract;
it deliberately does not invoke the full VCC scorer.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse

from scripts.cloud.audit_compact_pseudobulks import audit as audit_compact
from scripts.evaluation.context_transfer_blog import (
    DERIVED,
    EPSILON,
    REPORT,
    SEED,
    ContextData,
    ContextSpec,
    bootstrap_coefficients,
    bootstrap_differences,
    crossfit_predictions,
    deterministic_folds,
    evaluate_models,
    normalized_effects,
    sha256,
    summarize_edge,
)
from scripts.evaluation.prepare_paired_linear_response_h1 import canonical_source_rows
from scripts.evaluation.prepare_replogle_h1_phase1 import truth_arrays
from scripts.evaluation.profile_h1 import pds
from scripts.linear_response.kernel import log_pseudobulk
from scripts.linear_response.strong_de_recovery import signed_topk

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
H1_AGGREGATION = ROOT / "data/derived/replogle_h1_transfer/effects.npz"
H1_PHASE1 = ROOT / "data/derived/replogle_h1_transfer/phase1_effects.npz"
H1_CONTROLS = ROOT / "data/derived/vcc2026_h1/h1_controls.h5ad"
REPLOGLE_RAW = ROOT / "data/external/replogle2022/K562_gwps_raw_bulk_01.h5ad"
MIN_STRONG = 10
MODELS = ("unchanged", "direct", "global_median", "context_interaction")

SOURCE_SPECS = {
    "CD4T": ContextSpec(
        "CD4T",
        "Norman/Nadig cross-study anchor",
        "10x Perturb-seq CRISPRi",
        ROOT / "data/derived/context_atlas/final/CD4T",
        "NTC",
    ),
    "HCT116": ContextSpec(
        "HCT116",
        "X-Atlas/Orion cross-study anchor",
        "FiCS 10x GEM-X 5-prime CRISPRi",
        ROOT / "data/derived/xatlas_orion/final/HCT116",
        "Non-Targeting",
    ),
    "HEK293T": ContextSpec(
        "HEK293T",
        "X-Atlas/Orion cross-study anchor",
        "FiCS 10x GEM-X 5-prime CRISPRi",
        ROOT / "data/derived/xatlas_orion/final/HEK293T",
        "Non-Targeting",
    ),
    "KOLF2.1J": ContextSpec(
        "KOLF2.1J",
        "OpenProblems single-cell perturbations cross-study anchor",
        "10x Perturb-seq CRISPRi",
        ROOT / "data/derived/context_atlas/final/KOLF2.1J",
        "NTC",
    ),
}


def expected_lfc_decoded_sum(raw: sparse.spmatrix, lfc: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of the frozen expected multiplicative decoder."""
    raw = sparse.csr_matrix(raw, dtype=np.float64)
    weighted = raw.multiply(np.exp2(np.asarray(lfc, dtype=np.float64)))
    denominators = np.asarray(weighted.sum(axis=1)).ravel()
    totals = np.asarray(raw.sum(axis=1)).ravel()
    return np.asarray(weighted.T @ (totals / denominators)).ravel()


def compact_source(spec: ContextSpec) -> ContextData:
    from scripts.evaluation.context_transfer_blog import load_context

    return load_context(spec)


def h1_context() -> ContextData:
    with np.load(H1_AGGREGATION, allow_pickle=False) as saved:
        targets = saved["target_gene"].astype(str).tolist()
        counts = saved["h1_count_sum"].astype(np.int64)
        control = saved["h1_control_count_sum"].astype(np.float64)
    with np.load(H1_PHASE1, allow_pickle=False) as saved:
        genes = saved["output_gene"].astype(str).tolist()
    effects, effects_log1p, control_log1p = normalized_effects(counts, control)
    return ContextData(
        ContextSpec(
            "H1", "VCC 2025 H1", "10x CRISPRi", H1_AGGREGATION.parent, "non-targeting"
        ),
        genes,
        targets,
        sparse.csr_matrix(counts),
        control,
        effects,
        effects_log1p,
        control_log1p,
        np.full(len(targets), 400, dtype=int),
        counts.sum(axis=1),
    )


def replogle_gwps_context() -> ContextData:
    with np.load(H1_AGGREGATION, allow_pickle=False) as saved:
        targets = saved["matched_target_gene"].astype(str).tolist()
        counts = saved["replogle_count_sum"].astype(np.int64)
        control = saved["replogle_control_count_sum"].astype(np.float64)
    raw = ad.read_h5ad(REPLOGLE_RAW, backed="r")
    try:
        symbols = raw.var["gene_name"].astype(str).tolist()
    finally:
        raw.file.close()
    codes, unique = pd.factorize(np.asarray(symbols), sort=False)
    if len(unique) != counts.shape[1] or codes.max() + 1 != counts.shape[1]:
        raise ValueError("Replogle collapsed count and symbol axes differ")
    genes = unique.astype(str).tolist()
    effects, effects_log1p, control_log1p = normalized_effects(counts, control)
    spec = ContextSpec(
        "K562_GWPS",
        "Replogle 2022 cross-study anchor",
        "10x Perturb-seq CRISPRi",
        REPLOGLE_RAW.parent,
        "non-targeting",
    )
    return ContextData(
        spec,
        genes,
        targets,
        sparse.csr_matrix(counts),
        control,
        effects,
        effects_log1p,
        control_log1p,
        np.full(len(targets), -1, dtype=int),
        counts.sum(axis=1),
    )


def anchor_arrays(source: ContextData, h1: ContextData) -> dict[str, object]:
    targets = sorted(set(source.targets) & set(h1.targets))
    genes = sorted(set(source.genes) & set(h1.genes))
    source_target = {name: index for index, name in enumerate(source.targets)}
    h1_target = {name: index for index, name in enumerate(h1.targets)}
    source_gene = {name: index for index, name in enumerate(source.genes)}
    h1_gene = {name: index for index, name in enumerate(h1.genes)}
    source_rows = np.asarray([source_target[name] for name in targets])
    h1_rows = np.asarray([h1_target[name] for name in targets])
    source_columns = np.asarray([source_gene[name] for name in genes])
    h1_columns = np.asarray([h1_gene[name] for name in genes])
    return {
        "targets": targets,
        "genes": genes,
        "source_rows": source_rows,
        "h1_rows": h1_rows,
        "h1_columns": h1_columns,
        "source": source.effects[source_rows][:, source_columns],
        "truth": h1.effects[h1_rows][:, h1_columns],
        "difference": h1.control_log1p[h1_columns]
        - source.control_log1p[source_columns],
    }


def fit_anchor(
    source: ContextData, h1: ContextData, rng: np.random.Generator
) -> dict[str, object]:
    arrays = anchor_arrays(source, h1)
    targets = arrays["targets"]
    genes = arrays["genes"]
    folds = deterministic_folds(targets)
    predictions, per_target_coefficients, fold_coefficients = crossfit_predictions(
        targets,
        genes,
        arrays["source"],
        arrays["truth"],
        arrays["difference"],
        folds,
    )
    per_target = evaluate_models(
        targets,
        genes,
        arrays["source"],
        arrays["truth"],
        predictions,
        source.cells[arrays["source_rows"]],
        h1.cells[arrays["h1_rows"]],
        folds,
    )
    edge = f"{source.spec.name}->H1"
    for frame in (per_target, per_target_coefficients, fold_coefficients):
        frame.insert(0, "edge", edge)
        frame.insert(1, "source", source.spec.name)
    summary = summarize_edge(per_target, rng)
    summary.insert(0, "edge", edge)
    summary.insert(1, "source", source.spec.name)
    differences = bootstrap_differences(per_target, rng)
    differences.insert(0, "edge", edge)
    differences.insert(1, "source", source.spec.name)
    coefficient_bootstrap = bootstrap_coefficients(per_target_coefficients, rng)
    coefficient_bootstrap.insert(0, "edge", edge)
    coefficient_bootstrap.insert(1, "source", source.spec.name)
    coefficient_bootstrap["context_difference_sd"] = np.std(arrays["difference"])

    destination = DERIVED / edge.replace("->", "_to_")
    destination.mkdir(parents=True, exist_ok=True)
    prediction_path = destination / "heldout_predictions.npz"
    np.savez_compressed(
        prediction_path,
        target_gene=np.asarray(targets),
        shared_gene=np.asarray(genes),
        fold=folds,
        source_lfc=np.asarray(arrays["source"], dtype=np.float32),
        target_lfc=np.asarray(arrays["truth"], dtype=np.float32),
        context_difference=np.asarray(arrays["difference"], dtype=np.float32),
        prediction_global=predictions["global_median"].astype(np.float32),
        prediction_interaction=predictions["context_interaction"].astype(np.float32),
    )
    return {
        **arrays,
        "source_name": source.spec.name,
        "source_study": source.spec.study,
        "source_assay": source.spec.assay,
        "predictions": predictions,
        "folds": folds,
        "source_cells": source.cells[arrays["source_rows"]],
        "target_cells": h1.cells[arrays["h1_rows"]],
        "per_target": per_target,
        "summary": summary,
        "differences": differences,
        "per_target_coefficients": per_target_coefficients,
        "fold_coefficients": fold_coefficients,
        "coefficient_bootstrap": coefficient_bootstrap,
        "prediction_path": prediction_path,
    }


def frozen_h1_metrics(
    result: dict[str, object], h1: ContextData, controls: ad.AnnData
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets = result["targets"]
    shared_genes = result["genes"]
    h1_rows = result["h1_rows"]
    gene_lookup = {gene: index for index, gene in enumerate(h1.genes)}
    full_indices = np.asarray([gene_lookup[gene] for gene in shared_genes])
    source_rows = canonical_source_rows(controls.n_obs, targets)
    de_truth, stable, de_valid = truth_arrays(targets, h1.genes)
    de_genes = np.asarray(h1.genes)[de_valid].tolist()
    de_lookup = {gene: index for index, gene in enumerate(de_genes)}
    truth_profiles = np.vstack([log_pseudobulk(h1.counts[row]) for row in h1_rows])
    control_profile = log_pseudobulk(h1.control_counts)
    intended_profiles = {
        model: np.empty((len(targets), len(h1.genes)), dtype=np.float64)
        for model in MODELS
    }
    null_profiles = np.empty_like(truth_profiles)
    frozen_rows = []
    for target_index, (target, selected) in enumerate(
        zip(targets, source_rows, strict=True)
    ):
        raw = controls.X[selected].tocsr()
        baseline = np.asarray(raw.sum(axis=0)).ravel().astype(np.float64)
        null_profiles[target_index] = log_pseudobulk(baseline)
        for model in MODELS:
            full = np.zeros(len(h1.genes), dtype=np.float64)
            full[full_indices] = result["predictions"][model][target_index]
            pooled = expected_lfc_decoded_sum(raw, full)
            realized = np.log2((pooled + EPSILON) / (baseline + EPSILON))
            intended_profiles[model][target_index] = log_pseudobulk(pooled)
            score = signed_topk(
                realized[de_valid],
                stable[target_index, de_valid],
                np.sign(de_truth[target_index, de_valid]),
                de_lookup.get(target),
            )
            eligible = score["n_strong"] >= MIN_STRONG
            frozen_rows.append(
                {
                    "source": result["source_name"],
                    "model": model,
                    "target_gene": target,
                    **score,
                    "strong_de_lfc_nmae": (
                        np.abs(
                            realized[stable[target_index]]
                            - de_truth[target_index, stable[target_index]]
                        ).sum()
                        / np.abs(de_truth[target_index, stable[target_index]]).sum()
                        if eligible
                        else np.nan
                    ),
                }
            )
    frozen = pd.DataFrame(frozen_rows)
    eligible = frozen.n_strong >= MIN_STRONG
    frozen_summary = (
        frozen.loc[eligible]
        .groupby(["source", "model"], sort=False)
        .agg(
            eligible_targets=("target_gene", "size"),
            mean_signed_recovery=("signed_recovery", "mean"),
            mean_unsigned_recall=("unsigned_recall", "mean"),
            mean_sign_given_recovered=("sign_given_recovered", "mean"),
            mean_strong_de_lfc_nmae=("strong_de_lfc_nmae", "mean"),
        )
        .reset_index()
    )
    profile_frames = []
    profile_summaries = []
    null_pds = pds(null_profiles, truth_profiles, control_profile, targets, h1.genes)
    null_effects = null_profiles - control_profile
    truth_effects = truth_profiles - control_profile
    for model in MODELS:
        predicted_pds = pds(
            intended_profiles[model], truth_profiles, control_profile, targets, h1.genes
        )
        predicted_effects = intended_profiles[model] - control_profile
        rows = []
        for index, target in enumerate(targets):
            keep = np.ones(len(h1.genes), dtype=bool)
            keep[gene_lookup[target]] = False
            observed = truth_effects[index, keep]
            predicted = predicted_effects[index, keep]
            null = null_effects[index, keep]
            rows.append(
                {
                    "target_gene": target,
                    "expected_pds": predicted_pds[target],
                    "null_pds": null_pds[target],
                    "expected_squared_error": np.square(predicted - observed).sum(),
                    "null_squared_error": np.square(null - observed).sum(),
                    "control_squared_error": np.square(observed).sum(),
                }
            )
        frame = pd.DataFrame(rows)
        frame.insert(0, "source", result["source_name"])
        frame.insert(1, "model", model)
        profile_frames.append(frame)
        profile_summaries.append(
            {
                "source": result["source_name"],
                "model": model,
                "targets": len(frame),
                "mean_pds": frame.expected_pds.mean(),
                "expression_mse_ratio": frame.expected_squared_error.sum()
                / frame.control_squared_error.sum(),
                "null_expression_mse_ratio": frame.null_squared_error.sum()
                / frame.control_squared_error.sum(),
            }
        )
    return (
        frozen,
        frozen_summary,
        pd.concat(profile_frames, ignore_index=True),
        pd.DataFrame(profile_summaries),
    )


def common_subset_evaluation(
    results: list[dict[str, object]], common_targets: set[str], rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    summaries = []
    for result in results:
        selected = np.asarray(
            [target in common_targets for target in result["targets"]], dtype=bool
        )
        targets = np.asarray(result["targets"])[selected].tolist()
        predictions = {
            model: values[selected] for model, values in result["predictions"].items()
        }
        frame = evaluate_models(
            targets,
            result["genes"],
            result["source"][selected],
            result["truth"][selected],
            predictions,
            result["source_cells"][selected],
            result["target_cells"][selected],
            result["folds"][selected],
        )
        frame.insert(0, "edge", f"{result['source_name']}->H1")
        frame.insert(1, "source", result["source_name"])
        frames.append(frame)
        summary = summarize_edge(frame, rng)
        summary.insert(0, "edge", f"{result['source_name']}->H1")
        summary.insert(1, "source", result["source_name"])
        summaries.append(summary)
    return pd.concat(frames, ignore_index=True), pd.concat(summaries, ignore_index=True)


def plot_h1_anchors(available: pd.DataFrame, common: pd.DataFrame) -> None:
    direct = available.query("model == 'direct'").set_index("source")
    common_direct = common.query("model == 'direct'").set_index("source")
    sources = direct.index.tolist()
    metrics = (
        ("mean_effect_cosine", "Target-excluded cosine"),
        ("mean_retrieval_score", "Correct-target retrieval"),
        ("mean_signed_recovery_100", "Signed top-100 recovery"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    x = np.arange(len(sources))
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        axis.bar(x - 0.18, direct.loc[sources, metric], 0.36, label="available")
        axis.bar(
            x + 0.18,
            common_direct.loc[sources, metric],
            0.36,
            label="all-source common",
        )
        axis.set_title(title)
        axis.set_xticks(x, sources, rotation=30, ha="right")
        if "retrieval" in metric:
            axis.axhline(0.5, color="black", lw=0.7, alpha=0.5)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Descriptive cross-study H1 anchors: direct transfer")
    figure.savefig(REPORT / "h1_anchors.png", dpi=180)
    figure.savefig(REPORT / "h1_anchors.svg")
    plt.close(figure)


def update_manifest(paths: list[Path], inputs: dict[str, str]) -> None:
    manifest_path = REPORT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["h1_anchors"] = {
        "status": "completed after final KOLF compact audit passed",
        "role": "descriptive cross-study anchors; not additional context replicates",
        "sources": ["K562_GWPS", "CD4T", "HCT116", "HEK293T", "KOLF2.1J"],
        "fast_evaluation": "canonical H1 controls, expected multiplicative decoder, PDS and expression-MSE ratio; full scorer not run",
        "mse_interpretation": "contract diagnostic only; never a model-selection gate",
        "inputs": inputs,
    }
    manifest["inputs"].update(inputs)
    for path in paths:
        manifest["outputs"][str(path.relative_to(ROOT))] = sha256(path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if not (REPORT / "manifest.json").exists():
        raise FileNotFoundError("run context_transfer_blog before the H1 anchors")
    kolf_audit = audit_compact(SOURCE_SPECS["KOLF2.1J"].directory)
    if kolf_audit["status"] != "passed":
        raise RuntimeError("final KOLF compact audit did not pass")
    sources = [replogle_gwps_context()]
    for spec in SOURCE_SPECS.values():
        sources.append(compact_source(spec))
    h1 = h1_context()
    rng = np.random.default_rng(SEED + 1)
    results = [fit_anchor(source, h1, rng) for source in sources]

    controls = ad.read_h5ad(H1_CONTROLS, backed="r")
    try:
        if controls.var_names.astype(str).tolist() != h1.genes:
            raise ValueError("H1 control and aggregate gene axes differ")
        frozen = [frozen_h1_metrics(result, h1, controls) for result in results]
    finally:
        controls.file.close()

    common_targets = set.intersection(*(set(result["targets"]) for result in results))
    per_target = pd.concat(
        [result["per_target"] for result in results], ignore_index=True
    )
    available_summary = pd.concat(
        [result["summary"] for result in results], ignore_index=True
    )
    common_per_target, common_summary = common_subset_evaluation(
        results, common_targets, rng
    )
    edge_rows = []
    for result in results:
        edge_rows.append(
            {
                "edge": f"{result['source_name']}->H1",
                "source": result["source_name"],
                "source_study": result["source_study"],
                "source_assay": result["source_assay"],
                "available_targets": len(result["targets"]),
                "all_source_common_targets": len(common_targets),
                "shared_genes": len(result["genes"]),
                "role": "descriptive cross-study anchor",
                "prediction_artifact": str(result["prediction_path"].relative_to(ROOT)),
            }
        )
    tables = {
        "h1_edges.csv": pd.DataFrame(edge_rows),
        "h1_per_target.csv": per_target,
        "h1_per_edge_available.csv": available_summary,
        "h1_per_edge_common.csv": common_summary,
        "h1_per_target_common.csv": common_per_target,
        "h1_performance_bootstrap.csv": pd.concat(
            [result["differences"] for result in results], ignore_index=True
        ),
        "h1_per_target_coefficients.csv": pd.concat(
            [result["per_target_coefficients"] for result in results],
            ignore_index=True,
        ),
        "h1_fold_coefficients.csv": pd.concat(
            [result["fold_coefficients"] for result in results], ignore_index=True
        ),
        "h1_coefficient_bootstrap.csv": pd.concat(
            [result["coefficient_bootstrap"] for result in results],
            ignore_index=True,
        ),
        "h1_frozen_strong_de_per_target.csv": pd.concat(
            [item[0] for item in frozen], ignore_index=True
        ),
        "h1_frozen_strong_de_summary.csv": pd.concat(
            [item[1] for item in frozen], ignore_index=True
        ),
        "h1_fast_profile_per_target.csv": pd.concat(
            [item[2] for item in frozen], ignore_index=True
        ),
        "h1_fast_profile_summary.csv": pd.concat(
            [item[3] for item in frozen], ignore_index=True
        ),
        "h1_all_source_common_targets.csv": pd.DataFrame(
            {"target_gene": sorted(common_targets)}
        ),
    }
    paths = []
    for name, frame in tables.items():
        path = REPORT / name
        frame.to_csv(path, index=False)
        paths.append(path)
    plot_h1_anchors(available_summary, common_summary)
    paths.extend([REPORT / "h1_anchors.png", REPORT / "h1_anchors.svg"])
    paths.extend(result["prediction_path"] for result in results)
    input_paths = [H1_AGGREGATION, H1_PHASE1, H1_CONTROLS, REPLOGLE_RAW]
    input_paths.extend(
        path
        for spec in SOURCE_SPECS.values()
        for path in (
            spec.target_path,
            spec.guide_path,
            spec.manifest_path,
            spec.audit_path,
        )
    )
    input_paths.extend(SOURCE_SPECS["KOLF2.1J"].directory.glob("*_control_*.h5ad"))
    inputs = {str(path.relative_to(ROOT)): sha256(path) for path in input_paths}
    update_manifest(paths, inputs)
    print(pd.DataFrame(edge_rows).to_string(index=False))
    print(f"All-source common H1 target subset: {len(common_targets)}")


if __name__ == "__main__":
    main()
