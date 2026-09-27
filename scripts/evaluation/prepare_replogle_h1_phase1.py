"""Build and evaluate the pooled Replogle-to-H1 Phase 1 transfer arms."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from platformdirs import user_cache_path

from scripts.evaluation.prepare_paired_linear_response_h1 import canonical_source_rows
from scripts.evaluation.prepare_replogle_h1_transfer import loo_components
from scripts.linear_response.kernel import expected_lfc_decoded_sum
from scripts.linear_response.strong_de_recovery import signed_topk

ROOT = Path(__file__).resolve().parents[2]
AGGREGATION = ROOT / "data/derived/replogle_h1_transfer/effects.npz"
DE = ROOT / "data/derived/linear_response/eval_cache/de_wilcoxon_table-12f179b9e966af64.parquet"
STRONG = ROOT / "reports/linear-response-three-models/strong_de_truth_genes.csv"
CONTROLS = Path(user_cache_path("vcc2026-h1-benchmark")) / "h1_controls.h5ad"
CONTROL_MANIFEST = Path(user_cache_path("vcc2026-h1-benchmark")) / "h1_controls_manifest.json"
DERIVED = ROOT / "data/derived/replogle_h1_transfer/phase1_effects.npz"
REPORT = ROOT / "reports/replogle-h1-transfer/phase1"
EPSILON = 1e-9
MIN_STRONG = 10

ARMS = (
    "unchanged",
    "source_global",
    "source_residual",
    "source_effect",
    "h1_global",
    "h1_global_source_residual",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else np.nan


def build_arms(
    targets: list[str], matched: list[str], source: np.ndarray, h1: np.ndarray
) -> np.ndarray:
    """Build six arms, using the pooled source global for unmatched targets."""
    source_global_loo, source_residual = loo_components(source)
    h1_global_loo, _ = loo_components(h1)
    matched_lookup = {target: index for index, target in enumerate(matched)}
    source_global_all = source.mean(axis=0)
    global_126 = np.repeat(source_global_all[None, :], len(targets), axis=0)
    residual_126 = np.zeros_like(global_126)
    source_126 = global_126.copy()
    for row, target in enumerate(targets):
        if target not in matched_lookup:
            continue
        source_row = matched_lookup[target]
        global_126[row] = source_global_loo[source_row]
        residual_126[row] = source_residual[source_row]
        source_126[row] = source[source_row]
    zeros = np.zeros_like(global_126)
    return np.stack(
        [
            zeros,
            global_126,
            residual_126,
            source_126,
            h1_global_loo,
            h1_global_loo + residual_126,
        ]
    )


def truth_arrays(
    targets: list[str], genes: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    de_long = (
        pl.read_parquet(DE)
        .filter(pl.col("feature").is_in(genes))
        .select("target", "feature", "log2_fold_change")
        .to_pandas()
    )
    de_genes = set(de_long.feature)
    de_valid = np.asarray([gene in de_genes for gene in genes])
    de = (
        de_long
        .pivot(index="target", columns="feature", values="log2_fold_change")
        .reindex(index=targets, columns=genes)
        .fillna(0.0)
    )
    stable = np.zeros((len(targets), len(genes)), dtype=bool)
    target_lookup = {target: index for index, target in enumerate(targets)}
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    for row in pd.read_csv(STRONG).query("stable_strong").itertuples(index=False):
        if row.target_gene in target_lookup and row.feature in gene_lookup:
            stable[target_lookup[row.target_gene], gene_lookup[row.feature]] = True
    for row, target in enumerate(targets):
        if target in gene_lookup:
            stable[row, gene_lookup[target]] = False
    return de.to_numpy(dtype=np.float64), stable, de_valid


def expected_realized_effects(
    intended: np.ndarray,
    targets: list[str],
    genes: list[str],
    shared_genes: list[str],
    arm_names: tuple[str, ...] | list[str] = ARMS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    controls = ad.read_h5ad(CONTROLS, backed="r")
    try:
        if controls.var_names.astype(str).tolist() != genes:
            raise ValueError("H1 controls and output gene axes differ")
        gene_lookup = {gene: index for index, gene in enumerate(genes)}
        shared_indices = np.asarray([gene_lookup[gene] for gene in shared_genes])
        source_rows = canonical_source_rows(controls.n_obs, targets)
        realized = np.empty(
            (len(arm_names), len(targets), len(genes)), dtype=np.float64
        )
        for target_index, rows in enumerate(source_rows):
            raw = controls.X[rows].tocsr()
            baseline = np.asarray(raw.sum(axis=0)).ravel().astype(np.float64)
            for arm_index, arm in enumerate(arm_names):
                full = np.zeros(len(genes), dtype=np.float64)
                full[shared_indices] = intended[arm_index, target_index]
                pooled = expected_lfc_decoded_sum(raw, full)
                realized[arm_index, target_index] = np.log2(
                    (pooled + EPSILON) / (baseline + EPSILON)
                )
            print(
                f"Expected Phase 1 effects {target_index + 1}/{len(targets)}: "
                f"{targets[target_index]}",
                flush=True,
            )
    finally:
        controls.file.close()
    return realized, source_rows, shared_indices


def evaluate(
    intended: np.ndarray,
    realized: np.ndarray,
    targets: list[str],
    output_genes: list[str],
    shared_genes: list[str],
    shared_indices: np.ndarray,
    matched: set[str],
    h1_effect: np.ndarray,
    truth: np.ndarray,
    stable: np.ndarray,
    de_valid: np.ndarray,
    arm_names: tuple[str, ...] | list[str] = ARMS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared_gene_lookup = {gene: index for index, gene in enumerate(shared_genes)}
    de_genes = np.asarray(output_genes)[de_valid].tolist()
    de_gene_lookup = {gene: index for index, gene in enumerate(de_genes)}
    rows = []
    for arm_index, arm in enumerate(arm_names):
        for target_index, target in enumerate(targets):
            valid = np.ones(len(shared_genes), dtype=bool)
            shared_target_index = shared_gene_lookup.get(target)
            if shared_target_index is not None:
                valid[shared_target_index] = False
            prediction_full = realized[arm_index, target_index]
            prediction = prediction_full[shared_indices]
            reference = h1_effect[target_index]
            pred_valid, ref_valid = prediction[valid], reference[valid]
            pred_norm, ref_norm = np.linalg.norm(pred_valid), np.linalg.norm(ref_valid)
            topk = signed_topk(
                prediction_full[de_valid],
                stable[target_index, de_valid],
                np.sign(truth[target_index, de_valid]),
                de_gene_lookup.get(target),
            )
            eligible = topk["n_strong"] >= MIN_STRONG
            rows.append(
                {
                    "arm": arm,
                    "target_gene": target,
                    "source_overlap": target in matched,
                    "effect_cosine": cosine(pred_valid, ref_valid),
                    "predicted_to_truth_norm_ratio": (
                        float(pred_norm / ref_norm) if ref_norm else np.nan
                    ),
                    "truth_on_prediction_projection": (
                        float(pred_valid @ ref_valid / pred_norm**2)
                        if pred_norm
                        else np.nan
                    ),
                    "predicted_effect_norm": float(pred_norm),
                    "truth_effect_norm": float(ref_norm),
                    **topk,
                    "strong_de_lfc_nmae": (
                        float(
                            np.abs(
                                prediction_full[stable[target_index]]
                                - truth[target_index, stable[target_index]]
                            ).sum()
                            / np.abs(truth[target_index, stable[target_index]]).sum()
                        )
                        if eligible
                        else np.nan
                    ),
                    "mean_abs_intended_realized_lfc": float(
                        np.abs(prediction - intended[arm_index, target_index]).mean()
                    ),
                }
            )
    per_target = pd.DataFrame(rows)
    summaries = []
    for arm in arm_names:
        arm_frame = per_target[per_target.arm == arm]
        for population, selected in (
            ("shared_115", arm_frame.source_overlap),
            ("all_126", np.ones(len(arm_frame), dtype=bool)),
        ):
            frame = arm_frame.loc[selected]
            eligible = frame.n_strong >= MIN_STRONG
            summaries.append(
                {
                    "arm": arm,
                    "population": population,
                    "targets": len(frame),
                    "eligible_strong_de_targets": int(eligible.sum()),
                    "mean_effect_cosine": frame.effect_cosine.mean(),
                    "median_effect_cosine": frame.effect_cosine.median(),
                    "median_predicted_to_truth_norm_ratio": frame.predicted_to_truth_norm_ratio.median(),
                    "median_truth_on_prediction_projection": frame.truth_on_prediction_projection.median(),
                    "mean_signed_recovery": frame.loc[eligible, "signed_recovery"].mean(),
                    "mean_unsigned_recall": frame.loc[eligible, "unsigned_recall"].mean(),
                    "mean_sign_given_recovered": frame.loc[
                        eligible, "sign_given_recovered"
                    ].mean(),
                    "mean_strong_de_lfc_nmae": frame.loc[
                        eligible, "strong_de_lfc_nmae"
                    ].mean(),
                }
            )
    return per_target, pd.DataFrame(summaries)


def wrong_target_reference(
    residual: np.ndarray,
    targets: list[str],
    output_genes: list[str],
    shared_genes: list[str],
    shared_indices: np.ndarray,
    matched: list[str],
    h1_effect: np.ndarray,
    truth: np.ndarray,
    stable: np.ndarray,
    de_valid: np.ndarray,
) -> pd.DataFrame:
    target_lookup = {target: index for index, target in enumerate(targets)}
    shared_gene_lookup = {gene: index for index, gene in enumerate(shared_genes)}
    de_genes = np.asarray(output_genes)[de_valid].tolist()
    de_gene_lookup = {gene: index for index, gene in enumerate(de_genes)}
    rows = []
    for truth_target in matched:
        truth_index = target_lookup[truth_target]
        shared_target_index = shared_gene_lookup.get(truth_target)
        valid = np.ones(len(shared_genes), dtype=bool)
        if shared_target_index is not None:
            valid[shared_target_index] = False
        cosines, recoveries = [], []
        for predicted_target in matched:
            if predicted_target == truth_target:
                continue
            prediction_full = residual[target_lookup[predicted_target]]
            prediction = prediction_full[shared_indices]
            cosines.append(cosine(prediction[valid], h1_effect[truth_index, valid]))
            score = signed_topk(
                prediction_full[de_valid],
                stable[truth_index, de_valid],
                np.sign(truth[truth_index, de_valid]),
                de_gene_lookup.get(truth_target),
            )
            if score["n_strong"] >= MIN_STRONG:
                recoveries.append(score["signed_recovery"])
        rows.append(
            {
                "target_gene": truth_target,
                "wrong_targets": len(cosines),
                "median_wrong_target_cosine": np.nanmedian(cosines),
                "q95_wrong_target_cosine": np.nanquantile(cosines, 0.95),
                "median_wrong_target_signed_recovery": (
                    np.nanmedian(recoveries) if recoveries else np.nan
                ),
                "q95_wrong_target_signed_recovery": (
                    np.nanquantile(recoveries, 0.95) if recoveries else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    with np.load(AGGREGATION, allow_pickle=False) as saved:
        targets = saved["target_gene"].astype(str).tolist()
        matched = saved["matched_target_gene"].astype(str).tolist()
        shared_genes = saved["shared_gene"].astype(str).tolist()
        source = saved["replogle_lfc_native"].astype(np.float64)
        h1 = saved["h1_lfc_native"].astype(np.float64)
    intended = build_arms(targets, matched, source, h1)
    controls = ad.read_h5ad(CONTROLS, backed="r")
    try:
        output_genes = controls.var_names.astype(str).tolist()
    finally:
        controls.file.close()
    truth, stable, de_valid = truth_arrays(targets, output_genes)
    realized, source_rows, full_gene_index = expected_realized_effects(
        intended, targets, output_genes, shared_genes
    )
    per_target, summary = evaluate(
        intended,
        realized,
        targets,
        output_genes,
        shared_genes,
        full_gene_index,
        set(matched),
        h1,
        truth,
        stable,
        de_valid,
    )
    wrong = wrong_target_reference(
        realized[ARMS.index("source_residual")],
        targets,
        output_genes,
        shared_genes,
        full_gene_index,
        matched,
        h1,
        truth,
        stable,
        de_valid,
    )
    per_target.to_csv(REPORT / "expected_per_target.csv", index=False)
    summary.to_csv(REPORT / "expected_summary.csv", index=False)
    wrong.to_csv(REPORT / "wrong_target_reference.csv", index=False)

    np.savez_compressed(
        DERIVED,
        arm=np.asarray(ARMS),
        target_gene=np.asarray(targets),
        matched_target_gene=np.asarray(matched),
        shared_gene=np.asarray(shared_genes),
        output_gene=np.asarray(output_genes),
        full_gene_index=full_gene_index,
        source_rows=source_rows,
        intended_lfc=intended.astype(np.float32),
        expected_realized_lfc=realized.astype(np.float32),
    )
    controls_manifest = json.loads(CONTROL_MANIFEST.read_text())
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "stage": "Phase 1 expected-effect evaluation",
        "construct_handling": "pool target-transcript dual-guide populations by summing reconstructed counts",
        "missing_target_fallback": "zero source residual; source-global arms retain the equal-target source-global effect",
        "h1_global_eligibility": "diagnostic transductive ceiling only",
        "producer": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "inputs": {
            str(AGGREGATION.relative_to(ROOT)): sha256(AGGREGATION),
            str(DE.relative_to(ROOT)): sha256(DE),
            str(STRONG.relative_to(ROOT)): sha256(STRONG),
            str(CONTROLS): controls_manifest["artifact"]["sha256"],
        },
        "outputs": {
            str(DERIVED.relative_to(ROOT)): sha256(DERIVED),
            **{
                str(path.relative_to(ROOT)): sha256(path)
                for path in sorted(REPORT.glob("*.csv"))
            },
        },
        "software": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ("anndata", "numpy", "pandas", "polars", "scipy")
            },
        },
    }
    (REPORT / "expected_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
