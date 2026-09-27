"""Prepare clean paired H1 effects on the canonical all-control source cells."""

from __future__ import annotations

import hashlib
import json
import platform
import zlib
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from platformdirs import user_cache_path

from scripts.linear_response.kernel import (
    expected_decoded_sum,
    expected_lfc_decoded_sum,
)
from scripts.linear_response.strong_de_recovery import signed_topk

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = Path(user_cache_path("vcc2026-h1-benchmark"))
CONTROLS = BENCHMARK / "h1_controls.h5ad"
CONTROL_MANIFEST = BENCHMARK / "h1_controls_manifest.json"
BENCHMARK_MANIFEST = BENCHMARK / "benchmark/benchmark_manifest.json"
DE = BENCHMARK / "benchmark/reference_cache/de_wilcoxon_table-12f179b9e966af64.parquet"
MODEL = ROOT / "data/derived/linear_response/ladder/empirical.npz"
SCALES = ROOT / "reports/linear-response-ladder/downstream_amplitude_distribution.csv"
STRONG = ROOT / "reports/linear-response-three-models/strong_de_truth_genes.csv"
DERIVED = ROOT / "data/derived/linear_response/ladder/paired_h1_effects.npz"
REPORT = ROOT / "reports/linear-response-ladder"
SEED_NAMESPACE = "h1-control-baseline-v1"
CELLS_PER_TARGET = 400
EPSILON = 1e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_source_rows(n_controls: int, targets: list[str]) -> np.ndarray:
    pool = np.arange(n_controls)
    return np.vstack(
        [
            np.sort(
                np.random.default_rng(
                    zlib.crc32(f"{SEED_NAMESPACE}:{target}".encode())
                ).choice(pool, CELLS_PER_TARGET, replace=False)
            )
            for target in targets
        ]
    )


def truth_arrays(
    targets: list[str], output_genes: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    frame = pl.read_parquet(DE).to_pandas()
    truth = np.vstack(
        [
            frame.loc[frame["target"] == target]
            .set_index("feature")
            .loc[output_genes, "log2_fold_change"]
            .to_numpy()
            for target in targets
        ]
    )
    stable = np.zeros_like(truth, dtype=bool)
    target_lookup = {target: index for index, target in enumerate(targets)}
    gene_lookup = {gene: index for index, gene in enumerate(output_genes)}
    for row in pd.read_csv(STRONG).query("stable_strong").itertuples(index=False):
        stable[target_lookup[row.target_gene], gene_lookup[row.feature]] = True
    for target, target_gene in enumerate(targets):
        if target_gene in gene_lookup:
            stable[target, gene_lookup[target_gene]] = False
    return truth, stable


def log2_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.log2((numerator + EPSILON) / (denominator + EPSILON))


def main() -> None:
    with np.load(MODEL, allow_pickle=False) as model:
        targets = model["target_gene"].astype(str).tolist()
        output_genes = model["output_gene"].astype(str).tolist()
        fit_indices = model["full_gene_index"].astype(int)
        responses = model["response"].astype(np.float64)
    scales = (
        pd.read_csv(SCALES)
        .query("model == 'empirical'")
        .set_index("target_gene")
        .loc[targets, "loo_median_gamma"]
        .to_numpy()
    )
    truth, stable = truth_arrays(targets, output_genes)
    global_lfc = (truth.sum(axis=0) - truth) / (len(truth) - 1)

    controls = ad.read_h5ad(CONTROLS, backed="r")
    try:
        genes = controls.var_names.astype(str).tolist()
        if [genes[index] for index in fit_indices[: len(output_genes)]] != output_genes:
            raise ValueError("model and benchmark gene axes differ")
        source_rows = canonical_source_rows(controls.n_obs, targets)
        effects = np.zeros((3, len(targets), len(genes)), dtype=np.float32)
        expected = np.zeros((4, len(targets), len(output_genes)), dtype=np.float32)
        rows = []
        target_indices = [
            output_genes.index(target) if target in output_genes else None
            for target in targets
        ]
        for target, selected in enumerate(source_rows):
            raw = controls.X[selected].tocsr()
            baseline = np.asarray(raw.sum(axis=0)).ravel()
            direction = np.zeros(len(genes))
            direction[fit_indices] = responses[target]
            lr_lfc = log2_ratio(
                expected_decoded_sum(raw, direction, scales[target]), baseline
            )
            effects[0, target, fit_indices[: len(output_genes)]] = global_lfc[target]
            effects[1, target] = lr_lfc
            effects[2, target] = effects[0, target] + effects[1, target]

            intended = [np.zeros(len(genes)), *effects[:, target]]
            for arm, effect in enumerate(intended):
                realized = log2_ratio(expected_lfc_decoded_sum(raw, effect), baseline)
                prediction = realized[fit_indices[: len(output_genes)]]
                expected[arm, target] = prediction
                score = signed_topk(
                    prediction,
                    stable[target],
                    np.sign(truth[target]),
                    target_indices[target],
                )
                eligible = score["n_strong"] >= 10
                difference = prediction - effect[fit_indices[: len(output_genes)]]
                rows.append(
                    {
                        "arm": ["null", "global", "lr", "lr_global"][arm],
                        "target_gene": targets[target],
                        **score,
                        "strong_de_lfc_nmae": (
                            np.abs(prediction[stable[target]] - truth[target][stable[target]]).sum()
                            / np.abs(truth[target][stable[target]]).sum()
                            if eligible
                            else np.nan
                        ),
                        "mean_abs_intended_realized_lfc": np.abs(difference).mean(),
                        "median_intended_realized_lfc": np.median(difference),
                        "max_abs_intended_realized_lfc": np.abs(difference).max(),
                    }
                )
            print(f"Paired effects {target + 1}/{len(targets)}: {targets[target]}", flush=True)
    finally:
        controls.file.close()

    DERIVED.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DERIVED,
        arm=np.asarray(["global", "lr", "lr_global"]),
        target_gene=np.asarray(targets),
        gene=np.asarray(genes),
        output_gene=np.asarray(output_genes),
        source_rows=source_rows,
        lfc_effect=effects,
        expected_arm=np.asarray(["null", "global", "lr", "lr_global"]),
        expected_lfc=expected,
    )
    per_target = pd.DataFrame(rows)
    per_target.to_csv(REPORT / "paired_h1_expected_per_target.csv", index=False)
    eligible = per_target["n_strong"] >= 10
    summary = (
        per_target.loc[eligible]
        .groupby("arm", sort=False)
        .agg(
            eligible_targets=("target_gene", "size"),
            mean_signed_recovery=("signed_recovery", "mean"),
            mean_unsigned_recovery=("unsigned_recall", "mean"),
            mean_sign_given_recovered=("sign_given_recovered", "mean"),
            mean_strong_de_lfc_nmae=("strong_de_lfc_nmae", "mean"),
            mean_abs_intended_realized_lfc=(
                "mean_abs_intended_realized_lfc",
                "mean",
            ),
        )
        .reset_index()
    )
    summary.to_csv(REPORT / "paired_h1_expected_summary.csv", index=False)
    control_artifact = json.loads(CONTROL_MANIFEST.read_text())["artifact"]
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "design": "paired canonical-control H1 response comparison",
        "source_rows": {
            "pool": "all 38,176 H1 controls",
            "cells_per_target": CELLS_PER_TARGET,
            "seed_namespace": SEED_NAMESPACE,
            "shared_across_arms": True,
        },
        "effects": {
            "global": "equal-target mean truth LFC over the other 125 perturbations",
            "lr": "exact expected empirical LR LFC on the canonical source rows",
            "lr_global": "global + LR in LFC space",
            "off_panel_global_lfc": 0.0,
        },
        "decoder": "multiplicative LFC closure with deterministic largest-remainder counts",
        "inputs": {
            str(MODEL.relative_to(ROOT)): sha256(MODEL),
            str(SCALES.relative_to(ROOT)): sha256(SCALES),
            str(STRONG.relative_to(ROOT)): sha256(STRONG),
            str(DE): sha256(DE),
            str(BENCHMARK_MANIFEST): sha256(BENCHMARK_MANIFEST),
            str(CONTROLS): control_artifact["sha256"],
        },
        "outputs": {
            str(DERIVED.relative_to(ROOT)): sha256(DERIVED),
            "reports/linear-response-ladder/paired_h1_expected_per_target.csv": sha256(
                REPORT / "paired_h1_expected_per_target.csv"
            ),
            "reports/linear-response-ladder/paired_h1_expected_summary.csv": sha256(
                REPORT / "paired_h1_expected_summary.csv"
            ),
        },
        "software": {
            "python": platform.python_version(),
            **{name: version(name) for name in ["anndata", "numpy", "pandas", "polars", "scipy"]},
        },
    }
    (REPORT / "paired_h1_expected_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
