"""Prepare and evaluate expected H1 profiles for state-balanced covariances."""

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
from platformdirs import user_cache_path
from scipy.stats import spearmanr

from scripts.evaluation.prepare_paired_linear_response_h1 import (
    canonical_source_rows,
    log2_ratio,
    truth_arrays,
)
from scripts.linear_response.kernel import (
    expected_decoded_sum,
    expected_lfc_decoded_sum,
)
from scripts.linear_response.strong_de_recovery import (
    derangement_null,
    pairwise_wrong_target_scores,
    ranked_genes,
    signed_topk,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = Path(user_cache_path("vcc2026-h1-benchmark"))
CONTROLS = BENCHMARK / "h1_controls.h5ad"
CONTROL_MANIFEST = BENCHMARK / "h1_controls_manifest.json"
MODEL = (
    ROOT
    / "data/derived/linear_response/state_diversity/state_balanced_covariance.npz"
)
SCALES = ROOT / "reports/linear-response-ladder/downstream_amplitude_distribution.csv"
DERIVED = ROOT / "data/derived/linear_response/state_diversity/effects.npz"
REPORT = ROOT / "reports/linear-response-state-diversity"
N_NULL = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else 0.0


def main() -> None:
    with np.load(MODEL, allow_pickle=False) as saved:
        models = saved["model"].astype(str).tolist()
        targets = saved["target_gene"].astype(str).tolist()
        genes_fit = saved["fit_gene"].astype(str).tolist()
        output_genes = saved["output_gene"].astype(str).tolist()
        fit_indices = saved["full_gene_index"].astype(int)
        target_fold = saved["target_fold"].astype(int)
        responses = saved["response"].astype(np.float64)
    gamma = float(
        pd.read_csv(SCALES).query("model == 'empirical'")["loo_median_gamma"].median()
    )
    truth, stable = truth_arrays(targets, output_genes)
    truth_sign = np.sign(truth).astype(np.int8)
    target_indices = [
        output_genes.index(target) if target in output_genes else None
        for target in targets
    ]

    controls = ad.read_h5ad(CONTROLS, backed="r")
    try:
        genes = controls.var_names.astype(str).tolist()
        if [genes[index] for index in fit_indices] != genes_fit:
            raise ValueError("state-balanced model and control gene axes differ")
        source_rows = canonical_source_rows(controls.n_obs, targets)
        effects = np.empty((len(models), len(targets), len(genes)), dtype=np.float32)
        expected = np.empty(
            (len(models), len(targets), len(output_genes)), dtype=np.float32
        )
        for target_index, selected in enumerate(source_rows):
            raw = controls.X[selected].tocsr()
            baseline = np.asarray(raw.sum(axis=0)).ravel()
            for model_index in range(len(models)):
                direction = np.zeros(len(genes))
                direction[fit_indices] = responses[model_index, target_index]
                first_pool = expected_decoded_sum(raw, direction, gamma)
                effect = log2_ratio(first_pool, baseline)
                final_pool = expected_lfc_decoded_sum(raw, effect)
                effects[model_index, target_index] = effect
                expected[model_index, target_index] = log2_ratio(
                    final_pool, baseline
                )[fit_indices[: len(output_genes)]]
            print(
                f"expected profiles {target_index + 1}/{len(targets)}: "
                f"{targets[target_index]}",
                flush=True,
            )
    finally:
        controls.file.close()

    rows = []
    summaries = []
    for model_index, model in enumerate(models):
        predictions = expected[model_index].astype(np.float64)
        ranking = ranked_genes(predictions)
        pairwise = pairwise_wrong_target_scores(
            predictions, ranking, stable, truth_sign, target_indices
        )
        observed = []
        excess = []
        for target_index, target in enumerate(targets):
            result = signed_topk(
                predictions[target_index],
                stable[target_index],
                truth_sign[target_index],
                target_indices[target_index],
            )
            eligible = result["n_strong"] >= 10
            mask = np.ones(len(output_genes), dtype=bool)
            if target_indices[target_index] is not None:
                mask[target_indices[target_index]] = False
            wrong = np.delete(pairwise[target_index], target_index)
            wrong = wrong[np.isfinite(wrong)]
            nmae = (
                np.abs(
                    predictions[target_index, stable[target_index]]
                    - truth[target_index, stable[target_index]]
                ).sum()
                / np.abs(truth[target_index, stable[target_index]]).sum()
                if eligible
                else np.nan
            )
            if eligible:
                observed.append(result["signed_recovery"])
                excess.append(result["signed_recovery"] - wrong.mean())
            rows.append(
                {
                    "model": model,
                    "target_gene": target,
                    "fold": target_fold[target_index],
                    "transductive": model.startswith("all_"),
                    "eligible": eligible,
                    **result,
                    "strong_de_lfc_nmae": nmae,
                    "target_excluded_cosine": cosine(
                        predictions[target_index, mask], truth[target_index, mask]
                    ),
                    "target_excluded_spearman": spearmanr(
                        predictions[target_index, mask], truth[target_index, mask]
                    ).statistic,
                    "wrong_target_mean": wrong.mean() if eligible else np.nan,
                    "signed_recovery_excess_wrong": (
                        result["signed_recovery"] - wrong.mean()
                        if eligible
                        else np.nan
                    ),
                }
            )
        wrong_null = derangement_null(
            pairwise, np.random.default_rng([0, 112, model_index])
        )
        observed_mean = float(np.mean(observed))
        bootstrap = np.random.default_rng([0, 113, model_index]).choice(
            excess, size=(N_NULL, len(excess)), replace=True
        ).mean(axis=1)
        summaries.append(
            {
                "model": model,
                "transductive": model.startswith("all_"),
                "eligible_targets": len(observed),
                "mean_signed_recovery": observed_mean,
                "wrong_target_mean": wrong_null.mean(),
                "wrong_target_p": (1 + np.sum(wrong_null >= observed_mean))
                / (N_NULL + 1),
                "excess_wrong_q025": np.quantile(bootstrap, 0.025),
                "excess_wrong_q975": np.quantile(bootstrap, 0.975),
                "mean_strong_de_lfc_nmae": np.nanmean(
                    [row["strong_de_lfc_nmae"] for row in rows if row["model"] == model]
                ),
                "mean_target_excluded_cosine": np.mean(
                    [row["target_excluded_cosine"] for row in rows if row["model"] == model]
                ),
                "mean_target_excluded_spearman": np.mean(
                    [row["target_excluded_spearman"] for row in rows if row["model"] == model]
                ),
            }
        )

    DERIVED.parent.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DERIVED,
        model=np.asarray(models),
        target_gene=np.asarray(targets),
        gene=np.asarray(genes),
        output_gene=np.asarray(output_genes),
        source_rows=source_rows,
        gamma=np.asarray(gamma),
        lfc_effect=effects,
        expected_lfc=expected,
    )
    per_target = REPORT / "expected_profile_per_target.csv"
    summary = REPORT / "expected_profile_summary.csv"
    pd.DataFrame(rows).to_csv(per_target, index=False)
    pd.DataFrame(summaries).to_csv(summary, index=False)
    control_artifact = json.loads(CONTROL_MANIFEST.read_text())["artifact"]
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "design": "paired canonical-control expected-profile comparison",
        "amplitude": {
            "gamma": gamma,
            "source": "median control-only empirical downstream cross-fitted scale",
            "shared_across_models": True,
        },
        "decoder": "exact fractional multiplicative LFC closure",
        "inputs": {
            str(MODEL.relative_to(ROOT)): sha256(MODEL),
            str(SCALES.relative_to(ROOT)): sha256(SCALES),
            str(CONTROLS): control_artifact["sha256"],
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [DERIVED, per_target, summary]
        },
        "software": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ["anndata", "numpy", "pandas", "scipy"]
            },
        },
    }
    (REPORT / "expected_profile_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
