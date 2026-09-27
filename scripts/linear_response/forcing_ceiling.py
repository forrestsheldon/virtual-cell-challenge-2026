"""Test whether sparse multigene forcing rescues linear-response directions."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data/derived/linear_response"
REPORT = ROOT / "reports/linear-response-three-models"
MODEL1 = DERIVED / "model1_expected_profiles.npz"
MODEL2 = DERIVED / "model2_expected_profiles.npz"
DE_TABLE = (
    DERIVED / "eval_cache/de_wilcoxon_table-12f179b9e966af64.parquet"
)
STRONG_TRUTH = REPORT / "strong_de_truth_genes.csv"

RANK = 50
SUPPORT_SIZES = (1, 2, 5, 10)
MIN_STRONG = 10
N_NULL = 10_000
FOLD_SEED = 73


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_gene_folds(genes: list[str], seed: int = FOLD_SEED) -> np.ndarray:
    """Assign genes to two stable folds without consulting expression or truth."""
    return np.asarray(
        [
            hashlib.sha256(f"{seed}|{gene}".encode()).digest()[0] & 1
            for gene in genes
        ],
        dtype=np.int8,
    )


def bounded_fit(design: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Fit an anchored support, constraining its first (target) force negative."""
    lower = np.full(design.shape[1], -np.inf)
    upper = np.full(design.shape[1], np.inf)
    upper[0] = 0.0
    return lsq_linear(
        design,
        outcome,
        bounds=(lower, upper),
        method="bvls",
        tol=1e-10,
        max_iter=200,
    ).x


def anchored_omp_factorized(
    left_train: np.ndarray,
    left_test: np.ndarray,
    codes: np.ndarray,
    outcome: np.ndarray,
    target_candidate: int,
    support_sizes: tuple[int, ...] = SUPPORT_SIZES,
) -> tuple[dict[int, np.ndarray], list[dict[str, object]]]:
    """Target-anchored OMP for a dictionary represented as ``left @ codes``."""
    requested = set(support_sizes)
    support = [int(target_candidate)]
    predictions: dict[int, np.ndarray] = {}
    trace: list[dict[str, object]] = []
    maximum = max(support_sizes)
    for size in range(1, maximum + 1):
        design = left_train @ codes[:, support]
        coefficients = bounded_fit(design, outcome)
        fitted = design @ coefficients
        residual = outcome - fitted
        if size in requested:
            predictions[size] = left_test @ codes[:, support] @ coefficients
            trace.append(
                {
                    "support_size": size,
                    "support": support.copy(),
                    "coefficients": coefficients.copy(),
                    "train_rmse": float(np.sqrt(np.mean(np.square(residual)))),
                    "target_force_at_upper_bound": bool(
                        np.isclose(coefficients[0], 0.0, atol=1e-10)
                    ),
                }
            )
        if size == maximum:
            break
        correlations = codes.T @ (left_train.T @ residual)
        correlations[np.asarray(support)] = 0.0
        support.append(int(np.argmax(np.abs(correlations))))
    return predictions, trace


def dense_mode_ceiling(
    basis_train: np.ndarray,
    basis_test: np.ndarray,
    outcome: np.ndarray,
) -> np.ndarray:
    """Cross-fitted ceiling for an arbitrary response in the rank-50 mode span."""
    coefficients = np.linalg.lstsq(basis_train, outcome, rcond=1e-8)[0]
    return basis_test @ coefficients


def fold_score(
    prediction: np.ndarray,
    stable: np.ndarray,
    truth_sign: np.ndarray,
    folds: np.ndarray,
    excluded_gene: int | None,
) -> dict[str, float | int]:
    """Score each held-out gene fold separately and pool recovery counts."""
    strong_total = 0
    recovered_total = 0
    correct_total = 0
    for fold in (0, 1):
        valid = folds == fold
        if excluded_gene is not None:
            valid[excluded_gene] = False
        strong_fold = stable & valid
        k = int(strong_fold.sum())
        if k == 0:
            continue
        indices = np.flatnonzero(valid)
        top = indices[np.argpartition(np.abs(prediction[valid]), -k)[-k:]]
        recovered = stable[top]
        correct = recovered & (np.sign(prediction[top]) == truth_sign[top])
        strong_total += k
        recovered_total += int(recovered.sum())
        correct_total += int(correct.sum())
    return {
        "n_strong": strong_total,
        "signed_recovery": correct_total / strong_total if strong_total else np.nan,
        "unsigned_recall": recovered_total / strong_total
        if strong_total
        else np.nan,
        "sign_given_recovered": correct_total / recovered_total
        if recovered_total
        else np.nan,
    }


def heldout_lfc_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    excluded_gene: int | None,
) -> dict[str, float]:
    """Score cross-fitted predictions after pooling their held-out gene folds."""
    keep = np.ones(len(truth), dtype=bool)
    if excluded_gene is not None:
        keep[excluded_gene] = False
    predicted = prediction[keep]
    observed = truth[keep]
    denominator = np.linalg.norm(predicted) * np.linalg.norm(observed)
    cosine_value = float(predicted @ observed / denominator) if denominator else 0.0
    spearman_value = (
        float(spearmanr(predicted, observed).statistic)
        if np.ptp(predicted) and np.ptp(observed)
        else 0.0
    )
    return {
        "heldout_lfc_cosine": cosine_value,
        "heldout_lfc_spearman": spearman_value if np.isfinite(spearman_value) else 0.0,
        "heldout_lfc_nmae": float(
            np.abs(predicted - observed).sum() / np.abs(observed).sum()
        ),
        "heldout_lfc_explained_energy": float(
            1 - np.square(predicted - observed).sum() / np.square(observed).sum()
        ),
    }


def fold_gene_label_null(
    prediction: np.ndarray,
    stable: np.ndarray,
    truth_sign: np.ndarray,
    folds: np.ndarray,
    excluded_genes: list[int | None],
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute held-out operator rows while preserving ranks and signs."""
    aggregate = np.zeros(N_NULL)
    eligible = np.flatnonzero(stable.sum(axis=1) >= MIN_STRONG)
    for target in eligible:
        target_null = np.zeros(N_NULL)
        total = 0
        for fold in (0, 1):
            valid = folds == fold
            excluded = excluded_genes[target]
            if excluded is not None:
                valid[excluded] = False
            k = int((stable[target] & valid).sum())
            if not k:
                continue
            values = prediction[target, valid]
            top = np.argpartition(np.abs(values), -k)[-k:]
            signs = np.sign(values[top])
            predicted_up = int((signs > 0).sum())
            predicted_down = int((signs < 0).sum())
            truth_up = int((stable[target] & valid & (truth_sign[target] > 0)).sum())
            truth_down = int(
                (stable[target] & valid & (truth_sign[target] < 0)).sum()
            )
            universe = int(valid.sum())
            up_in_up = rng.hypergeometric(
                truth_up, universe - truth_up, predicted_up, N_NULL
            )
            down_in_up = rng.hypergeometric(
                truth_down,
                universe - truth_up - truth_down,
                predicted_up - up_in_up,
            )
            remaining_down = truth_down - down_in_up
            remaining = universe - predicted_up
            down_in_down = rng.hypergeometric(
                remaining_down,
                remaining - remaining_down,
                predicted_down,
            )
            target_null += up_in_up + down_in_down
            total += k
        aggregate += target_null / total
    return aggregate / len(eligible)


def pairwise_scores(
    predictions: np.ndarray,
    stable: np.ndarray,
    truth_sign: np.ndarray,
    folds: np.ndarray,
    excluded_genes: list[int | None],
) -> np.ndarray:
    """Compare every cross-fitted direction with every target truth set."""
    n_targets = len(predictions)
    scores = np.full((n_targets, n_targets), np.nan)
    rankings = [
        [
            np.flatnonzero(folds == fold)[
                np.argsort(-np.abs(predictions[target, folds == fold]))
            ]
            for fold in (0, 1)
        ]
        for target in range(n_targets)
    ]
    for truth_target in range(n_targets):
        if stable[truth_target].sum() < MIN_STRONG:
            continue
        excluded = excluded_genes[truth_target]
        for predicted_target in range(n_targets):
            correct_total = 0
            strong_total = 0
            for fold in (0, 1):
                valid = folds == fold
                if excluded is not None:
                    valid[excluded] = False
                k = int((stable[truth_target] & valid).sum())
                if k == 0:
                    continue
                order = rankings[predicted_target][fold]
                if excluded is not None:
                    order = order[order != excluded]
                top = order[:k]
                correct_total += int(
                    np.sum(
                        stable[truth_target, top]
                        & (
                            np.sign(predictions[predicted_target, top])
                            == truth_sign[truth_target, top]
                        )
                    )
                )
                strong_total += k
            scores[truth_target, predicted_target] = correct_total / strong_total
    return scores


def derangement_null(pairwise: np.ndarray, seed: int) -> np.ndarray:
    eligible = np.flatnonzero(np.isfinite(np.diag(pairwise)))
    rng = np.random.default_rng(seed)
    target_indices = np.arange(len(pairwise))
    values = np.empty(N_NULL)
    for replicate in range(N_NULL):
        while True:
            permutation = rng.permutation(len(pairwise))
            if np.all(permutation != target_indices):
                break
        values[replicate] = np.nanmean(pairwise[eligible, permutation[eligible]])
    return values


def load_truth(
    targets: list[str], genes: list[str]
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    de = pl.read_parquet(DE_TABLE).to_pandas()
    de_genes = sorted(de["feature"].astype(str).unique())
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    de_lookup = {gene: index for index, gene in enumerate(de_genes)}
    lfc = np.empty((len(targets), len(de_genes)), dtype=np.float64)
    for target_index, target in enumerate(targets):
        table = de.loc[de["target"] == target].set_index("feature").reindex(de_genes)
        if table["log2_fold_change"].isna().any():
            raise ValueError(f"incomplete DE truth for {target}")
        lfc[target_index] = table["log2_fold_change"].to_numpy()
    stable = np.zeros_like(lfc, dtype=bool)
    rows = pd.read_csv(STRONG_TRUTH)
    rows = rows[rows["stable_strong"]]
    target_lookup = {target: index for index, target in enumerate(targets)}
    for row in rows.itertuples(index=False):
        stable[target_lookup[row.target_gene], de_lookup[row.feature]] = True
    indices = np.asarray([gene_lookup[gene] for gene in de_genes], dtype=int)
    return de_genes, indices, lfc, stable


def run_operator(
    name: str,
    left: np.ndarray,
    codes: np.ndarray,
    candidate_genes: list[str],
    target_candidates: list[int],
    targets: list[str],
    de_genes: list[str],
    lfc: np.ndarray,
    stable: np.ndarray,
    folds: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    predictions = {
        f"{name}_fixed_target": np.zeros_like(lfc, dtype=np.float32),
        **{
            f"{name}_anchored_{size}": np.zeros_like(lfc, dtype=np.float32)
            for size in SUPPORT_SIZES
        },
    }
    support_rows: list[dict[str, object]] = []
    de_lookup = {gene: index for index, gene in enumerate(de_genes)}
    for target_index, target in enumerate(targets):
        target_output = de_lookup.get(target)
        fixed = -(left @ codes[:, target_candidates[target_index]])
        predictions[f"{name}_fixed_target"][target_index] = fixed
        for test_fold in (0, 1):
            train = folds != test_fold
            test = folds == test_fold
            if target_output is not None:
                train[target_output] = False
                test[target_output] = False
            fitted, trace = anchored_omp_factorized(
                left[train],
                left[test],
                codes,
                lfc[target_index, train],
                target_candidates[target_index],
            )
            for size, values in fitted.items():
                predictions[f"{name}_anchored_{size}"][target_index, test] = values
            for step in trace:
                support = step.pop("support")
                coefficients = step.pop("coefficients")
                support_rows.append(
                    {
                        "operator": name,
                        "target_gene": target,
                        "test_fold": test_fold,
                        **step,
                        "support_genes": ";".join(
                            candidate_genes[index] for index in support
                        ),
                        "force_coefficients": ";".join(
                            f"{coefficient:.8g}" for coefficient in coefficients
                        ),
                    }
                )
        if target_index % 20 == 0 or target_index + 1 == len(targets):
            print(f"{name}: targets {target_index + 1}/{len(targets)}", flush=True)
    return predictions, support_rows


def summarize_models(
    predictions: dict[str, np.ndarray],
    targets: list[str],
    de_genes: list[str],
    lfc: np.ndarray,
    stable: np.ndarray,
    folds: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    de_lookup = {gene: index for index, gene in enumerate(de_genes)}
    excluded = [de_lookup.get(target) for target in targets]
    signs = np.sign(lfc).astype(np.int8)
    per_target_rows = []
    summary_rows = []
    for model_index, (name, prediction) in enumerate(predictions.items()):
        observed = np.full(len(targets), np.nan)
        model_rows = []
        for target_index, target in enumerate(targets):
            result = fold_score(
                prediction[target_index],
                stable[target_index],
                signs[target_index],
                folds,
                excluded[target_index],
            )
            eligible = int(result["n_strong"]) >= MIN_STRONG
            profile = heldout_lfc_metrics(
                prediction[target_index],
                lfc[target_index],
                excluded[target_index],
            )
            if eligible:
                observed[target_index] = float(result["signed_recovery"])
            model_rows.append(
                {
                    "model": name,
                    "target_gene": target,
                    "eligible": eligible,
                    **result,
                    **profile,
                }
            )
        pairwise = pairwise_scores(prediction, stable, signs, folds, excluded)
        if not np.allclose(
            observed[np.isfinite(observed)],
            np.diag(pairwise)[np.isfinite(observed)],
        ):
            raise AssertionError(f"{name}: pairwise diagonal differs from direct score")
        for target_index, row in enumerate(model_rows):
            if not row["eligible"]:
                row["wrong_target_mean"] = np.nan
                row["wrong_target_p_upper"] = np.nan
                continue
            null = np.delete(pairwise[target_index], target_index)
            null = null[np.isfinite(null)]
            row["wrong_target_mean"] = float(null.mean())
            row["wrong_target_p_upper"] = float(
                (1 + np.sum(null >= observed[target_index])) / (1 + len(null))
            )
        per_target_rows.extend(model_rows)
        finite = observed[np.isfinite(observed)]
        null = derangement_null(pairwise, FOLD_SEED + model_index)
        random_operator = fold_gene_label_null(
            prediction,
            stable,
            signs,
            folds,
            excluded,
            np.random.default_rng(FOLD_SEED + 2_000 + model_index),
        )
        rng = np.random.default_rng(FOLD_SEED + 1_000 + model_index)
        bootstrap = rng.choice(
            finite, size=(N_NULL, len(finite)), replace=True
        ).mean(axis=1)
        observed_mean = float(finite.mean())
        summary_rows.append(
            {
                "model": name,
                "eligible_targets": len(finite),
                "mean_signed_recovery": observed_mean,
                "median_signed_recovery": float(np.median(finite)),
                "bootstrap_q025": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_q975": float(np.quantile(bootstrap, 0.975)),
                "wrong_target_mean": float(null.mean()),
                "wrong_target_q95": float(np.quantile(null, 0.95)),
                "wrong_target_p_upper": float(
                    (1 + np.sum(null >= observed_mean)) / (1 + len(null))
                ),
                "random_operator_mean": float(random_operator.mean()),
                "random_operator_q95": float(np.quantile(random_operator, 0.95)),
                "random_operator_p_upper": float(
                    (1 + np.sum(random_operator >= observed_mean))
                    / (1 + len(random_operator))
                ),
                "mean_heldout_lfc_cosine": float(
                    np.mean([row["heldout_lfc_cosine"] for row in model_rows])
                ),
                "mean_heldout_lfc_spearman": float(
                    np.mean([row["heldout_lfc_spearman"] for row in model_rows])
                ),
                "mean_heldout_lfc_nmae": float(
                    np.mean([row["heldout_lfc_nmae"] for row in model_rows])
                ),
                "mean_heldout_lfc_explained_energy": float(
                    np.mean(
                        [row["heldout_lfc_explained_energy"] for row in model_rows]
                    )
                ),
            }
        )
    return pd.DataFrame(per_target_rows), pd.DataFrame(summary_rows)


def main() -> None:
    with np.load(MODEL1, allow_pickle=False) as model1, np.load(
        MODEL2, allow_pickle=False
    ) as model2:
        targets = model1["target_gene"].astype(str).tolist()
        genes = model1["gene_names"].astype(str).tolist()
        source_order = model1["source_order"].astype(int)
        empirical = model1["directions"].astype(np.float64)[:, source_order]
        components = model2["components"].astype(np.float64)[:RANK]
        singular = model2["singular_values"].astype(np.float64)[:RANK]
    de_genes, de_indices, lfc, stable = load_truth(targets, genes)
    folds = deterministic_gene_folds(de_genes)
    if min(np.bincount(folds)) < 5_000:
        raise AssertionError("deterministic gene folds are unexpectedly imbalanced")
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    target_gene_indices = [gene_lookup[target] for target in targets]

    empirical_left = empirical[de_indices]
    empirical_codes = np.eye(len(targets), dtype=np.float64)
    empirical_predictions, support_rows = run_operator(
        "empirical150",
        empirical_left,
        empirical_codes,
        targets,
        list(range(len(targets))),
        targets,
        de_genes,
        lfc,
        stable,
        folds,
    )

    n_controls = json.loads((REPORT / "model1_manifest.json").read_text())[
        "control_cells"
    ]
    eigenvalues = np.square(singular) / (n_controls - 1)
    rank50_left = components[:, de_indices].T * eigenvalues
    rank50_codes = components
    rank50_predictions, rank50_support = run_operator(
        "rank50_allgenes",
        rank50_left,
        rank50_codes,
        genes,
        target_gene_indices,
        targets,
        de_genes,
        lfc,
        stable,
        folds,
    )
    support_rows.extend(rank50_support)

    dense = np.zeros_like(lfc, dtype=np.float32)
    mode_basis = components[:, de_indices].T
    de_lookup = {gene: index for index, gene in enumerate(de_genes)}
    for target_index, target in enumerate(targets):
        excluded = de_lookup.get(target)
        for test_fold in (0, 1):
            train = folds != test_fold
            test = folds == test_fold
            if excluded is not None:
                train[excluded] = False
                test[excluded] = False
            dense[target_index, test] = dense_mode_ceiling(
                mode_basis[train], mode_basis[test], lfc[target_index, train]
            )
    predictions = {
        **empirical_predictions,
        **rank50_predictions,
        "rank50_dense_mode_ceiling": dense,
    }
    per_target, summary = summarize_models(
        predictions, targets, de_genes, lfc, stable, folds
    )
    summary["wrong_target_p_bonferroni_models"] = np.minimum(
        1.0, len(summary) * summary["wrong_target_p_upper"]
    )
    summary["random_operator_p_bonferroni_models"] = np.minimum(
        1.0, len(summary) * summary["random_operator_p_upper"]
    )
    support = pd.DataFrame(support_rows)

    outputs = {
        "summary": REPORT / "forcing_ceiling_summary.csv",
        "per_target": REPORT / "forcing_ceiling_per_target.csv",
        "support": REPORT / "forcing_ceiling_support.csv",
    }
    summary.to_csv(outputs["summary"], index=False)
    per_target.to_csv(outputs["per_target"], index=False)
    support.to_csv(outputs["support"], index=False)
    manifest = {
        "kind": "cross-fitted sparse-forcing diagnostic ceiling; not a prediction model",
        "gene_split": {
            "method": "SHA-256 of seed and gene symbol, low bit",
            "seed": FOLD_SEED,
            "fold_counts": np.bincount(folds).tolist(),
        },
        "fit": {
            "truth_coordinate": "official-panel log2 CPM fold change",
            "target_force_constraint": "non-positive",
            "support_sizes": list(SUPPORT_SIZES),
            "rank": RANK,
            "own_target_excluded_from_fit_and_score": True,
        },
        "scores": [
            "held-out-fold signed top-K stable strong-DE recovery",
            "held-out LFC cosine",
            "held-out LFC Spearman",
            "held-out LFC NMAE",
            "held-out explained energy",
        ],
        "random_operator_null": "10,000 held-out row-label permutations preserving prediction ranks, signs, and operator row norms",
        "null_replicates": N_NULL,
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [MODEL1, MODEL2, DE_TABLE, STRONG_TRUTH]
        },
        "outputs": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for name, path in outputs.items()
        },
        "software": {
            "python": platform.python_version(),
            **{name: version(name) for name in ["numpy", "pandas", "polars", "scipy"]},
        },
    }
    (REPORT / "forcing_ceiling_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(summary.sort_values("mean_signed_recovery", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
