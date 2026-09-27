"""Fit Poisson-lognormal response columns by factorial moments.

For depth-corrected rates r_cg = Y_cg / s_c, Poisson sampling gives

    E[Y_cg (Y_ct - 1[g=t]) / s_c^2] = E[exp(x_cg + x_ct)].

If x is Gaussian, its covariance column is therefore the log of this
factorial second moment divided by the product of the mean rates.  This fits
only the requested response columns and needs no per-cell latent E-step.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scripts.linear_response.kernel import normalize_response_columns

ROOT = Path(__file__).resolve().parents[2]
H1 = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
EMPIRICAL = ROOT / "data/derived/linear_response/ladder/empirical.npz"
STATE_MODEL = (
    ROOT
    / "data/derived/linear_response/state_diversity/state_balanced_covariance.npz"
)
STRICT_GUIDES = ROOT / "reports/linear-response-three-models/h1_strict_controls.csv"
DERIVED = ROOT / "data/derived/linear_response/sequencing_model"
REPORT = ROOT / "reports/linear-response-sequencing-model"
ARTIFACT = DERIVED / "factorial_moment_response.npz"
CHUNK_SIZE = 512
MODEL_NAMES = ["control_factorial_pln", "all_combined_factorial_pln"]


@dataclass
class FactorialMoments:
    """Cell-equal first and factorial-second rate moments."""

    n: int
    rate_sum: np.ndarray
    second_sum: np.ndarray

    @classmethod
    def zeros(cls, genes: int, targets: int) -> FactorialMoments:
        return cls(0, np.zeros(genes), np.zeros((genes, targets)))

    def update(
        self,
        raw: np.ndarray,
        totals: np.ndarray,
        target_indices: np.ndarray,
    ) -> None:
        rates = raw / totals[:, None]
        self.n += len(raw)
        self.rate_sum += rates.sum(axis=0)
        self.second_sum += rates.T @ rates[:, target_indices]
        self.second_sum[target_indices, np.arange(len(target_indices))] -= (
            raw[:, target_indices] / np.square(totals[:, None])
        ).sum(axis=0)

    def add(self, other: FactorialMoments) -> FactorialMoments:
        return FactorialMoments(
            self.n + other.n,
            self.rate_sum + other.rate_sum,
            self.second_sum + other.second_sum,
        )

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        return self.rate_sum / self.n, self.second_sum / self.n


def lognormal_covariance_columns(
    mean_rate: np.ndarray,
    factorial_second: np.ndarray,
    target_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return latent log-rate covariance columns and their moment ratios."""
    denominator = np.outer(mean_rate, mean_rate[target_indices])
    ratio = factorial_second / denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        covariance = np.log(ratio)
    return covariance, ratio


def state_halves(
    batches: np.ndarray, rows: np.ndarray, state_index: int
) -> np.ndarray:
    """Deterministically split each state's cells within sequencing batches."""
    halves = np.empty(len(rows), dtype=np.int8)
    local_batches = batches[rows]
    for batch_index, batch in enumerate(sorted(set(local_batches))):
        positions = np.flatnonzero(local_batches == batch)
        shuffled = np.random.default_rng(
            np.random.SeedSequence([0, 211, state_index, batch_index])
        ).permutation(positions)
        halves[shuffled[::2]] = 0
        halves[shuffled[1::2]] = 1
    return halves


def fit_state(
    data: ad.AnnData,
    rows: np.ndarray,
    halves: np.ndarray,
    fit_indices: np.ndarray,
    target_indices: np.ndarray,
) -> list[FactorialMoments]:
    moments = [
        FactorialMoments.zeros(len(fit_indices), len(target_indices))
        for _ in range(2)
    ]
    for start in range(0, len(rows), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(rows))
        raw_full = data.X[rows[start:stop]].tocsr().astype(np.float64)
        totals = np.asarray(raw_full.sum(axis=1)).ravel()
        raw = raw_full[:, fit_indices].toarray()
        split = halves[start:stop]
        for half in (0, 1):
            take = split == half
            moments[half].update(raw[take], totals[take], target_indices)
    return moments


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    started = time.perf_counter()
    strict_guides = set(pd.read_csv(STRICT_GUIDES)["guide_id"].astype(str))
    with np.load(EMPIRICAL, allow_pickle=False) as empirical, np.load(
        STATE_MODEL, allow_pickle=False
    ) as state_model:
        targets = empirical["target_gene"].astype(str)
        output_genes = empirical["output_gene"].astype(str)
        fit_genes = empirical["fit_gene"].astype(str)
        fit_indices = empirical["full_gene_index"].astype(int)
        target_indices = empirical["target_fit_index"].astype(int)
        empirical_response = empirical["response"].astype(np.float64)
        state_models = state_model["model"].astype(str).tolist()
        combined_response = state_model["response"][
            state_models.index("all_combined")
        ].astype(np.float64)

    data = ad.read_h5ad(H1, backed="r")
    try:
        labels = data.obs["target_gene"].astype(str).to_numpy()
        guides = data.obs["guide_id"].astype(str).to_numpy()
        batches = data.obs["batch"].astype(str).to_numpy()
        perturbations = sorted(set(labels) - {"non-targeting"})
        states = ["non-targeting", *perturbations]
        rows_by_state = {
            "non-targeting": np.flatnonzero(
                (labels == "non-targeting") & np.isin(guides, list(strict_guides))
            ),
            **{state: np.flatnonzero(labels == state) for state in perturbations},
        }

        balanced_mean = [np.zeros(len(fit_genes)) for _ in range(3)]
        balanced_second = [
            np.zeros((len(fit_genes), len(targets))) for _ in range(3)
        ]
        control_moments: list[FactorialMoments] | None = None
        state_rows = []
        for state_index, state in enumerate(states):
            rows = rows_by_state[state]
            halves = state_halves(batches, rows, state_index)
            split_moments = fit_state(
                data, rows, halves, fit_indices, target_indices
            )
            full_moments = split_moments[0].add(split_moments[1])
            all_moments = [full_moments, *split_moments]
            if state == "non-targeting":
                control_moments = all_moments
            state_record = {
                "state": state,
                "n_cells": full_moments.n,
                "half0_cells": split_moments[0].n,
                "half1_cells": split_moments[1].n,
            }
            for split_index, moments in enumerate(all_moments):
                mean_rate, second = moments.finish()
                balanced_mean[split_index] += mean_rate / len(states)
                balanced_second[split_index] += second / len(states)
                if split_index == 0:
                    covariance, ratio = lognormal_covariance_columns(
                        mean_rate, second, target_indices
                    )
                    state_record["invalid_factorial_ratios"] = int(
                        np.count_nonzero(~np.isfinite(covariance))
                    )
                    state_record["negative_target_variances"] = int(
                        np.count_nonzero(
                            covariance[target_indices, np.arange(len(targets))] <= 0
                        )
                    )
                    state_record["minimum_factorial_ratio"] = float(ratio.min())
            state_rows.append(state_record)
            print(
                f"factorial moments {state_index + 1}/{len(states)}: "
                f"{state} ({len(rows)} cells)",
                flush=True,
            )
    finally:
        data.file.close()

    if control_moments is None:
        raise AssertionError("strict control state was not fitted")

    covariance_sets = []
    response_sets = []
    diagonal_sets = []
    ratio_sets = []
    mean_rate_sets = []
    split_responses = []
    model_moments = [
        [moments.finish() for moments in control_moments],
        list(zip(balanced_mean, balanced_second, strict=True)),
    ]
    for splits in model_moments:
        model_covariances = []
        model_responses = []
        for mean_rate, second in splits:
            covariance, ratio = lognormal_covariance_columns(
                mean_rate, second, target_indices
            )
            response, diagonal = normalize_response_columns(
                covariance, target_indices
            )
            model_covariances.append(covariance)
            model_responses.append(response.T)
            if len(model_covariances) == 1:
                mean_rate_sets.append(mean_rate)
                covariance_sets.append(covariance)
                response_sets.append(response.T)
                diagonal_sets.append(diagonal)
                ratio_sets.append(ratio)
        split_responses.append(model_responses[1:])

    covariance_sets = np.asarray(covariance_sets)
    response_sets = np.asarray(response_sets)
    diagonal_sets = np.asarray(diagonal_sets)
    ratio_sets = np.asarray(ratio_sets)
    mean_rate_sets = np.asarray(mean_rate_sets)
    if not np.isfinite(covariance_sets).all():
        raise RuntimeError("final factorial-moment covariance contains invalid values")
    if np.any(diagonal_sets <= 0):
        raise RuntimeError("final factorial-moment target variance is not positive")
    comparators = [empirical_response, combined_response]
    diagnostics = []
    for model_index, model in enumerate(MODEL_NAMES):
        for target_index, target in enumerate(targets):
            downstream = np.arange(len(fit_genes)) != target_indices[target_index]
            diagnostics.append(
                {
                    "model": model,
                    "target_gene": target,
                    "latent_variance": diagonal_sets[model_index, target_index],
                    "minimum_factorial_ratio": ratio_sets[
                        model_index, :, target_index
                    ].min(),
                    "downstream_norm": np.linalg.norm(
                        response_sets[model_index, target_index, downstream]
                    ),
                    "split_half_downstream_cosine": cosine(
                        split_responses[model_index][0][target_index, downstream],
                        split_responses[model_index][1][target_index, downstream],
                    ),
                    "log1cp10k_downstream_cosine": cosine(
                        response_sets[model_index, target_index, downstream],
                        comparators[model_index][target_index, downstream],
                    ),
                }
            )

    DERIVED.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ARTIFACT,
        model=np.asarray(MODEL_NAMES),
        response=response_sets.astype(np.float32),
        covariance=covariance_sets.astype(np.float32),
        diagonal=diagonal_sets.astype(np.float32),
        mean_rate=mean_rate_sets.astype(np.float32),
        factorial_ratio=ratio_sets.astype(np.float32),
        target_gene=targets,
        output_gene=output_genes,
        fit_gene=fit_genes,
        full_gene_index=fit_indices,
        target_fit_index=target_indices,
    )
    state_table = REPORT / "state_factorial_moment_diagnostics.csv"
    direction_table = REPORT / "direction_diagnostics.csv"
    summary_table = REPORT / "fit_summary.csv"
    pd.DataFrame(state_rows).to_csv(state_table, index=False)
    pd.DataFrame(diagnostics).to_csv(direction_table, index=False)
    summary = (
        pd.DataFrame(diagnostics)
        .groupby("model", sort=False)
        .agg(
            median_latent_variance=("latent_variance", "median"),
            median_downstream_norm=("downstream_norm", "median"),
            median_split_half_cosine=("split_half_downstream_cosine", "median"),
            median_log1cp10k_cosine=("log1cp10k_downstream_cosine", "median"),
        )
        .reset_index()
    )
    summary["runtime_seconds"] = time.perf_counter() - started
    summary.to_csv(summary_table, index=False)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "model": "factorial-moment Poisson-lognormal target-column estimator",
        "formula": "Sigma_gt = log(E[Y_g(Y_t-delta_gt)/s^2] / (E[Y_g/s] E[Y_t/s]))",
        "cell_size_factor": "observed full-panel UMI total",
        "genes": len(fit_genes),
        "targets": len(targets),
        "states": {
            "control": "strict 26-guide control pool",
            "all_combined": "equal mean of 151 state-specific first and factorial-second moments",
        },
        "fit": "two deterministic within-batch cell halves; no latent E-step and no iterative optimizer",
        "mixture_caveat": "the balanced all-state result is a moment-matched lognormal representation of a mixture, not a claim that the mixture is lognormal",
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [H1, EMPIRICAL, STATE_MODEL, STRICT_GUIDES]
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [ARTIFACT, state_table, direction_table, summary_table]
        },
        "software": {
            "python": platform.python_version(),
            **{name: version(name) for name in ["anndata", "numpy", "pandas"]},
        },
    }
    (REPORT / "fit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
