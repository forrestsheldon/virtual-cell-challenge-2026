"""Fit and validate the rank-50 Poisson-lognormal H1 diagnostic.

The global fit optimizes a differentiable Laplace approximation. Per-cell latent
modes use the exact positive-definite Hessian. A predeclared posterior-predictive
gate runs before any expected-profile artifact is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import pearsonr, spearmanr
from sklearn.utils.extmath import randomized_svd

from scripts.linear_response.kernel import (
    BULK_TARGET_SUM,
    CELL_TARGET_SUM,
    intended_target_shift,
    log1cp10k,
    log_pseudobulk,
    sample_balanced_controls,
)

jax.config.update("jax_enable_x64", False)

ROOT = Path(__file__).resolve().parents[2]
H1_PATH = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
REPORT_DIR = ROOT / "reports/linear-response-three-models"
DERIVED_DIR = ROOT / "data/derived/linear_response"
TARGET_TABLE = REPORT_DIR / "h1_target_counts.csv"
STRICT_CONTROLS = REPORT_DIR / "h1_strict_controls.csv"
CALIBRATION = REPORT_DIR / "model1_knockdown_calibration.csv"
INITIALIZATION = DERIVED_DIR / "model3_initialization.npz"
RANK = 50
BATCH_SIZE = 32
TRAIN_NEWTON_STEPS = 30
INFERENCE_NEWTON_STEPS = 30
MAX_NEWTON_STEP = 2.0
LEARNING_RATE = 2e-6
GRADIENT_CLIP_NORM = 100.0
LOADING_PENALTY = 1e-4
N_STARTS = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def normalize_population(m: jax.Array, loadings: jax.Array) -> jax.Array:
    correction = jax.scipy.special.logsumexp(m + 0.5 * jnp.square(loadings).sum(1))
    return m - correction


def _latent_modes(
    m: jax.Array,
    loadings: jax.Array,
    counts: jax.Array,
    totals: jax.Array,
    steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return Newton modes, rates, and positive posterior Hessians."""
    modes = jnp.zeros((counts.shape[0], loadings.shape[1]), dtype=loadings.dtype)
    identity = jnp.eye(loadings.shape[1], dtype=loadings.dtype)

    def update(current: jax.Array, _: None) -> tuple[jax.Array, None]:
        linear = m[None, :] + current @ loadings.T
        rates = totals[:, None] * jnp.exp(linear)
        gradient = current - (counts - rates) @ loadings
        hessian = identity[None, :, :] + jnp.einsum(
            "bg,gk,gl->bkl", rates, loadings, loadings
        )
        step = jnp.linalg.solve(hessian, gradient[..., None])[..., 0]
        step_norm = jnp.linalg.norm(step, axis=1, keepdims=True)
        step *= jnp.minimum(1.0, MAX_NEWTON_STEP / jnp.maximum(step_norm, 1e-12))
        return current - step, None

    modes, _ = jax.lax.scan(update, modes, None, length=steps)
    linear = m[None, :] + modes @ loadings.T
    rates = totals[:, None] * jnp.exp(linear)
    hessian = identity[None, :, :] + jnp.einsum(
        "bg,gk,gl->bkl", rates, loadings, loadings
    )
    return modes, rates, hessian


def training_latent_modes(
    m: jax.Array,
    loadings: jax.Array,
    counts: jax.Array,
    totals: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return _latent_modes(m, loadings, counts, totals, TRAIN_NEWTON_STEPS)


def inference_latent_modes(
    m: jax.Array,
    loadings: jax.Array,
    counts: jax.Array,
    totals: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return _latent_modes(m, loadings, counts, totals, INFERENCE_NEWTON_STEPS)


def laplace_loss(
    parameters: dict[str, jax.Array], counts: jax.Array, totals: jax.Array
) -> jax.Array:
    m = normalize_population(parameters["m"], parameters["loadings"])
    loadings = parameters["loadings"]
    modes, rates, hessian = training_latent_modes(m, loadings, counts, totals)
    linear = m[None, :] + modes @ loadings.T
    negative_joint = (
        rates.sum(1) - (counts * linear).sum(1) + 0.5 * jnp.square(modes).sum(1)
    )
    correction = 0.5 * jnp.linalg.slogdet(hessian)[1]
    penalty = LOADING_PENALTY * jnp.square(loadings).sum()
    return jnp.mean(negative_joint + correction) + penalty


def inference_laplace_loss(
    parameters: dict[str, jax.Array], counts: jax.Array, totals: jax.Array
) -> jax.Array:
    """Laplace loss evaluated at the converged final-inference modes."""
    m = normalize_population(parameters["m"], parameters["loadings"])
    loadings = parameters["loadings"]
    modes, rates, hessian = inference_latent_modes(m, loadings, counts, totals)
    linear = m[None, :] + modes @ loadings.T
    negative_joint = (
        rates.sum(1) - (counts * linear).sum(1) + 0.5 * jnp.square(modes).sum(1)
    )
    correction = 0.5 * jnp.linalg.slogdet(hessian)[1]
    penalty = LOADING_PENALTY * jnp.square(loadings).sum()
    return jnp.mean(negative_joint + correction) + penalty


@jax.jit
def train_step(
    parameters: dict[str, jax.Array],
    optimizer_state: optax.OptState,
    counts: jax.Array,
    totals: jax.Array,
) -> tuple[dict[str, jax.Array], optax.OptState, jax.Array]:
    loss, gradients = jax.value_and_grad(laplace_loss)(parameters, counts, totals)
    updates, optimizer_state = OPTIMIZER.update(gradients, optimizer_state, parameters)
    parameters = optax.apply_updates(parameters, updates)
    parameters["m"] = normalize_population(parameters["m"], parameters["loadings"])
    return parameters, optimizer_state, loss


OPTIMIZER = optax.chain(
    optax.clip_by_global_norm(GRADIENT_CLIP_NORM),
    optax.adam(LEARNING_RATE),
)
INFER = jax.jit(inference_latent_modes)
LOSS = jax.jit(laplace_loss)
INFERENCE_LOSS = jax.jit(inference_laplace_loss)


def laplace_mode_numpy(
    counts: np.ndarray,
    total: float,
    m: np.ndarray,
    loadings: np.ndarray,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    mode = np.zeros(loadings.shape[1])
    identity = np.eye(loadings.shape[1])
    for _ in range(100):
        rates = total * np.exp(m + loadings @ mode)
        gradient = mode - loadings.T @ (counts - rates)
        hessian = identity + loadings.T @ (rates[:, None] * loadings)
        step = np.linalg.solve(hessian, gradient)
        mode -= step
        if np.linalg.norm(step) < tolerance:
            break
    return mode, np.linalg.inv(hessian)


def tiny_laplace_validation() -> pd.DataFrame:
    """Compare a one-dimensional Laplace posterior with numerical quadrature."""
    m = np.log(np.array([0.25, 0.35, 0.40])) - 0.5 * np.array([0.04, 0.01, 0.0225])
    loadings = np.array([[0.20], [-0.10], [0.15]])
    counts = np.array([1260.0, 1660.0, 2080.0])
    total = counts.sum()
    mode, covariance = laplace_mode_numpy(counts, total, m, loadings)
    grid = np.linspace(mode.item() - 1.0, mode.item() + 1.0, 200_001)
    linear = m[None, :] + grid[:, None] * loadings[:, 0]
    log_density = (
        (counts[None, :] * linear).sum(1)
        - total * np.exp(linear).sum(1)
        - 0.5 * np.square(grid)
    )
    weights = np.exp(log_density - log_density.max())
    weights /= weights.sum()
    exact_mean = float(weights @ grid)
    exact_variance = float(weights @ np.square(grid - exact_mean))
    return pd.DataFrame(
        [
            {
                "laplace_mode": mode.item(),
                "quadrature_mean": exact_mean,
                "absolute_mean_error": abs(mode.item() - exact_mean),
                "laplace_variance": covariance.item(),
                "quadrature_variance": exact_variance,
                "relative_variance_error": abs(covariance.item() / exact_variance - 1),
            }
        ]
    )


def control_split(
    obs: pd.DataFrame, strict_guides: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    heldout_guides = sorted(
        np.random.default_rng(0).choice(strict_guides, size=2, replace=False).tolist()
    )
    strict = obs["target_gene"].astype(str).eq("non-targeting") & obs[
        "guide_id"
    ].astype(str).isin(strict_guides)
    heldout = strict & obs["guide_id"].astype(str).isin(heldout_guides)
    return (
        np.flatnonzero((strict & ~heldout).to_numpy()),
        np.flatnonzero(heldout.to_numpy()),
        heldout_guides,
    )


def initialize_model(
    data: ad.AnnData, train_rows: np.ndarray, work_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sums = np.zeros(data.n_vars)
    raw_sum = np.zeros(data.n_vars)
    for start in range(0, len(train_rows), 512):
        raw = data.X[train_rows[start : start + 512]].tocsr()
        sums += log1cp10k(raw).sum(0)
        raw_sum += np.asarray(raw.sum(0)).ravel()
    mean = sums / len(train_rows)
    matrix = np.memmap(
        work_path, dtype=np.float32, mode="w+", shape=(len(train_rows), data.n_vars)
    )
    for start in range(0, len(train_rows), 512):
        stop = min(start + 512, len(train_rows))
        matrix[start:stop] = log1cp10k(data.X[train_rows[start:stop]].tocsr()) - mean
    matrix.flush()
    _, singular, components = randomized_svd(
        matrix,
        n_components=RANK,
        n_oversamples=20,
        n_iter=2,
        random_state=0,
        flip_sign=True,
    )
    del matrix
    work_path.unlink(missing_ok=True)
    loadings = components.T * (singular / np.sqrt(len(train_rows) - 1))[None, :]
    q = np.maximum(raw_sum, 0.5)
    q /= q.sum()
    m = np.log(q) - 0.5 * np.square(loadings).sum(1)
    m -= np.log(np.exp(m + 0.5 * np.square(loadings).sum(1)).sum())
    return m.astype(np.float32), loadings.astype(np.float32), q


def dense_batch(data: ad.AnnData, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(rows)
    counts = data.X[ordered].toarray().astype(np.float32)
    return counts, counts.sum(1)


def fit_start(
    data: ad.AnnData,
    train_rows: np.ndarray,
    initial_m: np.ndarray,
    initial_loadings: np.ndarray,
    start_index: int,
    max_batches: int | None = None,
) -> tuple[dict[str, np.ndarray], float]:
    rng = np.random.default_rng(start_index)
    loadings = initial_loadings.copy()
    if start_index:
        scale = np.std(loadings, axis=0, keepdims=True)
        loadings += rng.normal(size=loadings.shape).astype(np.float32) * scale * 0.05
    parameters = {
        "m": jnp.asarray(initial_m),
        "loadings": jnp.asarray(loadings),
    }
    optimizer_state = OPTIMIZER.init(parameters)
    shuffled = rng.permutation(train_rows)
    losses = []
    n_batches = len(shuffled) // BATCH_SIZE
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)
    for batch_index in range(n_batches):
        rows = shuffled[batch_index * BATCH_SIZE : (batch_index + 1) * BATCH_SIZE]
        counts, totals = dense_batch(data, rows)
        parameters, optimizer_state, loss = train_step(
            parameters, optimizer_state, jnp.asarray(counts), jnp.asarray(totals)
        )
        value = float(loss)
        if not np.isfinite(value):
            raise FloatingPointError(
                f"Model 3 start {start_index} became non-finite at batch {batch_index}"
            )
        losses.append(value)
        if (batch_index + 1) % 100 == 0:
            print(
                f"Model 3 start {start_index + 1}/{N_STARTS}: "
                f"batch {batch_index + 1}/{n_batches}, loss {np.mean(losses[-100:]):.3f}"
            )
    fitted = {name: np.asarray(value) for name, value in parameters.items()}
    return fitted, float(np.mean(losses[-100:]))


def evaluate_loss(
    data: ad.AnnData,
    rows: np.ndarray,
    parameters: dict[str, np.ndarray],
    *,
    converged_modes: bool = False,
) -> float:
    losses = []
    scorer = INFERENCE_LOSS if converged_modes else LOSS
    for start in range(0, len(rows) - BATCH_SIZE + 1, BATCH_SIZE):
        counts, totals = dense_batch(data, rows[start : start + BATCH_SIZE])
        losses.append(
            float(
                scorer(
                    {name: jnp.asarray(value) for name, value in parameters.items()},
                    jnp.asarray(counts),
                    jnp.asarray(totals),
                )
            )
        )
    return float(np.mean(losses))


def posterior_predictive_diagnostics(
    data: ad.AnnData,
    rows: np.ndarray,
    m: np.ndarray,
    loadings: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    observed_sum = np.zeros(data.n_vars)
    observed_square_sum = np.zeros(data.n_vars)
    expected_sum = np.zeros(data.n_vars)
    expected_square_sum = np.zeros(data.n_vars)
    predicted_zero_sum = np.zeros(data.n_vars)
    observed_zero_count = 0
    rate_sums = []
    mode_gradient_norms = []
    count = 0
    parameters = (jnp.asarray(m), jnp.asarray(loadings))
    for start in range(0, len(rows), BATCH_SIZE):
        counts, totals = dense_batch(data, rows[start : start + BATCH_SIZE])
        modes, rates, _ = INFER(
            parameters[0], parameters[1], jnp.asarray(counts), jnp.asarray(totals)
        )
        modes = np.asarray(modes)
        rates = np.asarray(rates)
        gradients = modes - (counts - rates) @ loadings
        mode_gradient_norms.extend(np.linalg.norm(gradients, axis=1).tolist())
        observed_sum += counts.sum(0)
        observed_square_sum += np.square(counts).sum(0)
        observed_zero_count += int((counts == 0).sum())
        expected_sum += rates.sum(0)
        expected_square_sum += (rates + np.square(rates)).sum(0)
        predicted_zero_sum += np.exp(-rates).sum(0)
        rate_sums.extend((rates.sum(1) / totals).tolist())
        count += len(counts)
    observed_mean = observed_sum / count
    observed_variance = observed_square_sum / count - np.square(observed_mean)
    expected_mean = expected_sum / count
    expected_variance = expected_square_sum / count - np.square(expected_mean)
    observed_zero = observed_zero_count / (count * data.n_vars)
    predicted_zero = float((predicted_zero_sum / count).mean())
    rows_out = [
        {"metric": "observed_gene_zero_fraction", "value": observed_zero},
        {"metric": "posterior_predictive_gene_zero_fraction", "value": predicted_zero},
        {
            "metric": "mean_log1p_gene_pearson",
            "value": pearsonr(
                np.log1p(observed_mean), np.log1p(expected_mean)
            ).statistic,
        },
        {
            "metric": "variance_log1p_gene_spearman",
            "value": spearmanr(
                np.log1p(observed_variance), np.log1p(expected_variance)
            ).statistic,
        },
        {"metric": "posterior_total_rate_ratio_median", "value": np.median(rate_sums)},
        {
            "metric": "posterior_total_rate_ratio_q01",
            "value": np.quantile(rate_sums, 0.01),
        },
        {
            "metric": "posterior_total_rate_ratio_q99",
            "value": np.quantile(rate_sums, 0.99),
        },
        {
            "metric": "mode_gradient_norm_median",
            "value": np.median(mode_gradient_norms),
        },
        {
            "metric": "mode_gradient_norm_q99",
            "value": np.quantile(mode_gradient_norms, 0.99),
        },
    ]
    return pd.DataFrame(rows_out), np.asarray(rate_sums)


def source_expected_sum(
    data: ad.AnnData,
    rows: np.ndarray,
    m: np.ndarray,
    loadings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    expected = np.zeros(data.n_vars)
    raw_sum = np.zeros(data.n_vars)
    for start in range(0, len(rows), BATCH_SIZE):
        counts, totals = dense_batch(data, rows[start : start + BATCH_SIZE])
        _, rates, _ = INFER(
            jnp.asarray(m),
            jnp.asarray(loadings),
            jnp.asarray(counts),
            jnp.asarray(totals),
        )
        expected += np.asarray(rates).sum(0)
        raw_sum += counts.sum(0)
    return expected, raw_sum


def solve_amplitude(
    baseline_sum: np.ndarray,
    direction: np.ndarray,
    gene_index: int,
    target_shift: float,
) -> tuple[float, float, np.ndarray]:
    baseline_profile = log_pseudobulk(baseline_sum)

    def shifted(amplitude: float) -> np.ndarray:
        exponent = np.clip(amplitude * direction, -80, 80)
        return baseline_sum * np.exp(exponent)

    def residual(amplitude: float) -> float:
        return (
            log_pseudobulk(shifted(amplitude))[gene_index]
            - baseline_profile[gene_index]
            - target_shift
        )

    lower = -1.0
    while residual(lower) > 0 and lower > -1e6:
        lower *= 2
    if residual(lower) > 0:
        raise ValueError("could not bracket Model 3 amplitude")
    amplitude = float(brentq(residual, lower, 0.0, xtol=1e-10, rtol=1e-10))
    realized = target_shift + residual(amplitude)
    return amplitude, float(realized), shifted(amplitude)


def main(args: argparse.Namespace) -> None:
    tiny = tiny_laplace_validation()
    if (
        tiny["absolute_mean_error"].item() > 0.01
        or tiny["relative_variance_error"].item() > 0.05
    ):
        raise RuntimeError("tiny Laplace validation failed")

    target_table = pd.read_csv(TARGET_TABLE)
    benchmark = target_table[target_table["benchmark_candidate"]].copy()
    strict_guides = pd.read_csv(STRICT_CONTROLS)["guide_id"].astype(str).tolist()
    calibration = pd.read_csv(CALIBRATION).set_index("target_gene")
    work_path = DERIVED_DIR / "model3_centered_train.f32"
    data = ad.read_h5ad(H1_PATH, backed="r")
    try:
        genes = data.var_names.astype(str).tolist()
        gene_lookup = {gene: index for index, gene in enumerate(genes)}
        train_rows, heldout_rows, heldout_guides = control_split(
            data.obs, strict_guides
        )
        if INITIALIZATION.exists():
            with np.load(INITIALIZATION) as saved:
                initial_m = saved["m"]
                initial_loadings = saved["loadings"]
                saved_train_rows = saved["train_rows"]
                saved_heldout_rows = saved["heldout_rows"]
            if not np.array_equal(saved_train_rows, train_rows) or not np.array_equal(
                saved_heldout_rows, heldout_rows
            ):
                raise ValueError("cached Model 3 initialization has a different split")
        else:
            initial_m, initial_loadings, _ = initialize_model(
                data, train_rows, work_path
            )
            np.savez_compressed(
                INITIALIZATION,
                m=initial_m,
                loadings=initial_loadings,
                train_rows=train_rows,
                heldout_rows=heldout_rows,
            )
        if args.smoke_batches is not None:
            initial = {"m": initial_m, "loadings": initial_loadings}
            initial_loss = evaluate_loss(
                data, heldout_rows, initial, converged_modes=True
            )
            fitted, training_loss = fit_start(
                data,
                train_rows,
                initial_m,
                initial_loadings,
                0,
                max_batches=args.smoke_batches,
            )
            heldout_loss = evaluate_loss(
                data, heldout_rows, fitted, converged_modes=True
            )
            print(
                json.dumps(
                    {
                        "smoke_batches": args.smoke_batches,
                        "training_tail_laplace_nll": training_loss,
                        "initial_heldout_laplace_nll": initial_loss,
                        "fitted_heldout_laplace_nll": heldout_loss,
                    },
                    indent=2,
                )
            )
            return
        initial_heldout_loss = evaluate_loss(
            data,
            heldout_rows,
            {"m": initial_m, "loadings": initial_loadings},
            converged_modes=True,
        )
        if args.reuse_fit:
            with np.load(DERIVED_DIR / "model3_fit.npz") as saved:
                m = saved["m"]
                loadings = saved["loadings"]
            starts = pd.read_csv(REPORT_DIR / "model3_fit_starts.csv").to_dict(
                orient="records"
            )
            for row in starts:
                if "heldout_laplace_nll" in row:
                    row["heldout_training_approx_nll"] = row.pop("heldout_laplace_nll")
            best_index = int(
                np.argmin([row["heldout_training_approx_nll"] for row in starts])
            )
        else:
            starts = []
            fits = []
            for start_index in range(N_STARTS):
                fitted, training_loss = fit_start(
                    data, train_rows, initial_m, initial_loadings, start_index
                )
                heldout_loss = evaluate_loss(data, heldout_rows, fitted)
                fits.append(fitted)
                starts.append(
                    {
                        "start": start_index,
                        "training_tail_laplace_nll": training_loss,
                        "heldout_training_approx_nll": heldout_loss,
                    }
                )
            best_index = int(
                np.argmin([row["heldout_training_approx_nll"] for row in starts])
            )
            m = fits[best_index]["m"]
            loadings = fits[best_index]["loadings"]
        m = np.asarray(normalize_population(jnp.asarray(m), jnp.asarray(loadings)))
        selected_heldout_loss = evaluate_loss(
            data,
            heldout_rows,
            {"m": m, "loadings": loadings},
            converged_modes=True,
        )
        diagnostics, _ = posterior_predictive_diagnostics(
            data, heldout_rows, m, loadings
        )
        q = np.exp(m + 0.5 * np.square(loadings).sum(1))
        prior = np.random.default_rng(0).normal(size=(2_000, RANK)).astype(np.float32)
        prior_sums = []
        for start in range(0, len(prior), 50):
            prior_sums.extend(
                np.exp(m[None, :] + prior[start : start + 50] @ loadings.T)
                .sum(1)
                .tolist()
            )
        diagnostics = pd.concat(
            [
                diagnostics,
                pd.DataFrame(
                    [
                        {"metric": "sum_q", "value": q.sum()},
                        {
                            "metric": "q_transpose_L_norm",
                            "value": np.linalg.norm(q @ loadings),
                        },
                        {
                            "metric": "prior_total_rate_ratio_q01",
                            "value": np.quantile(prior_sums, 0.01),
                        },
                        {
                            "metric": "prior_total_rate_ratio_median",
                            "value": np.median(prior_sums),
                        },
                        {
                            "metric": "prior_total_rate_ratio_q99",
                            "value": np.quantile(prior_sums, 0.99),
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )
        diagnostic_lookup = diagnostics.set_index("metric")["value"]
        gate = {
            "population_normalized": bool(abs(diagnostic_lookup["sum_q"] - 1) < 1e-5),
            "prior_total_rate_q01_above_0.9": bool(
                diagnostic_lookup["prior_total_rate_ratio_q01"] > 0.9
            ),
            "prior_total_rate_q99_below_1.1": bool(
                diagnostic_lookup["prior_total_rate_ratio_q99"] < 1.1
            ),
            "posterior_total_rate_q01_above_0.9": bool(
                diagnostic_lookup["posterior_total_rate_ratio_q01"] > 0.9
            ),
            "posterior_total_rate_q99_below_1.1": bool(
                diagnostic_lookup["posterior_total_rate_ratio_q99"] < 1.1
            ),
            "mean_correlation_above_0.95": bool(
                diagnostic_lookup["mean_log1p_gene_pearson"] > 0.95
            ),
            "variance_correlation_above_0.5": bool(
                diagnostic_lookup["variance_log1p_gene_spearman"] > 0.5
            ),
            "zero_fraction_abs_error_below_0.05": bool(
                abs(
                    diagnostic_lookup["posterior_predictive_gene_zero_fraction"]
                    - diagnostic_lookup["observed_gene_zero_fraction"]
                )
                < 0.05
            ),
            "mode_gradient_q99_below_1e-3": bool(
                diagnostic_lookup["mode_gradient_norm_q99"] < 1e-3
            ),
            "heldout_likelihood_improves": bool(
                selected_heldout_loss < initial_heldout_loss
            ),
        }

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        DERIVED_DIR.mkdir(parents=True, exist_ok=True)
        tiny.to_csv(REPORT_DIR / "model3_tiny_laplace_validation.csv", index=False)
        pd.DataFrame(starts).to_csv(REPORT_DIR / "model3_fit_starts.csv", index=False)
        diagnostics.to_csv(REPORT_DIR / "model3_posterior_predictive.csv", index=False)
        pd.DataFrame(
            [{"check": name, "passed": passed} for name, passed in gate.items()]
        ).to_csv(REPORT_DIR / "model3_validation_gate.csv", index=False)
        np.savez_compressed(
            DERIVED_DIR / "model3_fit.npz",
            m=m,
            loadings=loadings,
            heldout_guides=np.asarray(heldout_guides),
        )
        fit_manifest = {
            "model": "rank-50 Poisson-lognormal with observed-total offsets and Laplace-MAP inference",
            "status": "validation_passed"
            if all(gate.values())
            else "stopped_at_validation_gate",
            "truth_cells_read": False,
            "rank": RANK,
            "heldout_guides": heldout_guides,
            "best_start": best_index,
            "initial_heldout_laplace_nll": initial_heldout_loss,
            "selected_heldout_laplace_nll": selected_heldout_loss,
            "start_selection_metric": "six-step training-approximation held-out Laplace NLL",
            "training": {
                "starts": N_STARTS,
                "epochs_per_start": 1,
                "batch_size": BATCH_SIZE,
                "training_newton_steps": TRAIN_NEWTON_STEPS,
                "inference_newton_steps": INFERENCE_NEWTON_STEPS,
                "maximum_newton_step": MAX_NEWTON_STEP,
                "learning_rate": LEARNING_RATE,
                "gradient_clip_norm": GRADIENT_CLIP_NORM,
                "loading_penalty": LOADING_PENALTY,
            },
            "population_normalization": "subtract logsumexp(m + diag(LL.T)/2) after every update",
            "shared_axis_filter": False,
            "validation_gate": gate,
            "fit_artifact": {
                "path": relative(DERIVED_DIR / "model3_fit.npz"),
                "sha256": sha256(DERIVED_DIR / "model3_fit.npz"),
            },
            "inputs": {
                "h1": {"path": relative(H1_PATH), "sha256": sha256(H1_PATH)},
                "calibration": {
                    "path": relative(CALIBRATION),
                    "sha256": sha256(CALIBRATION),
                },
            },
        }
        (REPORT_DIR / "model3_manifest.json").write_text(
            json.dumps(fit_manifest, indent=2) + "\n"
        )
        if not all(gate.values()):
            print("Model 3 validation gate failed; expected-profile generation stopped")
            return

        source_rng = np.random.default_rng(np.random.SeedSequence([0, 1, 2, 0, 0]))
        source_rows = np.sort(
            sample_balanced_controls(data.obs, strict_guides, source_rng)
        )
        baseline_sum, raw_sum = source_expected_sum(data, source_rows, m, loadings)
        baseline_profile = log_pseudobulk(baseline_sum)
        raw_profile = log_pseudobulk(raw_sum)
        expected_profiles = np.empty((len(benchmark), data.n_vars), dtype=np.float32)
        raw_null_profiles = np.repeat(
            raw_profile[None, :], len(benchmark), axis=0
        ).astype(np.float32)
        posterior_null_profiles = np.repeat(
            baseline_profile[None, :], len(benchmark), axis=0
        ).astype(np.float32)
        closure_rows = []
        for index, row in enumerate(benchmark.itertuples(index=False)):
            target = str(row.target_gene)
            gene_index = gene_lookup[target]
            direction = loadings @ loadings[gene_index]
            transferred = float(
                calibration.loc[target, "leave_one_out_knockdown_depth"]
            )
            intended = intended_target_shift(
                baseline_sum[gene_index] / baseline_sum.sum(), transferred
            )
            amplitude, realized, shifted_sum = solve_amplitude(
                baseline_sum, direction, gene_index, intended
            )
            expected_profiles[index] = log_pseudobulk(shifted_sum)
            closure_rows.append(
                {
                    "target_gene": target,
                    "source_order": int(row.source_order),
                    "observed_knockdown_depth": calibration.loc[
                        target, "observed_knockdown_depth"
                    ],
                    "leave_one_out_knockdown_depth": transferred,
                    "intended_target_shift": intended,
                    "expected_realized_shift": realized,
                    "amplitude": amplitude,
                    "direction_diagonal": direction[gene_index],
                    "expected_total_multiplier": float(
                        shifted_sum.sum() / baseline_sum.sum()
                    ),
                    "population_total_multiplier": float(
                        (q * np.exp(np.clip(amplitude * direction, -80, 80))).sum()
                    ),
                }
            )
    finally:
        data.file.close()
        work_path.unlink(missing_ok=True)

    closure_path = REPORT_DIR / "model3_on_target_closure.csv"
    pd.DataFrame(closure_rows).to_csv(closure_path, index=False)
    artifact = DERIVED_DIR / "model3_expected_profiles.npz"
    np.savez_compressed(
        artifact,
        target_gene=np.asarray(benchmark["target_gene"].astype(str).tolist()),
        source_order=benchmark["source_order"].to_numpy(dtype=np.int64),
        gene_names=np.asarray(genes),
        expected_log_bulk=expected_profiles,
        null_log_bulk=raw_null_profiles,
        posterior_null_log_bulk=posterior_null_profiles,
        source_rows=source_rows,
        directions=np.column_stack(
            [
                loadings @ loadings[gene_lookup[target]]
                for target in target_table["target_gene"]
            ]
        ).astype(np.float32),
    )
    manifest = fit_manifest | {
        "status": "expected_profiles_generated",
        "cell_target_sum": CELL_TARGET_SUM,
        "bulk_target_sum": BULK_TARGET_SUM,
        "artifact": {"path": relative(artifact), "sha256": sha256(artifact)},
        "outputs": {
            "tiny_validation": {
                "path": relative(REPORT_DIR / "model3_tiny_laplace_validation.csv"),
                "sha256": sha256(REPORT_DIR / "model3_tiny_laplace_validation.csv"),
            },
            "fit_starts": {
                "path": relative(REPORT_DIR / "model3_fit_starts.csv"),
                "sha256": sha256(REPORT_DIR / "model3_fit_starts.csv"),
            },
            "posterior_predictive": {
                "path": relative(REPORT_DIR / "model3_posterior_predictive.csv"),
                "sha256": sha256(REPORT_DIR / "model3_posterior_predictive.csv"),
            },
            "validation_gate": {
                "path": relative(REPORT_DIR / "model3_validation_gate.csv"),
                "sha256": sha256(REPORT_DIR / "model3_validation_gate.csv"),
            },
            "closure": {
                "path": relative(closure_path),
                "sha256": sha256(closure_path),
            },
        },
        "software": {
            "python": platform.python_version(),
            "anndata": version("anndata"),
            "jax": version("jax"),
            "numpy": version("numpy"),
            "optax": version("optax"),
            "pandas": version("pandas"),
            "scipy": version("scipy"),
            "scikit-learn": version("scikit-learn"),
        },
    }
    (REPORT_DIR / "model3_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-batches",
        type=int,
        help="run one finite-fit smoke test for this many batches, then stop",
    )
    parser.add_argument(
        "--reuse-fit",
        action="store_true",
        help="rerun diagnostics and profile generation from the saved best fit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
