"""Full rank-50 Poisson-lognormal model with diagonal residual variation.

This is the ``Sigma = L L.T + D`` successor to the pure-factor Model 3.  The
E-step uses a joint Laplace approximation over factors and gene log-rates.  The
M-step optimizes the expected Gaussian residual objective while enforcing the
population mean-rate normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import NamedTuple

import anndata as ad
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import pearsonr, spearmanr

from scripts.linear_response.kernel import (
    BULK_TARGET_SUM,
    CELL_TARGET_SUM,
    intended_target_shift,
    log_pseudobulk,
    sample_balanced_controls,
)
from scripts.linear_response.model3_poisson_lognormal import (
    INFER as PURE_INFER,
)
from scripts.linear_response.model3_poisson_lognormal import (
    fit_start as fit_pure_start,
)
from scripts.linear_response.model3_poisson_lognormal import (
    initialize_model as initialize_pca,
)
from scripts.linear_response.model3_poisson_lognormal import (
    normalize_population as normalize_pure_population,
)

jax.config.update("jax_enable_x64", True)

D_FLOOR = 1e-6
MODE_TOLERANCE = 1e-4
MAX_NEWTON_STEPS = 30
MAX_LINE_SEARCH_STEPS = 40
ARMIJO = 1e-4
LOADING_PENALTY = 1e-4
E_STEP_CHECKPOINT_BATCHES = 10
MIN_OUTER_ITERATIONS = 3
OUTER_CONVERGENCE_PATIENCE = 2
OUTER_NLL_TOLERANCE = 1e-2
OUTER_MEAN_TOLERANCE = 1e-3
OUTER_COVARIANCE_TOLERANCE = 1e-3
M_STEP_GRADIENT_RMS_TOLERANCE = 1e-2

ROOT = Path(__file__).resolve().parents[2]
H1_PATH = ROOT / "data/external/vcc2025_h1/adata_Training.h5ad"
REPORT = ROOT / "reports/linear-response-three-models"
DERIVED = ROOT / "data/derived/linear_response"
STRICT_CONTROLS = REPORT / "h1_strict_controls.csv"
TARGET_TABLE = REPORT / "h1_target_counts.csv"
CALIBRATION = REPORT / "model1_knockdown_calibration.csv"
INITIALIZATION = DERIVED / "model3_full_initialization.npz"
FIT = DERIVED / "model3_full_fit.npz"
INDEPENDENT_D_FIT = DERIVED / "model3_independent_d_fit.npz"
PURE_FIT = DERIVED / "model3_pure_factor_fit.npz"
RANK = 50
BATCH_SIZE = 64
N_SELECTION_GUIDES = 4
N_EVALUATION_GUIDES = 4
N_STARTS = 3
OUTER_ITERATIONS = 10
M_STEP_ITERATIONS = 1_000
RANDOM_MATRIX_DRAWS = 20


class ModeResult(NamedTuple):
    z: jax.Array
    x: jax.Array
    rates: jax.Array
    s: jax.Array
    k_inv: jax.Array
    r: jax.Array
    residual: jax.Array
    converged: jax.Array
    line_search_failed: jax.Array


@dataclass(frozen=True)
class MomentSums:
    n_cells: int
    x: np.ndarray
    z: np.ndarray
    zz: np.ndarray
    xz: np.ndarray
    x2: np.ndarray

    @classmethod
    def zeros(cls, genes: int, rank: int) -> MomentSums:
        return cls(
            n_cells=0,
            x=np.zeros(genes),
            z=np.zeros(rank),
            zz=np.zeros((rank, rank)),
            xz=np.zeros((genes, rank)),
            x2=np.zeros(genes),
        )

    def add(self, batch: dict[str, np.ndarray]) -> MomentSums:
        return MomentSums(
            n_cells=self.n_cells + int(batch["n_cells"]),
            x=self.x + batch["x"],
            z=self.z + batch["z"],
            zz=self.zz + batch["zz"],
            xz=self.xz + batch["xz"],
            x2=self.x2 + batch["x2"],
        )

    def means(self) -> dict[str, np.ndarray]:
        if self.n_cells == 0:
            raise ValueError("cannot average empty posterior moments")
        return {
            "x": self.x / self.n_cells,
            "z": self.z / self.n_cells,
            "zz": self.zz / self.n_cells,
            "xz": self.xz / self.n_cells,
            "x2": self.x2 / self.n_cells,
        }


def softplus_inverse(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    return np.where(value > 20, value, np.log(np.expm1(value)))


def make_parameters(
    mean_rates: np.ndarray, loadings: np.ndarray, residual_variance: np.ndarray
) -> dict[str, jax.Array]:
    mean_rates = np.asarray(mean_rates, dtype=np.float64)
    mean_rates /= mean_rates.sum()
    residual_variance = np.maximum(
        np.asarray(residual_variance, dtype=np.float32), D_FLOOR * 1.01
    )
    return {
        "alpha": jnp.asarray(np.log(mean_rates)),
        "loadings": jnp.asarray(loadings, dtype=jnp.float64),
        "raw_d": jnp.asarray(
            softplus_inverse(residual_variance - D_FLOOR), dtype=jnp.float64
        ),
    }


def population_parameters(
    parameters: dict[str, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    loadings = parameters["loadings"]
    d = D_FLOOR + jax.nn.softplus(parameters["raw_d"])
    log_p = jax.nn.log_softmax(parameters["alpha"])
    mu = log_p - 0.5 * (jnp.square(loadings).sum(1) + d)
    return mu, loadings, d


def negative_joint(
    z: jax.Array,
    x: jax.Array,
    counts: jax.Array,
    totals: jax.Array,
    mu: jax.Array,
    loadings: jax.Array,
    d: jax.Array,
) -> jax.Array:
    u = x - mu[None, :] - z @ loadings.T
    rates = totals[:, None] * jnp.exp(x)
    return (
        0.5 * jnp.square(z).sum(1)
        + 0.5 * (jnp.square(u) / d[None, :]).sum(1)
        + rates.sum(1)
        - (counts * x).sum(1)
    )


def negative_joint_change(
    z: jax.Array,
    x: jax.Array,
    delta_z: jax.Array,
    delta_x: jax.Array,
    counts: jax.Array,
    totals: jax.Array,
    mu: jax.Array,
    loadings: jax.Array,
    d: jax.Array,
) -> jax.Array:
    """Evaluate a trial objective change without subtracting large objectives."""
    u = x - mu[None, :] - z @ loadings.T
    delta_u = delta_x - delta_z @ loadings.T
    rates = totals[:, None] * jnp.exp(x)
    return (
        (z * delta_z).sum(1)
        + 0.5 * jnp.square(delta_z).sum(1)
        + ((u * delta_u + 0.5 * jnp.square(delta_u)) / d[None, :]).sum(1)
        + (rates * jnp.expm1(delta_x) - counts * delta_x).sum(1)
    )


def _mode_terms(
    z: jax.Array,
    x: jax.Array,
    counts: jax.Array,
    totals: jax.Array,
    mu: jax.Array,
    loadings: jax.Array,
    d: jax.Array,
) -> tuple[jax.Array, ...]:
    u = x - mu[None, :] - z @ loadings.T
    rates = totals[:, None] * jnp.exp(x)
    d_inv = 1.0 / d
    gradient_z = z - (u * d_inv[None, :]) @ loadings
    gradient_x = u * d_inv[None, :] + rates - counts
    k_inv = d[None, :] / (1.0 + d[None, :] * rates)
    r = 1.0 / (1.0 + d[None, :] * rates)
    weights = rates * r
    identity = jnp.eye(loadings.shape[1], dtype=loadings.dtype)
    s = identity[None, :, :] + jnp.einsum("bg,gr,gs->brs", weights, loadings, loadings)
    residual = jnp.maximum(
        jnp.linalg.norm(gradient_z, axis=1),
        jnp.max(jnp.abs(gradient_x), axis=1),
    )
    return rates, gradient_z, gradient_x, k_inv, r, s, residual


def joint_laplace_mode(
    mu: jax.Array,
    loadings: jax.Array,
    d: jax.Array,
    counts: jax.Array,
    totals: jax.Array,
    *,
    max_steps: int = MAX_NEWTON_STEPS,
    tolerance: float = MODE_TOLERANCE,
) -> ModeResult:
    """Find the unique joint mode using block Newton steps and Armijo search."""
    batch = counts.shape[0]
    z = jnp.zeros((batch, loadings.shape[1]), dtype=loadings.dtype)
    x = jnp.broadcast_to(mu, counts.shape)
    failed = jnp.zeros(batch, dtype=bool)

    def newton_step(
        state: tuple[jax.Array, jax.Array, jax.Array], _: None
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        current_z, current_x, current_failed = state
        (
            _,
            gradient_z,
            gradient_x,
            k_inv,
            r,
            s,
            residual,
        ) = _mode_terms(current_z, current_x, counts, totals, mu, loadings, d)
        active = (residual > tolerance) & ~current_failed
        rhs = -gradient_z - (r * gradient_x) @ loadings
        delta_z = jnp.linalg.solve(s, rhs[..., None])[..., 0]
        delta_x = -k_inv * gradient_x + r * (delta_z @ loadings.T)
        directional = (gradient_z * delta_z).sum(1) + (gradient_x * delta_x).sum(1)
        initial = (
            jnp.ones(batch, dtype=loadings.dtype),
            current_z,
            current_x,
            ~active,
        )

        def search_step(
            search: tuple[jax.Array, jax.Array, jax.Array, jax.Array], _: None
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], None]:
            step_size, accepted_z, accepted_x, accepted = search
            scaled_delta_z = step_size[:, None] * delta_z
            scaled_delta_x = step_size[:, None] * delta_x
            trial_z = current_z + scaled_delta_z
            trial_x = current_x + scaled_delta_x
            objective_change = negative_joint_change(
                current_z,
                current_x,
                scaled_delta_z,
                scaled_delta_x,
                counts,
                totals,
                mu,
                loadings,
                d,
            )
            sufficient = (
                jnp.isfinite(objective_change)
                & (objective_change <= ARMIJO * step_size * directional)
                & active
                & ~accepted
            )
            accepted_z = jnp.where(sufficient[:, None], trial_z, accepted_z)
            accepted_x = jnp.where(sufficient[:, None], trial_x, accepted_x)
            return (
                step_size * 0.5,
                accepted_z,
                accepted_x,
                accepted | sufficient,
            ), None

        (unused_step, next_z, next_x, accepted), _ = jax.lax.scan(
            search_step, initial, None, length=MAX_LINE_SEARCH_STEPS
        )
        del unused_step
        return (
            next_z,
            next_x,
            current_failed | (active & ~accepted),
        ), None

    (z, x, failed), _ = jax.lax.scan(
        newton_step, (z, x, failed), None, length=max_steps
    )
    rates, _, _, k_inv, r, s, residual = _mode_terms(
        z, x, counts, totals, mu, loadings, d
    )
    return ModeResult(
        z=z,
        x=x,
        rates=rates,
        s=s,
        k_inv=k_inv,
        r=r,
        residual=residual,
        converged=residual <= tolerance,
        line_search_failed=failed,
    )


def posterior_blocks(
    mode: ModeResult, loadings: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    identity = jnp.broadcast_to(
        jnp.eye(loadings.shape[1], dtype=loadings.dtype), mode.s.shape
    )
    v_zz = jnp.linalg.solve(mode.s, identity)
    v_xz = jnp.einsum("bg,gr,brs->bgs", mode.r, loadings, v_zz)
    shared_diagonal = jnp.einsum("gr,brs,gs->bg", loadings, v_zz, loadings)
    v_xx_diagonal = mode.k_inv + jnp.square(mode.r) * shared_diagonal
    return v_zz, v_xz, v_xx_diagonal


def batch_moment_sums(mode: ModeResult, loadings: jax.Array) -> dict[str, np.ndarray]:
    arrays = batch_moment_arrays(mode, loadings)
    return {
        "n_cells": np.asarray(mode.x.shape[0]),
        **{name: np.asarray(value, dtype=np.float64) for name, value in arrays.items()},
    }


def batch_moment_arrays(mode: ModeResult, loadings: jax.Array) -> dict[str, jax.Array]:
    v_zz, v_xz, v_xx_diagonal = posterior_blocks(mode, loadings)
    return {
        "x": mode.x.sum(0),
        "z": mode.z.sum(0),
        "zz": v_zz.sum(0) + mode.z.T @ mode.z,
        "xz": v_xz.sum(0) + mode.x.T @ mode.z,
        "x2": (v_xx_diagonal + jnp.square(mode.x)).sum(0),
    }


def laplace_nll_from_mode(
    mode: ModeResult,
    counts: jax.Array,
    totals: jax.Array,
    mu: jax.Array,
    loadings: jax.Array,
    d: jax.Array,
) -> jax.Array:
    phi = negative_joint(mode.z, mode.x, counts, totals, mu, loadings, d)
    logdet_s = jnp.linalg.slogdet(mode.s)[1]
    correction = 0.5 * (jnp.log1p(d[None, :] * mode.rates).sum(1) + logdet_s)
    return phi + correction


def laplace_nll(
    parameters: dict[str, jax.Array], counts: jax.Array, totals: jax.Array
) -> tuple[jax.Array, ModeResult]:
    """Return per-cell joint-Laplace NLL, omitting count-only constants."""
    mu, loadings, d = population_parameters(parameters)
    mode = joint_laplace_mode(mu, loadings, d, counts, totals)
    return (
        laplace_nll_from_mode(mode, counts, totals, mu, loadings, d),
        mode,
    )


def expected_residuals(
    mu: jax.Array,
    loadings: jax.Array,
    moments: dict[str, jax.Array],
) -> jax.Array:
    x_bar = moments["x"]
    z_bar = moments["z"]
    u_zz = moments["zz"]
    u_xz = moments["xz"]
    t_x = moments["x2"]
    loading_z = loadings @ z_bar
    return (
        t_x
        - 2.0 * mu * x_bar
        - 2.0 * (u_xz * loadings).sum(1)
        + jnp.square(mu)
        + 2.0 * mu * loading_z
        + (loadings @ u_zz * loadings).sum(1)
    )


def m_step_objective(
    parameters: dict[str, jax.Array],
    moments: dict[str, jax.Array],
    loading_penalty: float = LOADING_PENALTY,
) -> jax.Array:
    mu, loadings, d = population_parameters(parameters)
    residuals = expected_residuals(mu, loadings, moments)
    gaussian = 0.5 * jnp.sum(jnp.log(d) + residuals / d)
    return gaussian + loading_penalty * jnp.square(loadings).sum()


def fit_m_step(
    initial: dict[str, jax.Array],
    moments: dict[str, np.ndarray],
    *,
    max_iterations: int = 25,
    tolerance: float = 1e-4,
    loading_penalty: float = LOADING_PENALTY,
) -> tuple[dict[str, jax.Array], jaxopt.OptStep]:
    fixed = {name: jnp.asarray(value) for name, value in moments.items()}
    solver = jaxopt.LBFGS(
        fun=m_step_objective,
        maxiter=max_iterations,
        tol=tolerance,
        history_size=5,
        linesearch="zoom",
        maxls=30,
    )
    result = solver.run(initial, fixed, loading_penalty)
    centered = dict(result.params)
    centered["alpha"] = centered["alpha"] - centered["alpha"].mean()
    return centered, result


def fit_m_step_fixed_d(
    initial: dict[str, jax.Array],
    moments: dict[str, np.ndarray],
    *,
    max_iterations: int = 25,
    tolerance: float = 1e-4,
    loading_penalty: float = LOADING_PENALTY,
) -> tuple[dict[str, jax.Array], jaxopt.OptStep]:
    """Update mean and loadings while holding diagonal residual variance fixed."""
    fixed_moments = {name: jnp.asarray(value) for name, value in moments.items()}
    fixed_raw_d = initial["raw_d"]

    def objective(
        active: dict[str, jax.Array],
        posterior_moments: dict[str, jax.Array],
        penalty: float,
    ) -> jax.Array:
        return m_step_objective(
            {**active, "raw_d": fixed_raw_d}, posterior_moments, penalty
        )

    solver = jaxopt.LBFGS(
        fun=objective,
        maxiter=max_iterations,
        tol=tolerance,
        history_size=5,
        linesearch="zoom",
        maxls=30,
    )
    active = {name: initial[name] for name in ["alpha", "loadings"]}
    result = solver.run(active, fixed_moments, loading_penalty)
    centered = dict(result.params)
    centered["alpha"] = centered["alpha"] - centered["alpha"].mean()
    centered["raw_d"] = fixed_raw_d
    return centered, result


def posterior_mean_rates(mode: ModeResult, loadings: jax.Array) -> jax.Array:
    _, _, v_xx_diagonal = posterior_blocks(mode, loadings)
    return mode.rates * jnp.exp(0.5 * v_xx_diagonal)


def sample_posterior_predictive(
    mode: ModeResult,
    loadings: np.ndarray,
    totals: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one replicate from the fitted Gaussian posterior and Poisson model."""
    z = np.asarray(mode.z, dtype=np.float64)
    x = np.asarray(mode.x, dtype=np.float64)
    r = np.asarray(mode.r, dtype=np.float64)
    k_inv = np.asarray(mode.k_inv, dtype=np.float64)
    v_zz, _, _ = posterior_blocks(mode, jnp.asarray(loadings))
    v_zz = np.asarray(v_zz, dtype=np.float64)
    delta_z = np.stack(
        [
            np.linalg.cholesky(covariance) @ rng.normal(size=len(z[0]))
            for covariance in v_zz
        ]
    )
    delta_x = r * (delta_z @ loadings.T) + np.sqrt(k_inv) * rng.normal(size=x.shape)
    rates = totals[:, None] * np.exp(x + delta_x)
    if not np.isfinite(rates).all():
        raise FloatingPointError("posterior-predictive rates are non-finite")
    return rng.poisson(rates).astype(np.int32)


def covariance_columns(
    loadings: np.ndarray, residual_variance: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    columns = loadings @ loadings[indices].T
    columns[indices, np.arange(len(indices))] += residual_variance[indices]
    return columns


INFER_BATCH = jax.jit(joint_laplace_mode)
MOMENT_BATCH = jax.jit(batch_moment_arrays)
NLL_FROM_MODE = jax.jit(laplace_nll_from_mode)
POSTERIOR_MEAN_BATCH = jax.jit(posterior_mean_rates)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def display_path(path: Path) -> str:
    return relative(path) if path.is_relative_to(ROOT) else path.as_posix()


def arrays_digest(arrays: dict[str, np.ndarray | jax.Array]) -> str:
    """Hash named numerical arrays, including their dtype and shape."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def atomic_save_npz(path: Path, **arrays: np.ndarray | jax.Array) -> None:
    """Replace an NPZ only after its compressed temporary file is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def atomic_write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save_e_step_checkpoint(
    path: Path,
    sums: MomentSums,
    per_cell: np.ndarray,
    residuals: np.ndarray,
    usable_rows: np.ndarray,
    parameters: dict[str, jax.Array],
    completed_batches: int,
) -> None:
    atomic_save_npz(
        path,
        completed_batches=np.asarray(completed_batches),
        usable_rows=usable_rows,
        parameter_digest=np.asarray(arrays_digest(parameters)),
        moment_n_cells=np.asarray(sums.n_cells),
        moment_x=sums.x,
        moment_z=sums.z,
        moment_zz=sums.zz,
        moment_xz=sums.xz,
        moment_x2=sums.x2,
        per_cell=per_cell,
        residuals=residuals,
    )


def load_e_step_checkpoint(
    path: Path,
    usable_rows: np.ndarray,
    parameters: dict[str, jax.Array],
) -> tuple[int, MomentSums, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved["usable_rows"], usable_rows):
            raise ValueError(f"{display_path(path)} has a different E-step row set")
        expected_digest = arrays_digest(parameters)
        if str(saved["parameter_digest"].item()) != expected_digest:
            raise ValueError(f"{display_path(path)} has different E-step parameters")
        completed_batches = int(saved["completed_batches"])
        sums = MomentSums(
            n_cells=int(saved["moment_n_cells"]),
            x=saved["moment_x"],
            z=saved["moment_z"],
            zz=saved["moment_zz"],
            xz=saved["moment_xz"],
            x2=saved["moment_x2"],
        )
        per_cell = saved["per_cell"]
        residuals = saved["residuals"]
    expected_cells = completed_batches * BATCH_SIZE
    if not (
        sums.n_cells == expected_cells
        and len(per_cell) == expected_cells
        and len(residuals) == expected_cells
        and expected_cells <= len(usable_rows)
    ):
        raise ValueError(f"{display_path(path)} has inconsistent E-step progress")
    return completed_batches, sums, per_cell, residuals


def split_control_rows(
    obs: pd.DataFrame, strict_guides: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    permutation = np.random.default_rng(17).permutation(sorted(strict_guides))
    selection_guides = sorted(permutation[:N_SELECTION_GUIDES].tolist())
    evaluation_guides = sorted(
        permutation[
            N_SELECTION_GUIDES : N_SELECTION_GUIDES + N_EVALUATION_GUIDES
        ].tolist()
    )
    training_guides = sorted(
        permutation[N_SELECTION_GUIDES + N_EVALUATION_GUIDES :].tolist()
    )
    labels = obs["guide_id"].astype(str)
    controls = obs["target_gene"].astype(str).eq("non-targeting")

    def rows(guides: list[str]) -> np.ndarray:
        return np.flatnonzero((controls & labels.isin(guides)).to_numpy())

    return (
        rows(training_guides),
        rows(selection_guides),
        rows(evaluation_guides),
        {
            "training": training_guides,
            "selection": selection_guides,
            "evaluation": evaluation_guides,
        },
    )


def dense_counts(data: ad.AnnData, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = data.X[np.sort(rows)].toarray().astype(np.float64)
    return counts, counts.sum(1)


def factorial_initialization(
    data: ad.AnnData, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    count_sum = np.zeros(data.n_vars)
    factorial_sum = np.zeros(data.n_vars)
    exposure_sum = 0.0
    exposure_square_sum = 0.0
    for start in range(0, len(rows), 512):
        counts = data.X[rows[start : start + 512]].tocsr().astype(np.float64)
        totals = np.asarray(counts.sum(1)).ravel()
        count_sum += np.asarray(counts.sum(0)).ravel()
        factorial = counts.copy()
        factorial.data *= factorial.data - 1.0
        factorial_sum += np.asarray(factorial.sum(0)).ravel()
        exposure_sum += totals.sum()
        exposure_square_sum += np.square(totals).sum()
    mean_rates = np.maximum(count_sum, 0.5)
    mean_rates /= mean_rates.sum()
    second = factorial_sum / exposure_square_sum
    ratio = np.divide(
        second,
        np.square(mean_rates),
        out=np.ones_like(second),
        where=(second > 0) & (mean_rates > 0),
    )
    variance = np.maximum(np.log(ratio), 0.0)
    if not np.isclose(exposure_sum, count_sum.sum()):
        raise AssertionError("modeled-gene exposures do not match pooled counts")
    return mean_rates, variance


def initialize_full_model(
    data: ad.AnnData,
    train_rows: np.ndarray,
    work_path: Path,
) -> tuple[dict[str, jax.Array], np.ndarray, np.ndarray, np.ndarray]:
    _, pca_loadings, _ = initialize_pca(data, train_rows, work_path)
    mean_rates, total_variance = factorial_initialization(data, train_rows)
    factor_variance = np.square(pca_loadings).sum(1)
    capacity = 0.9 * total_variance
    scale = np.sqrt(np.minimum(1.0, capacity / np.maximum(factor_variance, 1e-12)))
    loadings = pca_loadings * scale[:, None]
    assigned = np.square(loadings).sum(1)
    d = np.maximum(total_variance - assigned, D_FLOOR * 1.01)
    return (
        make_parameters(mean_rates, loadings, d),
        pca_loadings,
        total_variance,
        mean_rates,
    )


def run_e_step(
    data: ad.AnnData,
    rows: np.ndarray,
    parameters: dict[str, jax.Array],
    *,
    collect_moments: bool,
    checkpoint_path: Path | None = None,
    resume_checkpoint: bool = False,
    checkpoint_every: int = E_STEP_CHECKPOINT_BATCHES,
) -> tuple[MomentSums | None, dict[str, float], np.ndarray]:
    if checkpoint_path is not None and not collect_moments:
        raise ValueError("E-step checkpoints require posterior moments")
    mu, loadings, d = population_parameters(parameters)
    usable = rows[: len(rows) // BATCH_SIZE * BATCH_SIZE]
    sums = MomentSums.zeros(data.n_vars, loadings.shape[1]) if collect_moments else None
    first_batch = 0
    residuals: list[np.ndarray] = []
    per_cell: list[np.ndarray] = []
    if resume_checkpoint and checkpoint_path is not None and checkpoint_path.exists():
        first_batch, loaded_sums, loaded_values, loaded_residuals = (
            load_e_step_checkpoint(checkpoint_path, usable, parameters)
        )
        sums = loaded_sums
        per_cell.append(loaded_values)
        residuals.append(loaded_residuals)
        print(
            f"resuming E-step at batch {first_batch}/{len(usable) // BATCH_SIZE}",
            flush=True,
        )
    for start in range(first_batch * BATCH_SIZE, len(usable), BATCH_SIZE):
        batch_rows = usable[start : start + BATCH_SIZE]
        counts, totals = dense_counts(data, batch_rows)
        counts_jax = jnp.asarray(counts)
        totals_jax = jnp.asarray(totals)
        mode = INFER_BATCH(mu, loadings, d, counts_jax, totals_jax)
        failed = np.asarray(mode.line_search_failed)
        converged = np.asarray(mode.converged)
        mode_residuals = np.asarray(mode.residual, dtype=np.float64)
        if failed.any() or not converged.all():
            if checkpoint_path is not None:
                atomic_write_json(
                    checkpoint_path.with_suffix(".failure.json"),
                    {
                        "created_utc": datetime.now(UTC).isoformat(),
                        "e_step_checkpoint": display_path(checkpoint_path),
                        "batch": start // BATCH_SIZE,
                        "source_rows": batch_rows.tolist(),
                        "line_search_failed_rows": batch_rows[failed].tolist(),
                        "unconverged_rows": batch_rows[~converged].tolist(),
                        "residuals": mode_residuals.tolist(),
                        "maximum_residual": float(mode_residuals.max()),
                        "parameter_digest": arrays_digest(parameters),
                    },
                )
            raise RuntimeError(
                f"joint mode did not converge in batch {start // BATCH_SIZE}; "
                f"line-search failures {int(failed.sum())}, "
                f"maximum residual {float(mode_residuals.max()):.6g}"
            )
        values = np.asarray(
            NLL_FROM_MODE(mode, counts_jax, totals_jax, mu, loadings, d),
            dtype=np.float64,
        )
        per_cell.append(values)
        residuals.append(mode_residuals)
        if sums is not None:
            arrays = {
                name: np.asarray(value, dtype=np.float64)
                for name, value in MOMENT_BATCH(mode, loadings).items()
            }
            arrays["n_cells"] = BATCH_SIZE
            sums = sums.add(arrays)
        batch_number = start // BATCH_SIZE + 1
        total_batches = len(usable) // BATCH_SIZE
        if (
            checkpoint_path is not None
            and sums is not None
            and (batch_number % checkpoint_every == 0 or batch_number == total_batches)
        ):
            save_e_step_checkpoint(
                checkpoint_path,
                sums,
                np.concatenate(per_cell),
                np.concatenate(residuals),
                usable,
                parameters,
                batch_number,
            )
            print(
                f"E-step checkpoint {batch_number}/{total_batches}: "
                f"{display_path(checkpoint_path)}",
                flush=True,
            )
        if len(usable) >= 100 * BATCH_SIZE and batch_number % 100 == 0:
            print(
                f"E-step batch {batch_number}/{total_batches}",
                flush=True,
            )
    per_cell_array = np.concatenate(per_cell)
    residual_array = np.concatenate(residuals)
    diagnostics = {
        "cells": float(len(usable)),
        "omitted_remainder_cells": float(len(rows) - len(usable)),
        "laplace_nll_mean": float(per_cell_array.mean()),
        "mode_residual_median": float(np.median(residual_array)),
        "mode_residual_q99": float(np.quantile(residual_array, 0.99)),
        "mode_residual_max": float(residual_array.max()),
    }
    return sums, diagnostics, per_cell_array


def distribution_change(
    before: dict[str, jax.Array], after: dict[str, jax.Array]
) -> tuple[float, float]:
    """Return RMS mean change and rotation-invariant covariance change."""
    before_mu, before_loadings, before_d = population_parameters(before)
    after_mu, after_loadings, after_d = population_parameters(after)
    before_mu = np.asarray(before_mu)
    after_mu = np.asarray(after_mu)
    before_loadings = np.asarray(before_loadings)
    after_loadings = np.asarray(after_loadings)
    before_d = np.asarray(before_d)
    after_d = np.asarray(after_d)

    mean_rms = float(np.sqrt(np.mean(np.square(after_mu - before_mu))))
    before_gram = before_loadings.T @ before_loadings
    after_gram = after_loadings.T @ after_loadings
    cross_gram = before_loadings.T @ after_loadings
    shared_difference_squared = (
        np.square(before_gram).sum()
        + np.square(after_gram).sum()
        - 2.0 * np.square(cross_gram).sum()
    )
    shared_diagonal_difference = np.square(after_loadings).sum(1) - np.square(
        before_loadings
    ).sum(1)
    diagonal_difference = after_d - before_d
    covariance_difference_squared = (
        shared_difference_squared
        + np.square(diagonal_difference).sum()
        + 2.0 * np.dot(shared_diagonal_difference, diagonal_difference)
    )
    covariance_norm_squared = (
        np.square(before_gram).sum()
        + np.square(before_d).sum()
        + 2.0 * np.dot(np.square(before_loadings).sum(1), before_d)
    )
    covariance_relative = float(
        np.sqrt(
            max(covariance_difference_squared, 0.0)
            / max(covariance_norm_squared, 1e-30)
        )
    )
    return mean_rms, covariance_relative


def fit_em(
    data: ad.AnnData,
    rows: np.ndarray,
    initial: dict[str, jax.Array],
    *,
    outer_iterations: int,
    m_step_iterations: int,
    update_d: bool = True,
    checkpoint_prefix: Path | None = None,
    trace_path: Path | None = None,
    resume: bool = False,
    require_convergence: bool = False,
) -> tuple[dict[str, jax.Array], pd.DataFrame]:
    parameters = initial
    trace: list[dict[str, object]] = []
    first_outer = 0

    def outer_checkpoint(iteration: int) -> Path:
        if checkpoint_prefix is None:
            raise ValueError("checkpoint prefix is unavailable")
        return checkpoint_prefix.with_name(
            f"{checkpoint_prefix.name}_checkpoint_outer_{iteration}.npz"
        )

    if checkpoint_prefix is not None:
        if resume:
            for completed in range(outer_iterations, -1, -1):
                path = outer_checkpoint(completed)
                if path.exists():
                    validate_fit_checkpoint(
                        path,
                        rows,
                        update_d=update_d,
                        m_step_iterations=m_step_iterations,
                    )
                    parameters = load_parameters(path)
                    first_outer = completed
                    break
            for completed in range(1, first_outer + 1):
                with np.load(outer_checkpoint(completed), allow_pickle=False) as saved:
                    trace.append(json.loads(str(saved["trace_record"].item())))
            if first_outer:
                print(
                    f"resuming EM after outer iteration "
                    f"{first_outer}/{outer_iterations}",
                    flush=True,
                )
        if first_outer == 0:
            save_parameters(
                outer_checkpoint(0),
                parameters,
                train_rows=rows,
                completed_outer_iterations=np.asarray(0),
                update_d=np.asarray(update_d),
                m_step_iterations=np.asarray(m_step_iterations),
                trace_record=np.asarray(""),
            )
        if trace_path is not None:
            atomic_write_frame(trace_path, pd.DataFrame(trace))

    if trace and bool(trace[-1].get("outer_converged", False)):
        return parameters, pd.DataFrame(trace)

    for outer in range(first_outer, outer_iterations):
        e_step_checkpoint = None
        if checkpoint_prefix is not None:
            e_step_checkpoint = checkpoint_prefix.with_name(
                f"{checkpoint_prefix.name}_outer_{outer + 1}_estep.npz"
            )
        sums, e_step, _ = run_e_step(
            data,
            rows,
            parameters,
            collect_moments=True,
            checkpoint_path=e_step_checkpoint,
            resume_checkpoint=resume,
        )
        if sums is None:
            raise AssertionError("E-step did not return posterior moments")
        moments = sums.means()
        fixed = {name: jnp.asarray(value) for name, value in moments.items()}
        before = float(m_step_objective(parameters, fixed))
        fitter = fit_m_step if update_d else fit_m_step_fixed_d
        updated, result = fitter(parameters, moments, max_iterations=m_step_iterations)
        after = float(m_step_objective(updated, fixed))
        if not np.isfinite(after) or after > before + 1e-4:
            raise RuntimeError(
                f"M-step failed to decrease fixed-posterior objective: {before} -> {after}"
            )
        mu, loadings, d = population_parameters(updated)
        q = np.exp(
            np.asarray(mu)
            + 0.5 * (np.square(np.asarray(loadings)).sum(1) + np.asarray(d))
        )
        mean_rms_change, covariance_relative_change = distribution_change(
            parameters, updated
        )
        nll_change = (
            float(trace[-1]["laplace_nll_mean"]) - e_step["laplace_nll_mean"]
            if trace
            else np.nan
        )
        parameter_count = sum(np.asarray(value).size for value in updated.values())
        gradient_rms = float(result.state.error) / np.sqrt(parameter_count)
        outer_stable = bool(
            outer + 1 >= MIN_OUTER_ITERATIONS
            and np.isfinite(nll_change)
            and abs(nll_change) <= OUTER_NLL_TOLERANCE
            and mean_rms_change <= OUTER_MEAN_TOLERANCE
            and covariance_relative_change <= OUTER_COVARIANCE_TOLERANCE
            and gradient_rms <= M_STEP_GRADIENT_RMS_TOLERANCE
        )
        prior_stable = 0
        for prior in reversed(trace):
            if not bool(prior.get("outer_stable", False)):
                break
            prior_stable += 1
        outer_converged = outer_stable and (
            prior_stable + 1 >= OUTER_CONVERGENCE_PATIENCE
        )
        record: dict[str, object] = {
            "outer_iteration": outer,
            **e_step,
            "laplace_nll_improvement_from_previous": nll_change,
            "m_step_before": before,
            "m_step_after": after,
            "m_step_iterations": int(result.state.iter_num),
            "m_step_gradient_error": float(result.state.error),
            "m_step_gradient_rms": gradient_rms,
            "residual_variance_updated": update_d,
            "sum_population_mean_rates": float(q.sum()),
            "population_log_mean_rms_change": mean_rms_change,
            "covariance_relative_frobenius_change": covariance_relative_change,
            "residual_variance_min": float(np.asarray(d).min()),
            "residual_variance_median": float(np.median(np.asarray(d))),
            "residual_variance_max": float(np.asarray(d).max()),
            "outer_stable": outer_stable,
            "outer_converged": outer_converged,
        }
        trace.append(record)
        parameters = updated
        if checkpoint_prefix is not None:
            save_parameters(
                outer_checkpoint(outer + 1),
                parameters,
                train_rows=rows,
                completed_outer_iterations=np.asarray(outer + 1),
                update_d=np.asarray(update_d),
                m_step_iterations=np.asarray(m_step_iterations),
                trace_record=np.asarray(json.dumps(record)),
            )
            if trace_path is not None:
                atomic_write_frame(trace_path, pd.DataFrame(trace))
        print(
            f"outer {outer + 1}/{outer_iterations}: "
            f"E NLL {e_step['laplace_nll_mean']:.3f}, "
            f"M {before:.3f} -> {after:.3f}",
            flush=True,
        )
        if outer_converged:
            print(
                f"EM converged after {outer + 1} outer iterations",
                flush=True,
            )
            break
    if require_convergence and not trace[-1]["outer_converged"]:
        location = (
            display_path(outer_checkpoint(len(trace)))
            if checkpoint_prefix is not None
            else "no checkpoint"
        )
        raise RuntimeError(
            f"EM did not converge in {outer_iterations} outer iterations; "
            f"latest parameters are preserved at {location}"
        )
    return parameters, pd.DataFrame(trace)


def parameter_arrays(
    parameters: dict[str, jax.Array],
) -> dict[str, np.ndarray]:
    return {name: np.asarray(value) for name, value in parameters.items()}


def save_parameters(
    path: Path,
    parameters: dict[str, jax.Array],
    **metadata: np.ndarray,
) -> None:
    atomic_save_npz(path, **parameter_arrays(parameters), **metadata)


def load_parameters(path: Path) -> dict[str, jax.Array]:
    with np.load(path, allow_pickle=False) as saved:
        return {
            name: jnp.asarray(saved[name]) for name in ["alpha", "loadings", "raw_d"]
        }


def validate_fit_checkpoint(
    path: Path,
    train_rows: np.ndarray,
    *,
    update_d: bool,
    m_step_iterations: int,
) -> None:
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved["train_rows"], train_rows):
            raise ValueError(f"{relative(path)} has a different training split")
        if bool(saved["update_d"]) != update_d:
            raise ValueError(f"{relative(path)} has a different D-update setting")
        if int(saved["m_step_iterations"]) != m_step_iterations:
            raise ValueError(f"{relative(path)} has a different M-step length")


def pure_parameters(
    mean_rates: np.ndarray, loadings: np.ndarray
) -> dict[str, np.ndarray]:
    variance = np.square(loadings).sum(1)
    m = np.log(mean_rates) - 0.5 * variance
    m = np.asarray(normalize_pure_population(jnp.asarray(m), jnp.asarray(loadings)))
    return {"m": m, "loadings": np.asarray(loadings)}


@jax.jit
def pure_cell_nll(
    m: jax.Array,
    loadings: jax.Array,
    counts: jax.Array,
    totals: jax.Array,
) -> jax.Array:
    normalized = normalize_pure_population(m, loadings)
    modes, rates, hessian = PURE_INFER(normalized, loadings, counts, totals)
    linear = normalized[None, :] + modes @ loadings.T
    joint = rates.sum(1) - (counts * linear).sum(1) + 0.5 * jnp.square(modes).sum(1)
    return joint + 0.5 * jnp.linalg.slogdet(hessian)[1]


def evaluate_pure_cells(
    data: ad.AnnData,
    rows: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> tuple[dict[str, float], np.ndarray]:
    usable = rows[: len(rows) // BATCH_SIZE * BATCH_SIZE]
    per_cell = []
    residuals = []
    m = jnp.asarray(parameters["m"])
    loadings = jnp.asarray(parameters["loadings"])
    for start in range(0, len(usable), BATCH_SIZE):
        counts, totals = dense_counts(data, usable[start : start + BATCH_SIZE])
        modes, rates, _ = PURE_INFER(
            m, loadings, jnp.asarray(counts), jnp.asarray(totals)
        )
        gradients = modes - (jnp.asarray(counts) - rates) @ loadings
        residuals.extend(np.linalg.norm(np.asarray(gradients), axis=1).tolist())
        per_cell.append(
            np.asarray(
                pure_cell_nll(
                    m,
                    loadings,
                    jnp.asarray(counts),
                    jnp.asarray(totals),
                )
            )
        )
    values = np.concatenate(per_cell).astype(np.float64)
    residuals_array = np.asarray(residuals)
    return (
        {
            "cells": float(len(usable)),
            "laplace_nll_mean": float(values.mean()),
            "mode_residual_median": float(np.median(residuals_array)),
            "mode_residual_q99": float(np.quantile(residuals_array, 0.99)),
            "mode_residual_max": float(residuals_array.max()),
        },
        values,
    )


def independent_poisson_cell_nll(
    data: ad.AnnData, rows: np.ndarray, mean_rates: np.ndarray
) -> np.ndarray:
    usable = rows[: len(rows) // BATCH_SIZE * BATCH_SIZE]
    log_rates = np.log(mean_rates)
    values = []
    for start in range(0, len(usable), BATCH_SIZE):
        counts, totals = dense_counts(data, usable[start : start + BATCH_SIZE])
        values.append(totals - counts @ log_rates)
    return np.concatenate(values)


def perturbed_start(
    initial: dict[str, jax.Array], start_index: int
) -> dict[str, jax.Array]:
    """Return one of three deterministic, nearby starts."""
    if start_index == 0:
        return {name: value.copy() for name, value in initial.items()}
    rng = np.random.default_rng(np.random.SeedSequence([17, start_index]))
    arrays = parameter_arrays(initial)
    scale = np.std(arrays["loadings"], axis=0, keepdims=True)
    arrays["loadings"] += (
        0.05 * scale * rng.normal(size=arrays["loadings"].shape)
    ).astype(np.float32)
    arrays["raw_d"] += rng.normal(scale=0.02, size=arrays["raw_d"].shape).astype(
        np.float32
    )
    return {name: jnp.asarray(value) for name, value in arrays.items()}


def load_or_create_initialization(
    data: ad.AnnData,
    train_rows: np.ndarray,
    selection_rows: np.ndarray,
    evaluation_rows: np.ndarray,
) -> tuple[dict[str, jax.Array], np.ndarray, np.ndarray, np.ndarray]:
    if INITIALIZATION.exists():
        with np.load(INITIALIZATION, allow_pickle=False) as saved:
            for name, expected in [
                ("train_rows", train_rows),
                ("selection_rows", selection_rows),
                ("evaluation_rows", evaluation_rows),
            ]:
                if not np.array_equal(saved[name], expected):
                    raise ValueError(f"cached full Model 3 {name} do not match")
            parameters = {
                name: jnp.asarray(saved[name])
                for name in ["alpha", "loadings", "raw_d"]
            }
            return (
                parameters,
                saved["pca_loadings"],
                saved["total_variance"],
                saved["mean_rates"],
            )
    work_path = DERIVED / "model3_full_centered_train.f32"
    try:
        parameters, pca_loadings, total_variance, mean_rates = initialize_full_model(
            data, train_rows, work_path
        )
    finally:
        work_path.unlink(missing_ok=True)
    save_parameters(
        INITIALIZATION,
        parameters,
        pca_loadings=pca_loadings,
        total_variance=total_variance,
        mean_rates=mean_rates,
        train_rows=train_rows,
        selection_rows=selection_rows,
        evaluation_rows=evaluation_rows,
    )
    return parameters, pca_loadings, total_variance, mean_rates


def validate_cached_rows(path: Path, train_rows: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as saved:
        if "train_rows" not in saved or not np.array_equal(
            saved["train_rows"], train_rows
        ):
            raise ValueError(f"{relative(path)} has a different training split")


def fit_model_ladder(
    data: ad.AnnData,
    train_rows: np.ndarray,
    selection_rows: np.ndarray,
    evaluation_rows: np.ndarray,
    *,
    resume: bool,
    outer_iterations: int,
    m_step_iterations: int,
) -> tuple[
    np.ndarray,
    dict[str, jax.Array],
    dict[str, np.ndarray],
    dict[str, jax.Array],
    pd.DataFrame,
]:
    initial, pca_loadings, total_variance, mean_rates = load_or_create_initialization(
        data, train_rows, selection_rows, evaluation_rows
    )
    REPORT.mkdir(parents=True, exist_ok=True)
    initialization_audit = pd.DataFrame(
        [
            {
                "genes": data.n_vars,
                "train_cells": len(train_rows),
                "factorial_variance_positive_fraction": np.mean(total_variance > 0),
                "factorial_variance_median": np.median(total_variance),
                "pca_factor_variance_median": np.median(np.square(pca_loadings).sum(1)),
                "full_initial_factor_variance_median": np.median(
                    np.square(np.asarray(initial["loadings"])).sum(1)
                ),
                "full_initial_residual_variance_median": np.median(
                    np.asarray(population_parameters(initial)[2])
                ),
            }
        ]
    )
    initialization_audit.to_csv(
        REPORT / "model3_full_initialization_audit.csv", index=False
    )

    if resume and INDEPENDENT_D_FIT.exists():
        validate_cached_rows(INDEPENDENT_D_FIT, train_rows)
        independent_d = load_parameters(INDEPENDENT_D_FIT)
    else:
        independent_initial = make_parameters(
            mean_rates,
            np.empty((data.n_vars, 0)),
            np.maximum(total_variance, D_FLOOR * 1.01),
        )
        independent_d, independent_trace = fit_em(
            data,
            train_rows,
            independent_initial,
            outer_iterations=outer_iterations,
            m_step_iterations=m_step_iterations,
        )
        independent_trace.insert(0, "model", "independent_poisson_lognormal_D")
        independent_trace.to_csv(
            REPORT / "model3_independent_d_fit_trace.csv", index=False
        )
        save_parameters(INDEPENDENT_D_FIT, independent_d, train_rows=train_rows)

    pure_start_rows = []
    pure_fits: list[dict[str, np.ndarray]] = []
    initial_pure = pure_parameters(
        mean_rates, np.asarray(pca_loadings, dtype=np.float64)
    )
    for start_index in range(N_STARTS):
        path = DERIVED / f"model3_pure_factor_start_{start_index}.npz"
        if resume and path.exists():
            validate_cached_rows(path, train_rows)
            with np.load(path, allow_pickle=False) as saved:
                fitted_pure = {name: saved[name] for name in ["m", "loadings"]}
            training_tail = np.nan
        else:
            fitted_pure, training_tail = fit_pure_start(
                data,
                train_rows,
                initial_pure["m"],
                initial_pure["loadings"],
                start_index,
            )
            np.savez_compressed(
                path,
                **fitted_pure,
                train_rows=train_rows,
            )
        selection_diagnostics, _ = evaluate_pure_cells(
            data, selection_rows, fitted_pure
        )
        pure_fits.append(fitted_pure)
        pure_start_rows.append(
            {
                "model": "pure_factor_LL_T",
                "start": start_index,
                "training_tail_laplace_nll": training_tail,
                "selection_laplace_nll": selection_diagnostics["laplace_nll_mean"],
                "selection_mode_residual_q99": selection_diagnostics[
                    "mode_residual_q99"
                ],
            }
        )
    pure_best = int(
        np.argmin([row["selection_laplace_nll"] for row in pure_start_rows])
    )
    pure = pure_fits[pure_best]
    np.savez_compressed(
        PURE_FIT,
        **pure,
        train_rows=train_rows,
        selected_start=np.asarray(pure_best),
    )

    full_start_rows = []
    full_fits = []
    traces = []
    for start_index in range(N_STARTS):
        path = DERIVED / f"model3_full_start_{start_index}.npz"
        trace_path = REPORT / f"model3_full_fit_trace_start_{start_index}.csv"
        if resume and path.exists():
            validate_cached_rows(path, train_rows)
            fitted_full = load_parameters(path)
        else:
            fitted_full, trace = fit_em(
                data,
                train_rows,
                perturbed_start(initial, start_index),
                outer_iterations=outer_iterations,
                m_step_iterations=m_step_iterations,
                checkpoint_prefix=DERIVED / f"model3_full_start_{start_index}",
                trace_path=trace_path,
                resume=resume,
                require_convergence=True,
            )
            trace.insert(0, "start", start_index)
            atomic_write_frame(trace_path, trace)
            save_parameters(path, fitted_full, train_rows=train_rows)
        if trace_path.exists():
            traces.append(pd.read_csv(trace_path))
        _, selection_diagnostics, _ = run_e_step(
            data, selection_rows, fitted_full, collect_moments=False
        )
        full_fits.append(fitted_full)
        full_start_rows.append(
            {
                "model": "full_LL_T_plus_D",
                "start": start_index,
                "selection_laplace_nll": selection_diagnostics["laplace_nll_mean"],
                "selection_mode_residual_q99": selection_diagnostics[
                    "mode_residual_q99"
                ],
            }
        )
    full_best = int(
        np.argmin([row["selection_laplace_nll"] for row in full_start_rows])
    )
    full = full_fits[full_best]
    save_parameters(
        FIT,
        full,
        train_rows=train_rows,
        selected_start=np.asarray(full_best),
    )
    if traces:
        atomic_write_frame(
            REPORT / "model3_full_fit_trace.csv",
            pd.concat(traces, ignore_index=True),
        )
    starts = pd.DataFrame(pure_start_rows + full_start_rows)
    starts["selected"] = (
        starts["model"].eq("pure_factor_LL_T") & starts["start"].eq(pure_best)
    ) | (starts["model"].eq("full_LL_T_plus_D") & starts["start"].eq(full_best))
    starts.to_csv(REPORT / "model3_full_fit_starts.csv", index=False)
    return mean_rates, independent_d, pure, full, starts


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = values.mean()
    indices = np.arange(2 ** len(values), dtype=np.uint64)[:, None]
    bit_positions = np.arange(len(values), dtype=np.uint64)
    signs = 2 * ((indices >> bit_positions) & 1).astype(np.int8) - 1
    null = (signs * values[None, :]).mean(1)
    return float(np.mean(null >= observed))


def benchmark_ladder(
    data: ad.AnnData,
    evaluation_rows: np.ndarray,
    mean_rates: np.ndarray,
    independent_d: dict[str, jax.Array],
    pure: dict[str, np.ndarray],
    full: dict[str, jax.Array],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = np.sort(evaluation_rows[: len(evaluation_rows) // BATCH_SIZE * BATCH_SIZE])
    independent_poisson = independent_poisson_cell_nll(data, usable, mean_rates)
    _, independent_d_diagnostics, independent_d_values = run_e_step(
        data, usable, independent_d, collect_moments=False
    )
    pure_diagnostics, pure_values = evaluate_pure_cells(data, usable, pure)
    _, full_diagnostics, full_values = run_e_step(
        data, usable, full, collect_moments=False
    )
    values = {
        "independent_poisson": independent_poisson,
        "independent_poisson_lognormal_D": independent_d_values,
        "pure_factor_LL_T": pure_values,
        "full_LL_T_plus_D": full_values,
    }
    cell_table = pd.DataFrame(
        {
            "source_row": usable,
            "guide_id": data.obs.iloc[usable]["guide_id"].astype(str).to_numpy(),
            "batch": data.obs.iloc[usable]["batch"].astype(str).to_numpy(),
            **values,
        }
    )
    long = cell_table.melt(
        id_vars=["source_row", "guide_id", "batch"],
        var_name="model",
        value_name="laplace_nll_without_count_constant",
    )
    by_guide_batch = (
        long.groupby(["model", "guide_id", "batch"], as_index=False)
        .agg(
            mean_nll=("laplace_nll_without_count_constant", "mean"),
            n_cells=("source_row", "size"),
        )
        .sort_values(["model", "guide_id", "batch"])
    )
    comparisons = [
        ("independent_poisson", "independent_poisson_lognormal_D"),
        ("independent_poisson_lognormal_D", "pure_factor_LL_T"),
        ("independent_poisson_lognormal_D", "full_LL_T_plus_D"),
        ("pure_factor_LL_T", "full_LL_T_plus_D"),
    ]
    summary_rows = [
        {
            "record": "model",
            "baseline": "",
            "model": name,
            "heldout_nll_mean": value.mean(),
            "mean_nll_improvement": np.nan,
            "guide_mean_improvement": np.nan,
            "guide_standard_error": np.nan,
            "exact_guide_sign_flip_p_one_sided": np.nan,
            "n_cells": len(value),
            "n_guides": cell_table["guide_id"].nunique(),
        }
        for name, value in values.items()
    ]
    for baseline, model in comparisons:
        difference = values[baseline] - values[model]
        guide_means = (
            pd.DataFrame({"guide_id": cell_table["guide_id"], "difference": difference})
            .groupby("guide_id")["difference"]
            .mean()
            .to_numpy()
        )
        summary_rows.append(
            {
                "record": "comparison",
                "baseline": baseline,
                "model": model,
                "heldout_nll_mean": values[model].mean(),
                "mean_nll_improvement": difference.mean(),
                "guide_mean_improvement": guide_means.mean(),
                "guide_standard_error": guide_means.std(ddof=1)
                / np.sqrt(len(guide_means)),
                "exact_guide_sign_flip_p_one_sided": exact_sign_flip_p(guide_means),
                "n_cells": len(difference),
                "n_guides": len(guide_means),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["independent_d_mode_residual_q99"] = independent_d_diagnostics[
        "mode_residual_q99"
    ]
    summary["pure_mode_residual_q99"] = pure_diagnostics["mode_residual_q99"]
    summary["full_mode_residual_q99"] = full_diagnostics["mode_residual_q99"]
    return summary, by_guide_batch


def random_matrix_benchmark(
    data: ad.AnnData,
    evaluation_rows: np.ndarray,
    full: dict[str, jax.Array],
    *,
    draws: int = RANDOM_MATRIX_DRAWS,
) -> pd.DataFrame:
    mu, loadings_jax, d_jax = population_parameters(full)
    loadings = np.asarray(loadings_jax)
    d = np.asarray(d_jax)
    mean_rates = np.exp(np.asarray(mu) + 0.5 * (np.square(loadings).sum(1) + d))
    labels = data.obs["guide_id"].astype(str).to_numpy()
    rng = np.random.default_rng(23)
    subset = []
    for guide in sorted(np.unique(labels[evaluation_rows])):
        available = evaluation_rows[labels[evaluation_rows] == guide]
        subset.extend(rng.choice(available, 16, replace=False).tolist())
    subset = np.sort(np.asarray(subset, dtype=np.int64))
    _, observed_diagnostics, _ = run_e_step(data, subset, full, collect_moments=False)
    norms = np.linalg.norm(loadings, axis=1)
    unit = np.divide(
        loadings,
        norms[:, None],
        out=np.zeros_like(loadings),
        where=norms[:, None] > 0,
    )
    nonzero = np.flatnonzero(norms > 0)
    records = [
        {
            "draw": -1,
            "model": "fitted_full_LL_T_plus_D",
            "mean_laplace_nll": observed_diagnostics["laplace_nll_mean"],
            "n_cells": len(subset),
        }
    ]
    for draw in range(draws):
        remapped = unit.copy()
        remapped[nonzero] = unit[rng.permutation(nonzero)]
        null_loadings = norms[:, None] * remapped
        null = make_parameters(mean_rates, null_loadings, d)
        _, diagnostics, _ = run_e_step(data, subset, null, collect_moments=False)
        records.append(
            {
                "draw": draw,
                "model": "diagonal_matched_random_loading_directions",
                "mean_laplace_nll": diagnostics["laplace_nll_mean"],
                "n_cells": len(subset),
            }
        )
    table = pd.DataFrame(records)
    fitted_nll = table.loc[table["draw"].eq(-1), "mean_laplace_nll"].item()
    null_nll = table.loc[table["draw"].ge(0), "mean_laplace_nll"]
    table["monte_carlo_p_null_at_least_as_good"] = (
        1 + np.count_nonzero(null_nll <= fitted_nll)
    ) / (draws + 1)
    return table


def sample_full_prior(
    parameters: dict[str, jax.Array],
    totals: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    mu, loadings, d = population_parameters(parameters)
    mu = np.asarray(mu)
    loadings = np.asarray(loadings)
    d = np.asarray(d)
    z = rng.normal(size=(len(totals), loadings.shape[1]))
    residual = rng.normal(size=(len(totals), len(mu))) * np.sqrt(d)[None, :]
    log_rates = mu[None, :] + z @ loadings.T + residual
    rates = totals[:, None] * np.exp(log_rates)
    if not np.isfinite(rates).all():
        raise FloatingPointError("prior-predictive rates are non-finite")
    return rng.poisson(rates).astype(np.int32)


def sample_pure_prior(
    parameters: dict[str, np.ndarray],
    totals: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    m = np.asarray(
        normalize_pure_population(
            jnp.asarray(parameters["m"]), jnp.asarray(parameters["loadings"])
        )
    )
    loadings = parameters["loadings"]
    z = rng.normal(size=(len(totals), loadings.shape[1]))
    rates = totals[:, None] * np.exp(m[None, :] + z @ loadings.T)
    if not np.isfinite(rates).all():
        raise FloatingPointError("pure-factor prior-predictive rates are non-finite")
    return rng.poisson(rates).astype(np.int32)


def sampled_count_metrics(
    generated: np.ndarray,
    observed: np.ndarray,
    feature_indices: np.ndarray,
    target_indices: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    generated_totals = generated.sum(1)
    observed_totals = observed.sum(1)

    def selected_log_counts(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
        totals = values.sum(1)
        return np.log1p(
            values[:, indices] * (CELL_TARGET_SUM / np.maximum(totals, 1))[:, None]
        )

    generated_feature = selected_log_counts(generated, feature_indices)
    observed_feature = selected_log_counts(observed, feature_indices)
    generated_target = selected_log_counts(generated, target_indices)
    observed_target = selected_log_counts(observed, target_indices)
    generated_feature -= generated_feature.mean(0)
    observed_feature -= observed_feature.mean(0)
    generated_target -= generated_target.mean(0)
    observed_target -= observed_target.mean(0)
    generated_covariance = generated_feature.T @ generated_target / (len(generated) - 1)
    observed_covariance = observed_feature.T @ observed_target / (len(observed) - 1)
    generated_norm = np.linalg.norm(generated_covariance, axis=0)
    observed_norm = np.linalg.norm(observed_covariance, axis=0)
    similarities = (generated_covariance.T @ observed_covariance) / np.maximum(
        generated_norm[:, None] * observed_norm[None, :], 1e-15
    )
    matched = np.diag(similarities)
    null = np.empty(1_000)
    columns = np.arange(len(target_indices))
    for draw in range(len(null)):
        null[draw] = np.median(similarities[columns, rng.permutation(columns)])
    generated_mean = generated.mean(0)
    observed_mean = observed.mean(0)
    generated_variance = generated.var(0)
    observed_variance = observed.var(0)
    total_ratio = generated_totals / observed_totals
    return {
        "zero_fraction_abs_error": float(
            abs(np.mean(generated == 0) - np.mean(observed == 0))
        ),
        "log1p_gene_mean_pearson": float(
            pearsonr(np.log1p(generated_mean), np.log1p(observed_mean)).statistic
        ),
        "log1p_gene_variance_spearman": float(
            spearmanr(
                np.log1p(generated_variance), np.log1p(observed_variance)
            ).statistic
        ),
        "downstream_covariance_median_cosine": float(np.median(matched)),
        "downstream_covariance_random_pairing_p": float(
            (1 + np.count_nonzero(null >= np.median(matched))) / (len(null) + 1)
        ),
        "total_count_ratio_q01": float(np.quantile(total_ratio, 0.01)),
        "total_count_ratio_median": float(np.median(total_ratio)),
        "total_count_ratio_q99": float(np.quantile(total_ratio, 0.99)),
    }


def prior_predictive_benchmark(
    data: ad.AnnData,
    evaluation_rows: np.ndarray,
    mean_rates: np.ndarray,
    independent_d: dict[str, jax.Array],
    pure: dict[str, np.ndarray],
    full: dict[str, jax.Array],
    *,
    n_cells: int = 256,
    replicates: int = 3,
) -> pd.DataFrame:
    rng = np.random.default_rng(73)
    sample_rows = np.sort(rng.choice(evaluation_rows, n_cells, replace=False))
    observed = data.X[sample_rows].toarray().astype(np.int32)
    totals = observed.sum(1)
    target_table = pd.read_csv(TARGET_TABLE)
    gene_lookup = {gene: index for index, gene in enumerate(data.var_names.astype(str))}
    target_indices = np.asarray(
        [gene_lookup[target] for target in target_table["target_gene"].astype(str)]
    )
    downstream = np.ones(data.n_vars, dtype=bool)
    downstream[target_indices] = False
    available = np.flatnonzero(downstream)
    feature_indices = available[np.argsort(mean_rates[available])[-1_024:]]
    generators = {
        "independent_poisson": lambda local_rng: local_rng.poisson(
            totals[:, None] * mean_rates[None, :]
        ).astype(np.int32),
        "independent_poisson_lognormal_D": lambda local_rng: sample_full_prior(
            independent_d, totals, local_rng
        ),
        "pure_factor_LL_T": lambda local_rng: sample_pure_prior(
            pure, totals, local_rng
        ),
        "full_LL_T_plus_D": lambda local_rng: sample_full_prior(
            full, totals, local_rng
        ),
    }
    records = []
    for model_index, (name, generator) in enumerate(generators.items()):
        for replicate in range(replicates):
            local_rng = np.random.default_rng(
                np.random.SeedSequence([73, model_index, replicate])
            )
            generated = generator(local_rng)
            records.append(
                {
                    "generator": name,
                    "replicate": replicate,
                    "n_cells": n_cells,
                    **sampled_count_metrics(
                        generated,
                        observed,
                        feature_indices,
                        target_indices,
                        local_rng,
                    ),
                }
            )
            print(
                f"prior-predictive {name} replicate {replicate + 1}/{replicates}",
                flush=True,
            )
    return pd.DataFrame(records)


def full_posterior_predictive_diagnostics(
    data: ad.AnnData,
    evaluation_rows: np.ndarray,
    full: dict[str, jax.Array],
    *,
    n_cells: int = 256,
) -> pd.DataFrame:
    rng = np.random.default_rng(79)
    rows = np.sort(rng.choice(evaluation_rows, n_cells, replace=False))
    mu, loadings_jax, d_jax = population_parameters(full)
    loadings = np.asarray(loadings_jax)
    d = np.asarray(d_jax)
    observed_blocks = []
    generated_blocks = []
    modal_total_ratios = []
    mean_total_ratios = []
    residuals = []
    s_minimum = []
    s_condition = []
    for start in range(0, len(rows), BATCH_SIZE):
        counts, totals = dense_counts(data, rows[start : start + BATCH_SIZE])
        mode = INFER_BATCH(
            mu, loadings_jax, d_jax, jnp.asarray(counts), jnp.asarray(totals)
        )
        if not np.asarray(mode.converged).all():
            raise RuntimeError("posterior-predictive mode did not converge")
        expected = np.asarray(POSTERIOR_MEAN_BATCH(mode, loadings_jax))
        generated = sample_posterior_predictive(mode, loadings, totals, rng)
        observed_blocks.append(counts.astype(np.int32))
        generated_blocks.append(generated)
        modal_total_ratios.extend((np.asarray(mode.rates).sum(1) / totals).tolist())
        mean_total_ratios.extend((expected.sum(1) / totals).tolist())
        residuals.extend(np.asarray(mode.residual).tolist())
        eigenvalues = np.linalg.eigvalsh(np.asarray(mode.s))
        s_minimum.extend(eigenvalues[:, 0].tolist())
        s_condition.extend((eigenvalues[:, -1] / eigenvalues[:, 0]).tolist())
    observed = np.vstack(observed_blocks)
    generated = np.vstack(generated_blocks)
    observed_mean = observed.mean(0)
    generated_mean = generated.mean(0)
    observed_variance = observed.var(0)
    generated_variance = generated.var(0)
    population_rates = np.exp(np.asarray(mu) + 0.5 * (np.square(loadings).sum(1) + d))
    metrics = {
        "sum_population_mean_rates": population_rates.sum(),
        "q_transpose_L_norm": np.linalg.norm(population_rates @ loadings),
        "observed_gene_zero_fraction": np.mean(observed == 0),
        "posterior_predictive_gene_zero_fraction": np.mean(generated == 0),
        "posterior_predictive_log1p_gene_mean_pearson": pearsonr(
            np.log1p(observed_mean), np.log1p(generated_mean)
        ).statistic,
        "posterior_predictive_log1p_gene_variance_spearman": spearmanr(
            np.log1p(observed_variance), np.log1p(generated_variance)
        ).statistic,
        "posterior_modal_total_rate_ratio_q01": np.quantile(modal_total_ratios, 0.01),
        "posterior_modal_total_rate_ratio_median": np.median(modal_total_ratios),
        "posterior_modal_total_rate_ratio_q99": np.quantile(modal_total_ratios, 0.99),
        "posterior_mean_total_rate_ratio_q01": np.quantile(mean_total_ratios, 0.01),
        "posterior_mean_total_rate_ratio_median": np.median(mean_total_ratios),
        "posterior_mean_total_rate_ratio_q99": np.quantile(mean_total_ratios, 0.99),
        "mode_residual_q99": np.quantile(residuals, 0.99),
        "mode_residual_max": np.max(residuals),
        "schur_complement_minimum_eigenvalue_q01": np.quantile(s_minimum, 0.01),
        "schur_complement_condition_q99": np.quantile(s_condition, 0.99),
    }
    return pd.DataFrame(
        [{"metric": name, "value": value} for name, value in metrics.items()]
    )


def source_posterior_mean_sum(
    data: ad.AnnData,
    rows: np.ndarray,
    parameters: dict[str, jax.Array],
) -> tuple[np.ndarray, np.ndarray]:
    mu, loadings, d = population_parameters(parameters)
    expected_sum = np.zeros(data.n_vars)
    raw_sum = np.zeros(data.n_vars)
    for start in range(0, len(rows), BATCH_SIZE):
        counts, totals = dense_counts(data, rows[start : start + BATCH_SIZE])
        mode = INFER_BATCH(mu, loadings, d, jnp.asarray(counts), jnp.asarray(totals))
        if not np.asarray(mode.converged).all():
            raise RuntimeError("source-cell posterior mode did not converge")
        expected_sum += np.asarray(POSTERIOR_MEAN_BATCH(mode, loadings)).sum(0)
        raw_sum += counts.sum(0)
    return expected_sum, raw_sum


def solve_lograte_amplitude(
    baseline_sum: np.ndarray,
    direction: np.ndarray,
    target_index: int,
    target_shift: float,
) -> tuple[float, float, np.ndarray]:
    """Match a bulk target shift without clipping the log-rate response."""
    log_baseline = np.log(np.asarray(baseline_sum, dtype=np.float64))
    baseline_target = log_pseudobulk(baseline_sum)[target_index]

    def residual(amplitude: float) -> float:
        log_weights = log_baseline + amplitude * direction
        target_fraction = np.exp(log_weights[target_index] - logsumexp(log_weights))
        shifted_target = np.log1p(BULK_TARGET_SUM * target_fraction)
        return float(shifted_target - baseline_target - target_shift)

    at_zero = residual(0.0)
    if abs(at_zero) < 1e-12:
        return 0.0, target_shift, baseline_sum.copy()
    lower = -1.0
    while residual(lower) > 0 and lower > -4096:
        lower *= 2
    if residual(lower) > 0:
        raise ValueError("could not bracket a non-positive log-rate amplitude")
    amplitude = float(brentq(residual, lower, 0.0, xtol=1e-10, rtol=1e-10))
    log_weights = log_baseline + amplitude * direction
    probabilities = np.exp(log_weights - logsumexp(log_weights))
    shifted = baseline_sum.sum() * probabilities
    return amplitude, target_shift + residual(amplitude), shifted


def generate_expected_profiles(
    full: dict[str, jax.Array], strict_guides: list[str]
) -> dict[str, Path]:
    target_table = pd.read_csv(TARGET_TABLE)
    benchmark = target_table[target_table["benchmark_candidate"]].copy()
    calibration = pd.read_csv(CALIBRATION).set_index("target_gene")
    data = ad.read_h5ad(H1_PATH, backed="r")
    try:
        genes = data.var_names.astype(str).tolist()
        lookup = {gene: index for index, gene in enumerate(genes)}
        rng = np.random.default_rng(np.random.SeedSequence([0, 3, 0]))
        source_rows = np.sort(sample_balanced_controls(data.obs, strict_guides, rng))
        baseline_sum, raw_sum = source_posterior_mean_sum(data, source_rows, full)
    finally:
        data.file.close()
    _, loadings_jax, d_jax = population_parameters(full)
    loadings = np.asarray(loadings_jax)
    d = np.asarray(d_jax)
    all_target_indices = np.asarray(
        [lookup[target] for target in target_table["target_gene"].astype(str)]
    )
    full_directions = covariance_columns(loadings, d, all_target_indices)
    shared_directions = loadings @ loadings[all_target_indices].T
    target_to_column = {
        target: index
        for index, target in enumerate(target_table["target_gene"].astype(str))
    }
    expected = np.empty((len(benchmark), len(genes)), dtype=np.float32)
    shared_expected = np.empty_like(expected)
    closure = []
    for output_index, row in enumerate(benchmark.itertuples(index=False)):
        target = str(row.target_gene)
        target_index = lookup[target]
        column = target_to_column[target]
        transferred = float(calibration.loc[target, "leave_one_out_knockdown_depth"])
        intended = intended_target_shift(
            baseline_sum[target_index] / baseline_sum.sum(), transferred
        )
        full_amplitude, full_realized, full_sum = solve_lograte_amplitude(
            baseline_sum,
            full_directions[:, column],
            target_index,
            intended,
        )
        shared_amplitude, shared_realized, shared_sum = solve_lograte_amplitude(
            baseline_sum,
            shared_directions[:, column],
            target_index,
            intended,
        )
        expected[output_index] = log_pseudobulk(full_sum)
        shared_expected[output_index] = log_pseudobulk(shared_sum)
        downstream = np.ones(len(genes), dtype=bool)
        downstream[target_index] = False
        full_response = full_amplitude * full_directions[:, column]
        shared_response = shared_amplitude * shared_directions[:, column]
        closure.append(
            {
                "target_gene": target,
                "source_order": int(row.source_order),
                "observed_knockdown_depth": calibration.loc[
                    target, "observed_knockdown_depth"
                ],
                "leave_one_out_knockdown_depth": transferred,
                "intended_target_shift": intended,
                "full_expected_realized_shift": full_realized,
                "shared_expected_realized_shift": shared_realized,
                "full_amplitude": full_amplitude,
                "shared_only_amplitude": shared_amplitude,
                "full_direction_diagonal": full_directions[target_index, column],
                "shared_direction_diagonal": shared_directions[target_index, column],
                "diagonal_residual_variance": d[target_index],
                "full_downstream_lograte_norm": np.linalg.norm(
                    full_response[downstream]
                ),
                "shared_downstream_lograte_norm": np.linalg.norm(
                    shared_response[downstream]
                ),
                "full_over_shared_downstream_norm": np.linalg.norm(
                    full_response[downstream]
                )
                / max(np.linalg.norm(shared_response[downstream]), 1e-15),
                "sampled_realized_shift": np.nan,
            }
        )
    raw_null = np.repeat(
        log_pseudobulk(raw_sum)[None, :], len(benchmark), axis=0
    ).astype(np.float32)
    posterior_null = np.repeat(
        log_pseudobulk(baseline_sum)[None, :], len(benchmark), axis=0
    ).astype(np.float32)
    artifact = DERIVED / "model3_full_expected_profiles.npz"
    closure_path = REPORT / "model3_full_on_target_closure.csv"
    np.savez_compressed(
        artifact,
        target_gene=benchmark["target_gene"].astype(str).to_numpy(),
        source_order=benchmark["source_order"].to_numpy(dtype=np.int64),
        gene_names=np.asarray(genes),
        expected_log_bulk=expected,
        shared_only_expected_log_bulk=shared_expected,
        null_log_bulk=raw_null,
        posterior_null_log_bulk=posterior_null,
        source_rows=source_rows,
        directions=full_directions.astype(np.float32),
        shared_directions=shared_directions.astype(np.float32),
    )
    pd.DataFrame(closure).to_csv(closure_path, index=False)
    return {"expected_profiles": artifact, "on_target_closure": closure_path}


def regularization_audit(
    independent_d: dict[str, jax.Array],
    pure: dict[str, np.ndarray],
    full: dict[str, jax.Array],
) -> pd.DataFrame:
    _, _, independent_variance = population_parameters(independent_d)
    _, full_loadings, full_variance = population_parameters(full)
    pure_loadings = pure["loadings"]
    return pd.DataFrame(
        [
            {
                "model": "independent_poisson",
                "covariance": "none",
                "structural_constraint": "independent genes",
                "penalty": "none",
                "sparsity_prior": False,
                "exact_loading_zero_fraction": np.nan,
                "residual_variance_median": 0.0,
            },
            {
                "model": "independent_poisson_lognormal_D",
                "covariance": "diagonal D",
                "structural_constraint": "independent gene log-rates",
                "penalty": "none",
                "sparsity_prior": False,
                "exact_loading_zero_fraction": np.nan,
                "residual_variance_median": np.median(np.asarray(independent_variance)),
            },
            {
                "model": "pure_factor_LL_T",
                "covariance": "L L.T",
                "structural_constraint": f"rank at most {RANK}",
                "penalty": f"L2 loadings coefficient {LOADING_PENALTY}",
                "sparsity_prior": False,
                "exact_loading_zero_fraction": np.mean(pure_loadings == 0),
                "residual_variance_median": 0.0,
            },
            {
                "model": "full_LL_T_plus_D",
                "covariance": "L L.T + D",
                "structural_constraint": f"rank-{RANK} shared plus diagonal",
                "penalty": f"L2 loadings coefficient {LOADING_PENALTY}",
                "sparsity_prior": False,
                "exact_loading_zero_fraction": np.mean(np.asarray(full_loadings) == 0),
                "residual_variance_median": np.median(np.asarray(full_variance)),
            },
        ]
    )


def run_full_pipeline(
    *,
    resume: bool,
    outer_iterations: int,
    m_step_iterations: int,
    random_matrix_draws: int,
) -> dict[str, object]:
    strict_guides = pd.read_csv(STRICT_CONTROLS)["guide_id"].astype(str).tolist()
    data = ad.read_h5ad(H1_PATH, backed="r")
    try:
        train_rows, selection_rows, evaluation_rows, guides = split_control_rows(
            data.obs, strict_guides
        )
        split_rows = []
        for role, role_guides in guides.items():
            for guide in role_guides:
                split_rows.append(
                    {
                        "role": role,
                        "guide_id": guide,
                        "n_cells": int(
                            np.sum(
                                data.obs["target_gene"].astype(str).eq("non-targeting")
                                & data.obs["guide_id"].astype(str).eq(guide)
                            )
                        ),
                    }
                )
        REPORT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(split_rows).to_csv(
            REPORT / "model3_full_control_split.csv", index=False
        )
        mean_rates, independent_d, pure, full, starts = fit_model_ladder(
            data,
            train_rows,
            selection_rows,
            evaluation_rows,
            resume=resume,
            outer_iterations=outer_iterations,
            m_step_iterations=m_step_iterations,
        )
        summary, by_guide_batch = benchmark_ladder(
            data,
            evaluation_rows,
            mean_rates,
            independent_d,
            pure,
            full,
        )
        random_matrix = random_matrix_benchmark(
            data,
            evaluation_rows,
            full,
            draws=random_matrix_draws,
        )
        prior_predictive = prior_predictive_benchmark(
            data,
            evaluation_rows,
            mean_rates,
            independent_d,
            pure,
            full,
        )
        posterior_predictive = full_posterior_predictive_diagnostics(
            data, evaluation_rows, full
        )
    finally:
        data.file.close()

    outputs = {
        "benchmark": REPORT / "model3_full_heldout_benchmark.csv",
        "benchmark_by_guide_batch": REPORT
        / "model3_full_heldout_benchmark_by_guide_batch.csv",
        "random_matrix": REPORT / "model3_full_random_matrix_benchmark.csv",
        "regularization": REPORT / "model3_full_regularization_audit.csv",
        "prior_predictive": REPORT / "model3_full_prior_predictive.csv",
        "posterior_predictive": REPORT / "model3_full_posterior_predictive.csv",
    }
    summary.to_csv(outputs["benchmark"], index=False)
    by_guide_batch.to_csv(outputs["benchmark_by_guide_batch"], index=False)
    random_matrix.to_csv(outputs["random_matrix"], index=False)
    regularization_audit(independent_d, pure, full).to_csv(
        outputs["regularization"], index=False
    )
    prior_predictive.to_csv(outputs["prior_predictive"], index=False)
    posterior_predictive.to_csv(outputs["posterior_predictive"], index=False)
    comparison = summary[
        summary["record"].eq("comparison")
        & summary["baseline"].eq("pure_factor_LL_T")
        & summary["model"].eq("full_LL_T_plus_D")
    ].iloc[0]
    full_over_independent = summary[
        summary["record"].eq("comparison")
        & summary["baseline"].eq("independent_poisson_lognormal_D")
        & summary["model"].eq("full_LL_T_plus_D")
    ].iloc[0]
    posterior = posterior_predictive.set_index("metric")["value"]
    validation_gate = {
        "full_improves_over_independent_D": bool(
            full_over_independent["mean_nll_improvement"] > 0
        ),
        "full_improves_over_pure_factor": bool(comparison["mean_nll_improvement"] > 0),
        "all_evaluation_modes_converged": bool(
            posterior["mode_residual_max"] <= MODE_TOLERANCE
        ),
        "population_mean_is_normalized": bool(
            abs(posterior["sum_population_mean_rates"] - 1) < 1e-8
        ),
        "posterior_zero_fraction_error_below_0.05": bool(
            abs(
                posterior["posterior_predictive_gene_zero_fraction"]
                - posterior["observed_gene_zero_fraction"]
            )
            < 0.05
        ),
        "posterior_mean_correlation_above_0.95": bool(
            posterior["posterior_predictive_log1p_gene_mean_pearson"] > 0.95
        ),
        "posterior_variance_correlation_above_0.5": bool(
            posterior["posterior_predictive_log1p_gene_variance_spearman"] > 0.5
        ),
    }
    if all(validation_gate.values()):
        outputs.update(generate_expected_profiles(full, strict_guides))
    result: dict[str, object] = {
        "status": (
            "validation_passed"
            if all(validation_gate.values())
            else "stopped_at_validation_gate"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "model": "rank-50 Poisson-lognormal with Sigma = L L.T + D",
        "truth_cells_read": False,
        "fit_controls": "strict 26-guide control pool only",
        "train_guides": guides["training"],
        "selection_guides": guides["selection"],
        "evaluation_guides": guides["evaluation"],
        "start_selection_used_evaluation_guides": False,
        "selected_pure_start": int(
            starts.loc[
                starts["model"].eq("pure_factor_LL_T") & starts["selected"],
                "start",
            ].item()
        ),
        "selected_full_start": int(
            starts.loc[
                starts["model"].eq("full_LL_T_plus_D") & starts["selected"],
                "start",
            ].item()
        ),
        "full_over_pure_mean_nll_improvement": float(
            comparison["mean_nll_improvement"]
        ),
        "full_over_pure_exact_guide_sign_flip_p_one_sided": float(
            comparison["exact_guide_sign_flip_p_one_sided"]
        ),
        "validation_gate": validation_gate,
        "random_matrix_test": {
            "draws": random_matrix_draws,
            "null": "permute learned loading directions across genes while preserving each gene's shared variance and D",
            "monte_carlo_p_null_at_least_as_good": float(
                random_matrix["monte_carlo_p_null_at_least_as_good"].iloc[0]
            ),
        },
        "sparsity_prior": False,
        "sparsity_note": "No covariance sparsity prior is used; rank 50, diagonal D, and weak L2 are the declared regularizers.",
        "response_direction": {
            "primary": "full fitted covariance column from L L.T + D",
            "sensitivity": "shared-only L L.T column with independently output-matched amplitude",
            "amplitude_calibration": "leave-one-target-out knockdown depth matched in the exact aggregate expected decoder",
        },
        "optimization": {
            "starts": N_STARTS,
            "outer_em_iterations": outer_iterations,
            "m_step_lbfgs_iterations": m_step_iterations,
            "mode_tolerance": MODE_TOLERANCE,
            "mode_max_newton_steps": MAX_NEWTON_STEPS,
            "mode_line_search": "per-cell Armijo backtracking",
            "precision": "float64",
        },
        "inputs": {
            "h1": {"path": relative(H1_PATH), "sha256": sha256(H1_PATH)},
            "strict_controls": {
                "path": relative(STRICT_CONTROLS),
                "sha256": sha256(STRICT_CONTROLS),
            },
        },
        "fit_artifacts": {
            name: {"path": relative(path), "sha256": sha256(path)}
            for name, path in {
                "independent_D": INDEPENDENT_D_FIT,
                "pure_factor": PURE_FIT,
                "full": FIT,
            }.items()
        },
        "outputs": {
            name: {"path": relative(path), "sha256": sha256(path)}
            for name, path in outputs.items()
        },
        "software": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ["anndata", "jax", "jaxopt", "numpy", "pandas"]
            },
        },
    }
    (REPORT / "model3_full_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def smoke_test(batches: int) -> dict[str, object]:
    if batches < 4:
        raise ValueError("the rank-50 smoke test needs at least four 16-cell batches")
    strict_guides = pd.read_csv(STRICT_CONTROLS)["guide_id"].astype(str).tolist()
    data = ad.read_h5ad(H1_PATH, backed="r")
    work_path = DERIVED / "model3_full_smoke_centered.f32"
    try:
        train_rows, selection_rows, _evaluation_rows, guides = split_control_rows(
            data.obs, strict_guides
        )
        rng = np.random.default_rng(17)
        smoke_rows = np.sort(
            rng.choice(train_rows, size=batches * BATCH_SIZE, replace=False)
        )
        smoke_selection = np.sort(
            rng.choice(selection_rows, size=BATCH_SIZE, replace=False)
        )
        initial, pca_loadings, total_variance, mean_rates = initialize_full_model(
            data, smoke_rows, work_path
        )
        _, initial_train, _ = run_e_step(
            data, smoke_rows, initial, collect_moments=False
        )
        _, initial_selection, _ = run_e_step(
            data, smoke_selection, initial, collect_moments=False
        )
        fitted, trace = fit_em(
            data,
            smoke_rows,
            initial,
            outer_iterations=1,
            m_step_iterations=10,
        )
        _, fitted_train, _ = run_e_step(data, smoke_rows, fitted, collect_moments=False)
        _, fitted_selection, _ = run_e_step(
            data, smoke_selection, fitted, collect_moments=False
        )
        _, loadings, d = population_parameters(fitted)
        result: dict[str, object] = {
            "status": "passed",
            "created_utc": datetime.now(UTC).isoformat(),
            "smoke_only_not_a_fit": True,
            "rank": int(loadings.shape[1]),
            "batch_size": BATCH_SIZE,
            "train_cells": len(smoke_rows),
            "selection_cells": len(smoke_selection),
            "training_guides": guides["training"],
            "selection_guides": guides["selection"],
            "evaluation_guides_untouched": guides["evaluation"],
            "initial_train_laplace_nll": initial_train["laplace_nll_mean"],
            "fitted_train_laplace_nll": fitted_train["laplace_nll_mean"],
            "initial_selection_laplace_nll": initial_selection["laplace_nll_mean"],
            "fitted_selection_laplace_nll": fitted_selection["laplace_nll_mean"],
            "fitted_mode_residual_q99": fitted_selection["mode_residual_q99"],
            "fitted_mode_residual_max": fitted_selection["mode_residual_max"],
            "initial_factor_variance_median": float(
                np.median(np.square(pca_loadings).sum(1))
            ),
            "initial_total_variance_median": float(np.median(total_variance)),
            "initial_positive_total_variance_fraction": float(
                np.mean(total_variance > 0)
            ),
            "initial_smallest_mean_rate": float(mean_rates.min()),
            "fitted_residual_variance_median": float(np.median(np.asarray(d))),
            "software": {
                name: version(name)
                for name in ["anndata", "jax", "jaxopt", "numpy", "pandas"]
            },
            "platform": platform.platform(),
            "input": {
                "path": relative(H1_PATH),
                "sha256": sha256(H1_PATH),
            },
        }
        REPORT.mkdir(parents=True, exist_ok=True)
        trace.to_csv(REPORT / "model3_full_smoke_trace.csv", index=False)
        (REPORT / "model3_full_smoke.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return result
    finally:
        data.file.close()
        work_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke-batches",
        type=int,
        help="run one smoke-only EM update on this many 16-cell batches",
    )
    mode.add_argument(
        "--fit",
        action="store_true",
        help="fit all four control-generative rungs and evaluate held-out guides",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse split-validated intermediate fits where present",
    )
    parser.add_argument("--outer-iterations", type=int, default=OUTER_ITERATIONS)
    parser.add_argument("--m-step-iterations", type=int, default=M_STEP_ITERATIONS)
    parser.add_argument("--random-matrix-draws", type=int, default=RANDOM_MATRIX_DRAWS)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.smoke_batches is not None:
        result = smoke_test(args.smoke_batches)
    else:
        result = run_full_pipeline(
            resume=args.resume,
            outer_iterations=args.outer_iterations,
            m_step_iterations=args.m_step_iterations,
            random_matrix_draws=args.random_matrix_draws,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(parse_args())
