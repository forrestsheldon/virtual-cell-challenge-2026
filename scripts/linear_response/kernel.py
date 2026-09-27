"""Small shared kernels for the H1 linear-response diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import brentq

CELL_TARGET_SUM = 10_000.0
BULK_TARGET_SUM = 50_000.0
CELLS_PER_TARGET = 400
CHUNK_SIZE = 1_024


def _dense_float(raw: sparse.spmatrix | np.ndarray) -> np.ndarray:
    dense = raw.toarray() if sparse.issparse(raw) else np.asarray(raw)
    return dense.astype(np.float64, copy=False)


def log1cp10k(raw: sparse.spmatrix | np.ndarray) -> np.ndarray:
    """Per-cell total normalization to 10,000 followed by log1p."""
    dense = _dense_float(raw)
    totals = dense.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("every source cell must have a positive total")
    return np.log1p(dense * (CELL_TARGET_SUM / totals)[:, None])


def log_pseudobulk(
    counts: sparse.spmatrix | np.ndarray, scale: float = BULK_TARGET_SUM
) -> np.ndarray:
    """Normalize pooled counts along the last axis and apply log1p."""
    if sparse.issparse(counts):
        counts = np.asarray(counts.sum(axis=0)).ravel()
    values = np.asarray(counts, dtype=np.float64)
    totals = values.sum(axis=-1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("every pseudobulk must have a positive total")
    return np.log1p(scale * values / totals)


def sample_balanced_controls(
    obs: pd.DataFrame,
    guide_order: Sequence[str],
    rng: np.random.Generator,
    *,
    cells_per_target: int = CELLS_PER_TARGET,
    control_label: str = "non-targeting",
) -> np.ndarray:
    """Sample controls as evenly as possible across the requested guides."""
    base, remainder = divmod(cells_per_target, len(guide_order))
    extras = set(rng.choice(len(guide_order), remainder, replace=False))
    guides = obs["guide_id"].astype(str).to_numpy()
    labels = obs["target_gene"].astype(str).to_numpy()
    rows: list[int] = []
    for guide_index, guide in enumerate(guide_order):
        available = np.flatnonzero((guides == guide) & (labels == control_label))
        take = base + (guide_index in extras)
        if len(available) < take:
            raise ValueError(f"guide {guide!r} has {len(available)} cells; need {take}")
        rows.extend(rng.choice(available, take, replace=False).tolist())
    rng.shuffle(rows)
    return np.asarray(rows, dtype=np.int64)


def split_control_halves(
    obs: pd.DataFrame, rows: np.ndarray, guide_order: Sequence[str]
) -> np.ndarray:
    """Deterministically split every guide-by-batch control stratum in half."""
    controls = obs.iloc[rows]
    batches = sorted(controls["batch"].astype(str).unique())
    batch_index = {batch: index for index, batch in enumerate(batches)}
    halves = np.full(len(rows), -1, dtype=np.int8)
    for guide_index, guide in enumerate(guide_order):
        for batch in batches:
            positions = np.flatnonzero(
                controls["guide_id"].astype(str).eq(guide).to_numpy()
                & controls["batch"].astype(str).eq(batch).to_numpy()
            )
            if len(positions) == 0:
                continue
            rng = np.random.default_rng(
                np.random.SeedSequence([0, 7, guide_index, batch_index[batch]])
            )
            shuffled = rng.permutation(positions)
            halves[shuffled[::2]] = 0
            halves[shuffled[1::2]] = 1
    if np.any(halves < 0):
        raise AssertionError("control split left cells unassigned")
    return halves


def _new_moments(n_genes: int, n_targets: int) -> dict[str, np.ndarray | int]:
    return {
        "n": 0,
        "sum": np.zeros(n_genes, dtype=np.float64),
        "cross": np.zeros((n_genes, n_targets), dtype=np.float64),
    }


def _update_moments(
    moments: dict[str, np.ndarray | int],
    values: np.ndarray,
    target_indices: Sequence[int],
) -> None:
    moments["n"] += len(values)
    moments["sum"] += values.sum(axis=0)
    moments["cross"] += values.T @ values[:, target_indices]


def _finish_moments(
    moments: dict[str, np.ndarray | int], target_indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    n = int(moments["n"])
    if n <= 1:
        raise ValueError("at least two cells are required")
    total = np.asarray(moments["sum"])
    mean = total / n
    centered_cross = np.asarray(moments["cross"]) - np.outer(
        total, total[np.asarray(target_indices)]
    ) / n
    return centered_cross / (n - 1), mean


def fit_covariance_columns(
    data: ad.AnnData,
    rows: np.ndarray,
    target_indices: Sequence[int],
    halves: np.ndarray | None = None,
    *,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Fit empirical covariance columns, optionally on two fixed cell halves."""
    moments = [_new_moments(data.n_vars, len(target_indices))]
    if halves is not None:
        moments.extend(_new_moments(data.n_vars, len(target_indices)) for _ in range(2))
    for start in range(0, len(rows), chunk_size):
        chunk_rows = rows[start : start + chunk_size]
        values = log1cp10k(data.X[chunk_rows].tocsr())
        _update_moments(moments[0], values, target_indices)
        if halves is not None:
            chunk_halves = halves[start : start + chunk_size]
            _update_moments(moments[1], values[chunk_halves == 0], target_indices)
            _update_moments(moments[2], values[chunk_halves == 1], target_indices)
    full, mean = _finish_moments(moments[0], target_indices)
    if halves is None:
        return full, None, None, mean
    half0, _ = _finish_moments(moments[1], target_indices)
    half1, _ = _finish_moments(moments[2], target_indices)
    return full, half0, half1, mean


def collect_raw_pseudobulks(
    data: ad.AnnData,
    target_order: Sequence[str],
    strict_guides: Sequence[str],
    *,
    control_label: str = "non-targeting",
    chunk_size: int = CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool strict controls and every labelled perturbation in raw-count space."""
    target_lookup = {target: index for index, target in enumerate(target_order)}
    target_sums = np.zeros((len(target_order), data.n_vars), dtype=np.float64)
    control_sum = np.zeros(data.n_vars, dtype=np.float64)
    labels = data.obs["target_gene"].astype(str).to_numpy()
    guides = data.obs["guide_id"].astype(str).to_numpy()
    strict = set(strict_guides)
    for start in range(0, data.n_obs, chunk_size):
        stop = min(start + chunk_size, data.n_obs)
        raw = data.X[start:stop].tocsr().astype(np.float64)
        chunk_labels = labels[start:stop]
        chunk_guides = guides[start:stop]
        control_mask = (chunk_labels == control_label) & np.isin(
            chunk_guides, list(strict)
        )
        if control_mask.any():
            control_sum += np.asarray(raw[control_mask].sum(axis=0)).ravel()
        for target in np.unique(chunk_labels[chunk_labels != control_label]):
            target_sums[target_lookup[target]] += np.asarray(
                raw[chunk_labels == target].sum(axis=0)
            ).ravel()
    return control_sum, target_sums


def decoder_probabilities(
    raw_controls: sparse.spmatrix | np.ndarray,
    direction: np.ndarray,
    amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source totals and exact multinomial probabilities after a shift."""
    raw = _dense_float(raw_controls)
    totals = raw.sum(axis=1)
    probabilities = shifted_probabilities(log1cp10k(raw), direction, amplitude)
    return totals, probabilities


def shifted_probabilities(
    log_controls: np.ndarray,
    direction: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    """Decode probabilities from an already normalized log1p control matrix."""
    shifted = np.asarray(log_controls) + float(amplitude) * np.asarray(
        direction, dtype=np.float64
    )
    if not np.isfinite(shifted).all():
        raise ValueError("shifted expression is not finite")
    weights = np.clip(np.expm1(shifted), 0.0, None)
    weight_totals = weights.sum(axis=1)
    if not np.isfinite(weights).all() or np.any(weight_totals <= 0):
        raise ValueError("decoder produced invalid or zero total weights")
    return weights / weight_totals[:, None]


def expected_cpm_from_log(
    log_controls: np.ndarray,
    direction: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    """Expected mean per-cell CPM from cached log1p-normalized controls."""
    return 1_000_000.0 * shifted_probabilities(
        log_controls, direction, amplitude
    ).mean(axis=0)


def expected_restricted_cpm(
    raw_panel: sparse.spmatrix | np.ndarray,
    full_totals: np.ndarray,
    direction: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    """Exact expected CPM on a panel when the response is zero off-panel."""
    raw = _dense_float(raw_panel)
    totals = np.asarray(full_totals, dtype=np.float64)
    baseline_weights = raw * (CELL_TARGET_SUM / totals)[:, None]
    outside_weights = CELL_TARGET_SUM - baseline_weights.sum(axis=1)
    shifted_weights = np.clip(
        np.expm1(np.log1p(baseline_weights) + amplitude * np.asarray(direction)),
        0.0,
        None,
    )
    denominators = outside_weights + shifted_weights.sum(axis=1)
    return 1_000_000.0 * np.mean(shifted_weights / denominators[:, None], axis=0)


def expected_decoded_sum(
    raw_controls: sparse.spmatrix | np.ndarray,
    direction: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    """Exact expected pooled counts of the multinomial decoder."""
    totals, probabilities = decoder_probabilities(raw_controls, direction, amplitude)
    return np.einsum("i,ij->j", totals, probabilities)


def expected_decoded_cpm_mean(
    raw_controls: sparse.spmatrix | np.ndarray,
    direction: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    """Expected mean per-cell CPM after applying a deterministic log-space shift."""
    _, probabilities = decoder_probabilities(raw_controls, direction, amplitude)
    return 1_000_000.0 * probabilities.mean(axis=0)


def expected_decoded_lfc(
    raw_controls: sparse.spmatrix | np.ndarray,
    direction: np.ndarray,
    amplitude: float,
    *,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """Expected DE log2 fold change against the identical unshifted source cells."""
    baseline = expected_decoded_cpm_mean(raw_controls, direction, 0.0)
    shifted = expected_decoded_cpm_mean(raw_controls, direction, amplitude)
    return np.log2((shifted + epsilon) / (baseline + epsilon))


def decoder_lfc_tangent(
    raw_controls: sparse.spmatrix | np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """One-sided derivative of expected per-cell-CPM log2FC at zero."""
    return decoder_lfc_tangents(raw_controls, np.asarray(direction)[None, :])[0]


def decoder_lfc_tangents(
    raw_controls: sparse.spmatrix | np.ndarray,
    directions: np.ndarray,
    *,
    full_totals: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized expected DE-log2FC tangents for several response directions.

    ``full_totals`` permits ``raw_controls`` to contain only the genes on which
    every direction is nonzero while retaining the exact full-cell denominator.
    """
    raw = sparse.csr_matrix(raw_controls, dtype=np.float64)
    totals = (
        np.asarray(raw.sum(axis=1)).ravel()
        if full_totals is None
        else np.asarray(full_totals, dtype=np.float64)
    )
    directions = np.asarray(directions, dtype=np.float64)
    presence = raw.copy()
    presence.data.fill(1)
    fractions = raw.multiply((1 / totals)[:, None])
    mean_fraction = np.asarray(fractions.mean(axis=0)).ravel()
    positive = np.maximum(directions, 0)
    negative = np.minimum(directions, 0)
    d_weight_total = (
        positive.sum(axis=1)[None, :]
        + np.asarray(presence @ negative.T)
        + CELL_TARGET_SUM / totals[:, None] * np.asarray(raw @ directions.T)
    )
    first = (
        positive / CELL_TARGET_SUM
        + negative
        * np.asarray(presence.mean(axis=0)).ravel()[None, :]
        / CELL_TARGET_SUM
        + directions * mean_fraction[None, :]
    )
    second = np.asarray(
        raw.T
        @ (
            d_weight_total
            / (totals[:, None] * CELL_TARGET_SUM * raw.shape[0])
        )
    ).T
    derivative = first - second
    return np.divide(
        derivative,
        mean_fraction[None, :] * np.log(2),
        out=np.zeros_like(derivative),
        where=mean_fraction[None, :] > 0,
    )


def expected_log_rate_cpm(
    raw_panel: sparse.spmatrix | np.ndarray,
    full_totals: np.ndarray,
    direction: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    """Expected CPM after shifting latent log rates on a restricted panel."""
    raw = sparse.csr_matrix(raw_panel, dtype=np.float64)
    totals = np.asarray(full_totals, dtype=np.float64)
    fractions = raw.multiply((1 / totals)[:, None])
    outside = 1 - np.asarray(fractions.sum(axis=1)).ravel()
    factors = np.exp(np.clip(amplitude * np.asarray(direction), -80, 80))
    shifted = fractions.multiply(factors)
    shifted_total = outside + np.asarray(shifted.sum(axis=1)).ravel()
    return 1_000_000 * np.asarray(
        shifted.multiply((1 / shifted_total)[:, None]).mean(axis=0)
    ).ravel()


def expected_log_rate_cpms(
    raw_panel: sparse.spmatrix | np.ndarray,
    full_totals: np.ndarray,
    directions: np.ndarray,
    amplitudes: np.ndarray,
    *,
    batch_size: int = 32,
) -> np.ndarray:
    """Vectorized expected CPMs for several latent log-rate shifts."""
    raw = sparse.csr_matrix(raw_panel, dtype=np.float64)
    totals = np.asarray(full_totals, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    fractions = raw.multiply((1 / totals)[:, None])
    outside = 1 - np.asarray(fractions.sum(axis=1)).ravel()
    result = np.empty_like(directions)
    for start in range(0, len(directions), batch_size):
        stop = min(start + batch_size, len(directions))
        factors = np.exp(
            np.clip(amplitudes[start:stop, None] * directions[start:stop], -80, 80)
        )
        shifted_totals = outside[:, None] + np.asarray(fractions @ factors.T)
        result[start:stop] = (
            1_000_000
            * np.asarray(fractions.T @ (1 / shifted_totals)).T
            * factors
            / len(totals)
        )
    return result


def log_rate_lfc_tangents(
    raw_panel: sparse.spmatrix | np.ndarray,
    full_totals: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    """DE-log2FC derivatives for several latent log-rate directions."""
    raw = sparse.csr_matrix(raw_panel, dtype=np.float64)
    totals = np.asarray(full_totals, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    fractions = raw.multiply((1 / totals)[:, None])
    mean_fraction = np.asarray(fractions.mean(axis=0)).ravel()
    cell_shift = np.asarray(fractions @ directions.T)
    normalization = np.asarray(fractions.T @ cell_shift).T / len(totals)
    derivative = mean_fraction[None, :] * directions - normalization
    return np.divide(
        derivative,
        mean_fraction[None, :] * np.log(2),
        out=np.zeros_like(derivative),
        where=mean_fraction[None, :] > 0,
    )


def decode_multinomial(
    raw_controls: sparse.spmatrix | np.ndarray,
    direction: np.ndarray,
    amplitude: float,
    rng: np.random.Generator,
) -> sparse.csr_matrix:
    """Draw one sparse decoded count matrix while preserving source totals."""
    totals, probabilities = decoder_probabilities(raw_controls, direction, amplitude)
    generated = np.vstack(
        [
            rng.multinomial(int(total), probability)
            for total, probability in zip(totals, probabilities, strict=True)
        ]
    ).astype(np.int32, copy=False)
    matrix = sparse.csr_matrix(generated)
    matrix.eliminate_zeros()
    if not np.array_equal(np.asarray(matrix.sum(axis=1)).ravel(), totals):
        raise AssertionError("multinomial decoder did not preserve source totals")
    return matrix


def decode_largest_remainder(
    raw_controls: sparse.spmatrix | np.ndarray,
    direction: np.ndarray,
    amplitude: float,
) -> sparse.csr_matrix:
    """Deterministically decode integer counts while preserving every cell total."""
    raw = sparse.csr_matrix(raw_controls)
    if amplitude == 0:
        result = raw.astype(np.int32)
        result.eliminate_zeros()
        return result
    totals, probabilities = decoder_probabilities(raw, direction, amplitude)
    expected = probabilities * totals[:, None]
    generated = np.floor(expected).astype(np.int32)
    genes = np.arange(generated.shape[1])
    for row, total in enumerate(totals.astype(np.int64)):
        remainder = int(total - generated[row].sum())
        if remainder:
            fractions = expected[row] - generated[row]
            selected = np.lexsort((genes, -fractions))[:remainder]
            generated[row, selected] += 1
    result = sparse.csr_matrix(generated)
    result.eliminate_zeros()
    if not np.array_equal(np.asarray(result.sum(axis=1)).ravel(), totals):
        raise AssertionError("deterministic decoder did not preserve source totals")
    return result


def expected_lfc_decoded_sum(
    raw_controls: sparse.spmatrix | np.ndarray, lfc: np.ndarray
) -> np.ndarray:
    """Expected pooled counts after a multiplicative LFC perturbation."""
    raw = sparse.csr_matrix(raw_controls, dtype=np.float64)
    factors = np.exp2(np.asarray(lfc, dtype=np.float64))
    pooled = np.zeros(raw.shape[1], dtype=np.float64)
    for row in range(raw.shape[0]):
        start, stop = raw.indptr[row : row + 2]
        indices = raw.indices[start:stop]
        counts = raw.data[start:stop]
        total = counts.sum()
        weights = counts * factors[indices]
        pooled[indices] += total * weights / weights.sum()
    return pooled


def decode_lfc_largest_remainder(
    raw_controls: sparse.spmatrix | np.ndarray, lfc: np.ndarray
) -> sparse.csr_matrix:
    """Apply an LFC vector without adding sampling noise or new nonzero entries."""
    raw = sparse.csr_matrix(raw_controls)
    if not np.any(lfc):
        result = raw.astype(np.int32)
        result.eliminate_zeros()
        return result
    factors = np.exp2(np.asarray(lfc, dtype=np.float64))
    data: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    indptr = [0]
    for row in range(raw.shape[0]):
        start, stop = raw.indptr[row : row + 2]
        row_indices = raw.indices[start:stop]
        counts = raw.data[start:stop]
        total = int(counts.sum())
        weights = counts * factors[row_indices]
        expected = total * weights / weights.sum()
        generated = np.floor(expected).astype(np.int32)
        remainder = total - int(generated.sum())
        if remainder:
            fractions = expected - generated
            selected = np.lexsort((row_indices, -fractions))[:remainder]
            generated[selected] += 1
        keep = generated > 0
        data.append(generated[keep])
        indices.append(row_indices[keep])
        indptr.append(indptr[-1] + int(keep.sum()))
    result = sparse.csr_matrix(
        (np.concatenate(data), np.concatenate(indices), np.asarray(indptr)),
        shape=raw.shape,
    )
    if not np.array_equal(
        np.asarray(result.sum(axis=1)).ravel(), np.asarray(raw.sum(axis=1)).ravel()
    ):
        raise AssertionError("LFC decoder did not preserve source totals")
    return result


def decode_lfc_dependent_round(
    raw_controls: sparse.spmatrix | np.ndarray,
    lfc: np.ndarray,
    rng: np.random.Generator,
) -> sparse.csr_matrix:
    """Unbiasedly integerize an LFC shift while preserving every cell total."""
    raw = sparse.csr_matrix(raw_controls)
    if not np.any(lfc):
        result = raw.astype(np.int32)
        result.eliminate_zeros()
        return result
    factors = np.exp2(np.asarray(lfc, dtype=np.float64))
    data: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    indptr = [0]
    for row in range(raw.shape[0]):
        start, stop = raw.indptr[row : row + 2]
        row_indices = raw.indices[start:stop]
        counts = raw.data[start:stop]
        total = int(counts.sum())
        weights = counts * factors[row_indices]
        expected = total * weights / weights.sum()
        generated = np.floor(expected).astype(np.int32)
        remainder = total - int(generated.sum())
        if remainder:
            cumulative = np.cumsum(expected - generated)
            cumulative[-1] = remainder
            selected = np.searchsorted(
                cumulative,
                rng.random() + np.arange(remainder),
                side="right",
            )
            generated[selected] += 1
        keep = generated > 0
        data.append(generated[keep])
        indices.append(row_indices[keep])
        indptr.append(indptr[-1] + int(keep.sum()))
    result = sparse.csr_matrix(
        (np.concatenate(data), np.concatenate(indices), np.asarray(indptr)),
        shape=raw.shape,
    )
    if not np.array_equal(
        np.asarray(result.sum(axis=1)).ravel(), np.asarray(raw.sum(axis=1)).ravel()
    ):
        raise AssertionError("dependent rounding did not preserve source totals")
    return result


def normalize_response_columns(
    covariance_columns: np.ndarray,
    target_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert covariance columns into knockdown directions with target value -1."""
    columns = np.asarray(covariance_columns, dtype=np.float64)
    targets = np.asarray(target_indices, dtype=int)
    diagonal = columns[targets, np.arange(len(targets))]
    if np.any(diagonal <= 0):
        raise ValueError("response columns require positive target variances")
    response = -columns / diagonal
    response[targets, np.arange(len(targets))] = -1.0
    return response, diagonal


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the lower weighted median of finite values with positive weights."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return 0.0
    order = np.argsort(values[keep], kind="stable")
    ordered_values = values[keep][order]
    ordered_weights = weights[keep][order]
    index = np.searchsorted(
        np.cumsum(ordered_weights), ordered_weights.sum() / 2, side="left"
    )
    return float(ordered_values[index])


def nonnegative_l1_scale(
    direction: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Minimize absolute error for ``scale * direction`` on a selected gene set."""
    direction = np.asarray(direction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool) & (direction != 0)
    scale = weighted_median(
        truth[selected] / direction[selected], np.abs(direction[selected])
    )
    return max(0.0, scale)


def intended_target_shift(
    control_fraction: float,
    knockdown_depth: float,
    *,
    bulk_target_sum: float = BULK_TARGET_SUM,
) -> float:
    """Translate a log-fraction knockdown into the scored pseudobulk coordinate."""
    if not (0 < control_fraction <= 1) or not np.isfinite(knockdown_depth):
        raise ValueError("control fraction and knockdown depth must be finite and valid")
    after = control_fraction * np.exp(knockdown_depth)
    return float(
        np.log1p(bulk_target_sum * after)
        - np.log1p(bulk_target_sum * control_fraction)
    )


def solve_output_matched_amplitude(
    raw_controls: sparse.spmatrix | np.ndarray,
    direction: np.ndarray,
    target_index: int,
    target_shift: float,
    *,
    bulk_target_sum: float = BULK_TARGET_SUM,
) -> tuple[float, float]:
    """Find a non-positive amplitude matching the exact expected target shift."""
    raw_sum = np.asarray(raw_controls.sum(axis=0)).ravel() if sparse.issparse(raw_controls) else np.asarray(raw_controls).sum(axis=0)
    baseline = log_pseudobulk(raw_sum, bulk_target_sum)[target_index]

    def residual(amplitude: float) -> float:
        expected = expected_decoded_sum(raw_controls, direction, amplitude)
        realized = log_pseudobulk(expected, bulk_target_sum)[target_index] - baseline
        return float(realized - target_shift)

    at_zero = residual(0.0)
    if abs(at_zero) < 1e-12:
        return 0.0, float(target_shift)
    lower = -1.0
    at_lower = residual(lower)
    while np.signbit(at_lower) == np.signbit(at_zero) and lower > -4096:
        lower *= 2
        at_lower = residual(lower)
    if np.signbit(at_lower) == np.signbit(at_zero):
        raise ValueError("could not bracket a non-positive output-matched amplitude")
    amplitude = float(brentq(residual, lower, 0.0, xtol=1e-10, rtol=1e-10))
    realized = target_shift + residual(amplitude)
    return amplitude, float(realized)


def knockdown_depths(
    target_order: Sequence[str],
    target_indices: Sequence[int],
    control_counts: np.ndarray,
    target_counts: np.ndarray,
) -> pd.DataFrame:
    """Compute observed and leave-one-target-out log-fraction knockdown depths."""
    control = np.asarray(control_counts, dtype=np.float64)
    perturbed = np.asarray(target_counts, dtype=np.float64)
    indices = np.asarray(target_indices, dtype=int)
    control_fraction = control[indices] / control.sum()
    perturbed_fraction = perturbed[np.arange(len(indices)), indices] / perturbed.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        observed = np.log(perturbed_fraction / control_fraction)
    eligible = (
        np.isfinite(observed)
        & (control_fraction > 0)
        & (perturbed_fraction > 0)
        & (observed < 0)
    )
    loo = np.full(len(indices), np.nan)
    for index in range(len(indices)):
        donors = eligible.copy()
        donors[index] = False
        if donors.any():
            loo[index] = np.median(observed[donors])
    return pd.DataFrame(
        {
            "target_gene": list(target_order),
            "target_index": np.arange(len(indices)),
            "control_fraction": control_fraction,
            "perturbed_fraction": perturbed_fraction,
            "observed_knockdown_depth": observed,
            "eligible_knockdown_donor": eligible,
            "leave_one_out_knockdown_depth": loo,
        }
    )


def pca_covariance_columns(
    components: np.ndarray,
    singular_values: np.ndarray,
    n_cells: int,
    target_indices: Sequence[int],
) -> np.ndarray:
    """Construct selected columns of V diag(lambda) V.T without forming it."""
    vectors = np.asarray(components, dtype=np.float64)
    eigenvalues = np.asarray(singular_values, dtype=np.float64) ** 2 / (n_cells - 1)
    return vectors.T @ (eigenvalues[:, None] * vectors[:, np.asarray(target_indices)])


def write_prediction(
    counts_by_target: Mapping[str, sparse.spmatrix | np.ndarray],
    genes: Sequence[str],
    path: Path,
) -> None:
    """Write the canonical H1 prediction seam with no explicitly stored zeros."""
    targets = list(counts_by_target)
    blocks: list[sparse.csr_matrix] = []
    labels: list[str] = []
    for target in targets:
        block = sparse.csr_matrix(counts_by_target[target])
        block.eliminate_zeros()
        if block.shape != (CELLS_PER_TARGET, len(genes)):
            raise ValueError(f"{target}: expected {(CELLS_PER_TARGET, len(genes))}, got {block.shape}")
        if (block.data == 0).any():
            raise AssertionError("prediction contains explicitly stored zeros")
        blocks.append(block)
        labels.extend([target] * CELLS_PER_TARGET)
    matrix = sparse.vstack(blocks, format="csr")
    obs = pd.DataFrame(
        {"target_gene": labels}, index=pd.Index(np.arange(len(labels)).astype(str))
    )
    var = pd.DataFrame(index=pd.Index(np.asarray(genes, dtype=str)))
    output = ad.AnnData(matrix, obs=obs, var=var)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.write_h5ad(path)
