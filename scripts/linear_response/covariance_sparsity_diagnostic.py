"""Describe empirical off-target coefficients before choosing a sparsity prior."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from scripts.linear_response.ladder_sparse import (
    heldout_error,
    regression_coefficients,
)

ROOT = Path(__file__).resolve().parents[2]
EMPIRICAL = ROOT / "data/derived/linear_response/ladder/empirical.npz"
REPORT = ROOT / "reports/linear-response-ladder"
TAUS = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
QUANTILES = np.asarray([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coefficients(
    covariance: np.ndarray,
    variance: np.ndarray,
    targets: np.ndarray,
    cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    target_variance = variance[targets]
    beta = covariance / target_variance
    residual = np.maximum(
        variance[:, None] - np.square(covariance) / target_variance, 0
    )
    se = np.sqrt(residual / ((cells - 2) * target_variance[None, :]))
    z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    return beta, z


def histogram_rows(name: str, values: np.ndarray, *, signed: bool) -> list[dict]:
    if signed:
        limit = np.quantile(np.abs(values), 0.999)
        transformed = np.clip(values, -limit, limit)
        edges = np.linspace(-limit, limit, 101)
        axis = "linear_winsorized_0.1pct"
    else:
        positive = values[values > 0]
        lower, upper = np.quantile(positive, [0.001, 0.999])
        transformed = np.log10(np.clip(positive, lower, upper))
        edges = np.linspace(np.log10(lower), np.log10(upper), 101)
        axis = "log10_winsorized_0.1pct"
    counts, edges = np.histogram(transformed, bins=edges)
    return [
        {
            "measure": name,
            "axis": axis,
            "left": left,
            "right": right,
            "count": count,
        }
        for left, right, count in zip(edges[:-1], edges[1:], counts, strict=True)
    ]


def main() -> None:
    with np.load(EMPIRICAL, allow_pickle=False) as saved:
        covariance = saved["covariance"].astype(np.float64)
        variance = saved["variance"].astype(np.float64)
        targets = saved["target_fit_index"].astype(int)
        target_names = saved["target_gene"].astype(str)
        output_count = len(saved["output_gene"])
        cells = int(saved["strict_cells"])
        half_covariance = [
            saved["half0_covariance"].astype(np.float64),
            saved["half1_covariance"].astype(np.float64),
        ]
        half_variance = [
            saved["half0_variance"].astype(np.float64),
            saved["half1_variance"].astype(np.float64),
        ]
        half_cells = [int(saved["half0_cells"]), int(saved["half1_cells"])]

    beta, z = coefficients(covariance, variance, targets, cells)
    half = [
        coefficients(half_covariance[index], half_variance[index], targets, half_cells[index])
        for index in (0, 1)
    ]
    mask = np.ones((output_count, len(targets)), dtype=bool)
    for column, target in enumerate(targets):
        if target < output_count:
            mask[target, column] = False

    values = {
        "C_gt": covariance[:output_count][mask],
        "abs_C_gt": np.abs(covariance[:output_count][mask]),
        "beta_Cgt_over_Ctt": beta[:output_count][mask],
        "abs_beta": np.abs(beta[:output_count][mask]),
        "abs_z": np.abs(z[:output_count][mask]),
    }
    quantiles = pd.DataFrame(
        [
            {"measure": name, "quantile": quantile, "value": value}
            for name, array in values.items()
            for quantile, value in zip(
                QUANTILES, np.quantile(array, QUANTILES), strict=True
            )
        ]
    )

    threshold_rows = []
    for tau in TAUS:
        selected = (np.abs(z[:output_count]) > tau) & mask
        half_selected = [
            (np.abs(result[1][:output_count]) > tau) & mask for result in half
        ]
        union = half_selected[0] | half_selected[1]
        threshold_rows.append(
            {
                "tau": tau,
                "retained_fraction": selected.sum() / mask.sum(),
                "independent_normal_retained_fraction": 2 * norm.sf(tau),
                "split_half_sign_agreement": np.mean(
                    np.sign(half[0][0][:output_count][selected])
                    == np.sign(half[1][0][:output_count][selected])
                ),
                "split_half_support_jaccard": (
                    (half_selected[0] & half_selected[1]).sum() / union.sum()
                ),
                "heldout_control_error_ratio": np.mean(
                    [
                        heldout_error(
                            regression_coefficients(
                                half_covariance[fit],
                                half_variance[fit],
                                targets,
                                half_cells[fit],
                                tau,
                            ),
                            half_covariance[1 - fit],
                            half_variance[1 - fit],
                            targets,
                        )
                        for fit in (0, 1)
                    ]
                ),
            }
        )
    thresholds = pd.DataFrame(threshold_rows)

    target_summary = pd.DataFrame(
        {
            "target_gene": target_names,
            "C_tt": covariance[targets, np.arange(len(targets))],
            "median_abs_C_gt": [
                np.median(np.abs(covariance[:output_count, column][mask[:, column]]))
                for column in range(len(targets))
            ],
            "median_abs_beta": [
                np.median(np.abs(beta[:output_count, column][mask[:, column]]))
                for column in range(len(targets))
            ],
            "fraction_abs_z_gt_2": [
                np.mean(np.abs(z[:output_count, column][mask[:, column]]) > 2)
                for column in range(len(targets))
            ],
        }
    )
    histograms = pd.DataFrame(
        [
            *histogram_rows("C_gt", values["C_gt"], signed=True),
            *histogram_rows(
                "beta_Cgt_over_Ctt", values["beta_Cgt_over_Ctt"], signed=True
            ),
            *histogram_rows("abs_beta", values["abs_beta"], signed=False),
            *histogram_rows("abs_z", values["abs_z"], signed=False),
        ]
    )

    outputs = {
        "quantiles": REPORT / "covariance_coefficient_quantiles.csv",
        "thresholds": REPORT / "covariance_sparsity_thresholds.csv",
        "targets": REPORT / "covariance_target_summary.csv",
        "histograms": REPORT / "covariance_coefficient_histograms.csv",
    }
    quantiles.to_csv(outputs["quantiles"], index=False)
    thresholds.to_csv(outputs["thresholds"], index=False)
    target_summary.to_csv(outputs["targets"], index=False)
    histograms.to_csv(outputs["histograms"], index=False)
    manifest = {
        "kind": "pre-prior empirical off-target covariance diagnostic",
        "entries": int(mask.sum()),
        "target_entries_excluded": True,
        "analytic_null": "independent simple regressions: |z| exceeds tau with probability 2*Normal.sf(tau)",
        "input": {str(EMPIRICAL.relative_to(ROOT)): sha256(EMPIRICAL)},
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path) for path in outputs.values()
        },
        "software": {
            "python": platform.python_version(),
            **{name: version(name) for name in ["numpy", "pandas", "scipy"]},
        },
    }
    (REPORT / "covariance_sparsity_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
