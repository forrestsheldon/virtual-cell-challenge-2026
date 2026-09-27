from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.linear_response.control_dose_response import (
    nuisance_design,
    residualize,
    solve_quadratic_coefficients,
)


def test_residualize_removes_nuisance_columns() -> None:
    rng = np.random.default_rng(4)
    nuisance = np.column_stack([np.ones(100), rng.normal(size=(100, 3))])
    orthonormal = np.linalg.qr(nuisance, mode="reduced")[0]
    values = nuisance @ rng.normal(size=(4, 2)) + rng.normal(size=(100, 2))
    result = residualize(orthonormal, values)
    assert np.max(np.abs(orthonormal.T @ result)) < 1e-12


def test_quadratic_solver_recovers_known_coefficients() -> None:
    rng = np.random.default_rng(5)
    residual = rng.normal(size=(500, 2))
    squared = np.square(residual) - np.square(residual).mean(axis=0)
    beta_linear = np.array([[2.0, -1.0], [0.5, 3.0]])
    beta_squared = np.array([[-0.4, 0.7], [1.2, -0.8]])
    outcomes = np.empty((2, 500, 2))
    for target in range(2):
        outcomes[target] = (
            residual[:, target, None] * beta_linear[target]
            + squared[:, target, None] * beta_squared[target]
        )
    cross_linear = np.stack(
        [residual[:, target] @ outcomes[target] for target in range(2)]
    )
    cross_squared = np.stack(
        [squared[:, target] @ outcomes[target] for target in range(2)]
    )
    observed_linear, observed_squared = solve_quadratic_coefficients(
        residual, squared, cross_linear, cross_squared
    )
    assert np.allclose(observed_linear, beta_linear)
    assert np.allclose(observed_squared, beta_squared)


def test_nuisance_design_includes_requested_state_axes() -> None:
    import pandas as pd

    obs = pd.DataFrame(
        {
            "batch": ["a", "a", "b", "b"],
            "guide_id": ["x", "y", "x", "y"],
        }
    )
    scores = np.arange(12, dtype=float).reshape(4, 3)
    without = nuisance_design(obs, scores, include_state=False)
    with_state = nuisance_design(obs, scores, include_state=True)
    assert without.shape == (4, 3)
    assert with_state.shape == (4, 6)
