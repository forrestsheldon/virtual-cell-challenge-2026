"""Audit what the Model 3 traces do and do not establish about convergence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/linear-response-three-models"
MODE_TOLERANCE = 1e-4
M_STEP_TOLERANCE = 1e-4
OUTER_NLL_STABILITY_TOLERANCE = 0.1


def audit_trace(model: str, trace: pd.DataFrame) -> dict[str, object]:
    ordered = trace.sort_values("outer_iteration")
    nll = ordered["laplace_nll_mean"].to_numpy()
    outer_change = float(nll[-1] - nll[-2]) if len(nll) > 1 else np.nan
    return {
        "model": model,
        "outer_iterations": len(ordered),
        "all_cell_modes_converged": bool(
            ordered["mode_residual_max"].max() <= MODE_TOLERANCE
        ),
        "maximum_mode_residual": ordered["mode_residual_max"].max(),
        "every_m_step_decreased_fixed_posterior_objective": bool(
            (ordered["m_step_after"] <= ordered["m_step_before"] + 1e-4).all()
        ),
        "final_m_step_gradient_error": ordered["m_step_gradient_error"].iloc[-1],
        "final_m_step_met_declared_tolerance": bool(
            ordered["m_step_gradient_error"].iloc[-1] <= M_STEP_TOLERANCE
        ),
        "last_outer_laplace_nll_change": outer_change,
        "outer_laplace_nll_stable_within_0.1": bool(
            np.isfinite(outer_change)
            and abs(outer_change) <= OUTER_NLL_STABILITY_TOLERANCE
        ),
        "outer_laplace_nll_nonincreasing": bool(np.all(np.diff(nll) <= 0)),
        "population_normalized": bool(
            np.max(np.abs(ordered["sum_population_mean_rates"] - 1)) < 1e-8
        ),
        "final_parameter_update_re_evaluated_on_training_rows": False,
        "parameter_convergence_demonstrated": bool(
            ordered["m_step_gradient_error"].iloc[-1] <= M_STEP_TOLERANCE
            and np.isfinite(outer_change)
            and abs(outer_change) <= OUTER_NLL_STABILITY_TOLERANCE
        ),
    }


def main() -> None:
    records = []
    independent_path = REPORT / "model3_independent_d_fit_trace.csv"
    if independent_path.exists():
        independent = pd.read_csv(independent_path)
        records.append(
            audit_trace("independent_poisson_lognormal_D", independent)
        )
    full_path = REPORT / "model3_full_fit_trace.csv"
    if full_path.exists():
        full = pd.read_csv(full_path)
        for start, trace in full.groupby("start"):
            records.append(audit_trace(f"full_LL_T_plus_D_start_{start}", trace))
    records.append(
        {
            "model": "pure_factor_LL_T",
            "outer_iterations": np.nan,
            "all_cell_modes_converged": np.nan,
            "maximum_mode_residual": np.nan,
            "every_m_step_decreased_fixed_posterior_objective": np.nan,
            "final_m_step_gradient_error": np.nan,
            "final_m_step_met_declared_tolerance": False,
            "last_outer_laplace_nll_change": np.nan,
            "outer_laplace_nll_stable_within_0.1": False,
            "outer_laplace_nll_nonincreasing": np.nan,
            "population_normalized": True,
            "final_parameter_update_re_evaluated_on_training_rows": False,
            "parameter_convergence_demonstrated": False,
            "note": "fixed one-epoch Adam fit; mode convergence is checked separately, parameter convergence is not claimed",
        }
    )
    table = pd.DataFrame(records)
    path = REPORT / "model3_convergence_audit.csv"
    table.to_csv(path, index=False)
    summary = {
        "kind": "optimization convergence audit, separate from predictive validation",
        "mode_tolerance": MODE_TOLERANCE,
        "m_step_gradient_tolerance": M_STEP_TOLERANCE,
        "outer_nll_absolute_stability_tolerance": OUTER_NLL_STABILITY_TOLERANCE,
        "all_fitted_models_demonstrate_parameter_convergence": bool(
            table["parameter_convergence_demonstrated"].all()
        ),
        "interpretation": "A false value does not invalidate the finite fit, but it requires continuation or sensitivity analysis before calling the optimizer converged.",
        "output": str(path.relative_to(ROOT)),
    }
    (REPORT / "model3_convergence_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
