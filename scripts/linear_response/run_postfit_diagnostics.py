"""Run the queued raw-count ablation and perturbation-level H1 figure."""

from __future__ import annotations

from argparse import Namespace

from scripts.evaluation.profile_h1 import evaluate
from scripts.linear_response.audit_model3_convergence import main as audit_convergence
from scripts.linear_response.model1_raw_counts import ARTIFACT as RAW_ARTIFACT
from scripts.linear_response.model1_raw_counts import DERIVED, REPORT, ROOT
from scripts.linear_response.model1_raw_counts import main as run_raw_ablation
from scripts.linear_response.plot_perturbation_performance import OUTPUT
from scripts.linear_response.plot_perturbation_performance import main as plot


def evaluate_profile(artifact, output) -> None:
    evaluate(
        Namespace(
            artifact=artifact,
            output=output,
            h1=ROOT / "data/external/vcc2025_h1/adata_Training.h5ad",
            reference_cells=ROOT / "reports/vcc2026-h1/reference_cells.csv",
            skip_ceiling=True,
        )
    )


def main() -> None:
    audit_convergence()
    run_raw_ablation()
    evaluate_profile(RAW_ARTIFACT, REPORT / "profile_scores/model1_raw_count")
    full_artifact = DERIVED / "model3_full_expected_profiles.npz"
    if full_artifact.exists():
        evaluate_profile(full_artifact, REPORT / "profile_scores/model3_full")
    plot(Namespace(output=OUTPUT))


if __name__ == "__main__":
    main()
