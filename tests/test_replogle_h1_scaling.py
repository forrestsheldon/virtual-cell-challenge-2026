import numpy as np

from scripts.evaluation.analyze_replogle_h1_scaling import projection_diagnostics


def test_projection_diagnostics_separate_parallel_scale_and_orthogonal_error() -> None:
    source = np.array([1.0, 0.0])
    target = np.array([2.0, 3.0])

    result = projection_diagnostics(source, target)

    assert result["oracle_scale"] == 2
    assert np.isclose(result["cosine"], 2 / np.sqrt(13))
    assert np.isclose(result["source_to_target_norm_ratio"], 1 / np.sqrt(13))
    assert result["orthogonal_norm"] == 3
    assert np.isclose(result["orthogonal_energy_fraction"], 9 / 13)
