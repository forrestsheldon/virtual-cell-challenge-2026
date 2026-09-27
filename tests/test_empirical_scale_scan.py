from __future__ import annotations

import numpy as np

from scripts.evaluation.scan_empirical_scale import leave_one_out_choices


def test_leave_one_out_scale_selection_excludes_held_target() -> None:
    errors = np.array(
        [
            [0.0, 100.0, 100.0],
            [10.0, 0.0, 100.0],
            [10.0, 0.0, 100.0],
        ]
    )
    choices, heldout = leave_one_out_choices(errors)
    assert np.array_equal(choices, [1, 0, 0])
    assert np.array_equal(heldout, [100.0, 10.0, 10.0])
