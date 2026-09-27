import importlib.util
from pathlib import Path

import numpy as np
import pyarrow as pa

SCRIPT = Path(__file__).parents[1] / "scripts/cloud/aggregate_xatlas_shards.py"
SPEC = importlib.util.spec_from_file_location("aggregate_xatlas", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_aggregate_table_groups_targets_guides_and_controls(monkeypatch):
    monkeypatch.setattr(MODULE, "N_GENES", 4)
    table = pa.table(
        {
            "gene_token_id": [[0, 2], [1, 2], [0, 3], [2]],
            "gene_expression": [[2.0, 1.0], [3.0, 4.0], [5.0, 1.0], [9.0]],
            "gene_target": ["A", "A", "Non-Targeting", "B"],
            "guide_target": ["A-1", "A-2", "NT-1", "B-1"],
            "sample": ["batch1"] * 4,
            "pass_guide_filter": [1, 1, 1, 0],
        }
    )

    arrays, audit = MODULE.aggregate_table(table, {"A", "B"})

    target = MODULE.sparse.csr_matrix(
        (
            arrays["target_data"],
            arrays["target_indices"],
            arrays["target_indptr"],
        ),
        shape=arrays["target_shape"],
    ).toarray()
    assert arrays["target_labels"].tolist() == ["A", "Non-Targeting"]
    np.testing.assert_array_equal(target, [[2, 3, 5, 0], [5, 0, 0, 1]])
    np.testing.assert_array_equal(arrays["control_sum"], [5, 0, 0, 1])
    assert audit["n_input_cells"] == 4
    assert audit["n_pass_guide_filter"] == 3
    assert audit["n_selected_cells"] == 3
    assert audit["n_control_cells"] == 1
