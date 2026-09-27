import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/cloud/prepare_kolf_recovery.py"
SPEC = importlib.util.spec_from_file_location("prepare_kolf_recovery", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_recovery_contract_accepts_complete_union() -> None:
    h1 = {"H1_1", "H1_2", "BOTH"}
    vcc = {"VCC_1", "BOTH"}
    panel = pd.DataFrame({"target_gene": sorted(h1 | vcc)})
    panel["in_h1"] = panel.target_gene.isin(h1)
    panel["in_vcc2026"] = panel.target_gene.isin(vcc)

    summary = MODULE.coverage(panel, h1 | vcc)

    assert summary["h1_requested"] == 3
    assert summary["vcc_requested"] == 2
    assert summary["panel_overlap"] == 1
    assert summary["union_available"] == 4


def test_recovery_contract_rejects_former_vcc_only_panel() -> None:
    panel = pd.DataFrame(
        {"target_gene": ["A"], "in_h1": [False], "in_vcc2026": [True]}
    )
    summary = MODULE.coverage(panel, {"A"})

    with pytest.raises(ValueError, match="h1_requested"):
        MODULE.assert_expected(summary)
