import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/cloud/merge_xatlas_shards.py"
SPEC = importlib.util.spec_from_file_location("merge_xatlas", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("guide", "target"),
    [
        ("CAST_P1-1|CAST_P1-2", "CAST"),
        ("ABCD1_P1P2-1|ABCD1_P1P2-2", "ABCD1"),
        ("KDM2B_ENST00000377069.4-1|KDM2B_ENST00000377069.4-2", "KDM2B"),
        ("non-targeting_00998|non-targeting_00141", "Non-Targeting"),
    ],
)
def test_guide_gene_target(guide, target):
    assert MODULE.guide_gene_target(guide) == target
