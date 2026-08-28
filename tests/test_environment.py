from importlib import import_module
from importlib.metadata import version


def test_analysis_stack_imports() -> None:
    assert import_module("pertpy")
    assert import_module("scanpy")
    assert import_module("pyarrow")
    assert version("pertpy")
    assert version("scanpy")
    assert version("pyarrow")
