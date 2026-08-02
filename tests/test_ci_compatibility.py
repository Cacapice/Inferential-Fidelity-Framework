"""CI runtime and Python 3.10 compatibility regressions."""

from enum import Enum
from pathlib import Path

from transfermod.validation import CouplingConclusion

ROOT = Path(__file__).resolve().parents[1]


def test_coupling_conclusion_uses_python310_compatible_enum():
    assert issubclass(CouplingConclusion, str)
    assert issubclass(CouplingConclusion, Enum)
    assert CouplingConclusion.COUPLED_NEGATIVE.value == "coupled_negative"


def test_workflow_uses_node24_native_actions():
    workflow = (ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    assert "actions/checkout@v6" in workflow
    assert "actions/setup-python@v6" in workflow
    assert "actions/checkout@v4" not in workflow
    assert "actions/setup-python@v5" not in workflow
