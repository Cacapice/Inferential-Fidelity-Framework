"""The trained-operator result must remain framed as a negative finding."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_validation_report_states_no_hidden_failure():
    text = (ROOT / "VALIDATION_REPORT.md").read_text(encoding="utf-8")
    assert "no decision failure invisible to aggregate validation was found" in text
    assert "not Silent Risk" in text


def test_roadmap_preserves_negative_result():
    text = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    assert "negative result for decoupling" in text
    assert "no post-result substitution" in text


def test_external_protocol_freezes_confirmatory_choices():
    text = (ROOT / "LEARNED_OPERATOR_PROTOCOL.md").read_text(encoding="utf-8")
    for phrase in (
        "model repositories, checkpoints, and hashes",
        "aggregate validation metric",
        "decision functional",
        "search algorithm, budget, seeds, and stopping rules",
        "A failed confirmatory search is reported",
    ):
        assert phrase in text
