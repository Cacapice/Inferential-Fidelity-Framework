"""Documentation regressions for theorem-scope corrections."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_methods_separates_family_and_full_moduli():
    text = (ROOT / "methods.md").read_text(encoding="utf-8")
    assert "ω_{D,Θ} ≤ τ" in text
    assert "Under (C1)" in text
    assert "family-restricted modulus" in text


def test_methods_does_not_claim_differentiability_equivalence():
    text = (ROOT / "methods.md").read_text(encoding="utf-8")
    assert "the converse need not hold" in text
    assert "differentiable at zero exactly when" not in text


def test_methods_qualifies_effective_computability():
    text = (ROOT / "methods.md").read_text(encoding="utf-8")
    assert "(C4) Effective directional control" in text
    assert "not by themselves an effective\nfinite algorithm" in text


def test_u1_docs_do_not_claim_untruncated_exact_solution():
    files = [
        ROOT / "README.md",
        ROOT / "RELEASE_MANIFEST.md",
        ROOT / "methods.md",
        ROOT / "src/transfermod/spectral/lattice/u1.py",
    ]
    combined = "\n".join(p.read_text(encoding="utf-8") for p in files)
    assert "exactly solvable compact U(1)" not in combined
    assert "Everything here is exact" not in combined
