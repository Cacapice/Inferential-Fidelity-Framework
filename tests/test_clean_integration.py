import math
import os
from pathlib import Path
import subprocess
import sys
import warnings

import pytest


def geometry():
    from transfermod.modulus import CertificationGeometry
    return CertificationGeometry.from_standard_metric(metric="MAE", reference_q=0.0)


def test_canonical_certification_import_does_not_eagerly_load_compat():
    code = (
        "import sys; import transfermod.certification as c; "
        "assert c.CoverageTier.PROVEN_EXACT.exact; "
        "assert 'transfermod.compat' not in sys.modules"
    )
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", code], check=False, env=env
    )
    assert completed.returncode == 0


def test_deprecated_export_is_resolved_lazily():
    import transfermod.certification as certification
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coverage = certification.UnprovenCoverage("legacy")
    assert coverage.tier is certification.CoverageTier.CERTIFIED_FLOOR
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)


def test_modulus_result_rejects_nan_and_negative_epsilon():
    from transfermod.certification import Coverage, modulus_result
    with pytest.raises(ValueError, match="not NaN"):
        modulus_result(
            math.nan,
            coverage=Coverage.certified_floor("f"),
            perturbation_family="f",
            geometry=geometry(),
        )
    with pytest.raises(ValueError, match="epsilon"):
        modulus_result(
            0.1,
            coverage=Coverage.certified_floor("f"),
            perturbation_family="f",
            geometry=geometry(),
            epsilon=-0.1,
        )


def test_family_profile_distinguishes_families_and_information_bases():
    from transfermod.certification import Coverage, modulus_result
    from transfermod.family_profile import FamilySearchProfile
    a = modulus_result(
        0.1,
        coverage=Coverage.certified_floor("a"),
        perturbation_family="ray-a",
        information_basis="gap_series",
        geometry=geometry(),
    )
    b = modulus_result(
        0.2,
        coverage=Coverage.certified_floor("b"),
        perturbation_family="ray-b",
        information_basis="case_identity",
        geometry=geometry(),
    )
    profile = FamilySearchProfile((a, b))
    assert profile.perturbation_families == ("ray-a", "ray-b")
    assert profile.information_bases == ("gap_series", "case_identity")
    assert profile.best_certified_floor == pytest.approx(0.2)


def test_exploratory_only_profile_has_no_certified_floor():
    from transfermod.certification import Coverage, modulus_result
    from transfermod.family_profile import FamilySearchProfile
    result = modulus_result(
        0.3,
        coverage=Coverage.exploratory("grid", "ten points"),
        perturbation_family="grid",
        geometry=geometry(),
    )
    profile = FamilySearchProfile((result,))
    assert profile.best_certified_floor is None
    assert "Strongest certified floor: none" in profile.render()
