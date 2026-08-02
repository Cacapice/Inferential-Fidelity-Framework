import math

import numpy as np
import pytest

from transfermod.certification import (
        CoverageTier,
        )
from transfermod.exact_coverage import (
    EllipsoidalLinearCoverage,
    HilbertLinearFunctionalCoverage,
    LeadingEigenvalueFrobeniusCoverage,
    SketchNullspaceCoverage,
)
from transfermod.family_profile import FamilySearchProfile
from transfermod.modulus import CertificationGeometry
from transfermod.certification import Coverage, modulus_result


def geometry():
    return CertificationGeometry.from_standard_metric(
        metric="MAE", reference_q=0.0
    )


def test_hilbert_linear_exact_coverage():
    theorem = HilbertLinearFunctionalCoverage([3.0, 4.0])
    assert theorem.exact_modulus(0.2) == pytest.approx(1.0)
    assert np.allclose(theorem.extremal_direction(), [0.6, 0.8])
    result = theorem.certify(0.2, geometry=geometry())
    assert result.tier is CoverageTier.PROVEN_EXACT
    assert result.tier is CoverageTier.PROVEN_EXACT


def test_ellipsoidal_linear_exact_value():
    theorem = EllipsoidalLinearCoverage(
        [[4.0, 0.0], [0.0, 1.0]],
        [2.0, 1.0],
    )
    expected = math.sqrt(2.0)
    assert theorem.exact_modulus(1.0) == pytest.approx(expected)
    h = theorem.extremal_direction(1.0)
    M = np.array([[4.0, 0.0], [0.0, 1.0]])
    assert h @ M @ h == pytest.approx(1.0)


def test_ellipsoidal_kernel_mismatch_is_infinite():
    theorem = EllipsoidalLinearCoverage(
        [[1.0, 0.0], [0.0, 0.0]],
        [0.0, 1.0],
    )
    assert math.isinf(theorem.exact_modulus(0.1))


def test_sketch_nullspace_exact_coverage():
    P = np.array([[1.0, 0.0, 0.0]])
    q = np.array([0.0, 3.0, 4.0])
    theorem = SketchNullspaceCoverage(P, q)
    assert theorem.exact_modulus(0.2) == pytest.approx(1.0)
    h = theorem.extremal_direction(0.2)
    assert np.linalg.norm(P @ h) < 1e-12
    assert np.linalg.norm(h) == pytest.approx(0.2)


def test_sketch_query_in_rowspace_has_zero_exposure():
    theorem = SketchNullspaceCoverage([[1.0, 0.0]], [2.0, 0.0])
    assert theorem.exact_modulus(1.0) == pytest.approx(0.0)


def test_leading_eigenvalue_rank_one_exact():
    A = np.diag([1.0, 3.0])
    theorem = LeadingEigenvalueFrobeniusCoverage(A)
    Delta = theorem.extremal_perturbation(0.4)
    assert np.linalg.norm(Delta, "fro") == pytest.approx(0.4)
    shift = np.linalg.eigvalsh(A + Delta)[-1] - np.linalg.eigvalsh(A)[-1]
    assert shift == pytest.approx(0.4)


def test_exploratory_tier():
    result = modulus_result(
        0.3,
        coverage=Coverage.exploratory("grid", "10 amplitudes"),
        perturbation_family="grid",
        geometry=geometry(),
    )
    assert result.tier is CoverageTier.EXPLORATORY_SAMPLE
    assert result.tier is CoverageTier.EXPLORATORY_SAMPLE


def test_family_search_profile_uses_max_floor_not_mean():
    a = modulus_result(
        0.2,
        coverage=Coverage.certified_floor("a"),
        perturbation_family="a",
        geometry=geometry(),
    )
    b = modulus_result(
        0.5,
        coverage=Coverage.certified_floor("b"),
        perturbation_family="b",
        geometry=geometry(),
    )
    profile = FamilySearchProfile((a, b))
    assert profile.best_certified_floor == pytest.approx(0.5)
    assert profile.between_family_range == pytest.approx(0.3)
