import importlib

import pytest

from transfermod.certification import Coverage, CoverageTier, modulus_result
from transfermod.modulus import CertificationGeometry


def geometry():
    return CertificationGeometry.from_standard_metric(metric="MAE", reference_q=0.0)


def test_canonical_result_stores_only_tier_via_coverage():
    result = modulus_result(
        0.2,
        coverage=Coverage.certified_floor("f"),
        perturbation_family="f",
        geometry=geometry(),
    )
    assert result.tier is CoverageTier.CERTIFIED_FLOOR
    assert "status" not in result.__dict__


def test_deprecated_coverage_facades_derive_canonical_objects():
    certification = importlib.import_module("transfermod.certification")
    proven = getattr(certification, "ProvenCoverage")
    floor_constructor = getattr(certification, "UnprovenCoverage")
    with pytest.deprecated_call(match="ProvenCoverage"):
        coverage = proven("T")
    assert coverage.tier is CoverageTier.PROVEN_EXACT
    with pytest.deprecated_call(match="UnprovenCoverage"):
        floor = floor_constructor("f")
    assert floor.tier is CoverageTier.CERTIFIED_FLOOR


def test_deprecated_status_and_constructor_remain_source_compatible():
    certification = importlib.import_module("transfermod.certification")
    constructor = getattr(certification, "restricted_modulus_result")
    with pytest.deprecated_call(match="restricted_modulus_result"):
        result = constructor(
            0.2,
            coverage=Coverage.certified_floor("f"),
            perturbation_family="f",
            geometry=geometry(),
        )
    with pytest.deprecated_call(match="BoundStatus"):
        bound_status = certification.BoundStatus
    with pytest.deprecated_call(match="status"):
        assert result.status is bound_status.LOWER_BOUND
