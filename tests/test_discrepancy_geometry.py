import pytest

from transfermod.modulus import (
    CertificationGeometry,
    absolute_discrepancy,
    decision_diameter,
    named_discrepancy,
    ray_decision_modulus,
    relative_discrepancy,
    silent_risk_measure,
    stabilized_relative_discrepancy,
    symmetric_relative_discrepancy,
)


def test_absolute_discrepancy():
    assert absolute_discrepancy(0.2, 0.8) == pytest.approx(0.6)


def test_relative_discrepancy():
    assert relative_discrepancy(1.2, 0.8) == pytest.approx(0.5)


def test_relative_discrepancy_rejects_zero_and_declared_near_zero():
    with pytest.raises(ValueError):
        relative_discrepancy(0.1, 0.0)
    with pytest.raises(ValueError):
        relative_discrepancy(0.1, 1e-8, minimum_reference=1e-6)


def test_stabilized_relative_discrepancy():
    assert stabilized_relative_discrepancy(
        0.01, 0.001, stabilization=0.05
    ) == pytest.approx(0.18)


def test_symmetric_relative_discrepancy_zero_pair():
    assert symmetric_relative_discrepancy(0.0, 0.0) == 0.0


def test_named_discrepancy_factory():
    metric = named_discrepancy("stabilized_relative", stabilization=0.1)
    assert metric(0.2, 0.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        named_discrepancy("unknown")


def test_decision_diameter_absolute_scalar():
    assert decision_diameter([0.2, 0.8, 0.5]) == pytest.approx(0.6)


def test_decision_diameter_singleton_and_empty():
    assert decision_diameter([0.3]) == 0.0
    with pytest.raises(ValueError):
        decision_diameter([])


def test_silent_risk_empirical_measure():
    risk = silent_risk_measure(
        [0.0, 0.1, 0.4, 0.7],
        reference_q=0.0,
        tolerance=0.2,
    )
    assert risk == pytest.approx(0.5)


def test_silent_risk_weighted_and_admitted():
    risk = silent_risk_measure(
        [0.0, 0.5, 0.9],
        reference_q=0.0,
        tolerance=0.4,
        weights=[1.0, 2.0, 7.0],
        admitted=[True, True, False],
    )
    assert risk == pytest.approx(2.0 / 3.0)


def test_ray_decision_modulus_uses_declared_geometry():
    q = lambda w: 2.0 + w
    agg = lambda w: w
    out = ray_decision_modulus(
        q,
        agg,
        eps=0.1,
        reference_q=2.0,
        discrepancy=absolute_discrepancy,
        lo=1e-12,
        hi=1.0,
    )
    assert out == pytest.approx(0.1, rel=1e-6)


def test_certification_geometry_records_choice():
    geometry = CertificationGeometry(
        discrepancy="relative",
        reference_type="declared_target",
        reference_q=2.0,
        tolerance=0.1,
    )
    assert geometry.discrepancy == "relative"
    assert geometry.reference_q == 2.0
    with pytest.raises(ValueError):
        CertificationGeometry("", "declared_target", 2.0)
