"""Targeted branch tests for public utility modules."""

import numpy as np
import pytest

from transfermod.certification import (
    Coverage,
                    modulus_result,
)
from transfermod.exact_coverage import (
    EllipsoidalLinearCoverage,
    HilbertLinearFunctionalCoverage,
    LeadingEigenvalueFrobeniusCoverage,
    SketchNullspaceCoverage,
)
from transfermod.family_profile import FamilySearchProfile
from transfermod.modulus import CertificationGeometry
from transfermod.modulus.discrepancy import decision_diameter, silent_risk_measure
from transfermod.perturbations import (
    AdditivePerturbation,
    GraphEdgeWeightPerturbation,
    ParametricResidualPerturbation,
    SpectralModeInjection,
)
from transfermod.pipeline import ContractivityCertificate, compose_moduli


def geom():
    return CertificationGeometry.from_standard_metric(metric="MAE", reference_q=0.0)


def test_family_profile_empty_and_exact_paths():
    with pytest.raises(ValueError):
        FamilySearchProfile(())
    a = modulus_result(
        0.1, coverage=Coverage.proven("T"), perturbation_family="a", geometry=geom()
    )
    b = modulus_result(
        0.2, coverage=Coverage.proven("T"), perturbation_family="a", geometry=geom()
    )
    p = FamilySearchProfile((a, b))
    assert p.all_exact
    assert p.perturbation_families == ("a",)
    assert p.information_bases == ()
    assert "Families evaluated" in p.render()


def test_pipeline_validation_zero_and_render_paths():
    with pytest.raises(ValueError):
        ContractivityCertificate(1.1, "bad")
    with pytest.raises(ValueError):
        compose_moduli(lambda e: e, lambda e: e, -1)
    with pytest.raises(ValueError):
        compose_moduli(lambda e: e, lambda e: e, 1, openness_constant=0)
    zero = modulus_result(
        0.0, coverage=Coverage.proven("T"), perturbation_family="z", geometry=geom()
    )
    r = compose_moduli(lambda e: e, lambda e: 2*e, 1, direct_composite=zero, openness_constant=0.5)
    assert r.slack_factor is None
    assert "no finite slack ratio" in r.render().lower()
    assert "2x" in r.render()


def test_exact_coverage_error_and_zero_branches():
    with pytest.raises(ValueError):
        HilbertLinearFunctionalCoverage([[1.0]]).exact_modulus(1)
    h = HilbertLinearFunctionalCoverage([0.0, 0.0])
    assert np.allclose(h.extremal_direction(), [0, 0])
    with pytest.raises(ValueError):
        h.exact_modulus(-1)

    with pytest.raises(ValueError):
        EllipsoidalLinearCoverage([[1, 0]], [1, 0]).exact_modulus(1)
    with pytest.raises(ValueError):
        EllipsoidalLinearCoverage([[1, 2], [0, 1]], [1, 0]).exact_modulus(1)
    with pytest.raises(ValueError):
        EllipsoidalLinearCoverage([[1, 0], [0, -1]], [1, 0]).exact_modulus(1)
    e = EllipsoidalLinearCoverage([[1, 0], [0, 1]], [0, 0])
    assert np.allclose(e.extremal_direction(1), [0, 0])
    with pytest.raises(ValueError):
        e.extremal_direction(-1)

    s = SketchNullspaceCoverage(np.eye(2), [1, 0])
    assert np.allclose(s.extremal_direction(1), [0, 0])
    with pytest.raises(ValueError):
        SketchNullspaceCoverage([[1, 0, 0]], [1, 0]).exact_modulus(1)
    with pytest.raises(ValueError):
        s.exact_modulus(-1)

    with pytest.raises(ValueError):
        LeadingEigenvalueFrobeniusCoverage([[1, 2, 3]]).extremal_perturbation(1)
    with pytest.raises(ValueError):
        LeadingEigenvalueFrobeniusCoverage([[1, 2], [0, 1]]).extremal_perturbation(1)
    with pytest.raises(ValueError):
        LeadingEigenvalueFrobeniusCoverage(np.eye(2)).exact_modulus(-1)


def test_discrepancy_edge_branches():
    assert decision_diameter([1.0]) == 0.0
    with pytest.raises(ValueError):
        decision_diameter([])
    with pytest.raises(ValueError):
        silent_risk_measure([], 0, 1)
    with pytest.raises(ValueError):
        silent_risk_measure([1], 0, -1)
    with pytest.raises(ValueError):
        silent_risk_measure([1, 2], 0, 1, weights=[1])
    with pytest.raises(ValueError):
        silent_risk_measure([1], 0, 1, weights=[-1])
    with pytest.raises(ValueError):
        silent_risk_measure([1, 2], 0, 1, admitted=[True])
    with pytest.raises(ValueError):
        silent_risk_measure([1], 0, 1, admitted=[False])


def test_perturbation_bounds_and_metric_branches():
    add = AdditivePerturbation(
        direction=(1.0, -1.0),
        validation_metric=lambda a, b: sum(abs(x-y) for x, y in zip(a, b)),
    )
    assert add.perturb((1.0, 1.0), 0.5) == (1.5, 0.5)
    assert add.parameter_bounds() == (0.0, 1.0)
    with pytest.raises(ValueError):
        add.perturb((1, 1), 2)

    residual = ParametricResidualPerturbation(
        residual_generator=lambda w: (w,),
        validation_metric=lambda a, b: abs(a[0]-b[0]),
    )
    assert residual.admissibility_error((0,), residual.perturb((0,), 0.5)) == 0.5
    assert residual.parameter_bounds() == (0.0, 1.0)
    with pytest.raises(ValueError):
        residual.perturb((0,), -1)

    spectral = SpectralModeInjection(
        times=[0, 1],
        mode_rate=1,
        validation_metric=lambda a, b: max(abs(x-y) for x,y in zip(a,b)),
    )
    assert spectral.parameter_bounds() == (0.0, 1.0)
    with pytest.raises(ValueError):
        spectral.perturb([1], 0.5)
    with pytest.raises(ValueError):
        spectral.perturb([1, 1], 2)

    W = np.zeros((2, 2))
    graph = GraphEdgeWeightPerturbation([])
    assert graph.parameter_bounds() == (0.0, 1.0)
    with pytest.raises(ValueError):
        graph.perturb(W, 2)
    with pytest.raises(ValueError):
        graph.admissibility_error(W, W)
