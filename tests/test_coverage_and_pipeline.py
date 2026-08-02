import math

import numpy as np
import pytest

from transfermod.certification import (
    CoverageTier,
    Coverage,
                require_exact,
    modulus_result,
)
from transfermod.modulus import (
    CertificationGeometry,
    certify_ray_modulus,
)
from transfermod.perturbations import (
    AdditivePerturbation,
    GraphEdgeWeightPerturbation,
    ParametricResidualPerturbation,
    SpectralModeInjection,
)
from transfermod.pipeline import ContractivityCertificate, compose_moduli


def test_unproven_coverage_returns_lower_bound():
    geometry = CertificationGeometry.from_standard_metric(
        metric="MAE", reference_q=1.0
    )
    result = modulus_result(
        0.2,
        coverage=Coverage.certified_floor("test family"),
        perturbation_family="test family",
        geometry=geometry,
    )
    assert result.tier is CoverageTier.CERTIFIED_FLOOR
    assert not result.exact
    assert "LOWER BOUND" in result.render()


def test_proven_coverage_returns_exact():
    geometry = CertificationGeometry.from_standard_metric(
        metric="MAE", reference_q=1.0
    )
    result = modulus_result(
        0.2,
        coverage=Coverage.proven("Theorem X"),
        perturbation_family="complete family",
        geometry=geometry,
    )
    assert result.tier is CoverageTier.PROVEN_EXACT
    assert require_exact(result) is result


def test_require_exact_rejects_unproven():
    geometry = CertificationGeometry.from_standard_metric(
        metric="MAE", reference_q=1.0
    )
    result = modulus_result(
        0.2,
        coverage=Coverage.certified_floor("ray"),
        perturbation_family="ray",
        geometry=geometry,
    )
    with pytest.raises(ValueError):
        require_exact(result)


def test_certify_ray_modulus_preserves_coverage_status():
    geometry = CertificationGeometry.from_standard_metric(
        metric="MAE", reference_q=1.0
    )
    result = certify_ray_modulus(
        q=lambda w: 1.0 + 2.0 * w,
        agg=lambda w: w,
        eps=0.1,
        geometry=geometry,
        coverage=Coverage.certified_floor("one ray"),
        perturbation_family="one ray",
    )
    assert result.value == pytest.approx(0.2, rel=1e-6)
    assert result.tier is CoverageTier.CERTIFIED_FLOOR


def test_metric_factory_near_zero_guard():
    with pytest.raises(ValueError, match="ill-conditioned"):
        CertificationGeometry.from_standard_metric(
            metric="MAPE",
            reference_q=1e-15,
            minimum_reference=1e-12,
        )


def test_metric_factory_stabilized_percentage():
    geometry = CertificationGeometry.from_standard_metric(
        metric="stabilized_percentage",
        reference_q=0.0,
        stabilization=0.1,
    )
    assert geometry.discrepancy == "stabilized_relative"
    assert geometry.discrepancy_function()(0.02, 0.0) == pytest.approx(0.2)


def test_additive_perturbation():
    family = AdditivePerturbation(
        direction=np.array([1.0, -1.0]),
        validation_metric=lambda a, b: float(np.linalg.norm(b - a)),
    )
    reference = np.array([2.0, 2.0])
    alternative = family.perturb(reference, 0.5)
    assert np.allclose(alternative, [2.5, 1.5])
    assert family.admissibility_error(reference, alternative) == pytest.approx(
        math.sqrt(0.5)
    )


def test_parametric_residual_perturbation():
    family = ParametricResidualPerturbation(
        residual_generator=lambda w: [w, -w],
        validation_metric=lambda a, b: sum(abs(x-y) for x, y in zip(a, b)),
    )
    assert family.perturb([1.0, 1.0], 0.25) == [1.25, 0.75]


def test_spectral_mode_injection():
    family = SpectralModeInjection(
        times=[0.0, 1.0],
        mode_rate=1.0,
        validation_metric=lambda a, b: max(abs(x-y) for x, y in zip(a, b)),
    )
    alt = family.perturb([1.0, 0.5], 0.1)
    assert alt[0] == pytest.approx(1.1)
    assert alt[1] > 0.5


def test_graph_edge_weight_perturbation_conserves_redistributed_weight():
    W = np.zeros((4, 4))
    W[0, 2] = W[2, 0] = 1.0
    W[0, 1] = W[1, 0] = 1.0
    family = GraphEdgeWeightPerturbation(
        weakened_edges=[(0, 2)],
        redistribution_edges=[(0, 1)],
    )
    alt = family.perturb(W, 0.5)
    assert alt[0, 2] == pytest.approx(0.5)
    assert alt[0, 1] == pytest.approx(1.5)


def test_pipeline_composition_with_exact_direct_result():
    geometry = CertificationGeometry.from_standard_metric(
        metric="MAE", reference_q=0.0
    )
    direct = modulus_result(
        0.1,
        coverage=Coverage.proven("Exact enumeration"),
        perturbation_family="all directions",
        geometry=geometry,
    )
    result = compose_moduli(
        lambda eps: 2 * eps,
        lambda eps: 3 * eps,
        0.1,
        direct_composite=direct,
    )
    assert result.stagewise_bound == pytest.approx(0.6)
    assert result.slack_factor == pytest.approx(6.0)
    assert result.slack_upper_bound is None


def test_pipeline_composition_lower_bound_ratio_is_not_called_exact_slack():
    geometry = CertificationGeometry.from_standard_metric(
        metric="MAE", reference_q=0.0
    )
    direct = modulus_result(
        0.1,
        coverage=Coverage.certified_floor("sampled rays"),
        perturbation_family="sampled rays",
        geometry=geometry,
    )
    result = compose_moduli(
        lambda eps: 2 * eps,
        lambda eps: 3 * eps,
        0.1,
        direct_composite=direct,
    )
    assert result.slack_factor is None
    assert result.slack_upper_bound == pytest.approx(6.0)


def test_contractivity_reduces_stagewise_bound():
    plain = compose_moduli(lambda e: 2*e, lambda e: 3*e, 0.1)
    contracted = compose_moduli(
        lambda e: 2*e,
        lambda e: 3*e,
        0.1,
        contractivity=ContractivityCertificate(0.5, "Lipschitz theorem"),
    )
    assert contracted.stagewise_bound < plain.stagewise_bound


def test_censored_modulus_downgrades_tier_and_serializes_bracket():
    geometry = CertificationGeometry.from_standard_metric(
        metric="MAE", reference_q=1.0
    )
    result = certify_ray_modulus(
        q=lambda w: 1.0 + w,
        agg=lambda w: 1e-12 * w,
        eps=1.0,
        geometry=geometry,
        coverage=Coverage.proven("Exact family theorem"),
        perturbation_family="complete family",
    )
    payload = result.to_dict()
    assert result.tier is CoverageTier.CERTIFIED_FLOOR
    assert result.censored
    assert payload["bracket"]["w_agg"]["status"] == "above_bracket"
    assert payload["bracket"]["w_agg"]["bracket"] == [1e-14, 1.0]
    with pytest.raises(ValueError):
        require_exact(result)


def test_left_censored_modulus_is_exploratory_and_not_publishable():
    geometry = CertificationGeometry.from_standard_metric(
        metric="MAE", reference_q=1.0
    )
    result = certify_ray_modulus(
        q=lambda w: 1.0 + w,
        agg=lambda w: 1.0 + w,
        eps=0.5,
        geometry=geometry,
        coverage=Coverage.proven("Exact family theorem"),
        perturbation_family="complete family",
    )
    payload = result.to_dict()
    assert result.tier is CoverageTier.EXPLORATORY_SAMPLE
    assert payload["estimate_kind"] == "indeterminate"
    assert payload["publishable"] is False
    assert "CENSORED (indeterminate)" in result.render()


def test_strict_publish_rejects_indeterminate_result():
    from transfermod.certification import Coverage, modulus_result, strict_publish
    from transfermod.modulus import CertificationGeometry, CensoredScalar, Threshold, BELOW_BRACKET

    value = CensoredScalar(
        1.0,
        thresholds={"w_agg": Threshold(1e-14, BELOW_BRACKET, 1e-14, 1.0, "library_default")},
        estimate_kind="indeterminate",
    )
    result = modulus_result(
        value,
        coverage=Coverage.certified_floor("ray"),
        perturbation_family="ray",
        geometry=CertificationGeometry("absolute", "fixed", 0.0),
    )
    with pytest.raises(ValueError, match="not publishable"):
        strict_publish(result)


def test_to_dict_promotes_effective_tier_to_top_level():
    from transfermod.certification import Coverage, modulus_result
    from transfermod.modulus import CertificationGeometry

    result = modulus_result(
        1.0,
        coverage=Coverage.certified_floor("ray"),
        perturbation_family="ray",
        geometry=CertificationGeometry("absolute", "fixed", 0.0),
    )
    payload = result.to_dict()
    assert payload["tier"] == result.tier.value
    assert payload["tier"] == payload["coverage_tier"]


def test_strict_publish_promotes_effective_tier_to_top_level():
    from transfermod.certification import Coverage, modulus_result, strict_publish
    from transfermod.modulus import CertificationGeometry

    result = modulus_result(
        1.0,
        coverage=Coverage.certified_floor("ray"),
        perturbation_family="ray",
        geometry=CertificationGeometry("absolute", "fixed", 0.0),
    )
    payload = strict_publish(result)
    assert payload["tier"] == result.tier.value
    assert payload["tier"] == payload["coverage_tier"]


def test_strict_publish_returns_serialized_publishable_result():
    from transfermod.certification import Coverage, modulus_result, strict_publish
    from transfermod.modulus import CertificationGeometry

    result = modulus_result(
        1.0,
        coverage=Coverage.certified_floor("ray"),
        perturbation_family="ray",
        geometry=CertificationGeometry("absolute", "fixed", 0.0),
    )
    assert strict_publish(result) == result.to_dict()
    assert result.strict_publish() == result.to_dict()
