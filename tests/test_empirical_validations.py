"""Regression tests for the three empirical validation tracks."""

import pytest

from transfermod.certification import CoverageTier
from transfermod.validation import (
    CouplingConclusion,
    run_bayesian_silent_risk_validation,
    run_learned_operator_validation,
    run_pde_scaling_validation,
)


def test_learned_operator_validation_reports_negative_decoupling_result():
    result = run_learned_operator_validation(
        n_train=300,
        n_validation=100,
    )
    assert result.validation_passed
    assert result.validation_p95_relative_l2 < result.aggregate_error_threshold
    assert result.conclusion is CouplingConclusion.COUPLED_NEGATIVE
    assert result.hidden_failure_count == 0
    assert (
        result.smallest_aggregate_point.relative_global_l2_error
        > result.aggregate_error_threshold
    )
    assert result.ratio_min >= 0.0
    assert result.ratio_max < 0.5
    assert result.strongest_decision_point.decision_error > 0.02
    assert result.result.tier is CoverageTier.CERTIFIED_FLOOR
    assert result.result.information_basis == (
        "trained_operator_localized_challenge"
    )


def test_learned_operator_family_profile_is_fully_reported():
    result = run_learned_operator_validation(
        n_train=300,
        n_validation=100,
    )
    assert len(result.family_points) == result.family_evaluations == 45
    assert result.to_dict()["conclusion"] == "coupled_negative"
    assert all(
        point.relative_global_l2_error > result.aggregate_error_threshold
        for point in result.family_points
    )


def test_pde_scaling_validation_runs_in_two_and_three_dimensions():
    validation = run_pde_scaling_validation()
    assert [item.dimension for item in validation.results] == [2, 3]
    for item in validation.results:
        assert item.spatial_points > 0
        assert item.uniform_evaluations > 0
        assert item.adaptive_evaluations > 0
        assert item.uniform_best_q_error > 0
        assert item.adaptive_best_q_error > 0
        assert item.adaptive_fraction_of_uniform >= 0.75


def test_bayesian_silent_risk_matches_probability_semantics():
    result = run_bayesian_silent_risk_validation(n_draws=5_000)
    assert 0.0 <= result.posterior_silent_risk <= 1.0
    assert 0.0 <= result.recurrence_probability <= 1.0
    assert result.posterior_interval[0] <= result.posterior_median_q
    assert result.posterior_median_q <= result.posterior_interval[1]
    assert result.posterior_fidelity_radius >= 0.0
    assert result.credible_set_worst_case >= result.posterior_fidelity_radius
    assert result.credible_set_decision_diameter >= 0.0


def test_validation_argument_guards():
    with pytest.raises(ValueError):
        run_bayesian_silent_risk_validation(n_draws=10)
    with pytest.raises(ValueError):
        run_learned_operator_validation(validation_threshold=0.0)
