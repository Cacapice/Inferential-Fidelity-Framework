"""Executable empirical validations for TransferMod."""

from transfermod.validation.learned_operator import (
    CouplingConclusion,
    LearnedOperatorFamilyPoint,
    LearnedOperatorValidation,
    run_learned_operator_validation,
)
from transfermod.validation.pde_scaling import (
    PDEScalingValidation,
    run_pde_scaling_validation,
)
from transfermod.validation.bayesian import (
    BayesianSilentRiskValidation,
    run_bayesian_silent_risk_validation,
)

__all__ = [
    "CouplingConclusion",
    "LearnedOperatorFamilyPoint",
    "LearnedOperatorValidation",
    "run_learned_operator_validation",
    "PDEScalingValidation",
    "run_pde_scaling_validation",
    "BayesianSilentRiskValidation",
    "run_bayesian_silent_risk_validation",
]
