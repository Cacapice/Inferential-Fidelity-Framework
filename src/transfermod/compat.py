"""Deprecated v1.x compatibility API; removed in v2.0."""

from __future__ import annotations

from enum import Enum
import warnings

from transfermod.certification.coverage import (
    Coverage,
    CoverageTier,
    RestrictedModulusResult,
    modulus_result,
)
from transfermod.modulus.discrepancy import CertificationGeometry


class BoundStatus(str, Enum):
    EXACT = "exact"
    LOWER_BOUND = "lower_bound"
    EXPLORATORY = "exploratory"


_REPLACEMENTS = {
    "BoundStatus": "CoverageTier",
    "CoverageProof": "Coverage",
    "ProvenCoverage": "Coverage.proven",
    "UnprovenCoverage": "Coverage.certified_floor",
    "ExploratorySampleCoverage": "Coverage.exploratory",
    "restricted_modulus_result": "modulus_result",
    "exploratory_modulus_result": "modulus_result",
}


def _deprecated(name: str, replacement: str | None = None) -> None:
    target = _REPLACEMENTS.get(name) if replacement is None else replacement
    warnings.warn(
        f"{name} is deprecated; use {target}. It will be removed in v2.0.",
        DeprecationWarning,
        stacklevel=3,
    )


def warn_deprecated_access(name: str) -> None:
    """Warn when a deprecated type-like facade is resolved."""
    _deprecated(name)


def ProvenCoverage(theorem: str, scope: str = "family covers the admissible set") -> Coverage:
    _deprecated("ProvenCoverage", "Coverage.proven")
    return Coverage.proven(theorem, scope)


def UnprovenCoverage(
    family: str,
    reason: str = "No theorem or exhaustive argument establishes full coverage.",
) -> Coverage:
    _deprecated("UnprovenCoverage", "Coverage.certified_floor")
    return Coverage.certified_floor(family, reason)


def ExploratorySampleCoverage(family: str, sample_description: str) -> Coverage:
    _deprecated("ExploratorySampleCoverage", "Coverage.exploratory")
    return Coverage.exploratory(family, sample_description)


def restricted_modulus_result(*args, **kwargs) -> RestrictedModulusResult:
    _deprecated("restricted_modulus_result", "modulus_result")
    return modulus_result(*args, **kwargs)


def exploratory_modulus_result(
    value: float,
    *,
    coverage: Coverage,
    perturbation_family: str,
    geometry: CertificationGeometry,
    epsilon: float | None = None,
    notes: tuple[str, ...] = (),
    information_basis: str | None = None,
) -> RestrictedModulusResult:
    _deprecated("exploratory_modulus_result", "modulus_result")
    if coverage.tier is not CoverageTier.EXPLORATORY_SAMPLE:
        raise ValueError("exploratory results require Coverage.exploratory(...)")
    return modulus_result(
        value,
        coverage=coverage,
        perturbation_family=perturbation_family,
        geometry=geometry,
        epsilon=epsilon,
        notes=notes,
        information_basis=information_basis,
    )

CoverageProof = Coverage
