"""Coverage-aware certification results with one canonical status type."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import warnings

from transfermod.modulus.discrepancy import CertificationGeometry


class CoverageTier(str, Enum):
    """Epistemic status of a transfer calculation."""

    PROVEN_EXACT = "proven_exact"
    CERTIFIED_FLOOR = "certified_floor"
    EXPLORATORY_SAMPLE = "exploratory_sample"

    @property
    def exact(self) -> bool:
        return self is CoverageTier.PROVEN_EXACT

    @property
    def certified(self) -> bool:
        return self is not CoverageTier.EXPLORATORY_SAMPLE

    @property
    def label(self) -> str:
        return {
            CoverageTier.PROVEN_EXACT: "EXACT",
            CoverageTier.CERTIFIED_FLOOR: "CERTIFIED FLOOR (LOWER BOUND)",
            CoverageTier.EXPLORATORY_SAMPLE: "EXPLORATORY SAMPLE",
        }[self]


@dataclass(frozen=True)
class Coverage:
    """Coverage tier and human-auditable provenance."""

    tier: CoverageTier
    family: str
    provenance: str

    @classmethod
    def proven(cls, theorem: str, scope: str = "family covers the admissible set") -> "Coverage":
        return cls(CoverageTier.PROVEN_EXACT, theorem, scope)

    @classmethod
    def certified_floor(
        cls,
        family: str,
        reason: str = "No theorem or exhaustive argument establishes full coverage.",
    ) -> "Coverage":
        return cls(CoverageTier.CERTIFIED_FLOOR, family, reason)

    @classmethod
    def exploratory(cls, family: str, sample_description: str) -> "Coverage":
        return cls(CoverageTier.EXPLORATORY_SAMPLE, family, sample_description)

    @property
    def description(self) -> str:
        prefix = {
            CoverageTier.PROVEN_EXACT: "Coverage proved",
            CoverageTier.CERTIFIED_FLOOR: "Coverage unproved",
            CoverageTier.EXPLORATORY_SAMPLE: "Exploratory sample",
        }[self.tier]
        return f"{prefix} for {self.family}: {self.provenance}"


@dataclass(frozen=True)
class RestrictedModulusResult:
    """A transfer value with one coverage tier and its provenance."""

    value: float
    coverage: Coverage
    perturbation_family: str
    geometry: CertificationGeometry
    epsilon: float | None = None
    notes: tuple[str, ...] = ()
    information_basis: str | None = None
    censored: bool = False
    bracket: dict[str, dict[str, object]] | None = None
    estimate_kind: str = "measured_crossing"
    publishable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.coverage, Coverage):
            raise TypeError("coverage must be a Coverage instance")
        if math.isnan(self.value) or self.value < 0:
            raise ValueError("modulus value must be nonnegative and not NaN")
        if self.epsilon is not None and (math.isnan(self.epsilon) or self.epsilon < 0):
            raise ValueError("epsilon must be nonnegative and not NaN")

    @property
    def tier(self) -> CoverageTier:
        return self.coverage.tier

    @property
    def exact(self) -> bool:
        return self.tier.exact

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "tier": self.tier.value,
            "coverage_tier": self.tier.value,
            "coverage": self.coverage.description,
            "perturbation_family": self.perturbation_family,
            "discrepancy": self.geometry.discrepancy,
            "reference_type": self.geometry.reference_type,
            "reference_q": self.geometry.reference_q,
            "epsilon": self.epsilon,
            "information_basis": self.information_basis,
            "censored": self.censored,
            "bracket": self.bracket,
            "estimate_kind": self.estimate_kind,
            "publishable": self.publishable,
            "notes": list(self.notes),
        }


    def strict_publish(self) -> dict[str, object]:
        """Serialize only a result whose scalar semantics permit publication.

        This guard is intentionally applied at the reporting boundary rather
        than to ordinary arithmetic, preserving backward-compatible float
        behavior while preventing censored/indeterminate values from being
        silently laundered into a published payload.
        """
        if not self.publishable:
            raise ValueError(
                "result is not publishable: estimate_kind="
                f"{self.estimate_kind!r}; inspect censoring status and bracket"
            )
        return self.to_dict()

    def render(self) -> str:
        """Compatibility convenience; prefer ``transfermod.reporting``."""
        from transfermod.reporting import format_modulus_result
        return format_modulus_result(self)

    @property
    def status(self):
        """Deprecated compatibility view; use ``tier``."""
        warnings.warn(
            "RestrictedModulusResult.status is deprecated; use .tier.",
            DeprecationWarning,
            stacklevel=2,
        )
        from transfermod.compat import BoundStatus
        return {
            CoverageTier.PROVEN_EXACT: BoundStatus.EXACT,
            CoverageTier.CERTIFIED_FLOOR: BoundStatus.LOWER_BOUND,
            CoverageTier.EXPLORATORY_SAMPLE: BoundStatus.EXPLORATORY,
        }[self.tier]


def modulus_result(
    value: float,
    *,
    coverage: Coverage,
    perturbation_family: str,
    geometry: CertificationGeometry,
    epsilon: float | None = None,
    notes: tuple[str, ...] = (),
    information_basis: str | None = None,
) -> RestrictedModulusResult:
    censored = bool(getattr(value, "censored", False))
    bracket = getattr(value, "bracket", None)
    estimate_kind = getattr(value, "estimate_kind", "measured_crossing")
    publishable = bool(getattr(value, "publishable", True))
    if censored:
        if coverage.tier is CoverageTier.PROVEN_EXACT:
            coverage = Coverage.certified_floor(
                coverage.family,
                "Threshold crossing was censored; the reported value is one-sided. "
                + coverage.provenance,
            )
        elif coverage.tier is CoverageTier.CERTIFIED_FLOOR:
            coverage = Coverage.exploratory(
                coverage.family,
                "Threshold crossing was censored; the reported value is one-sided. "
                + coverage.provenance,
            )
        if estimate_kind == "indeterminate" and coverage.tier is not CoverageTier.EXPLORATORY_SAMPLE:
            coverage = Coverage.exploratory(
                coverage.family,
                "Censor direction does not identify a valid scalar bound. "
                + coverage.provenance,
            )
        notes = tuple(notes) + (
            "At least one threshold crossing was censored; inspect estimate_kind and bracket before reporting the scalar.",
        )
    return RestrictedModulusResult(
        value=float(value),
        coverage=coverage,
        perturbation_family=perturbation_family,
        geometry=geometry,
        epsilon=epsilon,
        notes=tuple(notes),
        information_basis=information_basis,
        censored=censored,
        bracket=bracket,
        estimate_kind=estimate_kind,
        publishable=publishable,
    )


def require_exact(result: RestrictedModulusResult) -> RestrictedModulusResult:
    if not result.exact:
        raise ValueError(
            "exact certification requested, but coverage is not proven. "
            + result.coverage.description
        )
    return result


def strict_publish(result: RestrictedModulusResult) -> dict[str, object]:
    """Return a publication payload or raise for a non-publishable result."""
    if not isinstance(result, RestrictedModulusResult):
        raise TypeError("strict_publish requires a RestrictedModulusResult")
    return result.strict_publish()
