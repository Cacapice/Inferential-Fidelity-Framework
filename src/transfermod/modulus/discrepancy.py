"""Decision-discrepancy geometries for inferential fidelity.

A fidelity statement is not determined by an admissible family and decision
functional alone. It also requires a declared geometry on the decision space.
This module provides common scalar discrepancies and lightweight metadata for
making that choice explicit in certification reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

DecisionDiscrepancy = Callable[[float, float], float]


def absolute_discrepancy(q: float, reference_q: float) -> float:
    """Absolute decision discrepancy ``|q - reference_q|``."""
    return abs(float(q) - float(reference_q))


def relative_discrepancy(
    q: float,
    reference_q: float,
    *,
    minimum_reference: float = 0.0,
) -> float:
    """Relative decision discrepancy.

    Raises when the reference magnitude is at or below ``minimum_reference``.
    This makes near-zero ill-conditioning explicit instead of silently changing
    the definition of relative error.
    """
    denominator = abs(float(reference_q))
    if denominator <= minimum_reference:
        raise ValueError(
            "relative discrepancy is undefined or ill-conditioned at the "
            f"declared reference scale: |reference_q|={denominator:g} <= "
            f"minimum_reference={minimum_reference:g}"
        )
    return abs(float(q) - float(reference_q)) / denominator


def stabilized_relative_discrepancy(
    q: float,
    reference_q: float,
    *,
    stabilization: float,
) -> float:
    """Relative discrepancy with a declared positive denominator floor."""
    if stabilization <= 0:
        raise ValueError("stabilization must be positive")
    denominator = max(abs(float(reference_q)), float(stabilization))
    return abs(float(q) - float(reference_q)) / denominator


def symmetric_relative_discrepancy(
    q: float,
    reference_q: float,
    *,
    stabilization: float = 0.0,
) -> float:
    """Symmetric relative discrepancy with optional positive stabilization."""
    if stabilization < 0:
        raise ValueError("stabilization must be nonnegative")
    numerator = 2.0 * abs(float(q) - float(reference_q))
    denominator = abs(float(q)) + abs(float(reference_q)) + stabilization
    if denominator == 0:
        return 0.0
    return numerator / denominator


def named_discrepancy(
    name: str,
    *,
    stabilization: float | None = None,
    minimum_reference: float = 0.0,
) -> DecisionDiscrepancy:
    """Return a scalar discrepancy by name.

    Supported names are ``absolute``, ``relative``, ``stabilized_relative``,
    and ``symmetric_relative``.
    """
    if name == "absolute":
        return absolute_discrepancy
    if name == "relative":
        return lambda q, q0: relative_discrepancy(
            q, q0, minimum_reference=minimum_reference
        )
    if name == "stabilized_relative":
        if stabilization is None:
            raise ValueError("stabilization is required")
        return lambda q, q0: stabilized_relative_discrepancy(
            q, q0, stabilization=stabilization
        )
    if name == "symmetric_relative":
        return lambda q, q0: symmetric_relative_discrepancy(
            q, q0, stabilization=0.0 if stabilization is None else stabilization
        )
    raise ValueError(f"unknown discrepancy: {name!r}")


@dataclass(frozen=True)
class CertificationGeometry:
    """Declared geometry and reference metadata for a certification result."""

    discrepancy: str
    reference_type: str
    reference_q: float
    tolerance: float | None = None
    stabilization: float | None = None

    def __post_init__(self) -> None:
        if not self.discrepancy:
            raise ValueError("discrepancy must be nonempty")
        if not self.reference_type:
            raise ValueError("reference_type must be nonempty")
        if self.tolerance is not None and self.tolerance < 0:
            raise ValueError("tolerance must be nonnegative")
        if self.stabilization is not None and self.stabilization <= 0:
            raise ValueError("stabilization must be positive")


    @classmethod
    def from_standard_metric(
        cls,
        *,
        metric: str,
        reference_q: float,
        tolerance: float | None = None,
        stabilization: float | None = None,
        minimum_reference: float = 1e-12,
        reference_type: str = "declared_target",
    ) -> "CertificationGeometry":
        """Construct a geometry from familiar metric terminology.

        Scalar aliases are interpreted as follows:

        - ``MAE`` / ``absolute`` -> absolute discrepancy;
        - ``MAPE`` / ``APE`` / ``relative`` -> relative discrepancy;
        - ``stabilized_mape`` / ``stabilized_percentage`` ->
          stabilized relative discrepancy;
        - ``sMAPE`` / ``symmetric_percentage`` -> symmetric relative
          discrepancy.

        ``MAPE`` is accepted as a familiar practitioner alias, although for one
        scalar quantity the implemented object is an absolute percentage error.
        """
        key = metric.strip().lower().replace("-", "_")
        aliases = {
            "mae": "absolute",
            "absolute": "absolute",
            "absolute_error": "absolute",
            "mape": "relative",
            "ape": "relative",
            "relative": "relative",
            "absolute_percentage": "relative",
            "stabilized_mape": "stabilized_relative",
            "stabilized_percentage": "stabilized_relative",
            "stabilized_relative": "stabilized_relative",
            "smape": "symmetric_relative",
            "symmetric_percentage": "symmetric_relative",
            "symmetric_relative": "symmetric_relative",
        }
        if key not in aliases:
            raise ValueError(
                f"unsupported standard metric {metric!r}; supported aliases are "
                f"{sorted(aliases)}"
            )
        discrepancy = aliases[key]
        if discrepancy == "relative" and abs(reference_q) <= minimum_reference:
            raise ValueError(
                "Cannot construct relative/percentage geometry because "
                f"|reference_q|={abs(reference_q):g} is at or below the "
                f"configured floor {minimum_reference:g}. Percentage error is "
                "ill-conditioned near zero. Use metric='absolute' or "
                "metric='stabilized_percentage' with a substantive "
                "stabilization scale."
            )
        if discrepancy == "stabilized_relative" and stabilization is None:
            raise ValueError(
                "stabilization is required for stabilized percentage geometry"
            )
        return cls(
            discrepancy=discrepancy,
            reference_type=reference_type,
            reference_q=float(reference_q),
            tolerance=tolerance,
            stabilization=stabilization,
        )

    def discrepancy_function(
        self,
        *,
        minimum_reference: float = 1e-12,
    ) -> DecisionDiscrepancy:
        """Build the callable discrepancy represented by this metadata."""
        return named_discrepancy(
            self.discrepancy,
            stabilization=self.stabilization,
            minimum_reference=minimum_reference,
        )


def decision_diameter(
    values: Iterable[float],
    discrepancy: DecisionDiscrepancy = absolute_discrepancy,
) -> float:
    """Reference-free sampled diameter of a decision-valued family.

    This is the maximum pairwise discrepancy among supplied values. For scalar
    values under absolute discrepancy it equals ``max(values)-min(values)``.
    """
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("values must be nonempty")
    if len(vals) == 1:
        return 0.0
    return max(discrepancy(a, b) for a in vals for b in vals)


def silent_risk_measure(
    values: Sequence[float],
    reference_q: float,
    tolerance: float,
    *,
    discrepancy: DecisionDiscrepancy = absolute_discrepancy,
    weights: Sequence[float] | None = None,
    admitted: Sequence[bool] | None = None,
) -> float:
    """Measure of admitted alternatives whose decision loss exceeds tolerance.

    With equal weights this is an empirical proportion. With supplied weights
    it is the normalized weighted mass. ``admitted`` can restrict the measure
    to alternatives that pass a certification gate.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("values must be nonempty")
    n = len(vals)
    if weights is None:
        w = [1.0] * n
    else:
        if len(weights) != n:
            raise ValueError("weights must match values")
        w = [float(x) for x in weights]
        if any(x < 0 for x in w):
            raise ValueError("weights must be nonnegative")
    if admitted is None:
        mask = [True] * n
    else:
        if len(admitted) != n:
            raise ValueError("admitted must match values")
        mask = [bool(x) for x in admitted]

    total = sum(weight for weight, keep in zip(w, mask) if keep)
    if total <= 0:
        raise ValueError("admitted alternatives must have positive total weight")
    bad = sum(
        weight
        for value, weight, keep in zip(vals, w, mask)
        if keep and discrepancy(value, reference_q) > tolerance
    )
    return bad / total
