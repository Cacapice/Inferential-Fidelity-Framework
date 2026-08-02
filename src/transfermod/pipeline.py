"""Diagnostics for composing stage-wise transfer moduli."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from transfermod.certification import CoverageTier, RestrictedModulusResult

ModulusFn = Callable[[float], float]


@dataclass(frozen=True)
class ContractivityCertificate:
    """Certificate that an intermediate map contracts the declared geometry."""

    factor: float
    provenance: str

    def __post_init__(self) -> None:
        if not 0 <= self.factor <= 1:
            raise ValueError("contractivity factor must lie in [0, 1]")


@dataclass(frozen=True)
class PipelineCompositionResult:
    """Stage-wise composition plus any direct comparison and slack provenance."""

    epsilon: float
    first_stage_value: float
    propagated_intermediate_value: float
    stagewise_bound: float
    direct_composite_value: float | None
    direct_tier: CoverageTier | None
    slack_factor: float | None
    slack_upper_bound: float | None
    openness_slack_bound: float | None
    warnings: tuple[str, ...]

    def render(self) -> str:
        lines = [
            f"Input tolerance: {self.epsilon:.8g}",
            f"First-stage modulus: {self.first_stage_value:.8g}",
            f"Propagated intermediate tolerance: "
            f"{self.propagated_intermediate_value:.8g}",
            f"Stage-wise pipeline bound: {self.stagewise_bound:.8g}",
        ]
        if self.direct_composite_value is None:
            lines.append("Direct composite modulus: not computed")
            lines.append("Slack: unknown")
        else:
            label = (
                "exact" if self.direct_tier is CoverageTier.PROVEN_EXACT
                else "family-restricted lower bound"
            )
            lines.append(
                f"Direct composite value: {self.direct_composite_value:.8g} "
                f"({label})"
            )
            if self.slack_factor is not None:
                lines.append(f"Composition slack: {self.slack_factor:.3g}x")
            if self.slack_upper_bound is not None:
                lines.append(
                    "Composition slack upper bound from restricted denominator: "
                    f"{self.slack_upper_bound:.3g}x"
                )
        if self.openness_slack_bound is not None:
            lines.append(
                f"Theorem-backed openness slack bound: "
                f"{self.openness_slack_bound:.3g}x"
            )
        lines.extend(f"Warning: {w}" for w in self.warnings)
        return "\n".join(lines)


def compose_moduli(
    first: ModulusFn,
    second: ModulusFn,
    epsilon: float,
    *,
    direct_composite: RestrictedModulusResult | None = None,
    openness_constant: float | None = None,
    contractivity: ContractivityCertificate | None = None,
) -> PipelineCompositionResult:
    """Compose two modulus bounds and diagnose observable slack.

    ``second(first(epsilon))`` is the generic stage-wise bound. A proven
    contractivity certificate scales the intermediate tolerance before the
    second stage. A direct composite result enables a slack comparison; when
    that result is only a lower bound, the resulting ratio is only an upper
    bound on the true slack.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    if openness_constant is not None and openness_constant <= 0:
        raise ValueError("openness_constant must be positive")

    first_value = float(first(epsilon))
    propagated = first_value
    warnings: list[str] = []
    if contractivity is not None:
        propagated *= contractivity.factor
        warnings.append(
            "Intermediate tolerance reduced using contractivity certificate: "
            + contractivity.provenance
        )
    stagewise = float(second(propagated))

    direct_value = None
    direct_tier = None
    slack = None
    slack_upper = None
    if direct_composite is not None:
        direct_value = direct_composite.value
        direct_tier = direct_composite.tier
        if direct_value > 0:
            ratio = stagewise / direct_value
            if direct_composite.exact:
                slack = ratio
            else:
                slack_upper = ratio
                warnings.append(
                    "Direct composite value is a lower bound because coverage "
                    "is unproven; the displayed ratio is an upper bound on "
                    "composition slack, not an estimate."
                )
        elif stagewise > 0:
            warnings.append(
                "Direct composite value is zero; no finite slack ratio can be "
                "reported."
            )
    else:
        warnings.append(
            "No direct composite search supplied. The stage-wise value is a "
            "valid bound, not a calibrated pipeline error budget."
        )

    openness_bound = None
    if openness_constant is not None:
        openness_bound = 1.0 / openness_constant

    return PipelineCompositionResult(
        epsilon=float(epsilon),
        first_stage_value=first_value,
        propagated_intermediate_value=propagated,
        stagewise_bound=stagewise,
        direct_composite_value=direct_value,
        direct_tier=direct_tier,
        slack_factor=slack,
        slack_upper_bound=slack_upper,
        openness_slack_bound=openness_bound,
        warnings=tuple(warnings),
    )
