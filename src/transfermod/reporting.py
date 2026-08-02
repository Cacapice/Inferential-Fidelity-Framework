"""Formatting helpers kept separate from scientific result state."""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from transfermod.certification import RestrictedModulusResult


def format_modulus_result(result: "RestrictedModulusResult") -> str:
    lines = [
        f"Restricted modulus: {result.value:.8g}",
        f"Status: {result.tier.label}",
        f"Coverage tier: {result.tier.value}",
        f"Perturbation family: {result.perturbation_family}",
        f"Coverage: {result.coverage.description}",
        f"Discrepancy: {result.geometry.discrepancy}",
        f"Reference: {result.geometry.reference_type} "
        f"(Q0={result.geometry.reference_q:.8g})",
    ]
    if result.censored:
        lines.append(f"Threshold status: CENSORED ({result.estimate_kind})")
        lines.append(f"Publishable scalar: {'yes' if result.publishable else 'no'}")
        if result.bracket:
            for name, threshold in result.bracket.items():
                bracket = threshold.get("bracket")
                lines.append(
                    f"Threshold {name}: {threshold.get('status')} bracket={bracket}"
                )
    if result.epsilon is not None:
        lines.append(f"Validation tolerance: {result.epsilon:.8g}")
    if result.information_basis is not None:
        lines.append(f"Information basis: {result.information_basis}")
    lines.extend(f"Note: {note}" for note in result.notes)
    return "\n".join(lines)
