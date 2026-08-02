"""Cross-family search profiles.

Agreement among families is useful robustness evidence but is not a proof of
coverage. The maximum certified floor is the strongest one-sided statement.
"""

from __future__ import annotations

from dataclasses import dataclass

from transfermod.certification import RestrictedModulusResult


@dataclass(frozen=True)
class FamilySearchProfile:
    """Summary of several perturbation-family searches."""

    results: tuple[RestrictedModulusResult, ...]

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("at least one family result is required")

    @property
    def certified_results(self) -> tuple[RestrictedModulusResult, ...]:
        return tuple(result for result in self.results if result.tier.certified)

    @property
    def best_certified_floor(self) -> float | None:
        certified = self.certified_results
        return max((result.value for result in certified), default=None)

    @property
    def between_family_range(self) -> float:
        values = [result.value for result in self.results]
        return max(values) - min(values)

    @property
    def perturbation_families(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(result.perturbation_family for result in self.results))

    @property
    def information_bases(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                result.information_basis
                for result in self.results
                if result.information_basis is not None
            )
        )

    @property
    def all_exact(self) -> bool:
        return all(result.exact for result in self.results)

    def render(self) -> str:
        floor = self.best_certified_floor
        floor_text = "none" if floor is None else f"{floor:.8g}"
        lines = [
            f"Families evaluated: {len(self.results)}",
            f"Certified families: {len(self.certified_results)}",
            f"Strongest certified floor: {floor_text}",
            f"Between-family range: {self.between_family_range:.8g}",
            "Interpretation: convergence is robustness evidence, not proof of coverage.",
        ]
        return "\n".join(lines)
