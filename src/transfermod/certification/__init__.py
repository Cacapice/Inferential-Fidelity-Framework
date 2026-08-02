"""Domain-independent certification vocabulary and result types."""

from __future__ import annotations

CERTIFIED = "CERTIFIED"
CONFOUND = "CONFOUND"
REJECTED = "REJECTED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
VERDICTS = (CERTIFIED, CONFOUND, REJECTED, INSUFFICIENT_EVIDENCE)
EXPLANATION = {
    CERTIFIED: "conventional pass, spectral pass",
    CONFOUND: "conventional pass, spectral FAIL",
    REJECTED: "conventional FAIL",
    INSUFFICIENT_EVIDENCE: "conventional pass, spectral tests unmeasured",
}

from transfermod.certification.coverage import (
    Coverage,
    CoverageTier,
    RestrictedModulusResult,
    modulus_result,
    require_exact,
    strict_publish,
)

__all__ = [
    "CERTIFIED",
    "CONFOUND",
    "REJECTED",
    "INSUFFICIENT_EVIDENCE",
    "VERDICTS",
    "EXPLANATION",
    "Coverage",
    "CoverageTier",
    "RestrictedModulusResult",
    "modulus_result",
    "require_exact",
    "strict_publish",
]

_DEPRECATED_EXPORTS = {
    "BoundStatus",
    "CoverageProof",
    "ProvenCoverage",
    "UnprovenCoverage",
    "ExploratorySampleCoverage",
    "restricted_modulus_result",
    "exploratory_modulus_result",
}


def __getattr__(name: str):
    """Resolve deprecated v1.x exports lazily through :mod:`transfermod.compat`.

    Constructor/function facades warn when called. Type-like facades warn when
    accessed because access itself is the deprecated operation.
    """
    if name not in _DEPRECATED_EXPORTS:
        raise AttributeError(name)
    from transfermod import compat
    if name in {"BoundStatus", "CoverageProof"}:
        compat.warn_deprecated_access(name)
    return getattr(compat, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | _DEPRECATED_EXPORTS)
