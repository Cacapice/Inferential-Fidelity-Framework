"""transfermod — transfer guarantees and fidelity moduli.

The mathematical core is domain-independent. Given an admissible family, a
decision functional Q, and a declared discrepancy geometry ell on the decision
space, the fidelity modulus

    omega_ell(eps) = sup{ ell(Q(u'), Q(u)) : ||u' - u||_A <= eps }

is the pointwise-minimal transfer guarantee at u. Relative error is one special
case, not a universal normalization. A certification procedure induces a
restricted modulus; Silent Risk is the measure of admitted alternatives whose
decision discrepancy exceeds a declared tolerance.

Layout:

    transfermod.modulus         fidelity moduli and discrepancy geometry
    transfermod.certification   coverage tier and result provenance
    transfermod.exact_coverage exact theorem implementations
    transfermod.perturbations   reusable adversarial-family templates
    transfermod.pipeline        composition diagnostics
    transfermod.applications    reference-application facades
    transfermod.validation      executable empirical validations
    transfermod.compat          deprecated v1.x facade (removed in v2.0)

The former top-level name ``sfbench`` remains importable as a deprecated alias
for ``transfermod.spectral``.
"""

from transfermod import certification, modulus, spectral  # noqa: F401
from transfermod import perturbations, pipeline  # noqa: F401
from transfermod import exact_coverage, family_profile, applications, validation  # noqa: F401

__all__ = ["modulus", "certification", "spectral", "perturbations", "pipeline", "exact_coverage", "family_profile", "applications", "validation"]
__version__ = "1.2.2"
