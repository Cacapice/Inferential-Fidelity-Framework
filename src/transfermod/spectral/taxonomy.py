"""Taxonomy of scientific surrogates: the four certification classes.

A spectral-fidelity benchmark is only useful if it is extensible, so the
approximations it judges are organised into four classes rather than an ad-hoc
list of examples. Each class fails spectral fidelity for a *different structural
reason*, which is what makes the taxonomy worth having:

===========  ==================================  ====================================
Class        Family                              Characteristic failure mode
===========  ==================================  ====================================
I            Numerical approximations            truncation / conditioning error
             (Krylov, sparse eigensolvers,
             exact diagonalisation)
II           Reduced representations             subspace truncation; Rayleigh--Ritz
             (POD, Galerkin, KL truncation,      bounds bias the extremal spectrum
             reduced transfer operators)
III          Learned surrogates                  optimisation targets an aggregate
             (NQS, neural operators, DeepONet,   loss that is blind to low-signal
             FNO, diffusion samplers)            modes; noise floors hide light states
IV           Adversarial controls                deliberate construction: matches every
             (spectral shortcut)                 conventional observable by design
===========  ==================================  ====================================

Class I is the trust anchor: members must agree with the exact reference to
machine precision (validated in ``tests/``), otherwise nothing downstream is
meaningful. Classes II and III are the objects of study. Class IV supplies the
positive control -- a benchmark that never fires on anything is untested, so a
construction that *must* be caught is required to demonstrate sensitivity.

``classify(surrogate)`` returns the class of a registered surrogate; new families
register themselves via ``register``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CertificationClass:
    numeral: str
    name: str
    description: str
    members: tuple = field(default_factory=tuple)

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"Class {self.numeral}: {self.name}"


CLASS_I = CertificationClass(
    "I", "Numerical approximations",
    "Controlled numerics judged against the exact reference; the trust anchor.",
    ("ExactReference", "KrylovReference", "IdentitySurrogate"),
)
CLASS_II = CertificationClass(
    "II", "Reduced representations",
    "Projection onto a truncated subspace (POD/Galerkin, KL, reduced operators).",
    ("PODGalerkinSurrogate", "ReducedTransferSurrogate"),
)
CLASS_III = CertificationClass(
    "III", "Learned surrogates",
    "Models fit by optimisation (neural quantum states, neural operators, "
    "DeepONets, FNOs, diffusion samplers, multi-exponential fits).",
    ("NeuralQuantumStateSurrogate", "TrainedMultiExpSurrogate"),
)
CLASS_IV = CertificationClass(
    "IV", "Adversarial controls",
    "Deliberately constructed to pass every conventional check; the positive "
    "control that proves the gate has teeth.",
    ("SpectralShortcutControl",),
)

CLASSES = (CLASS_I, CLASS_II, CLASS_III, CLASS_IV)

# name -> class registry
_REGISTRY: dict[str, CertificationClass] = {}
for _c in CLASSES:
    for _m in _c.members:
        _REGISTRY[_m] = _c


def register(type_name: str, cls: CertificationClass) -> None:
    """Register a new surrogate/reference type under a certification class."""
    _REGISTRY[type_name] = cls


def classify(obj) -> CertificationClass | None:
    """Return the certification class of a surrogate/reference instance or type."""
    name = obj if isinstance(obj, str) else type(obj).__name__
    return _REGISTRY.get(name)


def summary() -> str:
    """Human-readable taxonomy table."""
    lines = []
    for c in CLASSES:
        lines.append(f"Class {c.numeral:<4} {c.name}")
        lines.append(f"           {c.description}")
        lines.append(f"           members: {', '.join(c.members) or '(none yet)'}")
    return "\n".join(lines)
