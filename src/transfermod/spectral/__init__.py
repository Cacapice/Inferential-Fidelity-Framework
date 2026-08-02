"""Spectral Fidelity Benchmark for Scientific Surrogate Models.

First benchmark: 2+1D compact U(1) lattice gauge theory (the mass-gap question).

Quickstart -- test any surrogate against a trusted exact reference in one line::

    from transfermod.spectral import certify, build_reference
    ref = build_reference(Lx=2, Ly=2, Lambda=2, g=1.0)
    verdict = certify(ref, my_surrogate)   # my_surrogate: a Surrogate subclass
    print(verdict.label)                   # CERTIFIED / CONFOUND / REJECTED

A surrogate is anything implementing ``sfbench.surrogates.base.Surrogate``:
either return a spectral triple ``(exp_val, deltas, weights)`` from ``spectral``,
or override ``correlator``/``exp_val``/``variance`` with measured quantities.
"""

from __future__ import annotations

__version__ = "1.1.0"

from .lattice.u1 import U1Lattice, U1Model
from .reference.exact import ExactReference, SpectralData
from .reference.krylov import KrylovReference
from .gate.tests import certify, run_gate, Verdict
from .gate.thresholds import Thresholds
from .surrogates.base import Surrogate, IdentitySurrogate
from . import taxonomy
from .taxonomy import CLASSES, CertificationClass, classify


def build_reference(Lx: int = 2, Ly: int = 2, Lambda: int = 2, g: float = 1.0,
                    method: str = "auto", basis_mode: str | None = None,
                    dense_dim_limit: int = 4000):
    """Build a finite-cutoff numerical reference for a 2+1D compact U(1) system.

    method:
        "auto"   -- dense exact diagonalisation for small physical dimension,
                    otherwise the sparse Krylov path. Dense diagonalisation returns the
                    selected-basis spectrum; Krylov returns low-energy and
                    operator-channel data.
        "dense"  -- force ``ExactReference`` (full eigh).
        "krylov" -- force ``KrylovReference`` (sparse eigsh + Lanczos).
    basis_mode:
        "full" (enumerate + Gauss filter) or "vacuum" (BFS flux sector). Defaults
        to "full" for the dense path and "vacuum" for the Krylov path.
    """
    if method not in ("auto", "dense", "krylov"):
        raise ValueError("method must be 'auto', 'dense', or 'krylov'")
    # The vacuum (BFS flux) sector reproduces the plaquette channel exactly and is
    # what keeps larger systems tractable, so it is the default basis here. (The
    # pilot uses the full basis explicitly for an all-sectors exactness check.)
    bmode = basis_mode or "vacuum"
    if method == "auto":
        probe = U1Model(U1Lattice(Lx, Ly, Lambda, basis_mode=bmode))
        method = "dense" if probe.dim <= dense_dim_limit else "krylov"
        model = probe
    else:
        model = U1Model(U1Lattice(Lx, Ly, Lambda, basis_mode=bmode))
    if method == "dense":
        return ExactReference(model, g=g)
    return KrylovReference(model, g=g)


__all__ = [
    "certify", "run_gate", "Verdict", "Thresholds", "build_reference",
    "U1Lattice", "U1Model", "ExactReference", "KrylovReference", "SpectralData",
    "Surrogate", "IdentitySurrogate", "__version__",
    "taxonomy", "CLASSES", "CertificationClass", "classify",
]
