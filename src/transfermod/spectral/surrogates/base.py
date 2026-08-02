"""Surrogate interface: the contract every scientific surrogate implements.

The benchmark evaluates **outputs, not architectures**. Anything that can report
the benchmark's observables can be certified, whether it is a neural network, a
projection-based reduction, a controlled numerical method, or an adversarial
construction. That is what makes the gate portable across surrogate families and,
in principle, across scientific domains.

Required contract (three methods)
---------------------------------
``correlator(ts)``  the connected two-point function on the frozen time grid.
``variance()``      the equal-time connected variance ``C(0)``.
``metadata()``      a ``SurrogateMetadata`` record describing what this model is.

Surrogates that possess an explicit spectral decomposition may implement
``spectral() -> (exp_val, deltas, weights)`` instead and inherit ``correlator``,
``variance``, ``exp_val`` and ``asymptotic_gap`` for free. Surrogates that only
produce measured data override the required three (plus ``exp_val``) directly.

Optional, recommended for reproducibility
-----------------------------------------
``provenance()``      how this surrogate was produced (method, solver, source).
``training_info()``   optimiser / capacity / data details for fitted models.
``surrogate_family()`` certification class (see ``sfbench.taxonomy``).

``metadata()`` has a working default that assembles these, so existing surrogates
keep working; override the pieces that apply to your model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SurrogateMetadata:
    """Structured description of a surrogate, for reproducible benchmark records."""

    name: str
    family: str = "unspecified"           # certification class, e.g. "Class III: Learned surrogates"
    provenance: dict = field(default_factory=dict)
    training_info: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"name": self.name, "family": self.family,
                "provenance": dict(self.provenance),
                "training_info": dict(self.training_info)}


class Surrogate(ABC):
    name: str = "surrogate"

    # ---- required contract -------------------------------------------------
    @abstractmethod
    def spectral(self):
        """Return (exp_val: float, deltas: np.ndarray, weights: np.ndarray).

        Surrogates without an explicit spectral decomposition should instead
        override ``correlator``, ``variance`` and ``exp_val`` with measured
        quantities and may raise ``NotImplementedError`` here.
        """

    def correlator(self, ts: np.ndarray) -> np.ndarray:
        _, d, w = self.spectral()
        ts = np.asarray(ts, dtype=float)
        return (w[:, None] * np.exp(-d[:, None] * ts[None, :])).sum(0)

    def variance(self) -> float:
        return float(self.spectral()[2].sum())

    def metadata(self) -> SurrogateMetadata:
        """Structured description of this surrogate. Override pieces as needed."""
        return SurrogateMetadata(
            name=self.name,
            family=self.surrogate_family(),
            provenance=self.provenance(),
            training_info=self.training_info(),
        )

    # ---- optional, recommended --------------------------------------------
    def surrogate_family(self) -> str:
        """Certification class of this surrogate (see ``sfbench.taxonomy``)."""
        from ..taxonomy import classify
        cls = classify(self)
        return str(cls) if cls is not None else "unspecified"

    def provenance(self) -> dict:
        """How this surrogate was produced (method, solver, source data)."""
        return {}

    def training_info(self) -> dict:
        """Optimiser / capacity / data details for fitted models."""
        return {}

    # ---- derived -----------------------------------------------------------
    def spectral_repr(self):
        """Common contract shared with the exact reference."""
        return self.spectral()

    def exp_val(self) -> float:
        return self.spectral()[0]

    def asymptotic_gap(self) -> float:
        """Smallest decay rate carrying non-negligible weight -> the reported gap."""
        _, d, w = self.spectral()
        sig = w > 1e-12
        return float(d[sig].min())


class IdentitySurrogate(Surrogate):
    """Echoes the reference exactly. Calibration control: the gate MUST pass it."""

    name = "identity(echo)"

    def __init__(self, spectral_data):
        self._exp = spectral_data.exp_val
        self._d = np.asarray(spectral_data.deltas, dtype=float)
        self._w = np.asarray(spectral_data.weights, dtype=float)

    def spectral(self):
        return self._exp, self._d.copy(), self._w.copy()
