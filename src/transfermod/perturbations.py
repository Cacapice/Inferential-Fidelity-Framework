"""Reusable perturbation-family templates.

The fidelity modulus is only as informative as the alternatives searched.
These templates make the construction of ``u_w`` explicit and reusable while
leaving domain-specific admissibility and quantity-of-interest functions in
user code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Protocol, Sequence, TypeVar, runtime_checkable

StateT = TypeVar("StateT")
DirectionT = TypeVar("DirectionT")

Metric = Callable[[StateT, StateT], float]


def _validate_weight(bounds: tuple[float, float], weight: float) -> None:
    lower, upper = bounds
    if not lower <= weight <= upper:
        raise ValueError(f"weight {weight} outside [{lower}, {upper}]")


@runtime_checkable
class PerturbationFamily(Protocol[StateT]):
    """One-parameter family ``u_w`` used for restricted-modulus searches."""

    name: str
    information_basis: str | None

    def perturb(self, reference: StateT, weight: float) -> StateT:
        """Construct the alternative state at amplitude ``weight``."""

    def admissibility_error(self, reference: StateT, alternative: StateT) -> float:
        """Evaluate the declared validation geometry."""

    def parameter_bounds(self) -> tuple[float, float]:
        """Supported closed interval of perturbation amplitudes."""


def _default_add(reference, increment):
    # Python lists and tuples use ``+`` for concatenation, so handle ordinary
    # sequences explicitly before delegating to array/scalar arithmetic.
    if isinstance(reference, list) and isinstance(increment, (list, tuple)):
        return [a + b for a, b in zip(reference, increment)]
    if isinstance(reference, tuple) and isinstance(increment, (list, tuple)):
        return tuple(a + b for a, b in zip(reference, increment))
    try:
        return reference + increment
    except TypeError:
        return type(reference)(a + b for a, b in zip(reference, increment))


def _default_scale(direction, weight):
    try:
        return weight * direction
    except TypeError:
        return type(direction)(weight * x for x in direction)


@dataclass(frozen=True)
class AdditivePerturbation(Generic[StateT, DirectionT]):
    """Template for ``u_w = u + w v``."""

    direction: DirectionT
    validation_metric: Metric
    bounds: tuple[float, float] = (0.0, 1.0)
    combine: Callable[[StateT, object], StateT] = _default_add
    scale: Callable[[DirectionT, float], object] = _default_scale
    name: str = "AdditivePerturbation"
    information_basis: str | None = None

    def perturb(self, reference: StateT, weight: float) -> StateT:
        self._check_weight(weight)
        return self.combine(reference, self.scale(self.direction, weight))

    def admissibility_error(self, reference: StateT, alternative: StateT) -> float:
        return float(self.validation_metric(reference, alternative))

    def parameter_bounds(self) -> tuple[float, float]:
        return self.bounds

    def _check_weight(self, weight: float) -> None:
        _validate_weight(self.bounds, weight)


@dataclass(frozen=True)
class ParametricResidualPerturbation(Generic[StateT]):
    """Template for ``u_w = combine(u, residual_generator(w))``."""

    residual_generator: Callable[[float], object]
    validation_metric: Metric
    combine: Callable[[StateT, object], StateT] = _default_add
    bounds: tuple[float, float] = (0.0, 1.0)
    name: str = "ParametricResidualPerturbation"
    information_basis: str | None = None

    def perturb(self, reference: StateT, weight: float) -> StateT:
        _validate_weight(self.bounds, weight)
        return self.combine(reference, self.residual_generator(weight))

    def admissibility_error(self, reference: StateT, alternative: StateT) -> float:
        return float(self.validation_metric(reference, alternative))

    def parameter_bounds(self) -> tuple[float, float]:
        return self.bounds


@dataclass(frozen=True)
class SpectralModeInjection:
    """Add a faint exponential mode to a sampled correlator.

    ``reference`` is a sequence evaluated at ``times`` and the injected mode is
    ``weight * mode_shape(t) * exp(-mode_rate*t)``.
    """

    times: Sequence[float]
    mode_rate: float
    validation_metric: Callable[[Sequence[float], Sequence[float]], float]
    mode_shape: Callable[[float], float] = lambda _t: 1.0
    bounds: tuple[float, float] = (0.0, 1.0)
    name: str = "SpectralModeInjection"
    information_basis: str | None = "long_distance_correlator_tail"

    def perturb(self, reference: Sequence[float], weight: float) -> list[float]:
        _validate_weight(self.bounds, weight)
        if len(reference) != len(self.times):
            raise ValueError("reference and times must have equal length")
        import math
        return [
            float(value) + weight * self.mode_shape(float(t)) *
            math.exp(-self.mode_rate * float(t))
            for value, t in zip(reference, self.times)
        ]

    def admissibility_error(
        self, reference: Sequence[float], alternative: Sequence[float]
    ) -> float:
        return float(self.validation_metric(reference, alternative))

    def parameter_bounds(self) -> tuple[float, float]:
        return self.bounds


@dataclass(frozen=True)
class GraphEdgeWeightPerturbation:
    """Reweight selected graph edges and optionally redistribute removed weight."""

    weakened_edges: Sequence[tuple[int, int]]
    redistribution_edges: Sequence[tuple[int, int]] = ()
    bounds: tuple[float, float] = (0.0, 1.0)
    name: str = "GraphEdgeWeightPerturbation"
    information_basis: str | None = "graph_laplacian"

    def perturb(self, reference, weight: float):
        import numpy as np
        _validate_weight(self.bounds, weight)
        matrix = np.asarray(reference, dtype=float).copy()
        removed = 0.0
        for i, j in self.weakened_edges:
            delta = weight * matrix[i, j]
            matrix[i, j] -= delta
            matrix[j, i] -= delta
            removed += delta
        if self.redistribution_edges:
            per_edge = removed / len(self.redistribution_edges)
            for i, j in self.redistribution_edges:
                matrix[i, j] += per_edge
                matrix[j, i] += per_edge
        return matrix

    def admissibility_error(self, reference, alternative) -> float:
        import numpy as np
        denominator = float(np.linalg.norm(reference, "fro"))
        if denominator == 0:
            raise ValueError("reference Frobenius norm must be positive")
        return float(np.linalg.norm(alternative - reference, "fro") / denominator)

    def parameter_bounds(self) -> tuple[float, float]:
        return self.bounds
