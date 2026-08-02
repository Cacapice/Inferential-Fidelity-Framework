"""Computational-scaling validation for localized 2D and 3D PDE fields."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import time

import numpy as np


@dataclass(frozen=True)
class DimensionScalingResult:
    dimension: int
    spatial_points: int
    uniform_evaluations: int
    adaptive_evaluations: int
    uniform_runtime_seconds: float
    adaptive_runtime_seconds: float
    uniform_best_q_error: float
    adaptive_best_q_error: float
    adaptive_fraction_of_uniform: float

    def to_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class PDEScalingValidation:
    results: tuple[DimensionScalingResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {"results": [result.to_dict() for result in self.results]}


def _coordinates(dimension: int, n_grid: int) -> tuple[np.ndarray, ...]:
    axis = np.linspace(0.0, 1.0, n_grid)
    return tuple(np.meshgrid(*([axis] * dimension), indexing="ij"))


def _low_pass(field: np.ndarray, cutoff: int) -> np.ndarray:
    spectrum = np.fft.fftn(field)
    mask = np.ones(field.shape, dtype=bool)
    for axis, size in enumerate(field.shape):
        frequencies = np.fft.fftfreq(size) * size
        shape = [1] * field.ndim
        shape[axis] = size
        mask &= np.abs(frequencies).reshape(shape) <= cutoff
    return np.fft.ifftn(spectrum * mask).real


def _evaluate(
    mesh: tuple[np.ndarray, ...],
    center: tuple[float, ...],
    *,
    width: float,
    amplitude: float,
    cutoff: int,
) -> tuple[float, float]:
    radius_squared = sum(
        (coordinate - coordinate_center) ** 2
        for coordinate, coordinate_center in zip(mesh, center)
    )
    exact = amplitude * np.exp(-radius_squared / (2.0 * width * width))
    surrogate = _low_pass(exact, cutoff)
    global_l2 = float(np.sqrt(np.mean((exact - surrogate) ** 2)))
    q_error = float(abs(np.max(exact) - np.max(surrogate)))
    return global_l2, q_error


def _search(
    mesh: tuple[np.ndarray, ...],
    centers: list[tuple[float, ...]],
    *,
    admissible_l2: float,
) -> tuple[int, float]:
    best = 0.0
    evaluations = 0
    for center in centers:
        for width in (0.045, 0.060):
            evaluations += 1
            global_l2, q_error = _evaluate(
                mesh, center, width=width, amplitude=0.03, cutoff=3
            )
            if global_l2 <= admissible_l2:
                best = max(best, q_error)
    return evaluations, best


def _adaptive_centers(
    dimension: int,
    coarse_best: tuple[float, ...],
) -> list[tuple[float, ...]]:
    offsets = (-0.15, 0.0, 0.15)
    centers = []
    for delta in product(offsets, repeat=dimension):
        centers.append(
            tuple(
                float(np.clip(value + shift, 0.1, 0.9))
                for value, shift in zip(coarse_best, delta)
            )
        )
    return list(dict.fromkeys(centers))


def run_pde_scaling_validation(
    *,
    configurations: tuple[tuple[int, int], ...] = ((2, 32), (3, 18)),
    admissible_l2: float = 0.002,
) -> PDEScalingValidation:
    """Compare uniform and coarse-to-fine searches in 2D and 3D.

    The field is a short-time localized diffusion response; the surrogate is a
    low-frequency spectral truncation. The returned values are empirical
    scaling measurements for the grid-search layer of Theorem 5, not a proof of
    asymptotic tractability.
    """
    records: list[DimensionScalingResult] = []
    for dimension, n_grid in configurations:
        mesh = _coordinates(dimension, n_grid)

        uniform_axis = np.linspace(0.2, 0.8, 5)
        uniform_centers = list(product(uniform_axis, repeat=dimension))
        started = time.perf_counter()
        uniform_evaluations, uniform_best = _search(
            mesh, uniform_centers, admissible_l2=admissible_l2
        )
        uniform_runtime = time.perf_counter() - started

        coarse_axis = np.linspace(0.2, 0.8, 3)
        coarse_centers = list(product(coarse_axis, repeat=dimension))
        started = time.perf_counter()
        _, coarse_best_value = _search(
            mesh, coarse_centers, admissible_l2=admissible_l2
        )
        # Select a deterministic best coarse center by direct evaluation.
        scored = []
        for center in coarse_centers:
            _, q_error = _evaluate(
                mesh, center, width=0.045, amplitude=0.03, cutoff=3
            )
            scored.append((q_error, center))
        best_center = max(scored, key=lambda item: item[0])[1]
        refined_centers = _adaptive_centers(dimension, best_center)
        refined_evaluations, refined_best = _search(
            mesh, refined_centers, admissible_l2=admissible_l2
        )
        adaptive_runtime = time.perf_counter() - started
        adaptive_evaluations = 2 * len(coarse_centers) + refined_evaluations
        adaptive_best = max(coarse_best_value, refined_best)

        records.append(
            DimensionScalingResult(
                dimension=dimension,
                spatial_points=n_grid**dimension,
                uniform_evaluations=uniform_evaluations,
                adaptive_evaluations=adaptive_evaluations,
                uniform_runtime_seconds=float(uniform_runtime),
                adaptive_runtime_seconds=float(adaptive_runtime),
                uniform_best_q_error=float(uniform_best),
                adaptive_best_q_error=float(adaptive_best),
                adaptive_fraction_of_uniform=(
                    float(adaptive_best / uniform_best)
                    if uniform_best > 0
                    else 1.0
                ),
            )
        )
    return PDEScalingValidation(tuple(records))
