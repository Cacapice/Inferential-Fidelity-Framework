"""Coupling audit on a trained neural operator for one-dimensional diffusion.

The model is a deterministic random-feature neural operator trained on smooth
initial conditions. The localized challenge family is evaluated without
assuming that aggregate and decision errors decouple. The primary empirical
question is whether any family member occupies a predeclared
small-aggregate/large-decision region.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time

import numpy as np

from transfermod.certification import Coverage, RestrictedModulusResult, modulus_result
from transfermod.modulus import CertificationGeometry


class CouplingConclusion(str, Enum):
    """Empirical conclusion for a fixed trained model and challenge family."""

    DECOUPLED = "decoupled"
    COUPLED_NEGATIVE = "coupled_negative"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class LearnedOperatorFamilyPoint:
    """One evaluated member of the localized challenge family."""

    center: float
    width: float
    amplitude: float
    absolute_global_l2_error: float
    relative_global_l2_error: float
    decision_error: float

    @property
    def decision_to_absolute_global_ratio(self) -> float:
        if self.absolute_global_l2_error == 0.0:
            return float("inf") if self.decision_error > 0.0 else 0.0
        return self.decision_error / self.absolute_global_l2_error

    def to_dict(self) -> dict[str, float]:
        return {
            "center": self.center,
            "width": self.width,
            "amplitude": self.amplitude,
            "absolute_global_l2_error": self.absolute_global_l2_error,
            "relative_global_l2_error": self.relative_global_l2_error,
            "decision_error": self.decision_error,
            "decision_to_absolute_global_ratio": (
                self.decision_to_absolute_global_ratio
            ),
        }


@dataclass(frozen=True)
class LearnedOperatorValidation:
    validation_mean_relative_l2: float
    validation_p95_relative_l2: float
    aggregate_error_threshold: float
    decision_error_threshold: float
    validation_passed: bool
    trained_examples: int
    family_evaluations: int
    conclusion: CouplingConclusion
    hidden_failure_count: int
    pearson_error_correlation: float
    spearman_error_correlation: float
    ratio_min: float
    ratio_median: float
    ratio_max: float
    smallest_aggregate_point: LearnedOperatorFamilyPoint
    strongest_decision_point: LearnedOperatorFamilyPoint
    family_points: tuple[LearnedOperatorFamilyPoint, ...]
    runtime_seconds: float
    result: RestrictedModulusResult

    @property
    def validation_threshold(self) -> float:
        """Backward-compatible alias for ``aggregate_error_threshold``."""
        return self.aggregate_error_threshold

    @property
    def strongest_q_error(self) -> float:
        """Backward-compatible strongest decision error."""
        return self.strongest_decision_point.decision_error

    @property
    def strongest_global_l2_error(self) -> float:
        """Backward-compatible absolute global error at strongest decision point."""
        return self.strongest_decision_point.absolute_global_l2_error

    def to_dict(self) -> dict[str, object]:
        return {
            "validation_mean_relative_l2": self.validation_mean_relative_l2,
            "validation_p95_relative_l2": self.validation_p95_relative_l2,
            "aggregate_error_threshold": self.aggregate_error_threshold,
            "decision_error_threshold": self.decision_error_threshold,
            "validation_passed": self.validation_passed,
            "trained_examples": self.trained_examples,
            "family_evaluations": self.family_evaluations,
            "conclusion": self.conclusion.value,
            "hidden_failure_count": self.hidden_failure_count,
            "pearson_error_correlation": self.pearson_error_correlation,
            "spearman_error_correlation": self.spearman_error_correlation,
            "ratio_min": self.ratio_min,
            "ratio_median": self.ratio_median,
            "ratio_max": self.ratio_max,
            "smallest_aggregate_point": self.smallest_aggregate_point.to_dict(),
            "strongest_decision_point": self.strongest_decision_point.to_dict(),
            "family_points": [point.to_dict() for point in self.family_points],
            "runtime_seconds": self.runtime_seconds,
            "result": self.result.to_dict(),
        }


class RandomFeatureDiffusionOperator:
    """One-hidden-layer random-feature operator with a fitted linear readout."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        ridge: float = 1e-6,
        seed: int = 7,
    ) -> None:
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(scale=0.02, size=(hidden_dim, input_dim))
        self.ridge = float(ridge)
        self.readout: np.ndarray | None = None

    def _features(self, inputs: np.ndarray) -> np.ndarray:
        return np.tanh(np.asarray(inputs, dtype=float) @ self.weights.T)

    def fit(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        X = np.asarray(inputs, dtype=float)
        Y = np.asarray(outputs, dtype=float)
        if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
            raise ValueError("inputs and outputs must be aligned two-dimensional arrays")
        H = self._features(X)
        gram = H.T @ H + self.ridge * np.eye(H.shape[1])
        self.readout = np.linalg.solve(gram, H.T @ Y)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if self.readout is None:
            raise RuntimeError("operator must be fitted before prediction")
        return self._features(np.asarray(inputs, dtype=float)) @ self.readout


def _heat_operator(
    n_grid: int,
    *,
    diffusivity: float,
    time_horizon: float,
) -> tuple[np.ndarray, np.ndarray]:
    if n_grid < 8:
        raise ValueError("n_grid must be at least 8")
    x = np.linspace(0.0, 1.0, n_grid + 2)[1:-1]
    h = 1.0 / (n_grid + 1)
    laplacian = (
        np.diag(-2.0 * np.ones(n_grid))
        + np.diag(np.ones(n_grid - 1), 1)
        + np.diag(np.ones(n_grid - 1), -1)
    ) / (h * h)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    propagator = eigenvectors @ np.diag(
        np.exp(diffusivity * time_horizon * eigenvalues)
    ) @ eigenvectors.T
    return x, propagator


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks, including ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def run_learned_operator_validation(
    *,
    seed: int = 7,
    n_grid: int = 48,
    n_train: int = 500,
    n_validation: int = 160,
    validation_threshold: float = 0.01,
    decision_error_threshold: float = 0.02,
) -> LearnedOperatorValidation:
    """Train and audit a neural diffusion operator for error decoupling.

    Training and ordinary validation use smooth six-mode sine fields. The
    challenge family consists of narrow localized inputs outside that training
    distribution.

    The confirmatory endpoint is fixed before the family is evaluated:

    ``relative_global_l2_error <= validation_threshold`` and
    ``decision_error >= decision_error_threshold``.

    A family point in that quadrant supports decoupling for this model/family.
    No such point is an explicit negative result; it must not be reframed as a
    hidden validation failure.
    """
    if validation_threshold <= 0.0 or decision_error_threshold <= 0.0:
        raise ValueError("error thresholds must be positive")

    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    x, propagator = _heat_operator(
        n_grid, diffusivity=0.01, time_horizon=0.02
    )
    smooth_basis = np.stack(
        [np.sin(np.pi * k * x) for k in range(1, 7)], axis=1
    )
    coefficient_scale = np.linspace(1.0, 0.3, smooth_basis.shape[1])

    def sample_smooth(count: int) -> np.ndarray:
        coefficients = rng.normal(size=(count, smooth_basis.shape[1]))
        coefficients *= coefficient_scale
        return coefficients @ smooth_basis.T

    train_inputs = sample_smooth(n_train)
    train_outputs = train_inputs @ propagator.T
    model = RandomFeatureDiffusionOperator(
        n_grid, hidden_dim=128, ridge=1e-6, seed=seed + 1
    )
    model.fit(train_inputs, train_outputs)

    validation_inputs = sample_smooth(n_validation)
    validation_truth = validation_inputs @ propagator.T
    validation_prediction = model.predict(validation_inputs)
    relative_l2 = np.linalg.norm(
        validation_prediction - validation_truth, axis=1
    ) / np.maximum(np.linalg.norm(validation_truth, axis=1), 1e-12)
    mean_error = float(np.mean(relative_l2))
    p95_error = float(np.quantile(relative_l2, 0.95))
    validation_passed = p95_error <= validation_threshold

    points: list[LearnedOperatorFamilyPoint] = []
    for center in (0.35, 0.50, 0.65):
        for width in (0.010, 0.015, 0.020, 0.030, 0.050):
            bump = np.exp(-0.5 * ((x - center) / width) ** 2)
            bump /= np.linalg.norm(bump)
            for amplitude in (0.10, 0.15, 0.20):
                initial = amplitude * bump
                truth = propagator @ initial
                prediction = model.predict(initial[None, :])[0]
                absolute_global = float(np.linalg.norm(prediction - truth))
                relative_global = absolute_global / max(
                    float(np.linalg.norm(truth)), 1e-12
                )
                q_index = int(np.argmax(truth))
                decision_error = float(
                    abs(prediction[q_index] - truth[q_index])
                )
                points.append(
                    LearnedOperatorFamilyPoint(
                        center=float(center),
                        width=float(width),
                        amplitude=float(amplitude),
                        absolute_global_l2_error=absolute_global,
                        relative_global_l2_error=relative_global,
                        decision_error=decision_error,
                    )
                )

    aggregate = np.asarray(
        [point.relative_global_l2_error for point in points], dtype=float
    )
    decision = np.asarray([point.decision_error for point in points], dtype=float)
    ratios = np.asarray(
        [point.decision_to_absolute_global_ratio for point in points],
        dtype=float,
    )
    hidden = [
        point
        for point in points
        if point.relative_global_l2_error <= validation_threshold
        and point.decision_error >= decision_error_threshold
    ]

    pearson = _correlation(aggregate, decision)
    spearman = _correlation(_rankdata(aggregate), _rankdata(decision))
    if hidden:
        conclusion = CouplingConclusion.DECOUPLED
    else:
        # The confirmatory claim is existential: did the declared family reach
        # the small-aggregate/large-decision region? If not, the result is
        # negative for this model/family, irrespective of whether a single
        # correlation coefficient summarizes the full profile well.
        conclusion = CouplingConclusion.COUPLED_NEGATIVE

    smallest_aggregate = min(
        points, key=lambda point: point.relative_global_l2_error
    )
    strongest_decision = max(points, key=lambda point: point.decision_error)

    # This remains a valid family-restricted decision-error maximum, but it is
    # not described as Silent Risk because the same point has conspicuous
    # aggregate error.
    geometry = CertificationGeometry.from_standard_metric(
        metric="MAE",
        reference_q=0.0,
        tolerance=decision_error_threshold,
        reference_type="exact_diffusion_operator",
    )
    result = modulus_result(
        strongest_decision.decision_error,
        coverage=Coverage.certified_floor(
            "localized Gaussian initial-condition family",
            "the family maximum is exhaustive over the declared grid, but "
            "does not cover all trained-model failures",
        ),
        perturbation_family="localized Gaussian initial-condition family",
        geometry=geometry,
        epsilon=max(point.relative_global_l2_error for point in points),
        information_basis="trained_operator_localized_challenge",
        notes=(
            f"Coupling audit conclusion: {conclusion.value}.",
            "No hidden validation failure is claimed unless a point enters the "
            "predeclared small-aggregate/large-decision quadrant.",
        ),
    )
    return LearnedOperatorValidation(
        validation_mean_relative_l2=mean_error,
        validation_p95_relative_l2=p95_error,
        aggregate_error_threshold=validation_threshold,
        decision_error_threshold=decision_error_threshold,
        validation_passed=validation_passed,
        trained_examples=n_train,
        family_evaluations=len(points),
        conclusion=conclusion,
        hidden_failure_count=len(hidden),
        pearson_error_correlation=pearson,
        spearman_error_correlation=spearman,
        ratio_min=float(np.min(ratios)),
        ratio_median=float(np.median(ratios)),
        ratio_max=float(np.max(ratios)),
        smallest_aggregate_point=smallest_aggregate,
        strongest_decision_point=strongest_decision,
        family_points=tuple(points),
        runtime_seconds=float(time.perf_counter() - started),
        result=result,
    )
