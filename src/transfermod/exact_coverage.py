"""Exact-coverage theorems for common admissible families.

The results in this module are intentionally narrow. Each class encodes a
setting where the searched extremal family is mathematically exhaustive, so the
reported value is the full modulus rather than only a restricted lower bound.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from transfermod.certification import Coverage, modulus_result
from transfermod.modulus.discrepancy import CertificationGeometry


def _as_vector(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("expected a one-dimensional vector")
    return arr


def _certify_exact(model, epsilon: float, geometry: CertificationGeometry, *, family: str, scope: str):
    return modulus_result(
        model.exact_modulus(epsilon),
        coverage=Coverage.proven(model.theorem, scope),
        perturbation_family=family,
        geometry=geometry,
        epsilon=epsilon,
    )


@dataclass(frozen=True)
class HilbertLinearFunctionalCoverage:
    """Exact modulus for a continuous linear functional over an L2 ball.

    For ``Q(u+h)-Q(u)=<q,h>`` and ``||h||_2 <= epsilon``,

        omega(epsilon) = epsilon * ||q||_2.
    """

    representer: Sequence[float]
    theorem: str = "Hilbert-ball Riesz representation theorem"

    def exact_modulus(self, epsilon: float) -> float:
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        q = _as_vector(self.representer)
        return float(epsilon * np.linalg.norm(q))

    def extremal_direction(self) -> np.ndarray:
        q = _as_vector(self.representer)
        norm = float(np.linalg.norm(q))
        if norm == 0:
            return np.zeros_like(q)
        return q / norm

    def certify(self, epsilon: float, *, geometry: CertificationGeometry):
        return _certify_exact(
            self, epsilon, geometry,
            family="Riesz representer ray",
            scope="the Riesz direction attains the global L2-ball extremum",
        )


@dataclass(frozen=True)
class EllipsoidalLinearCoverage:
    """Exact modulus for a linear functional over ``h^T M h <= epsilon^2``.

    If the quantity representer has a component in ``ker(M)``, the admissible
    seminorm does not control the quantity and the full modulus is infinite.
    Otherwise,

        omega(epsilon) = epsilon * sqrt(q^T M^+ q).
    """

    matrix: Sequence[Sequence[float]]
    representer: Sequence[float]
    kernel_tolerance: float = 1e-10
    theorem: str = "Ellipsoidal dual-norm theorem"

    def _objects(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        M = np.asarray(self.matrix, dtype=float)
        q = _as_vector(self.representer)
        if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] != q.size:
            raise ValueError("matrix must be square and match representer size")
        if not np.allclose(M, M.T, atol=1e-10):
            raise ValueError("matrix must be symmetric")
        eigenvalues = np.linalg.eigvalsh(M)
        if np.min(eigenvalues) < -self.kernel_tolerance:
            raise ValueError("matrix must be positive semidefinite")
        return M, q, np.linalg.pinv(M, rcond=self.kernel_tolerance)

    def uncontrolled_kernel_component(self) -> float:
        M, q, pinv = self._objects()
        projector_kernel = np.eye(M.shape[0]) - pinv @ M
        return float(np.linalg.norm(projector_kernel @ q))

    def exact_modulus(self, epsilon: float) -> float:
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        M, q, pinv = self._objects()
        if self.uncontrolled_kernel_component() > self.kernel_tolerance:
            return math.inf
        return float(epsilon * math.sqrt(max(0.0, q @ pinv @ q)))

    def extremal_direction(self, epsilon: float) -> np.ndarray:
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        M, q, pinv = self._objects()
        if self.uncontrolled_kernel_component() > self.kernel_tolerance:
            raise ValueError("no finite extremizer: Q varies on the seminorm kernel")
        direction = pinv @ q
        dual_norm = math.sqrt(max(0.0, q @ pinv @ q))
        if dual_norm == 0:
            return np.zeros_like(q)
        return epsilon * direction / dual_norm

    def certify(self, epsilon: float, *, geometry: CertificationGeometry):
        value = self.exact_modulus(epsilon)
        scope = (
            "the modulus is infinite because Q varies on the seminorm kernel"
            if math.isinf(value)
            else "the ellipsoidal dual direction attains the global extremum"
        )
        return modulus_result(
            value,
            coverage=Coverage.proven(self.theorem, scope),
            perturbation_family="ellipsoidal dual direction",
            geometry=geometry,
            epsilon=epsilon,
        )


@dataclass(frozen=True)
class SketchNullspaceCoverage:
    """Exact sketch-conditioned corruption of a linear query.

    Under ``P h = 0`` and ``||h||_2 <= epsilon``,

        sup |<q,h>| = epsilon * ||Proj_ker(P) q||_2.
    """

    sketch_matrix: Sequence[Sequence[float]]
    query: Sequence[float]
    tolerance: float = 1e-10
    theorem: str = "Sketch-nullspace projection theorem"

    def nullspace_projection(self) -> np.ndarray:
        P = np.asarray(self.sketch_matrix, dtype=float)
        q = _as_vector(self.query)
        if P.ndim != 2 or P.shape[1] != q.size:
            raise ValueError("sketch matrix columns must match query dimension")
        _, singular_values, vh = np.linalg.svd(P, full_matrices=True)
        rank = int(np.sum(singular_values > self.tolerance))
        null_basis = vh[rank:].T
        if null_basis.size == 0:
            return np.zeros_like(q)
        return null_basis @ (null_basis.T @ q)

    def exact_modulus(self, epsilon: float) -> float:
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        return float(epsilon * np.linalg.norm(self.nullspace_projection()))

    def extremal_direction(self, epsilon: float) -> np.ndarray:
        projection = self.nullspace_projection()
        norm = float(np.linalg.norm(projection))
        if norm == 0:
            return np.zeros_like(projection)
        return epsilon * projection / norm

    def certify(self, epsilon: float, *, geometry: CertificationGeometry):
        return _certify_exact(
            self, epsilon, geometry,
            family="query projection onto sketch nullspace",
            scope="the projected query direction attains the sketch-nullspace extremum",
        )


@dataclass(frozen=True)
class LeadingEigenvalueFrobeniusCoverage:
    """Exact leading-eigenvalue shift under unrestricted symmetric Frobenius error.

    For symmetric ``A`` and symmetric ``Delta`` with ``||Delta||_F <= epsilon``,

        sup [lambda_max(A + Delta) - lambda_max(A)] = epsilon,

    attained by ``Delta = epsilon v v^T`` for a leading eigenvector ``v``.
    """

    matrix: Sequence[Sequence[float]]
    theorem: str = "Variational leading-eigenvalue/Frobenius theorem"

    def _matrix(self) -> np.ndarray:
        A = np.asarray(self.matrix, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("matrix must be square")
        if not np.allclose(A, A.T, atol=1e-10):
            raise ValueError("matrix must be symmetric")
        return A

    def exact_modulus(self, epsilon: float) -> float:
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        return float(epsilon)

    def extremal_perturbation(self, epsilon: float) -> np.ndarray:
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        A = self._matrix()
        _, vectors = np.linalg.eigh(A)
        v = vectors[:, -1]
        return epsilon * np.outer(v, v)

    def certify(self, epsilon: float, *, geometry: CertificationGeometry):
        return _certify_exact(
            self, epsilon, geometry,
            family="leading-eigenvector rank-one perturbation",
            scope="rank one is exhaustive for the symmetric Frobenius ball",
        )
