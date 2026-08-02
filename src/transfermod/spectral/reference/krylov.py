"""Krylov reference: sparse ground state + Lanczos spectral function.

Same physics and same ``SpectralData`` output as ``ExactReference``, but reached
without dense diagonalisation, so it scales to larger volumes and truncations.

- Ground state ``|0>`` and the first gap in the selected basis block come from ``eigsh``.
- The connected channel spectral data ``{(Delta_n, w_n)}`` for the operator come
  from a Lanczos run seeded by ``v = O_c |0>``. The Lanczos Ritz values/weights
  are exactly the spectral representation of ``<v| f(H) |v>``; with full
  reorthogonalisation and enough steps they reproduce the dense result to machine
  precision (see ``tests/test_krylov.py``).

The vacuum basis is the plaquette-connected zero-winding block. Published
reference settings compare its ground state and plaquette channel with the full
basis; its first basis-spectrum gap is not an all-sector gap.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh

from ..lattice.u1 import U1Model
from .exact import SpectralData


def lanczos_spectral(H, v: np.ndarray, m: int = 200, tol: float = 1e-11):
    """Return (ritz_energies, weights) for the spectral function of ``H`` seeded
    by ``v``: ``<v| f(H) |v> = sum_i weights_i f(ritz_i)``, weights summing to
    ``||v||^2``. Full reorthogonalisation for accuracy on small/medium systems."""
    beta0 = float(np.linalg.norm(v))
    if beta0 < 1e-300:
        return np.array([0.0]), np.array([0.0])
    q = v / beta0
    Q = [q]
    alphas: list[float] = []
    betas: list[float] = []
    q_prev = np.zeros_like(q)
    b = 0.0
    m = min(m, H.shape[0])
    for k in range(m):
        w = H @ q - b * q_prev
        a = float(q @ w)
        w = w - a * q
        for qq in Q:  # full reorthogonalisation
            w = w - (qq @ w) * qq
        b = float(np.linalg.norm(w))
        alphas.append(a)
        if b < tol or k == m - 1:
            break
        betas.append(b)
        q_prev = q
        q = w / b
        Q.append(q)
    T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
    ev, evec = np.linalg.eigh(T)
    weights = (beta0 ** 2) * (evec[0, :] ** 2)
    return ev, weights


class KrylovReference:
    """Sparse/Krylov analogue of ``ExactReference`` with the same interface."""

    def __init__(self, model: U1Model, g: float, operator_sparse=None,
                 m: int = 220, wtol: float = 1e-12):
        self.model = model
        self.g = float(g)
        H = model.hamiltonian_sparse(g)
        k = min(2, H.shape[0] - 1)
        evals, evecs = eigsh(H, k=k, which="SA")
        order = np.argsort(evals)
        self.E0 = float(evals[order[0]])
        self.psi0 = evecs[:, order[0]]
        self._global_gap = float(evals[order[1]] - evals[order[0]]) if k > 1 else float("nan")
        O = model.cos_plaquette_operator_sparse() if operator_sparse is None else operator_sparse

        exp_val = float(self.psi0 @ (O @ self.psi0))
        v = O @ self.psi0 - exp_val * self.psi0
        ritz, w = lanczos_spectral(H, v, m=m)
        d = ritz - self.E0
        keep = (d > 1e-6) & (w > wtol)
        dd, ww = d[keep], w[keep]
        o = np.argsort(dd)
        dd, ww = dd[o], ww[o]
        channel_gap = float(dd[0]) if dd.size else float("nan")
        self._spectral = SpectralData(
            exp_val=exp_val, variance=float(ww.sum()), deltas=dd, weights=ww,
            sector_gap=self._global_gap, channel_gap=channel_gap,
            basis_mode=self.model.lat.basis_mode,
            spectrum_scope=("all_winding_sectors" if self.model.lat.basis_mode == "full" else "zero_winding_sector"),
        )

    # -- identical interface to ExactReference ------------------------------
    @property
    def spectral(self) -> SpectralData:
        return self._spectral

    def spectral_repr(self):
        s = self._spectral
        return s.exp_val, s.deltas.copy(), s.weights.copy()

    def correlator(self, ts: np.ndarray) -> np.ndarray:
        return self._spectral.correlator(ts)

    def exp_val(self) -> float:
        return self._spectral.exp_val

    def variance(self) -> float:
        return self._spectral.variance

    def plaquette_expectation(self) -> float:
        return self._spectral.exp_val

    def equal_time_variance(self) -> float:
        return self._spectral.variance

    def mass_gap(self) -> float:
        """Finite-volume, finite-cutoff operator-channel gap in the selected sector."""
        return self._spectral.channel_gap

    def asymptotic_gap(self) -> float:
        return self._spectral.channel_gap
