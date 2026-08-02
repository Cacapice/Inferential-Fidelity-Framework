"""POD/Galerkin surrogate: proper-orthogonal-decomposition reduced-order model.

A second *learned* surrogate family, structurally unrelated to the neural
quantum state, included to show the gate evaluates scientific fidelity rather
than one model class.

This is the standard reduced-basis / POD-Galerkin construction used throughout
model-order reduction and scientific ML: build a low-dimensional subspace from
snapshot data, project the operator onto it, and evolve the reduced system. Here
the snapshots are the Krylov vectors ``{O_c|0>, H O_c|0>, H^2 O_c|0>, ...}`` --
i.e. the data an emulator actually sees when it observes the correlator's
short-time behaviour -- orthonormalised and truncated at rank ``r``. The reduced
operator ``H_r = V^T H V`` then supplies the surrogate's spectrum.

The rank ``r`` is the model's capacity, and it controls spectral fidelity in a
transparent way:

- Large ``r``: the subspace captures the low modes, the reduced spectrum matches,
  and the surrogate is CERTIFIED.
- Small ``r``: the truncation discards low-weight tail structure. Because a
  Krylov subspace converges to the *dominant-weight* modes first, an aggressive
  truncation preferentially keeps the high-overlap (heavier) modes -- the same
  bias that makes the weak light state hard to learn -- and the reported gap
  drifts. Whether that drift is caught conventionally or only spectrally depends
  on how much equal-time variance the truncation also loses.

Unlike ``SpectralShortcutControl`` this is not adversarial: nothing is
fabricated. It is an honest reduction whose fidelity is an emergent property of
its capacity, exactly as for a trained network.
"""

from __future__ import annotations

import numpy as np

from ..lattice.u1 import U1Model
from .base import Surrogate


class PODGalerkinSurrogate(Surrogate):
    name = "pod-galerkin(reduced)"

    def __init__(self, model: U1Model, g: float, rank: int = 8,
                 operator_sparse=None, reorth: bool = True):
        self.model = model
        self.g = float(g)
        self.rank = int(rank)
        H = model.hamiltonian_sparse(g)
        O = model.cos_plaquette_operator_sparse() if operator_sparse is None else operator_sparse

        # ground state (the "training data" generator)
        from scipy.sparse.linalg import eigsh
        evals, evecs = eigsh(H, k=1, which="SA")
        psi0 = evecs[:, 0]
        E0 = float(evals[0])

        exp_val = float(psi0 @ (O @ psi0))
        v0 = O @ psi0 - exp_val * psi0
        norm0 = float(np.linalg.norm(v0))

        # snapshot / Krylov basis, orthonormalised (POD-style reduced basis)
        r = max(1, min(self.rank, H.shape[0] - 1))
        V = np.zeros((H.shape[0], r))
        q = v0 / norm0
        V[:, 0] = q
        for j in range(1, r):
            w = H @ V[:, j - 1]
            if reorth:
                for k in range(j):  # modified Gram-Schmidt
                    w = w - (V[:, k] @ w) * V[:, k]
                for k in range(j):  # second pass for numerical stability
                    w = w - (V[:, k] @ w) * V[:, k]
            nw = np.linalg.norm(w)
            if nw < 1e-12:
                V = V[:, :j]
                break
            V[:, j] = w / nw
        self.effective_rank = V.shape[1]

        # Galerkin projection of the operator onto the reduced subspace
        Hr = V.T @ (H @ V)
        Hr = 0.5 * (Hr + Hr.T)
        theta, S = np.linalg.eigh(Hr)
        # reduced-model spectral weights: overlap of the (normalised) seed with
        # each reduced eigenvector, scaled by the true connected variance.
        c = S[0, :]  # V[:,0] is the seed direction
        w = (norm0 ** 2) * (c ** 2)
        d = theta - E0
        keep = (d > 1e-6) & (w > 1e-14)
        d, w = d[keep], w[keep]
        o = np.argsort(d)
        self._d, self._w = d[o], w[o]
        self._exp = exp_val
        self._var = float(self._w.sum())

    # -- Surrogate interface ------------------------------------------------
    def spectral(self):
        return self._exp, self._d.copy(), self._w.copy()

    def exp_val(self) -> float:
        return self._exp

    def variance(self) -> float:
        return self._var

    def provenance(self) -> dict:
        return {"method": "POD/Galerkin projection onto a Krylov snapshot subspace",
                "requested_rank": self.rank, "effective_rank": self.effective_rank,
                "basis_dim": int(self.model.dim), "coupling_g": self.g,
                "note": "Rayleigh-Ritz values converge from above"}
