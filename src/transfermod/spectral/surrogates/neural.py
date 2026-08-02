"""A real neural sampler: a Neural Quantum State (NQS) trained by VMC.

This is a genuine neural network -- a single-hidden-layer MLP wavefunction
``psi_theta(config) = exp(w2 . tanh(W1 config + b1) + b2)`` over the electric-field
basis -- trained by gradient descent (Adam) on the exact variational energy
``E(theta) = <psi_theta|H|psi_theta> / <psi_theta|psi_theta>``. The compact U(1)
ground state admits a positive variational ansatz, allowing deterministic
optimization over the full physical basis without Monte Carlo noise.

The trained state is then *measured* like any surrogate: equal-time expectation
and variance come straight from the learned amplitudes, and the connected
imaginary-time correlator is obtained by seeding the exact-``H`` Lanczos spectral
routine with ``v = O_c |psi_theta>``. The benchmark then asks the question it was
built for -- does this learned state preserve the low-energy spectral evidence? --
and the same gate judges it.

Empirically (see ``scripts/run_learned.py``): once the NQS is reasonably converged
(energy error <~1e-3) it is spectrally faithful and CERTIFIED; the gate does not
false-positive on a real learned model. Intermediate networks are rejected on a
conventional check (equal-time or aggregate correlator error) rather than passing
as a silent confound.
"""

from __future__ import annotations

import numpy as np

from ..lattice.u1 import U1Model
from ..reference.krylov import lanczos_spectral
from .base import Surrogate


class NeuralQuantumStateSurrogate(Surrogate):
    name = "neural-quantum-state(learned)"

    def __init__(self, model: U1Model, g: float, hidden: int = 24, epochs: int = 2000,
                 lr: float = 5e-2, seed: int = 0, operator_sparse=None,
                 leak_rate_floor: float = 1e-3):
        self.model = model
        self.g = float(g)
        H = model.hamiltonian_sparse(g)
        O = model.cos_plaquette_operator_sparse() if operator_sparse is None else operator_sparse
        X = np.asarray(model.basis, dtype=float) / model.lat.Lambda
        D, n = X.shape

        rng = np.random.default_rng(seed)
        P = [rng.standard_normal((n, hidden)) * 0.3, np.zeros(hidden),
             rng.standard_normal(hidden) * 0.3, 0.0]
        mom = [np.zeros_like(p) for p in P]
        vel = [np.zeros_like(p) for p in P]
        energies = []
        t = 0
        for _ in range(epochs):
            t += 1
            a1 = np.tanh(X @ P[0] + P[1])
            lp = a1 @ P[2] + P[3]
            lp = lp - lp.max()
            psi = np.exp(lp)
            psi /= np.linalg.norm(psi)
            Hpsi = H @ psi
            E = float(psi @ Hpsi)
            energies.append(E)
            local = Hpsi / psi
            c = 2.0 * (psi ** 2) * (local - E)           # d/d(logpsi) weight
            grads = [X.T @ (np.outer(c, P[2]) * (1 - a1 ** 2)),
                     (np.outer(c, P[2]) * (1 - a1 ** 2)).sum(0),
                     a1.T @ c, np.array(c.sum())]
            for i in range(4):
                mom[i] = 0.9 * mom[i] + 0.1 * grads[i]
                vel[i] = 0.999 * vel[i] + 0.001 * grads[i] ** 2
                mhat = mom[i] / (1 - 0.9 ** t)
                vhat = vel[i] / (1 - 0.999 ** t)
                P[i] = P[i] - lr * mhat / (np.sqrt(vhat) + 1e-8)

        self._hp = {"hidden": int(hidden), "epochs": int(epochs), "lr": float(lr),
                    "seed": int(seed), "optimizer": "adam"}
        # final state and measured spectral quantities
        a1 = np.tanh(X @ P[0] + P[1])
        lp = a1 @ P[2] + P[3]
        lp = lp - lp.max()
        psi = np.exp(lp)
        psi /= np.linalg.norm(psi)
        self.energy = float(psi @ (H @ psi))
        self.energy_trace = np.asarray(energies)
        self._exp = float(psi @ (O @ psi))
        self._var = float(psi @ (O @ (O @ psi))) - self._exp ** 2

        v = O @ psi - self._exp * psi
        ritz, w = lanczos_spectral(H, v, m=min(150, D - 1))
        rates = ritz - self.energy
        # drop the (near-zero / negative) vacuum-overlap artifact from the
        # imperfect variational vacuum; keep physical decaying modes.
        keep = (rates > leak_rate_floor) & (w > 1e-14)
        d, ww = rates[keep], w[keep]
        o = np.argsort(d)
        self._d, self._w = d[o], ww[o]

    # -- Surrogate interface ------------------------------------------------
    def spectral(self):
        return self._exp, self._d.copy(), self._w.copy()

    def provenance(self) -> dict:
        return {"method": "variational Monte Carlo (exact summation over basis)",
                "ansatz": "single-hidden-layer MLP log-amplitude wavefunction",
                "basis_dim": int(self.model.dim), "coupling_g": self.g,
                "spectral_extraction": "Lanczos spectral function seeded by O_c|psi>"}

    def training_info(self) -> dict:
        return {**self._hp, "final_variational_energy": float(self.energy),
                "n_epochs_recorded": int(self.energy_trace.size)}

    def exp_val(self) -> float:
        return self._exp

    def variance(self) -> float:
        return self._var

    def correlator(self, ts):
        ts = np.asarray(ts, dtype=float)
        return (self._w[:, None] * np.exp(-self._d[:, None] * ts[None, :])).sum(0)

    def asymptotic_gap(self) -> float:
        """Report the gap the way the gate reads it -- the effective-mass plateau
        of the measured correlator -- so negligible variational leakage (tiny-weight
        spurious modes) does not masquerade as the gap."""
        ts = np.arange(0, 9)
        C = self.correlator(ts)
        with np.errstate(divide="ignore", invalid="ignore"):
            em = np.log(np.where((C[:-1] > 0) & (C[1:] > 0), C[:-1] / C[1:], np.nan))
        tail = em[-3:]
        if not np.any(np.isfinite(tail)):
            return float("nan")
        return float(np.nanmean(tail))
