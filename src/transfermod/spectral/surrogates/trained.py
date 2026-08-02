"""A genuinely *trained* surrogate: a multi-exponential transfer-operator model
fit to noisy sampled correlator data under the conventional (signal-weighted)
objective an ML practitioner would use.

This is not a hand-constructed confound. It is fit by nonlinear least squares to
synthetic "measured" correlator data ``C(t) + noise`` and then queried for the
gap. Whether it is spectrally faithful is an *emergent* property of the fit:

- With low statistical noise and an informative tail, the fit recovers the true
  gap and is certified.
- With a realistic constant noise floor (the exponential signal-to-noise problem
  of lattice spectroscopy), the weak light state falls below the noise over the
  window, the fit locks onto the dominant heavier mode, and the reported gap is
  badly wrong -- while the aggregate correlator error stays tiny. The gate then
  flags it as a confound.

The surrogate exposes its *fitted* spectral pairs and its *fitted* correlator
through the standard ``Surrogate`` interface, so a learned model plugs into the
same gate as the analytic surrogates. ``scripts/run_scaling.py`` sweeps the
noise floor to exhibit the certified -> confound transition.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .base import Surrogate


class TrainedMultiExpSurrogate(Surrogate):
    name = "trained-multiexp(learned)"

    def __init__(self, reference, ts: np.ndarray, rel_noise: float = 0.01,
                 noise_model: str = "absolute_floor", K_values=(2, 3),
                 n_restarts: int = 8, seed: int = 0):
        """Fit to sampled correlator data from ``reference``.

        noise_model:
            "absolute_floor" -- sigma = rel_noise * C(0), constant across t
                (realistic: statistical error does not shrink with the signal;
                large-t points drown, hiding a weak light state).
            "relative"       -- sigma = rel_noise * C(t) (keeps the tail
                informative; the fit usually recovers the gap).
        """
        self.ts = np.asarray(ts, dtype=float)
        self._exp = reference.exp_val() if callable(getattr(reference, "exp_val", None)) \
            else reference.exp_val
        Cref = np.asarray(reference.correlator(self.ts), dtype=float)
        rng = np.random.default_rng(seed)
        if noise_model == "absolute_floor":
            sigma = rel_noise * Cref[0] * np.ones_like(Cref)
        elif noise_model == "relative":
            sigma = rel_noise * np.abs(Cref) + 1e-30
        else:
            raise ValueError(noise_model)
        Cdata = Cref + sigma * rng.standard_normal(Cref.shape)

        best = None
        for K in K_values:
            for _ in range(n_restarts):
                p0 = np.concatenate([rng.uniform(0.1, 1.0, K), rng.uniform(0.5, 3.0, K)])
                try:
                    sol = least_squares(
                        lambda p, K=K: (self._model(p, K, self.ts) - Cdata) / sigma,
                        p0, method="lm", max_nfev=5000,
                    )
                except (ValueError, FloatingPointError,
                        np.linalg.LinAlgError, RuntimeError):
                    continue
                if best is None or sol.cost < best[0]:
                    best = (sol.cost, sol.x, K)
        if best is None:
            raise RuntimeError("multi-exponential fit failed to converge")
        _, x, K = best
        a = x[:K] ** 2
        r = x[K:] ** 2
        o = np.argsort(r)
        self._d = r[o]
        self._w = a[o]
        self.fit_cost = best[0]
        self.K = K
        self._hp = {"noise_model": noise_model, "rel_noise": float(rel_noise),
                    "K_selected": int(K), "K_candidates": list(K_values),
                    "n_restarts": int(n_restarts), "seed": int(seed),
                    "loss": "signal-weighted nonlinear least squares"}

    @staticmethod
    def _model(params, K, ts):
        a = params[:K]
        r = params[K:]
        return (a[:, None] ** 2 * np.exp(-(r[:, None] ** 2) * ts[None, :])).sum(0)

    def spectral(self):
        return self._exp, self._d.copy(), self._w.copy()

    def provenance(self) -> dict:
        return {"method": "multi-exponential fit to noisy sampled correlator data",
                "fit_cost": float(self.fit_cost)}

    def training_info(self) -> dict:
        return dict(self._hp)
