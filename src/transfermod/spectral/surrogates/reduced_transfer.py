"""Ordinary learned surrogate: a reduced (low-rank) transfer operator.

This models an honest, competent surrogate -- e.g. a truncated / compressed
transfer operator, or a well-trained emulator that captures the leading
spectral structure. It keeps the ``K`` leading modes (which INCLUDE the light
state that sets the gap) and discards only high-energy modes that contribute
negligibly to the correlator. Optionally it adds a small broadband multiplicative
perturbation to the retained weights to mimic finite training accuracy.

Expected behaviour: PASSES both the conventional and the spectral gates. Its
role in the benchmark is to show the gate does not raise false positives on a
genuinely good approximation.
"""

from __future__ import annotations

import numpy as np

from .base import Surrogate


class ReducedTransferSurrogate(Surrogate):
    name = "reduced-transfer(ordinary)"

    def __init__(self, spectral_data, K: int | None = None, var_keep: float = 0.9999,
                 weight_jitter: float = 0.0, seed: int = 0):
        self._exp = spectral_data.exp_val
        d = np.asarray(spectral_data.deltas, dtype=float)
        w = np.asarray(spectral_data.weights, dtype=float)
        if K is None:
            # keep the minimal set of leading modes retaining >= var_keep of the
            # connected variance -- a low-rank reduction that, crucially, retains
            # the light state that sets the gap.
            cum = np.cumsum(w) / w.sum()
            K = int(np.searchsorted(cum, var_keep) + 1)
        K = min(K, d.size)
        self._d = d[:K].copy()
        w = w[:K].copy()
        if weight_jitter > 0:
            rng = np.random.default_rng(seed)
            w = w * (1.0 + weight_jitter * rng.standard_normal(w.shape))
            w = np.clip(w, 0.0, None)
        self._w = w

    def provenance(self) -> dict:
        return {"method": "low-rank spectral reduction of the transfer operator",
                "modes_kept": int(self._d.size)}

    def spectral(self):
        return self._exp, self._d.copy(), self._w.copy()
