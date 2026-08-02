"""Constructed spectral confound: an intentionally engineered adversarial surrogate.

This is not an ordinary learned surrogate. It is a *constructed adversarial
surrogate* -- the analogue of the Darcy confound-imitating construction,
transported to lattice gauge theory -- engineered to match every conventional
observable an ML surrogate is trained on and reports (the equal-time variance and
the signal-weighted / aggregate correlator error) while distorting the spectral
quantity that actually estimates the mass gap.

Two provable mechanisms are provided.

``spurious_slow`` (default; universally constructible)
    Fabricate a faint extra mode at ``rho * gap`` (``rho < 1``) carrying a small
    fraction ``frac`` of the variance, and renormalise the true weights so the
    equal-time variance ``C(0)`` is preserved *exactly*. Because that spurious
    mode is the slowest, it dominates the long-time tail and pulls the
    effective-mass plateau down to ``rho * gap`` -- an under-estimated gap --
    even though its contribution stays below the aggregate-error floor over the
    measured window. Physically: a fabricated long-range correlation, i.e. a
    spurious near-zero-eigenvalue in a reduced transfer operator or a slow
    autocorrelation mode in a neural sampler.

``attenuate_tail`` (requires a small-overlap light state)
    Delete / attenuate the true light-state weight so the tail decays at the
    next-heaviest rate -- an over-estimated gap. This is the "unsmeared operator
    misses the light glueball" failure. It is only feasible when the light state
    carries little weight in the chosen channel (otherwise attenuating it is
    visible in the conventional observables); the benchmark checks feasibility
    and reports it.

Note on direction: a surrogate whose correlator matches the reference across the
measured window *cannot* over-estimate the gap (the window bounds the slowest
rate from below), which is exactly why ``attenuate_tail`` needs a weak light
state. ``spurious_slow`` has no such requirement.
"""

from __future__ import annotations

import numpy as np

from .base import Surrogate


class SpectralShortcutControl(Surrogate):
    name = "spectral-shortcut(constructed)"

    def __init__(
        self,
        spectral_data,
        mode: str = "spurious_slow",
        rho: float = 0.6,
        frac: float = 0.03,
        attenuation: float = 0.0,
    ):
        self._exp = spectral_data.exp_val
        d = np.asarray(spectral_data.deltas, dtype=float)
        w = np.asarray(spectral_data.weights, dtype=float)
        self.mode = mode
        self.rho = float(rho)
        self.frac = float(frac)
        self.attenuation = float(attenuation)
        if mode == "spurious_slow":
            gap = float(d[0])
            var = float(w.sum())
            eps = frac * var
            d_spur = rho * gap
            self._d = np.concatenate([[d_spur], d])
            self._w = np.concatenate([[eps], w * (1.0 - frac)])  # C(0) preserved exactly
        elif mode == "attenuate_tail":
            # attenuate the lightest mode; renormalise the rest to preserve C(0).
            wc = w.copy()
            removed = (1.0 - attenuation) * wc[0]
            wc[0] = attenuation * wc[0]
            if wc[1:].sum() > 0:
                wc[1:] *= 1.0 + removed / wc[1:].sum()
            self._d = d.copy()
            self._w = wc
        else:
            raise ValueError(f"unknown mode {mode!r}")

    def provenance(self) -> dict:
        return {"method": "constructed adversarial spectral confound (not learned)",
                "mechanism": self.mode, "rho": getattr(self, "rho", None),
                "frac": getattr(self, "frac", None),
                "attenuation": getattr(self, "attenuation", None),
                "invariant": "equal-time variance C(0) preserved exactly"}

    def spectral(self):
        return self._exp, self._d.copy(), self._w.copy()
