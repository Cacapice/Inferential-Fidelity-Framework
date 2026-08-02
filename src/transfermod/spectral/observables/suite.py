"""Frozen observable suite.

All grids and fit windows are fixed here and must not be tuned per surrogate --
that is what makes the gate a preregistered test rather than a post-hoc fit.
Any change to these constants is an amendment (see ``PREREGISTRATION.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- frozen grids ---------------------------------------------------------
T_MAX = 8
TS = np.arange(0, T_MAX + 1, dtype=float)          # integer Euclidean times 0..8
PLATEAU_WINDOW = slice(-3, None)                    # last 3 effective-mass points
SHORT_WINDOW = TS <= 1.0                            # "short-distance" region
N_MOMENTS = 4                                       # matched short-time moments mu_0..mu_3


def effective_mass(correlator: np.ndarray) -> np.ndarray:
    """m_eff(t) = log[C(t)/C(t+1)] on the frozen unit-spaced grid.

    Guarded against non-positive / underflowing correlator values (which can
    arise for a badly-fit surrogate); such points return NaN rather than raising.
    """
    c = np.asarray(correlator, dtype=float)
    num, den = c[:-1], c[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((num > 0) & (den > 0), num / den, np.nan)
        return np.log(ratio)


def plateau_mass(correlator: np.ndarray) -> float:
    em = effective_mass(correlator)
    window = em[PLATEAU_WINDOW]
    if not np.any(np.isfinite(window)):
        return float("nan")
    with np.errstate(invalid="ignore"):
        return float(np.nanmean(window))


def moments(deltas: np.ndarray, weights: np.ndarray, n: int = N_MOMENTS) -> np.ndarray:
    d = np.asarray(deltas, dtype=float)
    w = np.asarray(weights, dtype=float)
    return np.array([(w * d ** k).sum() for k in range(n)])


@dataclass
class ObservableSet:
    exp_val: float
    variance: float
    correlator: np.ndarray
    eff_mass: np.ndarray
    plateau_gap: float
    asymptotic_gap: float
    moments: np.ndarray


class FrozenObservableSuite:
    """Compute the frozen observable set for a reference or a surrogate."""

    ts = TS

    def measure(self, obj) -> ObservableSet:
        # obj exposes: exp_val()/exp_val attr, variance(), correlator(ts), spectral()
        exp_val = obj.exp_val() if callable(getattr(obj, "exp_val", None)) else obj.exp_val
        variance = obj.variance() if callable(getattr(obj, "variance", None)) else obj.variance
        c = np.asarray(obj.correlator(self.ts), dtype=float)
        em = effective_mass(c)
        if hasattr(obj, "spectral_repr"):
            _, d, w = obj.spectral_repr()
            mom = moments(d, w)
            asy = obj.asymptotic_gap()
        else:  # a measured surrogate that only exposes a correlator
            mom = np.full(N_MOMENTS, np.nan)
            asy = plateau_mass(c)
        return ObservableSet(
            exp_val=float(exp_val),
            variance=float(variance),
            correlator=c,
            eff_mass=em,
            plateau_gap=plateau_mass(c),
            asymptotic_gap=float(asy),
            moments=mom,
        )
