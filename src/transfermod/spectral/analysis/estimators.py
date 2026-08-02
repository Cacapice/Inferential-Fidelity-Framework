"""Alternative mass-gap estimators.

Practitioners do not read the gap off a single effective-mass plateau; they use
tail fits, multi-exponential fits, linear spectral reconstruction (Prony /
linear prediction), and model averaging. This module implements a representative
spread so the benchmark can show that the constructed confound fools *all* of
them in the same direction: the failure is a property of the correlator tail,
not an artifact of one extraction algorithm. These procedures are distinct
estimators of a shared information basis, so their agreement is a robustness
check rather than independent corroboration.

Every estimator takes a correlator sampled on the frozen integer grid ``ts`` and
returns a single gap estimate (the slowest real decay rate it infers).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def plateau_gap(ts: np.ndarray, C: np.ndarray, window: int = 3) -> float:
    """Effective-mass plateau: mean of log[C(t)/C(t+1)] over the last ``window``."""
    C = np.asarray(C, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        em = np.log(np.where((C[:-1] > 0) & (C[1:] > 0), C[:-1] / C[1:], np.nan))
    tail = em[-window:]
    if not np.any(np.isfinite(tail)):
        return float("nan")
    return float(np.nanmean(tail))


def single_exp_tail_gap(ts: np.ndarray, C: np.ndarray, t_min: int = 3) -> float:
    """Log-linear single-exponential fit over the tail ``t >= t_min``."""
    ts = np.asarray(ts, float)
    C = np.asarray(C, float)
    m = (ts >= t_min) & (C > 0)
    if m.sum() < 2:
        return float("nan")
    slope = np.polyfit(ts[m], np.log(C[m]), 1)[0]
    return float(-slope)


def multi_exp_gap(ts: np.ndarray, C: np.ndarray, K: int = 3,
                  n_restarts: int = 8, seed: int = 0) -> float:
    """Nonlinear ``K``-exponential fit; return the slowest fitted rate."""
    ts = np.asarray(ts, float)
    C = np.asarray(C, float)
    rng = np.random.default_rng(seed)

    def model(p):
        a = p[:K]
        r = p[K:]
        return (a[:, None] ** 2 * np.exp(-(r[:, None] ** 2) * ts[None, :])).sum(0)

    best = None
    scale = C[0] if C[0] > 0 else 1.0
    for _ in range(n_restarts):
        p0 = np.concatenate([rng.uniform(0.1, 1.0, K), rng.uniform(0.3, 3.0, K)])
        try:
            sol = least_squares(lambda p: (model(p) - C) / scale, p0,
                                method="lm", max_nfev=4000)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError):
            continue
        if best is None or sol.cost < best[0]:
            best = (sol.cost, sol.x)
    if best is None:
        return float("nan")
    r = best[1][K:] ** 2
    a = best[1][:K] ** 2
    r = r[a > 1e-8]
    return float(np.min(r)) if r.size else float("nan")


def cosh_gap(ts: np.ndarray, C: np.ndarray, T: float | None = None,
             seed: int = 0) -> float:
    """Single-state cosh fit ``C(t) = A cosh(m (t - T/2))``.

    The standard periodic-time estimator. With an open (non-periodic) correlator
    and ``T`` large this reduces to a single exponential; included because it is
    a routine choice and is fooled identically.
    """
    ts = np.asarray(ts, float)
    C = np.asarray(C, float)
    if T is None:
        T = 2.0 * ts[-1]
    rng = np.random.default_rng(seed)
    scale = C[0] if C[0] > 0 else 1.0

    def resid(p):
        A, m = p
        return (A * np.cosh(m * (ts - T / 2.0)) - C) / scale

    best = None
    for _ in range(6):
        p0 = np.array([rng.uniform(1e-4, 1.0), rng.uniform(0.3, 3.0)])
        try:
            sol = least_squares(resid, p0, method="lm", max_nfev=4000)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError):
            continue
        if best is None or sol.cost < best[0]:
            best = (sol.cost, sol.x)
    if best is None:
        return float("nan")
    return float(abs(best[1][1]))


def prony_gap(ts: np.ndarray, C: np.ndarray, order: int = 3) -> float:
    """Prony / linear-prediction spectral reconstruction.

    Fits ``C(n) = sum_j c_j C(n-j)`` by least squares, then takes the decay rates
    from the roots of the prediction polynomial. A purely linear-algebra
    estimator (no nonlinear optimisation), representative of Backus--Gilbert /
    linear spectral-reconstruction approaches. Returns the slowest real positive
    rate.
    """
    C = np.asarray(C, float)
    n = C.size
    p = min(order, (n - 1) // 2)
    if p < 1:
        return float("nan")
    # Hankel linear-prediction system: C[p+i] = sum_j c_j C[p-1-j+i]
    rows = n - p
    A = np.empty((rows, p))
    b = np.empty(rows)
    for i in range(rows):
        A[i, :] = C[i:i + p][::-1]
        b[i] = C[i + p]
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    # roots of x^p - c_1 x^{p-1} - ... - c_p
    poly = np.concatenate([[1.0], -coeffs])
    roots = np.roots(poly)
    rates = []
    for z in roots:
        if abs(z.imag) < 1e-6 and z.real > 0:
            r = -np.log(z.real)
            if r > 1e-6:
                rates.append(r)
    return float(min(rates)) if rates else float("nan")


def bayesian_model_avg_gap(ts: np.ndarray, C: np.ndarray, seed: int = 0) -> float:
    """AIC-weighted average of the single- and two-exponential gap estimates.

    A lightweight stand-in for Bayesian model averaging over fit models. Fooled
    the same way because every member sees the same corrupted tail.
    """
    ts = np.asarray(ts, float)
    C = np.asarray(C, float)
    scale = C[0] if C[0] > 0 else 1.0

    def fit_K(K):
        rng = np.random.default_rng(seed + K)

        def model(p):
            a = p[:K]
            r = p[K:]
            return (a[:, None] ** 2 * np.exp(-(r[:, None] ** 2) * ts[None, :])).sum(0)

        best = None
        for _ in range(8):
            p0 = np.concatenate([rng.uniform(0.1, 1.0, K), rng.uniform(0.3, 3.0, K)])
            try:
                sol = least_squares(lambda p: (model(p) - C) / scale, p0,
                                    method="lm", max_nfev=4000)
            except Exception:
                continue
            if best is None or sol.cost < best[0]:
                best = (sol.cost, sol.x)
        if best is None:
            return None
        rss = float(np.sum((model(best[1]) - C) ** 2))
        r = best[1][K:] ** 2
        a = best[1][:K] ** 2
        gap = float(np.min(r[a > 1e-8])) if np.any(a > 1e-8) else float("nan")
        k_params = 2 * K
        aic = C.size * np.log(max(rss, 1e-300) / C.size) + 2 * k_params
        return gap, aic

    fits = [f for f in (fit_K(1), fit_K(2)) if f is not None]
    if not fits:
        return float("nan")
    gaps = np.array([g for g, _ in fits])
    aics = np.array([a for _, a in fits])
    w = np.exp(-0.5 * (aics - aics.min()))
    w = w / w.sum()
    return float(np.sum(w * gaps))


ESTIMATORS = {
    "effective-mass plateau": plateau_gap,
    "single-exp tail fit": single_exp_tail_gap,
    "multi-exp fit": multi_exp_gap,
    "cosh fit": cosh_gap,
    "Prony / linear prediction": prony_gap,
    "Bayesian model average": bayesian_model_avg_gap,
}
