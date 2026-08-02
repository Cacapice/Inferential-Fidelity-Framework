"""Fidelity moduli along monotone one-parameter perturbation families.

This module is a utility for **monotone one-parameter families**, not a generic
numerical implementation of the fidelity modulus. The caller supplies a family
``u_w`` implicitly, through three scalar statistics of the amplitude ``w``:

    agg(w)   the aggregate discrepancy  ||u_w - u||_A
    err(w)   a declared decision discrepancy in the quantity of interest
    det(w)   the detection statistic of the certification procedure

Standing assumptions, all needed for the returned numbers to mean what their
names say:

  (A1) each statistic is non-decreasing in ``w``, so each threshold is unique;
  (A2) ``err`` is continuous (the hypothesis of Theorem 2);
  (A3) the family is downward feasible: every ``w' < w`` is admissible whenever
       ``w`` is, so the accepted set is a down-set;
  (A4) the bracket ``[lo, hi]`` covers the relevant crossing. When it does not,
       the result is *censored* and says so — see ``Threshold``.

What is computed is the family-restricted quantity written ``omega_{D,Theta}``
in the methods note: exact for the family, a lower bound for the modulus over
all of ``D_eps``. Nothing here proves any family extremal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

Stat = Callable[[float], float]

DEFAULT_BRACKET_LO = 1e-14
DEFAULT_BRACKET_HI = 1.0

#: threshold statuses
CROSSED = "crossed"              # a genuine crossing was bracketed
ABOVE_BRACKET = "above_bracket"  # right-censored: stat(hi) < target
BELOW_BRACKET = "below_bracket"  # left-censored: stat(lo) >= target


class CensoredScalar(float):
    """Float-compatible result carrying threshold and bound semantics.

    ``estimate_kind`` is one of ``measured_crossing``, ``lower_bound``,
    ``upper_bound``, or ``indeterminate``. Arithmetic on this float subclass
    may discard the metadata; callers publishing results should preserve the
    object or serialize its attributes before coercing it to ``float``.
    """

    def __new__(
        cls,
        value: float,
        *,
        thresholds: dict[str, "Threshold"],
        estimate_kind: str | None = None,
    ):
        obj = float.__new__(cls, value)
        obj.thresholds = dict(thresholds)
        obj.censored = any(not threshold.crossed for threshold in thresholds.values())
        obj.estimate_kind = estimate_kind or _estimate_kind(thresholds)
        obj.publishable = obj.estimate_kind != "indeterminate"
        return obj

    @property
    def bracket(self) -> dict[str, dict[str, object]]:
        return {name: threshold.to_dict() for name, threshold in self.thresholds.items()}


def _estimate_kind(thresholds: dict[str, "Threshold"]) -> str:
    """Conservative scalar semantics implied by threshold censoring."""
    statuses = {threshold.status for threshold in thresholds.values()}
    if statuses == {CROSSED}:
        return "measured_crossing"
    if BELOW_BRACKET in statuses:
        return "indeterminate"
    if statuses <= {CROSSED, ABOVE_BRACKET}:
        return "lower_bound"
    return "indeterminate"


@dataclass(frozen=True)
class Threshold:
    """A bracketed threshold, with its censoring status.

    ``value`` is a genuine crossing only when ``status == CROSSED``. For a
    censored result ``value`` is the bracket endpoint, which is a bound on the
    true threshold, not an estimate of it. Downstream widths computed from a
    censored threshold are correspondingly one-sided; ``float(t)`` is provided
    for convenience but discards that information deliberately, so prefer
    checking ``t.crossed`` in any code that reports a certification result.
    """

    value: float
    status: str
    lo: float
    hi: float
    bracket_source: str = "caller"
    search_lo: float | None = None
    search_hi: float | None = None

    @property
    def crossed(self) -> bool:
        return self.status == CROSSED

    def __float__(self) -> float:
        return float(self.value)

    @property
    def bracket(self) -> tuple[float, float]:
        return (self.lo, self.hi)

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "status": self.status,
            "crossed": self.crossed,
            "bracket": [self.lo, self.hi],
            "bracket_source": self.bracket_source,
            "search_domain": [
                self.lo if self.search_lo is None else self.search_lo,
                self.hi if self.search_hi is None else self.search_hi,
            ],
            "model_assumption": (
                "Library-default modeling assumption for the search domain; "
                "confirm that this amplitude scale is appropriate for the perturbation family."
                if self.bracket_source == "library_default"
                else "Search-domain endpoints explicitly supplied by the caller."
            ),
        }


def bisect_threshold(stat: Stat, target: float,
                     lo: float | None = None, hi: float | None = None,
                     iters: int = 60) -> Threshold:
    """Amplitude at which a non-decreasing ``stat`` crosses ``target``.

    Log-bisection, O(log 1/delta) evaluations. Returns a :class:`Threshold`
    whose ``status`` records whether the crossing was actually bracketed. When
    ``lo`` or ``hi`` is omitted, the serialized threshold marks the search
    domain as a library-default modeling assumption rather than a caller-declared
    amplitude range.
    """
    used_default_bracket = lo is None or hi is None
    if lo is None:
        lo = DEFAULT_BRACKET_LO
    if hi is None:
        hi = DEFAULT_BRACKET_HI
    bracket_source = "library_default" if used_default_bracket else "caller"
    search_lo, search_hi = lo, hi

    if not all(math.isfinite(value) for value in (target, lo, hi)):
        raise ValueError("target, lo, and hi must be finite")
    if not (0 < lo < hi):
        raise ValueError("bisect_threshold requires 0 < lo < hi")
    if iters < 0:
        raise ValueError("iters must be nonnegative")

    def evaluate(amplitude: float) -> float:
        value = float(stat(amplitude))
        if not math.isfinite(value):
            raise ValueError(
                f"stat returned a non-finite value at amplitude {amplitude}: {value}"
            )
        return value

    lo_value = evaluate(lo)
    hi_value = evaluate(hi)
    def decreases(left: float, right: float) -> bool:
        return left > right and not math.isclose(
            left, right, rel_tol=1e-12, abs_tol=1e-15
        )

    if decreases(lo_value, hi_value):
        raise ValueError(
            "sampled values violate the required non-decreasing assumption: "
            f"stat({lo})={lo_value} > stat({hi})={hi_value}"
        )
    if hi_value < target:
        return Threshold(hi, ABOVE_BRACKET, lo, hi, bracket_source, search_lo, search_hi)
    if lo_value >= target:
        return Threshold(lo, BELOW_BRACKET, lo, hi, bracket_source, search_lo, search_hi)
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        mid_value = evaluate(mid)
        if decreases(lo_value, mid_value) or decreases(mid_value, hi_value):
            raise ValueError(
                "sampled values violate the required non-decreasing assumption: "
                f"stat({lo})={lo_value}, stat({mid})={mid_value}, "
                f"stat({hi})={hi_value}"
            )
        if mid_value < target:
            lo, lo_value = mid, mid_value
        else:
            hi, hi_value = mid, mid_value
    return Threshold(math.sqrt(lo * hi), CROSSED, lo, hi, bracket_source, search_lo, search_hi)


def ray_modulus(err: Stat, agg: Stat, eps: float, **kw) -> float:
    """Largest declared decision discrepancy at aggregate discrepancy <= ``eps``. 

    This is a ray-restricted quantity. The modulus over a multi-parameter family
    is the supremum of this over directions (Corollary 5); the modulus over a
    space is not computed here at all.
    """
    w = bisect_threshold(agg, eps, **kw)
    value = float(err(float(w) * (1 - 1e-9)))
    if not math.isfinite(value) or value < 0:
        raise ValueError("err must return a finite, nonnegative discrepancy")
    estimate_kind = {
        CROSSED: "measured_crossing",
        ABOVE_BRACKET: "lower_bound",
        BELOW_BRACKET: "indeterminate",
    }[w.status]
    return CensoredScalar(
        value,
        thresholds={"w_agg": w},
        estimate_kind=estimate_kind,
    )


#: Deprecated alias. ``modulus`` overstated what is computed.
modulus = ray_modulus


def ray_decision_modulus(q: Stat, agg: Stat, eps: float, reference_q: float,
                         discrepancy, **kw) -> float:
    """Ray-restricted modulus from a decision functional and discrepancy.

    ``q(w)`` returns the decision value along the perturbation ray. The
    discrepancy is evaluated against the declared ``reference_q``. This is the
    general form; :func:`ray_modulus` remains the backward-compatible API when
    callers have already constructed an error statistic.
    """
    return ray_modulus(lambda w: discrepancy(q(w), reference_q), agg, eps, **kw)


def certify_ray_modulus(
    q: Stat,
    agg: Stat,
    eps: float,
    *,
    geometry,
    coverage,
    perturbation_family: str,
    require_proven_coverage: bool = False,
    **kw,
):
    """Compute a ray-restricted modulus and return a coverage-aware result."""
    from transfermod.certification import (
        require_exact,
        modulus_result,
    )

    discrepancy = geometry.discrepancy_function()
    value = ray_decision_modulus(
        q,
        agg,
        eps,
        geometry.reference_q,
        discrepancy,
        **kw,
    )
    result = modulus_result(
        value,
        coverage=coverage,
        perturbation_family=perturbation_family,
        geometry=geometry,
        epsilon=eps,
    )
    return require_exact(result) if require_proven_coverage else result


def _width(num: Threshold, den: Threshold) -> CensoredScalar:
    if float(den) <= 0:
        value = float("inf")
    else:
        value = max(0.0, math.log10(float(num) / float(den)))
    # Ratios of censored thresholds are not generally direction-identifiable:
    # numerator and denominator censoring can reverse the bound direction.
    estimate_kind = "measured_crossing" if num.crossed and den.crossed else "indeterminate"
    return CensoredScalar(
        value,
        thresholds={"numerator": num, "denominator": den},
        estimate_kind=estimate_kind,
    )


def interval_width(agg: Stat, det: Stat, eps: float,
                   det_tol: float, **kw) -> float:
    """|I| = log10(w_agg / w_det), clipped at zero. Exposure, in decades."""
    return _width(bisect_threshold(agg, eps, **kw),
                  bisect_threshold(det, det_tol, **kw))


def blind_spot_width(err: Stat, det: Stat, tau: float,
                     det_tol: float, **kw) -> float:
    """Log-amplitude Silent Risk: ``log10(w_det / w_corrupt)`` decades.

    By Theorem 2, ``|B| == 0`` is equivalent to ``omega_{D,Theta} <= tau``.
    """
    return _width(bisect_threshold(det, det_tol, **kw),
                  bisect_threshold(err, tau, **kw))


# Explicit name for the repository-specific measure of Silent Risk.
log_amplitude_silent_risk_width = blind_spot_width


def signature(err: Stat, agg: Stat, det: Stat, eps: float, tau: float,
              det_tol: float, **kw) -> tuple[float, float]:
    """R = (|I|, |B|): exposure and log-amplitude Silent Risk, in decades."""
    return (interval_width(agg, det, eps, det_tol, **kw),
            blind_spot_width(err, det, tau, det_tol, **kw))


def condition_number(err: Stat, agg: Stat, w0: float = 1e-9) -> float:
    """Relative condition number along a ray: ``kappa_rel = omega'(0+)``.

    Estimated as the ratio of the two statistics' slopes at a small amplitude,
    which is exact when both are differentiable at ``w = 0``. By Theorem 3,
    ``omega(eps) = kappa_rel * eps + o(eps)``: the fidelity modulus is the
    finite-eps extension of the classical relative condition number, and this
    is its linearisation.
    """
    a, e = agg(w0), err(w0)
    return float("inf") if a <= 0 else e / a


def linearity_ratio(err: Stat, agg: Stat, eps: float, w0: float = 1e-9,
                    **kw) -> float:
    """``L(eps) = omega(eps) / (kappa_rel * eps)``.

    ``L`` near 1 means condition-number reasoning is valid at ``eps``; ``L`` far
    from 1 means the linearisation has broken down and ``kappa_rel`` cannot be
    used to certify or to predict the error, only to screen. Row 2 of the
    trichotomy is exactly the regime where ``kappa_rel`` is finite and
    ``L(eps_0)`` is far from 1 at the operating tolerance.
    """
    k = condition_number(err, agg, w0)
    if not math.isfinite(k) or k <= 0:
        return float("nan")
    return ray_modulus(err, agg, eps, **kw) / (k * eps)


def second_order(err: Stat, agg: Stat, eps_hi: float | None = None,
                  n: int = 6, **kw) -> tuple[float, float, float]:
    """Fit ``omega(eps) = a*eps + b*eps**2 + o(eps**2)`` along one ray.

    Returns ``(a, b, eps_crossover)`` where ``a = kappa_theta`` is the ray
    condition number (Theorem 3), ``b`` is the *ray* second-order coefficient
    (Proposition 2). ``b`` is signed, because only ``w >= 0`` is admissible along
    a one-sided family; the corresponding coefficient of the full symmetric
    modulus is non-negative (Theorem 4), so a negative ``b`` here says nothing
    about ``omega`` itself. ``eps_crossover`` is
    the scale ``a / |b|`` at which the quadratic term
    matches the linear one. Empirically ``eps_crossover`` locates the half-linearity
    point (``L(eps) = 1/2``) to within about a factor of two.

    The fit window defaults to a range scaled by ``1/a`` so that it sits inside
    the linear regime whatever the conditioning.
    """
    import numpy as np

    a0 = condition_number(err, agg)
    if not math.isfinite(a0) or a0 <= 0:
        return float("nan"), float("nan"), float("nan")
    hi = eps_hi if eps_hi is not None else 1e-5 / a0
    grid = np.geomspace(hi / 50.0, hi, n)
    vals = np.array([ray_modulus(err, agg, float(e), **kw) for e in grid])
    A = np.vstack([grid, grid ** 2]).T
    a, b = np.linalg.lstsq(A, vals, rcond=None)[0]
    cross = float(a / abs(b)) if b != 0 else float("inf")
    return float(a), float(b), cross


def certified_grid_max(ray_value: Callable[[float], float],
                       lo: float, hi: float, step: float,
                       lipschitz: float | None = None) -> tuple[float, float, float]:
    """Two-sided bound on a supremum over directions, from a finite grid.

    ``ray_value(d)`` returns the per-ray quantity (for instance the gated
    modulus along direction ``d``). Under Theorem 5's hypotheses — coverage of
    the admissible set by the family, ray monotonicity, and Lipschitz dependence
    on the direction with constant ``L`` — a grid of spacing ``step`` gives

        max_grid  <=  sup_d ray_value(d)  <=  max_grid + L * step.

    Returns ``(lower, upper, L)``. If ``lipschitz`` is None, ``L`` is *estimated*
    from the largest slope between adjacent nodes; that estimate is plausible,
    not certified, and a certified upper bound requires an a priori smoothness
    bound on the family. Coverage is the one hypothesis no computation can check:
    without it the result is a lower bound only.
    """
    import numpy as np

    grid = np.arange(lo, hi, step)
    vals = np.array([ray_value(float(d)) for d in grid])
    if lipschitz is None:
        lipschitz = (float(np.max(np.abs(np.diff(vals)) / np.diff(grid)))
                     if len(grid) > 1 else 0.0)
    lower = float(vals.max())
    return lower, lower + lipschitz * step, float(lipschitz)


def censoring(err: Stat, agg: Stat, det: Stat, eps: float, tau: float,
              det_tol: float, **kw) -> dict[str, str]:
    """Censoring status of each threshold behind a signature.

    Report this alongside ``signature``: a value computed from a censored
    threshold is a bound, not a measurement.
    """
    return {
        "w_agg": bisect_threshold(agg, eps, **kw).status,
        "w_det": bisect_threshold(det, det_tol, **kw).status,
        "w_corrupt": bisect_threshold(err, tau, **kw).status,
    }
