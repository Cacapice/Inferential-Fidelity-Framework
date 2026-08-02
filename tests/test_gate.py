

def test_unmeasured_is_not_a_pass():
    """A surrogate that does not expose the leading spectral weight cannot be
    CERTIFIED on that basis: unknown maps to INSUFFICIENT_EVIDENCE, not pass."""
    import numpy as np

    from transfermod.spectral.gate.tests import GateResult, Verdict, _status

    assert _status(float("nan"), 0.15) == "unmeasured"
    assert _status(0.1, 0.15) == "pass"
    assert _status(0.2, 0.15) == "fail"

    conv = [GateResult(f"C{i}", "conventional", 0.0, 1.0, "pass") for i in range(3)]
    spec_ok = [GateResult("S1", "spectral", 0.0, 1.0, "pass")]
    spec_unk = [GateResult("S2", "spectral", float("nan"), 1.0, "unmeasured")]

    from transfermod import certification

    assert Verdict("x", conv + spec_ok).code == certification.CERTIFIED
    v = Verdict("x", conv + spec_ok + spec_unk)
    assert v.code == certification.INSUFFICIENT_EVIDENCE
    assert "S2" in v.explanation
    assert not v.spectral_pass
    assert not v.spectral_fail
    assert GateResult("S2", "spectral", float("nan"), 1.0, "unmeasured").passed is False


def test_domain_free_modulus_api():
    """transfermod.modulus works with no reference to any physical model."""
    from transfermod.modulus import (ABOVE_BRACKET, CROSSED, bisect_threshold,
                                     blind_spot_width, censoring,
                                     interval_width, ray_modulus, signature)

    err = lambda w: 3.0 * w      # noqa: E731
    agg = lambda w: 0.085 * w    # noqa: E731
    det = lambda w: 8.0 * w      # noqa: E731

    assert abs(float(bisect_threshold(agg, 5e-3)) - 5e-3 / 0.085) < 1e-6
    assert ray_modulus(err, agg, 5e-3) > 0.17
    expo, silent = signature(err, agg, det, eps=5e-3, tau=0.05, det_tol=0.15)
    assert expo == interval_width(agg, det, 5e-3, 0.15)
    assert silent == blind_spot_width(err, det, 0.05, 0.15)
    assert expo > 0 and silent >= 0

    # censoring is reported, not hidden
    assert bisect_threshold(agg, 5e-3).status == CROSSED
    assert bisect_threshold(lambda w: 1e-12 * w, 1.0).status == ABOVE_BRACKET
    assert set(censoring(err, agg, det, eps=5e-3, tau=0.05,
                         det_tol=0.15)) == {"w_agg", "w_det", "w_corrupt"}


def test_condition_number_linearises_the_modulus():
    """Theorem 3: omega(eps) = kappa_rel * eps + o(eps), and L -> 1 as eps -> 0."""
    from transfermod.modulus import (condition_number, linearity_ratio,
                                     ray_modulus)

    # a smooth ray with known slopes
    err = lambda w: 3.0 * w + 12.0 * w ** 2   # noqa: E731
    agg = lambda w: 0.085 * w                 # noqa: E731

    k = condition_number(err, agg)
    assert abs(k - 3.0 / 0.085) < 1e-4

    # linearisation is exact to first order
    for eps in (1e-8, 1e-9):
        assert abs(linearity_ratio(err, agg, eps) - 1.0) < 1e-5
    # and degrades at large eps, where the quadratic term dominates
    assert linearity_ratio(err, agg, 1e-2) > 1.0

    assert abs(ray_modulus(err, agg, 1e-9) - k * 1e-9) < 1e-12


def test_spectral_family_is_row_two_in_condition_number_terms():
    """kappa_rel finite, but the linear regime ends far below the acceptance floor."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_feasibility import make_metrics

    from transfermod.modulus import condition_number, linearity_ratio
    from transfermod.spectral import build_reference

    m = make_metrics(build_reference(2, 2, 2, g=1.0))
    err = lambda w: m(0.6, w)[1]   # noqa: E731
    agg = lambda w: m(0.6, w)[0]   # noqa: E731

    k = condition_number(err, agg)
    assert 1e3 < k < 1e5, "finite condition number: not row 3"
    assert linearity_ratio(err, agg, 5e-3) < 0.1, (
        "linearisation must have broken down at the acceptance floor: row 2")


def test_certified_grid_max_brackets_the_supremum():
    """Theorem 5: a finite grid gives a two-sided bound, not just a lower one."""
    import numpy as np

    from transfermod.modulus import certified_grid_max

    # a smooth unimodal ray function with interior maximiser
    f = lambda d: 1.0 - (d - 0.37) ** 2   # noqa: E731
    truth = 1.0

    for step in (0.2, 0.1, 0.05, 0.01):
        lo, hi, L = certified_grid_max(f, 0.05, 0.999, step)
        assert lo <= truth <= hi, f"bracket failed at step={step}"
        assert L > 0
    # tighter grids give tighter brackets
    widths = [certified_grid_max(f, 0.05, 0.999, s)[1]
              - certified_grid_max(f, 0.05, 0.999, s)[0] for s in (0.2, 0.05, 0.01)]
    assert widths[0] > widths[1] > widths[2]


def test_second_order_expansion_recovers_known_coefficients():
    """Theorem 4: omega(eps) = a*eps + b*eps^2 + o(eps^2), fitted exactly."""
    from transfermod.modulus import second_order

    # omega(eps) is exactly 3/0.085 * eps - 12/0.085^2 * eps^2 for this ray
    err = lambda w: 3.0 * w - 12.0 * w ** 2   # noqa: E731
    agg = lambda w: 0.085 * w                 # noqa: E731

    a, b, cross = second_order(err, agg, eps_hi=1e-3)
    assert abs(a - 3.0 / 0.085) / (3.0 / 0.085) < 1e-4
    assert abs(b - (-12.0 / 0.085 ** 2)) / (12.0 / 0.085 ** 2) < 1e-3
    assert abs(cross - a / abs(b)) < 1e-12


def test_crossover_predicts_where_linearisation_fails():
    """eps_crossover locates the half-linearity point to within a factor of ~2."""
    import sys
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_feasibility import make_metrics

    from transfermod.modulus import linearity_ratio, second_order
    from transfermod.spectral import build_reference

    m = make_metrics(build_reference(2, 2, 2, g=1.0))
    for rho in (0.9, 0.6):
        err = lambda w, r=rho: m(r, w)[1]   # noqa: E731
        agg = lambda w, r=rho: m(r, w)[0]   # noqa: E731
        _, b, cross = second_order(err, agg)
        assert b < 0, "saturation: the quadratic correction is negative"
        # half-linearity point, by bisection on L
        lo, hi = 1e-12, 0.5
        for _ in range(50):
            mid = np.sqrt(lo * hi)
            if linearity_ratio(err, agg, mid) > 0.5:
                lo = mid
            else:
                hi = mid
        half = np.sqrt(lo * hi)
        assert 0.4 < half / cross < 3.0, f"crossover off by {half / cross:.2f}x"


def test_bisect_threshold_rejects_non_monotone_statistic():
    import pytest

    from transfermod.modulus import bisect_threshold

    with pytest.raises(ValueError, match="non-decreasing"):
        bisect_threshold(lambda w: -w, target=-0.5, lo=0.1, hi=1.0)


def test_width_preserves_censoring_metadata():
    from transfermod.modulus import interval_width

    width = interval_width(
        lambda w: 1e-12 * w,
        lambda w: w,
        eps=1.0,
        det_tol=0.1,
    )
    assert isinstance(width, float)
    assert width.censored
    assert width.bracket["numerator"]["status"] == "above_bracket"


def test_bisect_threshold_rejects_nonfinite_values():
    import math
    import pytest
    from transfermod.modulus import bisect_threshold

    with pytest.raises(ValueError, match="must be finite"):
        bisect_threshold(lambda w: w, target=math.nan)
    with pytest.raises(ValueError, match="non-finite"):
        bisect_threshold(lambda w: math.nan if w > 0.2 else w, target=0.5)


def test_ray_modulus_exposes_directional_censor_semantics():
    from transfermod.modulus import ray_modulus

    right = ray_modulus(lambda w: w, lambda w: 1e-12 * w, 1.0)
    assert right.estimate_kind == "lower_bound"
    assert right.publishable

    left = ray_modulus(lambda w: w, lambda w: 1.0 + w, 0.5)
    assert left.estimate_kind == "indeterminate"
    assert not left.publishable
