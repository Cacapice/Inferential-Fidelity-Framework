"""Tests for the operational feasibility interval (PREREG 10.2, quantitative form).

These pin the three claims the Operational Structural Asymmetry Principle rests
on: (i) both gate metrics are monotone in the fabricated weight, which is what
makes the feasible set a single interval and earns the biconditional; (ii) the
interval closes above a critical rho, so the gate is complete against mild
under-estimates; (iii) the interval widens with the observation window.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from run_feasibility import endpoints, make_metrics  # noqa: E402

from transfermod.spectral import build_reference  # noqa: E402


@pytest.fixture(scope="module")
def ref():
    return build_reference(Lx=2, Ly=2, Lambda=2, g=1.0)


def test_gate_metrics_monotone_in_weight(ref):
    """The lemma behind the 'only if': each constraint cuts the axis once."""
    metrics = make_metrics(ref)
    ws = np.logspace(-7, np.log10(0.5), 30)
    for rho in (0.3, 0.6, 0.9):
        g2 = np.array([metrics(rho, w)[0] for w in ws])
        g4 = np.array([metrics(rho, w)[1] for w in ws])
        assert np.all(np.diff(g2) > 0), f"G2 not monotone at rho={rho}"
        assert np.all(np.diff(g4) > 0), f"G4 not monotone at rho={rho}"


def test_aggregate_error_is_linear_in_weight(ref):
    """G2 is exactly proportional to w, so w_G2 is a closed-form endpoint."""
    metrics = make_metrics(ref)
    ws = np.logspace(-6, -2, 12)
    g2 = np.array([metrics(0.6, w)[0] for w in ws])
    ratio = g2 / ws
    assert np.allclose(ratio, ratio[0], rtol=1e-6)


def test_plateau_interval_closes_but_full_gate_interval_does_not(ref):
    """rho=0.95: no mode can move the plateau while hiding -- but G5 still sees it."""
    metrics = make_metrics(ref)
    w_g4, w_g5, w_g2 = endpoints(metrics, rho=0.95)
    assert w_g4 > w_g2, "plateau-only interval should be empty at a 5% under-estimate"
    assert w_g5 < w_g2, "the full gate interval should remain open (G5 is binding)"

    w_g4, w_g5, w_g2 = endpoints(metrics, rho=0.60)
    assert w_g2 > min(w_g4, w_g5), "interval non-empty at the preregistered rho"


def test_early_warning_tail_gate_is_strictly_more_sensitive(ref):
    """w_G5 < w_G4 at every rho: the tail rejects before the plateau does."""
    metrics = make_metrics(ref)
    for rho in (0.9, 0.7, 0.5, 0.3):
        w_g4, w_g5, _ = endpoints(metrics, rho)
        assert w_g5 < w_g4, f"tail gate should bind first at rho={rho}"


def test_interval_widens_with_observation_window(ref):
    """Longer tail -> more exposure to hidden slow modes, not less."""
    widths = []
    for tmax in (4, 8, 16):
        m = make_metrics(ref, ts=np.arange(0, tmax + 1, dtype=float))
        w_g4, w_g5, wg = endpoints(m, rho=0.6)
        wd = min(w_g4, w_g5)
        assert wg > wd
        widths.append(np.log10(wg / wd))
    assert widths[0] < widths[1] < widths[2]


def test_preregistered_control_sits_inside_the_interval(ref):
    """The pilot's frac=0.03 at rho=0.6 must be a feasible confound."""
    metrics = make_metrics(ref)
    w_g4, w_g5, wg = endpoints(metrics, rho=0.6)
    assert min(w_g4, w_g5) < 0.03 < wg


def test_blind_spot_empty_at_loose_science_tolerance(ref):
    """B(rho, tau) = 0 for tau >= 5%: the gate fires before the science breaks."""
    from run_feasibility import _bisect_rising

    metrics = make_metrics(ref)
    for rho in (0.9, 0.6, 0.2):
        _, w_det, _ = endpoints(metrics, rho)
        for tau in (0.10, 0.05):
            w_corrupt = _bisect_rising(lambda w: metrics(rho, w)[1], tau)
            assert w_corrupt >= w_det, (
                f"blind spot should be empty at rho={rho}, tau={tau}")


def test_blind_spot_opens_but_stays_bounded_at_tight_tolerance(ref):
    """B opens below ~2% and remains under half a decade."""
    from run_feasibility import _bisect_rising

    metrics = make_metrics(ref)
    _, w_det, _ = endpoints(metrics, rho=0.6)
    w_corrupt = _bisect_rising(lambda w: metrics(0.6, w)[1], 0.01)
    assert w_corrupt < w_det, "blind spot should open at a 1% science tolerance"
    assert np.log10(w_det / w_corrupt) < 0.5, "blind spot should stay bounded"


def test_scalar_interval_accessors(ref):
    """|I| and |B| are single scalars, clipped at zero when closed."""
    from run_feasibility import blind_spot_width, interval_width

    metrics = make_metrics(ref)
    assert interval_width(metrics, 0.60) > interval_width(metrics, 0.95) > 0
    assert blind_spot_width(metrics, 0.60, 0.05) == 0.0
    assert 0.0 < blind_spot_width(metrics, 0.60, 0.01) < 0.5


def test_blind_spot_theorem_reference_instance(ref):
    """|B| = 0 across the swept rho range at tau >= 2.36%."""
    from run_feasibility import blind_spot_width

    metrics = make_metrics(ref)
    for rho in np.arange(0.05, 1.0, 0.05):
        assert blind_spot_width(metrics, float(rho), 0.025) == 0.0


def test_certification_signature(ref):
    """R = (|I|, |B|) is the reported pair."""
    from run_feasibility import signature

    exposure, silent = signature(make_metrics(ref), rho=0.6, tau=0.05)
    assert exposure > 2.5 and silent == 0.0


def test_signature_converges_under_truncation_refinement(ref):
    """R is a property of the problem, not the discretisation."""
    from transfermod.spectral import build_reference

    from run_feasibility import signature

    widths = []
    for Lam in (2, 3, 4):
        m = make_metrics(build_reference(2, 2, Lam, g=1.0))
        expo, silent = signature(m, rho=0.6, tau=0.05)
        widths.append(expo)
        assert silent == 0.0
    assert max(widths) - min(widths) < 0.01, "exposure should be converged by Lambda=2"


def test_inferential_modulus_decays_slowly(ref):
    """omega(eps) >> eps: the aggregate norm does not control Q."""
    from run_modulus import modulus

    om_loose, _ = modulus(ref, 5e-3)
    om_tight, _ = modulus(ref, 1e-6)
    assert om_loose > 0.9, "Q essentially unconstrained at the acceptance floor"
    assert om_tight > 0.4, "six orders tighter still leaves Q badly controlled"
    assert om_tight < om_loose


def test_asymptotic_functional_is_discontinuous(ref):
    """Q_asymptotic = rho*gap for every w > 0, however small."""
    from transfermod.spectral.surrogates.spectral_shortcut import SpectralShortcutControl

    gap = ref.mass_gap()
    for w in (1e-2, 1e-6, 1e-10):
        ctl = SpectralShortcutControl(ref.spectral, "spurious_slow", rho=0.6, frac=w)
        assert abs(ctl.asymptotic_gap() - 0.6 * gap) < 1e-12


def test_gating_bounds_the_modulus(ref):
    """omega_gate << omega, and is flat in eps: certification, not accuracy, controls Q."""
    from run_modulus import gated_modulus, modulus

    for eps in (5e-3, 1e-5):
        raw, _ = modulus(ref, eps)
        gat = gated_modulus(ref, eps)
        assert gat < raw / 10, f"gating should reduce the modulus by >10x at eps={eps}"
    assert abs(gated_modulus(ref, 5e-3) - gated_modulus(ref, 1e-5)) < 1e-3


def test_gated_modulus_matches_tau_crit(ref):
    """|B(tau)| = 0 iff tau >= omega_gate -- the two computations must agree."""
    from run_feasibility import blind_spot_width
    from run_modulus import gated_modulus

    metrics = make_metrics(ref)
    og = gated_modulus(ref, 5e-3)
    assert all(blind_spot_width(metrics, float(r), og * 1.05) == 0.0
               for r in np.arange(0.1, 1.0, 0.1))
