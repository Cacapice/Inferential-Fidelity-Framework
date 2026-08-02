"""Validation tests for the spectral-fidelity benchmark.

Physics tests establish that the reference is a correct compact-U(1)
Kogut--Susskind theory; gate tests establish the headline behaviour
(control is a confound; honest surrogates are certified).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transfermod.spectral.gate.tests import certify
from transfermod.spectral.lattice.u1 import U1Lattice, U1Model
from transfermod.spectral.reference.exact import ExactReference
from transfermod.spectral.surrogates import (
    IdentitySurrogate,
    ReducedTransferSurrogate,
    SpectralShortcutControl,
)


@pytest.fixture(scope="module")
def model():
    return U1Model(U1Lattice(Lx=2, Ly=2, Lambda=2))


@pytest.fixture(scope="module")
def ref(model):
    return ExactReference(model, g=1.0)


# -- physics correctness ---------------------------------------------------
def test_basis_satisfies_gauss_law(model):
    assert all(model._gauss_ok(cfg) for cfg in model.basis)


def test_plaquette_preserves_gauss_law(model):
    # Gauge invariance: magnetic term maps physical -> physical.
    assert model.plaquette_preserves_gauss_law()


def test_hamiltonian_hermitian(model):
    H = model.hamiltonian(1.3)
    assert np.allclose(H, H.T, atol=1e-12)


def test_strong_coupling_gap(model):
    # As g -> infinity the minimal Gauss-law excitation on the 2x2 torus is a
    # 2-link winding loop with energy g^2; the global gap must approach g^2.
    g = 8.0
    ref = ExactReference(model, g=g)
    rel = abs(ref.spectral.sector_gap - g * g) / (g * g)
    assert rel < 0.01, f"strong-coupling gap {ref.spectral.sector_gap} != g^2={g*g}"


def test_variational_lambda_convergence():
    # Enlarging the electric truncation can only lower the ground energy.
    g = 1.0
    e1 = ExactReference(U1Model(U1Lattice(2, 2, 1)), g=g).E0
    e2 = ExactReference(U1Model(U1Lattice(2, 2, 2)), g=g).E0
    assert e2 <= e1 + 1e-9


def test_channel_gap_positive(ref):
    assert ref.mass_gap() > 0
    assert ref.spectral.variance > 0


# -- surrogate construction ------------------------------------------------
def test_control_preserves_equal_time_variance(ref):
    ctl = SpectralShortcutControl(ref.spectral, mode="spurious_slow", rho=0.6, frac=0.03)
    _, _, w = ctl.spectral()
    assert abs(w.sum() - ref.spectral.variance) < 1e-12


def test_control_reports_wrong_gap(ref):
    ctl = SpectralShortcutControl(ref.spectral, mode="spurious_slow", rho=0.6, frac=0.03)
    # spurious slow mode at 0.6*gap must set the asymptotic decay rate.
    assert ctl.asymptotic_gap() < 0.8 * ref.mass_gap()


# -- gate behaviour (headline) --------------------------------------------
def test_gate_certifies_identity(ref):
    v = certify(ref, IdentitySurrogate(ref.spectral))
    assert v.conventional_pass and v.spectral_pass


def test_gate_certifies_ordinary(ref):
    v = certify(ref, ReducedTransferSurrogate(ref.spectral, var_keep=0.9999))
    assert v.conventional_pass and v.spectral_pass


def test_gate_flags_control_as_confound(ref):
    v = certify(ref, SpectralShortcutControl(ref.spectral, mode="spurious_slow"))
    assert v.conventional_pass, "control must pass conventional gates"
    assert not v.spectral_pass, "control must fail at least one spectral gate"
    assert v.label.startswith("CONFOUND")


@pytest.mark.parametrize("g", [0.8, 1.0, 1.2])
def test_confound_stable_over_coupling(model, g):
    ref = ExactReference(model, g=g)
    v = certify(ref, SpectralShortcutControl(ref.spectral, mode="spurious_slow"))
    assert v.conventional_pass and not v.spectral_pass
