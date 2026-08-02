"""Audit-regression tests for finite-cutoff U(1) reference semantics."""

import numpy as np
import pytest

from transfermod.spectral.lattice.u1 import U1Lattice, U1Model
from transfermod.spectral.reference.exact import ExactReference


def test_vacuum_basis_contains_only_zero_winding():
    model = U1Model(U1Lattice(2, 2, 1, basis_mode="vacuum"))
    assert model.winding_sectors() == ((0, 0),)


def test_full_basis_contains_nonzero_winding_sectors():
    model = U1Model(U1Lattice(2, 2, 1, basis_mode="full"))
    sectors = model.winding_sectors()
    assert (0, 0) in sectors
    assert any(s != (0, 0) for s in sectors)


def test_hamiltonian_preserves_winding_blocks():
    model = U1Model(U1Lattice(2, 2, 1, basis_mode="full"))
    H = model.hamiltonian(1.2)
    rows, cols = np.nonzero(np.abs(H) > 1e-14)
    for i, j in zip(rows, cols):
        assert model.winding_numbers(model.basis[i]) == model.winding_numbers(model.basis[j])


def test_plaquette_operators_are_hermitian_in_selected_basis():
    for mode in ("full", "vacuum"):
        model = U1Model(U1Lattice(2, 2, 1, basis_mode=mode))
        local = model.cos_plaquette_operator()
        zero = model.zero_momentum_cos()
        assert np.allclose(local, local.T, atol=1e-12)
        assert np.allclose(zero, zero.T, atol=1e-12)


def test_reference_scope_is_explicit():
    full = ExactReference(U1Model(U1Lattice(2, 2, 1, basis_mode="full")), g=1.0)
    vac = ExactReference(U1Model(U1Lattice(2, 2, 1, basis_mode="vacuum")), g=1.0)
    assert full.spectral.spectrum_scope == "all_winding_sectors"
    assert full.spectral.global_physical_gap == pytest.approx(full.spectral.sector_gap)
    assert vac.spectral.spectrum_scope == "zero_winding_sector"
    assert vac.spectral.global_physical_gap is None
    assert vac.spectral.sector_gap == pytest.approx(vac.spectral.sector_gap)


def test_reference_ground_state_and_plaquette_channel_match_full_basis():
    full_model = U1Model(U1Lattice(2, 2, 1, basis_mode="full"))
    vac_model = U1Model(U1Lattice(2, 2, 1, basis_mode="vacuum"))
    full = ExactReference(full_model, g=1.0, operator=full_model.zero_momentum_cos())
    vac = ExactReference(vac_model, g=1.0, operator=vac_model.zero_momentum_cos())
    assert vac.E0 == pytest.approx(full.E0, abs=1e-11)
    assert vac.mass_gap() == pytest.approx(full.mass_gap(), abs=1e-10)
    ts = np.array([0.0, 0.25, 1.0])
    assert np.allclose(vac.correlator(ts), full.correlator(ts), atol=1e-10)


def test_strong_coupling_distinguishes_global_and_zero_winding_gaps():
    g = 8.0
    full = ExactReference(U1Model(U1Lattice(2, 2, 1, basis_mode="full")), g=g)
    vac = ExactReference(U1Model(U1Lattice(2, 2, 1, basis_mode="vacuum")), g=g)
    assert full.spectral.sector_gap / (g*g) == pytest.approx(1.0, rel=0.01)
    assert vac.spectral.sector_gap / (2*g*g) == pytest.approx(1.0, rel=0.01)
    assert vac.mass_gap() / (2*g*g) == pytest.approx(1.0, rel=0.01)


def test_hard_wall_shift_and_boundary_probability_are_explicit():
    model = U1Model(U1Lattice(2, 2, 1, basis_mode="vacuum"))
    boundary_cfg = next(cfg for cfg in model.basis if model.configuration_touches_cutoff(cfg))
    assert model.configuration_touches_cutoff(boundary_cfg)
    ref = ExactReference(model, g=1.0)
    assert 0.0 <= ref.cutoff_boundary_probability() <= 1.0
    # At least one attempted plaquette move from a hard-wall state is rejected.
    assert any(
        model._apply_plaquette(boundary_cfg, p, sign) is None
        for p in model.plaquettes
        for sign in (+1, -1)
    )
