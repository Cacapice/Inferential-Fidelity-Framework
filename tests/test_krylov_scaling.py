"""Tests for the sparse/Krylov path, the vacuum-sector basis, and the trained surrogate."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transfermod.spectral.gate.tests import certify
from transfermod.spectral.lattice.u1 import U1Lattice, U1Model
from transfermod.spectral.observables.suite import TS
from transfermod.spectral.reference.exact import ExactReference
from transfermod.spectral.reference.krylov import KrylovReference
from transfermod.spectral.surrogates import SpectralShortcutControl, TrainedMultiExpSurrogate


def test_vacuum_basis_matches_full_channel():
    # The plaquette channel lives in the vacuum sector: gap must match the full basis.
    full = ExactReference(U1Model(U1Lattice(2, 2, 2, basis_mode="full")), g=1.0)
    vac = ExactReference(U1Model(U1Lattice(2, 2, 2, basis_mode="vacuum")), g=1.0)
    assert abs(full.mass_gap() - vac.mass_gap()) < 1e-9
    assert abs(full.exp_val() - vac.exp_val()) < 1e-9


def test_krylov_matches_dense():
    dense = ExactReference(U1Model(U1Lattice(2, 2, 2, basis_mode="full")), g=1.0)
    kry = KrylovReference(U1Model(U1Lattice(2, 2, 2, basis_mode="vacuum")), g=1.0)
    assert abs(dense.mass_gap() - kry.mass_gap()) < 1e-8
    assert abs(dense.variance() - kry.variance()) < 1e-8
    c_d, c_k = dense.correlator(TS), kry.correlator(TS)
    assert np.abs(c_d - c_k).max() / c_d[0] < 1e-8


def test_larger_volume_weak_light_state():
    # 3x3 (Lambda=1) at g=0.8 has a naturally weak light state (< 10% weight).
    ref = KrylovReference(U1Model(U1Lattice(3, 3, 1, basis_mode="vacuum")), g=0.8)
    sd = ref.spectral
    assert sd.weights[0] / sd.weights.sum() < 0.10


def test_natural_attenuate_tail_confound():
    ref = KrylovReference(U1Model(U1Lattice(3, 3, 1, basis_mode="vacuum")), g=0.8)
    ctl = SpectralShortcutControl(ref.spectral, mode="attenuate_tail", attenuation=0.0)
    v = certify(ref, ctl)
    assert v.conventional_pass and not v.spectral_pass
    # over-estimate: reported gap exceeds the true gap
    from transfermod.spectral.observables.suite import FrozenObservableSuite
    assert FrozenObservableSuite().measure(ctl).asymptotic_gap > ref.mass_gap()


def test_trained_surrogate_certified_with_tail_signal():
    ref = KrylovReference(U1Model(U1Lattice(3, 3, 1, basis_mode="vacuum")), g=0.8)
    sur = TrainedMultiExpSurrogate(ref, TS, rel_noise=0.001, noise_model="relative", seed=3)
    v = certify(ref, sur)
    assert v.conventional_pass and v.spectral_pass  # recovers the gap


def test_trained_surrogate_confounded_under_noise_floor():
    ref = KrylovReference(U1Model(U1Lattice(3, 3, 1, basis_mode="vacuum")), g=0.8)
    sur = TrainedMultiExpSurrogate(ref, TS, rel_noise=0.002, noise_model="absolute_floor", seed=3)
    v = certify(ref, sur)
    assert v.conventional_pass and not v.spectral_pass  # confound emerges from training


def test_neural_quantum_state_certified():
    # A real VMC-trained NQS should converge and be certified (no false positive).
    from transfermod.spectral.surrogates import NeuralQuantumStateSurrogate
    m = U1Model(U1Lattice(2, 2, 2, basis_mode="vacuum"))
    ref = ExactReference(m, g=1.0)
    nqs = NeuralQuantumStateSurrogate(m, g=1.0, hidden=24, epochs=1500, seed=0)
    assert abs(nqs.energy - ref.E0) < 5e-3          # trained to a good ground state
    assert abs(nqs.exp_val() - ref.exp_val()) < 1e-2
    v = certify(ref, nqs)
    assert v.conventional_pass and v.spectral_pass  # spectrally faithful


def test_neural_quantum_state_interface():
    from transfermod.spectral.surrogates import NeuralQuantumStateSurrogate
    m = U1Model(U1Lattice(2, 2, 2, basis_mode="vacuum"))
    nqs = NeuralQuantumStateSurrogate(m, g=1.0, hidden=8, epochs=200, seed=1)
    C = nqs.correlator(TS)
    assert C.shape == TS.shape and np.all(np.isfinite(C)) and C[0] > 0
    e, d, w = nqs.spectral()
    assert d.size == w.size and np.all(d > 0)


def test_reduced_order_model_certified_at_high_rank():
    from transfermod.spectral.surrogates import PODGalerkinSurrogate
    m = U1Model(U1Lattice(2, 2, 2, basis_mode="vacuum"))
    ref = ExactReference(m, g=1.0)
    rom = PODGalerkinSurrogate(m, g=1.0, rank=20)
    v = certify(ref, rom)
    assert v.conventional_pass and v.spectral_pass
    assert abs(rom.asymptotic_gap() - ref.mass_gap()) / ref.mass_gap() < 0.02


def test_underresolved_rom_is_a_natural_confound():
    # An honest but under-resolved ROM matches low moments exactly (conventional
    # pass) while Rayleigh-Ritz over-estimates the gap (spectral fail).
    from transfermod.spectral.surrogates import PODGalerkinSurrogate
    m = U1Model(U1Lattice(2, 2, 2, basis_mode="vacuum"))
    ref = ExactReference(m, g=1.0)
    rom = PODGalerkinSurrogate(m, g=1.0, rank=3)
    # equal-time variance preserved by Krylov moment matching
    assert abs(rom.variance() - ref.variance()) / ref.variance() < 1e-8
    v = certify(ref, rom)
    assert v.conventional_pass and not v.spectral_pass
    # Rayleigh-Ritz converges from above -> over-estimate
    assert rom.asymptotic_gap() > ref.mass_gap()
