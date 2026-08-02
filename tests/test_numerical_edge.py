"""Numerical-precision and edge-case tests.

Spectral estimators and the Lanczos/eigsh path can be sensitive to conditioning:
near-degenerate spectra, low-rank collapse, vanishing or extreme weights, and
correlators that underflow to zero. These tests assert the pipeline degrades
gracefully (well-defined output or NaN, never an exception or a spurious
"CERTIFIED") rather than silently producing garbage.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transfermod.spectral.analysis.estimators import ESTIMATORS, prony_gap
from transfermod.spectral.gate.tests import certify
from transfermod.spectral.lattice.u1 import U1Lattice, U1Model
from transfermod.spectral.observables.suite import TS, effective_mass, plateau_mass
from transfermod.spectral.reference.exact import ExactReference, SpectralData
from transfermod.spectral.reference.krylov import KrylovReference
from transfermod.spectral.surrogates import IdentitySurrogate, ReducedTransferSurrogate


def _spectral(deltas, weights, exp_val=0.5):
    d = np.asarray(deltas, float)
    w = np.asarray(weights, float)
    return SpectralData(exp_val=exp_val, variance=float(w.sum()), deltas=d, weights=w,
                        sector_gap=float(d.min()), channel_gap=float(d.min()))


# -- correlator / effective-mass guards ------------------------------------
def test_effective_mass_handles_nonpositive():
    # A correlator that hits zero / negative must yield NaN, not raise.
    C = np.array([1.0, 0.5, 0.0, -0.1, 1e-300, 1e-320, 0.0, 0.0, 0.0])
    em = effective_mass(C)
    assert em.shape == (8,)
    assert np.isnan(em[1:]).any()  # nonpositive entries -> NaN
    assert np.isfinite(em[0])


def test_plateau_all_nan_returns_nan():
    C = np.zeros(9)
    assert np.isnan(plateau_mass(C))


# -- near-degenerate spectrum ----------------------------------------------
def test_near_degenerate_modes_do_not_crash_estimators():
    # Two almost-coincident low modes: ill-conditioned for Prony / multi-exp.
    sd = _spectral([1.0000, 1.0000001, 3.0], [0.2, 0.2, 0.6])
    C = sd.correlator(TS)
    for name, fn in ESTIMATORS.items():
        val = fn(TS, C)
        assert val is not None and (np.isnan(val) or val > 0), name
    # the well-behaved tail estimators should still land near the true slow rate
    assert abs(prony_gap(TS, C) - 1.0) < 0.2 or np.isnan(prony_gap(TS, C))


def test_identity_certified_on_degenerate_spectrum():
    sd = _spectral([1.0, 1.0000001, 3.0], [0.2, 0.2, 0.6])
    v = certify_from_spectral(sd)
    assert v.conventional_pass and v.spectral_pass


def certify_from_spectral(sd):
    class _Ref:
        def __init__(self, s):
            self._s = s
        spectral = property(lambda self: self._s)
        def spectral_repr(self):
            return self._s.exp_val, self._s.deltas.copy(), self._s.weights.copy()
        def correlator(self, ts):
            return self._s.correlator(ts)
        def exp_val(self):
            return self._s.exp_val
        def variance(self):
            return self._s.variance
        def mass_gap(self):
            return self._s.channel_gap
        def asymptotic_gap(self):
            return self._s.channel_gap
    ref = _Ref(sd)
    return certify(ref, IdentitySurrogate(sd))


# -- extreme / vanishing weights -------------------------------------------
def test_tiny_weight_mode_is_filtered_not_fatal():
    # A mode with negligible weight must not dominate or break the pipeline.
    sd = _spectral([0.05, 2.0], [1e-15, 0.5])
    C = sd.correlator(TS)
    assert np.all(np.isfinite(C))
    # identity still certifies (the tiny mode is physically present but negligible)
    v = certify_from_spectral(sd)
    assert v.conventional_pass


def test_extreme_weight_ratio_correlator_finite():
    sd = _spectral([0.5, 5.0], [1e6, 1e-6])
    C = sd.correlator(TS)
    assert np.all(np.isfinite(C)) and C[0] > 0


# -- low-rank collapse ------------------------------------------------------
def test_reduced_transfer_rank_one_collapse_is_handled():
    ref = ExactReference(U1Model(U1Lattice(2, 2, 2, basis_mode="vacuum")), g=1.0)
    # Force collapse to a single retained mode.
    sur = ReducedTransferSurrogate(ref.spectral, K=1)
    v = certify(ref, sur)  # must not raise
    # a rank-1 truncation loses variance -> should NOT be silently certified
    assert not (v.conventional_pass and v.spectral_pass)


def test_prony_short_or_constant_data_returns_nan_gracefully():
    assert np.isnan(prony_gap(np.array([0.0, 1.0]), np.array([1.0, 1.0])))
    assert np.isnan(prony_gap(TS, np.ones_like(TS, dtype=float)))  # no decay


# -- Krylov conditioning on the smallest system ----------------------------
def test_krylov_matches_dense_small_illconditioned():
    # 2x2 Lambda=1 (dim 19): few modes, Lanczos must still match dense.
    dense = ExactReference(U1Model(U1Lattice(2, 2, 1, basis_mode="vacuum")), g=1.0)
    kry = KrylovReference(U1Model(U1Lattice(2, 2, 1, basis_mode="vacuum")), g=1.0)
    assert abs(dense.mass_gap() - kry.mass_gap()) < 1e-8


# -- taxonomy / extension points -------------------------------------------
def test_taxonomy_classifies_each_family():
    from transfermod.spectral import classify
    from transfermod.spectral.reference.krylov import KrylovReference as _K
    from transfermod.spectral.surrogates import (PODGalerkinSurrogate, NeuralQuantumStateSurrogate,
                                    SpectralShortcutControl)
    assert classify("ExactReference").numeral == "I"
    assert classify("KrylovReference").numeral == "I"
    assert classify("PODGalerkinSurrogate").numeral == "II"
    assert classify("NeuralQuantumStateSurrogate").numeral == "III"
    assert classify("SpectralShortcutControl").numeral == "IV"


def test_extension_points_raise_not_implemented():
    from transfermod.spectral.surrogates import (NeuralOperatorSurrogate, DeepONetSurrogate,
                                    FNOSurrogate)
    for cls in (NeuralOperatorSurrogate, DeepONetSurrogate, FNOSurrogate):
        with pytest.raises(NotImplementedError):
            cls()


# -- surrogate interface contract ------------------------------------------
def test_surrogate_contract_and_metadata():
    """Every concrete surrogate must satisfy the three-method contract."""
    from transfermod.spectral import build_reference
    from transfermod.spectral.surrogates import (IdentitySurrogate, ReducedTransferSurrogate,
                                    PODGalerkinSurrogate, SpectralShortcutControl)
    from transfermod.spectral.surrogates.base import SurrogateMetadata
    ref = build_reference(2, 2, 2, 1.0)
    surrogates = [
        IdentitySurrogate(ref.spectral),
        ReducedTransferSurrogate(ref.spectral),
        PODGalerkinSurrogate(ref.model, 1.0, rank=6),
        SpectralShortcutControl(ref.spectral, mode="spurious_slow"),
    ]
    for s in surrogates:
        C = s.correlator(TS)
        assert C.shape == TS.shape and np.all(np.isfinite(C)), s.name
        assert s.variance() > 0, s.name
        md = s.metadata()
        assert isinstance(md, SurrogateMetadata) and md.name == s.name
        d = md.as_dict()
        assert set(d) == {"name", "family", "provenance", "training_info"}
        assert md.family != "unspecified", f"{s.name} not registered in taxonomy"


def test_metadata_carries_provenance_for_constructed_control():
    from transfermod.spectral import build_reference
    from transfermod.spectral.surrogates import SpectralShortcutControl
    ref = build_reference(2, 2, 2, 1.0)
    ctl = SpectralShortcutControl(ref.spectral, mode="spurious_slow", rho=0.6, frac=0.03)
    prov = ctl.metadata().provenance
    assert prov["mechanism"] == "spurious_slow"
    assert prov["rho"] == 0.6 and prov["frac"] == 0.03


def test_default_bracket_is_declared_as_library_assumption():
    from transfermod.modulus import bisect_threshold

    threshold = bisect_threshold(lambda w: w, 0.5)
    payload = threshold.to_dict()
    assert payload["search_domain"] == [1e-14, 1.0]
    assert payload["bracket"][0] < 0.5 <= payload["bracket"][1]
    assert payload["bracket_source"] == "library_default"
    assert "modeling assumption" in payload["model_assumption"]


def test_explicit_bracket_is_declared_as_caller_supplied():
    from transfermod.modulus import bisect_threshold

    threshold = bisect_threshold(lambda w: w, 0.5, lo=1e-6, hi=2.0)
    payload = threshold.to_dict()
    assert payload["bracket_source"] == "caller"
    assert "explicitly supplied" in payload["model_assumption"]
