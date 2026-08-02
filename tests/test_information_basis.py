"""Information-basis metadata must remain structural and inspectable."""

import numpy as np

from transfermod.spectral import build_reference, certify
from transfermod.spectral.gate import (
    BASIS_CORRELATOR_TAIL,
    BASIS_CORRELATOR_WINDOW,
    BASIS_EQUAL_TIME,
    BASIS_SPECTRAL_DECOMPOSITION,
)
from transfermod.spectral.surrogates import IdentitySurrogate


def test_gate_results_expose_information_basis():
    ref = build_reference(Lx=2, Ly=2, Lambda=1, g=1.0)
    verdict = certify(ref, IdentitySurrogate(ref.spectral))
    assert all(result.information_basis != "unspecified" for result in verdict.results)


def test_expected_information_bases_are_present():
    ref = build_reference(Lx=2, Ly=2, Lambda=1, g=1.0)
    verdict = certify(ref, IdentitySurrogate(ref.spectral))
    assert set(verdict.information_bases) == {
        BASIS_EQUAL_TIME,
        BASIS_CORRELATOR_WINDOW,
        BASIS_CORRELATOR_TAIL,
        BASIS_SPECTRAL_DECOMPOSITION,
    }


def test_repeated_basis_labels_do_not_change_verdict_semantics():
    ref = build_reference(Lx=2, Ly=2, Lambda=1, g=1.0)
    verdict = certify(ref, IdentitySurrogate(ref.spectral))
    assert verdict.code == "CERTIFIED"
    assert len(verdict.results) > len(verdict.information_bases)


def test_measured_information_bases_exclude_unmeasured_results():
    ref = build_reference(Lx=2, Ly=2, Lambda=1, g=1.0)
    verdict = certify(ref, IdentitySurrogate(ref.spectral))
    assert verdict.measured_information_bases == verdict.information_bases
