import numpy as np
import pytest

from transfermod.spectral.lattice.u1 import U1Lattice, U1Model
from transfermod.spectral.reference.exact import ExactReference, SpectralData


def test_spectral_data_exact_gap_property_warns():
    data = SpectralData(
        exp_val=0.0,
        variance=1.0,
        deltas=np.array([1.0]),
        weights=np.array([1.0]),
        sector_gap=0.75,
        channel_gap=1.0,
        basis_mode="vacuum",
    )
    with pytest.deprecated_call(match="alias for `sector_gap`"):
        assert data.exact_gap == data.sector_gap


def test_legacy_exact_gap_constructor_warns_and_maps_to_sector_gap():
    with pytest.deprecated_call(match="alias for `sector_gap`"):
        data = SpectralData(
            exp_val=0.0,
            variance=1.0,
            deltas=np.array([1.0]),
            weights=np.array([1.0]),
            exact_gap=0.5,
            channel_gap=1.0,
        )
    assert data.sector_gap == pytest.approx(0.5)


def test_conflicting_gap_names_are_rejected():
    with pytest.raises(ValueError, match="disagree"):
        SpectralData(
            exp_val=0.0,
            variance=1.0,
            deltas=np.array([1.0]),
            weights=np.array([1.0]),
            sector_gap=0.5,
            exact_gap=0.6,
            channel_gap=1.0,
        )


def test_exact_reference_alias_warns():
    ref = ExactReference(U1Model(U1Lattice(2, 2, 1, basis_mode="vacuum")), g=1.0)
    with pytest.deprecated_call(match="deprecated"):
        assert ref.exact_gap == ref.sector_gap
