"""Uniform runtime warnings for deprecated certification facades."""

import importlib
import warnings

import pytest


def test_bound_status_access_warns():
    certification = importlib.import_module("transfermod.certification")
    with pytest.deprecated_call(match="BoundStatus"):
        status_type = certification.BoundStatus
    assert status_type.EXACT.value == "exact"


def test_coverage_proof_access_warns():
    certification = importlib.import_module("transfermod.certification")
    with pytest.deprecated_call(match="CoverageProof"):
        coverage_type = certification.CoverageProof
    assert coverage_type.__name__ == "Coverage"


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("ProvenCoverage", ("Theorem X",)),
        ("UnprovenCoverage", ("family",)),
        ("ExploratorySampleCoverage", ("family", "sample")),
    ],
)
def test_constructor_facades_warn(name, args):
    certification = importlib.import_module("transfermod.certification")
    constructor = getattr(certification, name)
    with pytest.deprecated_call(match=name):
        coverage = constructor(*args)
    assert coverage is not None
