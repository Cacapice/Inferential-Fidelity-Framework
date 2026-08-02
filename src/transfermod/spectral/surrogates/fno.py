"""Fourier Neural Operator surrogates (Class III) -- extension point, not yet implemented.

See ``sfbench.surrogates.operator`` for the full Class III contract. This module
marks the intended plug-in site for Fourier Neural Operator models; instantiating the class raises
``NotImplementedError`` rather than returning a fake result.
"""

from __future__ import annotations

from .base import Surrogate


class FNOSurrogate(Surrogate):
    """Extension point for Fourier Neural Operator surrogates. Not yet implemented."""

    name = "fno(learned)"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FNOSurrogate is an extension point, not an implementation. "
            "Subclass sfbench.surrogates.base.Surrogate and provide measured "
            "exp_val(), variance(), and correlator(ts); see "
            "sfbench/surrogates/operator.py for the contract."
        )

    def spectral(self):  # pragma: no cover - unreachable until implemented
        raise NotImplementedError
