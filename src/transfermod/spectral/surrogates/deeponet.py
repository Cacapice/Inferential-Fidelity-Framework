"""DeepONet (branch/trunk operator-learning) surrogates (Class III) -- extension point, not yet implemented.

See ``sfbench.surrogates.operator`` for the full Class III contract. This module
marks the intended plug-in site for DeepONet (branch/trunk operator-learning) models; instantiating the class raises
``NotImplementedError`` rather than returning a fake result.
"""

from __future__ import annotations

from .base import Surrogate


class DeepONetSurrogate(Surrogate):
    """Extension point for DeepONet (branch/trunk operator-learning) surrogates. Not yet implemented."""

    name = "deeponet(learned)"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "DeepONetSurrogate is an extension point, not an implementation. "
            "Subclass sfbench.surrogates.base.Surrogate and provide measured "
            "exp_val(), variance(), and correlator(ts); see "
            "sfbench/surrogates/operator.py for the contract."
        )

    def spectral(self):  # pragma: no cover - unreachable until implemented
        raise NotImplementedError
