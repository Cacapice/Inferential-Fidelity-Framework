"""Neural-operator surrogates (Class III) -- extension point, not yet implemented.

This module is a deliberate placeholder marking where operator-learning families
plug into the benchmark. It is *not* a stub pretending to work: instantiating
anything here raises ``NotImplementedError`` with the contract a contributor
must satisfy.

Contract for any Class III surrogate
------------------------------------
Subclass ``transfermod.spectral.surrogates.base.Surrogate`` and supply the *measured*
quantities your model produces -- never the reference's:

1. ``exp_val()``   -- equal-time expectation of the benchmark operator.
2. ``variance()``  -- equal-time connected variance ``C(0)``.
3. ``correlator(ts)`` -- the connected two-point function on the frozen grid.
   (Or, if your model exposes an explicit spectral decomposition, implement
   ``spectral()`` returning ``(exp_val, deltas, weights)`` and inherit the rest.)

The gate is agnostic to how these are obtained: forward-evolve your learned
operator, sample your learned distribution, or read them off a learned spectrum.
See ``neural.py`` (variational state) and ``pod_galerkin.py`` (projection) for
worked surrogate examples with different internal structure.

Why this file exists
--------------------
The benchmark is intended to be architecture-agnostic. Naming the intended
extension points makes the framework's scope explicit and keeps the barrier to
contributing a new family low.
"""

from __future__ import annotations

from .base import Surrogate


class NeuralOperatorSurrogate(Surrogate):
    """Extension point for neural-operator surrogates. Not yet implemented."""

    name = "neural-operator(learned)"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "NeuralOperatorSurrogate is an extension point, not an implementation. "
            "Subclass transfermod.spectral.surrogates.base.Surrogate and provide measured "
            "exp_val(), variance(), and correlator(ts); see this module's docstring "
            "for the full contract and neural.py / pod_galerkin.py for examples."
        )

    def spectral(self):  # pragma: no cover - unreachable until implemented
        raise NotImplementedError
