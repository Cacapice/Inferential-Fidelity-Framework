"""Surrogate families judged by the benchmark (see ``sfbench.taxonomy``)."""

from __future__ import annotations

from .base import Surrogate, IdentitySurrogate

# Class II -- reduced representations
from .reduced_transfer import ReducedTransferSurrogate
from .pod_galerkin import PODGalerkinSurrogate

# Class III -- learned surrogates
from .trained import TrainedMultiExpSurrogate
from .neural import NeuralQuantumStateSurrogate

# Class IV -- adversarial controls
from .spectral_shortcut import SpectralShortcutControl

# Class III extension points (not yet implemented; see module docstrings)
from .operator import NeuralOperatorSurrogate
from .deeponet import DeepONetSurrogate
from .fno import FNOSurrogate

__all__ = [
    "Surrogate", "IdentitySurrogate",
    "ReducedTransferSurrogate", "PODGalerkinSurrogate",
    "TrainedMultiExpSurrogate", "NeuralQuantumStateSurrogate",
    "SpectralShortcutControl",
    "NeuralOperatorSurrogate", "DeepONetSurrogate", "FNOSurrogate",
]
