"""Deprecated alias for :mod:`transfermod.spectral`.

The package was renamed when the mathematics was generalised from spectral
fidelity to transfer guarantees; the spectral benchmark is now one reference
application of the framework. This shim keeps existing imports working.
"""

import sys
import warnings

from transfermod import spectral as _spectral

warnings.warn(
    "sfbench has been renamed; import transfermod.spectral instead.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _spectral
