"""Domain-independent fidelity moduli.

Nothing here refers to a lattice, a correlator, or a spectrum. The objects are
the ones defined in the methods note: given a declared discrepancy geometry on decisions and a
divergence on inputs, the fidelity modulus is the pointwise-minimal transfer
guarantee, and bisection locates its thresholds when the relevant statistics are
monotone in a scalar amplitude.
"""

from transfermod.modulus.core import (
    ABOVE_BRACKET,
    BELOW_BRACKET,
    CROSSED,
    CensoredScalar,
    DEFAULT_BRACKET_LO,
    DEFAULT_BRACKET_HI,
    Threshold,
    bisect_threshold,
    blind_spot_width,
    censoring,
    certified_grid_max,
    condition_number,
    interval_width,
    linearity_ratio,
    modulus,
    ray_modulus,
    ray_decision_modulus,
    certify_ray_modulus,
    log_amplitude_silent_risk_width,
    second_order,
    signature,
)
from transfermod.modulus.discrepancy import (
    CertificationGeometry,
    DecisionDiscrepancy,
    absolute_discrepancy,
    decision_diameter,
    named_discrepancy,
    relative_discrepancy,
    silent_risk_measure,
    stabilized_relative_discrepancy,
    symmetric_relative_discrepancy,
)

__all__ = ["ray_modulus", "ray_decision_modulus", "certify_ray_modulus", "modulus", "interval_width", "blind_spot_width",
           "signature", "log_amplitude_silent_risk_width", "second_order", "censoring", "certified_grid_max", "condition_number", "linearity_ratio", "bisect_threshold", "Threshold",
           "CROSSED", "ABOVE_BRACKET", "BELOW_BRACKET", "CensoredScalar", "DEFAULT_BRACKET_LO", "DEFAULT_BRACKET_HI",
           "DecisionDiscrepancy", "CertificationGeometry",
           "absolute_discrepancy", "relative_discrepancy",
           "stabilized_relative_discrepancy",
           "symmetric_relative_discrepancy", "named_discrepancy",
           "decision_diameter", "silent_risk_measure"]
