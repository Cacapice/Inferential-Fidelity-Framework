"""Frozen certification thresholds.

These are preregistered. Justification for each is in ``PREREGISTRATION.md``.
Changing any value is a substantive amendment and must be logged formally.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    # -- conventional gates (short-distance / equal-time / aggregate) --------
    tau_equal_time_rel: float = 1.0e-2   # G1: |<O>_sur - <O>_ref| / |<O>_ref|
    tau_variance_rel: float = 1.0e-2     # G1: relative equal-time variance error
    tau_aggregate_rmse: float = 5.0e-3   # G2: signal-normalised aggregate correlator RMSE

    # -- spectral gates (long-distance / gap) --------------------------------
    tau_gap_rel: float = 0.10            # G3: relative channel-gap error vs reference
    tau_plateau_rel: float = 0.10        # G4: relative effective-mass plateau error
    tau_longdist_rel: float = 0.15       # G5: max relative correlator error over the tail
    tau_overlap_rel: float = 0.15        # G6: relative leading-overlap (channel) error


DEFAULT = Thresholds()
