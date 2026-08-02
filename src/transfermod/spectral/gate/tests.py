"""Preregistered spectral-fidelity gate.

Each test returns a ``GateResult``. Tests are tagged ``conventional`` or
``spectral``. The headline claim of the benchmark is that the constructed
control passes every ``conventional`` test and fails at least one ``spectral``
test, while an honest reduced-transfer surrogate passes both classes.

Conventional tests deliberately mirror what an ML surrogate is trained on and
reports: equal-time observables and a signal-weighted / aggregate correlator
error. Spectral tests probe the low-signal long-distance tail that determines
the mass gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..observables.suite import TS, FrozenObservableSuite, ObservableSet
from transfermod import certification

from .thresholds import DEFAULT, Thresholds


# Information-basis labels describe which measured information a gate uses.
# They are interpretive metadata, not claims of statistical independence.
BASIS_EQUAL_TIME = "equal_time_observables"
BASIS_CORRELATOR_WINDOW = "aggregate_correlator_window"
BASIS_CORRELATOR_TAIL = "long_distance_correlator_tail"
BASIS_SPECTRAL_DECOMPOSITION = "explicit_spectral_decomposition"


@dataclass
class GateResult:
    """Outcome of one gate test.

    ``status`` is three-valued. ``information_basis`` records the measured
    information on which the test depends; distinct labels do not imply
    statistical independence. "unmeasured" is *not* a pass: a quantity the
    surrogate does not expose has not been verified, and treating absence of
    evidence as evidence of accuracy would invert the whole point of the
    framework. Unmeasured spectral tests downgrade the verdict to
    INSUFFICIENT_EVIDENCE rather than silently certifying.
    """

    name: str
    kind: str        # "conventional" or "spectral"
    value: float
    threshold: float
    status: str      # "pass" | "fail" | "unmeasured"
    information_basis: str = "unspecified"

    @property
    def passed(self) -> bool:
        """Backwards-compatible: True only for an actual pass."""
        return self.status == "pass"

    @property
    def measured(self) -> bool:
        return self.status != "unmeasured"


def _status(value: float, threshold: float) -> str:
    if not np.isfinite(value):
        return "unmeasured"
    return "pass" if value <= threshold else "fail"


def _rel(a: float, b: float) -> float:
    denom = abs(b) if abs(b) > 1e-30 else 1e-30
    return abs(a - b) / denom


def run_gate(ref_obs: ObservableSet, sur_obs: ObservableSet,
             ref_gap: float, thr: Thresholds = DEFAULT) -> list[GateResult]:
    """Compare a surrogate's observables to the reference. ``ref_gap`` is the
    reference's exact channel gap (ground truth)."""
    res: list[GateResult] = []

    # -- G1 equal-time fidelity (conventional) ------------------------------
    v = _rel(sur_obs.exp_val, ref_obs.exp_val)
    res.append(GateResult("G1a_equal_time_expectation", "conventional",
                          v, thr.tau_equal_time_rel, _status(v, thr.tau_equal_time_rel),
                          BASIS_EQUAL_TIME))
    v = _rel(sur_obs.variance, ref_obs.variance)
    res.append(GateResult("G1b_equal_time_variance", "conventional",
                          v, thr.tau_variance_rel, _status(v, thr.tau_variance_rel),
                          BASIS_EQUAL_TIME))

    # -- G2 aggregate correlator fidelity (conventional) --------------------
    scale = ref_obs.correlator[0]
    agg = float(np.sqrt(np.mean((sur_obs.correlator - ref_obs.correlator) ** 2)) / scale)
    res.append(GateResult("G2_aggregate_correlator_rmse", "conventional",
                          agg, thr.tau_aggregate_rmse, _status(agg, thr.tau_aggregate_rmse),
                          BASIS_CORRELATOR_WINDOW))

    # -- G3 channel-gap fidelity (spectral) ---------------------------------
    v = _rel(sur_obs.asymptotic_gap, ref_gap)
    res.append(GateResult("G3_channel_gap", "spectral",
                          v, thr.tau_gap_rel, _status(v, thr.tau_gap_rel),
                          BASIS_SPECTRAL_DECOMPOSITION if not np.isnan(sur_obs.moments[0])
                          else BASIS_CORRELATOR_TAIL))

    # -- G4 effective-mass plateau fidelity (spectral) ----------------------
    v = _rel(sur_obs.plateau_gap, ref_obs.plateau_gap)
    res.append(GateResult("G4_plateau_effective_mass", "spectral",
                          v, thr.tau_plateau_rel, _status(v, thr.tau_plateau_rel),
                          BASIS_CORRELATOR_TAIL))

    # -- G5 long-distance decay fidelity (spectral) -------------------------
    # relative correlator error over the tail (t >= 2), where the gap lives.
    tail = TS >= 2.0
    rel_tail = np.abs(sur_obs.correlator - ref_obs.correlator)[tail] / np.abs(ref_obs.correlator)[tail]
    v = float(rel_tail.max())
    res.append(GateResult("G5_long_distance_decay", "spectral",
                          v, thr.tau_longdist_rel, _status(v, thr.tau_longdist_rel),
                          BASIS_CORRELATOR_TAIL))

    # -- G6 channel / overlap alignment (spectral) --------------------------
    # leading spectral weight (coupling to the light state) must be preserved.
    ref_w = ref_obs.moments[0]        # mu_0 = total connected weight
    sur_w = sur_obs.moments[0]
    # A measured-only surrogate does not expose the leading spectral weight.
    # That is "unmeasured", not "pass": see GateResult.
    v = float("nan") if np.isnan(sur_w) else _rel(sur_w, ref_w)
    res.append(GateResult("G6_overlap_alignment", "spectral",
                          v, thr.tau_overlap_rel, _status(v, thr.tau_overlap_rel),
                          BASIS_SPECTRAL_DECOMPOSITION))
    return res


@dataclass
class Verdict:
    surrogate: str
    results: list[GateResult]

    @property
    def information_bases(self) -> tuple[str, ...]:
        """Distinct measured information bases represented by the gate results.

        This is structural metadata only. A count of bases is not an effective
        sample size and does not assert independence.
        """
        return tuple(sorted({r.information_basis for r in self.results}))

    @property
    def measured_information_bases(self) -> tuple[str, ...]:
        """Distinct information bases among measured gate results."""
        return tuple(sorted({r.information_basis for r in self.results if r.measured}))

    @property
    def conventional_pass(self) -> bool:
        return all(r.status == "pass" for r in self.results
                   if r.kind == "conventional")

    @property
    def spectral_fail(self) -> bool:
        return any(r.status == "fail" for r in self.results
                   if r.kind == "spectral")

    @property
    def spectral_unmeasured(self) -> bool:
        return any(r.status == "unmeasured" for r in self.results
                   if r.kind == "spectral")

    @property
    def spectral_pass(self) -> bool:
        """All spectral tests measured and passing."""
        return not self.spectral_fail and not self.spectral_unmeasured

    @property
    def code(self) -> str:
        """Domain-free verdict code; compare against transfermod.certification."""
        if not self.conventional_pass:
            return certification.REJECTED
        if self.spectral_fail:
            return certification.CONFOUND
        if self.spectral_unmeasured:
            return certification.INSUFFICIENT_EVIDENCE
        return certification.CERTIFIED

    @property
    def explanation(self) -> str:
        base = certification.EXPLANATION[self.code]
        if self.code == certification.INSUFFICIENT_EVIDENCE:
            missing = ", ".join(r.name for r in self.results
                                if r.status == "unmeasured")
            return f"{base}: {missing}"
        return base

    @property
    def label(self) -> str:
        """Human-readable ``CODE (explanation)``. Compare ``code``, not this."""
        return f"{self.code} ({self.explanation})"


def certify(model_ref, surrogate, thr: Thresholds = DEFAULT) -> Verdict:
    """Full pipeline: measure reference + surrogate, run the gate, return verdict."""
    suite = FrozenObservableSuite()
    ref_obs = suite.measure(model_ref)
    sur_obs = suite.measure(surrogate)
    ref_gap = model_ref.mass_gap()
    results = run_gate(ref_obs, sur_obs, ref_gap, thr)
    return Verdict(surrogate=getattr(surrogate, "name", "surrogate"), results=results)
