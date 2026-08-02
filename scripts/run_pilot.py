"""Run the pilot benchmark end-to-end.

Builds the exact reference, three surrogates (identity echo, honest reduced
transfer operator, constructed spectral-shortcut control), runs the frozen
observable suite and the preregistered gate, prints a verdict table, saves
``results/pilot.json`` and ``results/pilot.png``, and runs a small nuisance
sweep over coupling ``g`` and truncation ``Lambda`` to show verdict stability.

Usage::

    python scripts/run_pilot.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transfermod.spectral.gate.tests import certify, run_gate  # noqa: E402
from transfermod.spectral.lattice.u1 import U1Lattice, U1Model  # noqa: E402
from transfermod.spectral.observables.suite import TS, FrozenObservableSuite  # noqa: E402
from transfermod.spectral.reference.exact import ExactReference  # noqa: E402
from transfermod.spectral.surrogates import (  # noqa: E402
    IdentitySurrogate,
    ReducedTransferSurrogate,
    SpectralShortcutControl,
)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def _json_native(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def build_surrogates(ref: ExactReference):
    sd = ref.spectral
    return [
        IdentitySurrogate(sd),
        ReducedTransferSurrogate(sd, var_keep=0.9999, weight_jitter=0.0),
        SpectralShortcutControl(sd, mode="spurious_slow", rho=0.6, frac=0.03),
    ]


def verdict_table(ref, surrogates):
    suite = FrozenObservableSuite()
    ref_obs = suite.measure(ref)
    ref_gap = ref.mass_gap()
    rows = []
    for s in surrogates:
        sur_obs = suite.measure(s)
        results = run_gate(ref_obs, sur_obs, ref_gap)
        conv = all(r.passed for r in results if r.kind == "conventional")
        spec = all(r.passed for r in results if r.kind == "spectral")
        rows.append((s.name, conv, spec, results, sur_obs))
    return ref_obs, ref_gap, rows


def print_report(ref, ref_obs, ref_gap, rows):
    print("=" * 78)
    print("SPECTRAL-FIDELITY BENCHMARK -- PILOT")
    print("=" * 78)
    print(f"Reference: 2+1D compact U(1) Kogut-Susskind, exact diagonalisation")
    print(f"  lattice {ref.model.lat.Lx}x{ref.model.lat.Ly}, Lambda={ref.model.lat.Lambda}, "
          f"g={ref.g}, phys. dim={ref.model.dim}")
    print(f"  exact channel mass gap = {ref_gap:.4f}   "
          f"(selected-basis E1-E0 = {ref.spectral.sector_gap:.4f})")
    print(f"  <cos(plaq)> = {ref_obs.exp_val:.4f}   equal-time var = {ref_obs.variance:.5f}")
    print("-" * 78)
    header = f"{'surrogate':<28}{'conv':>6}{'spec':>6}  {'gap':>8}  verdict"
    print(header)
    print("-" * 78)
    for name, conv, spec, results, sur_obs in rows:
        label = ("CERTIFIED" if conv and spec else
                 "CONFOUND" if conv and not spec else "REJECTED")
        print(f"{name:<28}{_yn(conv):>6}{_yn(spec):>6}  {sur_obs.asymptotic_gap:>8.4f}  {label}")
    print("-" * 78)
    # detailed gate readout for the control
    ctl = [r for r in rows if "shortcut" in r[0]][0]
    print(f"\nGate detail for {ctl[0]}:")
    for r in ctl[3]:
        mark = "pass" if r.passed else "FAIL"
        print(f"  [{r.kind:<12}] {r.name:<30} value={r.value:.3e}  thr={r.threshold:.1e}  {mark}")
    print("=" * 78)


def _yn(b: bool) -> str:
    return "PASS" if b else "fail"


def make_figure(ref, surrogates, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    suite = FrozenObservableSuite()
    ref_obs = suite.measure(ref)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {"identity(echo)": ("o", "#888888"),
              "reduced-transfer(ordinary)": ("s", "#1f77b4"),
              "spectral-shortcut(constructed)": ("^", "#d62728")}
    ax1.semilogy(TS, ref_obs.correlator, "k-", lw=2, label="reference (exact)")
    for s in surrogates:
        o = suite.measure(s)
        m, c = styles.get(s.name, ("x", "gray"))
        ax1.semilogy(TS, o.correlator, m, ms=5, color=c, alpha=0.8, label=s.name)
    ax1.set_xlabel("Euclidean time t"); ax1.set_ylabel("C(t)")
    ax1.set_title("Two-point correlator"); ax1.legend(fontsize=8)
    ax1.axvspan(0, 1, color="green", alpha=0.06)
    ax1.text(0.3, ax1.get_ylim()[1]*0.3, "conventional\n(short-t)", fontsize=7, color="green")

    tsm = TS[:-1] + 0.5
    ax2.plot(tsm, ref_obs.eff_mass, "k-", lw=2, label="reference (exact)")
    ax2.axhline(ref.mass_gap(), color="k", ls=":", lw=1, alpha=0.6, label="true gap")
    for s in surrogates:
        o = suite.measure(s)
        m, c = styles.get(s.name, ("x", "gray"))
        ax2.plot(tsm, o.eff_mass, m + "-", ms=5, color=c, alpha=0.8, label=s.name)
    ax2.set_xlabel("Euclidean time t"); ax2.set_ylabel(r"$m_{\rm eff}(t)$")
    ax2.set_title("Effective mass (plateau = gap)"); ax2.legend(fontsize=8)
    fig.suptitle("Confound-imitating surrogate: conventional accuracy passes, spectral gap fails",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"figure -> {path}")


def nuisance_sweep(lattice_kwargs, gs, lambdas):
    print("\nNuisance sweep (verdict on the constructed control should be stable):")
    print(f"  {'g':>5} {'Lambda':>7} {'dim':>6}  {'conv':>5} {'spec':>5}  gap_ref  gap_ctl")
    out = []
    for Lam in lambdas:
        model = U1Model(U1Lattice(Lambda=Lam, **lattice_kwargs))
        for g in gs:
            ref = ExactReference(model, g)
            ctl = SpectralShortcutControl(ref.spectral, mode="spurious_slow", rho=0.6, frac=0.03)
            v = certify(ref, ctl)
            gap_ctl = FrozenObservableSuite().measure(ctl).asymptotic_gap
            print(f"  {g:>5.2f} {Lam:>7d} {model.dim:>6d}  "
                  f"{_yn(v.conventional_pass):>5} {_yn(v.spectral_pass):>5}  "
                  f"{ref.mass_gap():>7.3f}  {gap_ctl:>7.3f}")
            out.append(dict(g=g, Lambda=Lam, dim=model.dim,
                            conventional_pass=v.conventional_pass,
                            spectral_pass=v.spectral_pass,
                            gap_ref=ref.mass_gap(), gap_ctl=gap_ctl))
    return out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    model = U1Model(U1Lattice(Lx=2, Ly=2, Lambda=2))
    ref = ExactReference(model, g=1.0)
    surrogates = build_surrogates(ref)
    ref_obs, ref_gap, rows = verdict_table(ref, surrogates)
    print_report(ref, ref_obs, ref_gap, rows)
    make_figure(ref, surrogates, os.path.join(RESULTS, "pilot.png"))
    sweep = nuisance_sweep(dict(Lx=2, Ly=2), gs=[0.8, 1.0, 1.2], lambdas=[1, 2])

    payload = {
        "reference": {
            "theory": "2+1D compact U(1) Kogut-Susskind (exact diagonalisation)",
            "lattice": [model.lat.Lx, model.lat.Ly],
            "Lambda": model.lat.Lambda, "g": ref.g, "dim": model.dim,
            "exact_channel_gap": ref_gap, "selected_basis_gap_E1_E0": ref.spectral.sector_gap,
            "plaquette_expectation": ref_obs.exp_val, "equal_time_variance": ref_obs.variance,
        },
        "surrogates": [
            {"name": name, "conventional_pass": conv, "spectral_pass": spec,
             "asymptotic_gap": sur_obs.asymptotic_gap,
             "gate": [{"name": r.name, "kind": r.kind, "value": r.value,
                       "threshold": r.threshold, "passed": r.passed} for r in results]}
            for name, conv, spec, results, sur_obs in rows
        ],
        "nuisance_sweep": sweep,
    }
    with open(os.path.join(RESULTS, "pilot.json"), "w") as f:
        json.dump(payload, f, indent=2, default=_json_native)
    print(f"\nresults -> {os.path.join(RESULTS, 'pilot.json')}")


if __name__ == "__main__":
    main()
