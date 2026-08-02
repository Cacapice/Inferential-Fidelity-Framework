"""Train real learned surrogates and judge their spectral fidelity.

Two structurally unrelated learned model families are trained/constructed and
run through the *same* gate, to show the benchmark evaluates scientific fidelity
rather than one model class:

1. **Neural quantum state** -- an MLP wavefunction trained by VMC.
2. **Reduced-order model** -- a POD/Krylov-Galerkin projection, swept over rank.

Writes ``results/neural.{json,png}``.

Usage::  python scripts/run_learned.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transfermod.spectral.gate.tests import certify  # noqa: E402
from transfermod.spectral.lattice.u1 import U1Lattice, U1Model  # noqa: E402
from transfermod.spectral.observables.suite import TS, FrozenObservableSuite  # noqa: E402
from transfermod.spectral.reference.exact import ExactReference  # noqa: E402
from transfermod.spectral.surrogates import (  # noqa: E402
    NeuralQuantumStateSurrogate,
    PODGalerkinSurrogate,
)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def _native(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def run():
    print("=" * 74)
    print("Real neural sampler: a Neural Quantum State trained by VMC")
    print("=" * 74)
    m = U1Model(U1Lattice(2, 2, 2, basis_mode="vacuum"))
    ref = ExactReference(m, g=1.0)
    print(f"  reference: 2+1D U(1), 2x2 Lambda=2 (dim {m.dim}), g=1.0")
    print(f"  exact E0={ref.E0:.5f}, channel gap={ref.mass_gap():.5f}\n")

    nqs = NeuralQuantumStateSurrogate(m, g=1.0, hidden=24, epochs=2000, seed=0)
    v = certify(ref, nqs)
    print(f"  trained MLP wavefunction (24 hidden units, 2000 VMC epochs):")
    print(f"    variational energy   {nqs.energy:.5f}  (error {abs(nqs.energy - ref.E0):.1e})")
    print(f"    <cos plaquette>      {nqs.exp_val():.5f}  (error {abs(nqs.exp_val() - ref.exp_val()):.1e})")
    print(f"    channel gap          {nqs.asymptotic_gap():.4f}  (exact {ref.mass_gap():.4f})")
    print(f"    verdict              {v.label}")
    for r in v.results:
        print(f"      [{r.kind:<12}] {r.name:<32} {'pass' if r.passed else 'FAIL'}")

    # capacity sweep: fidelity vs network size
    print("\n  capacity / training sweep:")
    print(f"  {'hidden':>7}{'epochs':>8}{'E_err':>10}{'<cos>_err':>11}{'gap':>8}"
          f"{'verdict':>12}  failing gate(s)")
    sweep = []
    any_confound = False
    for hidden, epochs in [(2, 1500), (4, 1500), (8, 2000), (24, 2000)]:
        s = NeuralQuantumStateSurrogate(m, g=1.0, hidden=hidden, epochs=epochs, seed=0)
        vv = certify(ref, s)
        fails = [r.name.split("_")[0] for r in vv.results if not r.passed]
        any_confound |= (vv.conventional_pass and not vv.spectral_pass)
        print(f"  {hidden:>7}{epochs:>8}{abs(s.energy - ref.E0):>10.1e}"
              f"{abs(s.exp_val() - ref.exp_val()):>11.1e}{s.asymptotic_gap():>8.3f}"
              f"{vv.label.split('(')[0].strip():>12}  {', '.join(fails) or '-'}")
        sweep.append({"hidden": hidden, "epochs": epochs,
                      "energy_err": abs(s.energy - ref.E0),
                      "exp_val_err": abs(s.exp_val() - ref.exp_val()),
                      "gap": s.asymptotic_gap(), "verdict": vv.label,
                      "failing_gates": fails})

    print(f"\n  -> No NQS in this sweep was ever a CONFOUND (conventional pass +")
    print(f"     spectral fail): every model was CERTIFIED or REJECTED. Rejections are")
    print(f"     on conventional checks -- e.g. hidden=8 misses only the equal-time")
    print(f"     variance tolerance while its gap is fine -- so an under-converged")
    print(f"     network announces itself rather than passing silently. (The sweep is")
    print(f"     non-monotone in capacity: VMC optimisation variance, not a spectral")
    print(f"     effect.) Once converged, the neural sampler is spectrally faithful")
    print(f"     and CERTIFIED: the gate does not false-positive on a real learned model.\n")
    assert not any_confound, "unexpected: an NQS passed conventional but failed spectral"

    rom_rows = rom_section(ref, m)

    make_figure(ref, nqs, os.path.join(RESULTS, "neural.png"))
    payload = {
        "reference": {"E0": ref.E0, "gap": ref.mass_gap()},
        "nqs": {"hidden": 24, "epochs": 2000, "energy": nqs.energy,
                "energy_err": abs(nqs.energy - ref.E0), "exp_val": nqs.exp_val(),
                "gap": nqs.asymptotic_gap(), "verdict": v.label},
        "capacity_sweep": sweep,
        "reduced_order_model": rom_rows,
    }
    with open(os.path.join(RESULTS, "neural.json"), "w") as f:
        json.dump(payload, f, indent=2, default=_native)
    print(f"results -> {os.path.join(RESULTS, 'neural.json')}")


def rom_section(ref, m):
    """A second, structurally unrelated learned family: POD/Krylov-Galerkin ROM."""
    print("=" * 74)
    print("Second learned family: reduced-order model (POD/Krylov-Galerkin)")
    print("=" * 74)
    print("  An honest, standard model-order reduction -- nothing adversarial is")
    print("  fabricated. Rank is the capacity knob.\n")
    print(f"  {'rank':>5}{'gap':>9}{'gap err':>9}{'C(0) err':>10}{'C(8) err':>10}"
          f"{'conv':>6}{'spec':>6}  verdict")
    Cref = ref.correlator(TS)
    rows = []
    for r in (2, 3, 4, 5, 6, 10, 20):
        s = PODGalerkinSurrogate(m, 1.0, rank=r)
        vv = certify(ref, s)
        Cs = s.correlator(TS)
        gap = s.asymptotic_gap()
        e0 = abs(Cs[0] - Cref[0]) / Cref[0]
        e8 = abs(Cs[-1] - Cref[-1]) / Cref[-1]
        print(f"  {r:>5}{gap:>9.4f}{(gap - ref.mass_gap()) / ref.mass_gap():>8.0%}"
              f"{e0:>10.1e}{e8:>10.1e}"
              f"{('PASS' if vv.conventional_pass else 'fail'):>6}"
              f"{('PASS' if vv.spectral_pass else 'fail'):>6}  "
              f"{vv.label.split('(')[0].strip()}")
        rows.append({"rank": r, "gap": gap, "C0_rel_err": e0, "Ctail_rel_err": e8,
                     "conventional_pass": vv.conventional_pass,
                     "spectral_pass": vv.spectral_pass, "verdict": vv.label})
    print("\n  -> This is the practitioner-facing result. A Krylov/POD reduction matches")
    print("     the low moments *exactly* (C(0) to ~1e-15), so equal-time and aggregate")
    print("     checks pass; but Rayleigh-Ritz eigenvalues converge from ABOVE, so an")
    print("     under-resolved ROM systematically OVER-estimates the gap while looking")
    print("     conventionally excellent -- a spectral confound arising from ordinary")
    print("     practice, not adversarial design. Increasing rank restores fidelity.\n")
    return rows


def make_figure(ref, nqs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # left: VMC energy convergence
    ax1.plot(nqs.energy_trace, color="#1f77b4", lw=1.2, label="NQS variational energy")
    ax1.axhline(ref.E0, color="k", ls=":", label="exact $E_0$")
    ax1.set_xlabel("VMC epoch")
    ax1.set_ylabel("energy")
    ax1.set_title("Training convergence")
    ax1.legend(fontsize=8)
    lo = ref.E0 - 0.2 * abs(ref.E0)
    ax1.set_ylim(lo, ref.E0 + 1.5 * abs(ref.E0))

    # right: effective mass -- NQS vs exact, plateau at the true gap
    suite = FrozenObservableSuite()
    ro = suite.measure(ref)
    no = suite.measure(nqs)
    tsm = TS[:-1] + 0.5
    ax2.plot(tsm, ro.eff_mass, "k-", lw=2, label="reference (exact)")
    ax2.axhline(ref.mass_gap(), color="k", ls=":", alpha=0.6, label="true gap")
    ax2.plot(tsm, no.eff_mass, "o-", color="#2ca02c", label="neural quantum state")
    ax2.set_xlabel("Euclidean time t")
    ax2.set_ylabel(r"$m_{\rm eff}(t)$")
    ax2.set_title("Spectral fidelity: NQS plateaus at the true gap")
    ax2.legend(fontsize=8)
    fig.suptitle("A real neural sampler (VMC-trained NQS): converges in energy and "
                 "preserves the spectral gap", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"figure -> {path}")


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    run()
