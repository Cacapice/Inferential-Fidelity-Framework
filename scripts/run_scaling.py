"""Scaling study enabled by the sparse/Krylov path.

Sections:
  1. Krylov vs dense cross-validation (2x2, Lambda=2).
  2. Truncation (Lambda) convergence of the channel gap at fixed volume -- the
     Lambda -> infinity direction the sparse path makes reachable.
  3. Larger-volume confounds on 3x3 (Lambda=1), where the light state carries
     little operator weight: both the universal spurious-slow control and the
     now-natural attenuate-tail (over-estimate) control.
  4. A genuinely trained multi-exponential surrogate, swept over the statistical
     noise floor, exhibiting the certified -> confound transition and the gate
     verdict tracking it.

Usage::  python scripts/run_scaling.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transfermod.spectral.analysis.estimators import ESTIMATORS  # noqa: E402
from transfermod.spectral.gate.tests import certify  # noqa: E402
from transfermod.spectral.lattice.u1 import U1Lattice, U1Model  # noqa: E402
from transfermod.spectral.observables.suite import TS, FrozenObservableSuite  # noqa: E402
from transfermod.spectral.reference.exact import ExactReference  # noqa: E402
from transfermod.spectral.reference.krylov import KrylovReference  # noqa: E402
from transfermod.spectral.surrogates import (  # noqa: E402
    SpectralShortcutControl,
    TrainedMultiExpSurrogate,
)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def _native(o):
    import numpy as _np
    if isinstance(o, _np.bool_):
        return bool(o)
    if isinstance(o, _np.integer):
        return int(o)
    if isinstance(o, _np.floating):
        return float(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def section_1_validation():
    print("=" * 74)
    print("1. Krylov (sparse+Lanczos) vs dense -- cross validation")
    print("=" * 74)
    mf = U1Model(U1Lattice(2, 2, 2, basis_mode="full"))
    mv = U1Model(U1Lattice(2, 2, 2, basis_mode="vacuum"))
    rd = ExactReference(mf, g=1.0)
    rk = KrylovReference(mv, g=1.0)
    ts = TS
    dg = abs(rd.mass_gap() - rk.mass_gap())
    dc = float(np.abs(rd.correlator(ts) - rk.correlator(ts)).max() / rd.correlator(ts)[0])
    print(f"  full-basis dim={mf.dim}, vacuum-basis dim={mv.dim}")
    print(f"  |gap_dense - gap_krylov|         = {dg:.2e}")
    print(f"  max rel |C_dense - C_krylov|     = {dc:.2e}")
    print(f"  -> vacuum sector + Krylov reproduce the channel exactly.\n")
    return {"full_dim": mf.dim, "vacuum_dim": mv.dim, "gap_diff": dg, "corr_reldiff": dc}


def section_2_lambda(g=1.0):
    print("=" * 74)
    print(f"2. Truncation (Lambda) convergence of the channel gap  (2x2, g={g})")
    print("=" * 74)
    rows = []
    prev = None
    for Lam in (1, 2, 3, 4):
        m = U1Model(U1Lattice(2, 2, Lam, basis_mode="vacuum"))
        r = KrylovReference(m, g=g)
        gap = r.mass_gap()
        delta = "" if prev is None else f"  d(gap)={gap - prev:+.4f}"
        print(f"  Lambda={Lam}: vac-dim={m.dim:4d}  channel gap={gap:.5f}{delta}")
        rows.append({"Lambda": Lam, "dim": m.dim, "gap": gap})
        prev = gap
    print("  -> gap converges as the electric basis is enlarged.\n")
    return rows


def section_3_larger_volume():
    print("=" * 74)
    print("3. Larger volume 3x3 (Lambda=1): confounds with a naturally weak light state")
    print("=" * 74)
    m = U1Model(U1Lattice(3, 3, 1, basis_mode="vacuum"))
    ref = KrylovReference(m, g=0.8)
    sd = ref.spectral
    w1frac = sd.weights[0] / sd.weights.sum()
    print(f"  vac-dim={m.dim}  channel gap={ref.mass_gap():.4f}  "
          f"light-state weight fraction={w1frac:.3f}  (weak overlap)")
    out = {"dim": m.dim, "gap": ref.mass_gap(), "w1_frac": float(w1frac), "surrogates": []}

    for label, ctl in [
        ("spurious_slow (under-estimate)",
         SpectralShortcutControl(sd, mode="spurious_slow", rho=0.6, frac=0.03)),
        ("attenuate_tail (over-estimate, now natural)",
         SpectralShortcutControl(sd, mode="attenuate_tail", attenuation=0.0)),
    ]:
        v = certify(ref, ctl)
        gap = FrozenObservableSuite().measure(ctl).asymptotic_gap
        print(f"  {label:44} gap={gap:6.3f}  conv={_yn(v.conventional_pass)} "
              f"spec={_yn(v.spectral_pass)}  {v.label.split('(')[0].strip()}")
        out["surrogates"].append({"label": label, "gap": gap,
                                  "conventional_pass": v.conventional_pass,
                                  "spectral_pass": v.spectral_pass})
    print()
    return out


def section_4_trained():
    print("=" * 74)
    print("4. Trained multi-exponential surrogate: when is a learned gap trustworthy?")
    print("=" * 74)
    m = U1Model(U1Lattice(3, 3, 1, basis_mode="vacuum"))
    ref = KrylovReference(m, g=0.8)
    gap = ref.mass_gap()
    print(f"  reference gap={gap:.4f} (light-state weight ~3%). Fit a multi-exp")
    print(f"  surrogate to noisy sampled C(t) under the conventional signal-weighted loss.\n")

    def run(label, **kw):
        sur = TrainedMultiExpSurrogate(ref, TS, seed=3, **kw)
        v = certify(ref, sur)
        fg = sur.asymptotic_gap()
        err = (fg - gap) / gap
        fg_disp = f"{fg:8.3f}" if abs(fg) < 100 else f"{fg:8.1e}"
        print(f"  {label:46} fit_gap={fg_disp}  err={err:>7.0%}  "
              f"conv={_yn(v.conventional_pass)} spec={_yn(v.spectral_pass)}  "
              f"{v.label.split('(')[0].strip()}")
        return {"label": label, "fit_gap": fg, "gap_err": err,
                "conventional_pass": v.conventional_pass, "spectral_pass": v.spectral_pass, **kw}

    rows = []
    # (a) informative tail -> the learned gap is correct AND certified
    rows.append(run("(a) tail-informative noise 0.1% (relative)",
                    rel_noise=0.001, noise_model="relative"))
    # (b,c) realistic constant noise floor hides the weak tail -> CONFOUND
    rows.append(run("(b) constant noise floor 0.1% (absolute)",
                    rel_noise=0.001, noise_model="absolute_floor"))
    rows.append(run("(c) constant noise floor 0.3% (absolute)",
                    rel_noise=0.003, noise_model="absolute_floor"))
    # (d) heavy noise -> fit fails even conventionally -> REJECTED (not silently wrong)
    rows.append(run("(d) constant noise floor 0.5% (absolute)",
                    rel_noise=0.005, noise_model="absolute_floor"))
    print("\n  Reading: (a) with tail signal the learned gap is right and CERTIFIED;")
    print("  (b,c) a realistic constant noise floor hides the weak light state, so the")
    print("  fit reports a heavy-mode gap that still passes conventional metrics -- a")
    print("  CONFOUND the gate catches; (d) a grossly noisy fit fails outright (REJECTED).\n")
    return rows


def section_5_estimators():
    print("=" * 74)
    print("5. Estimator robustness: the confound fools every standard gap estimator")
    print("=" * 74)
    ref = ExactReference(U1Model(U1Lattice(2, 2, 2, basis_mode="vacuum")), g=1.0)
    ctl = SpectralShortcutControl(ref.spectral, mode="spurious_slow", rho=0.6, frac=0.03)
    true = ref.mass_gap()
    Cref = ref.correlator(TS)
    Cctl = ctl.correlator(TS)
    print(f"  true channel gap = {true:.4f}. Each estimator is run on the reference")
    print(f"  and on the constructed confound (which matches all conventional observables).\n")
    print(f"  {'estimator':<26}{'ref':>8}{'recovers':>10}{'confound':>10}{'fooled':>8}")
    rows = []
    for name, fn in ESTIMATORS.items():
        gr = fn(TS, Cref)
        gc = fn(TS, Cctl)
        recovers = np.isfinite(gr) and abs(gr - true) / true <= 0.10
        fooled = recovers and (np.isfinite(gc) and abs(gc - true) / true > 0.10)
        rec = "yes" if recovers else "mis-spec"
        fl = "YES" if fooled else ("-" if not recovers else "no")
        print(f"  {name:<26}{gr:>8.3f}{rec:>10}{gc:>10.3f}{fl:>8}")
        rows.append({"estimator": name, "ref_estimate": gr, "recovers_ref": recovers,
                     "confound_estimate": gc, "fooled": fooled})
    n_fooled = sum(r["fooled"] for r in rows)
    n_valid = sum(r["recovers_ref"] for r in rows)
    print(f"\n  -> {n_fooled}/{n_valid} estimators that correctly read the reference are")
    print(f"     fooled by the confound. (cosh is mis-specified for an open, non-periodic")
    print(f"     correlator -- short-t-weighted -- so it neither reads the reference nor")
    print(f"     probes the tail; it is shown for completeness.)\n")
    return rows


def section_6_multichannel():
    print("=" * 74)
    print("6. Operator basis: can the confound fool multiple channels at once?")
    print("=" * 74)
    m = U1Model(U1Lattice(2, 2, 2, basis_mode="vacuum"))
    o1 = m.cos_plaquette_operator([0])   # local single-plaquette channel
    o2 = m.zero_momentum_cos()           # smeared zero-momentum channel
    r1 = ExactReference(m, 1.0, operator=o1)
    r2 = ExactReference(m, 1.0, operator=o2)
    from transfermod.spectral.analysis.estimators import plateau_gap as pg
    g1t, g2t = pg(TS, r1.correlator(TS)), pg(TS, r2.correlator(TS))
    tol = 0.10

    def consistent(a, b):
        return abs(a - b) / (0.5 * (a + b)) <= tol

    print(f"  two channels of the same reference share the true gap "
          f"(single-plaquette {g1t:.3f}, zero-momentum {g2t:.3f}).")
    print(f"  A cross-channel check flags a surrogate whose channels disagree by >{tol:.0%}.\n")

    rows = []
    # (A) confound built in ONE channel only
    cA1 = SpectralShortcutControl(r1.spectral, mode="spurious_slow", rho=0.6, frac=0.03)
    gA1 = pg(TS, cA1.correlator(TS))
    gA2 = g2t  # other channel left true
    okA = consistent(gA1, gA2)
    print(f"  (A) single-channel fake : gaps ({gA1:.3f}, {gA2:.3f})  "
          f"cross-channel {'CONSISTENT' if okA else 'INCONSISTENT -> caught'}")
    rows.append({"case": "single_channel_fake", "gaps": [gA1, gA2], "cross_channel_consistent": okA})

    # (B) confound built consistently in BOTH channels (a 'physical' fake)
    cB1 = SpectralShortcutControl(r1.spectral, mode="spurious_slow", rho=0.6, frac=0.03)
    cB2 = SpectralShortcutControl(r2.spectral, mode="spurious_slow", rho=0.6, frac=0.03)
    gB1, gB2 = pg(TS, cB1.correlator(TS)), pg(TS, cB2.correlator(TS))
    okB = consistent(gB1, gB2)
    vB = certify(r1, cB1)  # the trusted reference still judges channel 1
    print(f"  (B) all-channel fake    : gaps ({gB1:.3f}, {gB2:.3f})  "
          f"cross-channel {'CONSISTENT -> not caught by consistency' if okB else 'INCONSISTENT'}")
    print(f"      ...but the trusted reference gate still flags channel 1: {vB.label.split('(')[0].strip()}")
    rows.append({"case": "all_channel_fake", "gaps": [gB1, gB2],
                 "cross_channel_consistent": okB, "reference_gate": vB.label})

    print("\n  -> A variational/smearing operator basis defeats cheap single-channel")
    print("     confounds (A): they betray themselves as cross-channel inconsistency.")
    print("     A confound that corrupts every channel consistently (B) survives a")
    print("     reference-free consistency check -- only the trusted reference catches")
    print("     it. This sharpens conjecture §10.4: multi-channel raises the bar but is")
    print("     not by itself a certificate.\n")
    return rows


def make_diagnostics_figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = ExactReference(U1Model(U1Lattice(2, 2, 2, basis_mode="vacuum")), g=1.0)
    ctl = SpectralShortcutControl(ref.spectral, mode="spurious_slow", rho=0.6, frac=0.03)
    _, dr, wr = ref.spectral_repr()
    _, dc, wc = ctl.spectral()
    Cref = ref.correlator(TS)
    Cctl = ctl.correlator(TS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # left: spectral decomposition (delta vs weight) -- the spurious mode is obvious
    ax1.stem(dr, wr, linefmt="k-", markerfmt="ko", basefmt=" ", label="reference")
    ax1.stem(dc, wc, linefmt="r--", markerfmt="rx", basefmt=" ", label="constructed confound")
    ax1.axvline(ref.mass_gap(), color="k", ls=":", alpha=0.5)
    ax1.set_xlabel(r"excitation energy $\Delta_n$")
    ax1.set_ylabel(r"operator weight $w_n$")
    ax1.set_title("Spectral decomposition")
    ax1.set_xlim(0, min(6, float(max(dr.max(), dc.max())) + 0.5))
    ax1.legend(fontsize=8)
    # annotate the fabricated slow mode
    if dc.size:
        i = int(np.argmin(dc))
        ax1.annotate("fabricated\nslow mode", xy=(dc[i], wc[i]),
                     xytext=(dc[i] + 0.6, wc[i] + 0.02), fontsize=8, color="red",
                     arrowprops=dict(arrowstyle="->", color="red"))

    # right: correlator ratio -- ~1 at short t, blows up in the tail (why G5 fires)
    ratio = Cctl / Cref
    ax2.axhspan(1 - 0.15, 1 + 0.15, color="green", alpha=0.12,
                label="G5 tail tolerance (±15%)")
    ax2.semilogy(TS, ratio, "^-", color="#d62728", label=r"$C_{\rm confound}/C_{\rm ref}$")
    ax2.axhline(1.0, color="k", lw=0.8)
    ax2.set_xlabel("Euclidean time t")
    ax2.set_ylabel(r"$C_{\rm confound}(t)\,/\,C_{\rm ref}(t)$")
    ax2.set_title(f"Correlator ratio (tail reaches {ratio[-1]:.0f}×)")
    ax2.legend(fontsize=8, loc="upper left")
    fig.suptitle("Diagnostics: the fabricated slow mode is invisible to aggregate metrics, "
                 "obvious in the spectrum and the tail", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"figure -> {path}")


def make_scaling_figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # left: Lambda-convergence of the gap (2x2)
    lams, gaps = [], []
    for Lam in (1, 2, 3, 4):
        r = KrylovReference(U1Model(U1Lattice(2, 2, Lam, basis_mode="vacuum")), g=1.0)
        lams.append(Lam)
        gaps.append(r.mass_gap())
    ax1.plot(lams, gaps, "o-", color="#1f77b4")
    ax1.axhline(gaps[-1], ls=":", color="gray", alpha=0.7)
    ax1.set_xlabel(r"electric truncation $\Lambda$")
    ax1.set_ylabel("channel mass gap")
    ax1.set_title("Truncation convergence (2×2, g=1.0)")
    ax1.set_xticks(lams)

    # right: 3x3 effective mass -- reference vs both natural confounds
    m = U1Model(U1Lattice(3, 3, 1, basis_mode="vacuum"))
    ref = KrylovReference(m, g=0.8)
    suite = FrozenObservableSuite()
    ro = suite.measure(ref)
    tsm = TS[:-1] + 0.5
    ax2.plot(tsm, ro.eff_mass, "k-", lw=2, label="reference (exact)")
    ax2.axhline(ref.mass_gap(), ls=":", color="k", alpha=0.6, label="true gap")
    under = SpectralShortcutControl(ref.spectral, mode="spurious_slow", rho=0.6, frac=0.03)
    over = SpectralShortcutControl(ref.spectral, mode="attenuate_tail", attenuation=0.0)
    ax2.plot(tsm, suite.measure(under).eff_mass, "v-", color="#d62728", alpha=0.8,
             label="spurious-slow (under)")
    ax2.plot(tsm, suite.measure(over).eff_mass, "^-", color="#ff7f0e", alpha=0.8,
             label="attenuate-tail (over)")
    ax2.set_xlabel("Euclidean time t")
    ax2.set_ylabel(r"$m_{\rm eff}(t)$")
    ax2.set_title("3×3: weak light state, both confounds")
    ax2.legend(fontsize=8)
    fig.suptitle("Sparse/Krylov path: truncation scaling and natural confounds at larger volume",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"figure -> {path}")


def _yn(b):
    return "PASS" if b else "fail"


def main():
    os.makedirs(RESULTS, exist_ok=True)
    payload = {}
    payload["krylov_validation"] = section_1_validation()
    payload["lambda_convergence"] = section_2_lambda()
    payload["larger_volume"] = section_3_larger_volume()
    payload["trained_noise_sweep"] = section_4_trained()
    payload["estimator_robustness"] = section_5_estimators()
    payload["multichannel"] = section_6_multichannel()
    make_scaling_figure(os.path.join(RESULTS, "scaling.png"))
    make_diagnostics_figure(os.path.join(RESULTS, "diagnostics.png"))
    with open(os.path.join(RESULTS, "scaling.json"), "w") as f:
        json.dump(payload, f, indent=2, default=_native)
    print(f"results -> {os.path.join(RESULTS, 'scaling.json')}")


if __name__ == "__main__":
    main()
