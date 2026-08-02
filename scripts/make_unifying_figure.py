"""Render the unifying figure: why these scientific domains are the same problem.

A single diagram showing the shared inferential pipeline -- system, surrogate,
aggregate validation, low-signal scientific quantity, certification gate -- with
the domain instantiations underneath.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

INK = "#2b2b2b"
MUTED = "#6d6d6d"
PASS_C = "#2ca02c"
FAIL_C = "#d62728"
ACCENT = "#1f77b4"
WARN = "#b8860b"


def box(ax, cx, cy, w, h, text, fc, ec, fs=8.6, tc="white", weight="normal"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.014", linewidth=1.3,
                                facecolor=fc, edgecolor=ec, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
            weight=weight, zorder=3, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=MUTED, style="-|>", lw=1.3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=12, linewidth=lw, color=color,
                                 zorder=1))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.6, 8.4))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 8.4)
    ax.axis("off")

    cx = 4.05          # spine x for the pipeline
    bw, bh = 4.5, 0.62
    ys = [7.72, 6.90, 6.08, 5.26, 4.44, 3.62]

    box(ax, cx, ys[0], bw, bh, "Scientific system", INK, INK, fs=9, weight="bold")
    box(ax, cx, ys[1], bw, bh, "High-dimensional representation", "#4a4a4a", "#4a4a4a")
    box(ax, cx, ys[2], bw, bh, "Surrogate\n(learned · reduced · numerical)", ACCENT, ACCENT)
    box(ax, cx, ys[3], bw, bh, "Aggregate validation  \u2713", PASS_C, PASS_C)
    box(ax, cx, ys[4], bw, bh, "Low-signal scientific quantity", WARN, WARN)
    box(ax, cx, ys[5], bw, bh, "Certification gate", INK, INK, fs=9, weight="bold")

    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax, cx, a - bh / 2, cx, b + bh / 2)

    # verdict branches
    by = ys[5] - bh / 2 - 0.62
    arrow(ax, cx, ys[5] - bh / 2, cx - 1.28, by + 0.22)
    arrow(ax, cx, ys[5] - bh / 2, cx + 1.28, by + 0.22)
    box(ax, cx - 1.28, by, 1.75, 0.54, "CERTIFIED", PASS_C, PASS_C, fs=8.6, weight="bold")
    box(ax, cx + 1.28, by, 1.75, 0.54, "CONFOUND", FAIL_C, FAIL_C, fs=8.6, weight="bold")

    # annotation: the failure everyone misses
    ax.annotate("aggregate accuracy does not\nconstrain this quantity",
                xy=(cx + bw / 2, (ys[3] + ys[4]) / 2), xytext=(cx + bw / 2 + 0.35, (ys[3] + ys[4]) / 2),
                fontsize=8.2, color=FAIL_C, va="center", linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=FAIL_C, lw=1.0))

    # domain instantiation table
    ax.text(1.0, 2.15, "The same pipeline, five domains", fontsize=10, weight="bold",
            color=INK)
    ax.plot([1.0, 11.6], [1.99, 1.99], color="#cccccc", lw=1.0)
    ax.text(1.0, 1.78, "domain", fontsize=8.4, style="italic", color=MUTED)
    ax.text(5.3, 1.78, "low-signal scientific quantity", fontsize=8.4, style="italic",
            color=MUTED)
    rows = [
        ("Yang\u2013Mills / lattice gauge theory", "mass gap (spectral tail)", True),
        ("Darcy-flow inverse problem", "Karhunen\u2013Lo\u00e8ve coefficient recovery", False),
        ("Electrical impedance tomography", "physiological interpretation", False),
        ("Climate emulators", "extreme-event statistics", False),
        ("Molecular dynamics", "rare-event kinetics / slow modes", False),
    ]
    y = 1.50
    for dom, q, done in rows:
        mark = "\u25cf" if done else "\u25cb"
        col = ACCENT if done else MUTED
        ax.text(1.0, y, f"{mark}  {dom}", fontsize=8.6, color=col,
                weight="bold" if done else "normal")
        ax.text(5.3, y, "\u2192   " + q, fontsize=8.6, color=col,
                weight="bold" if done else "normal")
        y -= 0.30

    ax.text(1.0, 0.04,
            "\u25cf implemented in this repository       \u25cb same structure, "
            "not yet instantiated",
            fontsize=8.0, color=MUTED)

    fig.suptitle("Inferential fidelity: one failure mode across scientific surrogates",
                 fontsize=12.5, weight="bold", y=0.985)
    ax.text(6.3, 8.02,
            "A surrogate can pass every aggregate check and still corrupt the "
            "low-signal mode the science depends on.",
            ha="center", fontsize=9, color=MUTED)
    fig.tight_layout()
    out = os.path.join(RESULTS, "inferential_fidelity.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
