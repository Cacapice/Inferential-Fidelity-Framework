"""Render the benchmark-logic figure: reference -> surrogate families -> verdicts."""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

CERT = "#2ca02c"
CONF = "#d62728"
REJ = "#ff7f0e"
REFC = "#333333"
LEARN = "#1f77b4"


def box(ax, x, y, w, h, text, fc, ec, fontsize=8.5, weight="normal", tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                linewidth=1.4, facecolor=fc, edgecolor=ec, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=tc, weight=weight, zorder=3, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color="#666666"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.2,
                                 color=color, zorder=1))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rows = [
        ("Class IV: constructed\nadversarial surrogate", "matches every\nconventional observable",
         "CONFOUND", CONF, "gap 1.18 vs true 1.96", CONF),
        ("Class III: neural quantum state\n(VMC-trained MLP)", "converged\n(energy err 7e-5)",
         "CERTIFIED", CERT, "gap 1.97  \u2713", LEARN),
        ("Class II: POD/Galerkin ROM\n(rank 3, under-resolved)", "moments matched\nexactly",
         "CONFOUND", CONF, "gap 2.39 vs true 1.96", LEARN),
        ("Class II: POD/Galerkin ROM\n(rank 10, resolved)", "subspace captures\nlow modes",
         "CERTIFIED", CERT, "gap 1.96  \u2713", LEARN),
        ("Class III: fit under a\nrealistic noise floor", "conventional\nchecks fail",
         "REJECTED", REJ, "not silently wrong", LEARN),
    ]
    n = len(rows)
    row_h, gap_h = 0.78, 0.30
    top = 0.55
    fig_h = 1.5 + n * (row_h + gap_h)
    fig, ax = plt.subplots(figsize=(12.2, fig_h))
    total = n * (row_h + gap_h)
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, total + 1.15)
    ax.axis("off")

    ys = [total - top - i * (row_h + gap_h) for i in range(n)]
    spine_x = 2.15
    # reference box centred vertically
    mid_y = 0.5 * (ys[0] + ys[-1]) + row_h / 2
    box(ax, 0.10, mid_y - 0.58, 1.85, 1.16,
        "CLASS I\nTRUSTED REFERENCE\n(exact spectrum)", REFC, REFC, fontsize=9, weight="bold")
    # vertical spine
    ax.plot([spine_x, spine_x], [ys[-1] + row_h / 2, ys[0] + row_h / 2],
            color="#999999", lw=1.2, zorder=1)
    arrow(ax, 1.95, mid_y, spine_x, mid_y)

    for y, (name, mid, verdict, color, note, namec) in zip(ys, rows):
        cy = y + row_h / 2
        ax.plot([spine_x, 2.75], [cy, cy], color="#999999", lw=1.1, zorder=1)
        arrow(ax, 2.62, cy, 2.80, cy)
        box(ax, 2.85, y, 2.70, row_h, name, namec, "#00000022", fontsize=8)
        arrow(ax, 5.60, cy, 6.05, cy)
        box(ax, 6.10, y, 1.85, row_h, mid, "#f2f2f2", "#bbbbbb", fontsize=7.6, tc="#333333")
        arrow(ax, 8.00, cy, 8.45, cy)
        box(ax, 8.50, y, 1.45, row_h, verdict, color, color, fontsize=8.6, weight="bold")
        ax.text(10.05, cy, note, fontsize=7.4, color="#555555", va="center")

    hy = ys[0] + row_h + 0.30
    for x, t in [(4.20, "surrogate family"), (7.02, "conventional gates"),
                 (9.22, "spectral gates \u2192 verdict")]:
        ax.text(x, hy, t, ha="center", fontsize=8.6, color="#333333", style="italic")

    ax.text(5.9, 0.16,
            "One trusted reference, many surrogate families, one gate: it flags adversarial "
            "and under-resolved\nsurrogates while certifying faithful learned models.",
            ha="center", fontsize=8.6, color="#333333", linespacing=1.5)
    fig.suptitle("Benchmark logic", fontsize=12, y=0.995, weight="bold")
    fig.tight_layout()
    out = os.path.join(RESULTS, "benchmark_logic.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
