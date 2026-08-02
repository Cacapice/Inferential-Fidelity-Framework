#!/usr/bin/env python3
"""Exact sketch-nullspace instance.

A linear sketch can be identical for two source vectors while a downstream
linear query changes. With a source-space radius, the exact modulus is the norm
of the query projected onto the sketch nullspace.
"""

from __future__ import annotations

import numpy as np

from transfermod.exact_coverage import SketchNullspaceCoverage
from transfermod.modulus import CertificationGeometry


def main() -> None:
    rng = np.random.default_rng(7)
    d, m = 12, 5
    P = rng.normal(size=(m, d))
    q = rng.normal(size=d)
    epsilon = 0.25

    theorem = SketchNullspaceCoverage(P, q)
    geometry = CertificationGeometry.from_standard_metric(
        metric="MAE",
        reference_q=0.0,
        tolerance=epsilon,
    )
    result = theorem.certify(epsilon, geometry=geometry)
    h_star = theorem.extremal_direction(epsilon)

    print("Sketch-nullspace exact coverage")
    print("=" * 42)
    print(result.render())
    print()
    print(f"||P h*||_2: {np.linalg.norm(P @ h_star):.3e}")
    print(f"||h*||_2:   {np.linalg.norm(h_star):.6f}")
    print(f"|<q,h*>|:   {abs(q @ h_star):.6f}")
    print()
    print("Large exact exposure means the sketch does not identify the query.")
    print("A zero value means the query lies entirely in the sketch row space.")


if __name__ == "__main__":
    main()
