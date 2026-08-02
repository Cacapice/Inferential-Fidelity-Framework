#!/usr/bin/env python3
"""POD hot-spot instance with a certified adversarial floor.

A truncated global basis captures nearly all L2 energy while missing a narrow
localized feature. The Gaussian bump family is scientifically motivated but
does not prove full coverage, so the result is reported as a certified floor.
"""

from __future__ import annotations

import numpy as np

from transfermod.certification import Coverage, modulus_result
from transfermod.family_profile import FamilySearchProfile
from transfermod.modulus import CertificationGeometry
from transfermod.perturbations import ParametricResidualPerturbation


def l2_error(a: np.ndarray, b: np.ndarray, dx: float) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2) * dx))


def main() -> None:
    x = np.linspace(0.0, 1.0, 1001)
    dx = float(x[1] - x[0])
    reference = 0.15 + 0.2 * np.sin(np.pi * x)
    x_star = 0.73
    epsilon = 0.01

    geometry = CertificationGeometry.from_standard_metric(
        metric="MAE",
        reference_q=float(np.max(reference)),
        tolerance=epsilon,
    )

    results = []
    for width in (0.008, 0.015, 0.03):
        bump = np.exp(-0.5 * ((x - x_star) / width) ** 2)
        bump /= np.sqrt(np.sum(bump**2) * dx)

        family = ParametricResidualPerturbation(
            residual_generator=lambda w, bump=bump: w * bump,
            validation_metric=lambda a, b: l2_error(
                np.asarray(a), np.asarray(b), dx
            ),
            bounds=(0.0, epsilon),
            name=f"GaussianHotspot(width={width})",
            information_basis="localized_pod_complement",
        )
        alternative = family.perturb(reference, epsilon)
        value = abs(float(np.max(alternative)) - float(np.max(reference)))

        results.append(
            modulus_result(
                value,
                coverage=Coverage.certified_floor(
                    family.name,
                    "one normalized Gaussian bump width; full PDE-constrained "
                    "coverage has not been proved",
                ),
                perturbation_family=family.name,
                geometry=geometry,
                epsilon=epsilon,
            )
        )

    profile = FamilySearchProfile(tuple(results))

    print("POD hot-spot certified floors")
    print("=" * 42)
    for result in results:
        print(result.render())
        print()
    print(profile.render())
    print()
    print("Large floors condemn global-L2 validation for this local quantity.")
    print("Small floors would not establish safety because coverage is unproven.")


if __name__ == "__main__":
    main()
