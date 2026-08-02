"""Posterior Silent Risk validation for a drift--diffusion decision."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from transfermod.modulus.discrepancy import (
    CertificationGeometry,
    decision_diameter,
    silent_risk_measure,
)


@dataclass(frozen=True)
class BayesianSilentRiskValidation:
    posterior_draws: int
    reference_q: float
    posterior_mean_q: float
    posterior_median_q: float
    posterior_interval: tuple[float, float]
    posterior_silent_risk: float
    posterior_fidelity_radius: float
    credible_set_worst_case: float
    credible_set_decision_diameter: float
    recurrence_probability: float
    geometry: CertificationGeometry

    def to_dict(self) -> dict[str, object]:
        return {
            "posterior_draws": self.posterior_draws,
            "reference_q": self.reference_q,
            "posterior_mean_q": self.posterior_mean_q,
            "posterior_median_q": self.posterior_median_q,
            "posterior_interval": self.posterior_interval,
            "posterior_silent_risk": self.posterior_silent_risk,
            "posterior_fidelity_radius": self.posterior_fidelity_radius,
            "credible_set_worst_case": self.credible_set_worst_case,
            "credible_set_decision_diameter": self.credible_set_decision_diameter,
            "recurrence_probability": self.recurrence_probability,
            "geometry": {
                "discrepancy": self.geometry.discrepancy,
                "reference_type": self.geometry.reference_type,
                "reference_q": self.geometry.reference_q,
                "tolerance": self.geometry.tolerance,
                "stabilization": self.geometry.stabilization,
            },
        }


def _return_probability(mu: np.ndarray, sigma2: np.ndarray, distance: float) -> np.ndarray:
    return np.where(
        mu <= 0.0,
        1.0,
        np.exp(-2.0 * mu * distance / sigma2),
    )


def run_bayesian_silent_risk_validation(
    *,
    seed: int = 19,
    n_observations: int = 80,
    n_draws: int = 20_000,
    distance: float = 5.0,
    credibility: float = 0.90,
    decision_tolerance: float = 0.15,
) -> BayesianSilentRiskValidation:
    """Fit a conjugate posterior and evaluate decision-level posterior exposure."""
    if n_observations < 2 or n_draws < 100:
        raise ValueError("n_observations >= 2 and n_draws >= 100 are required")
    if not 0.0 < credibility < 1.0:
        raise ValueError("credibility must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    increments = rng.normal(loc=0.12, scale=0.70, size=n_observations)

    mu0, kappa0, alpha0, beta0 = 0.0, 0.25, 2.0, 0.8
    mean = float(np.mean(increments))
    centered_ss = float(np.sum((increments - mean) ** 2))
    kappa_n = kappa0 + n_observations
    mu_n = (kappa0 * mu0 + n_observations * mean) / kappa_n
    alpha_n = alpha0 + n_observations / 2.0
    beta_n = (
        beta0
        + 0.5 * centered_ss
        + 0.5 * (kappa0 * n_observations / kappa_n) * (mean - mu0) ** 2
    )

    sigma2 = 1.0 / rng.gamma(shape=alpha_n, scale=1.0 / beta_n, size=n_draws)
    mu = rng.normal(loc=mu_n, scale=np.sqrt(sigma2 / kappa_n))
    q = _return_probability(mu, sigma2, distance)

    plugin_sigma2 = beta_n / (alpha_n - 1.0)
    reference_q = float(
        _return_probability(
            np.asarray([mu_n]), np.asarray([plugin_sigma2]), distance
        )[0]
    )
    geometry = CertificationGeometry.from_standard_metric(
        metric="MAE",
        reference_q=reference_q,
        tolerance=decision_tolerance,
        reference_type="posterior_plugin_center",
    )
    discrepancy = geometry.discrepancy_function()
    losses = np.asarray([discrepancy(value, reference_q) for value in q])
    silent_risk = silent_risk_measure(
        q.tolist(),
        reference_q,
        decision_tolerance,
        discrepancy=discrepancy,
    )
    fidelity_radius = float(np.quantile(losses, credibility))

    # Approximate a joint HPD subset by ranking draws under the conjugate
    # posterior density in (mu, sigma^2) coordinates.
    log_density = (
        -(alpha_n + 1.0) * np.log(sigma2)
        - beta_n / sigma2
        - 0.5 * np.log(sigma2 / kappa_n)
        - 0.5 * kappa_n * (mu - mu_n) ** 2 / sigma2
    )
    keep_count = max(1, int(math.ceil(credibility * n_draws)))
    retained = np.argpartition(log_density, -keep_count)[-keep_count:]
    q_set = q[retained]
    loss_set = losses[retained]

    alpha = (1.0 - credibility) / 2.0
    interval = (
        float(np.quantile(q, alpha)),
        float(np.quantile(q, 1.0 - alpha)),
    )
    return BayesianSilentRiskValidation(
        posterior_draws=n_draws,
        reference_q=reference_q,
        posterior_mean_q=float(np.mean(q)),
        posterior_median_q=float(np.median(q)),
        posterior_interval=interval,
        posterior_silent_risk=float(silent_risk),
        posterior_fidelity_radius=fidelity_radius,
        credible_set_worst_case=float(np.max(loss_set)),
        credible_set_decision_diameter=float(
            decision_diameter(q_set.tolist(), discrepancy)
        ),
        recurrence_probability=float(np.mean(mu <= 0.0)),
        geometry=geometry,
    )
