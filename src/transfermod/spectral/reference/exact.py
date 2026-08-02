"""Finite-matrix reference by full numerical diagonalisation.

The reference contains no Monte Carlo sampling error: it diagonalises the
declared finite-cutoff Hamiltonian and computes operator-channel observables
from its spectral decomposition. Values remain subject to floating-point
roundoff, finite volume, electric cutoff, and selected winding sector.

All temporal correlators are computed from the exact spectral decomposition::

    C_O(t) = <0| O_c e^{-H t} O_c |0>
           = sum_{n >= 1} |<0| O_c |n>|^2  e^{-(E_n - E_0) t}

where ``O_c = O - <0|O|0>`` is the connected operator.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from ..lattice.u1 import U1Model


@dataclass(frozen=True, init=False)
class SpectralData:
    """Connected spectral representation of one operator channel.

    ``sector_gap`` is the canonical first excitation gap of the selected basis
    block. ``exact_gap`` is retained only as a deprecated compatibility alias
    and will be removed in TransferMod 2.0.
    """

    exp_val: float
    variance: float
    deltas: np.ndarray
    weights: np.ndarray
    _sector_gap: float
    channel_gap: float
    basis_mode: str = "unknown"
    spectrum_scope: str = "selected_basis"

    def __init__(
        self,
        *,
        exp_val: float,
        variance: float,
        deltas: np.ndarray,
        weights: np.ndarray,
        sector_gap: float | None = None,
        exact_gap: float | None = None,
        channel_gap: float,
        basis_mode: str = "unknown",
        spectrum_scope: str = "selected_basis",
    ) -> None:
        """Create spectral data using ``sector_gap`` or legacy ``exact_gap``.

        Supplying ``exact_gap`` is deprecated. Supplying both names with
        different values is rejected.
        """
        if sector_gap is None and exact_gap is None:
            raise TypeError("one of sector_gap or deprecated exact_gap is required")
        if sector_gap is not None and exact_gap is not None:
            if float(sector_gap) != float(exact_gap):
                raise ValueError("sector_gap and exact_gap disagree")
        if exact_gap is not None:
            warnings.warn(
                "`exact_gap` is deprecated and is an alias for `sector_gap`; "
                "use `sector_gap` with `spectrum_scope` and "
                "`global_physical_gap`. The alias will be removed in "
                "TransferMod 2.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        canonical_gap = sector_gap if sector_gap is not None else exact_gap
        object.__setattr__(self, "exp_val", float(exp_val))
        object.__setattr__(self, "variance", float(variance))
        object.__setattr__(self, "deltas", np.asarray(deltas, dtype=float))
        object.__setattr__(self, "weights", np.asarray(weights, dtype=float))
        object.__setattr__(self, "_sector_gap", float(canonical_gap))
        object.__setattr__(self, "channel_gap", float(channel_gap))
        object.__setattr__(self, "basis_mode", str(basis_mode))
        object.__setattr__(self, "spectrum_scope", str(spectrum_scope))

    @property
    def sector_gap(self) -> float:
        """First excitation gap in the selected basis block."""
        return self._sector_gap

    @property
    def exact_gap(self) -> float:
        """Deprecated alias for :attr:`sector_gap`.

        The name historically suggested a global physical gap even when the
        selected basis represented only one winding sector.
        """
        warnings.warn(
            "`exact_gap` is deprecated and is an alias for `sector_gap`; "
            "use `sector_gap` with `spectrum_scope` and "
            "`global_physical_gap`. The alias will be removed in "
            "TransferMod 2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._sector_gap

    @property
    def global_physical_gap(self) -> float | None:
        """All-sector finite-cutoff gap, available only for a full basis."""
        return self._sector_gap if self.basis_mode == "full" else None

    def correlator(self, ts: np.ndarray) -> np.ndarray:
        ts = np.asarray(ts, dtype=float)
        return (
            self.weights[:, None]
            * np.exp(-self.deltas[:, None] * ts[None, :])
        ).sum(0)


class ExactReference:
    """Diagonalise the model and expose exact spectral observables for one operator."""

    def __init__(self, model: U1Model, g: float, operator: np.ndarray | None = None):
        self.model = model
        self.g = float(g)
        H = model.hamiltonian(g)
        # Dense eigh gives full numerical diagonalisation of the selected finite matrix.
        self.energies, self.vectors = np.linalg.eigh(H)
        self.E0 = float(self.energies[0])
        self.psi0 = self.vectors[:, 0]
        self.operator = model.cos_plaquette_operator() if operator is None else operator
        self._spectral = self._build_spectral(self.operator)

    def _build_spectral(self, O: np.ndarray, wtol: float = 1e-12) -> SpectralData:
        D = O.shape[0]
        exp_val = float(self.psi0 @ O @ self.psi0)
        Oc = O - exp_val * np.eye(D)
        overlaps = self.vectors.T @ (Oc @ self.psi0)  # <n|O_c|0>
        w = overlaps ** 2
        d = self.energies - self.E0
        keep = (d > 1e-9) & (w > wtol)
        dd, ww = d[keep], w[keep]
        order = np.argsort(dd)
        dd, ww = dd[order], ww[order]
        sector_gap = float(self.energies[1] - self.energies[0])
        channel_gap = float(dd[0]) if dd.size else float("nan")
        return SpectralData(
            exp_val=exp_val,
            variance=float(ww.sum()),
            deltas=dd,
            weights=ww,
            sector_gap=sector_gap,
            channel_gap=channel_gap,
            basis_mode=self.model.lat.basis_mode,
            spectrum_scope=("all_winding_sectors" if self.model.lat.basis_mode == "full" else "zero_winding_sector"),
        )

    # -- public observables ----------------------------------------------
    @property
    def spectral(self) -> SpectralData:
        return self._spectral

    @property
    def sector_gap(self) -> float:
        """First excitation gap in the selected basis block."""
        return self._spectral.sector_gap

    @property
    def exact_gap(self) -> float:
        """Deprecated alias for ``spectral.sector_gap``; removed in v2.0."""
        warnings.warn(
            "`ExactReference.exact_gap` is deprecated; use `sector_gap` or "
            "`spectral.sector_gap` with `spectrum_scope`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._spectral.sector_gap

    @property
    def global_physical_gap(self) -> float | None:
        """All-sector finite-cutoff gap, available only for ``basis_mode='full'``."""
        return self._spectral.global_physical_gap

    def correlator(self, ts: np.ndarray) -> np.ndarray:
        return self._spectral.correlator(ts)

    def plaquette_expectation(self) -> float:
        """Equal-time <cos(theta_p)> per plaquette in the operator."""
        return self._spectral.exp_val

    def equal_time_variance(self) -> float:
        return self._spectral.variance

    def mass_gap(self) -> float:
        """Finite-volume, finite-cutoff operator-channel gap in the selected sector."""
        return self._spectral.channel_gap

    def cutoff_boundary_probability(self) -> float:
        """Ground-state probability on configurations touching ``|E|=Lambda``."""
        mask = np.array(
            [self.model.configuration_touches_cutoff(cfg) for cfg in self.model.basis],
            dtype=bool,
        )
        return float(np.sum(np.abs(self.psi0[mask]) ** 2))

    # -- duck-typed interface shared with surrogates (for the observable suite) --
    def spectral_repr(self):
        """(exp_val, deltas, weights) -- the common contract with surrogates."""
        s = self._spectral
        return s.exp_val, s.deltas.copy(), s.weights.copy()

    def exp_val(self) -> float:
        return self._spectral.exp_val

    def variance(self) -> float:
        return self._spectral.variance

    def asymptotic_gap(self) -> float:
        return self._spectral.channel_gap
