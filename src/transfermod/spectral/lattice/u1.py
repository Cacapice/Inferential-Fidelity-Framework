"""Finite-cutoff compact U(1) Kogut--Susskind lattice gauge theory.

The module constructs the Gauss-law physical Hilbert space and Hamiltonian
deterministically for a declared finite lattice, electric-field cutoff, and
topological sector. Matrix assembly is exact for that finite model; eigenvalues
are subsequently obtained by floating-point numerical diagonalisation.

This is not an exact solution of the untruncated compact-U(1) rotor theory. The
hard-wall cutoff ``|E_l| <= Lambda`` makes the Hilbert space finite and causes
link raising/lowering operators to terminate at the cutoff boundary.

With lattice spacing set to ``a = 1`` and an irrelevant additive plaquette
constant omitted, the implemented Hamiltonian is

    H = (g^2 / 2) * sum_links E_l^2
        - (1 / (2 g^2)) * sum_plaquettes (P + P^dagger).

The finite lattice, cutoff, basis sector, and operator channel must accompany
any reported spectral gap. No continuum or infinite-volume claim is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class U1Lattice:
    """Specification of a compact-U(1) gauge theory on an ``Lx x Ly`` torus."""

    Lx: int = 2
    Ly: int = 2
    Lambda: int = 2  # electric-field truncation |E| <= Lambda
    periodic: bool = True
    basis_mode: str = "full"  # "full" (enumerate + Gauss filter) or "vacuum" (BFS from E=0)

    def __post_init__(self) -> None:
        if not self.periodic:
            raise NotImplementedError("Only periodic (toroidal) lattices are implemented.")
        if self.Lx < 2 or self.Ly < 2:
            raise ValueError("Use Lx, Ly >= 2; a 1x1 periodic lattice has trivial magnetic dynamics.")
        if self.basis_mode not in ("full", "vacuum"):
            raise ValueError("basis_mode must be 'full' or 'vacuum'")


class U1Model:
    """Builds the Gauss-law physical basis, Hamiltonian, and observables.

    The link ordering is deterministic: for each site ``(x, y)`` in row-major
    order we assign an x-link (pointing to ``(x+1, y)``) then a y-link (pointing
    to ``(x, y+1)``), all modulo the torus size.
    """

    def __init__(self, lattice: U1Lattice):
        self.lat = lattice
        self._build_geometry()
        self._build_plaquette_index()
        self._build_basis()

    # -- geometry ---------------------------------------------------------
    def _build_geometry(self) -> None:
        L = self.lat
        self.sites = [(x, y) for x in range(L.Lx) for y in range(L.Ly)]
        self.link_index: dict[tuple[str, tuple[int, int]], int] = {}
        n = 0
        for s in self.sites:
            self.link_index[("x", s)] = n
            n += 1
            self.link_index[("y", s)] = n
            n += 1
        self.n_links = n

    def _gauss_ok(self, cfg: tuple[int, ...]) -> bool:
        """No static charges: lattice divergence of E vanishes at every vertex."""
        L = self.lat
        li = self.link_index
        for (x, y) in self.sites:
            xm = ((x - 1) % L.Lx, y)
            ym = (x, (y - 1) % L.Ly)
            div = (
                cfg[li[("x", (x, y))]]
                + cfg[li[("y", (x, y))]]
                - cfg[li[("x", xm)]]
                - cfg[li[("y", ym)]]
            )
            if div != 0:
                return False
        return True

    def _build_basis(self) -> None:
        if self.lat.basis_mode == "full":
            self._build_basis_full()
        else:
            self._build_basis_vacuum()
        self.bmap: dict[tuple[int, ...], int] = {c: i for i, c in enumerate(self.basis)}
        self.dim = len(self.basis)

    def _build_basis_full(self) -> None:
        """Enumerate all electric configs and keep the Gauss-law-satisfying ones.

        Exact but costs ``(2 Lambda + 1) ** n_links`` before filtering; use only
        for small systems. Includes all electric-flux (winding) sectors.
        """
        L = self.lat
        vals = range(-L.Lambda, L.Lambda + 1)
        self.basis: list[tuple[int, ...]] = [
            cfg for cfg in product(vals, repeat=self.n_links) if self._gauss_ok(cfg)
        ]

    def _build_basis_vacuum(self) -> None:
        """Build the plaquette-connected zero-winding sector from ``E=0``.

        Plaquette operators preserve winding sectors, so plaquette observables
        cannot leave this block. For the published reference configurations the
        block ground state and plaquette-channel correlator agree with the
        corresponding full-basis quantities; callers changing lattice, cutoff,
        or Hamiltonian should verify that comparison explicitly.

        The block omits disconnected nonzero-winding sectors. Its first spectral
        gap is therefore a sector gap, not automatically the global finite-torus
        gap.
        """
        Lam = self.lat.Lambda
        start = tuple([0] * self.n_links)
        seen = {start}
        order = [start]
        frontier = [start]
        while frontier:
            nxt = []
            for cfg in frontier:
                for p in self.plaquettes:
                    for sign in (+1, -1):
                        c = self._shift(cfg, p, sign, Lam)
                        if c is not None and c not in seen:
                            seen.add(c)
                            order.append(c)
                            nxt.append(c)
            frontier = nxt
        self.basis = order

    @staticmethod
    def _shift(cfg, plaq, sign, Lam):
        b, r, t, l = plaq
        c = list(cfg)
        c[b] += sign
        c[r] += sign
        c[t] -= sign
        c[l] -= sign
        if any(abs(v) > Lam for v in c):
            return None
        return tuple(c)

    def _build_plaquette_index(self) -> None:
        L = self.lat
        li = self.link_index
        plaqs = []
        for (x, y) in self.sites:
            b = li[("x", (x, y))]
            r = li[("y", ((x + 1) % L.Lx, y))]
            t = li[("x", (x, (y + 1) % L.Ly))]
            l = li[("y", (x, y))]
            plaqs.append((b, r, t, l))
        self.plaquettes: list[tuple[int, int, int, int]] = plaqs

    # -- operators --------------------------------------------------------
    def _apply_plaquette(self, cfg, plaq, sign):
        """Raise (sign=+1) or lower (sign=-1) the plaquette. Returns new config or None."""
        b, r, t, l = plaq
        c = list(cfg)
        c[b] += sign
        c[r] += sign
        c[t] -= sign
        c[l] -= sign
        Lam = self.lat.Lambda
        if any(abs(v) > Lam for v in c):
            return None
        return tuple(c)

    def hamiltonian(self, g: float) -> np.ndarray:
        """Dense Hermitian Hamiltonian matrix at coupling ``g``."""
        D = self.dim
        H = np.zeros((D, D))
        inv2g2 = 0.5 / (g * g)
        halfg2 = 0.5 * g * g
        for i, cfg in enumerate(self.basis):
            H[i, i] += halfg2 * sum(e * e for e in cfg)  # electric term
            for p in self.plaquettes:
                for sign in (+1, -1):  # P and P^dagger
                    c = self._apply_plaquette(cfg, p, sign)
                    if c is None:
                        continue
                    j = self.bmap.get(c)
                    if j is not None:
                        H[j, i] += -inv2g2
        # symmetrise against floating error; H is exactly symmetric by construction
        return 0.5 * (H + H.T)

    def hamiltonian_sparse(self, g: float):
        """Sparse (CSR) Hamiltonian at coupling ``g`` for the Krylov path."""
        from scipy.sparse import csr_matrix

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        inv2g2 = 0.5 / (g * g)
        halfg2 = 0.5 * g * g
        for i, cfg in enumerate(self.basis):
            rows.append(i)
            cols.append(i)
            vals.append(halfg2 * sum(e * e for e in cfg))
            for p in self.plaquettes:
                for sign in (+1, -1):
                    c = self._apply_plaquette(cfg, p, sign)
                    if c is None:
                        continue
                    j = self.bmap.get(c)
                    if j is not None:
                        rows.append(j)
                        cols.append(i)
                        vals.append(-inv2g2)
        return csr_matrix((vals, (rows, cols)), shape=(self.dim, self.dim))

    def cos_plaquette_operator(self, which=None) -> np.ndarray:
        """Gauge-invariant operator ``sum_{p in which} cos(theta_p)``.

        With ``which=None`` uses the single plaquette 0 (a deliberately *local*,
        unsmeared interpolator). Pass a list of plaquette indices for smeared /
        zero-momentum operators.
        """
        if which is None:
            which = [0]
        D = self.dim
        M = np.zeros((D, D))
        for i, cfg in enumerate(self.basis):
            for pidx in which:
                p = self.plaquettes[pidx]
                for sign in (+1, -1):
                    c = self._apply_plaquette(cfg, p, sign)
                    if c is None:
                        continue
                    j = self.bmap.get(c)
                    if j is not None:
                        M[j, i] += 0.5
        return M

    def zero_momentum_cos(self) -> np.ndarray:
        """Zero-momentum (smeared) scalar operator: sum over all plaquettes."""
        return self.cos_plaquette_operator(list(range(len(self.plaquettes))))

    def cos_plaquette_operator_sparse(self, which=None):
        """Sparse (CSR) version of ``cos_plaquette_operator`` for the Krylov path."""
        from scipy.sparse import csr_matrix

        if which is None:
            which = [0]
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for i, cfg in enumerate(self.basis):
            for pidx in which:
                p = self.plaquettes[pidx]
                for sign in (+1, -1):
                    c = self._apply_plaquette(cfg, p, sign)
                    if c is None:
                        continue
                    j = self.bmap.get(c)
                    if j is not None:
                        rows.append(j)
                        cols.append(i)
                        vals.append(0.5)
        return csr_matrix((vals, (rows, cols)), shape=(self.dim, self.dim))

    def electric_energy_operator(self, plaq_idx: int = 0) -> np.ndarray:
        """Diagonal gauge-invariant operator: sum of E^2 over one plaquette's links."""
        D = self.dim
        M = np.zeros((D, D))
        links = self.plaquettes[plaq_idx]
        for i, cfg in enumerate(self.basis):
            M[i, i] = sum(cfg[k] ** 2 for k in links)
        return M


    def winding_numbers(self, cfg: tuple[int, ...]) -> tuple[int, int]:
        """Return electric flux through fixed non-contractible torus cuts.

        Gauss-law configurations have cut-independent totals. Plaquette moves
        preserve this pair, which labels the winding block represented by a
        basis configuration.
        """
        L = self.lat
        wx = sum(cfg[self.link_index[("x", (0, y))]] for y in range(L.Ly))
        wy = sum(cfg[self.link_index[("y", (x, 0))]] for x in range(L.Lx))
        return int(wx), int(wy)

    def winding_sectors(self) -> tuple[tuple[int, int], ...]:
        """Sorted winding sectors represented in the selected basis."""
        return tuple(sorted({self.winding_numbers(cfg) for cfg in self.basis}))

    def configuration_touches_cutoff(self, cfg: tuple[int, ...]) -> bool:
        """Whether any link in ``cfg`` lies on the hard-wall cutoff."""
        return any(abs(e) == self.lat.Lambda for e in cfg)

    # -- diagnostics ------------------------------------------------------
    def plaquette_preserves_gauss_law(self) -> bool:
        """Every plaquette operator must map physical states to physical states."""
        for cfg in self.basis:
            for p in self.plaquettes:
                for sign in (+1, -1):
                    c = self._apply_plaquette(cfg, p, sign)
                    if c is not None and not self._gauss_ok(c):
                        return False
        return True
