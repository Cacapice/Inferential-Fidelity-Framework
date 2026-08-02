# v1.0.0 audit resolution — TransferMod v1.0.1

## Spectral / compact-U(1) corrections

- Replaced broad “exact” language with finite-cutoff, finite-volume,
  selected-sector numerical-diagonalisation language.
- Stated `a=1`, the omitted additive plaquette constant, and the hard-wall
  truncation explicitly.
- Retained `exact_gap` only as a compatibility field; added explicit
  `sector_gap`, `basis_mode`, `spectrum_scope`, and `global_physical_gap`.
- Added winding-sector labels and winding-block preservation tests.
- Added full-basis versus zero-winding ground-state and plaquette-channel tests.
- Added separate strong-coupling tests for the all-sector gap (`g^2`) and the
  zero-winding plaquette-sector gap (`2g^2`) on the 2x2 torus.
- Added ground-state hard-wall boundary-occupation diagnostics.

## Proof corrections

- Proposition 1 now treats the guarantee set as a principal upper set and
  defines additive slack only on the finite locus.
- Theorems 2 and 2-prime now certify `omega_{D,Theta}` unless coverage condition
  C1 is proved.
- The differentiability section states only the proved implication from
  Frechet differentiability of `Q` to a first-order expansion of the modulus.
- Proposition 2 separates exact positive homogeneity from asymptotic linearity.
- Theorem 5 separates existence/attainment from effective finite computation
  and adds an effective continuity hypothesis C4.

## Verification

- 118 tests pass.
- Combined measured line/branch coverage is recorded in `docs/releases/1.0/COVERAGE.md`.

### Spectral-gap API migration

`exact_gap` is deprecated and will be removed in TransferMod 2.0. It is a
warning-emitting compatibility alias for `sector_gap`, not a separate global
quantity. New code should use:

| Legacy name | Canonical replacement |
|---|---|
| `exact_gap` | `sector_gap` |
| — | `global_physical_gap` when the full all-sector basis is available |
| — | `spectrum_scope` and `basis_mode` to state interpretation |

