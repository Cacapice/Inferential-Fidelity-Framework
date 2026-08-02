# Migrations

## v1.1 coverage API

The canonical model now stores one status: `CoverageTier`.

```python
from transfermod.certification import Coverage, modulus_result

coverage = Coverage.certified_floor(
    "Fiedler-directed family",
    "coverage of the full Laplacian ball is unproved",
)
result = modulus_result(value, coverage=coverage, ...)
```

Deprecated through v1.x and removed in v2.0:

| Deprecated | Replacement |
|---|---|
| `BoundStatus` / `result.status` | `CoverageTier` / `result.tier` |
| `ProvenCoverage(...)` | `Coverage.proven(...)` |
| `UnprovenCoverage(...)` | `Coverage.certified_floor(...)` |
| `ExploratorySampleCoverage(...)` | `Coverage.exploratory(...)` |
| `restricted_modulus_result(...)` | `modulus_result(...)` |

## Spectral gap naming

`exact_gap` remains a deprecated alias for `sector_gap` until v2.0. Use `sector_gap`, `global_physical_gap`, and `spectrum_scope`.

## v1.1.1 integration notes

Deprecated v1.x names remain available from `transfermod.certification`, but are
resolved lazily through `transfermod.compat`. Canonical imports no longer load
the compatibility module. `FamilySearchProfile.information_bases` now reports
only explicitly declared `RestrictedModulusResult.information_basis` values; use
`perturbation_families` for family names.


## v1.2 deprecation warnings

Every deprecated certification facade now gives a runtime nudge:

- `result.status` warns on property access;
- `BoundStatus` and `CoverageProof` warn when resolved from
  `transfermod.certification`;
- deprecated coverage constructors and result factories warn when called.

Use `CoverageTier`, `Coverage`, and `modulus_result` directly.
