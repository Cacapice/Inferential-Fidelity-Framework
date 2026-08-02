# Release manifest — TransferMod v1.2.2

## Identity

- Package: `transfermod`
- Version: `1.2.2`
- License: AGPL-3.0-only
- Citation: `CITATION.cff`

## Primary documents

- `README.md` — orientation
- `methods.md` — theory and proofs
- `TUTORIAL_PERTURBATION_FAMILIES.md` — family construction
- `MIGRATIONS.md` — API migration
- `VALIDATION_REPORT.md` — measured empirical validation results
- `LEARNED_OPERATOR_PROTOCOL.md` — confirmatory external-model study protocol
- `ROADMAP.md` — validation programme
- `docs/releases/1.0/` — v1.0 audit records
- `docs/releases/1.2.2/RELEASE_NOTES.md` — interpretation correction and verification

## Reproduction commands

| Command | Output |
|---|---|
| `python scripts/run_graph_instance.py` | graph application and composition slack |
| `python scripts/run_sketch_nullspace_instance.py` | exact sketch-nullspace instance |
| `python scripts/run_pde_flux_instance.py` | POD hot-spot floor |
| `python scripts/run_pilot.py` | U(1) pilot result |
| `python scripts/run_scaling.py` | U(1) scaling and truncation checks |
| `python scripts/run_modulus.py` | spectral modulus results |
| `python scripts/run_feasibility.py` | feasibility and gate diagnostics |
| `python scripts/run_learned.py` | learned/reduced surrogate sweep |
| `python scripts/run_learned_operator_validation.py` | trained-operator aggregate/decision coupling audit |
| `python scripts/run_high_dimensional_pde_validation.py` | 2D/3D PDE grid-scaling validation |
| `python scripts/run_bayesian_silent_risk_validation.py` | posterior Silent Risk validation |

## Validation

Run:

```bash
pip install -e ".[test]"
pytest -q
python -m coverage run --branch -m pytest -q
python -m coverage report -m
```

The archive excludes caches and generated build artifacts.

## Verified release checks

- `147` tests passed.
- The latest completed branch-aware coverage measurement is the v1.2.0 baseline in `docs/releases/1.2/COVERAGE.md`; v1.2.2 adds semantic regression tests and does not claim a newly measured percentage.
- Editable installation (`--no-build-isolation`) and all six reference/validation smoke tests passed.
