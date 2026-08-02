# Test coverage — TransferMod v1.2.0

Measured with:

```bash
python -m coverage run --branch -m pytest -q
python -m coverage report -m --include='src/transfermod/*'
```

```text
141 passed
1,959 statements
111 missed statements
406 branches
73 partial branches
92% combined line/branch coverage
```

The validation modules are included in the measured source tree:

| Module | Combined coverage |
|---|---:|
| `validation/learned_operator.py` | 88% |
| `validation/pde_scaling.py` | 97% |
| `validation/bayesian.py` | 95% |

Coverage measures exercised code paths; it does not establish theorem,
physics, or empirical generality. Those claims are tested separately through
semantic regression tests and the scoped validation report.
