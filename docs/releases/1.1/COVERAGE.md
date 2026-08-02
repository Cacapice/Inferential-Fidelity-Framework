# Test coverage — TransferMod v1.1.1

Measured with:

```bash
python -m coverage run --branch -m pytest -q
python -m coverage report -m --include='src/transfermod/*'
```

Test result:

```text
........................................................................ [ 54%]
............................................................             [100%]
132 passed in 13.33s
```

Coverage report:

```text
Name                                                       Stmts   Miss Branch BrPart  Cover   Missing
------------------------------------------------------------------------------------------------------
src/transfermod/__init__.py                                    5      0      0      0   100%
src/transfermod/applications/__init__.py                       2      0      0      0   100%
src/transfermod/applications/u1.py                             4      0      0      0   100%
src/transfermod/certification/__init__.py                     17      2      2      1    84%   53, 59
src/transfermod/certification/coverage.py                     75      2      8      1    96%   85, 100
src/transfermod/compat.py                                     29      6      2      0    74%   45-46, 64-67
src/transfermod/exact_coverage.py                            127      9     38      4    92%   105, 116, 124-130, 178, 210, 214, 221
src/transfermod/family_profile.py                             34      0      2      0   100%
src/transfermod/modulus/__init__.py                            3      0      0      0   100%
src/transfermod/modulus/core.py                               81      5     16      5    90%   57, 74, 149, 206, 231, 263->266
src/transfermod/modulus/discrepancy.py                        97     11     52     11    85%   52, 65, 70, 87, 92, 97, 117, 119, 121, 166, 181
src/transfermod/perturbations.py                             114      4     16      2    95%   52-53, 141, 167->172, 179
src/transfermod/pipeline.py                                   71      4     26      5    91%   49-50, 61, 63, 67->72, 127->138
src/transfermod/reporting.py                                  10      2      4      2    71%   22, 24
src/transfermod/spectral/__init__.py                          23      3      6      3    79%   48, 58, 61
src/transfermod/spectral/analysis/__init__.py                  1      0      0      0   100%
src/transfermod/spectral/analysis/estimators.py              124     12     36      7    88%   29, 39, 63-64, 68, 85->87, 99-100, 104, 166-167, 171, 182
src/transfermod/spectral/gate/__init__.py                      2      0      0      0   100%
src/transfermod/spectral/gate/tests.py                       101      1     10      1    98%   176
src/transfermod/spectral/gate/thresholds.py                   12      0      0      0   100%
src/transfermod/spectral/lattice/__init__.py                   1      0      0      0   100%
src/transfermod/spectral/lattice/u1.py                       206     10     84      9    93%   42, 44, 46, 196->191, 220->215, 245->240, 257->259, 270->265, 278-283, 314
src/transfermod/spectral/observables/__init__.py               1      0      0      0   100%
src/transfermod/spectral/observables/suite.py                 48      2      4      1    94%   77-78
src/transfermod/spectral/reference/__init__.py                 1      0      0      0   100%
src/transfermod/spectral/reference/exact.py                  102      4      8      2    95%   63, 65->67, 183, 190, 193
src/transfermod/spectral/reference/krylov.py                  76      3      8      2    94%   33, 41->55, 112, 115
src/transfermod/spectral/surrogates/__init__.py               11      0      0      0   100%
src/transfermod/spectral/surrogates/base.py                   48      0      0      0   100%
src/transfermod/spectral/surrogates/deeponet.py                6      0      0      0   100%
src/transfermod/spectral/surrogates/fno.py                     6      0      0      0   100%
src/transfermod/spectral/surrogates/neural.py                 78      3      6      1    95%   103, 109, 132
src/transfermod/spectral/surrogates/operator.py                6      0      0      0   100%
src/transfermod/spectral/surrogates/pod_galerkin.py           56      2     10      2    94%   68->73, 75-76
src/transfermod/spectral/surrogates/reduced_transfer.py       24      3      4      1    86%   40-42
src/transfermod/spectral/surrogates/spectral_shortcut.py      33      1      6      2    92%   74->76, 79
src/transfermod/spectral/surrogates/trained.py                49      6     12      2    87%   56, 68-70, 74, 98, 102
src/transfermod/spectral/taxonomy.py                          29      7      6      0    74%   84, 95-100
------------------------------------------------------------------------------------------------------
TOTAL                                                       1713    102    366     64    92%
```

Coverage is measured over `src/transfermod` with branch coverage enabled.
Coverage percentage is not a substitute for theorem or domain validation;
semantic regression tests separately cover proof scope, coverage status, and
compact-U(1) sector interpretation.
