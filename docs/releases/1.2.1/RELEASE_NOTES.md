# TransferMod v1.2.1

## Learned-operator interpretation correction

The first trained-operator instance is a negative result for aggregate/decision decoupling. Across the frozen localized family:

- no point satisfies relative global L2 error `<= 0.01` and decision error `>= 0.02`;
- the smallest relative global L2 error is approximately `0.482`;
- the strongest decision error is approximately `0.083`, at relative global L2 error approximately `1.363`;
- decision error remains proportionate to absolute global error, with ratios approximately `0.20` to `0.39`.

The release no longer describes this as a failure hidden beneath passing aggregate validation. It demonstrates visible out-of-distribution degradation.

## Future empirical protocol

`LEARNED_OPERATOR_PROTOCOL.md` freezes external checkpoints, metrics, quantities, thresholds, perturbation families, search budgets, and reporting rules before confirmatory evaluation. Negative and coupled outcomes are published without post-hoc family replacement.

## Verification

- 145 tests passed.
- The complete 45-point profile is included in `results/learned_operator_validation.json`.
- The latest completed branch-aware coverage measurement remains the v1.2.0 baseline in `docs/releases/1.2/COVERAGE.md`.
