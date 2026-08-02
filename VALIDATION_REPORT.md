# Empirical validation report — TransferMod v1.2.1

The three validations answer different questions. Their outcomes are not all
positive, and they are not aggregated into a single claim of success.

## Learned neural operator: negative decoupling result

A random-feature neural operator was trained on smooth initial conditions for a
one-dimensional diffusion map and challenged with a frozen 45-point family of
localized inputs.

The confirmatory region was declared as:

```text
relative global L2 error <= 0.010
decision error          >= 0.020
```

| Quantity | Value |
|---|---:|
| in-distribution validation p95 relative L2 | 0.000545 |
| localized family evaluations | 45 |
| points in confirmatory decoupling region | 0 |
| smallest localized relative global L2 error | 0.481730 |
| decision error at that point | 0.019916 |
| strongest decision error | 0.083332 |
| relative global L2 error at strongest decision point | 1.362591 |
| Pearson aggregate–decision correlation | 0.5615 |
| Spearman aggregate–decision correlation | 0.5816 |
| decision / absolute-global ratio range | 0.201–0.391 |

**Conclusion:** no decision failure invisible to aggregate validation was found
in this trained model and declared family. The largest decision error occurs
with conspicuous aggregate error, and the decision/global ratio remains bounded
over a narrow range. This instance demonstrates proportionate
out-of-distribution degradation, not Silent Risk.

The result is valuable as a negative finding. It limits the empirical claim:
TransferMod can represent and test decoupling, but this first trained-model
instance does not establish that decoupling is present or common in learned
operators.

The full 45-point profile is stored in
`results/learned_operator_validation.json`.

## Higher-dimensional PDE scaling

| Dimension | Spatial points | Uniform evals | Adaptive evals | Uniform best Q error | Adaptive best Q error |
|---:|---:|---:|---:|---:|---:|
| 2 | 1024 | 50 | 36 | 0.016477 | 0.016661 |
| 3 | 5832 | 250 | 108 | 0.021524 | 0.021524 |

The compact 3D adaptive search recovers the uniform-grid maximum with
108 rather than
250 evaluations. This supports
implementation feasibility at the reference scale, not a general scaling
claim for solver-coupled 3D workflows.

## Bayesian posterior Silent Risk

| Quantity | Value |
|---|---:|
| posterior draws | 20000 |
| posterior Silent Risk | 0.2008 |
| posterior fidelity radius | 0.3704 |
| credible-set worst case | 0.9489 |
| credible-set decision diameter | 0.9992 |
| recurrence probability | 0.0353 |

This validates the software separation between posterior weighting, decision
geometry, anchored fidelity, and reference-free variation. It does not by
itself establish behavior for non-conjugate or multimodal posteriors.

## Empirical status

The present empirical record is mixed:

- exact structured cases close mathematically;
- the U(1) and graph applications exhibit strong constructed or directed
  transfer failures;
- the first trained-operator study is a negative decoupling result;
- external pretrained-operator prevalence remains unknown.

The next study is governed by `LEARNED_OPERATOR_PROTOCOL.md`. Its outcome will
be reported whether decoupling is found or not.
