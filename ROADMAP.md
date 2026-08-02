# Validation programme

TransferMod v1.2.1 treats empirical scope as an open question rather than an
assumption.

| Track | Current evidence | Interpretation |
|---|---|---|
| Learned operators | the trained random-feature diffusion operator shows no point with relative global L2 error at or below 0.01 and decision error at or above 0.02 | negative result for decoupling in this model/family; aggregate and decision degradation remain coupled |
| Higher-dimensional PDEs | compact 2D/3D uniform and coarse-to-fine searches execute successfully | implementation-feasibility result only; production solver scaling remains open |
| Bayesian Silent Risk | conjugate drift–diffusion posterior cleanly separates posterior weighting, anchored fidelity, worst-case fidelity, and decision diameter | successful implementation validation, not evidence of broad posterior robustness |

## Immediate confirmatory priority

Run the frozen protocol in `LEARNED_OPERATOR_PROTOCOL.md` on externally trained
operators that were neither designed nor trained for TransferMod.

The first confirmatory study should include:

- publicly released FNO and DeepONet checkpoints;
- model and checkpoint hashes fixed before analysis;
- predeclared aggregate metrics, quantities of interest, thresholds, families,
  and search budgets;
- complete reporting of coupled, decoupled, and inconclusive outcomes;
- no post-result substitution of perturbation families in the confirmatory
  analysis.

## Remaining scale-up work

- external learned-operator replication;
- solver-coupled two- and three-dimensional PDE studies with measured hardware
  cost;
- non-conjugate, multimodal, and MCMC posterior Silent Risk;
- preregistered comparative studies across models, metrics, quantities, and
  information bases.

The scope question is now explicit: **under what conditions do aggregate and
decision fidelity decouple in real trained systems?** A negative answer for a
particular model–metric–quantity–family combination is a first-class result.
