# TransferMod

**Release:** v1.2.2 · stable research API

**Canonical project name:** **TransferMod**. The GitHub repository and source archive retain the historical alias **Inferential Fidelity Framework**; the installable Python package is `transfermod`.

TransferMod computes finite-tolerance error in a declared quantity of interest over the approximations still admitted by a validation contract. It returns an exact restricted modulus when coverage is proved and a certified one-sided floor when it is not.

> Given a validated approximation, what conclusions are actually licensed?

### Declared search domains and publication guards

Threshold serialization distinguishes the final numerical ``bracket`` from the original ``search_domain``. When ``lo`` or ``hi`` is omitted, ``bracket_source`` is ``"library_default"`` and the payload records that the amplitude domain is a modeling assumption that should be checked against the perturbation family. Pass both endpoints explicitly to declare the domain as caller supplied.

Use ``strict_publish(result)`` (or ``result.strict_publish()``) at export time. It returns the ordinary serialized payload for publishable results and raises when censoring leaves the scalar indeterminate. This preserves float-compatible arithmetic while preventing metadata-laundered values from crossing a realistic publication boundary.


## Thirty-second example

A graph sparsifier can preserve a Laplacian in aggregate while materially changing algebraic connectivity.

```bash
python scripts/run_graph_instance.py
```

| Detection strategy | Log-amplitude Silent Risk |
|---|---:|
| 32 sampled Rayleigh directions | ≈ 0.891 decades |
| Fiedler-directed check | ≈ 0.013 decades |

The aggregate/output distinction is established in goal-oriented error estimation and certified reduced-basis methods. TransferMod focuses on a complementary question: what can an arbitrary validation-and-gating procedure certify at a finite operating tolerance, especially when the searched family does not cover every admitted approximation?

## One conceptual model

TransferMod begins with three declared objects:

```text
Admissible family D   Decision map Q   Decision discrepancy ℓ
         \              |              /
          \             |             /
           └──── fidelity modulus ω ───┘
```

Everything else is derived:

| Object | Meaning |
|---|---|
| `ω` | worst-case transfer from validation error to decision error |
| coverage | whether the searched family equals the full admissible family |
| tier | epistemic status of the returned value |
| gate | restriction of `D` |
| decision diameter | reference-free variation of `Q(D)` |
| Silent Risk | measure of admitted alternatives exceeding a decision tolerance |

## Quick start

```bash
pip install -e .
pytest -q
```

```python
from transfermod.certification import Coverage
from transfermod.modulus import CertificationGeometry, certify_ray_modulus

geometry = CertificationGeometry.from_standard_metric(
    metric="MAE", reference_q=1.7, tolerance=0.10
)

result = certify_ray_modulus(
    q=q_along_family,
    agg=validation_error_along_family,
    eps=0.005,
    geometry=geometry,
    coverage=Coverage.certified_floor(
        "Fiedler-directed family",
        "full-family coverage is unproved",
    ),
    perturbation_family="Fiedler-directed family",
)

print(result.tier.value)
print(result.to_dict())
```

## Coverage tiers

| Tier | Claim |
|---|---|
| `PROVEN_EXACT` | the returned value is the full modulus |
| `CERTIFIED_FLOOR` | the full modulus is at least this large |
| `EXPLORATORY_SAMPLE` | empirical sensitivity only |

> **Large floors condemn; small floors do not acquit.**

## Exact theorem implementations

- continuous linear functionals over Hilbert balls;
- linear functionals over ellipsoidal or seminorm balls;
- sketch-nullspace corruption with a source-space radius;
- leading-eigenvalue shift under unrestricted symmetric Frobenius perturbations.

```bash
python scripts/run_sketch_nullspace_instance.py
```

## Scientific and empirical validations

| Validation | Quantity of interest | Status |
|---|---|---|
| Compact U(1), finite cutoff and declared winding sector | operator-channel spectral gap | certified adversarial floor |
| Graph sparsification | algebraic connectivity `λ₂` | certified adversarial floor |
| POD hot spot | localized maximum temperature | certified adversarial floor |
| Trained random-feature diffusion operator | aggregate/decision coupling under localized inputs | negative result: no small-aggregate/large-decision point found |
| 2D/3D localized PDE fields | local maximum under uniform versus adaptive grid search | computational-scaling validation |
| Drift–diffusion posterior | posterior Silent Risk and credible-set decision exposure | probabilistic validation |

```bash
python scripts/run_learned_operator_validation.py  # full coupling profile
python scripts/run_high_dimensional_pde_validation.py
python scripts/run_bayesian_silent_risk_validation.py
```

The U(1) implementation is a finite-matrix numerical reference, not an exact solution of the untruncated or continuum theory.

## Relation to prior work

TransferMod builds on, rather than replaces, the established literature on output-oriented error analysis, including goal-oriented estimation, certified reduced-basis methods, Optimal Uncertainty Quantification, and backward error analysis. Rather than revisiting the distinction between aggregate and output error, its contribution is to refine how output fidelity itself is decomposed and certified under explicit perturbation families and coverage assumptions.

- **Goal-oriented and dual-weighted-residual estimation** bound error in a user-declared output instead of only a global solution norm.
- **Certified reduced-basis methods** provide rigorous output bounds for parametrized approximations.
- **Optimal Uncertainty Quantification** computes sharp objective bounds over scenarios compatible with a declared assumptions-and-information set.
- **Backward error and conditioning** connect nearby admissible problems to forward output error.

TransferMod's narrower contribution is the explicit separation of the full modulus, gate-restricted modulus, and family-restricted computable floor; the coverage condition that distinguishes exact certification from one-sided condemnation; and finite-tolerance diagnostics for regimes where local condition-number reasoning is no longer predictive. The constructed-confound search shares the constrained-maximization skeleton of adversarial examples, but its objects are scientific approximations and quantities of interest. See [`methods.md`](methods.md) for the detailed comparison and references.

## Repository map

| Path | Purpose |
|---|---|
| `methods.md` | complete theory and proofs |
| `src/transfermod/` | domain-independent implementation |
| `src/transfermod/applications/` | reference-application facades |
| `TUTORIAL_PERTURBATION_FAMILIES.md` | building a domain-specific family |
| `MIGRATIONS.md` | deprecated API and upgrade guide |
| `VALIDATION_REPORT.md` | measured learned-operator, PDE-scaling, and Bayesian results |
| `LEARNED_OPERATOR_PROTOCOL.md` | confirmatory external-model study protocol |
| `ROADMAP.md` | completed validations and remaining scale-up work |
| `RELEASE_MANIFEST.md` | archive inventory and reproduction commands |
| `docs/releases/1.0/` | v1.0 audit records |
| `docs/releases/1.2/COVERAGE.md` | latest completed coverage baseline (v1.2.0) |

## Mathematical status

The methods note treats the order structure as supporting organization. Its main results concern finite-tolerance conditioning, family-restricted versus full certification, coverage-aware effective computation, and measure-qualified Silent Risk. Family-restricted statements become full-modulus statements only under a coverage proof.

## License and citation

AGPL-3.0-only. Citation metadata are in `CITATION.cff`.
