# How to Construct a Perturbation Family for Your Surrogate

TransferMod computes a worst-case statement over an admissible family. The
practical bottleneck is therefore not evaluating a supremum; it is constructing
a family of alternatives that reflects how the validation geometry can fail.

## 1. Identify the validated object

Write down the object the validation actually constrains: a state vector,
operator, graph Laplacian, correlator, PDE field, reduced basis, or learned
surrogate output.

## 2. Declare the validation geometry

Specify the quantity called `admissibility_error` in the API. A seminorm is
allowed, but its kernel must be treated as scientifically meaningful: if the
quantity of interest varies along directions assigned zero validation cost, the
transfer problem is ill-posed.

## 3. Find weakly penalized directions

Look for alternatives that are small in validation geometry but influential for
the conclusion:

- low-energy spectral modes hidden by aggregate loss;
- sparse graph bridges missed by generic sampled directions;
- residual fields concentrated outside the validation window;
- parameter combinations lying in a sloppy or unidentified subspace;
- representation changes invisible to a probe but relevant to behavior.

## 4. Select a template

### Additive family

```python
family = AdditivePerturbation(
    direction=v,
    validation_metric=validation_error,
)
```

implements `u_w = u + w v`.

### Parametric residual family

```python
family = ParametricResidualPerturbation(
    residual_generator=lambda w: make_residual(w, parameters),
    validation_metric=validation_error,
)
```

supports nonlinear residual generators.

### Spectral mode injection

```python
family = SpectralModeInjection(
    times=t,
    mode_rate=gap_candidate,
    validation_metric=window_error,
)
```

adds a faint exponential mode.

### Graph edge reweighting

```python
family = GraphEdgeWeightPerturbation(
    weakened_edges=bridges,
    redistribution_edges=intra_cluster_edges,
)
```

tests whether aggregate Laplacian fidelity hides connectivity loss.

## 5. Evaluate the family

For each amplitude `w`:

1. construct `u_w = family.perturb(u, w)`;
2. compute `family.admissibility_error(u, u_w)`;
3. evaluate the scientific quantity `Q(u_w)`;
4. evaluate the declared decision discrepancy.

## 6. State coverage honestly

```python
from transfermod.certification import Coverage
```

A searched family is rarely the whole admissible set. Use:

```python
coverage = Coverage.certified_floor(
    family.name,
    "The family explores one physically motivated direction.",
)
```

The certification result will then be tagged `LOWER_BOUND`.

Use `Coverage.proven(...)` only when a theorem, exhaustive enumeration, or certified
cover proves that the family spans every relevant admissible alternative.

## 7. Add directions before adding confidence

If a result is alarming, add structurally different families and information
bases. Agreement among several parameterizations of the same direction is a
robustness check, not proof of full coverage.


## Coverage tier checklist

Before publishing a result, decide which statement is justified:

- **Tier 1 — Proven Exact:** a theorem or exhaustive finite enumeration proves
  coverage.
- **Tier 2 — Certified Adversarial Floor:** the family is defined and searched
  exhaustively, but no theorem proves it covers the full admissible set.
- **Tier 3 — Exploratory Sample:** a grid, random sample, or optimizer run may
  not even attain the family supremum.

Use Tier 2 aggressively for falsification: a large floor is already decisive.
Do not use a small Tier-2 or Tier-3 result as evidence of safety.
