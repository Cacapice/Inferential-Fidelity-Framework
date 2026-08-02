# Confirmatory learned-operator study protocol

## Research question

Does decision error decouple from the aggregate validation metric in trained
operators that were not designed or trained for TransferMod?

The confirmatory endpoint is the existence and magnitude of a
**small-aggregate / large-decision** region. Ordinary out-of-distribution
degradation, where both errors rise together, is a negative result for this
question.

## Separation of stages

### Exploratory stage

The exploratory stage may develop candidate quantities of interest,
perturbation parameterizations, and computational search methods. Every change
made after observing results is logged. Exploratory findings do not count as
confirmatory evidence.

### Confirmatory stage

Before evaluating held-out models, freeze:

1. model repositories, checkpoints, and hashes;
2. benchmark datasets and data splits;
3. aggregate validation metric \(A\);
4. decision functional \(Q\);
5. aggregate threshold \(\varepsilon_0\);
6. decision threshold \(\tau\);
7. perturbation families and parameter domains;
8. search algorithm, budget, seeds, and stopping rules;
9. exclusion and numerical-failure rules;
10. primary and secondary analyses.

A failed confirmatory search is reported without replacing the family or
quantity of interest post hoc.

## Model selection

Use externally trained and publicly released operator checkpoints. The first
study should include at least:

- one Fourier Neural Operator;
- one DeepONet or related branch–trunk operator;
- two PDE families or parameter regimes;
- multiple independently trained checkpoints where available.

Selection must occur before inspecting TransferMod profiles. Models should be
chosen for scientific relevance and reproducibility, not because preliminary
searches suggest decoupling.

## Primary endpoint

For each model, evaluate the predeclared family

\[
\Theta=\{\theta: u_\theta\text{ satisfies all perturbation constraints}\}.
\]

Define

\[
H_{\varepsilon_0,\tau}
=
\{\theta\in\Theta:
A(u_\theta,u)\le\varepsilon_0,\;
\ell(Q(u_\theta),Q(u))\ge\tau
\}.
\]

Report:

- whether \(H_{\varepsilon_0,\tau}\) is empty;
- the largest decision error within \(A\le\varepsilon_0\);
- the smallest aggregate error among points with decision error at least
  \(\tau\);
- the empirical aggregate–decision frontier;
- family coverage tier and search completeness.

The headline is determined by this endpoint, not by in-distribution validation
performance.

## Secondary analyses

Report without promoting them to the primary claim:

- Pearson and rank correlation between aggregate and decision error;
- decision-to-aggregate error ratios;
- results under alternative predeclared thresholds;
- cross-checks across information bases;
- computational cost and convergence diagnostics.

## Interpretation rules

- **Decoupling found:** at least one confirmatory point lies in
  \(H_{\varepsilon_0,\tau}\).
- **No decoupling found:** the confirmatory family search is complete and
  \(H_{\varepsilon_0,\tau}=\varnothing\).
- **Inconclusive:** numerical failure, incomplete search, or invalidated
  preconditions prevent evaluation.

“No decoupling found” is a first-class finding about the tested
model–metric–quantity–family combination. It is not evidence that decoupling is
impossible elsewhere.

## Reporting commitment

Release all outcomes, including null and coupled profiles. Do not replace a
negative confirmatory instance with a newly designed successful family.
Subsequent family design belongs to a separately labeled exploratory study.

## Scope inference

Claims must remain indexed by:

- trained model and checkpoint;
- training distribution;
- aggregate metric;
- quantity of interest;
- perturbation family;
- threshold pair;
- search coverage.

The study tests the empirical prevalence and conditions of decoupling. It does
not test the mathematical validity of the fidelity modulus itself.
