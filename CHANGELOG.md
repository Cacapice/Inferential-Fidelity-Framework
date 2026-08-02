# Changelog

- Publication payloads now expose the effective epistemic `tier` as a top-level field; the existing `coverage_tier` key remains for backward compatibility.


- Declared threshold search domains separately from converged numerical brackets, including whether endpoints came from library defaults or the caller.
- Added ``strict_publish`` to reject indeterminate censored results at serialization boundaries without breaking float-compatible arithmetic.

## Unreleased

_No changes yet._


## 1.2.2

### Manuscript and repository editorial revision

- Recentered the methods note on finite-tolerance, coverage-aware certification.
- Added a result roadmap emphasizing conditioning crossover, restricted moduli,
  coverage, and effective computation.
- Positioned TransferMod against established Optimal Uncertainty Quantification,
  goal-oriented/DWR error estimation, certified reduced-basis output bounds, and
  backward error analysis.
- Clarified in the README that output-oriented error analysis is established
  prior art and that TransferMod's contribution is narrower.
- Compressed repeated order-theoretic exposition and presented the structural
  calculus as basic properties used later.
- Normalized theorem numbering by section and clarified the hypotheses required
  by the second-order full-modulus expansion.
- Renamed `methods_reordered.md` to `methods.md` and updated repository links.
- Restored Python 3.10 compatibility by replacing `enum.StrEnum` with the
  equivalent `str, Enum` declaration for `CouplingConclusion`.
- Updated GitHub Actions to `actions/checkout@v6` and `actions/setup-python@v6`,
  which use the Node.js 24 runtime.
- Added CI compatibility regressions covering supported enum semantics and
  workflow action versions.

## 1.2.1

- Reclassified the learned-operator study as a negative decoupling result rather than a hidden validation failure.
- Added the complete 45-point aggregate–decision profile, matching relative-L2 aggregate errors, decision/global ratios, Pearson and rank correlations, and a predeclared small-aggregate/large-decision endpoint.
- Recorded that no family member enters the confirmatory decoupling region; the smallest localized relative-L2 error is itself far above the aggregate threshold.
- Added `LEARNED_OPERATOR_PROTOCOL.md`, a confirmatory protocol for externally trained FNO/DeepONet studies with frozen checkpoints, metrics, quantities, thresholds, families, search budgets, and reporting rules.
- Revised the README, methods note, validation report, and roadmap so null and coupled outcomes are first-class findings and cannot be replaced post hoc by a successful constructed family.

## 1.2.0

- Made deprecation warnings uniform across `exact_gap`, `result.status`, type-like certification facades, constructors, and result factories.
- Added a trained random-feature neural diffusion-operator validation that passes smooth-distribution validation but exposes a localized out-of-distribution decision failure.
- Added executable 2D and 3D PDE grid-scaling validation comparing uniform and coarse-to-fine searches.
- Added a conjugate Bayesian drift–diffusion validation for posterior Silent Risk, fidelity radius, credible-set worst case, recurrence probability, and decision diameter.
- Promoted the three former roadmap items into `transfermod.validation` and added reproduction scripts and regression tests.

## 1.1.1

- Integrated v1.1 cleanup patches into one canonical implementation.
- Corrected the methods-note overview to match finite-locus slack semantics.
- Made deprecated certification exports lazy so the canonical namespace no longer imports the compatibility layer eagerly.
- Added result validation for NaN/negative values and optional information-basis metadata.
- Corrected `FamilySearchProfile.information_bases` and made all-exploratory profiles well-defined.
- Added integration regressions for canonical imports, metadata, and exploratory-only profiles.

## 1.1.0

- Reduced the README to one conceptual model and moved roadmap, migration, release, audit, and coverage material to dedicated documents.
- Replaced duplicated stored `BoundStatus`/`CoverageTier` state with one canonical `CoverageTier`; deprecated names are derived compatibility facades through v1.x.
- Unified coverage provenance in `Coverage` with `proven`, `certified_floor`, and `exploratory` constructors.
- Added `modulus_result`, `to_dict`, and a separate reporting module.
- Consolidated exact-certification boilerplate and perturbation weight validation.
- Added the `transfermod.applications` namespace while retaining the historical spectral path.
- Replaced `MANIFEST.md` with the release-focused `RELEASE_MANIFEST.md`; moved audit records under `docs/releases/1.0`.

## 1.0.2

- Made `sector_gap` the canonical stored spectral gap.
- Converted `exact_gap` into a warning-emitting compatibility alias scheduled for removal in v2.0.
- Added constructor compatibility for legacy `exact_gap=` callers while warning on use.
- Migrated internal code, examples, tests, and result keys to sector-aware terminology.
- Added migration documentation and deprecation regression tests.

## 1.0.1

- Corrected compact-U(1) exactness language: references are full numerical diagonalizations of declared finite-cutoff basis blocks, not exact solutions of the untruncated theory.
- Made spectral scope explicit through `basis_mode`, `spectrum_scope`, `sector_gap`, and `global_physical_gap`.
- Added winding-sector and hard-wall-cutoff diagnostics, including ground-state boundary occupation.
- Corrected Proposition 1 on extended-real slack, Theorems 2/2′ on family coverage, Theorem 3's one-way differentiability implication, Proposition 2's asymptotic slack statement, and Theorem 5's effective-computability hypotheses.
- Added regression tests for winding-block preservation, full-versus-zero-winding channel agreement, strong-coupling sector scaling, cutoff diagnostics, proof wording, and previously uncovered API branches.
- Added measured line/branch coverage reporting to the release validation workflow.

## 1.0.0

- Declared the stable research-reference release.
- Reconciled README, methods note, results provenance, manifest, package metadata, and public layout with the three-tier instance hierarchy.
- Removed stale test-count and “two worked instances” claims.
- Corrected the manifest so exact Tier-1 results are distinguished from Tier-2 certified adversarial floors and Tier-3 exploratory samples.
- Added `CITATION.cff` and project metadata.
- Added automated release-consistency tests for versioning, local documentation links, manifest paths, and license metadata.

## 0.9.2

- Replaced the stale “Two worked instances” manifest section with an explicit instance hierarchy.
- Distinguished Tier-1 exact coverage demonstrations from Tier-2 scientific reference applications and Tier-3 planned empirical validations.
- Added sketch-nullspace, leading-eigenvalue, and POD hot-spot entries to the manifest.
- Updated stale manifest version and test-count claims.

## 0.9.1

- Added an explicit empirical-validation roadmap for learned neural operators and trained PDE surrogates.
- Added a higher-dimensional PDE scaling agenda for the finite-grid procedure in Theorem 5.
- Added a Bayesian posterior instance specification for measure-qualified Silent Risk.
- Clarified which claim each extension would test and that none is presented as completed validation in this release.

## 0.9.0

- Added three coverage tiers: proven exact, certified adversarial floor, and exploratory sample.
- Added exact-coverage theorems for Hilbert-ball linear functionals, ellipsoidal linear functionals, sketch-nullspace queries, and leading eigenvalues under unrestricted symmetric Frobenius perturbations.
- Added explicit infinite-modulus detection when the quantity of interest varies on a validation seminorm kernel.
- Added `FamilySearchProfile` for strongest-floor and between-family comparisons without treating convergence as proof of coverage.
- Added exact sketch-nullspace and Tier-2 POD hot-spot scripts.
- Added the one-sided interpretation: “Large floors condemn; small floors do not acquit.”
- Added eight regression tests for exact coverage, coverage tiers, infinite exposure, and family-profile semantics.

## 0.8.0

- Added coverage-aware `RestrictedModulusResult` values with explicit `EXACT` versus `LOWER_BOUND` status.
- Added `CoverageProof`, `ProvenCoverage`, and `UnprovenCoverage`, plus strict exactness checks.
- Added `certify_ray_modulus` as the coverage-aware certification boundary.
- Added perturbation-family templates and a surrogate-family construction tutorial.
- Added `PipelineCompositionResult`, direct-versus-stagewise slack diagnostics, openness metadata, and contractivity certificates.
- Added `CertificationGeometry.from_standard_metric` with practitioner aliases and informative near-zero percentage-error guards.
- Reordered the README around the graph-sparsification example and retained compact U(1) as the advanced exact stress test.
- Added an intuition box after Definition 2.
- Added 13 regression tests for coverage status, metric factories, perturbation templates, and pipeline diagnostics.

## 0.7.1

- Positioned the constructive confound as adversarial scientific computing and connected it to adversarial-example and robust-optimization literature.
- Distinguished deterministic admissible-set transfer bounds from probabilistic and Bayesian uncertainty quantification.
- Added a practitioner translation table for the core terminology.
- Added the graph-sparsification instance to the methods paper.
- Added scoped open questions on contractive intermediate maps and the tensor-product `log(1 + ω)` transform.
- Added selected adjacent references without changing theorem statements or APIs.

## 0.7.0

- Generalized inferential fidelity to an explicitly declared decision-discrepancy geometry.
- Added absolute, relative, stabilized-relative, and symmetric-relative scalar discrepancies.
- Added a near-zero guard for relative discrepancy.
- Added a reference-free sampled decision diameter as a companion to anchored moduli.
- Unified Silent Risk as the measure of admitted decision-corrupting alternatives; retained the existing log-amplitude width under an explicit name.
- Added certification geometry metadata recording discrepancy, reference, tolerance, and stabilization.
- Preserved all prior relative-error APIs as backward-compatible special cases.

## 0.6.0

- Added information-basis metadata to spectral gate results.
- Distinguished geometric gate composition from evidential independence.
- Reframed alternative spectral estimators as cross-estimator robustness checks.
- Added the relationship to Benchmark Stewardship and the unified information/transfer perspective.
- Corrected stale methods-note paths and replaced hard-coded test-count claims with release verification.

## 0.5.0

- Prior public reference release.
