# Certifying quantities of interest from aggregate approximation guarantees

*Methods note: a coverage-aware, finite-tolerance theory of how approximation
contracts constrain errors in derived scientific quantities. Compact U(1) spectral
inference, graph sparsification, and a POD hot-spot example provide qualified
reference applications; numerical values belong only to their stated instances.*

**Main contribution.** The note distinguishes the full fidelity modulus, a
certification-restricted modulus, and a family-restricted computable floor. It
identifies when a searched family yields an exact certificate, when it yields only
a one-sided condemnation result, and when local condition-number reasoning has
already failed at the operating tolerance.

**Supporting structure.** Elementary order properties organize the admissible
guarantees and restrictions. They are used as bookkeeping and composition tools,
not claimed as the principal mathematical novelty.

---

## 1. Problem

Approximations are validated in norms that aggregate error over the state space;
conclusions are drawn from a small set of derived quantities. A reduced-order
model is accepted on residual error and read for a stability margin; a surrogate
is accepted on held-out loss and read for a decay rate. The validated quantity
and the relied-on quantity differ, and nothing in the validation says whether the
first controls the second.

This note studies what can be certified at an operating tolerance, not merely
in the infinitesimal limit. It separates the full admissible discrepancy from
the discrepancy remaining after certification gates and from the value found by
a searched perturbation family. The main questions are whether that search has
coverage, whether local conditioning remains predictive at the tolerance in
use, and whether the resulting restricted modulus can be computed with a
certified numerical error.


### 1.1 Closest prior work and the remaining gap

The premise that global or aggregate approximation error need not control a
specific output is established prior art. Goal-oriented a posteriori error
estimation, including the dual-weighted residual method, estimates error in a
user-declared functional rather than only in an energy or residual norm
(Becker and Rannacher, 2001; Oden and Prudhomme, 2001; Prudhomme and Oden,
1999). Certified reduced-basis methods likewise provide rigorous output bounds
for parametrized models and use those bounds to drive efficient approximation
(Patera and Rozza, 2007; Rozza, Huynh, and Patera, 2008). TransferMod therefore
does **not** claim to discover the distinction between state error and output
error.

Optimal Uncertainty Quantification (OUQ) is the closest optimization-level
relative. OUQ places the objective and the assumptions/information set at the
center and computes optimal bounds by extremizing over compatible scenarios
(Owhadi et al., 2013). The fidelity modulus has the same broad max-over-an-
admissible-set form. The difference is one of declared object and operational
use: here the admissible set is induced by an approximation contract and,
optionally, by certification gates; the objective is discrepancy in a specified
quantity of interest at a finite operating tolerance; and an incomplete search
is reported explicitly as a family-restricted lower bound rather than as the
full optimum.

Backward error analysis supplies another close conceptual precedent. It asks
whether a computed answer is exact for a nearby problem, while conditioning
translates that nearby-problem statement into forward error. TransferMod begins
from the corresponding approximation neighborhood and studies the complete
finite-tolerance output discrepancy, including regimes in which the local
condition number is finite but no longer predictive.

The contribution claimed here is consequently narrower:

1. a single notation separating the **full modulus** `ω`, the gate-restricted
   modulus `ω_D`, and a computable family-restricted value `ω_{D,Θ}`;
2. a coverage criterion distinguishing exact certification from a certified
   one-sided floor;
3. finite-tolerance diagnostics for the regime in which first-order
   conditioning has failed;
4. gates treated as restrictions of the admissible family, with Silent Risk as
   a measure-qualified summary of admitted decision-corrupting alternatives;
5. constructive and empirical reference cases that report negative as well as
   positive findings.

Classical UQ, robust optimization, and adversarial robustness remain relevant
but are not the closest novelty comparators. Classical probabilistic UQ
propagates a distribution, posterior, or ensemble to output uncertainty. Robust
optimization chooses a decision under an uncertainty set. Adversarial examples
construct small perturbations that alter a classifier. TransferMod instead asks
for the sharpest finite-tolerance implication from a declared approximation and
certification contract to a declared scientific conclusion. A probabilistic
credible set may define the admissible family, and a goal-oriented estimator may
supply an upper bound, but neither identification changes the epistemic status
of an uncovered family search.

### 1.2 Roadmap of the substantive results

Section 2 fixes notation and only the structural facts needed later. Section 3
separates four finite-tolerance regimes and identifies the crossover beyond
which first-order conditioning is no longer predictive. Sections 4–6 develop
restricted moduli, coverage-aware certification, and effective computation.
Section 7 tests those claims on constructed and empirical references, including
negative results. The exact structured cases then show where coverage can be
proved rather than assumed.

## 2. Transfer guarantees and the fidelity modulus

### 2.0 Declared decision geometry

The general inferential object is specified by an approximation seminorm, a
decision functional `Q`, a target `u`, and a declared discrepancy
`ℓ : Y × Y → [0,∞]` on the decision space. Define

`ω_{Q,ℓ}(ε;u) = sup{ ℓ(Q(u′),Q(u)) : ‖u′−u‖_A ≤ ε }`.

For fixed `ℓ`, a transfer guarantee is any `g` that dominates every displayed
discrepancy on each approximation ball. The proof of Proposition 2.1 uses only
the least-upper-bound property and therefore carries over verbatim: `ω_{Q,ℓ}`
is the pointwise-minimal transfer guarantee. Admissible-set restriction is also
geometry-agnostic.

The choice of `ℓ` is substantive. Absolute, relative, and stabilized-relative
geometries encode different decision semantics. Relative discrepancy is
ill-conditioned near a zero reference and must not be treated as universal.
Composition, naturality, and differential claims require assumptions appropriate
to the chosen geometry. In particular, the condition-number results below are
statements about the relative specialization.

A reference-free companion is the decision diameter

`diam_{Q,ℓ}(D)=sup{ℓ(Q(u′),Q(u″)):u′,u″∈D}`.

It measures variation across the admissible family rather than deviation from a
declared center.

### 2.1 Relative-error specialization used in the reference application

Let `X` be a normed space, `‖·‖_A` a seminorm on `X` (the **approximation
seminorm**), `Q : X → ℝ` a **quantity of interest**, and `u ∈ X` a target. This
section specializes to `ℓ_rel(q,q₀)=|q-q₀|/|q₀|`, so **the standing assumption is `Q(u) ≠ 0`**.

`‖·‖_A` is a seminorm and its kernel is central, not technical: if `Q` is
non-constant on `{u′ : ‖u′ − u‖_A = 0}` then `ω(ε) ≥ c > 0` for every ε, so `Q`
is uncertifiable by Corollary 3.2. Continuity in Theorem 3.1 is continuity on the
pseudometric space `(X, d_A)`, equivalently on `X/∼_A`: `Q` is certifiable only
if it is constant, or approximately controlled, on observational equivalence
classes.

**Definition 2.1 (transfer guarantee).** `g : (0,∞) → [0,∞]` is a **transfer guarantee** for
`(Q, ‖·‖_A)` at `u` if `‖u′ − u‖_A ≤ ε ⟹ |Q(u′) − Q(u)|/|Q(u)| ≤ g(ε)` for all
`u′`, `ε`. These form a set `𝒢` ordered pointwise; smaller is stronger. `𝒢` is
never empty — `g ≡ ∞` belongs to it — so asking for a least element is not
vacuous.

A transfer guarantee is exactly what licenses "validated to aggregate accuracy ε,
therefore the conclusion is accurate to `g(ε)`".

**Definition 2.2 (fidelity modulus).** ω(ε) = sup{ |Q(u′) − Q(u)|/|Q(u)| : ‖u′ − u‖_A ≤ ε }.

> **Intuition — an exchange rate between errors.**  
> Think of \(\omega(\varepsilon)\) as the worst-case exchange rate between
> validation currency and conclusion currency. If validation permits
> \(\varepsilon\) units of error, \(\omega(\varepsilon)\) is the greatest
> downstream discrepancy still compatible with that contract. Proposition 2.1
> shows that no uniformly smaller guarantee can be valid.


**Remark.** In the general formulation, ω depends on four declared objects — the
approximation relation, the quantity of interest, the decision discrepancy, and
the reference point `u` — and is independent of any certification procedure.
This relative specialization fixes the discrepancy and therefore displays only
the other three. The restriction ω_D of §4 adds a fourth, the certified
set `D`.

**Definition 2.3 (certifiability).** `Q` is **(ε₀, τ)-certifiable** by `‖·‖_A` at `u` if ω(ε₀) ≤ τ.

**Proposition 2.1 (least guarantee and finite-locus slack).** The complete
lattice of transfer guarantees is the principal upper set

  **`𝒢 = {g : g ≥ ω}`.**

Hence `ω ∈ 𝒢` and is the unique least transfer guarantee. On the finite locus

  `F_g = {ε : ω(ε) < ∞ and g(ε) < ∞}`,

each `g ∈ 𝒢` has the unique additive decomposition
`g(ε)=ω(ε)+s_g(ε)` with `s_g(ε)=g(ε)-ω(ε) ≥ 0`.

*Proof.* For `g ∈ 𝒢`, `g(ε)` is an upper bound for the set defining `ω(ε)`.
Because a supremum is the least upper bound, `ω(ε)≤g(ε)` for every ε. Conversely,
every `g≥ω` dominates every admissible decision discrepancy and is therefore a
transfer guarantee. Thus `𝒢={g:g≥ω}`, and antisymmetry gives uniqueness of its
least element. Where both values are finite, ordinary subtraction defines the
unique non-negative slack. No unique additive slack is claimed at points with
extended-real value `∞`, where `∞-∞` is undefined. ∎

**Corollary 2.1 (minimal transfer assignment).** Let `T′` assign to each
quantity of interest a transfer guarantee for `(Q, ‖·‖_A)` at `u`, and require
that `T′(Q)` be the least element of `𝒢(Q)` for every `Q`. Then `T′ = T`.

*Proof.* Least elements of a partial order are unique, and Proposition 2.1
identifies that of `𝒢(Q)` as `ω_Q`. ∎

So `T` is canonical, not merely convenient — but the uniqueness is *relative to
minimality*. Drop that requirement and other operators into `𝒢` exist: `Q ↦ 2ω_Q`
assigns a valid transfer guarantee to every `Q`, and satisfies several items of
Proposition 2.2. What singles out `T` is that its value is the least valid guarantee at
each `Q`, which is exactly what Proposition 2.1 supplies. Whether the structural
axioms *alone* — without reference to `𝒢` — pin `T` down is the open question of
§9.

**Interpretation.** Proposition 2.1 is an organizing fact: `ω` is the
smallest valid guarantee, and any finite alternative guarantee differs from it
by non-negative slack. The paper does not claim this supremum argument as its
main novelty; it fixes the reference object used in the finite-tolerance and
coverage results below.

### Basic properties used later

The **transfer operator** is the canonical assignment sending each quantity of
interest to the least transfer guarantee compatible with the approximation
geometry:

  `T : Q ↦ ω_Q`,  at fixed `u` and `‖·‖_A`.

Write also `R : D ↦ ω_D` for the map sending an admissible set to the modulus
restricted to it. The items below are properties of these two operators,
collected into one theorem because they describe one object, not six.

**Proposition 2.2 (basic properties).** The following elementary properties are
collected for later use. Items (7) and (8) are the substantive composition and
product estimates; the remaining items follow directly from the definition as a
supremum over nested or restricted sets.

*(a) Dependence on the tolerance.*
1. ω is non-decreasing in ε; hence `ω(0⁺) := lim_{ε↓0} ω(ε) = inf_ε ω(ε)` exists
   in `[0,∞]` and ω has at most countably many discontinuities.
2. `ω(0⁺) = 0` iff `Q` is `‖·‖_A`-continuous at `u` (Theorem 3.1).

*(b) Dependence on the data.*
3. `T(αQ) = T(Q)` for `α ≠ 0`; `T` is not translation invariant, `ω[Q + β]`
   rescaling by `|Q(u)|/|Q(u) + β|`.
4. If `‖·‖_A ≤ K‖·‖_{A′}` then `ω_{A′}(ε) ≤ ω_A(Kε)`: a stronger validation
   seminorm can only shrink the modulus.

5. `ω|_{(0,ε₀]}` depends only on `Q|_{B_{ε₀}}`.
6. If `Q′(u) = Q(u)` and `sup_{B_ε}|Q′ − Q| ≤ η` then `|ω′(ε) − ω(ε)| ≤ η/|Q(u)|`:
   `T` is 1-Lipschitz in `Q` under the sup-norm on the ball.

*(c) Operations on problems.*
7. *Composition.* `ω̂[id](ε) = ε`, and with `V_ε = Q₁({u′ : ‖u′ − u‖_A ≤ ε})`,

     `ω̂[Q₂ ∘ Q₁](ε) ≤ sup_{v ∈ V_ε} ω̂[Q₂, ‖·‖_B, v]( ω̂[Q₁](ε) )`,

   which reduces to `ω̂₂ ∘ ω̂₁` when `ω̂[Q₂]` is uniform over `V_ε`. So `T` is lax
   functorial, the failure of strictness being the composition slack.
8. `1 + ω_{Q₁⊗Q₂} ≤ (1 + ω_{Q₁})(1 + ω_{Q₂})`; equivalently `log(1 + ω)` is
   subadditive under tensor products.

9. `R` is monotone: `D ⊆ D′ ⟹ ω_D ≤ ω_{D′}`, with `ω_X = ω`.
10. **`R` preserves joins exactly and meets only laxly:**
    `ω_{D ∪ D′} = ω_D ∨ ω_{D′}` but in general `ω_{D ∩ D′} ≤ ω_D ∧ ω_{D′}`,
    with strict inequality possible.

*Proof.* (1) balls nest. (2) Theorem 3.1. (3) the ratio is unchanged under scaling
of `Q`; under translation the numerator is unchanged and the denominator becomes
`|Q(u) + β|`. (4) the `‖·‖_{A′}`-ball of radius ε sits inside the `‖·‖_A`-ball of
radius `Kε`. (5) `B_ε ⊆ B_{ε₀}`. (6) `|Q′(u′) − Q′(u)| ≤ |Q(u′) − Q(u)| + η` and
symmetrically. (7) `‖u′ − u‖_A ≤ ε` gives `‖Q₁u′ − Q₁u‖_B ≤ ω̂[Q₁](ε)`, so `Q₁u′` lies in the
`ω̂[Q₁](ε)`-ball about `Q₁u ∈ V_ε`; apply the definition of `ω̂[Q₂]` at that base
point and take the supremum over `V_ε`. (8) `|Q₁′Q₂′ − Q₁Q₂| ≤ |Q₁′ − Q₁||Q₂′| +
|Q₁||Q₂′ − Q₂|`, divided by `|Q₁Q₂|`. (9) a supremum over a subset is no larger.
(10) `sup_{D ∪ D′} = max(sup_D, sup_{D′})` exactly, whereas `sup_{D ∩ D′}` is
bounded by each and may be strictly below both. ∎

The proposition is included to make later dependencies explicit. In particular,
restriction of an admissible set cannot increase the modulus, while composition
produces a generally non-tight upper bound whose slack is analyzed next.

**Corollary 2.2 (why stage-wise certification helps).** Item 10 is the structural
reason gates compose usefully. Adding a test intersects admissible sets, and
intersection can drive the modulus strictly below the minimum of the two
component moduli — a stricter guarantee than either test provides alone. Union,
by contrast, gives nothing: certifying against a wider family is exactly as weak
as its weakest part. This is a geometric statement about restriction of the
admissible set, not a probabilistic statement about independence among tests.
Two gates derived from the same information basis can tighten the restricted
modulus without constituting independent confirmations.

**Proposition 2.3 (reparameterisation of `Q`).** Let `h : ℝ → ℝ`
with `h(Q(u)) ≠ 0`, and write `T(Q) = ω_Q`.

1. *Affine `h`, exactly.* If `h(y) = ay + b` with `a ≠ 0` then
   `T(h ∘ Q) = c · T(Q)` with `c = |a| |Q(u)| / |aQ(u) + b|`. Proposition 2.2(3) is the
   case `b = 0`, where `c = 1`.
2. *Lipschitz `h`, laxly.* If `h` is `L`-Lipschitz near `Q(u)` then
   `T(h ∘ Q) ≤ (L |Q(u)| / |h(Q(u))|) · T(Q)`.
3. *Differentiable `h`, asymptotically.* If `h` is `C¹` near `Q(u)` with
   `h′(Q(u)) ≠ 0` and `Q` is `‖·‖_A`-continuous at `u`, then
   `ω_{h∘Q}(ε) / ω_Q(ε) → |h′(Q(u))| |Q(u)| / |h(Q(u))|` as `ε ↓ 0`.

*Proof.* (1) `|h(Q(u′)) − h(Q(u))| = |a| |Q(u′) − Q(u)|`, and the denominator is
the constant `|aQ(u) + b|`; the ratio is a fixed multiple of `e_Q` pointwise, so
the suprema over the same ball scale by that multiple. (2) replace `|a|` by the
Lipschitz bound. (3) `h(Q(u′)) − h(Q(u)) = h′(Q(u))(Q(u′) − Q(u)) + o(|Q(u′) −
Q(u)|)`, uniformly over `B_ε` because `Q(B_ε) → Q(u)` by continuity; divide and
take suprema. ∎

So `T` is natural in the quantity of interest: exactly under affine
reparameterisation, laxly under Lipschitz, and asymptotically under `C¹`. This is
the `T(F(Q)) = Φ(T(Q))` law for post-composition, with `Φ` multiplication by the
relative derivative `|h′(Q(u))| |Q(u)| / |h(Q(u))|`. The corresponding
pre-composition law — `Q ∘ G` for a map `G` of the space — is Proposition 2.2(7), and is
only lax; exactness there would require the worst directions of `G` and `Q` to
align, which is the composition slack of §5.1.

**Proposition 2.4 (a-priori bound on composition slack).** Say `Q₁` is
**`c`-open at `u`** (`0 < c ≤ 1`) if for all small ε the image `Q₁(B_ε)` contains
the `‖·‖_B`-ball of radius `c · ω̂[Q₁](ε)` about `Q₁u`. Then

  `ω̂[Q₂]( c · ω̂[Q₁](ε) ) ≤ ω̂[Q₂ ∘ Q₁](ε) ≤ ω̂[Q₂]( ω̂[Q₁](ε) )`,

so the slack in Proposition 2.2(7) is at most
`ω̂[Q₂](ω̂₁) / ω̂[Q₂](c·ω̂₁)`. If `ω̂[Q₂]` is exactly positively homogeneous of
degree one, this ratio is **at most `1/c`**, and the bound can be attained. If
instead `ω̂[Q₂](η)=κη+o(η)` with `κ>0`, the displayed ratio tends to `1/c` as
`η→0`; this is an asymptotic statement, not a finite-scale homogeneous bound.

*Proof.* The upper bound is Proposition 2.2(7). For the lower, `c`-openness puts the
`c·ω̂₁(ε)`-ball inside the image, and `ω̂[Q₂∘Q₁](ε)` is the supremum of `Q₂`'s
displacement over that image, hence at least its supremum over the contained
ball, which is `ω̂[Q₂](c·ω̂₁(ε))`. Exact degree-one homogeneity gives the ratio `1/c`. Attainment:
`Q₁ = diag(1, c)` on Euclidean `ℝ²`, with the unit input ball and `Q₂` the second coordinate, gives `ω̂₁ = 1`,
`ω̂[Q₂∘Q₁] = c`, upper bound `1`. ∎

**Alignment, characterised.** The slack is governed by how much the image
`Q₁(B_ε)` fails to fill the ball containing it, *in the directions `Q₂` is most
sensitive to*. Tightness is `c = 1` along the worst direction of `Q₂` — the
alignment condition, now a quantity rather than a description. The measured
3.6–4.5× slack corresponds to an effective `c ≈ 0.22–0.28` for that pair.

**Proposition 2.5 (restriction criterion).** Let `𝒟` be the poset of
admissible sets ordered by inclusion and `𝒢↑` the poset of *non-decreasing*
transfer guarantees ordered pointwise. Define

  `R : 𝒟 → 𝒢↑`,  `R(D) = ω_D`,
  `Σ : 𝒢↑ → 𝒟`,  `Σ(g) = { u′ ∈ X : e_Q(u′, u) ≤ g(‖u′ − u‖_A) }`.

Both are monotone, and for all `D ∈ 𝒟`, `g ∈ 𝒢↑`

  **`ω_D ≤ g  ⟺  D ⊆ Σ(g)`.**

Equivalently, this relation induces a closure operator on admissible sets. The
order-theoretic formulation is recorded for completeness; operationally, the
criterion says exactly when a gate-admissible family supports a proposed
transfer guarantee.

*Proof.* (⇒) Let `u′ ∈ D` and put `ε = ‖u′ − u‖_A`. Then `u′ ∈ D_ε`, so
`e_Q(u′, u) ≤ ω_D(ε) ≤ g(ε)`, i.e. `u′ ∈ Σ(g)`. (⇐) Fix ε and `u′ ∈ D_ε`. Then
`u′ ∈ Σ(g)` gives `e_Q(u′, u) ≤ g(‖u′ − u‖_A) ≤ g(ε)`, using that `g` is
non-decreasing; take the supremum over `D_ε`. Monotonicity of `R` is Proposition 2.2(9);
monotonicity of `Σ` is immediate. The composite `Σ ∘ R` is inflationary
(`D ⊆ Σ(ω_D)` by (⇒) with `g = ω_D`), monotone, and idempotent, hence a closure
operator. ∎

**Corollary 2.3 (closure-equivalent certification procedures).** `Σ(ω_D)` is the
largest admissible set with modulus no greater than `ω_D`, so replacing `D` by
its closure changes nothing certifiable. Two procedures with the same closure are
indistinguishable by any transfer guarantee; the closure is the invariant content
of a gate.

**Corollary 2.4 (certification in restriction form).** `|𝓑(τ)| = 0` says `ω_D ≤ τ`
pointwise on the relevant range, which by Proposition 2.5 is `D ⊆ Σ(τ)` — the accepted
set lies inside the τ-tolerant set. Certification is thus an operation *on the
family of transfer guarantees* rather than a separate apparatus: gating moves `D`
down `𝒟`, and `R` carries that motion to a smaller element of `𝒢↑`.

---

## 2.2 Contribution boundary

The worst-case sensitivity `ω(ε)` and the distinction between state error and
output error are established ideas. Proposition 2.1 and Proposition 2.5 organize
that familiar object but are not presented as deep standalone discoveries.

The paper's substantive claims begin at finite tolerance:

- local conditioning can be finite yet unusable at the accepted tolerance;
- a certification gate changes the admissible set and therefore the relevant
  restricted modulus;
- a searched perturbation family yields the full restricted modulus only under
  a coverage argument, and otherwise yields a one-sided floor;
- under effective directional regularity, the restricted modulus can be
  enclosed with a certified finite computation;
- Silent Risk summarizes the measure of gate-admitted alternatives that violate
  a declared decision tolerance.

The remainder of the paper is organized around those claims. The order language
is retained only where it shortens proofs or clarifies how gates and pipelines
compose.

---

## 3. Characterisation

**Theorem 3.1 (continuity criterion).** ω(ε) → 0 as ε → 0 iff `Q` is `‖·‖_A`-continuous at `u`.

*Proof.* (⇒) Given δ > 0, pick ε with ω(ε) ≤ δ. If `‖u′ − u‖_A ≤ ε` then the
ratio lies in `S(ε)`, so `|Q(u′) − Q(u)| ≤ ω(ε)|Q(u)| ≤ δ|Q(u)|`: continuity at
`u`. (⇐) Given δ > 0, continuity supplies ε with `|Q(u′) − Q(u)| ≤ δ|Q(u)|` for
all `u′` in the ε-ball, so δ is an upper bound for `S(ε)` and ω(ε) ≤ δ by
leastness of the supremum. Since ω is non-decreasing, ω(ε′) ≤ δ for ε′ ≤ ε. ∎

**Corollary 3.1 (Lipschitz sufficient condition).** If `Q` is `L`-Lipschitz in `‖·‖_A` near `u` then
ω(ε) ≤ Lε/|Q(u)|, so `Q` is (ε₀, τ)-certifiable whenever ε₀ ≤ τ|Q(u)|/L.

*Proof.* Every ratio in `S(ε)` is at most `L‖u′ − u‖_A/|Q(u)| ≤ Lε/|Q(u)|`, which
is therefore an upper bound; apply leastness. ∎

Lipschitz continuity is sufficient but not necessary: Hölder continuity of order
α gives ω(ε) ≤ Cε^α/|Q(u)| → 0, and by Theorem 3.1 bare continuity suffices for the
asymptotic statement.

**Corollary 3.2 (discontinuity obstruction).** If `Q` is `‖·‖_A`-discontinuous at `u` then ω(ε) ≥ c > 0 for all
ε, and `Q` is (ε₀, τ)-certifiable for no τ < c at any ε₀.

*Proof.* Contrapositive of Theorem 3.1: if ω(ε) → 0 fails then, ω being
non-decreasing, `inf_ε ω(ε) = c > 0`. Certifiability at τ < c would force
ω(ε₀) ≤ τ < c. ∎

Theorem 3.1 is a restatement of continuity and is not itself a contribution; its
role is to identify the governing object. It also settles only the limit, and the
practical question is finite-ε — a pair can satisfy Theorem 3.1 and be useless at
every ε a practitioner would accept, which is what §§4–5 address.

### 3.1 Differentiability of the fidelity modulus

Fréchet differentiability of `Q` implies a first-order expansion of ω at zero;
the converse need not hold. Its resulting right derivative is a quantity
numerical analysis already names. **The classical relative condition
number is the differential of the fidelity modulus at zero** — not an object the
theory extends, but the infinitesimal shadow of the more fundamental one.

**Theorem 3.2 (differentiability at zero).** If `Q` is Fréchet differentiable at `u` with
respect to `‖·‖_A`, then

  `ω(ε) = κ_rel · ε + o(ε)`,  where `κ_rel = ‖DQ(u)‖_* / |Q(u)|`,

so ω is right-differentiable at `0` with `ω′(0⁺) = κ_rel`.

**Corollary 3.3 (condition numbers).** The classical relative condition number of
`Q` at `u` is `κ = ‖u‖_A · ω′(0⁺)`. It is therefore a derived quantity: the
first-order coefficient of ω, carrying no information ω does not.

*Proof.* Differentiability gives `Q(u′) − Q(u) = DQ(u)[u′ − u] + o(‖u′ − u‖_A)`.
Taking the supremum over the ε-ball, `sup |DQ(u)[v]| = ε‖DQ(u)‖_*` by
homogeneity, and the remainder is `o(ε)`. Divide by `|Q(u)|`. The stated identity
with `κ` is the definition of the relative condition number, in which the
perturbation is measured relative to `‖u‖_A`. ∎

**Corollary 3.4 (finite-tolerance regimes).** Let `ε₀` be the
operating tolerance and `τ` the required accuracy.

- If `κ_rel < ∞` and the linearisation is still valid at `ε₀`, then
  `κ_rel ε₀ ≲ τ` certifies — row 1.
- If `κ_rel < ∞` but `ε₀ ≫ 1/κ_rel`, the condition number exists and is not
  predictive at the tolerance actually used — row 2.
- Row 3 implies that no finite condition-number linearisation can certify the
  quantity. **The converse is false.**

The converse fails because an infinite or undefined `κ_rel` does not imply
discontinuity. Take `Q(x) = 1 + √|x|` on `ℝ` at `u = 0`, where `Q(u) = 1`: then
`ω(ε) = √ε → 0`, so `Q` is continuous and asymptotically certifiable by
Theorem 3.1, while `ω(ε)/ε = ε^{-1/2} → ∞`, so no finite first-order condition
number exists. Continuous-but-non-Lipschitz transfer is a distinct regime.

So the classification by condition number is **four**-fold, not three-fold:

| regime | `ω(ε) → 0`? | `κ_rel` | certifiable? |
|---|---|---|---|
| well-conditioned, in the linear regime | yes, `∼ κ_rel ε` | finite | by `κ_rel` alone |
| finite `κ_rel`, outside the linear regime | yes | finite | only by computing ω |
| continuous but non-Lipschitz | yes, sublinearly | ∞ or undefined | only by computing ω |
| discontinuous / unidentifiable | no | ∞ | never |

The third row is the sharpest case for the theory: a classical condition
number cannot even classify it correctly — it reports `∞`, the same answer it
gives for genuine discontinuity — while ω distinguishes the two and certifies the
Hölder case at a computable tolerance.

**In the reference application.** Directional condition numbers along the
perturbation family:

| ρ | 0.9 | 0.6 | 0.3 | 0.1 | 0.01 |
|---|---|---|---|---|---|
| `κ_rel` | 4.3e1 | 3.9e3 | 2.0e5 | 2.0e6 | 4.4e6 |
| `1/κ_rel` | 2.3e-2 | 2.6e-4 | 5.1e-6 | 5.0e-7 | 2.3e-7 |
| `L(5e-3)` | 3.4e-1 | 2.0e-2 | 7.2e-4 | 8.9e-5 | 4.5e-5 |

`sup_ρ κ_rel ≈ 4.4 × 10⁶`, so the linear regime extends only to `ε ≲ 2.3 × 10⁻⁷`
— and the acceptance floor of `5 × 10⁻³` sits four orders of magnitude beyond it.
The pair is row 2 by Corollary 3.4, and the condition number, though finite, is
inapplicable by a factor of `2.2 × 10⁴` at the tolerance in use. Computed by
`transfermod.modulus.condition_number` and `linearity_ratio`.

The quantity a practitioner needs from second order is not the coefficient but
the scale at which first-order reasoning stops applying. Define the **crossover**

  `ε_× = κ_rel / |κ₂|`,

where `κ₂` is the second-order coefficient below. For `ε ≪ ε_×` the linearisation
of Theorem 3.2 governs; for `ε ≫ ε_×` it does not, and `ω` must be computed. The
expansion exists to justify and compute `ε_×`.

*Why this is not stated as the primary asymptotic result, with Theorem 3.2 as its
corollary:* Theorem 3.3 needs twice-differentiability and a unique non-degenerate
first-order maximiser, strictly more than Theorem 3.2, which needs only Fréchet
differentiability. Deriving Theorem 3.2 from it would import hypotheses it does not
require.

**Theorem 3.3 (second-order expansion, full modulus).** Suppose `Q` is twice
Fréchet differentiable at `u` and the first-order maximiser `v*` of `|DQ(u)[v]|`
over the unit sphere is unique up to sign and non-degenerate. Write
`κ_rel = ‖DQ(u)‖_*/|Q(u)|` and `β = D²Q(u)[v*, v*] / (2|Q(u)|)`. Then

  `ω(ε) = κ_rel ε + |β| ε² + o(ε²)`.

*Proof qualification.* The displayed expansion is immediate along the two rays
`±v*`. Equality with the full supremum additionally requires a smooth constrained
maximum: the unit sphere must be smooth near `±v*`, the first-order maximum must
be isolated with a negative-definite tangential Hessian, and the Taylor remainder
must be uniform on the unit sphere. Under those conditions the maximizing
direction varies smoothly by `O(ε)` and the envelope expansion gives the stated
coefficient. In a general normed space, or without this isolation condition, the
ray calculation remains a lower bound and the theorem should not be invoked. ∎

**The second-order coefficient of the full modulus is therefore non-negative** in
the scalar case: symmetry of the ball forecloses cancellation. A *signed*
quadratic coefficient is a property of a one-sided family, not of ω.

**Proposition 3.1 (second-order expansion, one-sided ray).** *(Not a consequence of
Theorems A–B: those describe how ω behaves under operations and restriction, and
carry no asymptotic content. This needs twice-differentiability along the ray.)* For a fixed
parameterised ray `u_w`, `w ≥ 0`, with `Q` twice differentiable along it,

  `ω_θ(ε) = κ_θ ε + κ_{2,θ} ε² + o(ε²)`,

and `κ_{2,θ}` may have either sign, since only `w ≥ 0` is admissible. The
crossover scale is `ε_×^θ = κ_θ / |κ_{2,θ}|`, and
`L_θ(ε) = 1 + (κ_{2,θ}/κ_θ)ε + o(ε)`.

**Corollary 3.5 (where first-order conditioning stops working).** Row 2 of the
trichotomy is, quantitatively, `κ_rel < ∞` and `ε₀ ≫ ε_×`; row 1 is `ε₀ ≪ ε_×`
with `κ_rel ε₀ ≤ τ`.

**Measured — for the ray-restricted modulus `ω_θ`, not for ω.** Fitting
`ω_θ(ε) = aε + bε²` on each ray, and locating the half-linearity point
`L_θ(ε) = ½` independently by bisection:

| ρ | `κ_θ = a` | `κ_{2,θ} = b` | `ε_×^θ = a/|b|` | `L_θ = ½` at | ratio |
|---|---|---|---|---|---|
| 0.9 | 4.33e1 | −1.74e4 | 2.49e-3 | 2.54e-3 | 1.02 |
| 0.6 | 3.85e3 | −5.28e7 | 7.30e-5 | 8.40e-5 | 1.15 |
| 0.3 | 1.95e5 | −1.18e11 | 1.66e-6 | 2.07e-6 | 1.25 |

`ε_×^θ` predicts where the linearisation half-fails to within 25%, across three
decades of conditioning. **`b < 0` on every ray**, which is a statement about the
one-sided family: the relative error saturates near 1 while the linear
extrapolation does not, so along these rays first-order conditioning
*over*-predicts — `κ_θ ε₀ = 2.2 × 10⁴` against a true `ω_θ(ε₀) = 0.95`. By
Theorem 3.3 the corresponding full-modulus coefficient is `+|β|`, so no such
conservatism can be inferred for ω itself.

Computed by `transfermod.modulus.second_order`, which fits the *ray* expansion.


## 4. Restricted moduli

Let `D ⊆ X` be the **certified set**, defined by the procedure alone and not by
ε, and put `D_ε = D ∩ B_ε(u)`. With `e_Q(u′,u) = |Q(u′) − Q(u)|/|Q(u)|`,

  `ω_D(ε) = sup{ e_Q(u′, u) : u′ ∈ D_ε } ≤ ω(ε)`.

**Restriction is monotone in the admissible set.** If `D ⊆ D′` then
`ω_D(ε) ≤ ω_{D′}(ε)` for every ε, since the supremum is over a subset; taking
`D′ = X` recovers `ω_D ≤ ω`. So `D ↦ ω_D` is an order-preserving map from the
poset of admissible sets to the poset of transfer guarantees, and certification
acts on ω by restriction along the inclusion `D ↪ X` rather than by replacing it.

Over a parameterised family, `ω_{D,Θ}(ε) = sup{ e_Q(u_θ, u) : u_θ ∈ D_ε }`. Every
computed number below is a `ω_{D,Θ}` — exact for the family, a lower bound for
`ω_D` — and no family is claimed extremal over `D_ε`.

Along a perturbation family `u_w`, `w ≥ 0`, with `u_0 = u`, write

  `A = { w : u_w ∈ D }`  (accepted),  `C(τ) = { w : |Q(u_w) − Q(u)| > τ|Q(u)| }`  (τ-corrupt),

and define `w_det = sup A`, `w_corrupt(τ) = inf C(τ)` (with the conventions
`sup ∅ = 0`, `inf ∅ = ∞`). Defining the thresholds as a supremum and an infimum
rather than as least elements avoids any question of whether the sets are open or
closed at their endpoints. The **log-amplitude Silent Risk** is
`|𝓑(τ)| = log₁₀(w_det/w_corrupt(τ))⁺`; the **exposure** is
`|𝓘| = log₁₀(w_agg/w_det)⁺`, where `w_agg = sup{ w : ‖u_w − u‖_A ≤ ε }`.

### 4.1 Measure-qualified Silent Risk

For a general measure `ν` on alternatives and tolerance `τ`, let
`B_τ={x:x is admitted and ℓ(Q(x),Q₀)>τ}`. Define

`SR_{ν,τ}=ν(B_τ)`.

This is the umbrella definition of Silent Risk: the measure of admitted
decision-corrupting alternatives. The one-parameter spectral construction uses
logarithmic amplitude measure and reports a width in decades. A Bayesian
construction may use posterior measure and report an exceedance probability.
The measure must always be named.

## 5. Certification

**Theorem 5.1 (family-restricted certification).** For an arbitrary family
`{u_θ}_{θ∈Θ}`, with `A = {θ : u_θ ∈ D}` and
`C(τ) = {θ : |Q(u_θ) − Q(u)| > τ|Q(u)|}`,

  `ω_{D,Θ} ≤ τ`  ⟺  `A ∩ C(τ) = ∅`.

*Proof.* `ω_{D,Θ}` is the supremum of the relative error over the accepted
parameters `A`. It exceeds τ iff some `θ∈A` has relative error above τ, which is
equivalent to `A∩C(τ)≠∅`. Negate. ∎

**Coverage corollary.** If condition (C1) proves that the family covers `D`, then
`ω_{D,Θ}=ω_D`, and the same equivalence holds for the full restricted modulus.

**Theorem 5.2 (scalar Silent Risk criterion).** Suppose `w ↦ |Q(u_w) − Q(u)|` is continuous and non-decreasing,
and the detection statistic is non-decreasing. Then `|𝓑(τ)| = 0 ⟺ ω_{D,Θ} ≤ τ`. Under (C1), this is also equivalent to `ω_D≤τ`.

*Proof.* Monotonicity of the detection statistic makes `A` a **down-set** in
`[0,∞)`: if `w′ < w` and `u_w` is accepted, so is `u_{w′}`. Monotonicity and
continuity of the error make `C(τ) = {w : error(w) > τ|Q(u)|}` a **relatively
open up-set**, so `C(τ) = (w_corrupt(τ), ∞)` and `w_corrupt(τ) ∉ C(τ)`. Hence
`A ∩ C(τ) = ∅` iff `A ⊆ [0, w_corrupt(τ)]` iff `w_det = sup A ≤ w_corrupt(τ)`,
which is `log₁₀(w_det/w_corrupt(τ)) ≤ 0`, i.e. `|𝓑(τ)| = 0`. Apply Theorem 5.1. ∎

**The continuity hypothesis is not removable.** Without it the boundary case
defeats the equivalence: if `A = [0,1]` and `C(τ) = [1,∞)` then
`w_det = w_corrupt(τ) = 1`, so `|𝓑(τ)| = 0`, yet `u_1` is accepted and τ-corrupt,
so `ω_D > τ`. Monotonicity alone gives only `A ∩ C(τ) = ∅ ⟹ w_det ≤ w_corrupt(τ)`
and the strict converse `w_det < w_corrupt(τ) ⟹ A ∩ C(τ) = ∅`. Every family
computed in this note is analytic in `w`, so the hypothesis costs nothing there;
it is stated because the scalar summary, not the equivalence of Theorem 5.1, is
what needs it.

Theorem 5.1 shows the equivalence is set-theoretic; monotonicity enters only to
make `A` and `C(τ)` intervals, so that disjointness collapses to a comparison of
two numbers and the discrepancy to the single quantity `|𝓑(τ)|`. Monotonicity is
therefore a hypothesis on the *computable scalar summary*, not on the result.
**This is the note's central claim: a certification procedure computes a
family-restricted modulus; it computes the full restricted modulus only when
coverage is proved.** Without
monotonicity only `ω_{D,Θ} ≤ τ ⟹ |𝓑(τ)| = 0` survives, `A` need not be an interval.

**Corollary 5.1 (limits of gating).** If `|Q(u_w) − Q(u)| ≥ c|Q(u)|` for all
`w > 0`, then any `D` with `w_det > 0` has `ω_D ≥ c`.

*Proof.* Let `c′ < c` and take the constant guarantee `g ≡ c′`. By hypothesis no
`u_w` with `w > 0` lies in `Σ(c′)`. Since `w_det > 0`, `D` contains such a `u_w`,
so `D ⊄ Σ(c′)`, and Proposition 2.5 gives `ω_D > c′`. Let `c′ ↑ c`. ∎

Gating therefore helps in the continuous but ill-conditioned regime, not the
discontinuous one.

**Trichotomy.** Lipschitz with modest `L` → refinement certifies, gating
unnecessary. Continuous with poor modulus → refinement fails at usable ε, gating
certifies. Discontinuous → neither. The contribution lies in the middle row.

**Composition, in brief.** For a map `F : (X, ‖·‖_A) → (X₂, ‖·‖_B)` write the
**absolute modulus** `ω̂[F](ε) = sup{ ‖F(u′) − F(u)‖_B : ‖u′ − u‖_A ≤ ε }`, so
that the relative modulus of Definition 2.2 is `ω = ω̂/|Q(u)|`. Proposition 2.2(7) bounds
`ω̂[Q₂ ∘ Q₁]` by `ω̂₂ ∘ ω̂₁`, and Proposition 2.4 bounds the slack of that bound by
`1/c` for a `c`-open upstream map. Consequences: pipelines compose
multiplicatively, so a single flat-modulus stage caps the chain; stage-wise
gating bounds what end-to-end validation does not, though the measured 3.6–4.5×
slack means it bounds rather than budgets; and composition preserves validity but
not minimality — the slack is well defined precisely because Proposition 2.1 pins
the target.

## 6. Computation

ω is a supremum over an infinite-dimensional ball. Two reductions: restrict to a
perturbation family (a lower bound, exact for the family); and, under
monotonicity, locate each threshold by bisection in `O(log 1/δ)` evaluations.
`R = (|𝓘|, |𝓑|)` costs three bisections plus one — measured, 180 evaluations and
20 ms at the reference point.

### 6.1 Sufficient conditions for computing a restricted modulus

`ω_D(ε)` is a supremum over an infinite-dimensional set, and §6 so far has only
produced `ω_{D,Θ}`, a lower bound. This subsection gives conditions under which
the gap closes and the computation is certified.

Let the admissible perturbations be parameterised as `u_{d,w}` with direction
`d ∈ S` and amplitude `w ≥ 0`, `u_{d,0} = u`.

**Theorem 6.1 (attainment and conditional effective computation).** Assume

- **(C1) Coverage.** Every `u′ ∈ D_ε` equals `u_{d,w}` for some
  `(d,w)∈S×[0,∞)`.
- **(C2) Ray regularity.** For each fixed `d`, the error, aggregate, and
  detection statistics are non-decreasing in `w`, and the error is continuous.
- **(C3) Directional regularity.** `S` is compact and
  `d↦ω_D^{(d)}(ε)` is continuous.

Then the full restricted modulus is attained:

  `ω_D(ε)=max_{d∈S}ω_D^{(d)}(ε)`.

Compactness and continuity prove existence, not by themselves an effective
finite algorithm. For certified computation to a requested tolerance, additionally
assume:

- **(C4) Effective directional control.** A finite `δ`-net of `S` can be
  constructed and a known modulus of continuity `μ` satisfies
  `|ω_D^{(d)}-ω_D^{(g)}|≤μ(dist(d,g))`.

Then for any `δ`-net `G⊂S`,

  `max_{g∈G}ω_D^{(g)}(ε) ≤ ω_D(ε)
   ≤ max_{g∈G}ω_D^{(g)}(ε)+μ(δ)`.

In the Lipschitz special case `μ(r)=Λr`, this recovers the implemented
`Λδ` bound. If ray thresholds are computable by certified bisection to error
`δ′`, the total cost is `|G|·O(log(1/δ′))` ray evaluations.

*Proof.* (C1) permits decomposition of the supremum by directions. (C2) reduces
each ray value to a threshold search. (C3) gives attainment of the outer
supremum by the extreme-value theorem. Under (C4), choose for each `d∈S` a
`g∈G` with `dist(d,g)≤δ`; then
`ω_D^{(d)}≤ω_D^{(g)}+μ(δ)≤max_G+μ(δ)`. The grid lower bound is immediate.
The complexity statement follows from the certified ray solver. ∎

**Corollary 6.1 (coverage is the remaining structural obstruction).** Under (C2)–(C4),
everything except (C1) is a finite computation with a certified error bound.
Coverage is the remaining *structural* obstruction once ray regularity,
effective compactness, and a certified continuity modulus have been supplied;
none of those computational hypotheses is free. What
distinguishes coverage is that no computation can verify it: it is a claim that
the chosen family exhausts what the certification procedure will accept. Absent a
proof of (C1), every reported value is a lower bound on `ω_D` — which is why the
quantities in this document are written `ω_{D,Θ}`.

`Λ` is likewise not free: estimating it from the largest slope between adjacent
grid nodes is plausible but not certified, and a certified upper bound requires
an a priori smoothness bound on the family. `transfermod.modulus.certified_grid_max`
returns `(lower, upper, Λ)` and documents which of these is which.

**In the reference application.** Directions are indexed by `ρ`, and
`ω_D^{(ρ)}(5e-3)` is Lipschitz with estimated `Λ ≈ 0.149`:

| grid spacing δ | nodes | lower | upper = lower + Λδ |
|---|---|---|---|
| 0.200 | 5 | 0.023538 | 0.053310 |
| 0.100 | 10 | 0.023538 | 0.038424 |
| 0.050 | 19 | 0.023538 | 0.030981 |
| 0.010 | 95 | 0.023538 | 0.025027 |

Every bracket contains the fine-grid value, and the interval tightens as `Λδ`.
One caveat visible here: `ω_D^{(ρ)}` increases as `ρ ↓ 0` and saturates —
0.023407 at `ρ=0.2`, 0.023553 at `0.02`, 0.023561 at `0.002` — so the direction
set `(0,1)` is not compact and the supremum is attained only in the limit. (C3)
holds on the compactification `[0,1]`, where the limit exists; the reported
0.0235 is the value there. That is a concrete instance of why compactness is a
hypothesis and not a formality.


## 7. Reference application: numerically diagonalised finite-cutoff compact U(1) spectral inference

### 7.1 Setup

`X` = spectral decompositions of a two-point response; `‖·‖_A` = signal-normalised
correlator residual with equal-time moments; `Q` = extremal decay rate (a mass
gap). The target is the reference application throughout — 2+1D compact U(1)
Kogut–Susskind gauge theory on a small
periodic lattice with a hard-wall-truncated electric basis, solved by full
floating-point diagonalisation of the selected Gauss-law basis block. Thus `Q`
is a finite-matrix eigenvalue difference with no Monte Carlo sampling error,
but it remains sector-, volume-, cutoff-, and channel-dependent. Correlators follow from
`C(t) = Σ_n |⟨0|O_c|n⟩|² e^{−(E_n−E_0)t}`. Reference diagnostics check the all-sector strong-coupling gap (`→g²` on the
2×2 torus), the zero-winding plaquette-sector gap (`→2g²`), gauge invariance,
Hermiticity, winding-block preservation, full-versus-sector channel agreement,
and cutoff-boundary occupation.

`w_corrupt(τ)` is the least fabricated amplitude shifting the estimated `Q` by
more than τ.

### 7.2 The perturbation family

*(formerly "Confound construction" — unchanged, now realising §4's `u_w`)*

The family fabricates a single faint slow mode at rate `ρ·Δ` (`ρ < 1`) carrying a
fraction `w` of the operator variance, renormalising the true weights so `C(0)`
is preserved exactly. Being slowest it dominates the long-time tail and biases
the effective-mass plateau downward, while its contribution stays below the
aggregate error floor over the measured window. Physically: a fabricated
long-range correlation — a spurious near-zero eigenvalue in a reduced operator,
or a slow autocorrelation mode in a sampler.

The alternative direction is structurally constrained: a surrogate matching the
reference on the window to precision ε cannot bias the leading rate upward
freely, since window data bound the slowest rate from below. Upward bias requires
a genuine light state with small operator overlap — the unsmeared-operator
regime — supplied as a secondary mechanism.

**Proposition 7.1 (constructibility of faint slow modes).** *(Realisation-specific:
an existence statement about one perturbation family, not a structural result.)*
Let the reference have spectral representation `C(t) = Σ_n w_n e^{−Δ_n t}` with
`w_n > 0`, total weight `W ∈ (0,∞)` and slowest rate `Δ`. Fix a bounded window
`[0, t_max]`, an aggregate tolerance ε > 0 and `0 < ρ < 1`. Adding a mode at rate
`ρΔ` with weight `wW` and rescaling the original weights by `(1 − w)` — so
`C_w(0) = C(0)` exactly — there exists `w > 0` with `‖C_w − C‖_A ≤ ε` while the
asymptotic rate of `C_w` equals `ρΔ`.

*Proof.* The asymptotic rate is `ρΔ` for every `w > 0`, the added mode being
strictly slowest with positive weight. And `C_w − C = w(W e^{−ρΔ·} − C)` is linear
in `w` on the bounded window, so `‖C_w − C‖_A = w‖W e^{−ρΔ·} − C‖_A → 0`; take
`w ≤ ε/‖W e^{−ρΔ·} − C‖_A`. ∎

The window bound is load-bearing: over `[0,∞)` the aggregate discrepancy no
longer vanishes with `w` uniformly, which is the same fact as ω growing with
`t_max`.


### 7.2.1 Adversarial interpretation

Proposition 7.1 can be written as an adversarial optimization problem. For an
admissible family

```math
D_\varepsilon(u)=\{u' : \|u'-u\|_A\le\varepsilon\},
```

the constructed slow mode seeks a member of \(D_\varepsilon(u)\) that maximizes
the declared downstream discrepancy. This is mathematically parallel to the
robust-optimization view of adversarial examples: a perturbation remains small
in the validation geometry while producing a large change after the downstream
map. The scientific distinction is that the perturbation is a surrogate model
or spectral measure and the target is a derived physical conclusion.

The connection supplies useful language and algorithms—inner maximization,
attack families, certified radii, and robust training—but does not identify the
two problems. Scientific admissibility is set by a validation contract, and the
quantity of interest may be asymptotic, operator-valued, or physically
constrained in ways absent from standard input-space attacks.

### 7.3 Where the pair sits

Finite-window `Q` is `‖·‖_A`-continuous (row 2 of the trichotomy, by the
`w → 0` table), so Theorem 3.1 gives certifiability at *some* tolerance. The
modulus says at which:

| ε | 5e-3 | 1e-4 | 1e-6 | 1e-8 |
|---|---|---|---|---|
| ω(ε) | 0.950 | 0.928 | 0.497 | 0.030 |

At the acceptance floor `Q` is essentially unconstrained; (ε₀, 3%)-certifiability
needs ε₀ ~ 1e-8, five orders tighter. Gating restores control, flat in ε as
Theorem 5.2 predicts: ω_D = 0.0235 at every ε above, matching the independently
computed family threshold τ\* = 2.36% below which silent risk becomes positive. Taking `Q` as the
true `t → ∞` rate instead puts the *same system and seminorm* in row 3:
`Q(u_w) = ρQ(u)` for every `w > 0`, discontinuous, and ω_D = 0.950 — uncertifiable
by Corollary 3.2 and unrescued by Corollary 5.1.

### 7.4 The certified set, and results

*(formerly "Benchmark design", "Gate", "Pilot result", "Robustness")*

`D` is defined by a preregistered gate with conventional tests (equal-time
expectation and variance; signal-normalised aggregate RMSE) and spectral tests
(channel-gap error; effective-mass plateau; relative tail-decay error;
leading-overlap alignment), thresholds frozen before interpretation. **`D` was
fixed before ω was defined**, which is why the agreement in §7.3 between ω_D and
τ\* is evidence rather than construction.

The gate evaluates *outputs, not architectures*: the contract is
`correlator(ts)`, `variance()`, `metadata()`, with optional `provenance()` and
`training_info()`; models with an explicit spectral decomposition supply
`spectral()` and inherit the rest. Four families are implemented, one
representative each, because each fails for a different structural reason:
(I) numerical — exact diagonalisation and Krylov/Lanczos, the trust anchor,
required to agree with the reference to machine precision (gap 8.9e-16,
correlator 1.4e-15); (II) reduced — POD/Galerkin and reduced transfer operators,
where subspace truncation and Rayleigh–Ritz bias the extremal spectrum;
(III) learned — a VMC-trained neural quantum state and a fitted multi-exponential
model, where aggregate loss is blind to low-signal modes; (IV) constructed — the
positive control, without which sensitivity would be untested. Confidence
originates in Class I and transfers downward.

**Pilot.** The honest reduced-transfer surrogate and an echo control are certified
(no false positive). The constructed control passes every conventional test —
identical equal-time variance, aggregate RMSE 2.6e-3 below a 5e-3 floor — and
fails the spectral tests, reporting `Q` ~40% below truth. Stable across a
coupling and truncation sweep.

**Reachable from ordinary practice.** A POD/Krylov–Galerkin reduction exhibits the
failure with no adversarial design. Krylov moment matching reproduces equal-time
and short-time data essentially exactly, so conventional checks cannot fail;
Rayleigh–Ritz converges from above, so an under-resolved reduction over-estimates
systematically and single-signed. Ranks 2–5 are confounds (gap error 3–33%,
`C(0)` error ≲ 2e-15); rank ≥ 6 is certified. At ranks 4–5 the gap error is
*inside* tolerance and the tail test alone drives the verdict — long-time fidelity
is the leading indicator.

**Cross-estimator robustness.** Effective-mass plateau, single- and multi-exponential
fits, Prony/linear-prediction reconstruction, and AIC model averaging all recover
the reference `Q` on clean data and are biased identically by the constructed
perturbation, because each reads the same corrupted tail. These are distinct
functionals of a shared correlator information basis: their agreement shows that
the failure is not estimator-specific, but it is not independent corroboration.
A variational operator
basis catches single-channel confounds as cross-channel inconsistency, but not
those applied consistently across channels, which still require the reference.

**Information bases of gates.** A gate result can be labelled by the measured
information on which it depends. Formally, let `G` be the set of gate tests and
let `b : G → 𝓘` map each test to an information-basis label. Then gate diversity
`|{g₁,…,gₘ}|` and information-basis diversity `|{b(g₁),…,b(gₘ)}|` are distinct
quantities. The map `b` is metadata for interpreting a certification report; it
does not enter the restricted-modulus theorems and does not assert statistical
independence. In the implementation the labels distinguish equal-time
observables, an aggregate correlator window, a long-distance correlator tail,
and an explicit spectral decomposition.

**Stability.** `R` converges under truncation refinement as `Q` does
(`Λ = 1,2,3,4` → |𝓘| = 3.18, 2.92, 2.92, 2.92) and moves with the physics, so it
characterises the inference problem rather than the discretisation.


### 7.5 A non-physics instance: graph sparsification

The transfer construction is not specific to spectral physics. The executable
graph instance in `scripts/run_graph_instance.py` takes:

- \(X\): weighted graphs on a fixed vertex set;
- \(\|\cdot\|_A\): relative Frobenius error of the graph Laplacian;
- \(Q\): algebraic connectivity, the Fiedler value \(\lambda_2\).

A perturbation weakens inter-cluster bridge edges while conserving total edge
weight. Aggregate Laplacian error remains small, yet \(\lambda_2\)—and therefore
conclusions about mixing, robustness, and clustering—can change substantially.
The instance reproduces the same logical pattern as the spectral example:

```text
small aggregate approximation error
        does not imply
small error in a structurally sensitive derived quantity.
```

This second domain is intentionally compact rather than a full graph-theoretic
case study. Its role is to demonstrate that the fidelity-modulus construction
depends on an admissible set, a quantity of interest, and a discrepancy
geometry—not on the physics of correlator tails.



### Coverage tiers and one-sided interpretation

A computed family-restricted value has one of three epistemic statuses:

1. **Proven exact**: \(C1\) is proved and \(\omega_{D,\Theta}=\omega_D\).
2. **Certified adversarial floor**: the family supremum is valid but coverage is
   unproved, so \(\omega_{D,\Theta}\le\omega_D\).
3. **Exploratory sample**: even the family supremum has not been established.

The one-sided interpretation is essential:

> Large floors condemn; small floors do not acquit.

A large certified floor proves substantial exposure. A small floor cannot rule
out an unsearched confound and therefore must not be described as a negative
result or a safety certificate.

Agreement among several perturbation families is useful robustness evidence,
but it does not prove \(C1\); distinct families may share the same blind spot.

## Exact coverage in structured families

The coverage condition \(C1\) is generally substantive, but it can be proved in
important structured settings.

### Theorem — continuous linear functional over a Hilbert ball

Let \(H\) be a real Hilbert space, let
\(D_\varepsilon(u)=\{u+h:\|h\|_H\le\varepsilon\}\), and let \(Q\) be continuous
and linear with Riesz representer \(q\). Then

```math
\sup_{\|h\|_H\le\varepsilon}|Q(u+h)-Q(u)|
=
\varepsilon\|q\|_H,
```

attained by \(h=\pm\varepsilon q/\|q\|_H\) when \(q\neq0\).

For a nonlinear \(Q\), the representer of \(DQ(u)\) gives only a local
first-order adversarial direction unless stronger global assumptions are
proved.

### Theorem — ellipsoidal linear coverage and seminorm kernels

Let \(M\succeq0\), let \(D_\varepsilon=\{h:h^\top Mh\le\varepsilon^2\}\), and
let \(Q(h)=q^\top h\). If \(q\) has a nonzero component in \(\ker M\), then the
modulus is infinite. Otherwise,

```math
\omega(\varepsilon)
=
\varepsilon\sqrt{q^\top M^\dagger q}.
```

This theorem makes the seminorm-kernel warning exact: an uncontrolled quantity
direction is not merely numerically unstable; it is unbounded under the stated
validation contract.

### Corollary — sketch-nullspace coverage

Under \(Ph=0\) and \(\|h\|_2\le\varepsilon\),

```math
\sup |\langle q,h\rangle|
=
\varepsilon\|\Pi_{\ker P}q\|_2.
```

The projected query direction is the exact extremizer.

### Theorem — leading eigenvalue under Frobenius perturbations

For symmetric \(A\) and unrestricted symmetric \(\Delta\) satisfying
\(\|\Delta\|_F\le\varepsilon\),

```math
\sup_\Delta
\left[
\lambda_{\max}(A+\Delta)-\lambda_{\max}(A)
\right]
=
\varepsilon.
```

A rank-one perturbation \(\Delta=\varepsilon vv^\top\), with \(v\) a leading
eigenvector, attains the bound. This statement does not automatically extend to
spectral gaps, Fiedler values, trace-constrained density matrices, or
physically constrained Laplacian perturbations.

## Known tensions

Four limitations are structural rather than incidental. They are stated here
together because each qualifies a headline number elsewhere in the document.

**1. Coverage is unproved, so the spectral-family values are `ω_{D,Θ}` lower bounds.** Corollary 6.1
is exact about this: without a proof that the single-mode family exhausts the
certified set, what is computed is `ω_{D,Θ}`. The quantitative claims — the 40×
gating reduction, `ω_{D,Θ} = 0.0235`, the `|𝓑| = 0` certification below τ\* = 2.36% —
are therefore **family-dependent**. A richer family could only raise `ω_D`, never
lower it, so the direction of the error is known; its size is not.

**2. The linear regime is narrow enough that first-order analysis is useless
where the theory operates.** `κ_rel ≈ 4 × 10⁶` and the linearisation holds only
for `ε ≲ 2 × 10⁻⁷`, against a `5 × 10⁻³` acceptance floor — four orders of
magnitude out. This is the case *for* computing the modulus rather than quoting a
condition number, but it also means the interesting regime is exactly the one
where classical first-order reasoning has already stopped applying. Row 2 is not
an edge case of the classical theory; it is outside it.

**3. The composition slack is bounded but still too large for a stage-wise error
budget.** The slack is now controlled a priori by the openness constant of the
upstream map (`slack ≤ 1/c` in the homogeneous case), but `c` must be estimated,
and the measured 3.6–4.5× gap between `ω̂₂ ∘ ω̂₁` and the true composite means
stage-wise gating bounds a pipeline but does not yet *budget* one: a four-stage
chain could inherit two orders of magnitude of slack. Estimating `c` without
solving the composite problem (open question 4) is the highest-leverage
theoretical follow-up here. A natural refinement is to characterize
intermediate maps that are contractive in the relevant validation and decision
geometries. If one stage strictly contracts admissible perturbations, the
composite bound should inherit less slack than the general openness estimate;
the exact improvement depends on how contraction interacts with the declared
discrepancies at the two stages.

**4. The verdict depends on how the quantity of interest is read.** The same
system is row 2 for a finite-window estimator and row 3 for the true `t → ∞`
rate, so the practical claim is sensitive to the choice of estimator. The
cross-estimator robustness checks (plateau, exponential fits, Prony, AIC
averaging, all biased identically) mitigate this but do not remove it: they show
that several extraction algorithms applied to the same correlator tail agree,
not that the tail supplies independent evidence or that a finite window is the
right reading.


## 8. Scope

Demonstrated on a finite-volume, finite-cutoff system with numerically diagonalised reference matrices; this establishes a
detectable failure mode and a non-false-positive gate, not any continuum or
Yang–Mills statement. ω is computed over a single-mode family, hence a lower
bound on the modulus over all perturbations. Certification is a threshold test,
not a validated bound.

## 9. Open questions

A finite-ε refinement of Theorem 3.1 (the row-1/row-2 boundary); estimating ω
without a reference of known `Q`; the cost of estimating ω_D in general;
tightness of Proposition 2.2(7) beyond the openness bound of the slack proposition —
estimating `c` without solving the composite problem, and the non-homogeneous
case; whether contractive intermediate maps yield a strictly sharper
composition calculus than the current openness bound; whether the tensor-product
subadditivity of `log(1 + ω)` reflects a deeper information-geometric or
resource-additive structure, or is only the algebraic linearization of a
multiplicative error bound; what replaces Corollary 3.1
for order-type discrepancies; making ω_D a construction objective, stage-wise per
Proposition 2.2(7); and whether sketching or sparsification supplies a practically
interesting row-2 pair.

**The operator viewpoint.** Proposition 2.2 is really a theorem about the transfer
operator `T : Q ↦ ω_Q` rather than about any single modulus. Identifying the
class of maps `T` preserves, and whether `T` is determined by its behaviour on a
generating family of quantities, would make the theory more compact than it is
here — where `T` appears as an organising device rather than an object of study.
That is the direction in which the structural layer would next be strengthened,
and it generates its own questions: which transformations preserve `T`, which
contract it, what the complexity of evaluating `T` is on different classes of
quantities, and whether `T` admits an axiomatic characterisation (below).

**An axiomatization theorem?** With `𝒢` given by Definition 2.1, the
question is trivial: Proposition 2.1 says `T(Q)` is the least element of `𝒢(Q)`,
which is unique, so `T` is already determined. The interesting version drops `𝒢`.
Start instead from an abstract assignment `T : Q ↦ g_Q` into non-decreasing
functions, required only to satisfy the structural axioms — a decomposition
property, the calculus of Proposition 2.2, an adjunction with restriction, and
naturality under reparameterisation — with no reference to the defining
supremum. Is `T` then unique? An affirmative answer would be an *axiomatization
theorem*: it would characterise the fidelity modulus by its behaviour rather than by its
construction, so that any operator satisfying the least-guarantee
property, the basic calculus, restriction–saturation relation and reparameterisation behavior is
necessarily `T`. A negative answer would exhibit a second, inequivalent transfer
calculus, which would be at least as interesting.



## Empirical extensions required for external validation

The mathematical results above do not by themselves establish that the
framework is effective across contemporary surrogate-model pipelines. Three
classes of experiments remain especially important.

### Learned operators and trained neural surrogates

Constructed confounds establish possibility and, in some settings, exact worst
cases. Learned-operator experiments address a different empirical question:
whether aggregate and decision error actually decouple in trained surrogates
that were not designed to exhibit the phenomenon.

The first trained-model audit in this repository is negative. A random-feature
diffusion operator passes its ordinary smooth-distribution validation, but every
member of the frozen localized challenge family has relative global L2 error
far above the declared aggregate threshold. The largest decision error rises
with conspicuous aggregate error; no point enters the predeclared
small-aggregate/large-decision region. The instance therefore demonstrates
proportionate out-of-distribution degradation, not a failure hidden from
aggregate validation.

This distinction is substantive. Passing an in-distribution validation set does
not make a later out-of-distribution error “silent” when the same aggregate
metric detects that error on the challenged input. The empirical object is the
joint aggregate–decision profile, not the fact that the model once passed a
different sample.

External validation must therefore test discovery rather than constructibility.
The confirmatory protocol in `LEARNED_OPERATOR_PROTOCOL.md` freezes external
checkpoints, metrics, quantities, thresholds, families, budgets, and reporting
rules before evaluation. Coupled and null profiles are first-class outcomes;
a negative result must not be replaced post hoc by a newly designed successful
family.

### Computational scaling in higher-dimensional PDEs

The finite-grid procedure of Theorem 6.1 is transparent, but its cost grows with
the dimension and resolution of the perturbation parameterization. Large-scale
two- and three-dimensional PDE instances should therefore report complexity
empirically: objective evaluations, admissibility evaluations, convergence of
the supremum estimate, and benefits from adaptive search or adjoint-informed
directions.

This is a question about the practical attainability of the restricted
supremum, not the validity of the theorem.

### Posterior-measure Silent Risk

The measure-qualified definition permits a posterior measure
\(\Pi(\mathrm d u\mid Y)\). In that case Silent Risk is the posterior mass of
admissible alternatives whose decision discrepancy exceeds a declared
tolerance:

```math
\mathrm{SR}_{\Pi,\tau}
=
\Pi\!\left(
\ell(Q(u),Q_0)>\tau
\mid Y
\right).
```

A Bayesian instance should keep three objects distinct:

1. the posterior law constructing or weighting the admissible family;
2. the decision discrepancy \(\ell\);
3. the transfer functional \(Q\).

The experiment should be compared against posterior summaries that can conceal
decision instability, including plug-in estimates and parameter-level credible
intervals.

Version 1.2.2 includes compact executable validations for each track
(numerical results: `VALIDATION_REPORT.md`). Their outcomes are mixed: the
learned-operator instance is negative for decoupling, the PDE study addresses
only compact-grid feasibility, and the Bayesian study validates software
separation of probabilistic and decision-level quantities. None replaces an
external pretrained-operator, solver-coupled 3D, or non-conjugate Bayesian
study.


### Spectral-gap API migration

`exact_gap` is deprecated and will be removed in TransferMod 2.0. It is a
warning-emitting compatibility alias for `sector_gap`, not a separate global
quantity. New code should use:

| Legacy name | Canonical replacement |
|---|---|
| `exact_gap` | `sector_gap` |
| — | `global_physical_gap` when the full all-sector basis is available |
| — | `spectrum_scope` and `basis_mode` to state interpretation |

## References

Becker, R., and Rannacher, R. (2001). “An Optimal Control Approach to A
Posteriori Error Estimation in Finite Element Methods.” *Acta Numerica* 10:1–102.
doi:10.1017/S0962492901000010.

Bertsimas, D., and Sim, M. (2004). “The Price of Robustness.”
*Operations Research* 52(1):35–53.

Goodfellow, I. J., Shlens, J., and Szegedy, C. (2014).
“Explaining and Harnessing Adversarial Examples.” arXiv:1412.6572.

Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*,
2nd ed. SIAM.

Madry, A., Makelov, A., Schmidt, L., Tsipras, D., and Vladu, A. (2017).
“Towards Deep Learning Models Resistant to Adversarial Attacks.”
arXiv:1706.06083.

Oden, J. T., and Prudhomme, S. (2001). “Goal-Oriented Error Estimation and
Adaptivity for the Finite Element Method.” *Computers & Mathematics with
Applications* 41(5–6):735–756. doi:10.1016/S0898-1221(00)00317-5.

Oden, J. T., and Prudhomme, S. (2002). “Estimation of Modeling Error in
Computational Mechanics.” *Journal of Computational Physics* 182(2):496–515.
doi:10.1006/jcph.2002.7183.

Owhadi, H., Scovel, C., Sullivan, T. J., McKerns, M., and Ortiz, M. (2013).
“Optimal Uncertainty Quantification.” *SIAM Review* 55(2):271–345.
doi:10.1137/10080782X.

Patera, A. T., and Rozza, G. (2007). *Reduced Basis Approximation and A
Posteriori Error Estimation for Parametrized Partial Differential Equations*.
MIT Pappalardo Graduate Monographs in Mechanical Engineering.

Prudhomme, S., and Oden, J. T. (1999). “On Goal-Oriented Error Estimation for
Elliptic Problems: Application to the Control of Pointwise Errors.” *Computer
Methods in Applied Mechanics and Engineering* 176:313–331.
doi:10.1016/S0045-7825(98)00343-0.

Rozza, G., Huynh, D. B. P., and Patera, A. T. (2008). “Reduced Basis
Approximation and A Posteriori Error Estimation for Affinely Parametrized
Elliptic Coercive Partial Differential Equations.” *Archives of Computational
Methods in Engineering* 15(3):229–275.
doi:10.1007/s11831-008-9019-9.

Smith, R. C. (2013). *Uncertainty Quantification: Theory, Implementation, and
Applications*. SIAM.

## Appendix A. Extensions

### A.0 Why a minimum exists

The ambient poset `[0,∞]^{(0,∞)}` under the pointwise
order is a complete lattice, `[0,∞]` being one and products of complete lattices
being complete. `𝒢` is upward closed (weakening a valid guarantee keeps it valid)
and closed under arbitrary pointwise infima (an infimum of upper bounds of `S(ε)`
is an upper bound of `S(ε)`), hence is a principal up-set generated by its own
infimum. Existence is therefore structural rather than incidental; uniqueness is
trivial; the content of Proposition 2.1 is that this abstractly guaranteed
infimum *is* ω, and is attained — so `𝒢` has a least element, not merely a
greatest lower bound.

### A.1 Further extensions


Arbitrary discrepancies on `Q` (spectral, order-type, Hausdorff — covering
operator-valued `Q`) and arbitrary divergences on `X` (Bregman, KL, quantile):
Definitions 1–2, Proposition 2.1, Theorem 6.1 hold verbatim; Theorem 3.1 needs
`d(u,u) = 0` and a convergence notion. Locality and uniform classes;
vector-valued and Banach-valued codomains; multi-parameter families by ray
decomposition; randomised approximations via a high-probability form of
Theorem 5.2, under which a published approximation guarantee is itself a transfer
guarantee and therefore, by Proposition 2.1, a computable upper bound on the gated
modulus (the gap being the guarantee's slack for that quantity of interest);
Theorem 5.2; and which row classical embedding results place standard pairs in.
