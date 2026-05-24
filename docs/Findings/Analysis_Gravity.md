# Gravitational Theory in the Common Governance Model: Causal Preservation of Ancestry through Identity and Individuality

## 1. Introduction

Physics provides accurate field equations for gravity but treats three foundational inputs as assumptions: the rest frame, the dimensionality of space, and the numerical value of the gravitational coupling constant. In Newtonian gravity, the 4π factor in the field law is a direct consequence of enclosing a mass in three spatial dimensions, yet the origin of those three dimensions remains unexplained [1]. In general relativity, the coupling κ = 8πG/c⁴ is fixed by requiring the Newtonian limit, leaving G as an externally measured parameter [4]. Similar dependence on unexplained prior assumptions occurs in Nordström's scalar theory [3], gravitoelectromagnetism [2], and linearized spin-2 formulations [5].

The Common Governance Model (CGM) rests on a single axiom: reality is organized by common capacity for freedom. The conditions for this freedom to manifest generate the observable features of spacetime and information. Physical conservation and informational coherence emerge as expressions of a single underlying order, the Preservation of Ancestry. To preserve this ancestry, operations must follow a strict sequence. Causality may be classified as a four-phase operational cycle that enforces this order. Gravity, including its coupling strength, dimensional profile, and causal structure, arises solely from the requirement to complete this cycle.

For the mathematical and computational realization of the conditions CGM defines, we have implemented a finite-state algorithm called the algebraic Quantum Processing Unit Kernel (aQPU). This computational medium provides the exact combinatorial invariants needed to anchor a continuous field theory, extracting precise physical constants from purely formal foundations.

This analysis establishes the following results:

*   The quantum of gravity emerges as the invariant **Q_G = 4π**, representing the complete solid angle necessary for coherent observation in three dimensions (Sections 2 and 3)
*   The framework identifies **gravity as the dynamical origin of rest mass** through the Virial condition, deriving the relativistic rest frame from operational closure (Section 4)
*   The aQPU kernel supplies the exact combinatorial invariants that fix **G_kernel**, the discrete Gauss law, and the continuum normalization required by the field equations (Section 5)
*   The **Poisson equation, gravitoelectromagnetic decomposition, nonlinear metric extension, and a closed-form expression for the dimensionless coupling Gv²/G_kernel** follow from the six degrees of freedom and aQPU kernel invariants; with the electroweak scale as energy anchor, numerical evaluation matches the CODATA reference value for G to within 0.074 ppm (Section 6)
*   **Causality is redefined as a four-phase operational cycle** (Source, Act, Retrieve, Commit). The standard light cone is the geometric projection of the commit phase. The first three phases are algebraically guaranteed, explaining how identity persists at gravitational horizons where propagation fails (Sections 6 and 8)
*   The framework yields **testable observational signatures**, including reduced black hole shadow sizes, coupling suppression in neutron star interiors, and strict constant-product falsification thresholds (Section 7)

Companion analyses provide supporting results, including the dimensional proof [15], the fine-structure constant calculation [24, 27], the UV-IR energy ladder [25, 26], the compact opacity construction [16], and the byte formalism [17]. These companion documents are archived at Zenodo under the same DOI as the core CGM framework [14].

### 1.1 Units
Throughout this manuscript, natural units c = ℏ = 1 are used except where SI is explicitly noted in observational predictions.

## 2. Foundations
The Common Governance Model is formalized as a propositional modal logic with two primitive modal operators representing recursive operational transitions.

**Primitive symbols:**

| Symbol | Description |
|--------|-------------|
| S | A propositional constant: the horizon constant |
| ¬ | Logical connectives: negation |
| → | Logical connectives: material implication |
| [L] | Modal operators: left transition |
| [R] | Modal operators: right transition |

Modal depth refers to the nesting level of modal operators. For instance, [L][R]S has depth two (two nested operators), while [L][R][L][R]S has depth four. Modal depth plays a critical role in CGM: depth-two operations exhibit contingent behavior (non-absolute unity and opposition), while depth-four operations achieve necessary closure (universal balance).

Throughout the logical development we reserve the symbol S for the designated propositional constant anchoring the horizon worlds. When this constant is realized in the Hilbert-space representation via the GNS construction, its expectation value in the cyclic vector state equals the horizon normalization Q_G = 4π. This value arises as the ratio λ/τ of two parameters fixed by the operational conditions [14] and is imposed as the GNS normalization that selects the compact real form, rather than derived as an eigenvalue of S.

#### Core Definitions
Four formulas capture the structural properties required by the Common Governance Model, all anchored to the horizon constant S:

| Concept | Formula | Description |
|---------|---------|-------------|
| Unity (U) | [L]S ↔ [R]S | Unity holds when left and right transitions yield equivalent results at the horizon constant. |
| Two-step Equality (E) | [L][R]S ↔ [R][L]S | Two-step equality holds when depth-two modal compositions commute at the horizon constant. |
| Opposition (O) | [L][R]S ↔ ¬[R][L]S | Opposition holds when depth-two modal compositions yield contradictory results at the horizon constant. |
| Balance (B) | [L][R][L][R]S ↔ [R][L][R][L]S | Balance holds when depth-four modal compositions commute at the horizon constant. |

#### The Five Foundational Conditions
The framework relies on five foundational conditions: one assumption (CS), two lemmas (UNA, ONA), and two propositions (BU-Egress, BU-Ingress). For independence analysis in the core modal system we treat all five as primitives. In the operational regime, the continuous flows, reachability from S, and simple Lie closure allow UNA and ONA to be obtained from CS (hence the lemma designation). The conjunction of BU-Egress and BU-Ingress defines universal balance.

### 2.1 Ancestry: Common Source (CS) Assumption
The Common Source (CS) assumption requires that all distinguishable physical structure preserve ancestry through common origination.

```text
S → ([R]S ↔ S ∧ ¬([L]S ↔ S))
```

This establishes fundamental chirality in the system, as the reference state behaves asymmetrically under the two types of transitions. The horizon constant S is preserved under right transitions but altered under left transitions.

Such traceability requires an ancestral parity violation, manifesting physically as chirality. This asymmetry governs the emergence of individuality, forming the dimensional identity called gravity, and enables coherent observation by distinguishing operational paths.

> **Definition:** Gravity is the emergent balance establishing preservation of ancestry through freedom of identity and individuality.

Gravitational alignment requires a distinct identity with accountable individuality (Unity and Opposition Non Absolute). Ancestry preserves a balance between these extreme operational modes, which necessitates energy conservation through gyration.

Composing displacements in a curved geometry yields a non-associative operation, which gyrogroup algebra corrects via the gyration operator. Accumulated gyration manifests as rotational structure in the continuous physical limit. Angular momentum emerges as the physical expression of this conserved gyration, preserving directional distinction and ancestry through translation.

### 2.2 Identity of Individuality: Unity-Non-Absolute (UNA) Lemma
Non-absolute unity (¬□E) ensures informational variety while maintaining ancestry preservation, preventing homogeneous collapse.

```text
S → ¬□E    where E := [L][R]S ↔ [R][L]S
```

At depth two, the order of transitions matters, yet this non-commutativity remains contingent across all accessible states, holding in some and failing in others. This non-absolute behavior expands the initial chirality into a rotational structure. The BCH expansion of the depth-four commutator condition forces the algebra to close on three generators as sl(2) [15]. SU(2) provides the minimal algebraic framework for this expansion, yielding exactly three rotational degrees of freedom.

### 2.3 Individuality of Identity: Opposition-Non-Absolute (ONA) Lemma
Non-absolute opposition (¬□¬E) ensures accountability of inference. Different operational paths remain comparable even when yielding different results, preventing structural fragmentation.

```text
S → ¬□¬E
```

Depth-two operations may yield opposite results, yet this opposition remains contingent. The system avoids complete agreement and complete contradiction. Opposition-Non-Absolute operates as the next transformation, introducing three translational degrees of freedom that inherently encompass the prior rotational ones. This sequential generation produces the algebra of rigid body motions, SE(3).

### 2.4 Balance Universal Proposition
Balance Universal governs the interaction between these six degrees of freedom. As displacement increases, the operational cost of sustaining coherence escalates, manifesting as gravitational attraction. Mass and energy represent the accumulated memory of this balance. In the relativistic limit, this structure maps directly to the gravitoelectric and gravitomagnetic fields.

Time emerges from the sequential ordering required by Balance Universal. Achieving depth-four balance demands first passing through the non-absolute stages. This fixed prerequisite imposes a directional structure on the sequence, yielding the arrow of time.

#### 2.4.1 BU-Egress: Depth-Four Closure
Egress Balance (BU-Egress) mandates that four-step compositions commute across all accessible configurations.

```text
S → □B    where B := [L][R][L][R]S ↔ [R][L][R][L]S
```

This absolute closure (□B) is the minimal depth at which balance occurs while preserving depth-two variety. Depth three still permits asymmetry, while depth four forces commutative convergence. Egress represents the centrifugal limit of this outward expansion.

#### 2.4.2 BU-Ingress: Memory Reconstruction
Ingress Balance (BU-Ingress) requires the closed state to retain the information necessary to reconstruct the original asymmetry and earlier contingencies.

```text
S → (□B → ([R]S ↔ S ∧ ¬([L]S ↔ S) ∧ ¬□E ∧ ¬□¬E))
```

The depth-four state preserves the full operational history, including the original chirality, rotational variety, and translational expansion. Balance implies memory, and ingress represents the centripetal binding reconstructing the original context without erasing structural distinctions.

The following results are theorems of the modal system proven in [15]. The BCH expansion of the depth-four commutator condition forces the generated Lie algebra to close on three generators as sl(2) (Hall word exclusion of bracket length ≥ 3). The simplicity requirement from BU-Ingress excludes direct-sum algebras such as so(4). The GNS construction selects the compact real form su(2). ONA's bi-gyrogroup consistency forces the semidirect product SE(3) = SU(2) ⋉ R³ with exactly three translational parameters. The dimensional proof in [15] establishes n = 3 as the unique dimension satisfying all five conditions simultaneously, with constructive exclusions of n = 2 and n ≥ 4. The aQPU kernel in Section 5 is a concrete finite realization of these theorems, not an independent postulate. In the GNS representation of [14], the modal operators [L] and [R] are realized as one-parameter unitary groups U_L(t) = exp(itX) and U_R(t) = exp(itY) on L²(S²), providing the Lie algebra elements X and Y on which the BCH expansion operates.

---

Gravity operates as the continuous capacity for conserving ancestry, functioning independently of discrete internal states. Ancestry is not localized to a single spatial point; each local measurement must remain accountable to the full operational history without directional bias. A sphere is the only geometry satisfying this requirement, demanding exactly 4π steradians for complete angular closure in a spatially closed and finite measurement domain. This geometric necessity establishes the quantum of gravity, Q_G = 4π. Mass and energy emerge as the accumulated operational memory required to sustain this spherical alignment against displacement.

## 3. The Quantum of Gravity

The geometric requirement of 4π steradians established in Section 2 dictates the quantum of gravity. Q_G = 4π is the ratio λ/m_a of two parameters fixed by the operational conditions before any geometric structure is assumed [14]. The solid angle of a sphere in three dimensions is a downstream realization of this algebraic fact. Gravity also requires quantization, meaning this continuous sphere must resolve into discrete operational passes. The aperture parameter m_a governs this resolution. Normalizing the directional asymmetry across both transition directions yields the bridge identity:

```text
m_a² × (2π)_L × (2π)_R = π/2
m_a² = 1/(8π)
Q_G × m_a² = 1/2
```

Here (2π)_L and (2π)_R denote the angular coverage contributed by the left and right transition directions respectively. Each contributes 2π because a full cycle in either direction covers one complete angular revolution.

The continuous 4π sphere thus resolves into a half-integer quantum pass, reflecting the underlying SU(2) double-cover structure established by the rotational degrees of freedom. The closure ratio ρ measures how close the balanced state comes to filling the aperture, while the aperture gap Δ = 1 − ρ measures the deviation from full closure. Numerically, m_a ≈ 0.199471 and Δ ≈ 0.0207.

This 4π invariant appears naturally in the field equation for a point source:

```text
∇²(1/r) = −4π δ³(r)
```

The Laplacian operator (∇²) measures how a field spreads outward from a source, and the delta function (δ³) isolates the source to a single point. Both sides reflect the same geometric fact as the sphere: complete angular coverage of three-dimensional space requires 4π steradians. Standard physics inherits this factor from pre-existing spatial geometry, whereas CGM derives it from the closure requirement.

## 4. Gravitational Causality

Gravity arises as a causal necessity of preserving operational coherence against dispersion. The system requires a strict causal sequence to maintain the memory of its origin, and gravity constitutes the physical structure that enforces this sequence. 

While standard physics treats zero momentum as an arbitrary coordinate choice, the non-absoluteness of unity and opposition makes kinetic individuality foundational and demands an active dynamical mechanism for pure rest. Because memory reconstruction fails if the system accumulates unbounded net displacement, coherent operational closure forces this displacement to zero over every operational cycle. The kernel verifies this algebraically via the shell displacement invariant D = 24 across all 64 mass configurations as detailed in Section 5.2. Vanishing net displacement forces the net momentum flux to zero and yields the Virial condition.

```text
2T + V = 0
```

This condition is a strict logical consequence of ancestry preservation dictating that total energy equals T + V = -T. What vanishes is the net unbalanced displacement and momentum flux rather than the total energy. Identifying gravity as the dynamic anchor that forces momentum to zero derives the rest energy relation E = mc² rather than merely selecting it as a coordinate choice.

Stage resolution maps the Virial sectors to the CGM recursion. Kinetic energy T arises from ONA and binding potential V from UNA. UNA and ONA are lemmas that generate the six degrees of freedom. BU is the depth-four closure proposition governing these six degrees of freedom, not an additional degree of freedom. BU-Egress guarantees algebraic closure of the identity channel, while BU-Ingress requires geometric propagation to reconstruct the prior state. At gravitational horizons, Egress holds while Ingress fails, which is why the first three causal phases survive but the commit phase is blocked.

Halting momentum requires continuous internal stress, which is the physical exertion of preserving the causal sequence. Confining directional momentum generates a symmetric stress tensor.

```text
σ^{ij} = p δ^{ij} + π^{ij}
```

Isotropic pressure occupies the trace sector while five independent trace-free components of π correspond to the ℓ = 2 representation of SO(3). These five components map to bulk shells 1 through 5 (defined in Section 5.1), which carry symmetric trace-free orientational degrees, while shells 0 and 6 are horizons with zero anisotropy. Gravity couples exclusively to this anisotropic sector because that is the directional exertion of the causal sequence. Computing the exact physical weight of this exertion and the limit where it exhausts spatial capacity requires the field equations and coupling structure derived in Section 6.

## 5. The aQPU Kernel

The continuous field theory requires exact combinatorial normalization. The Common Governance Model achieves this through a compact holographic algorithm called the algebraic Quantum Processing Unit (aQPU). The aQPU is a finite-state, deterministic kernel that turns the modal axioms of Section 2 into an executable integer algebra over a finite field. It does not simulate the continuous physics but provides the exact discrete manifold from which the continuous fields emerge.

### 5.1 The 6-Bit Runtime and State Geometry

The kernel organizes its state space around the six degrees of freedom derived in Section 2. An 8-bit input byte decomposes into two structural domains comprised of 2 boundary bits governing spinorial closure and 6 payload bits governing the SE(3) generators. The 6 payload bits map exactly to the 6 oriented dipole pairs of the 24-bit macro state tensor (GENE_Mac).

From the rest state, the transition law generates a reachable manifold Ω of exactly 4,096 states. The manifold distributes across seven concentric shells indexed by the Hamming distance between the active and passive gyrophases (ab_distance). Shell populations follow |shell_k| = C(6,k) × 64, where C(6,k) counts payload configurations at popcount k and 64 counts the boundary-bit and micro-reference configurations per payload class. The ergodic weight is C(6,k)/64.

| Shell | ab_distance | Population | Characterization |
|-------|-------------|------------|------------------|
| 0 | 0 | 64 | Equality horizon |
| 1 | 2 | 384 | Near unity |
| 2 | 4 | 960 | Intermediate |
| 3 | 6 | 1280 | Equatorial maximum |
| 4 | 8 | 960 | Intermediate |
| 5 | 10 | 384 | Near opposition |
| 6 | 12 | 64 | Complement horizon |

Two shells are horizons (shells 0 and 6) where all directional bias cancels and anisotropy vanishes. Five are bulk shells (shells 1 through 5) which carry nonzero anisotropy. This partition directly produces the gravitational attenuation profile because gravity couples exclusively to the five symmetric trace-free (STF) orientational degrees of freedom carried by these bulk shells. The trace component provides isotropic pressure but does not carry the gravitational signal.

### 5.2 Depth-4 Closure and the Discrete Gauss Law

The minimal closure unit in the kernel is a 4-byte frame (Prefix, Present, Past, Future) mapping directly to the four CGM stages (CS, UNA, ONA, BU). This depth-4 frame is the exact discrete container for the Baker-Campbell-Hausdorff (BCH) commutator cancellation required by BU-Egress.

The shell displacement D measures the total distance traversed through shell space during a complete Z2 holonomy cycle. Each depth-4 half-word performs a pole swap mapping shell s to 6−s (Theorem T2), so the shell path for a carrier starting at shell k follows [k, 6, k, 0] per half-cycle. The Hamming distance traversed is (6−k) + (6−k) + k + k = 12 per half-cycle, independent of k. The full Z2 cycle composes two half-cycles, giving D = 24, verified for all 64 micro-refs across all 4096 states.

The kernel Gauss map converts this integer to the dimensionless coupling.

```text
G_kernel = Q_G / D = π/6
```

The product D × G_kernel = Q_G = 4π gives the total flux per cycle in solid-angle units. This establishes the discrete Gauss law where the flux through any closed surface is quantized in units of Q_G and the coupling is fixed by the ratio of the quantum to the displacement.

### 5.3 Z2 Holonomy Completion and Holographic Identity

Depth-four achieves egress closure (BU-Egress as the W₂ involution). BU-Ingress is the depth-four spectral property that the balanced state retains memory of fundamental chirality, UNA variety, and ONA opposition. Completing the 8-byte holonomy word F ∘ F (K4 composition of two depth-four factors, not a new modal depth) returns the carrier to rest on the opposite Z2 sheet. This two-pass Z2 structure dictates the spin-2 character of gravitational radiation and supplies the factor 2 in the standard normalization 8π = 2 Q_G.

Every mass configuration reaches the equality horizon at the midpoint of the depth-four half-cycle, independent of mass. This fixed midpoint contact is the kernel expression of the Weak Equivalence Principle. At the equality horizon, all six directional bias components resolve simultaneously into a definite state. At the complement horizon, all six are zero. The two boundary horizons with 64 states each and the full manifold with 4096 states satisfy the holographic identity.

```text
|H|² = |Ω|   (64² = 4096)
```

The entropy relation ln|Ω| = 2 ln|H| identifies the Bekenstein-Hawking entropy factor of 2 with the two-pass structure of the gravitational closure cycle. Each pass through the depth-four order point contributes one unit of ln|H| to the total entropy budget.

### 5.4 Gauss Law Bridge

Embedding the seven shell layers into a radial coordinate with the binomial mass profile verifies that the discrete quantities produce the continuum Poisson equation. The boundary flux equals −Q_G G_kernel to relative precision 10⁻¹⁶. In the continuum limit, spherical symmetry and the substitution G = G_kernel/E_CS² matching the kernel profile to continuum mass-energy density ρ yield the Poisson equation derived from the kernel Gauss law rather than assumed from rotational invariance alone.

```text
div g = −Q_G G ρ
```

Three independent numerical checks confirm the inverse-square behavior. The product |g|r² is constant across the exterior to machine precision. A least-squares fit of log|g| versus log r gives an exponent of −2.000000 with uncertainty 9 × 10⁻¹⁶.

### 5.5 Axiom Derivation of Kernel Features

| Kernel feature | CGM condition that forces it | Not a free choice because |
|---|---|---|
| 6 payload bits | 6 DOF from SE(3) = SU(2) ⋉ R³ (Lemmas 1, 2 in [15]) | Changing DOF count would violate the BCH closure or bi-gyrogroup consistency |
| 2 boundary bits | SU(2) double cover gives 4 family phases | SU(2) is the unique double cover of SO(3) selected by the GNS construction |
| L-step mutates A only | CS axiom: left transitions alter S while right transitions preserve it | The asymmetry between active and passive faces is the natural discrete realization of CS chirality. Alternative implementations preserving the L/R asymmetry would produce isomorphic kernel structures |
| R-step is complement-and-swap | [R] preserves the horizon: right transitions must return the carrier to the S-sector | Any R-step that did not preserve the horizon would violate CS |
| 4096-state manifold | BFS enumeration from rest under the transition law | The manifold size is a consequence of the transition law, not a parameter |
| Binomial shell populations | 6-bit payload structure and pair-diagonal mask code | The binomial distribution follows from the 6 independent chirality bits |
| K4 gate algebra | Depth-4 closure forces four-gate structure (Theorem T1) | The Klein four-group is the unique group of order 4 that is not cyclic |

The kernel contains no free parameters. Every structural feature traces to a specific CGM condition or to a mathematical consequence of the transition law.

## 6. Gravitational Field Theory

The quantum of gravity and the aperture geometry of Section 3 fix the coupling structure, while the Virial condition of Section 4 establishes that this coupling arises from the mass-energy stress required to enforce rest mass. This section derives the field equations governing how this stress is sourced, extends them to the nonlinear regime using the aQPU kernel invariants, constructs the effective metric, and computes the exact mass dressing and horizon limits.

Section 6 derives the field theory in six steps, each entailed by the previous. Step 1 computes G_kernel = Q_G/D = π/6 from the horizon invariant Q_G (Section 3) and the shell displacement D = 24 (Section 5.2), both established as kernel theorems. Step 2 integrates the STF attenuation across the five bulk shells to obtain the Refractive Depth τ_G = |Ω|Δρ⁵(1 − 4ρΔ² + c₄Δ⁴) with c₄ = −7/4 fixed by two independent kernel routes (Appendix C). Step 3 evaluates the weak-field coupling G₀ = G_kernel exp(−τ_G)/v². Step 4 derives the position-dependent coupling G(ψ) = G₀ exp(g₁ψ) from the Refractive Depth gradient and E_ref(ψ). Step 5 constructs the effective metric f = 1 − 2ψ from the exact point-mass solution. Step 6 verifies the Einstein tensor, modified Gauss law, and equivalence principles. The dimensionless coupling Gv²/G_kernel = exp(−τ_G) is determined entirely by kernel invariants. Converting to the dimensional G requires the electroweak scale v as an energy anchor. This anchor is the single measured input to the prediction. Q_G = 4π is fixed by the GNS normalization as the ratio λ/m_a of two representation-independent parameters [14]. With Q_G fixed, the bridge identity Q_G m_a² = 1/2 determines m_a uniquely. The remaining kernel invariants (|Ω|, D, ρ, Δ, c₄) are all theorems about the computational system. The parameter count is: zero free parameters from the axioms, zero free parameters from the kernel, one measured input (v) to set the dimensional scale.

### 6.1 Continuum Limit of the Kernel Gauss Law

In CGM, mass is accumulated ancestry structure. Mass-energy density ρ measures how much traceable structure is present per unit volume. Gravitational potential Φ measures the concentration of this structure at a point. With Φ ≤ 0 when the reference at infinity sets Φ = 0, the acceleration field is g = −∇Φ. Approaching greater concentration lowers the action required to maintain relational traceability, causing test bodies to accelerate toward mass concentrations.

From the kernel Gauss law in Section 5.4, the continuum limit gives div g = −Q_G G ρ with G = G_kernel / E_CS². With g = −∇Φ this integrates to ∇²Φ = Q_G G ρ.

```text
∇² Φ = Q_G G ρ
```

Here Q_G = 4π is the quantum of gravity derived in Section 3, matching the standard Newtonian form. Curvature, in this framing, is the observable gradient of mass-energy density. Uniform density yields a flat geometry with a vanishing gradient despite the presence of gravity, whereas varying density produces curvature.

### 6.2 The Gravitoelectromagnetic Decomposition

The six degrees of freedom decompose the gravitational field into two sectors where the gravitoelectric field g = −∇Φ carries the three translational degrees of freedom and the gravitomagnetic field B_g = curl A_g carries the three rotational degrees of freedom. Together they satisfy the gravitoelectromagnetic system.

```text
∇ · g   = −Q_G G ρ
∇ × g   = −∂B_g / ∂τ
∇ · B_g = 0
∇ × B_g = −(Q_G G / c²) J + (1/c²)(∂g / ∂τ)
```

Here J is the mass-energy current and τ denotes the physical time parameter. Heaviside [2] wrote these equations in 1893 as the gravitational analog of Maxwell's equations, and they emerge rigorously from the weak-field limit of general relativity. The CGM derivation identifies the two sectors with the translational and rotational degrees of freedom forced by the closure conditions.

The decomposition also follows from the algebraic structure of displacement composition. Composing non-collinear displacements in a curved geometry yields a non-associative operation, which gyrogroup algebra corrects via the gyration automorphism. Accumulated gyration produces a circulation field in the continuous limit, manifesting as the gravitomagnetic vector potential A_g.

In the weak-field regime, the gravitoelectromagnetic system implies wave propagation with characteristic speed c. Taking the curl of the gravitomagnetic equation and substituting the remaining identities yields a wave equation with the characteristic speed fixed by the same constant c that appears in the source response normalization. The multimessenger event GW170817 bounds any difference between the gravitational and electromagnetic propagation speeds to below 3 × 10⁻¹⁵ of c [28], consistent with this prediction. Static density gradients extend across space without wave propagation, while perturbations propagate at c through the gravitomagnetic sector.

### 6.3 The Position-Dependent Coupling

The linear theory treats G as constant, which cannot hold self-consistently when the gravitational field is strong because mass-energy density modifies the geometry through which the field is sourced. Coupling must therefore depend on position. The gravitational potential ratio ψ = |Φ|/Φ_Planck measures field strength relative to the Planck scale. The symbol u is reserved for the radial wave function in the Regge-Wheeler analysis (Appendix E.2). In dimensionless units with r_g = GM/c², the coordinate s = r/r_g gives ψ(s) = GM/(rc²) in the Newtonian limit. The potential ratio ranges from 0 in the weak field to approximately 0.5 near compact-object horizons.

Coupling at a given point depends on how much ancestry structure has accumulated there. This dependence enters through a reference energy scale E_ref(ψ) that shifts with gravitational depth. Optical conjugacy requires the UV and IR energy conjugates to satisfy E_UV E_IR = E_CS v/(4π²), establishing the Planck scale and the electroweak scale as paired foci of the system [25, 26]. Optical conjugacy is derived in [25, 26] from the CGM stage structure and the energy ladder, not postulated independently. The energy ladder positions energy scales on a ruler with tick spacing Δ, such that n(E) = ln(E_CS/E)/(Δ ln 2). The Refractive depth gradient, derived from the aQPU kernel's 8-byte Z2 holonomy transport (Section 5), dictates that the accumulated depth scales as τ(ψ) = τ_G(1 − ψ). Premise 3 is verified to machine precision against the kernel's spectral accumulation but is not derived from the modal axioms alone. Its status is that of a kernel-verified empirical regularity that enters the E_ref derivation as a premise. A heuristic argument for the linear form: τ measures position on the energy ladder with tick spacing Δ and total length τ_G from v to E_CS. Gravitational redshift shifts E_ref from v toward E_CS by a fraction ψ of the total span, leaving τ_G(1−ψ) ticks remaining. The linearity follows from the first-order redshift (1−ψ) common to both CGM and GR in the weak-field limit.

Combining these three premises yields the reference energy as a function of gravitational depth.

```text
E_ref(ψ) = E_CS (v / E_CS)^(1−ψ)
```

At ψ = 0 corresponding to the weak field, E_ref = v which is the electroweak scale. At ψ = 1 corresponding to the Planck scale, E_ref = E_CS. The factor (1 − ψ) in the Refractive depth gradient reflects the first-order gravitational redshift common to CGM and GR in the weak-field limit. The E_ref formula is derived from three premises and does not depend on the exact redshift formula, while the exact CGM redshift follows from the metric. The reference energy is a ruler quantile representing the energy at position τ(ψ) on the ladder. On a logarithmic scale spanning approximately 17 decades, centroid and quantile differ substantially. The quantile is the correct object because the Refractive Depth measures position along the ladder rather than an average over it. The formal proof appears in Appendix E.

Substituting τ(ψ) and E_ref(ψ) into the coupling formula gives the position-dependent coupling.

```text
G(ψ) = G_kernel exp(−τ_G(1−ψ)) / E_ref(ψ)² 
      = G₀ exp(g₁ ψ)
```

Here G₀ = G_kernel exp(−τ_G)/v² is the weak-field coupling and g₁ = τ_G + 2η with η = ln(v/E_CS). Here v = 246.22 GeV denotes the Higgs vacuum expectation value in the standard electroweak normalization v = (√2 G_F)^{-1/2}, related to the Fermi constant. The sensitivity is ∂G/∂v = −2G/v, so the loop-level ambiguity in v at the MeV level contributes approximately 20 ppm to G, below the current experimental uncertainty. Numerically, g₁ = −0.6456.

Since d ln G / dψ = g₁ < 0, the coupling decreases with increasing ψ. As mass accumulates and the local potential deepens, E_ref(ψ) shifts from the electroweak scale toward the Planck scale, weakening G(ψ) where ψ is largest. At the electroweak anchor (ψ = 0), G = G₀. At the Planck anchor (ψ = 1), G ≈ 0.524 G₀.

Because gravity couples exclusively to the five bulk shells identified in Section 5.1, coherent survival across this sector produces an attenuation factor of exactly five powers of the closure ratio ρ per holonomy cycle. The three routes to the exponent 5 share the 6-bit kernel structure as a common axiom but verify the exponent through different mathematical objects (STF dimension, shell count, carrier trace polynomial), providing mathematical cross-checks rather than axiomatic independence. The Refractive Depth τ_G is the integral of this symmetric trace-free attenuation accumulated across Z2 holonomy cycles (8 bytes per cycle) between the Planck and electroweak anchors. Substituting the Refractive Depth gradient and E_ref(ψ) into the coupling formula yields G(ψ) = G₀ exp(g₁ψ). This shares the mathematical structure of Beer-Lambert transmission, though the physical mechanism is STF attenuation through the kernel shell structure rather than electromagnetic absorption. The transmission law identifies the vacuum as a polarizable medium with position-dependent gravitational permittivity. The STF attenuation governs how the coupling constant G is determined and how the gravitational signal propagates, while the Poisson equation sources from total mass-energy density including all stages.

```text
τ_G⁰ = |Ω| Δ ρ⁵ (1 − 4ρ Δ²)
```

Here |Ω| = 4096, Δ = 1 − ρ, and ρ is the closure ratio from Section 3. The factor (1 − 4ρΔ²) is the lowest-order symmetric correction from the four-stage depth structure. An additive correction reduces the residual further.

```text
δτ = |Ω| Δ ρ⁵ c₄ Δ⁴,    c₄ = −7/4
```

The full model is τ_G = τ_G⁰ + δτ. The constant c₄ = −7/4 is fixed by two independent routes (Appendix C.4). The Z2 involution of the holonomy cycle eliminates all odd-order corrections in Δ, enforcing exact symmetry at leading order. The c₄ Δ⁴ correction represents a soft breaking of this Z2 symmetry by the isotropic pressure component of the stress tensor, providing the monopole contribution to the mass-energy stress budget that the five STF components miss.

Numerical evaluation gives τ_G⁰ alone a 25 ppm offset in G relative to the reference measurement. Adding δτ with c₄ = −7/4 leaves a residual of 7.36 × 10⁻⁸ in τ, corresponding to a difference of 0.074 ppm between prediction and reference measurement. This agreement is far tighter than present experimental uncertainty on G, which is of order 10⁻⁵ relative in CODATA 2018 [13]. The comparison is a consistency check against the chosen reference value, distinct from a metrological verification at that precision. The experimental uncertainty on G in CODATA 2018 is approximately 22 ppm, orders of magnitude larger than the 0.074 ppm residual. The sub-ppm agreement therefore tests the internal consistency of the dimensionless formula rather than providing a metrological verification at that precision. A decisive test requires substantially improved G measurements or an independent observable that constrains the same τ_G structure. Because the prediction equals G_kernel exp(−τ_G)/v², a fractional change in τ maps directly to the same fractional change in G with opposite sign. The prediction is stable at the precision to which Δ and ρ are fixed, and the sub-ppm residual is free of fine-tuned cancellation among poorly determined inputs.

### 6.4 The Effective Metric and Einstein Equations

For a point mass, the potential satisfies dψ/ds = −exp(g₁ψ)/s². This equation has the exact solution

```text
ψ(s) = −(1/g₁) ln(1 − g₁/s)
```

which reduces to ψ = 1/s in the Newtonian limit g₁ → 0. The solution remains real and finite for all s > 1/g₁. The maximum potential attained is ψ_max = −1/g₁ ≈ 0.4996 at the minimum radius, which lies just inside the conventional Schwarzschild radius.

For extended density distributions ρ(r), the full coupled system comprises the potential ratio ψ, the field equation g = −dΦ/dr = G(ψ) M(r)/r², the mass equation dM/dr = Q_G ρ(r) r², and the coupling function G(ψ) from above. This system closes for any specified density profile.

Geometry responds to mass-energy density through the potential ratio ψ. The effective metric for static spherical configurations is

```text
ds² = −f dt² + f⁻¹ dr² + r² dΩ²
f(r) = 1 − 2ψ(r)
```

The horizon occurs where f = 0 at ψ = 1/2. For the exact point-mass solution, the horizon radius is s_h = 1.695 r_g, a 15.3% inward shift from the Schwarzschild radius. The photon sphere occurs at s_ph = 2.586 r_g compared to 3.0 in general relativity.

The position-dependent coupling modifies the Gauss law to

```text
∇ · [(G₀/G(x)) g] = −Q_G G₀ ρ
```

For a point mass, (G₀/G(ψ)) × 4π s² g = 4π at all radii, verified to relative precision 2.83 × 10⁻¹⁶. The modified flux is exactly conserved.

The Einstein tensor for the metric f = 1 − 2ψ satisfies the component identity G_rr = G_tt/f² (exact by construction for this metric, verified numerically to 4.4 × 10⁻¹⁶ as a computational consistency check) across all sampled radii. The position-dependent coupling introduces an effective anisotropic stress-energy in the exterior vacuum. The gradient of G(ψ) acts as a source term, producing a tangential pressure that structurally supports the coupling gradient. The modified Bianchi identity governs a continuous energy exchange between mass-energy density and the gravitational field.

```text
∇_μ T^μ_ν = −(∂_ν G / G) T^μ_μ
```

This exchange is negligible in the weak field (s > 100) and becomes significant near compact objects (s < 10), where the tangential pressure dominates the effective stress tensor.

The continuum limit admits a scalar-tensor representation in which ψ appears as a position-dependent coupling functional.

```text
S = (1/16πG₀) ∫ R exp(−g₁ψ) √(−g) d⁴x + ∫ L_m √(−g) d⁴x
```

Unlike Brans-Dicke constructions, ψ has no independent dynamical degree of freedom and is fixed algebraically by the closure structure through ψ = |Φ|/Φ_Planck. It is therefore a derived quantity, free of its own kinetic term and its own equation of motion. In the point-mass exterior, R = 0 and the modified Gauss law holds.

```text
div[exp(−g₁ψ)∇ψ] = 0
∇²ψ = g₁|∇ψ|²   (for g₁ ≠ 0)
```

The modified vacuum equation identifies exp(−g₁ψ) as an effective gravitational permittivity. The vacuum responds to the potential with position-dependent screening, weakening the effective coupling where the field is strongest. This is conceptually distinct from the Newtonian and GR formulations. The constraint is elliptic rather than hyperbolic, preventing ψ from propagating as an independent scalar wave. Energy conditions (null, weak, dominant) are satisfied for ψ ∈ [0, ½).

The general field equations for non-static, non-spherical sources follow from the scalar-tensor action of Appendix E.4 but require derivation of the dynamical consistency conditions for the algebraic constraint ψ = |Φ|/Φ_Planck under time-dependent sources. This derivation is outside the scope of the present work. The static spherically symmetric case developed here establishes the core coupling structure, metric form, and observational predictions that can be tested independently of the general equations.

### 6.5 Self-Energy and Mass Dressing

The self-energy of a CGM point mass equals exactly −Mc²/4, replacing the divergent integral of Newtonian gravity. The position-dependent coupling G(ψ) weakens where the gravitational field is strongest, providing a natural geometric regulator. The proof follows directly from the exterior ordinary differential equation. Refractive stress density u(r) = |g(r)|² / (8π G(ψ(r))) measures the positive cost of maintaining the field per unit volume, and rest-frame energy is the volume integral of this density. For the spherical exterior solution, the modified Gauss law identity gives |g| = exp(g₁ψ)/s². Evaluating the rest-frame energy requires computing the exterior integral.

```text
I = ∫_{s_h}^∞ exp(g₁ψ)/s² ds
```

By the ODE dψ/ds = −exp(g₁ψ)/s², the integrand equals −dψ/ds, causing the integral to evaluate to ψ(s_h) − ψ(∞) = 1/2. This holds for any g₁ because the ODE is satisfied by construction, and applying this identity to the self-energy yields the formal relation.

```text
E_self = −(1/2) M_obs ψ_max c²
```

At the horizon f = 1 − 2ψ = 0 fixes ψ_max = 1/2.

```text
E_self = −M_obs c²/4
```

Rest-frame energy equals +M_obs c²/4 to balance the self-energy locally, and self-consistent dressing gives the mass split.

```text
M_obs = M_bare + E_self/c² = M_bare − M_obs/4
M_obs = (4/5) M_bare
```

Observable mass is exactly 80 percent of bare mass, with 20 percent bound into gravitational self-energy. The fractional binding E_self/M_obs c² = −1/4 is independent of this mass split. Observable mass is what remains after the system pays the structural cost of coherence.

At the horizon, identity survives because coupling G/G₀ = 0.724 remains finite while individuality is blocked. Self-energy (−Mc²/4) is the finite residual energy stored when Egress is achieved but Ingress fails: the identity channel closes algebraically via Egress, but Ingress cannot propagate because the tortoise coordinate diverges and escape probability vanishes. Space converts to time at gravitational horizons because preserving operational memory consumes all available capacity for spatial extension, forcing spatial extension to resolve entirely into the temporal curvature of the causal sequence.

#### 6.5.1 Stage-Mass Decomposition

The self-energy theorem uses total bare mass M_bare at the field-equation level. Each CGM condition carries a mass-equivalent energy from the UV ladder, but the conditions have distinct categorical statuses. UNA and ONA are lemmas that generate the six degrees of freedom. BU is the depth-four closure proposition governing those degrees of freedom, with Egress and Ingress as dual spectral readings of the W₂ involution.

The bare mass budget reflects this structure. UV energy ratios from the stage actions yield mass fractions:

```text
f_UNA = E_UNA / (E_UNA + E_ONA + E_BU) ≈ 0.462
f_ONA = E_ONA / (E_UNA + E_ONA + E_BU) ≈ 0.513
f_closure = E_BU / (E_UNA + E_ONA + E_BU) ≈ 0.026

M_bare = M_UNA + M_ONA + M_closure
```

The six degrees of freedom arise from UNA and ONA, forming the gravitating sector that carries f_STF = f_UNA + f_ONA ≈ 0.974 of M_bare. At the stress-tensor level this sector splits into five symmetric trace-free components that carry the gravitational signal and one trace component that provides isotropic pressure without coupling attenuation. The trace component is distinct from the closure overhead; the trace arises from the 6-DOF stress decomposition, while M_closure is the energy cost of enforcing the depth-4 balance across all six degrees of freedom. The constraint governing motion costs less than the motion itself.

Within the gravitating sector, the UNA/ONA split is energy-weighted rather than proportional to a simple DOF headcount. Virial stage resolution assigns T to ONA and V to UNA. The check 2f_ONA − f_UNA ≈ 0.564 reflects that UV ladder ratios are bare-mass fractions while the virial condition operates on dressed quantities in a bound gravitational configuration. Gravitational dressing is uniform across all components, meaning M_obs_UNA = (4/5) M_UNA and similarly for ONA and closure, so the mass fractions are invariant under dressing.

Compact geometry places gravitational coupling on the same Delta ruler as electroweak masses. Stage-mass decomposition bridges the self-energy dressing M_obs = (4/5) M_bare to this particle mass ladder at compact-object potentials without altering the numerical kernel invariants.

### 6.6 Antimatter Gravitational Interaction

Antimatter corresponds to the involution 𝒞 that exchanges the conjugate 12-bit faces of the macro state and reverses the family order of the canonical word. Under this involution, the 8-byte Z2 holonomy closure invariants exhibit a precise parity split between the gravitoelectric and gravitomagnetic sectors.

The gravitoelectric sector is even under 𝒞. The displacement invariant D = 24 and the return-to-carrier-rest condition hold identically for standard and reversed word paths, while the coupling function G(ψ) depends only on the potential ratio ψ, which remains invariant under 𝒞. The face-swap involution preserves the mass observable across all 4096 states of the reachable manifold Ω, ensuring antimatter possesses positive gravitational mass identical to matter.

The gravitomagnetic sector is odd under 𝒞. The chirality register χ₆, defined as the 6-bit collapse of the conjugate face difference, transforms as χ₆ → χ₆ ⊕ 63 under the radial reflection that maps shell N to shell 6 − N. The signed observable H_spin = 3 − popcount(χ₆) measures the distance from the equatorial shell N = 3. Under the gravitomagnetic involution, H_spin changes sign. Exhaustive computational verification across the full 4096 states of Ω confirms H_spin(𝒞(s)) = −H_spin(s) for all states (`aqpu_gravity_analysis_6.py`). The 1280 equatorial states with H_spin = 0 are the fixed points of this sign reversal, leaving 2816 states where the sign flip is nontrivial. This odd parity dictates that the spin-gravity coupling for antimatter has the opposite sign to that of matter. The magnitude of the chiral correction scales as (4/75)ψ², derived from the constant anisotropy ratio ‖π‖²/Tr(σ)² = 2/75 across the bulk shells. At a neutron star surface where ψ ≈ 0.15, this correction reaches 0.12%. The same kernel invariant ‖π‖²/Tr(σ)² = 2/75 that sets the angular distribution of gravitational radiation also fixes the magnitude of this gravitomagnetic correction, unifying radiation structure and matter-antimatter distinction in a single combinatorial constant.

## 7. Observational Signatures
The nonlinear field theory derived in Section 6 produces distinct observational consequences across gravitational regimes. This section quantifies these signatures, moving from weak-field solar system tests to strong-field compact objects and constant-product falsification thresholds.

### 7.1 Weak-Field Regime
The CGM gravitational redshift follows from the metric coefficient f = 1 − 2ψ. A photon emitted at radius r and received at infinity is shifted by

```text
z_CGM = 1 / √(1 − 2ψ_CGM) − 1
z_GR  = 1 / √(1 − 2ψ_Newton) − 1
```

The CGM deviation arises because ψ_CGM < ψ_Newton at every finite radius. The position-dependent coupling G(ψ) weakens the accumulated potential relative to a constant-G theory, producing a smaller redshift for the same mass and radius. This is a direct gravitational signature of the decreasing coupling.

The weak-field limit of both formulas is z ≈ ψ, which is the first-order expansion common to CGM and GR. This first-order form, 1 − ψ, appears in the physical motivation for the Refractive depth gradient τ(ψ) = τ_G(1 − ψ) used in the E_ref derivation. The E_ref formula depends only on its three founding premises, independent of the exact redshift formula, while the exact CGM redshift follows directly from the metric via the equivalence principle.

Three equivalence principles follow from the CGM structure. The Weak Equivalence Principle holds because free-fall acceleration depends on the local field ψ alone, independent of the test body composition. The Einstein Equivalence Principle holds because the rotational-translational structure at each point guarantees local isotropy and frame equivalence, with the gravitational redshift matching the metric prediction. The Strong Equivalence Principle holds because G(ψ) depends on position alone, independent of the internal structure of the body. The Nordtvedt parameter η_N = 0, consistent with lunar laser ranging bounds |η_N| < 2.2 × 10⁻⁵ [29]. For the Earth-Moon system in the solar field, both bodies have ψ ≈ 10⁻⁹, so the self-potential contribution to ψ is negligible compared to the external potential. G(ψ) is therefore identical for both bodies at the same position to a precision far exceeding the lunar ranging bound.

The perturbative expansion of the exact point-mass solution in powers of 1/s gives

```text
ψ(s) = 1/s + a₂/s² + a₃/s³ + ⋯
a_n = g₁^(n−1) / n
```

The coefficients are a₁ = 1, a₂ = −0.3228, a₃ = 0.1389, a₄ = −0.0673, a₅ = 0.0347. From this expansion, γ = 1 exactly, consistent with the Cassini measurement γ = 1 ± 2.3 × 10⁻⁵ [29]. The leading deflection term is 4GM/c²b, identical to GR.

The coefficient a₂ = g₁/2 yields β = 1 − g₁/2 = 1.3228 at leading order. This value does not govern weak-field perihelion precession, and the standard PPN formula (2+2γ−β)/3 cannot be applied to CGM. The Parameterized Post-Newtonian framework assumes constant parameters and a metric whose expansion truncates at 1/s² order. CGM violates both conditions because the position-dependent coupling G(ψ) generates an infinite series of independent correction terms. Applying the constant-β formula with β = 1.3228 would predict a precession 11% below the GR value, which is excluded by observation. The resolution is that perihelion precession must be computed from geodesics in the full CGM metric f = 1 − 2ψ(s) rather than from a truncated expansion.

At weak field the exact potential ψ(s) = −(1/g₁) ln(1 − g₁/s) reduces to 1/s with corrections of order g₁/s. The leading correction to perihelion precession relative to GR is

```text
δφ_CGM / δφ_GR = 1 + g₁/(6 s_a) + O(s_a⁻²)
```

where s_a is the semi-latus rectum in units of r_g. For Mercury, s_a = 3.92 × 10⁷ and ψ ≈ 2.55 × 10⁻⁸, giving

```text
GR:  5.0185806280 × 10⁻⁷ rad/orbit = 42.9793 arcsec/century
CGM: 5.0185806143 × 10⁻⁷ rad/orbit = 42.9793 arcsec/century
CGM/GR = 0.9999999973  (0.003 ppm)
```

The CGM nonlinearity is controlled by the dimensionless potential ψ rather than by a constant PPN parameter. An effective β that characterizes the precession at a given gravitational depth is

```text
β_eff(ψ) = 1 + |g₁| ψ / 2
```

At Mercury's potential this evaluates to β_eff = 1 + 1.6 × 10⁻⁹, indistinguishable from the GR value of unity. The value β = 1.3228 extracted from the 1/s² coefficient reflects higher-order structure that activates only when ψ is appreciable. This depth-dependent activation is a structural property of CGM with no analogue in constant-parameter theories. The nonlinearity becomes observationally significant at ψ > 0.01, placing the observable deviation firmly in the compact-object regime. Table 1 summarizes the effective parameters and precession ratios across the full range of gravitational depth.

**Table 1.** Effective PPN parameters and perihelion precession by gravitational depth. β_eff is defined for comparison with PPN analyses and is not a fundamental parameter of the theory.

| ψ | β_eff | Precession/GR |
|--------|-------|---------------|
| 10⁻⁸ | 1.0000 | 1.0000 |
| 10⁻⁴ | 1.0000 | 1.0000 |
| 10⁻² | 1.0032 | 0.9989 |
| 5 × 10⁻² | 1.0161 | 0.9946 |
| 10⁻¹ | 1.0323 | 0.9892 |
| 2 × 10⁻¹ | 1.0646 | 0.9785 |
| 3 × 10⁻¹ | 1.0968 | 0.9677 |

The coefficient a₂ = g₁/2 yielding β = 1.3228 at leading order is a property of the 1/s² term in the exact potential expansion. It has no direct physical interpretation as a constant PPN parameter because CGM's position-dependent coupling generates an infinite series of independent correction terms. The physically meaningful object is the full metric f = 1 − 2ψ(s), from which all observable effects must be computed.

### 7.2 Gravitational Radiation
The canonical operational cycle traces a shell path with two symmetric excursions per half-cycle. Fourier decomposition of the shell displacement signal reveals the dominant spectral component at k = 2, with the shell modulation factor reaching 2C(6,2)/C(6,3) = 1.5 for the depth-four half-cycle. The full Z2 holonomy path gives |A₂| ≈ 1.25 as the dominant non-DC mode (`aqpu_gravity_analysis_6.py`). The next-strongest mode at k = 4 has |A₄| ≈ 1.02, about 82% of |A₂|. This mode is the discrete precursor of the hexadecapole (ℓ = 4) correction to gravitational radiation, subdominant to but structurally paired with the quadrupole channel. The quadrupolar character of gravitational radiation arises directly as the dominant spectral mode of the finite kernel's shell dynamics, independent of symmetry arguments or continuum limits. Two equal peaks per cycle identify the quadrupole structure at the level of the finite shell dynamics. In the continuous limit, a spin-2 field radiates predominantly through quadrupole emission, matching the kernel dominant spectral mode at k = 2.

Gravitational memory in CGM manifests as a permanently embedded gyration resulting from an interrupted Z2 holonomy cycle. The cycle requires depth-four egress closure (W₂) followed by completion of the second depth-four factor (W₂') in the 8-byte word F ∘ F. A gravitational wave interrupting the cycle after the first F-cycle but before F ∘ F restores carrier rest leaves an unresolved orientation correction permanently embedded in the system state. In the continuum limit, this residual gyration appears as the static metric displacement recorded by gravitational wave memory.

The scalar-tensor representation of the CGM action excludes any propagating scalar degree of freedom. The potential ratio ψ acts as an auxiliary field algebraically slaved to the metric through a Lagrange multiplier constraint ψ = |Φ|/Φ_Planck. The action takes the form

```text
S = (1/16π G₀) ∫ R exp(−g₁ψ) √(−g) d⁴x 
  − ∫ λ(ψ − |Φ|/Φ_Planck) √(−g) d⁴x 
  + ∫ L_m √(−g) d⁴x
```

Effective kinetic terms for ψ vanish upon imposing the slaving constraint, preventing independent wave propagation. In the vacuum exterior, the modified Gauss law is satisfied exactly. The identity s² exp(−g₁ψ) |dψ/ds| = 1 holds to machine precision across all radii. CGM predicts exactly two tensor polarization modes, consistent with LIGO and Virgo observations.

The position-dependent coupling modifies the orbital dynamics at all post-Newtonian orders. The leading correction to the orbital phase scales as δΦ/Φ_GR ≈ (5g₁/8)(v/c)². For GW150914 at peak orbital velocity v/c ≈ 0.4, this phase correction reaches −6.5%. Extracting this signal requires dedicated CGM waveform templates, as the correction is currently degenerate with mass parameters in standard analyses.

The quasinormal mode spectrum shifts due to the modified photon sphere and effective potential. Computing the tortoise coordinate r* reveals that the peak of the Regge-Wheeler axial effective potential shifts inward to s ≈ 2.84 r_g compared to the Schwarzschild value of 3.28 r_g. A Pöschl-Teller fit to the barrier shape yields an estimate that the fundamental ringdown frequency is about 12.5% above the exact general relativistic value, representing an approximation rather than an exact CGM QNM spectrum. A full Regge-Wheeler analysis on the CGM background would be required for precise bounds.

The modified Gauss law acts as a Klein-Gordon equation on the CGM metric. For a scalar perturbation, the radial equation in tortoise coordinates yields a Regge-Wheeler potential V_l(s). Matching the wavefunction and its first derivative across a sharp step in the metric yields a requirement that the scalar wave impedance Z = f k = ω remains constant across the step. Standard Fresnel equations for the electromagnetic field do not apply here; those equations describe vector waves crossing a refractive boundary. For scalar and gravitational perturbations, impedance continuity is an exact mathematical property of the Klein-Gordon equation on this specific metric. Consequently, all vacuum reflection comes from smooth tunneling through the Regge-Wheeler potential, and zero reflection occurs at sharp metric interfaces. Numerical integration across multiple frequencies and angular momenta confirms exact flux conservation (R + T = 1) in the vacuum exterior.

Gravitational wave strain is calibrated against the Hulse-Taylor binary pulsar. For the Hulse-Taylor system, ψ_orbital ≈ 2.1 × 10⁻⁶, giving G(ψ)/G₀ = 0.9999986. The resulting orbital period derivative differs from the GR prediction by 0.0003%, below current observational precision. Strong-field sources exhibit larger corrections. A neutron-star binary at 20 km orbital separation has ψ ≈ 0.10, yielding G/G₀ ≈ 0.935 and a 6.5% reduction in gravitational wave luminosity.

### 7.3 Strong-Field Compact Objects
The coupling function G(ψ) produces distinct observational consequences across a range of compact objects. The photon sphere yields an impact parameter b/r_g = 4.648, giving a shadow area 80% of the GR Schwarzschild prediction. Table 2 gives shadow angular diameters for two Event Horizon Telescope targets. CGM predictions fall 1.7 to 2.0 standard deviations below EHT measurements [31, 32]. At this precision both CGM and GR remain consistent with the data. Improved EHT observations at 1 μas or better would distinguish the models at the 3σ level. If next-generation EHT observations at ≤1 μas precision confirm shadow diameters consistent with GR Kerr and exclude CGM predictions at the 3σ level, the specific coupling function G(ψ) = G₀ exp(g₁ψ) with the kernel-derived g₁ would be falsified for the Schwarzschild sector. For the Kerr sector, the current spin correction is valid for a* < 0.3. A full Kerr-CGM metric for higher spin parameters is required before a quantitative falsification threshold can be established. The direction of the correction (shadow sizes below GR Kerr) is robust, but the magnitude at high spin remains approximate. Spin corrections use the wavefunction two-pass deficit method, with helix activation factor 0.5 and metric κ = 4.5, valid for spin parameter a* < 0.3. Both M87* and Sgr A* are estimated to have a* > 0.5 based on GR-based analyses. The CGM spin corrections in Table 2 should therefore be treated as order-of-magnitude estimates rather than precise predictions. A full Kerr-CGM metric derived from the position-dependent coupling G(ψ) applied to axisymmetric spacetimes is required for definitive comparison with EHT data at high spin.

**Table 2.** Shadow angular diameters (μas).

| Source | CGM (Schwarzschild) | CGM (with spin) | GR (Kerr) | EHT measurement |
|--------|---------------------|-----------------|-----------|-----------------|
| M87*   | 35.5               | 36.2            | 36.0      | 42.0 ± 3.0      |
| Sgr A* | 45.3               | 48.0            | 47.2      | 51.8 ± 2.3      |

Perihelion precession around compact objects provides an additional strong-field diagnostic. For an orbit at s = 10 r_g around a neutron star (ψ ≈ 0.1), the CGM precession is 1.1% below the GR prediction. For an orbit at s = 6 r_g around a stellar black hole (ψ ≈ 0.16), the reduction reaches 2.2%. These deviations scale as |g₁|ψ/2 and are accessible through precision timing of compact binary pulsars with orbital separations below 20 r_g.

A neutron star with M = 1.4 M☉ and R = 12 km has surface potential ψ_surface ≈ 0.153. The position-dependent coupling gives G/G₀ ≈ 0.906 at the surface. The core coupling remains near G₀ because the central potential is small for a polytropic equation of state. A self-consistent Tolman-Oppenheimer-Volkoff integration with G(ψ) confirms this. For a γ = 2 polytrope with central density 8 × 10¹⁷ kg/m³, the integration yields R ≈ 15.4 km and enclosed mass 1.25 M☉. The coupling increases from G/G₀ ≈ 0.906 at the surface to 1.000 at the center. The CGM gravitational redshift at the surface is z_CGM ≈ 0.200, which is 15% below the GR prediction of z_GR ≈ 0.235 for the same mass and radius. This redshift reduction is a direct signature of the position-dependent coupling and is testable with X-ray spectroscopy of neutron star surfaces.

Table 3 summarizes the coupling reduction across astrophysical objects. Earth and the Sun are in the linear regime with negligible corrections. White dwarfs show a 0.02% reduction. Neutron stars enter the strongly nonlinear regime: −10.5% at the Newtonian surface potential, −9.4% at the self-consistent CGM TOV surface (ψ ≈ 0.153). Stellar black holes at the horizon show a 27.6% reduction.

**Table 3.** Astrophysical coupling reduction. ψ values use the Newtonian estimate GM/(Rc²) unless noted.

| Object | ψ | G/G₀ | δG/G |
|--------|---|------|------|
| Earth surface | 7.0 × 10⁻¹⁰ | 1.000000 | −0.45 ppb |
| Sun surface | 2.1 × 10⁻⁶ | 0.999999 | −1.4 ppm |
| White dwarf (1 M☉) | 3.0 × 10⁻⁴ | 0.999809 | −0.019% |
| Neutron star (1.4 M☉, 12 km; Newtonian ψ) | 0.172 | 0.895 | −10.5% |
| Neutron star (1.4 M☉; CGM TOV ψ) | 0.153 | 0.906 | −9.4% |
| Stellar BH (10 M☉) | 0.501 | 0.724 | −27.6% |

Orbital period corrections scale as T/T_Newton = 1/√(G/G₀). For Earth at 1 AU, the correction is negligible. For a neutron star close orbit at s = 11.6, T/T_N = 1.027 and v/v_N = 0.974. For a black hole innermost stable circular orbit at s = 6.0, T/T_N = 1.052 and v/v_N = 0.951.

The CGM horizon at s_h ≈ 1.69 r_g (ψ = ½) yields surface gravity κ_CGM ≈ 1.01 κ_GR (temperature ~0.9% higher) but horizon area ~72% of the Schwarzschild value at r = 2 r_g. The estimated Hawking luminosity scales as κ⁴ times area, giving L_CGM/L_GR ≈ 0.74 (~26% lower than GR). This is an order-of-magnitude estimate from the exterior metric, and greybody factors on the CGM background would refine it.

Table 4 details the redshift comparison across the neutron star and compact object regime. The deviation grows with compactness because the potential gap ψ_Newton − ψ_CGM increases with depth. At solar potentials (ψ ~ 10⁻⁸) the two theories are indistinguishable. The redshift difference becomes observationally accessible at ψ > 0.1, placing it firmly in the neutron star regime.

**Table 4.** Gravitational redshift: CGM versus GR for a 1.4 M☉ source.

| R (km) | ψ_Newton | ψ_CGM | z_GR | z_CGM | δz/z_GR |
|--------|----------|-------|------|-------|---------|
| 15 | 0.138 | 0.122 | 0.161 | 0.147 | −8.7% |
| 12 | 0.172 | 0.153 | 0.235 | 0.200 | −15% |
| 10 | 0.207 | 0.183 | 0.306 | 0.254 | −17% |
| 8 | 0.258 | 0.227 | 0.439 | 0.342 | −22% |

Existing neutron star redshift measurements (e.g., EXO 0748-676 at z = 0.35, measured via Fe absorption lines) probe systems with higher compactness than the 1.4 M☉ models in Table 4. A systematic comparison of CGM redshift predictions against NICER and XMM-Newton observations is a priority for future work.

### 7.4 Coupling and Constant Signatures
Electromagnetic and gravitational couplings share the aperture geometry, producing a testable relationship between them. With ζ = 8/(m_a √3) = 16√(2π/3) ≈ 23.155 and α₀ = δ_BU⁴/m_a, the product

```text
α₀ ζ = ρ⁴ / (π √3) = 0.169025920321
```

cancels m_a entirely. Independent measurements of α and G can therefore falsify CGM should their product violate ρ⁴/(π√3). The laboratory fine-structure constant α_CODATA differs from α₀ by +319 ppm. The transport-corrected value from [24, 27] matches α_CODATA to sub-ppb accuracy. Gravity and the kernel invariant use α₀ and do not incorporate that correction chain, so the product test applies to α₀ specifically. Given α_CODATA, the product α ζ = ρ⁴/(π√3) predicts ζ ≈ 23.163. Any independent constraint on G or ζ that disagrees with this propagation, after explicit identification of which α definition is used, falsifies the stated layer of the framework.

The shell opacity structure modulates the effective electromagnetic coupling across cosmological depth. Mapped to redshift via the energy ladder, this produces an oscillation in the fine-structure constant with period Δ_z ≈ 0.0143 in ln(1+z) and peak-to-peak fractional amplitude approximately 4.8 × 10⁻⁴. Seven sub-cycles per main period arise from the shell structure, giving a sub-cycle period of approximately 0.0020. A survey spanning at least one full period in ln(1+z) and detecting no oscillation at 3σ confidence with the stated period would falsify this prediction at the stated amplitude scale.

Different experimental methods for measuring G yield systematically different values [13]. CGM predicts that systematic offsets among methods correlate with the effective shell weighting of the experimental configuration. The shell structure distributes coupling strength non-uniformly across the seven shells according to the binomial weight C(6,k)/64. Experiments that preferentially activate different geometric configurations will systematically measure different effective values of G. The per-family Refractive Depth variance is exactly zero across all four families, confirming that the variation emerges between experimental geometries rather than within them. Deriving the method-to-shell projection map for each experimental type would convert this qualitative prediction into a quantitative one. This concerns path-dependence rather than time-dependence, and supernova conditions on time variation of G remain consistent with CGM.

The CGM correction to gravitational wave luminosity is consistent with Hulse-Taylor observations. Strong-field corrections reach several percent for neutron-star mergers.

Dual-pole symmetry requires the next correction to the fine-structure constant prediction to be negative. A positive O(δ_BU⁶) correction at the Thomson limit would falsify the geometric identification.

## 8. Implications and Conclusion
This manuscript derived the gravitational field equation, the coupling constant structure, the spin-2 character of gravitational interaction, and the nonlinear extension to the Einstein equations from a single requirement: the Preservation of Ancestry.

Three results anchor the derivation. First, the shell displacement invariant D = 24, verified exhaustively across all 64 mass configurations, establishes the discrete Gauss law and fixes G_kernel = π/6. Second, the depth-four / 8-byte holonomy distinction proves that the gravitational cycle requires two depth-four passes (F then F), supplying the factor 2 in 8π = 2 Q_G and the spin-2 angular momentum structure. Third, the Refractive Depth model

```text
τ_G = |Ω| Δ ρ⁵ [(1 − 4ρΔ²) + c₄Δ⁴]  with c₄ = −7/4
```

yields G_pred within 0.074 ppm of the reference measurement, with the exponent 5 confirmed by three independent arguments.

As established in Section 4, the Virial condition ties Refractive Depth to the mass-energy stress required to bind kinetic individuality into rest mass, with gravitationally bound systems carrying negative total energy while net displacement per cycle vanishes exactly.

The nonlinear extension is complete. The position-dependent coupling G(ψ) = G₀ exp(g₁ψ) derives from three premises. The exact point-mass solution ψ(s) = −(1/g₁) ln(1 − g₁/s) closes analytically. The effective metric f = 1 − 2ψ satisfies the Einstein field equations and admits a scalar-tensor representation in which ψ is algebraically slaved to the closure structure. Equivalence principles follow from the CGM structure, while PPN parameters γ = 1 and β = 1.3228 satisfy Cassini and lunar-ranging conditions. Observational signatures include a 15.3% horizon shift, a 10% shadow diameter reduction, a 9.4% coupling reduction at neutron star surfaces, and Hawking luminosity ~26% below GR.

Quantum gravity programs face a structural obstacle in perturbative quantization. Gravity corresponds to the 8-byte Z2 holonomy closure of the operational cycle (F ∘ F), making the quantization of h_μν as an independent excitation circular. The kernel construction encodes this holonomy in carrier recovery to complement-horizon rest. The balance invariants Q_G, δ_BU, and m_a set the aperture structure, with Q_G = 4π as the geometric quantum. The framework is quantum-mechanical at foundation, independent of a graviton propagator. The relevant question becomes how this holonomy manifests in the regime where both quantum and gravitational effects are significant.

The derivation treats three-dimensional spatial structure as logically prior to four-dimensional spacetime. Time corresponds to the ordering of modal depth required to pass through CS, Unity-Non-Absolute (UNA), Opposition-Non-Absolute (ONA), and Balance Universal (BU) in sequence. The fourth coordinate remains a computational device in the continuum packaging rather than an ontological primitive. The algebraic kernel forces the 3+1 split through the SO(3)/SU(2) shadow projection. Four-dimensional tensors remain a valid packaging of the continuum limit, encoding a distinction between spatial and temporal domains that the kernel makes algebraically explicit.

Coupling constants emerge from the BU invariants Q_G, δ_BU, and m_a as derived parameters. The gravitational coupling follows from kernel invariants, aperture geometry, and one energy anchor. The electromagnetic coupling follows from the same aperture geometry at a different depth. Their product is fixed by the closure ratio alone. Should this pattern extend, other couplings may yield to similar constructions. The shell structure produces both a monopole (1 component) and quadrupole (5 components) decomposition, suggesting that the kernel contains sufficient spectral richness to accommodate the full Standard Model.

The UV-IR interface density ρ_MU scales in a gravitational field as ρ_MU(ψ) = ρ_MU(0) (v/E_ref(ψ))². Near a black hole horizon, this depletes by a factor of order 10⁻⁶, reflecting the extreme redshift of the UV-IR conjugate pair. The product ρ_MU E_ref² is preserved across all ψ. At the Sun's surface, the depletion is 0.016%. At a neutron star surface, it reaches 99.9998%.

The kernel structure provides an intrinsic gravitational clock. The period of the 8-byte Z2 holonomy cycle is T_Z2 = (6/π)GM/c³. Multiplying this period by the CGM surface gravity κ_CGM yields an advance rate per cycle of κ_CGM × T_Z2. Evaluating this product at flat space (ψ = 0) yields exactly 3c/(2π), tying the kernel displacement invariant D = 24 directly to the speed of light. At the horizon (ψ = 1/2), this conversion rate vanishes entirely. The standard light cone is the geometric projection of the commit phase alone; causal influence in the full four-phase cycle requires all four phases to be operational.

The framework derives a refractive index for the vacuum, n = 1/sqrt(1 - 2ψ), originating from the two Z2 gyrophase sheets. For scalar and gravitational perturbations, the wave impedance is constant across any sharp metric step, resulting in zero interface reflection. All vacuum reflection is strictly a smooth tunneling phenomenon governed by the Regge-Wheeler potential.

The companion compact geometry analysis 'docs\Findings\Analysis_Compact_Geometry.md' derives electroweak, lepton, and quark mass coordinates from the same carrier-trace polynomial on the Delta ruler, with the W/Z ratio recovering Delta to 8.34 x 10^-10. Active research leads include the CKM closure ansatz and full SU(3) colour derivation, pending predeclared null-model audits as specified in that analysis.

The physical structure derived throughout this manuscript admits a specific interpretation regarding the nature of intelligence as the capacity for coherent self-reference. The physical manifestation of this capacity is the strict causal sequence required to maintain the memory of an origin without erasing structural distinctions. The four foundational conditions map directly to the operational stages of this self-reference: Common Source provides governance traceability, Unity Non-Absolute provides information variety, Opposition Non-Absolute provides inference accountability, and Balance Universal provides integrity. In this framework gravity is not merely a metric curvature but rather the geometry of memory refusing to disperse. The gravitational field equations derived here describe the exact physical mechanism through which coherent self-reference preserves its own causal ancestry.

---
## Appendix A: Formal Modal Logic
The Common Governance Model is formalized as a propositional modal logic with two primitive modal operators [L] and [R] representing recursive operational transitions, evaluated over Kripke frames F = (W, R_L, R_R, V) with serial accessibility relations and nonempty valuation V(S).

**Primitive symbols:**

| Symbol | Description |
|--------|-------------|
| S | A propositional constant: the horizon constant |
| ¬ | Logical connective: negation |
| → | Logical connective: material implication |
| [L] | Modal operator: left transition |
| [R] | Modal operator: right transition |

**Defined symbols:**

| Symbol | Definition |
|--------|------------|
| φ ∧ ψ | ¬(φ → ¬ψ) |
| φ ∨ ψ | ¬φ → ψ |
| φ ↔ ψ | (φ → ψ) ∧ (ψ → φ) |
| ⟨L⟩φ | ¬[L]¬φ |
| ⟨R⟩φ | ¬[R]¬φ |
| □φ | [L]φ ∧ [R]φ |
| ◇φ | ⟨L⟩φ ∨ ⟨R⟩φ |

[L]φ means φ holds after a left transition. [R]φ means φ holds after a right transition. □φ means φ holds after both transitions. Modal depth refers to the nesting level of modal operators.

**Core definitions:**

| Concept | Formula | Description |
|---------|---------|-------------|
| Unity (U) | [L]S ↔ [R]S | Left and right transitions yield equivalent results at the horizon |
| Two-step Equality (E) | [L][R]S ↔ [R][L]S | Depth-two compositions commute at the horizon |
| Opposition (O) | [L][R]S ↔ ¬[R][L]S | Depth-two compositions yield contradictory results at the horizon |
| Balance (B) | [L][R][L][R]S ↔ [R][L][R][L]S | Depth-four compositions commute at the horizon |

Absoluteness: Abs(φ) = □φ (φ is invariant under both transitions). NonAbs(φ) = ¬□φ.

**The five foundational conditions:**

| Constraint | Name | Type | Formula |
|------------|------|------|---------|
| CS | Common Source | Assumption | S → ([R]S ↔ S ∧ ¬([L]S ↔ S)) |
| UNA | Unity-Non-Absolute | Lemma | S → ¬□E |
| ONA | Opposition-Non-Absolute | Lemma | S → ¬□¬E |
| BU-Egress | Balance Universal (egress) | Proposition | S → □B |
| BU-Ingress | Balance Universal (ingress) | Proposition | S → (□B → ([R]S ↔ S ∧ ¬([L]S ↔ S) ∧ ¬□E ∧ ¬□¬E)) |

The CS axiom establishes fundamental chirality. Right transitions preserve S while left transitions alter it. Unity-Non-Absolute (UNA) prevents homogeneous collapse by ensuring non-commutativity is contingent at depth two. Opposition-Non-Absolute (ONA) prevents irreconcilable contradiction by ensuring opposition is contingent at depth two. Balance Universal egress (BU-Egress) achieves commutative closure at depth four. Balance Universal ingress (BU-Ingress) guarantees that the balanced state contains sufficient information to reconstruct all prior conditions.

In the core modal system with Kripke semantics, all five conditions are logically independent. Each admits counterexample frames falsifying it while preserving the others. Consistency is verified via a three-world Kripke frame satisfying all five simultaneously. In the operational regime with continuous flows, reachability from S, and simple Lie closure, UNA and ONA follow from CS. The two-layer structure (modal axioms plus operational requirements) prevents circular reasoning.

## Appendix B: Kernel Topology and Combinatorics
### B.1 The Reachable Manifold
The kernel is a finite algebraic system with 4096 reachable states organized into seven shells. Shell k has ab-distance 2k for k = 0 through 6, with shell 0 at the equality horizon and shell 6 at the complement horizon. Shell populations follow |shell_k| = C(6,k) × 64.

### B.2 The K4 Operator Algebra
The depth-4 canonical half-word W₂ maps shell s to 6 − s, producing a pole swap. Its square is the identity on all 4096 states, with eigenspace dimensions dim(+1) = dim(−1) = 2048. The full canonical word F = W₂ ∘ W₂' composes two such involutions through the Klein four-group (K4) algebra {id, W₂, W₂', F}. Gate F preserves shell while acting as a two-state flip on the positional coordinate within each shell. The holonomy cycle requires F ∘ F = id.

### B.3 Carrier Trace Theorems
The shell transition operator M_q for byte weight q acts on the seven-dimensional shell space. The transition probability from shell w to shell t under weight q is

```text
P(w → t | q) = C(w,t) C(6−w, q−t) / C(6,q)
```

For even weights q = 2k, the Chu-Vandermonde identity yields the diagonal sum

```text
Tr(M_{2k}) = [Σ_w C(w,k) C(6−w,k)] / C(6,2k) 
            = C(7, 2k+1) / C(6,2k) 
            = 7/(2k+1)
```

For odd weights, the byte swap alters shell parity, forcing Tr(M_q) = 0. The return trace C(q) = Tr(M_q²) is obtained from the Krawtchouk spectral decomposition. The shell transition matrices are μ-symmetric with respect to the binomial measure μ(w) = C(6,w)/64. The Krawtchouk polynomials K_w(k) form an orthogonal eigenbasis, and the eigenvalues λ_k are computed exactly as rationals. Summing λ_k² yields the odd-weight traces. These values are verified by three independent computational routes: two-hop transition product, matrix squaring, and spectral eigenvalue summation.

### B.4 Chirality Inversion
Each depth-4 half-word fully inverts chirality: q(W₂) = q(W₂') = 63 for all 64 micro-reference configurations m. The full canonical word composes two such inversions, yielding q(F) = 63 ⊕ 63 = 0. Gate F preserves chirality while acting on the positional state, verified exhaustively across all micro-references.

### B.5 Holographic Identity
The boundary horizons H with 64 states each and the full manifold Ω with 4096 states satisfy |H|² = |Ω|. This follows from the self-dual [12,6,2] code structure of the kernel [16]. The entropy relation ln|Ω| = 2 ln|H| is driven by the two-pass holonomy identified in Section 5.

## Appendix C: Refractive Depth Construction
### C.1 Exact Per-Cycle Depth
The exact per-cycle Refractive Depth is derived from the binomial-weighted holonomy transport over the 64 micro-references. For a micro-reference at popcount k, all four bulk steps land on shell k, contributing 4 × C(6,k)/64 per step. Weighting by the ergodic measure and summing gives

```text
τ_cycle / Δ = 4 Σ_k C(6,k)³ / (64 Σ_k C(6,k)²)
```

With Σ_{k=1}^{5} C(6,k)³ = 15182 and Σ_{k=0}^{6} C(6,k)² = 924 = C(12,6), this evaluates to 60728/59136 = 7591/7392.

### C.2 The K Factor
The transport measure τ_cycle/Δ = 7591/7392 and the anisotropy measure 5/99 (from the trace-free magnitude with the same binomial weighting) differ by the exact rational factor K = 22773/1120. This factor decomposes as K = (3 × 7591)/1120, where 1120 = 224 × 5 and 224 = 7392/33. The numerator is three times the half-cube-sum. The denominator derives from binomial moment identities.

### C.3 Series Expansion
Expanding the closed form τ_G = |Ω| Δ (1−Δ)⁵ (1−4(1−Δ)Δ²) as a polynomial in Δ produces a finite series through degree nine with coefficients c_n/|Ω| = [0, 1, −5, 6, 14, −55, 79, −60, 24, −4]. The series converges to the closed form exactly at machine precision.

### C.4 The c₄ Correction
The additive correction δτ = |Ω| Δ ρ⁵ c₄ Δ⁴ with c₄ = −7/4 is fixed by two independent routes. Route A gives c₄ = −(1 + Tr(σ_iso)) = −7/4 from the isotropic stress trace. Route B gives c₄ = q_W from the closure charge on gyroscopic edge increments, yielding the same value. The two routes are mathematically independent: Route A derives c₄ from the second-moment structure of the payload bit distribution, while Route B derives it from the edge-increment structure of the K4 gate composition. They share the kernel as a common framework but use different theorems about different objects. Including δτ reduces the τ residual from 2.46 × 10⁻⁵ to 7.36 × 10⁻⁸.

### C.5 Cycle Count
The number of Z2 holonomy cycles (8 bytes each) is N_cycles = |Ω| ρ⁵ (f_K4 + c₄ Δ⁴) / (τ_cycle/Δ), where f_K4 = 1 − 4ρΔ². This evaluates to N_cycles ≈ 3586.5. The product N_cycles × τ_cycle = τ_G confirms exact agreement with the closed form.

### C.6 Per-Family Uniformity
The per-family depth-4 Refractive Depth is identical across all four family phases, with τ_word = 0.009408891 and zero variance. This uniformity supports the equal-weight assignment in the f_K4 correction factor.

## Appendix D: Translational Payload Stress
### D.1 Definition
The translational payload stress σ is computed from the translational payload bits. With the translational activation vector v = (b₆, b₅, b₄) and components in {0,1}, the stress is the centered second moment

```text
σ^{ij} = ⟨v^i v^j⟩ − ⟨v^i⟩⟨v^j⟩
```

where the averages are taken over the micro-reference ensemble within a cell.

### D.2 Decomposition
The tensor decomposes into isotropic and trace-free parts:

```text
σ^{ij} = p δ^{ij} + π^{ij}
```

where p = (1/3) Tr(σ) and π^{ij} is symmetric and trace-free. The trace component represents isotropic pressure. The five independent components of π form the trace-free sector corresponding to the ℓ = 2 representation of SO(3).

### D.3 Shell-Conditioned Values
Conditioning on popcount w, the trace evaluates to Tr(σ(w)) = w(6−w)/12. The anisotropy ratio ‖π‖² / Tr(σ)² is constant across all bulk shells. Over the uniform ensemble, the unconditional trace is Tr(σ_iso) = 3/4, decomposing as E[Tr(σ|w)] + 3 Var(E[v^i|w]) = 5/8 + 3/24 = 3/4.

### D.4 Nariai Bound
The interior-shell anisotropy ratio equals √6/9 ≈ 0.2722, matching the Nariai ultracold mass bound for stable extremal compact objects [23]. The significance of this match is discussed in Section 5. A dynamical derivation linking the two lies beyond the scope of this manuscript.

## Appendix E: Nonlinear Extension Details
### E.1 E_ref(ψ) Proof
Define L(E) = ln(E_CS/E) as the position on the energy ladder. At ψ = 0, L(v) = |η| and τ(0) = τ_G. At general ψ, τ(ψ) = τ_G(1−ψ) by Premise 3. The Refractive Depth per unit ladder length is α = τ_G/|η|. Setting τ(ψ) = α L(ψ) gives L(ψ) = |η|(1−ψ), hence

```text
E_ref(ψ) = E_CS exp(−|η|(1−ψ)) 
          = E_CS (v/E_CS)^(1−ψ)
```

Verification: E_ref(0) = v and E_ref(1) = E_CS, both matching the required endpoints.

### E.2 Einstein Tensor Components
For the metric f = 1 − 2ψ(s) with s = r/r_g, the Einstein tensor components are

```text
G_tt = −2f(ψ + s ψ′) / s²
G_rr = −2(ψ + s ψ′) / (s² f)
G_θθ = (s/2f) [f″ + f′/s − f′²/(2f)]
```

The component identity G_rr = G_tt/f² is verified in Section 6.4.

### E.3 Modified Bianchi Identity
The modified Bianchi identity ∇_μ T^μ_ν = −(∂_ν G / G) T^μ_μ is verified numerically. The exchange magnitude is computed at each sampled radius, confirming the behavior described in Section 6.4.

### E.4 Scalar-Tensor Representation
The continuum limit admits the following scalar-tensor representation:

```text
S = (1/16πG₀) ∫ R exp(−g₁ψ) √(−g) d⁴x
  − ∫ λ(ψ − |Φ|/Φ_Planck) √(−g) d⁴x
  + ∫ L_m √(−g) d⁴x
```

Here ψ is a position-dependent coupling functional constrained algebraically to the metric through the Lagrange multiplier λ. Unlike Brans-Dicke constructions, ψ is free of its own kinetic term and its own equation of motion. The constraint enforces ψ = |Φ|/Φ_Planck at the action level, eliminating any propagating scalar degree of freedom. In the point-mass exterior, R = 0 and div[exp(−g₁ψ)∇ψ] = 0 (Section 6.4). The field ψ remains slaved to the metric and does not propagate independently.

### E.5 PPN Derivation
The perturbative expansion of ψ(s) = −(1/g₁) ln(1 − g₁/s) in powers of 1/s gives

```text
ψ(s) = 1/s + a₂/s² + a₃/s³ + ⋯
a_n = g₁^(n−1) / n
```

The coefficients are a₁ = 1, a₂ = −0.3228, a₃ = 0.1389, a₄ = −0.0673, a₅ = 0.0347. The leading deflection term is 4GM/c²b, identical to GR. The parameter β at leading order follows as β = 1 − g₁/2 (Section 7.1).

### E.6 Neutron Star TOV Integration
The self-consistent TOV equation with G(ψ) is

```text
dP/dr = −G(ψ) ρ m/r² × [(1 + P/ρ)(1 + 4πr³P/m)] / [1 − 2G(ψ)m/(rc²)]
```

with dm/dr = 4πr²ρ and ψ solved self-consistently from ψ = G(ψ)m/(rc²). For a γ = 2 polytrope with central density 8 × 10¹⁷ kg/m³, the integration yields R ≈ 15.4 km, M ≈ 1.25 M☉, and ψ_surface ≈ 0.153.

### E.7 Gravitational Wave Strain
For a binary with total mass M_total, chirp mass M_chirp, and orbital separation a, the strain ratio h_CGM/h_GR = G(ψ)/G₀ where ψ = GM_total/(a c²). For the Hulse-Taylor binary: ψ = 2.14 × 10⁻⁶, ratio = 0.9999986, difference = 0.0003%. For NS-NS at 20 km: ψ = 0.103, ratio = 0.935, difference = 6.5%.

### E.8 Self-Energy Theorem
For the metric f = 1 − 2ψ(s) with s = r/r_g, the self-energy of a point mass equals E_self = −(1/2) M ψ_max c². The horizon condition f = 0 requires ψ_max = ½. Therefore E_self = −Mc²/4.

The operational rest-frame energy equals (M_obs c²/2) I, where I = ∫_{s_h}^∞ exp(g₁ψ)/s² ds. From the ODE dψ/ds = −exp(g₁ψ)/s², the integrand equals −dψ/ds. Thus I = ψ(s_h) − ψ(∞) = ½ for any g₁. The rest-frame energy equals +M_obs c²/4, balancing E_self = −M_obs c²/4 locally. Self-consistent dressing gives M_obs/M_bare = 4/5, where M_bare includes contributions from all stages (Section 6.5.1).

## Appendix F: References
[1] Newton, I. (1687). Philosophiae Naturalis Principia Mathematica. London: Royal Society.
[2] Heaviside, O. (1893). A gravitational and electromagnetic analogy. The Electrician, 31, 281-282 and 359.
[3] Nordström, G. (1913). Zur Theorie der Gravitation vom Standpunkt des Relativitätsprinzips. Annalen der Physik, 347(13), 533-554.
[4] Einstein, A. (1915). Die Feldgleichungen der Gravitation. Sitzungsberichte der Preussischen Akademie der Wissenschaften zu Berlin, 844-847.
[5] Fierz, M. and Pauli, W. (1939). On relativistic wave equations for particles of arbitrary spin in an electromagnetic field. Proceedings of the Royal Society of London A, 173(953), 211-232.
[6] Lovelock, D. (1971). The Einstein tensor and its generalizations. Journal of Mathematical Physics, 12(3), 498-501.
[7] Bekenstein, J. D. and Milgrom, M. (1984). Does the missing mass problem signal the breakdown of Newtonian gravity? Astrophysical Journal, 286, 7-14.
[8] Ungar, A. A. (2008). Analytic Hyperbolic Geometry and Albert Einstein's Special Theory of Relativity (2nd ed.). Singapore: World Scientific.
[9] Bruck, R. H. (1958). A Survey of Binary Systems. Berlin: Springer-Verlag.
[10] Hall, B. C. (2015). Lie Groups, Lie Algebras, and Representations (2nd ed.). New York: Springer.
[11] Morel, L., Yao, Z., Clade, P. and Guellati-Khelifa, S. (2020). Determination of the fine-structure constant with an accuracy of 81 parts per trillion. Nature, 588, 61-65.
[12] Christodoulou, D. (1991). Nonlinear nature of gravitation and gravitational-wave memory. Physical Review Letters, 67(12), 1486-1489.
[13] Tiesinga, E., Mohr, P. J., Newell, D. B. and Taylor, B. N. (2021). CODATA recommended values of the fundamental physical constants: 2018. Reviews of Modern Physics, 93(2), 025010.
[14] Korompilias, B. (2025). Common Governance Model: Mathematical Physics Framework. Zenodo. DOI: 10.5281/zenodo.17521384.
[15] Korompilias, B. (2025). Formal proof of three-dimensional necessity and six degrees of freedom in the Common Governance Model. Companion analysis (docs/Findings/Analysis_3D_6DOF_Proof.md).
[16] Korompilias, B. (2025). Compact geometry: Spectral algebra of the electroweak mass spectrum. Companion analysis (docs/Findings/Analysis_Compact_Geometry.md).
[17] Korompilias, B. (2025). Gyroscopic byte formalism: The 6-bit runtime and depth-4 closure. Companion specification (docs/Gyroscopic_Computational_Theory/Gyroscopic_ASI_Specs.md).
[18] Mashhoon, B. (2003). Gravitoelectromagnetism: A Brief Review. In The Measurement of Gravitomagnetism: A Challenging Enterprise, 31-42. New York: Nova Science.
[19] Mashhoon, B. (2008). Gravitoelectromagnetism. In Reference Frames, 29-39. Berlin: Springer.
[20] Arnowitt, R., Deser, S. and Misner, C. W. (1962). The dynamics of general relativity. In Gravitation: An Introduction to Current Research, 227-265. New York: Wiley.
[21] Poincaré, H. (1912). L'espace et le temps. Scientia, 12, 159-170.
[22] Pais, A. (1982). 'Subtle is the Lord...': The Science and the Life of Albert Einstein. Oxford: Oxford University Press.
[23] Chen, C., Li, B. and Wang, A. (2024). Mass bounds of compact objects in gravity theories. Physical Review D, 109, 084025.
[24] Korompilias, B. (2025). The fine-structure constant from geometric first principles. Companion analysis (docs/Findings/Analysis_Fine_Structure.md).
[25] Korompilias, B. (2025). Energy scale structure in the Common Governance Model. Companion analysis (docs/Findings/Analysis_Energy_Scales.md).
[26] Korompilias, B. (2025). CGM units from observational geometry. Companion analysis (docs/Findings/Analysis_CGM_Units.md).
[27] Korompilias, B. (2025). CGM constants and aperture geometry. Companion analysis (docs/Findings/Analysis_CGM_Constants.md).
[28] Abbott, B. P. et al. (2017). GW170817: Observation of gravitational waves from a binary neutron star inspiral. Physical Review Letters, 119(16), 161101.
[29] Bertotti, B., Iess, L. and Tortora, P. (2003). A test of general relativity using radio links with the Cassini spacecraft. Nature, 425, 374-376.
[30] Weisberg, J. M. and Taylor, J. H. (2005). The relativistic binary pulsar B1913+16: Thirty years of observations and analysis. In Binary Radio Pulsars, 25-32. San Francisco: Astronomical Society of the Pacific.
[31] The Event Horizon Telescope Collaboration (2019). First M87 Event Horizon Telescope results. I. The shadow of the supermassive black hole. Astrophysical Journal Letters, 875(1), L1.
[32] The Event Horizon Telescope Collaboration (2022). First Sagittarius A* Event Horizon Telescope results. I. The shadow of the supermassive black hole in the center of the Milky Way. Astrophysical Journal Letters, 930(2), L12.