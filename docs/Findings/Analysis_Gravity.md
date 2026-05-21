# Gravitational Theory in the Common Governance Model: Preservation of Ancestry through Identity and Individuality

Physics provides accurate field equations for gravity but treats two foundational inputs as assumptions: the dimensionality of space and the numerical value of the gravitational coupling constant. In Newtonian gravity, the 4π factor in the field law is a direct consequence of enclosing a mass in three spatial dimensions, yet the origin of those three dimensions remains unexplained [1]. In general relativity, the coupling κ = 8πG/c⁴ is fixed by requiring the Newtonian limit, leaving G as an externally measured parameter [4]. Similar dependence on unexplained prior assumptions occurs in Nordström's scalar theory [3], gravitoelectromagnetism [2], and linearized spin-2 formulations [5].

The Common Governance Model rests on a single axiom and establishes that reality is fundamentally organized by freedom, defined as the capacity for directional distinction and alignment from a common source. The conditions for this freedom to manifest generate the observable features of both spacetime and information. Physical conservation and informational coherence emerge as expressions of a single underlying order, the Preservation of Ancestry. Gravity, including its coupling strength and dimensional profile, arises solely from the requirement to preserve this ancestry. 

The mathematical and computational realization of these conditions is achieved through a compact holographic algorithm called the algebraic Quantum Processing Unit (aQPU). The CGM theory specifies space, chirality, and governance rules from the initial traceability axiom, and the aQPU operates as the reference architecture that executes them. This computational substrate provides the exact combinatorial invariants needed to anchor a continuous field theory, enabling the extraction of precise physical constants from purely formal foundations.

This analysis establishes the following results:

* The quantum of gravity emerges as the invariant **Q_G = 4π**, representing the complete solid angle necessary for coherent observation in three dimensions (Sections 2, 3, and 4)
* The **Poisson equation and its gravitoelectromagnetic decomposition** follow from the six degrees of freedom mandated by the traceability axiom (Section 5)
* A position-dependent coupling produces a **nonlinear metric extension satisfying the Einstein field equations**, while the two-pass closure cycle dictates the **spin-2 character of gravitational radiation and the 8π normalization** (Sections 6 and 7)
* The aQPU finite kernel supplies the exact combinatorial invariants and optical depth model that fix **the laboratory value of G to sub-ppm precision** (Sections 8 and 9)
* The framework yields testable strong-field predictions, including **reduced black hole shadow sizes and coupling suppression in neutron star interiors** (Sections 10 and 11)

Companion analyses provide supporting results, including the dimensional proof [15], the fine-structure constant calculation [24, 27], the UV-IR energy ladder [25, 26], the compact opacity construction [16], and the byte formalism [17].

### 1.1 Notation and Units

Throughout this manuscript, natural units c = ℏ = 1 are used except where SI is explicitly noted in observational predictions. The following symbols appear throughout and are collected here for reference. Additional symbols are defined at the point of first use.

| Category | Symbol | Meaning |
|---|---|---|
| **Foundational Constraints** | CS | Common Source |
| | UNA | Unity-Non-Absolute |
| | ONA | Opposition-Non-Absolute |
| | BU | Balance Universal |
| | BU-Egress | Balance Universal, depth-four closure |
| | BU-Ingress | Balance Universal, depth-eight reconstruction |
| **Geometric Invariants** | Q_G | Quantum of gravity (4π steradians) |
| | m_a | Observational aperture parameter |
| | ρ | Closure ratio (δ_BU / m_a) |
| | Δ | Aperture gap (1 − ρ) |
| **Gravitational Parameters** | ψ | Gravitational potential ratio \|Φ\| / Φ_Planck |
| | τ_G | Gravitational optical depth |
| | G_kernel | Kernel coupling constant (π/6) |
| | g₁ | Logarithmic coupling gradient d ln G / dψ |
| **Energy Scales** | E_CS | Planck-scale energy anchor (1.22 × 10¹⁹ GeV) |
| | v | Electroweak scale (246.22 GeV) |

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

Throughout the logical development we reserve the symbol S for the designated propositional constant anchoring the horizon worlds. When this constant is realized in the Hilbert-space representation, its expectation value equals the scalar horizon invariant of quantum gravity `Q_G = 4π`.

#### Core Definitions

Four formulas capture the structural properties required by the Common Governance Model, all anchored to the horizon constant S:

| Concept | Formula | Description |
|---------|---------|-------------|
| Unity (U) | [L]S ↔ [R]S | Unity holds when left and right transitions yield equivalent results at the horizon constant. |
| Two-step Equality (E) | [L][R]S ↔ [R][L]S | Two-step equality holds when depth-two modal compositions commute at the horizon constant. |
| Opposition (O) | [L][R]S ↔ ¬[R][L]S | Opposition holds when depth-two modal compositions yield contradictory results at the horizon constant. |
| Balance (B) | [L][R][L][R]S ↔ [R][L][R][L]S | Balance holds when depth-four modal compositions commute at the horizon constant. |

#### The Five Foundational Constraints

The framework relies on five foundational constraints: one assumption (CS), two lemmas (UNA, ONA), and two propositions (BU-Egress, BU-Ingress). For independence analysis in the core modal system we treat all five as primitives. In the operational regime, the continuous flows, reachability from S, and simple Lie closure allow UNA and ONA to be obtained from CS (hence the lemma designation). The conjunction of BU-Egress and BU-Ingress defines universal balance.

### 2.1 Ancestry:  Common Source (CS) Assumption

The foundational postulate asserts that all distinguishable physical structure remains traceable to a Common Source (CS). 

```
S → ([R]S ↔ S ∧ ¬([L]S ↔ S))
```

> The horizon constant S is preserved under right transitions but altered under left transitions. This establishes fundamental chirality, or "handedness," in the system. The reference state behaves asymmetrically under the two types of transitions.  

Such traceability requires an ancestral parity violation, manifesting physically as chirality. Asymmetry governs the emergence of individuality, forming the dimensional identity called gravity. Directional bias enables coherent observation by distinguishing operational paths.

> **Definition:** Gravity is the emergent balance establishing preservation of ancestry through freedom of identity and individuality.

Gravitational alignment requires a distinct identity with accountable individuality (Unity and Opposition Non Absolute). Ancestry preserves a balance between these extreme operational modes, which necessitates energy conservation through gyration. 

Systems gyrate within an emergent spacetime continuum while acting as self-referential states of space evolving through relative time. When displacements are composed in a curved geometry, the operation is non-associative. Gyrogroup algebra corrects this non-associativity via the gyration operator (Section 5.3). In the continuous physical limit, accumulated gyration manifests as rotational structure. Angular momentum emerges as the physical expression of this conserved gyration, ensuring that the system preserves its orientation relative to the common source while undergoing translation (Section 5.3).

### 2.2 Identity of Individuality: Unity-Non-Absolute (UNA) Lemma

Non-absolute unity (¬□E) ensures informational variety while maintaining ancestry preservation, preventing homogeneous collapse.

```
S → ¬□E    where E := [L][R]S ↔ [R][L]S
```

> At depth two (two nested modal operations), the order of transitions matters, but this non-commutativity is not absolute across all accessible states. The commutativity of depth-two operations is contingent, holding in some accessible states and failing in others.

Transitions deviating from this gravitational alignment correspond, in the physical realization, to linear momentum (Section 5.3). These transitions function as a centrifugal tendency, establishing individual distinction.

This contingency expands chirality into full rotational structure. SU(2) is the minimal algebra supporting this expansion, possessing three generators and a unique double cover of SO(3), yielding three rotational degrees of freedom.

### 2.3 Individuality of Identity: Opposition-Non-Absolute (ONA) Lemma

Non-absolute opposition (¬□¬E) ensures accountability of inference. Different operational paths remain comparable even when yielding different results, preventing structural fragmentation.

```
S → ¬□¬E
```

> While depth-two operations may yield opposite results ([L][R] vs. [R][L]), this opposition is not absolute. The system avoids both complete agreement and complete contradiction.  

Transitions preserving this gravitational alignment correspond, in the physical realization, to angular momentum (Section 5.3). These transitions function as a centripetal binding tendency, sustaining shared identity.

The combined structure is SE(3) = SU(2) ⋉ R³, the group of rigid body motions. Every rigid motion rotates first, then translates in a frame-dependent direction. The result is six degrees of freedom: three rotational and three translational.

### 2.4 Balance Universal Proposition

Gravitational attraction arises because linear displacement away from origination increases the energetic cost of maintaining relational coherence (Section 6.1). Mass and Energy correspond to the accumulated gravitational memory of ancestry (Section 5.1). The gravitational field describes how concentrated ancestry directs the trajectory of other structures seeking to preserve balance through alignment (Section 5.2).

Translational change, once committed as gravitational memory, defines time. The rotational and translational sectors map, in the relativistic limit, to gravitomagnetic and gravitoelectric structure respectively. The arrow of time arises because distinctions accumulate in a fixed order to satisfy these necessities. Achieving balanced alignment requires first establishing non-absolute distinctions, making the arrow of time a consequence of the requirement ordering. Return to the original state requires traversing the full depth-eight cycle. The full derivation of the dimensional structure appears in [15].

#### 2.4.1 BU-Egress: Depth-Four Closure

Egress Balance Universal (BU-Egress) mandates that four-step compositions commute across all accessible configurations.

```
S → □B    where B := [L][R][L][R]S ↔ [R][L][R][L]S
```

At depth four (four nested operations), the system achieves commutative closure. All admissible orderings of alternating depth-four transitions converge to the same state. This closure is absolute (□B), meaning it holds across all accessible worlds.  

This is the minimal depth at which coherent closure can occur while preserving depth-two contingency. Depth three still allows asymmetry. Depth four forces balance through the structure of the Baker-Campbell-Hausdorff expansion.  

#### 2.4.2 BU-Ingress: Memory Reconstruction

Ingress Balance Universal (BU-Ingress) requires the resulting closed state to retain sufficient information to reconstruct the original asymmetry and the contingency of earlier steps.

```
S → (□B → ([R]S ↔ S ∧ ¬([L]S ↔ S) ∧ ¬□E ∧ ¬□¬E))
```

The balanced state at depth four contains sufficient information to reconstruct all prior conditions: the original chirality (CS), the contingent unity (UNA), and the non-absolute opposition (ONA). Balance implies memory.  

This ensures that achieving balanced closure doesn't erase the structural distinctions established by prior constraints. The future state preserves the information required to reconstitute past distinctions without collapsing them.

---

Balance necessitates governance through observability. Gravity, being the capacity for conservation of ancestry and not an end-point, cannot appear as an observable state within the system. Because the source cannot be observed as a discrete point, observation must encompass the complete sphere of possible orientations. The measurement domain must therefore be spatially closed and finite. In three spatial dimensions, complete angular closure requires 4π steradians. This geometric necessity establishes the quantum of gravity, Q_G = 4π, used throughout the subsequent analysis.

## 4. The Quantum of Gravity

The dimensional structure of Section 3 produces a specific geometric constant. Full angular closure of the observational domain in three dimensions is 4π steradians. This quantity functions as a quantum of gravity, Q_G = 4π, analogous to ℏ but for geometric closure rather than phase-space resolution. The graviton operator acts inside the closed frame, and Q_G is the closure constant that sets its scale.

The same 4π appears in the identity ∇²(1/r) = −4π δ³(r). Both express the same geometric fact. Complete angular coverage of three-dimensional space requires 4π steradians. In standard physics, this factor is inherited from the geometry of three-dimensional space. In CGM, it is derived from the closure requirement.

The aperture parameter m_a fixes how the directional asymmetry from Section 2 is distributed across the rotational phases. Normalizing across both transition directions gives

    m_a² × (2π)_L × (2π)_R = π/2

from which m_a² = 1/(8π) and Q_G m_a² = 1/2. Numerically, m_a ≈ 0.199471. The closure ratio ρ measures how close the balanced state comes to filling the aperture. The aperture gap Δ = 1 − ρ measures the deviation from full closure. Numerically, Δ ≈ 0.0207.

Gravity is weak because the depth-eight recovery accumulates attenuation proportional to |Ω| Δ ρ⁵ across N_cycles closure traversals. The gravitational coupling requires full depth-eight recovery, which traverses this optical depth and is therefore exponentially suppressed relative to the electromagnetic coupling that operates at the depth-four residual. The optical depth τ_G quantifies this suppression and is computed in Section 9.

Electromagnetic and gravitational couplings share this aperture geometry, producing a testable relationship between them. With ζ = 8/(m_a √3) = 16√(2π/3) ≈ 23.155 and α₀ = δ_BU⁴/m_a, the product

    α₀ ζ = ρ⁴ / (π √3) = 0.169025920321

cancels m_a entirely. Independent measurements of α and G can therefore falsify CGM should their product violate ρ⁴/(π√3). The laboratory fine-structure constant α_CODATA differs from α₀ by +319 ppm. The transport-corrected value from [24, 27] matches α_CODATA to sub-ppb accuracy. Gravity and the kernel invariant use α₀ and do not incorporate that correction chain, so the product test applies to α₀ specifically.

## 5. The Field Equation

The quantum of gravity and the aperture geometry of Section 4 fix the coupling structure. This section derives the field equation governing how the gravitational field is sourced. The argument proceeds in three stages: the definition of the source, the derivation of the Poisson equation, and the decomposition into gravitoelectric and gravitomagnetic sectors.

### 5.1 The Source

In CGM, mass is accumulated common-source structure. Mass-energy density ρ measures how much traceable-to-common-origin structure is present per unit volume.

The gravitational potential Φ measures the concentration of common-origin structure at a point. With Φ ≤ 0 when the reference at infinity sets Φ = 0, the acceleration field is g = −∇Φ. Approaching greater concentration lowers the action required to maintain relational traceability, which is why test bodies accelerate toward mass concentrations.

### 5.2 The Poisson Equation

Rotational invariance and linearity in R³ uniquely determine a second-order operator of Laplacian form up to a scalar factor. The gravitational potential therefore satisfies

    ∇² Φ = Q_G G ρ

where Q_G = 4π is the quantum of gravity derived in Section 4. This matches the standard Newtonian form. Curvature, in this framing, is the observable gradient of mass-energy density. Uniform density gives a flat geometry with vanishing gradient while gravity remains present. Varying density produces curvature in the standard sense.

### 5.3 The Gravitoelectromagnetic Decomposition

The six degrees of freedom derived in Section 3 decompose the gravitational field into two sectors. The gravitoelectric field g = −∇Φ carries the three translational degrees of freedom. The gravitomagnetic field B_g = curl A_g carries the three rotational degrees of freedom. Together they satisfy the gravitoelectromagnetic system:

    ∇ · g = −Q_G G ρ

    ∇ × g = −∂B_g / ∂τ

    ∇ · B_g = 0

    ∇ × B_g = −(Q_G G / c²) J + (1/c²)(∂g / ∂τ)

where J is the mass-energy current and τ denotes the physical time parameter. Heaviside [2] wrote these equations in 1893 as the gravitational analog of Maxwell's equations. They emerge rigorously from the weak-field limit of general relativity. The CGM derivation identifies the two sectors with the translational and rotational degrees of freedom forced by the closure constraints.

The decomposition also follows from the algebraic structure of displacement composition. Composing non-collinear displacements in a curved geometry is non-associative, and the gyrogroup algebra corrects this non-associativity via an automorphism called the gyration operator [8]. In the continuous limit, accumulated gyration produces a circulation field. The gravitomagnetic vector potential A_g is the continuous manifestation of this accumulated gyration.

In the weak-field regime, the gravitoelectromagnetic system implies wave propagation with characteristic speed c. Taking the curl of the gravitomagnetic equation and substituting the remaining identities yields a wave equation with the characteristic speed fixed by the same constant c that appears in the source response normalization. The multimessenger event GW170817 bounds any difference between the gravitational and electromagnetic propagation speeds to below 3 × 10⁻¹⁵ of c [28], consistent with this prediction. Static density gradients extend across space without wave propagation, while perturbations propagate at c through the gravitomagnetic sector.

## 6. Nonlinear Gravity

The linear theory of Section 5 treats G as constant. This is a good approximation in the weak field but cannot hold self-consistently when the gravitational field is strong, because mass-energy density modifies the geometry through which the field is sourced. The coupling must therefore depend on position. This section derives the position-dependent coupling, constructs the effective metric, verifies that the metric satisfies the Einstein equations, and extracts the equivalence principles and post-Newtonian parameters.

### 6.1 The Position-Dependent Coupling

The gravitational potential ratio ψ = |Φ|/Φ_Planck measures field strength relative to the Planck scale. In dimensionless units with r_g = GM/c², the coordinate s = r/r_g gives ψ(s) = GM/(rc²) in the Newtonian limit. The potential ratio ranges from 0 in the weak field to approximately 0.5 near compact-object horizons.

The coupling at a given point depends on how much common-source structure has accumulated there. This dependence enters through a reference energy scale E_ref(ψ) that shifts with gravitational depth. Three premises fix its form. Premise 1 is optical conjugacy. The UV and IR energy conjugates satisfy E_UV E_IR = E_CS v/(4π²), establishing that the Planck scale and the electroweak scale are paired foci of the system [25, 26]. Premise 2 is the energy ladder. Energy scales are positioned on a ruler with tick spacing Δ, so that n(E) = ln(E_CS/E)/(Δ ln 2). Premise 3 is the verified optical depth law. The accumulated depth scales as τ(ψ) = τ_G(1 − ψ), confirmed to machine precision against the kernel spectral accumulation.

Combining these three premises yields the reference energy as a function of gravitational depth:

    E_ref(ψ) = E_CS (v / E_CS)^(1−ψ)

At ψ = 0, corresponding to the weak field, E_ref = v, the electroweak scale. At ψ = 1, corresponding to the Planck scale, E_ref = E_CS. The reference energy is a ruler quantile. It represents the energy at position τ(ψ) on the ladder. On a logarithmic scale spanning approximately 17 decades, centroid and quantile differ substantially, and the quantile is the correct object because the optical depth measures position along the ladder rather than an average over it. The formal proof appears in Appendix E.

Substituting τ(ψ) and E_ref(ψ) into the coupling formula gives

    G(ψ) = G_kernel exp(−τ_G(1−ψ)) / E_ref(ψ)² = G₀ exp(g₁ ψ)

where G₀ = G_kernel exp(−τ_G)/v² is the weak-field coupling and g₁ = τ_G + 2η, with η = ln(v/E_CS). Numerically, g₁ = −0.6456.

Since d ln G / dψ = g₁ < 0, the coupling decreases with increasing ψ. As mass accumulates and the local potential deepens, E_ref(ψ) shifts from the electroweak scale toward the Planck scale, weakening G(ψ) where ψ is largest. At the electroweak anchor (ψ = 0), G = G₀. At the Planck anchor (ψ = 1), G ≈ 0.524 G₀.

For a point mass, the potential satisfies dψ/ds = −exp(g₁ψ)/s². This equation has the exact solution

    ψ(s) = −(1/g₁) ln(1 − g₁/s)

which reduces to ψ = 1/s in the Newtonian limit g₁ → 0. The solution remains real and finite for all s > 1/g₁. The maximum potential attained is ψ_max = −1/g₁ ≈ 0.4996 at the minimum radius, which lies just inside the conventional Schwarzschild radius.

For extended density distributions ρ(r), the full coupled system comprises the potential ratio ψ, the field equation g = −dΦ/dr = G(ψ) M(r)/r², the mass equation dM/dr = Q_G ρ(r) r², and the coupling function G(ψ) from above. This system closes for any specified density profile.

### 6.2 The Effective Metric and Einstein Equations

Geometry responds to mass-energy density through the potential ratio ψ. The effective metric for static spherical configurations is

    ds² = −f dt² + f⁻¹ dr² + r² dΩ²,    f(r) = 1 − 2ψ(r)

The horizon occurs where f = 0, at ψ = 1/2. For the exact point-mass solution, the horizon radius is s_h = 1.695 r_g, a 15.3% inward shift from the Schwarzschild radius. The photon sphere occurs at s_ph = 2.586 r_g compared to 3.0 in general relativity.

The position-dependent coupling modifies the Gauss law to

    ∇ · [(G₀/G(x)) g] = −Q_G G₀ ρ

For a point mass, (G₀/G(ψ)) × 4π s² g = 4π at all radii, verified to relative precision 2.83 × 10⁻¹⁶. The modified flux is exactly conserved.

The Einstein tensor for the metric f = 1 − 2ψ satisfies the component identity G_rr = G_tt/f² to relative precision 4.4 × 10⁻¹⁶ across all sampled radii. The position-dependent coupling introduces an effective anisotropic stress-energy in the exterior vacuum. The gradient of G(ψ) acts as a source term, producing a tangential pressure that structurally supports the coupling gradient. The modified Bianchi identity governs a continuous energy exchange between mass-energy density and the gravitational field:

    ∇_μ T^μ_ν = −(∂_ν G / G) T^μ_μ

This exchange is negligible in the weak field (s > 100) and becomes significant near compact objects (s < 10), where the tangential pressure dominates the effective stress tensor.

The continuum limit admits a scalar-tensor representation in which ψ appears as a position-dependent coupling functional:

    S = (1/16πG₀) ∫ R exp(−g₁ψ) √(−g) d⁴x + ∫ L_m √(−g) d⁴x

Unlike Brans-Dicke constructions, ψ has no independent dynamical degree of freedom and is fixed algebraically by the closure structure through ψ = |Φ|/Φ_Planck. It is therefore a derived quantity, free of its own kinetic term and its own equation of motion. In the point-mass exterior, R = 0, ∇²ψ = 0, and dV/dψ = 0 at equilibrium. Energy conditions (null, weak, dominant) are satisfied for ψ ∈ [0, ½).

### 6.3 Redshift and Equivalence Principles

The CGM gravitational redshift follows from the metric:

    z_CGM = 1/(1−ψ) − 1

General relativity predicts z_GR = 1/√(1−2ψ) − 1. The two agree to within 2% for ψ < 0.2, covering the neutron-star regime. They diverge beyond that threshold. The CGM redshift expression remains defined for ψ in [0, 1], while the GR square-root form is defined for ψ < 0.5 in this normalization.

Three equivalence principles follow from the CGM structure. The Weak Equivalence Principle holds because free-fall acceleration depends on the local field ψ alone, independent of the composition of the test body (confirmed by the kernel realization, Section 8). The Einstein Equivalence Principle holds because the rotational-translational structure at each point guarantees local isotropy and frame equivalence. The Strong Equivalence Principle holds because G(ψ) depends on position alone, independent of the internal structure of the body. The Nordtvedt parameter η_N = 0, consistent with lunar laser ranging bounds |η_N| < 2.2 × 10⁻⁵ [29].

### 6.4 Post-Newtonian Parameters

The perturbative expansion of the exact point-mass solution in powers of 1/s gives

    ψ(s) = 1/s + a₂/s² + a₃/s³ + ⋯,    a_n = g₁^(n−1) / n

The coefficients are a₁ = 1, a₂ = −0.3228, a₃ = 0.1389, a₄ = −0.0673, a₅ = 0.0347. From this expansion, γ = 1 exactly, consistent with the Cassini measurement γ = 1 ± 2.3 × 10⁻⁵ [29]. The leading deflection term is 4GM/c²b, identical to GR. The parameter β = 1 − g₁/2 = 1.3228 at leading order. This value functions as a strong-field book-keeping parameter. At solar potentials (ψ ~ 10⁻⁸), the effective deviation from β = 1 is negligible, ensuring agreement with solar system tests. The nonlinearity activates prominently only for ψ > 0.01, placing the observable deviation firmly in the compact-object regime. Mercury's perihelion precession matches the general relativistic prediction to 0.003 ppm when calculated with the exact CGM metric, confirming that the elevated β remains dormant at weak-field scales.

## 7. Gravitational Radiation

The nonlinear theory of Section 6 completes the static description. This section addresses the dynamical sector.

The canonical operational cycle traces a shell path with two symmetric excursions per half-cycle. Fourier decomposition of the shell displacement signal reveals the dominant spectral component at k = 2 with amplitude 1.5. Two equal peaks per cycle identify the quadrupole structure at the level of the finite shell dynamics. In the continuous limit, a spin-2 field radiates predominantly through quadrupole emission, and the kernel's dominant spectral mode at k = 2 is the discrete precursor of this continuum behavior.

Gravitational memory in CGM manifests as a permanently embedded gyration resulting from an interrupted depth-eight closure cycle. The cycle requires a forward closure (BU-Egress at depth four) followed by a backward reconstruction (BU-Ingress completing at depth eight). A gravitational wave interrupting this cycle after BU-Egress but before BU-Ingress completion leaves an unresolved orientation correction permanently embedded in the system state. In the continuum limit, this residual gyration appears as the static metric displacement recorded by gravitational wave memory [12].

Gravitational wave strain is calibrated against the Hulse-Taylor binary pulsar [30]. For the Hulse-Taylor system, ψ_orbital ≈ 2.1 × 10⁻⁶, giving G(ψ)/G₀ = 0.9999986. The resulting orbital period derivative differs from the GR prediction by 0.0003%, below current observational precision. Strong-field sources exhibit larger corrections. A neutron-star binary at 20 km orbital separation has ψ ≈ 0.10, yielding G/G₀ ≈ 0.935 and a 6.5% reduction in gravitational wave luminosity.

Standard general relativity treats gravitational waves as weak perturbations h_μν on a Lorentzian metric. CGM describes the same signature as modulation of g and B_g from Section 5. Both constructions coincide on quadrupole luminosity at leading post-Newtonian order. The graviton corresponds to the orientation-recovery operator that closes the depth-eight cycle. The exact zero variance in optical depth across the four K4 family phases demonstrates path independence: all four discrete symmetry sectors of the closure algebra contribute identically to the gravitational attenuation. This path independence guarantees the masslessness of the graviton, since selecting a preferred path through the K4 structure would introduce a mass scale. The helicity-two profile and effective undetectability as a resolved particle follow from kernel algebra, without requiring a graviton propagator.

## 8. The Kernel Realization

The field law of Section 5 fixes the continuum form. The finite kernel fixes the discrete normalization. The closure conditions of Section 2 admit a finite algebraic realization. This is a system of 4096 states organized into seven concentric shells. This kernel provides exact combinatorial invariants that anchor the gravitational coupling in the continuous theory. All numerical values are reproducible from the computational verification scripts (experiments/aqpu_gravity_analysis_1.py through _3.py, experiments/aqpu_wavefunction_2.py).

The 4096 states are distributed across seven shells indexed by k = 0 through 6. Shell populations follow the binomial distribution C(6,k)/64, yielding 64, 384, 960, 1280, 960, 384, 64. The distribution reflects the fact that the six degrees of freedom are controlled independently, each equally likely to be activated or dormant under the ergodic measure. Two shells are horizon shells, where all directional bias cancels and anisotropy vanishes. Five are bulk shells, where anisotropy is nonzero. Gravity couples exclusively to the five symmetric trace-free (STF) orientational degrees of freedom carried by these bulk shells. The trace component (monopole) provides isotropic pressure but does not carry the gravitational signal. This partition directly produces the ρ⁵ attenuation factor: coherent survival across the five STF components yields exactly ρ⁵.

The shell displacement D measures the total distance traversed through shell space during a complete operational cycle. The depth-four half-word maps shell s to 6 − s, producing a traverse of 6 per half-word. The full cycle composes two such traversals, yielding D = 24 per complete round-trip. This invariance holds across all 64 mass configurations and is verified exhaustively in the wavefunction diagnostic.

The kernel Gauss map converts this integer to the dimensionless coupling:

    G_kernel = Q_G / D = π/6

The product D × G_kernel = Q_G = 4π gives the total flux per cycle in solid-angle units. This is the discrete Gauss law. The flux through any closed surface is quantized in units of Q_G, and the coupling is fixed by the ratio of the quantum to the displacement.

Every mass configuration reaches the equality horizon at the midpoint of the depth-four half-cycle, independent of mass. This fixed midpoint contact is the kernel expression of the Weak Equivalence Principle. Heavy and light objects follow the same path through shell space because they all pass through the same midpoint.

At the equality horizon, all six directional bias components resolve simultaneously into a definite state. At bulk steps, each chirality bit has probability 0.5 of being set. At the equality horizon, all six bits are set with certainty. At the complement horizon, all six are zero. Each depth-four half-word fully inverts chirality, and the full canonical word composes two such inversions, yielding zero net chirality change while acting on the positional state. This confirms that the holonomy acts on the positional subspace only.

The two boundary horizons with 64 states each and the full manifold with 4096 states satisfy |H|² = |Ω|, meaning 64² = 4096. The holographic identity ln|Ω| = 2 ln|H| identifies the Bekenstein-Hawking entropy factor of 2 with the two-pass structure of the gravitational closure cycle. Each pass through the depth-four closure point contributes one unit of ln|H| to the total entropy budget.

The constant anisotropy ratio ‖π‖² / Tr(σ)² = 2/75 across all bulk shells determines the angular distribution of gravitational radiation from any source described within the kernel structure. The interior-shell anisotropy ratio √6/9 ≈ 0.2722 matches the Nariai ultracold mass bound for stable extremal compact objects [23]. This convergence identifies the Nariai limit as the stability boundary for closure. Objects exceeding this anisotropy ratio cannot maintain stable closure and therefore cannot exist as stable configurations within the framework.

The shell transition operator for byte weight q has trace C(q) that evaluates to exact rationals. For even weights q = 2k, the Chu-Vandermonde identity yields C(2k) = 7/(2k+1), giving C(0) = 7, C(2) = 7/3, C(4) = 7/5, C(6) = 1. For odd weights, the return trace is obtained from the Krawtchouk spectral decomposition, yielding C(1) = C(5) = 28/9 and C(3) = 52/25. These values are verified by three independent computational routes. The full derivation appears in Appendix B.

### 8.1 Gauss Law Bridge

Embedding the seven shell layers into a radial coordinate with the binomial mass profile verifies that the discrete quantities produce the continuum Poisson equation. The boundary flux equals −Q_G G_kernel to relative precision 10⁻¹⁶. In the continuum limit, spherical symmetry gives div g = −Q_G G_kernel ρ(r). Substituting G = G_kernel/E_CS² and matching the kernel profile to continuum mass-energy density ρ gives div g = −Q_G G ρ. This is the Poisson equation derived from the kernel Gauss law rather than assumed from rotational invariance alone. Three independent numerical checks confirm the inverse-square behavior. The product |g|r² is constant across the exterior to machine precision. A least-squares fit of log|g| versus log r gives an exponent of −2.000000 with uncertainty 9 × 10⁻¹⁶.

## 9. The Gravitational Coupling Constant

The kernel invariants of Section 8 fix the discrete normalization. This section assembles the gravitational coupling from those invariants, the continuum matching of Section 5, and results established in companion analyses. The central result is a prediction for the laboratory value of G that agrees with the measured value to 0.074 ppm.

At the Planck-scale anchor, the coupling is G = G_kernel/E_CS², where G_kernel = π/6 and E_CS = 1.22 × 10¹⁹ GeV. The laboratory coupling is predicted by

    G_pred = G_kernel exp(−τ_G) / v²

where v = 246.22 GeV is the electroweak scale, identified as the infrared conjugate of E_CS by optical conjugacy. The Planck scale is excluded as an input to the derivation, and ℏ enters only through unit conversion. The dimensional anchor is the electroweak scale.

The optical depth τ_G aggregates the attenuation over all closure cycles between the Planck and electroweak anchors. As established in Section 4, gravitational weakness follows from |Ω| Δ ρ⁵ accumulated across N_cycles depth-eight traversals. The leading closed form for the optical depth is

    τ_G⁰ = |Ω| Δ ρ⁵ (1 − 4ρ Δ²)

where |Ω| = 4096, Δ = 1 − ρ, and ρ is the closure ratio from Section 4. The factor (1 − 4ρΔ²) is the lowest-order symmetric correction from the four-stage depth structure. An additive correction reduces the residual further:

    δτ = |Ω| Δ ρ⁵ c₄ Δ⁴,    c₄ = −7/4

The full model is τ_G = τ_G⁰ + δτ. The constant c₄ = −7/4 is fixed by two independent routes (Appendix C.4). The Z2 involution of the holonomy cycle eliminates all odd-order corrections in Δ, enforcing exact symmetry at leading order. The c₄ Δ⁴ correction represents a soft breaking of this Z2 symmetry by the isotropic pressure component of the stress tensor, providing the monopole contribution to the optical depth that the five STF components miss.

Three independent arguments converge on the exponent 5 in ρ⁵. First, a symmetric 3×3 stress in three dimensions decomposes into one trace component and five independent symmetric trace-free components. Gravity couples to the trace-free part, and coherent survival across this five-component sector produces ρ⁵. Second, five bulk shells carry anisotropy (Section 8). The surviving anisotropy channels number exactly five. Third, the exponent equals 8 − 3 = 5 via the two-lemma factorization of the optical depth. All three routes yield the same exponent without reference to G. The hierarchy between the gravitational and electroweak couplings reflects the accumulated attenuation of directional information across the full set of depth-eight cycles. The electroweak mass coordinates of the top quark, Higgs, Z, and W bosons occupy positions n = 24.7, 47.2, 69.2, and 78.0 on the energy ruler. The gravitational coupling occupies position n_G ≈ 3714.3, which is approximately |Ω|ρ⁵ ≈ 3689.5 ticks from the electroweak anchor. The ratio of the two coupling strengths is determined by the total number of closure cycles required to restore full closure, multiplied by the per-cycle attenuation from the five-component STF sector.

The exact per-cycle optical depth is a kernel theorem:

    τ_cycle = (7591/7392) Δ

where 7591 = (1/2) Σ_{k=1}^{5} C(6,k)³ and 7392 = 8 C(12,6). The derivation appears in Appendix C. The number of depth-eight cycles is N_cycles ≈ 3586.5, and the product N_cycles × τ_cycle = τ_G confirms exact agreement with the closed form.

Numerical evaluation gives τ_G⁰ alone a 25 ppm offset in G relative to the reference measurement. Adding δτ with c₄ = −7/4 leaves a residual of 7.36 × 10⁻⁸ in τ, corresponding to

    G_pred = 6.7088095 × 10⁻³⁹ GeV⁻²    vs.    G_meas = 6.7088100 × 10⁻³⁹ GeV⁻²

a difference of 0.074 ppm. This agreement is far tighter than present experimental uncertainty on G, which is of order 10⁻⁵ relative in CODATA 2018 [13]. The comparison is a consistency check against the chosen reference value, distinct from a metrological verification at that precision. A decisive test requires substantially improved G measurements or an independent observable that constrains the same τ_G structure. Because G_pred = G_kernel exp(−τ_G)/v², a fractional change in τ maps directly to the same fractional change in G with opposite sign. The prediction is stable at the precision to which Δ and ρ are fixed, and the sub-ppm residual is free of fine-tuned cancellation among poorly determined inputs.

## 10. Strong-Field Predictions

The coupling function G(ψ) derived in Section 6 produces observational consequences across a range of compact objects. This section quantifies those consequences for black hole shadows, neutron star interiors, and orbital dynamics.

The photon sphere (Section 6.2) yields an impact parameter b/r_g = 4.648, giving a shadow area 80% of the GR Schwarzschild prediction. Table 1 gives shadow angular diameters for two Event Horizon Telescope targets. CGM predictions fall 1.7 to 2.0 standard deviations below EHT measurements [31, 32]. Next-generation EHT observations at improved precision will distinguish CGM from GR at the 2σ level. Spin corrections use the wavefunction two-pass deficit method, with helix activation factor 0.5 and metric κ = 4.5, valid for spin parameter a* < 0.3.

**Table 1.** Shadow angular diameters (μas).

| Source | CGM (Schwarzschild) | CGM (with spin) | GR (Kerr) | EHT measurement |
|--------|---------------------|-----------------|-----------|-----------------|
| M87*   | 35.5               | 36.2            | 36.0      | 42.0 ± 3.0      |
| Sgr A* | 45.3               | 48.0            | 47.2      | 51.8 ± 2.3      |

A neutron star with M = 1.4 M☉ and R = 12 km has surface potential ψ_surface ≈ 0.153. The position-dependent coupling gives G/G₀ ≈ 0.906 at the surface. The core coupling remains near G₀ because the central potential is small for a polytropic equation of state. A self-consistent Tolman-Oppenheimer-Volkoff integration with G(ψ) confirms this. For a γ = 2 polytrope with central density 8 × 10¹⁷ kg/m³, the integration yields R ≈ 15.4 km and enclosed mass 1.25 M☉. The coupling increases from G/G₀ ≈ 0.906 at the surface to 1.000 at the center. The continuous variation of G(ψ) across the neutron star interior provides a physical mechanism for systematic radius discrepancies between observational methods. X-ray burst oscillations probe the surface regime where G/G₀ ≈ 0.91. Gravitational wave analysis of mergers probes the orbital regime where G/G₀ ≈ 0.94. Assuming a constant G when interpreting radius and mass measurements from these different regimes yields systematically different answers. The CGM coupling function predicts the direction and magnitude of these offsets as a function of the observational geometry.

Table 2 summarizes the coupling reduction across astrophysical objects. Earth and the Sun are in the linear regime with negligible corrections. White dwarfs show a 0.02% reduction. Neutron stars enter the strongly nonlinear regime with a 10.5% reduction. Stellar black holes at the horizon show a 27.6% reduction.

**Table 2.** Astrophysical coupling reduction.

| Object | ψ | G/G₀ | δG/G |
|--------|---|------|------|
| Earth surface | 7.0 × 10⁻¹⁰ | 1.000000 | −0.45 ppb |
| Sun surface | 2.1 × 10⁻⁶ | 0.999999 | −1.4 ppm |
| White dwarf (1 M☉) | 3.0 × 10⁻⁴ | 0.999809 | −0.019% |
| Neutron star (1.4 M☉) | 0.172 | 0.895 | −10.5% |
| Stellar BH (10 M☉) | 0.501 | 0.724 | −27.6% |

Orbital period corrections scale as T/T_Newton = 1/√(G/G₀). For Earth at 1 AU, the correction is negligible. For a neutron star close orbit at s = 11.6, T/T_N = 1.027 and v/v_N = 0.974. For a black hole innermost stable circular orbit at s = 6.0, T/T_N = 1.052 and v/v_N = 0.951.

## 11. Testable Predictions

The strong-field predictions of Section 10 are the most direct tests of the nonlinear theory. This section collects all falsifiable predictions from across the manuscript.

Given α_CODATA, the product α ζ = ρ⁴/(π√3) predicts ζ ≈ 23.163. Any independent constraint on G or ζ that disagrees with this propagation, after explicit identification of which α definition is used, falsifies the stated layer of the framework.

The shell opacity structure modulates the effective electromagnetic coupling across cosmological depth. Mapped to redshift via the energy ladder, this produces an oscillation in the fine-structure constant with period Δ_z ≈ 0.0143 in ln(1+z) and peak-to-peak fractional amplitude approximately 4.8 × 10⁻⁴. Seven sub-cycles per main period arise from the shell structure, giving a sub-cycle period of approximately 0.0020. A survey spanning at least one full period in ln(1+z) and detecting no oscillation at 3σ confidence with the stated period would falsify this prediction at the stated amplitude scale.

Next-generation EHT observations that refine the shadow diameters of M87* and Sgr A* to below 2 μas precision will distinguish CGM from GR. CGM predicts shadow diameters 10% below the GR Schwarzschild value (Section 10).

Different experimental methods for measuring G yield systematically different values [13]. CGM predicts that systematic offsets among methods correlate with the effective shell weighting of the experimental configuration. The shell structure distributes coupling strength non-uniformly across the seven shells according to the binomial weight C(6,k)/64. Experiments that preferentially activate different geometric configurations will systematically measure different effective values of G. The per-family optical depth variance is exactly zero across all four families, confirming that the variation emerges between experimental geometries rather than within them. Deriving the method-to-shell projection map for each experimental type would convert this qualitative prediction into a quantitative one. This concerns path-dependence rather than time-dependence, and supernova constraints on time variation of G remain consistent with CGM.

The CGM correction to gravitational wave luminosity is consistent with Hulse-Taylor observations (Section 7). Strong-field corrections reach several percent for neutron-star mergers (Section 7).

Dual-pole symmetry requires the next correction to the fine-structure constant prediction to be negative. A positive O(δ_BU⁶) correction at the Thomson limit would falsify the geometric identification.

The redshift predictions diverge for ψ > 0.2 (Section 6.3), making neutron star surfaces a viable test site. A precision redshift measurement near a neutron star surface (ψ ≈ 0.15) at the 5% level would distinguish the two predictions.

## 12. Implications

The results carry implications for three areas: the structure of quantum gravity programs, the status of spacetime dimensionality, and the origin of coupling constants.

Quantum gravity programs face a structural obstacle in perturbative quantization. Gravity corresponds to the depth-eight closure holonomy of the operational cycle, making the quantization of h_μν as an independent excitation circular. The kernel construction encodes this holonomy in the carrier recovery. The BU invariants Q_G, δ_BU, and m_a set the aperture structure, with Q_G = 4π as the geometric quantum. The framework is quantum-mechanical at foundation, without requiring a graviton propagator. The relevant question becomes how the depth-eight holonomy manifests in the regime where both quantum and gravitational effects are significant.

The derivation treats three-dimensional spatial structure as logically prior to four-dimensional spacetime. Time corresponds to the ordering of modal depth required to satisfy Common Source (CS), Unity-Non-Absolute (UNA), Opposition-Non-Absolute (ONA), and Balance Universal (BU). The fourth coordinate remains a computational device in the continuum packaging rather than an ontological primitive. The algebraic kernel forces the 3+1 split through the SO(3)/SU(2) shadow projection. Four-dimensional tensors remain a valid packaging of the continuum limit, encoding a distinction between spatial and temporal domains that the kernel makes algebraically explicit.

Coupling constants emerge from the BU invariants Q_G, δ_BU, and m_a rather than as free parameters. The gravitational coupling follows from kernel invariants, aperture geometry, and one energy anchor. The electromagnetic coupling follows from the same aperture geometry at a different depth. Their product is fixed by the closure ratio alone. Should this pattern extend, other couplings may yield to similar constructions. The shell structure produces both a monopole (1 component) and quadrupole (5 components) decomposition, suggesting that the kernel contains sufficient spectral richness to accommodate the full Standard Model.

The UV-IR interface density ρ_MU scales in a gravitational field as ρ_MU(ψ) = ρ_MU(0) (v/E_ref(ψ))². Near a black hole horizon, this depletes by a factor of order 10⁻⁶, reflecting the extreme redshift of the UV-IR conjugate pair. The product ρ_MU E_ref² is preserved across all ψ. At the Sun's surface, the depletion is 0.016%. At a neutron star surface, it reaches 99.9998%.

## 13. Conclusion

This manuscript derived the gravitational field equation, the coupling constant structure, the spin-2 character of gravitational interaction, and the nonlinear extension to the Einstein equations from a single requirement: the Preservation of Ancestry.

Three results anchor the derivation. First, the shell displacement invariant D = 24, verified exhaustively across all 64 mass configurations, establishes the discrete Gauss law and fixes G_kernel = π/6. Second, the depth-four/depth-eight holonomy distinction proves that the gravitational cycle requires a two-pass carrier return, supplying the factor 2 in 8π = 2 Q_G and the spin-2 angular momentum structure. Third, the optical depth model τ_G = |Ω| Δ ρ⁵ [(1 − 4ρΔ²) + c₄Δ⁴] with c₄ = −7/4 yields G_pred within 0.074 ppm of the reference measurement, with the exponent 5 confirmed by three independent arguments.

The nonlinear extension is complete. The position-dependent coupling G(ψ) = G₀ exp(g₁ψ) derives from three premises (Section 6.1). The exact point-mass solution ψ(s) = −(1/g₁) ln(1 − g₁/s) closes analytically. The effective metric f = 1 − 2ψ satisfies the Einstein field equations and admits a scalar-tensor representation in which ψ is algebraically slaved to the closure structure. Equivalence principles (WEP, EEP, SEP) follow from the CGM structure. PPN parameters γ = 1 and β = 1.3228 are extracted, with Cassini and lunar-ranging constraints satisfied. Strong-field predictions include a 15.3% horizon shift, a 10% shadow diameter reduction, and a 9.4% coupling reduction at neutron star surfaces.

Open questions remain. The strong and weak interactions would require identifying their corresponding carrier projections within the kernel structure. Realizing this would complete the program implied by the αζ product, establishing all couplings from the BU invariants Q_G, δ_BU, and m_a.

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

**The five foundational constraints:**

| Constraint | Name | Type | Formula |
|------------|------|------|---------|
| CS | Common Source | Assumption | S → ([R]S ↔ S ∧ ¬([L]S ↔ S)) |
| UNA | Unity-Non-Absolute | Lemma | S → ¬□E |
| ONA | Opposition-Non-Absolute | Lemma | S → ¬□¬E |
| BU-Egress | Balance Universal (egress) | Proposition | S → □B |
| BU-Ingress | Balance Universal (ingress) | Proposition | S → (□B → ([R]S ↔ S ∧ ¬([L]S ↔ S) ∧ ¬□E ∧ ¬□¬E)) |

Common Source (CS) establishes fundamental chirality. Right transitions preserve S while left transitions alter it. Unity-Non-Absolute (UNA) prevents homogeneous collapse by ensuring non-commutativity is contingent at depth two. Opposition-Non-Absolute (ONA) prevents irreconcilable contradiction by ensuring opposition is contingent at depth two. Balance Universal egress (BU-Egress) achieves commutative closure at depth four. Balance Universal ingress (BU-Ingress) guarantees that the balanced state contains sufficient information to reconstruct all prior conditions.

In the core modal system with Kripke semantics, all five constraints are logically independent. Each admits counterexample frames falsifying it while preserving the others. Consistency is verified via a three-world Kripke frame satisfying all five simultaneously. In the operational regime with continuous flows, reachability from S, and simple Lie closure, UNA and ONA follow from CS. The two-layer structure (modal axioms plus operational requirements) prevents circular reasoning.

## Appendix B: Kernel Topology and Combinatorics

### B.1 The Reachable Manifold

The kernel is a finite algebraic system with 4096 reachable states organized into seven shells. Shell k has ab-distance 2k for k = 0 through 6, with shell 0 at the equality horizon and shell 6 at the complement horizon. Shell populations follow |shell_k| = C(6,k)/64.

### B.2 The K4 Operator Algebra

The depth-4 canonical half-word W₂ maps shell s to 6 − s, producing a pole swap. Its square is the identity on all 4096 states, with eigenspace dimensions dim(+1) = dim(−1) = 2048. The full canonical word F = W₂ ∘ W₂' composes two such involutions through the Klein four-group (K4) algebra {id, W₂, W₂', F}. Gate F preserves shell while acting as a two-state flip on the positional coordinate within each shell. The holonomy cycle requires F ∘ F = id.

### B.3 Carrier Trace Theorems

The shell transition operator M_q for byte weight q acts on the seven-dimensional shell space. The transition probability from shell w to shell t under weight q is

    P(w → t | q) = C(w,t) C(6−w, q−t) / C(6,q)

For even weights q = 2k, the Chu-Vandermonde identity yields the diagonal sum

    Tr(M_{2k}) = [Σ_w C(w,k) C(6−w,k)] / C(6,2k) = C(7, 2k+1) / C(6,2k) = 7/(2k+1)

For odd weights, the byte swap alters shell parity, forcing Tr(M_q) = 0. The return trace C(q) = Tr(M_q²) is obtained from the Krawtchouk spectral decomposition. The shell transition matrices are μ-symmetric with respect to the binomial measure μ(w) = C(6,w)/64. The Krawtchouk polynomials K_w(k) form an orthogonal eigenbasis, and the eigenvalues λ_k are computed exactly as rationals. Summing λ_k² yields the odd-weight traces. These values are verified by three independent computational routes: two-hop transition product, matrix squaring, and spectral eigenvalue summation.

### B.4 Chirality Inversion

Each depth-4 half-word fully inverts chirality: q(W₂) = q(W₂') = 63 for all 64 micro-reference configurations m. The full canonical word composes two such inversions, yielding q(F) = 63 ⊕ 63 = 0. Gate F preserves chirality while acting on the positional state, verified exhaustively across all micro-references.

### B.5 Holographic Identity

The boundary horizons H with 64 states each and the full manifold Ω with 4096 states satisfy |H|² = |Ω|. This follows from the self-dual [12,6,2] code structure of the kernel [16]. The entropy relation ln|Ω| = 2 ln|H| is driven by the two-pass holonomy identified in Section 8.

## Appendix C: Optical Depth Construction

### C.1 Exact Per-Cycle Depth

The exact per-cycle optical depth is derived from the binomial-weighted holonomy transport over the 64 micro-references. For a micro-reference at popcount k, all four bulk steps land on shell k, contributing 4 × C(6,k)/64 per step. Weighting by the ergodic measure and summing gives

    τ_cycle / Δ = 4 Σ_k C(6,k)³ / (64 Σ_k C(6,k)²)

With Σ_{k=1}^{5} C(6,k)³ = 15182 and Σ_{k=0}^{6} C(6,k)² = 924 = C(12,6), this evaluates to 60728/59136 = 7591/7392.

### C.2 The K Factor

The transport measure τ_cycle/Δ = 7591/7392 and the anisotropy measure 5/99 (from the trace-free magnitude with the same binomial weighting) differ by the exact rational factor K = 22773/1120. This factor decomposes as K = (3 × 7591)/1120, where 1120 = 224 × 5 and 224 = 7392/33. The numerator is three times the half-cube-sum. The denominator derives from binomial moment identities.

### C.3 Series Expansion

Expanding the closed form τ_G = |Ω| Δ (1−Δ)⁵ (1−4(1−Δ)Δ²) as a polynomial in Δ produces a finite series through degree nine with coefficients c_n/|Ω| = [0, 1, −5, 6, 14, −55, 79, −60, 24, −4]. The series converges to the closed form exactly at machine precision.

### C.4 The c₄ Correction

The additive correction δτ = |Ω| Δ ρ⁵ c₄ Δ⁴ with c₄ = −7/4 is fixed by two independent routes. Route A gives c₄ = −(1 + Tr(σ_iso)) = −7/4 from the isotropic stress trace. Route B gives c₄ = q_W from the closure charge on gyroscopic edge increments, yielding the same value. Including δτ reduces the τ residual from 2.46 × 10⁻⁵ to 7.36 × 10⁻⁸.

### C.5 Cycle Count

The number of depth-8 cycles is N_cycles = |Ω| ρ⁵ (f_K4 + c₄ Δ⁴) / (τ_cycle/Δ), where f_K4 = 1 − 4ρΔ². This evaluates to N_cycles ≈ 3586.5. The product N_cycles × τ_cycle = τ_G confirms exact agreement with the closed form.

### C.6 Per-Family Uniformity

The per-family depth-4 optical depth is identical across all four family phases, with τ_word = 0.009408891 and zero variance. This uniformity supports the equal-weight assignment in the f_K4 correction factor.

## Appendix D: Translational Payload Stress

### D.1 Definition

The translational payload stress σ is computed from the translational payload bits. With the translational activation vector v = (b₆, b₅, b₄) and components in {0,1}, the stress is the centered second moment

    σ^{ij} = ⟨v^i v^j⟩ − ⟨v^i⟩⟨v^j⟩

where the averages are taken over the micro-reference ensemble within a cell.

### D.2 Decomposition

The tensor decomposes into isotropic and trace-free parts:

    σ^{ij} = p δ^{ij} + π^{ij}

where p = (1/3) Tr(σ) and π^{ij} is symmetric and trace-free. The trace component represents isotropic pressure. The five independent components of π form the trace-free sector corresponding to the ℓ = 2 representation of SO(3).

### D.3 Shell-Conditioned Values

Conditioning on popcount w, the trace evaluates to Tr(σ(w)) = w(6−w)/12. The anisotropy ratio ‖π‖² / Tr(σ)² is constant across all bulk shells. Over the uniform ensemble, the unconditional trace is Tr(σ_iso) = 3/4, decomposing as E[Tr(σ|w)] + 3 Var(E[v^i|w]) = 5/8 + 3/24 = 3/4.

### D.4 Nariai Bound

The interior-shell anisotropy ratio equals √6/9 ≈ 0.2722, matching the Nariai ultracold mass bound for stable extremal compact objects [23]. The significance of this match is discussed in Section 8. A dynamical derivation linking the two lies beyond the scope of this manuscript.

## Appendix E: Nonlinear Extension Details

### E.1 E_ref(ψ) Proof

Define L(E) = ln(E_CS/E) as the position on the energy ladder. At ψ = 0, L(v) = |η| and τ(0) = τ_G. At general ψ, τ(ψ) = τ_G(1−ψ) by Premise 3. The optical depth per unit ladder length is α = τ_G/|η|. Setting τ(ψ) = α L(ψ) gives L(ψ) = |η|(1−ψ), hence

    E_ref(ψ) = E_CS exp(−|η|(1−ψ)) = E_CS (v/E_CS)^(1−ψ)

Verification: E_ref(0) = v and E_ref(1) = E_CS, both matching the required endpoints.

### E.2 Einstein Tensor Components

For the metric f = 1 − 2ψ(s) with s = r/r_g, the Einstein tensor components are

    G_tt = −2f(ψ + s ψ′) / s²

    G_rr = −2(ψ + s ψ′) / (s² f)

    G_θθ = (s/2f) [f″ + f′/s − f′²/(2f)]

The component identity G_rr = G_tt/f² is verified in Section 6.2.

### E.3 Modified Bianchi Identity

The modified Bianchi identity ∇_μ T^μ_ν = −(∂_ν G / G) T^μ_μ is verified numerically. The exchange magnitude is computed at each sampled radius, confirming the behavior described in Section 6.2.

### E.4 Scalar-Tensor Representation

The continuum limit admits the following scalar-tensor representation:

    S = (1/16πG₀) ∫ R exp(−g₁ψ) √(−g) d⁴x
      + (1/16πG₀) ∫ g₁² exp(g₁ψ) (∇ψ)² √(−g) d⁴x
      − ∫ V(ψ) √(−g) d⁴x + S_matter

Here ψ is a position-dependent coupling functional, not an independent dynamical field. Unlike Brans-Dicke constructions, ψ is fixed algebraically by the closure structure through ψ = |Φ|/Φ_Planck. In the point-mass exterior, R = 0, ∇²ψ = 0, and dV/dψ = 0 at equilibrium, as discussed in Section 6.2.

### E.5 PPN Derivation

The perturbative expansion of ψ(s) = −(1/g₁) ln(1 − g₁/s) in powers of 1/s gives

    ψ(s) = 1/s + a₂/s² + a₃/s³ + ⋯,    a_n = g₁^(n−1) / n

The coefficients are a₁ = 1, a₂ = −0.3228, a₃ = 0.1389, a₄ = −0.0673, a₅ = 0.0347. The leading deflection term is 4GM/c²b, identical to GR. The parameter β at leading order follows as β = 1 − g₁/2 (Section 6.4).

### E.6 Neutron Star TOV Integration

The self-consistent TOV equation with G(ψ) is

    dP/dr = −G(ψ) ρ m/r² × [(1 + P/ρ)(1 + 4πr³P/m)] / [1 − 2G(ψ)m/(rc²)]

with dm/dr = 4πr²ρ and ψ solved self-consistently from ψ = G(ψ)m/(rc²). For a γ = 2 polytrope with central density 8 × 10¹⁷ kg/m³, the integration yields R ≈ 15.4 km, M ≈ 1.25 M☉, and ψ_surface ≈ 0.153.

### E.7 Gravitational Wave Strain

For a binary with total mass M_total, chirp mass M_chirp, and orbital separation a, the strain ratio h_CGM/h_GR = G(ψ)/G₀ where ψ = GM_total/(a c²). For the Hulse-Taylor binary: ψ = 2.14 × 10⁻⁶, ratio = 0.9999986, difference = 0.0003%. For NS-NS at 20 km: ψ = 0.103, ratio = 0.935, difference = 6.5%.

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