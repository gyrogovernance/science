# CGM Gravity Analysis: Ancestry, Quantization, and Compact Code Structure

## 1. Introduction

All gravitational theories inherit structural assumptions about space, motion, and identity from their predecessors. From Newtonian gravity to general relativity, physical bodies are treated as persistent entities moving within a pre-existing geometric framework whose dimensionality and symmetry are assumed rather than derived.

The Common Governance Model (CGM) derives spatial dimensionality and gravitational coupling from logical constraints required for coherent emergence and relational consistency. Gravitational behaviour is the preservation of relational traceability: distinguishable structure remains accountable to common origination through time as a consequence of closure.

Individuality is distinguishable variation across states; identity is continuity of structure across that variation.

**Definition:**

> Gravity is the identity of individuality originating from a common source.

Mass is accumulated common-origin structure. The field equations below express how that structure constrains geometry and coupling.

In Newtonian gravity [1], the inverse-square law and Poisson equation inherit the factor 4pi from the surface geometry of three-dimensional space. In general relativity, Einstein [4] replaced gravitational force with spacetime curvature, but the coupling constant kappa = 8piG/c^4 was still fixed by requiring consistency with the Newtonian limit. The same geometric inheritance appears in Nordstrom's scalar theory [3], Heaviside's gravitoelectromagnetism [2], Fierz-Pauli linearized gravity [5], and related continuum formulations. In each case, three-dimensionality and spherical closure are presupposed.

CGM derives these geometric facts from closure constraints. The framework yields the following primary structural consequences:

*   **Three-dimensional closure and Spin-2:** Three dimensions follow from the minimal consistency requirements of relational distinction, while the spin-2 character of gravitational interaction follows from a two-pass structure of orientation recovery under closure.
*   **The Quantum of Gravity:** The invariant Q_G = 4pi arises from full angular closure of the observational domain, acting as a geometric quantum analogous to Planck's constant but applied to closure rather than phase-space resolution.
*   **The Einstein Coupling Decomposition:** Every component of kappa = 8piG/c^4 is derived. The factor 4pi emerges from closure normalization (Q_G x m_a^2 = 1/2), the factor 2 emerges from Z2 holonomy in the algebraic kernel, and c^4 emerges from the four-stage depth structure.

The manuscript organizes these derivations into four main arcs. Sections 2 through 4 establish the foundational principles, the unobservability principle, and the shared aperture geometry linking light and gravity. Sections 5 through 9 derive the field equation and extract the discrete kernel invariants governing mass, chirality cancellation, and quadrupole radiation. Sections 10 and 11 assemble these invariants to reconstruct the relativistic continuum limits and predict the value of G to 0.074 ppm. Finally, Sections 12 through 14 outline testable predictions and implications.

Companion analyses supply independent results used here, including the dimensional proof [15] that CGM forces three dimensions and six degrees of freedom, the fine-structure constant derivation [24, 27], the UV-IR energy ladder [25, 26], the compact opacity construction [16], and the byte formalism [17]. The present manuscript is self-contained for the gravity-specific derivation while citing those results at the points of use.

### 1.1 Historical Context: The Three-Dimensional Stance

Einstein treated three-dimensional space and ordinary time as the natural descriptive basis for inertia and motion, while employing four-dimensional covariance as a calculational tool [22]. Poincare likewise regarded four-dimensional coordinate systems as geometrically legitimate yet held that three-dimensional language remained the appropriate medium for physical description [21].

CGM adopts that organizational stance and derives it from closure constraints. Three-dimensional space with six kinematic freedoms follows from the axiom chain; time enters as the sequential depth ladder CS to UNA to ONA to BU. Classical four-dimensional tensors remain a compact encoding of continuum limits, but the operative split here is spatial bookkeeping versus causal depth sequencing.

Gravitoelectromagnetism [2, 18, 19] and the ADM formalism [20] already formulate gravitational kinematics in spatial language. CGM adds a derivation-level account: the gyroscopic-carrier symmetry is fixed by contradiction-avoidance inside the algebraic kernel.

### 1.2 Notation and Invariants

The following symbols and invariants are used throughout this manuscript.

| Symbol | Meaning | Role |
|--------|---------|------|
| Q_G | Quantum of Gravity (4pi steradians) | Minimal solid-angle closure for coherent observation |
| m_a | Observational aperture parameter | Fixes closure normalization via Q_G x m_a^2 = 1/2 |
| Delta | Aperture gap (1 - rho) | Measures deviation from full closure |
| rho | Closure ratio (delta_BU / m_a) | Ratio of balanced holonomy to aperture |
| shell_k | Shell index k in {0,...,6} | ab_distance = 2k; shell_0 equality horizon, shell_6 complement horizon [16] |
| arch_shell | Kernel holonomy coordinate | arch_shell = 6 - shell_k; 0 = complement, 6 = equality |
| zeta | Gravitational coupling scale | zeta = 8/(m_a sqrt(3)) = 16 sqrt(2pi/3); equivalently Q_G/S_geo [27] |
| S_geo | Geometric mean action scale | S_geo = m_a pi sqrt(3)/2 = Q_G/zeta [27] |
| G_kernel | Kernel coupling constant (pi/6) | Normalized coupling appearing in boundary field and Einstein decomposition |
| tau_G | Gravitational optical depth | Attenuation factor for the gravitational coupling at the electroweak anchor |
| E_CS | Planck-scale UV anchor (E_CS = E_Planck) | CS stage energy from optical conjugacy; 1.22 x 10^19 GeV [25, 26] |
| v | Electroweak scale (246.22 GeV) | Infrared conjugate of the Planck scale and dimensional anchor for G |
| H | Boundary horizons (64 states each) | Define the holographic boundary of the kernel |
| Omega | Reachable manifold (4096 states) | Full state space of the finite algebraic kernel |

Shell indices follow the code-chart convention of the compact geometry analysis [16]: shell_k has ab_distance = 2k for k = 0,...,6, with shell_0 at the equality horizon and shell_6 at the complement horizon. Holonomy and gravity scripts use arch_shell = 6 - shell_k, so arch_shell labels run from complement (0) to equality (6). Popcount k in the binomial shell distribution C(6,k)/64 coincides with this shell index.

## 2. Foundations and the Unobservability Principle

### 2.1 Common Origination and Dimensional Structure

All distinguishable structure arises from a common origin while remaining traceable. Dimensionality, symmetry, and coupling emerge as conditions for maintaining traceability under variation.

Distinction requires differentiated states and coherent preservation across them. Chirality supplies a minimal directional asymmetry; further constraints prevent collapse into trivial identity.

Excluding absolute unity expands chirality into rotational structure. SU(2) [10] is the minimal non-abelian group supporting this, with three generators and a unique double cover of SO(3). Excluding isolated rotational frames requires translation, yielding SE(3) = SU(2) semidirect R^3. Orientation and displacement are coupled: every rigid motion rotates, then translates in a frame-dependent direction. Six degrees of freedom result (three rotational, three translational). The full derivation from the five CGM constraints appears in [15]; here we record the consequences.

Translational change, once committed to record, defines time. Rotational and translational sectors map, in the relativistic limit, to gravitomagnetic and gravitoelectric structure. Gravity preserves relational continuity from asymmetry to closure. Section 2.2 states the observational limits that follow.

### 2.2 The Unobservability Principle

Observation presupposes distinction, so common origination cannot appear as an observational object. The unobservability principle states that origination is the condition that enables observation and therefore cannot be observed directly. Decompositions that treat the observational frame as independent of the object break this constraint.

Perturbative quantum gravity treats gravity as a field on a fixed background, yet the background participates in defining gravity, so the split is inconsistent. The graviton is a structural feature of the closed frame, the closure relation itself. Companion work extends the same logic to other non-observable sectors.

Finite closure over the measurement domain is required. With Q_G = 4pi steradians as full angular closure, the minimal observational unit is fixed and conjugate quantities cannot be simultaneously definite without bound.

### 2.3 BU Duality: Egress Closure and Ingress Reconstruction

CGM is driven by a single axiom (CS) and the necessity chain that follows from it. BU has two logically distinct aspects that are simultaneous.

BU-Egress (closure) is the forward depth-4 balance condition that guarantees operational closure. BU-Ingress (reconstruction) is the requirement that the balanced state retains sufficient information to reconstruct the preceding distinctions (CS, UNA, ONA). Ingress is the reverse-read constraint implied by closure when common origination must remain traceable, acting as a structural necessity.

In the kernel realization, this duality takes a precise spectral form. The depth-4 canonical half-word W2 is an involution (W2^2 = id on Omega) that maps each complement-horizon state to a unique equality-horizon shadow. Egress reads this as closure achieved: the depth-4 commutator vanishes in the S-sector projection. Ingress reads it as memory preserved: each shadow uniquely reconstructs its origin via W2. These are two simultaneous readings of the same spectral property. The full canonical word F = W2 o W2' composes two depth-4 involutory passes through the K4 algebra, achieving carrier-level Z2 encoding (the distinction between rest and swapped coordinates on the complement horizon). Depth-8 is therefore K4 composition of two depth-4 involutions at the composition depth, adding no new modal depth.

## 3. The Quantum of Gravity

Q_G = 4pi steradians is the quantum of gravity: the minimal solid-angle closure for coherent observation in three dimensions, analogous to h-bar but for geometric closure rather than phase space. The graviton operator acts inside the closed frame; Q_G is the closure constant.

Normalizing the chiral seed pi/2 across SU(2) phases gives m_a^2 x (2pi)_L x (2pi)_R = pi/2, hence m_a^2 = 1/(8pi) and m_a = 1/(2 sqrt(2pi)) approximately 0.199471, and Q_G x m_a^2 = 1/2.

At depth four, 48 x Delta = 0.993578587835 with 48 = 4 x 12. The aperture is roughly one quantum per closure frame; at byte resolution the gap is 5/256. The ratio (1/48)/(1/32) = 2/3 encodes the tension between two chiral frames and three spatial axes.

The same 4pi appears in nabla^2(1/r) = -4pi delta^3(r) and in the closure requirement here: full angular completion of a spherical domain.

## 4. Light and Gravity

Light is abelian phase transport preserved by BU-Egress at the S-sector. Gravity is full orientation recovery under BU-Ingress.

> Light is how the universe explores its degrees of freedom, while Gravity is how the universe remembers and realizes the principle of freedom itself.

Electromagnetic and gravitational couplings share aperture geometry. With zeta = 8/(m_a sqrt(3)) = 16 sqrt(2pi/3) approximately 23.155 (equivalently Q_G/S_geo, S_geo = m_a pi sqrt(3)/2 [27]) and alpha_0 = delta_BU^4/m_a,

alpha_0 x zeta = rho^4 / (pi sqrt(3)) = 0.169025920321,

where rho = delta_BU/m_a and Delta = 1 - rho. The product cancels m_a: alpha_0 carries m_a in the denominator, zeta in the numerator. Laboratory alpha differs from alpha_0 by transport corrections in [24, 27]; here we keep alpha_0 for the kernel invariant.

Independent measurements of alpha and G can falsify CGM if their product violates rho^4/(pi sqrt(3)).

Gravity is weak because the aperture is small at Delta approximately 0.0207. The gravitational coupling requires full depth-8 recovery, which traverses a large optical depth and is therefore exponentially suppressed relative to the electromagnetic coupling that operates at the depth-4 residual.

### 4.1 The Speed of Gravity

In the weak-field regime, the gravitoelectromagnetic system of Section 5.4 implies wave propagation with characteristic speed c. Taking the curl of the gravitomagnetic equation and using the remaining identities yields a wave operator of the form (spatial Laplacian) minus (1 over c squared) times the second derivative in physical time tau. The characteristic speed is therefore fixed by the same constant c that appears in the source response normalization.

The multimessenger event GW170817 bounds any difference between the gravitational and electromagnetic propagation speeds to below 3 x 10^-15 of c, consistent with this weak-field prediction.

Static closure-density gradients in the gravitoelectric sector (Section 5.4) extend across space without wave propagation. Perturbations propagate at c through the gravitomagnetic sector via the orientation-correction operator. Gravitoelectromagnetism therefore separates a non-propagating static field from radiation at finite speed.

BU-Egress holds uniformly across continuous transition parameters, fixing one-parameter unitary groups and hence the propagation speed c.

## 5. Ancestry and the Field Equation

Ancestry density, ancestry credit, and a Poisson-type field equation formalize the preserving condition from the introduction.

### 5.1 Ancestry Density

Ancestry density rho_A is defined as accumulated common-origin structure per unit volume, representing the amount of structure that traces to common origination concentrated in a region of space. In standard physics this corresponds to mass-energy density. Energy is the measure of common-origin structure that has been committed to temporal record.

### 5.2 Ancestry Credit and the Gravitational Potential

The ancestry credit field C_A is defined as the non-negative scalar measuring accumulated common-origin structure at a point. The conventional gravitational potential is Phi = -C_A, with Phi <= 0 when the reference at infinity is C_A = 0. The acceleration field is g = -grad Phi = grad C_A. Approaching greater ancestry credit lowers the action required to maintain relational traceability.

### 5.3 The Local Divergence Law

Rotational invariance and linearity in R^3 uniquely determine a second-order operator of Laplacian form up to a scalar factor. This yields a Poisson-type equation for the gravitational potential.

The gravitational potential satisfies nabla^2 Phi = Q_G x G x rho_A, where Q_G = 4pi is the Quantum of Gravity. Equivalently, the ancestry credit field satisfies nabla^2 C_A = -Q_G x G x rho_A. With Q_G = 4pi from Section 3, this matches the standard Newtonian form. The derivation from the kernel Gauss law, including numerical verification of flux closure, is presented in Section 6.6.

Curvature is the observable gradient of closure-density. Uniform closure-density gives a flat geometry with vanishing gradient while gravity remains present. Varying closure-density produces curvature in the standard sense. Saturation occurs when closure-density is so high that the aperture closes and outward propagation is suppressed.

### 5.4 The Six-Degree-of-Freedom Extension

The SE(3) = SU(2) semidirect R^3 symmetry derived from the five CGM constraints decomposes the gravitational field into two sectors corresponding to the non-absoluteness of unity and the non-absoluteness of opposition. The gravitoelectric field g = -grad Phi carries the three translational degrees of freedom from R^3, which originate from the non-absoluteness of opposition. The gravitomagnetic field B_g = curl A_g carries the three rotational degrees of freedom from SU(2), which originate from the non-absoluteness of unity.

Here tau denotes the physical time parameter used in the weak-field limit, while the structural emergence of time is treated separately in Section 5.5. Together they satisfy the gravitoelectromagnetic system. The divergence of g equals -Q_G G rho_A. The curl of g equals -dB_g/dtau. The divergence of B_g equals 0. The curl of B_g equals -(Q_G G / c^2) J_A + (1/c^2)(dg/dtau), where J_A is the ancestry current.

In the weak-field regime, gravitational lensing is recovered through the same spin-2 mapping that connects the kernel to the linearized relativistic limit. A detailed derivation is part of the continuum mapping program listed in Section 13.

Heaviside [2] wrote these equations in 1893 as the gravitational analog of Maxwell's equations. They emerge rigorously from the weak-field limit of general relativity and reproduce the structural decomposition of the slow-motion gravitational limit.

The decomposition into gravitoelectric and gravitomagnetic sectors follows from the algebraic structure of displacement composition in the kernel [8]. Composing non-collinear displacements in a curved geometry is non-associative. The gyrogroup algebra corrects this non-associativity via an automorphism called the gyration operator [8]. In the continuous limit, accumulated gyration produces a circulation field.

The non-absoluteness of unity (UNA) activates this gyrocommutative structure. It ensures that the order of displacement composition matters, but in a bounded, coherent way governed by the gyration operator. The gravitomagnetic vector potential A_g is the continuous manifestation of accumulated gyration, encoding the orientation corrections required for translational displacements to compose coherently across extended domains. Without this accumulated gyration, displacement composition would fail to close, leaving the gravitoelectric sector algebraically incomplete.

The semidirect product SE(3) = SU(2) semidirect R^3 therefore decomposes the gravitational field into a displacement component from R^3 representing the gravitoelectric sector and an orientation-correction component from SU(2) representing the gravitomagnetic sector. The latter exists because displacement in a non-flat geometry necessarily generates rotational memory.

### 5.5 The Emergence of Time

Rotational freedom defines the space of possible orientations. Translational commitment defines irreversible relational history. Time emerges as preserved displacement memory, meaning once a translation is committed to record, erasing the trace reverses the commitment. The conversion from spatial freedom to temporal memory occurs at universal balance, where translational displacement becomes committed into recoverable record instead of remaining reversible variation.

Time has an arrow because it records the irreversible accumulation of committed distinctions. This allows the system to return to its original state, as the depth-8 cycle returns the carrier to rest with zero holonomy residue while preserving complete memory of the traversal. The irreversibility lies in the ordering: achieving balanced closure requires first establishing non-absolute distinctions, and those distinctions require a traceable common source. The arrow of time is a consequence of constraint ordering rather than energy dissipation.

The BU closure condition coordinates this accumulation into coherent cycles, giving time its periodic structure at the fundamental level. At the kernel level, this periodicity appears as the depth-8 carrier-return cycle. In the continuum, it appears as the causal structure of relativistic spacetime.

## 6. Kernel Invariants

The field law fixes the continuum form, while the finite kernel fixes the discrete normalization. The following sections extract the exact combinatorial invariants from the CGM kernel that anchor the gravitational coupling in the continuous theory. The kernel is a finite algebraic realization of the CGM closure conditions [17], providing exact combinatorial invariants verified by the aQPU wavefunction diagnostic (experiments/cgm_aqpu_wavefunction.py) and the gravitational coupling analysis (experiments/cgm_gravity_analysis_1.py). All numerical values are reproducible from these scripts.

### 6.1 The Reachable Manifold

The kernel is a finite algebraic system with 4096 reachable states organized into seven shells shell_k (k = 0,...,6) with ab_distance = 2k [16]. Shell_0 is the equality horizon (ab_distance 0); shell_6 is the complement horizon (ab_distance 12); intermediate shells carry increasing then decreasing distinction. The reachable manifold Omega contains these states. Shell populations follow |shell_k| = C(6,k)/64, yielding 64, 384, 960, 1280, 960, 384, 64 for k = 0 through 6. Gravity scripts record arch_shell = 6 - k when reporting holonomy paths.

### 6.2 The Shell Displacement Invariant

The shell displacement measures total distance traversed through shell space during a complete operational cycle. The wavefunction analysis (experiments/cgm_aqpu_wavefunction_2.py) establishes the structural origin of this invariant on all 4096 states. The depth-4 half-word W2 maps shell s to 6-s, producing a traverse of 6 per half-word from either constitutional pole. Gate F = W2 o W2' preserves shell, giving a path traverse of 12 per F-cycle (two half-words of traverse 6 each, with net displacement zero because the second half-word returns to the original pole). The holonomy cycle (carrier return) requires F o F = id, yielding a total path traverse of 24 per Z2 round-trip. The invariance of D = 24 across all 64 micro-reference configurations is therefore a consequence of the K4 operator algebra and the pole-swap structure of W2, confirmed by exhaustive verification.

The holonomy path traverse D = 24 is verified as a topological invariant across all 64 micro-reference configurations. It fixes the closed-cycle displacement normalization used in the Gauss law bridge (Section 6.6) and in the optical depth construction (Section 11.3).

In the finite kernel, the number of activated degrees of freedom serves as the discrete mass coordinate. More activated degrees of freedom mean more structure that has been committed to record and must be preserved. Mass, as accumulated common-origin structure, naturally increases with the number of active generators, making the count of activated degrees of freedom the kernel's discrete mass coordinate. Mass changes the shape of the trajectory through the shells while leaving the total closed-cycle displacement unchanged, because displacement is defined over closed cycles rather than state occupancy. The result is the discrete Gauss-law form. In standard gravity, Gauss's law relates flux through a closed surface to the enclosed source. Here, the total shell traversal per cycle is fixed at 24 independent of mass.

The kernel Gauss map connects this invariant to the Quantum of Gravity through Q_G = (pi/6) x 24 = 4pi.

Two independent derivations meet here. Q_G = 4pi comes from the horizon normalization Q_G m_a^2 = 1/2. The displacement invariant 24 comes from exhaustive verification across all 64 mass configurations. The kernel Gauss map G_kernel = Q_G/24 = pi/6 is the normalized coupling constant that appears in both the boundary field of the potential profile (Section 9) and the Einstein coupling decomposition (Section 10.1).

### 6.3 The Mass-Shell Relationship

The maximum shell excursion depth equals the number of activated degrees of freedom, meaning the discrete mass coordinate controls the intermediate shell excursion while the midpoint horizon contact remains fixed. The gravitational cycle circulates through all four closure phases, and every mass configuration reaches the equality horizon at the midpoint of the depth-4 half-cycle with arch_shell path 0, pop, 6, pop, 0 (arch_shell 6 at the equality horizon).

This fixed midpoint contact is independent of mass and serves as the kernel expression of the Weak Equivalence Principle. The kinematic trajectory's independence from accumulated mass structure guarantees the universality of free fall at the discrete level: all bodies reach the same equality-horizon condition at the midpoint of their closure cycle. In the continuous theory, gravitational and inertial mass coincide because both measure the same structural quantity. Local Lorentz invariance of the Strong Equivalence Principle enters through the SE(3) continuous limit (Section 5.4).

### 6.4 The Equality Horizon and Chirality Cancellation

At the midpoint of the gravitational closure cycle, all six chirality components flip simultaneously, producing a cancellation of directional bias. This simultaneous cancellation is the finite-kernel signature of the equality horizon in continuous physics, representing a condition where all directional biases cancel, all six degrees of freedom become transparent at once, and the system is maximally open to relational coupling.

At the equality horizon, the system's active and passive components become indistinguishable, and all six degrees of freedom simultaneously cancel their chirality. Gravity couples to the full 6-DOF cancellation at this horizon, while electromagnetism couples to the residual U(1) at depth four before full cancellation occurs.

The chirality bit statistics along the canonical depth-8 word confirm this structure. At bulk steps (1, 3, 5, 7), each of the six chirality bits has probability 0.5 of being set. At the complement horizon (steps 2, 6), all six bits are zero. At the equality horizon (steps 4, 8), all six bits are one. Both horizon states are pure trace, carrying zero STF amplitude. Bulk steps carry mixed chirality, producing the nonzero STF excitation.

The wavefunction analysis (experiments/cgm_aqpu_wavefunction_2.py) confirms that each depth-4 half-word fully inverts chirality: q(W2) = q(W2') = 63 for all micro-refs m. The full canonical word composes two such inversions, yielding q(F) = 63 xor 63 = 0. Gate F preserves chirality while acting non-trivially on the carrier, confirming that holonomy acts on the carrier subspace only, not on chirality.

### 6.5 Carrier Trace Theorems

The shell transition operator M_q for byte weight q acts on the seven-dimensional shell space. The carrier trace C(q) is Tr(M_q) when that trace is nonzero, and Tr(M_q^2) otherwise. The transition probability from shell w to shell t under weight q is P(w -> t | q) = C(w,t) C(6-w, q-t) / C(6,q), where the binomial term C(w,t) is zero when t lies outside the valid range.

For even weights q = 2k, the Chu-Vandermonde identity yields the diagonal sum:

Tr(M_{2k}) = sum_{w=0}^{6} P(w -> w | 2k) = [sum_w C(w,k) C(6-w,k)] / C(6,2k) = C(7, 2k+1) / C(6,2k) = 7/(2k+1).

This gives the even-shell carrier traces as exact rationals: C(0) = 7, C(2) = 7/3, C(4) = 7/5, C(6) = 1.

For odd weights, the byte swap alters shell parity, forcing Tr(M_q) = 0. The return trace C(q) = Tr(M_q^2) is obtained from the Krawtchouk spectral decomposition. The shell transition matrices are mu-symmetric with respect to the binomial measure mu(w) = C(6,w)/64, so the Krawtchouk polynomials K_w(k) form an orthogonal eigenbasis. The eigenvalues lambda_k of M_q are computed exactly as rationals, and C(q) = sum_k lambda_k^2 yields C(1) = C(5) = 28/9 and C(3) = 52/25. These values are verified in experiments/cgm_gravity_analysis_3.py by three independent routes: two-hop transition product (sequential shell transitions, then trace), matrix squaring, and spectral eigenvalue summation.

The odd-to-even trace ratios at corresponding shells are C(1)/C(2) = 4/3 and C(3)/C(4) = 52/35. The symmetry C(1) = C(5) = 28/9 reflects shell-space symmetry about the equator shell w = 3.

For bulk weights q = 1 through 5, Tr(M_q^8) converges to 2 under repeated application while horizon weights q = 0 and q = 6 retain Tr(M_0^8) = 7 and Tr(M_6^8) = 1. This is consistent with Z2 holonomy F o F = id: depth-8 carrier return splits the bulk shell dynamics into two invariant sectors, while the horizon shells are fixed by the holonomy template.

### 6.6 Gauss Law Bridge: From Kernel to Field Equation

The shell displacement invariant (Section 6.2) establishes that every closed operational cycle traverses total shell distance D = 24 independent of the mass configuration. This invariance is the discrete expression of the generatedness requirement of Assumption CS stating that all structure traces to a common source. The kernel realizes this by generating the full manifold Omega from the boundary H, yielding the holographic identity |H|^2 = |Omega|. In the continuum, generatedness becomes Gauss's law, tying exterior flux directly to the enclosed source.

The kernel Gauss map (Section 6.2) converts this pure integer to the dimensionless coupling G_kernel = Q_G/D = pi/6. The product D x G_kernel = Q_G = 4pi gives the total flux per cycle in solid-angle units.

To verify that this discrete Gauss law produces the continuum Poisson equation, the seven shell layers are embedded into a radial coordinate r in [0,1] with the binomial mass profile M(r) from Section 9. The gravitational field g(r) is computed from the potential gradient, and the boundary flux 4pi r^2 g(r) is evaluated at the outer shell.

The computed flux at the outer boundary is -6.579736. The expected value from the kernel Gauss law is -Q_G x G_kernel = -4pi x pi/6 = -2pi^2/3 approximately -6.579736. The ratio is 1.000000, confirming exact closure of the discrete flux through the boundary.

In the continuum limit, spherical symmetry gives div g = (1/r^2) d(r^2 g)/dr. From the discrete Gauss law, r^2 g(r) = -G_kernel M(r). Differentiating with dM/dr = Q_G r^2 rho(r) yields div g = -Q_G x G_kernel x rho(r).

Substituting G = G_kernel/E_CS^2 (Section 11.1) and identifying rho(r) as the ancestry density rho_A produces the CGM field equation div g = -Q_G x G x rho_A.

Equivalently, since g = -grad Phi and Q_G = 4pi, nabla^2 Phi = Q_G x G x rho_A.

This is the Poisson equation of Newtonian gravity derived from the kernel Gauss law rather than assumed from rotational invariance alone. The value Q_G = 4pi matches the aperture normalization from Section 3, confirming consistency between the continuous and discrete layers of the framework.

For a spherically symmetric mass distribution, the shell theorem guarantees an exterior field of the form g = G M_total/r^2. The binomial shell profile has spherical symmetry by construction, so this result applies exactly. Three independent numerical checks confirm the inverse-square behavior on the extended radial grid used in the gravity analysis script. First, the product |g| r^2 is constant across the exterior to machine precision with a relative standard deviation of 1.8 x 10^-16. Second, a least-squares fit of log|g| versus log r gives an exponent of -2.000000 plus or minus 9 x 10^-16. Third, the mean exterior flux 4pi r^2 g equals M x (-Q_G x G_kernel), where M = 0.611815 is the integrated continuum mass on this embedding.

## 7. Orientation Recovery and the Gravitational Factor Two

### 7.1 The W2 Involution and Z2 Holonomy

The kernel realizes BU-Egress and BU-Ingress through the spectral structure of the K4 operator algebra verified in experiments/cgm_aqpu_wavefunction_2.py.

The depth-4 canonical half-word W2 is an involution on Omega with eigenspace dimensions dim(+1) = dim(-1) = 2048. It maps each complement-horizon state to a unique equality-horizon state, and vice versa, establishing a perfect shadow pairing between the two constitutional poles. BU-Egress reads this involution as closure: W2^2 = id means the depth-4 operation squares to identity, confirming closure in the S-sector. BU-Ingress reads it as memory: the equality-horizon shadow of any complement-horizon state uniquely reconstructs that state via W2, confirming that the balanced state encodes the full prior chain.

Gate F = W2 o W2' composes two such involutions through the K4 algebra. On the carrier, F acts as the Z2 flip exchanging rest and swapped coordinates within each shell while preserving shell index. The holonomy cycle requires two applications of F to return the carrier to rest: F o F = id. This Z2 structure is the algebraic origin of the factor 2 in 8pi = 2 x Q_G and the spin-2 character of the gravitational sector.

For single-family words (families 00 or 11 alone), the depth-4 word acts as the identity on the carrier with zero holonomy signature. These achieve BU-Egress trivially without the Z2 encoding. The canonical four-family word is required for the holographic structure, where the carrier distinguishes between rest and swapped positions on the same complement-horizon shell. This distinction is not visible to chirality (both positions share the same chi6 value) and constitutes the holographic content of BU.

### 7.2 The Origin of the Factor Two

The Z2 holonomy F o F = id supplies the structural origin of the factor 2 in the Einstein coupling decomposition 8pi = 2 x Q_G. One application of F exchanges rest and swapped on the complement horizon. Two applications return to rest. The gravitational sector acts as the operator of recovery under closure, where one pass establishes the closed act and the second recovers the identity of the carrier.

In the wavefunction decomposition, the Hilbert space C^4096 splits into equal +1 and -1 eigenspaces of dimension 2048 under F. The rest state decomposes as |rest> = (|+> + |->)/sqrt(2). Under F, this becomes |swapped> = (|+> - |->)/sqrt(2). The holonomy phase (the sign distinguishing rest from swapped) resides in the relative sign of the +/-1 eigencomponents. The Z2 oscillation |rest> -> |swapped> -> |rest> is the carrier-level manifestation of spin-2 behavior.

The closure-phase family sequence for the canonical word is 0, 1, 2, 3, 0, 1, 2, 3. Its egress half produces net circulation +2, and its ingress half produces -2, canceling to zero. The net angular momentum per complete holonomy cycle is the sum of these circulations, yielding a field that transforms under the spin-2 representation.

On the finite manifold, ln|Omega| = 2 ln|H|, where the factor 2 is the Z2 holonomy from F o F = id (verified for all 64 micro-refs). The holographic entropy relation and the Einstein coupling factor share the same algebraic origin; Section 9 develops the Bekenstein-Hawking interpretation.

### 7.3 Masslessness

The graviton corresponds to the orientation-recovery operator. Mass is accumulated temporal depth representing what remains after conversion has been committed into the carrier. The graviton carries the conversion relation itself and is therefore massless. In standard physics, rest mass is associated with stored field excitations. Here the graviton is the operator of recovery.

## 8. Gravitational Radiation and Memory

The canonical word traces arch_shell path 0, 1, 6, 1, 0, 1, 6, 1 over a depth-8 cycle (arch_shell 6 is the equality horizon). This path exhibits two symmetric excursions to the equality horizon per half-cycle. Fourier decomposition of the shell displacement signal reveals the dominant spectral component at k = 2 with amplitude 1.5. Two equal peaks per cycle identify the quadrupole structure at the level of the finite shell dynamics. In the continuous limit, a spin-2 field radiates predominantly through quadrupole emission, and the kernel's dominant spectral mode at k = 2 is the discrete precursor of this continuum behavior.

Averaging over all 64 micro-references, the trace-free stress magnitude is nonzero on bulk steps and vanishes on horizon steps. This is consistent with the k = 2 quadrupole mode being the dominant radiative signature of the depth-8 cycle.

The depth-4 cycle terminates at the swapped coordinate with full Z2 holonomy signature (all bits set to 0xFFFFFF). This records complete retention of displacement memory, meaning the system has moved to the complementary orientation without returning. The depth-8 cycle returns to rest with zero holonomy residue at 0x000000, canceling all displacement memory.

An incomplete depth-8 recovery leaves a residual carrier displacement corresponding to the depth-4 Z2 holonomy signature at 0xFFFFFF. In gyrogroup terms, this residual is an unresolved gyration. The system has undergone egress but not ingress, leaving the orientation correction permanently embedded in the carrier state.

Gravitational memory in the CGM account is retained gyration from an incomplete closure cycle. In a Bruck loop, coherent closure requires resolving the gyration operator. An incomplete cycle leaves a gyration residual that persists because no subsequent operation has canceled it. In the continuous theory, this retained gyration manifests as the permanent metric displacement recorded by gravitational wave memory [12]. The spacetime geometry retains a twist from the passage of gravitational radiation, exactly analogous to the kernel retaining a swapped orientation from an incomplete depth-8 cycle.

Gravitational memory corresponds to incomplete BU-Ingress. BU-Egress alone can terminate with a non-zero Z2 holonomy signature representing a retained gyration. BU-Ingress is the resolving pass that cancels this signature. When the resolving pass is not completed, the residual persists as permanent displacement in the carrier bookkeeping. This aligns with the qualitative content of gravitational-wave memory, where a lasting geometric offset is produced by a process that has not returned the carrier to its pre-event orientation.

### 8.1 Gravitational Radiation as Three-Dimensional Field Oscillation

Standard general relativity describes gravitational waves as weak perturbations h_mu_nu on a Lorentzian metric. CGM describes the same signature as modulation of g and B_g from Section 5.4. Because both constructions coincide on quadrupole luminosity at leading post-Newtonian order, observable strain bookkeeping matches familiar GR formulas. The carrier semantics diverge precisely where perturbative quantization of h_mu_nu is ordinarily discussed.

Whereas standard general relativity treats the wave as a ripple in the spacetime metric itself, CGM identifies it as a coordinated modulation of ancestry credit and gyrational bookkeeping that corrects how translational composition closes on curved carrier data.

That semantic shift reframes the graviton puzzle. Perturbative quantization of h_mu_nu yields a massless spin-two particle whose coupling is dimensionful and whose scattering series is non-renormalizable in the usual power-counting sense [5]. Inside CGM, the quantum bookkeeping is embedded at the carrier level. Q_G = 4pi enforces closure quanta and the aperture gap Delta approximately 0.0207 sets the finest resolved observation grain. The graviton is therefore identified with the orientation-recovery operator that closes the depth-eight carrier cycle (Section 7), replacing the plane-wave mode atop a separately classical metric. Its masslessness, helicity-two profile, and effective undetectability as a resolved particle follow from kernel algebra rather than from a second-quantization recipe applied to h_mu_nu.

This three-dimensional formulation aligns gravity with the spatial-slice structure of quantum field theory, as noted in Section 1.1.

## 9. The Potential Profile

The finite shell structure gives a regularized interior potential profile, with ancestry distributed across the seven binomial shells. The resulting profile has finite central credit, a boundary field equal to G_kernel, and the expected sign relation Phi = -C_A.

The two boundary horizons H with 64 states each and the full reachable manifold Omega with 4096 states satisfy the holographic identity |H|^2 = |Omega|, meaning 64^2 = 4096. This identity follows from the self-dual [12,6,2] code structure of the kernel developed in the companion compact geometry analysis [16]. The discrete entropy relation is ln|Omega| = 2 ln|H|. This is the exact discrete analogue of the Bekenstein-Hawking entropy bound: bulk degrees of freedom (ln|Omega|) scale strictly with boundary degrees of freedom (ln|H|), driven by the Z2 holonomy. The factor 2 is the Z2 holonomy from F o F = id, the same algebraic origin as the factor 2 in 8pi = 2 x Q_G. The boundary carries the full combinatorial content of the interior configuration.

The kernel mass distribution follows the binomial profile C(6,k)/64 across the seven shells. The six degrees of freedom are controlled independently. In the absence of external bias, each degree of freedom is equally available for activation. The binomial distribution reflects the structural equiprobability of the six independent SE(3) generators.

The continuous profile computed from the binomial density yields a finite value at the origin. M(0) = 0.0156 is the enclosed binomial shell fraction at the innermost sampled radius, regularized by discrete shell occupancy rather than a point-mass delta. The field at the boundary equals G_kernel = pi/6, preparing the matching condition for the exterior solution.

The binomial mass distribution produces a regularized interior potential with finite central value and a 1/r far field at distances exceeding the effective source radius, consistent with the standard Newtonian limit for extended sources.

## 10. Connection to General Relativity

### 10.1 The 8pi and c^4 Decomposition

The Einstein coupling is kappa = 8piG / c^4. CGM decomposes the numerator as 8piG = 2 x Q_G x G, so that kappa = (2 x Q_G x G) / c^4. Here Q_G = 4pi is the quantum of gravity (Section 3). The factor 2 comes from the Z2 holonomy F o F = id, where the canonical word F = W2 o W2' exchanges rest and swapped coordinates on the complement horizon and must be applied twice for carrier return (experiments/cgm_aqpu_wavefunction_2.py).

In SI units, the stress-energy component T_00 has dimensions of energy density. Mass density carries one fewer power of c than energy density, so expressing the gravitational source as energy density introduces a factor c^2 relative to mass density. The Einstein field equation couples T_{mu nu} to curvature (via the Einstein tensor). The proportionality constant kappa = 8piG/c^4 is fixed so that, in the Newtonian limit, the potential equation nabla^2 Phi = 4pi G rho_mass holds with rho_mass the mass density. The c^4 in the denominator is therefore the standard unit conversion between energy-density sources and curvature response, not an additional CGM postulate. Within CGM, the four-stage depth structure offers a separate interpretation of why two quadratic conversions appear in the same closure chain.

The CGM decomposition of the Einstein coupling separates into four structural contributions.

| Term | CGM role |
|------|----------|
| 2 | Z2 holonomy: F o F = id for carrier return |
| Q_G = 4pi | closure solid angle |
| G | ancestry coupling |
| c^4 | SI matching: energy-density source to curvature in kappa = 8piG/c^4 |

The integer-to-constant chain proceeds as follows. The shell displacement invariant fixes the pure integer 24 through exhaustive verification (Section 6.2). The Gauss map converts this to the dimensionless coupling G_kernel = Q_G/24 = pi/6 (Section 6.2). Dimensional analysis then gives G = G_kernel/E_CS^2 where E_CS is the energy scale at which the gravitational coupling is evaluated (Section 11.1). Every step from the integer 24 to the dimensionful G passes through geometric invariants (Q_G) and dimensional anchors (E_CS) with no free coefficients.

### 10.2 Newtonian and Linearized Limits

The static limit of the field equation yields nabla^2 Phi = 4pi G rho_A with the standard Newtonian potential as solution. The full six-degree-of-freedom gravitoelectromagnetic system has the same structural decomposition as the weak-field, slow-motion limit of general relativity. In general relativity, kappa = 8piG/c^4. Here, kappa = 2 Q_G G / c^4, which matches the Fierz-Pauli normalization [5] through 16piG / c^4 = 2 kappa.

### 10.3 Nonlinear Extension

The nonlinear self-consistency condition, where ancestry density modifies the geometry through which ancestry accumulates, is the natural extension of the linear theory [6]. This extension follows the same path from Poisson to Einstein that the historical lineage traverses.

## 11. The Coupling Constant

This section assembles the gravitational coupling from kernel invariants (Sections 6-7), continuum matching (Sections 5 and 10), and results established in companion analyses. The UV source scale and optical-conjugacy ladder are fixed geometrically with E_CS at the Planck scale and v at the electroweak scale [25, 26]. The fine-structure constant and its laboratory value are derived in [24, 27]. The tau_G exponents align with the five shell checkpoints and four transitions of the depth-4 word [17] and with the six-mode opacity grading of [16]. The forward prediction G_pred = G_kernel x exp(-tau_G) / v^2 uses only those geometric inputs and v. Measured G is used for validation and for the inverse scale check below, not as a fit parameter in the prediction chain. The electromagnetic-gravitational link through alpha x zeta = rho^4/(pi sqrt(3)) is established in Section 4 and tested in Section 12.1.

The Planck scale is excluded as an input to the derivation, and h-bar enters only through unit conversion into GeV units. The dimensional anchor for the predicted gravitational constant is the electroweak scale v = 246.22 GeV.

### 11.1 Source Scale

The structural relation G = G_kernel / E_CS^2 holds at the UV focus, where G_kernel = Q_G / 24 = pi/6 and E_CS = E_Planck = 1.22 x 10^19 GeV is the CS-stage anchor from optical conjugacy E_i^UV x E_i^IR = (E_CS x E_EW)/(4pi^2) with E_EW = v = 246.22 GeV [25, 26]. This identity must not be inverted with measured G to obtain G: Planck units presuppose G and that route is circular. The laboratory coupling is predicted by G_pred = G_kernel x exp(-tau_G) / v^2 with tau_G from kernel invariants (Section 11.3); measured G validates that prediction only.

### 11.2 Dimensionless Gravitational Coupling at the Electroweak Anchor

At the electroweak scale, the dimensionless gravitational coupling is alpha_G(v) = G_meas x v^2 = 4.067168 x 10^-34. The electroweak scale is the appropriate anchor because the optical conjugacy relation identifies it as the infrared conjugate of E_CS. The gravitational coupling at v measures the optical attenuation between these two conjugate foci of the same closure structure. In optical-depth form, alpha_G(v) = G_kernel x exp(-tau_G). The Z2 holonomy from F o F = id assigns depth 2 ln(E_CS / v) to the conjugacy ladder. The leading-order kernel form fixes tau_G to within 2.46 x 10^-5 of tau_required; the full form including the Delta^4 term (Section 11.3) closes the residual to 7.36 x 10^-8. The fourth-degree conversion factor c^4 in the Einstein coupling belongs to the source-response matching described in Section 10.1.

On the Delta ruler anchored at the electroweak scale, the dimensionless gravitational coupling occupies position n_G = -log2(alpha_G(v))/Delta = 3714.3. This is approximately |Omega| x rho^5 = 3689.5 ticks from the anchor, with ratio n_G / (|Omega| x rho^5) = 1.007, placing the gravitational sector deep in the relational bulk of the reachable manifold rather than near a horizon shell. For comparison, the electroweak mass coordinates of the top quark, Higgs, Z, and W bosons occupy positions n = 24.7, 47.2, 69.2, and 78.0 on the same ruler (compact geometry analysis, order-5 Delta placement).

### 11.3 Aperture Optical Depth and Gravitational Coupling

The optical-depth construction for G operates on the full BU dual. Electromagnetic dynamics associates with the depth-4 residual sector, where closure proceeds without full carrier recovery. The gravitational coupling requires the depth-8 completion (BU-Egress followed by BU-Ingress), accumulating the full attenuation associated with reconstruction under closure. Gravity is weak because it is tied to the completion of recovery, which carries a large optical depth.

Define tau_G as a cumulative per-cycle leakage cost along the depth-8 holonomy. Independent closure cycles compose multiplicatively in survival amplitude: if each cycle transmits a factor exp(-tau_cycle), then N_cycles compose as exp(-N_cycles tau_cycle). Equivalently, log-survival adds across cycles, giving a macroscopic factor exp(-tau_G) when tau_G aggregates the total depth. This is the same structural rule as optical depth in radiative transfer, where tau is additive and intensity scales as exp(-tau). The identification alpha_G(v) = G_kernel exp(-tau_G) and G_pred = G_kernel exp(-tau_G)/v^2 is therefore a modelling step that maps kernel leakage depth to the dimensionless coupling at the electroweak anchor. By contrast, tau_cycle/Delta = 7591/7392 is a kernel theorem (Appendix C.1).

The leading closed-form optical depth is

tau_G0 = |Omega| Delta rho^5 (1 - 4 rho Delta^2),

where |Omega| = 4096, Delta = 1 - rho, and rho = delta_BU/m_a. The factor (1 - 4 rho Delta^2) is the K4-stage symmetric correction (Appendix C).

A separate kernel-derived additive correction reduces the residual further:

delta_tau = |Omega| Delta rho^5 c_4 Delta^4,    c_4 = -7/4,

with c_4 fixed by ancestry stress (Route A) and K4 closure charge (Route B), independent of the (1 - 4 rho Delta^2) factor. Odd-order corrections such as Delta^3 vanish identically under Z2 holonomy parity; Delta^4 is the leading term from soft Z2 breaking by the trace sector (experiments/cgm_gravity_analysis_2.py, Section S1). The full model is

tau_G = tau_G0 + delta_tau = |Omega| Delta rho^5 [(1 - 4 rho Delta^2) + c_4 Delta^4].

Three independent arguments converge on the exponent 5 in rho^5. From representation theory, a symmetric 3 x 3 stress in three spatial dimensions decomposes into one trace component and five independent symmetric trace-free components, and coherent survival across this five-dimensional sector produces rho^5. From the kernel shell structure, two of the seven shell layers are horizon shells with vanishing trace-free magnitude and five are bulk shells with nonzero trace-free magnitude. The wavefunction analysis (experiments/cgm_aqpu_wavefunction_2.py) confirms this directly: gate F preserves shell (Z2 within pole), while the shell-population census shows horizon shells shell_0 and shell_6 (equality and complement) with ||pi|| = 0 and bulk shells shell_1 through shell_5 with ||pi|| > 0, giving exactly 5 anisotropy-carrying channels. From the cycle geometry, the rho exponent equals 8 - 3 = 5 via the two-lemma factorization of the optical depth (Appendix C). All three routes yield the same exponent without reference to G.

The K4 correction factor (1 - 4 rho Delta^2) is the lowest-order symmetric correction consistent with the depth-4 stage structure. It is even in Delta and assigns equal weight to the four stage transitions. The per-family depth-4 optical depth computation gives identical tau_word across all four family phases with zero variance, supporting equal contribution per stage channel. The numerical magnitude matches the observed attenuation shift between |Omega| Delta rho^5 and tau_G.

Route A gives c_4 = -(1 + Tr(sigma_iso)) = -(1 + 3/4). Route B gives the same value from the K4 closure charge on gyroscopic edge increments.

Numerical evaluation: tau_G0 alone gives a 25 ppm offset in G relative to the reference measurement; adding delta_tau with c_4 = -7/4 leaves a residual of 7.36 x 10^-8 in tau, hence 0.074 ppm in G_pred = 6.7088095 x 10^-39 GeV^-2 versus G_meas = 6.7088100 x 10^-39 GeV^-2.

**Sensitivity.** Because G_pred = G_kernel exp(-tau_G)/v^2, a fractional change in tau maps directly to the same fractional change in G with opposite sign: d ln G_pred / d tau_G = -1. At the nominal ledger (delta_BU, m_a from Section 1.2), numerical evaluation gives d tau_G/d Delta approximately 3.67 x 10^3 and d tau_G/d rho approximately 3.89 x 10^2, so a relative perturbation of 10^-6 in Delta shifts tau_G by roughly 5 x 10^-5 relative, and the same perturbation in rho shifts tau_G by roughly 5 x 10^-6 relative. The prediction is therefore stable at the precision to which Delta and rho are fixed in the constant ledger; the sub-ppm G residual is not driven by fine-tuned cancellation among poorly determined inputs.

This agreement is far tighter than present experimental uncertainty on G (order 10^-5 relative in CODATA 2018 [13]). The comparison is a consistency check against the chosen reference value, not a metrological verification at 0.074 ppm. A decisive test requires substantially improved G measurements or an independent observable that constrains the same tau_G structure.

## 12. Testable Predictions

### 12.1 The alpha-zeta Product

The kernel invariant is alpha_0 x zeta_geom = rho^4 / (pi sqrt(3)) = 0.169025920321, where alpha_0 = delta_BU^4 / m_a is the kernel electromagnetic coupling and zeta_geom = 8/(m_a sqrt(3)) = 16 sqrt(2pi/3) is the gravitational coupling scale [27]. The product is exact at the kernel layer: alpha_0 and zeta_geom are defined together so that m_a cancels and only rho and geometric constants remain.

The laboratory fine-structure constant alpha_CODATA (alpha^-1 = 137.035999084(21) [11]) differs from alpha_0 by +319 ppm (+0.0316%). Inverting the same identity with alpha_CODATA gives zeta = rho^4/(pi sqrt(3) alpha_CODATA), which exceeds zeta_geom by 3.19 x 10^-4 relative. That offset equals (alpha_0/alpha_CODATA - 1) and is not a violation of the kernel product. The transport-corrected fine-structure value from [24, 27] matches alpha_CODATA to sub-ppb accuracy; gravity and the alpha-zeta kernel invariant use alpha_0 and do not incorporate that correction chain.

Given alpha_CODATA, the product predicts zeta approximately 23.1626. The gravitational coupling scale zeta determines G through G = G_kernel x exp(-tau_G) / v^2. Any independent constraint on G or zeta that disagrees with this propagation, after explicit identification of which alpha definition is used, falsifies the stated layer of the framework.

### 12.2 The alpha(z) Oscillation

The shell opacity structure modulates the effective electromagnetic coupling across cosmological optical depth. Mapped to redshift via the Delta-ruler, this produces a predicted oscillation in the fine-structure constant. The predicted alpha(z) oscillation has period Delta-z approximately 0.0143 and peak-to-peak fractional amplitude approximately 5 x 10^-4 relative to alpha_0, with seven sub-cycles per main period from the shell structure yielding a sub-cycle period of Delta-z approximately 0.0020. Precision spectroscopy of quasar absorption systems provides the primary test. The period is fixed by the aperture gap Delta, while the amplitude depends on the projection from shell weighting to the laboratory coupling.

A survey spanning at least one full period in ln(1+z) and detecting no oscillation at 3-sigma confidence with the stated period would falsify this prediction at the stated leading-order amplitude scale.

### 12.3 Variance in G Measurements

Different experimental methods for measuring G yield systematically different values. Such scatter is also central to the missing-mass and modified-gravity literature [7]. The 2018 CODATA value [13] is 6.67430(15) x 10^-11 m^3 kg^-1 s^-2, while atom interferometry gives 6.67191(99) x 10^-11, and torsion balance experiments cluster near 6.674 x 10^-11. CGM predicts that systematic offsets among G measurement methods could correlate with effective shell weighting of the experimental configuration. Deriving the method-to-shell map would sharpen this into a quantitative prediction. This prediction concerns path-dependence rather than time-dependence, and supernova constraints on the time variation of G remain consistent with CGM.

### 12.4 The Sign of Higher-Order Corrections to alpha

Dual-pole symmetry requires the next correction to the fine-structure constant prediction to be negative. A positive O(delta_BU^6) correction at the Thomson limit would falsify the geometric identification.

### 12.5 Binary Pulsar Orbital Decay

The k = 2 spectral mode identified in Section 8 supplies the required quadrupole structure for gravitational radiation. In the weak-field limit, this reproduces the standard quadrupole emission form. Binary pulsar timing provides a precision test of the CGM-to-relativistic mapping once the strain normalization in the nonlinear extension is fixed.

## 13. Implications

The results above carry implications beyond the immediate gravitational coupling.

Quantum gravity programs face a structural obstacle in perturbative quantization: gravity acts as the closure condition of the frame, making the quantization of h_mu_nu as an independent excitation circular. The kernel construction encodes gravity in the holonomy of the carrier recovery instead. This renders quantum structure gravitational. Q_G = 4pi functions as a geometric quantum, and the aperture Delta sets the minimal observational grain. The framework is quantum-mechanical at foundation, independent of a graviton propagator.

Philosophically, the derivation treats three-dimensional spatial structure as logically prior to four-dimensional spacetime. Time enters through the irreversible commitment of displacement into record, maintaining the fourth coordinate as a computational device rather than an ontological primitive. This stance aligns with Poincare [21] and the ADM formalism [20], and CGM provides a derivation-level justification. The algebraic kernel forces the 3+1 split through the SO(3)/SU(2) shadow projection. Four-dimensional tensors remain a valid packaging of the continuum limit, encoding a distinction between spatial and temporal domains that the kernel makes algebraically explicit.

Experimentally, the alpha-zeta product and the alpha(z) oscillation provide two independent falsification channels using existing instrumentation. Quasar absorption spectra and precision G measurements suffice for initial tests. The alpha(z) prediction is particularly sharp, featuring a specific period tied to the aperture gap and an amplitude at leading order determined by shell-weight variance. A null result at the predicted period and amplitude scale would eliminate the geometric connection between shell opacity and electromagnetic coupling.

More broadly, the derivation suggests that coupling constants may be consequences of observational closure rather than free parameters. The gravitational coupling follows from kernel invariants, aperture geometry, and one energy anchor. The electromagnetic coupling follows from the same aperture geometry at a different depth. Their product is fixed by the closure ratio alone. If this pattern extends, other couplings may yield to similar constructions. The strong and weak interactions would correspond to deeper shell structures or different carrier projections, each constrained by the same kernel grammar. This is speculative, but the systematic nature of the construction makes it a concrete research program.

## 14. Conclusion

This manuscript derives the gravitational field equation, the coupling constant structure, and the spin-2 character of gravitational interaction from CGM closure constraints. It also derives spatial dimensionality, spherical geometry, and gravitational coupling from first principles.

The derivation rests on three pillars. First, the shell displacement invariant D = 24 verified exhaustively across all 64 mass configurations establishes the discrete Gauss law and fixes G_kernel = pi/6. Second, the depth-4/depth-8 holonomy distinction proves that the gravitational cycle requires a two-pass carrier return, supplying the factor 2 in 8pi = 2 x Q_G and the spin-2 angular momentum structure. Third, the optical-depth model tau_G = tau_G0 + |Omega| Delta rho^5 c_4 Delta^4 with tau_G0 = |Omega| Delta rho^5 (1 - 4 rho Delta^2) and c_4 = -7/4 yields G_pred within 0.074 ppm of the reference measurement, with the exponent 5 confirmed by three independent arguments.

What remains open is the nonlinear extension. The present framework recovers the Poisson equation and the gravitoelectromagnetic system, corresponding to the Newtonian and weak-field relativistic limits. The full Einstein equations require showing that ancestry density modifies the spatial geometry through which ancestry accumulates, producing a self-consistent curved manifold. This is the natural continuation, moving from Poisson to Einstein following the same logical path that the historical development traversed but now with the coupling constants and dimensional structure already in hand.

A further open question is whether the kernel grammar extends to the other fundamental interactions. The electromagnetic coupling is already accounted for through the depth-4 residual sector. The strong and weak interactions would require identifying their corresponding carrier projections within the kernel structure. The fact that the six se(3) generators separate cleanly into rotational and translational sectors, and that the shell structure produces both a monopole with 1 component and quadrupole with 5 components decomposition, suggests that the kernel contains sufficient spectral richness to accommodate the full Standard Model. Realizing this would complete the program implied by the alpha-zeta product, establishing all couplings as consequences of observational closure geometry.

The unobservability principle imposes a final constraint on the entire program. Any theory of gravity must reckon with the fact that the conditions enabling observation cannot themselves be fully observed. CGM makes this principle explicit and derives its consequences. The result is a theory of gravity as the preserving condition for distinguishable structure to remain traceable to common origination. Whether this reframing proves productive beyond the results already obtained depends on the outcome of the testable predictions stated in Section 12 and on the completion of the nonlinear extension.

---

## Appendix A: CGM Foundational Constraints

The Common Governance Model is formalized as a propositional modal logic with two primitive modal operators [L] and [R] representing recursive operational transitions, evaluated over Kripke frames F = (W, R_L, R_R, V) with serial accessibility relations and nonempty valuation V(S). The system consists of five foundational constraints: one assumption (CS), two lemmas (UNA, ONA), and two propositions (BU-Egress, BU-Ingress). The conjunction of the two propositions defines universal balance (BU).

**Primitive symbols:**

| Symbol | Description |
|--------|-------------|
| S | A propositional constant: the horizon constant |
| not | Logical connective: negation |
| implies | Logical connective: material implication |
| [L] | Modal operator: left transition |
| [R] | Modal operator: right transition |

**Defined symbols:**

| Symbol | Definition |
|--------|------------|
| phi and psi | not(phi implies not psi) |
| phi or psi | not phi implies psi |
| phi iff psi | (phi implies psi) and (psi implies phi) |
| (L)phi | not[L]not phi |
| (R)phi | not[R]not phi |
| Box phi | [L]phi and [R]phi |
| Diamond phi | (L)phi or (R)phi |

**Reading the expressions:**

[L]phi means phi holds after a left transition.
[R]phi means phi holds after a right transition.
Box phi means phi holds after both transitions.

Modal depth refers to the nesting level of modal operators. Depth-two operations exhibit contingent behavior (non-absolute unity and opposition), while depth-four operations achieve necessary closure (universal balance).

**Core definitions:**

| Concept | Formula | Description |
|---------|---------|-------------|
| Unity (U) | [L]S iff [R]S | Left and right transitions yield equivalent results at the horizon constant |
| Two-step Equality (E) | [L][R]S iff [R][L]S | Depth-two modal compositions commute at the horizon constant |
| Opposition (O) | [L][R]S iff not[R][L]S | Depth-two modal compositions yield contradictory results at the horizon constant |
| Balance (B) | [L][R][L][R]S iff [R][L][R][L]S | Depth-four modal compositions commute at the horizon constant |

**Absoluteness:** Abs(phi) = Box phi (phi is invariant under both transitions). NonAbs(phi) = not Box phi.

**The five foundational constraints:**

| Constraint | Type | Formula |
|------------|------|---------|
| CS | Assumption | S implies ([R]S iff S and not([L]S iff S)) |
| UNA | Lemma | S implies not Box E |
| ONA | Lemma | S implies not Box not E |
| BU-Egress | Proposition | S implies Box B |
| BU-Ingress | Proposition | S implies (Box B implies ([R]S iff S and not([L]S iff S) and not Box E and not Box not E)) |

**CS (The Source is Common):** Right transitions preserve the horizon constant S while left transitions alter it. This establishes fundamental chirality.

**UNA (Unity is Non-Absolute):** At depth two, the order of transitions matters but this non-commutativity is not absolute, which prevents homogeneous collapse.

**ONA (Opposition is Non-Absolute):** At depth two, opposition is not absolute, which prevents irreconcilable contradiction.

**BU-Egress (Depth-Four Closure):** At depth four, the system achieves commutative closure. This closure is absolute (holds across all accessible worlds).

**BU-Ingress (Memory Reconstruction):** The balanced state at depth four contains sufficient information to reconstruct all prior conditions, including the original chirality (CS), the contingent unity (UNA), and the non-absolute opposition (ONA).

**Independence:** In the core modal system (Kripke semantics), all five constraints are logically independent. Each admits counterexample frames falsifying it while preserving the others. Consistency is verified via a three-world Kripke frame satisfying all five simultaneously.

**In the operational regime** with continuous flows, reachability from S, and simple Lie closure, UNA and ONA follow from CS. The two-layer structure (modal axioms plus operational requirements) prevents circular reasoning. The modal axioms do not assume continuous physics, but continuous physics requires satisfying the operational requirements that the modal axioms specify.

## Appendix B: Ancestry Stress Tensor and Source Decomposition

The ancestry stress sigma_A is computed from the translational payload bits. In the reference implementation, payload bit 6 drives P_x, bit 5 drives P_y, and bit 4 drives P_z. Define the translational activation vector v = (b6, b5, b4) with components in {0, 1}. The ancestry stress is the centered second moment:

sigma_A^{ij} = average(v^i v^j) - average(v^i) x average(v^j),

where the averages are taken over the micro-reference ensemble within a cell.

With this definition, diagonal entries measure per-axis variance, and off-diagonal entries measure correlations among axes. In the uniform ensemble over all 64 micro-references, sigma_A is isotropic with diagonal entries 1/4 and trace 3/4 in kernel units, and its off-diagonal entries vanish. In shell-conditioned ensembles, the off-diagonal entries are nonzero and negative because the total six-bit population is fixed, which induces anticorrelation among the translational activations.

The ancestry current J_A is the mean signed translational activation vector. Using the signed encoding v_pm = (2b6 - 1, 2b5 - 1, 2b4 - 1), the uniform ensemble gives J_A = 0 by symmetry, while shell-conditioned ensembles produce nonzero J_A with magnitude determined by shell index.

The rotational payload bits (bits 3, 2, 1) label the SU(2) generator sector and determine the structure of the gravitomagnetic potential A_g in Section 5.4. They are not used in the translational covariance definition of sigma_A.

The tensor decomposes into isotropic and trace-free parts:

sigma_A^{ij} = p_A delta^{ij} + pi_A^{ij},

where p_A = (1/3) Tr(sigma_A) and pi_A^{ij} is symmetric and trace-free.

The trace component represents isotropic pressure. The five independent components of pi_A^{ij} form the trace-free sector corresponding to the l = 2 representation of SO(3). The constancy of the fractional anisotropy across all interior shells reflects the spectral uniformity of this trace-free sector within the discrete chirality sphere.

This 1+5 decomposition matches the 3+1 ADM source split at the level of spatial tensors, with rho_A as scalar density, J_A as vector current, and sigma_A^{ij} as spatial stress.

Conditioning on micro-reference popcount w, the translational activation vector has expectation E[v_i | w] = w/6 and covariance Cov(v_i, v_j | w) = -w(6-w)/180 for i not equal to j. The ancestry stress trace evaluates to the exact rational Tr(sigma(w)) = w(6-w)/12. The anisotropy ratio ||pi||^2 / Tr(sigma)^2 = 2/75 is constant across all bulk shells. Over the uniform 64 micro-reference ensemble, the unconditional trace is Tr(sigma_iso) = 3/4, decomposing as E[Tr(sigma|w)] + 3 Var(E[v_i|w]) = 5/8 + 3/24 = 3/4.

## Appendix C: Optical Depth Construction

### C.1 The Two-Lemma Factorization

The exact per-cycle optical depth is a kernel theorem derived from the binomial-weighted holonomy transport. Over the 64 micro-references with Z2 holonomy bulk steps, the cycle depth evaluates to the exact rational:

tau_cycle = (7591/7392) Delta,

where 7591 = (1/2) sum_{k=1}^{5} C(6,k)^3 and 7392 = 8 C(12,6).

The per-cycle depth sums shell weights over the four bulk steps of the Z2 holonomy template. For a micro-reference at popcount k, all four bulk steps land on shell k, contributing 4 x C(6,k)/64 per step. Weighting by the ergodic measure C(6,k)/64 and summing over the C(6,k) micro-refs at each popcount gives:

tau_cycle / Delta = [4 x sum_k C(6,k)^3 / 64^2] / [sum_k C(6,k)^2 / 64]
                  = 4 x sum_k C(6,k)^3 / (64 x sum_k C(6,k)^2).

With sum_{k=1}^{5} C(6,k)^3 = 15182 and sum_{k=0}^{6} C(6,k)^2 = 924, this evaluates to 60728 / 59136 = 7591/7392. The identity sum_k C(6,k)^2 = C(12,6) = 924 follows from the Vandermonde convolution with a = b = 0, confirming 7392 = 8 x C(12,6).

The transport measure tau_cycle/Delta = 7591/7392 and the STF anisotropy measure 5/99 (from ||pi||^2 with the same binomial weighting) differ by the exact rational factor K = 22773/1120. The structural origin of K from the Hamming spectrum remains open; it encodes the relationship between shell-transition return probability and dipole-pair anisotropy under the binomial weighting.

The earlier uniform approximation tau_cycle = 48 Delta^2 / rho^3 exceeds the exact value by approximately 3%, with ratio tau_exact / tau_uniform = 0.9707.

The number of depth-8 cycles is N_cycles = (|Omega| rho^5 (1 - 4 rho Delta^2)) / (7591/7392). The product N_cycles x tau_cycle = |Omega| Delta rho^5 (1 - 4 rho Delta^2) = tau_G, confirming exact agreement with the leading-order closed form.

### C.2 Series Expansion in Delta

Expanding the closed form tau_G = |Omega| Delta (1 - Delta)^5 (1 - 4(1-Delta) Delta^2) as a polynomial in Delta produces a finite series through degree nine. The coefficient sequence is:

c_n / |Omega| = [0, 1, -5, 6, 14, -55, 79, -60, 24, -4]

The series converges to the closed form exactly (maximum absolute difference zero at machine precision). The coefficients satisfy the identity c_n = |Omega| x [coefficient of Delta^n in (1 - Delta)^5 (1 - 4 Delta^2 + 4 Delta^3)].

### C.3 Delta-Four Correction

The additive correction delta_tau = |Omega| Delta rho^5 c_4 Delta^4 with c_4 = -7/4 is an exact kernel invariant (Route A and Route B in Section 11.3). Including delta_tau reduces the tau residual from 2.46 x 10^-5 to 7.36 x 10^-8, corresponding to 0.074 ppm in G relative to the reference value.

### C.4 Per-Family Optical Depth

The per-family depth-4 optical depth is identical across all four family phases, with tau_word = 0.009408891 and zero variance. This uniformity supports the equal-weight assignment in the (1 - 4 rho Delta^2) correction factor.

### C.5 Cycle Count

The holonomy-cycle count (F o F round-trips) is N = tau_G / tau_cycle approximately 3586.5, where tau_cycle is the optical depth per Z2 round-trip.

## Appendix D: Anisotropy Bounds and the Nariai Limit

The maximum anisotropy ratio of 1/sqrt(3) approximately 0.577 arises when exactly one or two translational degrees of freedom are active. This value follows directly from the three translational degrees of freedom of SE(3). In d spatial dimensions, the corresponding maximum fractional anisotropy is 1/sqrt(d).

For the full six-bit shell structure, the interior-shell anisotropy ratio equals sqrt(6)/9 approximately 0.272166. This numerical value matches the Nariai ultracold mass bound for stable extremal compact objects [23].

The Nariai solution represents a limiting configuration in which the black hole and cosmological horizons coincide, corresponding to maximal isotropy of the stress distribution. The equality recorded here is a numerical identity between a kernel invariant and an extremal bound in gravitational theory. A dynamical derivation linking the two lies beyond the scope of this manuscript.

## Appendix E: References

[1] Newton, I. (1687). Philosophiae Naturalis Principia Mathematica. London: Royal Society.

[2] Heaviside, O. (1893). A gravitational and electromagnetic analogy. The Electrician, 31, 281-282 and 359.

[3] Nordstrom, G. (1913). Zur Theorie der Gravitation vom Standpunkt des Relativitatsprinzips. Annalen der Physik, 347(13), 533-554.

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

[21] Poincare, H. (1912). L'espace et le temps. Scientia, 12, 159-170.

[22] Pais, A. (1982). 'Subtle is the Lord...': The Science and the Life of Albert Einstein. Oxford: Oxford University Press.

[23] Chen, C., Li, B. and Wang, A. (2024). Mass bounds of compact objects in gravity theories. Physical Review D, 109, 084025.

[24] Korompilias, B. (2025). The fine-structure constant from geometric first principles. Companion analysis (docs/Findings/Analysis_Fine_Structure.md).

[25] Korompilias, B. (2025). Energy scale structure in the Common Governance Model. Companion analysis (docs/Findings/Analysis_Energy_Scales.md).

[26] Korompilias, B. (2025). CGM units from observational geometry. Companion analysis (docs/Findings/Analysis_CGM_Units.md).

[27] Korompilias, B. (2025). CGM constants and aperture geometry. Companion analysis (docs/Findings/Analysis_CGM_Constants.md).