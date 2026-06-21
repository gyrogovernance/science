# Geometric Prerequisites for Quadratic-Gravity Inflation from Combinatorial Axioms

**Basil Korompilias**

*Technical Note for Discussion, March 2026*

---

## Abstract

Quantum quadratic gravity (QQG) with running couplings produces viable inflation but takes seven structural inputs as assumptions: the presence of R² and C² terms, the direction of RG flow, the renormalization scale μ ~ |R|^{1/2}, a large effective field count N ~ 10⁵–10⁶, the IR emergence of GR, the initiation of inflation, and the confinement of the Weyl ghost. The Common Governance Model (CGM) [7], a Hilbert-style axiomatization that derives physical structure from a single axiom through a four-stage operational sequence, provides constructions for the first six. The Weyl ghost is partially addressed. The sharpest quantitative disagreement with QQG is the tensor-to-scalar ratio. CGM predicts r ≈ 0.0024, below the QQG strong-coupling bound r ≳ 0.01, yielding a two-way experimental test.

**Keywords:** quadratic gravity, inflation, asymptotic freedom, variable coupling, combinatorial derivation

---

## 1. The Problem

Quadratic gravity extends the Einstein-Hilbert action with R² and C² terms [1]. Liu, Quintin, and Afshordi (LQA) [6] showed that QQG with running couplings produces slow-roll inflation compatible with current CMB constraints [10, 11], followed by a kination phase and IR emergence of GR. The scenario requires N ~ 10⁵–10⁶ effective matter fields and predicts r ≳ 0.01 to avoid strong coupling.

CGM proposes an origin for seven structural ingredients appearing in the QQG inflationary scenario: (i) the presence of R² and C² terms, (ii) the direction of RG flow, (iii) the renormalization scale μ ~ |R|^{1/2}, (iv) a large effective field count N ~ 10⁵–10⁶, (v) the IR emergence of GR, (vi) the initiation of inflation, and (vii) the confinement of the Weyl ghost. CGM provides constructions for (i) through (vi). Input (vii) is partially addressed.

## 2. CGM and the aQPU Kernel

CGM [7] is a Hilbert-style axiomatization of fundamental physics and information science. From a single axiom (the Common Source, CS), three further conditions follow as lemmas and propositions: Unity-Non-Absolute (UNA), Opposition-Non-Absolute (ONA), and Balance Universal (BU, decomposed into BU-Egress and BU-Ingress). Together they form a four-stage operational sequence that generates physical structure as a logical necessity. CGM contains a derivation of three spatial dimensions and six degrees of freedom. Within the CGM modal system, the BCH expansion of the depth-four commutator forces the generated Lie algebra to close on three generators as sl(2); the simplicity requirement from BU-Ingress excludes direct-sum algebras such as so(4); and the GNS construction selects the compact real form su(2), giving UNA three rotational generators [7]. The bi-gyrogroup consistency required by ONA forces the semidirect product SE(3) = SU(2) ⋉ R³ with three translational parameters. The gyrotriangle closure condition δ = π − (π/2 + π/4 + π/4) = 0 excludes n = 2 and n ≥ 4 [7]. The unique dimension is n = 3, with six degrees of freedom.

The solid angle required for coherent observation in three dimensions is the quantum-gravity invariant Q_G = 4π steradians. In CGM, Q_G = 4π is the ratio of two parameters fixed by the operational conditions before any geometric structure is assumed; the solid angle of a sphere in three dimensions is a downstream realization of this algebraic fact, not its source. Q_G is the flux quantum that fixes the discrete Gauss law in the kernel.

The aQPU kernel [8] is the executable specification of the four-stage sequence (CS, UNA, ONA, BU). It is a deterministic 24-bit system whose state (A, B) consists of two 12-bit components, each a 2×3×2 binary grid encoding three spatial axes (the rows X, Y, Z) crossed with two chirality frames. The six oriented bit-pairs of this grid (the dipole pairs) are the six degrees of freedom. The kernel's input unit is a byte, an 8-bit instruction that decomposes into a 6-bit payload driving the six dipole pairs and a 2-bit family controlling the chirality phase. The reachable state space Ω contains |Ω| = 4,096 = 64² states, organized into seven concentric shells indexed by k = 0,…,6. A shell is a constant-Hamming-distance stratum of Ω; the binomial profile C(6,k) × 64 reflects the six dipole pairs of the chirality register. The two boundary shells (k = 0, where A and B are bitwise complements and chirality is maximal, and k = 6, where A = B and chirality vanishes) carry zero symmetric trace-free (STF) anisotropy. The five intermediate shells (k = 1,…,5) carry nonzero STF anisotropy. Gravity couples to these five shells only. Every invariant cited below is confirmed by exhaustive enumeration of Ω.

The derivations below use the kernel's combinatorial invariants, derived from the four-stage sequence, to fix the structure of the gravitational field.

## 3. Variable Coupling and Asymptotic Freedom

In CGM, gravity is formally defined as the emergent balance that preserves ancestry through freedom of identity (UNA) and individuality (ONA). BU is the depth-four closure that lets these two extremes coexist; preserving ancestry under operational displacement requires that this closure be algebraically complete. The closure fixes the dimensionless coupling through the discrete Gauss law G_kernel = Q_G / D = π/6, where D = 24 is the shell-displacement invariant, that is, the total Hamming distance through shell space traversed by a complete holonomy cycle averaged over the 64 chirality configurations. The dimensional scale is G = G_kernel exp(−τ_G) / v², where v is the electroweak scale and τ_G is the refractive depth. A constant coupling cannot realize this closure when the local field strength varies; the coupling must acquire position dependence.

The derivation proceeds in three steps.

**Step 1: The discrete Gauss law.** The continuum limit of CGM's discrete flux law produces a modified Poisson equation ∇ · g = −Q_G G(ψ) ρ, where ψ = |Φ|/Φ_Pl is the gravitational depth, that is, the dimensionless ratio between the Newtonian potential Φ and its Planck-normalized reference, ranging from 0 at flat space to 1 at the Planck scale. The flux quantum Q_G = 4π is the CGM invariant introduced in Section 2.

The construction uses ψ as the coupling coordinate. The linearity in ψ follows from the first-order redshift factor (1 − ψ) common to CGM and GR in the weak-field limit, and ψ connects the electroweak and Planck anchors of CGM's energy scale structure through the UV-IR pairing relation E^UV × E^IR = (E_CS × v)/(4π²).

**Step 2: Refractive depth and position dependence.** G acquires position dependence because gravitational propagation through the kernel's shell structure is attenuating. The five bulk shells carry STF anisotropy; the two boundary shells do not. Coherent propagation across the five bulk shells attenuates the signal by a factor of ρ⁵ per holonomy cycle, where ρ ≈ 0.9793 is the closure ratio of the four-stage cycle. Accumulating this attenuation across the full depth gives the refractive depth

    τ_G = |Ω| Δ ρ⁵ (1 − 4ρΔ² + c₄Δ⁴) ≈ 76.24,

where Δ = 1 − ρ ≈ 0.02070 is the aperture gap, the residual 1 − ρ after the four-stage cycle, and c₄ = −7/4 is fixed by two independent kernel computations (the isotropic stress trace and the closure charge on gyroscopic edge increments). The closed form is the Regge sum of plaquette deficit angles over the five bulk shells, evaluated analytically. The attenuation varies with gravitational depth as τ(ψ) = τ_G(1 − ψ). Compounding the per-cycle attenuation yields the exponential coupling

    G(ψ) = G_kernel exp(−τ_G) exp(τ_G ψ) / E_ref(ψ)² = G₀ exp(g₁ ψ),

where E_ref(ψ) is the reference-energy function of Section 5. The factor 2 ln(v/E_CS) in

    g₁ = τ_G + 2 ln(v/E_CS) = −0.6456 < 0

comes from the E_ref(ψ)² denominator: at ψ = 0 the reference energy is v (electroweak), at ψ = 1 it is E_CS (Planck). The sign g₁ < 0 follows because the attenuation from STF propagation exceeds the scale-shifting effect of E_ref(ψ). The weak-field coupling G₀ = G_kernel exp(−τ_G)/v², with v the single measured input; the predicted value lies within 0.074 ppm of the CODATA value [7].

**Step 3: Asymptotic freedom.** With μ = E_ref(ψ) (Section 5),

    β_ln G = d ln(G/G₀) / d ln μ = −τ_G / ln(E_CS/v) ≈ −0.0168 < 0.

The sign follows from the attenuation direction. The numerical value differs from the perturbative one-loop coefficient.

This addresses prerequisites (ii) and (vi). Asymptotic freedom follows from the attenuation direction, and inflation is initiated at ψ = 1 (the Planck-scale end of the energy scale) where curvature is maximal.

Embedding the seven shells into a radial coordinate, with the shell population C(6,k)×64 as the radial mass profile, reproduces the continuum Poisson equation. The boundary flux matches the product of Q_G and G_kernel to relative precision 10⁻¹⁶. Three independent numerical checks confirm the inverse-square behaviour: the product of the field magnitude and the square of the radius is constant across the exterior.

## 4. R² Terms and IR Recovery of GR

A position-dependent coupling cannot be written as pure Einstein-Hilbert gravity. The minimal covariant action is

    S = (1/16πG₀) ∫ R exp(−g₁ ψ) √(−g) d⁴x,

where ψ = |Φ|/Φ_Pl is algebraically determined by the metric through the Newtonian potential Φ. Because ψ is a function of the metric rather than an independent field, the theory has no dilaton.

On a homogeneous background the de Sitter relation μ = R^(1/2) makes ψ linear in ln R, so the Taylor expansion of the exponential gives

    S_eff ≈ (1/16πG₀) ∫ [ R + c_R² R² + O(R³) ] √(−g) d⁴x,

with c_R² = −g₁ / (2 ln(E_CS/v)) ≈ 8.40 × 10⁻³. The R² term is the leading correction that any position-dependent coupling necessarily produces. At ψ = 0.95 (UV), |R²|/|EH| = 0.61 and R² dominates. At ψ = 0.05 (IR), |R²|/|EH| = 0.03 and Einstein-Hilbert dominates. The effective quadratic coupling is ξ_eff = 1/c_R² ≈ 119.1.

This addresses prerequisites (i) and (v).

## 5. The Renormalization Scale

LQA identifies μ = |R|^{1/2} [6]. CGM provides an explicit reference-energy function

    E_ref(ψ) = E_CS (v / E_CS)^{1−ψ},

interpolating between the electroweak scale v (ψ = 0) and the Planck scale E_CS (ψ = 1). The functional form follows from CGM's UV-IR pairing relation E^UV × E^IR = (E_CS × v)/(4π²): the energy at logarithmic position ψ along the scale running from v to E_CS, with tick spacing set by the aperture gap Δ, is E_ref(ψ). On a near-de Sitter background this reduces to μ = |R|^{1/2}, and the identification ψ(R) is invertible.

This addresses prerequisite (iii).

## 6. The Field Count and Its Weights

LQA requires N ~ 10⁵–10⁶ matter fields weighted by N = (1/60) N_scalar + (1/5) N_vector + (1/20) N_fermion. These weights are standard in one-loop effective actions, where they arise from graviton-matter vertex counting.

CGM produces the same weights from the kernel's pairwise defect algebra. The derivation of N_eff relies on the assumption that gravitational and matter degrees of freedom are governed by the same combinatorial algebra. Under that assumption, the number of effective field degrees of freedom is fixed by the kernel's symmetry quotient.

When two byte operations are composed, the difference between the actual result and the commutative expectation is the defect, measured by the Hamming weight of the XOR difference between the two 6-bit chirality words A⊕B. The number of ordered byte pairs at defect weight 2 is 4 · C(6,2) = 60, where the factor 4 comes from the 4-to-1 fiber structure (each of the 64 chirality codewords is the image of exactly 4 bytes). This 60 decomposes as 1 + 3 + 3: one trace component (the scalar part of the stress decomposition, spin-0), three translational components (vector, spin-1), and three rotational components (fermion, spin-1/2). The fractions 1/60, 3/60, 3/60 reproduce the QFT weights.

Applying these weights to a 32-bit extension of the kernel yields N_eff ≈ 1.15 × 10⁵, within the LQA window. The 24-bit carrier fails the first-order spectral triple condition, the SU(3) sextet bracket, and the sixth-grade W-boundary closure; the 32-bit lift restores all three. The 8-bit lifting carries the family-index and holographic-projection registers that the gravity-only kernel does not need but the field count does. The Klein four-group K₄ = {id, S, C, F} of involutions (gate S swaps A and B; gate C complement-swaps them; gate F is their composition, the global complement; gate id is the depth-four alternation identity) acts on the extended space, and quotienting by this symmetry, the family fiber, and the holographic projection produces the final count.

This addresses prerequisite (iv).

## 7. The Weyl Sector and the Ghost

The kernel's Z₂ holonomy gate F, the global complement (A, B) → (A ⊕ 0xFFF, B ⊕ 0xFFF), is one of the four intrinsic K₄ gates of the kernel. It splits the state space into even- and odd-parity eigenspaces of dimension 2,048 each. The even-parity sector carries R² content; the odd-parity sector carries C² content at a computed anisotropy ratio of 2/75 relative to the even sector, fixing λ = 75 ξ. The ratio 2/75 is computed from the sum of STF weights over odd-parity shell configurations. On FRW backgrounds, the Weyl tensor vanishes identically by conformal flatness, so the C² sector does not affect the inflationary background. It enters tensor perturbations at suppressed weight 2/75. The spin-2 ghost is not eliminated covariantly.

## 8. Observables and the Testable Discrepancy

The UV effective action is f(R) = R + a₂ R² with a₂ = 8.40 × 10⁻³. Standard Einstein-frame machinery gives N_e ≈ 55.7, n_s ≈ 0.972, r ≈ 0.0024. The scalar amplitude A_s ≈ 1.59 × 10⁻⁹ follows from the holographic projection Π_H = ρ⁸ Δ⁴ / (π² |Ω|) ≈ 3.84 × 10⁻¹², where |Ω| = 4,096 is the size of the reachable state space. Π_H is the kernel realization of the |H|² = |Ω| identity, with H the 64-state boundary.

The effective theory is Starobinsky-like, and the prediction r ≈ 0.0024 is close to the standard Starobinsky value r ~ 12/N_e² ≈ 0.003. The shift from the Starobinsky baseline arises because CGM's ξ_eff ≈ 119.1 differs from the canonical Starobinsky value ξ = 1/(8 a₂) that would give r ~ 0.003, and the variable-coupling structure contributes an additional tensor suppression factor exp(2 g₁ ψ_infl) ≈ 0.28, where ψ_infl ≈ 0.95 is ψ during inflation. The combined effect shifts r from 0.003 to 0.0024.

The tensor ratio r ≈ 0.0024 differs from the LQA bound r ≳ 0.01 by a factor of ~4. This is a two-way experimental test. If next-generation CMB experiments detect r ≳ 0.01, the CGM mapping developed here is excluded. If they measure r ≈ 0.002–0.004, the strong-coupling bound of [6] is in tension with observation.

**Sensitivity.** The prediction r ≈ 0.0024 depends on τ_G through g₁ and ξ_eff. A 1% shift in τ_G shifts g₁ by ~0.76 and ξ_eff by ~0.8%, moving r by ~1.6%.

## 9. What CGM Derives and What It Assumes

| Claim | CGM derivation | External input |
|:---|:---|:---|
| Q_G = 4π as flux quantum | D · G_kernel = Q_G is an exact combinatorial identity | Identification with gravitational flux |
| G(ψ) = G₀ exp(g₁ ψ) | Exponential from compounding attenuation; g₁ < 0 from attenuation direction | ψ = \|Φ\|/Φ_Pl as coupling coordinate |
| Asymptotic freedom | g₁ < 0 fixes β < 0 | None |
| μ → \|R\|^{1/2} | Functional form E_ref(ψ) | UV-IR pairing relation E^UV × E^IR = (E_CS × v)/(4π²) |
| Weights 1/60, 1/5, 1/20 | Partition 1+3+3 of 60 from defect algebra | Spin mapping (trace→scalar, translational→vector, rotational→fermion) |
| N_eff ≈ 1.15 × 10⁵ | Count from 32-bit lifted algebra | Holographic projection Π_H = ρ⁸Δ⁴/(π²\|Ω\|) |
| R² dominance in UV | From variable coupling + Taylor expansion on homogeneous background | Homogeneous-background reduction ψ = ψ(R) |
| r ≈ 0.0024 | ξ_eff and exp(2 g₁ ψ) suppression | Variable-coupling action |

---

*Reproducibility.* All numerical results are generated by `experiments/aqpu_gravity_analysis_8.py` through `aqpu_gravity_analysis_10.py` in the archived repository [7]. The kernel invariants cited above are verified by exhaustive enumeration of the 4,096-state space.

## References

[1] K. S. Stelle, Phys. Rev. D **16**, 953 (1977).

[2] E. S. Fradkin and A. A. Tseytlin, Nucl. Phys. B **201**, 469 (1982).

[3] A. Codello and R. Percacci, Phys. Rev. Lett. **97**, 221301 (2006).

[4] M. R. Niedermaier, Phys. Rev. Lett. **103**, 101303 (2009).

[5] D. Buccio, J. F. Donoghue, G. Menezes, and R. Percacci, Phys. Rev. Lett. **133**, 021604 (2024).

[6] R. Liu, J. Quintin, and N. Afshordi, Phys. Rev. Lett. **136**, 111501 (2026).

[7] B. Korompilias, Common Governance Model: Mathematical Physics Framework, Zenodo (2025). DOI: 10.5281/zenodo.17521384.

[8] B. Korompilias, Gyroscopic ASI aQPU Kernel Specification (2025).

[9] B. Korompilias, CGM Energy Scale Structure and Sterile Neutrino Non-Observability (2025).

[10] Y. Akrami et al. (Planck Collaboration), Astron. Astrophys. **641**, A10 (2020).

[11] T. Louis et al. (Atacama Cosmology Telescope Collaboration), J. Cosmol. Astropart. Phys. **11**, 062 (2025).
