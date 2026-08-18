# Analysis: CGM Precession

## Frame Transport in the Common Governance Model: Pairwise Precession on Stage Geometry

**Citation:** Korompilias, B. (2025). Common Governance Model: Mathematical Physics Framework. Zenodo. https://doi.org/10.5281/zenodo.17521384

**Verification:** `experiments/cgm_precession_analysis_run.py` (companions `_1.py`, `_2.py`). Results are written to `experiments/cgm_precession_analysis_results.txt`.

---

## Abstract

Precession is the rotation of a carried reference frame that accumulates when a system traverses a closed path. In special relativity the same effect appears as Thomas precession, whose geometric source is the constant negative curvature of relativistic velocity space. A closed loop in that space encloses hyperbolic area, and the frame transported around the loop rotates by an angle equal to that area. This document establishes the corresponding precession structure of the Common Governance Model by placing the model thresholds in the Einstein velocity ball and measuring the holonomies that result.

The Common Governance Model supplies four thresholds from its foundational conditions. The Unity Non-Absolute, Opposition Non-Absolute, and Balance Universal thresholds are realized as three mutually orthogonal Einstein speeds, while the Common Source threshold serves as the orientation frame because its value lies outside the open unit ball. Composition of each orthogonal pair of payload boosts produces one elementary Wigner rotation. The three elementary angles are the Ungar gyrotriangle defects of the origin-based stage triangles. Each angle obeys the half-angle product identity whose factors are the Poincaré half-rapidity radii of the two stages involved. Inversion of the three product identities recovers the three stage radii from the three measured angles, so a carried frame that records the pair precessions reconstructs the full radial configuration of the payload placement.

The dual-pole loop that visits the Opposition Non-Absolute stage and the two Balance poles doubles the Opposition–Balance corner. The resulting holonomy is the balance-channel angle δ_BU, which serves as the Regge deficit unit in the gravitational construction and as the base of the leading electromagnetic coupling in the fine-structure analysis. The dual-pole loop rooted at Unity Non-Absolute doubles the Unity–Balance corner and supplies the orthogonal rotation-channel holonomy. The palindromic traversal through all payload stages preserves the balance angle while conjugating its axis by the Unity–Opposition elementary precession. Path memory therefore separates into two independent components: Opposition–Balance fixes the magnitude, and Unity–Opposition fixes the orientation.

The same loops are evaluated under three transport prescriptions. Fermi–Walker transport, origin gyration, geodesic mass-shell transvection, the Ungar defect, and the Cartesian Thomas path-ordered exponential agree on the intrinsic holonomy. Composition of successive boosts in one fixed inertial frame leaves a residual velocity on every bent path and yields a finer angle spectrum. Spherical-chart readouts of the Thomas connection differ from the intrinsic angle by a coordinate offset at the chart poles. Enumeration of all closed walks of length two through five on the four payload points produces exactly six Fermi–Walker holonomy values, each realized as a word in the three elementary generators and the identity.

The balance holonomy depends on the translational threshold and the balance amplitude through a closed constitutive relation. The closure ratio ρ is that holonomy measured in units of the balance amplitude, and the aperture gap Δ is the complementary rate deficit. Linear, secant, and tangent gains decompose the aperture into a baseline fixed by Opposition geometry, a finite-amplitude correction, and a residual opening at the operating point. Realization controls that replace Einstein speeds by rapidities, or that equalize the two large stage speeds, change the numerical angles while preserving axis orthogonality, confirming that the reported constellation belongs to the Einstein-speed placement.

The analysis therefore supplies the pairwise precession geometry of the Common Governance Model, the intrinsic holonomy of its balance channel, the steering of that holonomy under palindromic transport, and a transport classification aligned with standard relativistic practice. The shared geometric invariants are the inputs consumed by the gravitational and electromagnetic constructions of the wider program.

---

## 1. Precession as Physics

A gyroscope carried around a closed orbit returns pointing in a different direction. This fact appears throughout physics under several names. Thomas precession, identified in 1926, supplies the factor of one half in the spin-orbit coupling of atomic fine structure. The Wigner rotation, formalized in 1939, is the spatial rotation that accompanies the composition of two non-collinear Lorentz boosts. Geodetic precession, measured by Gravity Probe B, is the corresponding effect for a gyroscope in orbit around a mass. Muon storage rings measure spin precession rates to parts per billion. In every case the observable is the same kind of quantity, an angle of frame rotation accumulated per closed cycle of motion.

The geometric source of relativistic precession has been understood since Borel's 1913 observation that the space of relativistic velocities is a hyperbolic space of constant negative curvature. A closed loop in velocity space encloses hyperbolic area, and the frame carried around the loop rotates by an angle equal to that area. Thomas precession is therefore a holonomy, the parallel-transport rotation of the curved mass shell, and Palge and Pfeifer (Physical Review A 109, 032206, 2024) give the modern statement of this identification for spin-1/2 particles.

The Common Governance Model derives four stage thresholds from its foundational conditions. The Common Source threshold is θ_CS = π/2, the Unity Non-Absolute threshold is u_p = 1/√2, the Opposition Non-Absolute threshold is o_p = π/4, and the Balance Universal amplitude is m_a = 1/(2√(2π)). The Einstein-ball realization places the three payload thresholds as speeds in the open unit ball of relativistic velocity space, with UNA on the first axis, ONA on the second, and the two BU poles on the third. The Common Source threshold is the orientation frame. Its value π/2 lies outside the open unit ball, so CS enters as the common frame of the three payload velocities.

The holonomy angle has two inseparable roles. It is the physical rotation experienced by the carried frame, and it is the measurement of the transport memory retained by the closed path. The aperture amplitude m_a converts that angle into a rate on the geometric clock of the stage hierarchy. Physical time enters through the measured Planck constant.

The same structure appears under several established names. Thomas precession describes the rotation of a carried frame under continuous acceleration. Wigner rotation describes the finite rotation generated by composing boosts. Geodetic precession describes frame transport through curved spacetime. Gauge holonomy describes the rotation of an internal state under transport through a gauge connection. CGM places these phenomena within one operational requirement. A displaced state must preserve its ancestry while remaining distinguishable from its origin. Translation produces displacement, and rotation preserves the directional record of that displacement. Holonomy carries the residual memory when a sequence closes.

The object under transport is the local rest frame of a massive state, its spin frame, and the local tetrad of the gravitational continuum.

---

## 2. Stage Thresholds and Kinematics

The three stage angles close a Euclidean triangle with vanishing defect.

```
θ_CS  = π/2
θ_UNA = arccos(u_p) = π/4
θ_ONA = π/4

θ_CS + θ_UNA + θ_ONA = π
```

Here θ_CS, θ_UNA, and θ_ONA are the constitutional stage angles. Their sum equals π, so the triangle CS–UNA–ONA has zero Euclidean defect. This identity fixes the relation among the stage angles before motion is introduced.

The continuous realization then assigns the UNA, ONA, and BU threshold numbers to three orthogonal Einstein velocities,

```
UNA = (u_p, 0, 0)
ONA = (0, o_p, 0)
BU+ = (0, 0, +m_a)
BU- = (0, 0, -m_a)
```

with u_p = 1/√2, o_p = π/4, and m_a = 1/(2√(2π)). The two BU poles occupy the same axis with opposite sign. The constitutional triangle has zero Euclidean defect. The velocity-space triangles formed by these points have positive hyperbolic defects. That distinction is the separation between operational constitution and relativistic composition. Precession is the physical expression of the positive hyperbolic defects.

Each payload threshold, read as an Einstein speed β in units of the speed of light, carries the standard relativistic kinematic dictionary. The Lorentz factor is γ = 1/√(1 − β²). The rapidity is η = atanh(β), the hyperbolic distance of the stage from rest. The Poincaré half-rapidity radius is

```
k(β) = β / (1 + √(1 − β²)) = tanh(η / 2).
```

The function k maps an Einstein speed to the Poincaré-ball coordinate of that stage. It is the natural variable of the precession formulas below, because the Wigner half-angle of two perpendicular boosts is the product of their k values.

| Stage | β | γ | η | k |
|---|---:|---:|---:|---:|
| UNA | 0.707107 | 1.414214 | 0.881374 | 0.414214 |
| ONA | 0.785398 | 1.615533 | 1.059306 | 0.485116 |
| BU | 0.199471 | 1.020508 | 0.202182 | 0.100748 |

The UNA radius has the closed form k(1/√2) = √2 − 1, the silver-ratio conjugate. The UNA proper velocity γβ equals unity, so UNA sits at the point where coordinate velocity and proper velocity coincide in magnitude. At this point p = mc, the reduced de Broglie wavelength equals the reduced Compton wavelength. UNA is the Compton-momentum condition of the carried state. These closed forms follow directly from the threshold values.

---

## 3. Pairwise Stage Precessions

### 3.1 Elementary angles

The elementary precession of a stage pair is the Wigner rotation produced by composing the two stage boosts from the origin. For each of the three payload pairs UNA–ONA, ONA–BU, and BU–UNA, this rotation angle coincides with the Ungar gyrotriangle defect of the triangle formed by the origin and the two stage points, which is the hyperbolic area of that triangle in curvature units.

```
ω_UO = defect(origin, UNA, ONA) = 0.396601502221 rad  (22.7236°)
ω_OB = defect(origin, ONA, BU+) = 0.097671089129 rad  ( 5.5961°)
ω_UB = defect(origin, BU+, UNA) = 0.083413894169 rad  ( 4.7793°)
```

The symbols ω_UO, ω_OB, and ω_UB are the three elementary pair angles. Each is the holonomy of the triangle that joins the origin to one orthogonal pair of payload velocities.

Because the three payload vectors occupy three mutually orthogonal directions, the three rotations act about three complementary axes. UNA and ONA occupy the x–y plane and generate rotation about z. ONA and BU occupy the y–z plane and generate rotation about x. BU and UNA occupy the z–x plane and generate rotation about y. Every directed edge of the stage graph acts as one of these three generators or as the identity, the identity case being the collinear crossing between the two BU poles.

The constitutional stage triangle CS–UNA–ONA closes with zero Euclidean defect. That triangle fixes the angular thresholds before motion. The three pairwise precessions above are the hyperbolic holonomies of the payload placement in velocity space. The three pairwise rotations furnish rotations about the three orthogonal spatial axes. Together with the three stage displacements, they provide the six kinematic degrees of freedom used by the CGM realization. Orientation is produced by the stage pairs themselves.

### 3.2 The half-angle product identity

Each elementary angle satisfies a closed-form product identity in the Poincaré radii of its two stages.

```
tan(ω_UO / 2) = k(UNA) k(ONA) = 0.200941569628
tan(ω_OB / 2) = k(ONA) k(BU)  = 0.048874404435
tan(ω_UB / 2) = k(BU)  k(UNA) = 0.041731146576
```

These three identities are the Wigner formula tan(ω/2) = tanh(η₁/2) tanh(η₂/2) for perpendicular boosts, evaluated at the CGM thresholds. The half-angle tangent of each pair precession is the product of the two Poincaré radii.

The three angles share a single multiplicative grammar, and their pairwise ratios reduce to ratios of stage radii.

```
tan(ω_UB / 2) / tan(ω_OB / 2) = k(UNA) / k(ONA) = 0.853844605530
```

The ratio of the two balance-coupled half-angle tangents equals the ratio of the UNA and ONA radii. The BU radius cancels, so the small BU amplitude scales both the ONA–BU and BU–UNA precessions linearly at leading order, while the large UNA–ONA precession is independent of the balance amplitude.

### 3.3 Recovering the Poincaré radii

The three product identities invert. Writing a = tan(ω_UO/2), b = tan(ω_OB/2), and c = tan(ω_UB/2),

```
k_UNA = √(a c / b)
k_ONA = √(a b / c)
k_BU  = √(b c / a).
```

Each Poincaré radius is recovered from the three half-angle tangents. A carried frame that records the three pair angles therefore reconstructs the three displacement magnitudes of the payload placement.

The three angles are three sectional fluxes of one mass-shell curvature through the planes UNA–ONA, ONA–BU, and BU–UNA. One universal curvature runs through three constitutional surfaces, as one magnetic field yields different fluxes through surfaces of different area and orientation.

The ratio k_UNA / k_ONA isolates the asymmetry between the rotational and translational thresholds. Equal Einstein speeds on those two stages would equalize ω_UB and ω_OB. Their measured inequality preserves the distinction between rotational individuality and translational accountability.

---

## 4. Dual-Pole Channels

### 4.1 The gravitational channel

ONA introduces displacement within a shared geometry. BU preserves ancestry through balanced closure. Their composition measures the rotation required to preserve a common origin under displacement. The dual-pole loop follows the path ONA → BU+ → BU− → ONA. The central pole crossing is collinear and contributes zero gyration. The two corners contribute equal rotations about the same axis, so

```
δ_BU = 2 ω_OB = 4 arctan(k(o_p) k(m_a)) = 0.195342178258.
```

The holonomy δ_BU is twice the ONA–BU corner. The closed form follows because each corner has half-angle tangent k(o_p) k(m_a) and the two corners add on a common axis. This is the angle of the channel that combines displacement with balance. Gravity is the continuum preservation of ancestry under displacement, so the gravitational Regge deficit is this ONA–BU dual-pole holonomy. In the Regge discretization, each plaquette carries the deficit angle δ_BU.

The closure ratio and the aperture gap are the secant fill of that angle into the observational aperture,

```
ρ = δ_BU / m_a,    Δ = 1 − ρ.
```

The ratio ρ measures how much of the aperture amplitude is occupied by the dual-pole holonomy. The remainder Δ defines the residual aperture after the dual-pole holonomy is normalized by m_a.

The gravitational refractive depth [8]

```
τ_G = |Ω| Δ ρ⁵ (1 − 4ρΔ²) ≈ 76.237914
```

integrates plaquette curvature across the bulk manifold. Here |Ω| = 4096 is the number of reachable states of the discrete carrier. The factor ρ⁵ is the attenuation through the five bulk shells, and (1 − 4ρΔ²) is the geometric correction at even order in the aperture. The depth τ_G enters the position-dependent coupling G(ψ). The gravitational continuum accumulates the same precession across its plaquettes. The quantities ρ, Δ, and τ_G connect the local precession δ_BU to the scale-dependent coupling G(ψ).

The same angle is the base of the electromagnetic coupling [9]

```
α₀ = δ_BU⁴ / m_a = 0.007299683573.
```

The quartic power arises from two dual-pole corners and two commutators. Division by m_a measures the resulting coupling in aperture units.

The ONA–BU corner is the elementary angle of that channel, and the dual-pole loop doubles it.

### 4.2 The rotational aperture channel

The BU–UNA pair produces the third elementary angle ω_UB = 0.083413894169. Its dual-pole loop UNA → BU+ → BU− → UNA has angle

```
δ_UNA_BU = 2 ω_UB = 0.166827788338.
```

This is the dual-pole holonomy of the rotation channel, doubling the UNA–BU corner in the same way that δ_BU doubles the ONA–BU corner.

The ONA-rooted loop is the holonomy of the translation channel. The UNA-rooted loop is the holonomy of the rotation channel. Their dual-pole angles stand in the ratio

```
δ_UNA_BU / δ_BU = ω_UB / ω_OB = 0.854028504372.
```

The half-angle tangents stand in the Poincaré-radius ratio

```
tan(ω_UB / 2) / tan(ω_OB / 2) = k(UNA) / k(ONA) = 0.853844605530.
```

These are distinct identities. They agree in the small-angle limit, and at the physical thresholds they differ by about two parts in 10⁴. The UNA-rooted dual-pole holonomy is orthogonal in axis to the ONA-rooted holonomy. The ONA–BU channel supplies the displacement-associated component and the UNA–BU channel supplies the rotation-associated component.

### 4.3 Spinorial branch

A spatial rotation through angle θ lifts to a spinorial transformation with half-angle θ/2. The dual-pole spatial angle and the ONA–BU corner therefore stand in the double-cover relation

```
δ_BU / 2 = ω_OB.
```

The SO(3) holonomy of the dual-pole loop is twice the elementary corner, and the corresponding SU(2) transformation retains the half-angle phase. Two spatial corners form one spatial holonomy. The continuum records that double cover through this half-angle lift.

---

## 5. Palindrome Steering

The palindromic traversal UNA → ONA → BU+ → BU− → ONA → UNA carries the frame through all payload stages and back. Its intrinsic angle equals δ_BU, and its rotation operator satisfies the conjugation identity

```
R_pal = A⁻¹ R_BU A,    A = gyr(UNA, ONA).
```

Here R_BU is the dual-pole rotation, R_pal is the palindrome rotation, and A is the gyration generated by the UNA–ONA pair. Conjugation by A preserves the rotation angle and transports the rotation axis. The measured transport angle between the dual-pole axis and the palindrome axis is ω_UO. The dual-pole axis is perpendicular to the steering axis, so the conjugation acts at full strength.

Iterating the conjugation gives A⁻ⁿ R_BU Aⁿ, which preserves the angle δ_BU and rotates the axis by n ω_UO. The axis returns to its starting orientation after

```
2π / ω_UO = 15.8426 conjugations.
```

The period is the number of UNA–ONA steerings needed to complete a full turn of the balance axis.

The ONA–BU dual-pole loop fixes the magnitude δ_BU of the balance holonomy. The UNA–ONA gyration fixes the axis rotation ω_UO that steers this holonomy under palindromic conjugation. The two quantities therefore separate the magnitude of the retained rotation from its orientation in space. The UNA–ONA threshold difference is o_p − u_p = 0.078291. The same pair also enters the angular-storage ratios 4/3 and 5/3 [12].

This structure is the gyroscopic separation between spin precession and axis transport, derived here from the stage geometry. It is also the geometric phase content of the palindrome. A carried frame receives a definite balance rotation and a definite axis steering, both fixed by the thresholds.

---

## 6. Closed Walks of Length Two through Five

Closed loops through the stage points produce rotation angles that are words in the three generators. Let UO, OB, and UB denote the elementary rotations of the corresponding stage pairs and I the identity. Every closed walk of length two through five on the four payload points was enumerated, and the Fermi–Walker holonomy of each walk was computed. At those lengths the holonomy takes only six distinct values, and each value is realized by at least one walk. The table below lists the six classes with the shortest walk in each class.

| Class | Angle | Multiplicity | min L | Representative |
|---|---:|---:|---:|---|
| Identity | 0 | 72 | 2 | out-and-back |
| UNA dual-pole | 0.166827788338 | 66 | 3 | UNA–BU+–BU− |
| BU dual-pole | 0.195342178258 | 66 | 3 | ONA–BU+–BU− |
| Crossed four-cycle | 0.256712834405 | 8 | 4 | UNA–BU+–ONA–BU− |
| Four-stage cycle | 0.412719054050 | 16 | 4 | UNA–ONA–BU+–BU− |
| UNA–ONA–BU triangle | 0.420475081676 | 132 | 3 | UNA–ONA–BU+ |

The identity class collects all out-and-back walks, which cancel under Fermi–Walker transport by the inversion identity gyr(v, u) = gyr(u, v)⁻¹. The two dual-pole classes double their corner angles because the pole crossing contributes no rotation. The three-stage triangle δ_UOB = 0.420475081676 is the ordered product of the three elementary rotations. The scalar sum of the three elementary magnitudes is 0.577686485519, and the difference δ_UOB − (ω_UO + ω_OB + ω_UB) = −0.157211403843 measures the noncommutativity of rotations about distinct axes.

The remaining two classes are generator words on the payload graph. The four-stage cycle has generator word UO⁻¹ OB⁻¹ I UB and angle 0.412719054050. The crossed four-cycle has word UB OB OB UB and angle 0.256712834405. The nearest combination n₀ ω_UO + n₁ ω_OB + n₂ ω_UB with integer coefficients of magnitude at most two differs from the crossed four-cycle by 7.8 × 10⁻³. These two angles are Fermi–Walker holonomies of those 4-cycles, with axes and spinorial lifts belonging to the full rotation product.

Cyclic permutation of the starting vertex on the BU loop leaves the conjugacy angle invariant and equal to δ_BU. The same vertex permutations, evaluated as boost products in a single inertial frame, span about 0.010 in rotation angle.

---

## 7. Three Transport Prescriptions

An angle of precession depends on the rule by which the frame is carried. The analysis distinguishes three prescriptions and measures each on the same loops.

### 7.1 Intrinsic transport

The intrinsic prescription is Fermi–Walker transport, the torque-free carriage of a gyroscope along its worldline. Four independent constructions realize it: the origin-gyration word built from Ungar gyrations, the product of rotation-free boosts along the geodesic edges of the mass shell, the Ungar gyrotriangle defect, and the path-ordered integral of the Thomas one-form

```
ω_T = (γ² / (γ + 1)) β × dβ
```

in Cartesian velocity coordinates, with Richardson extrapolation. The one-form ω_T is the infinitesimal Thomas rotation generated by a change of Einstein velocity β. On all 360 enumerated walks, origin gyration and geodesic transvection give the same holonomy. On the BU loop and the palindrome, the Cartesian Thomas integral also agrees with those constructions. On the BU loop, the gyrotriangle defect, the dual-pole word, origin gyration, geodesic transvection, and the Cartesian Thomas integral all give δ_BU. The Cartesian path-ordered integral equals the geodesic angle to nine decimal places on both the BU loop and the palindrome. The intrinsic angle is invariant under cyclic refactorization of the loop and vanishes identically on every out-and-back path. This is the precession a comoving gyroscope accumulates, and it is the quantity that enters the gravitational Regge sum.

If TW(a, b) denotes the Thomas–Wigner angle for boosts along velocities a and b, then δ_BU = 2 TW(ONA, BU+). The finite pair rotation and the continuous path-ordered connection describe the same precession at complementary resolutions.

### 7.2 Boost composition in a fixed inertial frame

The second prescription composes the successive relative boosts of the loop in one fixed inertial frame, which is Thomas's original 1926 construction. The composition factorizes as a single net boost times a net rotation,

```
B(d₁) B(d₂) ⋯ B(d_n) = B(u_final) R_inert.
```

Each B(d_i) is a pure boost along a stage displacement, u_final is the leftover velocity after the product, and R_inert is the net rotation relative to that inertial frame. Only the four collinear out-and-back exchanges between the two BU poles close with vanishing net boost, and those carry zero rotation. Every bent walk leaves a residual velocity.

| Path | Rotation angle | Final boost magnitude |
|---|---:|---:|
| ONA–BU out-back | 0.06313650 | 0.127193 |
| UNA–ONA out-back | 0.46417802 | 0.657716 |
| UNA–BU out-back | 0.03689497 | 0.086532 |
| BU triangle | 0.25847867 | 0.127193 |
| Palindrome | 0.53360891 | 0.670179 |

A worldline that bends can return to its starting velocity. A finite product of pure boosts along the same vertices arrives with a leftover velocity that a further boost cancels, and that cancellation contributes its own Wigner rotation. The inertial-frame angles record how a particular external observer assembles the journey. Because that assembly depends on the leftover boost at each vertex, walks of length five produce 147 distinct rotation angles, while Fermi–Walker transport on those same walks takes only those six intrinsic values.

Let R_can be the intrinsic rotation and R_inert the rotation obtained by composing boosts in one inertial frame. The scalar difference of their angles is F = θ_inert − θ_can. The relative group element is R_F = R_can⁻¹ R_inert, with conjugacy angle θ_RF. For out-and-back paths θ_can = 0, so F and θ_RF both equal θ_inert. On the BU loop, F equals the ONA–BU out-and-back inertial-frame rotation, while θ_RF equals θ_can + θ_inert. F on the dual-pole triangles equals the inertial-frame rotation of the corresponding out-and-back leg.

### 7.3 Chart readout

In Cartesian velocity coordinates the path-ordered exponential equals the intrinsic angle. In spherical momentum coordinates the same connection acquires coordinate singularities at the chart poles and at rest. For the z-oriented chart, with BU poles at θ = 0 and θ = π, the spherical readout is 0.2466038230 on the BU loop, so the offset from δ_BU is G_z = 0.0512616448. A diagonal spherical chart moves the stage vertices away from its poles and differs from the z-oriented chart by approximately one radian. These differences are properties of the coordinate system. The Cartesian evaluation is regular on the full path and returns the intrinsic holonomy. The spherical residuals G quantify the chart-dependent contribution produced by coordinate singularities.

The three prescriptions place the CGM observables on the same footing as standard relativistic practice. A comoving gyroscope measures the intrinsic angles. An inertial observer who multiplies successive boosts measures angles that depend on that assembly and exceed the intrinsic ones on bent paths, and that close to a pure rotation on collinear pole exchange. A regular coordinate chart measures the intrinsic angle. These are the three situations of classical Thomas precession, held in closed form at the CGM thresholds.

---

## 8. Closure Response

The BU dual-pole angle is a function of the ONA magnitude and the BU amplitude,

```
δ_BU(θ, m) = 4 arctan[k(θ) k(m)].
```

The arguments θ and m are the translational threshold and the balance amplitude. At the physical thresholds this expression is a constitutive relation for the balance channel. The closure ratio ρ = δ_BU / m_a and the aperture gap Δ = 1 − ρ are secant quantities, ratios evaluated across the full balance amplitude from zero to m_a. The constitutive relation admits two further response coefficients, and the three together decompose the aperture.

The linear gain is the derivative of the dual-pole holonomy with respect to the balance amplitude, evaluated at zero amplitude:

```
ρ₀ = d(δ_BU)/d(m_a) at m_a = 0 = 2 k(o_p) = 0.970231725282.
```

The slope at vanishing amplitude equals twice the Poincaré radius of ONA. The corresponding linear opening is Δ₀ = 1 − ρ₀ = 0.029768274718. This is the opening supplied by ONA geometry before finite BU amplitude contributes additional closure. The value follows from o_p = π/4 and the derivative formula above.

The physical secant gain is ρ = 0.979300454497, so finite BU amplitude adds ρ − ρ₀ = 0.009068729215, and the remaining secant aperture is Δ = 0.020699545503. The tangent gain at the physical amplitude is ρ_tan = 0.997796177590, with incremental opening Δ_tan = 0.002203822410. The nonlinear closure gain is ρ_tan − ρ = 0.018495723092.

Two budget identities close without remainder:

```
1 − ρ₀ = (ρ − ρ₀) + (1 − ρ)
Δ      = Δ_tan + (ρ_tan − ρ).
```

The first identity splits the linear-regime opening into the closure purchased by finite balance amplitude plus the residual secant aperture. The second splits the secant aperture into the tangent aperture plus the nonlinear gain accumulated between zero amplitude and the operating point. The complete split is 1 − ρ₀ = (ρ − ρ₀) + (ρ_tan − ρ) + Δ_tan.

The closure response is a constitutive curve with three regimes. The ONA geometry alone, at vanishing balance amplitude, already converts 97.02 percent of each unit of amplitude into holonomy. Driving the amplitude to its physical value raises the average conversion to 97.93 percent and the marginal conversion to 99.78 percent. The system operates on the steep, nearly saturated part of its constitutive curve, where an infinitesimal additional oscillation converts to holonomy almost without loss while the full excursion retains a secant aperture of 2.07 percent.

Processes that engage the full balance excursion use the secant aperture. That is the scale of the gravitational Regge transport. Processes linearized about the operating point use the tangent aperture, which is smaller by the factor Δ / Δ_tan ≈ 9.393. The discrete companion 48Δ = 0.993578 is the unit quantization of the aperture structure. Combined with the model grand-unification energy E_GUT, it sets the seesaw scale M_R = E_GUT / 48² [10].

The fractional elasticities of the dual-pole holonomy at the physical point are

```
E_ONA = (o_p / δ_BU) (∂δ_BU / ∂o_p) = 1.612965279633
E_BU  = (m_a / δ_BU) (∂δ_BU / ∂m_a) = 1.018886668547
E_ONA / E_BU = (o_p / m_a) (d ln k / d o_p) / (d ln k / d m_a) = 1.583066428706.
```

The elasticity E_ONA is the logarithmic response of δ_BU to the translational threshold. The elasticity E_BU is the corresponding response to the balance amplitude. Their ratio follows from the logarithmic derivatives of the Poincaré radii. The dual-pole holonomy responds superlinearly to the translational geometry and almost linearly to the vibrational amplitude. The elasticity E_ONA is larger, so changes in the translational threshold steer the dual-pole holonomy more strongly than changes in the balance amplitude.

---

## 9. Equal-Speed Wigner Response

The quantity ω₀ belongs to a calibrated equal-speed Wigner configuration,

```
ω₀ = TW(u_p, u_p; separation = o_p) = 0.215549910153.
```

This is the Thomas–Wigner angle of two equal boosts of magnitude u_p separated by the ONA angle. The stage angle ω_UO uses the actual UNA and ONA speeds on orthogonal axes. The equal-speed calibration measures how Wigner rotation responds when the common boost magnitude and the separation angle vary around the canonical point. The derivatives are

```
∂ω₀/∂β = (12√2 − 4)/17 = 0.762974279322
∂ω₀/∂θ = (21 − 12√2)/17 = 0.237025720678
```

The first derivative is the response of ω₀ to the common boost speed. The second is the response to the directional separation. They sum to 1. At this operating point, changes in boost magnitude carry about three quarters of the first-order response, and changes in directional separation carry about one quarter.

---

## 10. Rate and the Aperture Clock

An angle per cycle becomes a precession rate once a clock is fixed. For a physical carrier that completes a closed loop in proper time T, the corresponding rate is Ω_prec = θ_hol / T. The stage geometry determines the angle. The carrier and its environment determine the proper time. The angle per completed cycle is already physical. Frequency appears when that cycle repeats.

The observational aperture supplies the intrinsic clock of the hierarchy. The aperture time is the balance amplitude itself,

```
t_aperture = m_a = 1 / (2 √(2π)).
```

This is a dimensionless interval. One unit of aperture time is the geometric duration associated with the observational opening.

The horizon length is L_horizon = √(2π). Their ratio is the quantum-gravity invariant Q_G = 4π, the complete solid angle of coherent observation in three dimensions. The spinorial identity Q_G m_a² = 1/2 holds to the numerical precision of the calculation. Stage actions are given by

```
S_CS  = (π/2) / m_a ≈ 7.874805
S_UNA = u_p / m_a   ≈ 3.544908
S_ONA = o_p / m_a   ≈ 3.937402 = K_QG
S_BU  = m_a         ≈ 0.199471
S_GUT = 1 / (1/S_UNA + 1/S_ONA + 1/S_CS) ≈ 1.508167   (η = 1)
```

The CS, UNA, and ONA actions are threshold values expressed in aperture units. The BU quantity is the aperture amplitude itself. The ONA action S_ONA is the commutator scale of the relation [X, P] = i K_QG [11]. The electromagnetic duality angle is arctan(S_ONA / S_UNA) ≈ 48.003°. The GUT action is the parallel combination of UNA, ONA, and CS with complete memory weight η = 1.

A holonomy angle θ has dimensionless action S(θ) = θ / t_aperture. For the BU loop this action is the closure ratio,

```
S(δ_BU) = δ_BU / t_aperture = ρ = 0.979300454497.
```

The aperture-normalized angular rate of the dual-pole holonomy is ρ. The aperture gap Δ = 1 − ρ is the rate deficit per tick, the amount by which the balance precession falls short of the unit rate. The gravitational refractive depth compounds this slightly sub-unit rate across the bulk shells.

On the Δ-ruler the coordinate of an angle is n = θ / Δ, the number of aperture ticks of memory.

| Object | θ | S = θ / t_aperture | n = θ / Δ |
|---|---:|---:|---:|
| ω_UB | 0.083413894169 | 0.418175 | 4.030 |
| ω_OB | 0.097671089129 | 0.489650 | 4.719 |
| δ_UNA_BU | 0.166827788338 | 0.836351 | 8.059 |
| δ_BU | 0.195342178258 | 0.979300 = ρ | 9.437 |
| ω_UO | 0.396601502221 | 1.988265 | 19.160 |
| φ_SU2 | 0.587900762654 | 2.947297 | 28.402 |

Physical time enters through the measured Planck constant. For a carrier of rest mass m the Compton time is T_C = ℏ / (m c²), and the dimensional aperture interval is the same clock scaled by the aperture amplitude,

```
T_aperture = m_a T_C = m_a ℏ / (m c²).
```

The interval T_aperture is the Compton time of the carrier reduced by the aperture amplitude. A holonomy angle θ then has the associated angular frequency Ω = θ / T_aperture = θ m c² / (m_a ℏ). For the balance channel this is Ω_BU = ρ ω_C, with Compton frequency ω_C = m c² / ℏ. The associated energy ρ m c² and the aperture energy Δ m c² are the same rest energy scaled by ρ and by Δ. Heavier carriers have shorter Compton times and higher rates at the same stage angle, so frequency scales linearly with mass while the geometry remains common.

For the electron the Compton conversion of the rest energy is

```
T_C        = 1.288089 × 10⁻²¹ s
T_aperture = 2.569365 × 10⁻²² s
f_BU       = 1.210014 × 10²⁰ Hz
Ω_BU       = 7.602741 × 10²⁰ rad/s
E_BU       = 5.004215 × 10⁵ eV
E_aperture = 1.057745 × 10⁴ eV
```

These quantities are T_C, m_a T_C, ρ ω_C / 2π, ρ ω_C, ρ m_e c², and Δ m_e c². They convert the electron rest energy through the geometric factors ρ and Δ.

The weak-field gravitational coupling is G₀ = (π/6) exp(−τ_G) / v² [8], where v is the electroweak vacuum expectation value. Combined with measured ℏ this coupling reconstructs the CS time interval √(ℏ G₀ / c⁵). That interval is an output of the same geometry that already supplies ρ and Δ. Orbit periods and storage-ring cycles supply the closed-loop proper time T in those transport contexts.

Thomas precession per revolution on a circle at speed V is 2π(γ(V) − 1). Inverting this relation assigns to each CGM angle the circular speed whose ordinary Thomas precession matches it per revolution.

| Angle | Equivalent circular speed |
|---|---:|
| ω_UB | 0.161344 |
| ω_OB | 0.174297 |
| δ_BU | 0.243712 |
| ω_UO | 0.339443 |
| φ_SU2 | 0.404725 |

The stage precessions occupy the range of atomic and astrophysical Thomas precession for matter at moderate fractions of the speed of light. The frequencies Ω_UO, Ω_OB, and Ω_UB scale linearly with carrier mass.

---

## 11. Compact Fiber

The pairwise precessions of the preceding sections live on the mass shell, the hyperbolic space of velocities. The framework also carries a compact rotational fiber. The CGM compact rotational sector is represented by SU(2). The commutator of two orthogonal SU(2) rotations through the stage angle π/4 has the closed-form conjugacy angle

```
φ_SU2 = 2 arccos((1 + 2√2) / 4) = 0.587900762654
```

This is the group angle of the compact commutator at the constitutional stage angles. Three BU dual-pole angles give 3 δ_BU = 0.586026534773. The scalar residual is

```
ε_CH = φ_SU2 − 3 δ_BU = 0.001874227881.
```

The residual is the mismatch between one compact commutator and three hyperbolic dual-pole holonomies.

The compact commutator measures rotational memory in the SU(2) fiber. The three BU loops measure displacement-balance memory in hyperbolic velocity space. Their residual measures the phase mismatch between compact rotation and threefold hyperbolic closure. The ratio φ_SU2 / (3 δ_BU) and the residual ε_CH form the commutator-transport factor of the electromagnetic coupling chain [9].

The compact commutator axis and the dual-pole holonomy axis are distinct directions, with inner product −0.357407, corresponding to transverse axis separation χ_CH = arccos(0.357407) ≈ 69.059°. Aligning the axes reduces their relative group angle to the scalar residual. Directly composing the original group elements retains the additional axis mismatch: rel(C, U_BU³) = 0.962562. The scalar relation governs phase transport after orientation has been identified. The complete group relation contains both phase and axis information.

The aperture-normalized form is σ = ε_CH / m_a = 0.009395985199. That is the same residual expressed in aperture units, the conversion that reads ρ as a rate.

---

## 12. Universality

The precession structure is universal because the same mechanism holds wherever a distinguishable state is transported while its origin remains recoverable. The universal sequence is oriented source, distinguishable displacement, noncommutative composition, frame rotation, balanced closure, and retained holonomy.

Different physical systems express different aspects of this sequence. A gyroscope is spatial frame rotation. A spinor is the half-angle SU(2) phase. A gauge state is internal parallel transport. An orbit is accumulated curvature through apsidal advance. The gravitational continuum is the accumulated ONA–BU curvature through the Regge deficit and refractive depth. Mercury perihelion advance, geodetic precession, and muon storage-ring spin rotation each involve accumulated angular change along a closed or periodic trajectory. CGM relates these observables through the shared geometry of transport, curvature, and cycle time.

The measurement and the phenomenon coincide because the transported frame is part of the physical system. The frame rotates, and its rotation records the geometry that acted upon it. Holonomy is both dynamics and record.

Within CGM, angular momentum expresses conserved transport memory, mass expresses the accumulated cost of maintaining that memory through displacement, and gravity expresses the continuum balance that preserves it. The three pairwise precessions are the elementary rotations. The dual-pole loop is the scalar holonomy of gravity and electromagnetism. The palindrome is the steered orientation. The aperture clock converts angle into rate. These are the same precession geometry at different scales and in different carriers.

---

## 13. Realization

The measured constellation depends on how the threshold numbers are placed in velocity space. Placing the thresholds as Einstein speeds yields the pairwise precessions reported above. Placing them as rapidities, η = atanh(threshold), yields a dual-pole angle 0.148518. Placing UNA at the ONA angle to equalize the two large speeds forces ω_UB = ω_OB and raises the steering quantum to 0.462263. Among the three tested placements, the Einstein-speed assignment gives the dual-pole angle used as the gravitational deficit unit [8] and in the electromagnetic coupling α₀ [9]. The Einstein-speed assignment supplies the shared value δ_BU used in those constructions. The alternative placements are controls showing that the angle values are properties of the realization. The orthogonality of the three axes persists under every tested placement, since it follows from the orthogonal seating of the stage vectors. Palge–Pfeifer Cartesian evaluation equals geodesic holonomy for any ball speeds.

---

## 14. Falsification Criteria

The following outcomes would falsify the precession realization.

1. A discrepancy between the geodesic holonomy, origin-gyration product, dual-pole word, or gyrotriangle defect for (ONA, BU+, BU−) and δ_BU = 4 arctan(k(π/4) k(m_a)) beyond numerical precision.
2. A discrepancy between the Cartesian Thomas path-ordered exponential on the BU loop or the palindrome and δ_BU beyond 10⁻⁸.
3. A discrepancy between origin-gyration and geodesic holonomy on any closed walk of length at most 5.
4. A Fermi–Walker angle other than the six enumerated values, or a generator other than UO, OB, UB, or the identity, on a closed walk of length at most 5.
5. Any departure from mutual orthogonality of the three elementary axes, or any discrepancy between an elementary angle and its gyrotriangle defect.
6. A discrepancy between the palindrome conjugation R_pal = A⁻¹ R_BU A and the measured operators, or a discrepancy between the axis-transport angle and ω_UO.
7. A product of successive boosts of length at most 5, evaluated in one inertial frame, that closes to a pure rotation on a path other than the collinear BU+ → BU− out-and-back paths.
8. A discrepancy between tan(ω_UO/2) and k(UNA) k(ONA), or between the companion identities for OB and UB and their Poincaré products, or between the inverted radii and k_UNA, k_ONA, and k_BU.
9. A discrepancy between Q_G = L_horizon / m_a and 4π, or between Q_G m_a² and 1/2, or between δ_BU / m_a and ρ.
10. A discrepancy larger than 10⁻¹² between the values δ_BU, ρ, and Δ and the values used in the gravitational and electromagnetic sectors [8, 9].

---

## 15. Reproducibility

```
python experiments/cgm_precession_analysis_run.py
```

The command runs parts 1 and 2, prints the full report, and writes `experiments/cgm_precession_analysis_results.txt`. The run is deterministic, uses `mpmath` for the analytic layer, and exits with a nonzero code if any gate fails. Companions are `cgm_precession_analysis_1.py` and `cgm_precession_analysis_2.py`. Circular Thomas calibration at several speeds is gated in `cgm_holonomy_analysis_2.py`.

---

## 16. Conclusion

The CGM precession analysis is the physical realization of three-dimensional rotational structure and six kinematic degrees of freedom. Three orthogonal stage displacements generate three orthogonal Thomas–Wigner rotations. Their angles reconstruct the radial stage geometry. Every Fermi–Walker holonomy among closed walks of length two through five is one of the six enumerated values.

The ONA–BU channel produces the dual-pole holonomy δ_BU. This channel combines displacement with ancestry-preserving closure and is the curvature unit of gravity [8] and the base of α₀ [9]. The UNA–BU channel is the holonomy of the rotation channel. The UNA–ONA channel steers the balance axis and produces a secondary precession of that axis under palindromic transport, with algebraic lift o_p − u_p. The same pair enters the angular-storage ratios 4/3 and 5/3 [12].

The closure function δ_BU(o_p, m_a) defines a nonlinear physical response. Its secant aperture controls integrated transport across the full BU excursion. Its tangent aperture controls incremental response at the operating point. The compact SU(2) commutator is the internal rotational memory, and its residual relative to three BU loops is the commutator-transport factor of the electromagnetic coupling chain, equivalently the aperture-normalized σ.

The canonical connection, the complete Thomas connection, the geodesic transvection, and the gyrotriangle defect describe the same physical rotation. Multiplication of successive boosts in one inertial frame records the additional acceleration history of that observer. The spherical connection records the behavior of local coordinates through singular frame patches.

Precession is the continuum action by which CGM preserves directional ancestry under displacement. Its angle is the retained memory of a closed path. Its axis records how that memory is oriented. Its rate on the aperture clock is the closure ratio ρ. The three pairwise precessions are the rotational structure of space, and the ONA–BU channel is the elementary curvature of gravity. Within CGM, angular momentum, mass, and gravity are the large-scale expressions of the same transport memory: orientation conserved, cost accumulated, balance preserved across the continuum.

---

## Appendix. Supplementary Measurements

The body sections state the primary holonomies and their physical roles. This appendix collects the remaining measured quantities from `cgm_precession_analysis_results.txt` that support transport classification, spectrum structure, and comparison between transport prescriptions. Verification gates remain in the reproducibility script.

### A.1 Extended stage kinematics

| Stage | γβ | Doppler (1+β)/(1−β) | Poincaré k |
|---|---:|---:|---:|
| UNA | 1.000000 | 2.414214 | 0.414214 |
| ONA | 1.268836 | 2.884369 | 0.485116 |
| BU | 0.203562 | 1.224070 | 0.100748 |

The Poincaré column equals k(β). UNA sits at γβ = 1.

### A.2 Loop topology and axis geometry

The BU dual-pole triangle spans a plane (rank 2). The palindrome spans three axes (rank 3, non-planar). On the palindrome, the corner Wigner rotations sum to 1.891 rad while the net inertial-frame rotation is 0.534 rad. Non-planarity makes the corner sum differ from the net rotation.

Palindrome conjugacy preserves the canonical angle with axis alignment |R_pal − A⁻¹ R_BU A| below 10⁻⁴⁹. The axis dot between BU and palindrome rotations is 0.922379. The axis-transport angle between BU and palindrome axes equals ω_UO. The steering axis is perpendicular to the BU axis, so their inner product vanishes.

### A.3 Generator words and pair-angle independence

| Word | Generators | θ | θ/2 |
|---|---|---:|---:|
| BU dual-pole | OB⁻¹ I OB⁻¹ | 0.195342178258 | 0.097671089129 |
| UNA dual-pole | UB I UB | 0.166827788338 | 0.083413894169 |
| UNA–ONA–BU+ | UO⁻¹ OB⁻¹ UB⁻¹ | 0.420475081676 | 0.210237540838 |
| four-stage cycle | UO⁻¹ OB⁻¹ I UB | 0.412719054050 | 0.206359527025 |
| crossed four-cycle | UB OB OB UB | 0.256712834405 | 0.128356417202 |

The nearest integer combination n₀ ω_UO + n₁ ω_OB + n₂ ω_UB with |nᵢ| ≤ 2 that best matches the crossed four-cycle is (0, 1, 2) at 0.264498877467. The residual 7.786 × 10⁻³ is the difference between that combination and the holonomy of the cycle.

### A.4 Inertial-frame boost residuals

| Quantity | Value |
|---|---:|
| F_BU = θ_inert − θ_can on the BU loop | 0.063136496377 |
| θ_RF on the BU loop | 0.45382085 |
| θ_can + θ_inert on the BU loop | 0.45382085 |
| F_pal = θ_inert − θ_can on the palindrome | 0.338266729989 |
| θ_RF on the palindrome | 0.63853091 |
| hypot(F_ONA–BU, F_UNA–ONA, F_UNA–BU) | 0.469902854347 |

F_BU equals the ONA–BU+ out-and-back rotation in the inertial-frame product. On the BU loop, θ_RF equals θ_can + θ_inert. The Euclidean combination of the three elementary inertial-frame out-and-back angles is 0.131636124358 away from F_pal.

Among walks of length at most 5, the inertial-frame products include additional angles 0.670604 (6 walks), 0.212018 (6), 0.609965 (6), and 0.268956 (6) besides the six Fermi–Walker values enumerated above. Cyclic permutation of the BU loop leaves the intrinsic angle unchanged and spans 0.0105 rad in the inertial-frame product.

### A.5 Chart readouts

Cartesian Thomas Pexp with Richardson extrapolation: BU n-step 0.1953217487, 2n-step 0.1953370698, extrapolated 0.1953421768 (equals δ_BU). Spherical z-chart readout on BU: θ = 0.24660382, offset G_z = 0.05126164. Diagonal chart: θ = 1.24655632, G_diag = 1.05121414. G_diag − G_z = 0.99995249 rad.

| Chart | θ_BU | θ_pal | G_BU | G_pal |
|---|---:|---:|---:|---:|
| z | 0.24660382 | 0.24673566 | 0.05126164 | 0.05139348 |
| x | 1.80164501 | 1.79935086 | 1.60630283 | 1.60400868 |
| y | 1.65033595 | 1.68769083 | 1.45499377 | 1.49234865 |
| diag | 1.24655632 | 1.26349311 | 1.05121414 | 1.06815093 |

BU poles sit at θ = 0 and θ = π in the z chart. Every stage vertex lies away from the poles of the diagonal chart.

### A.6 Closure response detail

| Quantity | Value |
|---|---:|
| two_1_rho0 = 2(1 − ρ₀) | 0.059536549435 |
| d(δ_BU)/dm at m_a | 0.997796177590 |
| d²(δ_BU)/dm² | 0.285536795041 |
| d(δ_BU)/dθ_ONA | 0.401172508240 |
| 1/Δ | 48.310239463 |
| d ln k/d ln o_p | 2.056960062373 |
| d ln k/d ln m_a | 5.116070631790 |

### A.7 Equivalent circular Thomas speeds

| Angle | θ | turn fraction θ/2π | equivalent β |
|---|---:|---:|---:|
| ω_UB | 0.08341389 | 0.013276 | 0.161344 |
| ω_OB | 0.09767109 | 0.015545 | 0.174297 |
| δ_BU | 0.19534218 | 0.031090 | 0.243712 |
| ω₀ | 0.21554991 | 0.034306 | 0.255413 |
| ω_UO | 0.39660150 | 0.063121 | 0.339443 |
| φ_SU2 | 0.58790076 | 0.093567 | 0.404725 |
| θ_inert BU | 0.25847867 | 0.041138 | 0.278324 |
| θ_inert pal | 0.53360891 | 0.084927 | 0.387853 |
| θ_chart complete | 0.19534218 | 0.031090 | 0.243712 |
| θ_chart sph z | 0.24660382 | 0.039248 | 0.272224 |

### A.8 Compact fiber detail

Compact commutator axis: (+0.357407, −0.357407, +0.862856). Dual-pole holonomy axis: (−1, 0, 0). Axis dot: −0.357407. rel(C, U(n_C, 3 δ_BU)) equals |φ_SU2 − 3 δ_BU| = 0.001874228. rel(C, U_BU³) = 0.962562. rel(C, Ux Uy Uz δ) = 0.482226. ε_CH / m_a = σ = 0.009395985.

### A.9 Realization controls

| Placement | ω_UO | ω_OB | ω_UB | δ_BU |
|---|---:|---:|---:|---:|
| Einstein β | 0.396601502221 | 0.097671089129 | 0.083413894169 | 0.195342178258 |
| Rapidity | 0.252400660335 | 0.074259006747 | 0.067475778654 | 0.148518013495 |
| UNA at ONA angle | 0.462263357559 | 0.097671089129 | 0.097671089129 | 0.195342178258 |

ω₀ = TW(u_p, u_p; o_p) = 0.215549910153. TW(o_p, o_p; π/2) = 0.462263357559 equals the UNA-at-ONA-angle steering quantum. Among the three tested placements, the Einstein-speed assignment gives δ_BU equal to the gravitational and electromagnetic anchor value.

---

## References

[1] L. H. Thomas, The motion of the spinning electron, Nature 117, 514 (1926); The kinematics of an electron with an axis, Philosophical Magazine 3, 1–22 (1927).

[2] E. P. Wigner, On unitary representations of the inhomogeneous Lorentz group, Annals of Mathematics 40, 149 (1939).

[3] E. Borel, La théorie de la relativité et la cinématique, Comptes Rendus 156, 215 (1913).

[4] A. A. Ungar, Analytic Hyperbolic Geometry and Albert Einstein's Special Theory of Relativity, 2nd ed., World Scientific, Singapore (2008).

[5] V. Palge, C. Pfeifer, Thomas–Wigner rotation as a holonomy for spin-1/2 particles, Physical Review A 109, 032206 (2024), arXiv:2310.08121.

[6] C. W. F. Everitt et al., Gravity Probe B: Final results of a space experiment to test general relativity, Physical Review Letters 106, 221101 (2011).

[7] Korompilias, B. (2025). Common Governance Model: Mathematical Physics Framework. Zenodo. https://doi.org/10.5281/zenodo.17521384

[8] Korompilias, B. (2025). Gravitational theory in the Common Governance Model: causal preservation of ancestry through identity and individuality. https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_Gravity.md

[9] Korompilias, B. (2025). The fine-structure constant from geometric first principles. https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_Fine_Structure.md

[10] Korompilias, B. (2025). Energy scale structure in the Common Governance Model: a geometric approach to unification. https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_Energy_Scales.md

[11] Korompilias, B. (2025). CGM units analysis: geometric foundation of physical reality. https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_CGM_Units.md

[12] Korompilias, B. (2025). CGM geometry coherence analysis. https://github.com/gyrogovernance/science/blob/main/docs/Findings/Analysis_Geometric_Coherence.md
