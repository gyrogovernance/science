# Analysis: CGM Holonomy

## Path Memory in the Common Governance Model: Continuous Structure and Finite Realization

**Citation:** Korompilias, B. (2025). Common Governance Model: Mathematical Physics Framework. Zenodo. https://doi.org/10.5281/zenodo.17521384

**Verification:** `experiments/cgm_holonomy_analysis.py` (37 integrity checks, all passing). Results are written to `experiments/cgm_holonomy_analysis_results.txt`.

---

## Abstract

Holonomy is the residual transformation that remains when a system is transported around a closed path in a curved space. The traversal returns to its starting point, while the orientation carried along the path does not. This document establishes the holonomy structure of the Common Governance Model (CGM) in three layers. The first layer contains exact algebraic results that follow from the CGM threshold angles alone, including a closed form for the SU(2) commutator holonomy. The second layer places the CGM stages at their gyrovector coordinates and derives the BU dual-pole holonomy in closed form as an elementary function of two thresholds, from which the closure ratio ρ ≈ 0.9793 and the aperture gap Δ ≈ 0.0207 follow as definitions. The central structural result of this layer is a conjugation theorem. The palindromic traversal of all payload stages preserves the holonomy angle while transporting its axis, which separates the magnitude of path memory from its orientation. The third layer verifies the finite realization of the same architecture in the Holonomic Quantum Virtual Machine (hQVM), where holonomy appears as an order-two operator structure on a 4096-state manifold.

---

## 1. Scope and Terminology

This document treats one subject, which is the memory that closed paths leave behind in the CGM state geometry. It establishes the definitions, the closed forms, the invariance properties, and the finite realization of that memory. Physical applications of the quantities derived here, including the fine-structure constant and the gravitational coupling, are treated in separate documents (Analysis_Fine_Structure.md and Analysis_Gravity.md) and are outside the present scope.

The following vocabulary is used throughout.

A **path** is an ordered sequence of states. A **loop** is a path whose first and last states coincide. Each step of a path contributes a transport operator, and the composition of these operators around a loop is the **holonomy** of that loop. When the holonomy is a three-dimensional rotation, its conjugacy-invariant rotation angle is the **holonomy angle**, and a scalar measurement of nontrivial return is called a **defect**. The holonomy element is characterized by its angle, its axis, its unit quaternion, and its conjugacy class, equivalently by the eigenvalue set {1, exp(+iδ), exp(−iδ)}. The angle is the conjugacy-class invariant. The axis is the oriented realization in a chosen frame.

The production of holonomy follows from the joint role of the four CGM conditions. UNA permits order-dependent composition, so distinct routes to a shared endpoint can differ. ONA keeps those routes mutually comparable within one structure. BU closes the observable configuration while retaining a residual transformation that records which route was taken. Closure of the projected state therefore coexists with a nontrivial transport operator, and that residual is the geometric carrier of path memory.

Two mathematical settings appear. The continuous setting is a **gyrovector space**, which is the algebraic structure formed by relativistic velocity addition inside the open ball of radius c. Velocity addition in this space is neither commutative nor associative, and the correction operator that repairs composition is called the **gyration**. Gyrations are rotations, and they are the source of all continuous holonomy in this analysis. The finite setting is the hQVM, a computational machine defined in Analysis_hQVM_Wavefunction.md, whose relevant features are introduced in Section 13 before they are used.

---

## 2. The CGM Thresholds

CGM is built from four foundational conditions, named Common Source (CS), Unity Non-Absolute (UNA), Opposition Non-Absolute (ONA), and Balance Universal (BU). Their construction is given in CGM_Logic.md. For the present analysis, each condition contributes one dimensionless threshold, and the analysis depends only on these numbers.

| Condition | Threshold | Value | Character |
|---|---|---|---|
| CS | s_p = π/2 | 1.5707963... | angle |
| UNA | u_p = 1/√2 | 0.7071067... | amplitude, with associated angle arccos(u_p) = π/4 |
| ONA | o_p = π/4 | 0.7853981... | angle |
| BU | m_a = 1/(2√(2π)) | 0.1994711... | amplitude scale |

The UNA threshold is an amplitude whose associated angle is π/4, and the distinction between the amplitude and the angle is maintained throughout. Two exact identities hold at these values and are verified to full working precision.

```
π/2 + arccos(1/√2) + π/4 = π

Q_G · m_a² = 1/2,   where Q_G = 4π
```

The first identity states that the three stage angles close a triangle with zero angular defect. The second links the complete solid angle Q_G = 4π to the BU amplitude scale and yields the half-integer associated with the double cover of the rotation group. Both are consequences of the threshold definitions.

---

## 3. Exact SU(2) Commutator Holonomy

The first holonomy result uses only the threshold angles and the algebra of the group SU(2), the double cover of the rotation group.

Let U be the SU(2) rotation by π/4 about the x axis and let V be the SU(2) rotation by π/4 about the y axis. These are the UNA and ONA stage angles applied about orthogonal axes. The commutator

```
C = U V U† V†
```

measures the failure of the two rotations to commute. For two SU(2) rotations through angles β and γ whose axes have separation δ, the conjugacy angle φ of the commutator satisfies

```
cos(φ/2) = 1 − 2 sin²(δ) sin²(β/2) sin²(γ/2)
```

The threshold configuration sets δ = π/2 and β = γ = π/4. Using sin²(π/8) = (1 − 1/√2)/2, the half-angle identity reduces to

```
cos(φ_SU2/2) = (1 + 2√2) / 4
```

and therefore

```
φ_SU2 = 2 · arccos((1 + 2√2) / 4) = 0.5879007626540203 rad = 33.6842°
```

The script computes the commutator with 80-digit matrix arithmetic and confirms the closed form with a residual of 7.4 × 10⁻⁸¹. This angle is the exact continuous benchmark of the analysis. Two rotations whose individual angles are fixed by the CGM thresholds generate, through their commutator alone, a rotation of about 33.7 degrees. Order of operations carries geometric content at these thresholds.

---

## 4. Calibration of the Rotation Machinery

The BU dual-pole and palindrome results below are computed with a software implementation of the gyration operator. Before that implementation is used at the CGM stage coordinates, it is validated against an independent analytic standard.

The standard is the Thomas-Wigner rotation of special relativity. When two boosts with velocities u and v are composed, the result is a boost combined with a spatial rotation, and for small speeds the rotation angle approaches ||u × v|| / (2c²). The calibration evaluates the implemented gyration angle against this formula on a fixed deterministic grid of 576 velocity pairs per speed bound, for maximum speeds from 0.02c to 0.10c.

| max speed (units of c) | fitted slope | max residual |
|---|---|---|
| 0.02 | 1.000132 | 2.0 × 10⁻⁸ |
| 0.03 | 1.000298 | 1.0 × 10⁻⁷ |
| 0.05 | 1.000830 | 8.0 × 10⁻⁷ |
| 0.08 | 1.002128 | 5.2 × 10⁻⁶ |
| 0.10 | 1.003330 | 1.3 × 10⁻⁵ |

The slope error scales with the square of the speed bound (measured order 2.003) and the absolute residual scales with the fourth power (measured order 4.006). Both orders match the known series structure of the Wigner angle, so the implementation reproduces the analytic behavior across the tested range rather than at a single tolerance point. At the CGM stage coordinate magnitudes the matrix layer agrees with the analytic formulae to about 10⁻⁸, and matrix-layer comparisons in this document use that tolerance.

---

## 5. CGM Stage Coordinates in the Gyrovector Space

The BU dual-pole path of Analysis_CGM_Constants.md is the loop ONA → BU+ → BU− → ONA. In the Einstein gyrovector model with c = 1 the CGM stages occupy the coordinates

```
UNA  = (1/√2, 0, 0)
ONA  = (0, π/4, 0)
BU+  = (0, 0, +m_a)
BU-  = (0, 0, -m_a)
```

BU appears as a pair of opposite poles on the third axis, which is the dual character of the balance condition. The ONA threshold enters as the coordinate magnitude on the second axis. CS supplies the reference frame within which the other stages are defined and is not a location that transport visits; the CS threshold π/2 also exceeds the open unit ball and so cannot serve as a velocity coordinate. The script enforces the ball constraint and confirms that all four payload vectors lie inside it.

The four stages on the path are the **payload stages**, and CS is the **gauge frame**. Section 10 shows that this split matches the structure of the finite machine, where an 8-bit instruction consists of 6 payload bits framed by 2 gauge bits.

---

## 6. The BU Dual-Pole Holonomy in Closed Form

The central loop visits the ONA stage, crosses to the positive BU pole, crosses to the negative BU pole, and returns.

```
ONA → BU+ → BU- → ONA
```

This loop consists of two gyration corners joined by a pole crossing. The pole crossing contributes no rotation because BU+ and BU- are collinear, so the holonomy is generated entirely at the two ONA-BU corners. Each corner is a gyration of two boosts, one of magnitude π/4 and one of magnitude m_a, separated by a right angle. The Wigner angle for boosts of unequal magnitudes β₁ and β₂ separated by an angle θ is

```
ω(β₁, β₂, θ) = 2 · arctan( sin(θ) k(β₁) k(β₂) / (1 + cos(θ) k(β₁) k(β₂)) )
```

where the half-rapidity function

```
k(β) = β / (1 + √(1 - β²))
```

equals tanh(atanh(β)/2). At θ = π/2 one has sin(θ) = 1 and cos(θ) = 0, so the corner angle reduces to

```
ω = 2 · arctan( k(π/4) · k(m_a) )
```

and since the two corners share the same rotation axis, their angles add without correction. The full loop holonomy angle, named the **BU dual-pole holonomy** and written δ_BU, has the closed form

```
δ_BU = 4 · arctan( k(π/4) · k(m_a) )
```

which is the same quantity written in Analysis_CGM_Constants.md as δ_BU = 2 × ω(ONA ↔ BU). Evaluated at 80-digit precision,

```
k(π/4)  = 0.4851158626411627
k(m_a)  = 0.1007479000361957
ω       = 0.0976710891288310  rad
δ_BU    = 0.1953421782576621  rad  = 11.19°
```

The rotation axis of the loop is the direction ONA × BU, which for these coordinates is the negative x axis. The holonomy therefore consists of a scalar angle together with an oriented axis, and Section 9 shows that these two components behave differently under transport.

---

## 7. The Closure Ratio and the Aperture Gap

Two derived quantities compare the loop defect to the BU amplitude scale.

```
ρ = δ_BU / m_a = 0.9793004544973297

Δ = 1 - ρ      = 0.0206995455026703
```

The **closure ratio** ρ states that the accumulated dual-pole defect fills about 97.93 percent of the aperture scale m_a. The **aperture gap** Δ is the remaining fraction, about 2.07 percent. Within CGM these two numbers carry the balance interpretation developed in CGM_Logic.md, where near-closure provides structural stability and the residual gap keeps reconstruction of the system's history possible. In the present document they are definitions. Once δ_BU and m_a are fixed, ρ and Δ contain no further freedom.

The closed form permits an expansion of ρ in the BU amplitude. In the limit of vanishing m_a, the arctangent linearizes and

```
ρ(m_a → 0) = 2 · k(π/4) = 0.9702317252823254
```

The full value exceeds this baseline by a finite-amplitude correction.

```
baseline gap      1 - ρ(0)  = 0.0297682747176746
finite correction ρ - ρ(0)  = 0.0090687292150043
final gap         Δ          = 0.0206995455026703
```

The decomposition shows where the aperture comes from. The ONA geometry alone already fixes a closure of 97.02 percent, leaving a baseline gap near 2.98 percent. The finite BU amplitude closes a further 0.91 percentage points, producing the final gap of 2.07 percent. The correction is of second order in m_a with coefficients determined by k(π/4), so the aperture gap is an analytic function of the thresholds with no adjustable content.

---

## 8. Verification of the Matrix Realization

The closed form is checked against a direct matrix computation of the loop. The gyration operator produces one 3 by 3 rotation matrix per leg, the matrices are composed in path order, and the angle of the product is extracted through a quaternion representation.

The matrix loop angle is 0.1953421832831577, which differs from the analytic value by 5.0 × 10⁻⁹, within the matrix-layer tolerance. The two corner legs report angles of 0.0976710883 and 0.0976710950, equal to each other within 6.7 × 10⁻⁹, and the middle leg reports an angle of zero to machine precision, confirming the flat pole crossing. The sum of the two corner angles reproduces the total to 6.3 × 10⁻¹⁵, confirming the shared-axis additivity stated in Section 6. This additivity is a property of the orthogonal configuration, in which both corners rotate about the same axis, and does not hold for general loop compositions.

The product matrix itself passes structural audits. Its deviation from orthogonality is 5.3 × 10⁻¹⁶, its determinant differs from 1 by 4.4 × 10⁻¹⁶, its trace matches 1 + 2cos(δ_BU) to 2.0 × 10⁻⁹, and its eigenvalues are 1 together with the conjugate pair exp(±i δ_BU). The same orthogonality and determinant audit is applied to all ten canonical gyration matrices used anywhere in the analysis, with worst-case residuals near 10⁻¹⁵. The rotation machinery therefore produces genuine rotations at the actual threshold coordinates, beyond the small-velocity regime tested in Section 4.

Three invariance properties are then verified at the matrix level. Traversing the loop in reverse order produces the transpose of the forward holonomy, with the same angle, which confirms that the loop operator composes as a group element and records orientation. Starting the same cycle from the BU+ pole instead of from ONA leaves the angle and the trace unchanged, which confirms that the holonomy angle is a property of the loop as a conjugacy class rather than of the chosen starting point. Rotating all four stage coordinates by a global rotation Q transforms the holonomy operator to Q H Qᵀ while leaving the angle unchanged, which confirms that the angle is independent of the coordinate basis. Residuals for all three checks sit at or below the 10⁻⁸ matrix-layer tolerance.

---

## 9. The Palindromic Conjugation Result

The BU dual-pole loop uses two of the four payload stages. The full payload traversal visits all of them in a palindromic order.

```
UNA → ONA → BU+ → BU- → ONA → UNA
```

This path places six payload positions on a five-edge closed walk, moving outward from UNA through ONA to the BU pole pair and returning through the same stages in reverse. In the eight-position phase layout of the finite instruction unit the same structure appears as CS | UNA ONA BU | BU ONA UNA | CS, with CS occupying the two outer gauge positions and the six internal positions matching the continuous payload walk. The measured holonomy angle of this path equals the BU dual-pole angle to machine precision, while the rotation axis differs. The BU loop axis is (-1, 0, 0), and the palindrome axis is (-0.9224, 0.3863, 0).

The equality of angles together with the change of axis follows from a theorem of gyrogroup theory. Ungar's inversion identity states that for any two gyrovectors u and v,

```
gyr(v, u) = gyr(u, v)⁻¹
```

so the return leg through a stage pair applies the inverse of the outbound gyration. Writing A for the outbound gyration from UNA to ONA and H_BU for the dual-pole loop operator, the palindrome operator factors as

```
H_pal = A · H_BU · A⁻¹
```

The palindrome conjugates the BU holonomy by the UNA-ONA transport. Conjugation preserves the rotation angle and maps the rotation axis by A, so

```
angle(H_pal) = δ_BU
axis(H_pal)  = A · axis(H_BU)
```

The script verifies each component of this statement. The implemented reverse gyration matches the transpose of A to 5.9 × 10⁻⁹, which validates the inversion identity in the implementation. The conjugated operator matches the directly composed palindrome to the same residual. The transported BU axis matches the measured palindrome axis with alignment 1 - 2.2 × 10⁻¹⁶, and the scalar component of the quaternion, which encodes the angle, agrees between the two loops to machine precision.

The structural content of this result is the separation of path memory into two channels. The magnitude of the memory, the angle δ_BU, is created at the BU pole structure and is invariant under the surrounding traversal. The orientation of the memory, the axis, is transported by the UNA-ONA gyration. The outer stages relocate where the memory points without altering how much memory there is.

---

## 10. Dependency Structure

The closed form makes the dependency of the holonomy on the thresholds explicit. The angle δ_BU is a function of θ_ONA and m_a alone. The UNA threshold does not appear in the formula, and its derivative is zero identically because the BU loop never visits the UNA stage. UNA influences the holonomy only through the conjugation of Section 9, acting on the axis. CS enters only as the gauge frame.

The continuous layer therefore has a rank-two parameter dependence. Two thresholds set the magnitude of the path memory, one threshold steers its orientation, and one threshold fixes the frame. The sensitivity of the magnitude to its two parameters is quantified by logarithmic derivatives at the canonical point.

```
(θ_ONA / δ_BU) · d(δ_BU)/dθ_ONA = 1.6146
(m_a  / δ_BU) · d(δ_BU)/dm_a    = 1.0190
```

The response to m_a is close to linear, with the excess above 1 accounted for by the finite-amplitude correction of Section 7. The response to the ONA threshold is superlinear. Finite-difference derivatives of the matrix realization match derivatives of the analytic formula to a few parts in 10⁴, consistent with the matrix-layer tolerance.

---

## 11. The Wigner Map at the Canonical Thresholds

The equal-speed Wigner rotation evaluated at the UNA and ONA thresholds, with u_p as the common boost speed and θ_ONA as the separation angle, takes the value

```
ω(u_p, θ_ONA) = 0.2155499101533235
```

At the same point the local geometry of the Wigner map admits closed forms for the partial derivatives

```
dω/dβ = (12√2 - 4) / 17 = 0.7629742793221847
dω/dθ = (21 - 12√2) / 17 = 0.2370257206778153
```

and their sum equals 1 as an algebraic identity. The response of the Wigner angle at the canonical thresholds therefore splits into a boost-magnitude share of 76.3 percent and an angular share of 23.7 percent. The numerical derivatives match the closed forms to better than 10⁻⁴⁰ at working precision.

---

## 12. Precision Governance

The angle δ_BU feeds downstream analyses, including the fine-structure derivation in Analysis_Fine_Structure.md, where the leading expression scales as the fourth power of δ_BU. A relative change ε in δ_BU therefore produces a relative change of about 4ε in that expression.

The analytic closed form of Section 6 is the canonical value. The shared constant BU_HOLONOMY_ANGLE in the repository constants module is defined from this closed form, and the script verifies that its own 80-digit evaluation matches the shared constant with zero relative difference. Downstream modules import the shared constant.

---

## 13. The Finite Realization in the hQVM

The Holonomic Quantum Virtual Machine is a finite computational machine, specified in Analysis_hQVM_Wavefunction.md, that realizes the CGM architecture in exact integer arithmetic. Three of its features are relevant here and are summarized before use.

The instruction unit of the machine is an 8-bit byte whose bit positions carry the four CGM stage labels in palindromic order,

```
CS | UNA ONA BU | BU ONA UNA | CS
```

The two outer bits carry the CS label and act as a frame selector, and the six inner bits carry the payload stages. Reading the four stage labels forward through the first half and comparing with the reverse reading through the second half defines the **fold** of the byte. The four phase pairs are (CS, CS), (UNA, UNA), (ONA, ONA), and (BU, BU). A byte whose two readings agree at every stage position is called flat, and the count k of disagreeing positions, from 0 to 4, measures the byte's internal curvature.

For a fixed set of k disagreeing pairs there are 16 assignments of the common binary values on those pairs, and there are C(4, k) ways to choose which pairs disagree. The number of bytes at disagreement grade k is therefore

```
N(k) = 16 · C(4, k)
```

which produces the distribution

| Disagreeing pairs k | Byte count N(k) |
|---|---|
| 0 | 16 |
| 1 | 64 |
| 2 | 96 |
| 3 | 64 |
| 4 | 16 |

with total 256. The match establishes that the four stage-position comparisons behave as four independent binary observables, each disagreeing in half of all cases. Sixteen bytes are flat and 240 carry curvature. The central BU-to-BU comparison disagrees in 128 of 256 bytes, a fraction of one half.

The machine state lives on a manifold of 4096 states with two six-bit coordinates. The six-bit chirality word χ = u XOR v grades the state between two constitutional horizon sectors. Certain distinguished instruction words of length two, named W2 and W2', act on this manifold as involutions, meaning operators that square to the identity. In the six-bit chart, W2 flips all six chirality bits by the mask 63 = 2⁶ − 1, exchanging a chirality word with its complement, and maps the shell grade s to 6 − s, thereby exchanging the two extremal regions of the state space. Together with the identity and their product they form a Klein four-group, a commutative group of four elements each of order at most two, referred to as K4.

The canonical W2 and W2' certificate passes in full. The operators are involutions, they exchange the extremal state regions as required, and the associated shell and chirality transformations hold on all 4096 states. These finite results carry no numerical tolerance because the arithmetic is exact. The full K4 composition table and its permutation spectrum are developed in Analysis_hQVM_Wavefunction.md.

---

## 14. Byte-Horizon Aperture Quantization

Section 7 fixes the continuous aperture gap Δ from the holonomy ratio. Section 13 fixes the instruction unit as an 8-bit byte, so the natural discrete scale at that horizon is 256 ticks. The byte-horizon quantization of the aperture is the nearest integer number of ticks to 256 · Δ.

```
256 · Δ = 5.299083648683975

round(256 · Δ) = 5

Q_256(Δ) = 5/256 = 0.01953125
```

Thus the finite aperture at the byte horizon is five ticks open out of 256. Relative to the continuous gap,

```
|Δ − 5/256| / Δ = 0.05645
```

so the dyadic value lies about 5.6 percent below Δ. This is the quantity written A_kernel = 5/256 in the measurement and QuBEC reports, and stored in the shared constants module as APERTURE_GAP_Q256 = 5.

The same gap participates in the depth-four closure count of the machine. Four successive bytes contribute 48 chirality bits (4 · 12), and

```
48 · Δ = 0.9935781841281744
```

which sits 0.64 percent below unity. The reciprocal scale 1/48 is therefore the depth-four companion of Δ, while 5/256 is its expression at the single-byte horizon. The turn-normalized holonomy δ_BU / (2π) ≈ 1/32 supplies the third natural scale; the ratio (1/48) / (1/32) = 2/3 is the chirality-to-space factor developed in hQVM_QuBEC_Theory.md. In the present document the operative finite statement is the byte-horizon identity

```
Q_256(Δ) = 5/256
```

with Δ taken from Section 7.

---

## 15. Continuous-Finite Correspondence

The continuous and finite layers realize the same architecture at different formal levels. The following table records the correspondence of structural roles.

| Continuous layer | Finite hQVM layer |
|---|---|
| closed path in the gyrovector space | operator word on the 4096-state manifold |
| holonomy angle as conjugacy invariant | nontrivial finite involution |
| BU dual-pole loop | W2 pole exchange |
| closure under return | W2 squared equals the identity |
| aperture gap Δ | byte-horizon dyadic 5/256 |
| palindromic payload path | byte fold across the central BU boundary |
| 6 payload positions | 6 payload bits |
| CS gauge frame | byte bits 0 and 7 as frame selector |
| conjugacy spectrum 1, exp(±iδ) | involution spectrum +1, -1 |

The continuous BU holonomy carries a general rotational phase with eigenvalues 1 and exp(±i δ_BU). The finite carrier operators, under their permutation-matrix lift, carry an order-two phase distinction with eigenvalues in {+1, −1}.

The layers localize curvature at related but distinct sites. In the continuous realization the holonomy is generated at the two ONA-BU corners while the pole crossing is flat. In the finite realization the fold disagreement is counted at the central BU boundary of the byte. Both descriptions organize a dual return through the balance stage.

The six payload positions of the continuous palindrome and the six payload bits of the byte both count degrees of freedom within the three-dimensional, six-degree-of-freedom framework that CGM derives. They are combinatorial structures inside that framework rather than additional spatial dimensions.

---

## 16. Falsification Criteria

The analysis fails if any of the following occurs.

1. The 80-digit SU(2) matrix computation departs from the closed form 2 · arccos((1 + 2√2)/4).
2. An independent implementation of the CGM stage coordinates yields a δ_BU that differs from 4 · arctan(k(π/4) · k(m_a)) beyond its stated numerical floor.
3. The loop holonomy angle changes under path reversal, cyclic re-rooting, or a global rotation of the stage coordinates.
4. The palindrome holonomy departs from the conjugacy class of the BU dual-pole holonomy, in angle or in transported axis.
5. The byte fold distribution departs from 16 · C(4, k), or the W2 and W2' certificate fails on any of the 4096 states.
6. The nearest 8-bit dyadic to Δ departs from 5/256, or the shared constant APERTURE_GAP_Q256 departs from 5.

---

## 17. Reproducibility

```
python experiments/cgm_holonomy_analysis.py
python experiments/hqvm_wavefunction_kernel.py --k4-only
```

The first command runs all integrity checks, prints the full report, and writes `experiments/cgm_holonomy_analysis_results.txt`. The run is deterministic, uses 80-digit arithmetic for the analytic layer, and exits with a nonzero code if any check fails. The second command reproduces the finite certificate independently.

---

## 18. Conclusion

Three results form the foundation established here. The CGM threshold angles generate a nontrivial SU(2) commutator holonomy with the exact closed form 2 · arccos((1 + 2√2)/4). On the BU dual-pole path the holonomy angle is δ_BU = 4 · arctan(k(π/4) · k(m_a)), an elementary function of two thresholds, from which the closure ratio of 97.93 percent and the aperture gap of 2.07 percent follow as definitions, with the gap decomposing into a 2.98 percent baseline fixed by the ONA geometry and a 0.91 percentage point closure supplied by the finite BU amplitude. The palindromic traversal of the payload stages conjugates this holonomy, preserving the angle while transporting the axis, so the magnitude of path memory is set at the balance stage and its orientation is steered by the surrounding stages.

The finite machine realizes the same architecture in exact arithmetic, with byte-level fold curvature distributed binomially and the balance-stage exchange operators verified as involutions on the full state manifold. At the byte horizon the continuous aperture quantizes as Q_256(Δ) = 5/256. The quantities established here, in particular δ_BU, ρ, Δ, and the dyadic aperture 5/256, are the fixed inputs that downstream analyses of physical couplings and the hQVM kernel consume.