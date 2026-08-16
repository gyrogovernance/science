# Aperture Corrections Guide

We made a refinement on one of our most foundational invariant, the aperture. In the past we were using the value 0.195342176580 which was a wrong rounding. Now we use this equation δ_BU = 2ω = 4·arctan(r(θ_ONA)·r(m_a)). Like π, the decimal expansion is infinite; numerical work on scripts therefore must evaluate the equation rather than substituting a truncated literal.

```
k(π/4)  = 0.4851158626411627
k(m_a)  = 0.1007479000361957
ω       = 0.0976710891288310  rad
δ_BU    = 0.1953421782576621  rad  = 11.19°
```

This file defines the aperture quantities, the closed sources of truth, and how to propagate them through the CGM program (findings, specs, constants modules, and experiments).

Things we are looking for:
- direct corrections about the equation and value.
- indirect impact on other calculations.

How do we correct:
- on scripts we use the equation than a magic number. Define δ_BU by the equation or by importing/evaluating the shared constant.
- Do not keep a second continuous aperture source (truncated literal, float64 gyration theater, side-by-side “old digits”) as a competing definition.
- in docs we correct only what we should.
- we do not go obsessively add on every single doc of the other analyses too much about the aperture, but we do add the proper derivations when they are indeed necessary for self-contained reviews and reading by people who do not know anything about CGM, which are all people who read our work.
- No em dashes, no latex, no repetitions, but proper surgical integrations.

Our new Holonomy analysis defined the aperture in a more grounded and rounded way using more elaborate gyrogroup theory. As we will pass through our docs and scripts it is important to consider whether they uncover new insights or context to be added properly and formally in appropriate places.

Keep "holonomy" only when referring to the operator H_BU, or when using the general term "holonomy" philosophically (as in "holonomy is path memory").


| Old | New |
|---|---|
| BU dual-pole holonomy δ_BU | BU dual-pole loop angle δ_BU |
| The BU dual-pole holonomy is... | The BU dual-pole loop angle is... |
| the holonomy angle of Section 6 | the loop angle of Section 6 |
| the dual-pole aperture | the dual-pole loop angle (or just "the loop angle" when context is clear to avoid overusing the term) |


What I forbid:
- proposals for changes that convey editorial meta-comments. This is a constant bias of AI model reviewers that insist on adding prose about the corrections. I don't want any prose. We need to write what is, as if there is no history of another aperture. Do not write migration vocabulary into the artifact (corrected, historical, legacy, former, previous, superseded, before/after). Present the equation and the evaluation.
- repetitions.

[ Do not change the text before this bracket ]
===
---

## Sources of truth

| Quantity | Definition |
|---|---|
| m_a | `1 / (2√(2π))` |
| k(β) | `β / (1 + √(1 − β²))` |
| δ_BU | `4 · arctan( k(π/4) · k(m_a) )` |
| ρ | `δ_BU / m_a` |
| Δ | `1 − ρ` |
| φ_SU2 | `2 · arccos((1 + 2√2) / 4)` |
| α₀ | `δ_BU⁴ / m_a` |
| Q_256(Δ) | `5/256` (hQVM finite aperture. Nearest byte-horizon dyadic to Δ) |

Shared float evaluation: `gyroscopic.hQVM.constants.bu_holonomy_angle()` / `BU_HOLONOMY_ANGLE`.

Derivation and high-precision checks: `docs/Findings/Analysis_Holonomy.md`, `experiments/cgm_holonomy_analysis_{common,1,2,run}.py`.

Reference display digits (closed-form δ_BU):

```
δ_BU ≈ 0.195342178258
ρ    ≈ 0.979300454497
Δ    ≈ 0.020699545503
48Δ  ≈ 0.993578
α₀   ≈ 0.007299683573
α    ≈ 0.007297352816   (≈ 33.8 ppb vs CODATA 2018 α = 1/137.035999084)
```

Decimals in docs and logs are display truncations of these evaluations. They are not alternate definitions.

Do not mention about what is truncated or not - this is a violation of the editorial meta-comment condition.

---

## Case 1 — δ_BU pasted or dual-sourced

**Symptom.** Hardcoded `0.195342176580` (or nearby) as the definition of δ_BU; float64 `GyroVectorSpace.gyration` kept as a second continuous aperture beside the closed form.

**Handle.**

1. Replace the literal with the equation or the shared evaluator.
2. Recompute all dependents listed under Sources of truth.
3. Remove parallel continuous aperture sections kept only to compare truncated or float sources to the equation.

---

## Case 2 — Stale ρ, Δ, 48Δ

**Symptom.** Digits from a truncated δ_BU (for example ρ ≈ 0.979300446087, Δ ≈ 0.020699553913).

**Handle.** Re-evaluate `ρ = δ_BU / m_a` and `Δ = 1 − ρ` from Case 1. Keep Q_256(Δ) = 5/256 unless continuous Δ moves enough to change the nearest tick (it does not under the closed form).

---

## Case 3 — Fine-structure α chain

**Symptom.** α₀ / α / ppm / ppb figures from truncated δ_BU; claims of exact CODATA match that fail under the closed form.

**Handle.**

1. Recompute the full chain from Case 1 δ_BU, with `diff = φ_SU2 − 3 δ_BU` and `1/ρ` derived rather than pasted.
2. Report the actual residual vs the stated CODATA reference (closed-form chain ≈ 33.8 ppb vs CODATA 2018).
3. Update quoted α₀ ≈ 0.007299683573 and α ≈ 0.007297352816.

Primary finding still to align: `Analysis_Fine_Structure.md`. Related: `hqvm_corrections_analysis_1.py`, Gravity appendix H.

---

## Case 4 — Aperture meta in scripts or prints

**Symptom.** Comments or printed lines that narrate aperture migration policy, or that frame δ_BU / ρ / Δ as replaced stock.

**Handle.** Delete the meta. Print the equation, the evaluation, and neutral method labels only. This guide holds the policy.

---

## Add a case

When a new aperture-propagation pattern appears in the program, append:

```
## Case N — <short name>

**Symptom**

**Handle**

**Typical locations**

**Corrected values**

```

It is important to not recompute every time what we have corrected already in previous scripts/docs.

Keep this file about aperture quantities and their dependents across the CGM program. Do not turn it into a general writing or review handbook.

===

## Log

### Analysis_CGM_Constants.md
- δ_BU digits already matched closed form (ρ, Δ, α₀, α, W_residual/diff).
- Removed truncation-meta sentence at the δ_BU definition.
- Renamed BU dual-pole holonomy wording to loop angle where δ_BU is meant; α₂ label to Commutator transport.

### Analysis_CGM_Units.md
- δ_BU / ω stated by equation with guide display digits 0.195342178258 / 0.097671089129.
- ρ, Δ to guide digits.
- α₀ ≈ 0.007299683573; corrected α ≈ 0.007297352816 (33.8 ppb vs CODATA 2018); dropped old 0.007299734 and nine-figure match claim.
- Abstract and §8.1 aligned to those figures.
- BU stage / §5.2 wording: loop angle for δ_BU; kept philosophical “holonomy” in §5.3 title sense.

### Analysis_3D_6DOF_Proof.md
- Section 2.3: stage-angle defect delta = 0 distinguished from delta_BU equation (k, m_a).
- Corollary 6.1: m_a vs residual aperture gap ~2.07%; memory = loop angle delta_BU equation.
- Section 7 BU: stage-angle delta = 0 vs nonzero delta_BU; avoided aperture symbol Delta (BCH already uses Delta).
- experiments/cgm_3D_6DoF_analysis.py: imports shared BU_HOLONOMY_ANGLE / M_A; BU print distinguishes stage-angle defect from loop angle.

### Analysis_48_States.md
- delta_BU equation and guide digits; loop-angle wording.
- Continuous Delta = 1 - rho; 48*Delta ~ 0.993578; 1/48 as discrete companion (not Delta = 1/48 exact).
- lambda_0 = Delta/sqrt(5); alpha_0 digit aligned.
- angle_45_48_aperture_analysis.py: shared BU_HOLONOMY_ANGLE / M_A.
- test_exact_48delta.py: reports continuous Delta vs 1/48 (no forced 48*Delta = 1).

### cgm_bsm_analysis.py
- Imports shared BU_HOLONOMY_ANGLE / M_A; phi_SU2 from closed form.
- Delta = 1 - rho from loop angle (no forced Delta = 1/48); lambda_0 = Delta/sqrt(5).
- Print labels: 48Delta without exact-1 claim.

### Cross-doc 48Delta (near unity)
- CGM_Program.md: 48Delta ≈ 0.993578; 1/48 as discrete companion (was exact 48Delta = 1).
- Analysis_Compact_Geometry.md: 48*Delta / epsilon / eta / continuous Delta digits from closed-form aperture.
- cgm_byte_formalism_analysis.py: shared BU_HOLONOMY_ANGLE; print near-unit 48*Delta (not exact 1).
- Already aligned: Analysis_CGM_Constants, Analysis_48_States, Analysis_Holonomy, YM/Mass-Gap/Multiplication (≈ 1 language).

### Analysis_Axiomatization.md
- No Case 1-3 aperture hits in the finding or cgm_axiomatization_analysis.py (BCH Delta only). No aperture text added.

### Analysis_BH_Aperture.md
- Grounded rho/Delta from delta_BU equation in Section 2.1; intro cites those digits.
- cgm_bh_aperture_analysis.py: m_a from shared M_A.

### Analysis_BH_Universe.md
- Abstract, Section 2.2, diagnostics, Appendix C: delta_BU equation and guide digits for rho/Delta.
- cgm_bh_universe_analysis.py: shared BU_HOLONOMY_ANGLE / M_A / Delta; removed pasted 0.0207.

### Analysis_Capacity_Concepts.md
- Section 1.3 / 4.2 / 5 / 6: delta_BU equation and guide rho/Delta digits; loop-angle wording for memory capacity.
- Aligned inconsistent 97.9%/2.1% memory lines to 97.93%/2.07%.
- Target A identified with Delta (guide digits). No companion script.

### Analysis_Energy_Scales.md
- m_a display to guide digits. No delta_BU / rho / Delta / alpha chain in this finding. No companion aperture script.
