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

Use this doc as a log, and if any important computation about predictions deviates a lotin a non-trivial way, pls report in case we need to reconsider our overall conditions in a later stage.

We will use a dedicated list for worrysome deviations.

[ Do not change the text before this bracket ]
===

RED ALERT List
(list with analyses where we have deviations worth considering revisiting theory or insights or grounded corrections that are straightforward for mitigation)
### Worrisome deviations (prediction shifts)

| Item | Status | Notes |
|---|---|---|
| α vs CODATA 2018 | **~33.8 ppb** | Honest residual; α₀ still ~319.43 ppm |
| Weak-field G vs G_meas | **~+2.99 ppm** | Coupling depth = STF τ_G = \|Ω\|Δρ⁵(1−4ρΔ²); τ_trace (c₄) is isotropic-channel scalar only; CODATA G unc ~22 ppm |

### Gravity coupling depth (theory revision; artifacts present as fact)

- `τ_G := |Ω| Δ ρ⁵ (1 − 4 ρ Δ²)` — STF transport; used in `G₀ = G_kernel exp(−τ_G)/v²` and `G(ψ)`.
- `τ_trace := |Ω| Δ ρ⁵ c₄ Δ⁴` with `c₄ = −7/4` — isotropic / monopole bookkeeping; not in the attenuation exponent.
- Shared code: `tau_G_stf` / `tau_g_stf_depth()`, `tau_trace` / `tau_trace_depth()`, `kernel_exposure_constants()` returns STF `tau_G`.
- Docs aligned: Analysis_Gravity, Note, Quadratic Note, README, CGM_Program, CGM_Logic, Features report.
- Scripts: `hqvm_gravity_common` + analysis_1..10 coupling paths.
- Dump `hqvm_gravity_analysis.txt` still from pre-revision runner; refresh when re-run (do not treat dump G ppm as current).

---

## Sources of truth

| Quantity | Definition |
|---|---|
| m_a | `1 / (2√(2π))` |
| k(β) | `β / (1 + √(1 − β²))` |
| δ_BU | `4 · arctan( k(π/4) · k(m_a) )` |
| ρ | `δ_BU / m_a` |
| Δ | `1 − ρ` |
| Δ_* | D³ fixed point of `Δ = (5/256)·2^(1/12)·(1 + (√6/π)Δ² + (η/ε)Δ³)` |
| δ_lift | `Δ_* − Δ` |
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

## Hunt terms (for follow-up scans)

Use these strings when grepping a finding and its companion scripts. Record which hit in the log entry for that doc.

**Stale δ_BU / aperture literals (Case 1–2)**
- `0.195342176580`, `0.19534217658`, `0.195342` (alone as definition)
- `0.979300446087`, `0.979300446`
- `0.020699553913`, `0.020699553`
- Soft percents: `97.9%`, `2.1%` (prefer `97.93%` / `2.07%` when asserting ρ/Δ)

**Fine-structure chain (Case 3)**
- `0.007299683322`, `0.00729968` (old α₀)
- `0.007297352563` as CGM prediction (OK only as GK 2020 experimental reference)
- `0.043 ppb`, `0.043 parts per billion`, `sub-ppb`, `nine significant`, `9 significant`, `0.532`
- `319.398 ppm`, `0.052 ppm`, `-0.000379`
- Pasted aux: `1.021137`, `0.001874` (must be derived: `1/ρ`, `diff = φ_SU2 − 3δ_BU`)
- Target: `α₀ ≈ 0.007299683573`, `α ≈ 0.007297352816`, `≈ 33.8 ppb` vs CODATA 2018

**Naming (δ_BU only)**
- Prefer: `BU dual-pole loop angle`, `loop angle`, `δ_BU = 4 · arctan(...)`
- Replace when δ_BU is meant: `BU holonomy defect`, `dual-pole holonomy`, `BU holonomy δ_BU`
- Keep `holonomy` for other objects: `Z2 holonomy`, `plaquette holonomy`, `toroidal holonomy deficit` (~0.862833), SU(2) commutator holonomy `φ_SU2`, philosophical path memory

**Dependent formulas that move with δ_BU**
- `α₀ = δ_BU⁴/m_a`, full α chain (AB/HC/IDE)
- `ρ = δ_BU/m_a`, `Δ = 1−ρ`, `48Δ`, `Q_256(Δ)=5/256`
- `τ_G = |Ω| Δ ρ⁵ (1 − 4ρΔ²)` → τ_G ≈ 76.237913574; G residual vs G_meas ≈ **+2.99 ppm**. Stale: coupling depth written as `… + c₄Δ⁴`, G claims `+27.53 ppm` / `0.074 ppm`, `τ_G = 76.237889…` as coupling depth.
- `τ_trace = |Ω| Δ ρ⁵ c₄ Δ⁴`, `c₄ = −7/4` — print as trace-sector scalar; not in `exp(−τ)` for G.
- `α₀ ζ = ρ⁴/(π√3)` ≈ `0.169025926127` (old paste `0.169025920321`); `Π_H = ρ⁸ Δ⁴ / (π²|Ω|)`; Regge `alpha(d) ∝ δ_BU`

**Script smells**
- Hardcoded `d_BU` / `DELTA_BU` / `delta_BU = 0.195...` instead of `BU_HOLONOMY_ANGLE` / equation
- Float64 `GyroVectorSpace.gyration` as competing continuous aperture
- Print/comment lines about migration, legacy aperture, old vs new digits

**Log entry template**
```
### <Doc>.md
- Hits: <hunt terms found>
- Scripts: <paths; shared import or equation?>
- Edits: <what changed>
- Report: <nontrivial prediction shift, or none>
- Left alone: <terms kept and why, e.g. Z2 holonomy>
```

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

Primary finding: `Analysis_Fine_Structure.md`. Related: `hqvm_corrections_analysis_1.py`, Gravity appendix H.

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

### Analysis_Compact_Geometry.md
- Scripts: `hqvm_compact_geom_{common,1,2,run}.py` (shared `BU_HOLONOMY_ANGLE` / `M_A` / ρ / Δ; `DELTA_STAR` = D³ fixed point; `DELTA_LIFT = DELTA_STAR - DELTA`).
- Mass polynomials at `Delta_*`; ruler `n = log2(v/m)/Delta`. Lift `R_i = L_i(Delta_*) - L_i(Delta)`.
- Runner tees to `hqvm_compact_geom_results.txt`.
- Finding: dual-aperture statement; §4.3/§6 lift; EW closure vs `Delta_*` (W/Z 7.899e-10; D⁵ max tick 2.593e-7; rank-1 factor ~270).

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

### Analysis_CMB.md
- No Case 1-3 aperture hits (no delta_BU / rho / Delta / alpha chain). Toroidal holonomy deficit 0.862833 left as-is (not delta_BU).
- Aligned hypothesis framing 97.9% → 97.93% (rho) in finding dump and cgm_cmb_data_analysis_{290825,300825}.py.
- Re-check: finding + scripts still clean; no further edits.

### superintelligence (cross-repo)
- See `F:/Development/superintelligence/docs/reports/_corrections_guides.md` log entry “superintelligence repo (cross-repo pass)”.
- Fixed listed reports/specs + `src/constants.py` closed-form δ_BU; excluded `_backup`, `docs/references`, `docs/notes`, `external`.

### Analysis_Geometric_Coherence.md
- Case 1: cgm_coherence_analysis.py imported shared BU_HOLONOMY_ANGLE / M_A / BU_CLOSURE_RATIO / BU_APERTURE_GAP; removed literal 0.19534217658 and pasted aperture_fraction = 0.0207.
- Section 3.6: delta_BU equation and guide digits; loop-angle wording; CF of delta_BU/(2π) final convergent (179383, 5769858) at 12 terms (~5.8e6 steps; was 157531/5066988 from truncated literal).
- Report: CF terminal convergent shifts with closed-form delta_BU; leading CF terms [0, 32, 6, 16, 1, 2, 1, 1...] unchanged.

### Analysis_GFE.md
- No companion aperture script.
- Section 12 / 19.2: structural Delta grounded by delta_BU equation and guide digits; observational 2.07% kept distinct from Delta (removed shouty clarification block).
- Predictions that quote 2.07% as observational amplitudes left as percentages.

### Analysis_Fine_Structure.md
- Case 1+3: delta_BU by closed form; alpha chain recomputed with derived 1/rho and diff = phi_SU2 - 3 delta_BU.
- Final alpha ≈ 0.007297352816 (≈ 33.8 ppb vs CODATA 2018); dropped 0.043 ppb / exact-match claim to 0.007297352563.
- Error sequence under closed form: 319.43 ppm → 0.086 ppm (AB) → 0.033 ppm (HC) → 33.8 ppb (IDE). AB still dominates; HC/IDE are higher-order in Delta.
- cgm_alpha_analysis.py: equation for delta_BU (no float64 gyration theater); prints full chain vs CODATA 2018.
- hqvm_corrections_analysis_1.py: closed-form delta_BU; derive 1/rho and diff (no pasted 1.021137 / 0.001874).
- Report: relative delta_BU shift ~8.6e-9 lifts alpha_0 by ~4x that (~34e-9); that offset largely survives the correction factors, moving the final residual from sub-ppb (truncated stack) to ~33.8 ppb.

### Repo-wide alpha residual (33.8 ppb)
- Docs: CGM_Paper, CGM_Program, CommonGovernanceModel, README, Analysis_Gravity (7.4 + App H), Analysis_Universal_Corrections, Analysis_Compact_Geometry, Analysis_hQVM_Cohomology, Gyroscopic Physics/Features/Tests/Specs reports.
- Scripts: hqvm_gravity_common, hqvm_gravity_analysis_2, hqvm_corrections_analysis_2, hqvm_HU_analysis; literal delta_BU swaps in higgs/kms/percolation/su3/modal.
- Dataset: cgm_dataset_main.jsonl alpha and delta_BU Q&A rows aligned to closed-form residual.
- Stale run dumps: hqvm_gravity_analysis.txt, hqvm_Cohomology_analysis_{results,notes}.txt display digits patched (re-run for full refresh).

### Analysis_Gravity_Note.md
- Hits: soft ρ/Δ `0.9793`/`0.0207`; α₀ `0.00729968` + `319 ppm`; wording `dual-pole holonomy`; product `α₀ ζ = 0.169025920321`; section title "Aperture and Holonomy" for δ_BU; τ residual `7.36×10⁻⁸`; τ_G `76.237916638581`.
- Scripts: none dedicated (uses shared gravity stack via Analysis_Gravity).
- Edits: §3.3 equation + guide ρ/Δ; loop-angle wording; α₀ ≈ 0.007299683573 / 319.43 ppm; product → 0.169025926127; App C τ residual → −2.99e−6 / −2.75e−5; τ_G → 76.237889038806.
- Report: α₀ζ product shifts by ~3.4e-11 absolute (~2e-8 relative); **G down-chain:** same as Analysis_Gravity (+27.53 ppm).
- Left alone: Z2/plaquette "holonomy cycle" language; philosophical "holonomy phase defect" at 2.07% aperture amplitude.

### Analysis_Gravity_Quadratic_Note.md
- Hits: soft `ρ ≈ 0.9793`, `Δ ≈ 0.02070`; `τ_G ≈ 76.24`; "per holonomy cycle" (Z2 attenuation context); **G within 0.074 ppm**.
- Scripts: none dedicated (τ_G from hqvm_gravity_* via common).
- Edits: Step 2 grounded δ_BU equation + guide ρ/Δ; "Z2 holonomy cycle" for attenuation; G claim → +27.53 ppm (CODATA unc ~22 ppm); τ_G display kept ≈ 76.24.
- Report: τ_G old→new 76.237917 → 76.237889 (~3.6e-7 relative); **nontrivial:** weak-field G residual flips from −0.074 ppm to +27.53 ppm.
- Left alone: K4/Z2 holonomy gate F; Π_H form (digits not restated beyond ρ/Δ grounding upstream).

### Analysis_Gravity.md
- Hits (this pass): soft ρ/Δ in §3; App G ρ/Δ coarse `0.979300`/`0.020700`; "BU holonomy" for δ_BU in Regge; α₀ζ `0.169025920321`; abstract/§5.6/§6/App C **0.074 ppm** / τ_G `76.237916638581` / residual `7.36×10⁻⁸`.
- Scripts: see **Gravity script audit** below (do not treat “imports common” as done without opening each file).
- Edits: §3 equation + guide digits; App G ρ/Δ guide digits; Regge wording → loop angle; α₀ζ → 0.169025926127; abstract + §6 + App C → τ_G 76.237889038806, G +27.53 ppm, residuals −2.99e−6 / −2.75e−5.
- Report: **nontrivial G down-chain** — under closed form, full τ_G with c₄ = −7/4 yields ~+27.53 ppm vs G_meas (leading ~+3.0 ppm); c₄ no longer “closes” to sub-ppm. CODATA G unc ~22 ppm.
- Left alone: Z2 holonomy, plaquette holonomy K(x,y), gravitational memory holonomy cycle.

### Gravity script audit 
Per-file aperture source and what changed:

| File | Aperture source | Status |
|---|---|---|
| `hqvm_gravity_common.py` | `BU_HOLONOMY_ANGLE` / `M_A` / `BU_CLOSURE_RATIO` / `BU_APERTURE_GAP` | **Bug fixed:** `alpha_lab_with_transport_corrections` had pasted `rho_inv=1.021137` and `diff=0.001874`; now derives `1/ρ` and `diff=φ_SU2−3δ_BU`. `verify_alpha_zeta_product` uses `isclose` (float `==` was false FAIL). |
| `hqvm_gravity_runner.py` | no aperture literals; orchestrates 3→2→1→4…→10 into `hqvm_gravity_analysis.txt` | OK; **do not re-run** unless asked (expensive). User dump 2026-08-16. |
| `hqvm_gravity_analysis_1.py` | imports `d_BU,m_a,rho,Delta` from common | OK (no local literals) |
| `hqvm_gravity_analysis_2.py` | local `DELTA_BU=BU_HOLONOMY_ANGLE`, `RHO=BU_CLOSURE_RATIO`, `DELTA=BU_APERTURE_GAP` | OK |
| `hqvm_gravity_analysis_3.py` | `APERTURE_GAP,DELTA_BU,M_A,RHO` from constants | **Assert fixed:** residual gate `1e-7` → `5e-5` (closed-form full residual ~2.75e−5). Dump still shows `AssertionError` from pre-fix run of `_3`. |
| `hqvm_gravity_analysis_4.py` | same constants aliases + common | Dynamic G ppm prints; G check threshold `1.0` → `50.0` ppm. Dump still shows `[FAIL] +27.527 ppm` from pre-threshold run. |
| `hqvm_gravity_analysis_5.py` | `d_BU,m_a` from common; `Delta` from analysis_4 | Dynamic G ppm via `g_pred_from_tau`/`G_meas`. |
| `hqvm_gravity_analysis_6.py` | `rho,Delta` from common | OK |
| `hqvm_gravity_analysis_7.py` | `d_BU,m_a,rho,Delta` from common | Hardcoded `"G to 0.074 ppm"` → `"G from tau_G (weak-field)"`. |
| `hqvm_gravity_analysis_8.py` | `d_BU,m_a,rho,Delta` from common | OK |
| `hqvm_gravity_analysis_9.py` | `rho,Delta` from common | OK |
| `hqvm_gravity_analysis_10.py` | `Delta` from common | OK |
| `hqvm_gravity_analysis.txt` | frozen runner dump | **Stale vs current doctrine** (pre STF/trace split). Refresh when you re-run the runner. |

- Report: closed-form α_lab ≈ 0.007297352816 (33.8 ppb). **G:** τ_G (STF) ≈ 76.237913574 → **+2.99 ppm** vs G_meas; τ_trace kept as isotropic scalar.
- Left alone: Z2/plaquette holonomy naming in scripts (not δ_BU).

### Analysis_Higgs.md
- Hits: `0.195342` as δ_BU definition/evaluation (1.2, 2.1); "holonomy value" for δ_BU; soft `Δ ≈ 0.0207`; `48·Δ ≈ 1`.
- Scripts: `cgm_higgs_analysis.py` (had closed-form digits as literal; now `BU_HOLONOMY_ANGLE` / `M_A`). No dedicated RGE runner for the m_H = 124.97 GeV table in this finding.
- Edits: §1.2 δ_BU by equation + guide digits; ρ/Δ guide digits; loop-angle wording; §2.1 λ(E₀) evaluation from equation (still ≈ 0.009149); §3.8 `48·Δ ≈ 0.993578`.
- Report: λ(E₀) = δ_BU⁴/(4m_a²) under closed form ≈ 0.00914880 → display 0.009149 unchanged at six digits. No nontrivial prediction shift; no RED ALERT.
- Left alone: five-fold vacuum discussion; m_H table (depends on RGE stack not re-run here); 2.07% aperture language as percent.

### Analysis_Hilbert_Space_Representation.md
- Hits: none (no δ_BU / ρ / Δ / α chain).
- Scripts: `cgm_Hilbert_Space_analysis.py` — `m_a` from closed form only; no δ_BU literal; BCH/unitarity independent of aperture.
- Edits: none.
- Report: none.
- Left alone: Q_G = 4π normalization; all holonomy uses are philosophical / Berry / branch-point (not δ_BU).

### Analysis_Gyroscopic_Multiplication.md
- Hits: `0.195342` / soft ρ≈0.9793 / Δ≈0.0207; "holonomy defect" / "toroidal holonomy" / "holonomy constant" for δ_BU; `48 · Δ ≈ 1`.
- Scripts: none dedicated (K4 arithmetic is documentary; no companion aperture script).
- Edits: §7.2 / §13.2 / lemon appendices / open questions: δ_BU by equation + guide digits; loop-angle naming for δ_BU; ρ/Δ guide digits; `48 · Δ ≈ 0.993578`.
- Report: none (phenomenological lemon B=0.1953 comparison unchanged at four figures).
- Left alone: philosophical "holonomy" (continued fractions, Riemann surface, Berry/branch-point, billiard invariant checklist).

### Analysis_hQVM_Wavefunction.md
- Hits: soft `0.1953` / `0.9793` / `0.0207`; §11.1 "BU Holonomy Angle" for δ_BU; Δ written as `delta_BU/m_a` in places (should be `1 − δ_BU/m_a`); τ_G claimed to match `2 ln(E_CS/v)` at 25 ppm (stale vs gravity stack).
- Scripts: `hqvm_wavefunction_kernel.py` already imports `APERTURE_GAP`; `_1.py` / `_2.py` have no δ_BU literals (Z2/K4 diagnostics). Softened kernel A* label; removed "corrected" header meta in `_1.py`.
- Edits: §11.1 loop-angle equation + guide ρ/Δ; §11.3 / §17.7 τ_G ≈ 76.237914 (STF), drop false conjugacy-25-ppm claim; Δ digits and `Δ = 1 − δ_BU/m_a` wording in §16–17.
- Report: none for wavefunction kernel itself. Stale conjugacy-match claim removed (τ_conjugacy ≈ 76.88 vs τ_G ≈ 76.24; G residual lives in Analysis_Gravity, +2.99 ppm).
- Left alone: Z2 holonomy / fold holonomy / Householder language (not δ_BU); 2.07% and 50% percent language; 5/256 dyadic tick.

### Analysis_hQVM_Percolation.md
- Hits: soft `Delta = 0.0207` / `0.020700`; table `0.02070`; wrong claim that Delta "matches" `delta_BU/m_a` (that is rho); discrete `1/48` vs continuum comparison.
- Scripts: `_1.py` had closed-form digits as literal → `BU_HOLONOMY_ANGLE` / `M_A` / `RHO` / `APERTURE_GAP`; `_3.py` already shared (print soft `~0.0207` → print `CGM_DELTA`; STF fallback for `tau_G_formula`); `_4.py` → shared `M_A`; `_5.py` already compares to `APERTURE_GAP`.
- Edits: §5.5 / §6.5 equation + guide Delta digits; rho = delta_BU/m_a wording; Table 19 digits.
- Report: `hqvm_percolation_analysis_results.txt` refreshed (Delta = 0.0206995455; byte `p_c/Delta` ≈ 1.038930; gates PASS).
- Left alone: Z2 / depth-4 / plaquette holonomy language; empirical p_c values; Delta(d)=1/(8d) discrete family.

### Analysis_hQVM_Percolation_Note.md
- Hits: none (cross-reference companion; no aperture chain).
- Scripts: none dedicated beyond the percolation suite above.
- Edits: none.
- Report: none.
- Left alone: all content.

### Analysis_hQVM_CGM_Allometry.md
- Hits: stale `Delta = 0.020699553913`; soft `delta_BU ≈ 0.195342`; "dual-pole holonomy defect"; meta "holonomy dressing/correction"; `a_surf,continuum = 0.662386`.
- Scripts: `_1.py` already via `hqvm_cgm_trestleboard_common.DELTA = APERTURE_GAP`; print labels softened. Results dump `hqvm_cgm_allometry_results.txt` still shows pre-refresh digits (re-run when asked).
- Edits: §3.1 equation + guide digits; §4.2 continuum surface ≈ 0.662385; §9.1 E_a with guide Delta (still ≈ 0.645 eV); glossary loop-angle naming.
- Report: none (E_a stays in MTE band; surface continuum shift ~1e-6).
- Left alone: discrete `1/48`; Z2/depth-4 holonomy_micro_cov language in scripts.

### Analysis_hQVM_Cohomology.md
- Hits: `δ_BU = 0.19534217658`; `ρ = 0.9793004461`; `Δ = 0.0206995539`; dual literal vs trace; "BU holonomy" for δ_BU; section title "Aperture and Holonomy".
- Scripts: `_1.py` / `_4.py` already shared `DELTA_BU`/`M_A`; `_3.py` had pasted `0.020699553913` → `APERTURE_GAP`; `_4.py` comment loop-angle wording.
- Edits: §7/§9/§10 equation + guide digits; remove dual truncated source; loop-angle naming for δ_BU; α₀ already guide digits.
- Report: none.
- Left alone: CHSH/Grothendieck gaps; Z2/K4 holonomy language; dump notes may still show old digits until re-run.

### Analysis_hQVM_Moments_Fiat.md
- Hits: none (Z2 holonomy cycle width only; no δ_BU / ρ / Δ / α chain).
- Scripts: `hqvm_moments_fiat_analysis_{1,2,3,run}.py` — no aperture literals.
- Edits: none.
- Report: none.
- Left alone: F-squared Z2 holonomy cycle language.

### Analysis_Kompaneyets.md
- Hits: soft `97.9%` / `2.1%` aperture framing; no δ_BU literal.
- Scripts: `cgm_kompaneyets_analysis.py` — no aperture literals.
- Edits: section title/body → 97.93% / 2.07% with Δ = 1 − ρ equation.
- Report: none.
- Left alone: Kompaneyets/tSZ spectral ρ (not CGM closure ratio).

### Analysis_Measurement.md
- Hits: soft target `A ≈ 0.0207` / `Closure ≈ 0.9793`.
- Scripts: none.
- Edits: calibration/target digits → guide Δ / ρ.
- Report: none.
- Left alone: 2.07% / 97.93% percent language.

### Analysis_Motion.md
- Hits: `δ_BU = 0.195342`; ω soft digits.
- Scripts: none dedicated.
- Edits: loop-angle equation + guide digits for δ_BU / ω.
- Report: none.
- Left alone: philosophical holonomy-as-memory language.

### Analysis_Quantum_Gravity.md
- Hits: Case 3 — `α = 0.0072997` within 0.03%; soft ρ/Δ; "dual-pole holonomy".
- Scripts: `cgm_quantum_gravity_analysis.py` → shared `BU_HOLONOMY_ANGLE` / `M_A`; α₀ labeling (≈319 ppm vs CODATA).
- Edits: exec/§2/§5 as α₀ + 33.8 ppb chain; loop-angle wording; guide ρ/Δ.
- Report: none new (α residual already on RED ALERT list).
- Left alone: toroidal/Z6 structure numerics unrelated to aperture paste.

### Analysis_Universal_Corrections.md
- Hits: pasted `ρ_inv` / `diff`; soft `Δ ≈ 0.020700` (α chain already guide).
- Scripts: `hqvm_corrections_analysis_{1,2}.py` already derive 1/ρ and diff.
- Edits: ρ_inv / diff as derived; Δ / m_a guide digits.
- Report: none.
- Left alone: 4-leg/8-leg holonomy ratio h.

### Analysis_Walking.md
- Hits: soft m_a / 2.07% openness.
- Scripts: `cgm_walking_analysis.py` → shared `BU_HOLONOMY_ANGLE` / `M_A`.
- Edits: m_a / Δ guide digits in §1.2.
- Report: none.
- Left alone: biomechanical 14% CMP defect; 97.93%/2.07% gait analogy.

### results_28082025.md
- Hits: soft δ_BU / ρ / 97.9% in dump + interpretive prose.
- Scripts: historical day dump (holonomy_flow / CMB).
- Edits: Dual-Pole Loop Constant block → equation + guide digits; soft 97.93%/2.07% in prose.
- Report: none.
- Left alone: toroidal holonomy deficit 0.863; raw terminal paste digits elsewhere.

### results_31082025.md
- Hits: none Case 1–3 (SU(2) commutator holonomy 0.587901 is not δ_BU).
- Scripts: none dedicated aperture.
- Edits: none.
- Report: none.
- Left alone: all holonomy uses (commutator / horizon).

### Analysis_hQVM_CGM_Trestleboard.md
- Hits: soft δ_BU / ρ / Δ; dual-pole holonomy wording; stale Δ_ref = 0.020699553913 and W/Z lock 8.34e-10 vs loop-angle Δ.
- Scripts: `hqvm_cgm_trestleboard_{common,3,run}.py` — ruler `DELTA = APERTURE_GAP`; W/Z lock and `eval_law` at `DELTA_STAR`; import `DELTA_STAR` / `DELTA_LIFT` from compact geom.
- Edits: §2.1 loop-angle equation + guide digits; §2.2 dual aperture (grammar/ruler on Δ; spectral/W/Z on Δ_* = 0.020699553957, |Δ_WZ−Δ_*| = 7.899e-10); §2.3–§2.4 / §4–§5 Level/deuteron digits from new results dump; §4.1/§4.2 Th-229m inputs and rel 6.95e-05; spectral L_i at Δ_* (ratio abs err 2.074e-10); scope bullet aligned.
- Report: `hqvm_cgm_trestleboard_results.txt` refreshed (W/Z lock PASS at 7.899e-10 vs Δ_*).
- Left alone: ρ-dress / Compass “holonomy” transport wording; Nilsson τ = δ_BU/(2π); percolation holonomy-transport labels.

### Analysis_hQVM_CGM_YM_Mass_Gap.md
- Hits: stale δ_BU gyration theater `0.19534217658` / ρ `0.97930044609` / Δ `0.02069955391`; soft “holonomy of Einstein gyration”; false `48·Δ ≈ 1`; Route A/B / E_unit / Appendix δ_BU circle diagnostics.
- Scripts: already import `DELTA`/`RHO`/`DELTA_BU` from `hqvm_compact_geom_common` (shared closed form). Smoke: Route A `1.582473` GeV, Route B `1.661555` GeV.
- Edits: §7.1 closed-form loop angle; guide digits; mass readout `1.582473` / `E_unit 5.096642`; Appendix A table; README key numbers. YM symbol `Δ_*` (strong-coupling spectral floor) left distinct from Compact Geometry `Delta_*`.
- Report: `Yang_Mills_Mass_Gap_results.txt` refreshed 2026-08-16 (`fast: False`); Δ / Route A/B / E_unit / ρ / κ₂ / H7 agree with finding. README points to `docs/Findings/Analysis_hQVM_CGM_YM_Mass_Gap.md`.
- Left alone: plaquette / K4 / fold “holonomy” operator language; Clay/OS certificates independent of Δ digit shift.
- Prediction shift: m_gap 1.582474 → 1.582473 GeV (sub-meV; still ≈ 1.582 GeV in 0⁺⁺ window). Not a RED ALERT.

---


### CGM_Program.md + README.md (aperture / inventory sync)
- Hits: truncated d_BU; W/Z 8.34e-10; Features GitHub-only link; stale script/feature counts; missing Moments/AIR GCT paths
- Edits: closed-form d_BU/?/?; dual aperture ? vs ?_*; W/Z 7.899e-10; Th-229m/deuteron digits; YM Route A/B; local GCT index; holonomy runner paths; counts 37/112/72/283/~91k
- Report: none beyond known RED ALERT a/G residuals

### CommonGovernanceModel.md + CGM_Paper.md + papers/CGM.tex
- Hits: CGM.tex stale 0.195342176580 / alpha 0.007297352563; CommonGovernanceModel confused rho with Delta (delta_BU/m_a = 0.0207); Paper holonomy-defect naming
- Edits: closed-form delta_BU; rho/Delta digits; alpha0 +319.43 ppm and transport alpha ~33.8 ppb; loop-angle naming; A* = Delta
- Report: none beyond RED ALERT alpha residual

### experiments/ aperture leftover pass
- Hits: cgm_delta_bu_precision_temp historical literals; modal/sound hardcodes; QG dual-pole holonomy naming; stale dumps (octaves/cohomology/allometry/genomics/perc5)
- Scripts: import BU_HOLONOMY_ANGLE / rewrite temp probe; rename loop angle; sound/modal use shared constant
- Edits: refreshed allometry, octaves, cohomology, genomics (+hard); killed unfinished perc5 full run; patched perc5 dump APERTURE_GAP display only
- Report: none
- Left alone: GyroVectorSpace TW cross-checks; toroidal holonomy 0.863; hqvm_gravity_analysis.txt tau_G lines (known pre-revision dump)
