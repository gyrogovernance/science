# Analysis: hQVM CGM Trestleboard

**Citation:** Korompilias, B. (2025). Common Governance Model: Mathematical Physics Framework. Zenodo. https://doi.org/10.5281/zenodo.17521384

**Reproducibility:** `experiments/hqvm_cgm_trestleboard_results.txt`. Scripts: `hqvm_cgm_trestleboard_1.py` through `hqvm_cgm_trestleboard_4.py`, `hqvm_cgm_trestleboard_common.py`, `hqvm_cgm_trestleboard_run.py`. External data catalogs: `data/catalogs/ensdf/` and `data/catalogs/fusion/` (SOURCE files in each directory). Local PDF copies of primary isomer references: `docs/references/` (`SOURCE_Th229m.txt`, `SOURCE_U235m.txt`).

**Subject classes (arXiv-style):** nucl-th; nucl-ex; physics.plasm-ph; math-ph

**Keywords:** Common Governance Model, hQVM trestleboard, nuclear isomer, Th-229m, Delta-ruler, fusion S-factor, percolation hierarchy, alpha decay, beta decay, Coulomb barrier

## 1. Scope and Claims

### 1.1 Scope

The Common Governance Model (CGM) is a Hilbert-style axiomatization of fundamental physics and information science. It begins from a single foundational principle and develops subsequent structure through a four-stage operational sequence whose governing requirement is that ancestry remain preservable under recursive operations. The stages are CS (chirality), UNA (rotational three-axis), ONA (translational three-axis), and BU (depth-four closure). Within that construction, the finite kernel is a discrete algebraic system with 4096 reachable states, organised into seven shells by a binomial distribution and carrying a self-dual [12,6,2] binary code. The Holonomic Quantum Virtual Machine (hQVM) is the discrete realization of CGM on that register. The executable kernel supplies exact combinatorial, spectral, and percolation data with no freely adjustable parameters.

This document applies three kernel instruments—Square (percolation hierarchy), Compass (holonomy dress), and Level (Delta-ruler placement)—to nuclear excitation, deuteron binding, alpha and beta decay on the kernel carrier, and fusion barrier and resonance geometry on the same ruler that carries electroweak structure. The nuclear claims it establishes are

```
result                                       value
aperture gap Δ from W/Z masses              0.0206995539 (|Δ_WZ − Δ_ref| = 8.34e-10)
minimum isomeric excitation E_min          8.3563 eV vs Th-229m 8.3557 eV
deuteron binding E_d                        2.2242 MeV (|rel| 8.9e-05 vs 2.2240 MeV)
alpha shell / parity / |N−Z|mod7           314/314 preserved
beta shell-parity + daughter-shell closure 801/801
fusion barrier on strong ladder             k = 3 for all seven fuels
resonance map (viable fuels)                5/5 pass; null p ≈ 0
barrier radius r0 ∈ [1.1, 1.4] fm           all on k = 3
```

Forced grammar classes carry the empirical claims; optional landmarks are scaffolding; Appendix A is design hypothesis only and carries no PASS gate.

### 1.2 External Data and Empirical Anchors

The trestleboard predicts placements on a grammar fixed by the kernel. Empirical anchors and holdout tables enter only as verification and as falsifiable targets. The catalogs used in this analysis are the following.

For nuclear structure (ENSDF / IAEA LiveChart), frozen snapshots reside in `data/catalogs/ensdf/` with provenance recorded in `data/catalogs/ensdf/SOURCE.txt`. Ground-state spins, parities, alpha and beta parents, Q-values, and half-lives are read from `iaea_livechart_ground_states.csv` (IAEA Nuclear Data Section LiveChart API, underlying evaluations ENSDF). The eV-band isomer census is the filtered table `ensdf_ev_band_levels.csv` (0 < E <= 200 eV from 214 actinide level files, Z = 88-98, A = 220-250), and first-excited actinide energies are in `ensdf_first_excited_actinides.csv`. The API endpoint is https://nds.iaea.org/relnsd/v1/data.

The primary optical-isomer gate for the Th-229m energy uses Zhang et al., Nature 633, 63-70 (2024), DOI 10.1038/s41586-024-07839-6, at 8.3557335(8) eV in CaF2 (local copy and provenance in `docs/references/SOURCE_Th229m.txt`), and the trestleboard supersedes the ENSDF Adopted listing for Th-229, which still records approximately 7.6 eV, with the Zhang value for tick checks.

The U-235m contrast enters because the eV-band census includes U-235m near 76 eV (Ponce et al., Phys. Rev. C 97, 054310, 2018; Shigekawa et al., arXiv:2603.01699, 2026), an isomer that lies outside the forced (6,2) window and is unclassified under the optical isomer census (local copies in `docs/references/SOURCE_U235m.txt`).

Reference astrophysical S-factors for holdout tests reside in `data/catalogs/fusion/` with provenance in `data/catalogs/fusion/SOURCE.txt`. D-T, D-D, D-3He, and T-T use the Bosch-Hale Padé fits (Nucl. Fusion 32, 611, 1992); p-11B uses the Tentori-Belloni piecewise fit (Nucl. Fusion 63, 086001, 2023); 3He-3He uses the Solar Fusion II quadratic (Adelberger et al., Rev. Mod. Phys. 83, 195, 2011); and p-6Li uses the Trojan Horse Method quadratic fit recorded in that catalog.

The electroweak anchor `v` and the W/Z masses that lock `Delta` follow the PDG 2024 review (Navas et al., Phys. Rev. D 110, 030001, 2024), and the deuteron binding energy used for the strong-scale check is 2.2240 MeV (PDG few-nucleon summary).

The terrestrial fusion viability boundary is the Rider cutoff `Z1 Z2 >= 7` (with `Z1 Z2 >= 8` absolute) together with the p-11B bremsstrahlung-to-fusion power ratio 1.19, both taken from Rider, LLNL High Energy Density Science seminar, 19 January 2023, "Is There a Better Route to Fusion?" (slides: https://heds-center.llnl.gov/sites/heds_center/files/2023-03/01-19-23_slides_-_rider_.pdf).

### 1.3 Dependency Map

The trestleboard inherits four layers of prior result. Each layer is a separate document and is cited by name where it enters.

1. **Aperture and closure constants.** `Analysis_CGM_Constants.md` fixes the observational aperture `m_a`, the BU dual-pole monodromy `δ_BU`, the closure ratio `ρ = δ_BU / m_a`, and the aperture gap `Δ = 1 − ρ`. Section 1.4 restates that chain only far enough that `Δ` is not a free nuclear parameter; the full derivation lives in the constants analysis.

2. **Finite kernel and electroweak spectrum.** `Analysis_Compact_Geometry.md` establishes the reachable manifold Omega (4096 states), the dual 64-state horizons, the seven-shell binomial chart, the enumerator weights C1 = 6, C2 = 15, C3 = 20, the reduced shell moment M_shell = 192, and the spectral expansion `L_i(Delta)` that recovers the four electroweak masses from the aperture gap Delta. The nuclear scales in this document are driven by that Delta and electroweak anchor `v`.

3. **Percolation and the square-root cluster theorem.** `Analysis_hQVM_Percolation.md` establishes the generator-restricted percolation hierarchy on the kernel: the exact coverage fraction `theta(p)`, the rank thresholds `p_c(r)`, the square-root identity `|Reach(A)| = (2^r(A))^2`, and the root-completion hierarchy from bare spanning through defect-spectrum completion to depth-four holonomy closure. The Square, the fusion resonance map, and beta branching depth read this hierarchy.

4. **Minimum-necessity codex.** `docs/CGM_Logic.md` records the chain of minimal necessities by which each CGM stage exists only to resolve the failure mode of the previous one. The nuclear minimum-isomeric-excitation gate is a member of that chain.

### 1.4 Origin of the Aperture Gap Δ

The spacing constant Δ of the ruler descends from two closed-form constants of the kernel (`Analysis_CGM_Constants.md`) through one measured geometric angle.

The observational aperture is the closed-form scale

```
m_a = 1 / (2 √(2π)) ≈ 0.199471
```

set by the requirement that coherent observation fit inside one π-radian phase horizon; equivalently `Q_G · m_a² = 1/2` with `Q_G = 4π`. It is the yardstick against which structural closure is measured.

The BU dual-pole monodromy `δ_BU ≈ 0.195342` rad is the geometric phase accumulated around a closed loop on the kernel, recovered from the SU(2) half-loop trace (`Analysis_Monodromy.md`).

The closure ratio is the fraction of `m_a` filled by that loop,

```
ρ = δ_BU / m_a ≈ 0.979300
```

so about 97.93% of `m_a` closes. The residual

```
Δ = 1 − ρ = 1 − δ_BU / m_a ≈ 0.020699
```

is the aperture gap — the ≈2.07% left open — and it is the ruler unit of Section 2.1. The label “aperture Δ” in this document always means this gap, distinct from `m_a` itself. The same Δ is locked independently by the W/Z mass ratio (Section 2.2) and by the electroweak spectral expansion (Section 4.4), agreeing to `8.34e-10`. Its powers `ρ^ℓ` are the dress factors of the grammar classes (Section 2.3), and `ρ^5` is the gravity-bulk factor behind the structural alpha preformation `P_α = 5 / 2^20` (Section 4.1).

## 2. The Three Instruments and the Delta-Ruler

### 2.1 The Delta-Ruler Coordinate

The aperture gap `Delta` derived in Section 1.4 fixes a logarithmic energy coordinate. For a state at energy `E`, the ruler reading is

```
n(E) = log2(v / E) / Delta
```

The anchor `v` (246.22 GeV) sits at `n = 0`. Higher energies fall to negative `n`. Lower energies rise to positive `n`. The inverse relation is `E(n) = v / 2^(n Delta)`. One tick on this ruler is the unit `Delta` in the exponent. Conversion constants measured on the kernel are `ticks/K = 270.26` for one aperture grade, `ticks/rho = 1.458` for one holonomy layer, and `ticks/oct = 48.31` for one octave (dyadic energy halving). The equatorial code factor is `2^(C3 Delta^2) = 1.00595755` with `C3 Delta = 0.4140`.

The same ruler carries electroweak masses at small `n`, nuclear optical structure at large `n`, and atomic levels at the largest `n`. With `Delta` locked by the W/Z mass ratio, the ruler is a single shared coordinate across all energy scales.

### 2.2 W/Z Lock of the Aperture

The aperture is recovered from the measured W and Z masses before any nuclear claim is made. It is fixed by the electroweak mass-coordinate expansion of `Analysis_Compact_Geometry.md`: each channel mass is a carrier-trace polynomial

```
L_i = a_i*Delta + b_i + c_i*Delta^2 + p_i*Delta^3/sqrt(5) + q_i*Delta^4 + r5_i*Delta^5
```

with coefficients `a_i, b_i, c_i, p_i, q_i, r5_i` drawn from the kernel grammar (`|H| = 64`, `C1 = 6, C2 = 15, C3 = 20`, `M_shell = 192`, the K4 stage flags, and the trace-free edge increments). At fifth order the four channels (Top, Higgs, Z, W) recover their PDG masses with tick error below `6.15e-9`, and the recovered `m_W / m_Z` locks `Delta` to absolute error `8.34e-10`. That locked value is the `Delta_ref` used as the ruler unit throughout this document.

With the PDG ratio `m_Z / m_W = 1.134470`,

```
log2(m_Z / m_W) = 0.182019026 = n_W - n_Z
```

The W/Z code gap is `C2 - C1 = 9`, promoted through the D4 kernel identity

```
log2(m_Z / m_W) = Δ · S_WZ(Δ)
S_WZ(Δ) = (C2 − C1) − (C3/2)·Δ + 2·Δ²/√5 − Δ³
```

with every coefficient a kernel constant. Newton inversion of that identity from the measured masses yields a second, independent determination of the same aperture:

```
Delta_WZ = 0.020699554747
Delta_ref = 0.020699553913
|Delta_WZ - Delta_ref| = 8.340e-10
```

relative error `4.029e-08`. The two determinations, one from the full four-channel `L_i` expansion and one from the W/Z gap alone, agree to the fourth-order D4 target of the compact-geometry analysis (absolute error `8.34e-10`). All nuclear and fusion placements in this document use that locked `Delta`.

### 2.3 Closure Grammar

A grammar class is a pair of non-negative integers `(k, ℓ)`. The first integer `k` counts aperture grades `Delta^k` below the anchor. The second integer `ℓ` counts the holonomy dress `rho^ℓ`. Dress meanings used in the trestleboard are

```
ℓ = 0   bare (no holonomy)
ℓ = 2   Z2 two-pass spinorial (F = W2 o W2')
ℓ = 4   EM-depth dress (dual commutator scale)
ℓ = 5   STF gravity bulk (five shells, rho^5)
```

The class predicts an energy through

```
E(k, ℓ) = v · Δ^k · ρ^ℓ · (1/√5 if STF) · (2^(C3 Δ²) if equatorial tick)
```

with the STF factor applied when the class carries trace-free quadrupole dress and the equatorial tick `2^(C3 Δ²)` applied when the class sits on the nuclear/spinorial equator (both flags are fixed grammar properties of the class). Two classes are forced by the grammar. The remaining classes are optional boundary and sector landmarks used by the Compass and Square.

```
status    (k,ℓ)   n          E              role
FORCED    (3,0)   810.78     2.1838 MeV     Strong bare (v·Δ³)
FORCED    (6,2)  1680.15     8.3563 eV      Nuclear spinorial (v·ρ²·Δ⁶/√5·2^(C3Δ²))
optional  (6,0)  1621.56    19.368 eV       Boundary bare (v·Δ⁶)
optional  (6,4)  1683.48     7.9665 eV      Boundary EM (v·ρ⁴·Δ⁶/√5)
optional  (6,5)  1684.93     7.8016 eV      Boundary gravity (v·ρ⁵·Δ⁶/√5)
optional  (4,0)  1081.04    45.203 keV      keV bare (v·Δ⁴)
optional  (4,2)  1140.04    19.387 keV      keV spinorial (v·ρ²·Δ⁴/√5)
optional  (3,2)   869.78   936.60 keV       Strong spinorial (v·ρ²·Δ³/√5)
optional  (3,5)   874.15   879.63 keV       Strong gravity (v·ρ⁵·Δ³/√5)
```

### 2.4 The Level

The Level places an absolute energy on the ruler. Given an energy `E`, it returns `n(E)` and the named grammar class whose predicted energy `E(k,ℓ)` minimizes the absolute tick residual `|n(E) - n(E(k,ℓ))|`. Against the forced classes,

```
Th-229m     n = 1680.15   class (6,2)   tick = +0.00501   |rel| = 7.189e-05
Deuteron BE n =  809.51   class (3,0)   tick = -1.27239   |rel| bare = 1.809e-02
```

The deuteron full formula `v·Δ³ + v·Δ⁴·(2/√5)` closes at `|rel| = 8.891e-05` against the measured 2.2240 MeV. The 1.272-tick Level residual is a code-atom resolution limit of the Greek-triangle lattice.

### 2.5 The Square and Sector Placement

The Square reads the percolation hierarchy and the named sector of the energy ladder. Given an energy `E`, it reports `n(E)` and the sector band cut by the ruler thresholds

```
n < 0        Planck / CS
0 ≤ n < 200  EW / UV
200 ≤ n < 900  Strong / IR
900 ≤ n < 1200 keV / Plasma
1200 ≤ n < 1900 Nuclear / Boundary
n ≥ 1900     Atomic / Deep IR
```

Measured placements on the shared ruler are

```
object          n         sector
EW v              0.00    EW/UV
Z                69.23    EW/UV
W                78.02    EW/UV
Deuteron        809.51    Strong/IR
10 keV plasma  1186.18    keV/Plasma
Th-229m        1680.15    Nuclear/Boundary
Cs hyperfine   2537.48    Atomic/Deep IR
```

In the fusion module the Square becomes the coverage dial. The inclusion probability is `p = E / V_b`, and the coverage `theta(p)` sets a resonance-independent lower bound on the fusion rate.

### 2.6 Optical Conjugacy and the Horizon Lemma

Optical conjugacy on the ruler pairs an ultraviolet energy with an infrared conjugate through the kernel constant

```
K = E_CS · v · (1/(4π²)) ,   E_conj = K / E ,   OPTICAL_DILUTION = 1/(4π²)
```

with Planck-scale reference `E_CS = 1.22 × 10^28` eV and electroweak anchor `v` in eV, so that `E · E_conj = K` and `n_UV + n_IR = −log2(K/v²)/Δ` is constant for every stage. For the electroweak anchors the residuals vanish to machine precision:

```
object   E·E_conj resid   n_sum resid
EW v     0.00e+00         0.000e+00
Z        0.00e+00         0.000e+00
W        0.00e+00         0.000e+00
```

Sample conjugates are `EW v → 3.090296e+26 eV` at `n = -2423.08` and `Z → 8.344256e+26 eV` at `n = -2492.31`.

The Horizon Lemma places dyadic and predecessor horizons on the `2^a · 3^b` table (`hQVM_Specs_Formalism.md`). Verified on-table values include

```
n      factorization   role
6      2^1 · 3^1       predecessor
9      2^0 · 3^2       2^a 3^b
24     2^3 · 3^1       predecessor P_k
64     2^6 · 3^0       dyadic (|H|)
192    2^6 · 3^1       predecessor
1536   2^9 · 3^1       predecessor
4096   2^12 · 3^0      dyadic (|Ω|)
```

Octave moves on the Compass are Horizon-Lemma dyadic jumps of `ticks/oct = 48.31`.

### 2.7 The Compass and Explicit Paths

The Compass traces the holonomy dress between two energies and returns the ordered list of moves that carry the anchor class of the start energy to the anchor class of the end energy. Anchor selection prefers the Level class when the absolute tick residual is below half an aperture grade (`|n − n_cls| < ½ · ticks/K`), and otherwise snaps to the nearest bare grade `k = round(n / ticks/K)`. Dress ranks are restricted to the ordered ladder `DRESS_ORDER = (0, 2, 4, 5)`.

Five move types appear, and the Compass applies them in a fixed routing order:

1. Undress along `DRESS_ORDER` until `ℓ = 0` (remove holonomy layers `rho^(-1)`).
2. Delta-step along bare grades until `k` matches the target (`Delta^(+-1)`).
3. Dress along `DRESS_ORDER` until `ℓ` matches the target (`rho^(+1)`).
4. Octave if the residual tick gap equals one dyadic jump within tolerance (`E → E/2` or `E → 2E`, Horizon-Lemma dyadic).
5. Code or offset: snap the residual to the nearest named code atom in `{C1, C2, C3, halves, differences, sums}`, or record the residual to the measured energy when no code atom fits.

Compass offsets are measurement residuals relative to the discrete code-atom set. They are not additive fit parameters and are never used to retune `Δ`, `v`, `ρ`, or any upstream constant.

Each dress move cites its operator. The two-pass spinorial closure is `F = W2 o W2'`, with `W2 = (0xaa, 0xab)` and `W2' = (0x2a, 0x2b)`. The word lifts the 24-bit carrier to the 32-bit spinorial frame and preserves the chirality shell. The involution `F^2 = id` holds on all 64 micro-refs.

Five measured Compass paths connect the nuclear and fusion scales.

The path from 10 keV to the deuteron starts at the `(4,2)` keV spinorial class and ends at the `(3,0)` strong bare class:

```
1. undress  ρ^-1   ℓ=2→0    Δticks=59.00   E 19.39 keV → 45.20 keV    F^-1 = F
2. Δ-step   Δ^-1   k=4→3    Δticks=270.26  E 45.20 keV → 2.184 MeV   aperture
3. offset   -1.272 ticks    Δticks=1.27    E 2.184 MeV → 2.224 MeV   bound-state residual
```

The path from the deuteron to Th-229m starts at `(3,0)` strong bare and ends at `(6,2)` nuclear spinorial with no residual remaining:

```
1. Δ-step   Δ^+1   k=3→4    Δticks=270.26  E 2.184 MeV → 45.20 keV
2. Δ-step   Δ^+1   k=4→5    Δticks=270.26  E 45.20 keV → 935.7 eV
3. Δ-step   Δ^+1   k=5→6    Δticks=270.26  E 935.7 eV → 19.37 eV
4. dress    ρ^+2   ℓ=0→2    Δticks=58.59   E 19.37 eV → 8.356 eV     F = W2∘W2′
```

The path from EW to the deuteron takes three aperture steps from the anchor to `(3,0)` and then the same 1.272-tick bound-state residual. The path from the barrier (about 0.44 MeV) to the deuteron starts at `(3,5)` strong gravity, applies three undress layers (`ℓ = 5 → 4 → 2 → 0`) to return to the bare strong scale, and then applies the deuteron residual. The longest measured path, from 10 keV to the barrier, undresses from `(4,2)` to bare keV, takes one aperture step to strong bare, dresses through `ℓ = 2, 4, 5`, applies one octave `E → E/2`, and finishes with a 0.660-tick residual to the barrier energy.

```
1. undress  ρ^-1   ℓ=2→0    Δticks=59.00
2. Δ-step   Δ^-1   k=4→3    Δticks=270.26
3. dress    ρ^+2   ℓ=0→2    Δticks=59.00
4. dress    ρ^+2   ℓ=2→4    Δticks=2.92
5. dress    ρ^+1   ℓ=4→5    Δticks=1.46
6. octave   E→E/2           Δticks=48.31
7. offset   -0.660 ticks    Δticks=0.66
```

Self-checks confirm that dress ranks stay in `{0, 2, 4, 5}`, that the Deuteron→Th path has no residual offset, that the 10 keV→Deuteron path carries the bound-state residual, and that the 10 keV→Barrier path uses an octave.

### 2.8 The Three Instruments as One Geometry

The Square, Compass, and Level are three readings of one object, the CGM energy grammar on the finite kernel. The Level gives the absolute coordinate. The Square gives the channel-accessibility structure at that coordinate. The Compass gives the move sequence that the kernel executes between coordinates. A fusion or decay prediction in this document states where on the ruler a transition lands and which percolation event opens there.

## 3. Kernel Percolation Foundation

### 3.1 The Reachable Manifold and the Square-Root Cluster Theorem

The kernel reachable set Omega contains 4096 states. Ancestry preservation forces Omega to factorize as a product of two conjugate faces `U` and `V`, each of size 64, so `|Omega| = |H|^2 = 4096` with `|H| = 64` the constitutional horizon. Under fiber-complete restriction, the reachable cluster from rest satisfies the square-root cluster theorem:

```
|Reach(A)| = root(A)^2 = (2^r(A))^2
```

Here `r(A)` is the GF(2) transport rank of the allowed byte set `A`, and `root(A) = 2^r(A)` is the surviving root dimension. In log2 coordinates this is the linear identity `log2|Reach(A)| = 2 r(A)`, slope 2 fixed by the product geometry. The identity holds at every transport rank under fiber-complete restriction and across the hQVM(`d_χ`) kernel family, where chirality dimension `d_χ` generalizes the physical instance `d_χ = 6` studied here.

The shell census verifies the factorization. With shell index `s` the Hamming weight of a byte, shell populations are `64 * C(6, s)` for `s = 0..6`:

```
pops = [64, 384, 960, 1280, 960, 384, 64]
```

summing to 4096 with mean shell `⟨S⟩ = 3.000`. Holographic balance `|H|^2 = |Omega|` holds exactly. Rank-by-rank reachability under fiber-complete restriction is

```
r   |Reach|   θ = |Reach|/|Ω|   note
0       2     0.000488          not fiber-complete (gauge doublet)
1       4     0.000977          |Reach| = (2^1)^2
2      16     0.003906          |Reach| = (2^2)^2
3      64     0.015625          |Reach| = (2^3)^2
4     256     0.062500          |Reach| = (2^4)^2
5    1024     0.250000          even-weight plateau (parity-obstructed)
6    4096     1.000000          full manifold
```

The rank-5 plateau follows directly from the parity functional `parity(q) = popcount(q) mod 2`, which is a homomorphism from the transport group to GF(2) whose kernel is the rank-5 subspace of even-weight transport values. A rank-5 set confined to that kernel drives only even-weight transport, so from the shell-6 anchor it reaches only even shells and the cluster closes at `32^2 = 1024`. Full reachability therefore requires odd-shell access beyond the mere rank condition `r = 6`, and the same parity cohomology class is what separates `E_span` from `E_full` in the coverage hierarchy.

### 3.2 The Exact Coverage Fraction theta(p)

The percolation parameter `p` is the independent probability that each of the 256 byte operators is included in the allowed set `A`. Restricting the byte alphabet degrades the transport rank on the chirality register, and the reachable cluster shrinks as the square of the surviving root. For the micro-reference payload protocol, the full coverage distribution admits a closed form

```
theta(p) = sum_k P(rank = k) * (2^k)^2 / 2^(2d)
```

The sum runs over the exact rank probability mass function. Physical coverage uses the conditional form for a nonempty generator set. Exact coverage at the hierarchy thresholds (unconditional audit form) is

```
p       θ(p)      event
0.0219  0.025530  E_span (weak transport, p/Δ ≈ 1.04)
0.0273  0.043530  E_full (strong, r = 6)
0.0402  0.112578  E_spectrum (defect completion)
0.0908  0.579325  P(rank = d) = 1/2 (micro-ref p_c)
0.3086  0.999831  E_word (holonomy transport)
```

Coverage saturates by `p ≈ 0.30`, which is a property of the kernel graph and is independent of the electrostatic barrier. At the exact micro-reference rank threshold, `theta(p_c_rank = 0.0908) = 0.5793`, a fuel-independent value defined at `p = p_c`.

The rank-ladder thresholds are exact. For integer rank `r`, `p_c(r)` is the inclusion probability at which the reachable set first reaches rank `r` (the probability that the rank equals `r` is one half). Distinct values after de-duplication are

```
p_c(rank ladder) = {0.293, 0.219, 0.146, 0.091}
```

with the exact micro-reference rank threshold `p_c(rank) = 0.0908`. The ratio `p_c(span) / Delta ≈ 1.056` matches the compact-geometry target near 1.04.

### 3.3 The Coverage Hierarchy

The coverage events are successively stronger conditions on the same rank-six root. Full transport rank `r(A) = 6` is necessary for full reachability but not sufficient for the finer structure the fusion module reads, because a root can be full-dimensional while remaining sparsely populated, anisotropically branched, or uncomposed into closure operators. Each additional event demands one more of these properties, so the events turn on at separable generator fractions as `p` increases.

```
p_c(span)     = 0.0219   at least one path from horizon 6 to horizon 0
p_c(full)     = 0.0273   full transport rank r = 6 with odd-shell access
p_c(spectrum) = 0.0402   all seven transport-defect weights present
p_c(rank)     = 0.0908   exact micro-ref rank threshold, P(rank = d) = 1/2
p_c(word)     = 0.3086   depth-four closure words available (holonomy transport)
```

Span requires only that the reachable set touch the opposite horizon, which can occur along a low-dimensional transport subspace, whereas full reachability adds the requirement of odd-shell transport. Spectrum completion requires the root to be uniformly covered by all seven defect weights, exceeding the spanning condition. The word event is the strongest condition, requiring the root to be composed into the depth-four closure operators that carry holonomy, and its availability follows `1 - (1 - p^4)^64` because each closure word needs four independent byte inclusions. The ordering `p_c(span) < p_c(full) < p_c(spectrum) < p_c(rank) < p_c(word)` therefore reflects increasing structural demand on a single fixed root (PASS).

### 3.4 Protocol Sensitivity

Under the default fusion model the inclusion probability that feeds `theta` is the Delta-dial value `p_Delta = E / V_b`. A second protocol, generator inclusion by q6 payload, produces a parallel coverage curve. At sample D-T energies the two protocols give

```
E_keV    τ       T        p_Δ     θ_micro     θ_q6
10.0     9.213   0.0001   0.0225  3.569e-02   1.239e-02
20.0     6.038   0.0024   0.0450  1.558e-01   7.795e-02
30.0     4.632   0.0097   0.0676  3.582e-01   2.307e-01
50.0     3.221   0.0399   0.1126  7.481e-01   6.359e-01
72.5     2.399   0.0908   0.1633  9.436e-01   9.049e-01
100.0    1.801   0.1651   0.2252  9.941e-01   9.888e-01
```

At `E_rank = 72.5` keV, where bare transmission reaches `p_c(rank)`, both protocols already sit above `theta = 0.90`. The fusion calculations use the micro-reference protocol with conditional coverage.

### 3.5 Discrete Grammar and Continuous Observables

The discrete kernel fixes the grammar through the classes `(k, ℓ)`, the thresholds `p_c(r)`, and the coverage `theta(p)`. Continuous observables enter as ruler readings. Once `Delta` and `v` are fixed, `n(E)` is a smooth function of `E`. Once the exact rank distribution is fixed, `theta(p)` is a smooth function of `p`. The kernel supplies the combinatorial structure and the energy coordinate supplies the continuity. A given nucleus or resonance is predicted to land at a specific tick on a ruler whose unit is fixed by the W/Z ratio. The falsifiable quantity is the tick residual.

## 4. Forced Minimum Nuclear Excitation

### 4.1 Minimum Isomeric Excitation

The formal gate of this section is the minimum half-life-tagged, optically addressable nuclear excitation on the forced grammar class `(k, ℓ) = (6, 2)`:

```
E_min = v * rho^2 * Delta^6 / sqrt(5) * 2^(C3 Delta^2)
```

What the ENSDF half-life-tagged eV-band census can falsify is the absence of any such isomer below the tolerance window around `E_min`. The stronger reading—that this residual is the absolute minimum nuclear excitation of the ground-state sector—is a physics interpretation of the Δ⁶ W-boundary, not the formal theorem checked by the census.

The derivation uses only upstream quantities. The electroweak sector closes at `Delta^5` in the compact-geometry five-order expansion, which is the BU-balanced ground relative to the representation boundary. The sixth grade is the W-channel representation boundary, the unique full-flag K4 endpoint. The first excitation beyond ground is that residual. Any lower structure would have been required to close at `Delta^5` (`docs/CGM_Logic.md`). The energy is read from the ruler with the spinorial dress `rho^2` (Gate F), the STF equipartition normalizer `1/sqrt(5)`, and the equatorial tick factor `2^(C3 Delta^2)` with `C3 = C(6, 3) = 20`. The inputs are fixed. `Delta` is locked by the W/Z ratio, and `(k, ℓ) = (6, 2)` is the forced nuclear class.

Numerically,

```
E_min = 8.3563 eV
```

### 4.2 Verification Against Th-229m

The lowest established optically addressable nuclear excited state is the Th-229m isomer at 8.3557335(8) eV in CaF2 (Zhang et al.). This is more than 10^5 times lower than typical nuclear excitations in the keV-MeV range, and it is the unique known nuclear excitation in the laser and VUV window.

The forced prediction and the measurement agree to a relative error of 7.19e-05, with a ruler residual of 0.005 ticks against the forced class `(6, 2)` (tick tolerance 0.1). The Level assigns Th-229m to `(6, 2)` (PASS). Free parameters in the prediction are zero, exponent scanning is disabled, and the grammar rank is 1. The forced-class energy window at tolerance `tol` ticks is

```
E_lo = E_min / 2^(tol · Δ) ,   E_hi = E_min · 2^(tol · Δ)
```

which at `tol = 0.1` is `[8.3444, 8.3683]` eV. The ENSDF eV-band isomer census in `data/catalogs/ensdf/ensdf_ev_band_levels.csv`, filtered to half-life-tagged entries via the IAEA LiveChart levels API (ENSDF underlying evaluations; `data/catalogs/ensdf/SOURCE.txt`, 214 actinide level files), contains no isomer below that window. The prediction is therefore the empirical minimum among known excitations. Census status on the filtered band is

```
status         label                   E_eV     near(k,ℓ)  tick
PASS           Th-229m (Zhang CaF2)    8.3557   (6,2)      +0.005
UNCLASSIFIED   U-235 ENSDF/iso        76.0000   (6,0)     -95.283
```

with `PASS = 1`, `UNCLASSIFIED = 1`, `SUPERSEDED = 0`. The null probability that a random energy in the eV band `[0.1, 200]` eV lands within `tol` ticks of `E_min` under log-uniform measure is

```
p_one = (2 · tol · Δ) / log2(band_max / band_lo)
```

and with census size `N = 1` in the forced window the probability of at least one hit is `1 − (1 − p_one)^N = 0.0004`. The ENSDF Adopted listing for Th-229 still records approximately 7.6 eV. That value is superseded here by Zhang for tick checks. Pu-239 first excitation lies above 1 keV (PASS).

### 4.3 The Strong Bare Scale

The strong bare scale is the other forced anchor that feeds the holographic product and the fusion module:

```
E_str = v * Delta^3
```

Substituting the locked parameters, `E_str = 2.1838 MeV`. The minimum excitation is related to it by the holographic product

```
E_min = E_str * (rho^2 * Delta^3 / sqrt(5)) * 2^(C3 Delta^2)
```

The strong bare scale is the anchor on which the Coulomb barrier and the tau-dial (formalized in Section 7.2) are built.

### 4.4 Spectral Bridge from the Wavefunction Kernel

The nuclear residual inherits the spectral structure that recovers the electroweak masses. The wavefunction is the state on the finite kernel manifold,

```
H = l²(Ω),   dim Ω = 4096,   |horizon| = 64
```

with shell-number operator `D_shell` whose reduced spectral moment is `M_shell = Tr(D_code) = 192`. The four electroweak channels are the K4 operator group `{id, W2, W2', F}`, where K4 denotes the Klein four-group reached by byte words on Omega. Cumulative fold-crossing depth fixes each channel's flag tuple `(base, rot, bal)`:

```
channel   operator   flags (base, rot, bal)
Top       id         (0, 0, 0)
Higgs     W2         (1, 0, 0)
Z         W2'        (1, 1, 0)
W         F          (1, 1, 1)
```

The W channel is the full-flag endpoint and carries the largest positive sixth-grade residual. Each mass is a spectral expansion in the aperture,

```
L_i(Δ) = a_i·Δ + b_i + c_i·Δ² + p_i·(Δ/√5)·Δ² + q_i·Δ⁴ + r5_i·Δ⁵
m_i = v / 2^{L_i(Δ)}
```

with coefficients fixed by kernel algebra. Recovered masses are

```
Top    172.7600 GeV
Higgs  125.1000 GeV
Z       91.1876 GeV
W       80.3790 GeV
```

The W/Z ratio from the spectral expansion is `0.88146853`, matching the PDG value, with absolute error on `Delta` recovered from that ratio equal to `2.203e-11`. Carrier traces `C(q)` enter the channel corrections and the nuclear matrix-element proxies.

`E_min` is the nuclear residual of this structure. The sixth grade is the W-channel representation boundary, so the nuclear scale inherits the kernel spectral anchor rather than introducing a new parameter. The 32-bit spinorial lift closes through K4 algebra, Gate F (a shell-preserving involution on the carrier, `F² = id`), and exact rank-lock (PMF match `0.999667`).

## 5. Deuteron Binding: Strong Bare Plus Tensor

### 5.1 The Two-Term Decomposition

The deuteron binding energy is reconstructed from the strong bare scale and a tensor correction:

```
E_d = v * Delta^3 + v * Delta^4 * (2 / sqrt(5))
```

The bare term `v * Delta^3` is `E_str`, approximately 2.1838 MeV. The tensor term uses the coefficient `2/sqrt(5)`, identical to the W/Z p-charge difference `(p_W - p_Z)/sqrt(5)` in the electroweak expansion. The tensor coefficient is the discrete trace-free quadrupole correction from the kernel grammar that fixes the electroweak masses.

Using the kernel constants,

```
E_bare   = 2.1838 MeV
E_tensor = 0.0404 MeV
E_total  = 2.2242 MeV
```

The measured deuteron binding is 2.2240 MeV (Particle Data Group few-nucleon summary, Navas et al., Phys. Rev. D 110, 030001, 2024), so the full formula closes to a relative error of 8.89e-05 (PASS, threshold 5e-04). The bare term alone has relative error 1.81e-02, which the tensor correction removes.

### 5.2 The Tensor Fraction and the Discrete Pion

The tensor term is 1.82 percent of the total binding. In the discrete CGM frame the tensor correction `v * Delta^4 * (2/sqrt(5))` arises from the Delta-shell-2 isospin-flip carrier move, the minimal spin-0, isospin-1 carrier excitation. That move is the discrete counterpart of the pion, the Goldstone boson of chiral symmetry breaking in the continuous theory. The Delta^4 gap, the step from the bare `Delta^3` to the tensor `Delta^4`, sets the chiral-symmetry-breaking scale in the discrete frame. The carrier quantity `C(2) = 7/3` is the trace of that move. This identification is a structural parallel between the discrete carrier move and the continuous Goldstone mode.

### 5.3 The Level Residual of the Deuteron

The Level assigns the deuteron to the strong bare class `(3, 0)` at a ruler residual of minus 1.272 ticks. This residual is the code-atom resolution limit of the Greek-triangle lattice, the discrete set of named code atoms `{C1, C2, C3, halves, differences, sums}` on the ruler, among which no 1.272-tick code atom exists. The binding formula closes to 1e-04, so the residual is a discretization artifact of the Level readout. The deuteron sits off the integer grid while Th-229m sits on it, and both placements are consistent with a discrete lattice of named code atoms.

## 6. Alpha and Beta Decay on the Kernel Carrier

### 6.1 The Three Element-Changing Paths

Nuclear structure contains three element-changing transitions. Alpha decay changes charge by minus two. Beta decay changes charge by plus one. Fusion combines two nuclei. All three execute on the kernel carrier through CGM stage payloads and K4 depth-four words, and this section treats alpha and beta while fusion follows in Section 7.

The chirality shell of a nucleus is `|N - Z| mod 7`, carried in the six-bit register chi6. Charge `Z`, neutron count `N`, spin `J`, and parity are mapped to a 32-bit atom in which the oriented chi6 gives the shell and the intron (an eight-bit companion word) gives `J` and parity. Orientation is built from the nuclear data before the weight is forced to the shell,

```
chi_rot = (Z mod 8) ⊕ (2J mod 8)     (Frame-0 bits 0–2)
chi_tr  = (N mod 8) ⊕ parity_bit       (Frame-1 bits 3–5)
chi6    = set_weight(chi_rot | chi_tr, |N − Z| mod 7)
```

with `u6 = Z mod 64` (parity bit optionally set) and `v6 = u6 ⊕ chi6` assembling the 24-bit Mac. The intron encodes `2J` in the six payload bits plus the family-high bit (capacity `2J ≤ 127`) and parity in the family-low bit. The intron is also the byte XOR the micro archetype `GENE_MIC = 0xAA`, with the eight bit positions grouped into the palindromic CGM stage pairs L0/LI/FG/BG → CS/UNA/ONA/BU (`hQVM_Specs_Formalism.md`). Alpha and beta operators act on this atom through byte words on the kernel graph.

### 6.2 Alpha Decay: Gate F, Shell-Preserving

Alpha emission ejects a `^4He` cluster (`N = Z`), so `N - Z` is conserved and the daughter stays on the same chirality shell as the parent. The operator is the Gate F word, four bytes whose shared payload micro is `(FG | BG) >> 1` (ONA/BU) taken once per K4 family index `0..3`, so that Gate F = W2 ∘ W2′. Gate F is the global-inversion element of the K4 holonomy algebra {id, S, C, F} (`hQVM_Features_Report.md`), an involution that preserves chirality and therefore preserves shell (verified 200 of 200 sampled states) while flipping the Z2 carrier sheet. Applying F twice returns the carrier to rest.

The bulk census over the IAEA LiveChart ground states (`data/catalogs/ensdf/iaea_livechart_ground_states.csv`, 2572 catalog entries with usable `J, P`) reports 314/314 for all three metrics on the 314 alpha parents with a catalogued daughter, confirming that Gate F preserves shell, shell-parity, and the daughter `|N−Z| mod 7` formula:

```
shell preserved (Gate F) .... 314/314
shell-parity conserved ...... 314/314
shell = |N-Z| mod 7 daughter  314/314
```

The alpha half-life is assembled from the tau-dial tunnel transmission `T = exp(−τ)` of the alpha on the daughter barrier, the carrier-trace hindrance `H_L = C(L) / C(0)`, the assault frequency `ν = 10^21 s^−1`, and the structural preformation `P_α = 5 / 2^20 = 4.7684e-06`,

```
T½ = ln 2 · P_α / (ν · T · H_L)
```

with `P_α` the five bulk STF shells over the operator-state phase space `|Ω| · |Alphabet| = 2^20`.

For Th-229 → Ra-225 the Gate F word is `(0x96, 0x97, 0x16, 0x17)`. The carrier maps `0xaaa555 → 0x555aaa` with shell 6 preserved. With `Q_α = 5.168 MeV`, `L = 2`, tunnel factor `T_tunnel = 4.135e-38`, and `H_L = C(2)/C(0) = 1/3`, the structural half-life is `2.397e11 s` against the measured `2.498e11 s` (ratio 0.96, residual −4.0 percent).

Across 310 alpha parents that carry both Q-value and half-life in the catalog, the structural estimator yields

```
ratio within [0.5, 2] ..... 113/310
ratio within [0.1, 10] .... 271/310
median ratio .............. 0.697
```

Sample closures include Nd-144 → Ce-140 (ratio 1.81) and Sm-146 → Nd-142 (ratio 1.10). The estimator is a structural lower bound from tunnel transmission and carrier hindrance. Absolute rates still require the ordinary nuclear preformation physics that sits outside the kernel rationals.

Spot checks on five alpha parents (Th, U, Ra, Po, Pu chains) all preserve shell and shell-parity under Gate F.

### 6.3 Beta Decay: UNA Byte, Shell-Parity

Beta-minus decay (`n -> p`) is a change of `N - Z` by minus two at fixed mass number `A`. The isospin axis is the UNA stage, the Frame-0 rotational degree of freedom. The transition is a byte (the kernel primitive) that flips both LI dipole pairs, advancing the carrier by two chirality shells with no Delta-ruler mass step. Three beta branches are the bytes that flip exactly the LI pair, namely `0x29` (forward LI only, `Delta J = +1`), `0x6b` (reverse LI only, `Delta J = -1`), and `0x69` (both LI, `Delta J = 0`, the isospin `|Delta shell| = 2` operator). The `Delta J` label is read from the intron LI bits of the byte (`intron = byte ⊕ 0xAA`): bit 1 (forward LI) contributes `+1` and bit 6 (reverse LI) contributes `−1`, so both set yields `0`.

The bulk census over 801 beta-minus parents from the same LiveChart ground-state catalog reports:

```
parent J round-trip .......... 801/801
shell-parity conserved ...... 801/801
decoded J-rule vs parent .... 801/801
catalog |dJ| <= 1 ........... 402/402
daughter-shell closure ...... 801/801
```

All 801 cases pass the round-trip, parity, and closure checks. The gated claims are distinct:

```
claim                                      domain                         result
shell-parity conserved                     β− parents with catalog daughter  801/801
daughter shell reachable (FWD/REFL/SRCH)   same                             801/801
parent J round-trip                        same                             801/801
daughter J for |ΔJ|≤1 (allowed stratum)    depth-1 subset                   402/402
daughter J vs catalog (all depths)         all 801                          402/801
```

The 402/801 overall J figure is the full catalog including higher-|ΔJ| compositions. It is not a failure rate of the allowed theory: on the depth-1 stratum `|ΔJ| ≤ 1` the agreement is 402/402. The daughter spin is operator-emitted (`J → J + dJ`) in the intron. The parent Mac (the 24-bit carrier shadow) alone does not select a unique branch, because the carrier is a two-to-one shadow and branch provenance lives in the intron or word composition (individuality-under-ancestry).

For tritium (`³H → ³He`) the intron is `L0|LI = 0xc3`, the byte is `intron ⊕ GENE_MIC = 0x69`, and the carrier maps `0xaaa555 → 0xaaa956` (shell 6 → 4, `|Δshell| = 2`). The half-life estimate uses the ordinary Fermi integral and an empirical superallowed anchor,

```
f(Z, Q) = ∫₁^{W₀} F(Z,W) · p · W · (W₀ − W)² dW ,   T½ = ln 2 · ft / (f · |M|²)
```

with `W₀ = (Q + m_e)/m_e`, nonrelativistic Fermi function `F = 2πη/(1 − e^{−2πη})`, `η = α Z W / p`, kernel `|M|^2 = 1.0`, and `ft = 10^{3.05}`. With `Q_β = 18.591` keV (LNHB) and `f(Z=2, Q) = 2.880e-06`, the estimated half-life is `2.700e8 s` against the measured `3.885e8 s` (ratio 0.69).

Decoded daughter-J agreement with the catalog is `402/801` overall. On the depth-1 subset `|dJ| ≤ 1` the agreement is `402/402`. Branch-shell match on any UNA branch is `87/801`, and joint shell-plus-J match is `44/801`. Those lower rates reflect the individuality-under-ancestry structure, because the Mac shadow does not uniquely select a branch until the intron is fixed.

### 6.4 The Daughter-Shell Closure and Branching Depth

The daughter shell is routed deterministically by three classes of move, which together close all 801 beta-minus parents:

```
FWD  (3 UNA branches) .......... 243/801
REFL (W2 byte 0x2A, w -> 6-w) .. 70/801
SRCH (derived byte, XOR-transport rule)  488/801
CLOSED total ..................... 801/801
```

The SRCH byte is derived by a kernel-grammar rule. It selects the smallest-|q| UNA-family byte whose transport mask `q` satisfies `|chi ^ q| = wp + |q| - 2|chi & q| = daughter_shell`. The closure therefore follows from that rule. The shell-transport identity verified on all 64 chi words is `|chi'| = |chi| + |q| - 2|chi & q|`, where the overlap `|chi & q|` makes the shell change state-dependent. The Hamming ladder carries no mod-7 cycle. Wrap cases use W2 reflection `w ↦ 6 − w`.

The REFL residue (70 parents) consists of cases whose daughter shell equals `6 − wp`. Representative parents include Be-12, B-14, C-16, N-18, Na-26, Mg-28, and Si-32. The SRCH residue (488 parents) requires a derived UNA-family byte outside the three LI branches. Representative parents include n-1 (`need_byte = 0x04`), H-3 (`0x03`), He-6 (`0x10`), Li-9 (`0x04`), and C-14 (`0x27`). Derived-byte `dJ` matches the catalog `dJ` on `145/801` cases overall and on `145/402` of the depth-1 subset.

Branching by catalog `|Delta J|` reads the root-coverage depth on the UNA sector. A single UNA half-cycle emits `Delta J` in `{-1, 0, +1}`, the depth-1 stratum. Larger catalog `|Delta J|` are compositions of half-cycles, so higher `|Delta J|` sits at deeper strata:

```
depth-1 (|dJ| <= 1) ......... 402/801
depth-2 (|dJ| <= 2) ......... 198/801
depth-3+ (|dJ| > 2) ........ 201/801
catalog dJ resolved at depth <= 2 ... 600/801
```

The catalog `|Delta J|` histogram splits as `0: 139, 1: 263, 2: 198, 3: 98, 4: 48, 5: 24, 6: 12, 7: 9, 8: 8` (plus two outliers at 23). The two largest low-`Delta J` bins (139 + 263 = 402) match the 2:1 prediction of the two-UNA-reference family at ratio 1.89:1. Beta branching and percolation depth are the root coverage that the Square reads in the fusion module.

Spot checks on four beta parents (H-3, C-14 family, Co-60 family, Sr/Y chain) all close the daughter shell under the FWD/REFL/SRCH routing.

### 6.5 Carrier Traces as Beta Matrix Elements

The kernel shell-transition matrix `M_q` (Krawtchouk shell-mixing on the six-bit chirality register) supplies exact rational carrier traces. For even `q` the diagonal trace is nonzero and equals `C(q) = Tr(M_q) = 7/(q+1)`. For odd `q` the diagonal vanishes and `C(q) = Tr(M_q²)` is the return-trace:

```
C(0) = 7
C(1) = 28/9
C(2) = 7/3
C(3) = 52/25
C(4) = 7/5
C(5) = 28/9
C(6) = 1
```

The Fermi proxy is `|M_F|^2 = C(0) = 7`. The Gamow-Teller proxy is `|M_GT|^2 = C(1) = 28/9`. The forbidden ladder is `C(3)/C(1) = 117/175` and `C(5)/C(1) = 1`. The alpha hindrance is `H_L = C(2)/C(0) = 1/3`, the discrete result for an `L = 2` transition. These are kernel rationals. The Fermi integral `f(Z, Q)` and the superallowed `ft` that convert them to a half-life are ordinary nuclear physics, as demonstrated in the tritium estimate above.

### 6.6 Magic Numbers as Structural Coincidence

The shell capacities `C(6, s) * 64` are `[64, 384, 960, 1280, 960, 384, 64]`, with cumulative sums `[64, 448, 1408, 2688, 3648, 4032, 4096]`. Comparing the nuclear magic numbers `[2, 8, 20, 28, 50, 82, 126]` to the structural set (capacities, cumulative sums, and code-gap arithmetic `{C1, C2, C3, WZ_gap = 9, horizon = 64, omega = 4096, M_shell = 192, predecessor horizons 24, 192, 1536}`) yields one coincidence: 20 lands on a code-gap arithmetic value, while the other six magic numbers fall outside any CGM structural limit. The comparison is recorded as a structural parallel.

### 6.7 Atomic Spectroscopy Parallel

Same-element spectral line pairs align to compact-geometry code levels on the ruler. Conversion of the full spectroscopic catalog into a self-check lies outside the trestleboard gate set. The measured alignments reported with the compact-geometry findings include

```
level   compact role                        best pair         err(ticks)
12      constitutional diameter             He 10917/12968    0.001
16      mask-code weight 2                  Cs 8047/10124     0.001
32      mask-code weight 4                  Na 2839/4494      0.000
48      mask-code weight 6 / depth-4        Na 3094/6161      0.008
64      mask-code weight 8 / |H|            Cs 5466/13693     0.006
80      mask-code weight 10                 Na 2905/9154      0.001
96      mask-code weight 12                 He 4713/18685     0.001
```

Antihydrogen mirror-tick sensitivity is recorded as `eta_X = log2(nu_H / nu_Hbar) / Delta` with sigma-tick scale `9.4e-3`. These alignments sit on the atomic/deep-IR sector of the Square and are structural parallels to the nuclear placements.

## 7. The Fusion Module

### 7.1 The Coulomb Barrier as a Placed Grammar Coordinate

Fusion of light nuclei is exothermic on the rising flank of the nuclear binding-energy curve, with ^4He among the most tightly bound products. Before the short-range nuclear attraction can act, the nuclei must approach through the long-range Coulomb repulsion. The classical barrier height for that approach is the Coulomb barrier

```
V_b = 1.44 · Z1 · Z2 / r_fm ,   r_fm = 1.2 · (A1^(1/3) + A2^(1/3))
```

with energies in MeV and radii in fm. Quantum tunneling allows fusion at kinetic energies below `V_b` (Gamow, 1928, applied tunneling first to alpha decay and then to fusion as the inverse process, and Atkinson and Houtermans, 1929, used that penetration to estimate stellar fusion rates). The Gamow energy of the reduced-mass two-body problem is

```
μ = A1 · A2 / (A1 + A2) · m_N ,   E_G = 2 μ (π α Z1 Z2)²
```

with `m_N = 931.494 MeV` and `α = 1/137.036`, and the Gamow penetration factor is `P_Gamow(E) = exp(−√(E_G / E))`. The trestleboard takes `V_b` from the formula above (default `r0 = 1.2` fm) and reads it as a placed coordinate on the strong-family ladder. Its ruler tick `n(V_b)` lands on a class with `k = 3` (the strong bare scale `v * Delta^3`), and the dress rank `ℓ` varies with `Z1 Z2` so that heavier charge products sit lower on the strong ladder.

The barrier-placement gate tests two claims:

1. The barrier tick `n(V_b)` lands on a strong-family class (`k = 3`).
2. On a fine energy grid below the barrier, the truncated-barrier transmission

```
τ_b(E) = 2π · √(E_G / E) · (1 − √(E / V_b)) ,   s(E) = (1/E) · θ(E/V_b) · exp(−τ_b)
```

attains its maximum at an energy whose tick coincides with `n(V_b)` within tolerance (7 ticks, about 10 percent in energy). Because `τ_b → 0` as `E → V_b`, the peak sits near the barrier by construction of the truncated form. The nontrivial grammar claim is the `k = 3` placement of `V_b` itself.

Per-fuel barrier placement is

```
fuel      Z1Z2   V_b(MeV)  n(V_b)   n_peak  Δn     class (k,ℓ)
D-T          1     0.444    921.79   921.86  +0.07  (3,5) Strong gravity
D-D          1     0.476    916.92   916.99  +0.07  (3,5) Strong gravity
D-3He        2     0.888    873.48   873.55  +0.07  (3,5) Strong gravity
T-T          1     0.416    926.34   926.41  +0.07  (3,5) Strong gravity
3He-3He      4     1.664    829.72   829.79  +0.07  (3,0) Strong bare
p-6Li        3     1.278    848.13   848.19  +0.07  (3,2) Strong spinorial
p-B11        5     1.861    821.92   821.99  +0.07  (3,0) Strong bare
```

Every barrier lands on a `k = 3` class (PASS). The truncated-barrier peak coincides with the barrier tick within 0.1 ticks for all seven fuels (PASS). True resonances appear as measured offsets below the barrier: D-T at 50 keV sits `+152.22` ticks above the barrier tick on the IR side of the ruler, and p-B11 at 600 keV sits `+78.89` ticks above its barrier tick.

### 7.1.1 Barrier Radius Sensitivity

Because `V_b ∝ 1/r0`, a change of nuclear-radius prefactor shifts every barrier tick by the same amount, `Δn = −log2(r0 / 1.2) / Δ`, independent of fuel. Sweeping `r0 ∈ {1.1, 1.2, 1.3, 1.4}` fm on the seven holdout fuels (`hqvm_cgm_trestleboard_4.py`, section I) yields

```
r0 (fm)   Δn vs 1.2 (ticks)   all on k=3
1.1       −6.06               yes
1.2        0.00               yes
1.3       +5.58               yes
1.4      +10.74               yes
```

Strong-ladder placement (`k = 3`) survives the full sweep (PASS). Within `r0 ∈ [1.1, 1.3]` the tick shift stays inside the 7-tick peak-coincidence tolerance. At `r0 = 1.4` the shift exceeds that tolerance while the ladder class remains `k = 3`. Barrier-class claims are therefore robust to the usual nuclear-radius band; peak-coincidence at the default `r0 = 1.2` is the sharper, radius-sensitive statement.

### 7.2 The Two Dials and the Cross-Section Formula

Two inclusion dials feed the coverage `theta` in the fusion cross-section. The tau-dial sets `p = p_c · T` with `T = exp(−τ)`, the Beer-Lambert form of Gamow barrier transmission,

```
τ = √(E_G / E) − √(E_G / V_b)   (E < V_b),   τ = 0 otherwise
```

so that inverting `T = p_target` below the barrier recovers the landmark energy

```
√(E_G / E) = √(E_G / V_b) − ln(p_target) ,   E_τ = E_G / [√(E_G/V_b) − ln(p_c)]²
```

At and above the barrier, `τ = 0` so `p_τ = p_c`. The Delta-dial sets `p_Δ = E / V_b`, with twin landmark `E_Δ = p_c · V_b`. The astrophysical S-factor convention factors the Coulomb penetration from the nuclear matrix element, writing `σ(E) = (S(E)/E) P_Gamow(E)`. The default CGM model (Model 2, dial = delta) multiplies that baseline by the kernel coverage,

```
sigma ~ (S / E) * P_Gamow * theta(p_Delta)
```

with `p_Delta = E / V_b`. The effective transport rank read from coverage is the exact inverse of the square-root cluster identity for `r ≥ 1`,

```
θ(r) = (2^r / |H|)^2 ,   r_eff = d_χ + ½ log2(θ)   (clipped to [0, d_χ], with θ ≤ 2/|Ω| mapping to r = 0)
```

The Gamow factor is kept separately from `theta`, so barrier penetration is counted once. Model 1 instead takes `theta` as the tunneling factor and drops the separate Gamow factor. The native model drops the Gamow factor entirely and tests whether the exact coverage `theta(p)` alone reproduces the measured cross-section. Under each of these choices, `theta(p)` remains the exact kernel coverage and supplies a lower bound on the fusion rate. Reference S-factors are the Bosch-Hale Padé fits for D-T, D-D, D-3He, and T-T, Tentori-Belloni for p-11B, Solar Fusion II for 3He-3He, and the Trojan Horse Method fit for p-6Li.

In a fusion calculation the trestleboard is used as a plug-in factor, not as a replacement for R-matrix. Given `(Z1, Z2, A1, A2)` and a dial choice, it returns `θ(E/V_b)`, the rank landmarks `E_r = p_c(r)·V_b`, the susceptibility width proxy `Γ_struct(r)`, and the cutoff discriminant `R`. The baseline remains `σ_base = (S/E) P_Gamow`, the CGM-modulated form is `σ = σ_base · θ(E/V_b)`, and any Breit–Wigner or R-matrix resonance term is an optional overlay on that floor.

The dual dial covers all four fuels in the test set. `E_tau` is the energy where `T = exp(-tau) = p_c`. `E_Delta` is the energy where `p_Delta = p_c`.

```
fuel     E_τ(keV)  hitτ   E_Δ(keV)  hitΔ   Res(keV)  TOL   best
D-T         72.5     Y       40.3     Y      64.0    25.0  both
D-D         66.6     Y       43.2     N     100.0    40.0  τ
D-3He      212.8     Y       80.6     N     250.0   100.0  τ
p-B11      650.9     N      169.0     Y     148.0    60.0  Δ
```

Dual-dial coverage is 4/4. Light fuels sit on the tau-band. p-B11 sits on the Delta-dial. The ordering `E_τ(D-T) < E_τ(D-3He) < E_τ(p-B11)` holds. For D-T, `V_b ≈ 0.444 MeV`, `E_G ≈ 1.175 MeV`, and `E_rank ≈ 72.50 keV` lies inside the 5 to 500 keV band.

### 7.3 D-T Cross-Section Grid

Among candidate terrestrial fuels, D-T has the largest low-temperature reactivity because the reaction `D + T → ^4He (3.5 MeV) + n (14.1 MeV)` liberates 17.6 MeV, and a low-energy resonance (identified in the wartime cross-section program; see Chadwick and Reed, 2024) raises its cross-section by about two orders of magnitude relative to naive D-D scaling. That resonance is why D-T is the power-fuel reference in the resonance map and why a pure geometric baseline cannot absorb the 50 keV peak.

On the D-T energy grid the Model-2 cross-section, normalized to `σ₀ = 1` at the 10 keV reference, is

```
E_cm   n       p_inc  r_eff  θ         P_Gamow   σG/σ0     σCGM/σ0
1.0    1346.7  0.0023  2.34  6.29e-03  1.30e-15  6.62e-10  1.17e-10
5.0    1234.5  0.0113  2.83  1.23e-02  2.20e-07  2.24e-02  7.72e-03
10.0   1186.2  0.0225  3.60  3.57e-02  1.96e-05  1.00e+00  1.00e+00
20.0   1137.9  0.0450  4.66  1.56e-01  4.69e-04  1.20e+01  5.22e+01
50.0   1074.0  0.1126  5.79  7.48e-01  7.85e-03  8.00e+01  1.68e+03
100.0  1025.7  0.2252  6.00  9.94e-01  3.25e-02  1.66e+02  4.61e+03
300.0   949.1  0.6755  6.00  1.00e+00  1.38e-01  2.35e+02  6.58e+03
500.0   913.5  1.0000  6.00  1.00e+00  2.16e-01  2.20e+02  6.17e+03
```

Self-checks confirm `θ(10) < 0.5`, monotone growth `θ(10) < θ(30) < θ(100)`, and `θ(10)/θ(100) < 0.5`. The pure-Gamow peak and the CGM-model peak both sit at 300.0 keV on this grid (`σG/σ0 = 234.9`, `σCGM/σ0 = 6582`). The model maximum remains unshifted by `theta` on D-T because coverage has already saturated near the Gamow peak. The analytic Gamow-only maximum `E_G/4 = 293.7` keV agrees with the grid peak to within one bin.

For p-B11 the barrier is higher (`V_b ≈ 1.861 MeV`, `E_G ≈ 22.438 MeV`). Coverage rises more slowly. At 100 keV, `θ ≈ 0.228` and `σCGM/σ0 = 1` by normalization. At 600 keV, `θ ≈ 1` and `σCGM/σ0 ≈ 5.18e+03`. The CGM enhancement relative to pure Gamow is therefore concentrated at intermediate energies where coverage is turning on.

### 7.4 The Resonance Map on the Percolation Hierarchy

The resonance map is the falsifiable fusion claim. Measured fusion resonances are placed on the percolation hierarchy. Declared landmarks per fuel are the union of six structural events, the Gamow-peak energy, and the rank-ladder twins on both dials:

```
E_span, E_full, E_spec, E_τ, E_word   from p_c(event) via τ-inversion
E_Δ                                   = p_c(rank) · V_b
E_Gamow                               = E_G / 4
E_τ_r{r}, E_Δ_r{r}                    rank-ladder twins for each predeclared p_c(r)
```

Seventeen landmarks are declared per fuel. Resonance energies `E_res` are center-of-mass peak positions taken from the literature sources cited with each fuel (Bosch–Hale / Tentori / Solar Fusion II / THM catalogs for the holdout set; for the map suite, D-T 50 keV, p-B11 600 keV, 10B-p 10 keV sub-threshold 11C, 12C-p 461 keV, 15N-p 325 keV with literature band 312–338 keV, 7Li-p 330 keV, 6Li-p 440 keV). The map tolerance converts the literature energy window (keV) into ticks. Landmark energies on the four-fuel stress suite are

```
fuel     Res    E_span  E_full  E_spec   E_τ    E_word   E_Δ
D-T       64.0    39.6    43.0    50.1    72.5   149.6    40.3
D-D      100.0    35.4    38.6    45.3    66.6   143.8    43.2
D-3He    250.0   125.3   134.9   154.6   212.8   389.0    80.6
p-B11    148.0   421.6   448.5   501.9   650.9  1038.6   169.0
```

Each resonance tick `n(E_res)` is compared to the nearest landmark tick. PASS requires the offset to lie within the literature tolerance and above the weakest rank threshold. Roles and placements for seven fuels with literature resonances are (CNO entries follow the solar CNO cycle rates of Adelberger et al., Solar Fusion II):

```
fuel     role        Z1Z2  E_res  landmark      off(ticks)  tol   status
D-T      power          1   50.0  E_spectrum      +0.20     6.64  PASS
p-B11    aneutronic     5  600.0  E_tau           +5.67     6.64  PASS
10B-p    aneutronic     5   10.0  E_tau_r0      +110.47    48.31  FAIL SUB
12C-p    CNO            6  461.0  E_delta_r1      +1.00     6.63  PASS
15N-p    CNO            7  325.0  E_delta_r2      +8.77     6.74  FAIL CUT
7Li-p    aneutronic     3  330.0  E_span          +2.56     7.97  PASS
6Li-p    aneutronic     3  440.0  E_tau           +3.75     6.06  PASS
```

The null model gives a single-hit probability under a log-uniform window of width `2 · tol` ticks over the sub-barrier band,

```
p_single = (2 · tol · Δ) / log2(V_b / E_band_lo)
```

equal to `0.0334` on the suite (expected hits 0.23). Five of seven pass (`P(K >= 5) = 0.0000`, Bonferroni `p` by 17 events also 0.0000). Among fuels with `Z1 Z2 < 7` and no sub-threshold flag, placement is 5/5. The two non-passing fuels remain in the report to illustrate boundary conditions. 10B-p at 10 keV is a center-of-mass sub-threshold 11C resonance (`p_Delta ≈ 0.007`), and 15N-p at 325 keV is the `Z1 Z2 = 7` Rider-cutoff fuel whose resonance sits in the integer-rank gap between `r5` (0.146) and `r4` (0.219). The grammar gap coincides with Rider's terrestrial viability boundary. Code `E_G` matches the literature Gamow table to 0.4–0.8 percent.

### 7.5 Reactivity and the Enhancement Growth

For a thermal plasma the fusion rate density is `f = n_1 n_2 ⟨σv⟩` (with `n^2/2` for like-particle fuels such as D-D), where the reactivity `⟨σv⟩` is the velocity-averaged product of cross-section and relative speed. Meaningful `⟨σv⟩` requires temperatures of order 10–100 keV, well above ionization, so the reactants are a plasma, and the Lawson criterion then states the `nTτ` triple product needed for net power. The CGM scan approximates the relative reactivity by a trapezoid integral of the Model-2 integrand against a Maxwellian weight,

```
I(T) = ∫ P_Gamow(E) · θ(E/V_b) · exp(−E/T) dE
```

(and the same integral without `θ` for the Gamow-only baseline), across a temperature grid. Relative reactivities for D-T, normalized at 10 keV, are

```
T_keV   ⟨σv⟩G / ⟨σv⟩G0   ⟨σv⟩CGM / ⟨σv⟩CGM0   R = CGM/G
1.0     3.57e-06          1.77e-07              0.0248
5.0     5.20e-02          2.53e-02              0.243
10.0    1.00e+00          1.00e+00              0.499
20.0    1.23e+01          1.92e+01              0.780
50.0    1.88e+02          3.57e+02              0.948
100.0   1.04e+03          2.05e+03              0.985
300.0   8.78e+03          1.76e+04              0.997
500.0   1.69e+04          3.38e+04              0.999
```

The absolute reactivity ratio `R(T) = ⟨σv⟩_CGM / ⟨σv⟩_G` rises monotonically toward 1 as `T → ∞`, because `theta` is a coverage-weighted average. Absolute peak locations sit at the grid edge (500 keV) for both Gamow and CGM, as expected for a monotone integrand. The falsifiable interior signal is the temperature of maximum `dR/dlnT`, the point where `theta` most rapidly reshapes the Maxwellian window. For D-T that temperature is 20 keV, inside the plasma band, and the enhancement growth is interior (PASS). The structural reading is that `theta` raises the low-energy tail of the fusion rate by a resonance-independent amount, on top of which a localized Breit-Wigner resonance (such as the D-T 50 keV peak) overlays as a compound-nucleus amplitude.

## 8. Quantitative Consequences for Fusion

### 8.1 Resonance Widths from Percolation Susceptibility

The susceptibility is the derivative of exact micro-reference coverage with respect to inclusion probability, evaluated by central difference on the closed form (`h = 10^{−6}`, no Monte Carlo),

```
χ(p) = [θ(p+h) − θ(p−h)] / (2h)
```

At each rank-ladder inclusion `p_c(r)` that susceptibility sets the structural width

```
Gamma_struct(r) = chi_ref / chi(p_c(r))
```

with reference `chi_ref = max_r χ(p_c(r)) = χ(p_c(4)) = 8.848690`. A sharp transition (large `chi`) is a narrow structural resonance. Per-rank scaling on the D-T barrier is

```
r   p_c(r)    E = p_c·V_b (keV)   chi(p_c)   Gamma_struct
1   0.292893  130.0702            0.015388   575.041
2   0.218779   97.1570            0.295041    29.991
3   0.145759   64.7298            3.035960     2.915
4   0.090795   40.3211            8.848690     1.000
```

The scaling is monotone inverse across rungs (PASS). Rank-1 landmarks are broad. Higher-rank landmarks are narrow. The same `Gamma_struct` values apply across fuels. Only the landmark energy `E = p_c(r) · V_b` changes:

```
fuel      E_r1 (keV)  E_r2     E_r3     E_r4
D-T         130.07     97.16    64.73    40.32
D-D         139.48    104.19    69.41    43.24
D-3He       260.14    194.31   129.46    80.64
T-T         121.85     91.02    60.64    37.77
3He-3He     487.39    364.06   242.55   151.09
p-6Li       374.29    279.58   186.27   116.03
p-B11       545.09    407.16   271.27   168.98
```

For a beam-target or colliding-beam system with a controlled energy profile, `chi` sets the required energy spread `delta E / E` to lock a given rank closure. A Maxwellian distribution clips the narrow landmarks weakly. A monoenergetic beam can force them.

### 8.2 The CGM Surrogate on Predictive Holdout

The CGM cross-section surrogate is tested on the reference S-factor tables in `data/catalogs/fusion/` with a single scale degree of freedom `C`:

```
sigma_CGM(E) = C * P_Gamow(E) * theta(E / V_b) / E
```

Calibration uses even CSV indices and fits `C` by least squares against the reference cross-section `σ_ref = S_ref · P_Gamow / E`, namely `C = Σ σ_ref · σ_raw / Σ σ_raw²` with `σ_raw` the unscaled CGM shape. Holdout uses odd indices. Per-fuel holdout metrics are

```
fuel      n_cal  n_hold  C            RMSE(log10)  Pearson r
D-T          48     48   6.981e+06    0.5777       -0.1175
D-D          48     48   1.295e+07    0.2308        0.9384
D-3He        48     48   1.099e+05    0.4526        0.9985
T-T          48     48   8.610e+04    0.2283        0.9888
3He-3He      48     48   6.392e+03    0.8175        0.9955
p-6Li        48     48   2.958e+03    0.7425        0.9904
p-B11        61     60   3.205e+05    0.7906        0.9710
```

Pooled RMSE in log10 is 0.5962. Pooled mean Pearson `r` is 0.8236. Non-resonant fuels give `r` above 0.93. The D-T holdout Pearson `r` is minus 0.12. That result is consistent with the role of `theta(p)` as the direct, non-resonant topological baseline. The Breit-Wigner peak near 50 keV is a localized compound-nucleus overlay that the baseline does not carry. The holdout therefore measures baseline shape. The decomposition used here is a geometric floor (`theta(p)` from the barrier tick, a resonance-independent lower bound on `⟨σv⟩`) together with a resonance boost at specific energies. If the geometric floor fails the Lawson criterion (Lawson, 1957) for a fuel, resonance structure does not restore viability.

### 8.3 The Rider Cutoff as an Internal Discriminant

The Rider cutoff (Rider, LLNL High Energy Density Science seminar, 19 January 2023) marks `Z1 Z2 >= 7` as the Coulomb barrier too high for terrestrial fusion, with `Z1 Z2 >= 8` absolute, and notes that p-11B already has a bremsstrahlung-to-fusion power ratio of 1.19 in equilibrium plasma. That fuel is the canonical aneutronic candidate (`p + ^11B → 3 ^4He + 8.7 MeV`), but advanced fuels pay a radiation-loss penalty that grows with `Z` of the non-hydrogenic reactant. The CGM supplies an internal discriminant from barrier placement alone:

```
R(Z1 Z2) = n(V_b(Z1 Z2)) - n_cut
```

with `n_cut = n(V_b)` at `Z1 Z2 = 7` (15N-p) equal to 828.86 ticks. Per-fuel values are

```
fuel     Z1Z2   n(V_b)   R        below cutoff
D-T         1   921.79  +92.93    True
D-D         1   916.92  +88.06    True
D-3He       2   873.48  +44.62    True
p-6Li       3   848.13  +19.26    True
p-B11       5   821.92   -6.94    True (anomaly)
10B-p       5   843.26  +14.39    True
12C-p       6   834.61   +5.74    True
15N-p       7   828.86   +0.00    False
```

`R` is positive on the accessible side and negative above the cutoff. Strict sign separation fails only on p-B11, where `R ≈ −6.94` despite `Z1 Z2 = 5`, because the barrier tick sits below the `Z1 Z2 = 7` reference. This discriminant is a barrier-tick correlator relative to that cutoff anchor. It does not compute bremsstrahlung or other radiation-loss physics, and it is not a derivation of Rider's `P_brem/P_fus` ratio. The geometric counterpart of Rider's radiation-loss marginality for p-B11 is that the barrier placement itself sits on the wrong side of the cutoff tick. The classification used here is that `R <= 0` reports a geometry-restricted channel relative to the cutoff reference.

### 8.4 Sparse-Data Prediction Targets

For each holdout fuel, the untested band is `[first CSV energy, p_c(1) * V_b]`. Rank-1 landmarks and untested widths are

```
fuel      first_CSV (keV)  landmark (keV)  untested width (keV)
D-T              10.0           130.07              120.1
D-D              10.0           139.48              129.5
D-3He            10.0           260.14              250.1
T-T              10.0           121.85              111.8
3He-3He          10.0           487.39              477.4
p-6Li            10.0           374.29              364.3
p-B11            10.0           545.09              535.1
```

These are targeted beam energies for structural rank transitions where standard S-factor tables are silent below the landmark.

## Appendix A. Design Hypotheses for Three Fusion Domains

This appendix is speculative. None of its claims carries a trestleboard PASS gate. It records which coordinate each experimental route acts on once the core modules have fixed `p = E/V_b`, `θ(p)`, and `p_c(r)`.

```
domain              coordinate changed              CGM lever
compact hot         raise E (hence p)               place power on broad Γ_struct bands
muon-catalyzed      ρ-dress (orbit shrink)          Gate F / carrier-trace scaling
lattice / LENR      supply missing generators       local rank-6 completion at defects
```

Kinetic heating raises `p` until thresholds are crossed stochastically. The other two routes change geometry while leaving the kinetic coordinate fixed.

### A.1 Compact Hot Fusion

The standard terrestrial routes, namely magnetic confinement (tokamak, stellarator) and inertial confinement (laser or beam drivers), raise kinetic energy until the plasma triple product approaches the Lawson threshold. Under the CGM reading, heating raises `p` until the plasma distribution crosses rank thresholds. Shaping RF or beam energy deposition onto the broad `Gamma_struct` landmarks (rank-1 and rank-2) instead of uniform heating targets the broad `chi` bands and can reduce auxiliary power for a given `theta`. The route remains kinetic, and the change is energy placement on the ruler.

### A.2 Muon-Catalyzed Fusion

Muon-catalyzed fusion proceeds at ordinary temperatures because the muon mass shrinks the Bohr orbit by about 207 times, so nuclei sit closer without MeV thermal `E`. Net energy production has remained unsuccessful: muon production is costly, the muon lifetime is 2.2 μs, and sticking of the muon to the daughter alpha terminates the catalysis chain (Jones, 1986). Under the CGM reading, the muon is a forced `rho`-dress, a spinorial mass scaling that shifts the bipartite carrier toward complement-horizon closure without adding kinetic `E`. Sticking is the muon trapped in the daughter Z2 holonomy after the Gate F word closes. An open question is whether the physical muon is required or only its operator signature. If an electromagnetic drive can match the Delta-step or carrier-trace scaling of the discrete pion parallel, including the forbidden modes via `C(3) = 52/25`, the geometric close would occur without particle production or sticking. That possibility is a hypothesis and has no trestleboard PASS gate yet.

### A.3 The Lattice Path

A standard objection to lattice fusion is that lattice thermal energy cannot overcome `V_b`. The same objection appears in beam-target fusion: fusion cross-sections are many orders of magnitude below Coulomb scattering, so most ions radiate or ionize before fusing. Under the CGM reading, `E` is one coordinate of `p = E / V_b`, and rank-6 generator completion on `GF(2)^{d_χ}` with `d_χ = 6` is the fusion transition condition. A metal lattice (Pd, Ni, and others) can supply fixed, pre-loaded chirality-transport bytes at defect sites (dislocations, grain boundaries, vacancies), so the local generator set reaches rank 6 without raising the kinetic energy through the barrier. The implied consequences are rank completion as a slow holonomic process (`theta -> 1` locally), replication failures from uncontrolled metallurgy when one defect geometry completes rank 6 and a nearby sample remains at rank 5, and a spectrum that may select aneutronic or low-energy alpha branches fixed by lattice holonomy. These implications require pathway enumeration against EXFOR and the contested LENR literature. They are not a claim that room-temperature thermal kinetics alone produce fusion.

## References

1. B. Korompilias, *Common Governance Model: Mathematical Physics Framework*, Zenodo (2025), https://doi.org/10.5281/zenodo.17521384.
2. Particle Data Group, S. Navas et al., *Review of Particle Physics*, Phys. Rev. D 110, 030001 (2024), https://doi.org/10.1103/PhysRevD.110.030001.
3. C. Zhang et al., *Frequency ratio of the 229mTh nuclear isomeric transition and the 87Sr atomic clock*, Nature 633, 63-70 (2024), https://doi.org/10.1038/s41586-024-07839-6; arXiv:2406.18719.
4. F. Ponce, E. Swanberg, J. Burke, R. Henderson, and S. Friedrich, *Accurate measurement of the first excited nuclear state in 235U*, Phys. Rev. C 97, 054310 (2018), https://doi.org/10.1103/PhysRevC.97.054310.
5. Y. Shigekawa et al., *Chemical effects on nuclear decay of 235U isomer in the uranyl form*, arXiv:2603.01699 (2026).
6. IAEA Nuclear Data Section, LiveChart of Nuclides Data Download API, https://nds.iaea.org/relnsd/v1/data; underlying evaluations: Evaluated Nuclear Structure Data File (ENSDF).
7. Laboratory National Henri Becquerel (LNHB), Recommended data for 3H beta decay.
8. H.-S. Bosch and G.M. Hale, *Improved formulas for fusion cross-sections and thermal reactivities*, Nucl. Fusion 32, 611 (1992), https://doi.org/10.1088/0029-5515/32/4/I07.
9. A. Tentori and F. Belloni, *Revisiting p-11B fusion cross section and reactivity, and their analytic approximations*, Nucl. Fusion 63, 086001 (2023), https://doi.org/10.1088/1741-4326/acda4b.
10. E.G. Adelberger et al., *Solar fusion cross sections. II. The pp chain and CNO cycles*, Rev. Mod. Phys. 83, 195 (2011), https://doi.org/10.1103/RevModPhys.83.195.
11. A. Tumino, C. Spitaleri, et al., *Indirect study of the astrophysically relevant 6Li(p, alpha)3He reaction by means of the Trojan Horse Method*, Prog. Theor. Phys. Suppl. 154, 341 (2004), https://doi.org/10.1143/ptps.154.341.
12. T.H. Rider, *Is There a Better Route to Fusion?*, LLNL High Energy Density Science Seminar, 19 January 2023, https://heds-center.llnl.gov/sites/heds_center/files/2023-03/01-19-23_slides_-_rider_.pdf.
13. G. Gamow, *Zur Quantentheorie des Atomkernes*, Z. Phys. 51, 204–212 (1928), https://doi.org/10.1007/BF01343196.
14. R. d'E. Atkinson and F.G. Houtermans, *Zur Frage der Aufbaumöglichkeit der Elemente in Sternen*, Z. Phys. 54, 656–665 (1929), https://doi.org/10.1007/BF01341595.
15. J.D. Lawson, *Some Criteria for a Power Producing Thermonuclear Reactor*, Proc. Phys. Soc. B 70, 6–10 (1957), https://doi.org/10.1088/0370-1301/70/1/303.
16. S.E. Jones, *Muon-Catalysed Fusion Revisited*, Nature 321, 127–133 (1986), https://doi.org/10.1038/321127a0.
17. M.B. Chadwick and B.C. Reed, *Introduction to Special Issue on the Early History of Nuclear Fusion*, Fusion Sci. Technol. 80, S1 (2024), https://doi.org/10.1080/15361055.2024.2346868.
18. Companion analyses: `docs/Findings/Analysis_Compact_Geometry.md`, `docs/Findings/Analysis_hQVM_Percolation.md`, `docs/Findings/Analysis_hQVM_Cohomology.md`, `docs/Findings/Analysis_Gravity_Note.md`, `docs/Findings/Analysis_Hilbert_Space_Representation.md`, and `docs/CGM_Logic.md`. Kernel specification layer: `docs/Gyroscopic_Computational_Theory/hQVM_Specs_Formalism.md`, `docs/Gyroscopic_Computational_Theory/hQVM_Features_Report.md`, `docs/Gyroscopic_Computational_Theory/hQVM_QuBEC_Theory.md`, and `docs/Gyroscopic_Computational_Theory/hQVM_SDK_Quantum_Computing.md`.
19. Data catalogs: `data/catalogs/ensdf/` and `data/catalogs/fusion/` (SOURCE files in each directory); local isomer PDFs in `docs/references/`.
