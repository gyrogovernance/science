# Yang–Mills Existence and Mass Gap — computational suite

This directory is the runnable certificate suite for the Clay Yang–Mills existence and mass gap problem in the Common Governance Model (CGM). It translates Formalism carrier identities into finite Jaffe–Witten / Osterwalder–Schrader / Wilson lattice vocabulary and records PASS/FAIL gates.

**Role split.** `Yang_Mills_Mass_Gap_Solution.md` is the mathematical solution paper (definitions, theorems, continuum reading). This README is the operational inventory of what the scripts contain and print, so you can know the suite without reading the code. `Yang_Mills_Mass_Gap_results.txt` is the last full-run log. Scripts print measurements and gates only; interpretation lives in the solution paper.

## How to run

From this directory:

```
python Yang_Mills_Mass_Gap_run.py
```

Optional flags:

```
python Yang_Mills_Mass_Gap_run.py --fast
python Yang_Mills_Mass_Gap_run.py -o path/to/out.txt
```

`--fast` skips some expensive charts inside `_1` / `_5` (still runs the full section outline). Exit code `0` iff `H7_closed` is true; otherwise `1`. Output is teed live to the console and to `Yang_Mills_Mass_Gap_results.txt` (or `-o`).

Individual modules are also runnable:

| Command | Entry |
|---------|--------|
| `python Yang_Mills_Mass_Gap_1.py [--fast]` | `run_jw_wilson` |
| `python Yang_Mills_Mass_Gap_2.py` | `run_curvature_3d` |
| `python Yang_Mills_Mass_Gap_3.py` | `run_sc_h6` |
| `python Yang_Mills_Mass_Gap_4.py` | `run_refine` |
| `python Yang_Mills_Mass_Gap_5.py [--fast]` | `run_formalism` |

Pipeline order in the orchestrator: `_1` → `_2` → `_3` → `_4` → `_5`.

## Directory inventory

| File | Role |
|------|------|
| `Yang_Mills_Mass_Gap_run.py` | Orchestrator; prints derivation outline; tees stdout; prints `DELIVERY SUMMARY`; exit from `H7_closed`. |
| `Yang_Mills_Mass_Gap_common.py` | Shared constants, Δ-ruler / unit map, K₄/Q₈ tables, Wilson–Kogut–Susskind 2D engines, Euclidean OS helpers. No section certificates. |
| `Yang_Mills_Mass_Gap_1.py` | Sections **0–10**: carrier / JW-A, Δ-ruler, shadow lock, Wilson lattice JW. |
| `Yang_Mills_Mass_Gap_2.py` | Section **11**: Euclidean OS Θ-Gram; also `LatticeYM3D`, Λ² span helpers used by `_3`/`_4`. |
| `Yang_Mills_Mass_Gap_3.py` | Sections **12–19**: two-plaquette conjugacy, SC0/SC1, Lemma L′, H6, D1/D2/D3, infinite-volume OS checklist. |
| `Yang_Mills_Mass_Gap_4.py` | Section **20**: magnetic degree-2 / Λ² lock, intertwiner, D0 transversality / dark intersection / κ₂ charts. |
| `Yang_Mills_Mass_Gap_5.py` | Sections **21–22**: Formalism Clay checklist (H0, G, R4, M, H, gap, Hopf) and H7 aggregate. |
| `Yang_Mills_Mass_Gap_results.txt` | Last full-run certificate log (overwrite on each orchestrated run). |
| `Yang_Mills_Mass_Gap_Solution.md` | Solution paper (not a script inventory). |
| `QUANTUM YANG–MILLS THEORY.md` | Jaffe–Witten problem statement (reference copy). |

## Shared layer (`Yang_Mills_Mass_Gap_common.py`)

**Formalism cardinalities (asserted):** `D_PAYLOAD = 6`, `|Byte| = 256`, `|K₄| = 4`, `|Q₈| = 8`, `|χ| = |C64| = 64`, `|Ω| = 4096`, `SHELLS7 = 7`, `A_kernel = 5/256`.

**Standing scalars:** `Δ`, `ρ`, `Q_G = 4π`, `QG_MA2 = Q_G m_a² = 1/2`, `C₂ = C(6,2) = 15`, `v = E_EW`, stage angles `S_CS` / `S_UNA` / `S_ONA`, `GENE_Mic`, `G_DEFINING_KS = 1`, `BETA_DEFINING = 1`, `SE3_DOF = 6`, `D_CONTINUUM_CGM = 4`, `BU_CLOSURE_DEPTH = 4`.

**Unit map:** `UNIT_MAP_MODE = "grade1_only"` with `E_unit := v·Δ`. The matching readout `m_gap/κ₂` is diagnostic only and is not used as the unit definition.

**Nuclear forced class:** `(k,ℓ) = (6,2)` with STF and equatorial tick flags (`NUCLEAR_CLASS`).

**Δ-ruler helpers:** `cgm_ym_gap_prediction()` (Routes A/B + IR optical conjugacy ladder), `curvature_index_kappa2`, `unit_map_trackd_to_gev`, `optical_conjugacy_ir_ladder`, `gap_record`, `matching_readout_only`.

**Printing:** `section`, `section_title`, `gate`, `progress`; `RUN_SECTIONS` is the publication outline (22 entries, sections 0–22).

**Lattice engines:** `K4()`, `Q8()`, `Q8_from_extension()`, Wilson weights, `LatticeYM` (2D KS), gauge-invariant spectrum / gap helpers, orbit-reduced He/Hm, correlator local mass, `Lattice2D` + Euclidean action / OS Gram / transfer matrix builders (`os_gram_matrix_exact`, `certify_os_rp_exact`, `build_transfer_matrix_exact`, …). SC constants: `C_SHARP_FINITE = 1/√3`, `C_G_SU2_CONT = 4/3`, `R_STAR_2D = 2`.

**Note:** ℓ²(Gᴱ) lattice Hilbert spaces are Clay-certificate spaces, not the kernel space. The kernel space is Ω (4096). Lattice OS/SC charts sit on top of the GNS vacuum from Ω.

## Certificate outline (publication order)

Printed at every orchestrated run (`RUN_SECTIONS`):

0. PREFLIGHT — hQVM kernel cross-check
1. JW-A — definitions (A, omega) → (H, pi, Omega)
2. DELTA-RULER — dimensionful mass-gap prediction (Routes A/B)
3. SHADOW Delta_W — formula n/(2(n−1)) on Omega [Track B]
4. SHADOW LOCK — Q_G m_a² vs collapsed B/C gap
5. GAUGE GROUP — K4 holonomy + Q8 central extension
6. WILSON H — Kogut–Susskind algebra certificates
7. Q8 PLAQUETTE — non-abelian JW gap
8. CORRELATOR / LOCALITY / TRANSFER / GENE_Mic
9. UNIT MAP — grade-1 E_unit = v Delta
10. SPACETIME — D=4 packaging
11. OS EUCLIDEAN GRAM (Θ measure)
12. TWO-PLAQUETTE CONJUGACY POSITIVITY
13. SC1 LOCAL EXCITATION (abelian K4)
14. SC0-Q FREE PLAQUETTE
15. LEMMA L′ + LOCAL GAUGE U_g + SC1-Q
16. H6 CLUSTERING FROM SC1 FLOOR
17. CGM NATIVE GAP CHECKS
18. D1/D2/D3 — Euclidean transfer / omega_inf / GENE_Mic
19. INFINITE-VOLUME / OS CHECKLIST
20. MAGNETIC EXCITATION DEGREE-2 + Lambda2 LOCK
21. FORMALISM CHECKLIST — G / R4 / measure / Hilbert / gap / Hopf fiber
22. H7 FORMALISM AGGREGATE (H7_closed := formalism_checklist_closed)

## Module contents

### `_1` — JW object + Wilson lattice (`run_jw_wilson`)

**Carrier block (0–4).** Kernel d=6 API cross-check; `|Ω| = 4096`; JW-A printout (GENE_Mic / GENE_Mac, `QG_MA2`, continuous Δ vs discrete `A_kernel`, C₂); Routes A/B mass predictions and IR ladder; shadow `Δ_W(n=256) = n/(2(n−1))`; D3-struct shadow lock (`lim Δ_W = 1/2 = QG_MA2`, oriented `|Δ − 1/2|` witness).

**Wilson block (5–10).** K₄ holonomy on Ω; Q₈ derived as central extension of K₄; KS algebra certificates on K₄/Q₈ (`A_v` projections, `[A_v, H_mag]`, Wilson V spectra); defining Q₈ 1×1 plaquette spectrum (`Δ_JW = E₁ − E₀`); correlator decay, multi-link clustering, locality commutators, transfer contraction vs `e^{−a·gap}`, GENE_Mic ω datum; grade-1 unit map (`κ₂ = gap/Δ`, `E_unit`, `m_phys`); D = n+1 = 4 packaging (dimensionless g at D=4).

### `_2` — OS Euclidean Gram + 3D / Λ² helpers (`run_curvature_3d`)

**Printed certificate:** section 11 only — exact OS Gram on Q₈, T×L = 2×2, `β = BETA_DEFINING`, temporal gauge off; reports `n_configs`, `n_ops`, `min eig M`, PSD gate.

**Also exported for later modules (not separate run sections):** Λ² bivector basis / GF(2) rank, `curvature_span_*`, magnetic uniqueness audit, `LatticeYM3D` (3D spatial Wilson), tree-reduced He/Hm, GI SVD basis, `os_positivity_matrix_euclid`.

### `_3` — SC scaffolding + H6 / D / infinite-volume OS (`run_sc_h6`)

**12–15 (SC scaffolding).** Two-plaquette conjugacy positivity (orbit census; unique local Wilson ray); SC1 local excitation on abelian K₄ and numeric Q₈ α\* vs `2/√3`; SC0-Q free-plaquette form at `C = 1/√3` (K₄ proved, Q₈ finite exhaustion); Lemma L′ (class-function V, conjugacy, local `U_g`, boundary-preserving) + SC1-Q incidence; C1 template (SC1-G = SC0-G + L′ + r_\*); SC0-G-cont continuous lemma; Formalism Q₈ root chart.

**16–19 (H6 / D / IV).** H6 clustering from SC1 floor (`Δ_*` vs gaps and `‖T‖ ≤ e^{−Δ_*}`); CGM native gap checks (Δ > 0, Routes A/B vs `m_phys`, κ₂ vs C₂); D2 ω_∞ gap pin (K₄ volume tower + Q₈ Lx=1→2), D1 Euclidean transfer PSD, D3 GENE_Mic orientation (`Q₈` gap ≠ shadow lock); infinite-volume OS checklist flags (`H4_time_RP_finite`, H6 abelian/Q₈, Cauchy on K₄ dens, Lemma IV uniqueness, `Theorem_OS_lat_hypercubic`, GENE_Mic unit map).

### `_4` — Λ² lock + intertwiner (`run_refine`)

**Single printed section 20**, internally many charts:

- Magnetic excitation degree-2 support on defining Q₈ chart (`dV` raise, dim_K = C₂ = 15).
- Slim carrier↔lattice intertwiner (dim_full 64 → dim_GI 28; He/Hm/h0/comm residuals; Layer A/C).
- D0-D local correlator mass κ₂ on finite charts; O_Λ² complete + N2 isotropy skeleton (`|I₂| = 15`, Γ = S₆ transitive).
- D0 2D plaquette transversality; D0-3D dark intersection empty (`S_xy ∩ S_yz ∩ S_zx = ∅`); κ₂→C₂ input checklist (steps 1–3 finite; step 4 continuum = Hopf).
- D0-D(2) Γ-isotropy bottom + Δ scaling; O_Λ² g-coupling tower at defining g = 1.

### `_5` — Formalism Clay checklist + H7 (`run_formalism`)

**Section 21** prints, in order:

| Block | What is certified |
|-------|-------------------|
| H0 Hopf fiber census | flat=16, curved=240, mean fold disagreement 1/2, mean S/d = 1/2, entanglement sum 12288, F²=id on rest, depth4=48, SO3 shadow=128, 128 partner pairs |
| Oriented shadow | GENE_Mic zero intron, Δ>0, fold holonomy, SO3 2-to-1, D3 `QG_MA2` lock |
| Hopf dictionary | phase pairs (fwd/rev/BU), same census + `Q_G`, D=n+1=4 |
| Admissible quotient | `|Byte|/|q6|=4`, `|Ω|/|χ|=64` |
| Gap on quotient | Δ>0, C₂=15, D0-3D, IsoSupport, N2=15; Route A readout |
| Lemma G | GENE_Mic + family×payload SE(3) DoF, K₄ families |
| Lemma R4 | 3+1 packaging, `Q_G = 4π` |
| Lemma M | shell populations sum to 4096; Z₁(β) closed form |
| Lemma H | horizons 64/64, `|H|² = |Ω|`, F involution ±1 eigenspaces 2048/2048, GENE_Mac rest |
| Lemma gap | ρ, Δ = 1−ρ, Q₂₅₆ = 5/256, C₂, Route A |
| Hopf horizon chart | SO3=128, C₂, Δ, `Q_G m_a² = 1/2`, Route A in (1,2) GeV |
| κ₂ spectral bound | Q₈ 1×1: `m_coupled(O_Λ²)`, below-threshold scan vs `15Δ` |
| Γ lift | intron+GENE_Mic on Ω; Aut(Q₈) order 24, H-symmetry on H_GI |

Aggregate boolean: `formalism_checklist_closed`.

**Section 22:** `H7_closed := formalism_checklist_closed`; reprints lemma flags and Route A `m_A_GeV`.

## Key numbers (last full run)

From `Yang_Mills_Mass_Gap_results.txt` (full run, `fast: False`, 2026-07-26 UTC). Re-run to refresh.

| Quantity | Value |
|----------|--------|
| `|Ω|` | 4096 |
| `Δ` | ≈ 0.020699553913 |
| `A_kernel = 5/256` | 0.01953125 |
| `QG_MA2` | 0.5 exactly |
| `Δ_W(n=256)` | ≈ 0.5019607843 |
| Route A `m_A = C₂·v·Δ²` | ≈ 1.582474 GeV |
| Route B `m_B` | ≈ 1.661556 GeV |
| Q₈ defining gap `Δ_JW` | ≈ 0.330221 |
| `E_unit = vΔ` | ≈ 5.096644 GeV |
| `m_phys` (grade-1) | ≈ 1.68302 GeV |
| `κ₂ = Δ_JW/Δ` | ≈ 15.953 (target C₂ = 15) |
| OS Gram `min eig M` (Q₈ 2×2) | ≈ 0.12118 (PSD) |
| SC1 floor α | `2/√3` ≈ 1.154701 |
| SC0 sharp C | `1/√3` ≈ 0.57735 |
| Hopf flat / curved | 16 / 240 |
| SO3 shadow pairs | 128 |
| Shell populations | `[64, 384, 960, 1280, 960, 384, 64]` |
| `formalism_checklist_closed` / `H7_closed` | True / True |

## Delivery summary

Orchestrator end block:

```
DELIVERY SUMMARY
-----
  formalism_checklist_closed : …
  H7_closed                  : …
```

Success: both true and process exit 0. Continuum reading of the finite certificates is the Hopf chart of the oriented quotient; see the solution paper.
