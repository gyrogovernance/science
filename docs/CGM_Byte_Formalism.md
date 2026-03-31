# Byte Boundaries and the 6-Bit Runtime

Reference note for the Router kernel's byte-boundary analysis and how it reduces effective processing to 6 bits at runtime via the GENE_Mic archetype and the 24-bit GENE_Mac tensor.

---

## 1. Depth-4 Closure and the 48-Bit Projection

The architecture is based on **depth-4 closure**: any 4 components are always known, whether they are bits, bytes, or 12-bit tensors.

**The 4-byte frame:** Prefix, Present, Past, Future. These four bytes form the minimal closure unit and map to the four CGM stages in the transition law:

| Layer | CGM Stage | Byte Role | Transition Law |
|-------|-----------|-----------|----------------|
| 0 | CS | Prefix (byte enters) | Common source of the mutation |
| 1 | UNA | Present | Mutation acts on A; variety introduced |
| 2 | ONA | Past | A_next = B ^ 0xFFF; B was previous A, now complemented |
| 3 | BU | Future | B_next = A_mut ^ 0xFFF; mutated present committed as future's passive record |

**Projection:** An 8-bit byte projects to a 12-bit tensor via the expansion function. The projection maps each byte to a unique 12-bit mask that operates on the tensor state.

**48-bit tensors:** Four bytes project to four 12-bit tensors: 4 x 12 = 48 bits. The 48-bit tensor is the full projection of the 4-byte frame (Prefix, Present, Past, Future). The 24-bit GENE_Mac (A12, B12) is one slice of this structure; the full 48-bit tensor extends it when all four byte positions are considered.

---

## 2. The 8-Bit Byte and CGM-Linked Bit Pairs

In the kernel, each input byte is first turned into an **intron** by XOR with the micro archetype:

```text
intron = byte ^ GENE_MIC_S   where  GENE_MIC_S = 0xAA
```

The 8 bit positions of the intron (and thus of the byte, up to the fixed XOR) are not uniform. They group into **4 paired bit groups** with distinct roles that align with the CGM stage structure:

| Bit  | Pair | Gyrogroup Role | CGM Stage |
|------|------|----------------|-----------|
| 0    | L0   | Left Identity | CS |
| 1    | LI   | Left Inverse | UNA |
| 2    | FG   | Forward Gyration | ONA |
| 3    | BG   | Backward Gyration | BU |
| 4    | BG   | Backward Gyration | BU |
| 5    | FG   | Forward Gyration | ONA |
| 6    | LI   | Left Inverse | UNA |
| 7    | L0   | Left Identity | CS |

So the byte has a **palindromic** structure: Left Identity at the boundaries (bits 0 and 7), Left Inverse next (1 and 6), then Forward Gyration (2, 5) and Backward Gyration (3, 4) in the middle. This reflects the cyclic CGM structure (CS -> UNA -> ONA -> BU -> ...) folded onto 8 positions.

### 2.1 Families and Bit Pairs

**Families** are defined by the **L0 boundary bits** (positions 0 and 7). These 2 bits give 4 combinations, partitioning the 256 introns into **4 families of 64** each.

The **bit pairs** (L0, LI, FG, BG) are groupings of bit **positions** by their gyrogroup role. They are NOT families — they describe the structural role of each bit position.

```python
L0_MASK = 0b10000001  # bits 0, 7 - Left Identity (boundary) -> defines families
LI_MASK = 0b01000010  # bits 1, 6 - Left Inverse
FG_MASK = 0b00100100  # bits 2, 5 - Forward Gyration
BG_MASK = 0b00011000  # bits 3, 4 - Backward Gyration
```

- **L0 bits** (0, 7): boundary anchors that **define the 4 families**. They do not flip tensor pairs.
- **LI bits** (1, 6): payload bits controlling 2 of the 6 tensor pairs.
- **FG bits** (2, 5): payload bits controlling 2 of the 6 tensor pairs.
- **BG bits** (3, 4): payload bits controlling 2 of the 6 tensor pairs.

### 2.2 The 6 DoF and Tensor Transformation

The **6 payload bits** (LI, FG, BG pairs at positions 1-6) are the spaces of active operations. Each payload bit correlates to **one pair** (one of the 6 DoF) in the tensor.

**The 6 DoF as pairs:**

```
Frame 0:  [-1, 1]  [-1, 1]  [-1, 1]   <- pairs 0, 1, 2
Frame 1:  [ 1,-1]  [ 1,-1]  [ 1,-1]   <- pairs 3, 4, 5
```

Each pair is 2 bits in the 12-bit representation. When a payload bit is set, it **flips that entire pair** — both bits together. The pair `[-1, 1]` (bits `10`) becomes `[1, -1]` (bits `01`).

**The algebra:** This organizes 256 introns into a structured space:

- **L0 bits** (bits 0, 7) = **4 families** (2 boundary bits → 4 combinations)
- **Payload bits** (bits 1-6) = **64 members per family** (6 bits → 64 transformations)
- **Total:** 4 families × 64 = 256 unique introns

### 2.3 Verified Mask Properties

The payload-to-mask mapping has the following verified properties:

1. **Dipole flip (PROVED):**
   For each payload bit `i` in {1..6}, toggling that bit changes exactly one pair in the 12-bit mask:
   - The 12 mask bits decompose into 6 pairs: `(0,1), (2,3), (4,5), (6,7), (8,9), (10,11)`.
   - Toggling payload bit `i` flips both bits in pair `i-1` and leaves all other pairs untouched.
   - Mapping: bit 1 -> pair 0, bit 2 -> pair 1, ..., bit 6 -> pair 5.

2. **Mask uniqueness:**
   - The 12-bit mask space contains exactly **64** distinct values (6 payload bits).
   - The combined pair `(family_idx, mask12)` yields **256** distinct values:
     - 4 families x 64 masks = 256.

3. **Families remain structural:**
   - The 2 L0 boundary bits (0 and 7) select one of 4 families but do not change which dipole pair each payload bit controls.
   - Transformation content lives entirely in the 6 payload bits; families provide spinorial/topological context.

---

## 3. Boundary Bits and the "Only 6 Bits" Idea

The key finding: **bits 0 and 7 are Left Identity (L0)**. They define identity and frame; they do not carry the dynamic transformation content. The **middle 6 bits (1..6)** carry the physical/chiral/dynamic information.

Consequences:

- If we **assign only the boundaries (0 and 7) to families** as fixed structural roles, the remaining **6 bits** are the ones that actually drive transformation.
- At runtime we can therefore **organize processing around 6 bits of dynamic content**; the two boundary bits fix the "frame" and can be handled by the expansion and mask structure rather than by full 8-bit state.

**Boundary bits are structural anchors, not dynamic content.** They do not change the tensor; they only select which family the transformation belongs to. This is the design choice that lets us treat the byte as **2 anchor bits + 6 payload bits**.

---

## 4. How This Mutates GENE_Mic and Produces the 12-Bit Mask

**GENE_Mic** is the 8-bit holographic archetype `0xAA`. Mutation is transcription:

- `intron = byte ^ 0xAA`

So every byte is mapped to a unique intron; `0xAA` is the reference byte (intron `0x00`).

The intron is then **expanded** into a **12-bit Type A mask**. The expansion should respect the L0/payload split established in Section 2:

**Correct decomposition (per Appendix G):**

- **Family index** (2 bits) = L0 boundary bits (positions 0 and 7):
  ```python
  family = ((intron >> 7) & 1) << 1 | (intron & 1)
  ```

- **Micro-reference** (6 bits) = payload bits (positions 1-6):
  ```python
  micro_ref = (intron >> 1) & 0x3F
  ```

This gives: 4 families x 64 micro-references = 256 unique introns.

### 4.1 What Families (Boundary Bits) Actually Do

The **payload bits (1-6)** define the transformation - which of the 64 micro-references to apply to A.

**Important:** Only A is mutated by the entering byte. B is NOT mutated pre-gyration (the mask bottom 12 bits are always 0). B only changes during gyration.

The **family bits (0,7)** do NOT define transformation content. Their role relates to the **L0 parity**:

```python
L0_parity = (bit0 XOR bit7)
```

The 4 families:
- Family 00 (bit7=0, bit0=0): L0_parity = 0
- Family 01 (bit7=0, bit0=1): L0_parity = 1
- Family 10 (bit7=1, bit0=0): L0_parity = 1
- Family 11 (bit7=1, bit0=1): L0_parity = 0

**Spin structure in GENE_Mac:**

```python
A12 = [[[-1, 1], [-1, 1], [-1, 1]], [[ 1, -1], [ 1, -1], [ 1, -1]]]  # spin +
B12 = [[[ 1, -1], [ 1, -1], [ 1, -1]], [[-1, 1], [-1, 1], [-1, 1]]]  # spin -
```

A12 and B12 are **anti-parallel** - opposite spin orientations. The entering byte mutates A toward or away from its archetype. The L0 parity (family) may define how this mutation relates to gyration behavior, but since B is not mutated by the input, the family does NOT directly select "which component to affect."

### 4.2 Families Provide 720° Spinorial Closure

The 2 family bits give exactly **4 values**, which correspond to the 4 layers of the SU(2) spinorial cycle:

| Family | Bits (7,0) | Layer | Phase | Role |
|--------|------------|-------|-------|------|
| 00 | 0,0 | CS | 0° | Identity |
| 01 | 0,1 | UNA | π | Global inversion |
| 10 | 1,0 | ONA | 2π | Minus identity (spinor sign flip) |
| 11 | 1,1 | BU | 3π | Return toward closure |

Closure occurs at 4π (720°) when the cycle returns to Layer 0. This is the **spinorial double-cover structure of SU(2)**: a spinor returns to identity only after 720°, not 360°.

**Key insight:** The family bits don't define transformation content. They define **which phase of the spinorial cycle** the transformation operates in. The payload (bits 1-6) transforms A; the family (bits 0,7) selects the closure layer.

This explains why we need exactly 2 boundary bits: fewer gives insufficient closure depth; more is redundant.

**Note on current implementation:** The codebase currently uses a different expansion formula that extracts bits 6,7 as the family index instead of bits 0,7. This is inconsistent with the L0 boundary bit definition and needs architectural review. See the NOTE in `atlas.py` (Category 2: GENE_Mic intron-stage priors).

---

## 5. How the Mask Affects the GENE_Mac Tensor (24-Bit State)

### 5.1 GENE_Mac as a Tensor with ±1 Values

GENE_Mac is fundamentally a **tensor with -1 and +1 values**, not merely a bit pattern. The canonical tensor definition:

```python
GENE_Mac = np.array([
    [[[-1, 1], [-1, 1], [-1, 1]], [[ 1, -1], [ 1, -1], [ 1, -1]]],  # A12
    [[[ 1, -1], [ 1, -1], [ 1, -1]], [[-1, 1], [-1, 1], [-1, 1]]]   # B12
], dtype=np.int8)  # Shape: [2, 2, 3, 2]
```

Shape: `[2 components, 2 frames, 3 rows, 2 cols]` = 24 elements, each ±1.

### 5.2 The 6 DoF Structure

Each 12-bit component has **2 frames** and **3 rows** per frame. Each row is a **pair** `[-1, 1]` or `[1, -1]`:

```
Frame 0:  [-1, 1]  [-1, 1]  [-1, 1]   <- 3 pairs (rows)
Frame 1:  [ 1,-1]  [ 1,-1]  [ 1,-1]   <- 3 pairs (rows)
```

**6 DoF = 6 pairs** (3 rows × 2 frames). Each pair represents ONE axis with its two oriented sides (negative, positive). The pair `[-1, 1]` is one axis; `[1, -1]` is that axis flipped.

**Micro vs macro archetype:**

- **Micro archetype** (12-bit): `0xAAA` = `101010101010` (the alternating bit pattern)
- **Macro archetype** (tensor): `[[[-1, 1], [-1, 1], [-1, 1]], [[ 1, -1], [ 1, -1], [ 1, -1]]]` (one component at rest)

**Bit-to-±1 packing:**

- **Bit = 0 → +1** (archetypal polarity)
- **Bit = 1 → -1** (mutated polarity)

### 5.3 6 Payload Bits → 6 DoF

The **6 inner bits** of the intron (bits 1-6, excluding boundaries 0 and 7) are the **spaces of active operations**. They map to the **6 DoF** (6 pairs) of the tensor:

- Each payload bit controls **one pair** (one axis)
- Flipping a payload bit flips **both bits** in that pair: `[-1, 1]` becomes `[1, -1]`

The **2 boundary bits** (0 and 7) do not flip tensor elements. They select the family.

This is why **6 bits define a complete mutation**: 6 payload bits control the 6 pairs; the boundaries are structural anchors.

### 5.4 Bit Packing

Each 12-bit component unpacks to a `[2, 3, 2]` tensor (2 frames x 3 rows x 2 cols). The micro archetype `101010101010` unpacks to the macro tensor form:

| Component | Hex | Binary (micro) | Macro tensor (6 DoF) |
|-----------|-----|----------------|----------------------|
| A12 (default) | `0xAAA` | `101010101010` | `[[[-1, 1], [-1, 1], [-1, 1]], [[ 1, -1], [ 1, -1], [ 1, -1]]]` |
| B12 (default) | `0x555` | `010101010101` | Complement of A12 |

The default state has A12 and B12 as **exact complements** (`A ^ B = 0xFFF`), meaning their tensor forms have opposite signs at every position. This encodes the fundamental **chirality** of the system.

### 5.5 The 24-Bit State

The "macro" state is the **24-bit GENE_Mac**: two 12-bit components (A12, B12), with default state:

- `ARCHETYPE_A12 = 0xAAA`
- `ARCHETYPE_B12 = 0x555`
- `ARCHETYPE_STATE24 = 0xAAA555`

The 12-bit mask acts **only on the A component**:

1. Mutate A (UNA): `A12_mut = A12 ^ mask_a12` — variety introduced
2. Gyration and complement:
   - `A12_next = B12 ^ 0xFFF` (ONA: B was past A, now complemented)
   - `B12_next = A12_mut ^ 0xFFF` (BU: mutated present committed as future's passive record)
3. Next state: `state24_next = (A12_next << 12) | B12_next`

So:

- **GENE_Mic** (0xAA) mutates the byte into an intron; the intron expands to a **12-bit mask** that encodes the 6-bit micro-reference plus the 2-bit family (boundary) index.
- **GENE_Mac** is the 24-bit state; the mask **only** touches the A half. The B component is updated by complement-and-swap. So the byte-boundary structure (and the 6-bit payload) affect the macro state **through this single 12-bit mask on A**, then the fixed gyration rule.

The 2x3x2 geometry of each 12-bit component (2 frames, 3 rows, 2 cols) is the same as in the expansion: frame 0 and frame 1 of the mask align with the two chirality frames of the state, so the "6 bits of dynamics" and "boundary/family" split are reflected in how the 24-bit state is updated.

---

## 6. Depth-4 Closure and Single-Step Projection

The architecture is depth-4: the minimal closure unit is a 4-byte frame (Prefix, Present, Past, Future), which projects to 4x12 = 48 bits in mask space and 4x32 = 128 bits in the full register-atom space.

### 6.1 Two Depth-4 Objects

Depth-4 structure appears in two distinct but related objects:

- **48-bit manifold projection:** 4x12-bit masks derived from payload (6 bits per byte), representing how four consecutive bytes act on the 12-bit manifold slices. This captures the mask-side geometry but discards family bits.

- **128-bit atom frame:** 4x32-bit register atoms (each 8-bit intron + 24-bit Mac state), representing the full execution context across a 4-byte frame. This is the level at which spinorial phase (families) and manifold updates are both visible, and at which 4-frame projections are bijective.

### 6.2 Single-Step 24-Bit Projection (Verified)

From a fixed 24-bit state (e.g. the archetype `0xAAA555`), applying all 256 bytes under the spinorial transition law produces **128** distinct next Mac states. This is not a defect; it reflects a 2-to-1 projection:

- The 24-bit Mac is one slice of the 48-bit depth-4 frame.
- Certain pairs of introns (differing in family + payload complement) map to the same 24-bit next state.
- Test result: `unique_states = 128/256` ("shadow projection").

**Physical Interpretation (SO(3) vs SU(2)):** The 24-bit Mac is a purely spatial SO(3) object. Because it only tracks 3D geometry (the 6 DoF), it suffers from the standard spinorial degeneracy where a spatial inversion is indistinguishable from a 360 degree phase rotation, collapsing 256 actions to 128 geometric states. The full 32-bit register atom (Mac + Intron) acts as the SU(2) spinor, retaining the 2-bit spin phase (family bits) to maintain the full 256-state bijection.

### 6.3 Depth-4 Projections (Verified)

- **48-bit mask-only (4x12):** Collisions appear as expected because family (L0) information is discarded. Test: 9995/10000 unique in random sampling.

- **32-bit intron sequence (4x8):** The mapping of 4 consecutive 8-bit input actions is **bijective**. For random 4-byte frames, `project_4byte_full` yields unique 32-bit values. Test: 10000/10000 unique. Single-byte bijective positions: 4/4. **Status: PROVED**.

**Interpretation:** The 24-bit Mac state is an SO(3) **shadow** (projection) of the full SU(2) register atom. Full depth-4 closure and spinorial phase information live in the 32-bit intron sequence (4x8) and the 48-bit projection (4x12). The apparent 128/256 degeneracy at 24 bits is the standard SO(3)/SU(2) double-cover and is resolved when we look at the full depth-4 objects.

---

## 7. Aperture Quantization and Horizons

The CGM aperture gap is defined continuously as:

- `delta_BU`: BU monodromy defect (radians) = 0.195342176580
- `m_a = 1/(2*sqrt(2*pi))` = 0.199471140201
- `rho = delta_BU / m_a` = 0.979300446087
- `Delta = 1 - rho` = 0.020699553913 (dimensionless aperture gap, ~2.07%)

### 7.1 Tick Spaces (Must Not Be Conflated)

We distinguish two 256-tick spaces:

- `T_256^(frac)`: 256-tick fraction line for dimensionless ratios (Delta, rho)
- `T_256^(turn)`: 256-tick circle for angles normalized by 2pi (delta_BU)

### 7.2 Quantization Results

**On `T_256^(frac)` (fractions):**
- `Q_256(Delta) = 5/256` = 0.0195312500
- 5/256 is the **best 8-bit dyadic approximation** of Delta.
- Quantization error: 0.001168303913
- This is the byte-horizon expression of aperture: 5 "ticks" open, 251 closed.

**On `T_256^(turn)` (turns):**
- `tau = delta_BU / (2*pi)`
- `Q_256(tau) = 8/256 = 1/32 turn`
- This matches delta_BU ~ pi/16 to within ~1e-3 radians.

**At depth-4 projection scale (48-bit):**
- `Q_48(Delta) ~ 1/48`
- 48 * Delta = 0.9936 ~ 1
- This aligns Delta with the 48-bit horizon (4x12) of the depth-4 projection.

### 7.3 The 2/3 Ratio: Chirality to Space

The ratio of these canonical approximants is:

- `(1/48) / (1/32) = 2/3`

This 2/3 factor is not merely a numerical fraction; it is the **ratio of Chirality to Space**:

- **2 = Chirality** (the two frames A and B; the spinorial double-cover)
- **3 = Spatial Dimensions** (the X, Y, Z axes; the 3 rows per frame)

The manifold consists of 2 chiral layers (spin states) projected across 3 spatial axes. The aperture exists precisely because mapping a 2-phase chiral spinor onto a 3-axis discrete space leaves a fractional geometric gap. This ratio dictates how purely topological spin (chirality) bridges into physical geometry (space).

### 7.4 Horizon Lemma (Arithmetic)

Consider sizes of the form `n = 2^a * 3^b` with a, b non-negative integers.

**Key facts:**

- `log2(n) = a + b*log2(3)`. Since `log2(3)` is irrational, `log2(n)` is an integer **iff b = 0** (pure powers of two).

- **Dyadic horizons** (b=0): 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, ...

- **Predecessor horizons** (b=1): For each `k >= 1`, define `P_k = 3 * 2^(k-1)`. Then:
  ```
  2^k < P_k < 2^(k+1)
  ```
  Moreover:
  ```
  P_k = (3/4) * 2^(k+1)
  ```
  So `P_k` is the **maximal 2^a * 3 size** that fits below the next dyadic horizon.

**Horizon table:**

| n    | Form        | log2(n) | Type        | Role |
|------|-------------|---------|-------------|------|
| 12   | 2^2 x 3^1   | 3.585   | predecessor | projection unit (3*4 bits) |
| 16   | 2^4 x 3^0   | 4.000   | dyadic      | 2^4 |
| 32   | 2^5 x 3^0   | 5.000   | dyadic      | 2^5 |
| 48   | 2^4 x 3^1   | 5.585   | predecessor | P_4 = 3*16 = (3/4)*64; depth-4 projection |
| 64   | 2^6 x 3^0   | 6.000   | dyadic      | cache line / payload space |
| 96   | 2^5 x 3^1   | 6.585   | predecessor | P_5 = 3*32 = (3/4)*128 |
| 128  | 2^7 x 3^0   | 7.000   | dyadic      | depth-4 atoms (4 x 32) |
| 256  | 2^8 x 3^0   | 8.000   | dyadic      | byte horizon |
| 384  | 2^7 x 3^1   | 8.585   | predecessor | P_7 = 3*128 = (3/4)*512 |
| 512  | 2^9 x 3^0   | 9.000   | dyadic      | cache line (64 bytes) |
| 768  | 2^8 x 3^1   | 9.585   | predecessor | P_8 = 3*256 = (3/4)*1024 |
| 1024 | 2^10 x 3^0  | 10.000  | dyadic      | 2^10 |
| 1536 | 2^9 x 3^1   | 10.585  | predecessor | P_9 = 3*512 = (3/4)*2048 |
| 2048 | 2^11 x 3^0  | 11.000  | dyadic      | 2^11 |
| 3072 | 2^10 x 3^1  | 11.585  | predecessor | P_10 = 3*1024 = (3/4)*4096 |
| 4096 | 2^12 x 3^0  | 12.000  | dyadic      | 12-bit mask |

**Byte-formalism note (micro-only):**

The intron's palindromic 4-pair partition (L0 / LI / FG / BG) naturally separates into:
- **1 boundary pair** (L0: bits 0, 7)
- **3 interior pairs** (LI, FG, BG: bits 1-6)

When scaling structures that preserve this **3+1 split** while staying aligned to dyadic (power-of-two) boundaries, the arithmetic pattern `3*2^k just below 4*2^k = 2^(k+2)` corresponds exactly to the predecessor horizons. This is why 48, 96, 384, etc. appear naturally as "one step before" 64, 128, 512, etc.

---

## 8. Hardware Alignment: 6-Bit Runtime and Cache Lines

The 6-bit runtime is not only a structural property of the byte; it also matches the native addressing structure of hardware cache lines.

### 8.1 Cache Line Structure

Typical L1 cache lines are 64 bytes (512 bits). Addressing 64 items requires 6 bits:

- `2^6 = 64` -> 6-bit offset selects one byte within a cache line.

### 8.2 Intron as Cache Address

The intron split maps directly to cache addressing:

- **Bits 1-6 (payload):** 6-bit field with 64 values -> **cache line offset** (which of 64 transformations)
- **Bits 0,7 (L0 anchors):** 2-bit field with 4 values -> **cache tag** (which of 4 families/lines)

This yields:

- 4 families x 64 payload offsets = 256 introns = full alphabet.
- Intron is a literal `[Family][Payload]` address in an L1-sized logical space.

The 6-bit runtime is therefore aligned both with the intrinsic CGM DoF structure and with the hardware's natural 64-element memory grain.

---

## 9. Summary

| Concept | Role |
|--------|------|
| Depth-4 closure | Any 4 components (bits, bytes, or 12-bit tensors) are always known. |
| 4-byte frame | Prefix, Present, Past, Future. Projects to 48-bit tensor (4 x 12). |
| Projection | 8-bit byte -> 12-bit tensor via expansion. |
| Bit pairs (L0, LI, FG, BG) | Groupings of bit **positions** by gyrogroup role. NOT families. |
| Families | Defined by **L0 boundary bits (0, 7)**. 4 families x 64 = 256. Provide 720 deg spinorial closure. |
| 6-bit payload (bits 1-6) | **Spaces of active operations**. Each bit controls one of the 6 pairs (6 DoF). Dipole flip PROVED. |
| GENE_Mic (0xAA) | Micro archetype (8-bit); mutation = `intron = byte ^ 0xAA`. |
| 12-bit mask | Expansion of 8-bit intron. 64 unique masks from 6 payload bits. |
| GENE_Mac (24-bit) | **SO(3) shadow**. 128/256 unique states (spatial geometry only). |
| 32-bit register atom | **SU(2) spinor**. Mac + intron retains spin phase. Full 256-state bijection. |
| 32-bit intron sequence | 4x8 depth-4 trace. PROVED bijective. |
| Aperture (Delta) | ~2.07%. Best 8-bit: 5/256. Ratio 2/3 = Chirality(2) / Space(3). |
| Horizon Lemma | P_k = 3*2^(k-1) = (3/4)*2^(k+1). Dyadic (b=0) vs predecessor (b=1) horizons. |
| 3+1 split | 1 boundary pair (L0) + 3 interior pairs (LI/FG/BG) -> 3*2^k horizons. |
| Cache alignment | Bits 1-6 = offset (64), bits 0,7 = tag (4). Intron = L1 cache address. |

**Key insight:** The 24-bit Mac is an SO(3) object (spatial geometry); the 32-bit register atom (Mac + intron) is the SU(2) spinor that retains spin phase via family bits. The 128/256 degeneracy at 24 bits is the standard spinorial double-cover. The 2/3 aperture ratio is Chirality/Space: mapping 2-phase spinors onto 3-axis geometry leaves a fractional gap. The Horizon Lemma shows that predecessor horizons (48, 96, 384...) arise naturally from the byte's 3+1 interior/boundary split when scaled to dyadic alignment. The Router replaces ALU arithmetic with pure structural routing into L1 cache.
