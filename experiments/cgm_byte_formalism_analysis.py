#!/usr/bin/env python3
"""
CGM Byte Formalism Analysis
===========================

Calculations for a byte-horizon (256) formalism based on CGM aperture structure.
Implements the mathematically justified quantization relations.

TICK SPACES (must not be conflated):
- T_256^(frac): 256-tick fraction line for dimensionless ratios (Delta, rho)
- T_256^(turn): 256-tick circle for angles normalized by 2*pi (delta_BU)
Cross-comparison requires an explicit homomorphism; none is canonical.

Architecture (from Byte_Boundaries_Reference):
- Depth-4 closure: 4 components (bytes or 12-bit tensors) always known
- Two depth-4 objects: 48 bits (4x12 projection) vs 128 bits (4x32 atoms)
- Bit pairs (L0, LI, FG, BG): groupings by gyrogroup role, NOT families
- Families: L0 boundary bits (0, 7) only. 4 families x 64 = 256 introns
- 6 payload bits (1-6): dipole flip PROVED (each bit controls one pair)

CGM definitions:
- delta_BU: BU monodromy defect (radians)
- m_a: BU aperture parameter = 1/(2*sqrt(2*pi))
- rho: closure ratio = delta_BU / m_a
- Delta: aperture gap = 1 - rho

Quantization: Q_N(x) = round(N*x) / N  for x in [0,1]

"""

import math
from typing import Dict, Any, Tuple
from dataclasses import dataclass

# Router architecture constants (Byte_Boundaries_Reference)
GENE_MIC_S = 0xAA
L0_MASK = 0b10000001  # bits 0, 7 - Left Identity (boundary) -> defines families
LI_MASK = 0b01000010  # bits 1, 6 - Left Inverse
FG_MASK = 0b00100100  # bits 2, 5 - Forward Gyration (ONA)
BG_MASK = 0b00011000  # bits 3, 4 - Backward Gyration (BU)

# Depth-4: two distinct objects
BYTES_PER_FRAME = 4
BITS_PER_PROJECTION = 12
BITS_PER_ATOM = 32  # register atom
DEPTH_4_PROJECTION_BITS = BYTES_PER_FRAME * BITS_PER_PROJECTION  # 48 (4x12)
DEPTH_4_ATOM_BITS = BYTES_PER_FRAME * BITS_PER_ATOM  # 128 (4x32)

# 4 families (L0 bits) x 64 members (6 payload bits) = 256
FAMILIES = 4
MEMBERS_PER_FAMILY = 64
INTRONS_TOTAL = FAMILIES * MEMBERS_PER_FAMILY  # 256


def expand_intron_to_mask_a12(intron: int) -> Tuple[int, int]:
    """
    Faithful implementation of Byte_Boundaries_Reference physics.

    Structure:
    - Bits 0, 7: L0 anchors (define 4 families - spinorial gyration phase)
    - Bits 1-6: Payload (6 DoF pairs - transformation content)

    Returns: (mask12, family_idx)
    - mask12: 12-bit mask with 64 unique values (6 payload bits -> 6 pairs)
    - family_idx: 0-3 (spinorial phase, controls gyration)

    Each payload bit flips one pair (both bits in the pair).
    Family does NOT modify the mask; it controls gyration phase.
    """
    # Extract 6 payload bits (positions 1-6)
    payload = (intron >> 1) & 0x3F

    # Build 12-bit mask: each payload bit i flips pair i
    # Pair i occupies bits (2*i, 2*i+1)
    mask12 = 0
    for i in range(6):
        if (payload >> i) & 1:
            mask12 |= (0b11 << (2 * i))

    # Extract family index from L0 bits (0 and 7)
    # bit0 -> position 0, bit7 -> position 1
    family_idx = (intron & 1) | ((intron >> 6) & 2)

    return mask12, family_idx


def step_state_spinorial(state24: int, byte_val: int) -> int:
    """
    The 24-bit transition law with 720-degree Spinorial Closure.

    - Payload (bits 1-6) defines mutation on A (64 masks)
    - L0 anchors (bits 0, 7) define the Spin Gyration phase (4 families)
    - Total: 64 x 4 = 256 unique transitions

    Spinorial structure (SU(2) double-cover):
    - bit 0: controls whether A_next is inverted (pi shift)
    - bit 7: controls whether B_next is inverted (pi shift)

    | Family | Bits (7,0) | Phase | Gyration |
    |--------|------------|-------|----------|
    | 00     | 0, 0       | 0     | A_next=B, B_next=A_mut |
    | 01     | 0, 1       | pi    | A_next=B^FFF, B_next=A_mut |
    | 10     | 1, 0       | 2pi   | A_next=B, B_next=A_mut^FFF |
    | 11     | 1, 1       | 3pi   | A_next=B^FFF, B_next=A_mut^FFF |

    Closure at 4pi (720 degrees) - spinorial return to identity.
    """
    A12 = (state24 >> 12) & 0xFFF
    B12 = state24 & 0xFFF

    intron = byte_val ^ 0xAA

    # 1. Active Operation (Payload -> A mutation)
    mask12, _ = expand_intron_to_mask_a12(intron)
    A_mut = A12 ^ mask12

    # 2. Topological Context (L0 Anchors -> Gyration phase)
    # bit 0 controls whether A_next inverts (pi shift on A)
    # bit 7 controls whether B_next inverts (pi shift on B)
    invert_A = 0xFFF if (intron & 0x01) else 0
    invert_B = 0xFFF if (intron & 0x80) else 0

    # 3. Apply Spinorial Gyration
    A_next = B12 ^ invert_A
    B_next = A_mut ^ invert_B

    return ((A_next & 0xFFF) << 12) | (B_next & 0xFFF)


def expand_intron_combined(intron: int) -> int:
    """
    Combined 14-bit expansion: family (2 bits) + mask (12 bits).
    Returns 14-bit value: (family_idx << 12) | mask12
    This gives 4 x 64 = 256 unique values.
    """
    mask12, family_idx = expand_intron_to_mask_a12(intron)
    return (family_idx << 12) | mask12


def test_dipole_flip_property() -> Dict[str, Any]:
    """
    Test: does toggling payload bit i flip exactly one pair in the 12-bit mask?

    The 12-bit mask has 6 pairs (2 bits each). Bit positions in mask:
    pairs 0,1,2 = frame0 (bits 0-5), pairs 3,4,5 = frame1 (bits 6-11).

    For each payload bit (1-6), toggle it and check how many mask bits change.
    CLAIM: each payload bit toggles exactly 2 mask bits (one pair).
    """
    # Payload bits are 1,2,3,4,5,6 (0 and 7 are L0 boundary)
    payload_bits = list(range(1, 7))
    results = []

    # Test in family 0 (L0 bits both 0)
    for bit_idx in payload_bits:
        base = 0
        toggled = base ^ (1 << bit_idx)
        mask_base, _ = expand_intron_to_mask_a12(base)
        mask_toggled, _ = expand_intron_to_mask_a12(toggled)
        xor_mask = mask_base ^ mask_toggled
        bits_changed = bin(xor_mask).count("1")

        # Which pair(s) changed?
        pairs_changed = []
        for p in range(6):
            lo, hi = 2 * p, 2 * p + 1
            if (xor_mask >> lo) & 1 or (xor_mask >> hi) & 1:
                pairs_changed.append(p)

        results.append({
            "bit": bit_idx,
            "bits_changed": bits_changed,
            "pairs_changed": pairs_changed,
            "exactly_one_pair": bits_changed == 2 and len(pairs_changed) == 1,
        })

    all_pass = all(r["exactly_one_pair"] for r in results)
    return {
        "results": results,
        "dipole_flip_holds": all_pass,
        "status": "PROVED" if all_pass else "COUNTEREXAMPLE",
    }


def test_mask_uniqueness() -> Dict[str, Any]:
    """
    Test uniqueness at different levels:
    - 12-bit mask alone: 64 unique (6 payload bits)
    - 14-bit combined (family + mask): 256 unique
    """
    masks_12 = {}
    masks_14 = {}
    for intron in range(256):
        mask12, family = expand_intron_to_mask_a12(intron)
        combined = expand_intron_combined(intron)
        masks_12[mask12] = masks_12.get(mask12, 0) + 1
        masks_14[combined] = masks_14.get(combined, 0) + 1

    return {
        "unique_masks_12bit": len(masks_12),
        "unique_masks_14bit": len(masks_14),
        "mask_12bit_expected": 64,
        "mask_14bit_expected": 256,
        "mask_12bit_ok": len(masks_12) == 64,
        "mask_14bit_ok": len(masks_14) == 256,
    }


def test_family_structure() -> Dict[str, Any]:
    """
    Verify family structure: L0 bits (0, 7) partition 256 into 4 families of 64.
    """
    families = {0: [], 1: [], 2: [], 3: []}
    for intron in range(256):
        _, family_idx = expand_intron_to_mask_a12(intron)
        families[family_idx].append(intron)

    sizes = {f: len(v) for f, v in families.items()}
    all_64 = all(s == 64 for s in sizes.values())

    return {
        "family_sizes": sizes,
        "all_64": all_64,
        "status": "PROVED" if all_64 else "COUNTEREXAMPLE",
    }


def test_spinorial_transition_24bit() -> Dict[str, Any]:
    """
    Test single-step 24-bit transition (expected: 128 unique states due to
    projection degeneracy - 24 bits is only half the 48-bit frame).
    """
    ARCHETYPE_STATE24 = 0xAAA555

    phase_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for byte_val in range(256):
        intron = byte_val ^ 0xAA
        family = (intron & 1) | ((intron >> 6) & 2)
        phase_counts[family] += 1

    next_states = {}
    for byte_val in range(256):
        next_state = step_state_spinorial(ARCHETYPE_STATE24, byte_val)
        if next_state not in next_states:
            next_states[next_state] = byte_val

    return {
        "unique_states": len(next_states),
        "expected": 128,
        "phase_counts": phase_counts,
        "note": "24-bit is 2D shadow of 48-bit; degeneracy expected",
    }


def project_4byte_to_48bit(b0: int, b1: int, b2: int, b3: int) -> int:
    """
    Project a 4-byte frame to 48-bit tensor space.
    Each byte produces a 12-bit mask; concatenate them.
    NOTE: This is mask-only (24 bits of information from 4x6 payload bits).
    """
    result = 0
    for i, b in enumerate([b0, b1, b2, b3]):
        intron = b ^ 0xAA
        mask12, _ = expand_intron_to_mask_a12(intron)
        result |= (mask12 << (12 * i))
    return result


def project_4byte_full(b0: int, b1: int, b2: int, b3: int) -> int:
    """
    Full 4-byte projection including family bits.
    Each byte: 12-bit mask (from 6 payload) + 2-bit family = 14 bits
    Total: 4 x 14 = 56 bits (or 4 x 8 = 32 bits raw)

    For complete uniqueness, we use the full 32-bit intron sequence.
    """
    result = 0
    for i, b in enumerate([b0, b1, b2, b3]):
        intron = b ^ 0xAA
        result |= (intron << (8 * i))
    return result


def test_depth4_48bit_uniqueness(sample_size: int = 10000) -> Dict[str, Any]:
    """
    Test depth-4 projection uniqueness at different levels:
    1. 48-bit mask-only (4 × 12-bit from payload): 64^4 = 16M unique max
    2. 32-bit full intron (4 × 8-bit): 256^4 = 4B unique, bijective

    The mask-only projection has collisions (family bits discarded).
    The full intron projection is perfectly bijective.
    """
    import random

    # Test 1: 48-bit mask-only (will have collisions)
    mask_projs: Dict[int, Tuple[int, int, int, int]] = {}
    mask_collisions = 0
    random.seed(42)
    for _ in range(sample_size):
        b0, b1, b2, b3 = [random.randint(0, 255) for _ in range(4)]
        frame = (b0, b1, b2, b3)
        proj = project_4byte_to_48bit(b0, b1, b2, b3)
        if proj in mask_projs:
            mask_collisions += 1
        else:
            mask_projs[proj] = frame

    # Test 2: 32-bit full intron (should be bijective)
    full_projs: Dict[int, Tuple[int, int, int, int]] = {}
    full_collisions = 0
    random.seed(42)
    for _ in range(sample_size):
        b0, b1, b2, b3 = [random.randint(0, 255) for _ in range(4)]
        frame = (b0, b1, b2, b3)
        proj = project_4byte_full(b0, b1, b2, b3)
        if proj in full_projs:
            full_collisions += 1
        else:
            full_projs[proj] = frame

    # Single-byte variation test with full projection
    base = [0xAA, 0x55, 0xAA, 0x55]
    single_unique = 0
    for pos in range(4):
        pos_projs: set[int] = set()
        for b in range(256):
            frame = base.copy()
            frame[pos] = b
            proj = project_4byte_full(frame[0], frame[1], frame[2], frame[3])
            pos_projs.add(proj)
        if len(pos_projs) == 256:
            single_unique += 1

    return {
        "sample_size": sample_size,
        "mask_only_48bit": {
            "unique": len(mask_projs),
            "collisions": mask_collisions,
            "note": "Expected collisions (family bits discarded)",
        },
        "full_32bit": {
            "unique": len(full_projs),
            "collisions": full_collisions,
            "note": "Bijective (full intron preserved)",
        },
        "single_byte_positions_bijective": f"{single_unique}/4",
        "status": "PROVED" if full_collisions == 0 else "COUNTEREXAMPLE",
    }


def analyze_cache_line_mapping() -> Dict[str, Any]:
    """
    Analyze the cache-line address mapping.

    L1 cache line = 64 bytes (512 bits)
    To address 64 items: 6 bits needed (2^6 = 64)

    The 8-bit intron maps directly to cache addressing:
    - 6 payload bits (1-6) = cache line offset (which of 64 bytes)
    - 2 boundary bits (0, 7) = cache tag / page selector (which of 4 lines)

    4 families x 64 bytes = 256 bytes = full alphabet horizon
    """
    CACHE_LINE_BYTES = 64
    CACHE_LINE_BITS = 512
    OFFSET_BITS = 6  # log2(64) = 6
    TAG_BITS = 2  # 4 cache lines

    # Verify the mapping
    total_addressable = (2 ** TAG_BITS) * (2 ** OFFSET_BITS)

    return {
        "cache_line_bytes": CACHE_LINE_BYTES,
        "cache_line_bits": CACHE_LINE_BITS,
        "offset_bits": OFFSET_BITS,
        "tag_bits": TAG_BITS,
        "total_addressable": total_addressable,
        "matches_alphabet": total_addressable == 256,
        "mapping": {
            "payload_bits_1_6": "cache line offset (6 bits -> 64 transformations)",
            "boundary_bits_0_7": "cache tag (2 bits -> 4 families/lines)",
        },
        "insight": "Intron is literal L1 cache address: [Family][Payload]",
    }


@dataclass
class CGMByteConstants:
    """CGM constants for byte formalism calculations."""

    delta_BU: float = 0.195342176580  # Measured from tw_closure_test.py
    m_a: float = 1.0 / (2.0 * math.sqrt(2.0 * math.pi))

    @property
    def rho(self) -> float:
        return self.delta_BU / self.m_a

    @property
    def Delta(self) -> float:
        return 1.0 - self.rho


def quantize(x: float, N: int) -> Tuple[float, int]:
    """
    Quantize x in [0,1] at resolution N.
    Returns (Q_N(x), numerator) where Q_N(x) = round(N*x)/N.
    """
    n = round(N * x)
    return n / N, n


def analyze_aperture_quantization(constants: CGMByteConstants) -> Dict[str, Any]:
    """
    Analyze Delta (aperture gap) quantization at various resolutions.
    Tests whether 5/256 is the best 8-bit quantization of Delta.
    """
    Delta = constants.Delta
    rho = constants.rho

    results = {}

    # 8-bit (256) quantization
    Q256_Delta, num_256 = quantize(Delta, 256)
    err_256 = Delta - Q256_Delta
    results["Delta"] = Delta
    results["Q_256_Delta"] = Q256_Delta
    results["Q_256_numerator"] = num_256
    results["Q_256_error"] = err_256
    results["5_over_256"] = 5 / 256
    results["is_5_over_256"] = num_256 == 5

    # 12-bit (4096) quantization
    Q4096_Delta, num_4096 = quantize(Delta, 4096)
    err_4096 = Delta - Q4096_Delta
    results["Q_4096_Delta"] = Q4096_Delta
    results["Q_4096_numerator"] = num_4096
    results["Q_4096_error"] = err_4096

    # 48 quantization (CGM geometric unit)
    Q48_Delta, num_48 = quantize(Delta, 48)
    err_48 = Delta - Q48_Delta
    results["Q_48_Delta"] = Q48_Delta
    results["Q_48_numerator"] = num_48
    results["Q_48_error"] = err_48
    results["1_over_48"] = 1 / 48
    results["48_Delta_exact"] = 48 * Delta

    # rho quantization (closure)
    Q256_rho, num_rho = quantize(rho, 256)
    results["rho"] = rho
    results["Q_256_rho"] = Q256_rho
    results["Q_256_rho_numerator"] = num_rho
    results["251_over_256"] = 251 / 256

    return results


def analyze_tick_spaces(constants: CGMByteConstants) -> Dict[str, Any]:
    """
    Explicit tick spaces - must not be conflated.

    T_256^(frac): fraction line for dimensionless ratios (Delta, rho)
    T_256^(turn): circle for angles normalized by 2*pi (delta_BU)

    The '10' relation (delta_BU=50/256 rad vs Delta=5/256) uses implicit
    '1 rad = 256 ticks' which is NOT canonical. Canonical angle scale is
    full-turn: ticks = round(256 * delta_BU/(2*pi)).
    """
    tau = constants.delta_BU / (2 * math.pi)
    Q256_tau, num_tau = quantize(tau, 256)
    pi_16 = math.pi / 16
    err_pi16 = constants.delta_BU - pi_16

    # T_256^(frac): Delta = 5 ticks
    Delta_q_frac = 5 / 256
    # T_256^(turn): delta_BU = 8 ticks (1/32 turn)
    tau_q_turn = num_tau / 256  # 8/256 = 1/32

    return {
        "delta_BU": constants.delta_BU,
        "tau": tau,
        "Q_256_tau": Q256_tau,
        "Q_256_tau_numerator": num_tau,
        "pi_over_16": pi_16,
        "delta_BU_minus_pi_16": err_pi16,
        "Delta_q_frac": Delta_q_frac,
        "tau_q_turn": tau_q_turn,
        "one_32_turn": 1 / 32,
        "one_48": 1 / 48,
    }


def analyze_32_48_ratio(constants: CGMByteConstants) -> Dict[str, Any]:
    """
    Canonically justified (32, 48) pair:
    - delta_BU ~ 1/32 turn (8 ticks on T_256^(turn))
    - Delta ~ 1/48 (depth-4 projection scale)

    Ratio (1/48)/(1/32) = 2/3 = the 2x3 skeleton of the manifold
    (two chirality layers, three axes).
    """
    tau = constants.delta_BU / (2 * math.pi)
    one_32 = 1 / 32
    one_48 = 1 / 48
    ratio_2_3 = one_48 / one_32

    return {
        "tau": tau,
        "one_32_turn": one_32,
        "one_48": one_48,
        "ratio_2_3": ratio_2_3,
        "tau_minus_1_32": tau - one_32,
        "Delta": constants.Delta,
        "Delta_minus_1_48": constants.Delta - one_48,
    }


def analyze_cross_scale_coherence(constants: CGMByteConstants) -> Dict[str, Any]:
    """
    48 * (5/256) = 0.9375; round(48 * 5/256) = 1.
    Structural compatibility: 5 ticks at 256-scale rounds to 1 tick at 48-scale.
    """
    product = 48 * (5 / 256)
    rounded = round(product)
    return {
        "product": product,
        "rounded": rounded,
        "coherent": rounded == 1,
    }


def analyze_48_vs_256(constants: CGMByteConstants) -> Dict[str, Any]:
    """
    Compare CGM 48-quantization with Router 48-bit projection.
    48 = 4 bytes x 12 bits (depth-4 closure projection).
    """
    Delta = constants.Delta
    return {
        "48_Delta": 48 * Delta,
        "48_Delta_target": 1.0,
        "48_deviation": abs(48 * Delta - 1.0),
        "256_Delta": 256 * Delta,
        "round_256_Delta": round(256 * Delta),
        "depth_4_projection_bits": DEPTH_4_PROJECTION_BITS,
        "depth_4_atom_bits": DEPTH_4_ATOM_BITS,
        "48_as_4x12": f"4 bytes x {BITS_PER_PROJECTION} bits = {DEPTH_4_PROJECTION_BITS}",
    }


def analyze_horizon_types() -> Dict[str, Any]:
    """
    Horizon Lemma (Arithmetic)

    Consider sizes n = 2^a * 3^b with a, b nonnegative integers.

    - log2(n) = a + b*log2(3). Since log2(3) is irrational, log2(n) is
      an integer iff b = 0 (i.e., n is a pure power of two).

    - Primary (dyadic) horizons: n = 2^a (b=0)
      16, 32, 64, 128, 256, 512, 1024, 2048, 4096, ...

    - Predecessor (triadic) horizons: P_k = 3*2^(k-1) = (3/4)*2^(k+1) (b=1)
      For k >= 1: 2^k < P_k < 2^(k+1)
      48, 96, 384, 768, 1536, 3072, ...

    The P_k are the maximal 2^a*3 sizes below each dyadic horizon.

    Byte-formalism note (micro-only):
    The intron's palindromic 4-pair partition (L0/LI/FG/BG) separates into
    1 boundary pair (L0) and 3 interior pairs (LI/FG/BG). When scaling
    structures that preserve this 3+1 split while staying dyadic-aligned,
    the 3*2^k horizons naturally appear as "largest interior block" sizes
    that fit below 4*2^k = 2^(k+2) containers.
    """
    log2_3 = math.log2(3)

    # Paired horizons: (predecessor P_k, dyadic 2^(k+1))
    paired_horizons = [
        # (P_k, 2^(k+1), k)
        (48, 64, 4),    # 3*2^4 = 48, 2^6 = 64
        (96, 128, 5),   # 3*2^5 = 96, 2^7 = 128
        (384, 512, 7),  # 3*2^7 = 384, 2^9 = 512
        (768, 1024, 8), # 3*2^8 = 768, 2^10 = 1024
    ]

    # Full horizon table with roles
    horizons = [
        (12, 2, 1, "3*4 bits (projection unit)"),
        (16, 4, 0, "2^4"),
        (32, 5, 0, "2^5"),
        (48, 4, 1, "P_4 = 3*16 = (3/4)*64"),
        (64, 6, 0, "2^6 (4 families * 16)"),
        (96, 5, 1, "P_5 = 3*32 = (3/4)*128"),
        (128, 7, 0, "2^7 (depth-4 atoms: 4*32)"),
        (256, 8, 0, "2^8 (byte horizon)"),
        (384, 7, 1, "P_7 = 3*128 = (3/4)*512"),
        (512, 9, 0, "2^9 (cache line: 64 bytes)"),
        (768, 8, 1, "P_8 = 3*256 = (3/4)*1024"),
        (1024, 10, 0, "2^10"),
        (4096, 12, 0, "2^12 (12-bit mask)"),
    ]

    results = []
    for n, a, b, role in horizons:
        log2_n = a + b * log2_3
        is_dyadic = b == 0
        htype = "dyadic" if is_dyadic else "predecessor"
        results.append({
            "n": n,
            "form": f"2^{a} x 3^{b}" if b > 0 else f"2^{a}",
            "log2_n": log2_n,
            "horizon_type": htype,
            "role": role,
        })

    # Verify the lemma: P_k = 3*2^(k-1) = (3/4)*2^(k+1)
    lemma_checks = []
    for P_k, dyadic, k in paired_horizons:
        computed_P = 3 * (2 ** (k - 1))
        ratio_check = P_k / dyadic
        lemma_checks.append({
            "k": k,
            "P_k": P_k,
            "dyadic": dyadic,
            "3*2^(k-1)": computed_P,
            "P_k/dyadic": ratio_check,
            "is_3/4": abs(ratio_check - 0.75) < 1e-10,
        })

    return {
        "log2_3": log2_3,
        "horizons": results,
        "paired_horizons": paired_horizons,
        "lemma_checks": lemma_checks,
        "lemma": "P_k = 3*2^(k-1) = (3/4)*2^(k+1) sits just below 2^(k+1)",
        "byte_note": "3+1 split (1 boundary L0, 3 interior LI/FG/BG) -> 3*2^k horizons",
    }


def analyze_router_architecture() -> Dict[str, Any]:
    """
    Router architecture: depth-4 closure, bit pairs, families, 6 DoF.
    """
    return {
        "gene_mic": GENE_MIC_S,
        "l0_mask": L0_MASK,
        "li_mask": LI_MASK,
        "fg_mask": FG_MASK,
        "bg_mask": BG_MASK,
        "depth_4_projection_bits": DEPTH_4_PROJECTION_BITS,
        "depth_4_atom_bits": DEPTH_4_ATOM_BITS,
        "families": FAMILIES,
        "members_per_family": MEMBERS_PER_FAMILY,
        "introns_total": INTRONS_TOTAL,
        "stage_mapping": {
            "Prefix": "CS",
            "Present": "UNA",
            "Past": "ONA",
            "Future": "BU",
        },
    }


def run_analysis(verbose: bool = True) -> Dict[str, Any]:
    """Run full byte formalism analysis."""
    c = CGMByteConstants()
    arch = analyze_router_architecture()

    if verbose:
        print("CGM BYTE FORMALISM ANALYSIS")
        print("-" * 40)
        print("\n0. ROUTER ARCHITECTURE (depth-4 closure)")
        print("-" * 40)
        print("   Depth-4: 4 components (bytes or 12-bit tensors) always known")
        print(f"   4-byte frame -> 4 x 12 = {arch['depth_4_projection_bits']} bits (CGM 48)")
        print("   Prefix(CS) | Present(UNA) | Past(ONA) | Future(BU)")
        print(f"   Bit pairs: L0={hex(arch['l0_mask'])} LI={hex(arch['li_mask'])} FG={hex(arch['fg_mask'])} BG={hex(arch['bg_mask'])}")
        print(f"   Families: L0 bits (0,7) -> {arch['families']} families x {arch['members_per_family']} = {arch['introns_total']}")
        print("   6 payload bits (1-6) -> 6 DoF (each bit controls one pair)")
        print("\n1. EXACT DEFINITIONS (from CGM construction)")
        print(f"   delta_BU = {c.delta_BU:.12f} rad")
        print(f"   m_a = 1/(2*sqrt(2*pi)) = {c.m_a:.12f}")
        print(f"   rho = delta_BU/m_a = {c.rho:.12f}")
        print(f"   Delta = 1 - rho = {c.Delta:.12f}")
        print(f"   (2.07%% is rounded percentage of Delta)")

    aq = analyze_aperture_quantization(c)
    if verbose:
        print("\n2. APERTURE GAP (Delta) QUANTIZATION")
        print("-" * 40)
        print(f"   256 * Delta = {256 * c.Delta:.10f}")
        print(f"   round(256 * Delta) = {aq['Q_256_numerator']}")
        print(f"   Q_256(Delta) = {aq['Q_256_numerator']}/256 = {aq['Q_256_Delta']:.10f}")
        print(f"   5/256 = {aq['5_over_256']:.10f}")
        print(f"   Is 5/256 the best 8-bit quantization? {aq['is_5_over_256']}")
        print(f"   Quantization error: Delta - Q_256(Delta) = {aq['Q_256_error']:.12f}")
        print(f"\n   For rho: Q_256(rho) = {aq['Q_256_rho_numerator']}/256 = {aq['Q_256_rho']:.10f}")
        print(f"   251/256 = {aq['251_over_256']:.10f}")
        print(f"\n   12-bit (4096): Q_4096(Delta) = {aq['Q_4096_numerator']}/4096 = {aq['Q_4096_Delta']:.12f}")
        print(f"   Error: {aq['Q_4096_error']:.2e}")

    ticks = analyze_tick_spaces(c)
    r32_48 = analyze_32_48_ratio(c)
    coherence = analyze_cross_scale_coherence(c)
    dipole = test_dipole_flip_property()

    if verbose:
        print("\n3. TICK SPACES (must not be conflated)")
        print("-" * 40)
        print("   T_256^(frac): fraction line for Delta, rho (dimensionless)")
        print("   T_256^(turn): circle for angles, normalized by 2*pi")
        print(f"\n   T_256^(frac): Delta = 5 ticks = 5/256")
        print(f"   T_256^(turn): delta_BU = {ticks['Q_256_tau_numerator']} ticks = {ticks['Q_256_tau_numerator']}/256 = 1/32 turn")
        print(f"   delta_BU - pi/16 = {ticks['delta_BU_minus_pi_16']:.2e}")
        print("\n   WARNING: '10' relation (50/256 rad vs 5/256) uses implicit 1 rad=256 ticks")
        print("   which is NOT canonical. Canonical angle scale is full-turn.")

    if verbose:
        print("\n4. (32, 48) PAIR - dimensionally clean ratio 2/3")
        print("-" * 40)
        print(f"   delta_BU ~ 1/32 turn, Delta ~ 1/48")
        print(f"   Ratio (1/48)/(1/32) = {r32_48['ratio_2_3']:.6f} = 2/3")
        print("   (2x3 skeleton: two chirality layers, three axes)")
        print(f"   tau - 1/32 = {r32_48['tau_minus_1_32']:.2e}")
        print(f"   Delta - 1/48 = {r32_48['Delta_minus_1_48']:.2e}")

    if verbose:
        print("\n5. CROSS-SCALE COHERENCE")
        print("-" * 40)
        print(f"   48 * (5/256) = {coherence['product']:.4f}")
        print(f"   round(48 * 5/256) = {coherence['rounded']}")
        print(f"   5 ticks at 256-scale -> 1 tick at 48-scale: {coherence['coherent']}")

    uniqueness = test_mask_uniqueness()
    family_struct = test_family_structure()
    spinorial_24 = test_spinorial_transition_24bit()
    spinorial_48 = test_depth4_48bit_uniqueness()
    cache = analyze_cache_line_mapping()

    if verbose:
        print("\n6. EXPANSION AND DEPTH-4 CLOSURE")
        print("-" * 40)
        print("   Dipole flip (payload bits 1-6 -> pairs 0-5):")
        print(f"      Status: {dipole['status']}")
        for r in dipole["results"]:
            pair_idx = r["pairs_changed"][0] if len(r["pairs_changed"]) == 1 else "?"
            print(f"        bit {r['bit']} -> pair {pair_idx}")
        print(f"   Mask uniqueness: {uniqueness['unique_masks_12bit']}/64 (6 payload bits)")
        print(f"   Family structure (4x64): {family_struct['status']}")

        print(f"\n   24-BIT SINGLE-STEP (shadow projection):")
        print(f"      Unique states: {spinorial_24['unique_states']}/256 (expected {spinorial_24['expected']})")
        print(f"      Note: {spinorial_24['note']}")

        print(f"\n   DEPTH-4 PROJECTION TESTS:")
        m = spinorial_48['mask_only_48bit']
        f = spinorial_48['full_32bit']
        print(f"      Mask-only (48-bit, 4x12): {m['unique']}/{spinorial_48['sample_size']} unique")
        print(f"         {m['note']}")
        print(f"      Full intron (32-bit, 4x8): {f['unique']}/{spinorial_48['sample_size']} unique")
        print(f"         {f['note']}")
        print(f"      Single-byte bijective positions: {spinorial_48['single_byte_positions_bijective']}")
        print(f"      Overall status: {spinorial_48['status']}")

    if verbose:
        print("\n7. CACHE-LINE HARDWARE MAPPING")
        print("-" * 40)
        print(f"   L1 cache line: {cache['cache_line_bytes']} bytes ({cache['cache_line_bits']} bits)")
        print(f"   Offset bits needed: {cache['offset_bits']} (2^6 = 64)")
        print(f"   Tag bits: {cache['tag_bits']} (4 cache lines)")
        print(f"   Total addressable: {cache['total_addressable']} = alphabet horizon")
        print(f"\n   INTRON AS NATIVE CACHE ADDRESS:")
        print(f"      Bits 1-6 (payload): {cache['mapping']['payload_bits_1_6']}")
        print(f"      Bits 0,7 (boundary): {cache['mapping']['boundary_bits_0_7']}")
        print(f"   {cache['insight']}")

    q48 = analyze_48_vs_256(c)
    if verbose:
        print("\n8. 48-BIT PROJECTION AND TWO DEPTH-4 OBJECTS")
        print("-" * 40)
        print(f"   Depth-4 projection: 4 x 12 = {q48['depth_4_projection_bits']} bits (manifold)")
        print(f"   Depth-4 atoms: 4 x 32 = {q48['depth_4_atom_bits']} bits (execution)")
        print(f"   CGM: 48*Delta = 1 (geometric quantization)")
        print(f"   Measured: 48*Delta = {q48['48_Delta']:.10f}")
        print(f"   Deviation from 1: {q48['48_deviation']:.2e}")
        print(f"   Byte: 256 = 4 families x 64; 256*Delta -> numerator {q48['round_256_Delta']}")

    horiz = analyze_horizon_types()
    if verbose:
        print("\n9. HORIZON LEMMA (n = 2^a * 3^b)")
        print("-" * 40)
        print(f"   log2(3) = {horiz['log2_3']:.6f} (irrational)")
        print("   log2(n) integer iff b=0 (pure power of two)")
        print()
        print("   Dyadic (b=0): 16, 32, 64, 128, 256, 512, ...")
        print("   Predecessor (b=1): P_k = 3*2^(k-1) = (3/4)*2^(k+1)")
        print()
        print("   | n    | Form      | log2(n) | Type        | Role")
        print("   |------|-----------|---------|-------------|-----")
        for h in horiz["horizons"]:
            print(f"   | {h['n']:4d} | {h['form']:9s} | {h['log2_n']:7.3f} | {h['horizon_type']:11s} | {h['role']}")
        print(f"\n   Lemma: {horiz['lemma']}")
        print(f"   Byte: {horiz['byte_note']}")

    return {
        "architecture": arch,
        "aperture_quantization": aq,
        "tick_spaces": ticks,
        "32_48_ratio": r32_48,
        "cross_scale_coherence": coherence,
        "dipole_flip": dipole,
        "mask_uniqueness": uniqueness,
        "family_structure": family_struct,
        "spinorial_24bit": spinorial_24,
        "spinorial_48bit": spinorial_48,
        "cache_line_mapping": cache,
        "48_vs_256": q48,
        "horizon_types": horiz,
        "constants": {"delta_BU": c.delta_BU, "m_a": c.m_a, "rho": c.rho, "Delta": c.Delta},
    }


def main():
    results = run_analysis(verbose=True)
    print("\n" + "-" * 40)
    print("SUMMARY")
    print("-" * 40)
    aq = results["aperture_quantization"]
    print(f"5/256 is best 8-bit quantization of Delta: {aq['is_5_over_256']}")
    print(f"Quantization error at 8-bit: {aq['Q_256_error']:.12f}")

    print("\nEXPANSION & DEPTH-4 CLOSURE:")
    print(f"- Dipole flip: {results['dipole_flip']['status']}")
    print(f"- 12-bit mask: {results['mask_uniqueness']['unique_masks_12bit']}/64 unique")
    print(f"- Family structure (4x64): {results['family_structure']['status']}")

    print("\nDEPTH-4 CLOSURE:")
    print(f"- 24-bit single-step: {results['spinorial_24bit']['unique_states']}/256 (shadow projection)")
    print(f"- 32-bit full 4-frame: {results['spinorial_48bit']['status']} (bijective)")
    print("- 720-degree closure operates across 4-byte frame")
    print("- Family bits encode spinorial phase, resolved over trajectory")

    print("\nCACHE-LINE MAPPING:")
    print("- Bits 1-6 (payload) = cache line offset (64 transformations)")
    print("- Bits 0,7 (boundary) = cache tag (4 families)")
    print("- Intron is literal L1 cache address: [Family][Payload]")
    return results


if __name__ == "__main__":
    main()
