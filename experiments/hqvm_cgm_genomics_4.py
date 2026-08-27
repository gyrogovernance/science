#!/usr/bin/env python3
"""
hqvm_cgm_genomics_4.py

Sections 7–11: stage palindrome through QuBEC dynamics.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from gyroscopic.hQVM.api import (
    OmegaSignature12,
    omega_word_signature,
    q_word6,
    q_word6_for_items,
)
from gyroscopic.hQVM.constants import (
    APERTURE_GAP,
    APERTURE_GAP_Q256,
    BU_HOLONOMY_ANGLE,
    M_A,
    RHO,
)
from hqvm_cgm_genomics_1 import (
    H34_MULTIPLICITIES,
    block_weight6,
    cumulative_S,
    ncbi_chart_s2_grid,
    rc_byte_keep_family,
    synonymous_homology,
    walsh_degree_energy,
    walsh_s2,
)
from hqvm_cgm_genomics_2 import load_named_fasta, orbit_reps
from hqvm_cgm_genomics_3 import (
    _product_words,
    payload_rc_star,
    theta_payload_rc,
)
from hqvm_cgm_genomics_common import (
    ANTIPODE_6,
    CODONS,
    NCBI_TABLE_IDS,
    NULL_SEED,
    ORBIT_ELEMENTARY,
    ORBIT_OTHER,
    ORBIT_PAIR_INV,
    STANDARD_CODE,
    STRONG,
    WC,
    NucleotideEncoding,
    affine_rank6,
    all_nucleotide_encodings,
    block_reverse6,
    encoding_orbit_name,
    extract_chr22_cds,
    encodings_in_orbit,
    fiber_components,
    gf2_rank6,
    fibers,
    in_W,
    iter_codons,
    one_base_neighbors,
    pack_codon_bits,
    print_table,
    report_checks,
    report_objects,
    report_section,
    reverse_complement_4mer_byte,
    translation_table,
)

KYTE_DOOLITTLE: Dict[str, float] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

RUMER_ROOT: Dict[str, str] = {"A": "C", "C": "A", "G": "T", "T": "G"}

STOP_SQUARE: Tuple[str, ...] = ("TAA", "TAG", "TGA", "TGG")

def _half_rapidity_k(beta: float) -> float:
    b = float(beta)
    return b / (1.0 + math.sqrt(1.0 - b * b))

def delta_bu_closed_form() -> float:
    return 4.0 * math.atan(_half_rapidity_k(math.pi / 4.0) * _half_rapidity_k(float(M_A)))

def gate_S(sig: OmegaSignature12) -> OmegaSignature12:
    return OmegaSignature12(sig.parity, sig.tau_v6, sig.tau_u6)

def gate_F(sig: OmegaSignature12) -> OmegaSignature12:
    return OmegaSignature12(sig.parity, sig.tau_u6 ^ ANTIPODE_6, sig.tau_v6 ^ ANTIPODE_6)

def gate_R(sig: OmegaSignature12) -> OmegaSignature12:
    return OmegaSignature12(sig.parity, block_reverse6(sig.tau_u6), block_reverse6(sig.tau_v6))

@dataclass
class StagePalindromeCensus:
    delta_bu: float
    rho: float
    aperture: float
    q256: int
    theta_is_SFR: int
    theta_total: int
    pair_payload_pole: int
    pair_kin_pole: int
    elem_payload_pole: int
    elem_kin_pole: int
    odd_shell_flip: Tuple[int, int]
    even_shell_keep: Tuple[int, int]
    gates: Dict[str, bool]

def stage_palindrome_census() -> StagePalindromeCensus:
    pair = encodings_in_orbit(ORBIT_PAIR_INV)[0][1]
    elem = encodings_in_orbit(ORBIT_ELEMENTARY)[0][1]

    n_sfr = 0
    for b in range(256):
        s = omega_word_signature([b])
        if theta_payload_rc(s) == gate_S(gate_F(gate_R(s))):
            n_sfr += 1

    def pole_count(enc: NucleotideEncoding, kin: bool) -> int:
        n = 0
        for b in range(256):
            q = q_word6(b)
            rb = reverse_complement_4mer_byte(b, enc) if kin else rc_byte_keep_family(b, enc)
            if q_word6(rb).bit_count() == 6 - q.bit_count():
                n += 1
        return n

    odd_ok = odd_n = even_ok = even_n = 0
    for n, stride in ((1, 1), (2, 8), (3, 16), (4, 32)):
        for w in _product_words(n, stride):
            q = q_word6_for_items(w)
            qr = q_word6_for_items(payload_rc_star(w, pair))
            if n % 2:
                odd_n += 1
                if qr.bit_count() == 6 - q.bit_count():
                    odd_ok += 1
            else:
                even_n += 1
                if qr.bit_count() == q.bit_count():
                    even_ok += 1

    pair_pay = pole_count(pair, False)
    pair_kin = pole_count(pair, True)
    elem_pay = pole_count(elem, False)
    elem_kin = pole_count(elem, True)
    delta = delta_bu_closed_form()
    gates = {
        "80_theta_is_SFR": n_sfr == 256,
        "81_pair_payload_is_W2": pair_pay == 256,
        "82_elem_payload_not_W2": elem_pay == 0,
        "83_odd_word_shell_flip": odd_ok == odd_n and odd_n > 0,
        "84_even_word_shell_keep": even_ok == even_n and even_n > 0,
        "85_delta_bu_closed_form": abs(delta - float(BU_HOLONOMY_ANGLE)) < 1e-15,
        "86_q256_is_5": APERTURE_GAP_Q256 == 5,
        "87_kin_not_the_pole": pair_kin < 256,
    }
    return StagePalindromeCensus(
        delta_bu=delta,
        rho=float(RHO),
        aperture=float(APERTURE_GAP),
        q256=int(APERTURE_GAP_Q256),
        theta_is_SFR=n_sfr,
        theta_total=256,
        pair_payload_pole=pair_pay,
        pair_kin_pole=pair_kin,
        elem_payload_pole=elem_pay,
        elem_kin_pole=elem_kin,
        odd_shell_flip=(odd_ok, odd_n),
        even_shell_keep=(even_ok, even_n),
        gates=gates,
    )

def print_stage_palindrome_census(c: StagePalindromeCensus) -> None:
    g = c.gates
    report_section("7. Theta = S o F o R_block; CHIRALITY POLE MAP")
    report_objects(
        (
            "Theta = S o F o R_block; pair_inversion pole map; Z_n; delta_BU, Q_256",
        )
    )
    print("  continuous constants")
    print(f"    delta_BU={c.delta_bu:.16f}  rho={c.rho:.16f}  Delta={c.aperture:.16f}  Q_256= {c.q256}/256")
    print()
    print("  Theta factorization")
    print(f"    Theta = S o F o R_block on letters: {c.theta_is_SFR}/{c.theta_total}")
    print()
    print("  chirality pole exchange s |-> 6-s")
    print(f"    pair payload-RC  {c.pair_payload_pole}/256")
    print(f"    pair kinematic   {c.pair_kin_pole}/256")
    print(f"    elem payload-RC  {c.elem_payload_pole}/256")
    print(f"    elem kinematic   {c.elem_kin_pole}/256")
    print()
    print("  word palindrome on XOR chirality (pair, payload-RC*)")
    print(f"    odd  n shell flip   {c.odd_shell_flip[0]}/{c.odd_shell_flip[1]}")
    print(f"    even n shell keep   {c.even_shell_keep[0]}/{c.even_shell_keep[1]}")
    print()
    report_checks((
        ('Theta is the stage palindrome S o F o R_block', g['80_theta_is_SFR'], f'{c.theta_is_SFR}/{c.theta_total}', '256/256'),
        ('pair_inversion payload-RC is the chirality pole map s |-> 6-s', g['81_pair_payload_is_W2'], f'{c.pair_payload_pole}/256', '256/256'),
        ('elementary payload-RC chirality-pole matches', g['82_elem_payload_not_W2'], f'{c.elem_payload_pole}/256', '0/256'),
        ('odd-length payload-RC* inverts XOR-chirality shell', g['83_odd_word_shell_flip'], f'{c.odd_shell_flip[0]}/{c.odd_shell_flip[1]}', 'all sampled'),
        ('even-length payload-RC* preserves XOR-chirality shell', g['84_even_word_shell_keep'], f'{c.even_shell_keep[0]}/{c.even_shell_keep[1]}', 'all sampled'),
        ('delta_BU equals kernel closed form 4 arctan(k(pi/4) k(m_a))', g['85_delta_bu_closed_form'], f'{c.delta_bu:.16f}', 'BU_HOLONOMY_ANGLE'),
        ('byte-horizon aperture is 5/256', g['86_q256_is_5'], f'{c.q256}/256', '5/256'),
        ('kinematic 4-mer RC chirality-pole matches on pair_inversion', g['87_kin_not_the_pole'], f'pair kin {c.pair_kin_pole}/256', '< 256'),
    ))

BYTE_STAGES: Tuple[str, ...] = ("CS", "UNA", "ONA", "BU", "BU", "ONA", "UNA", "CS")

CODON_PAYLOAD_BITS: Tuple[Tuple[int, int], ...] = (
    (4, 5),  # first
    (2, 3),  # middle = fold plane
    (0, 1),  # wobble
)

SER_TCN: Tuple[str, ...] = ("TCT", "TCC", "TCA", "TCG")

SER_AGY: Tuple[str, ...] = ("AGT", "AGC")

SER_PURINE_TCN: Tuple[str, ...] = ("TCA", "TCG")

OUTER_MASK = 0b110011

FOLD_MASK = 0b001100

def sense_edge_diffs(enc: NucleotideEncoding, code: Dict[str, str] = STANDARD_CODE) -> List[int]:
    out: List[int] = []
    for c in CODONS:
        if code[c] == "*":
            continue
        pc = pack_codon_bits(c, enc)
        for n in one_base_neighbors(c):
            if code[n] == code[c]:
                out.append((pc ^ pack_codon_bits(n, enc)) & 0x3F)
    return out

def stop_diff(enc: NucleotideEncoding) -> int:
    return (pack_codon_bits("TAA", enc) ^ pack_codon_bits("TGA", enc)) & 0x3F

def serine_chords(enc: NucleotideEncoding) -> List[int]:
    return [
        (pack_codon_bits(a, enc) ^ pack_codon_bits(b, enc)) & 0x3F
        for a in SER_TCN
        for b in SER_AGY
    ]

def span_elements(vecs: Sequence[int]) -> List[int]:
    """All linear combinations of a GF(2) generating set (rank ≤ 6)."""
    basis: Dict[int, int] = {}
    for v in vecs:
        x = int(v) & 0x3F
        while x:
            p = x.bit_length() - 1
            if p in basis:
                x ^= basis[p]
            else:
                basis[p] = x
                break
    keys = sorted(basis.keys(), reverse=True)
    cols = [basis[k] for k in keys]
    out = [0]
    for col in cols:
        out.extend([u ^ col for u in out])
    return out

def fold_intersection(vecs: Sequence[int]) -> Tuple[int, ...]:
    """Nonzero elements of span(vecs) that lie in P_fold."""
    hits = sorted({v for v in span_elements(vecs) if v != 0 and (v & ~FOLD_MASK) == 0})
    return tuple(hits)

def sense_component_sizes(code: Dict[str, str] = STANDARD_CODE) -> Tuple[int, ...]:
    fib = fibers(code)
    sizes: List[int] = []
    for aa, group in fib.items():
        if aa == "*":
            continue
        for comp in fiber_components(group):
            sizes.append(len(comp))
    return tuple(sorted(sizes, reverse=True))

@dataclass
class ChartWallRow:
    orbit: str
    rank_sense: int
    rank_stop: int
    rank_ser: int
    rank_full: int
    fold_sense: Tuple[int, ...]
    fold_stop: Tuple[int, ...]
    fold_ser: Tuple[int, ...]
    fold_full: Tuple[int, ...]
    n_ser_chords: int
    n_ser_antipode: int
    agy_is_antipode_of_purine_tcn: bool

@dataclass
class WallTransportCensus:
    rows: Tuple[ChartWallRow, ...]
    stage_map: Tuple[Tuple[str, Tuple[int, int], Tuple[str, str]], ...]
    component_sizes: Tuple[int, ...]
    n_sense_components: int
    gates: Dict[str, bool]

def wall_transport_census() -> WallTransportCensus:
    rows: List[ChartWallRow] = []
    for enc in all_nucleotide_encodings():
        orbit = encoding_orbit_name(enc)
        sense = sense_edge_diffs(enc)
        sd = stop_diff(enc)
        ch = serine_chords(enc)
        r_sense = gf2_rank6(sense)
        r_stop = gf2_rank6(sense + [sd])
        r_ser = gf2_rank6(sense + ch)
        r_full = gf2_rank6(sense + [sd] + ch)
        agy = {pack_codon_bits(c, enc) & 0x3F for c in SER_AGY}
        pur_tcn_A = {
            (pack_codon_bits(c, enc) ^ ANTIPODE_6) & 0x3F for c in SER_PURINE_TCN
        }
        rows.append(
            ChartWallRow(
                orbit=orbit,
                rank_sense=r_sense,
                rank_stop=r_stop,
                rank_ser=r_ser,
                rank_full=r_full,
                fold_sense=fold_intersection(sense),
                fold_stop=fold_intersection(sense + [sd]),
                fold_ser=fold_intersection(sense + ch),
                fold_full=fold_intersection(sense + [sd] + ch),
                n_ser_chords=len(ch),
                n_ser_antipode=sum(1 for d in ch if d == ANTIPODE_6),
                agy_is_antipode_of_purine_tcn=(agy == pur_tcn_A),
            )
        )

    stage_map: List[Tuple[str, Tuple[int, int], Tuple[str, str]]] = []
    for pos, (lo, hi) in enumerate(CODON_PAYLOAD_BITS):
        byte_lo, byte_hi = lo + 1, hi + 1
        stage_map.append(
            (
                ("first", "middle", "wobble")[pos],
                (lo, hi),
                (BYTE_STAGES[byte_lo], BYTE_STAGES[byte_hi]),
            )
        )
    stage_map_t = tuple(stage_map)

    comps = sense_component_sizes()
    expect_comps = (6, 6, 4, 4, 4, 4, 4, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1)

    all_sense_closed = all(r.rank_sense == 4 and r.fold_sense == () for r in rows)
    all_stop_dim1 = all(r.rank_stop == 5 and len(r.fold_stop) == 1 for r in rows)
    all_ser_dim1 = all(r.rank_ser == 5 and len(r.fold_ser) == 1 for r in rows)
    all_full = all(
        r.rank_full == 6 and set(r.fold_full) == {0b000100, 0b001000, 0b001100}
        for r in rows
    )

    pair_rows = [r for r in rows if r.orbit == ORBIT_PAIR_INV]
    elem_rows = [r for r in rows if r.orbit == ORBIT_ELEMENTARY]
    other_rows = [r for r in rows if r.orbit == ORBIT_OTHER]
    # On pair_inversion: AGY = {TCA, TCG} XOR antipode (exactly two antipodal pairs).
    pair_agy_pole = all(r.agy_is_antipode_of_purine_tcn for r in pair_rows)
    pair_two_antipode_chords = all(r.n_ser_antipode == 2 for r in pair_rows)
    nonpair_ser_free = all(r.n_ser_antipode == 0 for r in elem_rows + other_rows)
    nonpair_agy_free = all(not r.agy_is_antipode_of_purine_tcn for r in elem_rows + other_rows)

    stage_ok = stage_map_t == (
        ("first", (4, 5), ("ONA", "UNA")),
        ("middle", (2, 3), ("BU", "BU")),
        ("wobble", (0, 1), ("UNA", "ONA")),
    )

    gates = {
        "88_sense_rank4_fold_closed": all_sense_closed and len(rows) == 24,
        "89_stop_opens_fold_dim1": all_stop_dim1,
        "90_serine_opens_fold_dim1": all_ser_dim1,
        "91_joint_fills_fold_plane": all_full,
        "92_pair_agy_equals_purine_tcn_antipode": (
            pair_agy_pole and pair_two_antipode_chords and len(pair_rows) == 8
        ),
        "93_nonpair_serine_pole_free": nonpair_ser_free and nonpair_agy_free,
        "94_codon_stage_anatomy": stage_ok,
        "95_sense_component_census": comps == expect_comps and len(comps) == 21,
    }
    return WallTransportCensus(
        rows=tuple(rows),
        stage_map=stage_map_t,
        component_sizes=comps,
        n_sense_components=len(comps),
        gates=gates,
    )

def print_wall_transport_census(c: WallTransportCensus) -> None:
    g = c.gates
    report_section("8. NESTED WALL-BREACH; STAGE ANATOMY; SERINE ANTIPODE")
    report_objects(
        (
            "L_sense span; P_fold middle plane; stop/serine keys; stage anatomy",
        )
    )
    print("  stage anatomy (payload bit p -> byte bit p+1)")
    for name, (lo, hi), (s0, s1) in c.stage_map:
        print(f"    {name:7s}  payload bits {lo},{hi}  byte stages {s0}|{s1}")
    print()
    print("  nested ranks and fold intersections (24 charts)")
    by_orb: Dict[str, List[ChartWallRow]] = {
        ORBIT_ELEMENTARY: [],
        ORBIT_PAIR_INV: [],
        ORBIT_OTHER: [],
    }
    for r in c.rows:
        by_orb[r.orbit].append(r)
    for orbit, group in by_orb.items():
        ranks = Counter((r.rank_sense, r.rank_stop, r.rank_ser, r.rank_full) for r in group)
        folds = Counter(
            (len(r.fold_sense), len(r.fold_stop), len(r.fold_ser), len(r.fold_full))
            for r in group
        )
        n_anti = sum(r.n_ser_antipode for r in group)
        n_ch = sum(r.n_ser_chords for r in group)
        print(
        f"    {orbit}: charts={len(group)} "
        f"rank_profiles={dict(ranks)} fold_|I|_profiles={dict(folds)} "
        f"ser_antipode_chords={n_anti}/{n_ch} "
        f"agy=purine_tcn⊕A={sum(1 for r in group if r.agy_is_antipode_of_purine_tcn)}/{len(group)}"
    )
    print()
    print("  sense-component sizes (connectivity governs drift)")
    print(f"    n_components={c.n_sense_components}  sizes={c.component_sizes}")
    print()
    report_checks((
        ('sense span rank 4 and fold plane closed on all 24 charts', g['88_sense_rank4_fold_closed'], 'rank=4, |I_fold|=0 on 24/24', '4, empty intersection'),
        ('stop opens exactly dim-1 of the fold plane (rank 5)', g['89_stop_opens_fold_dim1'], 'rank=5, |I_fold|=1 on 24/24', '5, one nonzero fold vector'),
        ('serine opens exactly dim-1 of the fold plane (rank 5)', g['90_serine_opens_fold_dim1'], 'rank=5, |I_fold|=1 on 24/24', '5, one nonzero fold vector'),
        ('stop+serine jointly fill the fold plane (rank 6)', g['91_joint_fills_fold_plane'], 'rank=6, I_fold=P_fold\\{0} on 24/24', '6, three nonzero fold vectors'),
        ('pair_inversion: AGY = {TCA,TCG} XOR antipode (2 antipodal pairs)', g['92_pair_agy_equals_purine_tcn_antipode'], f'agy_match={sum((1 for r in c.rows if r.orbit == ORBIT_PAIR_INV and r.agy_is_antipode_of_purine_tcn))}/8; antipode_chords={sum((r.n_ser_antipode for r in c.rows if r.orbit == ORBIT_PAIR_INV))}/64', '8/8 charts; exactly 2 antipodal chords per chart'),
        ('elementary/other: serine split free of the antipodal AGY identity', g['93_nonpair_serine_pole_free'], '0 antipode chords and 0 AGY=purine_tcn⊕A on 16 charts', '0/16'),
        ('codon positions occupy the three interior stage-pairs of the byte', g['94_codon_stage_anatomy'], str(c.stage_map), 'first=ONA|UNA, middle=BU|BU, wobble=UNA|ONA'),
        ('sense synonymous-graph component census', g['95_sense_component_census'], f'n={c.n_sense_components} sizes={c.component_sizes}', '21 components; (6,6,4^6,3,2^10,1,1)'),
    ))

def hydropathy_eta2(pos: int, code: Dict[str, str]) -> float:
    vals = [(c[pos], KYTE_DOOLITTLE[code[c]]) for c in CODONS if code[c] != "*"]
    n = len(vals)
    mean = sum(v for _, v in vals) / n
    ss_tot = sum((v - mean) ** 2 for _, v in vals)
    groups: Dict[str, List[float]] = {}
    for base, v in vals:
        groups.setdefault(base, []).append(v)
    ss_between = sum(len(g) * (sum(g) / len(g) - mean) ** 2 for g in groups.values())
    return ss_between / ss_tot if ss_tot else 0.0

def doublet_boxes(code: Dict[str, str] = STANDARD_CODE) -> Dict[str, Tuple[str, ...]]:
    boxes: Dict[str, List[str]] = defaultdict(list)
    for c in CODONS:
        boxes[c[:2]].append(c)
    return {d: tuple(sorted(ms)) for d, ms in boxes.items()}

def complete_boxes(code: Dict[str, str] = STANDARD_CODE) -> Tuple[str, ...]:
    boxes = doublet_boxes(code)
    out = []
    for d, ms in boxes.items():
        labels = {code[c] for c in ms}
        if len(labels) == 1 and len(ms) == 4 and "*" not in labels:
            out.append(d)
    return tuple(sorted(out))

def max_degeneracy(doublet: str, code: Dict[str, str] = STANDARD_CODE) -> int:
    boxes = doublet_boxes(code)
    return max(Counter(code[c] for c in boxes[doublet]).values())

def gc_strength(doublet: str) -> str:
    n = sum(1 for b in doublet if b in "GC")
    if n == 2:
        return "strong"
    if n == 0:
        return "weak"
    return "mixed"

def rumer_map(doublet: str) -> str:
    return RUMER_ROOT[doublet[0]] + RUMER_ROOT[doublet[1]]

def wc_rc_doublet(doublet: str) -> str:
    return WC[doublet[1]] + WC[doublet[0]]

@dataclass
class IdentityAxesCensus:
    eta2_std: Tuple[float, float, float]
    eta2_tables: Tuple[Tuple[int, float, float, float], ...]
    n_tables_mid_dominates: int
    n_tables: int
    complete: Tuple[str, ...]
    rumer_complete_to_incomplete: int
    wc_strength_preserved: int
    wc_complete_to_complete: int
    stop_aff_rank: int
    stop_square_size: int
    trp_is_corner: bool
    pair_stop_square_affine: bool
    triplet_address: Tuple[int, int, int, int]
    gates: Dict[str, bool]

def identity_axes_census() -> IdentityAxesCensus:
    eta_std = (
        hydropathy_eta2(0, STANDARD_CODE),
        hydropathy_eta2(1, STANDARD_CODE),
        hydropathy_eta2(2, STANDARD_CODE),
    )
    table_rows: List[Tuple[int, float, float, float]] = []
    n_dom = 0
    for tid in NCBI_TABLE_IDS:
        code = translation_table(tid)
        e0, e1, e2 = hydropathy_eta2(0, code), hydropathy_eta2(1, code), hydropathy_eta2(2, code)
        table_rows.append((tid, e0, e1, e2))
        if e1 > e0 and e1 > e2:
            n_dom += 1

    complete = complete_boxes()
    boxes = doublet_boxes()
    rumer_out = sum(1 for d in complete if rumer_map(d) not in complete)
    wc_preserved = sum(
        1 for d in boxes if gc_strength(d) == gc_strength(wc_rc_doublet(d))
    )
    wc_keep = sum(1 for d in complete if wc_rc_doublet(d) in complete)

    # Stop square geometry on pair_inversion (affine ranks are chart-covariant
    # up to the WC polarity; we measure on the physical chart).
    pair = encodings_in_orbit(ORBIT_PAIR_INV)[0][1]
    sq_bits = [pack_codon_bits(c, pair) for c in STOP_SQUARE]
    stop_bits = [pack_codon_bits(c, pair) for c in ("TAA", "TAG", "TGA")]
    aff_sq = affine_rank6(sq_bits)
    aff_stop = affine_rank6(stop_bits)
    trp_corner = STANDARD_CODE["TGG"] == "W" and set(STOP_SQUARE[:3]) == {"TAA", "TAG", "TGA"}

    # Triplet addressability (pure arithmetic of the lift).
    n_labels = 21  # 20 AA + stop
    one_letter = 4
    two_letter = 16
    three_letter = 64
    payload_bits = 6

    gates = {
        "96_std_middle_hydropathy": (
            eta_std[1] > eta_std[0] > eta_std[2] and eta_std[1] > 0.5
        ),
        "97_ncbi_middle_dominates": n_dom == len(NCBI_TABLE_IDS) and len(NCBI_TABLE_IDS) == 22,
        "98_rumer_exchanges_complete": rumer_out == 8 and len(complete) == 8,
        "99_wc_preserves_gc_strength": wc_preserved == 16,
        "100_wc_keeps_complete_majority": wc_keep == 6,
        "101_stop_square_affine_rank2": aff_sq == 2 and len(STOP_SQUARE) == 4,
        "102_stop_tree_in_square": aff_stop == 2 and trp_corner,
        "103_triplet_minimal_address": (
            one_letter == 4
            and two_letter == 16
            and three_letter == 64
            and two_letter < n_labels <= three_letter
            and payload_bits == 6
        ),
    }
    return IdentityAxesCensus(
        eta2_std=eta_std,
        eta2_tables=tuple(table_rows),
        n_tables_mid_dominates=n_dom,
        n_tables=len(NCBI_TABLE_IDS),
        complete=complete,
        rumer_complete_to_incomplete=rumer_out,
        wc_strength_preserved=wc_preserved,
        wc_complete_to_complete=wc_keep,
        stop_aff_rank=aff_sq,
        stop_square_size=len(STOP_SQUARE),
        trp_is_corner=trp_corner,
        pair_stop_square_affine=aff_sq == 2,
        triplet_address=(one_letter, two_letter, three_letter, n_labels),
        gates=gates,
    )

def print_identity_axes_census(c: IdentityAxesCensus) -> None:
    g = c.gates
    report_section("9. IDENTITY AT THE FOLD; TWO AXES; STOP SQUARE; TRIPLETS")
    report_objects(
        (
            "hydropathy eta^2 by codon pos (KD external); Rumer vs WC; stop square; triplet address",
        )
    )
    print("  hydropathy eta^2 (standard code)")
    print(
        f"    first={c.eta2_std[0]:.4f}  middle={c.eta2_std[1]:.4f}  "
        f"wobble={c.eta2_std[2]:.4f}"
    )
    print(
        f"    NCBI tables with middle > first and middle > wobble: "
        f"{c.n_tables_mid_dominates}/{c.n_tables}"
    )
    mids = [row[2] for row in c.eta2_tables]
    print(f"    middle eta^2 range across tables: min={min(mids):.4f} max={max(mids):.4f}")
    print()
    print("  two root involutions on the 16 doublets")
    print(f"    complete (4-fold) boxes: {c.complete}")
    print(
        f"    Rumer: complete -> incomplete = {c.rumer_complete_to_incomplete}/"
        f"{len(c.complete)}"
    )
    print(f"    WC-RC: GC-strength preserved = {c.wc_strength_preserved}/16")
    print(
        f"    WC-RC: complete -> complete = {c.wc_complete_to_complete}/"
        f"{len(c.complete)}"
    )
    print()
    print("  stop as punctured affine square (pair_inversion)")
    print(
        f"    square {STOP_SQUARE} affine_rank={c.stop_aff_rank}  "
        f"Trp_is_corner={c.trp_is_corner}"
    )
    print()
    print("  triplet addressability")
    a1, a2, a3, nlab = c.triplet_address
    print(f"    4^1={a1}  4^2={a2}  4^3={a3}  labels_needed={nlab}  payload_bits=6")
    print()
    report_checks((
        ('standard-code middle base explains hydropathy (eta^2~0.756)', g['96_std_middle_hydropathy'], f'eta2=({c.eta2_std[0]:.4f},{c.eta2_std[1]:.4f},{c.eta2_std[2]:.4f})', 'middle > first > wobble and middle eta^2 > 0.5'),
        ('middle base dominates hydropathy on every NCBI table', g['97_ncbi_middle_dominates'], f'{c.n_tables_mid_dominates}/{c.n_tables}', '22/22'),
        ('Rumer root map sends all 8 complete boxes to incomplete', g['98_rumer_exchanges_complete'], f'{c.rumer_complete_to_incomplete}/{len(c.complete)}', '8/8'),
        ('WC reverse-complement preserves GC-strength on all 16 doublets', g['99_wc_preserves_gc_strength'], f'{c.wc_strength_preserved}/16', '16/16'),
        ('WC reverse-complement keeps 6 of 8 complete boxes complete', g['100_wc_keeps_complete_majority'], f'{c.wc_complete_to_complete}/8', '6/8'),
        ('stop+Trp form an affine rank-2 square', g['101_stop_square_affine_rank2'], f'aff={c.stop_aff_rank} size={c.stop_square_size}', 'aff=2 size=4'),
        ('stop occupies three vertices; Trp is the completing corner', g['102_stop_tree_in_square'], f'Trp_corner={c.trp_is_corner}', 'True'),
        ('triplets are the minimal 2-bit words addressing 21 labels on 6 bits', g['103_triplet_minimal_address'], f'4^2={a2} < {nlab} <= 4^3={a3}', '16 < 21 <= 64'),
    ))

def classify_diff(diff: int) -> str:
    d = int(diff) & 0x3F
    if d == 0:
        return "zero"
    o = d & OUTER_MASK
    f = d & FOLD_MASK
    if f == 0:
        return "outer_only"
    if o == 0:
        return "fold_only"
    return "mixed"

def theta_kin_census(enc: NucleotideEncoding) -> Tuple[int, int, int, int, int, int]:
    """Returns (byte_kin_neq, q_kin_neq, lambda_in_W, multi_image, max_images, byte_eq_q)."""
    byte_neq = 0
    q_neq = 0
    lambda_w = 0
    sig_to_qkin: Dict[OmegaSignature12, Set[int]] = defaultdict(set)
    for b in range(256):
        q = q_word6(b)
        bk = reverse_complement_4mer_byte(b, enc)
        bp = rc_byte_keep_family(b, enc)
        if bk != bp:
            byte_neq += 1
        qk = q_word6(bk)
        qp = q_word6(bp)
        if qk != qp:
            q_neq += 1
        lam = q ^ qk
        if in_W(lam):
            lambda_w += 1
        sig = omega_word_signature([b])
        sig_to_qkin[sig].add(qk)
    multi = sum(1 for qs in sig_to_qkin.values() if len(qs) > 1)
    max_img = max(len(qs) for qs in sig_to_qkin.values()) if sig_to_qkin else 0
    return byte_neq, q_neq, lambda_w, multi, max_img, 256 - byte_neq

def _syn_edges(code: Dict[str, str], enc: NucleotideEncoding) -> List[Tuple[int, int, int]]:
    """Directed syn edges (u, v, diff) with u < v."""
    out: List[Tuple[int, int, int]] = []
    for c in CODONS:
        if code[c] == "*":
            continue
        u = pack_codon_bits(c, enc)
        for n in one_base_neighbors(c):
            if code[n] != code[c]:
                continue
            v = pack_codon_bits(n, enc)
            if u >= v:
                continue
            out.append((u, v, (u ^ v) & 0x3F))
    return out

def _spanning_forest(vertices: Set[int], edges: Sequence[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    parent = {v: v for v in vertices}
    rank = {v: 0 for v in vertices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree: Set[Tuple[int, int]] = set()
    for u, v in sorted(edges):
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        tree.add((u, v))
        if rank[ru] < rank[rv]:
            parent[ru] = rv
        elif rank[ru] > rank[rv]:
            parent[rv] = ru
        else:
            parent[rv] = ru
            rank[ru] += 1
    return tree

def _tree_path_edges(
    tree: Set[Tuple[int, int]], u: int, v: int, adj: Dict[int, List[int]]
) -> Set[Tuple[int, int]]:
    if u == v:
        return set()
    prev: Dict[int, int] = {u: u}
    q = [u]
    while q:
        x = q.pop(0)
        for y in adj[x]:
            if y in prev:
                continue
            prev[y] = x
            if y == v:
                q = []
                break
            q.append(y)
    path_edges: Set[Tuple[int, int]] = set()
    x = v
    while x != u:
        p = prev[x]
        path_edges.add((min(p, x), max(p, x)))
        x = p
    return path_edges

def cycle_diff_vectors(code: Dict[str, str]) -> List[int]:
    """GF(2)^6 generators for the cycle space via fundamental cycles."""
    enc = encodings_in_orbit(ORBIT_PAIR_INV)[0][1]
    syn = _syn_edges(code, enc)
    edge_diff = {(u, v): d for u, v, d in syn}
    vertices = set()
    for u, v, _ in syn:
        vertices.add(u)
        vertices.add(v)
    undirected = [(u, v) for u, v, _ in syn]
    tree = _spanning_forest(vertices, undirected)
    adj: Dict[int, List[int]] = defaultdict(list)
    for u, v in undirected:
        adj[u].append(v)
        adj[v].append(u)
    vecs: List[int] = []
    for u, v, d in syn:
        e = (u, v)
        if e in tree:
            continue
        cyc = _tree_path_edges(tree, u, v, adj)
        cyc = set(cyc)
        cyc.add(e)
        acc = 0
        for eu, ev in cyc:
            acc ^= edge_diff[(eu, ev)]
        vecs.append(acc)
    return vecs

def intertwiner_cycle_rank(code: Dict[str, str] = STANDARD_CODE) -> Tuple[int, int, Tuple[int, ...]]:
    vecs = cycle_diff_vectors(code)
    _ne, _nc, b1 = synonymous_homology(code)
    rank = gf2_rank6(vecs)
    slot_tot = [0, 0, 0, 0]
    for d in vecs:
        assert d != 0, "fundamental cycle with zero XOR-diff"
        j = block_weight6(d)
        slot_tot[j] += 1
    return len(vecs), rank, tuple(slot_tot)

def substitution_cost_rows(code: Dict[str, str] = STANDARD_CODE) -> Tuple[Counter, Counter]:
    enc = encodings_in_orbit(ORBIT_PAIR_INV)[0][1]
    syn = Counter()
    all_mut = Counter()
    seen: Set[Tuple[str, str]] = set()
    for c in CODONS:
        pc = pack_codon_bits(c, enc)
        for n in one_base_neighbors(c):
            pair = (c, n) if c < n else (n, c)
            if pair in seen:
                continue
            seen.add(pair)
            d = (pc ^ pack_codon_bits(n, enc)) & 0x3F
            cls = classify_diff(d)
            all_mut[cls] += 1
            if code[c] == code[n] and code[c] != "*":
                syn[cls] += 1
    return syn, all_mut

def stop_leakage_map(code: Dict[str, str] = STANDARD_CODE) -> Dict[str, Tuple[str, ...]]:
    out: Dict[str, List[str]] = {c: [] for c in CODONS if code[c] == "*"}
    for stop in out:
        for n in one_base_neighbors(stop):
            if code[n] != "*":
                out[stop].append(n)
    return {k: tuple(sorted(set(v))) for k, v in out.items()}

@dataclass
class AlgebraCompletionCensus:
    byte_kin_neq: int
    q_kin_neq: int
    q_kin_eq: int
    lambda_in_W: int
    sig_multi_image: int
    max_kin_images: int
    n_cycles: int
    cycle_intertwiner_rank: int
    cycle_slot_tot: Tuple[int, ...]
    syn_cost: Tuple[Tuple[str, int], ...]
    stop_neighbors: Tuple[Tuple[str, Tuple[str, ...]], ...]
    beta1: int
    gates: Dict[str, bool]

def algebra_completion_census() -> AlgebraCompletionCensus:
    enc = encodings_in_orbit(ORBIT_PAIR_INV)[0][1]
    byte_neq, q_neq, lam_w, multi, max_img, q_eq = theta_kin_census(enc)
    n_cyc, c_rank, slot_tot = intertwiner_cycle_rank(STANDARD_CODE)
    syn, _all = substitution_cost_rows()
    leak = stop_leakage_map()
    _ne, _nc, b1 = synonymous_homology(STANDARD_CODE)

    gates = {
        "104_kin_neq_payload_bytes": byte_neq == 256,
        "105_lambda_in_W_all_bytes": lam_w == 256,
        "106_kin_multi_image_signatures": multi > 0 and max_img >= 2,
        "107_cycle_count_beta1": n_cyc == b1 == 27,
        "108_cycle_count_h34_j2": n_cyc == H34_MULTIPLICITIES[2],
        "109_synonymous_outer_only": syn.get("outer_only", 0) == 67 and syn.get("fold_only", 0) == 0,
        "110_stop_leakage_to_sense": all(len(v) > 0 for v in leak.values()) and len(leak) == 3,
        "111_trp_bridges_tga": "TGG" in leak.get("TGA", ()),
    }
    return AlgebraCompletionCensus(
        byte_kin_neq=byte_neq,
        q_kin_neq=q_neq,
        q_kin_eq=q_eq,
        lambda_in_W=lam_w,
        sig_multi_image=multi,
        max_kin_images=max_img,
        n_cycles=n_cyc,
        cycle_intertwiner_rank=c_rank,
        cycle_slot_tot=slot_tot,
        syn_cost=tuple(sorted(syn.items())),
        stop_neighbors=tuple(sorted(leak.items())),
        beta1=b1,
        gates=gates,
    )

def print_algebra_completion_census(c: AlgebraCompletionCensus) -> None:
    g = c.gates
    report_section("10. THETA_KIN; CYCLE INTERTWINER; SUBSTITUTION COST; STOP LEAKAGE")
    report_objects(
        (
            "Theta_kin; cycle space; intertwiner j-profile; sub cost; stop leakage",
        )
    )
    print("  Theta_kin (pair_inversion)")
    print(
        f"    byte kin!=payload {c.byte_kin_neq}/256  q kin!=payload {c.q_kin_neq}/256  "
        f"(q agree on {c.q_kin_eq} bytes)  lambda in W {c.lambda_in_W}/256  "
        f"multi-image signatures {c.sig_multi_image}  max images {c.max_kin_images}"
    )
    print()
    print("  cycle intertwiner")
    print(
        f"    n_cycles={c.n_cycles} beta1={c.beta1}  cycle_diff_rank={c.cycle_intertwiner_rank}  "
        f"slot_tot={c.cycle_slot_tot}  H34_j2={H34_MULTIPLICITIES[2]}"
    )
    print()
    print("  synonymous substitution cost")
    print(f"    {dict(c.syn_cost)}")
    print()
    print("  stop leakage map (stop -> adjacent sense codons)")
    for stop, sense in c.stop_neighbors:
        print(f"    {stop} -> {sense}")
    print()
    report_checks((
        ('kinematic RC differs from payload-RC on every byte', g['104_kin_neq_payload_bytes'], f'{c.byte_kin_neq}/256', '256/256'),
        ('kinematic q-residual lambda(b) lies in W for every byte', g['105_lambda_in_W_all_bytes'], f'{c.lambda_in_W}/256', '256/256'),
        ('some Omega signatures carry multiple kinematic q-images', g['106_kin_multi_image_signatures'], f'multi={c.sig_multi_image} max={c.max_kin_images}', 'multi>0, max>=2'),
        ('fundamental cycle count equals beta1=27', g['107_cycle_count_beta1'], f'n_cycles={c.n_cycles} beta1={c.beta1}', '27'),
        ('fundamental cycle count equals H(3,4) j=2 multiplicity (27)', g['108_cycle_count_h34_j2'], f'n_cycles={c.n_cycles}', '27'),
        ('sense-synonymous edges are outer-plane only (67 edges, 0 fold)', g['109_synonymous_outer_only'], str(dict(c.syn_cost)), 'outer_only=67, fold_only=0'),
        ('every stop codon leaks to sense by one-base adjacency', g['110_stop_leakage_to_sense'], f'stops={len(c.stop_neighbors)}', '3 stops, each with neighbors'),
        ('Trp codon TGG is one-base adjacent to TGA stop', g['111_trp_bridges_tga'], str(dict(c.stop_neighbors)), 'TGG in TGA neighbors'),
    ))

MAX_CDS = 2000

SHELL_NULL = tuple(math.comb(6, k) / 64.0 for k in range(7))

def shell_hist(codons: Sequence[str], enc) -> Tuple[int, ...]:
    hist = [0] * 7
    prev = None
    for c in codons:
        if prev is not None:
            q0 = pack_codon_bits(prev, enc)
            q1 = pack_codon_bits(c, enc)
            hist[(q0 ^ q1).bit_count()] += 1
        prev = c
    return tuple(hist)

def l1_to_null(hist: Tuple[int, ...]) -> float:
    n = sum(hist)
    if n == 0:
        return float("nan")
    obs = [h / n for h in hist]
    return sum(abs(obs[k] - SHELL_NULL[k]) for k in range(7))

def mean_shell(codons: Sequence[str], enc) -> float:
    n = 0
    s = 0.0
    prev = None
    for c in codons:
        if prev is not None:
            q0 = pack_codon_bits(prev, enc)
            q1 = pack_codon_bits(c, enc)
            s += (q0 ^ q1).bit_count()
            n += 1
        prev = c
    return s / n if n else float("nan")

def genome_rows() -> List[Tuple[str, str, Tuple[str, ...]]]:
    specs = (
        (("ecoli_k12",), "ecoli"),
        (("yeast_s288c",), "yeast"),
        (("chr22_cds",), "chr22"),
    )
    rows = []
    for keys, label in specs:
        recs = load_named_fasta(keys)
        if not recs:
            rows.append((label, "SKIP", ()))
            continue
        codons: List[str] = []
        for _id, seq in recs[:MAX_CDS]:
            codons.extend(iter_codons(seq))
        rows.append((label, "ok", tuple(codons)))
    return rows

def chart_s2_rankings(code: Dict[str, str]) -> Tuple[int, int]:
    """Return (n_tables pair_wins, n_tables)."""
    grid, encs = ncbi_chart_s2_grid()
    wins = 0
    for ti in range(grid.shape[0]):
        best_pair = -1.0
        best_elem = -1.0
        for ei, enc in enumerate(encs):
            s2 = float(grid[ti, ei])
            orbit = encoding_orbit_name(enc)
            if orbit == ORBIT_PAIR_INV:
                best_pair = max(best_pair, s2)
            elif orbit == ORBIT_ELEMENTARY:
                best_elem = max(best_elem, s2)
        if best_pair > best_elem:
            wins += 1
    return wins, grid.shape[0]

@dataclass
class DynamicsRow:
    name: str
    status: str
    n_pairs: int
    mean_shell: float
    l1_null: float
    low_shell_frac: float

@dataclass
class DynamicsCensus:
    rows: Tuple[DynamicsRow, ...]
    pair_wins: int
    n_tables: int
    per_chart_pair_top: int
    gates: Dict[str, bool]

def dynamics_census() -> DynamicsCensus:
    reps = orbit_reps()
    enc_p = reps[ORBIT_PAIR_INV][1]
    dyn_rows: List[DynamicsRow] = []
    ok_l1 = []
    ok_low = []
    for name, status, codons in genome_rows():
        if status != "ok" or len(codons) < 2:
            dyn_rows.append(
                DynamicsRow(name, status, 0, float("nan"), float("nan"), float("nan"))
            )
            continue
        hist = shell_hist(codons, enc_p)
        n_pairs = sum(hist)
        ms = mean_shell(codons, enc_p)
        l1 = l1_to_null(hist)
        low = (hist[0] + hist[1]) / n_pairs if n_pairs else float("nan")
        dyn_rows.append(DynamicsRow(name, status, n_pairs, ms, l1, low))
        ok_l1.append(l1)
        ok_low.append(low)

    pair_wins, n_tab = chart_s2_rankings(STANDARD_CODE)
    grid, encs = ncbi_chart_s2_grid()
    table1_idx = NCBI_TABLE_IDS.index(1)
    s2_by_chart = [float(grid[table1_idx, ei]) for ei in range(len(encs))]
    pair_s2 = max(
        s2_by_chart[ei]
        for ei, enc in enumerate(encs)
        if encoding_orbit_name(enc) == ORBIT_PAIR_INV
    )
    n_pair_top = sum(1 for s2 in s2_by_chart if s2 <= pair_s2 + 1e-15)

    gates = {
        "112_orf_shell_deviates_from_null": all(l1 > 0.02 for l1 in ok_l1) and len(ok_l1) >= 2,
        "113_mean_shell_below_null": all(ms < 3.0 for ms in [r.mean_shell for r in dyn_rows if r.status == "ok"]),
        "114_pair_s2_wins_all_tables": pair_wins == n_tab and n_tab > 0,
        "115_pair_orbit_S2_gt_elementary": pair_s2 > max(
            s2_by_chart[ei]
            for ei, enc in enumerate(encs)
            if encoding_orbit_name(enc) == ORBIT_ELEMENTARY
        ),
    }
    return DynamicsCensus(
        rows=tuple(dyn_rows),
        pair_wins=pair_wins,
        n_tables=n_tab,
        per_chart_pair_top=n_pair_top,
        gates=gates,
    )

def print_dynamics_census(c: DynamicsCensus) -> None:
    g = c.gates
    report_section("11. QU BEC WALK; GAUGE SELECTION; SHELL VELOCITY")
    report_objects(
        (
            "chi-shell hist; binomial null; low-shell frac; S2 gauge selection",
        )
    )
    print("  ORF shell meters (pair_inversion)")
    print(f"    {'name':<8} {'status':<6} {'pairs':>8} {'mean_sh':>8} {'L1_null':>8} {'low_frac':>8}")
    for r in c.rows:
        print(
            f"    {r.name:<8} {r.status:<6} {r.n_pairs:>8} "
            f"{r.mean_shell:>8.4f} {r.l1_null:>8.4f} {r.low_shell_frac:>8.4f}"
        )
    print()
    print("  gauge selection")
    print(
        f"    S2(pair) > S2(elem) on NCBI tables: {c.pair_wins}/{c.n_tables}  "
        f"pair octet tied top on table 1: {c.per_chart_pair_top}/24"
    )
    print()
    report_checks((
        ('real ORFs deviate from binomial shell null (L1 > 0.02)', g['112_orf_shell_deviates_from_null'], f"L1={[round(r.l1_null, 4) for r in c.rows if r.status == 'ok']}", 'all ok genomes > 0.02'),
        ('mean chi-shell below binomial expectation (3.0) on real ORFs', g['113_mean_shell_below_null'], f"mean={[round(r.mean_shell, 4) for r in c.rows if r.status == 'ok']}", 'all ok genomes < 3.0'),
        ('pair_inversion S2 wins every NCBI table vs elementary', g['114_pair_s2_wins_all_tables'], f'{c.pair_wins}/{c.n_tables}', f'{c.n_tables}/{c.n_tables}'),
        ('pair_inversion orbit S2 exceeds elementary on table 1', g['115_pair_orbit_S2_gt_elementary'], f'pair octet tied top tier {c.per_chart_pair_top}/24', 'pair S2 > elem S2'),
    ))


def _qubec_from_mean_shell(mean_shell: float) -> Tuple[float, float, float, float]:
    """Moment fit: E[N]=6*rho => rho=mean/6; lambda=rho/(1-rho); eta; M2."""
    if not (mean_shell == mean_shell) or mean_shell <= 0.0 or mean_shell >= 6.0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rho = mean_shell / 6.0
    if rho >= 1.0 - 1e-15:
        return float("nan"), float("nan"), float("nan"), float("nan")
    lam = rho / (1.0 - rho)
    eta = (1.0 - lam) / (1.0 + lam)
    m2 = 4096.0 / ((1.0 + eta * eta) ** 6)
    return lam, rho, eta, m2


def _gc_shuffle_codons4(codons: Sequence[str], rng: random.Random) -> List[str]:
    s = list("".join(codons))
    idx_gc = [i for i, b in enumerate(s) if b in STRONG]
    idx_at = [i for i, b in enumerate(s) if b in {"A", "T"}]
    gcb = [s[i] for i in idx_gc]
    atb = [s[i] for i in idx_at]
    rng.shuffle(gcb)
    rng.shuffle(atb)
    for i, b in zip(idx_gc, gcb):
        s[i] = b
    for i, b in zip(idx_at, atb):
        s[i] = b
    s2 = "".join(s)
    return [s2[i : i + 3] for i in range(0, len(s2) - 2, 3)]


@dataclass
class QubecRow:
    name: str
    n_genes: int
    n_pairs: int
    mean_shell: float
    lam: float
    rho: float
    eta: float
    m2: float
    null_eta: float
    null_m2: float
    p_eta_high: int
    n_null: int


@dataclass
class QubecCensus:
    rows: Tuple[QubecRow, ...]
    gates: Dict[str, bool]


def qubec_order_census() -> QubecCensus:
    enc = orbit_reps()[ORBIT_PAIR_INV][1]
    rng = random.Random(NULL_SEED + 4)
    n_null = 12
    specs = (
        (("ecoli_k12",), "ecoli", None),
        (("yeast_s288c",), "yeast", None),
        (("chr22_cds",), "chr22", [s for _h, s, _st in extract_chr22_cds(200)]),
    )
    rows: List[QubecRow] = []
    for keys, name, extra in specs:
        genes: List[List[str]] = []
        recs = load_named_fasta(keys) if extra is None else None
        if recs:
            for _h, s in recs[:400]:
                cs = iter_codons(s)
                if len(cs) >= 30:
                    genes.append(cs)
        if extra:
            for s in extra:
                cs = iter_codons(s)
                if len(cs) >= 30:
                    genes.append(cs)
        if not genes:
            continue
        tot = 0.0
        n_pairs = 0
        for g in genes:
            ms = mean_shell(g, enc)
            if ms == ms:
                np_ = len(g) - 1
                tot += ms * np_
                n_pairs += np_
        if n_pairs == 0:
            continue
        obs_mean = tot / n_pairs
        lam, rho, eta, m2 = _qubec_from_mean_shell(obs_mean)

        null_etas = []
        null_m2s = []
        for _ in range(n_null):
            nt = 0.0
            npn = 0
            for g in genes[:120]:
                sg = _gc_shuffle_codons4(g, rng)
                ms = mean_shell(sg, enc)
                if ms == ms:
                    np_ = len(sg) - 1
                    nt += ms * np_
                    npn += np_
            if npn:
                _l, _r, ne, nm = _qubec_from_mean_shell(nt / npn)
                null_etas.append(ne)
                null_m2s.append(nm)
        null_eta = sum(null_etas) / len(null_etas) if null_etas else float("nan")
        null_m2 = sum(null_m2s) / len(null_m2s) if null_m2s else float("nan")
        # condensed => |eta| larger than thermal (0); coding smoother => mean shell < 3 => eta > 0
        p_eta_high = sum(1 for x in null_etas if eta >= x)
        rows.append(
            QubecRow(
                name, len(genes), n_pairs, obs_mean, lam, rho, eta, m2,
                null_eta, null_m2, p_eta_high, n_null,
            )
        )

    gates = {
        "148_qubec_eta_positive": all(r.eta > 0 for r in rows) and bool(rows),
        "149_qubec_m2_below_thermal": all(r.m2 < 4096.0 for r in rows) and bool(rows),
        "150_qubec_eta_above_null": all(r.p_eta_high >= r.n_null - 2 for r in rows) and bool(rows),
    }
    return QubecCensus(rows=tuple(rows), gates=gates)


def print_qubec_order_census(c: QubecCensus) -> None:
    g = c.gates
    report_section("22. QuBEC ORDER PARAMETERS ON ORFs")
    report_objects(("shell hist -> lambda fit; rho=lam/(1+lam); eta; M2; GC-matched null",))
    print_table(
        ("name", "genes", "pairs", "mean", "lam", "rho", "eta", "M2", "null_eta", "p_eta"),
        (8, 6, 8, 7, 7, 6, 7, 8, 8, 6),
        [
            (
                r.name,
                r.n_genes,
                r.n_pairs,
                f"{r.mean_shell:.4f}",
                f"{r.lam:.4f}",
                f"{r.rho:.4f}",
                f"{r.eta:.4f}",
                f"{r.m2:.1f}",
                f"{r.null_eta:.4f}",
                f"{r.p_eta_high}/{r.n_null}",
            )
            for r in c.rows
        ],
        aligns=("<", ">", ">", ">", ">", ">", ">", ">", ">", ">"),
    )
    print()
    report_checks((
        ("eta > 0 on every genome (damped vs thermal)", g["148_qubec_eta_positive"], f"eta={[round(r.eta,4) for r in c.rows]}", "all > 0"),
        ("M2 < 4096 (below fully thermalized support)", g["149_qubec_m2_below_thermal"], f"M2={[round(r.m2,1) for r in c.rows]}", "all < 4096"),
        ("eta above GC-matched null (p_eta >= 10/12)", g["150_qubec_eta_above_null"], f"p={[r.p_eta_high for r in c.rows]}", "all >= n_null-2"),
    ))
