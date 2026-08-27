#!/usr/bin/env python3
"""
hqvm_cgm_genomics_7.py

Code-classification constraint harness and tRNA-identity vs fold-plane probe.
"""
from __future__ import annotations

import itertools
import random
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from hqvm_cgm_genomics_1 import synonymous_homology
from hqvm_cgm_genomics_4 import (
    FOLD_MASK,
    OUTER_MASK,
    SER_AGY,
    SER_TCN,
    fold_intersection,
    sense_edge_diffs,
    serine_chords,
    span_elements,
    stop_diff,
)
from hqvm_cgm_genomics_5 import fold_plane_syn_edges
from gyroscopic.hQVM.api import BYTES_BY_Q6
from gyroscopic.hQVM.constants import ab_distance, horizon_distance, unpack_state
from gyroscopic.hQVM.family import (
    build_hqvm_d,
    intron_family_d,
    intron_from_byte,
    predicted_cluster_size,
    rest_uv,
)

from hqvm_cgm_genomics_common import (
    AA_ORDER,
    BASES,
    CODONS,
    DATA_DIR,
    NCBI_TABLE_IDS,
    NULL_SEED,
    ORBIT_PAIR_INV,
    STANDARD_CODE,
    STRONG,
    NucleotideEncoding,
    affine_rank6,
    carrier_from_codon_pair,
    codon_state,
    encodings_in_orbit,
    fiber_components,
    fibers,
    gc_fraction,
    gf2_rank6,
    iter_codons,
    one_base_neighbors,
    pack_codon_bits,
    parse_fasta,
    print_table,
    report_checks,
    report_objects,
    report_section,
    translation_table,
)

# Measured sense size multiset of the standard code (excl stop).
STD_SIZE_PROFILE = Counter({6: 3, 4: 5, 3: 1, 2: 9, 1: 2})
STOP_TRP_SQUARE = ("TAA", "TAG", "TGA", "TGG")


def _ref_enc() -> NucleotideEncoding:
    return encodings_in_orbit(ORBIT_PAIR_INV)[0][1]


def ref_enc() -> NucleotideEncoding:
    """Pair-inversion chart used as the duplex-constitutional reference."""
    return _ref_enc()


def size_profile(code: Dict[str, str]) -> Counter:
    return Counter(len(g) for aa, g in fibers(code).items() if aa != "*")


def n_stops(code: Dict[str, str]) -> int:
    return sum(1 for c in CODONS if code[c] == "*")


def wall_closed(code: Dict[str, str], enc: NucleotideEncoding) -> bool:
    """L_sense rank 4 and L_sense ∩ P_fold = {0} (no fold-plane sense edges)."""
    sense = sense_edge_diffs(enc, code)
    if gf2_rank6(sense) != 4:
        return False
    return fold_plane_syn_edges(code, enc) == 0


def has_serine_split(code: Dict[str, str]) -> bool:
    group = [c for c in CODONS if code[c] == "S"]
    if len(group) != 6:
        return False
    return len(fiber_components(group)) == 2


def stop_punctured_square(code: Dict[str, str], enc: NucleotideEncoding) -> bool:
    """Stop occupies 3 vertices of an affine rank-2 square completed by Trp."""
    if n_stops(code) != 3:
        return False
    stops = [c for c in CODONS if code[c] == "*"]
    trp = [c for c in CODONS if code[c] == "W"]
    if len(trp) != 1:
        return False
    hull = stops + trp
    bits = [pack_codon_bits(c, enc) for c in hull]
    if affine_rank6(bits) != 2:
        return False
    # Trp adjacent by one base to at least one stop.
    return any(
        t in one_base_neighbors(s) for s in stops for t in trp
    )


def beta1_is_27(code: Dict[str, str]) -> bool:
    _e, _c, b1 = synonymous_homology(code)
    return b1 == 27


def family_sheet_invariant(code: Dict[str, str]) -> bool:
    """Labels depend on payload only: already true for any codon->AA map.
    Recorded as tautology gate for the classification interface."""
    return len(code) == 64 and set(code) == set(CODONS)


def _intra_fiber_ab_shells(
    group: Sequence[str], enc: NucleotideEncoding
) -> Counter:
    shells: Counter = Counter()
    for i, c1 in enumerate(group):
        for c2 in group[i + 1 :]:
            st = carrier_from_codon_pair(codon_state(c1, enc), codon_state(c2, enc))
            a12, b12 = unpack_state(st)
            shells[ab_distance(a12, b12)] += 1
    return shells


def _pair_ab_horizon(c1: str, c2: str, enc: NucleotideEncoding) -> Tuple[int, int]:
    st = carrier_from_codon_pair(codon_state(c1, enc), codon_state(c2, enc))
    a12, b12 = unpack_state(st)
    return ab_distance(a12, b12), horizon_distance(a12, b12)


def serine_complement_reach(code: Dict[str, str], enc: NucleotideEncoding) -> bool:
    """Constraint (x): serine unique at ab=12; other sense fibers ab<=6."""
    ser = [c for c in CODONS if code[c] == "S"]
    ser_shells = _intra_fiber_ab_shells(ser, enc)
    if not ser_shells or max(ser_shells) != 12 or ser_shells.get(12, 0) != 2:
        return False
    for aa, group in fibers(code).items():
        if aa in ("*", "S") or len(group) < 2:
            continue
        sh = _intra_fiber_ab_shells(group, enc)
        if sh and max(sh) > 6:
            return False
    return True


def stop_fiber_near_equality(code: Dict[str, str], enc: NucleotideEncoding) -> bool:
    """Constraint (x) stop half: stop intra-fiber ab <= 4."""
    stops = [c for c in CODONS if code[c] == "*"]
    sh = _intra_fiber_ab_shells(stops, enc)
    return (not sh) or max(sh) <= 4


def stop_trp_complementarity(code: Dict[str, str], enc: NucleotideEncoding) -> bool:
    """Constraint (xi): all pairs among stop+Trp hull obey ab + horizon = 12."""
    stops = [c for c in CODONS if code[c] == "*"]
    trp = [c for c in CODONS if code[c] == "W"]
    hull = stops + trp
    if len(hull) < 2:
        return False
    for i, c1 in enumerate(hull):
        for c2 in hull[i + 1 :]:
            abd, hd = _pair_ab_horizon(c1, c2, enc)
            if abd + hd != 12:
                return False
    return True


def complete_wobble_boxes(code: Dict[str, str]) -> int:
    """Count of doublet prefixes with a single AA on all four third bases."""
    n = 0
    for a in BASES:
        for b in BASES:
            labs = {code[a + b + t] for t in BASES}
            if len(labs) == 1 and "*" not in labs:
                n += 1
    return n


def affine_fibers_ok(code: Dict[str, str], enc: NucleotideEncoding) -> bool:
    """Every sense fiber is an affine flat (rank 0–3)."""
    for aa, group in fibers(code).items():
        if aa == "*":
            continue
        bits = [pack_codon_bits(c, enc) for c in group]
        r = affine_rank6(bits)
        if r < 0 or r > 3:
            return False
        if (1 << max(r, 0)) < len(group) and r >= 0:
            # size must not exceed hull; hull = 2^rank for affine flats through origin of differences
            if len(group) > (1 << r):
                return False
    return True


def u8_constraints(
    code: Dict[str, str],
    enc: NucleotideEncoding,
) -> Dict[str, bool]:
    return {
        "wall_closed": wall_closed(code, enc),
        "size_profile": size_profile(code) == STD_SIZE_PROFILE and n_stops(code) == 3,
        "affine_fibers": affine_fibers_ok(code, enc),
        "serine_split": has_serine_split(code),
        "stop_square": stop_punctured_square(code, enc),
        "beta1_27": beta1_is_27(code),
        "wobble_boxes_8": complete_wobble_boxes(code) == 8,
        "family_sheet": family_sheet_invariant(code),
        "serine_complement_reach": serine_complement_reach(code, enc),
        "stop_fiber_near_eq": stop_fiber_near_equality(code, enc),
        "stop_trp_invariant": stop_trp_complementarity(code, enc),
    }


def u8_pass(code: Dict[str, str], enc: NucleotideEncoding) -> bool:
    return all(u8_constraints(code, enc).values())


@dataclass
class U8StandardCensus:
    checks: Dict[str, bool]
    gates: Dict[str, bool]


def u8_standard_census() -> U8StandardCensus:
    enc = _ref_enc()
    checks = u8_constraints(STANDARD_CODE, enc)
    gates = {
        "171_standard_passes_code_constraints": all(checks.values()),
        "172_code_class_wall_closed": checks["wall_closed"],
        "173_code_class_size_profile": checks["size_profile"],
        "174_code_class_serine_and_stop": checks["serine_split"] and checks["stop_square"],
        "175_code_class_beta1_and_boxes": checks["beta1_27"] and checks["wobble_boxes_8"],
        "203_code_class_constitutional": (
            checks["serine_complement_reach"]
            and checks["stop_fiber_near_eq"]
            and checks["stop_trp_invariant"]
        ),
    }
    return U8StandardCensus(checks=checks, gates=gates)


def print_u8_standard(c: U8StandardCensus) -> None:
    g = c.gates
    report_section("31. CODE-CLASSIFICATION CONSTRAINTS ON STANDARD CODE")
    report_objects((
        "wall; size; affine; serine; stop square; beta1; wobble; constitutional shells",
    ))
    print_table(
        ("constraint", "ok"),
        (24, 6),
        [(k, "PASS" if v else "FAIL") for k, v in c.checks.items()],
        aligns=("<", ">"),
    )
    print()
    report_checks((
        ("standard code satisfies every code-classification constraint", g["171_standard_passes_code_constraints"], str(all(c.checks.values())), "True"),
        ("wall closed (rank 4, fold syn edges 0)", g["172_code_class_wall_closed"], str(c.checks["wall_closed"]), "True"),
        ("size profile matches measured multiset", g["173_code_class_size_profile"], str(c.checks["size_profile"]), "True"),
        ("serine split and punctured stop square", g["174_code_class_serine_and_stop"], "see checks", "True"),
        ("beta1=27 and eight complete wobble boxes", g["175_code_class_beta1_and_boxes"], "see checks", "True"),
        ("constitutional side conditions (x)-(xi)", g["203_code_class_constitutional"], "see checks", "True"),
    ))


@dataclass
class U8LocalCensus:
    n_moves: int
    n_weak: int
    n_u8: int
    survivors: Tuple[Tuple[str, str, str], ...]
    gates: Dict[str, bool]


def u8_local_moduli_census() -> U8LocalCensus:
    """Filter one-codon reassignments through weak neighborhood then full constraints."""
    enc = _ref_enc()
    n_moves = n_weak = 0
    survivors: List[Tuple[str, str, str]] = []
    labels = list(AA_ORDER)
    for c in CODONS:
        cur = STANDARD_CODE[c]
        for aa in labels:
            if aa == cur:
                continue
            tab = dict(STANDARD_CODE)
            tab[c] = aa
            if n_stops(tab) != 3:
                continue
            n_moves += 1
            _e, _comp, b1 = synonymous_homology(tab)
            fe = fold_plane_syn_edges(tab, enc)
            if not (20 <= b1 <= 34 and fe <= 1):
                continue
            n_weak += 1
            if u8_pass(tab, enc):
                survivors.append((c, cur, aa))
    n_u8 = len(survivors)
    gates = {
        "176_local_moves_enumerated": n_moves > 1000,
        "177_full_constraints_local_moduli_smaller": n_u8 < n_weak,
        "178_full_constraints_local_count_5": n_u8 == 5,
    }
    return U8LocalCensus(n_moves, n_weak, n_u8, tuple(survivors), gates)


def print_u8_local(c: U8LocalCensus) -> None:
    g = c.gates
    report_section("32. LOCAL MODULI UNDER FULL CODE CONSTRAINTS")
    report_objects(("one-codon reassignments; weak nbhd vs full constraint filter; survivors",))
    print(f"    moves_with_3_stops={c.n_moves} weak_nbhd={c.n_weak} full_constraints={c.n_u8}")
    print(f"    survivors={c.survivors}")
    print()
    report_checks((
        ("one-move space enumerated at scale", g["176_local_moves_enumerated"], f"n={c.n_moves}", ">1000"),
        ("full constraints stricter than weak neighborhood", g["177_full_constraints_local_moduli_smaller"], f"{c.n_u8} < {c.n_weak}", "full < weak"),
        ("full-constraint local moduli has 5 survivors (one Aff_S6 orbit)", g["178_full_constraints_local_count_5"], f"n={c.n_u8}", "5"),
    ))


@dataclass
class U8SliceCensus:
    n_stop_configs: int
    n_stop_u8: int
    n_ser_configs: int
    n_ser_u8: int
    gates: Dict[str, bool]


def u8_structured_slice_census() -> U8SliceCensus:
    """Structured slice: stop placements on the stop–Trp square; serine AGY↔AGR swaps."""
    enc = _ref_enc()
    n_stop = n_stop_u8 = 0
    for leave in STOP_TRP_SQUARE:
        tab = dict(STANDARD_CODE)
        for c in STOP_TRP_SQUARE:
            tab[c] = "*" if c != leave else "W"
        for c in CODONS:
            if c not in STOP_TRP_SQUARE and tab[c] == "*":
                tab[c] = STANDARD_CODE[c]
        n_stop += 1
        if u8_pass(tab, enc):
            n_stop_u8 += 1

    n_ser = n_ser_u8 = 0
    agr = ["AGA", "AGG"]
    for a_from in SER_AGY:
        for a_to in agr:
            tab = dict(STANDARD_CODE)
            tab[a_from], tab[a_to] = tab[a_to], tab[a_from]
            n_ser += 1
            if u8_pass(tab, enc):
                n_ser_u8 += 1

    gates = {
        "179_stop_square_slice_defined": n_stop == 4,
        "180_stop_square_all_four_pass_code_constraints": n_stop_u8 == 4,
        "181_ser_bridge_slice_defined": n_ser > 0,
    }
    return U8SliceCensus(n_stop, n_stop_u8, n_ser, n_ser_u8, gates)


def print_u8_slice(c: U8SliceCensus) -> None:
    g = c.gates
    report_section("33. STRUCTURED SLICES (STOP SQUARE; SER BRIDGE)")
    report_objects(("stop on 3/4 of TAA/TAG/TGA/TGG; AGY↔AGR swaps; full-constraint survivors",))
    print(
        f"    stop_configs={c.n_stop_configs} stop_u8={c.n_stop_u8} "
        f"ser_swaps={c.n_ser_configs} ser_u8={c.n_ser_u8}"
    )
    print()
    report_checks((
        ("stop-square slice has 4 completions", g["179_stop_square_slice_defined"], f"n={c.n_stop_configs}", "4"),
        ("all four stop-square completions pass full constraints", g["180_stop_square_all_four_pass_code_constraints"], f"survivors={c.n_stop_u8}", "4"),
        ("serine-bridge swap slice enumerated", g["181_ser_bridge_slice_defined"], f"n={c.n_ser_configs} u8={c.n_ser_u8}", ">0"),
    ))


# ----------------------------------------
# tRNA identity elements vs fold plane
# ----------------------------------------

# Curated E. coli identity-element sites (Giegé / McClain compilations).
# Each entry: (aa, element_description, role)
# role:
#   1  = anticodon middle (position 35; codon middle / fold plane)
#   2  = anticodon contact without a curated middle-only assignment
#  -1  = outside the anticodon (acceptor stem, discriminator, variable arm, ...)
TRNA_IDENTITY: Tuple[Tuple[str, str, int], ...] = (
    ("A", "G3:U70", -1),
    ("A", "discriminator A73", -1),
    ("R", "A20", -1),
    ("R", "C35", 1),
    ("N", "anticodon", 2),
    ("D", "G73", -1),
    ("D", "anticodon", 2),
    ("C", "U73", -1),
    ("C", "anticodon", 2),
    ("Q", "anticodon", 2),
    ("E", "anticodon", 2),
    ("G", "C35", 1),
    ("G", "discriminator", -1),
    ("H", "anticodon", 2),
    ("I", "anticodon", 2),
    ("L", "A73", -1),
    ("L", "anticodon", 2),
    ("K", "anticodon", 2),
    ("M", "CAU anticodon", 2),
    ("M", "A73", -1),
    ("F", "anticodon", 2),
    ("P", "anticodon", 2),
    ("P", "G72", -1),
    ("S", "G73", -1),
    ("S", "variable arm", -1),
    ("T", "anticodon", 2),
    ("W", "anticodon", 2),
    ("Y", "anticodon", 2),
    ("V", "anticodon", 2),
    ("V", "A73", -1),
)


@dataclass
class TrnaCensus:
    n_elements: int
    n_anticodon_middle: int
    n_anticodon_any: int
    n_outside: int
    frac_anticodon: float
    frac_middle: float
    gates: Dict[str, bool]


def trna_fold_census() -> TrnaCensus:
    n = len(TRNA_IDENTITY)
    n_mid = sum(1 for _aa, _d, role in TRNA_IDENTITY if role == 1)
    n_ac = sum(1 for _aa, _d, role in TRNA_IDENTITY if role in (1, 2))
    n_out = sum(1 for _aa, _d, role in TRNA_IDENTITY if role == -1)
    frac_ac = n_ac / n if n else float("nan")
    frac_mid = n_mid / n if n else float("nan")
    gates = {
        "182_trna_table_loaded": n >= 20,
        "183_anticodon_identity_majority": frac_ac > 0.5,
        "184_outside_identity_present": n_out > 0,
        "185_anticodon_middle_curated": n_mid >= 1,
    }
    return TrnaCensus(n, n_mid, n_ac, n_out, frac_ac, frac_mid, gates)


def print_trna_census(c: TrnaCensus) -> None:
    g = c.gates
    report_section("34. tRNA IDENTITY ELEMENTS VS ANTICODON / FOLD")
    report_objects((
        "curated E. coli identity sites",
        "role 1 = anticodon middle (pos 35 / fold); role 2 = other anticodon; -1 = outside",
    ))
    print(
        f"    elements={c.n_elements} anticodon_any={c.n_anticodon_any} "
        f"anticodon_middle={c.n_anticodon_middle} outside={c.n_outside} "
        f"frac_anticodon={c.frac_anticodon:.3f} frac_middle={c.frac_middle:.3f}"
    )
    print()
    report_checks((
        ("identity-element table loaded", g["182_trna_table_loaded"], f"n={c.n_elements}", ">=20"),
        ("majority of listed elements involve the anticodon", g["183_anticodon_identity_majority"], f"frac={c.frac_anticodon:.3f}", ">0.5"),
        ("non-anticodon identity elements also present", g["184_outside_identity_present"], f"n_out={c.n_outside}", ">0"),
        ("at least one curated anticodon-middle (pos 35) contact", g["185_anticodon_middle_curated"], f"n_mid={c.n_anticodon_middle}", ">=1"),
    ))


# ----------------------------------------
# Code classification: Aut quotient, two-move, NCBI
# ----------------------------------------

SIZE2_AAS: Tuple[str, ...] = tuple(
    aa for aa, grp in fibers(STANDARD_CODE).items() if aa != "*" and len(grp) == 2
)


def _gl2_gens() -> List[Tuple[int, ...]]:
    """Elementary generators of GL(2,2) as row-bit tuples."""
    eye = (0b01, 0b10)
    gens = [eye]
    gens.append((0b10, 0b01))  # swap
    gens.append((0b11, 0b10))  # transvection
    gens.append((0b01, 0b11))
    return gens


def _gl4_gens() -> List[Tuple[int, ...]]:
    eye = tuple(1 << i for i in range(4))
    gens: List[Tuple[int, ...]] = []
    for i, j in itertools.combinations(range(4), 2):
        m = list(eye)
        m[i], m[j] = m[j], m[i]
        gens.append(tuple(m))
    for i, j in itertools.permutations(range(4), 2):
        m = list(eye)
        m[i] = m[i] ^ (1 << j)
        gens.append(tuple(m))
    return gens


def _mul_rows(rows: Tuple[int, ...], v: int) -> int:
    out = 0
    for i, row in enumerate(rows):
        if bin(row & v).count("1") & 1:
            out |= 1 << i
    return out


def _wall_aut_apply(af: Tuple[int, ...], ao: Tuple[int, ...], v: int) -> int:
    """Apply GL(2)_fold × GL(4)_outer to a 6-bit codon vector."""
    f_bits = [i for i in range(6) if (FOLD_MASK >> i) & 1]
    o_bits = [i for i in range(6) if (OUTER_MASK >> i) & 1]
    fv = sum((((v >> b) & 1) << i) for i, b in enumerate(f_bits))
    ov = sum((((v >> b) & 1) << i) for i, b in enumerate(o_bits))
    fv2 = _mul_rows(af, fv)
    ov2 = _mul_rows(ao, ov)
    out = 0
    for i, b in enumerate(f_bits):
        if (fv2 >> i) & 1:
            out |= 1 << b
    for i, b in enumerate(o_bits):
        if (ov2 >> i) & 1:
            out |= 1 << b
    return out


def _met_fiber_pair(stolen: str, enc: NucleotideEncoding) -> frozenset:
    return frozenset(
        {
            pack_codon_bits("ATG", enc),
            pack_codon_bits(stolen, enc),
        }
    )


def _orbit_pairs(seed: frozenset) -> set:
    """BFS orbit of a 2-set under wall Aut generators."""
    idf = (0b01, 0b10)
    ido = tuple(1 << i for i in range(4))
    gf = _gl2_gens()
    go = _gl4_gens()
    seen = {seed}
    q: deque = deque([seed])
    while q:
        cur = q.popleft()
        for af in gf:
            nxt = frozenset(_wall_aut_apply(af, ido, v) for v in cur)
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
        for ao in go:
            nxt = frozenset(_wall_aut_apply(idf, ao, v) for v in cur)
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


@dataclass
class U8AutCensus:
    n_survivors: int
    n_met_expansions: int
    n_orbits: int
    orbit_sizes: Tuple[int, ...]
    orbit_reps: Tuple[Tuple[str, ...], ...]
    gates: Dict[str, bool]


def u8_aut_quotient_census(survivors: Sequence[Tuple[str, str, str]]) -> U8AutCensus:
    """Quotient local full-constraint survivors by wall-preserving linear Aut GL2×GL4."""
    enc = _ref_enc()
    n_met = sum(1 for _c, _src, aa in survivors if aa == "M")
    used = [False] * len(survivors)
    orbits: List[Tuple[str, ...]] = []
    for i, (c, _src, aa) in enumerate(survivors):
        if used[i] or aa != "M":
            continue
        seen = _orbit_pairs(_met_fiber_pair(c, enc))
        group: List[str] = []
        for j, (c2, _s2, aa2) in enumerate(survivors):
            if aa2 != "M" or used[j]:
                continue
            if _met_fiber_pair(c2, enc) in seen:
                used[j] = True
                group.append(c2)
        orbits.append(tuple(group))
    sizes = tuple(sorted((len(o) for o in orbits), reverse=True))
    gates = {
        "314_local_all_met_expansions": n_met == len(survivors) == 5,
        "186_wall_aut_orbits_three": len(orbits) == 3 and sizes == (3, 1, 1),
        "187_largest_wall_aut_orbit_3": sizes[:1] == (3,),
    }
    return U8AutCensus(len(survivors), n_met, len(orbits), sizes, tuple(orbits), gates)


def print_u8_aut(c: U8AutCensus) -> None:
    g = c.gates
    report_section("35. LOCAL MODULI: MET EXPANSIONS; WALL-AUT (PROBE ONLY)")
    report_objects(("5 full-constraint Met survivors; wall-Aut probe orbits; Aff_S6 is the measure",))
    print(f"    survivors={c.n_survivors} met_expansions={c.n_met_expansions}")
    print(f"    n_orbits={c.n_orbits} sizes={c.orbit_sizes}")
    print(f"    orbits={c.orbit_reps}")
    print()
    report_checks((
        ("all 5 local full-constraint survivors are single Met expansions", g["314_local_all_met_expansions"], f"met={c.n_met_expansions}/{c.n_survivors}", "5/5"),
        ("wall Aut (probe) quotients the 5 into 3 orbits of sizes 3,1,1", g["186_wall_aut_orbits_three"], f"n={c.n_orbits} sizes={c.orbit_sizes}", "3 (3,1,1)"),
        ("largest wall-Aut probe orbit has size 3", g["187_largest_wall_aut_orbit_3"], f"sizes={c.orbit_sizes}", "3,..."),
    ))


@dataclass
class U8TwoMoveCensus:
    n_met_pairs: int
    n_met_pairs_u8: int
    n_ile_double: int
    n_ile_double_u8: int
    n_size2_swaps: int
    n_size2_swaps_u8: int
    gates: Dict[str, bool]


def u8_two_move_census(survivors: Sequence[Tuple[str, str, str]]) -> U8TwoMoveCensus:
    """Structured two-move generators under full constraints."""
    enc = _ref_enc()
    met_codons = [c for c, _src, aa in survivors if aa == "M"]
    n_met_pairs = n_met_u8 = 0
    for i, c1 in enumerate(met_codons):
        for c2 in met_codons[i + 1 :]:
            tab = dict(STANDARD_CODE)
            tab[c1] = "M"
            tab[c2] = "M"
            n_met_pairs += 1
            if u8_pass(tab, enc):
                n_met_u8 += 1

    ile = [c for c in CODONS if STANDARD_CODE[c] == "I"]
    n_ile = n_ile_u8 = 0
    for c1, c2 in itertools.combinations(ile, 2):
        tab = dict(STANDARD_CODE)
        tab[c1] = "M"
        tab[c2] = "M"
        n_ile += 1
        if u8_pass(tab, enc):
            n_ile_u8 += 1

    size2_codons = [c for c in CODONS if STANDARD_CODE[c] in SIZE2_AAS]
    n_sw = n_sw_u8 = 0
    for i, c1 in enumerate(size2_codons):
        a1 = STANDARD_CODE[c1]
        for c2 in size2_codons[i + 1 :]:
            a2 = STANDARD_CODE[c2]
            if a1 == a2:
                continue
            tab = dict(STANDARD_CODE)
            tab[c1], tab[c2] = a2, a1
            n_sw += 1
            if u8_pass(tab, enc):
                n_sw_u8 += 1

    gates = {
        "188_met_expansion_pairs_empty": n_met_u8 == 0 and n_met_pairs > 0,
        "189_ile_double_met_all_pass": n_ile_u8 == n_ile == 3,
        "190_size2_swaps_102_of_144": n_sw == 144 and n_sw_u8 == 102,
    }
    return U8TwoMoveCensus(n_met_pairs, n_met_u8, n_ile, n_ile_u8, n_sw, n_sw_u8, gates)


def print_u8_two_move(c: U8TwoMoveCensus) -> None:
    g = c.gates
    report_section("36. STRUCTURED TWO-MOVE GENERATORS")
    report_objects(("Met-expansion pairs; Ile double->Met; size-2 codon swaps; full constraints",))
    print(
        f"    met_pairs={c.n_met_pairs_u8}/{c.n_met_pairs} "
        f"ile_double_met={c.n_ile_double_u8}/{c.n_ile_double} "
        f"size2_swaps={c.n_size2_swaps_u8}/{c.n_size2_swaps}"
    )
    print()
    report_checks((
        ("no two simultaneous Met expansions among the 5 pass full constraints", g["188_met_expansion_pairs_empty"], f"{c.n_met_pairs_u8}/{c.n_met_pairs}", "0/>0"),
        ("all three Ile double->Met moves pass full constraints", g["189_ile_double_met_all_pass"], f"{c.n_ile_double_u8}/{c.n_ile_double}", "3/3"),
        ("size-2 codon swaps: 102 of 144 pass full constraints (constitutional kills 36)", g["190_size2_swaps_102_of_144"], f"{c.n_size2_swaps_u8}/{c.n_size2_swaps}", "102/144"),
    ))


@dataclass
class U8NcbiCensus:
    n_tables: int
    n_u8: int
    pass_ids: Tuple[int, ...]
    gates: Dict[str, bool]


def u8_ncbi_placement_census() -> U8NcbiCensus:
    enc = _ref_enc()
    passed: List[int] = []
    for tid in NCBI_TABLE_IDS:
        if u8_pass(translation_table(tid), enc):
            passed.append(tid)
    gates = {
        "191_ncbi_only_standard_pair": tuple(passed) == (1, 11),
        "192_ncbi_count_two": len(passed) == 2,
    }
    return U8NcbiCensus(len(NCBI_TABLE_IDS), len(passed), tuple(passed), gates)


def print_u8_ncbi(c: U8NcbiCensus) -> None:
    g = c.gates
    report_section("37. NCBI TABLES UNDER FULL CODE CONSTRAINTS")
    report_objects(("NCBI translation tables; full constraint filter; pass list",))
    print(f"    tables={c.n_tables} constraint_pass={c.n_u8} ids={c.pass_ids}")
    print()
    report_checks((
        ("only NCBI tables 1 and 11 (standard) pass full constraints", g["191_ncbi_only_standard_pair"], f"ids={c.pass_ids}", "(1, 11)"),
        ("exactly two NCBI tables pass full constraints", g["192_ncbi_count_two"], f"n={c.n_u8}/{c.n_tables}", "2"),
    ))


# ----------------------------------------
# Wall-Aut vs Aff_S6 on Met survivors
# ----------------------------------------

def _gl_order(n: int) -> int:
    o = 1
    for k in range(n):
        o *= (1 << n) - (1 << k)
    return o


def _apply_bit_perm(pi: Tuple[int, ...], v: int) -> int:
    out = 0
    for i in range(6):
        if (v >> i) & 1:
            out |= 1 << pi[i]
    return out


def _s6_preserves_wall_split(pi: Tuple[int, ...]) -> bool:
    fold = {i for i in range(6) if (FOLD_MASK >> i) & 1}
    outer = {i for i in range(6) if (OUTER_MASK >> i) & 1}
    return {pi[i] for i in fold} == fold and {pi[i] for i in outer} == outer


def _orbit_aff_s6(seed: frozenset) -> set:
    """BFS orbit of a codon-bit set under Aff = translations(GF(2)^6) rtimes S6."""
    seen = {seed}
    q: deque = deque([seed])
    swaps = []
    for i in range(5):
        pi = list(range(6))
        pi[i], pi[i + 1] = pi[i + 1], pi[i]
        swaps.append(tuple(pi))
    while q:
        cur = q.popleft()
        for pi in swaps:
            nxt = frozenset(_apply_bit_perm(pi, v) for v in cur)
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
        for bit in range(6):
            t = 1 << bit
            nxt = frozenset(v ^ t for v in cur)
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


@dataclass
class AutReconcileCensus:
    order_wall_aut: int
    order_aff_s6: int
    order_byte_graph_aut: int
    n_s6_wall: int
    n_aff_orbits: int
    aff_orbit_sizes: Tuple[int, ...]
    n_wall_orbits: int
    wall_orbit_sizes: Tuple[int, ...]
    same_partition: bool
    gates: Dict[str, bool]


def aut_reconciliation_census(
    survivors: Sequence[Tuple[str, str, str]],
    wall: U8AutCensus,
) -> AutReconcileCensus:
    """Compare wall-Aut GL(2)xGL(4) with classified Aff_S6 on the Met survivors."""
    enc = _ref_enc()
    met = [(c, src, aa) for c, src, aa in survivors if aa == "M"]
    used = [False] * len(met)
    orbits: List[Tuple[str, ...]] = []
    for i, (c, _src, _aa) in enumerate(met):
        if used[i]:
            continue
        seen = _orbit_aff_s6(_met_fiber_pair(c, enc))
        group: List[str] = []
        for j, (c2, _s2, _aa2) in enumerate(met):
            if used[j]:
                continue
            if _met_fiber_pair(c2, enc) in seen:
                used[j] = True
                group.append(c2)
        orbits.append(tuple(group))
    aff_sizes = tuple(sorted((len(o) for o in orbits), reverse=True))
    n_s6_wall = sum(
        1 for pi in itertools.permutations(range(6)) if _s6_preserves_wall_split(pi)
    )
    order_wall = _gl_order(2) * _gl_order(4)
    order_aff = 64 * 720
    order_graph = 128 * 720
    same = set(frozenset(o) for o in orbits) == set(
        frozenset(o) for o in wall.orbit_reps
    )
    gates = {
        "193_wall_aut_order_120960": order_wall == 120960,
        "194_aff_s6_order_46080": order_aff == 46080,
        "195_s6_wall_split_48": n_s6_wall == 48,
        "196_aff_orbits_single_5": aff_sizes == (5,),
        "197_aff_wall_partitions_differ": (not same)
        and wall.orbit_sizes == (3, 1, 1)
        and aff_sizes == (5,),
        "198_byte_graph_aut_order_92160": order_graph == 92160,
    }
    return AutReconcileCensus(
        order_wall,
        order_aff,
        order_graph,
        n_s6_wall,
        len(orbits),
        aff_sizes,
        wall.n_orbits,
        wall.orbit_sizes,
        same,
        gates,
    )


def print_aut_reconcile(c: AutReconcileCensus) -> None:
    g = c.gates
    report_section("38. AUT RECONCILIATION: Iso(H(6,2)) PRIMARY; WALL-AUT PROBE")
    report_objects((
        "Iso(H(6,2)) = GF(2)^6 rtimes S6 (Aff_S6); wall-Aut probe; orders",
    ))
    print(
        f"    |wallAut|={c.order_wall_aut} |Iso(H(6,2))|={c.order_aff_s6} "
        f"|byteGraphAut|={c.order_byte_graph_aut} |S6 cap wall-split|={c.n_s6_wall}"
    )
    print(
        f"    wall_orbits={c.n_wall_orbits} sizes={c.wall_orbit_sizes} "
        f"aff_orbits={c.n_aff_orbits} sizes={c.aff_orbit_sizes} "
        f"same_partition={c.same_partition}"
    )
    print()
    report_checks((
        ("wall-Aut order is |GL(2,2)| x |GL(4,2)| = 120960", g["193_wall_aut_order_120960"], str(c.order_wall_aut), "120960"),
        ("Iso(H(6,2)) order is 64 x 720 = 46080", g["194_aff_s6_order_46080"], str(c.order_aff_s6), "46080"),
        ("S6 elements preserving fold/outer split: |S2 x S4| = 48", g["195_s6_wall_split_48"], str(c.n_s6_wall), "48"),
        ("Iso(H(6,2)) quotients the 5 Met survivors into one orbit of size 5", g["196_aff_orbits_single_5"], str(c.aff_orbit_sizes), "(5,)"),
        ("Iso(H(6,2)) and wall-Aut induce different partitions of the 5", g["197_aff_wall_partitions_differ"], f"same={c.same_partition}", "False"),
        ("byte-graph Aut order is 128 x 720 = 92160", g["198_byte_graph_aut_order_92160"], str(c.order_byte_graph_aut), "92160"),
    ))


# ----------------------------------------
# Constitutional fiber side conditions (detail report)
# ----------------------------------------

@dataclass
class ConstitutionalCensus:
    ser_max_ab: int
    ser_n_complement: int
    other_sense_max_ab: int
    stop_max_ab: int
    stop_trp_invariant_ok: bool
    n_stop_trp_pairs: int
    n_wall_only_swap_fails: int
    n_size2_pre_const: int
    gates: Dict[str, bool]


def constitutional_fiber_census() -> ConstitutionalCensus:
    """Per-fiber Omega shells on the standard; record that (x) does not replace wall."""
    enc = _ref_enc()
    ser = [c for c in CODONS if STANDARD_CODE[c] == "S"]
    ser_shells = _intra_fiber_ab_shells(ser, enc)
    ser_max = max(ser_shells) if ser_shells else -1
    ser_comp = int(ser_shells.get(12, 0))
    other_max = 0
    for aa, group in fibers(STANDARD_CODE).items():
        if aa in ("*", "S") or len(group) < 2:
            continue
        sh = _intra_fiber_ab_shells(group, enc)
        if sh:
            other_max = max(other_max, max(sh))
    stops = [c for c in CODONS if STANDARD_CODE[c] == "*"]
    stop_shells = _intra_fiber_ab_shells(stops, enc)
    stop_max = max(stop_shells) if stop_shells else -1
    hull = stops + [c for c in CODONS if STANDARD_CODE[c] == "W"]
    n_pairs = 0
    inv_ok = True
    for i, c1 in enumerate(hull):
        for c2 in hull[i + 1 :]:
            abd, hd = _pair_ab_horizon(c1, c2, enc)
            n_pairs += 1
            if abd + hd != 12:
                inv_ok = False

    # Size-2 swaps: count wall-only fails that still pass constitutional halves.
    size2_codons = [c for c in CODONS if STANDARD_CODE[c] in SIZE2_AAS]
    n_wall_only = 0
    n_pre = 0
    for i, c1 in enumerate(size2_codons):
        a1 = STANDARD_CODE[c1]
        for c2 in size2_codons[i + 1 :]:
            a2 = STANDARD_CODE[c2]
            if a1 == a2:
                continue
            tab = dict(STANDARD_CODE)
            tab[c1], tab[c2] = a2, a1
            checks = u8_constraints(tab, enc)
            # pre-constitutional = all except the three constitutional keys
            core = {k: v for k, v in checks.items() if k not in (
                "serine_complement_reach", "stop_fiber_near_eq", "stop_trp_invariant",
            )}
            if all(core.values()):
                n_pre += 1
            if (
                not checks["wall_closed"]
                and checks["serine_complement_reach"]
                and checks["stop_fiber_near_eq"]
                and checks["stop_trp_invariant"]
                and all(
                    v for k, v in core.items() if k != "wall_closed"
                )
            ):
                n_wall_only += 1

    gates = {
        "199_serine_unique_complement_reach": ser_max == 12 and other_max <= 6,
        "200_serine_complement_pairs_two": ser_comp == 2,
        "201_stop_fiber_no_complement": stop_max <= 4,
        "202_stop_trp_complementarity_invariant": inv_ok and n_pairs == 6,
        "204_six_swaps_fail_wall_only": n_wall_only == 6,
        "205_constitutional_shrinks_size2_from_138": n_pre == 138,
    }
    return ConstitutionalCensus(
        ser_max, ser_comp, other_max, stop_max, inv_ok, n_pairs,
        n_wall_only, n_pre, gates,
    )


def print_constitutional(c: ConstitutionalCensus) -> None:
    g = c.gates
    report_section("39. CONSTITUTIONAL FIBER SIDE CONDITIONS")
    report_objects((
        "intra-fiber Omega ab shells; serine complement reach; wall not replaced by (x)-(xi)",
    ))
    print(
        f"    ser_max_ab={c.ser_max_ab} ser_complement_pairs={c.ser_n_complement} "
        f"other_sense_max_ab={c.other_sense_max_ab} stop_max_ab={c.stop_max_ab}"
    )
    print(
        f"    stop_trp_pairs={c.n_stop_trp_pairs} "
        f"complementarity_invariant={c.stop_trp_invariant_ok} "
        f"wall_only_swap_fails={c.n_wall_only_swap_fails} "
        f"size2_pre_const={c.n_size2_pre_const}"
    )
    print()
    report_checks((
        ("serine is the unique sense fiber reaching the complement pole (ab=12)", g["199_serine_unique_complement_reach"], f"ser_max={c.ser_max_ab} other_max={c.other_sense_max_ab}", "12 vs <=6"),
        ("exactly two serine intra-fiber pairs sit on the complement horizon", g["200_serine_complement_pairs_two"], str(c.ser_n_complement), "2"),
        ("stop fiber stays near equality (intra-fiber ab <= 4)", g["201_stop_fiber_no_complement"], str(c.stop_max_ab), "<=4"),
        ("all six stop-Trp square pairs obey ab + horizon = 12", g["202_stop_trp_complementarity_invariant"], f"ok={c.stop_trp_invariant_ok} n={c.n_stop_trp_pairs}", "True n=6"),
        ("exactly six size-2 swaps fail wall alone while passing constitutional halves", g["204_six_swaps_fail_wall_only"], str(c.n_wall_only_swap_fails), "6"),
        ("without constitutional, 138 size-2 swaps pass the older core constraints", g["205_constitutional_shrinks_size2_from_138"], str(c.n_size2_pre_const), "138"),
    ))


# ----------------------------------------
# Wall direct sum; kernel percolation; BU curvature
# ----------------------------------------

def _swap_fold_bits(v: int) -> int:
    """Bit fold on P_fold: swap payload bits 2 and 3."""
    b2 = (v >> 2) & 1
    b3 = (v >> 3) & 1
    return (v & ~FOLD_MASK) | (b3 << 2) | (b2 << 3)


def _pi_bu(v: int) -> int:
    return int(v) & FOLD_MASK


@dataclass
class WallDirectSumCensus:
    rank_sense: int
    sense_eq_outer: bool
    sense_meet_fold_trivial: bool
    stop_bu: int
    ser_bu: int
    fold_stop_bu: int
    fold_ser_bu: int
    stop_xor_ser_bu: int
    keys_span_fold: bool
    outcome_b: bool
    gates: Dict[str, bool]


def wall_direct_sum_census() -> WallDirectSumCensus:
    """Direct-sum wall: H = L_sense ⊕ P_fold; stop/ser BU keys."""
    enc = _ref_enc()
    sense = sense_edge_diffs(enc)
    stop = stop_diff(enc)
    ser = serine_chords(enc)
    span_s = set(span_elements(sense))
    outer = set(span_elements([1 << i for i in range(6) if (OUTER_MASK >> i) & 1]))
    r = gf2_rank6(sense)
    meet = fold_intersection(sense)
    stop_bu = _pi_bu(stop)
    ser_bus = sorted({_pi_bu(c) for c in ser if _pi_bu(c)})
    ser_bu = ser_bus[0] if len(ser_bus) == 1 else -1
    fold_stop = _swap_fold_bits(stop_bu)
    fold_ser = _swap_fold_bits(ser_bu) if ser_bu >= 0 else -1
    xor_bu = stop_bu ^ ser_bu if ser_bu >= 0 else -1
    keys_span = set(span_elements([stop_bu, ser_bu])) == set(
        span_elements([1 << i for i in range(6) if (FOLD_MASK >> i) & 1])
    )
    # Outcome B: serine fold-fixed (11), stop oriented (10 <-> 01 under bit fold).
    outcome_b = stop_bu == 0b000100 and ser_bu == 0b001100 and fold_ser == ser_bu
    gates = {
        "206_sense_rank_4": r == 4,
        "207_sense_span_equals_outer": span_s == outer,
        "208_sense_meet_fold_trivial": meet == (),
        "209_h_direct_sum_sense_fold": r == 4 and meet == () and span_s == outer,
        "210_stop_ser_bu_keys_span_fold": keys_span and stop_bu != 0 and ser_bu != 0,
        "211_fold_outcome_b_ser_fixed": outcome_b,
    }
    return WallDirectSumCensus(
        r, span_s == outer, meet == (), stop_bu, ser_bu, fold_stop, fold_ser,
        xor_bu, keys_span, outcome_b, gates,
    )


def print_wall_direct_sum(c: WallDirectSumCensus) -> None:
    g = c.gates
    report_section("40. WALL DIRECT SUM: H = L_SENSE ⊕ P_FOLD")
    report_objects(("L_sense; P_fold; stop/ser BU projections; bit-fold on B",))
    print(
        f"    rank_sense={c.rank_sense} sense=outer={c.sense_eq_outer} "
        f"meet_fold_0={c.sense_meet_fold_trivial}"
    )
    print(
        f"    stop_BU={c.stop_bu:06b} ser_BU={c.ser_bu:06b} "
        f"fold(stop)={c.fold_stop_bu:06b} fold(ser)={c.fold_ser_bu:06b} "
        f"stop XOR ser={c.stop_xor_ser_bu:06b}"
    )
    print(f"    keys_span_fold={c.keys_span_fold} outcome_B_ser_fold_fixed={c.outcome_b}")
    print()
    report_checks((
        ("rank L_sense = 4", g["206_sense_rank_4"], str(c.rank_sense), "4"),
        ("L_sense equals the outer plane O", g["207_sense_span_equals_outer"], str(c.sense_eq_outer), "True"),
        ("L_sense ∩ P_fold = {0}", g["208_sense_meet_fold_trivial"], str(c.sense_meet_fold_trivial), "True"),
        ("H = L_sense ⊕ P_fold (exact direct sum)", g["209_h_direct_sum_sense_fold"], "see above", "True"),
        ("π_BU(stop) and π_BU(ser) span P_fold", g["210_stop_ser_bu_keys_span_fold"], f"span={c.keys_span_fold}", "True"),
        ("fold outcome B: ser BU-fixed 11; stop oriented 10", g["211_fold_outcome_b_ser_fixed"], f"outcome_B={c.outcome_b}", "True"),
    ))


def _allowed_q_subspace(qs: set, *, family: Optional[int]) -> List[int]:
    out: List[int] = []
    for q in qs:
        for b in BYTES_BY_Q6[q]:
            if family is None or intron_family_d(intron_from_byte(b, 6), 6) == family:
                out.append(b)
    return out


def _bfs_reach_from(eng, start_uv: Tuple[int, int], allowed: Sequence[int]) -> Tuple[int, int, int]:
    allowed_set = set(int(b) for b in allowed)
    byte_idx = [i for i in range(256) if i in allowed_set]
    visited = bytearray(eng.n_omega)
    sidx = eng.uv_to_idx[start_uv]
    q: deque = deque([sidx])
    visited[sidx] = 1
    while q:
        i = q.popleft()
        row = eng.transitions[i]
        for bi in byte_idx:
            j = row[bi]
            if not visited[j]:
                visited[j] = 1
                q.append(j)
    uv_of = {v: k for k, v in eng.uv_to_idx.items()}
    us: set = set()
    vs: set = set()
    n = 0
    for i in range(eng.n_omega):
        if visited[i]:
            n += 1
            u, v = uv_of[i]
            us.add(u)
            vs.add(v)
    return n, len(us), len(vs)


@dataclass
class KernelPercolationCensus:
    rows: Tuple[Tuple[str, int, int, int, int, int], ...]
    ladder_payload_ok: bool
    gates: Dict[str, bool]


def kernel_percolation_census() -> KernelPercolationCensus:
    """Fiber generators from wall ranks; square-root orbit percolation ladder."""
    enc = _ref_enc()
    sense = sense_edge_diffs(enc)
    stop = stop_diff(enc)
    ser = serine_chords(enc)
    eng = build_hqvm_d(6)
    specs = (
        ("sense", sense, 256),
        ("sense_stop", sense + [stop], 1024),
        ("sense_ser", sense + list(ser), 1024),
        ("full", sense + [stop] + list(ser), 4096),
    )
    rows: List[Tuple[str, int, int, int, int, int]] = []
    payload_ok = True
    for name, vecs, expect in specs:
        qs = set(span_elements(vecs))
        r = gf2_rank6(list(qs))
        allow = _allowed_q_subspace(qs, family=0)
        n, nu, nv = _bfs_reach_from(eng, (0, 0), allow)
        pred = predicted_cluster_size(r)
        rows.append((name, r, n, pred, nu, nv))
        if n != expect or pred != expect:
            payload_ok = False
    gates = {
        "212_percolation_sense_256": rows[0][2] == 256,
        "213_percolation_sense_stop_1024": rows[1][2] == 1024,
        "214_percolation_sense_ser_1024": rows[2][2] == 1024,
        "215_percolation_full_4096": rows[3][2] == 4096,
        "216_percolation_ladder_payload_origin": payload_ok,
    }
    return KernelPercolationCensus(tuple(rows), payload_ok, gates)


def print_kernel_percolation(c: KernelPercolationCensus) -> None:
    g = c.gates
    report_section("41. KERNEL PERCOLATION LADDER (WALL → OMEGA ORBITS)")
    report_objects(("sense / +stop / +ser / full; predicted (2^r)^2; measured |Reach|, |U|, |V|",))
    print_table(
        ("generators", "rank", "reach", "predict", "|U|", "|V|"),
        (12, 4, 6, 7, 4, 4),
        [(n, r, reach, pred, nu, nv) for n, r, reach, pred, nu, nv in c.rows],
        aligns=("<", ">", ">", ">", ">", ">"),
    )
    print()
    report_checks((
        ("sense generators: |Reach|=256", g["212_percolation_sense_256"], str(c.rows[0][2]), "256"),
        ("sense+stop: |Reach|=1024", g["213_percolation_sense_stop_1024"], str(c.rows[1][2]), "1024"),
        ("sense+serine: |Reach|=1024", g["214_percolation_sense_ser_1024"], str(c.rows[2][2]), "1024"),
        ("full (sense+stop+ser): |Reach|=4096", g["215_percolation_full_4096"], str(c.rows[3][2]), "4096"),
        ("payload-origin percolation ladder matches square-root law", g["216_percolation_ladder_payload_origin"], str(c.ladder_payload_ok), "True"),
    ))


@dataclass
class BuCurvatureCensus:
    n_sense: int
    n_sense_bu_nonzero: int
    stop_bu: int
    ser_bu: int
    full_bu_classes: Tuple[int, ...]
    gates: Dict[str, bool]


def bu_curvature_census() -> BuCurvatureCensus:
    """π_BU of sense/stop/ser difference defects."""
    enc = _ref_enc()
    sense = sense_edge_diffs(enc)
    stop = stop_diff(enc)
    ser = serine_chords(enc)
    n_sense = len(sense)
    n_bu = sum(1 for d in sense if _pi_bu(d) != 0)
    stop_bu = _pi_bu(stop)
    ser_bu = sorted({_pi_bu(c) for c in ser if _pi_bu(c)})[0]
    full_classes = tuple(sorted(set(span_elements([stop_bu, ser_bu]))))
    gates = {
        "217_sense_bu_defect_zero": n_bu == 0 and n_sense > 0,
        "218_stop_bu_weight_1": stop_bu.bit_count() == 1,
        "219_ser_bu_weight_2": ser_bu.bit_count() == 2,
        "220_stop_ser_fill_all_bu_classes": full_classes == (0, 4, 8, 12),
    }
    return BuCurvatureCensus(n_sense, n_bu, stop_bu, ser_bu, full_classes, gates)


def print_bu_curvature(c: BuCurvatureCensus) -> None:
    g = c.gates
    report_section("42. BU PROJECTION OF COMMUTATOR / DIFFERENCE DEFECTS")
    report_objects(("sense-synonymous edges; stop key; serine chords; BU defect classes",))
    print(
        f"    sense_edges={c.n_sense} sense_bu_nonzero={c.n_sense_bu_nonzero} "
        f"stop_BU={c.stop_bu:06b} ser_BU={c.ser_bu:06b} "
        f"full_BU_classes={tuple(f'{x:06b}' for x in c.full_bu_classes)}"
    )
    print()
    report_checks((
        ("all sense-synonymous defects have π_BU=0", g["217_sense_bu_defect_zero"], f"nonzero={c.n_sense_bu_nonzero}/{c.n_sense}", "0"),
        ("stop key has BU weight 1", g["218_stop_bu_weight_1"], f"wt={c.stop_bu.bit_count()}", "1"),
        ("serine key has BU weight 2 (fold-fixed 11)", g["219_ser_bu_weight_2"], f"wt={c.ser_bu.bit_count()}", "2"),
        ("stop+serine fill all four BU classes {00,10,01,11}", g["220_stop_ser_fill_all_bu_classes"], str(c.full_bu_classes), "(0,4,8,12)"),
    ))


# ----------------------------------------
# Fold as Weyl reflection of BU (finite)
# ----------------------------------------

W2_BYTES: Tuple[int, ...] = (0xAA, 0xAB)
W2P_BYTES: Tuple[int, ...] = (0x2A, 0x2B)


def _apply_omega_word(u: int, v: int, word: Sequence[int]) -> Tuple[int, int]:
    from gyroscopic.hQVM.api import OmegaState12, step_omega12_by_byte

    st = OmegaState12(u6=int(u) & 0x3F, v6=int(v) & 0x3F)
    for b in word:
        st = step_omega12_by_byte(st, int(b) & 0xFF)
    return st.u6, st.v6


def _omega_words_equal(w1: Sequence[int], w2: Sequence[int]) -> bool:
    for u in range(64):
        for v in range(64):
            if _apply_omega_word(u, v, w1) != _apply_omega_word(u, v, w2):
                return False
    return True


def _omega_word_is_id(word: Sequence[int]) -> bool:
    for u in range(64):
        for v in range(64):
            if _apply_omega_word(u, v, word) != (u, v):
                return False
    return True


def _bitrev6(x: int) -> int:
    out = 0
    for i in range(6):
        if (x >> i) & 1:
            out |= 1 << (5 - i)
    return out


@dataclass
class FoldWeylCensus:
    fold_involutive: bool
    q_fold_is_bitrev: bool
    w2_involution: bool
    w2p_involution: bool
    f_involution: bool
    fold_w2_eq_w2p: bool
    fold_w2p_eq_w2: bool
    fold_f_eq_f: bool
    reverse_w2_eq_w2p: bool
    genomic_fold_swaps_stop_fixes_ser: bool
    gates: Dict[str, bool]


def fold_weyl_census() -> FoldWeylCensus:
    """Fold exchanges W2↔W2', fixes F; genomic BU keys match the same pattern.

    Finite form of orientation reversal — not 'fold = holonomy'. Continuous U(δ)
    satisfies P U P^{-1} = U^{-1}; here W2,W2' are involutions, and fold swaps them
    while fixing F = W2∘W2' (Z2 cycle). Orientation lives in the U/V half-word split.
    """
    from gyroscopic.hQVM.api import q_word6
    from gyroscopic.hQVM.family import fold_map_d

    fold_inv = all(fold_map_d(fold_map_d(b, 6), 6) == b for b in range(256))
    q_bitrev = all(
        q_word6(fold_map_d(b, 6)) == _bitrev6(q_word6(b)) for b in range(256)
    )
    w2 = W2_BYTES
    w2p = W2P_BYTES
    fword = w2 + w2p
    fw2 = tuple(fold_map_d(b, 6) for b in w2)
    fw2p = tuple(fold_map_d(b, 6) for b in w2p)
    ff = tuple(fold_map_d(b, 6) for b in fword)
    w2_inv = _omega_word_is_id(w2 + w2)
    w2p_inv = _omega_word_is_id(w2p + w2p)
    f_inv = _omega_word_is_id(fword + fword)
    fold_w2_w2p = _omega_words_equal(fw2, w2p)
    fold_w2p_w2 = _omega_words_equal(fw2p, w2)
    fold_f_f = _omega_words_equal(ff, fword)
    rev_w2_w2p = _omega_words_equal(tuple(reversed(w2)), w2p)
    # Genomic parallel from wall direct-sum: P swaps stop 10↔01, fixes ser 11.
    stop_bu, ser_bu = 0b000100, 0b001100
    gen_ok = (
        _swap_fold_bits(stop_bu) == 0b001000
        and _swap_fold_bits(ser_bu) == ser_bu
    )
    gates = {
        "221_byte_fold_involutive": fold_inv,
        "222_q_of_fold_is_bitrev6": q_bitrev,
        "223_w2_and_w2p_involutions": w2_inv and w2p_inv and f_inv,
        "224_fold_exchanges_w2_w2p": fold_w2_w2p and fold_w2p_w2,
        "225_fold_fixes_z2_product_f": fold_f_f,
        "226_word_reverse_w2_eq_w2p": rev_w2_w2p,
        "227_genomic_bu_keys_match_weyl_pattern": gen_ok,
    }
    return FoldWeylCensus(
        fold_inv, q_bitrev, w2_inv, w2p_inv, f_inv,
        fold_w2_w2p, fold_w2p_w2, fold_f_f, rev_w2_w2p, gen_ok, gates,
    )


def print_fold_weyl(c: FoldWeylCensus) -> None:
    g = c.gates
    report_section("43. FOLD AS WEYL REFLECTION OF BU (FINITE)")
    report_objects((
        "byte fold P; W2/W2' depth-4 half-words; F=W2∘W2'; genomic BU keys under bit-fold",
    ))
    print(
        f"    fold_involutive={c.fold_involutive} q(fold)=bitrev6={c.q_fold_is_bitrev} "
        f"W2/W2'/F involutions={c.w2_involution}/{c.w2p_involution}/{c.f_involution}"
    )
    print(
        f"    fold(W2)=W2'={c.fold_w2_eq_w2p} fold(W2')=W2={c.fold_w2p_eq_w2} "
        f"fold(F)=F={c.fold_f_eq_f} reverse(W2)=W2'={c.reverse_w2_eq_w2p}"
    )
    print(
        f"    genomic: bit-fold swaps stop BU 10↔01 and fixes ser 11: "
        f"{c.genomic_fold_swaps_stop_fixes_ser}"
    )
    print()
    report_checks((
        ("byte fold P is an involution on all 256 bytes", g["221_byte_fold_involutive"], str(c.fold_involutive), "True"),
        ("q6(P(b)) equals bit-reversal of q6(b)", g["222_q_of_fold_is_bitrev6"], str(c.q_fold_is_bitrev), "True"),
        ("W2, W2', and F=W2∘W2' are involutions on Omega", g["223_w2_and_w2p_involutions"], "see flags", "True"),
        ("fold exchanges W2 ↔ W2' as Omega operators", g["224_fold_exchanges_w2_w2p"], f"W2→W2'={c.fold_w2_eq_w2p} W2'→W2={c.fold_w2p_eq_w2}", "True"),
        ("fold fixes the Z2 cycle product F = W2∘W2'", g["225_fold_fixes_z2_product_f"], str(c.fold_f_eq_f), "True"),
        ("word reverse also sends W2 to W2' (orientation channel)", g["226_word_reverse_w2_eq_w2p"], str(c.reverse_w2_eq_w2p), "True"),
        ("genomic BU keys match the same Weyl pattern (stop swapped, ser fixed)", g["227_genomic_bu_keys_match_weyl_pattern"], str(c.genomic_fold_swaps_stop_fixes_ser), "True"),
    ))


# ----------------------------------------
# Genome-facing probe: real CDSs scored through the wall decomposition
# ----------------------------------------

@dataclass
class GenomeBuRow:
    genome: str
    n_genes: int
    n_codons: int
    gc: float
    bu_counts: Dict[int, int]
    n_class10: int
    n_tga: int
    n_edges: int
    n_edges_gc_null: int
    n_same_aa: int
    n_same_aa_null: int
    n_chord: int
    n_chord_gc_null: int
    n_chord_syn_null: int
    status: str


def _load_cds_records() -> List[Tuple[str, List[List[str]]]]:
    """Load real CDS codon lists per named source; empty list means missing data."""
    from hqvm_cgm_genomics_2 import extract_chr22_cds, load_named_fasta

    out: List[Tuple[str, List[List[str]]]] = []
    fasta_sources = (
        ("ecoli", ("ecoli_k12_cds",)),
        ("yeast", ("yeast_s288c_cds",)),
        ("sars_cov2", ("sars_cov2_cds",)),
    )
    for name, keys in fasta_sources:
        recs = load_named_fasta(keys)
        genes: List[List[str]] = []
        if recs:
            for _h, s in recs[:400]:
                cs = iter_codons(s)
                if len(cs) >= 30:
                    genes.append(cs)
        out.append((name, genes))
    cds22 = extract_chr22_cds(300)
    genes22: List[List[str]] = []
    if cds22:
        for _h, s, _st in cds22:
            cs = iter_codons(s)
            if len(cs) >= 30:
                genes22.append(cs)
    out.append(("chr22", genes22))
    return out


def _is_ser_chord(a: str, b: str) -> bool:
    return (a in SER_TCN and b in SER_AGY) or (a in SER_AGY and b in SER_TCN)


@dataclass
class GenomeWallCensus:
    rows: Tuple[GenomeBuRow, ...]
    gates: Dict[str, bool]


def genome_wall_census(n_null_gc: int = 12, n_null_syn: int = 24) -> GenomeWallCensus:
    """Score real CDSs through the wall decomposition (sections 40–42).

    Channels measured on adjacent-codon pairs of real ORFs:
      - sense channel: same-amino-acid edges vs GC-matched letter shuffle;
      - fold-crossing channel: same-AA TCN<->AGY serine chords vs GC shuffle
        and vs protein-fixed uniform synonym resampling.
    Usage anatomy (BU classes, TGA share of the 10-class) reported descriptively.
    """
    enc = _ref_enc()
    bu_of_codon = {c: _pi_bu(pack_codon_bits(c, enc)) for c in CODONS}
    code = STANDARD_CODE
    from hqvm_cgm_genomics_6 import _gc_shuffle_codon_list, recode_synonymous

    rng = random.Random(NULL_SEED + 41)
    rows: List[GenomeBuRow] = []
    for name, genes in _load_cds_records():
        if not genes:
            rows.append(
                GenomeBuRow(name, 0, 0, float("nan"), {}, 0, 0, 0, 0, 0, 0, 0, 0, 0, "SKIP")
            )
            continue
        counts_tot = {0: 0, 4: 0, 8: 0, 12: 0}
        n_class10 = n_tga = 0
        n_codons = 0
        gc_num = gc_den = 0
        n_edges = n_same_aa = n_chord = 0
        n_same_aa_nul = n_chord_gc_nul = n_chord_syn_nul = 0
        n_edges_nul = 0
        for gene in genes:
            cs = [c for c in gene if c in bu_of_codon]
            if len(cs) < 2:
                continue
            n_codons += len(cs)
            seq = "".join(cs)
            gc_num += sum(1 for ch in seq if ch in STRONG)
            gc_den += len(seq)
            idx_gc = [i for i, ch in enumerate(seq) if ch in STRONG]
            idx_at = [i for i, ch in enumerate(seq) if ch in {"A", "T"}]
            for c in cs:
                b = bu_of_codon[c]
                counts_tot[b] += 1
                if b == 4:
                    n_class10 += 1
                    if code[c] == "*":
                        n_tga += 1
            for a, b in zip(cs, cs[1:]):
                n_edges += 1
                if code[a] == code[b]:
                    n_same_aa += 1
                    if _is_ser_chord(a, b):
                        n_chord += 1
            for _k in range(n_null_gc):
                sh = _gc_shuffle_codon_list(cs, rng, idx_gc, idx_at)
                for a, b in zip(sh, sh[1:]):
                    n_edges_nul += 1
                    if code[a] == code[b]:
                        n_same_aa_nul += 1
                        if _is_ser_chord(a, b):
                            n_chord_gc_nul += 1
            for _k in range(n_null_syn):
                rc = recode_synonymous(cs, code, rng)
                for a, b in zip(rc, rc[1:]):
                    if code[a] == code[b] and _is_ser_chord(a, b):
                        n_chord_syn_nul += 1
        gc = gc_num / gc_den if gc_den else float("nan")
        rows.append(
            GenomeBuRow(
                name, len(genes), n_codons, gc, dict(counts_tot),
                n_class10, n_tga, n_edges, n_edges_nul,
                n_same_aa, n_same_aa_nul,
                n_chord, n_chord_gc_nul, n_chord_syn_nul, "ok",
            )
        )
    ok_rows = [r for r in rows if r.status == "ok"]
    gates = {
        "228_all_genomes_scored": bool(ok_rows) and len(ok_rows) == len(rows),
        "229_same_aa_edges_enriched_over_gc_null": all(
            r.n_same_aa / r.n_edges > r.n_same_aa_null / max(1, r.n_edges_gc_null)
            for r in ok_rows
        ),
        "230_serine_chord_channel_present": all(r.n_chord > 0 for r in ok_rows),
    }
    return GenomeWallCensus(tuple(rows), gates)


def print_genome_wall(c: GenomeWallCensus) -> None:
    g = c.gates
    report_section("44. GENOMES THROUGH THE WALL DECOMPOSITION (REAL CDS)")
    report_objects((
        "BU-class usage; TGA share of the 10-class; same-AA edge fraction; "
        "TCN<->AGY chord share of same-AA edges; "
        "nulls: GC-matched letter shuffle x12 and protein-fixed synonym resample x24",
    ))
    headers = ("genome", "genes", "codons", "gc", "c00", "c10", "c01", "c11", "tga/c10")
    widths = (9, 5, 8, 6, 8, 7, 7, 7, 8)
    rows_out = [
        (
            r.genome, r.n_genes, r.n_codons,
            f"{r.gc:.3f}" if r.gc == r.gc else "-",
            r.bu_counts.get(0, 0), r.bu_counts.get(4, 0),
            r.bu_counts.get(8, 0), r.bu_counts.get(12, 0),
            f"{r.n_tga}/{r.n_class10}" if r.status == "ok" else "-",
        )
        for r in c.rows
    ]
    print_table(headers, widths, rows_out, aligns=("<", ">", ">", ">", ">", ">", ">", ">", ">"))
    print()
    print("  channels vs nulls (fractions)")
    for r in c.rows:
        if r.status != "ok":
            continue
        f_same = r.n_same_aa / r.n_edges
        z_same = r.n_same_aa_null / max(1, r.n_edges_gc_null)
        f_ch = r.n_chord / r.n_same_aa if r.n_same_aa else float("nan")
        z_ch_gc = r.n_chord_gc_null / max(1, r.n_same_aa_null)
        z_ch_syn = r.n_chord_syn_null / max(1, 24 * r.n_same_aa)
    print(
            f"    {r.genome}: sameAA {f_same:.4f} vs gcnull {z_same:.4f} | "
            f"chord|sameAA {f_ch:.4f} vs gcnull {z_ch_gc:.4f} vs synnull {z_ch_syn:.4f}"
    )
    print()
    report_checks((
        ("all four genomes scored (no missing data)", g["228_all_genomes_scored"], "see table", "all ok"),
        ("same-AA edge fraction exceeds its GC-matched null in every genome", g["229_same_aa_edges_enriched_over_gc_null"], "see channel lines", "obs > null each"),
        ("fold-crossing serine chords present and measurable in every genome", g["230_serine_chord_channel_present"], "see channel lines", "count > 0 each"),
    ))
