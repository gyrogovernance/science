#!/usr/bin/env python3
"""
hqvm_cgm_genomics_1.py

Chart and translation quotient. Nucleotide K4 torsor, 24 affine gauges,
two metric orbits, fold/RC commutation by orbit, synonymous-fiber geometry
and beta1, Aut inside the wreath action on NCBI tables, Walsh-Hadamard /
H(3,4) block-weight energy, grade-3 projection ranks, stop+Trp hull, and
N1-N5 nulls with p_hat = (hits+1)/(N+1).

print_chart_census emits the readable report block. Invoked by hqvm_cgm_genomics_run.py.
"""
from __future__ import annotations

import itertools
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from gyroscopic.hQVM.api import q_word6, shadow_partner_byte, walsh_sign6
from gyroscopic.hQVM.family import fold_disagreement_d, gf2_rank, phase_pairs_d

from hqvm_cgm_genomics_common import (
    AA_ORDER,
    AMINO,
    BASES,
    CHIRALITY_D,
    CODE_NAMES,
    CODONS,
    CODON_INDEX,
    LINEAR_BITS,
    N_CODONS,
    N_NUCLEOTIDE_ENCODINGS,
    NCBI_TABLE_IDS,
    NULL_SEED,
    ORBIT_ELEMENTARY,
    ORBIT_OTHER,
    ORBIT_PAIR_INV,
    PURINES,
    REF_TS_DELTA,
    REF_TV_DELTA,
    REF_WC_DELTA,
    STRONG,
    TRANSITION,
    TRANSVERSION,
    WC,
    NucleotideEncoding,
    all_nucleotide_encodings,
    binom,
    codon_state,
    degeneracy_multiset,
    encoding_orbit_name,
    fibers,
    fold_byte,
    fold_matches_phase_pairs,
    hamming6,
    involution_delta,
    kernel_dependency,
    krawtchouk,
    min_mean_max,
    monte_carlo_p,
    make_null_code_pools,
    mutation_class,
    mutation_q,
    one_base_neighbors,
    pack_byte,
    STANDARD_CODE,
    encodings_in_orbit,
    pack_codon_bits,
    predicted_q6,
    print_table,
    report_checks,
    report_objects,
    report_section,
    rc_bits,
    rc_is_affine,
    shuffle_code,
    shuffle_code_boxes,
    shuffle_equal_fiber_geometry,
    shuffle_preserve_stops,
    translation_table,
    unpack_byte,
    unpack_codon_bits,
)

NULL_N1 = 200
NULL_N2 = 200
NULL_N345 = 80


@dataclass(frozen=True)
class EncodingReport:
    index: int
    matrix: Tuple[int, int, int, int]
    translation: int
    phi: Tuple[int, int, int, int]
    orbit: str
    wc_ok: bool
    ts_ok: bool
    tv_ok: bool
    wc_delta: int
    ts_delta: int
    tv_delta: int
    wc_wt: int
    ts_wt: int
    tv_wt: int
    k4_span: int
    bipartition: Tuple[str, ...]
    rc_affine: bool
    rc_rank: int
    rc_shift: int
    rc_shift_wt: int
    q6_match: int
    q6_total: int
    fold_eq_rc: int
    fold_eq_id: int
    shadow_eq_rc: int
    fold_rank_plus_i: int
    rc_rank_plus_i: int
    similar_fold_rc: bool
    fold_phase_pairs_ok: int
    fold_phase_pairs_total: int
    map8_fold_involution: int
    map8_rc_involution: int
    map8_commute: int
    map8_total: int
    map8_fold_rank_plus_i: int
    map8_rc_rank_plus_i: int
    map8_fold_rc_rank_plus_i: int


@dataclass(frozen=True)
class FiberAtlasRow:
    aa: str
    size: int
    n_edges: int
    n_components: int
    transport_rank: int
    hull: int
    density: float
    cycle_rank: int
    boundary: int
    leakage: float
    krawtchouk: Tuple[int, ...]


@dataclass(frozen=True)
class AutTableRow:
    table_id: int
    name: str
    n_aut_pointwise: int
    n_aut_quotient: int
    n_aut_edge: int
    n_aut_stop: int
    n_stop: int
    wobble_k4_elem: int
    wobble_k4_pair: int
    diagonal_k4_elem: int
    diagonal_k4_pair: int


@dataclass(frozen=True)
class ChartEnergyRow:
    table_id: int
    enc_index: int
    orbit: str
    S1: float
    S2: float
    S3: float
    E: Tuple[int, ...]
    K: Tuple[float, ...]
    n1_hits: int
    n1_n: int
    n2_hits: int
    n2_n: int
    n1_p: float
    n2_p: float


@dataclass(frozen=True)
class NullLadderRow:
    name: str
    orbit: str
    statistic: str
    observed: float
    hits: int
    n: int
    p_hat: float


@dataclass(frozen=True)
class HomologyRow:
    table_id: int
    n_edges: int
    n_components: int
    beta1: int


@dataclass(frozen=True)
class ChartQuotientCensus:
    kernel_ok: bool
    kernel_note: str
    n_encodings: int
    n_distinct_phi: int
    n_wc_translation: int
    n_ts_translation: int
    n_tv_translation: int
    n_full_k4: int
    n_rc_affine: int
    n_q6_exact: int
    n_fold_eq_rc: int
    n_similar_fold_rc: int
    n_orbit_elem: int
    n_orbit_pair: int
    n_orbit_other: int
    wt2_assignment: Dict[str, int]
    reports: Tuple[EncodingReport, ...]
    degeneracy: Tuple[int, ...]
    fiber_rows: Tuple[FiberAtlasRow, ...]
    serine: FiberAtlasRow
    stop_row: FiberAtlasRow
    n_aa: int
    c63: int
    aut_rows: Tuple[AutTableRow, ...]
    aut_standard: Tuple[Tuple[Tuple[int, ...], Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]], ...]
    energy_rows: Tuple[ChartEnergyRow, ...]
    null_rows: Tuple[NullLadderRow, ...]
    fold_hist: Tuple[int, ...]
    homology_rows: Tuple[HomologyRow, ...]
    beta1_standard: int
    h34_mult: Tuple[int, ...]
    block_E: Tuple[int, ...]
    grade3_rank: int
    grade3_rank_with_stop: int
    stop_trp_rank: int
    stop_trp_hull: int
    stop_trp_has_trp: bool
    n_pair_s2_gt_elem: int
    n_tables_s2: int
    n_vert_edges: int
    n_horiz_edges: int
    gates: Dict[str, bool]


def _bipartition_bits(enc: NucleotideEncoding) -> Tuple[str, ...]:
    found = []
    targets = (
        ("R/Y", PURINES),
        ("M/K", AMINO),
        ("S/W", STRONG),
    )
    for name, ones in targets:
        hit = "none"
        for bit_name, fn in LINEAR_BITS:
            vals = {fn(enc.encode_base(b)) for b in ones}
            other = {fn(enc.encode_base(b)) for b in BASES if b not in ones}
            if len(vals) == 1 and len(other) == 1 and vals != other:
                v = next(iter(vals))
                hit = f"{bit_name}={v}"
                break
        found.append(f"{name}:{hit}")
    return tuple(found)


def _payload_fold_linear(family: int = 0) -> List[int]:
    cols = []
    for i in range(6):
        e = 1 << i
        _f, micro = unpack_byte(fold_byte(pack_byte(family, e)))
        cols.append(micro)
    return cols


def _rc_linear_columns(enc: NucleotideEncoding) -> List[int]:
    b0 = rc_bits(0, enc)
    return [rc_bits(1 << i, enc) ^ b0 for i in range(6)]


def _rank_m_plus_i(columns: List[int], dim: int) -> int:
    plus = [columns[i] ^ (1 << i) for i in range(dim)]
    return gf2_rank(plus, dim)


def rc_byte_keep_family(byte: int, enc: NucleotideEncoding) -> int:
    fam, pay = unpack_byte(byte)
    return pack_byte(fam, rc_bits(pay, enc))


def affine8_columns(fn) -> Tuple[bool, int, List[int]]:
    b0 = fn(0)

    def lin(x: int) -> int:
        return fn(x) ^ b0

    ok = True
    for i in range(8):
        e = 1 << i
        for j in range(i, 8):
            f = 1 << j
            if lin(e ^ f) != (lin(e) ^ lin(f)):
                ok = False
                break
        if not ok:
            break
    cols = [lin(1 << i) for i in range(8)]
    return ok, b0, cols


def classify_encoding(index: int, enc: NucleotideEncoding) -> EncodingReport:
    wc_ok, wc_delta = involution_delta(WC, enc)
    ts_ok, ts_delta = involution_delta(TRANSITION, enc)
    tv_ok, tv_delta = involution_delta(TRANSVERSION, enc)
    span = gf2_rank([wc_delta, ts_delta, tv_delta], 2)
    rc_ok, rc_rank, rc_shift = rc_is_affine(enc)

    rc_lut = tuple(rc_bits(p, enc) for p in range(N_CODONS))

    q_match = 0
    q_total = 0
    fold_eq_rc = 0
    fold_eq_id = 0
    shadow_eq_rc = 0
    phase_ok = 0
    fold_inv = 0
    rc_inv = 0
    commute = 0
    map8_total = 0
    for family in range(4):
        for codon in range(N_CODONS):
            byte = pack_byte(family, codon)
            q_total += 1
            if q_word6(byte) == predicted_q6(family, codon):
                q_match += 1
            if fold_matches_phase_pairs(byte):
                phase_ok += 1
            f2, c2 = unpack_byte(fold_byte(byte))
            rc = rc_lut[codon]
            if c2 == rc:
                fold_eq_rc += 1
            if c2 == codon:
                fold_eq_id += 1
            _sf, sc = unpack_byte(shadow_partner_byte(byte))
            if sc == rc:
                shadow_eq_rc += 1
            fb = fold_byte(byte)
            rb = rc_byte_keep_family(byte, enc)
            map8_total += 1
            if fold_byte(fb) == byte:
                fold_inv += 1
            if rc_byte_keep_family(rb, enc) == byte:
                rc_inv += 1
            if fold_byte(rb) == rc_byte_keep_family(fb, enc):
                commute += 1

    fold_cols = _payload_fold_linear(0)
    rc_cols = _rc_linear_columns(enc)
    fold_r = _rank_m_plus_i(fold_cols, 6)
    rc_r = _rank_m_plus_i(rc_cols, 6)

    def f8(x: int) -> int:
        return fold_byte(x)

    def r8(x: int) -> int:
        return rc_byte_keep_family(x, enc)

    def fr8(x: int) -> int:
        return fold_byte(r8(x))

    _ok_f, _b_f, cols_f = affine8_columns(f8)
    _ok_r, _b_r, cols_r = affine8_columns(r8)
    _ok_fr, _b_fr, cols_fr = affine8_columns(fr8)

    return EncodingReport(
        index=index,
        matrix=enc.matrix,
        translation=enc.translation,
        phi=enc.phi,
        orbit=encoding_orbit_name(enc),
        wc_ok=wc_ok,
        ts_ok=ts_ok,
        tv_ok=tv_ok,
        wc_delta=wc_delta,
        ts_delta=ts_delta,
        tv_delta=tv_delta,
        wc_wt=hamming6(wc_delta),
        ts_wt=hamming6(ts_delta),
        tv_wt=hamming6(tv_delta),
        k4_span=span,
        bipartition=_bipartition_bits(enc),
        rc_affine=rc_ok,
        rc_rank=rc_rank,
        rc_shift=rc_shift,
        rc_shift_wt=hamming6(rc_shift),
        q6_match=q_match,
        q6_total=q_total,
        fold_eq_rc=fold_eq_rc,
        fold_eq_id=fold_eq_id,
        shadow_eq_rc=shadow_eq_rc,
        fold_rank_plus_i=fold_r,
        rc_rank_plus_i=rc_r,
        similar_fold_rc=fold_r == rc_r,
        fold_phase_pairs_ok=phase_ok,
        fold_phase_pairs_total=q_total,
        map8_fold_involution=fold_inv,
        map8_rc_involution=rc_inv,
        map8_commute=commute,
        map8_total=map8_total,
        map8_fold_rank_plus_i=_rank_m_plus_i(cols_f, 8),
        map8_rc_rank_plus_i=_rank_m_plus_i(cols_r, 8),
        map8_fold_rc_rank_plus_i=_rank_m_plus_i(cols_fr, 8),
    )


def _fiber_krawtchouk(group: Sequence[str], enc: NucleotideEncoding) -> Tuple[int, ...]:
    origin = pack_codon_bits(group[0], enc)
    weights = [0] * 7
    for c in group:
        weights[hamming6(pack_codon_bits(c, enc) ^ origin)] += 1
    kvals = []
    for k in range(4):
        acc = 0
        for i, a_i in enumerate(weights):
            acc += a_i * krawtchouk(k, i, 6)
        kvals.append(acc)
    return tuple(kvals)


def fiber_atlas(code: Dict[str, str], enc: NucleotideEncoding) -> Tuple[FiberAtlasRow, ...]:
    rows = []
    fib = fibers(code)
    for aa in AA_ORDER:
        group = fib.get(aa, ())
        if not group:
            continue
        members = set(group)
        n_edges = 0
        adj = {c: [] for c in group}
        boundary = 0
        for c in group:
            for n in one_base_neighbors(c):
                if n in members:
                    n_edges += 1
                    adj[c].append(n)
                else:
                    boundary += 1
        n_edges //= 2
        seen = set()
        n_comp = 0
        for c in group:
            if c in seen:
                continue
            n_comp += 1
            stack = [c]
            seen.add(c)
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
        origin = pack_codon_bits(group[0], enc)
        vecs = [pack_codon_bits(c, enc) ^ origin for c in group]
        rank = gf2_rank(vecs, 6)
        hull = 1 << rank if rank >= 0 else 0
        density = (len(group) / hull) if hull else 0.0
        cycle = n_edges - len(group) + n_comp
        leakage = boundary / (9.0 * len(group)) if group else 0.0
        rows.append(
            FiberAtlasRow(
                aa=aa,
                size=len(group),
                n_edges=n_edges,
                n_components=n_comp,
                transport_rank=rank,
                hull=hull,
                density=density,
                cycle_rank=cycle,
                boundary=boundary,
                leakage=leakage,
                krawtchouk=_fiber_krawtchouk(group, enc),
            )
        )
    return tuple(rows)


def _letter_maps() -> Tuple[Dict[str, str], ...]:
    return tuple({BASES[i]: perm[i] for i in range(4)} for perm in itertools.permutations(BASES))


_LETTER_MAPS = _letter_maps()
_POS_PERMS = tuple(itertools.permutations(range(3)))


def apply_wreath(codon: str, pos_perm: Tuple[int, ...], maps: Tuple[Dict[str, str], ...]) -> str:
    chars = []
    for new_i in range(3):
        old_i = pos_perm[new_i]
        chars.append(maps[new_i][codon[old_i]])
    return "".join(chars)


@lru_cache(maxsize=1)
def wreath_index_perms() -> Tuple[Tuple[Tuple[int, ...], Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], Tuple[int, ...]], ...]:
    found = []
    for pos in _POS_PERMS:
        for m0 in _LETTER_MAPS:
            for m1 in _LETTER_MAPS:
                for m2 in _LETTER_MAPS:
                    maps = (m0, m1, m2)
                    perm = tuple(CODON_INDEX[apply_wreath(c, pos, maps)] for c in CODONS)
                    found.append(
                        (
                            pos,
                            tuple(m0[b] for b in BASES),
                            tuple(m1[b] for b in BASES),
                            tuple(m2[b] for b in BASES),
                            perm,
                        )
                    )
    return tuple(found)


def code_string(code: Dict[str, str]) -> str:
    return "".join(code[c] for c in CODONS)


def count_aut_lattice(code: Dict[str, str], do_edge: bool) -> Tuple[int, int, int, int, Tuple]:
    s = code_string(code)
    n_pt = n_q = n_edge = n_stop = 0
    pointwise = []
    for pos, t0, t1, t2, perm in wreath_index_perms():
        pt = all(s[perm[i]] == s[i] for i in range(N_CODONS))
        if pt:
            n_pt += 1
            pointwise.append((pos, t0, t1, t2))
        sigma: Dict[str, str] = {}
        q_ok = True
        for i in range(N_CODONS):
            a = s[i]
            b = s[perm[i]]
            prev = sigma.get(a)
            if prev is None:
                sigma[a] = b
            elif prev != b:
                q_ok = False
                break
        if q_ok:
            n_q += 1
        if all((s[i] == "*") == (s[perm[i]] == "*") for i in range(N_CODONS)):
            n_stop += 1
        if do_edge and pt:
            img_of = [CODONS[perm[i]] for i in range(N_CODONS)]
            e_ok = True
            for codon in CODONS:
                ci = CODON_INDEX[codon]
                img_c = img_of[ci]
                for neigh in one_base_neighbors(codon):
                    if mutation_class(img_c, img_of[CODON_INDEX[neigh]]) != mutation_class(codon, neigh):
                        e_ok = False
                        break
                if not e_ok:
                    break
            if e_ok:
                n_edge += 1
        elif not do_edge and pt:
            n_edge += 1
    return n_pt, n_q, n_edge, n_stop, tuple(pointwise)


def _xor_last_base(codon: str, enc: NucleotideEncoding, delta: int) -> str:
    bits = pack_codon_bits(codon, enc) ^ (int(delta) & 0x3)
    return unpack_codon_bits(bits, enc)


def _xor_diag(codon: str, enc: NucleotideEncoding, delta: int) -> str:
    d = int(delta) & 0x3
    q = d | (d << 2) | (d << 4)
    return unpack_codon_bits(pack_codon_bits(codon, enc) ^ q, enc)


def subgroup_preserves(code: Dict[str, str], enc: NucleotideEncoding, kind: str) -> int:
    n_ok = 0
    for delta in range(4):
        ok = True
        for codon in CODONS:
            if kind == "wobble":
                image = _xor_last_base(codon, enc, delta)
            else:
                image = _xor_diag(codon, enc, delta)
            if code[image] != code[codon]:
                ok = False
                break
        if ok:
            n_ok += 1
    return n_ok


def wht64(vec: List[int]) -> List[int]:
    """In-place Walsh–Hadamard transform on length-64 integer vectors."""
    out = list(vec)
    h = 1
    n = 64
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = out[j]
                b = out[j + h]
                out[j] = a + b
                out[j + h] = a - b
        h *= 2
    return out


@lru_cache(maxsize=1)
def _wht64_matrix() -> np.ndarray:
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < 64:
        h = np.block([[h, h], [h, -h]])
    return h


_WALSH_DEGREE_LUT = np.array([s.bit_count() for s in range(64)], dtype=np.int8)


_AA_TO_IDX: Dict[str, int] = {a: i for i, a in enumerate(AA_ORDER)}


def _code_aa_indices(code: Dict[str, str]) -> np.ndarray:
    """Encoding-independent AA row index per codon slot (0..63), -1 if not in AA_ORDER."""
    out = np.full(N_CODONS, -1, dtype=np.int16)
    for i, codon in enumerate(CODONS):
        aa = code[codon]
        idx = _AA_TO_IDX.get(aa)
        if idx is not None:
            out[i] = idx
    return out


@lru_cache(maxsize=None)
def _payload_lut(enc: NucleotideEncoding) -> Tuple[int, ...]:
    return tuple(pack_codon_bits(codon, enc) for codon in CODONS)


def _indicator_from_aa_indices(aa_idx: np.ndarray, payload_lut: Tuple[int, ...]) -> np.ndarray:
    mat = np.zeros((len(AA_ORDER), 64), dtype=np.float64)
    for ci, row in enumerate(aa_idx):
        if row >= 0:
            mat[int(row), payload_lut[ci]] = 1.0
    return mat


def _code_indicator_matrix(code: Dict[str, str], enc: NucleotideEncoding) -> np.ndarray:
    return _indicator_from_aa_indices(_code_aa_indices(code), _payload_lut(enc))


def _energy_from_indicator(ind: np.ndarray) -> Tuple[int, ...]:
    hat = ind @ _wht64_matrix().T
    energy = np.zeros(7, dtype=np.float64)
    for k in range(7):
        energy[k] = np.square(hat[:, _WALSH_DEGREE_LUT == k]).sum()
    return tuple(int(round(v)) for v in energy)


def _s2_from_indicator(ind: np.ndarray) -> float:
    energy = _energy_from_indicator(ind)
    return cumulative_S(energy, 2)


def _batched_s2_from_indicators(batch: np.ndarray) -> np.ndarray:
    hat = batch @ _wht64_matrix().T
    n = batch.shape[0]
    energy = np.zeros((n, 7), dtype=np.float64)
    for k in range(7):
        mask = _WALSH_DEGREE_LUT == k
        energy[:, k] = np.square(hat[:, :, mask]).sum(axis=(1, 2))
    tot = energy.sum(axis=1)
    cum = energy[:, :3].sum(axis=1)
    return np.where(tot > 0, cum / tot, 0.0)


def walsh_s2(code: Dict[str, str], enc: NucleotideEncoding) -> float:
    return _s2_from_indicator(_code_indicator_matrix(code, enc))


def walsh_s2_batch(codes: Sequence[Dict[str, str]], enc: NucleotideEncoding) -> np.ndarray:
    if not codes:
        return np.zeros(0, dtype=np.float64)
    batch = np.stack([_code_indicator_matrix(c, enc) for c in codes])
    return _batched_s2_from_indicators(batch)


@lru_cache(maxsize=1)
def ncbi_chart_s2_grid() -> Tuple[np.ndarray, Tuple[NucleotideEncoding, ...]]:
    """S2(table, enc) for every NCBI table and nucleotide encoding."""
    encs = all_nucleotide_encodings()
    grid = np.zeros((len(NCBI_TABLE_IDS), len(encs)), dtype=np.float64)
    for ti, tid in enumerate(NCBI_TABLE_IDS):
        table = translation_table(tid)
        for ei, enc in enumerate(encs):
            grid[ti, ei] = walsh_s2(table, enc)
    return grid, encs


def walsh_degree_energy(code: Dict[str, str], enc: NucleotideEncoding) -> Tuple[int, ...]:
    return _energy_from_indicator(_code_indicator_matrix(code, enc))


def krawtchouk_energy(energy: Tuple[int, ...]) -> Tuple[float, ...]:
    out = []
    tot = float(sum(energy)) or 1.0
    for k in range(4):
        acc = 0.0
        for i, e_i in enumerate(energy):
            acc += e_i * krawtchouk(k, i, 6)
        out.append(acc / tot)
    return tuple(out)


def cumulative_S(energy: Tuple[int, ...], k: int) -> float:
    tot = sum(energy)
    if tot == 0:
        return 0.0
    return sum(energy[: k + 1]) / tot


def synonymous_tv_edges(code: Dict[str, str]) -> int:
    n = 0
    for c in CODONS:
        for neigh in one_base_neighbors(c):
            if code[neigh] == code[c] and mutation_class(c, neigh) == "transversion":
                n += 1
    return n // 2


def orbit_rep_indices(encodings: Sequence[NucleotideEncoding]) -> Dict[str, int]:
    reps: Dict[str, int] = {}
    for i, enc in enumerate(encodings):
        name = encoding_orbit_name(enc)
        if name not in reps:
            reps[name] = i
    return reps


def chemical_k4_on_reference() -> Dict[str, int]:
    return {
        "ref_wc_delta": REF_WC_DELTA,
        "ref_ts_delta": REF_TS_DELTA,
        "ref_tv_delta": REF_TV_DELTA,
        "ref_wc_wt": hamming6(REF_WC_DELTA),
        "ref_ts_wt": hamming6(REF_TS_DELTA),
        "ref_tv_wt": hamming6(REF_TV_DELTA),
    }


def fold_disagreement_census(enc: NucleotideEncoding) -> Tuple[int, ...]:
    hist = [0] * 5
    for family in range(4):
        for codon in range(N_CODONS):
            byte = pack_byte(family, codon)
            d = fold_disagreement_d(byte, CHIRALITY_D)
            hist[d] += 1
    return tuple(hist)


def synonymous_homology(code: Dict[str, str]) -> Tuple[int, int, int]:
    """Return (n_syn_edges, n_components, beta1) of the synonymous one-base graph."""
    n_edges = 0
    n_comp = 0
    fib = fibers(code)
    for _aa, group in fib.items():
        members = set(group)
        adj = {c: [] for c in group}
        for c in group:
            for n in one_base_neighbors(c):
                if n in members and c < n:
                    n_edges += 1
                    adj[c].append(n)
                    adj[n].append(c)
        seen = set()
        for c in group:
            if c in seen:
                continue
            n_comp += 1
            stack = [c]
            seen.add(c)
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
    beta1 = n_edges - N_CODONS + n_comp
    return n_edges, n_comp, beta1


_BLOCK_WEIGHT_LUT: Tuple[int, ...] = tuple(
    sum(1 for i in range(3) if ((s >> (2 * i)) & 0x3) != 0) for s in range(64)
)


def block_weight6(s: int) -> int:
    """Number of nonzero 2-bit nucleotide blocks in a 6-bit Walsh index."""
    return _BLOCK_WEIGHT_LUT[int(s) & 0x3F]


def block_degree_energy(code: Dict[str, str], enc: NucleotideEncoding) -> Tuple[int, ...]:
    """Walsh energy regrouped by H(3,4) block weight j in {0,1,2,3}."""
    energy = [0] * 4
    for aa in AA_ORDER:
        vec = [0] * N_CODONS
        for codon in CODONS:
            if code[codon] == aa:
                vec[pack_codon_bits(codon, enc)] = 1
        hat = wht64(vec)
        for s, val in enumerate(hat):
            energy[block_weight6(s)] += val * val
    return tuple(energy)


def grade3_projection_ranks(code: Dict[str, str], enc: NucleotideEncoding) -> Tuple[int, int]:
    """Rank of AA indicators in Walsh grade-3 (20) and grade-0+3 (21) sectors."""
    import numpy as np

    grade3 = [s for s in range(N_CODONS) if s.bit_count() == 3]
    aa_list = [a for a in AA_ORDER if a != "*"]
    mats = []
    for aa in aa_list:
        vec = [0] * N_CODONS
        for codon in CODONS:
            if code[codon] == aa:
                vec[pack_codon_bits(codon, enc)] = 1
        hat = wht64(vec)
        mats.append([float(hat[s]) for s in grade3])
    M = np.asarray(mats, dtype=float)
    rank20 = int(np.linalg.matrix_rank(M, tol=1e-8))
    stop_vec = [0] * N_CODONS
    for codon in CODONS:
        if code[codon] == "*":
            stop_vec[pack_codon_bits(codon, enc)] = 1
    hat_stop = wht64(stop_vec)
    row_stop = [float(hat_stop[0])] + [float(hat_stop[s]) for s in grade3]
    mats21 = []
    for aa in aa_list:
        vec = [0] * N_CODONS
        for codon in CODONS:
            if code[codon] == aa:
                vec[pack_codon_bits(codon, enc)] = 1
        hat = wht64(vec)
        mats21.append([float(hat[0])] + [float(hat[s]) for s in grade3])
    mats21.append(row_stop)
    M21 = np.asarray(mats21, dtype=float)
    rank21 = int(np.linalg.matrix_rank(M21, tol=1e-8))
    return rank20, rank21


def stop_trp_hull(code: Dict[str, str], enc: NucleotideEncoding) -> Tuple[int, int, bool]:
    stops = [c for c in CODONS if code[c] == "*"]
    has_trp = any(code[c] == "W" for c in CODONS)
    trp = [c for c in CODONS if code[c] == "W"]
    pts = stops + trp
    if not pts:
        return 0, 0, False
    origin = pack_codon_bits(pts[0], enc)
    vecs = [pack_codon_bits(c, enc) ^ origin for c in pts]
    rank = gf2_rank(vecs, 6)
    return rank, 1 << rank, has_trp and len(trp) == 1 and len(stops) == 3


def vertical_horizontal_edges(code: Dict[str, str]) -> Tuple[int, int]:
    vert = horiz = 0
    for c in CODONS:
        for n in one_base_neighbors(c):
            if c >= n:
                continue
            if code[c] == code[n]:
                vert += 1
            else:
                horiz += 1
    return vert, horiz


H34_MULTIPLICITIES = (1, 9, 27, 27)


def conjugacy_and_quotient_census() -> ChartQuotientCensus:
    kernel_ok, kernel_note = kernel_dependency()
    encodings = all_nucleotide_encodings()
    reports = tuple(classify_encoding(i, enc) for i, enc in enumerate(encodings))
    reps = orbit_rep_indices(encodings)

    wt2 = Counter()
    for r in reports:
        wt2_matches = [n for n, w in (("wc", r.wc_wt), ("transition", r.ts_wt), ("transversion", r.tv_wt)) if w == 2]
        assert len(wt2_matches) <= 1, f"chart {r.index}: multiple weight-2 K4 involutions"
        wt2_name = wt2_matches[0] if wt2_matches else "none"
        wt2[wt2_name] += 1

    n_q6 = sum(1 for r in reports if r.q6_match == r.q6_total)
    n_fold = sum(1 for r in reports if r.fold_eq_rc == r.q6_total)
    n_sim = sum(1 for r in reports if r.similar_fold_rc)
    n_linear3 = sum(1 for r in reports if all("none" not in p for p in r.bipartition))
    n_elem = sum(1 for r in reports if r.orbit == ORBIT_ELEMENTARY)
    n_pair = sum(1 for r in reports if r.orbit == ORBIT_PAIR_INV)
    n_other = sum(1 for r in reports if r.orbit == ORBIT_OTHER)

    code1 = translation_table(1)
    deg = degeneracy_multiset(code1)
    fiber_rows = fiber_atlas(code1, encodings[0])
    ser = next(r for r in fiber_rows if r.aa == "S")
    stop_row = next(r for r in fiber_rows if r.aa == "*")
    n_aa = sum(1 for r in fiber_rows if r.aa != "*")

    aut_rows = []
    aut_standard = ()
    for tid in NCBI_TABLE_IDS:
        code = translation_table(tid)
        do_edge = tid == 1
        n_pt, n_q, n_edge, n_stop, pointwise = count_aut_lattice(code, do_edge)
        if tid == 1:
            aut_standard = pointwise
        enc_e = encodings[reps[ORBIT_ELEMENTARY]]
        enc_p = encodings[reps[ORBIT_PAIR_INV]]
        aut_rows.append(
            AutTableRow(
                table_id=tid,
                name=CODE_NAMES.get(tid, str(tid)),
                n_aut_pointwise=n_pt,
                n_aut_quotient=n_q,
                n_aut_edge=n_edge,
                n_aut_stop=n_stop,
                n_stop=sum(1 for c in CODONS if code[c] == "*"),
                wobble_k4_elem=subgroup_preserves(code, enc_e, "wobble"),
                wobble_k4_pair=subgroup_preserves(code, enc_p, "wobble"),
                diagonal_k4_elem=subgroup_preserves(code, enc_e, "diag"),
                diagonal_k4_pair=subgroup_preserves(code, enc_p, "diag"),
            )
        )

    rng = random.Random(NULL_SEED)
    energy_rows: List[ChartEnergyRow] = []
    null_rows: List[NullLadderRow] = []

    pools = make_null_code_pools(
        code1, rng, n1=NULL_N1, n2=NULL_N2, n345=NULL_N345
    )
    n1_pool = list(pools.n1)
    n2_pool = list(pools.n2)
    n3_pool = list(pools.n3)
    n4_pool = list(pools.n4)
    n5_obs = synonymous_tv_edges(code1)
    n5_hits = 0
    for nc in n1_pool[:NULL_N345]:
        if synonymous_tv_edges(nc) >= n5_obs:
            n5_hits += 1

    n1_aa = [_code_aa_indices(nc) for nc in n1_pool]
    n2_aa = [_code_aa_indices(nc) for nc in n2_pool]
    n3_aa = [_code_aa_indices(nc) for nc in n3_pool]
    n4_aa = [_code_aa_indices(nc) for nc in n4_pool]
    ref_aa = _code_aa_indices(code1)

    s2_by_orbit: Dict[str, List[float]] = defaultdict(list)
    p1_by_orbit: Dict[str, List[float]] = defaultdict(list)
    p2_by_orbit: Dict[str, List[float]] = defaultdict(list)

    for i, enc in enumerate(encodings):
        lut = _payload_lut(enc)
        ind_std = _indicator_from_aa_indices(ref_aa, lut)
        energy = _energy_from_indicator(ind_std)
        s2 = cumulative_S(energy, 2)
        null_inds = np.stack([_indicator_from_aa_indices(aa, lut) for aa in n1_aa])
        n1_hits = int((_batched_s2_from_indicators(null_inds) >= s2 - 1e-15).sum())
        null_inds2 = np.stack([_indicator_from_aa_indices(aa, lut) for aa in n2_aa])
        n2_hits = int((_batched_s2_from_indicators(null_inds2) >= s2 - 1e-15).sum())
        orbit = encoding_orbit_name(enc)
        p1 = monte_carlo_p(n1_hits, NULL_N1)
        p2 = monte_carlo_p(n2_hits, NULL_N2)
        s2_by_orbit[orbit].append(s2)
        p1_by_orbit[orbit].append(p1)
        p2_by_orbit[orbit].append(p2)
        energy_rows.append(
            ChartEnergyRow(
                table_id=1,
                enc_index=i,
                orbit=orbit,
                S1=cumulative_S(energy, 1),
                S2=s2,
                S3=cumulative_S(energy, 3),
                E=energy,
                K=krawtchouk_energy(energy),
                n1_hits=n1_hits,
                n1_n=NULL_N1,
                n2_hits=n2_hits,
                n2_n=NULL_N2,
                n1_p=p1,
                n2_p=p2,
            )
        )

    for tid in NCBI_TABLE_IDS:
        if tid == 1:
            continue
        code = translation_table(tid)
        for name, idx in reps.items():
            enc = encodings[idx]
            energy = walsh_degree_energy(code, enc)
            energy_rows.append(
                ChartEnergyRow(
                    table_id=tid,
                    enc_index=idx,
                    orbit=name,
                    S1=cumulative_S(energy, 1),
                    S2=cumulative_S(energy, 2),
                    S3=cumulative_S(energy, 3),
                    E=energy,
                    K=krawtchouk_energy(energy),
                    n1_hits=-1,
                    n1_n=0,
                    n2_hits=-1,
                    n2_n=0,
                    n1_p=float("nan"),
                    n2_p=float("nan"),
                )
            )

    for orbit in (ORBIT_ELEMENTARY, ORBIT_PAIR_INV, ORBIT_OTHER):
        if orbit not in s2_by_orbit:
            continue
        mn, mean, mx = min_mean_max(s2_by_orbit[orbit])
        p1mn, p1mean, p1mx = min_mean_max(p1_by_orbit[orbit])
        p2mn, p2mean, p2mx = min_mean_max(p2_by_orbit[orbit])
        null_rows.append(NullLadderRow("N1_S2_min_p", orbit, "S2", mn, -1, NULL_N1, p1mn))
        null_rows.append(NullLadderRow("N1_S2_mean_p", orbit, "S2", mean, -1, NULL_N1, p1mean))
        null_rows.append(NullLadderRow("N1_S2_max_p", orbit, "S2", mx, -1, NULL_N1, p1mx))
        null_rows.append(NullLadderRow("N2_S2_min_p", orbit, "S2", mn, -1, NULL_N2, p2mn))
        null_rows.append(NullLadderRow("N2_S2_mean_p", orbit, "S2", mean, -1, NULL_N2, p2mean))
        null_rows.append(NullLadderRow("N2_S2_max_p", orbit, "S2", mx, -1, NULL_N2, p2mx))

    s2_ref = next(r.S2 for r in energy_rows if r.table_id == 1 and r.enc_index == 0)
    ref_enc = encodings[0]
    ref_lut = _payload_lut(ref_enc)
    n3_inds = np.stack([_indicator_from_aa_indices(aa, ref_lut) for aa in n3_aa])
    n3_hits = int((_batched_s2_from_indicators(n3_inds) >= s2_ref - 1e-15).sum())
    n4_inds = np.stack([_indicator_from_aa_indices(aa, ref_lut) for aa in n4_aa])
    n4_hits = int((_batched_s2_from_indicators(n4_inds) >= s2_ref - 1e-15).sum())
    null_rows.append(
        NullLadderRow("N3_component", ORBIT_ELEMENTARY, "S2", s2_ref, n3_hits, NULL_N345, monte_carlo_p(n3_hits, NULL_N345))
    )
    null_rows.append(
        NullLadderRow("N4_stop_degeneracy", ORBIT_ELEMENTARY, "S2", s2_ref, n4_hits, NULL_N345, monte_carlo_p(n4_hits, NULL_N345))
    )
    null_rows.append(
        NullLadderRow("N5_tv_syn_edges", "chart_invariant", "tv_edges", float(n5_obs), n5_hits, NULL_N345, monte_carlo_p(n5_hits, NULL_N345))
    )

    r0 = reports[0]
    aut1 = next(r for r in aut_rows if r.table_id == 1)
    e0 = next(r for r in energy_rows if r.table_id == 1 and r.enc_index == 0)
    n_phase = sum(1 for r in reports if r.fold_phase_pairs_ok == r.fold_phase_pairs_total)
    n_fold_inv = sum(1 for r in reports if r.map8_fold_involution == r.map8_total)
    n_rc_inv = sum(1 for r in reports if r.map8_rc_involution == r.map8_total)
    n_commute_elem = sum(
        1 for r in reports if r.orbit == ORBIT_ELEMENTARY and r.map8_commute == r.map8_total
    )
    n_commute_pair = sum(
        1 for r in reports if r.orbit == ORBIT_PAIR_INV and r.map8_commute == r.map8_total
    )

    homology_rows = []
    for tid in NCBI_TABLE_IDS:
        ne, nc, b1 = synonymous_homology(translation_table(tid))
        homology_rows.append(HomologyRow(table_id=tid, n_edges=ne, n_components=nc, beta1=b1))
    beta1_std = next(h.beta1 for h in homology_rows if h.table_id == 1)
    block_E = block_degree_energy(code1, encodings[0])
    g3_rank, g3_stop = grade3_projection_ranks(code1, encodings[0])
    st_rank, st_hull, st_trp = stop_trp_hull(code1, encodings[0])
    n_vert, n_horiz = vertical_horizontal_edges(code1)

    n_pair_wins = 0
    n_tables_s2 = 0
    for tid in NCBI_TABLE_IDS:
        rows_t = [r for r in energy_rows if r.table_id == tid]
        if not rows_t:
            continue
        by_orb: Dict[str, List[float]] = defaultdict(list)
        for r in rows_t:
            by_orb[r.orbit].append(r.S2)
        if ORBIT_ELEMENTARY in by_orb and ORBIT_PAIR_INV in by_orb:
            n_tables_s2 += 1
            mean_elem = sum(by_orb[ORBIT_ELEMENTARY]) / len(by_orb[ORBIT_ELEMENTARY])
            mean_pair = sum(by_orb[ORBIT_PAIR_INV]) / len(by_orb[ORBIT_PAIR_INV])
            if mean_pair > mean_elem + 1e-15:
                n_pair_wins += 1

    dual_obstruction = True
    for r in reports:
        if r.orbit == ORBIT_ELEMENTARY and r.map8_commute == r.map8_total:
            dual_obstruction = False
        if r.wc_wt == 1 and r.ts_wt == 1 and r.map8_commute == r.map8_total:
            dual_obstruction = False

    gates = {
        "1_n24_encodings": len(reports) == N_NUCLEOTIDE_ENCODINGS,
        "2_wc_translation_all": all(r.wc_ok for r in reports),
        "3_ts_translation_all": all(r.ts_ok for r in reports),
        "4_tv_translation_all": all(r.tv_ok for r in reports),
        "5_k4_span_2_all": all(r.k4_span == 2 for r in reports),
        "6_codon_bijection": all(
            len({pack_codon_bits(c, enc) for c in CODONS}) == N_CODONS for enc in encodings
        ),
        "7_rc_affine_involution": all(r.rc_affine and r.rc_rank == 6 for r in reports),
        "8_q6_packing_consistency": n_q6 == len(reports),
        "9_fold_neq_rc": n_fold == 0,
        "10_fold_rc_rank_gap": n_sim == 0,
        "11_kernel_api": kernel_ok,
        "11b_chemical_bits_linear": n_linear3 == len(reports),
        "11c_orbits_8_8_8": n_elem == 8 and n_pair == 8 and n_other == 8,
        "11d_fold_phase_pairs": n_phase == len(reports),
        "11e_fold_8bit_involution": n_fold_inv == len(reports),
        "11f_rc_8bit_involution": n_rc_inv == len(reports),
        "11g_fold_rc_not_commute_elem": n_commute_elem == 0,
        "11g_fold_rc_commute_pair_inv": n_commute_pair == 8,
        "11j_dual_chart_obstruction": dual_obstruction and n_commute_elem == 0 and n_commute_pair == 8,
        "12_degeneracy_64": sum(deg) == 64,
        "12b_twenty_aa": n_aa == 20,
        "13_aut_wreath_order_2": aut1.n_aut_pointwise == 2,
        "14_wht_plancherel": sum(e0.E) == 64 * 64,
        "14c_block_plancherel": sum(block_E) == 64 * 64,
        "15_beta1_standard_27": beta1_std == 27,
        "15b_beta1_h34_match": beta1_std == H34_MULTIPLICITIES[2],
        "16_grade3_rank_lt20": 0 < g3_rank < 20,
        "16b_grade3_stop_gt_aa": g3_stop > g3_rank,
        "17_stop_trp_hull4": st_hull == 4 and st_rank == 2 and st_trp,
        "17b_pair_s2_gt_elem": n_pair_wins == n_tables_s2 and n_tables_s2 > 0,
        "18_wobble_not_full_k4": aut1.wobble_k4_elem < 4 and aut1.wobble_k4_pair < 4,
        "18b_serine_disconnected": ser.size == 6 and ser.n_components == 2,
        "18c_stop_boundary": stop_row.leakage > 0.0,
        "18d_mutation_is_codon_xor": all(
            mutation_q(c, n, encodings[0]).q6
            == (codon_state(c, encodings[0]).bits ^ codon_state(n, encodings[0]).bits)
            for c in CODONS
            for n in one_base_neighbors(c)
        ),
        "18f_vert_horiz_partition": n_vert + n_horiz == (64 * 9) // 2,
        "11h_phase_pair_count": len(phase_pairs_d(CHIRALITY_D)) == 4,
        "11i_ref_fold_rank3_rc_rank2": r0.fold_rank_plus_i == 3 and r0.rc_rank_plus_i == 2,
    }

    return ChartQuotientCensus(
        kernel_ok=kernel_ok,
        kernel_note=kernel_note,
        n_encodings=len(reports),
        n_distinct_phi=len({r.phi for r in reports}),
        n_wc_translation=sum(1 for r in reports if r.wc_ok),
        n_ts_translation=sum(1 for r in reports if r.ts_ok),
        n_tv_translation=sum(1 for r in reports if r.tv_ok),
        n_full_k4=sum(1 for r in reports if r.k4_span == 2),
        n_rc_affine=sum(1 for r in reports if r.rc_affine),
        n_q6_exact=n_q6,
        n_fold_eq_rc=n_fold,
        n_similar_fold_rc=n_sim,
        n_orbit_elem=n_elem,
        n_orbit_pair=n_pair,
        n_orbit_other=n_other,
        wt2_assignment=dict(wt2),
        reports=reports,
        degeneracy=deg,
        fiber_rows=fiber_rows,
        serine=ser,
        stop_row=stop_row,
        n_aa=n_aa,
        c63=binom(6, 3),
        aut_rows=tuple(aut_rows),
        aut_standard=aut_standard,
        energy_rows=tuple(energy_rows),
        null_rows=tuple(null_rows),
        fold_hist=fold_disagreement_census(encodings[0]),
        homology_rows=tuple(homology_rows),
        beta1_standard=beta1_std,
        h34_mult=H34_MULTIPLICITIES,
        block_E=block_E,
        grade3_rank=g3_rank,
        grade3_rank_with_stop=g3_stop,
        stop_trp_rank=st_rank,
        stop_trp_hull=st_hull,
        stop_trp_has_trp=st_trp,
        n_pair_s2_gt_elem=n_pair_wins,
        n_tables_s2=n_tables_s2,
        n_vert_edges=n_vert,
        n_horiz_edges=n_horiz,
        gates=gates,
    )


def print_chart_census(c: ChartQuotientCensus, ref_k4: Dict[str, int]) -> None:
    g = c.gates
    h_std = next(r for r in c.homology_rows if r.table_id == 1)

    report_section("1. NUCLEOTIDE CHART AND TRANSLATION QUOTIENT")
    report_objects(
        (
            "24 Aff charts; orbits elem/pair/other; synonymous beta1; fold vs RC; S2",
        )
    )
    print("  encodings")
    print(
        f"    n={c.n_encodings}  orbits elem/pair/other="
        f"{c.n_orbit_elem}/{c.n_orbit_pair}/{c.n_orbit_other}"
    )
    print(
        f"    chemical K4 on reference chart: "
        f"WC={ref_k4['ref_wc_delta']:02b}/wt{ref_k4['ref_wc_wt']} "
        f"TS={ref_k4['ref_ts_delta']:02b}/wt{ref_k4['ref_ts_wt']} "
        f"TV={ref_k4['ref_tv_delta']:02b}/wt{ref_k4['ref_tv_wt']}"
    )
    print(
        f"    WC/TS/TV act by translation on all charts: "
        f"{c.n_wc_translation}/{c.n_ts_translation}/{c.n_tv_translation} of {c.n_encodings}"
    )
    print(
        f"    fold==RC as maps: {c.n_fold_eq_rc}/{c.n_encodings}; "
        f"fold~RC (same GF2 rank pattern): {c.n_similar_fold_rc}/{c.n_encodings}"
    )
    n_comm_e = sum(
        1 for r in c.reports if r.orbit == ORBIT_ELEMENTARY and r.map8_commute == r.map8_total
    )
    n_comm_p = sum(
        1 for r in c.reports if r.orbit == ORBIT_PAIR_INV and r.map8_commute == r.map8_total
    )
    print(
        f"    fold o RC == RC o fold: elementary {n_comm_e}/8; pair_inversion {n_comm_p}/8"
    )
    print()
    print("  synonymous homology (NCBI tables)")
    print_table(
        ("tid", "name", "E", "C", "beta1"),
        (4, 22, 4, 3, 5),
        [
            (
                r.table_id,
                CODE_NAMES.get(r.table_id, str(r.table_id)),
                r.n_edges,
                r.n_components,
                r.beta1,
            )
            for r in c.homology_rows
        ],
        aligns=(">", "<", ">", ">", ">"),
    )
    print(
        f"    standard (tid=1): V={N_CODONS} E={h_std.n_edges} C={h_std.n_components} "
        f"beta1={h_std.beta1}  (= E-V+C)"
    )
    print(
        f"    H(3,4) Walsh-block multiplicities (j=0..3): {c.h34_mult}  "
        f"(slot j=2 equals {c.h34_mult[2]})"
    )
    print(
        f"    AA count={c.n_aa}  C(6,3)={c.c63}  serine size/components="
        f"{c.serine.size}/{c.serine.n_components}  stop leakage={c.stop_row.leakage:.4f}"
    )
    print(
        f"    vert/horiz one-base edges (tid=1): {c.n_vert_edges}/{c.n_horiz_edges} "
        f"(sum={c.n_vert_edges + c.n_horiz_edges}, H(3,4) undirected edges={(64 * 9) // 2})"
    )
    print(
        f"    grade-3 projection rank (20 AA / +stop): {c.grade3_rank}/{c.grade3_rank_with_stop}"
    )
    print(
        f"    stop+Trp hull size={c.stop_trp_hull} rank={c.stop_trp_rank} "
        f"contains Trp={c.stop_trp_has_trp}"
    )
    print(
        f"    mean S2(pair_inversion) > mean S2(elementary): "
        f"{c.n_pair_s2_gt_elem}/{c.n_tables_s2} NCBI tables"
    )
    print()
    report_checks((
        ('24 affine encodings', g['1_n24_encodings'], f'n={c.n_encodings}', f'n={N_NUCLEOTIDE_ENCODINGS}'),
        ('orbits 8/8/8', g['11c_orbits_8_8_8'], f'{c.n_orbit_elem}/{c.n_orbit_pair}/{c.n_orbit_other}', '8/8/8'),
        ('WC/TS/TV translations on every chart', g['2_wc_translation_all'] and g['3_ts_translation_all'] and g['4_tv_translation_all'], f'{c.n_wc_translation}/{c.n_ts_translation}/{c.n_tv_translation}', f'{c.n_encodings}/{c.n_encodings}/{c.n_encodings}'),
        ('K4 span=2 on every chart', g['5_k4_span_2_all'], f'n_full_k4={c.n_full_k4}', f'n_full_k4={c.n_encodings}'),
        ('codon packing is a bijection 64->64', g['6_codon_bijection'], 'all encodings bijective', 'all encodings bijective'),
        ('RC is an affine involution of rank 6', g['7_rc_affine_involution'], f'n_rc_affine={c.n_rc_affine}', f'n_rc_affine={c.n_encodings}'),
        ('fold and RC are distinct maps', g['9_fold_neq_rc'], f'fold_eq_rc={c.n_fold_eq_rc}', 'fold_eq_rc=0'),
        ('fold and RC have different GF2 ranks', g['10_fold_rc_rank_gap'], f'similar={c.n_similar_fold_rc}', 'similar=0'),
        ('fold/RC commute counts by orbit (pair full, elementary empty)', g['11j_dual_chart_obstruction'], f'elem_commute={n_comm_e}/8 pair_commute={n_comm_p}/8', 'elem_commute=0/8 pair_commute=8/8'),
        ('kernel d=6 API binds', g['11_kernel_api'], c.kernel_note, 'q_word, step_uv, gates match api at d=6'),
        ('standard-code beta1', g['15_beta1_standard_27'], f'beta1={c.beta1_standard} (E={h_std.n_edges}, C={h_std.n_components})', 'beta1=27'),
        ('beta1 equals H(3,4) multiplicity slot j=2', g['15b_beta1_h34_match'], f'beta1={c.beta1_standard} mult={c.h34_mult}', f'beta1={c.h34_mult[2]}'),
        ('20 amino acids (excl stop)', g['12b_twenty_aa'], f'n_aa={c.n_aa}', 'n_aa=20'),
        ('serine fiber disconnected', g['18b_serine_disconnected'], f'size={c.serine.size} components={c.serine.n_components}', 'size=6 components=2'),
        ('stop has positive leakage to sense codons', g['18c_stop_boundary'], f'leakage={c.stop_row.leakage:.4f}', 'leakage>0'),
        ('vert+horiz partition H(3,4) edges', g['18f_vert_horiz_partition'], f'{c.n_vert_edges}+{c.n_horiz_edges}={c.n_vert_edges + c.n_horiz_edges}', f'sum={64 * 9 // 2}'),
        ('pair_inversion S2 > elementary S2 on all NCBI tables', g['17b_pair_s2_gt_elem'], f'{c.n_pair_s2_gt_elem}/{c.n_tables_s2}', f'{c.n_tables_s2}/{c.n_tables_s2}'),
        ('grade-3 AA rank in (0,20)', g['16_grade3_rank_lt20'], f'rank={c.grade3_rank}', '0 < rank < 20'),
        ('adding stop raises grade-3 rank', g['16b_grade3_stop_gt_aa'], f'rank_aa={c.grade3_rank} rank_aa+stop={c.grade3_rank_with_stop}', 'rank_aa+stop > rank_aa'),
        ('q6 packing matches predicted', g['8_q6_packing_consistency'], f'n_q6_exact={c.n_q6_exact}', f'n={c.n_encodings}'),
        ('R/Y,M/K,S/W are linear bits', g['11b_chemical_bits_linear'], 'all charts', 'all charts'),
        ('fold matches phase-pair mask', g['11d_fold_phase_pairs'], 'all charts', 'all charts'),
        ('fold is an 8-bit involution', g['11e_fold_8bit_involution'], 'all charts', 'all charts'),
        ('RC is an 8-bit involution', g['11f_rc_8bit_involution'], 'all charts', 'all charts'),
        ('chirality phase-pair count', g['11h_phase_pair_count'], 'count from kernel', '4'),
        ('ref chart fold rank+I=3, RC rank+I=2', g['11i_ref_fold_rank3_rc_rank2'], 'see gates', '3 and 2'),
        ('degeneracy multiset sums to 64', g['12_degeneracy_64'], f'sum={sum(c.degeneracy)}', '64'),
        ('pointwise Aut order on standard code', g['13_aut_wreath_order_2'], 'n_aut_pointwise', '2'),
        ('Walsh Plancherel sum E=64^2', g['14_wht_plancherel'], 'sum E', '4096'),
        ('block-weight Plancherel sum=64^2', g['14c_block_plancherel'], f'sum={sum(c.block_E)}', '4096'),
        ('stop+Trp affine hull', g['17_stop_trp_hull4'], f'hull={c.stop_trp_hull} rank={c.stop_trp_rank}', 'hull=4 rank=2 with Trp'),
        ('wobble Aut not full K4', g['18_wobble_not_full_k4'], 'elem and pair <4', 'both <4'),
        ('one-base mutation = codon XOR', g['18d_mutation_is_codon_xor'], 'all neighbors', 'all neighbors'),
    ))


def _j2_character_matrix() -> Tuple[List[int], np.ndarray]:
    idxs: List[int] = []
    rows: List[List[int]] = []
    for r in range(64):
        if block_weight6(r) == 2:
            idxs.append(r)
            rows.append([walsh_sign6(q, r) for q in range(64)])
    return idxs, np.array(rows, dtype=float)


def _syn_edges_payload(code: Dict[str, str], enc: NucleotideEncoding) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for c in CODONS:
        if code[c] == "*":
            continue
        u = pack_codon_bits(c, enc)
        for n in one_base_neighbors(c):
            if code[n] != code[c]:
                continue
            v = pack_codon_bits(n, enc)
            if u < v:
                out.append((u, v))
    return out


def _spanning_forest_edges(vertices: Set[int], edges: Sequence[Tuple[int, int]]) -> Set[Tuple[int, int]]:
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


def _tree_path(tree: Set[Tuple[int, int]], u: int, v: int, adj: Dict[int, List[int]]) -> Set[Tuple[int, int]]:
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
    path: Set[Tuple[int, int]] = set()
    x = v
    while x != u:
        p = prev[x]
        path.add((min(p, x), max(p, x)))
        x = p
    return path


def fundamental_cycle_vertex_sets(code: Dict[str, str], enc: NucleotideEncoding) -> List[Set[int]]:
    edges = _syn_edges_payload(code, enc)
    vertices: Set[int] = set()
    for u, v in edges:
        vertices.add(u)
        vertices.add(v)
    tree = _spanning_forest_edges(vertices, edges)
    tadj: Dict[int, List[int]] = defaultdict(list)
    for u, v in tree:
        tadj[u].append(v)
        tadj[v].append(u)
    cycles: List[Set[int]] = []
    for u, v in edges:
        e = (u, v)
        if e in tree:
            continue
        cyc = set(_tree_path(tree, u, v, tadj))
        cyc.add(e)
        verts: Set[int] = set()
        for eu, ev in cyc:
            verts.add(eu)
            verts.add(ev)
        cycles.append(verts)
    return cycles


def _bridge_indicator(
    code: Dict[str, str],
    enc: NucleotideEncoding,
    aa: str,
    box_a: Set[str],
    box_b: Set[str],
) -> np.ndarray:
    ind = np.zeros(64)
    for c in box_a:
        if code.get(c) != aa:
            continue
        u = pack_codon_bits(c, enc)
        for n in one_base_neighbors(c):
            if n in box_b and code.get(n) == aa:
                ind[u] = 1.0
                ind[pack_codon_bits(n, enc)] = 1.0
    return ind


def _component_indicators(code: Dict[str, str], enc: NucleotideEncoding, aa: str) -> List[np.ndarray]:
    from hqvm_cgm_genomics_common import fiber_components

    group = [c for c in CODONS if code[c] == aa]
    comps = fiber_components(group)
    out = []
    for comp in comps:
        ind = np.zeros(64)
        for c in comp:
            ind[pack_codon_bits(c, enc)] = 1.0
        out.append(ind)
    return out


@dataclass
class CycleJ2BasisCensus:
    n_cycles: int
    n_j2: int
    rank: int
    dim_ker: int
    dim_coker: int
    ker_support: Tuple[Tuple[int, ...], ...]
    coker_j2_idxs: Tuple[Tuple[int, ...], ...]
    singular_names: Tuple[str, ...]
    singular_coker_rank: int
    singular_match: bool
    singular_relation: Tuple[int, ...]
    ser_stop_coker_equal: bool
    rank_hist: Dict[int, int]
    n_rank24: int
    singular_span_rank24: int
    gates: Dict[str, bool]


def _singular_indicators_payload(code: Dict[str, str], enc) -> Tuple[np.ndarray, ...]:
    leu_a = {"TTA", "TTG"}
    leu_b = {"CTA", "CTC", "CTG", "CTT"}
    arg_a = {"AGA", "AGG"}
    arg_b = {"CGA", "CGC", "CGG", "CGT"}
    ind_leu = _bridge_indicator(code, enc, "L", leu_a, leu_b)
    ind_arg = _bridge_indicator(code, enc, "R", arg_a, arg_b)
    ser_inds = _component_indicators(code, enc, "S")
    ind_ser = ser_inds[0] - ser_inds[1] if len(ser_inds) == 2 else np.zeros(64)
    ind_stop = np.zeros(64)
    for c in CODONS:
        if code[c] == "*":
            ind_stop[pack_codon_bits(c, enc)] = 1.0
    return ind_leu, ind_arg, ind_ser, ind_stop


def _cycle_j2_map(code: Dict[str, str], enc, B: np.ndarray, tol: float = 1e-8):
    cycles = fundamental_cycle_vertex_sets(code, enc)
    projs = []
    for verts in cycles:
        ind = np.zeros(64)
        for v in verts:
            ind[v] = 1.0
        projs.append(B @ ind)
    M = np.array(projs) if projs else np.zeros((0, B.shape[0]))
    U, s, Vt = np.linalg.svd(M, full_matrices=True)
    rank = int((s > tol).sum())
    dim_ker = M.shape[0] - rank
    dim_coker = M.shape[1] - rank
    ker_basis = Vt[rank:].T if dim_ker else np.zeros((M.shape[0], 0))
    coker_basis = U[:, rank:] if dim_coker else np.zeros((M.shape[1], 0))
    return cycles, M, rank, dim_ker, dim_coker, ker_basis, coker_basis


def cycle_j2_basis_census(code: Dict[str, str] = STANDARD_CODE) -> CycleJ2BasisCensus:
    """Canonical cycle-to-j=2 census for the genomics suite.

    Construction: sense-only synonymous edges (_syn_edges_payload), Kruskal
    spanning forest on sorted edges (fundamental_cycle_vertex_sets), vertex
    indicators projected onto Walsh characters of block-weight 2. Rank 24 with
    three-dimensional kernel and cokernel on the pair_inversion reference chart.
    Four named singular directions span the cokernel with one relation:
    Ser_sheet_split and stop_tree share the same cokernel class (reference).
    Across all 24 encodings the rank histogram is {23:8, 24:8, 25:8}, and on
    every rank-24 chart the singular span equals the cokernel.
    """
    enc = encodings_in_orbit(ORBIT_PAIR_INV)[0][1]
    j2_idxs, B = _j2_character_matrix()
    tol = 1e-8
    cycles, M, rank, dim_ker, dim_coker, ker_basis, coker_basis = _cycle_j2_map(
        code, enc, B, tol
    )

    ker_support: List[Tuple[int, ...]] = []
    for k in range(dim_ker):
        col = ker_basis[:, k]
        supp = tuple(i for i, x in enumerate(col) if abs(x) > 1e-6)
        ker_support.append(supp)

    coker_j2: List[Tuple[int, ...]] = []
    for k in range(dim_coker):
        col = coker_basis[:, k]
        supp = tuple(j2_idxs[i] for i, x in enumerate(col) if abs(x) > 1e-6)
        coker_j2.append(supp)

    ind_leu, ind_arg, ind_ser, ind_stop = _singular_indicators_payload(code, enc)
    singular = (
        ("Leu_TTR_CTN", ind_leu),
        ("Arg_AGR_CGN", ind_arg),
        ("Ser_sheet_split", ind_ser),
        ("stop_tree", ind_stop),
    )
    S = np.array([B @ ind for _n, ind in singular])
    if dim_coker > 0:
        coords = coker_basis.T @ S.T
        singular_coker_rank = int(np.linalg.matrix_rank(coords, tol=tol))
        ser_stop_eq = bool(np.allclose(coords[:, 2], coords[:, 3], atol=1e-6))
        _U2, s2, Vt2 = np.linalg.svd(coords, full_matrices=True)
        null_rows = [Vt2[i] for i, sv in enumerate(s2) if sv < tol]
        null_rows.extend(Vt2[i] for i in range(len(s2), Vt2.shape[0]))
        if null_rows:
            a = null_rows[0] / (np.max(np.abs(null_rows[0])) + 1e-15)
            best = np.rint(a)
            best_err = float(np.max(np.abs(a - best)))
            for scale in range(1, 13):
                cand = a * scale
                rounded = np.rint(cand)
                err = float(np.max(np.abs(cand - rounded)))
                if err < best_err:
                    best_err = err
                    best = rounded
            relation = tuple(int(x) for x in best.tolist())
        else:
            relation = ()
    else:
        singular_coker_rank = 0
        ser_stop_eq = False
        relation = ()
    singular_match = dim_coker > 0 and singular_coker_rank == dim_coker

    rank_hist: Dict[int, int] = Counter()
    n_rank24 = 0
    singular_span_rank24 = 0
    for enc_i in all_nucleotide_encodings():
        _cyc, _M, r_i, _dk, dc_i, _kb, cb_i = _cycle_j2_map(code, enc_i, B, tol)
        rank_hist[r_i] += 1
        if r_i != 24 or dc_i != 3:
            continue
        n_rank24 += 1
        inds = _singular_indicators_payload(code, enc_i)
        coords_i = cb_i.T @ np.column_stack([B @ ind for ind in inds])
        if int(np.linalg.matrix_rank(coords_i, tol=tol)) == 3:
            singular_span_rank24 += 1

    gates = {
        "140_cycle_j2_square_rank24": rank == 24 and dim_ker == 3 and dim_coker == 3,
        "141_singular_span_coker": singular_match,
        "311_ser_stop_coker_equal": ser_stop_eq and relation == (0, 0, -1, 1),
        "312_rank_hist_8_8_8": dict(rank_hist) == {23: 8, 24: 8, 25: 8},
        "313_singular_span_all_rank24": (
            n_rank24 == 8 and singular_span_rank24 == n_rank24
        ),
    }
    return CycleJ2BasisCensus(
        n_cycles=len(cycles),
        n_j2=len(j2_idxs),
        rank=rank,
        dim_ker=dim_ker,
        dim_coker=dim_coker,
        ker_support=tuple(ker_support),
        coker_j2_idxs=tuple(coker_j2),
        singular_names=tuple(n for n, _ in singular),
        singular_coker_rank=singular_coker_rank,
        singular_match=singular_match,
        singular_relation=relation,
        ser_stop_coker_equal=ser_stop_eq,
        rank_hist=dict(sorted(rank_hist.items())),
        n_rank24=n_rank24,
        singular_span_rank24=singular_span_rank24,
        gates=gates,
    )


def print_cycle_j2_basis_census(c: CycleJ2BasisCensus) -> None:
    g = c.gates
    report_section("19. CYCLE-TO-J2 KER/COKER BASES")
    report_objects(
        (
            "27 cycles -> Walsh j=2; ker/coker bases; Leu/Arg/Ser/stop singular "
            "directions; Ser~stop cokernel relation on pair_inversion; "
            "rank histogram and singular-span on all 24 encodings",
        )
    )
    print(
        f"    cycles={c.n_cycles} j2={c.n_j2} rank={c.rank} "
        f"dim_ker={c.dim_ker} dim_coker={c.dim_coker}"
    )
    print(f"    ker cycle-index supports: {c.ker_support}")
    print(f"    coker j2-index supports: {c.coker_j2_idxs}")
    print(
        f"    singular={c.singular_names} coker_rank={c.singular_coker_rank} "
        f"match={c.singular_match}"
    )
    print(
        f"    singular_relation={c.singular_relation} "
        f"ser_stop_coker_equal={c.ser_stop_coker_equal}"
    )
    print(
        f"    rank_hist={c.rank_hist} n_rank24={c.n_rank24} "
        f"singular_span_rank24={c.singular_span_rank24}/{c.n_rank24}"
    )
    print()
    report_checks((
        ("cycle->j=2 map square (27 cycles, 27 chars), rank 24", g["140_cycle_j2_square_rank24"], f"rank={c.rank} ker={c.dim_ker} coker={c.dim_coker}", "rank=24 ker=3 coker=3"),
        ("Leu/Arg/Ser/stop singular directions span cokernel (rank 3 of 3)", g["141_singular_span_coker"], f"singular_coker_rank={c.singular_coker_rank}", "=dim_coker=3"),
        ("Ser_sheet_split ~ stop_tree in cokernel (relation 0,0,-1,1)", g["311_ser_stop_coker_equal"], f"rel={c.singular_relation} eq={c.ser_stop_coker_equal}", "(0,0,-1,1)"),
        ("rank histogram over 24 encodings is 8+8+8 at 23/24/25", g["312_rank_hist_8_8_8"], str(c.rank_hist), "{23:8, 24:8, 25:8}"),
        ("singular span equals cokernel on every rank-24 chart", g["313_singular_span_all_rank24"], f"{c.singular_span_rank24}/{c.n_rank24}", "8/8"),
    ))
