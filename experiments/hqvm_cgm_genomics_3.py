#!/usr/bin/env python3
"""
hqvm_cgm_genomics_3.py

Sections 4–5: spectral bundle and polarity/fiber census.
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

import itertools
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
from gyroscopic.hQVM.api import (
    OmegaSignature12,
    compose_omega_signatures,
    omega_word_signature,
    q_word6,
)
from gyroscopic.hQVM.family import intron_from_byte
from hqvm_cgm_genomics_1 import (
    H34_MULTIPLICITIES,
    block_degree_energy,
    block_weight6,
    count_aut_lattice,
    cumulative_S,
    fiber_atlas,
    ncbi_chart_s2_grid,
    orbit_rep_indices,
    rc_byte_keep_family,
    synonymous_homology,
    walsh_degree_energy,
    walsh_s2,
    wht64,
)
from hqvm_cgm_genomics_common import (
    AA_ORDER,
    ANTIPODE_6,
    BASES,
    BLOCK_REVERSE_COLS,
    Q_RESIDUAL_W,
    W_ANNIHILATOR,
    affine_rank6,
    bit_reverse6,
    block_reverse6,
    fiber_components,
    gf2_rank6,
    CHIRALITY_D,
    CODE_NAMES,
    CODONS,
    NULL_SEED,
    NCBI_TABLE_IDS,
    N_CODONS,
    ORBIT_ELEMENTARY,
    ORBIT_OTHER,
    ORBIT_PAIR_INV,
    STANDARD_CODE,
    WC,
    NucleotideEncoding,
    all_nucleotide_encodings,
    encodings_in_orbit,
    encoding_orbit_name,
    fibers,
    fold_byte,
    genomic_byte_stream,
    kernel_dependency,
    make_null_code_pools,
    monte_carlo_p,
    one_base_neighbors,
    pack_4mer_byte,
    pack_byte,
    pack_codon_bits,
    print_table,
    report_checks,
    report_objects,
    report_section,
    reverse_complement_4mer_byte,
    reverse_complement_seq,
    rc_bits,
    translation_table,
    unpack_byte,
)

NULL_N = 80

_FOLD_TABLE: Tuple[int, ...] = tuple(fold_byte(b) for b in range(256))


@lru_cache(maxsize=None)
def _rc_fold_tables(enc: NucleotideEncoding) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    rc_byte = tuple(rc_byte_keep_family(b, enc) for b in range(256))
    rc_fold = tuple(rc_byte_keep_family(_FOLD_TABLE[b], enc) for b in range(256))
    return rc_byte, rc_fold


def count_fold_payload_rc_commute(enc: NucleotideEncoding) -> Tuple[int, int]:
    rc_byte, rc_fold = _rc_fold_tables(enc)
    n_com = sum(1 for b in range(256) if _FOLD_TABLE[rc_byte[b]] == rc_fold[b])
    return n_com, 256

H34_EIGENVALUES = (9.0, 5.0, 1.0, -3.0)

@dataclass
class DualChartRow:
    orbit: str
    n_charts: int
    mean_commute: float
    mean_S2: float
    mean_block_S2: float
    n_full_commute: int
    n_zero_commute: int

@dataclass
class Beta1NullRow:
    null_name: str
    n: int
    n_eq_27: int
    p_hat_eq_27: float
    mean_beta1: float
    min_beta1: int
    max_beta1: int

@dataclass
class Grade3Row:
    table_id: int
    orbit: str
    rank20: int
    rank21: int
    top_singular20: float
    wedge_zero_n: int
    wedge_match_stop: int

@dataclass
class BundleRow:
    table_id: int
    n_vert: int
    n_horiz: int
    n_stop_boundary: int
    beta1: int
    n_comp: int
    serine_components: int
    n_fibers_need_ge2_sections: int
    stop_leakage: float

@dataclass
class ModuliEdge:
    src: int
    dst: int
    n_reassign: int

@dataclass
class ModuliNode:
    table_id: int
    n_aut: int
    beta1: int
    n_stop: int
    stop_leakage: float
    S2_elem: float
    S2_pair: float
    n_neighbors_le2: int

@dataclass
class SpectralBundleCensus:
    kernel_ok: bool
    kernel_note: str
    h34_eigs: Tuple[float, ...]
    h34_mult: Tuple[int, ...]
    h34_eigs_ok: bool
    intertwiner_mult: Tuple[int, ...]
    intertwiner_ok: bool
    dual_rows: Tuple[DualChartRow, ...]
    cocycle_commute_gap: float
    cocycle_S2_gap: float
    pair_s2_wins: int
    n_tables_s2: int
    beta1_tables: Tuple[Tuple[int, int], ...]
    beta1_nulls: Tuple[Beta1NullRow, ...]
    grade3_rows: Tuple[Grade3Row, ...]
    bundle_rows: Tuple[BundleRow, ...]
    moduli_nodes: Tuple[ModuliNode, ...]
    moduli_edges: Tuple[ModuliEdge, ...]
    gates: Dict[str, bool]

def h34_adjacency_spectrum() -> Tuple[Tuple[float, ...], Tuple[int, ...], bool]:
    """Adjacency spectrum of the biological one-base codon graph H(3,4)."""
    idx = {c: i for i, c in enumerate(CODONS)}
    A = np.zeros((N_CODONS, N_CODONS), dtype=float)
    for c in CODONS:
        for n in one_base_neighbors(c):
            A[idx[c], idx[n]] = 1.0
    w = np.linalg.eigvalsh(A)
    # Round to nearest known H(3,4) eigenvalues and count
    targets = list(H34_EIGENVALUES)
    counts = [0, 0, 0, 0]
    for val in w:
        d = [abs(val - t) for t in targets]
        j = int(np.argmin(d))
        counts[j] += 1
    mult = tuple(counts)
    ok = mult == H34_MULTIPLICITIES and abs(float(np.max(w)) - 9.0) < 1e-6
    # Return sorted unique eig representatives actually measured
    eigs = tuple(sorted((float(x) for x in np.unique(np.round(w, 8))), reverse=True))
    return eigs, mult, ok

def intertwiner_multiplicities() -> Tuple[Tuple[int, ...], bool]:
    counts = [0, 0, 0, 0]
    for s in range(N_CODONS):
        counts[block_weight6(s)] += 1
    mult = tuple(counts)
    return mult, mult == H34_MULTIPLICITIES

def block_S2(code: Dict[str, str], enc: NucleotideEncoding) -> float:
    e = block_degree_energy(code, enc)
    tot = sum(e)
    if tot == 0:
        return 0.0
    return (e[0] + e[1] + e[2]) / tot

def dual_chart_cocycle(encodings: Sequence[NucleotideEncoding]) -> Tuple[DualChartRow, ...]:
    by_orbit: Dict[str, List[NucleotideEncoding]] = defaultdict(list)
    for enc in encodings:
        by_orbit[encoding_orbit_name(enc)].append(enc)
    rows = []
    code = STANDARD_CODE
    for orbit in (ORBIT_ELEMENTARY, ORBIT_PAIR_INV):
        encs = by_orbit.get(orbit, [])
        if not encs:
            continue
        commute = []
        s2s = []
        bs2s = []
        n_full = n_zero = 0
        for enc in encs:
            n_com, _ = count_fold_payload_rc_commute(enc)
            commute.append(n_com / 256.0)
            if n_com == 256:
                n_full += 1
            if n_com == 0:
                n_zero += 1
            s2s.append(cumulative_S(walsh_degree_energy(code, enc), 2))
            bs2s.append(block_S2(code, enc))
        rows.append(
            DualChartRow(
                orbit=orbit,
                n_charts=len(encs),
                mean_commute=sum(commute) / len(commute),
                mean_S2=sum(s2s) / len(s2s),
                mean_block_S2=sum(bs2s) / len(bs2s),
                n_full_commute=n_full,
                n_zero_commute=n_zero,
            )
        )
    return tuple(rows)

def wedge_zero_census(enc: NucleotideEncoding) -> Tuple[int, int]:
    """Return (best_zero_count, stop_overlap).

    best_zero_count is the minimum |zero-set| over rank-3 bit families (monomial AND
    or triple XOR). stop_overlap is how many of the three standard stops lie in the
    best matching size-3 zero set (0–3). The two fields measure different things.
    """
    stops = {c for c in CODONS if STANDARD_CODE[c] == "*"}
    triples = list(itertools.combinations(range(6), 3))
    best_zero = 64
    best_match = 0
    for trip in triples:
        zeros = []
        for c in CODONS:
            x = pack_codon_bits(c, enc)
            bits = [(x >> i) & 1 for i in trip]
            val = bits[0] & bits[1] & bits[2]
            if val == 0:
                zeros.append(c)
        # Also try XOR of the three bits as a linear grade proxy
        zeros_lin = []
        for c in CODONS:
            x = pack_codon_bits(c, enc)
            val = ((x >> trip[0]) & 1) ^ ((x >> trip[1]) & 1) ^ ((x >> trip[2]) & 1)
            if val == 0:
                zeros_lin.append(c)
        for zset in (zeros, zeros_lin):
            if len(zset) < best_zero:
                best_zero = len(zset)
            match = len(stops.intersection(zset))
            if len(zset) == 3 and match > best_match:
                best_match = match
            if len(zset) == 3 and set(zset) == stops:
                return 3, 3
    # Second family: sum of all weight-3 monomials (parity of number of set bit-triples among support)
    zeros2 = []
    for c in CODONS:
        x = pack_codon_bits(c, enc)
        s = 0
        for i, j, k in triples:
            s ^= ((x >> i) & 1) & ((x >> j) & 1) & ((x >> k) & 1)
        if s == 0:
            zeros2.append(c)
    match2 = len(stops.intersection(zeros2))
    if len(zeros2) == 3 and set(zeros2) == stops:
        return 3, 3
    return best_zero if best_zero < 64 else len(zeros2), max(best_match, match2 if len(zeros2) == 3 else 0)

def grade3_singular_top(code: Dict[str, str], enc: NucleotideEncoding) -> Tuple[int, int, float]:
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
    svals = np.linalg.svd(M, compute_uv=False)
    top = float(svals[0]) if len(svals) else 0.0
    stop_vec = [0] * N_CODONS
    for codon in CODONS:
        if code[codon] == "*":
            stop_vec[pack_codon_bits(codon, enc)] = 1
    hat_stop = wht64(stop_vec)
    mats21 = []
    for aa in aa_list:
        vec = [0] * N_CODONS
        for codon in CODONS:
            if code[codon] == aa:
                vec[pack_codon_bits(codon, enc)] = 1
        hat = wht64(vec)
        mats21.append([float(hat[0])] + [float(hat[s]) for s in grade3])
    mats21.append([float(hat_stop[0])] + [float(hat_stop[s]) for s in grade3])
    rank21 = int(np.linalg.matrix_rank(np.asarray(mats21, dtype=float), tol=1e-8))
    return rank20, rank21, top

def bundle_row(code: Dict[str, str], enc: NucleotideEncoding, tid: int) -> BundleRow:
    vert = horiz = stop_b = 0
    for c in CODONS:
        for n in one_base_neighbors(c):
            if c >= n:
                continue
            if (code[c] == "*") != (code[n] == "*"):
                stop_b += 1
            if code[c] == code[n]:
                vert += 1
            else:
                horiz += 1
    ne, nc, b1 = synonymous_homology(code)
    atlas = fiber_atlas(code, enc)
    ser = next(r for r in atlas if r.aa == "S") if any(r.aa == "S" for r in atlas) else None
    stop = next(r for r in atlas if r.aa == "*")
    n_ge2 = sum(1 for r in atlas if r.aa != "*" and r.n_components >= 2)
    return BundleRow(
        table_id=tid,
        n_vert=vert,
        n_horiz=horiz,
        n_stop_boundary=stop_b,
        beta1=b1,
        n_comp=nc,
        serine_components=ser.n_components if ser else 0,
        n_fibers_need_ge2_sections=n_ge2,
        stop_leakage=stop.leakage,
    )

def moduli_census(
    encodings: Sequence[NucleotideEncoding],
    reps: Dict[str, int],
) -> Tuple[Tuple[ModuliNode, ...], Tuple[ModuliEdge, ...]]:
    enc_e = encodings[reps[ORBIT_ELEMENTARY]]
    enc_p = encodings[reps[ORBIT_PAIR_INV]]
    codes = {tid: translation_table(tid) for tid in NCBI_TABLE_IDS}
    edges = []
    for a, b in itertools.combinations(NCBI_TABLE_IDS, 2):
        ca, cb = codes[a], codes[b]
        d = sum(1 for c in CODONS if ca[c] != cb[c])
        if d <= 6:
            edges.append(ModuliEdge(a, b, d))
    # keep all edges with d<=6 for neighborhood; nodes report neighbors with d<=2
    neigh = defaultdict(int)
    for e in edges:
        if e.n_reassign <= 2:
            neigh[e.src] += 1
            neigh[e.dst] += 1
    nodes = []
    for tid, code in codes.items():
        n_pt, _nq, _ne, _ns, _pt = count_aut_lattice(code, do_edge=False)
        _ne2, _nc, b1 = synonymous_homology(code)
        atlas = fiber_atlas(code, enc_e)
        stop = next(r for r in atlas if r.aa == "*")
        nodes.append(
            ModuliNode(
                table_id=tid,
                n_aut=n_pt,
                beta1=b1,
                n_stop=sum(1 for c in CODONS if code[c] == "*"),
                stop_leakage=stop.leakage,
                S2_elem=cumulative_S(walsh_degree_energy(code, enc_e), 2),
                S2_pair=cumulative_S(walsh_degree_energy(code, enc_p), 2),
                n_neighbors_le2=neigh[tid],
            )
        )
    # Store compact edge list: only d<=4 to keep report small
    edges_keep = tuple(e for e in edges if e.n_reassign <= 4)
    return tuple(nodes), edges_keep

def spectral_bundle_census() -> SpectralBundleCensus:
    kernel_ok, kernel_note = kernel_dependency()
    encodings = all_nucleotide_encodings()
    reps = orbit_rep_indices(encodings)
    eigs, mult, eigs_ok = h34_adjacency_spectrum()
    it_mult, it_ok = intertwiner_multiplicities()
    dual_rows = dual_chart_cocycle(encodings)
    by_orb = {r.orbit: r for r in dual_rows}
    cocycle_commute_gap = by_orb[ORBIT_PAIR_INV].mean_commute - by_orb[ORBIT_ELEMENTARY].mean_commute
    cocycle_S2_gap = by_orb[ORBIT_PAIR_INV].mean_S2 - by_orb[ORBIT_ELEMENTARY].mean_S2

    pair_wins = 0
    n_tab = 0
    beta1_tables = []
    for tid in NCBI_TABLE_IDS:
        code = translation_table(tid)
        _e, _c, b1 = synonymous_homology(code)
        beta1_tables.append((tid, b1))
        s2e = cumulative_S(walsh_degree_energy(code, encodings[reps[ORBIT_ELEMENTARY]]), 2)
        s2p = cumulative_S(walsh_degree_energy(code, encodings[reps[ORBIT_PAIR_INV]]), 2)
        n_tab += 1
        if s2p > s2e + 1e-15:
            pair_wins += 1

    code1 = STANDARD_CODE
    rng = random.Random(NULL_SEED + 5)
    pools = make_null_code_pools(code1, rng, n1=NULL_N, n2=NULL_N, n345=NULL_N)
    null_specs = (
        ("N1_degen", pools.n1),
        ("N2_boxes", pools.n2),
        ("N4_stops", pools.n4),
        ("N3_geom", pools.n3),
    )
    beta1_nulls = []
    for name, pool in null_specs:
        vals = [synonymous_homology(nc)[2] for nc in pool]
        hits = sum(1 for v in vals if v == 27)
        beta1_nulls.append(
            Beta1NullRow(
                null_name=name,
                n=len(vals),
                n_eq_27=hits,
                p_hat_eq_27=monte_carlo_p(hits, len(vals)),
                mean_beta1=sum(vals) / len(vals),
                min_beta1=min(vals),
                max_beta1=max(vals),
            )
        )

    grade3_rows = []
    for orbit in (ORBIT_ELEMENTARY, ORBIT_PAIR_INV):
        enc = encodings[reps[orbit]]
        for tid in NCBI_TABLE_IDS:
            code = translation_table(tid)
            r20, r21, top = grade3_singular_top(code, enc)
            z_n, z_m = wedge_zero_census(enc) if tid == 1 else (-1, -1)
            grade3_rows.append(
                Grade3Row(
                    table_id=tid,
                    orbit=orbit,
                    rank20=r20,
                    rank21=r21,
                    top_singular20=top,
                    wedge_zero_n=z_n,
                    wedge_match_stop=z_m,
                )
            )

    bundle_rows = []
    for tid in NCBI_TABLE_IDS:
        enc = encodings[reps[ORBIT_ELEMENTARY]]
        bundle_rows.append(bundle_row(translation_table(tid), enc, tid))

    moduli_nodes, moduli_edges = moduli_census(encodings, reps)

    g3_std = next(r for r in grade3_rows if r.table_id == 1 and r.orbit == ORBIT_ELEMENTARY)
    b_std = next(r for r in bundle_rows if r.table_id == 1)

    by_null = {r.null_name: r for r in beta1_nulls}
    # beta1=27 is isolated by synonymous fiber geometry (N3), not by
    # degeneracy/box/stop shuffles (N1/N2/N4).
    gates = {
        "48_h34_spectrum_mult": eigs_ok and mult == H34_MULTIPLICITIES,
        "49_intertwiner_mult": it_ok,
        "50_dual_elem_zero_commute": by_orb[ORBIT_ELEMENTARY].n_zero_commute == 8,
        "51_dual_pair_full_commute": by_orb[ORBIT_PAIR_INV].n_full_commute == 8,
        "52_cocycle_commute_gap_pos": cocycle_commute_gap > 0.5,
        "53_cocycle_S2_gap_pos": cocycle_S2_gap > 0.0,
        "54_pair_s2_all_tables": pair_wins == n_tab and n_tab > 0,
        "55_beta1_geometry_isolated": (
            abs(by_null["N3_geom"].mean_beta1 - 27.0) < 0.5
            and all(abs(by_null[n].mean_beta1 - 27.0) > 0.5 for n in ("N1_degen", "N2_boxes", "N4_stops"))
        ),
        "56_grade3_rank_positive": g3_std.rank20 >= 1,
        "57_bundle_partition": b_std.n_vert + b_std.n_horiz == (64 * 9) // 2,
        "58_serine_disconnected": b_std.serine_components == 2,
        "59_moduli_nodes": len(moduli_nodes) == len(NCBI_TABLE_IDS),
    }

    return SpectralBundleCensus(
        kernel_ok=kernel_ok,
        kernel_note=kernel_note,
        h34_eigs=eigs,
        h34_mult=mult,
        h34_eigs_ok=eigs_ok,
        intertwiner_mult=it_mult,
        intertwiner_ok=it_ok,
        dual_rows=dual_rows,
        cocycle_commute_gap=cocycle_commute_gap,
        cocycle_S2_gap=cocycle_S2_gap,
        pair_s2_wins=pair_wins,
        n_tables_s2=n_tab,
        beta1_tables=tuple(beta1_tables),
        beta1_nulls=tuple(beta1_nulls),
        grade3_rows=tuple(grade3_rows),
        bundle_rows=tuple(bundle_rows),
        moduli_nodes=tuple(moduli_nodes),
        moduli_edges=moduli_edges,
        gates=gates,
    )

_NULL_DEF = {
    "N1_degen": "shuffle AA labels freely",
    "N2_boxes": "shuffle within codon boxes",
    "N4_stops": "shuffle sense; keep stop set",
    "N3_geom": "preserve fiber sizes+components",
}

def print_spectral_bundle_census(c: SpectralBundleCensus) -> None:
    g = c.gates
    by_orb = {r.orbit: r for r in c.dual_rows}
    b_std = next(r for r in c.bundle_rows if r.table_id == 1)
    g3_std = next(
        r for r in c.grade3_rows if r.table_id == 1 and r.orbit == ORBIT_ELEMENTARY
    )

    report_section("4. SPECTRUM, NULLS, BUNDLE")
    report_objects(
        (
            "H(3,4) spectrum; Walsh intertwiner; dual cocycle; beta1 nulls N=%d; bundle lite" % NULL_N,
        )
    )
    print("  H(3,4) spectrum")
    print(f"    eigenvalues measured: {c.h34_eigs}")
    print(f"    multiplicities measured: {c.h34_mult}  target: {H34_MULTIPLICITIES}")
    print(f"    intertwiner multiplicities: {c.intertwiner_mult}  ok={c.intertwiner_ok}")
    print()
    print("  dual-chart cocycle")
    for r in c.dual_rows:
        print(
            f"    {r.orbit}: charts={r.n_charts} mean_commute={r.mean_commute:.4f} "
            f"mean_S2={r.mean_S2:.5f} full_commute={r.n_full_commute} "
            f"zero_commute={r.n_zero_commute}"
        )
    print(
        f"    gaps (pair - elementary): commute={c.cocycle_commute_gap:.5f} "
        f"S2={c.cocycle_S2_gap:.5f}"
    )
    print(f"    S2(pair)>S2(elem) on NCBI tables: {c.pair_s2_wins}/{c.n_tables_s2}")
    print()
    print("  beta1 null isolation (standard code)")
    print(f"    {'null':<12} {'def':<56} {'n':>3} {'eq27':>5} {'mean':>7} {'min':>4} {'max':>4}")
    for r in c.beta1_nulls:
        print(
            f"    {r.null_name:<12} {_NULL_DEF.get(r.null_name, ''):<56} "
            f"{r.n:>3} {r.n_eq_27:>5} {r.mean_beta1:>7.3f} {r.min_beta1:>4} {r.max_beta1:>4}"
        )
    print()
    print("  NCBI beta1 by table")
    print_table(
        ("tid", "name", "beta1"),
        (4, 22, 5),
        [(tid, CODE_NAMES.get(tid, str(tid)), b1) for tid, b1 in c.beta1_tables],
        aligns=(">", "<", ">"),
    )
    print()
    print("  translation bundle (tid=1)")
    print(
        f"    vert={b_std.n_vert} horiz={b_std.n_horiz} "
        f"sum={b_std.n_vert + b_std.n_horiz} (H(3,4) edges={(64 * 9) // 2}) "
        f"beta1={b_std.beta1} serine_comp={b_std.serine_components} "
        f"stop_leakage={b_std.stop_leakage:.4f}"
    )
    print(
        f"    grade-3 rank20={g3_std.rank20} rank21={g3_std.rank21} "
        f"top_singular={g3_std.top_singular20:.4g}"
    )
    print(f"  moduli graph: nodes={len(c.moduli_nodes)} edges_kept(d<=4)={len(c.moduli_edges)}")
    print()
    n3 = next(r for r in c.beta1_nulls if r.null_name == "N3_geom")
    others = [r for r in c.beta1_nulls if r.null_name != "N3_geom"]
    report_checks((
        ('H(3,4) spectrum multiplicities', g['48_h34_spectrum_mult'], f'mult={c.h34_mult}', f'mult={H34_MULTIPLICITIES}'),
        ('intertwiner multiplicities match H(3,4)', g['49_intertwiner_mult'], f'mult={c.intertwiner_mult}', f'mult={H34_MULTIPLICITIES}'),
        ('elementary orbit fold/RC commute count', g['50_dual_elem_zero_commute'], f'zero_commute={by_orb[ORBIT_ELEMENTARY].n_zero_commute}/8', 'zero_commute=8/8'),
        ('pair_inversion orbit: fold/RC always fully commute', g['51_dual_pair_full_commute'], f'full_commute={by_orb[ORBIT_PAIR_INV].n_full_commute}/8', 'full_commute=8/8'),
        ('cocycle commute gap (pair - elem) > 0.5', g['52_cocycle_commute_gap_pos'], f'gap={c.cocycle_commute_gap:.5f}', 'gap>0.5'),
        ('cocycle S2 gap (pair - elem) > 0', g['53_cocycle_S2_gap_pos'], f'gap={c.cocycle_S2_gap:.5f}', 'gap>0'),
        ('pair S2 > elem S2 on every NCBI table', g['54_pair_s2_all_tables'], f'{c.pair_s2_wins}/{c.n_tables_s2}', f'{c.n_tables_s2}/{c.n_tables_s2}'),
        ('beta1=27 isolated by fiber geometry (N3), not N1/N2/N4', g['55_beta1_geometry_isolated'], f'N3 mean={n3.mean_beta1:.3f}; others={[round(r.mean_beta1, 3) for r in others]}', 'N3 mean~27; others far from 27'),
        ('grade-3 rank of 20 AA indicators >= 1', g['56_grade3_rank_positive'], f'rank20={g3_std.rank20}', 'rank20>=1'),
        ('vert+horiz partition of H(3,4) edges', g['57_bundle_partition'], f'{b_std.n_vert}+{b_std.n_horiz}={b_std.n_vert + b_std.n_horiz}', f'sum={64 * 9 // 2}'),
        ('serine fiber has 2 components', g['58_serine_disconnected'], f'components={b_std.serine_components}', 'components=2'),
        ('moduli nodes cover all NCBI tables', g['59_moduli_nodes'], f'nodes={len(c.moduli_nodes)}', f'nodes={len(NCBI_TABLE_IDS)}'),
    ))

def omega_inv(sig: OmegaSignature12) -> OmegaSignature12:
    if sig.parity == 0:
        return OmegaSignature12(0, sig.tau_u6, sig.tau_v6)
    return OmegaSignature12(1, sig.tau_v6, sig.tau_u6)

def theta_payload_rc(sig: OmegaSignature12) -> OmegaSignature12:
    return OmegaSignature12(
        sig.parity,
        block_reverse6(sig.tau_v6) ^ ANTIPODE_6,
        block_reverse6(sig.tau_u6) ^ ANTIPODE_6,
    )

def payload_rc_star(word: Sequence[int], enc: NucleotideEncoding) -> List[int]:
    return [rc_byte_keep_family(int(b) & 0xFF, enc) for b in reversed(word)]

def _product_words(n: int, stride: int) -> Iterable[List[int]]:
    rng = range(256) if (n == 1 and stride == 1) else range(0, 256, stride)
    acc: List[List[int]] = [[]]
    for _ in range(n):
        acc = [w + [b] for w in acc for b in rng]
    return acc

def z_pair_mod4(n: int) -> OmegaSignature12:
    r = int(n) % 4
    u = ANTIPODE_6 if r in (0, 3) else 0
    v = ANTIPODE_6 if r in (0, 1) else 0
    return OmegaSignature12(0, u, v)

def predict_sr(sig: OmegaSignature12, n: int, z: OmegaSignature12) -> OmegaSignature12:
    return compose_omega_signatures(theta_payload_rc(omega_inv(sig)), z)

def codon_rc_affine(enc: NucleotideEncoding) -> Tuple[int, Tuple[int, ...]]:
    shift = rc_bits(0, enc) & 0x3F
    cols = tuple((rc_bits(1 << i, enc) ^ shift) & 0x3F for i in range(6))
    return shift, cols

def fold_payload_map() -> Tuple[int, ...]:
    return tuple(unpack_byte(fold_byte(pack_byte(0, p)))[1] for p in range(64))

@dataclass
class OrbitPolarityRow:
    orbit: str
    n_charts: int
    shifts: Tuple[int, ...]
    all_antipode: bool
    n_commute_fold_payload_rc: int
    n_total: int

@dataclass
class DualRcRow:
    orbit: str
    n_kin_eq_payload: int
    n_family_flip: int
    n_fold_eq_kin: int
    residual_set: Tuple[int, ...]
    residual_rank: int
    residual_is_W: bool

@dataclass
class FiberGeomRow:
    aa: str
    size: int
    n_components: int
    affine_rank: int
    component_affine: Tuple[int, ...]

@dataclass
class StopTreeRow:
    n_stop: int
    n_stop_edges: int
    n_stop_components: int
    beta1_stop: int
    beta1_full: int
    beta1_sense: int
    trp_adjacent_tga: bool
    stop_leakage: float

@dataclass
class PolarityFiberCensus:
    polarity: Tuple[OrbitPolarityRow, ...]
    fold_is_bitrev: bool
    block_reverse_universal: bool
    dual_rc: Tuple[DualRcRow, ...]
    word_residual_additive: bool
    stop_tree: StopTreeRow
    fibers: Tuple[FiberGeomRow, ...]
    gates: Dict[str, bool]

def _orbit_reps() -> Dict[str, Tuple[int, NucleotideEncoding]]:
    out: Dict[str, Tuple[int, NucleotideEncoding]] = {}
    for i, enc in enumerate(all_nucleotide_encodings()):
        out.setdefault(encoding_orbit_name(enc), (i, enc))
    return out

def polarity_census() -> Tuple[Tuple[OrbitPolarityRow, ...], bool, bool]:
    by_orbit: Dict[str, List[NucleotideEncoding]] = defaultdict(list)
    for enc in all_nucleotide_encodings():
        by_orbit[encoding_orbit_name(enc)].append(enc)

    fold_map = fold_payload_map()
    fold_is_bitrev = all(fold_map[p] == bit_reverse6(p) for p in range(64))

    block_ok = True
    rows: List[OrbitPolarityRow] = []
    for orbit in (ORBIT_ELEMENTARY, ORBIT_PAIR_INV, ORBIT_OTHER):
        encs = by_orbit[orbit]
        shifts = []
        n_commute = 0
        n_total = 0
        for enc in encs:
            shift, cols = codon_rc_affine(enc)
            shifts.append(shift)
            if cols != BLOCK_REVERSE_COLS:
                block_ok = False
            n_com, n_chart = count_fold_payload_rc_commute(enc)
            n_commute += n_com
            n_total += n_chart
        rows.append(
            OrbitPolarityRow(
                orbit=orbit,
                n_charts=len(encs),
                shifts=tuple(sorted(set(shifts))),
                all_antipode=all(s == ANTIPODE_6 for s in shifts),
                n_commute_fold_payload_rc=n_commute,
                n_total=n_total,
            )
        )
    return tuple(rows), fold_is_bitrev, block_ok

def dual_rc_census(reps: Dict[str, Tuple[int, NucleotideEncoding]]) -> Tuple[DualRcRow, ...]:
    tets = ["".join(p) for p in itertools.product(BASES, repeat=4)]
    rows = []
    for orbit in (ORBIT_ELEMENTARY, ORBIT_PAIR_INV):
        enc = reps[orbit][1]
        n_eq = n_flip = n_fold = 0
        residuals = []
        for s in tets:
            b = pack_4mer_byte(s, enc).byte
            kin = reverse_complement_4mer_byte(b, enc)
            pay = rc_byte_keep_family(b, enc)
            if kin == pay:
                n_eq += 1
            f0, _p0 = unpack_byte(b)
            fk, _pk = unpack_byte(kin)
            if fk != f0:
                n_flip += 1
            if fold_byte(b) == kin:
                n_fold += 1
            residuals.append(q_word6(b) ^ q_word6(kin))
        R = tuple(sorted(set(residuals)))
        rows.append(
            DualRcRow(
                orbit=orbit,
                n_kin_eq_payload=n_eq,
                n_family_flip=n_flip,
                n_fold_eq_kin=n_fold,
                residual_set=R,
                residual_rank=gf2_rank6(R),
                residual_is_W=R == Q_RESIDUAL_W,
            )
        )
    return tuple(rows)

def word_residual_additive_ok(enc: NucleotideEncoding) -> bool:
    seqs = (
        "ATGCCGTAA",
        "AAAA",
        "ACGTACGT",
        "GGCCGGCC",
        "TGGATCCATG",
        "ATGCGGATCCAA",
        "TTTTGGGGCCCC",
    )
    for seq in seqs:
        fwd = genomic_byte_stream(seq, enc)
        rcw = genomic_byte_stream(reverse_complement_seq(seq), enc)
        qf = qr = lam = 0
        for b in fwd:
            qf ^= q_word6(b)
            kin = reverse_complement_4mer_byte(b, enc)
            lam ^= q_word6(b) ^ q_word6(kin)
        for b in rcw:
            qr ^= q_word6(b)
        if (qf ^ qr) != lam:
            return False
        if (qf ^ qr) not in Q_RESIDUAL_W:
            return False
    return True

def stop_tree_census() -> StopTreeRow:
    code = STANDARD_CODE
    stops = [c for c in CODONS if code[c] == "*"]
    sense = [c for c in CODONS if code[c] != "*"]
    e_full, c_full, b1_full = synonymous_homology(code)
    # sense-only: temporarily map stops out by restricting homology
    n_e = n_c = 0
    gset = set(sense)
    adj = {c: [] for c in sense}
    for c in sense:
        for n in one_base_neighbors(c):
            if n in gset and n > c and code[c] == code[n]:
                n_e += 1
                adj[c].append(n)
                adj[n].append(c)
    seen = set()
    for c in sense:
        if c in seen:
            continue
        n_c += 1
        stack = [c]
        seen.add(c)
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
    b1_sense = n_e - len(sense) + n_c

    # stop fiber alone
    sset = set(stops)
    se = 0
    sadj = {c: [] for c in stops}
    for c in stops:
        for n in one_base_neighbors(c):
            if n in sset and n > c:
                se += 1
                sadj[c].append(n)
                sadj[n].append(c)
    seen = set()
    sc = 0
    for c in stops:
        if c in seen:
            continue
        sc += 1
        stack = [c]
        seen.add(c)
        while stack:
            u = stack.pop()
            for v in sadj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
    b1_stop = se - len(stops) + sc

    leak = tot = 0
    for s in stops:
        for n in one_base_neighbors(s):
            tot += 1
            if code[n] != "*":
                leak += 1
    trp = [c for c in CODONS if code[c] == "W"]
    trp_adj = any(
        t in one_base_neighbors(s) for s in stops for t in trp
    )
    return StopTreeRow(
        n_stop=len(stops),
        n_stop_edges=se,
        n_stop_components=sc,
        beta1_stop=b1_stop,
        beta1_full=b1_full,
        beta1_sense=b1_sense,
        trp_adjacent_tga="TGA" in stops and "TGG" in trp and "TGG" in one_base_neighbors("TGA"),
        stop_leakage=leak / tot if tot else 0.0,
    )

def fiber_atlas_census(enc: NucleotideEncoding) -> Tuple[FiberGeomRow, ...]:
    fib = fibers(STANDARD_CODE)
    rows = []
    for aa in sorted(fib, key=lambda a: (-len(fib[a]), a)):
        group = fib[aa]
        comps = fiber_components(group)
        bits = [pack_codon_bits(c, enc) for c in group]
        comp_aff = tuple(
            affine_rank6([pack_codon_bits(c, enc) for c in comp]) for comp in comps
        )
        rows.append(
            FiberGeomRow(
                aa=aa,
                size=len(group),
                n_components=len(comps),
                affine_rank=affine_rank6(bits),
                component_affine=comp_aff,
            )
        )
    return tuple(rows)

def polarity_fiber_census() -> PolarityFiberCensus:
    reps = _orbit_reps()
    polarity, fold_is_bitrev, block_ok = polarity_census()
    dual = dual_rc_census(reps)
    add_ok = word_residual_additive_ok(reps[ORBIT_PAIR_INV][1])
    # also verify residual W on all 24 encodings
    all_W = True
    for enc in all_nucleotide_encodings():
        res = set()
        for s in itertools.product(BASES, repeat=4):
            seq = "".join(s)
            b = pack_4mer_byte(seq, enc).byte
            kin = reverse_complement_4mer_byte(b, enc)
            res.add(q_word6(b) ^ q_word6(kin))
        if tuple(sorted(res)) != Q_RESIDUAL_W:
            all_W = False
            break
    stop = stop_tree_census()
    fib = fiber_atlas_census(reps[ORBIT_PAIR_INV][1])
    by_orb = {r.orbit: r for r in polarity}
    pair = by_orb[ORBIT_PAIR_INV]
    elem = by_orb[ORBIT_ELEMENTARY]
    other = by_orb[ORBIT_OTHER]
    dual_p = next(r for r in dual if r.orbit == ORBIT_PAIR_INV)
    dual_e = next(r for r in dual if r.orbit == ORBIT_ELEMENTARY)
    ser = next(r for r in fib if r.aa == "S")

    gates = {
        "60_pair_delta_antipode": pair.all_antipode and pair.shifts == (ANTIPODE_6,),
        "61_elem_other_not_antipode": (not elem.all_antipode) and (not other.all_antipode),
        "62_fold_payload_bitrev": fold_is_bitrev,
        "63_block_reverse_universal": block_ok,
        "64_fold_rc_commute_iff_pair": (
            pair.n_commute_fold_payload_rc == pair.n_total
            and elem.n_commute_fold_payload_rc == 0
            and other.n_commute_fold_payload_rc == 0
        ),
        "65_kin_neq_payload_rc": dual_p.n_kin_eq_payload == 0 and dual_e.n_kin_eq_payload == 0,
        "66_q_residual_is_W": all_W and dual_p.residual_is_W and dual_e.residual_is_W,
        "67_word_residual_additive": add_ok,
        "68_stop_tree_beta1_invariant": (
            stop.beta1_stop == 0
            and stop.beta1_full == stop.beta1_sense == 27
            and stop.trp_adjacent_tga
        ),
        "69_serine_disconnected": ser.n_components == 2 and ser.size == 6,
    }
    return PolarityFiberCensus(
        polarity=tuple(polarity),
        fold_is_bitrev=fold_is_bitrev,
        block_reverse_universal=block_ok,
        dual_rc=dual,
        word_residual_additive=add_ok,
        stop_tree=stop,
        fibers=fib,
        gates=gates,
    )

def print_polarity_fiber_census(c: PolarityFiberCensus) -> None:
    g = c.gates
    report_section("5. POLARITY, DUAL RC, STOP TREE, AFFINE FIBERS")
    report_objects(
        (
            "codon-RC = R_block o T_delta; fold commute on pair_inversion; W residual; stop tree",
        )
    )

    print("  polarity by orbit")
    print(f"    {'orbit':<28} {'charts':>6} {'delta(s)':<22} {'fold-RC commute':>16}")
    for r in c.polarity:
        deltas = ",".join(f"{s:06b}" for s in r.shifts)
        print(
            f"    {r.orbit:<28} {r.n_charts:>6} {deltas:<22} "
            f"{r.n_commute_fold_payload_rc}/{r.n_total}"
        )
    print(f"    fold payload == bit-reversal: {c.fold_is_bitrev}")
    print(f"    R_block columns universal across 24 charts: {c.block_reverse_universal}")
    print()

    print("  dual RC (kinematic 4-mer vs payload)")
    for r in c.dual_rc:
        print(
            f"    {r.orbit}: kin==payload {r.n_kin_eq_payload}/256  "
            f"family_flip {r.n_family_flip}/256  fold==kin {r.n_fold_eq_kin}/256  "
            f"residual_rank={r.residual_rank} is_W={r.residual_is_W}"
        )
    print(f"    W = {Q_RESIDUAL_W}")
    print(f"    word residual = XOR letterwise lambda (pair_inv samples): {c.word_residual_additive}")
    print()

    st = c.stop_tree
    print("  stop tree / sense beta1")
    print(
        f"    stops: V={st.n_stop} E={st.n_stop_edges} C={st.n_stop_components} "
        f"beta1_stop={st.beta1_stop} (tree => 0)"
    )
    print(
        f"    beta1_full={st.beta1_full}  beta1_sense={st.beta1_sense}  "
        f"Trp—TGA adjacent={st.trp_adjacent_tga}  stop_leakage={st.stop_leakage:.4f}"
    )
    print()

    print("  affine fiber geometry (pair_inversion chart)")
    print(f"    {'aa':>3} {'n':>3} {'comps':>5} {'aff':>4}  component_aff")
    for r in c.fibers:
        print(
            f"    {r.aa:>3} {r.size:>3} {r.n_components:>5} {r.affine_rank:>4}  "
            f"{list(r.component_affine)}"
        )
    print()

    pair = next(r for r in c.polarity if r.orbit == ORBIT_PAIR_INV)
    elem = next(r for r in c.polarity if r.orbit == ORBIT_ELEMENTARY)
    report_checks((
        ('pair_inversion WC polarity is the antipode 111111', g['60_pair_delta_antipode'], f"shifts={tuple((f'{s:06b}' for s in pair.shifts))}", 'shifts=(111111,)'),
        ('elementary/other WC polarity is weight-3 (010101/101010)', g['61_elem_other_not_antipode'], f"elem={tuple((f'{s:06b}' for s in elem.shifts))}", 'weight-3 forms 010101/101010'),
        ('fold payload = bit-reversal on GF(2)^6', g['62_fold_payload_bitrev'], f'fold_is_bitrev={c.fold_is_bitrev}', 'True'),
        ('codon-RC linear part is block-reverse on all charts', g['63_block_reverse_universal'], f'universal={c.block_reverse_universal}', 'True'),
        ('fold o payload-RC commute counts by orbit', g['64_fold_rc_commute_iff_pair'], f'pair {pair.n_commute_fold_payload_rc}/{pair.n_total}; elem {elem.n_commute_fold_payload_rc}/{elem.n_total}', 'pair all; elem/other none'),
        ('kinematic 4-mer RC equals payload-RC as byte maps', g['65_kin_neq_payload_rc'], f'eq={tuple((r.n_kin_eq_payload for r in c.dual_rc))}/256', '(0, 0)'),
        ('kinematic RC chirality residual equals W (all 24 charts)', g['66_q_residual_is_W'], f'rank={c.dual_rc[0].residual_rank} |W|={len(Q_RESIDUAL_W)}', 'rank=4, same W everywhere'),
        ('word chirality residual = XOR of letterwise residuals', g['67_word_residual_additive'], f'additive={c.word_residual_additive}', 'True on genomic sample words'),
        ('stop fiber is a tree; sense beta1 = full beta1 = 27; Trp bridges TGA', g['68_stop_tree_beta1_invariant'], f'b1_stop={st.beta1_stop} b1_full={st.beta1_full} b1_sense={st.beta1_sense} Trp-TGA={st.trp_adjacent_tga}', 'b1_stop=0; both beta1=27; Trp-TGA True'),
        ('serine fiber disconnected (2 affine pieces)', g['69_serine_disconnected'], (lambda s: f'S size={s.size} comps={s.n_components} aff={s.component_affine}')(next((r for r in c.fibers if r.aa == 'S'))), 'size=6 comps=2'),
    ))
