#!/usr/bin/env python3
"""
hqvm_cgm_genomics_2.py

Sections 2–3: kernel layer census and sequence-window audits.
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

import gzip
import itertools
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from gyroscopic.hQVM.api import (
    OmegaState12,
    OmegaSignature12,
    apply_omega_gate,
    chirality_word6,
    compose_word_signatures,
    k4_orbit,
    omega12_to_state24,
    omega_word_signature,
    pairdiag12_to_word6,
    q_word6,
    step_omega12_by_byte,
    word6_to_pairdiag12,
    word_signature,
)
from gyroscopic.hQVM.family import (
    HqvmD,
    bfs_reach,
    build_hqvm_d,
    enumerate_bytes,
    fiber_complete,
    fold_disagreement_d,
    gf2_rank,
    predicted_cluster_size,
)
from gyroscopic.hQVM.constants import ab_distance, horizon_distance, unpack_state
from hqvm_cgm_genomics_common import (
    BASES,
    CHIRALITY_D,
    CODONS,
    DATA_DIR,
    NULL_SEED,
    N_CODONS,
    ORBIT_ELEMENTARY,
    ORBIT_PAIR_INV,
    STANDARD_CODE,
    STRONG,
    WC,
    NucleotideEncoding,
    all_nucleotide_encodings,
    carrier_from_codon_pair,
    codon_state,
    encoding_orbit_name,
    extract_chr22_cds,
    fold_byte,
    fold_matches_phase_pairs,
    gc_fraction,
    genomic_byte_stream,
    gtf_path,
    hamming6,
    involution_delta,
    iter_codons,
    iter_kmers,
    kernel_dependency,
    load_chr22_sequence,
    monte_carlo_p,
    mutation_class,
    mutation_q,
    one_base_neighbors,
    pack_4mer_byte,
    pack_byte,
    pack_codon_bits,
    parse_fasta,
    print_table,
    report_checks,
    report_objects,
    report_section,
    reverse_complement_codon,
    reverse_complement_seq,
    unpack_4mer_byte,
    unpack_byte,
)

_ENGINE: Optional[HqvmD] = None


def _engine() -> HqvmD:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = build_hqvm_d(CHIRALITY_D)
    return _ENGINE

def orbit_reps() -> Dict[str, Tuple[int, NucleotideEncoding]]:
    out = {}
    for i, enc in enumerate(all_nucleotide_encodings()):
        name = encoding_orbit_name(enc)
        if name not in out:
            out[name] = (i, enc)
    return out

@dataclass
class EtaRow:
    source: str
    orbit: str
    gl2: Tuple[int, int, int, int]
    shift: int
    n_k4: int
    rc_cov: int
    rc_total: int
    occupancy: Tuple[int, ...]

@dataclass
class DualChartRow:
    orbit: str
    n: int
    fold_eq: int
    mean_hamming: float
    mean_disagreement: float
    palindrome_family_delta: int
    palindrome_n: int

@dataclass
class PackingRow:
    orbit: str
    n_4mer: int
    n_equal: int
    n_payload_agree: int
    hinge_between_23: int

@dataclass
class OmegaRow:
    n_pairs: int
    chi_eq_xor: int
    pairdiag_roundtrip: int
    step_q_eq_chi: int
    gate_f_eq_epsilon: int

@dataclass
class GeneratorReachRow:
    name: str
    orbit: str
    n_bytes: int
    rank: int
    predicted: int
    reach: int
    giant: bool
    full: bool
    fiber_ok: bool
    odd_shells: int
    even_q_only: bool
    shells: Tuple[int, ...]

@dataclass
class HolonomyRow:
    family_mode: str
    n_gen: int
    n_pairs: int
    n_word_hom: int
    n_commute: int
    n_q6_add: int
    n_noncommute: int

@dataclass
class KernelLayerCensus:
    kernel_ok: bool
    kernel_note: str
    eta_rows: Tuple[EtaRow, ...]
    n_eta_nondeg_rc: int
    dual_rows: Tuple[DualChartRow, ...]
    packing_rows: Tuple[PackingRow, ...]
    omega: OmegaRow
    reach_rows: Tuple[GeneratorReachRow, ...]
    holonomy: Tuple[HolonomyRow, ...]
    gates: Dict[str, bool]

def _gl2() -> Tuple[Tuple[int, int, int, int], ...]:
    return (
        (1, 0, 0, 1),
        (0, 1, 1, 0),
        (1, 1, 0, 1),
        (1, 0, 1, 1),
        (0, 1, 1, 1),
        (1, 1, 1, 0),
    )

def apply_gl2(matrix: Tuple[int, int, int, int], x: int) -> int:
    x0 = int(x) & 1
    x1 = (int(x) >> 1) & 1
    a, b, c, d = matrix
    y0 = (a * x0 + b * x1) & 1
    y1 = (c * x0 + d * x1) & 1
    return y0 | (y1 << 1)

def eta_apply(matrix: Tuple[int, int, int, int], shift: int, bits: int) -> int:
    return apply_gl2(matrix, bits) ^ (int(shift) & 3)

def eta_census(enc: NucleotideEncoding, orbit: str) -> List[EtaRow]:
    _ok, wc_delta = involution_delta(WC, enc)
    rows = []
    sources = ("five_prime", "three_prime", "fourmer_outer")
    for source in sources:
        for mat in _gl2():
            for shift in range(4):
                occ = [0, 0, 0, 0]
                rc_ok = 0
                rc_tot = 0
                for b in BASES:
                    bits = enc.encode_base(b)
                    fam = eta_apply(mat, shift, bits)
                    occ[fam] += 1
                    wc_bits = enc.encode_base(WC[b])
                    fam_wc = eta_apply(mat, shift, wc_bits)
                    rc_tot += 1
                    if fam_wc == (fam ^ wc_delta):
                        rc_ok += 1
                n_k4 = sum(1 for c in occ if c > 0)
                rows.append(
                    EtaRow(
                        source=source,
                        orbit=orbit,
                        gl2=mat,
                        shift=shift,
                        n_k4=n_k4,
                        rc_cov=rc_ok,
                        rc_total=rc_tot,
                        occupancy=tuple(occ),
                    )
                )
    strand_occ = [0, 0, 0, 0]
    for b in BASES:
        for strand in (0, 1):
            fam = (enc.encode_base(b) ^ (strand * 3)) & 3
            strand_occ[fam] += 1
    rows.append(
        EtaRow(
            source="strand_xor",
            orbit=orbit,
            gl2=(1, 0, 0, 1),
            shift=0,
            n_k4=sum(1 for c in strand_occ if c > 0),
            rc_cov=0,
            rc_total=4,
            occupancy=tuple(strand_occ),
        )
    )
    return rows

def dual_chart_census(enc: NucleotideEncoding, orbit: str) -> DualChartRow:
    n = 0
    fold_eq = 0
    hamm = 0
    disag = 0
    pal_n = 0
    pal_delta = None
    pal_ok = 0
    _ok, wc_delta = involution_delta(WC, enc)
    for codon in CODONS:
        rc = reverse_complement_codon(codon)
        payload = pack_codon_bits(codon, enc)
        rc_payload = pack_codon_bits(rc, enc)
        for five in BASES:
            for three in BASES:
                fam_p = enc.encode_base(five)
                fam_m = enc.encode_base(WC[three])
                bp = pack_byte(fam_p, payload)
                bm = pack_byte(fam_m, rc_payload)
                fb = fold_byte(bp)
                n += 1
                if fb == bm:
                    fold_eq += 1
                hamm += (fb ^ bm).bit_count()
                disag += fold_disagreement_d(bp, CHIRALITY_D)
                if five == WC[three]:
                    pal_n += 1
                    dlt = fam_p ^ fam_m
                    if pal_delta is None:
                        pal_delta = dlt
                    if dlt == pal_delta:
                        pal_ok += 1
    return DualChartRow(
        orbit=orbit,
        n=n,
        fold_eq=fold_eq,
        mean_hamming=hamm / n if n else 0.0,
        mean_disagreement=disag / n if n else 0.0,
        palindrome_family_delta=int(pal_delta if pal_delta is not None else -1),
        palindrome_n=pal_n,
    )

def packing_census(enc: NucleotideEncoding, orbit: str) -> PackingRow:
    tetramers = ["".join(p) for p in itertools.product(BASES, repeat=4)]
    n_eq = 0
    n_pay = 0
    hinge = 0
    for seq4 in tetramers:
        linear = pack_4mer_byte(seq4, enc).byte
        bracket = pack_byte(enc.encode_base(seq4[0]), pack_codon_bits(seq4[1:], enc))
        if linear == bracket:
            n_eq += 1
        _lf, lmicro = unpack_byte(linear)
        _bf, bmicro = unpack_byte(bracket)
        if lmicro == bmicro:
            n_pay += 1
        intron_bits_ok = unpack_4mer_byte(linear, enc) == seq4
        if intron_bits_ok:
            hinge += 1
    return PackingRow(
        orbit=orbit,
        n_4mer=len(tetramers),
        n_equal=n_eq,
        n_payload_agree=n_pay,
        hinge_between_23=hinge,
    )

def omega_census(enc: NucleotideEncoding) -> OmegaRow:
    n = 0
    chi_eq = 0
    roundtrip = 0
    step_ok = 0
    gate_f = 0
    for x_c in CODONS:
        x = codon_state(x_c, enc)
        pd = word6_to_pairdiag12(x.bits)
        if pairdiag12_to_word6(pd) == x.bits:
            roundtrip += 1
        for y_c in CODONS:
            y = codon_state(y_c, enc)
            n += 1
            om = OmegaState12(u6=x.bits, v6=y.bits)
            chi_prop = om.chirality6
            chi_word = chirality_word6(omega12_to_state24(om))
            xor = x.bits ^ y.bits
            if chi_prop == xor == chi_word:
                chi_eq += 1
            byte = pack_byte(0, xor)
            stepped = step_omega12_by_byte(OmegaState12(u6=x.bits, v6=x.bits), byte)
            if stepped.chirality6 == (q_word6(byte) ^ 0):
                step_ok += 1
        om = OmegaState12(u6=x.bits, v6=x.bits)
        gated = apply_omega_gate(om, "F")
        if gated.chirality6 == 0 and gated.u6 == (x.bits ^ 0x3F):
            gate_f += 1
    return OmegaRow(
        n_pairs=n,
        chi_eq_xor=chi_eq,
        pairdiag_roundtrip=roundtrip,
        step_q_eq_chi=step_ok,
        gate_f_eq_epsilon=gate_f,
    )

def _mutation_bytes(
    enc: NucleotideEncoding,
    predicate,
    families: Sequence[int] = (0, 1, 2, 3),
) -> Tuple[int, ...]:
    out = set()
    code = STANDARD_CODE
    for c in CODONS:
        for n in one_base_neighbors(c):
            if not predicate(c, n, code):
                continue
            q = mutation_q(c, n, enc).q6
            for fam in families:
                out.add(pack_byte(fam, q))
    return tuple(sorted(out))

def _is_transition(c: str, n: str, _code) -> bool:
    return mutation_class(c, n) == "transition"

def _is_synonymous(c: str, n: str, code) -> bool:
    return code[c] == code[n]

def _is_inframe(_c: str, _n: str, _code) -> bool:
    return True

def _is_stop_preserving(c: str, n: str, code) -> bool:
    return (code[c] == "*") == (code[n] == "*")

def reach_row(name: str, orbit: str, allowed: Sequence[int]) -> GeneratorReachRow:
    eng = _engine()
    alphabet = enumerate_bytes(eng.d)
    allowed_set = set(int(b) for b in allowed)
    qs = [eng.q_by_byte[b] for b in allowed]
    rank = gf2_rank(qs, CHIRALITY_D)
    pred = predicted_cluster_size(rank)
    reach, _spans, giant, full = bfs_reach(eng, allowed, max_depth=2 * CHIRALITY_D + 6)
    fib = fiber_complete(allowed, eng)
    even_only = all((eng.q_weight[b] % 2) == 0 for b in allowed)
    shells = [0] * 7
    visited = bytearray(eng.n_omega)
    q = [eng.start_idx]
    visited[eng.start_idx] = 1
    byte_idx = [i for i, b in enumerate(alphabet) if b in allowed_set]
    head = 0
    while head < len(q):
        i = q[head]
        head += 1
        shells[eng.shell[i]] += 1
        row = eng.transitions[i]
        for bi in byte_idx:
            j = row[bi]
            if not visited[j]:
                visited[j] = 1
                q.append(j)
    odd = sum(shells[k] for k in range(1, 7, 2))
    return GeneratorReachRow(
        name=name,
        orbit=orbit,
        n_bytes=len(allowed_set),
        rank=rank,
        predicted=pred,
        reach=reach,
        giant=giant,
        full=full,
        fiber_ok=fib,
        odd_shells=odd,
        even_q_only=even_only,
        shells=tuple(shells),
    )

def mutation_generators(enc: NucleotideEncoding) -> Tuple[int, ...]:
    gens = []
    for pos in range(3):
        for delta in (1, 2, 3):
            gens.append(delta << (2 * (2 - pos)))
    return tuple(gens)

def holonomy_census(enc: NucleotideEncoding, family_mode: str) -> HolonomyRow:
    gens = mutation_generators(enc)
    n_word_hom = 0
    n_commute = 0
    n_q6_add = 0
    n_non = 0
    n_pairs = 0
    contexts = list(itertools.product(BASES, BASES)) if family_mode == "derived_fiveprime" else [(None, None)]
    for five, _three in contexts:
        fam_q = enc.encode_base(five) if five is not None else 0
        fam_r = fam_q
        for q, r in itertools.product(gens, repeat=2):
            n_pairs += 1
            bq = pack_byte(fam_q, q)
            br = pack_byte(fam_r, r)
            sig_q = word_signature([bq])
            sig_r = word_signature([br])
            word_qr = word_signature([bq, br])
            composed = compose_word_signatures(sig_r, sig_q)
            if (
                composed.parity == word_qr.parity
                and composed.tau_a12 == word_qr.tau_a12
                and composed.tau_b12 == word_qr.tau_b12
            ):
                n_word_hom += 1
            crq = compose_word_signatures(sig_q, sig_r)
            same = (
                composed.parity == crq.parity
                and composed.tau_a12 == crq.tau_a12
                and composed.tau_b12 == crq.tau_b12
            )
            if same:
                n_commute += 1
            else:
                n_non += 1
            om0 = OmegaState12(u6=0, v6=0)
            om1 = step_omega12_by_byte(om0, bq)
            om2 = step_omega12_by_byte(om1, br)
            if om2.chirality6 == (q_word6(bq) ^ q_word6(br)):
                n_q6_add += 1
    return HolonomyRow(
        family_mode=family_mode,
        n_gen=len(gens),
        n_pairs=n_pairs,
        n_word_hom=n_word_hom,
        n_commute=n_commute,
        n_q6_add=n_q6_add,
        n_noncommute=n_non,
    )

def kernel_layer_census() -> KernelLayerCensus:
    kernel_ok, kernel_note = kernel_dependency()
    reps = orbit_reps()
    eta_rows: List[EtaRow] = []
    dual_rows: List[DualChartRow] = []
    packing_rows: List[PackingRow] = []
    reach_rows: List[GeneratorReachRow] = []

    predicates = (
        ("transition_only", _is_transition),
        ("synonymous_only", _is_synonymous),
        ("frame_preserving", _is_inframe),
        ("stop_preserving", _is_stop_preserving),
    )

    for name in (ORBIT_ELEMENTARY, ORBIT_PAIR_INV):
        _idx, enc = reps[name]
        eta_rows.extend(eta_census(enc, name))
        dual_rows.append(dual_chart_census(enc, name))
        packing_rows.append(packing_census(enc, name))
        for gname, pred in predicates:
            allowed = _mutation_bytes(enc, pred)
            reach_rows.append(reach_row(gname, name, allowed))

    enc0 = reps[ORBIT_ELEMENTARY][1]
    omega = omega_census(enc0)
    hol = (
        holonomy_census(enc0, "family0"),
        holonomy_census(enc0, "derived_fiveprime"),
    )

    n_eta = sum(1 for r in eta_rows if r.n_k4 == 4 and r.rc_cov == r.rc_total and r.rc_total > 0)
    h0 = hol[0]
    hd = hol[1]
    dual_e = next(r for r in dual_rows if r.orbit == ORBIT_ELEMENTARY)
    trans_e = next(r for r in reach_rows if r.name == "transition_only" and r.orbit == ORBIT_ELEMENTARY)

    gates = {
        "22_eta_nondeg_rc_exists": n_eta > 0,
        "23_4mer_unpack_roundtrip": all(r.hinge_between_23 == r.n_4mer for r in packing_rows),
        "24_4mer_neq_bracket": all(r.n_equal < r.n_4mer for r in packing_rows),
        "25_chi_eq_xor": omega.chi_eq_xor == omega.n_pairs,
        "26_word_hom_family0": h0.n_word_hom == h0.n_pairs,
        "27_word_hom_derived": hd.n_word_hom == hd.n_pairs,
        "28_signature_noncommute_family0": h0.n_commute < h0.n_pairs,
        "29_signature_noncommute_derived": hd.n_noncommute > 0,
        "30_q6_additive": h0.n_q6_add == h0.n_pairs,
        "31_srct_predicted_positive": all(r.predicted >= 2 for r in reach_rows),
        "32_transition_rank_le6": trans_e.rank <= 6,
        "35_dual_not_identity_fold": dual_e.fold_eq < dual_e.n,
        "37_omega_roundtrip_64": omega.pairdiag_roundtrip == N_CODONS,
    }

    return KernelLayerCensus(
        kernel_ok=kernel_ok,
        kernel_note=kernel_note,
        eta_rows=tuple(eta_rows),
        n_eta_nondeg_rc=n_eta,
        dual_rows=tuple(dual_rows),
        packing_rows=tuple(packing_rows),
        omega=omega,
        reach_rows=tuple(reach_rows),
        holonomy=hol,
        gates=gates,
    )

def print_kernel_layer_census(c: KernelLayerCensus) -> None:
    g = c.gates
    o = c.omega
    dual_e = next(r for r in c.dual_rows if r.orbit == ORBIT_ELEMENTARY)
    dual_p = next(r for r in c.dual_rows if r.orbit == ORBIT_PAIR_INV)
    h0 = c.holonomy[0]
    hd = c.holonomy[1]
    trans_e = next(
        r for r in c.reach_rows if r.name == "transition_only" and r.orbit == ORBIT_ELEMENTARY
    )
    pack_e = next(r for r in c.packing_rows if r.orbit == ORBIT_ELEMENTARY)

    report_section("2. KERNEL LAYER (PACKING, OMEGA, WORD SIGNATURES)")
    report_objects(
        (
            "4-mer pack vs bracket; Omega step_uv; chi XOR; word signatures",
        )
    )
    print("  packing (linear 4-mer vs bracket)")
    for r in c.packing_rows:
        print(
            f"    {r.orbit}: n_4mer={r.n_4mer} equal_as_bytes={r.n_equal} "
            f"payload_agree={r.n_payload_agree} hinge_roundtrip={r.hinge_between_23}"
        )
    print()
    print("  dual chart (5' vs 3'/RC packing on codon windows)")
    for r in c.dual_rows:
        print(
            f"    {r.orbit}: n={r.n} fold_eq={r.fold_eq} "
            f"mean_hamming={r.mean_hamming:.4f} mean_disagreement={r.mean_disagreement:.4f}"
        )
    print()
    print("  Omega census")
    print(
        f"    codon-pairs={o.n_pairs} chi==XOR={o.chi_eq_xor} "
        f"pairdiag_roundtrip={o.pairdiag_roundtrip}/{N_CODONS} "
        f"step_q==chi={o.step_q_eq_chi} gate_F==epsilon={o.gate_f_eq_epsilon}"
    )
    print()
    print("  word-signature holonomy")
    for h in c.holonomy:
        print(
            f"    {h.family_mode}: pairs={h.n_pairs} word_hom={h.n_word_hom} "
            f"q6_add={h.n_q6_add} commute={h.n_commute} noncommute={h.n_noncommute}"
        )
    print(f"  eta charts with full K4 and RC coverage: {c.n_eta_nondeg_rc}")
    print()
    report_checks((
        ('eta nondegenerate RC chart exists', g['22_eta_nondeg_rc_exists'], f'n={c.n_eta_nondeg_rc}', 'n>0'),
        ('4-mer unpack roundtrip (hinge)', g['23_4mer_unpack_roundtrip'], 'hinge==n_4mer on both orbits', 'hinge==n_4mer'),
        ('linear 4-mer packing != bracket packing', g['24_4mer_neq_bracket'], f'equal={pack_e.n_equal}/{pack_e.n_4mer} (elementary)', 'equal < n_4mer'),
        ('chi equals payload XOR on all codon pairs', g['25_chi_eq_xor'], f'{o.chi_eq_xor}/{o.n_pairs}', f'{o.n_pairs}/{o.n_pairs}'),
        ('Omega pairdiag roundtrip on 64 codons', g['37_omega_roundtrip_64'], f'{o.pairdiag_roundtrip}', f'{N_CODONS}'),
        ('word homomorphism (family0)', g['26_word_hom_family0'], f'{h0.n_word_hom}/{h0.n_pairs}', f'{h0.n_pairs}/{h0.n_pairs}'),
        ('word homomorphism (derived_fiveprime)', g['27_word_hom_derived'], f'{hd.n_word_hom}/{hd.n_pairs}', f'{hd.n_pairs}/{hd.n_pairs}'),
        ('q6 additive under word compose (family0)', g['30_q6_additive'], f'{h0.n_q6_add}/{h0.n_pairs}', f'{h0.n_pairs}/{h0.n_pairs}'),
        ('signature commute count (family0)', g['28_signature_noncommute_family0'], f'commute={h0.n_commute}/{h0.n_pairs}', 'commute < n_pairs'),
        ('signatures have noncommuting pairs (derived)', g['29_signature_noncommute_derived'], f'noncommute={hd.n_noncommute}', 'noncommute>0'),
        ('dual chart fold equals identity count', g['35_dual_not_identity_fold'], f'fold_eq={dual_e.fold_eq}/{dual_e.n} (elementary); pair={dual_p.fold_eq}/{dual_p.n}', 'fold_eq < n'),
        ('transition generator rank <= 6', g['32_transition_rank_le6'], f'rank={trans_e.rank}', 'rank<=6'),
        ('SRC/T predicted shell count >= 2 on all generator sets', g['31_srct_predicted_positive'], 'all reach_rows.predicted>=2', 'predicted>=2'),
    ))

NULL_N = 80

MAX_CDS = 4000

MAX_WINDOWS = 8000

@dataclass
class AlphabetRow:
    orbit: str
    n_4mer: int
    fold_hist: Tuple[int, ...]
    mean_disagreement: float
    dual_n: int
    dual_fold_eq: int
    dual_mean_hamming: float

@dataclass
class GenomeRow:
    name: str
    status: str
    n_records: int
    n_codons: int
    gc: float
    orbit: str
    mean_4mer_fd: float
    mean_dual_h: float
    mean_consensus_shell: float
    family_occ: Tuple[int, int, int, int]
    dicodon_mean_shell: float
    usage_l1_vs_gcnull: float
    n1_hits: int
    n1_n: int
    n1_p: float
    fd_null_hits: int
    fd_null_n: int
    fd_null_p: float

@dataclass
class SpliceRow:
    kind: str
    status: str
    n: int
    orbit: str
    mean_4mer_fd: float
    gt_ag_frac: float
    null_p: float

@dataclass
class SequenceCensus:
    alphabet: Tuple[AlphabetRow, ...]
    genomes: Tuple[GenomeRow, ...]
    splice: Tuple[SpliceRow, ...]
    gates: Dict[str, bool]

def _fasta_candidates(stem: str) -> List[Path]:
    names = [
        f"{stem}.fna.gz",
        f"{stem}.fa.gz",
        f"{stem}.fna",
        f"{stem}.fa",
        f"{stem}_cds.fna.gz",
    ]
    return [DATA_DIR / n for n in names]

def load_named_fasta(keys: Sequence[str]) -> Optional[List[Tuple[str, str]]]:
    for key in keys:
        for path in _fasta_candidates(key):
            if path.exists() and path.stat().st_size > 0:
                try:
                    recs = parse_fasta(path)
                except OSError:
                    continue
                if recs:
                    return recs
    return None

def alphabet_row(enc: NucleotideEncoding, orbit: str) -> AlphabetRow:
    tetramers = ["".join(p) for p in itertools.product(BASES, repeat=4)]
    hist = [0] * 5
    total_d = 0
    for seq4 in tetramers:
        b = pack_4mer_byte(seq4, enc).byte
        d = fold_disagreement_d(b, CHIRALITY_D)
        hist[d] += 1
        total_d += d
    dual_n = 0
    dual_eq = 0
    dual_h = 0
    for codon in CODONS:
        payload = pack_codon_bits(codon, enc)
        rc_payload = pack_codon_bits(reverse_complement_codon(codon), enc)
        for five in BASES:
            for three in BASES:
                bp = pack_byte(enc.encode_base(five), payload)
                bm = pack_byte(enc.encode_base(WC[three]), rc_payload)
                dual_n += 1
                fb = fold_byte(bp)
                if fb == bm:
                    dual_eq += 1
                dual_h += (fb ^ bm).bit_count()
    return AlphabetRow(
        orbit=orbit,
        n_4mer=len(tetramers),
        fold_hist=tuple(hist),
        mean_disagreement=total_d / len(tetramers),
        dual_n=dual_n,
        dual_fold_eq=dual_eq,
        dual_mean_hamming=dual_h / dual_n if dual_n else 0.0,
    )

def _gc_codon_probs(gc: float) -> Dict[str, float]:
    pg = pc = gc / 2.0
    pa = pt = (1.0 - gc) / 2.0
    p = {"A": pa, "C": pc, "G": pg, "T": pt}
    return {c: p[c[0]] * p[c[1]] * p[c[2]] for c in CODONS}

def _usage(codons: Sequence[str]) -> Dict[str, float]:
    n = len(codons) or 1
    cnt = Counter(codons)
    return {c: cnt[c] / n for c in CODONS}

def _l1(p: Dict[str, float], q: Dict[str, float]) -> float:
    return sum(abs(p[c] - q[c]) for c in CODONS)

def _consensus_shell(codons: Sequence[str], enc: NucleotideEncoding) -> float:
    by_aa: Dict[str, List[str]] = {}
    for c in codons:
        by_aa.setdefault(STANDARD_CODE[c], []).append(c)
    shells = []
    for _aa, group in by_aa.items():
        cons = Counter(group).most_common(1)[0][0]
        cb = pack_codon_bits(cons, enc)
        for c in group:
            shells.append(hamming6(pack_codon_bits(c, enc) ^ cb))
    return sum(shells) / len(shells) if shells else 0.0

def _family_occ(records: Sequence[Tuple[str, str]], enc: NucleotideEncoding) -> Tuple[int, int, int, int]:
    occ = [0, 0, 0, 0]
    for _h, seq in records:
        s = seq.upper().replace("U", "T")
        for i in range(0, len(s) - 3, 3):
            if i == 0:
                continue
            five = s[i - 1]
            codon = s[i : i + 3]
            if five in BASES and len(codon) == 3 and all(b in BASES for b in codon):
                occ[enc.encode_base(five)] += 1
    return (occ[0], occ[1], occ[2], occ[3])

def _mean_4mer_fd(records: Sequence[Tuple[str, str]], enc: NucleotideEncoding) -> float:
    total = 0
    n = 0
    for _h, seq in records:
        for mer in iter_kmers(seq, 4):
            total += fold_disagreement_d(pack_4mer_byte(mer, enc).byte, CHIRALITY_D)
            n += 1
            if n >= 200000:
                break
        if n >= 200000:
            break
    return total / n if n else float("nan")

def _mean_dual(records: Sequence[Tuple[str, str]], enc: NucleotideEncoding) -> float:
    total = 0
    n = 0
    for _h, seq in records:
        s = seq.upper().replace("U", "T")
        for i in range(3, len(s) - 3, 3):
            five, codon, three = s[i - 1], s[i : i + 3], s[i + 3]
            if five not in BASES or three not in BASES or any(b not in BASES for b in codon):
                continue
            bp = pack_byte(enc.encode_base(five), pack_codon_bits(codon, enc))
            bm = pack_byte(enc.encode_base(WC[three]), pack_codon_bits(reverse_complement_codon(codon), enc))
            total += (fold_byte(bp) ^ bm).bit_count()
            n += 1
            if n >= 50000:
                break
        if n >= 50000:
            break
    return total / n if n else float("nan")

def _dicodon_shell(codons: Sequence[str], enc: NucleotideEncoding) -> float:
    if len(codons) < 2:
        return float("nan")
    acc = 0
    n = 0
    for a, b in zip(codons, codons[1:]):
        acc += hamming6(pack_codon_bits(a, enc) ^ pack_codon_bits(b, enc))
        n += 1
    return acc / n if n else float("nan")

def _trim_records(recs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    cds = []
    for h, s in recs:
        seq = "".join(b for b in s.upper().replace("U", "T") if b in BASES or b == "N")
        if len(seq) >= 9:
            cds.append((h, seq))
        if len(cds) >= MAX_CDS:
            break
    return cds

def _shuffle_bases(seq: str, rng: random.Random, preserve_gc: bool) -> str:
    s = [b for b in seq if b in BASES]
    if preserve_gc:
        gc = [b for b in s if b in STRONG]
        at = [b for b in s if b not in STRONG]
        rng.shuffle(gc)
        rng.shuffle(at)
        out = []
        ig = ia = 0
        for b in s:
            if b in STRONG:
                out.append(gc[ig])
                ig += 1
            else:
                out.append(at[ia])
                ia += 1
        return "".join(out)
    rng.shuffle(s)
    return "".join(s)

def _motif_span(kind: str, n: int) -> Optional[Tuple[int, int]]:
    """Canonical GT/AG span inside extracted splice windows."""
    if kind == "donor" and n >= 4:
        return 2, 4
    if kind == "acceptor" and n >= 6:
        return 4, 6
    return None

def _shuffle_splice_window(w: str, kind: str, rng: random.Random) -> str:
    """Shuffle flanks; preserve GT/AG motif span and flank GC/AT classes."""
    s = list(w.upper().replace("U", "T"))
    if any(ch not in BASES for ch in s):
        s = [ch for ch in s if ch in BASES]
    span = _motif_span(kind, len(s))
    if span is None or not s:
        return _shuffle_bases("".join(s), rng, preserve_gc=True)
    a, b = span
    motif = s[a:b]
    flanks = s[:a] + s[b:]
    sh = list(_shuffle_bases("".join(flanks), rng, preserve_gc=True))
    return "".join(sh[:a] + motif + sh[a:])

def genome_row(
    name: str,
    recs: Optional[List[Tuple[str, str]]],
    enc: NucleotideEncoding,
    orbit: str,
) -> GenomeRow:
    if recs is None:
        return GenomeRow(
            name=name,
            status="SKIP",
            n_records=0,
            n_codons=0,
            gc=float("nan"),
            orbit=orbit,
            mean_4mer_fd=float("nan"),
            mean_dual_h=float("nan"),
            mean_consensus_shell=float("nan"),
            family_occ=(0, 0, 0, 0),
            dicodon_mean_shell=float("nan"),
            usage_l1_vs_gcnull=float("nan"),
            n1_hits=-1,
            n1_n=0,
            n1_p=float("nan"),
            fd_null_hits=-1,
            fd_null_n=0,
            fd_null_p=float("nan"),
        )
    recs = _trim_records(recs)
    joined = "".join(s for _h, s in recs)
    codons = []
    for _h, s in recs:
        codons.extend(iter_codons(s))
    gc = gc_fraction(joined)
    usage = _usage(codons)
    expected = _gc_codon_probs(gc if gc == gc else 0.5)
    l1 = _l1(usage, expected)
    fd = _mean_4mer_fd(recs, enc)
    dual = _mean_dual(recs, enc)
    cons = _consensus_shell(codons, enc)
    fam = _family_occ(recs, enc)
    dic = _dicodon_shell(codons, enc)
    rng = random.Random(NULL_SEED)
    n1_hits = 0
    fd_null_hits = 0
    sample = recs[: min(len(recs), 80)]
    for _ in range(NULL_N):
        null_codons = []
        null_recs = []
        for h, s in sample:
            sh = _shuffle_bases(s, rng, preserve_gc=True)
            null_recs.append((h, sh))
            null_codons.extend(iter_codons(sh))
        n_usage = _usage(null_codons)
        n_l1 = _l1(n_usage, expected)
        if n_l1 >= l1 - 1e-15:
            n1_hits += 1
        n_fd = _mean_4mer_fd(null_recs, enc)
        if n_fd >= fd - 1e-15:
            fd_null_hits += 1
    return GenomeRow(
        name=name,
        status="ok",
        n_records=len(recs),
        n_codons=len(codons),
        gc=gc,
        orbit=orbit,
        mean_4mer_fd=fd,
        mean_dual_h=dual,
        mean_consensus_shell=cons,
        family_occ=fam,
        dicodon_mean_shell=dic,
        usage_l1_vs_gcnull=l1,
        n1_hits=n1_hits,
        n1_n=NULL_N,
        n1_p=monte_carlo_p(n1_hits, NULL_N),
        fd_null_hits=fd_null_hits,
        fd_null_n=NULL_N,
        fd_null_p=monte_carlo_p(fd_null_hits, NULL_N),
    )

def _parse_gtf_chr22(path: Path) -> List[Tuple[str, int, int, str, str]]:
    exons = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9:
                continue
            chrom, _src, kind, start, end, _sc, strand = parts[:7]
            if kind != "exon":
                continue
            if chrom not in {"chr22", "22"}:
                continue
            exons.append((chrom, int(start) - 1, int(end), strand, parts[8][:80]))
            if len(exons) >= 30000:
                break
    return exons

def _load_chr22_seq() -> Optional[str]:
    return load_chr22_sequence()


@lru_cache(maxsize=None)
def splice_census(enc: NucleotideEncoding, orbit: str) -> Tuple[SpliceRow, ...]:
    path = gtf_path()
    seq = _load_chr22_seq()
    if seq is None or path is None:
        kinds = ("donor", "acceptor", "exon_flank", "intron_interior")
        return tuple(
            SpliceRow(kind=k, status="SKIP", n=0, orbit=orbit, mean_4mer_fd=float("nan"), gt_ag_frac=float("nan"), null_p=float("nan"))
            for k in kinds
        )
    exons = _parse_gtf_chr22(path)
    by_tx: Dict[str, List[Tuple[int, int, str]]] = {}
    for _c, a, b, strand, attr in exons:
        tx = "tx"
        for tok in attr.split(";"):
            if "transcript_id" in tok:
                tx = tok.split('"')[1] if '"' in tok else tx
        by_tx.setdefault(tx, []).append((a, b, strand))
    donors = []
    acceptors = []
    flanks = []
    interiors = []
    nseq = len(seq)
    for _tx, spans in by_tx.items():
        spans = sorted(spans)
        if not spans:
            continue
        strand = spans[0][2]
        for i, (a, b, _s) in enumerate(spans):
            if 0 <= a < nseq and 4 <= b <= nseq:
                flanks.append(seq[max(0, a) : min(nseq, a + 4)])
                flanks.append(seq[max(0, b - 4) : min(nseq, b)])
            if i + 1 < len(spans):
                na, nb, _ = spans[i + 1]
                if strand != "-":
                    if b + 2 <= nseq:
                        donors.append(seq[b - 2 : b + 6] if b >= 2 else "")
                    if na >= 6:
                        acceptors.append(seq[na - 6 : na + 2])
                    if na - b >= 12:
                        mid = (b + na) // 2
                        interiors.append(seq[mid : mid + 4])
                else:
                    if a >= 6:
                        donors.append(reverse_complement_seq(seq[max(0, a - 6) : a + 2]))
                    if na + 2 <= nseq:
                        acceptors.append(reverse_complement_seq(seq[nb - 2 : nb + 6] if nb >= 2 else ""))
        if len(donors) >= MAX_WINDOWS:
            break
    rng = random.Random(NULL_SEED + 3)

    def pack_mean(windows: List[str]) -> Tuple[float, float]:
        mers = []
        motif = 0
        for w in windows:
            w = w.upper().replace("U", "T")
            if len(w) >= 4 and all(ch in BASES for ch in w[:4]):
                mers.append(w[:4])
            if "GT" in w[:4] or "AG" in w[:4]:
                motif += 1
        if not mers:
            return float("nan"), float("nan")
        fd = sum(fold_disagreement_d(pack_4mer_byte(m, enc).byte, CHIRALITY_D) for m in mers) / len(mers)
        return fd, motif / len(windows) if windows else float("nan")

    rows = []
    for kind, wins in (
        ("donor", donors),
        ("acceptor", acceptors),
        ("exon_flank", flanks),
        ("intron_interior", interiors),
    ):
        fd, frac = pack_mean(wins[:MAX_WINDOWS])
        hits = 0
        n_null = NULL_N
        if wins:
            use = [w.upper().replace("U", "T") for w in wins[:MAX_WINDOWS] if w]
            use = [w for w in use if len(w) >= 4 and all(ch in BASES for ch in w)]
            for _ in range(n_null):
                fake = [_shuffle_splice_window(w, kind, rng) for w in use[:200]]
                nfd, _f = pack_mean(fake)
                if nfd == nfd and fd == fd and nfd >= fd - 1e-15:
                    hits += 1
            p = monte_carlo_p(hits, n_null)
            status = "ok"
        else:
            p = float("nan")
            status = "SKIP"
        rows.append(
            SpliceRow(
                kind=kind,
                status=status,
                n=len(wins),
                orbit=orbit,
                mean_4mer_fd=fd,
                gt_ag_frac=frac,
                null_p=p,
            )
        )
    return tuple(rows)

def sequence_census() -> SequenceCensus:
    reps = orbit_reps()
    alphabet = []
    genomes = []
    splice = []
    sources = {
        "ecoli_k12": ("ecoli_k12_cds", "ecoli_k12"),
        "yeast_s288c": ("yeast_s288c_cds", "yeast_s288c"),
        "sars_cov2": ("sars_cov2_cds", "sars_cov2"),
        "chr22_cds": ("chr22_cds", "chr22"),
    }
    loaded = {name: load_named_fasta(keys) for name, keys in sources.items()}
    cds22 = extract_chr22_cds(300)
    if cds22:
        loaded["chr22_cds"] = [(h, s) for h, s, _starts in cds22]
    for orbit in (ORBIT_ELEMENTARY, ORBIT_PAIR_INV):
        _i, enc = reps[orbit]
        alphabet.append(alphabet_row(enc, orbit))
        for name, recs in loaded.items():
            genomes.append(genome_row(name, recs, enc, orbit))
        splice.extend(splice_census(enc, orbit))
    gates = {
        "39_alphabet_256": all(r.n_4mer == 256 for r in alphabet),
        "40_fold_hist_sum_256": all(sum(r.fold_hist) == 256 for r in alphabet),
        "41_both_orbits_present": {r.orbit for r in alphabet} == {ORBIT_ELEMENTARY, ORBIT_PAIR_INV},
    }
    return SequenceCensus(
        alphabet=tuple(alphabet),
        genomes=tuple(genomes),
        splice=tuple(splice),
        gates=gates,
    )

def print_sequence_census(c: SequenceCensus) -> None:
    g = c.gates
    n_ok = sum(1 for row in c.genomes if row.status == "ok")
    n_skip = sum(1 for row in c.genomes if row.status == "SKIP")

    report_section("3. SEQUENCE WINDOWS ON REAL DNA")
    report_objects(
        (
            "256 4-mer alphabet; genome fd/dual/shell meters; GC-matched nulls; SKIP if missing FASTA",
        )
    )
    print("  alphabet")
    for r in c.alphabet:
        print(
            f"    {r.orbit}: n_4mer={r.n_4mer} fold_hist={r.fold_hist} "
            f"mean_fd={r.mean_disagreement:.4f} dual_fold_eq={r.dual_fold_eq}/{r.dual_n} "
            f"dual_mean_h={r.dual_mean_hamming:.4f}"
        )
    print()
    print("  genomes")
    print(
        f"    {'name':<14} {'orbit':<24} {'status':<6} {'cds':>5} {'gc':>6} "
        f"{'fd':>7} {'dual_h':>7} {'shell':>7} {'n1_p':>7} {'fd_null_p':>9}"
    )
    for r in c.genomes:
        if r.status != "ok":
            print(f"    {r.name:<14} {r.orbit:<24} {r.status:<6}")
            continue
        print(
            f"    {r.name:<14} {r.orbit:<24} {r.status:<6} {r.n_records:>5} {r.gc:>6.3f} "
            f"{r.mean_4mer_fd:>7.4f} {r.mean_dual_h:>7.4f} {r.mean_consensus_shell:>7.3f} "
            f"{r.n1_p:>7.4f} {r.fd_null_p:>8.4f}"
        )
    print(f"    totals: ok={n_ok} SKIP={n_skip}")
    print()
    if c.splice:
        print("  splice flanks")
        print(
            f"    {'kind':<18} {'orbit':<24} {'status':<6} {'n':>5} "
            f"{'fd':>7} {'gt_ag':>6} {'null_p':>7}"
        )
        for r in c.splice:
            print(
                f"    {r.kind:<18} {r.orbit:<24} {r.status:<6} {r.n:>5} "
                f"{r.mean_4mer_fd:>7.4f} {r.gt_ag_frac:>6.3f} {r.null_p:>7.4f}"
            )
        print()
    report_checks((
        ('alphabet covers all 256 4-mers', g['39_alphabet_256'], f'n_4mer={[r.n_4mer for r in c.alphabet]}', '[256, 256]'),
        ('fold_hist sums to 256', g['40_fold_hist_sum_256'], f'sums={[sum(r.fold_hist) for r in c.alphabet]}', '[256, 256]'),
        ('both metric orbits present', g['41_both_orbits_present'], f'orbits={sorted({r.orbit for r in c.alphabet})}', f'{{{ORBIT_ELEMENTARY}, {ORBIT_PAIR_INV}}}'),
    ))


def _family_hist(bytes_: Sequence[int]) -> Tuple[int, int, int, int]:
    h = [0, 0, 0, 0]
    for b in bytes_:
        fam, _ = unpack_byte(b)
        h[fam & 3] += 1
    return tuple(h)  # type: ignore[return-value]


def _mu_from_hist(h: Sequence[int]) -> Tuple[float, float, float, float]:
    n = sum(h) or 1
    return tuple(c / n for c in h)  # type: ignore[return-value]


def _l1_uniform(mu: Sequence[float]) -> float:
    return sum(abs(x - 0.25) for x in mu)


def _shuffle_seq_bases(seq: str, rng: random.Random) -> str:
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


@dataclass
class FamilyMuRow:
    stratum: str
    n: int
    mu: Tuple[float, float, float, float]
    l1_uniform: float
    null_l1: float


@dataclass
class FamilyMuCensus:
    rows: Tuple[FamilyMuRow, ...]
    gates: Dict[str, bool]


def family_mu_census() -> FamilyMuCensus:
    enc = orbit_reps()[ORBIT_PAIR_INV][1]
    rng = random.Random(NULL_SEED + 7)
    strata: List[Tuple[str, List[str]]] = []

    cds_seqs: List[str] = []
    for keys in (("ecoli_k12",), ("yeast_s288c",), ("chr22_cds",)):
        recs = load_named_fasta(keys)
        if recs:
            for _h, s in recs[:200]:
                if len(s) >= 4:
                    cds_seqs.append(s)
    if cds_seqs:
        strata.append(("cds", cds_seqs[:400]))

    path = gtf_path()
    seq = _load_chr22_seq()
    if seq and path and path.exists():
        exons = _parse_gtf_chr22(path)
        by_tx: Dict[str, List[Tuple[int, int, str]]] = {}
        for _c, a, b, strand, attr in exons:
            tx = "tx"
            for tok in attr.split(";"):
                tok = tok.strip()
                if tok.startswith("transcript_id"):
                    tx = tok.split()[-1].strip("\"'")
                    break
            by_tx.setdefault(tx, []).append((a, b, strand))
        donors: List[str] = []
        acceptors: List[str] = []
        interiors: List[str] = []
        for _tx, spans in by_tx.items():
            spans = sorted(spans, key=lambda t: t[0])
            if not spans:
                continue
            strand = spans[0][2]
            for i, (a, b, _s) in enumerate(spans):
                if strand == "+":
                    if b >= 2 and b + 6 <= len(seq):
                        donors.append(seq[b - 2 : b + 6])
                    if i + 1 < len(spans):
                        na = spans[i + 1][0]
                        if na >= 6:
                            acceptors.append(seq[na - 6 : na + 2])
                        if na - b >= 20:
                            interiors.append(seq[b:na][:120])
                else:
                    if a >= 6:
                        donors.append(reverse_complement_seq(seq[max(0, a - 6) : a + 2]))
                    if i + 1 < len(spans):
                        nb = spans[i + 1][1]
                        if nb + 6 <= len(seq):
                            acceptors.append(reverse_complement_seq(seq[nb - 2 : nb + 6]))
            if len(donors) >= 1500:
                break
        if donors:
            strata.append(("donor_flank", donors[:1500]))
        if acceptors:
            strata.append(("acceptor_flank", acceptors[:1500]))
        if interiors:
            strata.append(("intron_interior", interiors[:300]))
        wins = [
            seq[i : i + 120]
            for i in range(0, min(len(seq), 60_000), 120)
            if len(seq[i : i + 120]) == 120
        ]
        if wins:
            strata.append(("genome_windows", wins[:200]))

    rows: List[FamilyMuRow] = []
    for name, seqs in strata:
        bytes_all: List[int] = []
        for s in seqs:
            bytes_all.extend(genomic_byte_stream(s, enc))
        if len(bytes_all) < 16:
            continue
        h = _family_hist(bytes_all)
        mu = _mu_from_hist(h)
        l1 = _l1_uniform(mu)
        null_l1s = []
        sample = seqs[:80]
        for _ in range(12):
            nb: List[int] = []
            for s in sample:
                nb.extend(genomic_byte_stream(_shuffle_seq_bases(s, rng), enc))
            if nb:
                null_l1s.append(_l1_uniform(_mu_from_hist(_family_hist(nb))))
        null_mean = sum(null_l1s) / len(null_l1s) if null_l1s else float("nan")
        rows.append(FamilyMuRow(name, sum(h), mu, l1, null_mean))

    cds_row = next((r for r in rows if r.stratum == "cds"), None)
    donor_row = next((r for r in rows if r.stratum == "donor_flank"), None)
    acceptor_row = next((r for r in rows if r.stratum == "acceptor_flank"), None)
    gates = {
        "142_family_mu_defined": len(rows) >= 2,
        "143_cds_mu_near_uniform": (
            cds_row is not None and cds_row.l1_uniform < 0.15
        ),
        "144_donor_sheet_asymmetric": (
            donor_row is not None
            and donor_row.l1_uniform > donor_row.null_l1 + 0.05
        ),
        "145_acceptor_sheet_asymmetric": (
            acceptor_row is not None
            and acceptor_row.l1_uniform > acceptor_row.null_l1 + 0.05
        ),
    }
    return FamilyMuCensus(rows=tuple(rows), gates=gates)


# Junction taxonomy used across boundary censuses:
# hinge_egress / hinge_ingress = splice regime hinge
# open_transcript = TSS / CDS-upstream genealogy open
# open_peptide / close_peptide = Met / stop
_JUNCTION_KIND = {
    "donor_flank": "hinge_egress",
    "acceptor_flank": "hinge_ingress",
    "tss_box35": "open_transcript",
    "tss_m40": "open_transcript",
    "cds_upstream_pm3": "open_transcript",
    "cds_upstream_m10": "open_transcript",
    "cds_upstream_m40": "open_transcript",
    "cds_interior": "mid_transcript",
}

# RegulonDB PromoterSet confidence used for primary TSS strata.
_TSS_CONF_KEEP = frozenset({"C", "S"})


def print_family_mu_census(c: FamilyMuCensus) -> None:
    g = c.gates
    report_section("20. FAMILY-SHEET MU ON K4")
    report_objects((
        "family bits on packed 4-mers; mu vs uniform 1/4; composition shuffle null; "
        "donor=hinge_egress acceptor=hinge_ingress",
    ))
    print_table(
        ("stratum", "junction", "n", "mu0", "mu1", "mu2", "mu3", "L1_unif", "null_L1"),
        (16, 14, 8, 6, 6, 6, 6, 8, 8),
        [
            (
                r.stratum,
                _JUNCTION_KIND.get(r.stratum, "-"),
                r.n,
                f"{r.mu[0]:.3f}",
                f"{r.mu[1]:.3f}",
                f"{r.mu[2]:.3f}",
                f"{r.mu[3]:.3f}",
                f"{r.l1_uniform:.4f}",
                f"{r.null_l1:.4f}",
            )
            for r in c.rows
        ],
        aligns=("<", "<", ">", ">", ">", ">", ">", ">", ">"),
    )
    print()
    report_checks((
        ("family mu strata computed", g["142_family_mu_defined"], f"n_strata={len(c.rows)}", ">=2"),
        ("CDS family-sheet mu near uniform (symmetric K4 use)", g["143_cds_mu_near_uniform"], "see table L1_unif", "<0.15"),
        ("donor flank selects a family sheet (L1 above null)", g["144_donor_sheet_asymmetric"], "see donor row", "L1 > null+0.05"),
        ("acceptor flank selects a family sheet (L1 above null)", g["145_acceptor_sheet_asymmetric"], "see acceptor row", "L1 > null+0.05"),
    ))


def _load_ecoli_replicon() -> Optional[str]:
    path = DATA_DIR / "ecoli_k12_full.fna.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        chunks: List[str] = []
        for line in fh:
            if line.startswith(">"):
                continue
            chunks.append(line.strip())
    seq = "".join(chunks).upper()
    return seq if len(seq) > 100_000 else None


@dataclass
class IngressRow:
    stratum: str
    junction: str
    n_bytes: int
    l1_uniform: float
    null_l1_mean: float
    n_null_ge: int
    n_null: int
    mean_fd: float
    kin_rc_gap: float
    d4_frac0: float
    d4_n: int


@dataclass
class GenealogyIngressCensus:
    rows: Tuple[IngressRow, ...]
    n_starts: int
    n_tss: int
    gates: Dict[str, bool]


@dataclass
class RegulonPromoter:
    tss0: int
    strand: str
    conf: str
    box10: Optional[Tuple[int, int]]
    box35: Optional[Tuple[int, int]]


def _parse_rdb_span(text: str) -> Optional[Tuple[int, int]]:
    text = text.strip()
    if "-" not in text:
        return None
    a_s, b_s = text.split("-", 1)
    if not (a_s.isdigit() and b_s.isdigit()):
        return None
    a0, b_incl = int(a_s) - 1, int(b_s)
    if a0 < 0 or b_incl <= a0:
        return None
    return a0, b_incl


def _load_regulondb_promoters(
    *,
    conf_keep: frozenset = _TSS_CONF_KEEP,
) -> List[RegulonPromoter]:
    """Load RegulonDB PromoterSet rows (NC_000913 coords, 1-based file → 0-based)."""
    path = DATA_DIR / "regulondb_promoter_set.txt"
    if not path.exists():
        return []
    out: List[RegulonPromoter] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#") or line.startswith("1)id"):
            continue
        parts = line.split("\t")
        if len(parts) < 15:
            continue
        strand = parts[2].strip().lower()
        pos = parts[3].strip()
        conf = parts[14].strip().upper()
        if conf not in conf_keep:
            continue
        if not pos.isdigit() or strand not in ("forward", "reverse"):
            continue
        out.append(RegulonPromoter(
            tss0=int(pos) - 1,
            strand=strand,
            conf=conf,
            box10=_parse_rdb_span(parts[8]),
            box35=_parse_rdb_span(parts[10]),
        ))
    return out


def _transcript_window(
    seq: str,
    tss0: int,
    strand: str,
    left: int,
    right: int,
) -> str:
    """Transcript-oriented [left, right) relative to TSS (negative = upstream)."""
    L = len(seq)
    if right <= left:
        return ""
    if strand == "forward":
        a, b = tss0 + left, tss0 + right
        if a < 0 or b > L:
            return ""
        return seq[a:b]
    g_lo = tss0 - (right - 1)
    g_hi = tss0 - left + 1
    if g_lo < 0 or g_hi > L or g_lo >= g_hi:
        return ""
    return reverse_complement_seq(seq[g_lo:g_hi])


def _oriented_genomic_span(seq: str, strand: str, span: Tuple[int, int]) -> str:
    a, b = span
    if a < 0 or b > len(seq) or a >= b:
        return ""
    frag = seq[a:b]
    if strand == "reverse":
        return reverse_complement_seq(frag)
    return frag


def _ingress_window_stats(
    windows: Sequence[str],
    enc: NucleotideEncoding,
    rng: random.Random,
    *,
    n_null: int = 24,
) -> Optional[IngressRow]:
    if not windows:
        return None
    bytes_fwd: List[int] = []
    for w in windows:
        bytes_fwd.extend(genomic_byte_stream(w, enc))
    if len(bytes_fwd) < 32:
        return None
    h = _family_hist(bytes_fwd)
    mu = _mu_from_hist(h)
    l1 = _l1_uniform(mu)
    fd = sum(fold_disagreement_d(b, CHIRALITY_D) for b in bytes_fwd) / len(bytes_fwd)

    # Kinematic (sequence) RC on the packed 4-mer stream — not payload-RC.
    bytes_rc: List[int] = []
    for w in windows:
        bytes_rc.extend(genomic_byte_stream(reverse_complement_seq(w), enc))
    fd_rc = (
        sum(fold_disagreement_d(b, CHIRALITY_D) for b in bytes_rc) / len(bytes_rc)
        if bytes_rc
        else fd
    )
    kin_rc_gap = abs(fd - fd_rc)

    ok = tot = 0
    for w in windows:
        stream = genomic_byte_stream(w, enc)
        for i in range(0, len(stream) - 3):
            sig = omega_word_signature(stream[i : i + 4])
            tot += 1
            if sig.parity == 0:
                ok += 1
    d4_frac0 = (ok / tot) if tot else float("nan")

    null_l1s: List[float] = []
    sample = list(windows[:120])
    for _ in range(n_null):
        nb: List[int] = []
        for s in sample:
            nb.extend(genomic_byte_stream(_shuffle_seq_bases(s, rng), enc))
        if nb:
            null_l1s.append(_l1_uniform(_mu_from_hist(_family_hist(nb))))
    null_mean = sum(null_l1s) / len(null_l1s) if null_l1s else float("nan")
    n_ge = sum(1 for x in null_l1s if x >= l1)
    return IngressRow(
        stratum="",
        junction="",
        n_bytes=len(bytes_fwd),
        l1_uniform=l1,
        null_l1_mean=null_mean,
        n_null_ge=n_ge,
        n_null=len(null_l1s),
        mean_fd=fd,
        kin_rc_gap=kin_rc_gap,
        d4_frac0=d4_frac0,
        d4_n=tot,
    )


def genealogy_ingress_census() -> GenealogyIngressCensus:
    """open_transcript — RegulonDB promoter boxes + CDS-upstream control."""
    enc = orbit_reps()[ORBIT_PAIR_INV][1]
    seq = _load_ecoli_replicon()
    recs = load_named_fasta(("ecoli_k12",))
    empty_gates = {
        "242_ingress_strata_defined": False,
        "243_cds_start_pm3_sheet_above_null": False,
        "244_cds_upstream_m10_sheet_above_null": False,
        "246_ingress_fold_fd_near_cds": False,
        "247_ingress_kin_rc_gap_near_zero": False,
        "248_ingress_depth4_parity_zero": False,
        "257_tss_catalog_loaded": False,
        "258_tss_box35_sheet_above_null": False,
        "260_tss_fold_fd_near_cds": False,
        "261_tss_kin_rc_gap_near_zero": False,
    }
    empty = GenealogyIngressCensus(rows=(), n_starts=0, n_tss=0, gates=empty_gates)
    if seq is None or not recs:
        return empty

    starts: List[int] = []
    for _h, cds in recs[:1200]:
        cds_u = cds.upper().replace("U", "T")
        if len(cds_u) < 90:
            continue
        i = seq.find(cds_u[:60])
        if i >= 0:
            starts.append(i)

    promoters = [
        p for p in _load_regulondb_promoters()
        if 80 <= p.tss0 < len(seq) - 80
    ]

    strata_wins: List[Tuple[str, List[str]]] = []
    if promoters:
        box35 = [
            w for p in promoters
            if p.box35 is not None and (w := _oriented_genomic_span(seq, p.strand, p.box35))
        ]
        strata_wins.extend([
            ("tss_box35", box35),
            (
                "tss_m40",
                [w for p in promoters if (w := _transcript_window(seq, p.tss0, p.strand, -40, 0))],
            ),
        ])
    if len(starts) >= 100:
        strata_wins.extend([
            ("cds_upstream_pm3", [seq[max(0, i - 3) : i + 3] for i in starts]),
            ("cds_upstream_m10", [seq[max(0, i - 13) : max(0, i - 4)] for i in starts]),
            ("cds_upstream_m40", [seq[max(0, i - 40) : i] for i in starts]),
            ("cds_interior", [seq[i + 60 : i + 160] for i in starts]),
        ])
    if not strata_wins:
        return empty

    rows: List[IngressRow] = []
    for name, wins in strata_wins:
        # Independent null stream per stratum so adding rows does not reshuffle prior p_ge.
        st = _ingress_window_stats(
            wins, enc, random.Random(NULL_SEED + 19 + (sum(map(ord, name)) % 997))
        )
        if st is None:
            continue
        rows.append(IngressRow(
            stratum=name,
            junction=_JUNCTION_KIND.get(name, "-"),
            n_bytes=st.n_bytes,
            l1_uniform=st.l1_uniform,
            null_l1_mean=st.null_l1_mean,
            n_null_ge=st.n_null_ge,
            n_null=st.n_null,
            mean_fd=st.mean_fd,
            kin_rc_gap=st.kin_rc_gap,
            d4_frac0=st.d4_frac0,
            d4_n=st.d4_n,
        ))

    by = {r.stratum: r for r in rows}
    pm3 = by.get("cds_upstream_pm3")
    m10 = by.get("cds_upstream_m10")
    cds = by.get("cds_interior")
    core = by.get("cds_upstream_m40")
    tss_m40 = by.get("tss_m40")
    tss_box35 = by.get("tss_box35")
    fd_ref = core if core is not None else pm3
    tss_rows = [r for r in rows if r.stratum.startswith("tss_")]

    gates = {
        "242_ingress_strata_defined": len(rows) >= 4,
        "243_cds_start_pm3_sheet_above_null": (
            pm3 is not None and pm3.n_null_ge == 0 and pm3.l1_uniform > pm3.null_l1_mean
        ),
        "244_cds_upstream_m10_sheet_above_null": (
            m10 is not None and m10.n_null_ge == 0 and m10.l1_uniform > m10.null_l1_mean
        ),
        "246_ingress_fold_fd_near_cds": (
            fd_ref is not None
            and cds is not None
            and abs(fd_ref.mean_fd - cds.mean_fd) < 0.05
        ),
        "247_ingress_kin_rc_gap_near_zero": all(r.kin_rc_gap < 0.01 for r in rows),
        "248_ingress_depth4_parity_zero": all(
            r.d4_n > 0 and r.d4_frac0 == 1.0
            for r in rows
            if r.stratum in ("cds_upstream_m40", "cds_interior", "tss_m40")
        ),
        "257_tss_catalog_loaded": len(promoters) >= 200 and tss_box35 is not None,
        "258_tss_box35_sheet_above_null": (
            tss_box35 is not None
            and tss_box35.n_null_ge == 0
            and tss_box35.l1_uniform > tss_box35.null_l1_mean
        ),
        "260_tss_fold_fd_near_cds": (
            tss_m40 is not None
            and cds is not None
            and abs(tss_m40.mean_fd - cds.mean_fd) < 0.05
        ),
        "261_tss_kin_rc_gap_near_zero": (
            bool(tss_rows) and all(r.kin_rc_gap < 0.01 for r in tss_rows)
        ),
    }
    return GenealogyIngressCensus(
        rows=tuple(rows),
        n_starts=len(starts),
        n_tss=len(promoters),
        gates=gates,
    )


def print_genealogy_ingress_census(c: GenealogyIngressCensus) -> None:
    g = c.gates
    report_section("20b. GENEALOGY INGRESS (OPEN_TRANSCRIPT)")
    report_objects((
        "open_transcript: RegulonDB C+S annotated -35 box; "
        "TSS-m40 and CDS-upstream contrast; kinematic-RC; fold vs hinge",
    ))
    print(f"    n_tss_CS={c.n_tss}  mapped_cds_starts={c.n_starts}")
    print_table(
        ("stratum", "junction", "n", "L1", "null_L1", "p_ge", "fd", "kin_rc", "d4_0"),
        (16, 14, 8, 7, 8, 6, 7, 7, 6),
        [
            (
                r.stratum,
                r.junction,
                r.n_bytes,
                f"{r.l1_uniform:.4f}",
                f"{r.null_l1_mean:.4f}",
                f"{r.n_null_ge}/{r.n_null}",
                f"{r.mean_fd:.4f}",
                f"{r.kin_rc_gap:.4f}",
                f"{r.d4_frac0:.3f}" if r.d4_n else "n/a",
            )
            for r in c.rows
        ],
        aligns=("<", "<", ">", ">", ">", ">", ">", ">", ">"),
    )
    print()
    report_checks((
        ("open_transcript strata defined", g["242_ingress_strata_defined"], f"n={len(c.rows)}", ">=4"),
        ("cds_upstream_pm3 family-sheet L1 above all GC nulls", g["243_cds_start_pm3_sheet_above_null"], "see pm3 row", "p_ge=0"),
        ("cds_upstream_m10 family-sheet L1 above all GC nulls", g["244_cds_upstream_m10_sheet_above_null"], "see m10 row", "p_ge=0"),
        ("CDS-proxy fold-disagreement near CDS", g["246_ingress_fold_fd_near_cds"], "|fd_m40-fd_cds|<0.05", "True"),
        ("all strata kinematic-RC gap near zero", g["247_ingress_kin_rc_gap_near_zero"], "kin_rc<0.01 all", "True"),
        ("depth-4 parity 0 on m40 / TSS-m40 / CDS", g["248_ingress_depth4_parity_zero"], "frac0=1", "True"),
        ("RegulonDB promoter catalog loaded (C+S)", g["257_tss_catalog_loaded"], f"n_tss={c.n_tss}", ">=200"),
        ("annotated -35 box family-sheet above GC nulls", g["258_tss_box35_sheet_above_null"], "see tss_box35", "p_ge=0"),
        ("TSS-m40 fold-disagreement near CDS interior", g["260_tss_fold_fd_near_cds"], "|fd_tss-fd_cds|<0.05", "True"),
        ("TSS strata kinematic-RC gap near zero", g["261_tss_kin_rc_gap_near_zero"], "kin_rc<0.01 tss", "True"),
    ))


@dataclass
class PoleRow:
    name: str
    n_pairs: int
    mean_ab: float
    mean_horizon: float
    eq_frac: float
    comp_frac: float
    null_eq: float
    null_comp: float
    p_eq_low: int
    n_null: int


@dataclass
class PoleCensus:
    rows: Tuple[PoleRow, ...]
    gates: Dict[str, bool]


def _pole_stats(codons: Sequence[str], enc: NucleotideEncoding) -> Tuple[float, float, float, float, int]:
    if len(codons) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    sab = sh = 0.0
    n_eq = n_comp = n = 0
    for i in range(len(codons) - 1):
        anc = codon_state(codons[i], enc)
        pres = codon_state(codons[i + 1], enc)
        st = carrier_from_codon_pair(anc, pres)
        a12, b12 = unpack_state(st)
        abd = ab_distance(a12, b12)
        hd = horizon_distance(a12, b12)
        sab += abd
        sh += hd
        n += 1
        if abd == 0:
            n_eq += 1
        if hd == 0:
            n_comp += 1
    return sab / n, sh / n, n_eq / n, n_comp / n, n


def _gc_shuffle_codons(codons: Sequence[str], rng: random.Random) -> List[str]:
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


def pole_census() -> PoleCensus:
    enc = orbit_reps()[ORBIT_PAIR_INV][1]
    rng = random.Random(NULL_SEED + 8)
    n_null = 12
    specs = (
        (("ecoli_k12",), "ecoli", None),
        (("yeast_s288c",), "yeast", None),
        (("chr22_cds",), "chr22", [s for _h, s, _st in extract_chr22_cds(200)]),
    )
    rows: List[PoleRow] = []
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
        mean_ab = mean_h = 0.0
        n_pairs = 0
        eq_n = comp_n = 0
        for g in genes:
            mab, mh, fe, fc, np_ = _pole_stats(g, enc)
            if np_ == 0:
                continue
            mean_ab += mab * np_
            mean_h += mh * np_
            eq_n += int(round(fe * np_))
            comp_n += int(round(fc * np_))
            n_pairs += np_
        if n_pairs == 0:
            continue
        mean_ab /= n_pairs
        mean_h /= n_pairs
        eq = eq_n / n_pairs
        comp = comp_n / n_pairs

        null_eqs = []
        null_comps = []
        gene_sample = genes[:120]
        for _ in range(n_null):
            ne = nc = nt = 0
            for g in gene_sample:
                sg = _gc_shuffle_codons(g, rng)
                _a, _h, fe, fc, np_ = _pole_stats(sg, enc)
                ne += int(round(fe * np_))
                nc += int(round(fc * np_))
                nt += np_
            if nt:
                null_eqs.append(ne / nt)
                null_comps.append(nc / nt)
        null_eq = sum(null_eqs) / len(null_eqs) if null_eqs else float("nan")
        null_comp = sum(null_comps) / len(null_comps) if null_comps else float("nan")
        p_eq_low = sum(1 for x in null_eqs if eq <= x)
        rows.append(
            PoleRow(name, n_pairs, mean_ab, mean_h, eq, comp, null_eq, null_comp, p_eq_low, n_null)
        )

    gates = {
        "249_pole_complementarity_invariant_note7_xi": all(
            abs((r.mean_ab + r.mean_horizon) - 12.0) < 1e-6 for r in rows
        ) and bool(rows),
        "250_eq_horizon_above_gc_null": all(
            r.eq_frac > r.null_eq for r in rows
        ) and bool(rows),
        "251_eq_frac_p_value_recorded": all(r.p_eq_low <= r.n_null for r in rows) and bool(rows),
    }
    return PoleCensus(rows=tuple(rows), gates=gates)


def print_pole_census(c: PoleCensus) -> None:
    g = c.gates
    report_section("21. CONSTITUTIONAL POLES ON CODON PAIRS")
    report_objects(("carrier_from_codon_pair; ab/horizon distances; pole occupancy vs GC shuffle",))
    print_table(
        ("name", "pairs", "mean_ab", "mean_hor", "eq_frac", "comp_frac", "null_eq", "p_eq"),
        (8, 8, 8, 8, 8, 9, 8, 6),
        [
            (
                r.name,
                r.n_pairs,
                f"{r.mean_ab:.4f}",
                f"{r.mean_horizon:.4f}",
                f"{r.eq_frac:.4f}",
                f"{r.comp_frac:.4f}",
                f"{r.null_eq:.4f}",
                f"{r.p_eq_low}/{r.n_null}",
            )
            for r in c.rows
        ],
        aligns=("<", ">", ">", ">", ">", ">", ">", ">"),
    )
    print()
    report_checks((
        ("xi: ab + horizon = 12 on live genome traffic", g["249_pole_complementarity_invariant_note7_xi"], "see means", "sum=12 each"),
        ("equality-horizon occupancy exceeds GC-shuffle null", g["250_eq_horizon_above_gc_null"], "eq_frac > null_eq", "each genome"),
        ("eq-fraction p-value recorded (k+1)/(m+1) style", g["251_eq_frac_p_value_recorded"], "p_eq in table", "defined"),
    ))
