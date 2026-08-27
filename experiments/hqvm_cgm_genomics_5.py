#!/usr/bin/env python3
"""
hqvm_cgm_genomics_5.py

Sections 6 and 12–18: word geometry, probes, Theta_kin law.
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
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from gyroscopic.hQVM.api import (
    OmegaSignature12,
    compose_omega_signatures,
    omega_word_signature,
    q_word6,
)
from gyroscopic.hQVM.family import fold_disagreement_d, intron_from_byte
from hqvm_cgm_genomics_1 import synonymous_homology
from hqvm_cgm_genomics_2 import (
    MAX_WINDOWS,
    load_named_fasta,
    orbit_reps,
    splice_census,
)
from hqvm_cgm_genomics_3 import (
    _product_words,
    omega_inv,
    payload_rc_star,
    predict_sr,
    theta_payload_rc,
    z_pair_mod4,
)
from hqvm_cgm_genomics_4 import (
    OUTER_MASK,
    substitution_cost_rows,
)
from hqvm_cgm_genomics_common import (
    AA_ORDER,
    ANTIPODE_6,
    BASES,
    CHIRALITY_D,
    CODONS,
    CODON_INDEX,
    DATA_DIR,
    NULL_SEED,
    NCBI_TABLE_IDS,
    ORBIT_ELEMENTARY,
    ORBIT_PAIR_INV,
    Q_RESIDUAL_W,
    STANDARD_CODE,
    STRONG,
    W_ANNIHILATOR,
    WC,
    NucleotideEncoding,
    bit_reverse6,
    block_reverse6,
    fiber_components,
    gf2_rank6,
    NucleotideEncoding,
    encodings_in_orbit,
    extract_chr22_cds,
    fibers,
    hamming6,
    in_W,
    iter_codons,
    load_chr22_sequence,
    one_base_neighbors,
    pack_4mer_byte,
    pack_byte,
    pack_codon_bits,
    print_table,
    report_check,
    report_checks,
    report_objects,
    report_section,
    reverse_complement_4mer_byte,
    reverse_complement_seq,
    translation_table,
    unpack_byte,
)

def W_from_constraints() -> Tuple[int, ...]:
    return tuple(x for x in range(64) if in_W(x))

def word_rc_holds(
    enc: NucleotideEncoding,
    n: int,
    stride: int,
    z_fn,
) -> Tuple[int, int]:
    ok = tot = 0
    z = z_fn(n)
    for w in _product_words(n, stride):
        s = omega_word_signature(w)
        sr = omega_word_signature(payload_rc_star(w, enc))
        tot += 1
        if sr == predict_sr(s, n, z):
            ok += 1
    return ok, tot

def observed_z(enc: NucleotideEncoding, n: int) -> Tuple[int, int, int]:
    w = [0] * n
    s = omega_word_signature(w)
    sr = omega_word_signature(payload_rc_star(w, enc))
    z = compose_omega_signatures(omega_inv(theta_payload_rc(omega_inv(s))), sr)
    return z.parity, z.tau_u6, z.tau_v6

def wc_codon(codon: str) -> str:
    return "".join(WC[b] for b in codon)

@dataclass
class WRow:
    set_eq_constraints: bool
    rank: int
    fold_invariant: bool
    antipode_in: bool
    block_rev_invariant: bool

@dataclass
class WordRcRow:
    orbit: str
    n_charts: int
    n_ok: int
    n_total: int
    theta_involution: bool
    z_period4: Tuple[Tuple[int, int, int], ...]

@dataclass
class BoxBetaRow:
    beta1_sum: int
    n_k4: int
    ile_k3: bool
    leu_extra: int
    arg_extra: int
    ser_disconnected: bool

@dataclass
class SheetRow:
    family_is_eps: bool
    serine_wc_complements: bool
    serine_sig_paired: bool
    n_aa_all_sigs_distinct: int
    n_aa: int

@dataclass
class PosAnisoRow:
    n_pos: Tuple[int, int, int]
    pos1_edges: Tuple[Tuple[str, str, str], ...]
    sense_pos1: int

@dataclass
class WordGeometryCensus:
    W: WRow
    word_rc: Tuple[WordRcRow, ...]
    boxes: BoxBetaRow
    sheet: SheetRow
    pos: PosAnisoRow
    ncbi_moves: Tuple[Tuple[int, int, str], ...]
    gates: Dict[str, bool]

def w_census() -> WRow:
    derived = W_from_constraints()
    given = Q_RESIDUAL_W
    return WRow(
        set_eq_constraints=derived == given,
        rank=gf2_rank6(derived),
        fold_invariant=tuple(sorted(bit_reverse6(x) for x in derived)) == derived,
        antipode_in=ANTIPODE_6 in set(derived),
        block_rev_invariant=tuple(sorted(block_reverse6(x) for x in derived)) == derived,
    )

def word_rc_census() -> Tuple[WordRcRow, ...]:
    rows: List[WordRcRow] = []
    pair_charts = encodings_in_orbit(ORBIT_PAIR_INV)
    elem_charts = encodings_in_orbit(ORBIT_ELEMENTARY)
    theta_ok = True
    n_ok = n_tot = 0
    for _i, enc in pair_charts:
        for n, stride in ((1, 1), (2, 4), (3, 16), (4, 32)):
            ok, tot = word_rc_holds(enc, n, stride, z_pair_mod4)
            n_ok += ok
            n_tot += tot
        for b in range(0, 256, 17):
            s = omega_word_signature([b])
            if theta_payload_rc(theta_payload_rc(s)) != s:
                theta_ok = False
                break
    z_pair = tuple(observed_z(pair_charts[0][1], n) for n in range(1, 5))
    rows.append(
        WordRcRow(
            orbit=ORBIT_PAIR_INV,
            n_charts=len(pair_charts),
            n_ok=n_ok,
            n_total=n_tot,
            theta_involution=theta_ok,
            z_period4=z_pair,
        )
    )
    enc_e = elem_charts[0][1]
    e_ok, e_tot = word_rc_holds(enc_e, 1, 1, z_pair_mod4)
    z_elem = tuple(observed_z(enc_e, n) for n in range(1, 5))
    rows.append(
        WordRcRow(
            orbit=ORBIT_ELEMENTARY,
            n_charts=len(elem_charts),
            n_ok=e_ok,
            n_total=e_tot,
            theta_involution=theta_ok,
            z_period4=z_elem,
        )
    )
    return tuple(rows)

def box_census() -> BoxBetaRow:
    fib = fibers(STANDARD_CODE)
    total = 0
    details: Dict[str, Tuple[int, int, int, int]] = {}
    for aa, group in fib.items():
        g = list(group)
        e = 0
        for a in g:
            for b in g:
                if a < b and b in one_base_neighbors(a):
                    e += 1
        v = len(g)
        comps = fiber_components(g)
        b1 = e - v + len(comps)
        details[aa] = (v, e, len(comps), b1)
        total += b1
    n_prefix_k4 = 0
    for a in "ACGT":
        for b in "ACGT":
            box = [a + b + c for c in "ACGT"]
            aas = {STANDARD_CODE[c] for c in box}
            if len(aas) == 1 and "*" not in aas:
                e = sum(
                    1
                    for i, x in enumerate(box)
                    for y in box[i + 1 :]
                    if y in one_base_neighbors(x)
                )
                if e == 6:
                    n_prefix_k4 += 1
    ile = details["I"]
    leu = details["L"]
    arg = details["R"]
    ser = details["S"]
    return BoxBetaRow(
        beta1_sum=total,
        n_k4=n_prefix_k4,
        ile_k3=ile[0] == 3 and ile[1] == 3 and ile[3] == 1,
        leu_extra=leu[3] - 3,
        arg_extra=arg[3] - 3,
        ser_disconnected=ser[2] == 2,
    )

def sheet_census(enc: NucleotideEncoding) -> SheetRow:
    fam_ok = True
    for b in range(256):
        intron = intron_from_byte(b, CHIRALITY_D)
        eps_a = intron & 1
        eps_b = (intron >> 7) & 1
        fam, _pay = unpack_byte(b)
        if fam != ((eps_b << 1) | eps_a):
            fam_ok = False
            break
    ser = list(fibers(STANDARD_CODE)["S"])
    comps = fiber_components(ser)
    tcn = [c for c in ser if c.startswith("TC")]
    agy = [c for c in ser if c.startswith("AG")]
    wc_agy = sorted(wc_codon(c) for c in agy)
    serine_wc = set(wc_agy).issubset(tcn) and len(comps) == 2
    paired = True
    for c in agy:
        p = pack_codon_bits(c, enc)
        pc = pack_codon_bits(wc_codon(c), enc)
        for f in range(4):
            s = omega_word_signature([pack_byte(f, p)])
            sc = omega_word_signature([pack_byte(f ^ 2, pc)])
            if s != sc:
                paired = False
    n_aa = 0
    n_distinct = 0
    for aa, group in fibers(STANDARD_CODE).items():
        if aa == "*":
            continue
        n_aa += 1
        sigs = set()
        for c in group:
            p = pack_codon_bits(c, enc)
            for f in range(4):
                s = omega_word_signature([pack_byte(f, p)])
                sigs.add((s.parity, s.tau_u6, s.tau_v6))
        if len(sigs) == 4 * len(group):
            n_distinct += 1
    return SheetRow(
        family_is_eps=fam_ok,
        serine_wc_complements=serine_wc,
        serine_sig_paired=paired,
        n_aa_all_sigs_distinct=n_distinct,
        n_aa=n_aa,
    )

def pos_census() -> PosAnisoRow:
    by_pos: Dict[int, List[Tuple[str, str, str]]] = {0: [], 1: [], 2: []}
    for c in CODONS:
        for n in one_base_neighbors(c):
            if c < n and STANDARD_CODE[c] == STANDARD_CODE[n]:
                pos = next(i for i in range(3) if c[i] != n[i])
                by_pos[pos].append((c, n, STANDARD_CODE[c]))
    sense1 = [e for e in by_pos[1] if e[2] != "*"]
    return PosAnisoRow(
        n_pos=(len(by_pos[0]), len(by_pos[1]), len(by_pos[2])),
        pos1_edges=tuple(by_pos[1]),
        sense_pos1=len(sense1),
    )

def ncbi_move_census() -> Tuple[Tuple[int, int, str], ...]:
    t1 = translation_table(1)
    rows = []
    for tid in NCBI_TABLE_IDS:
        t = translation_table(tid)
        diffs = [(c, t1[c], t[c]) for c in CODONS if t1[c] != t[c]]
        kinds: Counter[str] = Counter()
        for _c, a, b in diffs:
            if a != "*" and b == "*":
                kinds["sense_to_stop"] += 1
            elif a == "*" and b != "*":
                kinds["stop_to_sense"] += 1
            else:
                kinds["sense_to_sense"] += 1
        kind = ",".join(f"{k}:{n}" for k, n in sorted(kinds.items())) or "identical"
        rows.append((tid, len(diffs), kind))
    return tuple(rows)

def word_geometry_census() -> WordGeometryCensus:
    wrow = w_census()
    wrc = word_rc_census()
    boxes = box_census()
    pair_enc = encodings_in_orbit(ORBIT_PAIR_INV)[0][1]
    sheet = sheet_census(pair_enc)
    pos = pos_census()
    ncbi = ncbi_move_census()
    pair = next(r for r in wrc if r.orbit == ORBIT_PAIR_INV)
    elem = next(r for r in wrc if r.orbit == ORBIT_ELEMENTARY)
    z_pair_expect = (
        (0, 0, ANTIPODE_6),
        (0, 0, 0),
        (0, ANTIPODE_6, 0),
        (0, ANTIPODE_6, ANTIPODE_6),
    )
    pos1_ok = (
        pos.n_pos[1] == 1
        and pos.sense_pos1 == 0
        and pos.pos1_edges == (("TAA", "TGA", "*"),)
    )
    gates = {
        "70_W_is_interior_diagonal": wrow.set_eq_constraints and wrow.rank == 4,
        "71_W_fold_fixed_antipode": wrow.fold_invariant and wrow.antipode_in,
        "72_payload_rc_word_law_pair": pair.n_ok == pair.n_total and pair.n_total > 0,
        "73_Z_period4_pair": pair.z_period4 == z_pair_expect,
        "74_theta_involution": pair.theta_involution,
        "75_beta1_box_sum_27": (
            boxes.beta1_sum == 27
            and boxes.n_k4 == 8
            and boxes.ile_k3
            and boxes.leu_extra == 1
            and boxes.arg_extra == 1
        ),
        "76_family_is_eps_k4": sheet.family_is_eps,
        "77_serine_wc_sheet_pairing": sheet.serine_wc_complements and sheet.serine_sig_paired,
        "78_pos1_syn_is_stop_path": pos1_ok,
        "79_pair_law_fails_on_elem": elem.n_ok != elem.n_total and elem.z_period4 != z_pair_expect,
    }
    return WordGeometryCensus(
        W=wrow,
        word_rc=wrc,
        boxes=boxes,
        sheet=sheet,
        pos=pos,
        ncbi_moves=ncbi,
        gates=gates,
    )

def print_word_geometry_census(c: WordGeometryCensus) -> None:
    g = c.gates
    report_section("6. W, PAYLOAD-RC WORD LAW, BOX beta1, FAMILY SHEET")
    report_objects(
        (
            "W rank-4 residual; payload-RC anti-action; Theta; Z_n; beta1=27; family K4",
        )
    )
    print("  W")
    print(
        f"    constraints match residual set: {c.W.set_eq_constraints}  rank={c.W.rank}  "
        f"R_bit(W)=W: {c.W.fold_invariant}  antipode in W: {c.W.antipode_in}  "
        f"R_block(W)=W: {c.W.block_rev_invariant}"
    )
    print(f"    annihilator masks (bit1+bit3, bit2+bit4) = {tuple(f'{m:06b}' for m in W_ANNIHILATOR)}")
    print()
    print("  payload-RC word law")
    print(f"    {'orbit':<28} {'charts':>6} {'Theta^2=id':>10} {'law holds':>16}  Z_n n=1..4")
    for r in c.word_rc:
        print(
            f"    {r.orbit:<28} {r.n_charts:>6} {str(r.theta_involution):>10} "
            f"{r.n_ok}/{r.n_total:<11}  {r.z_period4}"
        )
    print()
    print("  synonymous cycle rank (sum = beta1 of the code)")
    print(
        f"    sum={c.boxes.beta1_sum}  full K4 boxes={c.boxes.n_k4}  "
        f"Ile K3={c.boxes.ile_k3}  Leu extra={c.boxes.leu_extra}  "
        f"Arg extra={c.boxes.arg_extra}  Ser disconnected={c.boxes.ser_disconnected}"
    )
    print()
    print("  family sheet / serine")
    print(
        f"    family=(eps_b << 1)|eps_a: {c.sheet.family_is_eps}  "
        f"Ser fibers WC-complements: {c.sheet.serine_wc_complements}  "
        f"Ser signatures pair under eps_b flip: {c.sheet.serine_sig_paired}"
    )
    print(
        f"    amino acids with all 4*degeneracy length-1 signatures distinct: "
        f"{c.sheet.n_aa_all_sigs_distinct}/{c.sheet.n_aa}"
    )
    print()
    print("  synonymous edges by codon position")
    print(
        f"    pos0,pos1,pos2 = {c.pos.n_pos}  sense pos1 = {c.pos.sense_pos1}  "
        f"pos1 edges = {c.pos.pos1_edges}"
    )
    print()
    print("  NCBI table moves vs table 1")
    for tid, n, kind in c.ncbi_moves:
        print(f"    t{tid:02d}  n_diff={n:2d}  {kind}")
    print()
    pair = next(r for r in c.word_rc if r.orbit == ORBIT_PAIR_INV)
    elem = next(r for r in c.word_rc if r.orbit == ORBIT_ELEMENTARY)
    report_checks((
        ('W is the interior-diagonal 4-space {x1=x3, x2=x4}', g['70_W_is_interior_diagonal'], f'set_eq={c.W.set_eq_constraints} rank={c.W.rank}', 'True, rank 4'),
        ('W is fold-invariant and contains the antipode', g['71_W_fold_fixed_antipode'], f'R_bit={c.W.fold_invariant} antipode={c.W.antipode_in}', 'True, True'),
        ('payload-RC word law holds on pair_inversion', g['72_payload_rc_word_law_pair'], f'{pair.n_ok}/{pair.n_total}', 'all sampled words on all 8 charts'),
        ('pair_inversion residual Z_n is Klein period-4 (identity at n=2)', g['73_Z_period4_pair'], f'{pair.z_period4}', '((0,0,63),(0,0,0),(0,63,0),(0,63,63))'),
        ('Theta is an involution', g['74_theta_involution'], f'{pair.theta_involution}', 'True'),
        ('beta1=27 = 8*3 + Ile + Leu extra + Arg extra', g['75_beta1_box_sum_27'], f'sum={c.boxes.beta1_sum} K4={c.boxes.n_k4} IleK3={c.boxes.ile_k3} Leu+{c.boxes.leu_extra} Arg+{c.boxes.arg_extra}', '27 = 8*3 + 1 + 1 + 1'),
        ('family bits are the (eps_a, eps_b) K4', g['76_family_is_eps_k4'], f'{c.sheet.family_is_eps}', 'True'),
        ('serine fibers are WC complements and share signatures after eps_b flip', g['77_serine_wc_sheet_pairing'], f'wc={c.sheet.serine_wc_complements} paired={c.sheet.serine_sig_paired}', 'True, True'),
        ('unique pos1 synonymous edge is the stop path TAA—TGA; sense pos1 is empty', g['78_pos1_syn_is_stop_path'], f'pos={c.pos.n_pos} sense_pos1={c.pos.sense_pos1} edges={c.pos.pos1_edges}', 'pos1=1, sense 0, (TAA, TGA, *)'),
        ('elementary residual under pair_inversion Z table', g['79_pair_law_fails_on_elem'], f'elem n=1 with pair Z: {elem.n_ok}/{elem.n_total}; elem Z_n={elem.z_period4}', 'partial; Z_2 nonzero'),
    ))

TYPE_II_PALINDROMIC: Tuple[str, ...] = (
    "GAATTC", "GGATCC", "AAGCTT", "CTCGAG", "GGTACC", "CCGG", "GCGC", "AGCT",
    "CATG", "GTAC", "TATA", "ATAT", "GCAT", "ATGC", "AATT", "GATC", "CAGCTG",
    "ACTAGT", "TCTAGA", "AGATCT", "GAGCTC", "CTGCAG", "ACGTGT",
    "GTCGAC", "CCWGG", "GGWCC", "GCNGC", "RGCGCY", "GANTC", "CCSGG", "GCWGC",
)

def is_palindrome(seq: str) -> bool:
    s = seq.upper()
    return s == reverse_complement_seq(s)

def bracket_byte(seq4: str, enc) -> int:
    return pack_byte(enc.encode_base(seq4[0]), pack_codon_bits(seq4[1:], enc))

def packing_hinge(seq4: str, enc) -> int:
    lin = pack_4mer_byte(seq4, enc).byte
    br = bracket_byte(seq4, enc)
    return lin ^ br

W_UNIFORM_P0 = 15 / 63

def binom_two_sided(k: int, n: int, p0: float) -> float:
    """Exact two-sided binomial p: total mass of outcomes no more likely than k."""
    def pmf(x: int) -> float:
        return math.comb(n, x) * p0**x * (1.0 - p0) ** (n - x)
    cut = pmf(k) * (1.0 + 1e-9)
    return min(1.0, sum(pmf(x) for x in range(n + 1) if pmf(x) <= cut))


def splice_hinge_census(enc) -> Tuple[int, float, float, float]:
    """Donor/acceptor mean |H_J| on chr22 exon junction windows."""
    rows = splice_census(enc, ORBIT_PAIR_INV)
    by_kind = {r.kind: r for r in rows}
    donor = by_kind.get("donor")
    acc = by_kind.get("acceptor")
    if donor is None or donor.status == "SKIP":
        return 0, float("nan"), float("nan"), float("nan")
    return donor.n, donor.mean_4mer_fd, acc.mean_4mer_fd if acc else float("nan"), donor.null_p

def ncbi_surgery_moves() -> Tuple[Counter, List[Tuple[int, str, str, str]]]:
    ref = translation_table(1)
    move_types = Counter()
    examples: List[Tuple[int, str, str, str]] = []
    for tid in NCBI_TABLE_IDS:
        if tid == 1:
            continue
        tab = translation_table(tid)
        for c in CODONS:
            if ref[c] == tab[c]:
                continue
            r, t = ref[c], tab[c]
            if r == "*" and t != "*":
                kind = "stop_to_sense"
            elif r != "*" and t == "*":
                kind = "sense_to_stop"
            elif r == "*" and t == "*":
                kind = "stop_to_stop"
            else:
                kind = "sense_to_sense"
            move_types[kind] += 1
            if len(examples) < 12:
                examples.append((tid, c, r, t))
    return move_types, examples

def restriction_parity() -> Tuple[int, int, int, float, float, float]:
    """Parity of even-length strict palindromes (ACGT only).

    IUPAC-degenerate catalog entries (N, R, Y, W, S, …) are excluded because
    parity is undefined without resolving ambiguity; this biases toward simple sites.
    Returns (n, mod2, mod0, frac_mod2, frac_mod0, two_sided_binomial_p vs 1/2).
    """
    sites = [s.upper().replace("U", "T") for s in TYPE_II_PALINDROMIC]
    strict = [s for s in sites if len(s) % 2 == 0 and all(b in BASES for b in s) and is_palindrome(s)]
    mod2 = sum(1 for s in strict if len(s) % 4 == 2)
    mod0 = sum(1 for s in strict if len(s) % 4 == 0)
    n = len(strict)
    p2 = mod2 / n if n else 0.0
    p0 = mod0 / n if n else 0.0
    bp = binom_two_sided(mod2, n, 0.5) if n else float("nan")
    return n, mod2, mod0, p2, p0, bp

def genome_skew(name: str, seq: str) -> Tuple[float, float]:
    seq = seq.upper().replace("U", "T")
    g = seq.count("G")
    c = seq.count("C")
    a = seq.count("A")
    t = seq.count("T")
    gc = (g - c) / (g + c) if (g + c) else 0.0
    at = (a - t) / (a + t) if (a + t) else 0.0
    return gc, at

@dataclass
class SequenceProbesCensus:
    splice_n: int
    donor_fd: float
    acceptor_fd: float
    donor_gt_p: float
    hinge_w_hits: int
    hinge_n: int
    hinge_w_frac: float
    hinge_w_p: float
    surgery_moves: Tuple[Tuple[str, int], ...]
    restr_n: int
    restr_mod2: int
    restr_mod0: int
    restr_p2: float
    restr_p0: float
    restr_binom_p: float
    skew_rows: Tuple[Tuple[str, float, float], ...]
    gates: Dict[str, bool]

def sequence_probes_census() -> SequenceProbesCensus:
    enc = encodings_in_orbit(ORBIT_PAIR_INV)[0][1]
    splice_n, d_fd, a_fd, d_p = splice_hinge_census(enc)
    moves, _ex = ncbi_surgery_moves()
    rn, r2, r0, p2, p0, r_bp = restriction_parity()

    skew_rows: List[Tuple[str, float, float]] = []
    for keys, label in ((("ecoli_k12",), "ecoli"), (("yeast_s288c",), "yeast")):
        recs = load_named_fasta(keys)
        if not recs:
            skew_rows.append((label, float("nan"), float("nan")))
            continue
        seq = "".join(s for _, s in recs[:500])
        gc, at = genome_skew(label, seq)
        skew_rows.append((label, gc, at))

    # packing hinge on all 4-mers: W-membership count vs uniform 15/63 baseline
    hinges = [packing_hinge("".join(p), enc) for p in itertools.product(BASES, repeat=4)]
    nonzero = [h for h in hinges if h]
    w_hits = sum(1 for h in nonzero if in_W(h))
    n_nonzero = len(nonzero)
    w_frac = w_hits / n_nonzero if n_nonzero else 0.0
    w_p = binom_two_sided(w_hits, n_nonzero, W_UNIFORM_P0) if n_nonzero else float("nan")

    gates = {
        "116_splice_donor_gt_acceptor": (
            splice_n > 0 and d_fd > a_fd and d_fd > 2.5 and a_fd < 2.5
        ),
        "117_packing_hinge_W_consistent": w_p > 0.01,
        "118_surgery_low_complexity": sum(moves.values()) < 120,
        "119_restr_mod2_direction": r2 > r0,
        "120_skew_finite": all(abs(gc) <= 1 and abs(at) <= 1 for _, gc, at in skew_rows if gc == gc),
    }
    return SequenceProbesCensus(
        splice_n=splice_n,
        donor_fd=d_fd,
        acceptor_fd=a_fd,
        donor_gt_p=d_p,
        hinge_w_hits=w_hits,
        hinge_n=n_nonzero,
        hinge_w_frac=w_frac,
        hinge_w_p=w_p,
        surgery_moves=tuple(sorted(moves.items())),
        restr_n=rn,
        restr_mod2=r2,
        restr_mod0=r0,
        restr_p2=p2,
        restr_p0=p0,
        restr_binom_p=r_bp,
        skew_rows=tuple(skew_rows),
        gates=gates,
    )

def print_sequence_probes_census(c: SequenceProbesCensus) -> None:
    g = c.gates
    report_section("12. SPLICE HINGE; SURGERY; RESTRICTION; SKEW")
    report_objects(
        (
            "H_J hinge; splice fd; surgery moves; restriction parity; CDS skew",
        )
    )
    print("  splice (pair_inversion chr22)")
    print(
        f"    n_donor={c.splice_n}  mean_fd donor={c.donor_fd:.4f}  "
        f"acceptor={c.acceptor_fd:.4f}  donor_null_p={c.donor_gt_p:.4f}"
    )
    print()
    print("  NCBI surgery moves vs table 1")
    print(f"    {dict(c.surgery_moves)}")
    print()
    print("  restriction site length parity (strict palindromes)")
    print(
        f"    n={c.restr_n}  len=2 mod4={c.restr_mod2}  len=0 mod4={c.restr_mod0}  "
        f"p2={c.restr_p2:.3f}  p0={c.restr_p0:.3f}  "
        f"binomial_p_vs_1/2={c.restr_binom_p:.4f}"
    )
    print()
    print("  genome skew")
    for name, gc, at in c.skew_rows:
        print(f"    {name}: GC_skew={gc:.4f}  AT_skew={at:.4f}")
    print()
    print("  hinge W-membership vs uniform baseline")
    print(
        f"    W hits={c.hinge_w_hits}/{c.hinge_n}  W_frac={c.hinge_w_frac:.4f} "
        f"baseline={W_UNIFORM_P0:.4f} (15/63)  two-sided binomial p={c.hinge_w_p:.4f}"
    )
    print()
    report_checks((
        ('splice donor fold-disagreement exceeds acceptor on pair chart', g['116_splice_donor_gt_acceptor'], f'donor={c.donor_fd:.4f} acceptor={c.acceptor_fd:.4f}', 'donor > acceptor, donor > 2.5'),
        ('packing hinge W-membership consistent with uniform GF(2)^6 baseline', g['117_packing_hinge_W_consistent'], f'W_hits={c.hinge_w_hits}/{c.hinge_n} W_frac={c.hinge_w_frac:.4f} vs {W_UNIFORM_P0:.4f}', 'two-sided binomial p > 0.01 against p0 = 15/63'),
        ('NCBI variant tables are low-complexity surgery (<120 total moves)', g['118_surgery_low_complexity'], f'total={sum((v for _, v in c.surgery_moves))}', '<120'),
        ('Type II palindromes lean length n=2 mod 4 over n=0 mod 4 (direction only)', g['119_restr_mod2_direction'], f'mod2={c.restr_mod2} mod0={c.restr_mod0} binomial_p={c.restr_binom_p:.4f}', 'mod2 > mod0; significance deferred to REBASE-scale (S4)'),
        ('genome GC/AT skew meters are finite on ingested CDS', g['120_skew_finite'], str(c.skew_rows), '|skew|<=1'),
    ))

@dataclass
class TableModuliRow:
    tid: int
    beta1: int
    sense_outer_only: int
    sense_fold_only: int
    wall_closed: bool
    deg_match: bool
    serine_comps: int

@dataclass
class ModuliCensus:
    rows: Tuple[TableModuliRow, ...]
    n_wall_closed: int
    n_beta1_band: int
    n_deg_match: int
    n_fold_le1: int
    n_ser2: int
    n_small_diff: int
    breach_tids: Tuple[int, ...]
    gates: Dict[str, bool]

STD_DEG = Counter(STANDARD_CODE[c] for c in CODONS)

def moduli_census() -> ModuliCensus:
    rows: List[TableModuliRow] = []
    n_wall = 0
    n_b1 = 0
    n_deg = 0
    breaches: List[int] = []
    for tid in NCBI_TABLE_IDS:
        code = translation_table(tid)
        syn, _ = substitution_cost_rows(code)
        outer = syn.get("outer_only", 0)
        fold = syn.get("fold_only", 0)
        wall_ok = fold == 0 and outer >= 60
        if wall_ok:
            n_wall += 1
        else:
            breaches.append(tid)
        _ne, _nc, b1 = synonymous_homology(code)
        if 24 <= b1 <= 30:
            n_b1 += 1
        deg = Counter(code[c] for c in CODONS)
        deg_ok = deg == STD_DEG
        if deg_ok:
            n_deg += 1
        ser_comps = len(fiber_components(fibers(code)["S"]))
        rows.append(
            TableModuliRow(tid, b1, outer, fold, wall_ok, deg_ok, ser_comps)
        )

    n_fold_le1 = sum(1 for r in rows if r.sense_fold_only <= 1)
    n_ser2 = sum(1 for r in rows if r.serine_comps == 2)
    n_small = sum(1 for r in rows if sum(1 for c in CODONS if translation_table(r.tid)[c] != translation_table(1)[c]) <= 6)

    gates = {
        "121_variant_wall_almost_closed": n_fold_le1 == len(NCBI_TABLE_IDS),
        "122_all_tables_beta1_band": n_b1 == len(NCBI_TABLE_IDS),
        "123_surgery_low_diff": n_small >= len(NCBI_TABLE_IDS) - 2,
        "124_serine_two_components": n_ser2 >= len(NCBI_TABLE_IDS) - 1,
        "125_wall_closed_count_18": n_wall == 18,
        "126_wall_breaches_are_fold_openers": (
            len(breaches) == 4 and all(
                next(r for r in rows if r.tid == t).sense_fold_only >= 1 for t in breaches
            )
        ),
    }
    return ModuliCensus(
        rows=tuple(rows),
        n_wall_closed=n_wall,
        n_beta1_band=n_b1,
        n_deg_match=n_deg,
        n_fold_le1=n_fold_le1,
        n_ser2=n_ser2,
        n_small_diff=n_small,
        breach_tids=tuple(breaches),
        gates=gates,
    )

def print_moduli_census(c: ModuliCensus) -> None:
    g = c.gates
    report_section("13. NCBI MODULI; WALL CLOSURE; DEGENERACY PROFILE")
    report_objects(
        (
            "per-table wall (fold_syn=0); beta1 band [24,30]; fiber-size degeneracy vs standard; "
            "breaches = fold-plane openers on nonstandard tables",
        )
    )
    print_table(
        ("tid", "beta1", "outer", "fold", "wall", "deg_ok", "Ser_c"),
        (4, 5, 5, 4, 5, 7, 5),
        [
            (
                r.tid, r.beta1, r.sense_outer_only, r.sense_fold_only,
                str(r.wall_closed), str(r.deg_match), r.serine_comps,
            )
            for r in c.rows
        ],
    )
    print()
    print(
        f"    wall_closed={c.n_wall_closed}/{len(c.rows)}  "
        f"breaches={list(c.breach_tids)}  "
        f"beta1_band={c.n_beta1_band}/{len(c.rows)}  "
        f"deg_match={c.n_deg_match}/{len(c.rows)}"
    )
    print()
    report_checks((
        ('variant tables: at most one sense fold-plane syn edge (wall almost closed)', g['121_variant_wall_almost_closed'], f'fold<=1 on {c.n_fold_le1}/{len(c.rows)} tables', 'all'),
        ('every NCBI table beta1 in [24,30]', g['122_all_tables_beta1_band'], f'{c.n_beta1_band}/{len(c.rows)}', 'all'),
        ('NCBI tables differ from standard by <=6 codons (low-diff surgery)', g['123_surgery_low_diff'], f'{c.n_small_diff}/{len(c.rows)}', f'>={len(c.rows) - 2}'),
        ('serine two-component on >= n-1 tables', g['124_serine_two_components'], f'{c.n_ser2}/{len(c.rows)}', f'>={len(c.rows) - 1}'),
        ('exactly 18/22 tables wall-closed (L_sense off fold)', g['125_wall_closed_count_18'], f'{c.n_wall_closed}/{len(c.rows)}', '18/22'),
        ('four wall breaches open the fold plane (fold_syn>=1)', g['126_wall_breaches_are_fold_openers'], f'breaches={list(c.breach_tids)}', '4 fold openers'),
    ))

def signature_image_census(enc) -> Tuple[int, int, int, bool, bool]:
    """Returns (n_sigs, n_multi, lambda_value, lambda_uniform, byte_cover_ok)."""
    sig_images: Dict[object, Set[int]] = defaultdict(set)
    byte_pairs: Set[Tuple[object, int]] = set()
    cover_ok = True
    for b in range(256):
        bk = reverse_complement_4mer_byte(b, enc)
        sig = omega_word_signature([b])
        qk = q_word6(bk)
        sig_images[sig].add(qk)
        pair = (sig, qk)
        if pair in byte_pairs:
            cover_ok = False
        byte_pairs.add(pair)
    multi = {s: qs for s, qs in sig_images.items() if len(qs) > 1}
    lams: Set[int] = set()
    for s, qs in multi.items():
        ordered = sorted(qs)
        lams.add(ordered[0] ^ ordered[1])
    lam = next(iter(lams)) if len(lams) == 1 else -1
    cover_ok = cover_ok and len(byte_pairs) == 256 and len(sig_images) == 128
    return len(sig_images), len(multi), lam, len(lams) == 1, cover_ok

@dataclass
class ThetaLawCensus:
    n_sigs: int
    n_multi: int
    lambda_value: int
    lambda_uniform: bool
    images_disjoint: bool
    lambda_in_W: bool
    gates: Dict[str, bool]


def theta_law_census() -> ThetaLawCensus:
    """Kinematic signature cover: each Omega signature has two RC images
    differing by a constant lambda in the interior-diagonal space W.
    """
    enc = encodings_in_orbit(ORBIT_PAIR_INV)[0][1]
    n_sigs, n_multi, lam, lam_unif, cover_ok = signature_image_census(enc)
    gates = {
        "307_sig_two_kin_images": n_sigs == 128 and n_multi == 128,
        "308_lambda_global_constant": lam_unif and lam == 0x1F,
        "309_lambda_in_W": in_W(lam),
        "310_sig_byte_cover_bijective": cover_ok and n_sigs * 2 == 256,
    }
    return ThetaLawCensus(
        n_sigs=n_sigs,
        n_multi=n_multi,
        lambda_value=lam,
        lambda_uniform=lam_unif,
        images_disjoint=cover_ok,
        lambda_in_W=in_W(lam),
        gates=gates,
    )


def print_theta_law_census(c: ThetaLawCensus) -> None:
    g = c.gates
    report_section("15. THETA_KIN SIGNATURE LAW")
    report_objects(("Theta_kin signatures; lambda in W; two-image byte cover",))
    print("  Theta_kin signature law (pair_inversion)")
    print(
        f"    signatures={c.n_sigs} multi-image={c.n_multi} "
        f"lambda=0x{c.lambda_value:02X} uniform={c.lambda_uniform} "
        f"in_W={c.lambda_in_W} byte_cover={c.images_disjoint}"
    )
    print(f"    cover arithmetic: {c.n_sigs} sigs x 2 images = {c.n_sigs * 2} bytes")
    print()
    report_checks((
        ('every length-1 Omega signature carries exactly two kinematic images', g['307_sig_two_kin_images'], f'{c.n_multi}/{c.n_sigs}', '128/128'),
        ('image difference is the global constant lambda = 0x1F', g['308_lambda_global_constant'], f'lambda=0x{c.lambda_value:02X}', 'uniform, 0x1F (= antipode XOR e_top)'),
        ('lambda lies in the interior-diagonal space W', g['309_lambda_in_W'], f'in_W={c.lambda_in_W}', 'True'),
        ('256 bytes biject to (signature, kinematic q-image) pairs', g['310_sig_byte_cover_bijective'], f'cover_ok={c.images_disjoint} 128x2=256', 'True'),
    ))

MAX_CDS = 2000

MIN_GENE_CODONS = 30

N_NULL = 40

def genes_of(keys: Sequence[str], extra: Optional[List[str]] = None) -> List[List[str]]:
    out: List[List[str]] = []
    recs = load_named_fasta(tuple(keys))
    if recs:
        for _h, s in recs[:MAX_CDS]:
            cs = iter_codons(s)
            if len(cs) >= MIN_GENE_CODONS:
                out.append(cs)
    if extra:
        for s in extra:
            cs = iter_codons(s)
            if len(cs) >= MIN_GENE_CODONS:
                out.append(cs)
    return out

_POPCOUNT_LUT = np.array([i.bit_count() for i in range(64)], dtype=np.float64)


@lru_cache(maxsize=None)
def _payload_array(enc: NucleotideEncoding) -> np.ndarray:
    return np.array([pack_codon_bits(c, enc) for c in CODONS], dtype=np.uint8)


def gene_shell_stats(codons: Sequence[str], enc) -> Tuple[float, float]:
    if len(codons) < 2:
        return float("nan"), float("nan")
    lut = _payload_array(enc)
    idx = np.fromiter((CODON_INDEX[c] for c in codons), dtype=np.intp, count=len(codons))
    payloads = lut[idx]
    xor = np.bitwise_xor(payloads[:-1], payloads[1:])
    shells = _POPCOUNT_LUT[xor]
    n = len(shells)
    return float(shells.sum() / n), float((shells <= 1.0).sum() / n)

def _gc_at_indices(seq: str) -> Tuple[List[int], List[int]]:
    idx_gc = [i for i, b in enumerate(seq) if b in STRONG]
    idx_at = [i for i, b in enumerate(seq) if b in {"A", "T"}]
    return idx_gc, idx_at


def gc_shuffle_gene(
    codons: Sequence[str],
    rng: random.Random,
    idx_gc: Optional[List[int]] = None,
    idx_at: Optional[List[int]] = None,
) -> List[str]:
    """Permute bases within GC/AT classes inside the gene (preserves composition)."""
    s = "".join(codons)
    if idx_gc is None or idx_at is None:
        idx_gc, idx_at = _gc_at_indices(s)
    chars = list(s)
    gcb = [chars[i] for i in idx_gc]
    atb = [chars[i] for i in idx_at]
    rng.shuffle(gcb)
    rng.shuffle(atb)
    for i, b in zip(idx_gc, gcb):
        chars[i] = b
    for i, b in zip(idx_at, atb):
        chars[i] = b
    s2 = "".join(chars)
    return [s2[i : i + 3] for i in range(0, len(s2) - 2, 3)]

def corpus_mean_shell(genes: Sequence[Sequence[str]], enc) -> Tuple[float, int]:
    tot = 0.0
    n = 0
    for cs in genes:
        m, _low = gene_shell_stats(cs, enc)
        if m == m:
            tot += m * (len(cs) - 1)
            n += len(cs) - 1
    return (tot / n if n else float("nan"), n)

@dataclass
class WalkRow:
    name: str
    n_genes: int
    n_pairs: int
    obs_mean: float
    null_mean: float
    p_low: int
    n_null: int
    low_frac: float

@dataclass
class OmegaWalkCensus:
    rows: Tuple[WalkRow, ...]
    gates: Dict[str, bool]

def omega_walk_census() -> OmegaWalkCensus:
    enc = orbit_reps()[ORBIT_PAIR_INV][1]
    cds22 = extract_chr22_cds(300)
    specs = (
        (("ecoli_k12",), "ecoli", None),
        (("yeast_s288c",), "yeast", None),
        (("chr22_cds",), "chr22", [s for _h, s, _st in cds22]),
    )
    rows: List[WalkRow] = []
    all_smooth = True
    any_data = False
    for keys, label, extra in specs:
        genes = genes_of(keys, extra)
        if not genes:
            rows.append(WalkRow(label, 0, 0, float("nan"), float("nan"), -1, N_NULL, float("nan")))
            continue
        any_data = True
        obs_m, n_pairs = corpus_mean_shell(genes, enc)
        lows = [gene_shell_stats(cs, enc)[1] for cs in genes]
        low_frac = sum(lows) / len(lows)
        gene_idx = [_gc_at_indices("".join(cs)) for cs in genes]
        rng = random.Random(NULL_SEED + 11)
        hits = 0
        null_ms: List[float] = []
        for _ in range(N_NULL):
            ng = [gc_shuffle_gene(cs, rng, *idx) for cs, idx in zip(genes, gene_idx)]
            nm, _np2 = corpus_mean_shell(ng, enc)
            null_ms.append(nm)
            if nm <= obs_m + 1e-12:
                hits += 1
        smooth = hits == 0
        if not smooth:
            all_smooth = False
        rows.append(
            WalkRow(
                name=label,
                n_genes=len(genes),
                n_pairs=n_pairs,
                obs_mean=obs_m,
                null_mean=sum(null_ms) / len(null_ms),
                p_low=hits,
                n_null=N_NULL,
                low_frac=low_frac,
            )
        )
    gates = {
        "131_orfs_smoother_than_gc_shuffles": any_data and all_smooth,
        "132_low_shell_concentrated": all(
            r.low_frac > 0.10 for r in rows if r.n_genes > 0
        ),
    }
    return OmegaWalkCensus(rows=tuple(rows), gates=gates)

def print_omega_walk_census(c: OmegaWalkCensus) -> None:
    g = c.gates
    report_section("16. OMEGA WALK VS GC-MATCHED SHUFFLES; PER-GENE OCCUPATION")
    report_objects(
        (
            "per-gene mean chi-shell; GC/AT-class shuffle null (40); low-shell frac",
        )
    )
    print(
        f"    {'name':<8} {'genes':>6} {'pairs':>9} {'obs':>8} {'null':>8} "
        f"{'p_low':>7} {'lowfrac':>8}"
    )
    for r in c.rows:
        if r.n_genes == 0:
            print(f"    {r.name:<8} SKIP (no data)")
            continue
        print(
            f"    {r.name:<8} {r.n_genes:>6} {r.n_pairs:>9} {r.obs_mean:>8.4f} "
            f"{r.null_mean:>8.4f} {r.p_low:>3}/{r.n_null} {r.low_frac:>8.4f}"
        )
    print()
    rows_ok = [r for r in c.rows if r.n_genes > 0]
    report_checks((
        ('real ORFs smoother than every GC-matched shuffle replicate', g['131_orfs_smoother_than_gc_shuffles'], f'p_low per genome = {[r.p_low for r in rows_ok]} of {(rows_ok[0].n_null if rows_ok else N_NULL)}', 'p_low == 0 on every genome (obs below all nulls)'),
        ('low-shell fraction above 0.10 on every genome', g['132_low_shell_concentrated'], str({r.name: round(r.low_frac, 4) for r in rows_ok}), '> 0.10'),
    ))

WINDOW = 100_000

BLOCK = 50_000

def _chr22_splice_windows() -> Tuple[List[str], List[str]]:
    seq_path = DATA_DIR / "chr22.fa.gz"
    gtf = DATA_DIR / "gencode.v47.annotation.gtf.gz"
    if not seq_path.exists() or not gtf.exists():
        return [], []
    seq = load_chr22_sequence() or ""
    by_tx: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
    opener = gzip.open if str(gtf).endswith(".gz") else open
    with opener(gtf, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9 or parts[2] != "exon" or parts[0] not in {"chr22", "22"}:
                continue
            tx = "tx"
            for tok in parts[8].split(";"):
                if "transcript_id" in tok:
                    tx = tok.split('"')[1] if '"' in tok else tx
            by_tx[tx].append((int(parts[3]) - 1, int(parts[4]), parts[6]))
    nseq = len(seq)
    donors: List[str] = []
    acceptors: List[str] = []
    done = False
    for _tx, spans in list(by_tx.items())[:20000]:
        if done:
            break
        spans = sorted(spans)
        if not spans:
            continue
        strand = spans[0][2]
        for i, (a, b, _s) in enumerate(spans):
            if i + 1 < len(spans):
                na, nb, _ = spans[i + 1]
                if strand != "-":
                    if b >= 2 and b + 6 <= nseq:
                        donors.append(seq[b - 2 : b + 6])
                    if na >= 6:
                        acceptors.append(seq[na - 6 : na])
                else:
                    if a >= 6:
                        donors.append(reverse_complement_seq(seq[a - 6 : a + 2]))
                    if nb + 6 <= nseq:
                        acceptors.append(reverse_complement_seq(seq[nb - 2 : nb + 6]))
            if len(donors) >= MAX_WINDOWS:
                done = True
                break
    return donors, acceptors

def mean_fd(windows: List[str], enc) -> float:
    tot = 0.0
    n = 0
    for w in windows[:MAX_WINDOWS]:
        w = w.upper()
        if len(w) >= 4 and all(ch in BASES for ch in w[:4]):
            tot += fold_disagreement_d(pack_4mer_byte(w[:4], enc).byte, CHIRALITY_D)
            n += 1
    return tot / n if n else float("nan")

def window_screws(seq: str) -> Tuple[List[float], List[float]]:
    """Per-window |GC skew| and |AT skew|."""
    gcs: List[float] = []
    ats: List[float] = []

    def sk(s: str) -> Tuple[float, float]:
        g = s.count("G")
        c = s.count("C")
        a = s.count("A")
        t = s.count("T")
        gc = (g - c) / (g + c) if g + c else float("nan")
        at = (a - t) / (a + t) if a + t else float("nan")
        return gc, at

    for st in range(0, len(seq) - WINDOW, WINDOW):
        gc, at = sk(seq[st : st + WINDOW])
        if gc == gc:
            gcs.append(abs(gc))
            ats.append(abs(at))
    return gcs, ats

def block_walk(seq: str) -> Tuple[int, int]:
    """Sign switches of the per-block GC skew along the chromosome; block count."""
    blocks: List[float] = []

    def sk_g(s: str) -> float:
        g = s.count("G")
        c = s.count("C")
        return (g - c) / (g + c) if g + c else 0.0

    for st in range(0, len(seq) - BLOCK, BLOCK):
        blocks.append(sk_g(seq[st : st + BLOCK]))
    switches = sum(1 for x, y in zip(blocks, blocks[1:]) if x * y < 0)
    return switches, len(blocks)

@dataclass
class MirrorSkewCensus:
    n_donors: int
    d_fd: float
    a_fd: float
    d_rc_fd: float
    a_rc_fd: float
    gap_ident: float
    gap_mirror: float
    frac_gc_below_05: float
    median_abs_at: float
    sign_switches: int
    n_blocks: int
    gates: Dict[str, bool]

def mirror_skew_census() -> MirrorSkewCensus:
    enc = orbit_reps()[ORBIT_PAIR_INV][1]
    donors, acceptors = _chr22_splice_windows()
    d_fd = mean_fd(donors, enc)
    a_fd = mean_fd(acceptors, enc)
    d_rc_fd = mean_fd([reverse_complement_seq(w) for w in donors], enc)
    a_rc_fd = mean_fd([reverse_complement_seq(w) for w in acceptors], enc)
    gap_ident = abs(d_fd - a_fd)
    gap_mirror = min(abs(d_rc_fd - a_fd), abs(a_rc_fd - d_fd))

    seq = load_chr22_sequence()
    if seq:
        gcs, ats = window_screws(seq)
        switches, n_blocks = block_walk(seq)
        frac05 = sum(1 for x in gcs if x < 0.05) / len(gcs) if gcs else float("nan")
        med_at = sorted(ats)[len(ats) // 2] if ats else float("nan")
    else:
        frac05 = med_at = float("nan")
        switches = n_blocks = 0

    mirror_ok = (
        d_fd > a_fd and gap_mirror <= 0.25 * gap_ident and d_rc_fd == d_rc_fd and a_fd == a_fd
    )
    chargaff_ok = frac05 > 0.5 and med_at == med_at
    gates = {
        "133_splice_rc_mirror": mirror_ok,
        "134_chargaff_window_parity": chargaff_ok,
        "135_skew_walk_defined": n_blocks > 500 and switches > 0,
    }
    return MirrorSkewCensus(
        n_donors=len(donors),
        d_fd=d_fd,
        a_fd=a_fd,
        d_rc_fd=d_rc_fd,
        a_rc_fd=a_rc_fd,
        gap_ident=gap_ident,
        gap_mirror=gap_mirror,
        frac_gc_below_05=frac05,
        median_abs_at=med_at,
        sign_switches=switches,
        n_blocks=n_blocks,
        gates=gates,
    )

def print_mirror_skew_census(c: MirrorSkewCensus) -> None:
    g = c.gates
    report_section("17. SPLICE RC-MIRROR; CHARGAFF PARITY; SKEW WALK")
    report_objects(
        (
            "RC-mirror splice; Chargaff |GC| windows; chr22 skew-walk switches",
        )
    )
    print(f"    splice windows n={c.n_donors}")
    print(
        f"      donor={c.d_fd:.4f} acc={c.a_fd:.4f} RC(donor)={c.d_rc_fd:.4f} "
        f"RC(acc)={c.a_rc_fd:.4f}"
    )
    print(f"      gap identity={c.gap_ident:.4f} gap mirror={c.gap_mirror:.4f}")
    print(
        f"    windows: frac|GC|<0.05={c.frac_gc_below_05:.3f} median|AT|={c.median_abs_at:.4f}"
    )
    print(f"    walk: blocks={c.n_blocks} sign_switches={c.sign_switches}")
    print()
    report_checks((
        ('donor/acceptor asymmetry closes under reverse complementation', g['133_splice_rc_mirror'], f'gap_id={c.gap_ident:.4f} gap_mirror={c.gap_mirror:.4f}', 'gap_mirror <= 0.25 * gap_identity'),
        ('Chargaff second-parity holds in most 100 kb windows', g['134_chargaff_window_parity'], f'frac|GC|<0.05={c.frac_gc_below_05:.3f}', '> 0.5'),
        ('cumulative skew walk is defined with sign structure', g['135_skew_walk_defined'], f'blocks={c.n_blocks} switches={c.sign_switches}', 'blocks>500, switches>0'),
    ))

BETA1_LO = 24

BETA1_HI = 30

_FIBERS_STD = fibers(STANDARD_CODE)


def _fiber_beta1(codons: Sequence[str], label_fn) -> int:
    g = list(codons)
    if len(g) <= 1:
        return 0
    members = set(g)
    edges = 0
    adj: Dict[str, List[str]] = {c: [] for c in g}
    for c in g:
        lc = label_fn(c)
        for n in one_base_neighbors(c):
            if n in members and label_fn(n) == lc and c < n:
                edges += 1
                adj[c].append(n)
                adj[n].append(c)
    seen: Set[str] = set()
    comps = 0
    for c in g:
        if c in seen:
            continue
        comps += 1
        stack = [c]
        seen.add(c)
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
    return edges - len(g) + comps


def beta1_after_relabel(c: str, new_aa: str) -> int:
    cur = STANDARD_CODE[c]
    if new_aa == cur:
        _ne, _nc, b1 = synonymous_homology(STANDARD_CODE)
        return b1

    def label_fn(codon: str) -> str:
        return new_aa if codon == c else STANDARD_CODE[codon]

    total = 0
    for aa, grp in _FIBERS_STD.items():
        if aa == cur:
            g = [x for x in grp if x != c]
        elif aa == new_aa:
            g = list(grp) + [c]
        else:
            g = list(grp)
        if len(g) > 1:
            total += _fiber_beta1(g, label_fn)
    return total


def beta1_of(code: Dict[str, str]) -> int:
    _ne, _nc, b1 = synonymous_homology(code)
    return b1

def fold_plane_syn_edges(code: Dict[str, str], enc) -> int:
    """Sense-synonymous one-base edges whose diff lies purely in the fold plane."""
    cnt = 0
    seen: Set[Tuple[str, str]] = set()
    for c in CODONS:
        if code[c] == "*":
            continue
        pc = pack_codon_bits(c, enc)
        for nb in one_base_neighbors(c):
            if code[nb] != code[c] or code[nb] == "*":
                continue
            key = (c, nb) if c < nb else (nb, c)
            if key in seen:
                continue
            seen.add(key)
            d = (pc ^ pack_codon_bits(nb, enc)) & 0x3F
            if d != 0 and (d & OUTER_MASK) == 0:
                cnt += 1
    return cnt

def one_move_neighborhood(enc) -> Tuple[int, int, List[Tuple[str, str, str, int, int]]]:
    """Count single-codon reassignments that keep stops=3, fold<=1, beta1 band."""
    ok = 0
    tot = 0
    examples: List[Tuple[str, str, str, int, int]] = []
    labels = list(AA_ORDER)
    for c in CODONS:
        cur = STANDARD_CODE[c]
        for aa in labels:
            if aa == cur:
                continue
            tab = dict(STANDARD_CODE)
            tab[c] = aa
            stops = sum(1 for x in CODONS if tab[x] == "*")
            if stops != 3:
                continue
            tot += 1
            b1 = beta1_after_relabel(c, aa)
            fe = fold_plane_syn_edges(tab, enc)
            if BETA1_LO <= b1 <= BETA1_HI and fe <= 1:
                ok += 1
                if len(examples) < 8:
                    examples.append((c, cur, aa, b1, fe))
    return ok, tot, examples

def ncbi_tables_in_neighborhood(enc) -> Tuple[int, int]:
    inside = 0
    for tid in NCBI_TABLE_IDS:
        tab = translation_table(tid)
        b1 = beta1_of(tab)
        fe = fold_plane_syn_edges(tab, enc)
        if BETA1_LO <= b1 <= BETA1_HI and fe <= 1:
            inside += 1
    return inside, len(NCBI_TABLE_IDS)

def c63_canonical_test() -> Tuple[int, int, int]:
    """Best global translation t aligning fiber directions to the weight-3 shell."""
    enc = encodings_in_orbit(ORBIT_PAIR_INV)[0][1]
    dirs: List[Set[int]] = []
    for _aa, grp in fibers(STANDARD_CODE).items():
        if len(grp) >= 2:
            base = pack_codon_bits(grp[0], enc)
            dirs.append({pack_codon_bits(g, enc) ^ base for g in grp[1:]})
    best_cnt = -1
    best_t = -1
    for t in range(64):
        cnt = sum(1 for dv in dirs if all(((x ^ t).bit_count() == 3) for x in dv))
        if cnt > best_cnt:
            best_cnt = cnt
            best_t = t
    # the T5 annihilator mask: bits (1,3) and (2,4) equality masks -> 001010
    annihilator_mask = 0b001010
    return best_t, best_cnt, len(dirs)

@dataclass
class SurgeryModuliCensus:
    one_move_ok: int
    one_move_tot: int
    examples: Tuple[Tuple[str, str, str, int, int], ...]
    tables_inside: int
    n_tables: int
    best_translation: int
    best_fibers_hit: int
    n_fibers: int
    gates: Dict[str, bool]

def surgery_moduli_census() -> SurgeryModuliCensus:
    enc = encodings_in_orbit(ORBIT_PAIR_INV)[0][1]
    ok, tot, ex = one_move_neighborhood(enc)
    inside, ntab = ncbi_tables_in_neighborhood(enc)
    bt, bh, nf = c63_canonical_test()
    annihilator_mask = 0b001010
    gates = {
        "136_one_move_locally_nonrigid": ok == tot and tot > 0,
        "137_ncbi_tables_in_neighborhood": inside == ntab and ntab > 0,
        "138_c63_no_literal_translation": bh < nf,
        "139_c63_best_translation_is_annihilator": bt == annihilator_mask,
    }
    return SurgeryModuliCensus(
        one_move_ok=ok,
        one_move_tot=tot,
        examples=tuple(ex),
        tables_inside=inside,
        n_tables=ntab,
        best_translation=bt,
        best_fibers_hit=bh,
        n_fibers=nf,
        gates=gates,
    )

def print_surgery_moduli_census(c: SurgeryModuliCensus) -> None:
    g = c.gates
    report_section("18. SURGERY MODULI NEIGHBORHOOD; C(6,3) CANONICAL TEST")
    report_objects(
        (
            "one-move surgery moduli; C(6,3) fiber directions; T5 annihilator mask",
        )
    )
    print(
        f"    one-move admissible={c.one_move_ok}/{c.one_move_tot} "
        f"NCBI tables inside={c.tables_inside}/{c.n_tables}"
    )
    print(f"    examples {c.examples[:5]}")
    print(
        f"    C(6,3): best t=0b{c.best_translation:06b} hits {c.best_fibers_hit}/{c.n_fibers} "
        f"fiber direction sets"
    )
    print()
    report_checks((
        ('one-reassignment local constraints are non-rigid (all moves admissible)', g['136_one_move_locally_nonrigid'], f'{c.one_move_ok}/{c.one_move_tot}', 'admissible == total (negative local rigidity; U8)'),
        ('every NCBI table satisfies the neighborhood constraints', g['137_ncbi_tables_in_neighborhood'], f'{c.tables_inside}/{c.n_tables}', f'{c.n_tables}/{c.n_tables}'),
        ('no literal translation realizes all fibers as weight-3 shells', g['138_c63_no_literal_translation'], f'{c.best_fibers_hit}/{c.n_fibers}', '< all fibers (C(6,3) not literal)'),
        ('best-aligning translation equals the T5 annihilator mask', g['139_c63_best_translation_is_annihilator'], f't=0b{c.best_translation:06b}', 't=0b001010'),
    ))
