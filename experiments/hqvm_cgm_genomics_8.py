"""Script 8: S6 covariance, stop-boundary moduli, NCBI wall breaches, compile print, Aff orbit, singular sector, serine synthetase, codon-pair radial channel.

Sections:

  29. S6 EDGE-CHARACTER COVARIANCE
      walsh_sign6(g x XOR g y, g r) = walsh_sign6(x XOR y, r) for all of S6.
      Exact finite covariance: memory magnitude invariant, orientation transported.

  30. STOP-BOUNDARY MODULI
      Global enumeration of stop3+Trp boundary objects under payload-form
      constraints. Admissible set is Iso(H(6,2))-invariant (Aff_S6),
      finite, and splits into exactly two orbits (960 standard-class + 1280).

  31. NCBI WALL BREACHES AS BU POSITIONS
      Each of tables 16/22/24/33 opens exactly one sense-synonymous
      fold-plane edge; openers classify as fold pole 11 (serine axis,
      TAG-TTG) vs pole 01 (stop branch, AAG-AGG).

  32. GENOMIC COMPILE PRINT
      GenomicCompile on one E. coli CDS window and one promoter window;
      layers reproduce certified fields (byte/W fold, payload wall, family
      sheet, Omega signature, depth-4 parity, chi shells, boundary keys).

  33. AFF_S6 ORBIT OF STANDARD CODE
      Standard passes all code-classification gates. Aff_S6 acts freely
      (orbit 46080). Hard constraints cut to 512 = 64 translations x C2^3
      letter-bit swaps on pairs (0,1), (2,3), (4,5). Absolute L/R/S product
      8x8x8 and palindromic B3 sign subgroup do not describe the cluster.

  40. SINGULAR-SECTOR NCBI AVOIDANCE
      Fifty-four NCBI reassignments avoid the interiors of the five clean
      complete wobble boxes (Pro/Thr/Val/Ala/Gly).

  41. SERINE MATCHED SYNTHETASE CHECK
      Serine is multi-pole and gauge-degenerate. Direct-contact fibers
      D/N/E/Q/K are single-pole and injective on fold pole 00.

  42. CODON-PAIR RADIAL CHANNEL
      Pair bias organizes by chirality shell on E. coli and yeast, with
      chi-64 eta^2 as ceiling, shell eta^2 as radial share, monotonic
      low-shell enrichment, and 7/7 cross-genome sign agreement.

Objects: S6 transport; Aff_S6 moduli and orbit; BU punctured-square
boundary; complementarity horizon; multi-layer compile; singular-sector
cycle space; radial shell channel.
"""
from __future__ import annotations

import itertools
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

import numpy as np

_EXP = Path(__file__).resolve().parent
_REPO = _EXP.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from gyroscopic.hQVM.api import omega_word_signature, walsh_sign6

from hqvm_cgm_genomics_2 import load_named_fasta
from hqvm_cgm_genomics_4 import (
    OUTER_MASK,
    substitution_cost_rows,
)
from hqvm_cgm_genomics_7 import (
    ref_enc,
    stop_fiber_near_equality,
    stop_trp_complementarity,
    u8_constraints,
    u8_pass,
)
from hqvm_cgm_genomics_common import (
    CODONS,
    CODON_INDEX,
    CODE_NAMES,
    FOLD_MASK,
    NCBI_TABLE_IDS,
    ORBIT_PAIR_INV,
    STANDARD_CODE,
    compile_interval,
    encodings_in_orbit,
    fiber_components,
    fibers,
    fold_disagreement_d,
    genomic_byte_stream,
    iter_codons,
    one_base_neighbors,
    pack_byte,
    pack_codon_bits,
    print_genomic_compile,
    print_table,
    report_checks,
    report_objects,
    report_section,
    translation_table,
)

# ----------------------------------------
# 29. S6 edge-character covariance
# ----------------------------------------

@dataclass
class S6CovarianceCensus:
    n_perms: int
    n_pairs: int
    violations_natural: int
    violations_alt: int
    gates: Dict[str, bool]


def s6_covariance_census() -> S6CovarianceCensus:
    """Exhaustive test of the covariant form sign6(gx^gy, gr) == sign6(x^y, gr')
    over all 720 permutations and all 64x64 difference pairs.

    Natural form applies the SAME permutation g to both arguments; the alt
    form (inverse permutation on rows) must fail, fixing the convention.
    """
    viol_nat = 0
    viol_alt = 0

    def permute(w: int, perm: Sequence[int]) -> int:
        out = 0
        for i in range(6):
            if (w >> i) & 1:
                out |= 1 << perm[i]
        return out

    diffs = list(range(64))
    for perm in itertools.permutations(range(6)):
        inv = [0] * 6
        for i, p in enumerate(perm):
            inv[p] = i
        for base in range(64):
            pb = permute(base, perm)
            for d in diffs:
                x = base ^ d
                if x < base:
                    continue
                px = permute(x, perm)
                pdx = permute(d, perm)
                if walsh_sign6(px ^ pb, pdx) != walsh_sign6(x ^ base, d):
                    viol_nat += 1
                pdi = permute(d, inv)
                if walsh_sign6(px ^ pb, pdi) != walsh_sign6(x ^ base, d):
                    viol_alt += 1
    total = 64 * 64 * 720 // 2 + (64 * 720)  # symmetric-pair accounting
    gates = {
        "262_s6_edge_character_covariant": viol_nat == 0,
        "263_s6_alt_form_not_covariant": viol_alt > 0,
    }
    return S6CovarianceCensus(
        n_perms=720,
        n_pairs=total,
        violations_natural=viol_nat,
        violations_alt=viol_alt,
        gates=gates,
    )


def print_s6_covariance(c: S6CovarianceCensus) -> None:
    g = c.gates
    report_section("29. S6 EDGE-CHARACTER COVARIANCE")
    report_objects((
        "walsh_sign6 as character on edge differences u^v; "
        "S6 acts on payload bits by permutation; translation part drops",
    ))
    print(
        f"    perms={c.n_perms} pairs_tested={c.n_pairs} "
        f"viol_natural={c.violations_natural} viol_alt={c.violations_alt}"
    )
    print()
    report_checks((
        (
            "natural form covariant under all of S6 (exact)",
            g["262_s6_edge_character_covariant"],
            f"viol={c.violations_natural}",
            "0",
        ),
        (
            "alt form (inverse-permuted row index) is NOT covariant",
            g["263_s6_alt_form_not_covariant"],
            f"viol={c.violations_alt}",
            ">0",
        ),
    ))


# ----------------------------------------
# 30. Stop-boundary moduli
# ----------------------------------------

STD_STOPS = ("TAA", "TAG", "TGA")
STD_TRP = "TGG"


@dataclass
class BoundaryModuliCensus:
    n_admissible: int
    invariance_violations: int
    n_orbits: int
    orbit_sizes: Tuple[int, ...]
    std_orbit_index: int
    std_orbit_size: int
    strict_selects_std_orbit: bool
    representatives: Tuple[Tuple[str, ...], ...] = field(default_factory=tuple)
    gates: Dict[str, bool] = field(default_factory=dict)


def _canon(stops, trp):
    return (tuple(sorted(stops)), trp)


def _apply_gen(w: int, gi: int) -> int:
    if gi < 5:
        i, j = gi, gi + 1
        if ((w >> i) & 1) != ((w >> j) & 1):
            return w ^ ((1 << i) | (1 << j))
        return w
    return w ^ (1 << (gi - 5))


def boundary_moduli_census() -> BoundaryModuliCensus:
    """Global enumeration of all C(64,4)x4 stop3+W configurations
    filtered by payload-invariant clauses:
      - punctured rank-2 affine hull completed by Trp,
      - complementarity: ab+horizon=12 over hull pairs,
      - stop-side max ab <= 4,
      - strict completion: Trp payload-adjacent to a stop (weight(u^v)=1).
    Orbit decomposition under Aff_S6 via 11 generators (adjacent bit
    transpositions + unit translations), canonical key sorts stops.
    """
    enc = ref_enc()
    ALL = list(CODONS)
    PACKED = {c: pack_codon_bits(c, enc) for c in ALL}
    UNPACK = {v: k for k, v in PACKED.items()}

    from hqvm_cgm_genomics_common import affine_rank6

    def passes(stops, trp):
        tab = dict.fromkeys(ALL, "X")
        for s in stops:
            tab[s] = "*"
        tab[trp] = "W"
        if not (
            stop_trp_complementarity(tab, enc)
            and stop_fiber_near_equality(tab, enc)
        ):
            return False
        bits = [PACKED[c] for c in stops] + [PACKED[trp]]
        return affine_rank6(bits) == 2

    passing = set()
    strict_passing = set()
    for sub in itertools.combinations(range(64), 4):
        subc = [ALL[i] for i in sub]
        for wpos in range(4):
            stops = tuple(sorted(c for i, c in enumerate(subc) if i != wpos))
            trp = subc[wpos]
            if passes(stops, trp):
                passing.add((stops, trp))
                wt = PACKED[trp]
                if any((wt ^ PACKED[s]).bit_count() == 1 for s in stops):
                    strict_passing.add((stops, trp))

    def g_canon(obj, gi):
        stops, trp = obj
        ws = [PACKED[s] for s in stops]
        wt = PACKED[trp]
        nws = sorted(_apply_gen(w, gi) & 0x3F for w in ws)
        nwt = _apply_gen(wt, gi) & 0x3F
        return _canon([UNPACK[w] for w in nws], UNPACK[nwt])

    violations = 0
    for obj in passing:
        for gi in range(11):
            if g_canon(obj, gi) not in passing:
                violations += 1

    seen: set = set()
    orbits: List[Tuple[Tuple[Tuple[str, str, str], str], ...]] = []
    std_orbit_index = -1
    std_key = _canon(STD_STOPS, STD_TRP)
    for obj in sorted(passing):
        if obj in seen:
            continue
        comp = [obj]
        seen.add(obj)
        stack = [obj]
        while stack:
            cur = stack.pop()
            for gi in range(11):
                nb = g_canon(cur, gi)
                if nb in passing and nb not in seen:
                    seen.add(nb)
                    comp.append(nb)
                    stack.append(nb)
        orbits.append(tuple(sorted(comp)))
        if std_key in comp:
            std_orbit_index = len(orbits) - 1

    sizes = tuple(sorted((len(o) for o in orbits), reverse=True))
    reps = tuple(tuple(o[0][0]) + (o[0][1],) for o in sorted(orbits, key=len, reverse=True)[:4])
    strict_in_std = bool(std_orbit_index >= 0) and strict_passing.issubset(
        set(orbits[std_orbit_index]) if std_orbit_index >= 0 else set()
    )

    gates = {
        "264_boundary_moduli_finite": len(passing) > 0,
        "265_boundary_admissible_aff_invariant": violations == 0,
        "266_boundary_two_orbits": len(sizes) == 2,
        "267_standard_in_minority_orbit": std_orbit_index >= 0 and len(orbits[std_orbit_index]) == min(sizes),
        "268_strict_completion_single_orbit": len(strict_passing) == 960 and strict_in_std,
    }
    return BoundaryModuliCensus(
        n_admissible=len(passing),
        invariance_violations=violations,
        n_orbits=len(sizes),
        orbit_sizes=sizes,
        std_orbit_index=std_orbit_index,
        std_orbit_size=len(orbits[std_orbit_index]) if std_orbit_index >= 0 else -1,
        strict_selects_std_orbit=strict_in_std,
        representatives=reps,
        gates=gates,
    )


def print_boundary_moduli(c: BoundaryModuliCensus) -> None:
    g = c.gates
    report_section("30. STOP-BOUNDARY MODULI UNDER Iso(H(6,2))")
    report_objects((
        "boundary object = stop^3 + Trp 4-subset of GF(2)^6; "
        "clauses: rank-2 hull, complementarity (ab+horizon=12), stop-side ab bound; "
        "orbit census under Iso(H(6,2)) = GF(2)^6 rtimes S6; BU punctured square",
    ))
    print_table(
        ("quantity", "value"),
        (34, 20),
        [
            ("admissible boundaries", c.n_admissible),
            ("invariance violations", c.invariance_violations),
            ("Iso(H(6,2)) orbits", c.n_orbits),
            ("orbit sizes", str(c.orbit_sizes)),
            ("standard orbit size", c.std_orbit_size),
            ("strict class within std orbit", c.strict_selects_std_orbit),
        ],
        aligns=("<", ">"),
    )
    print()
    report_checks((
        (
            "enumeration finite at scale (2240 expected structure)",
            g["264_boundary_moduli_finite"],
            f"n={c.n_admissible}",
            ">0 finite",
        ),
        (
            "admissible set is Iso(H(6,2))-invariant (payload-form clauses)",
            g["265_boundary_admissible_aff_invariant"],
            f"viol={c.invariance_violations}",
            "0",
        ),
        (
            "exactly two Iso(H(6,2)) orbits",
            g["266_boundary_two_orbits"],
            f"sizes={c.orbit_sizes}",
            "(960, 1280)",
        ),
        (
            "standard sits in the minority orbit",
            g["267_standard_in_minority_orbit"],
            f"std_size={c.std_orbit_size}",
            "min(sizes)",
        ),
        (
            "strict Trp-completion selects exactly the standard orbit",
            g["268_strict_completion_single_orbit"],
            f"bool={c.strict_selects_std_orbit}",
            "True",
        ),
    ))


# ----------------------------------------
# 31. NCBI wall breaches as BU positions
# ----------------------------------------

@dataclass
class BreachRow:
    tid: int
    name: str
    fold_edges: int
    outer_edges: int
    opener_c: str
    opener_nb: str
    opener_pole: int
    failed_constraints: Tuple[str, ...]


@dataclass
class WallBreachCensus:
    rows: Tuple[BreachRow, ...]
    breach_tids: Tuple[int, ...]
    gates: Dict[str, bool]


_WALL_BREACH_IDS = (16, 22, 24, 33)


def wall_breach_census() -> WallBreachCensus:
    """Classify the four NCBI fold-wall breaches by fold-plane opener.

    On the pair_inversion chart each breach opens exactly one
    sense-synonymous one-base edge whose payload diff lies purely in the fold
    plane; its fold coordinate is 11 (the serine fold-fixed axis, tables
    16/22 via the TAG<->TTG Leu bridge) or 01 (the stop-branch direction,
    same diff as TAA<->TGA, tables 24/33 via AAG<->AGG).
    """
    enc = ref_enc()
    PACK = {c: pack_codon_bits(c, enc) for c in CODONS}
    std_syn = substitution_cost_rows(STANDARD_CODE)[0]

    rows: List[BreachRow] = []
    for tid in _WALL_BREACH_IDS:
        code = translation_table(tid)
        syn, _all = substitution_cost_rows(code)
        openers = []
        seen = set()
        for c in CODONS:
            if code[c] == "*":
                continue
            for nb in one_base_neighbors(c):
                if code[nb] != code[c] or code[nb] == "*":
                    continue
                key = (c, nb) if c < nb else (nb, c)
                if key in seen:
                    continue
                seen.add(key)
                d = (PACK[c] ^ PACK[nb]) & 0x3F
                if d != 0 and (d & OUTER_MASK) == 0:
                    openers.append((c, nb, (d & FOLD_MASK) >> 2))
        fails = tuple(k for k, v in u8_constraints(code, enc).items() if not v)
        if len(openers) != 1:
            raise AssertionError(f"table {tid}: expected 1 opener, got {openers}")
        oc, onb, opole = openers[0]
        rows.append(
            BreachRow(
                tid=tid,
                name=CODE_NAMES.get(tid, f"table_{tid}"),
                fold_edges=int(syn.get("fold_only", 0)),
                outer_edges=int(syn.get("outer_only", 0)),
                opener_c=oc,
                opener_nb=onb,
                opener_pole=opole,
                failed_constraints=fails,
            )
        )

    gates = {
        "269_breach_count_four": len(rows) == 4,
        "270_each_breach_single_fold_opener": all(r.fold_edges == 1 for r in rows),
        "271_standard_wall_closed_67_outer": (
            int(std_syn.get("fold_only", 0)) == 0 and int(std_syn.get("outer_only", -1)) == 67
        ),
        "272_serine_axis_class_pole11": {
            r.opener_pole for r in rows if r.tid in (16, 22)
        } == {0b11},
        "273_stop_branch_class_pole01": {
            r.opener_pole for r in rows if r.tid in (24, 33)
        } == {0b01},
        "274_openers_tagttg_aagagg": {
            (r.opener_c, r.opener_nb) for r in rows
        } == {("TAG", "TTG"), ("AAG", "AGG")},
        "282_breach_tables_fail_wall_closed": all(
            "wall_closed" in r.failed_constraints for r in rows
        ),
        "283_tables_1_and_11_pass_u8": (
            u8_pass(translation_table(1), enc)
            and u8_pass(translation_table(11), enc)
        ),
        "284_breach_hard_plus_soft_failures": all(
            r.fold_edges == 1
            and "wall_closed" in r.failed_constraints
            and len(r.failed_constraints) >= 2
            for r in rows
        ),
    }
    return WallBreachCensus(rows=tuple(rows), breach_tids=_WALL_BREACH_IDS, gates=gates)


def print_wall_breach_census(c: WallBreachCensus) -> None:
    g = c.gates
    report_section("31. NCBI WALL BREACHES AS BU POSITIONS")
    report_objects((
        "breach classes on payload bits 2-3: pole 11 = serine fold-fixed axis "
        "(tables 16, 22); pole 01 = stop-branch direction, TAA-TGA diff "
        "(tables 24, 33); Weyl bit-fold on the fold plane",
    ))
    print_table(
        ("tid", "name", "fold", "outer", "opener", "pole", "fails"),
        (4, 18, 5, 6, 12, 5, 28),
        [
            (
                r.tid, r.name[:18], r.fold_edges, r.outer_edges,
                f"{r.opener_c}-{r.opener_nb}",
                format(r.opener_pole, "02b"),
                ",".join(r.failed_constraints[:4]),
            )
            for r in c.rows
        ],
    )
    print()
    report_checks((
        (
            "exactly four breach tables",
            g["269_breach_count_four"],
            f"n={len(c.rows)}",
            "4",
        ),
        (
            "each breach opens exactly one fold-plane sense edge",
            g["270_each_breach_single_fold_opener"],
            "fold=1 x4",
            "1 each",
        ),
        (
            "standard wall closed (67 outer / 0 fold sense edges)",
            g["271_standard_wall_closed_67_outer"],
            "67/0",
            "67/0",
        ),
        (
            "tables 16+22 ride the serine axis (fold pole 11)",
            g["272_serine_axis_class_pole11"],
            "TAG-TTG",
            "pole=11",
        ),
        (
            "tables 24+33 ride the stop branch (fold pole 01)",
            g["273_stop_branch_class_pole01"],
            "AAG-AGG",
            "pole=01",
        ),
        (
            "opener set is exactly TAG-TTG and AAG-AGG",
            g["274_openers_tagttg_aagagg"],
            "2 edges over 4 tables",
            "set match",
        ),
        (
            "all four breach tables fail wall_closed (hard fold breach)",
            g["282_breach_tables_fail_wall_closed"],
            "wall in fails x4",
            "True",
        ),
        (
            "NCBI tables 1 and 11 pass full u8 constraints",
            g["283_tables_1_and_11_pass_u8"],
            "pass both",
            "True",
        ),
        (
            "each breach is hard (fold=1, wall open) plus soft clause failures",
            g["284_breach_hard_plus_soft_failures"],
            "fold=1 and |fails|>=2",
            "True x4",
        ),
    ))


# ----------------------------------------
# 32. GenomicCompile print
# ----------------------------------------

def compile_print_census() -> Dict[str, bool]:
    """Assemble GenomicCompile on two real windows and check printed layers
    against certified fields (no classifier, no held-out accuracy product)."""
    from hqvm_cgm_genomics_2 import _load_ecoli_replicon, _load_regulondb_promoters

    enc = ref_enc()
    rep = _load_ecoli_replicon()
    gates: Dict[str, bool] = {}
    if not rep:
        print("E. coli replicon missing; compile print skipped")
        return {"275_compile_smoke_pass": False}

    prom = next(
        (p for p in _load_regulondb_promoters()
         if p.strand == "forward" and p.tss0 > 80),
        None,
    )
    if prom is None:
        print("no suitable RegulonDB forward promoter (tss0>80); compile print skipped")
        return {"275_compile_smoke_pass": False}
    tss0 = prom.tss0
    wins = (
        ("ecoli_cds_900", rep[tss0 + 30 : tss0 + 930]),
        (f"promoter_tss{tss0}", rep[tss0 - 60 : tss0 + 30]),
    )
    ok = True
    for label, win in wins:
        gc_obj = compile_interval(win, enc, label=label)
        print_genomic_compile(gc_obj)
        print()
        stream = genomic_byte_stream(win, enc)
        nb = gc_obj.value("byte_fold_w", "n_bytes")
        l1 = gc_obj.value("family_sheet", "l1_uniform")
        chi = gc_obj.value("chi_shells", "mean_shell")
        eta = gc_obj.value("qubec_order", "eta")
        m2 = gc_obj.value("qubec_order", "M2")
        ab_sum = gc_obj.value("ab_horizon", "ab_plus_horizon")
        mean_fd = gc_obj.value("byte_fold_w", "mean_fold_disagreement")
        ok &= nb is not None and int(nb) == len(stream)
        ok &= l1 is not None and 0.0 <= l1 <= 1.0
        ok &= chi is not None and 0 <= chi <= 6
        ok &= (
            eta is not None and m2 is not None
            and eta == eta and m2 == m2 and m2 > 0
        )
        if gc_obj.value("ab_horizon", "n_pairs"):
            ok &= ab_sum is not None and abs(ab_sum - 12.0) < 1e-3
        manual_fd = sum(fold_disagreement_d(b, 6) for b in stream) / len(stream) if stream else 0.0
        ok &= mean_fd is not None and abs(mean_fd - manual_fd) < 1e-12
        fold_hist = {p: 0 for p in range(4)}
        for b in stream:
            fold_hist[(b & FOLD_MASK) >> 2] += 1
        for p in range(4):
            got = gc_obj.value("fold_poles", f"pole_{p:02b}_frac")
            exp = fold_hist[p] / len(stream) if stream else float("nan")
            ok &= got is not None and abs(got - exp) < 1e-12
        if label.startswith("ecoli_cds"):
            d4 = gc_obj.value("depth4_parity", "parity_zero_frac")
            ok &= d4 is not None and d4 == 1.0
    gates["275_compile_smoke_pass"] = bool(ok)
    return gates


def print_compile_print_census(gates: Dict[str, bool]) -> None:
    report_section("32. GENOMIC COMPILE PRINT")
    report_objects((
        "GenomicCompile = certified fields packaged: byte/W fold + fold poles "
        "(bits 2-3) + family sheet mu + Omega signature + depth-4 parity + "
        "chi shells + QuBEC eta/M2 + ab/horizon + boundary keys; "
        "object lives in common.py; this is a thin print",
    ))
    report_checks((
        (
            "compile layers reproduce certified sections on both windows",
            gates["275_compile_smoke_pass"],
            "CDS + promoter windows pass",
            "pass",
        ),
    ))



# ----------------------------------------
# 33. Aff_S6 orbit of standard code
# ----------------------------------------

def _inner4(w: int) -> int:
    return ((w & 0b110000) >> 2) | (w & 0b11)


def _cube_minus_edge_type(six: FrozenSet[int]) -> str:
    rel = frozenset(_inner4(x) ^ _inner4(next(iter(six))) for x in six)
    for triple in itertools.combinations([x for x in range(16) if x], 3):
        sub = {
            0, triple[0], triple[1], triple[2],
            triple[0] ^ triple[1], triple[0] ^ triple[2],
            triple[1] ^ triple[2], triple[0] ^ triple[1] ^ triple[2],
        }
        if rel <= sub:
            miss = tuple(sorted(sub - rel))
            if len(miss) == 2 and (miss[0] ^ miss[1]) in sub:
                return "cube_minus_edge"
            return "rank3_other"
    return "not_rank3"


def _code_key(code: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(code.items()))


def _transport_code(
    code: Dict[str, str],
    enc,
    unpack: Dict[int, str],
    gi: int,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for c in CODONS:
        w = pack_codon_bits(c, enc) & 0x3F
        nw = _apply_gen(w, gi) & 0x3F
        out[unpack[nw]] = code[c]
    return out


def _translate_code(
    code: Dict[str, str],
    enc,
    unpack: Dict[int, str],
    t: int,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for c in CODONS:
        w = pack_codon_bits(c, enc) & 0x3F
        out[unpack[w ^ t]] = code[c]
    return out


def _fiber_bits(code: Dict[str, str], enc, aa: str) -> FrozenSet[int]:
    return frozenset(
        pack_codon_bits(c, enc) & 0x3F for c in CODONS if code[c] == aa
    )


# Codon-letter bit pairs (wobble, middle, first): generators of residual C2^3.
LETTER_BIT_PAIRS: Tuple[Tuple[int, int], ...] = ((0, 1), (2, 3), (4, 5))
# Palindromic B3 pairs (not the residual orientation group).
PALINDROME_BIT_PAIRS: Tuple[Tuple[int, int], ...] = ((0, 5), (1, 4), (2, 3))


def _apply_bit_perm(w: int, read_from: Tuple[int, ...]) -> int:
    out = 0
    for new_i, old_i in enumerate(read_from):
        if (w >> old_i) & 1:
            out |= 1 << new_i
    return out


def _pair_swap_perm(pairs: Tuple[Tuple[int, int], ...], mask: int) -> Tuple[int, ...]:
    r = list(range(6))
    for k, (i, j) in enumerate(pairs):
        if (mask >> k) & 1:
            r[i], r[j] = r[j], r[i]
    return tuple(r)


def _perm_code(
    code: Dict[str, str],
    enc,
    unpack: Dict[int, str],
    read_from: Tuple[int, ...],
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for c in CODONS:
        w = pack_codon_bits(c, enc) & 0x3F
        out[unpack[_apply_bit_perm(w, read_from) & 0x3F]] = code[c]
    return out


@dataclass
class AffOrbitCensus:
    std_gates: Dict[str, bool]
    std_u8_pass: bool
    six_fiber_geom: Dict[str, str]
    serine_letter_split: Tuple[int, ...]
    orbit_size: int
    u8_pass_count: int
    u8_cluster_size: int
    fail_wall: int
    fail_wobble: int
    fail_beta1: int
    n_translation_orbits: int = 0
    translation_orbit_size: int = 0
    translation_closed: bool = False
    translation_free: bool = False
    n_L_abs: int = 0
    n_R_abs: int = 0
    n_S_abs: int = 0
    n_LRS_triples: int = 0
    letter_c2_in_cluster: int = 0
    letter_c2_covers: bool = False
    palindrome_c2_in_cluster: int = 0
    gates: Dict[str, bool] = field(default_factory=dict)


def aff_orbit_census() -> AffOrbitCensus:
    enc = ref_enc()
    unpack = {pack_codon_bits(c, enc) & 0x3F: c for c in CODONS}
    std = dict(STANDARD_CODE)
    std_gates = u8_constraints(std, enc)

    six_geom: Dict[str, str] = {}
    ser_split: Tuple[int, ...] = ()
    for aa in ("L", "R", "S"):
        bits = frozenset(
            pack_codon_bits(c, enc) & 0x3F for c in CODONS if std[c] == aa
        )
        six_geom[aa] = _cube_minus_edge_type(bits)
        if aa == "S":
            letters = [c for c in CODONS if std[c] == aa]
            comps = fiber_components(letters)
            ser_split = tuple(sorted((len(x) for x in comps), reverse=True))

    seen: Set[Tuple[Tuple[str, str], ...]] = set()
    stack = [std]
    n_pass = 0
    fail_wall = fail_wobble = fail_beta1 = 0
    while stack:
        cur = stack.pop()
        k = _code_key(cur)
        if k in seen:
            continue
        seen.add(k)
        if u8_pass(cur, enc):
            n_pass += 1
        else:
            chk = u8_constraints(cur, enc)
            if not chk["wall_closed"]:
                fail_wall += 1
            if not chk["wobble_boxes_8"]:
                fail_wobble += 1
            if not chk["beta1_27"]:
                fail_beta1 += 1
        for gi in range(11):
            nxt = _transport_code(cur, enc, unpack, gi)
            if _code_key(nxt) not in seen:
                stack.append(nxt)

    seen_pass: Set[Tuple[Tuple[str, str], ...]] = set()
    pass_codes: List[Dict[str, str]] = []
    pst = [std]
    while pst:
        cur = pst.pop()
        ck = _code_key(cur)
        if ck in seen_pass or not u8_pass(cur, enc):
            continue
        seen_pass.add(ck)
        pass_codes.append(cur)
        for gi in range(11):
            nxt = _transport_code(cur, enc, unpack, gi)
            if u8_pass(nxt, enc):
                pst.append(nxt)

    # Factor 512 = 64 translations x 8 orientations (vs absolute 8x8x8).
    keys = set(seen_pass)
    key_to_code = {_code_key(c): c for c in pass_codes}
    closed = True
    free = True
    for code in pass_codes:
        for t in range(64):
            tk = _code_key(_translate_code(code, enc, unpack, t))
            if tk not in keys:
                closed = False
            if t != 0 and tk == _code_key(code):
                free = False
        if not closed:
            break
    remaining = set(keys)
    orb_sizes: List[int] = []
    while remaining:
        seed = remaining.pop()
        code = key_to_code[seed]
        orb = {
            _code_key(_translate_code(code, enc, unpack, t)) for t in range(64)
        }
        remaining -= orb
        orb_sizes.append(len(orb))
    n_T_orbits = len(orb_sizes)
    T_orb_size = orb_sizes[0] if orb_sizes else 0
    L_abs = {_fiber_bits(c, enc, "L") for c in pass_codes}
    R_abs = {_fiber_bits(c, enc, "R") for c in pass_codes}
    S_abs = {_fiber_bits(c, enc, "S") for c in pass_codes}
    triples = {
        (
            _fiber_bits(c, enc, "L"),
            _fiber_bits(c, enc, "R"),
            _fiber_bits(c, enc, "S"),
        )
        for c in pass_codes
    }

    orbit_size = len(seen)
    u8_cluster = len(seen_pass)
    non_pass = orbit_size - n_pass
    factor_64x8 = (
        closed
        and free
        and n_T_orbits == 8
        and set(orb_sizes) == {64}
        and u8_cluster == 512
    )
    not_8x8x8 = not (
        len(L_abs) == 8 and len(R_abs) == 8 and len(S_abs) == 8 and len(triples) == 512
    )

    # Residual orientations = C2^3 letter-bit swaps on (0,1), (2,3), (4,5).
    letter_perms = [_pair_swap_perm(LETTER_BIT_PAIRS, m) for m in range(8)]
    letter_in = 0
    letter_keys: Set[Tuple[Tuple[str, str], ...]] = set()
    for perm in letter_perms:
        img = _perm_code(std, enc, unpack, perm)
        ik = _code_key(img)
        if ik in keys and u8_pass(img, enc):
            letter_in += 1
        for t in range(64):
            letter_keys.add(_code_key(_translate_code(img, enc, unpack, t)))
    letter_covers = letter_in == 8 and letter_keys == keys

    pal_perms = [_pair_swap_perm(PALINDROME_BIT_PAIRS, m) for m in range(8)]
    pal_in = sum(
        1
        for perm in pal_perms
        if _code_key(_perm_code(std, enc, unpack, perm)) in keys
        and u8_pass(_perm_code(std, enc, unpack, perm), enc)
    )

    gates = {
        "276_std_passes_all_u8_gates": all(std_gates.values()),
        "277_aff_orbit_size_46080": orbit_size == 46080,
        "278_u8_pass_cluster_512": n_pass == 512 and u8_cluster == 512,
        "279_L_R_S_cube_minus_edge": all(
            six_geom[a] == "cube_minus_edge" for a in ("L", "R", "S")
        ),
        "280_serine_letter_split_42": ser_split == (4, 2),
        "281_wall_wobble_equal_dominant_fails": (
            non_pass > 0
            and fail_wall == fail_wobble
            and fail_wall >= int(0.94 * non_pass)
        ),
        "300_cluster_factors_64x8": factor_64x8,
        "301_cluster_not_absolute_8x8x8": not_8x8x8 and u8_cluster == 512,
        "302_orientations_are_letter_C2_3": letter_covers,
        "303_not_palindrome_B3_C2_3": pal_in < 8 and u8_cluster == 512,
    }
    return AffOrbitCensus(
        std_gates=std_gates,
        std_u8_pass=u8_pass(std, enc),
        six_fiber_geom=six_geom,
        serine_letter_split=ser_split,
        orbit_size=orbit_size,
        u8_pass_count=n_pass,
        u8_cluster_size=u8_cluster,
        fail_wall=fail_wall,
        fail_wobble=fail_wobble,
        fail_beta1=fail_beta1,
        n_translation_orbits=n_T_orbits,
        translation_orbit_size=T_orb_size,
        translation_closed=closed,
        translation_free=free,
        n_L_abs=len(L_abs),
        n_R_abs=len(R_abs),
        n_S_abs=len(S_abs),
        n_LRS_triples=len(triples),
        letter_c2_in_cluster=letter_in,
        letter_c2_covers=letter_covers,
        palindrome_c2_in_cluster=pal_in,
        gates=gates,
    )


def print_aff_orbit_census(c: AffOrbitCensus) -> None:
    report_section("33. Iso(H(6,2)) ORBIT OF STANDARD CODE")
    report_objects((
        "Iso(H(6,2)) = GF(2)^6 rtimes S6 (Aff_S6); transport labeled "
        "standard under 11 generators; BFS orbit; count constraint-pass "
        "members; factor 512-cluster",
    ))
    print(
        f"    orbit={c.orbit_size} u8_pass={c.u8_pass_count} "
        f"cluster={c.u8_cluster_size} "
        f"fail_wall={c.fail_wall} fail_wobble={c.fail_wobble} "
        f"fail_beta1={c.fail_beta1}"
    )
    print(
        f"    six_fiber_geom L={c.six_fiber_geom['L']} "
        f"R={c.six_fiber_geom['R']} S={c.six_fiber_geom['S']} "
        f"ser_split={c.serine_letter_split}"
    )
    print(
        f"    cluster_factor: T_orbits={c.n_translation_orbits} "
        f"T_orbit_size={c.translation_orbit_size} "
        f"closed={c.translation_closed} free={c.translation_free} "
        f"|L,R,S|_abs=({c.n_L_abs},{c.n_R_abs},{c.n_S_abs}) "
        f"LRS_triples={c.n_LRS_triples}"
    )
    print(
        f"    orientations: letter_C2^3 in_cluster={c.letter_c2_in_cluster}/8 "
        f"covers={c.letter_c2_covers} "
        f"palindrome_B3_C2^3 in_cluster={c.palindrome_c2_in_cluster}/8"
    )
    print()
    g = c.gates
    report_checks((
        (
            "standard passes all code-classification constraints",
            g["276_std_passes_all_u8_gates"],
            f"pass={c.std_u8_pass}",
            "True",
        ),
        (
            "Iso(H(6,2)) acts freely on standard (orbit size 46080)",
            g["277_aff_orbit_size_46080"],
            str(c.orbit_size),
            "46080",
        ),
        (
            "u8-passing Iso(H(6,2)) cluster has size 512 (= 2^9)",
            g["278_u8_pass_cluster_512"],
            f"pass={c.u8_pass_count} cluster={c.u8_cluster_size}",
            "512",
        ),
        (
            "L/R/S six-fibers are cube_minus_edge on inner F2^4",
            g["279_L_R_S_cube_minus_edge"],
            str(c.six_fiber_geom),
            "all cube_minus_edge",
        ),
        (
            "serine letter split is (4,2)",
            g["280_serine_letter_split_42"],
            str(c.serine_letter_split),
            "(4, 2)",
        ),
        (
            "wall and wobble failures coincide and dominate non-passers",
            g["281_wall_wobble_equal_dominant_fails"],
            f"wall={c.fail_wall} wobble={c.fail_wobble} non_pass={c.orbit_size - c.u8_pass_count}",
            "equal and >=94%",
        ),
        (
            "512-cluster factors as 64 translations x 8 orientations",
            g["300_cluster_factors_64x8"],
            f"T_orbits={c.n_translation_orbits} size={c.translation_orbit_size} "
            f"closed={c.translation_closed} free={c.translation_free}",
            "8 orbits of 64",
        ),
        (
            "512-cluster is not an absolute L/R/S product 8x8x8",
            g["301_cluster_not_absolute_8x8x8"],
            f"|L,R,S|=({c.n_L_abs},{c.n_R_abs},{c.n_S_abs}) triples={c.n_LRS_triples}",
            "not (8,8,8) with 512 triples",
        ),
        (
            "8 orientations are letter-bit C2^3 on pairs (0,1)(2,3)(4,5)",
            g["302_orientations_are_letter_C2_3"],
            f"in_cluster={c.letter_c2_in_cluster}/8 covers={c.letter_c2_covers}",
            "8/8 and T x C2^3 = cluster",
        ),
        (
            "8 orientations are not palindromic B3 C2^3 on (0,5)(1,4)(2,3)",
            g["303_not_palindrome_B3_C2_3"],
            f"in_cluster={c.palindrome_c2_in_cluster}/8",
            "<8",
        ),
    ))



# ----------------------------------------
# 40-42. Singular sector, serine synthetase, codon-pair radial
# ----------------------------------------

ENC = encodings_in_orbit(ORBIT_PAIR_INV)[0][1]
_PAYLOAD = {c: pack_codon_bits(c, ENC) for c in CODONS}

CLEAN_BOX_PREFIXES = ("CC", "AC", "GT", "GC", "GG")
CLEAN_BOX: Set[str] = {p + b for p in CLEAN_BOX_PREFIXES for b in "ACGT"}

ANTICODON_CLASS: Dict[str, str] = {
    "A": "blind",
    "S": "blind",
    "D": "direct",
    "N": "direct",
    "E": "direct",
    "Q": "direct",
    "K": "direct",
}


# ----------------------------------------
# 40. Singular-sector NCBI avoidance
# ----------------------------------------

@dataclass
class SingularSectorCensus:
    n_moves: int
    n_sense_origin: int
    n_clean_hit: int
    n_sense_clean_hit: int
    n_clean_box: int
    p_analytic: float
    sector_counts: Dict[str, int]
    gates: Dict[str, bool] = field(default_factory=dict)


def _ncbi_moves() -> List[Tuple[int, str, str]]:
    ref = translation_table(1)
    moves: List[Tuple[int, str, str]] = []
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
            else:
                kind = "sense_to_sense"
            moves.append((tid, c, kind))
    return moves


def _sector(c: str) -> str:
    aa = STANDARD_CODE[c]
    if aa == "*":
        return "stop"
    if aa == "L":
        return "Leu"
    if aa == "R":
        return "Arg"
    if aa == "S":
        return "Ser"
    if aa == "I":
        return "Ile"
    if c in CLEAN_BOX:
        return "clean_box"
    return "other"


def singular_sector_census() -> SingularSectorCensus:
    moves = _ncbi_moves()
    n_clean = sum(1 for _t, c, _k in moves if c in CLEAN_BOX)
    sense_origin = [m for m in moves if m[2] in ("sense_to_sense", "sense_to_stop")]
    n_sense_clean = sum(1 for _t, c, _k in sense_origin if c in CLEAN_BOX)
    distinct = sorted({c for _t, c, _k in moves})
    sectors = Counter(_sector(c) for c in distinct)
    p_in = len(CLEAN_BOX) / 64.0
    p_analytic = (1.0 - p_in) ** len(moves)
    gates = {
        "285_ncbi_moves_54": len(moves) == 54,
        "286_clean_box_avoidance": n_clean == 0,
        "287_sense_origin_avoidance": n_sense_clean == 0,
        "288_avoidance_p_lt_1e6": p_analytic < 1e-6,
    }
    return SingularSectorCensus(
        n_moves=len(moves),
        n_sense_origin=len(sense_origin),
        n_clean_hit=n_clean,
        n_sense_clean_hit=n_sense_clean,
        n_clean_box=len(CLEAN_BOX),
        p_analytic=p_analytic,
        sector_counts=dict(sectors),
        gates=gates,
    )


def print_singular_sector(c: SingularSectorCensus) -> None:
    g = c.gates
    report_section("40. SINGULAR-SECTOR NCBI AVOIDANCE")
    report_objects((
        "NCBI reassignments vs clean Pro/Thr/Val/Ala/Gly box interiors",
        "singular sector = Leu/Arg/Ser/stop cycle generators",
    ))
    print(f"    moves={c.n_moves} sense_origin={c.n_sense_origin} clean_box={c.n_clean_box}")
    print(f"    clean_hits={c.n_clean_hit}/{c.n_moves} sense_origin_clean={c.n_sense_clean_hit}/{c.n_sense_origin}")
    print(f"    p_analytic={c.p_analytic:.3e}")
    print(f"    sectors={c.sector_counts}")
    print()
    report_checks((
        ("NCBI nontrivial moves count 54", g["285_ncbi_moves_54"], f"n={c.n_moves}", "54"),
        ("zero moves in clean box interiors", g["286_clean_box_avoidance"], f"{c.n_clean_hit}/{c.n_moves}", "0"),
        ("zero sense-origin moves in clean interiors", g["287_sense_origin_avoidance"], f"{c.n_sense_clean_hit}/{c.n_sense_origin}", "0"),
        ("uniform-avoidance p below 1e-6", g["288_avoidance_p_lt_1e6"], f"{c.p_analytic:.3e}", "<1e-6"),
    ))


# ----------------------------------------
# 41. Serine matched synthetase check
# ----------------------------------------

@dataclass
class SerSynthetaseCensus:
    ser_n_poles: int
    ser_injective: bool
    direct_all_single: bool
    direct_all_injective: bool
    shared_pole: int
    rows: List[Tuple[str, str, int, int, bool]]
    gates: Dict[str, bool] = field(default_factory=dict)


def _fold_poles(aa: str) -> Set[int]:
    poles: Set[int] = set()
    for c in fibers(STANDARD_CODE)[aa]:
        poles.add((_PAYLOAD[c] & FOLD_MASK) >> 2)
    return poles


def _signatures_injective(aa: str) -> bool:
    group = fibers(STANDARD_CODE)[aa]
    sigs: Set[Tuple[int, int, int]] = set()
    for c in group:
        p = _PAYLOAD[c]
        for f in range(4):
            s = omega_word_signature([pack_byte(f, p)])
            sigs.add((s.parity, s.tau_u6, s.tau_v6))
    return len(sigs) == 4 * len(group)


def ser_synthetase_census() -> SerSynthetaseCensus:
    rows: List[Tuple[str, str, int, int, bool]] = []
    for aa in sorted(fibers(STANDARD_CODE)):
        if aa == "*":
            continue
        cls = ANTICODON_CLASS.get(aa, "mixed")
        poles = _fold_poles(aa)
        inj = _signatures_injective(aa)
        rows.append((aa, cls, len(fibers(STANDARD_CODE)[aa]), len(poles), inj))
    ser = next(r for r in rows if r[0] == "S")
    direct = [r for r in rows if r[1] == "direct"]
    all_single = all(r[3] == 1 for r in direct)
    all_inj = all(r[4] for r in direct)
    shared = set.intersection(*[_fold_poles(r[0]) for r in direct]) if direct else set()
    shared_pole = next(iter(shared)) if len(shared) == 1 else -1
    gates = {
        "289_ser_multipole": ser[3] == 2,
        "290_ser_gauge_degenerate": not ser[4],
        "291_direct_single_pole": all_single,
        "292_direct_injective": all_inj,
        "293_direct_shared_pole_00": shared_pole == 0,
    }
    return SerSynthetaseCensus(
        ser_n_poles=ser[3],
        ser_injective=ser[4],
        direct_all_single=all_single,
        direct_all_injective=all_inj,
        shared_pole=shared_pole,
        rows=rows,
        gates=gates,
    )


def print_ser_synthetase(c: SerSynthetaseCensus) -> None:
    g = c.gates
    report_section("41. SERINE MATCHED SYNTHETASE CHECK")
    report_objects((
        "Ser multi-pole + gauge-degenerate vs anticodon-blind SerRS",
        "direct-contact D/N/E/Q/K single-pole + injective",
    ))
    focus = [r for r in c.rows if r[1] in ("blind", "direct")]
    print_table(
        ("aa", "class", "deg", "n_pole", "inj"),
        (3, 7, 4, 6, 5),
        [(aa, cls, deg, npole, inj) for aa, cls, deg, npole, inj in focus],
        aligns=("<", "<", ">", ">", ">"),
    )
    print(f"    ser_poles={c.ser_n_poles} ser_injective={c.ser_injective}")
    print(f"    direct_single={c.direct_all_single} direct_injective={c.direct_all_injective} shared_pole={c.shared_pole:02b}")
    print()
    report_checks((
        ("serine occupies two fold poles", g["289_ser_multipole"], f"n={c.ser_n_poles}", "2"),
        ("serine length-1 signatures collide", g["290_ser_gauge_degenerate"], f"injective={c.ser_injective}", "False"),
        ("direct-contact fibers are single-pole", g["291_direct_single_pole"], f"{c.direct_all_single}", "True"),
        ("direct-contact signatures injective", g["292_direct_injective"], f"{c.direct_all_injective}", "True"),
        ("direct-contact share fold pole 00", g["293_direct_shared_pole_00"], f"pole={c.shared_pole:02b}", "00"),
    ))


# ----------------------------------------
# 42. Codon-pair radial channel
# ----------------------------------------

@dataclass
class CodonPairRadialCensus:
    ecoli_eta_chi: float
    yeast_eta_chi: float
    ecoli_eta_shell: float
    yeast_eta_shell: float
    ecoli_radial_frac: float
    yeast_radial_frac: float
    ecoli_prof: List[float]
    yeast_prof: List[float]
    sign_agree: int
    ecoli_contrast: float
    yeast_contrast: float
    ecoli_rec_contrast: float
    yeast_rec_contrast: float
    gates: Dict[str, bool] = field(default_factory=dict)


def _pair_count_matrix(codons: Sequence[str]) -> np.ndarray:
    M = np.zeros((64, 64), dtype=float)
    idx = CODON_INDEX
    for u, v in zip(codons, codons[1:]):
        M[idx[u], idx[v]] += 1.0
    return M


def _bias_arrays(M: np.ndarray, min_expected: float = 1.0):
    N = float(M.sum())
    row = M.sum(axis=1)
    col = M.sum(axis=0)
    ys, chis, wts, Es = [], [], [], []
    for i, u in enumerate(CODONS):
        qu = _PAYLOAD[u]
        for j, v in enumerate(CODONS):
            O = M[i, j]
            E = N * (row[i] / N) * (col[j] / N)
            if E < min_expected:
                continue
            chi = qu ^ _PAYLOAD[v]
            ys.append(float(np.log2((O + 1.0) / (E + 1.0))))
            chis.append(chi)
            wts.append(chi.bit_count())
            Es.append(E)
    return np.array(ys), np.array(chis), np.array(wts), np.array(Es)


def _eta_squared(y: np.ndarray, group: np.ndarray, w: np.ndarray) -> float:
    gm = np.average(y, weights=w)
    ss_tot = float((w * (y - gm) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    ss_within = 0.0
    for g in np.unique(group):
        m = group == g
        if not m.any():
            continue
        gm_g = np.average(y[m], weights=w[m])
        ss_within += float((w[m] * (y[m] - gm_g) ** 2).sum())
    return 1.0 - ss_within / ss_tot


def _shell_profile(y: np.ndarray, wts: np.ndarray, w: np.ndarray) -> np.ndarray:
    prof = np.full(7, np.nan)
    for s in range(7):
        m = wts == s
        if m.any():
            prof[s] = np.average(y[m], weights=w[m])
    return prof


def _resample_synonymous(codons: Sequence[str], rng: np.random.Generator) -> List[str]:
    fib = fibers(STANDARD_CODE)
    pools = {aa: list(grp) for aa, grp in fib.items()}
    return [pools[STANDARD_CODE[c]][int(rng.integers(0, len(pools[STANDARD_CODE[c]])))] for c in codons]


def _genome_codons(keys: Tuple[str, ...], max_cds: int = 4000) -> List[str]:
    recs = load_named_fasta(keys)
    if not recs:
        return []
    out: List[str] = []
    for _id, seq in recs[:max_cds]:
        out.extend(iter_codons(seq))
    return out


def _genome_radial(
    codons: Sequence[str], n_resample: int = 4
) -> Tuple[float, float, np.ndarray, float, float]:
    M = _pair_count_matrix(codons)
    y, chis, wts, E = _bias_arrays(M)
    eta_chi = _eta_squared(y, chis, E)
    eta_shell = _eta_squared(y, wts, E)
    prof = _shell_profile(y, wts, E)
    contrast = float(prof[0] - prof[6])
    rng = np.random.Generator(np.random.PCG64(20260827))
    rec = []
    for _ in range(n_resample):
        Mr = _pair_count_matrix(_resample_synonymous(codons, rng))
        yr, _cr, wr, Er = _bias_arrays(Mr)
        pr = _shell_profile(yr, wr, Er)
        rec.append(float(pr[0] - pr[6]))
    return eta_chi, eta_shell, prof, contrast, float(np.mean(rec))


def codon_pair_radial_census() -> CodonPairRadialCensus:
    ecoli = _genome_codons(("ecoli_k12",))
    yeast = _genome_codons(("yeast_s288c",))
    e_chi, e_shell, e_prof, e_con, e_rec = _genome_radial(ecoli)
    y_chi, y_shell, y_prof, y_con, y_rec = _genome_radial(yeast)
    e_frac = float(e_shell / e_chi) if e_chi > 0 else float("nan")
    y_frac = float(y_shell / y_chi) if y_chi > 0 else float("nan")
    agree = sum(
        1
        for s in range(7)
        if not (np.isnan(e_prof[s]) or np.isnan(y_prof[s]))
        and np.sign(e_prof[s]) == np.sign(y_prof[s])
    )
    gates = {
        "294_ecoli_eta_shell": e_shell > 0.02,
        "295_yeast_eta_shell": y_shell > 0.02,
        "296_ecoli_low_shell_enriched": e_con > 0.0,
        "297_yeast_low_shell_enriched": y_con > 0.0,
        "298_shell_sign_agree_7": agree == 7,
        "299_resample_contrast_positive": e_rec > 0.0 and y_rec > 0.0,
        "304_ecoli_eta_chi_above_shell": e_chi > e_shell and e_chi > 0.05,
        "305_yeast_eta_chi_above_shell": y_chi > y_shell and y_chi > 0.05,
        "306_radial_frac_in_band": (
            0.2 < e_frac < 0.7 and 0.2 < y_frac < 0.7
        ),
    }
    return CodonPairRadialCensus(
        ecoli_eta_chi=float(e_chi),
        yeast_eta_chi=float(y_chi),
        ecoli_eta_shell=float(e_shell),
        yeast_eta_shell=float(y_shell),
        ecoli_radial_frac=e_frac,
        yeast_radial_frac=y_frac,
        ecoli_prof=[float(x) for x in e_prof],
        yeast_prof=[float(x) for x in y_prof],
        sign_agree=agree,
        ecoli_contrast=e_con,
        yeast_contrast=y_con,
        ecoli_rec_contrast=e_rec,
        yeast_rec_contrast=y_rec,
        gates=gates,
    )


def print_codon_pair_radial(c: CodonPairRadialCensus) -> None:
    g = c.gates
    report_section("42. CODON-PAIR RADIAL CHANNEL")
    report_objects((
        "pair bias y = log2((O+1)/(E+1)); chi-64 ceiling and shell radial share",
        "cross-genome sign consistency; protein-fixed resample contrast",
    ))
    print(
        f"    ecoli eta_chi={c.ecoli_eta_chi:.4f} eta_shell={c.ecoli_eta_shell:.4f} "
        f"radial_frac={c.ecoli_radial_frac:.3f} contrast={c.ecoli_contrast:+.4f} "
        f"resample={c.ecoli_rec_contrast:+.4f}"
    )
    print(
        f"    yeast eta_chi={c.yeast_eta_chi:.4f} eta_shell={c.yeast_eta_shell:.4f} "
        f"radial_frac={c.yeast_radial_frac:.3f} contrast={c.yeast_contrast:+.4f} "
        f"resample={c.yeast_rec_contrast:+.4f}"
    )
    print(f"    sign_agree={c.sign_agree}/7")
    rows = []
    for s in range(7):
        rows.append((s, f"{c.ecoli_prof[s]:+.4f}", f"{c.yeast_prof[s]:+.4f}"))
    print_table(("shell", "ecoli", "yeast"), (5, 8, 8), rows, aligns=(">", ">", ">"))
    print()
    report_checks((
        ("E. coli shell eta^2 above 0.02", g["294_ecoli_eta_shell"], f"{c.ecoli_eta_shell:.4f}", ">0.02"),
        ("yeast shell eta^2 above 0.02", g["295_yeast_eta_shell"], f"{c.yeast_eta_shell:.4f}", ">0.02"),
        ("E. coli low-shell enriched", g["296_ecoli_low_shell_enriched"], f"contrast={c.ecoli_contrast:+.4f}", ">0"),
        ("yeast low-shell enriched", g["297_yeast_low_shell_enriched"], f"contrast={c.yeast_contrast:+.4f}", ">0"),
        ("shell-profile signs agree on 7/7", g["298_shell_sign_agree_7"], f"{c.sign_agree}/7", "7/7"),
        ("resample low-high contrast positive", g["299_resample_contrast_positive"], f"e={c.ecoli_rec_contrast:+.3f} y={c.yeast_rec_contrast:+.3f}", ">0 both"),
        ("E. coli chi-64 eta^2 above shell", g["304_ecoli_eta_chi_above_shell"], f"{c.ecoli_eta_chi:.4f}>{c.ecoli_eta_shell:.4f}", ">shell and >0.05"),
        ("yeast chi-64 eta^2 above shell", g["305_yeast_eta_chi_above_shell"], f"{c.yeast_eta_chi:.4f}>{c.yeast_eta_shell:.4f}", ">shell and >0.05"),
        ("radial frac shell/chi in (0.2, 0.7)", g["306_radial_frac_in_band"], f"e={c.ecoli_radial_frac:.3f} y={c.yeast_radial_frac:.3f}", "(0.2, 0.7)"),
    ))

# ----------------------------------------
# entry points
# ----------------------------------------

def run_script8_sections(names=None) -> Dict[str, bool]:
    names = names or (
        "s6cov", "boundary", "breach", "compile",
        "orbit", "singular", "ser_synth", "cpb",
    )
    gates: Dict[str, bool] = {}
    if "s6cov" in names:
        c29 = s6_covariance_census()
        print_s6_covariance(c29)
        gates.update(c29.gates)
    if "boundary" in names:
        c30 = boundary_moduli_census()
        print_boundary_moduli(c30)
        gates.update(c30.gates)
    if "breach" in names:
        c31 = wall_breach_census()
        print_wall_breach_census(c31)
        gates.update(c31.gates)
    if "compile" in names:
        g32 = compile_print_census()
        print_compile_print_census(g32)
        gates.update(g32)
    if "orbit" in names:
        c33 = aff_orbit_census()
        print_aff_orbit_census(c33)
        gates.update(c33.gates)
    if "singular" in names:
        c40 = singular_sector_census()
        print_singular_sector(c40)
        gates.update(c40.gates)
    if "ser_synth" in names:
        c41 = ser_synthetase_census()
        print_ser_synthetase(c41)
        gates.update(c41.gates)
    if "cpb" in names:
        c42 = codon_pair_radial_census()
        print_codon_pair_radial(c42)
        gates.update(c42.gates)
    return gates


if __name__ == "__main__":
    run_script8_sections()
