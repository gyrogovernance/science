#!/usr/bin/env python3
"""
hqvm_cgm_genomics_common.py

Shared genomic charts, domain types, NCBI translation tables, kernel
adapters, byte packings, metric orbits, null generators, and provenance
helpers. Kernel transport, family, fold, rank, and Omega maps are imported
from gyroscopic.hQVM.

Companions: hqvm_cgm_genomics_1.py through _6.py, hqvm_cgm_genomics_run.py, hqvm_cgm_genomics_data_ingest.py.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import itertools
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypedDict

_REPO = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
DATA_DIR = _REPO / "data" / "catalogs" / "genomics"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))


def configure_stdio_utf8() -> None:
    """Re-wrap stdout/stderr as UTF-8 on Windows. Call from run.py, not at import."""
    if sys.platform != "win32":
        return
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        buf = getattr(stream, "buffer", None)
        if buf is not None and not isinstance(stream, io.TextIOWrapper):
            setattr(sys, stream_name, io.TextIOWrapper(buf, encoding="utf-8", errors="replace"))


from gyroscopic.hQVM.api import (
    OmegaState12,
    chirality_word6,
    omega12_to_state24,
    q_word6,
    word6_to_pairdiag12,
)
from gyroscopic.hQVM.constants import APERTURE_GAP, M_A, RHO, ab_distance, horizon_distance, unpack_state
from gyroscopic.hQVM.family import (
    byte_from_family_micro,
    byte_from_intron,
    fold_disagreement_d,
    fold_map_d,
    gf2_rank,
    intron_family_d,
    intron_from_byte,
    intron_micro_ref_d,
    phase_pairs_d,
    verify_d6_against_api,
)

CHIRALITY_D = 6
PAYLOAD_MASK = 0x3F
FOLD_MASK = 0b001100
FAMILY_MASK = 0x3
N_CODONS = 64
N_NUCLEOTIDE_ENCODINGS = 24
NULL_SEED = 20260814

# Chirality shell convention:
#   chi = u XOR v in GF(2)^6, N = wt(chi) = popcount of the six payload bits.
#   N = 0 is the equality horizon; N = 6 is the complement horizon.
# Genomics censuses use N = wt(chi), not the reversed shell (6 - wt).


def chirality_shell(chi: int) -> int:
    """Hamming shell of a six-bit chirality register (equality at 0)."""
    return (chi & PAYLOAD_MASK).bit_count()

BASES: Tuple[str, ...] = ("A", "C", "G", "T")
BASE_INDEX: Dict[str, int] = {b: i for i, b in enumerate(BASES)}

WC: Dict[str, str] = {"A": "T", "T": "A", "G": "C", "C": "G"}
TRANSITION: Dict[str, str] = {"A": "G", "G": "A", "T": "C", "C": "T"}
TRANSVERSION: Dict[str, str] = {"A": "C", "C": "A", "G": "T", "T": "G"}

PURINES = frozenset({"A", "G"})
AMINO = frozenset({"A", "C"})
STRONG = frozenset({"G", "C"})

REF_PHI: Dict[str, int] = {"A": 0b00, "G": 0b01, "T": 0b10, "C": 0b11}
REF_WC_DELTA = 0b10
REF_TS_DELTA = 0b01
REF_TV_DELTA = 0b11

GL2_MATRICES: Tuple[Tuple[int, int, int, int], ...] = (
    (1, 0, 0, 1),
    (0, 1, 1, 0),
    (1, 1, 0, 1),
    (1, 0, 1, 1),
    (0, 1, 1, 1),
    (1, 1, 1, 0),
)

CODONS: Tuple[str, ...] = tuple("".join(p) for p in itertools.product(BASES, repeat=3))
CODON_INDEX: Dict[str, int] = {c: i for i, c in enumerate(CODONS)}

AA_ORDER: Tuple[str, ...] = (
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "*",
)
AA_INDEX: Dict[str, int] = {a: i for i, a in enumerate(AA_ORDER)}
N_AA_CLASSES = len(AA_ORDER)

ORBIT_ELEMENTARY = "orbit_elementary_axes"
ORBIT_PAIR_INV = "orbit_pair_inversion"
ORBIT_OTHER = "orbit_other"


def _standard_map() -> Dict[str, str]:
    block = {
        "TT": ("F", "F", "L", "L"),
        "TC": ("S", "S", "S", "S"),
        "TA": ("Y", "Y", "*", "*"),
        "TG": ("C", "C", "*", "W"),
        "CT": ("L", "L", "L", "L"),
        "CC": ("P", "P", "P", "P"),
        "CA": ("H", "H", "Q", "Q"),
        "CG": ("R", "R", "R", "R"),
        "AT": ("I", "I", "I", "M"),
        "AC": ("T", "T", "T", "T"),
        "AA": ("N", "N", "K", "K"),
        "AG": ("S", "S", "R", "R"),
        "GT": ("V", "V", "V", "V"),
        "GC": ("A", "A", "A", "A"),
        "GA": ("D", "D", "E", "E"),
        "GG": ("G", "G", "G", "G"),
    }
    third = "TCAG"
    out: Dict[str, str] = {}
    for prefix, aas in block.items():
        for i, aa in enumerate(aas):
            out[prefix + third[i]] = aa
    return out


STANDARD_CODE: Dict[str, str] = _standard_map()

# NCBI transl_table overrides relative to table 1. Source: NCBI genetic codes.
CODE_OVERRIDES: Dict[int, Dict[str, str]] = {
    1: {},
    2: {"AGA": "*", "AGG": "*", "ATA": "M", "TGA": "W"},
    3: {"ATA": "M", "CTT": "T", "CTC": "T", "CTA": "T", "CTG": "T", "TGA": "W"},
    4: {"TGA": "W"},
    5: {"AGA": "S", "AGG": "S", "ATA": "M", "TGA": "W"},
    6: {"TAA": "Q", "TAG": "Q"},
    9: {"AAA": "N", "AGA": "S", "AGG": "S", "TGA": "W"},
    10: {"TGA": "C"},
    11: {},
    12: {"CTG": "S"},
    13: {"AGA": "G", "AGG": "G", "ATA": "M", "TGA": "W"},
    14: {"AAA": "Y", "AGA": "S", "AGG": "S", "TAA": "Y", "TGA": "W"},
    16: {"TAG": "L"},
    21: {"TGA": "W", "ATA": "M", "AGA": "S", "AGG": "S", "AAA": "N"},
    22: {"TCA": "*", "TAG": "L"},
    23: {"TTA": "*"},
    24: {"AGA": "S", "AGG": "K", "TGA": "W"},
    25: {"TGA": "G"},
    26: {"CTG": "A"},
    29: {"TAA": "Y", "TAG": "Y"},
    30: {"TAA": "E", "TAG": "E"},
    33: {"AGA": "S", "AGG": "K", "TAA": "Y", "TGA": "W"},
}

CODE_NAMES: Dict[int, str] = {
    1: "standard",
    2: "vertebrate_mito",
    3: "yeast_mito",
    4: "mold_protozoan_mito",
    5: "invertebrate_mito",
    6: "ciliate_nuclear",
    9: "echinoderm_mito",
    10: "euplotid_nuclear",
    11: "bacterial_plastid",
    12: "alt_yeast_nuclear",
    13: "ascidian_mito",
    14: "alt_flatworm_mito",
    16: "chlorophycean_mito",
    21: "trematode_mito",
    22: "scenedesmus_mito",
    23: "thraustochytrium_mito",
    24: "rhabdopleuridae_mito",
    25: "sr1_gracilibacteria",
    26: "pachysolen_nuclear",
    29: "mesodinium_nuclear",
    30: "peritrich_nuclear",
    33: "cephalodiscidae_mito",
}

NCBI_TABLE_IDS: Tuple[int, ...] = tuple(sorted(CODE_OVERRIDES))


def translation_table(ncbi_id: int) -> Dict[str, str]:
    return dict(_translation_table_cached(int(ncbi_id)))


@lru_cache(maxsize=None)
def _translation_table_cached(ncbi_id: int) -> Tuple[Tuple[str, str], ...]:
    frozen = DATA_DIR / "ncbi_genetic_codes.json"
    if frozen.exists():
        payload = json.loads(frozen.read_text(encoding="utf-8"))
        tables = payload.get("tables", {})
        key = str(ncbi_id)
        if key in tables:
            raw = tables[key]["aa"]
            if len(raw) == N_CODONS:
                return tuple((CODONS[i], raw[i]) for i in range(N_CODONS))
    if ncbi_id not in CODE_OVERRIDES:
        raise KeyError(f"NCBI genetic-code table {ncbi_id} is not loaded")
    merged = dict(STANDARD_CODE)
    merged.update(CODE_OVERRIDES[ncbi_id])
    return tuple((c, merged[c]) for c in CODONS)


def apply_gl2(matrix: Tuple[int, int, int, int], x: int) -> int:
    x0 = int(x) & 1
    x1 = (int(x) >> 1) & 1
    a, b, c, d = matrix
    y0 = (a * x0 + b * x1) & 1
    y1 = (c * x0 + d * x1) & 1
    return y0 | (y1 << 1)


def xor2(x: int, t: int) -> int:
    return (int(x) ^ int(t)) & 0x3


@dataclass(frozen=True)
class NucleotideEncoding:
    matrix: Tuple[int, int, int, int]
    translation: int
    phi: Tuple[int, int, int, int]

    @property
    def phi_map(self) -> Dict[str, int]:
        return {b: self.phi[BASE_INDEX[b]] for b in BASES}

    def encode_base(self, base: str) -> int:
        return self.phi[BASE_INDEX[base.upper()]]

    def decode_base(self, bits: int) -> str:
        x = int(bits) & 0x3
        for b in BASES:
            if self.phi[BASE_INDEX[b]] == x:
                return b
        raise ValueError(f"bits {x} not in encoding")


@dataclass(frozen=True)
class CodonState:
    bits: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bits", int(self.bits) & PAYLOAD_MASK)


@dataclass(frozen=True)
class MutationTransport:
    q6: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "q6", int(self.q6) & PAYLOAD_MASK)

    @property
    def shell(self) -> int:
        return self.q6.bit_count()

    @property
    def parity(self) -> int:
        return self.q6.bit_count() & 1


@dataclass(frozen=True)
class FamilyPhase:
    k4: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "k4", int(self.k4) & FAMILY_MASK)


@dataclass(frozen=True)
class ContextByte:
    byte: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "byte", int(self.byte) & 0xFF)


@dataclass(frozen=True)
class GenomicCarrierPair:
    ancestral: CodonState
    present: CodonState

    @property
    def transport(self) -> MutationTransport:
        return MutationTransport(self.ancestral.bits ^ self.present.bits)


def nucleotide_encoding(matrix: Tuple[int, int, int, int], translation: int) -> NucleotideEncoding:
    t = int(translation) & 0x3
    p0, p1, p2, p3 = (xor2(apply_gl2(matrix, REF_PHI[b]), t) for b in BASES)
    phi = (p0, p1, p2, p3)
    return NucleotideEncoding(matrix=matrix, translation=t, phi=phi)


_ENCODINGS: Optional[Tuple[NucleotideEncoding, ...]] = None


def all_nucleotide_encodings() -> Tuple[NucleotideEncoding, ...]:
    global _ENCODINGS
    if _ENCODINGS is not None:
        return _ENCODINGS
    out = [nucleotide_encoding(m, t) for m in GL2_MATRICES for t in range(4)]
    if len(out) != N_NUCLEOTIDE_ENCODINGS:
        raise RuntimeError(f"expected 24 encodings, got {len(out)}")
    if len({e.phi for e in out}) != N_NUCLEOTIDE_ENCODINGS:
        raise RuntimeError("affine census is not 24 distinct bijections")
    _ENCODINGS = tuple(out)
    return _ENCODINGS


def involution_delta(perm: Dict[str, str], enc: NucleotideEncoding) -> Tuple[bool, int]:
    deltas = [enc.encode_base(b) ^ enc.encode_base(perm[b]) for b in BASES]
    return all(d == deltas[0] for d in deltas), int(deltas[0]) & 0x3


def encoding_orbit_name(enc: NucleotideEncoding) -> str:
    _ok, wc = involution_delta(WC, enc)
    _ok2, ts = involution_delta(TRANSITION, enc)
    wc_wt = wc.bit_count()
    ts_wt = ts.bit_count()
    if wc == 0b11:
        return ORBIT_PAIR_INV
    if wc_wt == 1 and ts_wt == 1:
        return ORBIT_ELEMENTARY
    return ORBIT_OTHER


def encodings_in_orbit(name: str) -> Tuple[Tuple[int, NucleotideEncoding], ...]:
    return tuple((i, e) for i, e in enumerate(all_nucleotide_encodings()) if encoding_orbit_name(e) == name)


def orbit_indices(name: str) -> Tuple[int, ...]:
    return tuple(i for i, _e in encodings_in_orbit(name))


def bit0(x: int) -> int:
    return int(x) & 1


def bit1(x: int) -> int:
    return (int(x) >> 1) & 1


def bit_xor(x: int) -> int:
    return (int(x) ^ (int(x) >> 1)) & 1


LINEAR_BITS = (("b0", bit0), ("b1", bit1), ("b0xor1", bit_xor))


def pack_codon_bits(codon: str, enc: NucleotideEncoding) -> int:
    codon = codon.upper().replace("U", "T")
    if len(codon) != 3:
        raise ValueError(f"codon {codon!r} is not length 3")
    out = 0
    for i, base in enumerate(codon):
        out |= (enc.encode_base(base) & 0x3) << (2 * (2 - i))
    return out & PAYLOAD_MASK


def unpack_codon_bits(bits: int, enc: NucleotideEncoding) -> str:
    x = int(bits) & PAYLOAD_MASK
    bases = []
    for i in range(3):
        shift = 2 * (2 - i)
        bases.append(enc.decode_base((x >> shift) & 0x3))
    return "".join(bases)


def codon_state(codon: str, enc: NucleotideEncoding) -> CodonState:
    return CodonState(pack_codon_bits(codon, enc))


def mutation_q(src: str, dst: str, enc: NucleotideEncoding) -> MutationTransport:
    return MutationTransport(pack_codon_bits(src, enc) ^ pack_codon_bits(dst, enc))


def reverse_complement_codon(codon: str) -> str:
    codon = codon.upper().replace("U", "T")
    return "".join(WC[b] for b in reversed(codon))


def reverse_complement_seq(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    return "".join(WC.get(b, "N") for b in reversed(seq))


def rc_bits(bits: int, enc: NucleotideEncoding) -> int:
    return pack_codon_bits(reverse_complement_codon(unpack_codon_bits(bits, enc)), enc)


def rc_is_affine(enc: NucleotideEncoding) -> Tuple[bool, int, int]:
    b0 = rc_bits(0, enc)

    def lin(x: int) -> int:
        return rc_bits(x, enc) ^ b0

    ok = True
    for x in range(N_CODONS):
        if rc_bits(rc_bits(x, enc), enc) != x:
            ok = False
            break
    if ok:
        for i in range(6):
            e = 1 << i
            for j in range(6):
                f = 1 << j
                if lin(e ^ f) != (lin(e) ^ lin(f)):
                    ok = False
                    break
            if not ok:
                break
    return ok, gf2_rank([lin(1 << i) for i in range(6)], 6), b0


def pack_byte(family: int, payload_bits: int) -> int:
    return byte_from_family_micro(int(family) & FAMILY_MASK, int(payload_bits) & PAYLOAD_MASK, CHIRALITY_D)


def pack_bracket_byte(family: int, payload_bits: int) -> ContextByte:
    return ContextByte(pack_byte(family, payload_bits))


def unpack_byte(byte: int) -> Tuple[int, int]:
    intron = intron_from_byte(int(byte) & 0xFF, CHIRALITY_D)
    return intron_family_d(intron, CHIRALITY_D), intron_micro_ref_d(intron, CHIRALITY_D)


def pack_4mer_byte(seq4: str, enc: NucleotideEncoding) -> ContextByte:
    seq4 = seq4.upper().replace("U", "T")
    if len(seq4) != 4:
        raise ValueError(f"4-mer {seq4!r} is not length 4")
    intron = 0
    for i, base in enumerate(seq4):
        intron |= (enc.encode_base(base) & 0x3) << (2 * i)
    return ContextByte(byte_from_intron(intron, CHIRALITY_D))


def unpack_4mer_byte(byte: int, enc: NucleotideEncoding) -> str:
    intron = intron_from_byte(int(byte) & 0xFF, CHIRALITY_D)
    return "".join(enc.decode_base((intron >> (2 * i)) & 0x3) for i in range(4))


def clean_acgt(seq: str) -> str:
    return "".join(b for b in seq.upper().replace("U", "T") if b in BASES)


def genomic_byte_stream(
    seq: str,
    enc: NucleotideEncoding,
    *,
    stride: int = 1,
    frame: int = 0,
) -> Tuple[int, ...]:
    """Canonical genomic lift: overlapping 4-mers → hQVM bytes."""
    s = clean_acgt(seq)
    if stride <= 0:
        raise ValueError("stride must be positive")
    if len(s) < 4:
        return tuple()
    return tuple(
        pack_4mer_byte(s[i : i + 4], enc).byte
        for i in range(frame, len(s) - 3, stride)
    )


def reverse_complement_4mer_byte(byte: int, enc: NucleotideEncoding) -> int:
    return pack_4mer_byte(reverse_complement_seq(unpack_4mer_byte(byte, enc)), enc).byte


def verify_stream_rc_kinematic(seq: str, enc: NucleotideEncoding) -> bool:
    """RC(seq) byte stream equals reverse of RC-mapped forward 4-mer bytes (stride-1)."""
    s = clean_acgt(seq)
    if len(s) < 4:
        return True
    forward = genomic_byte_stream(s, enc, stride=1)
    rc_stream = genomic_byte_stream(reverse_complement_seq(s), enc, stride=1)
    expected = tuple(reverse_complement_4mer_byte(b, enc) for b in reversed(forward))
    return rc_stream == expected


def fold_byte(byte: int) -> int:
    return fold_map_d(int(byte) & 0xFF, CHIRALITY_D)


def fold_matches_phase_pairs(byte: int) -> bool:
    intron = intron_from_byte(int(byte) & 0xFF, CHIRALITY_D)
    folded = intron_from_byte(fold_map_d(byte, CHIRALITY_D), CHIRALITY_D)
    for i, j in phase_pairs_d(CHIRALITY_D):
        if ((intron >> i) & 1) != ((folded >> j) & 1):
            return False
        if ((intron >> j) & 1) != ((folded >> i) & 1):
            return False
    return True


def l0_of_family(family: int) -> int:
    f = int(family) & FAMILY_MASK
    return (f & 1) ^ ((f >> 1) & 1)


def predicted_q6(family: int, payload_bits: int) -> int:
    q = int(payload_bits) & PAYLOAD_MASK
    if l0_of_family(family):
        q ^= PAYLOAD_MASK
    return q


_POPCOUNT6_LUT: Tuple[int, ...] = tuple((i & PAYLOAD_MASK).bit_count() for i in range(64))


def hamming6(x: int) -> int:
    return _POPCOUNT6_LUT[int(x) & PAYLOAD_MASK]

# --- GF(2)^6 payload geometry (shared across scripts 3–5) ---

ANTIPODE_6 = 0b111111

Q_RESIDUAL_W: Tuple[int, ...] = (
    0, 1, 10, 11, 20, 21, 30, 31, 32, 33, 42, 43, 52, 53, 62, 63,
)

W_ANNIHILATOR: Tuple[int, ...] = (0b001010, 0b010100)

BLOCK_REVERSE_COLS: Tuple[int, ...] = (
    0b010000,
    0b100000,
    0b000100,
    0b001000,
    0b000001,
    0b000010,
)


def bit_reverse6(x: int) -> int:
    x = int(x) & 0x3F
    out = 0
    for i in range(6):
        if (x >> i) & 1:
            out |= 1 << (5 - i)
    return out


def block_reverse6(x: int) -> int:
    """Reverse the three 2-bit codon-position blocks (linear part of codon-RC)."""
    x = int(x) & 0x3F
    a = (x >> 4) & 0x3
    b = (x >> 2) & 0x3
    c = x & 0x3
    return ((c & 0x3) << 4) | ((b & 0x3) << 2) | (a & 0x3)


def gf2_rank6(vals: Iterable[int]) -> int:
    m = [0] * 6
    r = 0
    for v in vals:
        x = int(v) & 0x3F
        for i in range(6):
            bit = 1 << (5 - i)
            if not (x & bit):
                continue
            if m[i] == 0:
                m[i] = x
                r += 1
                break
            x ^= m[i]
    return r


def affine_rank6(vals: Sequence[int]) -> int:
    vals = [int(v) & 0x3F for v in vals]
    if not vals:
        return -1
    o = vals[0]
    return gf2_rank6(v ^ o for v in vals)


def fiber_components(group: Sequence[str]) -> Tuple[Tuple[str, ...], ...]:
    gset = set(group)
    adj = {c: [n for n in one_base_neighbors(c) if n in gset] for c in group}
    seen: set[str] = set()
    comps: List[Tuple[str, ...]] = []
    for c in group:
        if c in seen:
            continue
        stack = [c]
        seen.add(c)
        comp = [c]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
                    comp.append(v)
        comps.append(tuple(sorted(comp)))
    return tuple(comps)


def in_W(x: int) -> bool:
    x = int(x) & 0x3F
    return ((x >> 1) & 1) == ((x >> 3) & 1) and ((x >> 2) & 1) == ((x >> 4) & 1)


def one_base_neighbors(codon: str) -> Tuple[str, ...]:
    return _one_base_neighbors_cached(codon)


@lru_cache(maxsize=None)
def _one_base_neighbors_cached(codon: str) -> Tuple[str, ...]:
    out = []
    for i, b in enumerate(codon):
        for alt in BASES:
            if alt != b:
                out.append(codon[:i] + alt + codon[i + 1 :])
    return tuple(out)


def mutation_class(src: str, dst: str) -> str:
    diffs = [(a, b) for a, b in zip(src, dst) if a != b]
    if len(diffs) != 1:
        return "multi"
    a, b = diffs[0]
    if WC[a] == b:
        return "wc"
    if TRANSITION[a] == b:
        return "transition"
    if TRANSVERSION[a] == b:
        return "transversion"
    return "other"


def degeneracy_multiset(code: Dict[str, str]) -> Tuple[int, ...]:
    counts = [0] * N_AA_CLASSES
    for codon in CODONS:
        counts[AA_INDEX[code[codon]]] += 1
    return tuple(counts)


def fibers(code: Dict[str, str]) -> Dict[str, Tuple[str, ...]]:
    buckets: Dict[str, List[str]] = {a: [] for a in AA_ORDER}
    for codon in CODONS:
        buckets[code[codon]].append(codon)
    return {a: tuple(buckets[a]) for a in AA_ORDER if buckets[a]}


def prefix_box_profile(code: Dict[str, str]) -> Tuple[Tuple[str, ...], ...]:
    prefixes = [a + b for a in BASES for b in BASES]
    return tuple(tuple(sorted(code[p + t] for t in BASES)) for p in prefixes)


def tv_edge_census(code: Dict[str, str]) -> Tuple[int, int, int]:
    n_wc = n_ts = n_tv = 0
    for c in CODONS:
        for n in one_base_neighbors(c):
            cls = mutation_class(c, n)
            if cls == "wc":
                n_wc += 1
            elif cls == "transition":
                n_ts += 1
            elif cls == "transversion":
                n_tv += 1
    return n_wc // 2, n_ts // 2, n_tv // 2


def component_size_multiset(code: Dict[str, str]) -> Tuple[int, ...]:
    sizes = []
    fib = fibers(code)
    for _aa, group in fib.items():
        members = set(group)
        seen = set()
        for c in group:
            if c in seen:
                continue
            stack = [c]
            seen.add(c)
            n = 0
            while stack:
                u = stack.pop()
                n += 1
                for v in one_base_neighbors(u):
                    if v in members and v not in seen:
                        seen.add(v)
                        stack.append(v)
            sizes.append(n)
    return tuple(sorted(sizes))


def shuffle_code(code: Dict[str, str], rng: random.Random) -> Dict[str, str]:
    labels = [code[c] for c in CODONS]
    rng.shuffle(labels)
    return {c: labels[i] for i, c in enumerate(CODONS)}


def shuffle_code_boxes(code: Dict[str, str], rng: random.Random) -> Dict[str, str]:
    prefixes = [a + b for a in BASES for b in BASES]
    boxes = [[code[p + t] for t in BASES] for p in prefixes]
    rng.shuffle(boxes)
    out: Dict[str, str] = {}
    for p, box in zip(prefixes, boxes):
        labels = list(box)
        rng.shuffle(labels)
        for t, lab in zip(BASES, labels):
            out[p + t] = lab
    return out


def shuffle_preserve_component_sizes(code: Dict[str, str], rng: random.Random) -> Dict[str, str]:
    return shuffle_code_boxes(code, rng)


def shuffle_preserve_stops(code: Dict[str, str], rng: random.Random) -> Dict[str, str]:
    stops = [c for c in CODONS if code[c] == "*"]
    rest = [c for c in CODONS if code[c] != "*"]
    labels = [code[c] for c in rest]
    rng.shuffle(labels)
    out = {c: "*" for c in stops}
    for c, lab in zip(rest, labels):
        out[c] = lab
    return out


def shuffle_equal_fiber_geometry(code: Dict[str, str], rng: random.Random) -> Dict[str, str]:
    """Permute amino-acid labels within matched (fiber size, component count) buckets."""
    fib = fibers(code)
    buckets: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    atlas: Dict[str, Tuple[str, ...]] = {}
    for aa, group in fib.items():
        members = set(group)
        seen: Set[str] = set()
        n_comp = 0
        for c in group:
            if c in seen:
                continue
            n_comp += 1
            stack = [c]
            seen.add(c)
            while stack:
                u = stack.pop()
                for v in one_base_neighbors(u):
                    if v in members and v not in seen:
                        seen.add(v)
                        stack.append(v)
        key = (len(group), n_comp)
        buckets[key].append(aa)
        atlas[aa] = group
    out: Dict[str, str] = {}
    for _key, aas in buckets.items():
        perm = list(aas)
        rng.shuffle(perm)
        for src, dst in zip(aas, perm):
            for codon in atlas[src]:
                out[codon] = dst
    return out


@dataclass(frozen=True)
class NullCodePools:
    n1: Tuple[Dict[str, str], ...]
    n2: Tuple[Dict[str, str], ...]
    n3: Tuple[Dict[str, str], ...]
    n4: Tuple[Dict[str, str], ...]


def make_null_code_pools(
    code: Dict[str, str],
    rng: random.Random,
    *,
    n1: int = 200,
    n2: int = 200,
    n345: int = 80,
) -> NullCodePools:
    """Build Monte Carlo null code pools N1–N4 under standard shuffle semantics."""
    return NullCodePools(
        n1=tuple(shuffle_code(code, rng) for _ in range(n1)),
        n2=tuple(shuffle_code_boxes(code, rng) for _ in range(n2)),
        n3=tuple(shuffle_equal_fiber_geometry(code, rng) for _ in range(n345)),
        n4=tuple(shuffle_preserve_stops(code, rng) for _ in range(n345)),
    )


def monte_carlo_p(hits: int, n: int) -> float:
    return (hits + 1) / (n + 1)


def carrier_from_codon_pair(anc: CodonState, pres: CodonState) -> int:
    omega = OmegaState12(u6=anc.bits, v6=pres.bits)
    return omega12_to_state24(omega)


def chirality_of_pair(anc: CodonState, pres: CodonState) -> int:
    return chirality_word6(carrier_from_codon_pair(anc, pres))


def pairdiag_of_codon(state: CodonState) -> int:
    return word6_to_pairdiag12(state.bits)


def kernel_q6_of_byte(byte: int) -> int:
    return q_word6(int(byte) & 0xFF)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class KernelManifest(TypedDict):
    d6_api_ok: bool
    d6_api_note: str
    APERTURE_GAP: float
    RHO: float
    M_A: float
    hashes: Dict[str, Optional[str]]


def kernel_manifest() -> KernelManifest:
    ok, note = verify_d6_against_api()
    paths = [
        _REPO / "gyroscopic" / "hQVM" / "api.py",
        _REPO / "gyroscopic" / "hQVM" / "family.py",
        _REPO / "gyroscopic" / "hQVM" / "constants.py",
        _REPO / "gyroscopic" / "hQVM" / "kernel.py",
    ]
    return {
        "d6_api_ok": ok,
        "d6_api_note": note,
        "APERTURE_GAP": APERTURE_GAP,
        "RHO": RHO,
        "M_A": M_A,
        "hashes": {str(p.relative_to(_REPO)): sha256_file(p) for p in paths},
    }


class Tee:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def report_section(title: str) -> None:
    print(title)
    print("=" * 5)


def report_check(label: str, ok: bool, measured: str, expect: str) -> None:
    """One-line check: ok, title, measured, expect (all retained)."""
    status = "PASS" if ok else "FAIL"
    print(f"  {status}  {label}  |  {measured}  |  {expect}")


def report_objects(lines: Sequence[str]) -> None:
    """Short object keys only; keep derivation prose out of the report stream."""
    print("  objects")
    for line in lines:
        print(f"    {line}")
    print()


def report_checks(items: Sequence[Tuple[str, bool, str, str]]) -> None:
    print("  checks")
    print("    ok    check  |  measured  |  expect")
    for label, ok, measured, expect in items:
        status = "PASS" if ok else "FAIL"
        print(f"    {status}  {label}  |  {measured}  |  {expect}")
    print()


def print_table(
    headers: Sequence[str],
    widths: Sequence[int],
    rows: Sequence[Sequence[object]],
    *,
    aligns: Optional[Sequence[str]] = None,
    indent: str = "    ",
) -> None:
    """Fixed-width table: indent + space-separated cells. Cells may be pre-formatted strings."""
    al = list(aligns) if aligns is not None else [">"] * len(headers)
    print(indent + " ".join(f"{h:{a}{w}}" for h, w, a in zip(headers, widths, al)))
    for row in rows:
        print(indent + " ".join(f"{c:{a}{w}}" for c, w, a in zip(row, widths, al)))


def kernel_dependency() -> Tuple[bool, str]:
    return verify_d6_against_api()


# ----------------------------------------
# GenomicCompile - the multi-layer compile object
# ----------------------------------------

@dataclass(frozen=True)
class LayerReport:
    """One certified field of the compile, with its measurement values."""

    name: str
    values: Tuple[Tuple[str, float], ...]


@dataclass(frozen=True)
class GenomicCompile:
    """Multi-layer compile of one DNA interval.

    Assembles previously certified per-byte / per-codon objects into a
    single feature record for an interval or ORF:

      byte/W layer     : W-residual byte fraction and fold disagreement,
      fold poles       : payload bits 2-3 occupancy (fold coordinates),
      family sheet     : mu = per-family occupancies and L1 versus uniform 1/4,
      ORF signature    : whole-stream Omega signature (parity, tau_u, tau_v),
      depth-4 parity   : sliding-frame parity-zero fraction,
      chi shells       : mean codon-pair chirality shell N_i over frame-0 pairs,
      QuBEC order      : eta and M2 from mean shell (script 4 moment fit),
      ab / horizon     : mean ab and horizon on successive codon pairs (xi),
      boundary keys    : stop/start key presence flags (TAA/TAG/TGA/ATG).

    No classifier and no held-out accuracy product: certified fields packaged
    for application runs on sequence catalogs.
    """

    label: str
    seq_len: int
    n_bytes: int
    layers: Tuple[LayerReport, ...]

    def layer(self, name: str) -> Optional[LayerReport]:
        for lay in self.layers:
            if lay.name == name:
                return lay
        return None

    def value(self, name: str, key: str) -> Optional[float]:
        lay = self.layer(name)
        if lay is None:
            return None
        for k, v in lay.values:
            if k == key:
                return v
        return None


def _w_residual(byte: int) -> int:
    """W-membership residual used by gate 13 lineage: bits b1 XOR pairs."""
    x = byte & 0x3F
    return 1 if ((x >> 1) & 1) == ((x >> 3) & 1) and ((x >> 2) & 1) == ((x >> 4) & 1) else 0


def _qubec_from_mean_shell(mean_shell: float) -> Tuple[float, float, float, float]:
    """Moment fit from script 4: E[N]=6*rho => rho=mean/6; lambda; eta; M2."""
    if not (mean_shell == mean_shell) or mean_shell <= 0.0 or mean_shell >= 6.0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rho = mean_shell / 6.0
    if rho >= 1.0 - 1e-15:
        return float("nan"), float("nan"), float("nan"), float("nan")
    lam = rho / (1.0 - rho)
    eta = (1.0 - lam) / (1.0 + lam)
    m2 = 4096.0 / ((1.0 + eta * eta) ** 6)
    return lam, rho, eta, m2


def _interval_ab_horizon(
    codons: Sequence[str], enc: NucleotideEncoding
) -> Tuple[float, float, int]:
    """Mean ab and horizon on successive frame-0 codon pairs."""
    if len(codons) < 2:
        return float("nan"), float("nan"), 0
    sab = sh = 0.0
    n = 0
    for i in range(len(codons) - 1):
        anc = codon_state(codons[i], enc)
        pres = codon_state(codons[i + 1], enc)
        st = carrier_from_codon_pair(anc, pres)
        a12, b12 = unpack_state(st)
        sab += ab_distance(a12, b12)
        sh += horizon_distance(a12, b12)
        n += 1
    return sab / n, sh / n, n


def compile_interval(seq: str, enc: NucleotideEncoding, *, label: str = "interval") -> GenomicCompile:
    s = clean_acgt(seq)
    stream = genomic_byte_stream(s, enc)
    n = len(stream)

    # -- byte/W fold layer -------------------------------------------------
    if n:
        w_resid = sum(_w_residual(b) for b in stream)
        fd_mean = sum(fold_disagreement_d(b, CHIRALITY_D) for b in stream) / n
        wall_hist: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        for b in stream:
            wall_hist[(b & FOLD_MASK) >> 2] += 1
    else:
        w_resid, fd_mean, wall_hist = 0, 0.0, {0: 0, 1: 0, 2: 0, 3: 0}

    # -- family sheet (mu, L1 vs uniform 1/4) ------------------------------
    fam: List[int] = [0, 0, 0, 0]
    for b in stream:
        fam[intron_family_d(b, CHIRALITY_D)] += 1
    tot_fam = sum(fam) or 1
    mu = tuple(c / tot_fam for c in fam)
    l1_uniform = sum(abs(m - 0.25) for m in mu)

    # -- codon-pair chi shells (frame-0, stride-3 pairs) -------------------
    packed_codons = []
    codon_list: List[str] = []
    for i in range(0, len(s) - 2, 3):
        c3 = s[i : i + 3]
        if len(c3) == 3 and all(ch in BASE_INDEX for ch in c3):
            packed_codons.append(pack_codon_bits(c3, enc))
            codon_list.append(c3)
    chi_shells: List[int] = []
    for u, v in zip(packed_codons, packed_codons[1:]):
        chi_shells.append(chirality_shell(u ^ v))
    mean_shell = (sum(chi_shells) / len(chi_shells)) if chi_shells else float("nan")
    _lam, _rho, q_eta, q_m2 = _qubec_from_mean_shell(mean_shell)
    mean_ab, mean_hor, n_ab_pairs = _interval_ab_horizon(codon_list, enc)

    # -- ORF Omega signature + depth-4 sliding parity ----------------------
    ok = tot = 0
    par_sig = None
    if n >= 4:
        from gyroscopic.hQVM.api import omega_word_signature

        full = omega_word_signature(stream)
        par_sig = (int(full.parity), int(full.tau_u6), int(full.tau_v6))
        for i in range(0, n - 3):
            sig = omega_word_signature(stream[i : i + 4])
            tot += 1
            if sig.parity == 0:
                ok += 1

    d4_frac = (ok / tot) if tot else float("nan")

    # -- boundary keys ------------------------------------------------------
    stop_hits = {
        c: (c in s) for c in ("TAA", "TAG", "TGA", "ATG")
    }

    layers = (
        LayerReport("byte_fold_w", (
            ("n_bytes", float(n)),
            ("w_residual_frac", w_resid / n if n else float("nan")),
            ("mean_fold_disagreement", fd_mean),
        )),
        LayerReport("fold_poles", tuple(
            (f"pole_{p:02b}_frac", wall_hist[p] / n if n else float("nan"))
            for p in range(4)
        )),
        LayerReport("family_sheet", (
            ("mu_0", mu[0]),
            ("mu_1", mu[1]),
            ("mu_2", mu[2]),
            ("mu_3", mu[3]),
            ("l1_uniform", l1_uniform),
        )),
        LayerReport("omega_signature", (
            ("parity", par_sig[0] if par_sig else float("nan")),
            ("tau_u_popcount", bin(par_sig[1]).count("1") if par_sig else float("nan")),
            ("tau_v_popcount", bin(par_sig[2]).count("1") if par_sig else float("nan")),
        )),
        LayerReport("depth4_parity", (("parity_zero_frac", d4_frac),)),
        LayerReport("chi_shells", (
            ("n_pairs", float(len(chi_shells))),
            ("mean_shell", mean_shell),
        )),
        LayerReport("qubec_order", (
            ("eta", q_eta),
            ("M2", q_m2),
        )),
        LayerReport("ab_horizon", (
            ("n_pairs", float(n_ab_pairs)),
            ("mean_ab", mean_ab),
            ("mean_horizon", mean_hor),
            ("ab_plus_horizon", mean_ab + mean_hor if n_ab_pairs else float("nan")),
        )),
        LayerReport("boundary_keys", tuple(
            (f"{key}_present", float(v)) for key, v in sorted(stop_hits.items())
        )),
    )
    return GenomicCompile(
        label=label,
        seq_len=len(s),
        n_bytes=n,
        layers=layers,
    )


def print_genomic_compile(gc_obj: GenomicCompile) -> None:
    print(f"compile[{gc_obj.label}] len={gc_obj.seq_len} bytes={gc_obj.n_bytes}")
    print("-" * 5)
    for lay in gc_obj.layers:
        vals = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in lay.values)
        print(f"  {lay.name}: {vals}")


def min_mean_max(values: Sequence[float]) -> Tuple[float, float, float]:
    xs = [float(v) for v in values]
    if not xs:
        return float("nan"), float("nan"), float("nan")
    return min(xs), sum(xs) / len(xs), max(xs)


def krawtchouk(k: int, x: int, n: int = 6) -> int:
    total = 0
    for j in range(k + 1):
        if j > x or (k - j) > (n - x):
            continue
        sign = -1 if (j & 1) else 1
        total += sign * math.comb(x, j) * math.comb(n - x, k - j)
    return total


def binom(n: int, k: int) -> int:
    return math.comb(n, k)


def parse_fasta(path: Path) -> List[Tuple[str, str]]:
    recs: List[Tuple[str, str]] = []
    header = None
    chunks: List[str] = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                if header is not None:
                    recs.append((header, "".join(chunks)))
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
    if header is not None:
        recs.append((header, "".join(chunks)))
    return recs


def clean_dna(seq: str) -> str:
    return "".join(b if b in BASES else "N" for b in seq.upper().replace("U", "T"))


def gc_fraction(seq: str) -> float:
    s = clean_dna(seq)
    acgt = [b for b in s if b in BASES]
    if not acgt:
        return float("nan")
    return sum(1 for b in acgt if b in STRONG) / len(acgt)


def iter_codons(seq: str) -> List[str]:
    s = clean_dna(seq)
    out = []
    for i in range(0, len(s) - 2, 3):
        c = s[i : i + 3]
        if len(c) == 3 and all(b in BASES for b in c):
            out.append(c)
    return out


def iter_kmers(seq: str, k: int) -> List[str]:
    s = clean_dna(seq)
    out = []
    for i in range(0, len(s) - k + 1):
        w = s[i : i + k]
        if all(b in BASES for b in w):
            out.append(w)
    return out


def ols_lstsq(X: List[List[float]], y: List[float]) -> Tuple[List[float], float, int]:
    import numpy as np

    A = np.asarray(X, dtype=float)
    b = np.asarray(y, dtype=float)
    if A.ndim != 2 or A.shape[0] != b.shape[0] or A.shape[0] == 0:
        return [], float("nan"), 0
    beta, _resid, rank, _s = np.linalg.lstsq(A, b, rcond=None)
    yhat = A @ beta
    ss_res = float(((b - yhat) ** 2).sum())
    ss_tot = float(((b - b.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return [float(v) for v in beta], r2, int(rank)


def load_chr22_sequence() -> Optional[str]:
    for name in ("chr22.fa.gz", "chr22.fa", "chr22.fna.gz"):
        path = DATA_DIR / name
        if path.exists() and path.stat().st_size > 0:
            recs = parse_fasta(path)
            if recs:
                return recs[0][1].upper()
    return None


def gtf_path() -> Optional[Path]:
    for name in ("gencode.v47.annotation.gtf.gz", "gencode.annotation.gtf.gz"):
        path = DATA_DIR / name
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


@lru_cache(maxsize=2)
def extract_chr22_cds(max_tx: int = 300) -> Tuple[Tuple[str, str, Tuple[int, ...]], ...]:
    """Return (transcript_id, oriented_cds, codon_genomic_starts0). Honors GTF CDS phase."""
    seq = load_chr22_sequence()
    gtf = gtf_path()
    if seq is None or gtf is None:
        return tuple()
    by_tx: Dict[str, List[Tuple[int, int, str, int]]] = {}
    opener = gzip.open if str(gtf).endswith(".gz") else open
    with opener(gtf, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _src, kind, start, end, _sc, strand, phase, attr = parts[:9]
            if kind != "CDS" or chrom not in {"chr22", "22"}:
                continue
            tx = None
            for tok in attr.split(";"):
                tok = tok.strip()
                if tok.startswith("transcript_id") and '"' in tok:
                    tx = tok.split('"')[1]
                    break
            if tx is None:
                continue
            phase_n = 0 if phase == "." else int(phase)
            if phase_n not in (0, 1, 2):
                continue
            by_tx.setdefault(tx, []).append((int(start) - 1, int(end), strand, phase_n))
    out: List[Tuple[str, str, Tuple[int, ...]]] = []
    nseq = len(seq)
    for tx, spans in by_tx.items():
        if not spans:
            continue
        strand = spans[0][2]
        if any(s != strand for _, _, s, _ in spans):
            continue
        spans = sorted(spans, key=lambda t: t[0], reverse=(strand == "-"))
        oriented_bases: List[str] = []
        oriented_positions: List[int] = []
        for a, b, _s, phase_n in spans:
            a = max(0, a)
            b = min(nseq, b)
            if a >= b:
                continue
            if strand == "+":
                positions = list(range(a, b))
                bases = [seq[p] for p in positions]
            else:
                positions = list(range(b - 1, a - 1, -1))
                bases = [WC.get(seq[p], "N") for p in positions]
            positions = positions[phase_n:]
            bases = bases[phase_n:]
            oriented_positions.extend(positions)
            oriented_bases.extend(bases)
        cds = "".join(oriented_bases)
        usable = len(cds) - (len(cds) % 3)
        cds = cds[:usable]
        oriented_positions = oriented_positions[:usable]
        if usable < 9:
            continue
        codon_starts = tuple(oriented_positions[i] for i in range(0, usable, 3))
        out.append((tx, cds, codon_starts))
        if len(out) >= max_tx:
            break
    return tuple(out)


def assert_standard_code() -> None:
    if STANDARD_CODE["TTT"] != "F" or STANDARD_CODE["ATG"] != "M":
        raise RuntimeError("standard code anchors failed")
    if STANDARD_CODE["TAA"] != "*" or STANDARD_CODE["TGG"] != "W":
        raise RuntimeError("standard stop/W anchors failed")
    if sum(1 for c in CODONS if STANDARD_CODE[c] == "*") != 3:
        raise RuntimeError("standard code does not have 3 stops")
    if len({STANDARD_CODE[c] for c in CODONS if STANDARD_CODE[c] != "*"}) != 20:
        raise RuntimeError("standard code does not have 20 amino acids")


assert_standard_code()
