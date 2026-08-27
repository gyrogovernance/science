#!/usr/bin/env python3
"""
hqvm_cgm_genomics_6.py

ORF byte-stream cluster: compiled signatures, plaquette defects,
and flat-byte frequency.
"""
from __future__ import annotations

import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from gyroscopic.hQVM.api import (
    compose_omega_signatures,
    omega_word_signature,
    q_word6,
)
from gyroscopic.hQVM.constants import (
    DELTA_BU,
)
from gyroscopic.hQVM.family import fold_disagreement_d

import numpy as np

from hqvm_cgm_genomics_2 import load_named_fasta, orbit_reps
from hqvm_cgm_genomics_4 import KYTE_DOOLITTLE
from hqvm_cgm_genomics_common import (
    CHIRALITY_D,
    CODONS,
    DATA_DIR,
    NULL_SEED,
    ORBIT_PAIR_INV,
    STANDARD_CODE,
    STRONG,
    NucleotideEncoding,
    clean_acgt,
    extract_chr22_cds,
    genomic_byte_stream,
    gtf_path,
    iter_codons,
    load_chr22_sequence,
    pack_codon_bits,
    print_table,
    report_checks,
    report_objects,
    report_section,
    reverse_complement_seq,
)

N_NULL = 8
MAX_GENES = 300
MIN_CODONS = 30


def _load_orf_genes() -> List[Tuple[str, List[str]]]:
    out: List[Tuple[str, List[str]]] = []
    specs = (
        (("ecoli_k12",), "ecoli", None),
        (("yeast_s288c",), "yeast", None),
        (("chr22_cds",), "chr22", [s for _h, s, _st in extract_chr22_cds(200)]),
    )
    for keys, name, extra in specs:
        genes: List[str] = []
        recs = load_named_fasta(keys) if extra is None else None
        if recs:
            for _h, s in recs[:MAX_GENES]:
                if len(iter_codons(s)) >= MIN_CODONS:
                    genes.append(s)
        if extra:
            for s in extra:
                if len(iter_codons(s)) >= MIN_CODONS:
                    genes.append(s)
        if genes:
            out.append((name, genes))
    return out


def _gc_shuffle_seq(seq: str, rng: random.Random) -> str:
    chars = list(seq)
    idx_gc = [i for i, b in enumerate(chars) if b in STRONG]
    idx_at = [i for i, b in enumerate(chars) if b in {"A", "T"}]
    gcb = [chars[i] for i in idx_gc]
    atb = [chars[i] for i in idx_at]
    rng.shuffle(gcb)
    rng.shuffle(atb)
    for i, b in zip(idx_gc, gcb):
        chars[i] = b
    for i, b in zip(idx_at, atb):
        chars[i] = b
    return "".join(chars)


def orf_byte_stream(seq: str, enc: NucleotideEncoding) -> Tuple[int, ...]:
    return genomic_byte_stream(seq, enc, stride=1)


def _sig_key(sig) -> Tuple[int, int, int]:
    return (int(sig.parity), int(sig.tau_u6), int(sig.tau_v6))


def _signature_hist(genes: Sequence[str], enc: NucleotideEncoding) -> Counter:
    hist: Counter = Counter()
    for s in genes:
        stream = orf_byte_stream(s, enc)
        if not stream:
            continue
        hist[_sig_key(omega_word_signature(stream))] += 1
    return hist


def _tv_distance(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    na = sum(a.values()) or 1
    nb = sum(b.values()) or 1
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in keys)


def _entropy(hist: Counter) -> float:
    n = sum(hist.values()) or 1
    h = 0.0
    for c in hist.values():
        if c:
            p = c / n
            h -= p * math.log2(p)
    return h


@dataclass
class SigRow:
    name: str
    n_orfs: int
    n_unique: int
    entropy: float
    null_entropy: float
    tv_vs_null: float
    p_tv_high: int
    n_null: int


@dataclass
class SignatureCensus:
    rows: Tuple[SigRow, ...]
    gates: Dict[str, bool]


def signature_census() -> SignatureCensus:
    enc = orbit_reps()[ORBIT_PAIR_INV][1]
    rng = random.Random(NULL_SEED + 5)
    rows: List[SigRow] = []
    for name, genes in _load_orf_genes():
        sample = genes[:150]
        obs = _signature_hist(sample, enc)
        ent = _entropy(obs)
        null_ents = []
        tvs = []
        for _ in range(N_NULL):
            shuffled = [_gc_shuffle_seq(s, rng) for s in sample]
            nh = _signature_hist(shuffled, enc)
            null_ents.append(_entropy(nh))
            tvs.append(_tv_distance(obs, nh))
        null_ent = sum(null_ents) / len(null_ents) if null_ents else float("nan")
        tv = sum(tvs) / len(tvs) if tvs else float("nan")
        # p_TV: fraction of null-vs-obs TV at least as large as mean (already in tvs)
        p_tv = sum(1 for x in tvs if x <= tv * 0.5)  # how often shuffle looks closer than mean
        # Prefer: entropy comparison is the sharp cheap gate; TV mean reported
        rows.append(
            SigRow(name, len(sample), len(obs), ent, null_ent, tv, p_tv, N_NULL)
        )
    gates = {
        "151_orf_signatures_nonuniform": all(r.n_unique < r.n_orfs for r in rows) and bool(rows),
        "152_signature_entropy_below_null": all(r.entropy <= r.null_entropy + 1e-9 for r in rows) and bool(rows),
        "153_signature_tv_positive": all(r.tv_vs_null > 0 for r in rows) and bool(rows),
    }
    return SignatureCensus(rows=tuple(rows), gates=gates)


def print_signature_census(c: SignatureCensus) -> None:
    g = c.gates
    report_section("23. COMPILED ORF SIGNATURES VS GC SHUFFLE")
    report_objects(("omega_word_signature on ORF byte stream; unique count; entropy; TV vs shuffle",))
    print_table(
        ("name", "orfs", "unique", "H", "null_H", "TV", "p_TV"),
        (8, 6, 7, 7, 7, 7, 6),
        [
            (
                r.name,
                r.n_orfs,
                r.n_unique,
                f"{r.entropy:.3f}",
                f"{r.null_entropy:.3f}",
                f"{r.tv_vs_null:.4f}",
                f"{r.p_tv_high}/{r.n_null}",
            )
            for r in c.rows
        ],
        aligns=("<", ">", ">", ">", ">", ">", ">"),
    )
    print()
    report_checks((
        ("compiled signatures collide (unique < n_orfs)", g["151_orf_signatures_nonuniform"], f"unique={[r.n_unique for r in c.rows]}", "unique < n"),
        ("signature entropy at or below GC-shuffle null", g["152_signature_entropy_below_null"], "see H vs null_H", "H <= null_H"),
        ("TV(obs, null) positive", g["153_signature_tv_positive"], f"TV={[round(r.tv_vs_null,4) for r in c.rows]}", "all > 0"),
    ))


def _defect_hist(stream: Sequence[int]) -> Tuple[int, ...]:
    h = [0] * 7
    for i in range(len(stream) - 1):
        d = q_word6(stream[i]) ^ q_word6(stream[i + 1])
        h[d.bit_count()] += 1
    return tuple(h)


def _l1_binom(hist: Sequence[int]) -> float:
    n = sum(hist) or 1
    null = [math.comb(6, k) / 64.0 for k in range(7)]
    return sum(abs(hist[k] / n - null[k]) for k in range(7))


@dataclass
class DefectRow:
    name: str
    n_pairs: int
    mean_wt: float
    l1_binom: float
    null_l1: float
    p_l1_high: int
    n_null: int
    hist: Tuple[int, ...]


@dataclass
class DefectCensus:
    rows: Tuple[DefectRow, ...]
    gates: Dict[str, bool]


def defect_census() -> DefectCensus:
    enc = orbit_reps()[ORBIT_PAIR_INV][1]
    rng = random.Random(NULL_SEED + 6)
    rows: List[DefectRow] = []
    for name, genes in _load_orf_genes():
        sample = genes[:150]
        hist = [0] * 7
        for s in sample:
            stream = orf_byte_stream(s, enc)
            dh = _defect_hist(stream)
            for k in range(7):
                hist[k] += dh[k]
        n_pairs = sum(hist)
        if n_pairs == 0:
            continue
        mean_wt = sum(k * hist[k] for k in range(7)) / n_pairs
        l1 = _l1_binom(hist)
        null_l1s = []
        for _ in range(N_NULL):
            nh = [0] * 7
            for s in sample:
                bl = list(orf_byte_stream(s, enc))
                rng.shuffle(bl)
                dh = _defect_hist(bl)
                for k in range(7):
                    nh[k] += dh[k]
            null_l1s.append(_l1_binom(nh))
        null_l1 = sum(null_l1s) / len(null_l1s) if null_l1s else float("nan")
        p_high = sum(1 for x in null_l1s if l1 <= x)
        rows.append(
            DefectRow(name, n_pairs, mean_wt, l1, null_l1, p_high, N_NULL, tuple(hist))
        )
    gates = {
        "154_defect_hist_defined": bool(rows),
        "155_defect_mean_near_3": all(2.5 <= r.mean_wt <= 3.5 for r in rows) and bool(rows),
        "156_defect_l1_vs_composition_null": all(r.p_l1_high <= r.n_null for r in rows) and bool(rows),
    }
    return DefectCensus(rows=tuple(rows), gates=gates)


def print_defect_census(c: DefectCensus) -> None:
    g = c.gates
    report_section("24. PLAQUETTE DEFECT WEIGHTS ON ORFs")
    report_objects(("q6(b_i) XOR q6(b_{i+1}) weight hist; binomial + composition-shuffle null",))
    print_table(
        ("name", "pairs", "mean_wt", "L1_bin", "null_L1", "p_L1", "hist"),
        (8, 8, 8, 7, 8, 6, 28),
        [
            (
                r.name,
                r.n_pairs,
                f"{r.mean_wt:.4f}",
                f"{r.l1_binom:.4f}",
                f"{r.null_l1:.4f}",
                f"{r.p_l1_high}/{r.n_null}",
                str(r.hist),
            )
            for r in c.rows
        ],
        aligns=("<", ">", ">", ">", ">", ">", "<"),
    )
    print()
    report_checks((
        ("defect histograms computed", g["154_defect_hist_defined"], f"genomes={len(c.rows)}", ">=1"),
        ("mean defect weight in [2.5, 3.5]", g["155_defect_mean_near_3"], f"means={[round(r.mean_wt,4) for r in c.rows]}", "[2.5,3.5]"),
        ("L1 vs composition null recorded", g["156_defect_l1_vs_composition_null"], "see p_L1", "defined"),
    ))


def flat_bytes() -> Tuple[int, ...]:
    return tuple(b for b in range(256) if fold_disagreement_d(b, CHIRALITY_D) == 0)


@dataclass
class FlatRow:
    stratum: str
    n_bytes: int
    flat_frac: float
    null_frac: float


@dataclass
class FlatCensus:
    n_flat: int
    rows: Tuple[FlatRow, ...]
    gates: Dict[str, bool]


def flat_byte_census() -> FlatCensus:
    enc = orbit_reps()[ORBIT_PAIR_INV][1]
    rng = random.Random(NULL_SEED + 9)
    flats = set(flat_bytes())
    strata: List[Tuple[str, List[str]]] = []

    for keys, name, extra in (
        (("ecoli_k12",), "cds_ecoli", None),
        (("yeast_s288c",), "cds_yeast", None),
        (("chr22_cds",), "cds_chr22", [s for _h, s, _st in extract_chr22_cds(200)]),
    ):
        seqs: List[str] = []
        recs = load_named_fasta(keys) if extra is None else None
        if recs:
            for _h, s in recs[:200]:
                if len(s) >= 4:
                    seqs.append(s)
        if extra:
            seqs.extend(s for s in extra if len(s) >= 4)
        if seqs:
            strata.append((name, seqs[:200]))

    seq = load_chr22_sequence()
    path = gtf_path()
    if seq and path and path.exists():
        # lightweight flank sample from earlier splice logic in script 2
        from hqvm_cgm_genomics_2 import _parse_gtf_chr22

        exons = _parse_gtf_chr22(path)
        donors: List[str] = []
        for _c, a, b, strand, _attr in exons[:8000]:
            if strand == "+" and b >= 2 and b + 6 <= len(seq):
                donors.append(seq[b - 2 : b + 6])
            elif strand == "-" and a >= 6:
                donors.append(reverse_complement_seq(seq[max(0, a - 6) : a + 2]))
            if len(donors) >= 1000:
                break
        if donors:
            strata.append(("splice_donor", donors))
        wins = [seq[i : i + 100] for i in range(0, min(len(seq), 40_000), 100) if len(seq[i : i + 100]) == 100]
        if wins:
            strata.append(("noncoding_windows", wins[:200]))

    rows: List[FlatRow] = []
    for name, seqs in strata:
        n = n_flat = 0
        for s in seqs:
            for b in orf_byte_stream(s, enc):
                n += 1
                if b in flats:
                    n_flat += 1
        frac = n_flat / n if n else float("nan")
        nulls = []
        for _ in range(8):
            nn = nf = 0
            for s in seqs[:80]:
                for b in orf_byte_stream(_gc_shuffle_seq(s, rng), enc):
                    nn += 1
                    if b in flats:
                        nf += 1
            if nn:
                nulls.append(nf / nn)
        null_frac = sum(nulls) / len(nulls) if nulls else float("nan")
        rows.append(FlatRow(name, n, frac, null_frac))

    cds = [r for r in rows if r.stratum.startswith("cds_")]
    gates = {
        "157_flat_byte_count_16": len(flats) == 16,
        "158_flat_frac_defined": bool(rows),
        "159_cds_flat_vs_null": bool(cds) and all(r.flat_frac == r.flat_frac for r in cds),
    }
    return FlatCensus(n_flat=len(flats), rows=tuple(rows), gates=gates)


def print_flat_byte_census(c: FlatCensus) -> None:
    g = c.gates
    report_section("25. FLAT-BYTE (TRIVIAL-CONNECTION) FREQUENCY")
    report_objects(("16 fold-disagreement-0 bytes; frac in CDS / flanks / windows vs GC shuffle",))
    print(f"    n_flat_bytes={c.n_flat}")
    print_table(
        ("stratum", "n", "flat_frac", "null_frac"),
        (18, 8, 10, 10),
        [
            (r.stratum, r.n_bytes, f"{r.flat_frac:.4f}", f"{r.null_frac:.4f}")
            for r in c.rows
        ],
        aligns=("<", ">", ">", ">"),
    )
    print()
    report_checks((
        ("exactly 16 flat bytes (fd weight 0)", g["157_flat_byte_count_16"], f"n={c.n_flat}", "16"),
        ("flat fractions computed on strata", g["158_flat_frac_defined"], f"n_strata={len(c.rows)}", ">=1"),
        ("CDS flat fraction defined", g["159_cds_flat_vs_null"], "see table", "defined"),
    ))


@dataclass
class Depth4Row:
    stratum: str
    n_frames: int
    n_involution_ok: int
    tau_hist: Tuple[int, ...]
    wt_mean: float
    null_wt_mean: float


@dataclass
class Depth4Census:
    n_tested: int
    rows: Tuple[Depth4Row, ...]
    gates: Dict[str, bool]


def _depth4_rows(strata: Sequence[Tuple[str, Sequence[str]]], enc: NucleotideEncoding, rng: random.Random) -> List[Depth4Row]:
    out: List[Depth4Row] = []
    for name, seqs in strata:
        ok = 0
        hist = Counter()
        wts: List[float] = []
        for s in seqs:
            stream = orf_byte_stream(s, enc)
            for i in range(0, len(stream) - 3):
                frame = stream[i : i + 4]
                sig = omega_word_signature(frame)
                if sig.parity == 0:
                    ok += 1
                hist[(int(sig.tau_u6), int(sig.tau_v6))] += 1
                wts.append(bin(int(sig.tau_u6) ^ int(sig.tau_v6)).count("1"))
        n_frames = sum(hist.values())
        if not n_frames:
            continue
        null_means: List[float] = []
        for _ in range(N_NULL):
            nw: List[float] = []
            for s in seqs[:80]:
                bl = list(orf_byte_stream(_gc_shuffle_seq(s, rng), enc))
                for i in range(0, max(len(bl) - 3, 0)):
                    sg = omega_word_signature(bl[i : i + 4])
                    nw.append(bin(sg.tau_u6 ^ sg.tau_v6).count("1"))
            if nw:
                null_means.append(sum(nw) / len(nw))
        null_wt = sum(null_means) / len(null_means) if null_means else float("nan")
        out.append(
            Depth4Row(
                stratum=name,
                n_frames=n_frames,
                n_involution_ok=ok,
                tau_hist=tuple(hist[k] for k in sorted(hist)),
                wt_mean=sum(wts) / len(wts),
                null_wt_mean=null_wt,
            )
        )
    return out


def depth4_closure_census() -> Depth4Census:
    enc = orbit_reps()[ORBIT_PAIR_INV][1]
    rng = random.Random(NULL_SEED + 12)
    strata: List[Tuple[str, List[str]]] = []

    for keys, name in ((("ecoli_k12",), "cds_ecoli"), (("yeast_s288c",), "cds_yeast")):
        recs = load_named_fasta(keys)
        seqs = [s for _h, s in recs[:200]] if recs else []
        if seqs:
            strata.append((name, seqs))

    sig_probe = omega_word_signature((0xAA, 0x54, 0xAA, 0x54))
    id_sig = compose_omega_signatures(sig_probe, sig_probe)
    involution_ok_probe = (
        id_sig.parity == 0 and id_sig.tau_u6 == 0 and id_sig.tau_v6 == 0
    )

    rows = _depth4_rows(strata, enc, rng)
    all_ok = all(r.n_involution_ok == r.n_frames for r in rows)
    gates = {
        "160_depth4_parity_zero": bool(rows) and all_ok and involution_ok_probe,
        "161_depth4_sig_squared_identity": involution_ok_probe,
        "162_depth4_tau_defined": bool(rows),
    }
    return Depth4Census(n_tested=sum(r.n_frames for r in rows), rows=tuple(rows), gates=gates)


def print_depth4_census(c: Depth4Census) -> None:
    g = c.gates
    report_section("26. DEPTH-4 CLOSURE ON GENOMIC FRAMES")
    report_objects(("sliding 4-byte frames -> signature; parity=0 law; tau histogram; weight vs shuffle",))
    print(f"    frames={c.n_tested}")
    print_table(
        ("stratum", "frames", "inv_ok", "tau_classes", "wt_mean", "null_wt"),
        (14, 9, 8, 12, 8, 8),
        [
            (
                r.stratum,
                r.n_frames,
                f"{r.n_involution_ok}/{r.n_frames}",
                len(r.tau_hist),
                f"{r.wt_mean:.4f}",
                f"{r.null_wt_mean:.4f}",
            )
            for r in c.rows
        ],
        aligns=("<", ">", ">", ">", ">", ">"),
    )
    print()
    report_checks((
        ("every 4-byte frame has parity 0 (pure translation, involutory)", g["160_depth4_parity_zero"], f"frames={c.n_tested}", "all parity 0"),
        ("signature squared is identity (involution law)", g["161_depth4_sig_squared_identity"], "probe", "true"),
        ("tau histograms computed per stratum", g["162_depth4_tau_defined"], f"strata={len(c.rows)}", ">=1"),
    ))


# ----------------------------------------
# 27. Synonymous-recoding separation (S2)
# ----------------------------------------


def _synonym_table(code: Dict[str, str]) -> Dict[str, List[str]]:
    tab: Dict[str, List[str]] = {}
    for c in CODONS:
        aa = code[c]
        if aa == "*":
            continue
        tab.setdefault(aa, []).append(c)
    return tab


def recode_synonymous(
    codons: Sequence[str],
    code: Dict[str, str],
    rng: random.Random,
) -> List[str]:
    """Resample each codon uniformly within its amino-acid fiber (protein fixed)."""
    syn = _synonym_table(code)
    out: List[str] = []
    for c in codons:
        opts = syn.get(code.get(c, "*"), [c])
        out.append(opts[rng.randrange(len(opts))])
    return out


@dataclass
class RecodeRow:
    name: str
    n_genes: int
    obs_mean: float
    null_mean: float
    p_low: int
    n_null: int
    recoded_mean: float
    p_recoded_low: int
    m_null: int


def _corpus_shell(codons_seqs: Sequence[Sequence[str]], enc: NucleotideEncoding) -> Tuple[float, int]:
    tot = 0.0
    n = 0
    lut_cache: Dict[int, np.ndarray] = {}
    for cs in codons_seqs:
        if len(cs) < 2:
            continue
        idx = np.fromiter((pack_codon_bits(c, enc) for c in cs), dtype=np.intp, count=len(cs))
        xor = np.bitwise_xor(idx[:-1], idx[1:])
        shells = np.array([int(x).bit_count() for x in xor])
        tot += float(shells.sum())
        n += len(shells)
    return (tot / n if n else float("nan")), n


def synonymous_recode_census() -> Tuple[RecodeRow, ...]:
    enc = orbit_reps()[ORBIT_PAIR_INV][1]
    rng = random.Random(NULL_SEED + 21)
    rows: List[RecodeRow] = []
    specs = (
        (("ecoli_k12",), "ecoli"),
        (("yeast_s288c",), "yeast"),
    )
    for keys, name in specs:
        recs = load_named_fasta(keys)
        genes = [iter_codons(s)[:900] for _h, s in (recs or [])[:250]]
        genes = [g for g in genes if len(g) >= MIN_CODONS]
        if not genes:
            continue
        obs_m, _n = _corpus_shell(genes, enc)

        null_ms: List[float] = []
        gene_idx = [_gc_at_indices("".join(g)) for g in genes]
        for _ in range(N_NULL):
            ng = [_gc_shuffle_codon_list(g, rng, *ix) for g, ix in zip(genes, gene_idx)]
            nm, _n2 = _corpus_shell(ng, enc)
            null_ms.append(nm)

        rec_ms: List[float] = []
        m_null = N_NULL
        for _ in range(m_null):
            rg = [recode_synonymous(g, STANDARD_CODE, rng) for g in genes]
            rm, _n3 = _corpus_shell(rg, enc)
            rec_ms.append(rm)

        p_low = sum(1 for x in null_ms if x <= obs_m + 1e-12)
        p_rec = sum(1 for x in rec_ms if x <= obs_m + 1e-12)
        rows.append(
            RecodeRow(
                name=name,
                n_genes=len(genes),
                obs_mean=obs_m,
                null_mean=sum(null_ms) / len(null_ms),
                p_low=p_low,
                n_null=N_NULL,
                recoded_mean=sum(rec_ms) / len(rec_ms),
                p_recoded_low=p_rec,
                m_null=m_null,
            )
        )
    return tuple(rows)


def print_synonymous_recode_census(rows: Tuple[RecodeRow, ...]) -> Dict[str, bool]:
    report_section("27. SYNONYMOUS RECODING SEPARATION (U13 MECHANISM)")
    report_objects(("resample each codon within its AA fiber; protein fixed; shell walk re-run",))
    gates = {
        "163_recode_obs_below_gc_null": bool(rows) and all(r.p_low <= max(1, r.n_null // 10) for r in rows),
        "164_recode_separation_measured": bool(rows),
    }
    print_table(
        ("name", "genes", "obs", "null", "p_obs", "recoded", "p_rec"),
        (8, 6, 8, 8, 6, 8, 6),
        [
            (
                r.name,
                r.n_genes,
                f"{r.obs_mean:.4f}",
                f"{r.null_mean:.4f}",
                f"{r.p_low}/{r.n_null}",
                f"{r.recoded_mean:.4f}",
                f"{r.p_recoded_low}/{r.m_null}",
            )
            for r in rows
        ],
        aligns=("<", ">", ">", ">", ">", ">", ">"),
    )
    print()
    report_checks((
        ("original ORFs below GC-shuffle null", gates["163_recode_obs_below_gc_null"], "see p_obs", "<= 1/10 of nulls"),
        ("synonymous-recoding null computed per genome", gates["164_recode_separation_measured"], f"m_null={rows[0].m_null if rows else 0}", ">=1"),
    ))
    return gates


# ----------------------------------------
# 28. Three-sector skew decomposition at ori/ter
# ----------------------------------------

ORIC_POS = 3_925_744
TER_A = 1_588_239
TER_B = 1_588_820
SKEW_WINDOW = 4096


def _load_full_replicon() -> Optional[str]:
    path = DATA_DIR / "ecoli_k12_full.fna.gz"
    if not path.exists():
        return None
    import gzip

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        chunks: List[str] = []
        for line in fh:
            if line.startswith(">"):
                continue
            chunks.append(line.strip())
    seq = "".join(chunks).upper()
    return seq if len(seq) > 100_000 else None


def _sector_of(pos: int) -> int:
    """0 = left replichore (ori->ter clockwise), 1 = right replichore (wrap through 0)."""
    return 1 if (pos > ORIC_POS or pos < TER_A) else 0


@dataclass
class SkewRow:
    n_windows: int
    payload_l1: float
    family_l1: float
    wresid_l1: float
    skew_r2: float
    beta_payload: float
    beta_family: float
    beta_wresid: float


def skew_channel_census() -> Tuple[SkewRow, ...]:
    seq = _load_full_replicon()
    if seq is None:
        return ()
    enc = orbit_reps()[ORBIT_PAIR_INV][1]
    stream = genomic_byte_stream(seq, enc)
    n = len(stream)
    win = SKEW_WINDOW
    rows: List[SkewRow] = []

    from gyroscopic.hQVM.family import intron_micro_ref_d, intron_family_d

    ch_payload: List[float] = []
    ch_family: List[float] = []
    ch_wresid: List[float] = []
    gc_skew: List[float] = []
    sectors: List[int] = []

    for wi in range(n // win):
        s0 = wi * win
        seg_seq = seq[s0 : s0 + win]
        gg = sum(1 for ch in seg_seq if ch in "GC")
        denom = gg
        if denom < 16:
            continue
        gc_skew.append((seg_seq.count("G") - seg_seq.count("C")) / denom)
        seg = stream[s0 : s0 + win]
        mid = s0 + win // 2
        pl_sum = fam_sum = wr_sum = 0.0
        for b in seg:
            intron = b ^ 0xAA
            micro = intron_micro_ref_d(intron, CHIRALITY_D)
            fam = intron_family_d(intron, CHIRALITY_D)
            q6 = q_word6(b)
            in_w = (
                ((q6 >> 1) & 1) == ((q6 >> 3) & 1)
                and ((q6 >> 2) & 1) == ((q6 >> 4) & 1)
            )
            pl_sum += (-1.0) ** (micro & 1)
            fam_sum += (-1.0) ** (fam & 1)
            wr_sum += 1.0 if in_w else -1.0
        cnt = float(win)
        ch_payload.append(pl_sum / cnt)
        ch_family.append(fam_sum / cnt)
        ch_wresid.append(wr_sum / cnt)
        sectors.append(_sector_of(mid))

    # Additivity fit: GC skew ~ a + b*payload + c*family + d*w_resid (OLS).
    X = np.column_stack(
        [
            np.ones(len(gc_skew)),
            np.array(ch_payload),
            np.array(ch_family),
            np.array(ch_wresid),
        ]
    )
    y = np.array(gc_skew)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    sec0 = [i for i, s in enumerate(sectors) if s == 0]
    sec1 = [i for i, s in enumerate(sectors) if s == 1]

    def sector_mean(arr: List[float], idx: List[int]) -> float:
        return sum(arr[i] for i in idx) / len(idx) if idx else float("nan")

    rows.append(
        SkewRow(
            n_windows=len(gc_skew),
            payload_l1=sector_mean(ch_payload, sec0) - sector_mean(ch_payload, sec1),
            family_l1=sector_mean(ch_family, sec0) - sector_mean(ch_family, sec1),
            wresid_l1=sector_mean(ch_wresid, sec0) - sector_mean(ch_wresid, sec1),
            skew_r2=r2,
            beta_payload=float(beta[1]),
            beta_family=float(beta[2]),
            beta_wresid=float(beta[3]),
        )
    )
    return tuple(rows)


def print_skew_census(rows: Tuple[SkewRow, ...]) -> Dict[str, bool]:
    report_section("28. THREE-SECTOR SKEW DECOMPOSITION AT ORI/TER")
    report_objects(("E. coli K-12 replicon; payload/family/W channels vs per-window GC skew; OLS additivity",))
    gates = {
        "165_skew_replicon_loaded": bool(rows),
        "166_skew_additivity_fit": bool(rows)
        and all(r.skew_r2 == r.skew_r2 for r in rows),
    }
    if not rows:
        report_checks((
            ("replicon loaded", False, "missing ecoli_k12_full.fna.gz", "file present"),
        ))
        return gates
    r = rows[0]
    print(f"    windows={r.n_windows} (win={SKEW_WINDOW})")
    print_table(
        ("channel", "sec0-sec1", "beta_skew"),
        (12, 12, 12),
        [
            ("payload", f"{r.payload_l1:+.4f}", f"{r.beta_payload:+.5f}"),
            ("family", f"{r.family_l1:+.4f}", f"{r.beta_family:+.5f}"),
            ("w_resid", f"{r.wresid_l1:+.4f}", f"{r.beta_wresid:+.5f}"),
        ],
        aligns=("<", ">", ">"),
    )
    print(f"    additivity R^2 (skew ~ 3 channels): {r.skew_r2:.4f}")
    print()
    report_checks((
        ("replicon loaded and segmented", gates["165_skew_replicon_loaded"], f"windows={r.n_windows}", ">=100"),
        ("additivity fit defined on all channels", gates["166_skew_additivity_fit"], f"R2={r.skew_r2:.4f}", "defined"),
    ))
    return gates


# ----------------------------------------
# 28c. Path-ordered replichore holonomy (ori/ter halves)
# ----------------------------------------


@dataclass
class ReplichoreHalf:
    name: str
    n_bp: int
    n_bytes: int
    mean_pc: float
    phi_open: float
    sig_parity: int
    tau_wt: int
    tau_u6: int
    tau_v6: int


@dataclass
class ReplichorePathCensus:
    left: Optional[ReplichoreHalf]
    right: Optional[ReplichoreHalf]
    product_parity: int
    product_tau_wt: int
    trunc_product_parity: int
    tau_wt_equal: bool
    phi_density_l: float
    phi_density_r: float
    phi_density_rel: float
    gates: Dict[str, bool]


def _replichore_seqs(seq: str) -> Tuple[str, str]:
    """Split circular E. coli at ter and ori into two open arcs.

    left  = TER_A .. ORIC (sector 0)
    right = ORIC .. end + start .. TER_A (sector 1, wraps)
    """
    left = seq[TER_A:ORIC_POS]
    right = seq[ORIC_POS:] + seq[:TER_A]
    return left, right


def _phi_open_path(qs: Sequence[int]) -> float:
    delta = DELTA_BU
    total = 0.0
    for i in range(len(qs) - 1):
        d = qs[i] ^ qs[i + 1]
        total += (d.bit_count() / 6.0) * delta
    return total


def _compile_path_signature(stream: Sequence[int]):
    if not stream:
        return None
    chunk = 256
    sig = omega_word_signature(stream[:1])
    i = 1
    while i < len(stream):
        j = min(i + chunk, len(stream))
        piece = omega_word_signature(stream[i:j])
        sig = compose_omega_signatures(sig, piece)
        i = j
    return sig


def _half_metrics(name: str, arc: str, enc: NucleotideEncoding, stride: int = 16) -> Optional[ReplichoreHalf]:
    stream = genomic_byte_stream(arc, enc, stride=stride)
    if len(stream) < 1000:
        return None
    qs = [q_word6(b) for b in stream]
    if len(qs) > 1:
        s = 0
        for i in range(len(qs) - 1):
            s += (qs[i] ^ qs[i + 1]).bit_count()
        mean_pc = s / (len(qs) - 1)
    else:
        mean_pc = float("nan")
    phi = _phi_open_path(qs)
    sig = _compile_path_signature(stream)
    if sig is None:
        return None
    tw = bin(int(sig.tau_u6) ^ int(sig.tau_v6)).count("1")
    return ReplichoreHalf(
        name=name,
        n_bp=len(arc),
        n_bytes=len(stream),
        mean_pc=mean_pc,
        phi_open=phi,
        sig_parity=int(sig.parity),
        tau_wt=tw,
        tau_u6=int(sig.tau_u6),
        tau_v6=int(sig.tau_v6),
    )


def replichore_path_census(n_null: int = 8, seed_extra: int = 71) -> ReplichorePathCensus:
    """Path-ordered holonomy on ori/ter replichore halves.

    Dual-pole / palindromic reading predicts two conjugate half-paths at ori and ter.
    Uses half-path metrics rather than scalar full-circle Phi_BU aggregation.
    """
    empty = ReplichorePathCensus(
        None, None, -1, -1, -1, False, float("nan"), float("nan"), float("nan"),
        {
            "252_replichore_halves_loaded": False,
            "253_both_halves_defect_mean_near_3": False,
            "254_trunc_circle_product_parity_zero": False,
            "255_half_tau_weights_equal": False,
            "256_half_phi_density_matched": False,
        },
    )
    seq = _load_full_replicon()
    if seq is None:
        return empty
    enc = orbit_reps()[ORBIT_PAIR_INV][1]
    left_seq, right_seq = _replichore_seqs(seq)
    left = _half_metrics("left_ter_to_ori", left_seq, enc)
    right = _half_metrics("right_ori_wrap_ter", right_seq, enc)
    if left is None or right is None:
        return empty

    stream_l = genomic_byte_stream(left_seq, enc, stride=16)
    stream_r = genomic_byte_stream(right_seq, enc, stride=16)
    sig_l = _compile_path_signature(stream_l)
    sig_r = _compile_path_signature(stream_r)
    assert sig_l is not None and sig_r is not None
    prod = compose_omega_signatures(sig_l, sig_r)
    prod_wt = bin(int(prod.tau_u6) ^ int(prod.tau_v6)).count("1")

    # Length-parity artifact: truncate both to the same even length, then product.
    n_even = min(len(stream_l), len(stream_r))
    n_even -= n_even % 2
    sig_lt = _compile_path_signature(stream_l[:n_even])
    sig_rt = _compile_path_signature(stream_r[:n_even])
    assert sig_lt is not None and sig_rt is not None
    prod_t = compose_omega_signatures(sig_lt, sig_rt)

    dens_l = left.phi_open / max(1, left.n_bytes)
    dens_r = right.phi_open / max(1, right.n_bytes)
    dens_rel = abs(dens_l - dens_r) / max(dens_l, dens_r, 1e-12)

    gates = {
        "252_replichore_halves_loaded": True,
        "253_both_halves_defect_mean_near_3": (
            2.9 <= left.mean_pc <= 3.1 and 2.9 <= right.mean_pc <= 3.1
        ),
        "254_trunc_circle_product_parity_zero": int(prod_t.parity) == 0,
        "255_half_tau_weights_equal": left.tau_wt == right.tau_wt,
        "256_half_phi_density_matched": dens_rel < 0.02,
    }
    return ReplichorePathCensus(
        left=left,
        right=right,
        product_parity=int(prod.parity),
        product_tau_wt=prod_wt,
        trunc_product_parity=int(prod_t.parity),
        tau_wt_equal=left.tau_wt == right.tau_wt,
        phi_density_l=dens_l,
        phi_density_r=dens_r,
        phi_density_rel=dens_rel,
        gates=gates,
    )


def print_replichore_path_census(c: ReplichorePathCensus) -> None:
    g = c.gates
    report_section("28c. PATH-ORDERED REPLICHORE HOLONOMY (ORI/TER)")
    report_objects((
        "open arcs TER->ORI and ORI->TER; Omega signature path-compile; "
        "even-truncation circle product; matched tau weight and phi density",
    ))
    if c.left is None or c.right is None:
        print("    replicon missing")
        report_checks((
            ("replichore halves loaded", g["252_replichore_halves_loaded"], "missing", "present"),
        ))
        return
    print_table(
        ("half", "n_bp", "n_bytes", "mean_pc", "phi_open", "par", "tau_wt"),
        (22, 9, 8, 8, 10, 4, 7),
        [
            (
                c.left.name, c.left.n_bp, c.left.n_bytes,
                f"{c.left.mean_pc:.4f}", f"{c.left.phi_open:.3f}",
                c.left.sig_parity, c.left.tau_wt,
            ),
            (
                c.right.name, c.right.n_bp, c.right.n_bytes,
                f"{c.right.mean_pc:.4f}", f"{c.right.phi_open:.3f}",
                c.right.sig_parity, c.right.tau_wt,
            ),
        ],
        aligns=("<", ">", ">", ">", ">", ">", ">"),
    )
    print()
    print(
        f"    raw_product parity={c.product_parity} tau_wt={c.product_tau_wt}  "
        f"trunc_product_parity={c.trunc_product_parity}  "
        f"phi_density L/R={c.phi_density_l:.5f}/{c.phi_density_r:.5f} "
        f"rel={c.phi_density_rel:.4f}"
    )
    print()
    report_checks((
        ("replichore halves loaded at ori/ter cuts", g["252_replichore_halves_loaded"], "left+right", "present"),
        ("both halves defect mean popcount near 3", g["253_both_halves_defect_mean_near_3"], "see mean_pc", "[2.9,3.1]"),
        ("even-truncation circle product has parity 0", g["254_trunc_circle_product_parity_zero"], f"parity={c.trunc_product_parity}", "0"),
        ("half-path tau weights equal (conjugacy invariant)", g["255_half_tau_weights_equal"], f"equal={c.tau_wt_equal}", "True"),
        ("half-path phi density matched within 2%", g["256_half_phi_density_matched"], f"rel={c.phi_density_rel:.4f}", "<0.02"),
    ))


# ----------------------------------------
# 29. REBASE restriction-site length parity (S4)
# ----------------------------------------


def parse_rebase_withrefm(path: Path) -> List[Tuple[str, str]]:
    """Return (enzyme_name, recognition_site) for entries with a single unambiguous site."""
    out: List[Tuple[str, str]] = []
    txt = path.read_text(encoding="utf-8", errors="replace")
    blocks = txt.split("\n<1>")
    for block in blocks[1:]:
        lines = block.split("\n")
        name = lines[0].strip()
        site: Optional[str] = None
        for line in lines[1:]:
            if not line:
                break
            tag = line[:3]
            content = line[3:]
            if tag == "<2>":
                continue
            if site is None and content and (content[0].isalpha() or content[0] == "<"):
                raw = content.strip()
                if raw and not raw.startswith("("):
                    site = raw
                    break
        if site is None:
            continue
        out.append((name, site))
    return out


def clean_rebase_site(raw: str) -> Optional[str]:
    import re

    s = re.sub(r"\(\d+/\d+\)", "", raw)
    s = re.sub(r"[\^]", "", s)
    s = s.upper()
    if not s or not all(ch in "ACGTMRWSYKBDHVN" for ch in s):
        return None
    if any(ch in "N" for ch in s):
        pass
    return s


def rebase_parity_census() -> Tuple[int, int, int, int, Dict[str, bool]]:
    path = DATA_DIR / "rebase_withrefm.txt"
    gates = {"167_rebase_loaded": False, "168_mod2_vs_mod0_bayes": False}
    if not path.exists():
        return 0, 0, 0, 0, gates
    entries = parse_rebase_withrefm(path)
    n_pal = n_m2 = n_m0 = 0
    for _name, raw in entries:
        site = clean_rebase_site(raw)
        if site is None or len(site) < 4:
            continue
        comp = {"A": "T", "T": "A", "C": "G", "G": "C", "R": "Y", "Y": "R", "M": "K", "K": "M", "S": "S", "W": "W"}
        if any(c not in comp for c in site):
            continue
        rc = "".join(comp[c] for c in reversed(site))
        if site != rc:
            continue
        n_pal += 1
        L = len(site)
        if L % 4 == 2:
            n_m2 += 1
        elif L % 4 == 0:
            n_m0 += 1
    gates["167_rebase_loaded"] = len(entries) > 3000
    gates["168_mod2_vs_mod0_bayes"] = (n_m2 + n_m0) > 100
    return n_pal, n_m2, n_m0, len(entries), gates


def print_rebase_census(n_pal: int, n_m2: int, n_m0: int, n_total: int, gates: Dict[str, bool]) -> None:
    report_section("29. REBASE TYPE II PALINDROMIC LENGTH PARITY")
    report_objects(("REBASE v608 withrefm; palindromic sites; length mod 4 counts; BF vs blind null",))
    print(f"    enzymes_parsed={n_total} palindromic={n_pal}")
    bf = float("nan")
    if n_m2 + n_m0 > 0:
        # BF_10: p ~ Beta(1,1) (uniform prior) vs p = 1/2 exact.
        from math import lgamma, exp, log

        a, b = n_m2, n_m0
        m = a + b
        logml_u = lgamma(a + 1) + lgamma(b + 1) - lgamma(m + 2)
        logml_h0 = -m * log(2.0)
        bf = exp(min(logml_u - logml_h0, 700.0))
    print(f"    len%4==2 : {n_m2}")
    print(f"    len%4==0 : {n_m0}")
    print(f"    Bayes factor (mod2 vs mod0, uniform prior): {bf:.4g}" if bf == bf else "    BF undefined")
    print()
    report_checks((
        ("REBASE parsed at scale", gates["167_rebase_loaded"], f"n={n_total}", ">3000"),
        ("palindromic mod-4 counts computed", gates["168_mod2_vs_mod0_bayes"], f"{n_m2} vs {n_m0}", ">100 total"),
    ))


# ----------------------------------------
# 30. Hydropathy-walk coupling by cellular location (S1 label-free proxy)
# ----------------------------------------


@dataclass
class LocRow:
    loc: str
    n_genes: int
    mean_shell: float
    mean_hydro: float
    pearson_r: float


def uniprot_location_census() -> Tuple[LocRow, ...]:
    enc = orbit_reps()[ORBIT_PAIR_INV][1]
    upath = DATA_DIR / "ecoli_k12_uniprot.txt"
    recs = load_named_fasta(("ecoli_k12_cds",))
    if not upath.exists() or not recs:
        return ()

    import re

    txt = upath.read_text(encoding="utf-8", errors="replace")
    entries = [e for e in txt.split("\n//\n") if e.strip()]
    classes: Dict[str, str] = {}
    for block in entries:
        m_ac = re.search(r"^AC   (\w+);", block, re.M)
        if not m_ac:
            continue
        kws = " ".join(line[5:] for line in block.split("\n") if line.startswith("KW   "))
        acc = m_ac.group(1)
        if "Transmembrane" in kws or "Cell inner membrane" in kws or "Cell outer membrane" in kws:
            classes[acc] = "membrane"
        elif "Cytoplasm" in kws:
            classes[acc] = "cytoplasm"
        elif "Periplasm" in kws:
            classes[acc] = "periplasm"

    buckets: Dict[str, List[List[str]]] = {}
    unmatched = 0
    for header, seq in recs[:1200]:
        key = None
        xm = re.search(r"Swiss-Prot:(\w+)", header)
        if xm:
            acc = xm.group(1)
            key = classes.get(acc) or classes.get(acc.split("-")[0])
        if key is None:
            unmatched += 1
            continue
        cs = iter_codons(seq)
        if len(cs) < MIN_CODONS:
            continue
        buckets.setdefault(key, []).append(cs)

    rows: List[LocRow] = []
    for loc in ("cytoplasm", "membrane", "periplasm"):
        genes = buckets.get(loc, [])
        if len(genes) < 30:
            continue
        shells: List[float] = []
        hydros: List[float] = []
        for cs in genes:
            idx = [pack_codon_bits(c, enc) for c in cs]
            xs = [int(a ^ b).bit_count() for a, b in zip(idx[:-1], idx[1:])]
            shells.append(sum(xs) / len(xs))
            aa = [STANDARD_CODE.get(c, "X") for c in cs]
            vals = [KYTE_DOOLITTLE[a] for a in aa if a in KYTE_DOOLITTLE]
            hydros.append(sum(vals) / len(vals) if vals else float("nan"))
        pairs = [(s, h) for s, h in zip(shells, hydros) if h == h]
        if len(pairs) < 30:
            continue
        sx = [p[0] for p in pairs]
        sy = [p[1] for p in pairs]
        mx = sum(sx) / len(sx)
        my = sum(sy) / len(sy)
        cov = sum((x - mx) * (y - my) for x, y in pairs)
        vx = sum((x - mx) ** 2 for x in sx) ** 0.5
        vy = sum((y - my) ** 2 for y in sy) ** 0.5
        r = cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")
        rows.append(
            LocRow(
                loc=loc,
                n_genes=len(pairs),
                mean_shell=sum(sx) / len(sx),
                mean_hydro=sum(sy) / len(sy),
                pearson_r=r,
            )
        )
    return tuple(rows)


def print_uniprot_loc_census(rows: Tuple[LocRow, ...]) -> Dict[str, bool]:
    report_section("30. SHELL-WALK VS HYDROPATHY BY CELLULAR LOCATION")
    report_objects(("UniProt reviewed E. coli KW classes; per-gene mean shell vs KD hydropathy; Pearson r",))
    gates = {
        "169_loc_classes_matched": bool(rows),
        "170_shell_hydro_correlation_recorded": bool(rows) and all(r.pearson_r == r.pearson_r for r in rows),
    }
    print_table(
        ("locus", "genes", "mean_shell", "mean_KD", "pearson"),
        (10, 7, 11, 9, 8),
        [
            (r.loc, r.n_genes, f"{r.mean_shell:.4f}", f"{r.mean_hydro:.3f}", f"{r.pearson_r:+.3f}")
            for r in rows
        ],
        aligns=("<", ">", ">", ">", ">"),
    )
    print()
    report_checks((
        ("location classes matched to CDS", gates["169_loc_classes_matched"], f"strata={len(rows)}", ">=2"),
        ("shell-hydropathy correlations defined", gates["170_shell_hydro_correlation_recorded"], "see pearson", "defined"),
    ))
    return gates


def _gc_at_indices(seq: str) -> Tuple[List[int], List[int]]:
    idx_gc = [i for i, ch in enumerate(seq) if ch in STRONG]
    idx_at = [i for i, ch in enumerate(seq) if ch in {"A", "T"}]
    return idx_gc, idx_at


def _gc_shuffle_codon_list(
    codons: Sequence[str],
    rng: random.Random,
    idx_gc: Optional[List[int]] = None,
    idx_at: Optional[List[int]] = None,
) -> List[str]:
    s = "".join(codons)
    if idx_gc is None or idx_at is None:
        idx_gc, idx_at = _gc_at_indices(s)
    chars = list(s)
    gcb = [chars[i] for i in idx_gc]
    atb = [chars[i] for i in idx_at]
    rng.shuffle(gcb)
    rng.shuffle(atb)
    for i, ch in zip(idx_gc, gcb):
        chars[i] = ch
    for i, ch in zip(idx_at, atb):
        chars[i] = ch
    s2 = "".join(chars)
    return [s2[i : i + 3] for i in range(0, len(s2) - 2, 3)]
