#!/usr/bin/env python3
"""
hqvm_cgm_octaves_common.py

Shared adapter for the CGM-hQVM octaves program: frozen constants with
provenance, octave/tick/cents coordinates, musical-interval library,
dyadic word helpers, discrete metrics, and kernel gates.

Imports kernel / trestleboard / compact-geometry / wavefunction-kernel
octave primitives; does not rederive them. Companions: hqvm_cgm_octaves_1.py
through _3.py, _run.py.
"""
from __future__ import annotations

import hashlib
import io
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
RESULTS_PATH = _EXP / "hqvm_cgm_octaves_results.txt"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from gyroscopic.hQVM.api import (
    OmegaSignature12,
    compose_omega_signatures,
    omega_word_signature,
    q_word6,
    shadow_partner_byte,
    shell_index_from_chirality6,
    shell_population,
    word_signature,
)
from gyroscopic.hQVM.constants import (
    APERTURE_GAP,
    APERTURE_GAP_Q256,
    CHIRALITY_QUBITS_6,
    DELTA_BU,
    GENE_MIC_S,
    HORIZON_SIZE,
    M_A,
    OMEGA_SIZE,
    RHO,
    byte_family,
    byte_to_intron,
    step_state_by_byte,
)
from gyroscopic.hQVM.family import (
    byte_from_family_micro,
    fold_disagreement_d,
    verify_d6_against_api,
)

try:
    from hqvm_compact_geom_common import CODE_C1, CODE_C2, CODE_C3, E_EW_GEV, M_SHELL
except Exception:
    E_EW_GEV = 246.22
    CODE_C1, CODE_C2, CODE_C3 = math.comb(6, 1), math.comb(6, 2), math.comb(6, 3)
    M_SHELL = sum(k * math.comb(6, k) for k in range(7))

DELTA_CONT = float(APERTURE_GAP)  # continuum aperture 1 - rho
DELTA = DELTA_CONT  # alias used throughout program
DELTA_DEPTH4 = 1.0 / 48.0  # depth-4 dyadic approximant
DELTA_KERNEL = DELTA_DEPTH4  # alias
DELTA_DYADIC_8 = float(APERTURE_GAP_Q256) / 256.0  # best 8-bit dyadic 5/256
DELTA_BYTE = DELTA_DYADIC_8  # deprecated alias; prefer DELTA_DYADIC_8
CHIRALITY_D = int(CHIRALITY_QUBITS_6)
OMEGA = int(OMEGA_SIZE)
H_CARD = int(HORIZON_SIZE)
C1, C2, C3 = int(CODE_C1), int(CODE_C2), int(CODE_C3)
C6 = tuple(math.comb(6, k) for k in range(7))
Q_G = 4.0 * math.pi
D_SHELL = 24.0
G_KERNEL = Q_G / D_SHELL
V_GEV = float(E_EW_GEV)
V_EV = V_GEV * 1.0e9
E_CS_GEV = 1.22e19
TICKS_PER_OCTAVE = 1.0 / DELTA
TICKS_PER_K = math.log2(1.0 / DELTA) / DELTA
CHIRALITY_SPACE = (1.0 / 48.0) / (1.0 / 32.0)
NULL_SEED = 20260815
OCTAVE_TOL_TICKS = 2.0
W2_BYTES = (0xAA, 0xAB)
W2P_BYTES = (0x2A, 0x2B)
STAGE_OF_BIT = ("CS", "UNA", "ONA", "BU", "BU", "ONA", "UNA", "CS")
BIT_ROLES = ("L0", "LI", "FG", "BG", "BG", "FG", "LI", "L0")
S_P = math.pi / 2.0
U_P = 1.0 / math.sqrt(2.0)
O_P = math.pi / 4.0
EPSILON_6 = 0x3F
# Full even-index atlas (32) and dyadic spine O5 (six nodes / five octave intervals)
EVEN_H: Tuple[int, ...] = tuple(range(2, 65, 2))
DYADIC_SPINE: Tuple[int, ...] = tuple(2**k for k in range(1, 7))
PREDECESSOR_48 = 48
PHI_SU2 = 2.0 * math.acos((1.0 + 2.0 * math.sqrt(2.0)) / 4.0)
BULK_STATES = OMEGA - 2 * H_CARD  # 4096 - 128 = 3968
MASK_CODE_NOTE = (
    "pair-diagonal self-dual binary [12,6,2] with weight enumerator (1+z^2)^6; "
    "not identified with the classical GF(4) hexacode"
)

STATUS_KERNEL_EXACT = "KERNEL_EXACT"
STATUS_DERIVED = "DERIVED_GIVEN_IMPORTED_CONSTANTS"
STATUS_EMPIRICAL = "FINITE_EMPIRICAL_ALIGNMENT"
STATUS_CONTINUUM = "CONTINUUM_BRIDGE_DEPENDENT"
STATUS_HYP = "HYPOTHESIS_GENERATING"
STATUS_NULL = "NULL_FAILED_AUDIT_ONLY"
STATUS_UNRESOLVED = "UNRESOLVED"
# Back-compat aliases used by existing sections
STATUS_EXACT = STATUS_KERNEL_EXACT


@dataclass(frozen=True)
class ConstantRecord:
    name: str
    value: float
    source: str
    derivation: str
    status: str = "frozen"


OCTAVE_CONSTANTS: Tuple[ConstantRecord, ...] = (
    ConstantRecord("m_a", float(M_A), "gyroscopic.hQVM.constants", "imported"),
    ConstantRecord("delta_BU", float(DELTA_BU), "gyroscopic.hQVM.constants", "imported"),
    ConstantRecord("rho", float(RHO), "gyroscopic.hQVM.constants", "imported"),
    ConstantRecord("Delta_cont", DELTA_CONT, "APERTURE_GAP = 1-rho", "imported"),
    ConstantRecord("Delta_depth4", DELTA_DEPTH4, "1/48", "imported_identity"),
    ConstantRecord("Delta_dyadic_8", DELTA_DYADIC_8, "APERTURE_GAP_Q256/256=5/256", "imported"),
    ConstantRecord("Q_G", Q_G, "4*pi", "imported_identity"),
    ConstantRecord("v_EW_GeV", V_GEV, "hqvm_compact_geom_common.E_EW_GEV", "imported"),
    ConstantRecord("horizon_size", float(H_CARD), "HORIZON_SIZE", "imported"),
    ConstantRecord("omega_size", float(OMEGA), "OMEGA_SIZE", "imported"),
    ConstantRecord("chirality_dim", float(CHIRALITY_D), "CHIRALITY_QUBITS_6", "imported"),
    ConstantRecord("ticks_per_octave", TICKS_PER_OCTAVE, "1/Delta_cont", "derived"),
    ConstantRecord("chirality_space", CHIRALITY_SPACE, "(1/48)/(1/32)", "imported_identity"),
)


@dataclass(frozen=True)
class OctaveCoordinate:
    quantity: str
    raw_value: float
    octave_index: int
    residual: float
    base: float
    convention: str
    ticks: float
    octave_phase_ticks: float


@dataclass(frozen=True)
class IntervalRecord:
    name: str
    numerator: int
    denominator: int
    log2: float
    cents: float


@dataclass(frozen=True)
class WordProbe:
    name: str
    bytes: Tuple[int, ...]
    length: int
    parity: int
    tau_u6: int
    tau_v6: int
    packed_sig: int
    shell_tau: int
    is_identity: bool


def octaves_between(x: float, y: float) -> float:
    return math.log2(float(y) / float(x))


def cents_between(x: float, y: float) -> float:
    return 1200.0 * math.log2(float(y) / float(x))


def ticks_of_energy(E: float, E0: float = V_EV, Delta: float = DELTA) -> float:
    return math.log2(float(E0) / float(E)) / float(Delta)


def octave_coordinate(
    x: float,
    *,
    quantity: str = "x",
    x_ref: float = 1.0,
    convention: str = "log2_ratio",
    Delta: float = DELTA,
) -> OctaveCoordinate:
    z = math.log2(float(x) / float(x_ref))
    idx = int(math.floor(z))
    res = z - idx
    ticks = z / float(Delta)
    phase = ticks % (1.0 / float(Delta))
    return OctaveCoordinate(quantity, float(x), idx, res, float(x_ref), convention, ticks, phase)


def energy_octave_coordinate(E: float, E0: float = V_EV, Delta: float = DELTA) -> OctaveCoordinate:
    z = math.log2(float(E0) / float(E))
    idx = int(math.floor(z))
    res = z - idx
    ticks = z / float(Delta)
    phase = ticks % (1.0 / float(Delta))
    return OctaveCoordinate("E", float(E), idx, res, float(E0), "log2(v/E)", ticks, phase)


def circular_tick_distance(a: float, b: float, period: float = TICKS_PER_OCTAVE) -> float:
    d = abs(float(a) - float(b)) % period
    return min(d, period - d)


def nearest_octave_boundary_ticks(ticks: float, period: float = TICKS_PER_OCTAVE) -> Tuple[float, float]:
    phase = float(ticks) % period
    dist = min(phase, period - phase)
    return phase, dist


def _interval(name: str, p: int, q: int) -> IntervalRecord:
    r = Fraction(p, q)
    lg = math.log2(float(r))
    return IntervalRecord(name, r.numerator, r.denominator, lg, 1200.0 * lg)


INTERVALS: Tuple[IntervalRecord, ...] = (
    _interval("octave", 2, 1),
    _interval("fifth", 3, 2),
    _interval("fourth", 4, 3),
    _interval("major_tone", 9, 8),
    _interval("minor_tone", 10, 9),
    _interval("major_third_just", 5, 4),
    _interval("minor_third_just", 6, 5),
    _interval("major_sixth_just", 5, 3),
    _interval("minor_sixth_just", 8, 5),
    _interval("major_seventh_just", 15, 8),
    _interval("pythagorean_comma", 3**12, 2**19),
    _interval("syntonic_comma", 81, 80),
    _interval("schisma", 3**8 * 5, 2**15),
)


def interval_by_name(name: str) -> IntervalRecord:
    for it in INTERVALS:
        if it.name == name:
            return it
    raise KeyError(name)


def tet12_ratio(k: int) -> float:
    return 2.0 ** (int(k) / 12.0)


def aperture_comma_table() -> List[Dict[str, float]]:
    commas = {
        "PC": interval_by_name("pythagorean_comma").log2,
        "SC": interval_by_name("syntonic_comma").log2,
        "schisma": interval_by_name("schisma").log2,
    }
    apertures = {
        "Delta_cont": DELTA_CONT,
        "Delta": DELTA_CONT,
        "Delta_depth4": DELTA_DEPTH4,
        "Delta_kernel": DELTA_KERNEL,
        "Delta_dyadic_8": DELTA_DYADIC_8,
        "Delta_byte": DELTA_DYADIC_8,
        "delta_BU_over_pi": float(DELTA_BU) / math.pi,
    }
    rows: List[Dict[str, float]] = []
    for cname, cv in commas.items():
        for aname, av in apertures.items():
            d_oct = abs(cv - av)
            rows.append(
                {
                    "comma": cname,
                    "aperture": aname,
                    "comma_log2": cv,
                    "aperture_value": av,
                    "abs_diff_octaves": d_oct,
                    "abs_diff_cents": 1200.0 * d_oct,
                    "abs_diff_ticks": d_oct / DELTA,
                }
            )
    return rows


def shell_pair_ratios() -> List[Dict[str, object]]:
    just = [it for it in INTERVALS if "comma" not in it.name and it.name != "schisma"]
    rows: List[Dict[str, object]] = []
    for i in range(7):
        for j in range(i):
            if C6[j] == 0:
                continue
            p, q = C6[i], C6[j]
            g = math.gcd(p, q)
            pn, qn = p // g, q // g
            lg = math.log2(pn / qn)
            best = min(just, key=lambda it: abs(it.log2 - lg))
            rows.append(
                {
                    "i": i,
                    "j": j,
                    "Ci": C6[i],
                    "Cj": C6[j],
                    "ratio": f"{pn}/{qn}",
                    "log2": lg,
                    "cents": 1200.0 * lg,
                    "nearest": best.name,
                    "nearest_cents": best.cents,
                    "residual_cents": 1200.0 * (lg - best.log2),
                }
            )
    return rows


def _byte_fm(family: int, micro: int) -> int:
    return int(byte_from_family_micro(int(family) & 3, int(micro) & 0x3F, CHIRALITY_D)) & 0xFF


def W2(m: int) -> Tuple[int, ...]:
    return (_byte_fm(0, m), _byte_fm(1, m))


def W2p(m: int) -> Tuple[int, ...]:
    return (_byte_fm(2, m), _byte_fm(3, m))


def Wfull(m: int) -> Tuple[int, ...]:
    return W2(m) + W2p(m)


def z2_holonomy_word(m: int) -> Tuple[int, ...]:
    return Wfull(m) + Wfull(m)


def octave_aperture_residues() -> Dict[str, object]:
    """Three octave resolutions and the continuum–depth4 residue (octave topic)."""
    ticks = 1.0 / DELTA_CONT
    return {
        "Delta_dyadic_8": DELTA_DYADIC_8,
        "Delta_cont": DELTA_CONT,
        "Delta_depth4": DELTA_DEPTH4,
        "ordering_dyadic_lt_cont_lt_depth4": (
            DELTA_DYADIC_8 < DELTA_CONT < DELTA_DEPTH4
        ),
        "ticks_per_octave_cont": ticks,
        "ticks_depth4_frame": 48.0,
        "epsilon_oct_ticks": ticks - 48.0,
        "forty_eight_Delta": 48.0 * DELTA_CONT,
        "one_minus_48_Delta": 1.0 - 48.0 * DELTA_CONT,
        "Q_G_over_2pi": Q_G / (2.0 * math.pi),
        "log2_Q_G_over_2pi": math.log2(Q_G / (2.0 * math.pi)),
        "chirality_space_2_3": CHIRALITY_SPACE,
        "claim_status": STATUS_KERNEL_EXACT,
    }


def predecessor_horizon_ladder() -> List[Dict[str, object]]:
    """P_k = 3·2^(k-1) = (3/4)·2^(k+1); 48 is depth-4 predecessor, not dyadic."""
    rows = []
    for k in range(1, 7):
        p = 3 * (1 << (k - 1))
        rows.append(
            {
                "k": k,
                "P_k": p,
                "as_3_times_2_km1": p,
                "as_3_4_of_next_dyadic": (3.0 / 4.0) * (1 << (k + 1)),
                "is_48": p == 48,
                "is_dyadic": (p & (p - 1)) == 0,
            }
        )
    return rows


def octave_primitives_from_wavefunction() -> Dict[str, object]:
    """Import octave-scale primitives from hqvm_wavefunction_kernel (no rederive).

    Used only for dyadic hierarchy, word-period Z2 closure, half/full frame,
    and local 50% dual-reading vs global Delta — not as a curvature topic.
    """
    from hqvm_wavefunction_kernel import (
        aperture_collapse_curve,
        decompose_byte,
        fold_disagreement,
        holographic_hierarchy,
        verify_k4_w2,
    )

    hier = holographic_hierarchy()
    aper = aperture_collapse_curve()
    k4 = verify_k4_w2()

    flat = sum(1 for b in range(256) if decompose_byte(b).is_flat)
    fold_hist = Counter(fold_disagreement(b) for b in range(256))
    # Word-period octave: L=2 involution, L=4 canonical, L=8 = F^2 identity
    w2 = W2(0)
    wfull = Wfull(0)
    f2 = z2_holonomy_word(0)
    from gyroscopic.hQVM.constants import GENE_MAC_REST

    def _is_id(word: Sequence[int]) -> bool:
        return apply_word(GENE_MAC_REST, word) == GENE_MAC_REST

    word_periods = {
        "W2_len": len(w2),
        "Wfull_len": len(wfull),
        "F2_len": len(f2),
        "W2_sq_is_id_rest": _is_id(w2 + w2),
        "Wfull_sq_is_id_rest": _is_id(wfull + wfull),
        "F2_is_id_rest": _is_id(f2),
        "log2_word_ladder": [0, 1, 2, 3, 4],
        "word_lengths": [1, 2, 4, 8, 16],
    }
    return {
        "holographic_levels": [
            {
                "name": h.name,
                "dof": h.dof,
                "subspace": h.subspace,
                "space": h.space,
                "dimension": h.dimension,
                "log2_space_over_subspace": math.log2(h.space / h.subspace),
                "redundancy": h.redundancy,
            }
            for h in hier
        ],
        "carrier_is_horizon_squared": OMEGA == H_CARD * H_CARD,
        "log2_Omega_over_H": math.log2(OMEGA / H_CARD),
        "aperture_collapse": [
            {"depth": a.depth, "label": a.label, "aperture": a.aperture} for a in aper
        ],
        "local_dual_reading_aperture": 0.5,
        "global_Delta_cont": DELTA_CONT,
        "compression_ratio_50_to_Delta": 0.5 / DELTA_CONT,
        "flat_bytes": flat,
        "curved_bytes": 256 - flat,
        "fold_disagreement_hist": dict(sorted(fold_hist.items())),
        "half_frame_phases": 4,
        "full_frame_bits": 8,
        "k4_w2_all_pass": bool(k4.all_pass),
        "word_periods": word_periods,
        "source": "hqvm_wavefunction_kernel",
        "claim_status": STATUS_KERNEL_EXACT,
    }


def repeat_word(word: Sequence[int], times: int) -> Tuple[int, ...]:
    w = tuple(int(b) & 0xFF for b in word)
    return w * int(times)


def apply_word(state24: int, word: Sequence[int]) -> int:
    s = int(state24) & 0xFFFFFF
    for b in word:
        s = step_state_by_byte(s, int(b) & 0xFF)
    return s


def packed_omega_sig(sig: OmegaSignature12) -> int:
    return ((sig.parity & 1) << 12) | ((sig.tau_u6 & 0x3F) << 6) | (sig.tau_v6 & 0x3F)


def probe_word(name: str, word: Sequence[int]) -> WordProbe:
    items = tuple(int(b) & 0xFF for b in word)
    osig = omega_word_signature(items)
    shell = shell_index_from_chirality6(osig.tau_u6 ^ osig.tau_v6)
    ident = osig.parity == 0 and osig.tau_u6 == 0 and osig.tau_v6 == 0
    return WordProbe(
        name=name,
        bytes=items,
        length=len(items),
        parity=osig.parity,
        tau_u6=osig.tau_u6,
        tau_v6=osig.tau_v6,
        packed_sig=packed_omega_sig(osig),
        shell_tau=shell,
        is_identity=ident,
    )


def signature_hamming(a: OmegaSignature12, b: OmegaSignature12) -> int:
    return (
        (a.parity ^ b.parity)
        + (a.tau_u6 ^ b.tau_u6).bit_count()
        + (a.tau_v6 ^ b.tau_v6).bit_count()
    )


def signature_composition_exact(word: Sequence[int]) -> Dict[str, object]:
    """Homomorphism check: sig(w+w) == compose(sig(w), sig(w)). Always True if API consistent."""
    w = tuple(int(b) & 0xFF for b in word)
    s1 = omega_word_signature(w)
    s2 = omega_word_signature(w + w)
    lift = compose_omega_signatures(s1, s1)
    return {
        "length": len(w),
        "sig1": packed_omega_sig(s1),
        "sig2": packed_omega_sig(s2),
        "lift": packed_omega_sig(lift),
        "hamming_to_lift": signature_hamming(s2, lift),
        "hamming_to_id": signature_hamming(
            s2, OmegaSignature12(parity=0, tau_u6=0, tau_v6=0)
        ),
        "compose_exact": packed_omega_sig(s2) == packed_omega_sig(lift),
        "note": "signature_monoid_homomorphism_check",
    }


def octave_holonomy_defect(word: Sequence[int]) -> Dict[str, object]:
    """Deprecated alias for signature_composition_exact (was misnamed as defect)."""
    return signature_composition_exact(word)


def omega12_word_vs_signature_disagreement(word: Sequence[int]) -> Dict[str, object]:
    """Non-tautological chart defect: step_omega12 word action vs apply_omega_signature.

    Counts Omega12 states where replaying bytes disagrees with the affine signature.
    """
    from gyroscopic.hQVM.api import (
        OmegaState12,
        apply_omega_signature,
        step_omega12_by_byte,
    )

    w = tuple(int(b) & 0xFF for b in word)
    sig = omega_word_signature(w)
    disagree = 0
    for u in range(64):
        for v in range(64):
            cur = OmegaState12(u6=u, v6=v)
            for b in w:
                cur = step_omega12_by_byte(cur, b)
            pred = apply_omega_signature(OmegaState12(u6=u, v6=v), sig)
            if cur.u6 != pred.u6 or cur.v6 != pred.v6:
                disagree += 1
    return {
        "length": len(w),
        "n_omega12": 4096,
        "disagree": disagree,
        "agree_frac": 1.0 - disagree / 4096.0,
        "exact": disagree == 0,
        "status": STATUS_EXACT if disagree == 0 else STATUS_EMPIRICAL,
    }


def one_step_shadow_from_rest() -> Dict[str, object]:
    """Certificate: 256 bytes from GENE_MAC_REST -> 128 unique next states, fibres size 2."""
    from collections import defaultdict

    from gyroscopic.hQVM.api import state24_to_omega12
    from gyroscopic.hQVM.constants import GENE_MAC_REST

    buckets: Dict[int, List[int]] = defaultdict(list)
    for b in range(256):
        nxt = step_state_by_byte(GENE_MAC_REST, b)
        buckets[nxt].append(b)
    sizes = [len(v) for v in buckets.values()]
    n_unique = len(buckets)
    h_proj = conditional_entropy_bits(sizes, 256)
    return {
        "n_unique_next": n_unique,
        "n_unique_expected": 128,
        "fiber_min": min(sizes) if sizes else 0,
        "fiber_max": max(sizes) if sizes else 0,
        "all_fibres_size_2": all(s == 2 for s in sizes),
        "H_byte_given_next": h_proj,
        "H_expected_1bit": abs(h_proj - 1.0) < 1e-12,
        "status": STATUS_EXACT,
    }


def phi_isotropic_bit(u: int) -> float:
    """Exact Walsh multiplier for uniform weight-1 XOR transport: 1 - 2*wt(u)/6."""
    return 1.0 - (2.0 * popcount6(u)) / 6.0


def best_dyadic_denom256(x: float) -> Tuple[int, float]:
    """Best k/256 approximant to x under absolute error; returns (k, error)."""
    best_k = 0
    best_err = abs(x)
    for k in range(257):
        err = abs(x - k / 256.0)
        if err < best_err:
            best_err = err
            best_k = k
    return best_k, best_err


def random_word(length: int, rng: random.Random) -> Tuple[int, ...]:
    return tuple(rng.randrange(256) for _ in range(length))


def byte_charts(byte: int) -> Dict[str, int]:
    b = int(byte) & 0xFF
    intron = byte_to_intron(b)
    fam = byte_family(b)
    q6 = q_word6(b)
    return {
        "byte": b,
        "intron": intron,
        "family": fam,
        "q6": q6,
        "shell": shell_index_from_chirality6(q6),
        "shadow": shadow_partner_byte(b),
        "fold_d": fold_disagreement_d(b, CHIRALITY_D),
    }


@lru_cache(maxsize=1)
def shadow_fiber_sizes() -> Tuple[int, ...]:
    seen = set()
    sizes: List[int] = []
    for b in range(256):
        p = shadow_partner_byte(b)
        key = frozenset((b, p))
        if key in seen:
            continue
        seen.add(key)
        sizes.append(len(key))
    return tuple(sizes)


@lru_cache(maxsize=1)
def q6_fiber_sizes() -> Tuple[int, ...]:
    c = Counter(q_word6(b) for b in range(256))
    return tuple(sorted(c.values()))


@lru_cache(maxsize=1)
def shell_fiber_sizes() -> Tuple[int, ...]:
    c = Counter(shell_index_from_chirality6(q_word6(b)) for b in range(256))
    return tuple(c[k] for k in range(7))


def conditional_entropy_bits(fiber_sizes: Sequence[int], n_total: int) -> float:
    if n_total <= 0:
        return float("nan")
    h = 0.0
    for s in fiber_sizes:
        if s <= 0:
            continue
        p = s / n_total
        h += p * math.log2(s)
    return h


def total_variation(p: Sequence[float], q: Sequence[float]) -> float:
    if len(p) != len(q):
        raise ValueError("tv length mismatch")
    return 0.5 * sum(abs(float(a) - float(b)) for a, b in zip(p, q))


def kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-15) -> float:
    s = 0.0
    for a, b in zip(p, q):
        aa = max(float(a), eps)
        bb = max(float(b), eps)
        s += aa * math.log2(aa / bb)
    return s


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kernel_manifest() -> Dict[str, object]:
    ok, note = verify_d6_against_api()
    pops = tuple(shell_population(k) for k in range(7))
    pops_bin = tuple(p // H_CARD for p in pops) if all(p % H_CARD == 0 for p in pops) else ()
    paths = [
        _REPO / "gyroscopic" / "hQVM" / "api.py",
        _REPO / "gyroscopic" / "hQVM" / "family.py",
        _REPO / "gyroscopic" / "hQVM" / "constants.py",
    ]
    return {
        "d6_api_ok": ok,
        "d6_api_note": note,
        "APERTURE_GAP": DELTA,
        "RHO": float(RHO),
        "M_A": float(M_A),
        "OMEGA": OMEGA,
        "HORIZON": H_CARD,
        "shell_populations": pops,
        "shell_populations_bin": pops_bin,
        "shell_populations_ok": pops_bin == C6,
        "shell_populations_scaled_ok": pops == tuple(c * H_CARD for c in C6),
        "hashes": {str(p.relative_to(_REPO)): sha256_file(p) for p in paths},
    }


def exact_gates() -> List[Tuple[str, bool]]:
    man = kernel_manifest()
    pc = interval_by_name("pythagorean_comma")
    bulk = sum(c * H_CARD for c in C6[1:6])
    return [
        ("d6_api", bool(man["d6_api_ok"])),
        ("|Omega|=4096", OMEGA == 4096),
        ("|H|=64", H_CARD == 64),
        ("bulk_count_is_3968", bulk == 3968 and BULK_STATES == 3968),
        ("shell_census_C6", C6 == (1, 6, 15, 20, 15, 6, 1)),
        ("shell_pop=C6*|H|", bool(man["shell_populations_scaled_ok"])),
        ("log2|Omega|=12", abs(math.log2(OMEGA) - 12.0) < 1e-12),
        ("log2|H|=6", abs(math.log2(H_CARD) - 6.0) < 1e-12),
        ("Delta>0", DELTA > 0.0),
        ("PC_exact_log2", abs(pc.log2 - math.log2(3**12 / 2**19)) < 1e-15),
        ("chirality_space_2/3", abs(CHIRALITY_SPACE - 2.0 / 3.0) < 1e-15),
        ("wave_normalization_QG_ma2_half", abs(Q_G * float(M_A) ** 2 - 0.5) < 1e-12),
    ]


class Tee:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            try:
                stream.write(data)
            except (ValueError, OSError):
                continue
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            try:
                stream.flush()
            except (ValueError, OSError):
                continue


def min_mean_max(values: Sequence[float]) -> Tuple[float, float, float]:
    xs = [float(v) for v in values]
    if not xs:
        return float("nan"), float("nan"), float("nan")
    return min(xs), sum(xs) / len(xs), max(xs)


def fmt(x: float, n: int = 6) -> str:
    if x != x:
        return "nan"
    return f"{x:.{n}f}"


def wht64(vec: Sequence[float]) -> List[float]:
    out = [float(v) for v in vec]
    if len(out) != 64:
        raise ValueError("wht64 expects length 64")
    h = 1
    while h < 64:
        for i in range(0, 64, h * 2):
            for j in range(i, i + h):
                a = out[j]
                b = out[j + h]
                out[j] = a + b
                out[j + h] = a - b
        h *= 2
    return out


def krawtchouk(k: int, x: int, n: int = 6) -> int:
    total = 0
    for j in range(k + 1):
        if j > x or (k - j) > (n - x):
            continue
        sign = -1 if (j & 1) else 1
        total += sign * math.comb(x, j) * math.comb(n - x, k - j)
    return total


def normalize_dist(xs: Sequence[float]) -> List[float]:
    s = sum(float(x) for x in xs)
    if s <= 0:
        n = len(xs)
        return [1.0 / n] * n if n else []
    return [float(x) / s for x in xs]


def popcount6(x: int) -> int:
    return (int(x) & 0x3F).bit_count()


def walsh_band_of_mode(u: int, perm: Sequence[int] | None = None) -> int:
    """First filtration level where mode u is nontrivial under nested coords.

    Levels: 0 = {bit0}, 1 = {0,1}, 2 = {0..3}, 3 = {0..5} after optional perm.
    Band = min level whose support intersects bits of u; u=0 -> band -1 (DC).
    """
    if u == 0:
        return -1
    order = list(perm) if perm is not None else list(range(6))
    cuts = (1, 2, 4, 6)
    for level, cut in enumerate(cuts):
        mask = 0
        for i in range(cut):
            mask |= 1 << order[i]
        if u & mask:
            return level
    return 3


def even_weight_q6() -> Tuple[int, ...]:
    return tuple(q for q in range(64) if popcount6(q) % 2 == 0)


def odd_weight_q6() -> Tuple[int, ...]:
    return tuple(q for q in range(64) if popcount6(q) % 2 == 1)


def harmonic_to_even_q_iso() -> List[Dict[str, object]]:
    """Canonical iso: H[i] <-> even-weight q sorted ascending (32 rows)."""
    ev = even_weight_q6()
    rows: List[Dict[str, object]] = []
    for i, (n, q) in enumerate(zip(EVEN_H, ev)):
        rows.append(
            {
                "idx": i + 1,
                "harmonic_n": n,
                "even_q6": q,
                "wt": popcount6(q),
                "is_dyadic": n in DYADIC_SPINE,
                "is_predecessor_48": n == PREDECESSOR_48,
            }
        )
    return rows


def bytes_by_q_parity(even: bool) -> Tuple[int, ...]:
    out = []
    for b in range(256):
        w = popcount6(q_word6(b))
        if (w % 2 == 0) == even:
            out.append(b)
    return tuple(out)


def fold_flat_curved_census() -> Dict[str, int]:
    flat = sum(1 for b in range(256) if fold_disagreement_d(b, CHIRALITY_D) == 0)
    return {"flat": flat, "curved": 256 - flat, "n_bytes": 256}


def foundation_lock_scalars() -> Dict[str, float]:
    """Closed-form locks from §4 of the even-harmonics foundation note."""
    ma = float(M_A)
    rho = float(RHO)
    dbu = float(DELTA_BU)
    return {
        "m_a": ma,
        "Q_G": Q_G,
        "Q_G_m_a2": Q_G * ma * ma,
        "s_p": S_P,
        "s_p_over_m_a2": S_P / (ma * ma),
        "four_pi2": 4.0 * math.pi * math.pi,
        "gyro_sum": S_P + O_P + O_P,
        "u_p": U_P,
        "o_p": O_P,
        "delta_BU": dbu,
        "rho": rho,
        "Delta": DELTA,
        "48_Delta": 48.0 * DELTA,
        "48_Delta_residue": 1.0 - 48.0 * DELTA,
        "chirality_space": CHIRALITY_SPACE,
        "Delta_cont": DELTA_CONT,
        "Delta_depth4": DELTA_DEPTH4,
        "Delta_dyadic_8": DELTA_DYADIC_8,
        "Delta_byte": DELTA_BYTE,  # deprecated
        "Delta_kernel": DELTA_KERNEL,  # deprecated alias of depth4
        "alpha0": (dbu**4) / ma,
        "zeta": 8.0 / (ma * math.sqrt(3.0)),
        "alpha0_zeta": ((dbu**4) / ma) * (8.0 / (ma * math.sqrt(3.0))),
        "rho4_over_pi_sqrt3": (rho**4) / (math.pi * math.sqrt(3.0)),
        "S_geo": ma * math.pi * math.sqrt(3.0) / 2.0,
        "rho5": rho**5,
        "m_gap_RouteA_GeV": float(C2) * V_GEV * (DELTA**2),
        "E_grade2_GeV": V_GEV * (DELTA**2),
        "M_shell": float(M_SHELL),
        "M_shell_over_32": float(M_SHELL) / 32.0,
    }


def k4_horizon_cycle_types() -> Dict[str, Dict[str, object]]:
    """Cycle types of {id,S,C,F} on equality horizon, computed via apply_omega_gate."""
    from gyroscopic.hQVM.api import OmegaState12, apply_omega_gate

    def cycle_decomp(mapping: Dict[int, int]) -> Dict[str, object]:
        seen: set = set()
        fixed = 0
        cycles: List[Tuple[int, ...]] = []
        for x in range(64):
            if x in seen:
                continue
            y = mapping[x]
            if y == x:
                fixed += 1
                seen.add(x)
                continue
            cyc = [x]
            seen.add(x)
            while y not in seen:
                cyc.append(y)
                seen.add(y)
                y = mapping[y]
            cycles.append(tuple(cyc))
        return {
            "fixed_points": fixed,
            "cycles": cycles,
            "two_cycles": sum(1 for c in cycles if len(c) == 2),
            "cycle_type": sorted(len(c) for c in cycles),
        }

    out: Dict[str, Dict[str, object]] = {}
    for g in ("id", "S", "C", "F"):
        mapping: Dict[int, int] = {}
        stays_on_pole = True
        for u in range(64):
            s = OmegaState12(u6=u, v6=u)
            s2 = apply_omega_gate(s, g)
            if s2.u6 != s2.v6:
                stays_on_pole = False
            mapping[u] = s2.u6
        dec = cycle_decomp(mapping)
        pairs = [
            (min(c[0], c[1]), max(c[0], c[1]))
            for c in dec["cycles"]
            if len(c) == 2
        ]
        xor_check = all((a ^ b) == EPSILON_6 for a, b in pairs) if pairs else True
        out[g] = {
            "gate": g,
            "fixed_points": dec["fixed_points"],
            "two_cycles": dec["two_cycles"],
            "cycle_type": dec["cycle_type"],
            "stays_on_equality_pole": stays_on_pole,
            "pairs_sample": pairs[:5],
            "n_pairs_listed": len(pairs),
            "two_cycle_xor_is_epsilon": xor_check,
            "all_pairs": pairs if g in ("F", "C") else [],
        }
    return out


def f_cycle_index_table() -> List[Dict[str, object]]:
    """32-row table: even harmonic n <-> even_q <-> F two-cycle on equality pole."""
    iso = harmonic_to_even_q_iso()
    f_info = k4_horizon_cycle_types()["F"]
    pairs = list(f_info["all_pairs"])
    # Order pairs by min face index to align with sorted even_q polarization
    pairs_sorted = sorted(pairs, key=lambda t: t[0])
    rows = []
    for row, pair in zip(iso, pairs_sorted):
        rows.append(
            {
                **row,
                "f_cycle_lo": pair[0],
                "f_cycle_hi": pair[1],
                "f_xor": pair[0] ^ pair[1],
                "f_xor_is_epsilon": (pair[0] ^ pair[1]) == EPSILON_6,
            }
        )
    return rows


def d_q(x: int, y: int) -> int:
    """Pairwise transport difference q6(x) XOR q6(y); not a projected group commutator."""
    return q_word6(x) ^ q_word6(y)


def projected_chi_commutator_census() -> Dict[str, object]:
    """Gate: χ(T_x T_y T_x^{-1} T_y^{-1}(s)) == χ(s) for all x,y (abelian χ quotient).

    On GF(2)^6, every byte acts by XOR q6(b) and is order-2, so the projected
    commutator is identically the identity. d_q is retained as a defect label only.
    """
    from gyroscopic.hQVM.api import chirality_word6
    from gyroscopic.hQVM.constants import GENE_MAC_REST, step_state_by_byte

    # Exhaustive algebraic identity on the quotient
    bad_xy = 0
    for x in range(256):
        for y in range(256):
            qx, qy = q_word6(x), q_word6(y)
            if (qx ^ qy ^ qx ^ qy) != 0:
                bad_xy += 1

    # Byte orders on rest (for inverses on Omega)
    orders: Dict[int, int] = {}
    for b in range(256):
        s = GENE_MAC_REST
        ord_b = 0
        for k in range(1, 9):
            s = step_state_by_byte(s, b)
            if s == GENE_MAC_REST:
                ord_b = k
                break
        orders[b] = ord_b if ord_b else 0

    def apply_inv(state: int, b: int) -> int:
        o = orders[b]
        if o <= 1:
            return state if o == 1 else step_state_by_byte(state, b)
        for _ in range(o - 1):
            state = step_state_by_byte(state, b)
        return state

    def apply_b(state: int, b: int) -> int:
        return step_state_by_byte(state, b)

    # Exhaustive (x,y) at rest: chi preserved; Omega may move (fiber curvature)
    chi_fail = 0
    omega_moved = 0
    n_pairs = 256 * 256
    for x in range(256):
        for y in range(256):
            s = GENE_MAC_REST
            chi0 = chirality_word6(s)
            s = apply_b(s, x)
            s = apply_b(s, y)
            s = apply_inv(s, x)
            s = apply_inv(s, y)
            if chirality_word6(s) != chi0:
                chi_fail += 1
            if s != GENE_MAC_REST:
                omega_moved += 1

    return {
        "n_byte_pairs": n_pairs,
        "chi_algebraic_bad": bad_xy,
        "chi_rest_path_fail": chi_fail,
        "projected_commutator_is_identity": bad_xy == 0 and chi_fail == 0,
        "omega_rest_moved_frac": omega_moved / n_pairs,
        "omega_rest_identity_frac": 1.0 - omega_moved / n_pairs,
        "d_q_note": "d_q=q6(x)^q6(y) is transport difference / plaquette label, not projected commutator",
        "mask_code_note": MASK_CODE_NOTE,
        "byte_order_census": dict(Counter(orders.values())),
        "claim_status": STATUS_KERNEL_EXACT,
    }


def wave_dispersion_table() -> List[Dict[str, object]]:
    """Isotropic Walsh modes: phi_r=1-r/3, lambda_r=r/3, classical/discrete wave freqs."""
    rows = []
    for r in range(7):
        phi_r = 1.0 - r / 3.0
        lam = r / 3.0
        # Continuous-time wave: omega = c sqrt(lambda); report at c=1
        omega_c = math.sqrt(lam) if lam >= 0 else float("nan")
        # Discrete reversible: cos(omega)=phi_r when |phi|<=1
        if abs(phi_r) <= 1.0:
            omega_d = math.acos(phi_r)
        else:
            omega_d = float("nan")
        rows.append(
            {
                "r": r,
                "phi_r": phi_r,
                "lambda_r": lam,
                "omega_cont_c1": omega_c,
                "omega_disc": omega_d,
                "phi_closed": abs(phi_r - (1.0 - 2.0 * r / 6.0)) < 1e-15,
                "claim_status": STATUS_KERNEL_EXACT,
            }
        )
    # Group vs phase velocity for continuous omega=c*sqrt(r/3): dispersive.
    # For the damped Markov channel A_{t+1}=eta^r A_t with log form, vp=vg.
    # Report both readings.
    return rows


def isotropic_radial_velocity_gate() -> Dict[str, object]:
    """For A_{t+1}(r)=η^r A_t(r), define ω(r)=r·log(1/η); then v_p=v_g=log(1/η)."""
    eta = 0.5  # probe scale; identity is eta-independent
    log_inv = math.log(1.0 / eta)
    rows = []
    for r in range(1, 7):
        omega = r * log_inv
        v_p = omega / r
        v_g = log_inv  # dω/dr
        rows.append({"r": r, "omega": omega, "v_p": v_p, "v_g": v_g, "equal": abs(v_p - v_g) < 1e-15})
    return {
        "eta_probe": eta,
        "rows": rows,
        "all_vp_eq_vg": all(r["equal"] for r in rows),
        "note": "Markov/spectral damping channel; distinct from arccos wave dispersion",
        "claim_status": STATUS_KERNEL_EXACT,
    }


def standing_wave_shell_table() -> Dict[str, object]:
    """Shell amplitudes C(6,k)·64 vs cos² fundamental envelope on k=0..6."""
    pops = [c * H_CARD for c in C6]
    # Fundamental drum envelope on discrete interval [0,6]: cos^2(pi*k/12) shifted
    # Compare to normalized binomial profile.
    env = [math.cos(math.pi * k / 12.0) ** 2 for k in range(7)]
    # Correlation of pops vs C(6,k) is exact 1; vs cos^2 envelope:
    mu_p = sum(pops) / 7.0
    mu_e = sum(env) / 7.0
    num = sum((p - mu_p) * (e - mu_e) for p, e in zip(pops, env))
    den = math.sqrt(
        sum((p - mu_p) ** 2 for p in pops) * sum((e - mu_e) ** 2 for e in env)
    )
    corr = num / den if den > 0 else float("nan")
    rows = []
    for k in range(7):
        rows.append(
            {
                "k": k,
                "C6": C6[k],
                "pop": pops[k],
                "pi_k": C6[k] / 64.0,
                "cos2_envelope": env[k],
                "is_horizon_node": k in (0, 6),
                "is_equator_antinode": k == 3,
            }
        )
    return {
        "rows": rows,
        "corr_pop_vs_cos2": corr,
        "equator_pop": pops[3],
        "horizon_pops": (pops[0], pops[6]),
        "bulk_sum": sum(pops[1:6]),
        "bulk_is_3968": sum(pops[1:6]) == 3968,
        "claim_status": STATUS_KERNEL_EXACT,
    }


def k4_interference_map() -> List[Dict[str, object]]:
    return [
        {"gate": "id", "interference": "constructive_both_faces", "abelian": True},
        {"gate": "S", "interference": "constructive_A_swap_preserving", "abelian": True},
        {"gate": "C", "interference": "constructive_B_complement_preserving", "abelian": True},
        {"gate": "F", "interference": "destructive_fold_Householder", "abelian": True},
        {
            "gate": "note",
            "interference": "K4 is abelian holonomy; non-Abelian curvature requires Q8/SU(2) lift",
            "abelian": True,
        },
    ]


def curvature_manifestations_table() -> Dict[str, object]:
    """Six curvature readouts under different quotients; ratios only (no prose)."""
    dbu = float(DELTA_BU)
    ma = float(M_A)
    phi = PHI_SU2
    # Fold |F|^2 census on bytes (fold_disagreement levels -> squared scale)
    fold_counts = Counter(fold_disagreement_d(b, CHIRALITY_D) for b in range(256))
    fold_vals = {d: (d / 4.0) ** 2 for d in range(5)}  # 0, 0.0625, 0.25, ...
    try:
        from hqvm_gravity_common import tau_G_formula as _tg

        tau_g = float(_tg)
    except Exception:
        tau_g = float(OMEGA) * DELTA * float(RHO) ** 5 * (1.0 - 4.0 * float(RHO) * DELTA**2)

    rows = [
        {"name": "Delta_cont", "value": DELTA_CONT, "unit": "dimensionless", "claim": STATUS_KERNEL_EXACT},
        {"name": "delta_BU", "value": dbu, "unit": "rad", "claim": STATUS_KERNEL_EXACT},
        {"name": "phi_SU2", "value": phi, "unit": "rad", "claim": STATUS_KERNEL_EXACT},
        {"name": "phi_SU2_over_3", "value": phi / 3.0, "unit": "rad", "claim": STATUS_KERNEL_EXACT},
        {"name": "W_residual_phi_minus_3delta", "value": phi - 3.0 * dbu, "unit": "rad", "claim": STATUS_EMPIRICAL},
        {"name": "tau_G", "value": tau_g, "unit": "neper", "claim": STATUS_DERIVED},
        {"name": "K_G_R_2", "value": math.sqrt(2.0), "unit": "dimensionless", "claim": STATUS_KERNEL_EXACT},
        {"name": "rho", "value": float(RHO), "unit": "dimensionless", "claim": STATUS_KERNEL_EXACT},
    ]
    one_third = {
        "abs_delta_BU_minus_phi_over_3": abs(dbu - phi / 3.0),
        "rel_residual": abs(dbu - phi / 3.0) / max(dbu, 1e-15),
        "within_5e-4": abs(dbu - phi / 3.0) < 5e-4,
        "within_2e-3": abs(dbu - phi / 3.0) < 2e-3,
        "claim_status": STATUS_EMPIRICAL,
    }
    return {
        "rows": rows,
        "fold_disagreement_counts": dict(sorted(fold_counts.items())),
        "fold_F2_by_level": fold_vals,
        "one_third_identity": one_third,
        "claim_status": STATUS_DERIVED,
    }


def stf_quadrupole_layer() -> Dict[str, object]:
    """Five STF / √5 / ρ^5 / 5/256 occurrences as a consistency table."""
    inv_sqrt5 = 1.0 / math.sqrt(5.0)
    rows = [
        {"tag": "STF_bulk_shells", "value": 5.0, "note": "shells 1..5", "claim": STATUS_KERNEL_EXACT},
        {"tag": "dim_STF2_3D", "value": 5.0, "note": "l=2 multiplet", "claim": STATUS_KERNEL_EXACT},
        {"tag": "Delta_dyadic_8", "value": DELTA_DYADIC_8, "note": "5/256", "claim": STATUS_KERNEL_EXACT},
        {"tag": "inv_sqrt5", "value": inv_sqrt5, "note": "STF norm", "claim": STATUS_KERNEL_EXACT},
        {"tag": "rho5", "value": float(RHO) ** 5, "note": "attenuation", "claim": STATUS_KERNEL_EXACT},
        {"tag": "cmb_ell_37_times_5", "value": 37 * 5, "note": "exploratory marker", "claim": STATUS_HYP},
    ]
    return {
        "rows": rows,
        "all_five_shell_count": sum(C6[1:6]) * H_CARD == 3968,
        "dyadic_numerator_is_5": abs(DELTA_DYADIC_8 - 5.0 / 256.0) < 1e-15,
        "claim_status": STATUS_DERIVED,
    }
