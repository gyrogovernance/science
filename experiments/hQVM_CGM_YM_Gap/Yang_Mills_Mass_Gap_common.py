"""Shared constants, Delta-ruler, Formalism inventory, and lattice engines for YM scripts.

Formalism spaces (hQVM_Specs_Formalism): Byte256, K4, GF(2)^6, C64, Omega, shells7.
Primitives: GENE_Mic transcription, family/payload extract, q6, mask12, T_b on GENE_Mac.
No section certificates here — those live in numbered scripts.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_EXP = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from gyroscopic.hQVM.constants import (
    APERTURE_GAP_Q256,
    CHIRALITY_QUBITS_6,
    GATE_NAMES,
    GENE_MAC_REST,
    GENE_MIC_S,
    HORIZON_SIZE,
    OMEGA_SIZE,
    byte_to_intron,
    intron_family,
    intron_micro_ref,
)
from gyroscopic.hQVM.family import alphabet_size, byte_from_family_micro
from hqvm_compact_geom_core import (
    DELTA,
    RHO,
    DELTA_BU,
    E_EW_GEV,
    M_A as M_A_CORE,
    STF_DIMENSION,
    CODE_C1,
    CODE_C2,
    CODE_C3,
)

V = E_EW_GEV  # grammar_energy scale alias
SQRT5 = math.sqrt(float(STF_DIMENSION))
EQUATOR = 2.0 ** (CODE_C3 * DELTA**2)
GEV_TO_EV = 1.0e9

# Unit-map discipline (non-negotiable). Main runs must stay grade1_only.
# Proof-theoretic safeguard: E_unit := v·Δ is independent of the spectral gap.
# Using E_unit := m_gap/κ₂ would make the Clay gap circular (gap defines unit that
# defines gap). matching_readout_only() is a labeled diagnostic only — never the definition.
UNIT_MAP_MODE = "grade1_only"  # or "matching_readout_only" (never as definition)
ALLOWED_UNIT_MAP_MODES = frozenset({"grade1_only", "matching_readout_only"})

# Trestleboard nuclear forced class (grammar_energy); k=Δ⁶ aperture, ℓ=ρ² dress.
NUCLEAR_GRADE_K = 6
NUCLEAR_DRESS_L = 2
NUCLEAR_STF = True       # 1/√5 STF bulk equipartition
NUCLEAR_EQUATOR = True   # 2^(C₃Δ²) equatorial tick
NUCLEAR_CLASS = (
    "(6,2) nuclear forced",
    NUCLEAR_GRADE_K,
    NUCLEAR_DRESS_L,
    NUCLEAR_STF,
    NUCLEAR_EQUATOR,
)

Q_G = 4.0 * math.pi
M_A = float(M_A_CORE)
QG_MA2 = Q_G * (M_A ** 2)
# CGM stage thresholds (CGM_Logic §9): CS / UNA / ONA chirality angles.
S_P = math.pi / 2.0
U_P = 1.0 / math.sqrt(2.0)
O_P = math.pi / 4.0
S_CS = S_P / M_A
S_UNA = U_P / M_A
S_ONA = O_P / M_A
GENE_MIC = GENE_MIC_S

# Formalism inventory cardinalities (hQVM_Specs_Formalism § Spaces and primitive operations).
# Scripts must route through these named spaces / GENE_Mic transcription / family+payload
# extractors — not invent abstract Hilbert spaces or hand-rolled bit splits.
D_PAYLOAD = CHIRALITY_QUBITS_6  # GF(2)^6
BYTE256 = alphabet_size(D_PAYLOAD)  # 4 families × 2^6
K4_ORDER = len(GATE_NAMES)  # family gauge labels → {id,S,C,F}
Q8_ORDER = 2 * K4_ORDER  # central Z₂ extension of K4
TRANSPORT_SIZE = 1 << D_PAYLOAD  # |GF(2)^6| = |q6| = |χ|
SHELLS7 = D_PAYLOAD + 1
MASK_CODE_SIZE = TRANSPORT_SIZE  # |C64|
assert BYTE256 == 256 and K4_ORDER == 4 and Q8_ORDER == 8 and TRANSPORT_SIZE == 64
assert OMEGA_SIZE == TRANSPORT_SIZE * TRANSPORT_SIZE
assert HORIZON_SIZE == TRANSPORT_SIZE

# APERTURE_GAP_Q256 is the integer tick-count on T_256^(frac); it must be 0<tick<BYTE256.
assert isinstance(APERTURE_GAP_Q256, int) and 0 < APERTURE_GAP_Q256 < BYTE256
A_KERNEL = APERTURE_GAP_Q256 / float(BYTE256)  # = 5/256
# External empirical UV anchor: Planck scale E_CS (CGM_Units.md §4, optical conjugacy root)
E_PLANCK_GEV = 1.22e19  # GeV, PDG/CODATA-scale input

N_SPATIAL_CGM = 3
N_TEMPORAL_CGM = 1
D_CONTINUUM_CGM = N_SPATIAL_CGM + N_TEMPORAL_CGM
BU_CLOSURE_DEPTH = 4
SE3_DOF = 6

# Defining KS point: g_R(1)^2 = Delta^0 = 1 (Theorem AF-Ruler, k=1 on Delta-ruler)
G_DEFINING_KS: float = 1.0
BETA_DEFINING: float = 1.0 / (G_DEFINING_KS * G_DEFINING_KS)

# SC / SU(2) floor constants (finite K4/Q8 charts; continuous bound for SC0-G-cont).
CASIMIR_J_HALF = 0.75  # j(j+1) for j=1/2
WILSON_V_DEV_INF = 1.0  # |V − V(1)| ≤ 1 for fundamental Wilson V ∈ [0, 2]
C_G_SU2_CONT = WILSON_V_DEV_INF / CASIMIR_J_HALF  # = 4/3
C_SHARP_FINITE = 1.0 / math.sqrt(3.0)  # K4/Q8 free-plaquette sharp C
# 2D square: each link in exactly two plaquettes (use _r_star_incidence on 3D trees).
R_STAR_2D = 2

# Lattice Hilbert spaces (l^2(G^E), l^2(G^{E_3D})) are Clay-certificate spaces,
# NOT kernel spaces. The kernel space is Omega (4096 states). The lattice provides
# the constructive QFT framework (OS-RP, clustering, infinite-volume limit)
# on top of the GNS vacuum from Omega.


def permute_payload_byte(byte: int, perm: tuple[int, ...], d: int = D_PAYLOAD) -> int:
    """S_d action on payload bits via Formalism primitives; family (L0) fixed.

    intron = byte ⊕ GENE_Mic; family = L0 bits; payload = bits 1..d;
    permute payload indices; reassemble via byte_from_family_micro.
    """
    if len(perm) != d:
        raise ValueError(f"perm length {len(perm)} != d={d}")
    if d == D_PAYLOAD:
        intron = byte_to_intron(int(byte) & 0xFF)
        fam = intron_family(intron)
        payload = intron_micro_ref(intron)
    else:
        from gyroscopic.hQVM.family import intron_family_d, intron_from_byte, intron_micro_ref_d

        intron = intron_from_byte(int(byte), d)
        fam = intron_family_d(intron, d)
        payload = intron_micro_ref_d(intron, d)
    payload_perm = 0
    for i in range(d):
        payload_perm |= ((payload >> i) & 1) << perm[i]
    return int(byte_from_family_micro(fam, payload_perm, d))


def progress(label: str) -> None:
    """Live stage cue on stderr (also teed to results file by Yang_Mills_Mass_Gap_run.py)."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {label}", file=sys.stderr, flush=True)


# CGM Delta-ruler prediction (shared readout; section certificates live in numbered scripts)
# -----------------------------------------------------------------
def optical_conjugacy_ir_ladder() -> dict[str, float]:
    """IR conjugates from optical conjugacy (CGM_Units §4): E^UV * E^IR = (E_CS*E_EW)/(4 pi^2).

    Stage UV ratios from geometric thresholds (Units §3.2) for CS/UNA/ONA:
    E_UNA/E_CS = u_p/s_p, E_ONA/E_CS = 1/2. BU is the IR electroweak anchor
    (E_BU_IR := E_EW), so E_BU_UV = conjugacy_product / E_EW. Returns GeV values
    derived from E_PLANCK and E_EW; no free fit parameters.
    """
    s_p = S_P
    u_p = U_P
    o_p = O_P
    # Denominator is 4π², not (4π)² = Q_G² (Analysis_Energy_Scales §3.3; CGM_Units §4).
    conj = (E_PLANCK_GEV * E_EW_GEV) / (4.0 * math.pi ** 2)
    e_cs_uv = E_PLANCK_GEV
    e_una_uv = e_cs_uv * (u_p / s_p)
    e_ona_uv = e_cs_uv * (o_p / s_p)
    e_bu_uv = conj / E_EW_GEV  # BU IR is the EW anchor
    return {
        "E_CS_IR_GeV": conj / e_cs_uv,
        "E_UNA_IR_GeV": conj / e_una_uv,
        "E_ONA_IR_GeV": conj / e_ona_uv,
        "E_BU_IR_GeV": conj / e_bu_uv,  # = E_EW by construction
        "E_BU_UV_GeV": e_bu_uv,
        "conjugacy_product_GeV2": conj,
    }


def cgm_ym_gap_prediction() -> dict:
    """Dimensionful YM mass-gap candidates on the CGM Delta-ruler.

    All quantities are derived from existing corpus constants (hqvm_compact_geom_core,
    CGM_Units, Compact_Geometry §2.6). Two routes are reported:

    Route A (2-form curvature grade, Compact_Geometry C2=15):
        m_A = C2 * v * Delta^2
    Route B (CS action x spinorial-2 * Delta^2 strong grade):
        m_B = S_CS * (2 * v * Delta^2)

    Route A uses the forced 2-form multiplicity of the hexacode chart (C2 = C(6,2) = 15)
    on the first curvature moment Delta^2. Route B uses the CS stage action and the
    spinorial double-cover factor 2. Both land in the pure-glue GeV neighborhood;
    agreement is a cross-check, not a fit. The association "YM gap = first 2-form
    excitation on the Delta-ruler" is the theorem target to prove; these numbers are
    the a-priori readout the corpus already forces once that association is adopted.
    """
    v_d2 = float(E_EW_GEV) * (DELTA ** 2)
    m_route_a = float(CODE_C2) * v_d2
    lambda_strong = 2.0 * v_d2
    m_route_b = S_CS * lambda_strong
    ir = optical_conjugacy_ir_ladder()
    return {
        "v_GeV": float(E_EW_GEV),
        "Delta": float(DELTA),
        "Delta2": float(DELTA ** 2),
        "v_Delta2_GeV": v_d2,
        "C1": int(CODE_C1),
        "C2": int(CODE_C2),
        "C3": int(CODE_C3),
        "S_CS": float(S_CS),
        "route_A_C2_v_Delta2_GeV": m_route_a,
        "route_B_S_CS_times_2v_Delta2_GeV": m_route_b,
        "lambda_strong_2v_Delta2_GeV": lambda_strong,
        "rel_diff_A_vs_B": abs(m_route_a - m_route_b) / max(m_route_a, m_route_b),
        "IR": ir,
        "A_star_continuous": float(DELTA),
        "A_kernel_discrete": float(A_KERNEL),
        "collapsed_lock_0p5": float(QG_MA2),
        "QG_ma2": float(QG_MA2),
        "GENE_Mic": int(GENE_MIC),
    }


def gap_record(
    operator: str,
    gap: float,
    E0: float,
    vac_mult: int,
    *,
    n_distinct: int | None = None,
    status: str = "ok",
) -> dict[str, Any]:
    """Typed gap payload. operator must be H_phys or H_shadow (or labeled variant)."""
    if not (operator.startswith("H_phys") or operator.startswith("H_shadow")):
        raise ValueError(f"gap_record: unknown operator label {operator!r}")
    gap_f = float(gap)
    e0_f = float(E0)
    out: dict[str, Any] = {
        "operator": operator,
        "gap": gap_f,
        "E0": e0_f,
        "vac_mult": int(vac_mult),
        "status": status,
    }
    if n_distinct is not None:
        out["n_distinct"] = int(n_distinct)
    return out


def matching_readout_only(gap_dimless: float, pred: dict) -> dict:
    """MATCHING ONLY: E_unit_match = m_pred / gap. Never use as the unit definition."""
    if gap_dimless is None or not (gap_dimless > 0) or math.isnan(gap_dimless):
        return {"ok": False, "unit_map_mode": "matching_readout_only"}
    m_a = pred["route_A_C2_v_Delta2_GeV"]
    m_b = pred["route_B_S_CS_times_2v_Delta2_GeV"]
    return {
        "unit_map_mode": "matching_readout_only",
        "label": "MATCHING ONLY — not a unit definition",
        "E_unit_match_route_A_GeV": m_a / gap_dimless,
        "E_unit_match_route_B_GeV": m_b / gap_dimless,
        "m_match_route_A_GeV": m_a,
        "m_match_route_B_GeV": m_b,
        "ok": True,
    }


def curvature_index_kappa2(gap_dimless: float) -> dict[str, float]:
    """Curvature index κ₂: Track-D dimensionless gap in grade-2 ruler coordinates.

    E_unit := vΔ, m_phys := gap_dimless · E_unit, grade-2 energy := vΔ².
    κ₂ := m_phys / (vΔ²) = gap_dimless / Δ.
    Chart readout only. Continuum mass uses C₂ as multiplicity (Route A), not κ₂ → C₂.
    """
    gd = float(gap_dimless)
    if not (gd > 0.0) or math.isnan(gd):
        return {
            "kappa2": float("nan"),
            "kappa2_target_C2": float(CODE_C2),
            "kappa2_rel_err": float("nan"),
        }
    k2 = gd / float(DELTA)
    tgt = float(CODE_C2)
    rel = abs(k2 - tgt) / tgt if tgt > 0 else float("nan")
    return {"kappa2": k2, "kappa2_target_C2": tgt, "kappa2_rel_err": rel}


def unit_map_trackd_to_gev(
    gap_dimless: float,
    pred: dict,
    *,
    unit_map_mode: str | None = None,
) -> dict:
    """Grade-1 unit map: E_unit = v*Delta (independent of gap); m_phys = gap * E_unit.

    Routes A/B are a priori ruler placements. Matching E_unit = m_pred/gap is only
    available via matching_readout_only() and must print MATCHING ONLY.
    """
    mode = UNIT_MAP_MODE if unit_map_mode is None else unit_map_mode
    assert mode in ALLOWED_UNIT_MAP_MODES, f"unit_map_mode={mode!r} not allowed"
    if gap_dimless is None or not (gap_dimless > 0) or math.isnan(gap_dimless):
        return {"gap_dimless": gap_dimless, "ok": False, "unit_map_mode": mode}
    if mode == "matching_readout_only":
        mr = matching_readout_only(gap_dimless, pred)
        mr["gap_dimless"] = float(gap_dimless)
        return mr
    # grade1_only
    e_unit = float(E_EW_GEV) * float(DELTA)
    m_a = pred["route_A_C2_v_Delta2_GeV"]
    m_b = pred["route_B_S_CS_times_2v_Delta2_GeV"]
    m_phys = float(gap_dimless) * e_unit
    mr = matching_readout_only(gap_dimless, pred)
    k2 = curvature_index_kappa2(gap_dimless)
    gap_pred = float(CODE_C2) * float(DELTA)
    return {
        "gap_dimless": float(gap_dimless),
        "unit_map_mode": "grade1_only",
        "E_unit_GeV": e_unit,
        "m_phys_GeV": m_phys,
        "m_pred_A_GeV": m_a,
        "m_pred_B_GeV": m_b,
        "rel_to_route_A": abs(m_phys - m_a) / m_a if m_a > 0 else float("nan"),
        "rel_to_route_B": abs(m_phys - m_b) / m_b if m_b > 0 else float("nan"),
        # Matching arithmetic kept only under explicit label (never as definition)
        "E_unit_match_route_A_GeV": mr.get("E_unit_match_route_A_GeV"),
        "E_unit_match_route_B_GeV": mr.get("E_unit_match_route_B_GeV"),
        "m_match_route_A_GeV": m_a,
        "matching_label": mr.get("label"),
        "unit_method": "grade_1_vDelta",
        "ok": True,
        "m_phys_in_1_2_GeV": 1.0 < m_phys < 2.0,
        # Λ² lock diagnostics (report; do not force equality)
        "kappa2_gap_over_Delta": k2["kappa2"],
        "kappa2_target_C2": k2["kappa2_target_C2"],
        "kappa2_rel_err": k2["kappa2_rel_err"],
        "gap_pred_from_C2Delta": gap_pred,
        "rel_gap_vs_C2Delta": abs(float(gap_dimless) - gap_pred) / gap_pred,
    }


def section(number: int, title: str) -> None:
    """Numbered section header for publication-ordered output."""
    print("-" * 5)
    print(f"{number}. {title}")


def section_title(number: int) -> str:
    """Title body from RUN_SECTIONS[number] (strip leading 'N. ' prefix)."""
    line = RUN_SECTIONS[number]
    # "14. GATE SUMMARY" or "0. PREFLIGHT — ..."
    if ". " in line:
        return line.split(". ", 1)[1]
    return line


def gate(label: str, ok: bool, detail: str = "") -> None:
    """Single PASS/FAIL gate line."""
    status = "PASS" if ok else "FAIL"
    if detail:
        print(f"gate ({label}): {status}  {detail}")
    else:
        print(f"gate ({label}): {status}")


# Derivation order (deliverable certificates only)
RUN_SECTIONS: tuple[str, ...] = (
    # _1 — JW object + Wilson lattice
    "0. PREFLIGHT — hQVM kernel cross-check",
    "1. JW-A — definitions (A, omega) -> (H, pi, Omega)",
    "2. DELTA-RULER — dimensionful mass-gap prediction (Routes A/B)",
    "3. SHADOW Delta_W — formula n/(2(n-1)) on Omega [Track B]",
    "4. SHADOW LOCK — Q_G m_a^2 vs collapsed B/C gap",
    "5. GAUGE GROUP — K4 holonomy + Q8 central extension",
    "6. WILSON H — Kogut-Susskind algebra certificates",
    "7. Q8 PLAQUETTE — non-abelian JW gap",
    "8. CORRELATOR / LOCALITY / TRANSFER / GENE_Mic",
    "9. UNIT MAP — grade-1 E_unit = v Delta",
    "10. SPACETIME — D=4 packaging",
    # _2 — Euclidean OS Gram
    "11. OS EUCLIDEAN GRAM (Θ measure)",
    # _3 — SC + H6 + D
    "12. TWO-PLAQUETTE CONJUGACY POSITIVITY",
    "13. SC1 LOCAL EXCITATION (abelian K4)",
    "14. SC0-Q FREE PLAQUETTE",
    "15. LEMMA L' + LOCAL GAUGE U_g + SC1-Q",
    "16. H6 CLUSTERING FROM SC1 FLOOR",
    "17. CGM NATIVE GAP CHECKS",
    "18. D1/D2/D3 — Euclidean transfer / omega_inf / GENE_Mic",
    "19. INFINITE-VOLUME / OS CHECKLIST",
    # _4 — Λ² lock + intertwiner
    "20. MAGNETIC EXCITATION DEGREE-2 + Lambda2 LOCK",
    # _5 — Formalism Clay checklist + H7
    "21. FORMALISM CHECKLIST — G / R4 / measure / Hilbert / gap / Hopf fiber",
    "22. H7 FORMALISM AGGREGATE (H7_closed := formalism_checklist_closed)",
)

# -----------------------------------------------------------------
# Lattice engine (Wilson Kogut-Susskind) — shared 2D core + helpers
# -----------------------------------------------------------------
import itertools
import time
from typing import TypeAlias

import numpy as np
from scipy.sparse import csr_matrix, eye as sp_eye, diags
from scipy.sparse.linalg import eigsh, LinearOperator, ArpackNoConvergence

DENSE_QR_THRESHOLD = 4096
Site2: TypeAlias = tuple[int, int]

def make_group(mult_dict, identity: str | None = None):
    """Build (elements, index, mult_table, inv_table) from a dict mult[g][h]=gh.

    Robust identity detection: unique e such that e*g=g and g*e=g for all g.
    Identity is placed at index 0.
    """
    G0 = list(mult_dict.keys())

    if identity is None:
        for cand in G0:
            ok = True
            for g in G0:
                if mult_dict[cand][g] != g or mult_dict[g][cand] != g:
                    ok = False
                    break
            if ok:
                identity = cand
                break
        if identity is None:
            raise ValueError("make_group: could not detect identity element")

    G = [identity] + [g for g in G0 if g != identity]
    N = len(G)
    gi = {g: i for i, g in enumerate(G)}

    table = np.zeros((N, N), dtype=int)
    for a in G:
        for h in G:
            table[gi[a], gi[h]] = gi[mult_dict[a][h]]

    inv = np.zeros(N, dtype=int)
    e = identity
    for a in G:
        ia = gi[a]
        found = False
        for h in G:
            if mult_dict[a][h] == e and mult_dict[h][a] == e:
                inv[ia] = gi[h]
                found = True
                break
        if not found:
            raise ValueError(f"make_group: no inverse found for {a}")

    return G, gi, table, inv


def K4():
    """Klein four (Z/2)^2: the doc's gauge group {id,S,C,F}. Abelian."""
    M = {
        "1": {"1": "1", "a": "a", "b": "b", "c": "c"},
        "a": {"1": "a", "a": "1", "b": "c", "c": "b"},
        "b": {"1": "b", "a": "c", "b": "1", "c": "a"},
        "c": {"1": "c", "a": "b", "b": "a", "c": "1"},
    }
    return make_group(M)


def Q8():
    """Reference quaternion table for cross-checking Q8_from_extension().

    The construction used in the proof is Q8_from_extension() (derived from
    K4 + cocycle_k4). This hardcoded table validates that construction.
    """
    M = {
        "1": {"1": "1", "-1": "-1", "i": "i", "-i": "-i", "j": "j", "-j": "-j", "k": "k", "-k": "-k"},
        "-1": {"1": "-1", "-1": "1", "i": "-i", "-i": "i", "j": "-j", "-j": "j", "k": "-k", "-k": "k"},
        "i": {"1": "i", "-1": "-i", "i": "-1", "-i": "1", "j": "k", "-j": "-k", "k": "-j", "-k": "j"},
        "-i": {"1": "-i", "-1": "i", "i": "1", "-i": "-1", "j": "-k", "-j": "k", "k": "j", "-k": "-j"},
        "j": {"1": "j", "-1": "-j", "i": "-k", "-i": "k", "j": "-1", "-j": "1", "k": "i", "-k": "-i"},
        "-j": {"1": "-j", "-1": "j", "i": "k", "-i": "-k", "j": "1", "-j": "-1", "k": "-i", "-k": "i"},
        "k": {"1": "k", "-1": "-k", "i": "j", "-i": "-j", "j": "-i", "-j": "i", "k": "-1", "-k": "1"},
        "-k": {"1": "-k", "-1": "k", "i": "-j", "-i": "j", "j": "i", "-j": "-i", "k": "1", "-k": "-1"},
    }
    return make_group(M)


def cocycle_k4():
    """Nontrivial 2-cocycle on K4 = (Z/2)^2 giving the Q8 extension (not D4).

    Represent K4 elements as 2-bit vectors over the {a,b} basis: id=(0,0), a=(1,0),
    b=(0,1), c=a+b=(1,1). The cocycle is the bilinear form omega(x,y) = x^T Omega y
    (mod 2) with Omega = [[1,0],[1,1]]. This makes:
        i^2 = (a,0)^2 = (a^2, omega(a,a)) = (id,1) = -1,
        j^2 = (b,0)^2 = (id, omega(b,b)) = -1,
        k^2 = (c,0)^2 = (id, omega(c,c)) with omega(c,c)=1 -> -1,
        i j = (a,0)(b,0) = (c, omega(a,b)) = (c,0) = k,
        j i = (b,0)(a,0) = (c, omega(b,a)) = (c,1) = -k.
    So the extension is Q8, not the abelian K4 x Z2. The cocycle is the BU-fold
    holonomy sign (rest vs swapped, the Z2 sheet flip) that makes SU(2) the double
    cover of SO(3) in the corpus.
    """
    G, gi, table, inv = K4()
    basis_vec = {gi["1"]: (0, 0), gi["a"]: (1, 0), gi["b"]: (0, 1), gi["c"]: (1, 1)}
    Omega = np.array([[1, 0], [1, 1]], dtype=int)
    om = np.zeros((4, 4), dtype=int)
    for x in range(4):
        for y in range(4):
            vx, vy = np.array(basis_vec[x]), np.array(basis_vec[y])
            om[x, y] = int((vx @ Omega @ vy) % 2)
    return G, gi, om


def Q8_from_extension():
    """Construct Q8 as the central extension 1 -> Z2 -> Q8 -> K4 -> 1.

    Elements are pairs (k, z) with k in K4, z in Z2, multiplication
        (k1, z1) * (k2, z2) = (k1*k2, z1 ^ z2 ^ omega(k1, k2)).
    omega is the nontrivial cocycle (cocycle_k4). The result is isomorphic to Q8
    (not D4): center = {1,-1} = Z2, and every non-central element has order 4.
    Returns a make_group-compatible (G, gi, table, inv) with quaternion labels.
    """
    Gk, gi_k, om = cocycle_k4()
    k4_names = ["1", "a", "b", "c"]
    ki_of = {name: i for i, name in enumerate(k4_names)}
    # K4 product by name lookup
    _, _, table_k4, _ = K4()
    N = 8
    idx = {(ki, z): 2 * ki + z for ki in range(4) for z in range(2)}
    table = np.zeros((N, N), dtype=int)
    for (k1, z1), i in idx.items():
        for (k2, z2), j in idx.items():
            kp = int(table_k4[k1, k2])  # K4 product index (k1,k2 are 0..3)
            zp = (z1 ^ z2 ^ om[k1, k2]) & 1
            table[i, j] = idx[(kp, zp)]
    # quaternion labels
    z2name = {(0, 0): "1", (0, 1): "-1",
              (1, 0): "i", (1, 1): "-i",
              (2, 0): "j", (2, 1): "-j",
              (3, 0): "k", (3, 1): "-k"}
    G = [z2name[(ki, z)] for ki in range(4) for z in range(2)]
    inv = np.zeros(N, dtype=int)
    for i in range(N):
        for j in range(N):
            if table[i, j] == 0 and table[j, i] == 0:
                inv[i] = j
    mult = {G[i]: {G[j]: G[table[i, j]] for j in range(N)} for i in range(N)}
    return make_group(mult)


# -----------------------------------------------------------------
# Irrep characters / Wilson weights
# -----------------------------------------------------------------
def wilson_weight_Q8_2d():
    """Wilson plaquette weight V_R for Q8 using its 2D (SU(2)) irrep.

    chi_2d: chi(1)=2, chi(-1)=-2, chi(others)=0.  V_R(g) = 1 - Re chi(g)/2.
    -> 0 for hol=1 (flat), 2 for hol=-1 (central), 1 for i,j,k and negatives.
    Returns (group, array V_R indexed by group index).
    """
    G, gi, table, inv = Q8()
    chi = {"1": 2.0, "-1": -2.0, "i": 0.0, "-i": 0.0, "j": 0.0, "-j": 0.0, "k": 0.0, "-k": 0.0}
    dR = 2.0
    V = np.array([1.0 - chi[g] / dR for g in G])
    return (G, gi, table, inv), V


def wilson_weight_K4(char_signs=("a", "c")):
    """Wilson plaquette weight for K4 from a nontrivial 1D irrep.

    char_signs selects which K4 elements get character -1 (the rest +1). Default
    {a, c} gives character chi(1)=chi(b)=+1, chi(a)=chi(c)=-1. V_R = 1 - chi.
    Returns (group, array V_R indexed by group index).
    """
    G, gi, table, inv = K4()
    neg = set(char_signs)
    chi = {g: (-1.0 if g in neg else 1.0) for g in G}
    V = np.array([1.0 - chi[g] for g in G])
    return (G, gi, table, inv), V


# -----------------------------------------------------------------
# Lattice
# -----------------------------------------------------------------
class LatticeYM:
    """Lx x Ly lattice. periodic=True wraps edges (torus, topological degeneracy);
    periodic=False is open (unique vacuum).

    Edge set: Ux(i,j):(i,j)->(i+1,j); Uy(i,j):(i,j)->(i,j+1). Open BC only includes
    edges whose both endpoints exist: |E|=(Lx-1)Ly + Lx(Ly-1).

    Plaquette at (i,j): Ux(i,j) Uy(i+1,j) Ux(i,j+1)^{-1} Uy(i,j)^{-1}.
    """

    def __init__(self, Lx: int, Ly: int, group, periodic: bool = False):
        self.Lx = Lx
        self.Ly = Ly
        self.periodic = periodic
        self.G, self.gi, self.table, self.inv = group
        self.N = len(self.G)
        if self.periodic:
            self.h_coords = [(i, j) for i in range(Lx) for j in range(Ly)]
            self.v_coords = [(i, j) for i in range(Lx) for j in range(Ly)]
        else:
            self.h_coords = [(i, j) for i in range(Lx - 1) for j in range(Ly)]
            self.v_coords = [(i, j) for i in range(Lx) for j in range(Ly - 1)]
        self.nH = len(self.h_coords)
        self.nV = len(self.v_coords)
        self.nE = self.nH + self.nV
        self.he = {c: k for k, c in enumerate(self.h_coords)}
        self.ve = {c: self.nH + k for k, c in enumerate(self.v_coords)}
        self._abelian = bool(np.all(self.table == self.table.T))

    def vertex_edges(self, i, j):
        """Incident oriented edges at vertex (i,j): outgoing +1, incoming -1."""
        edges = []
        if self.periodic:
            im = (i - 1) % self.Lx
            edges.append((self.he[(im, j)], -1))
            edges.append((self.he[(i, j)], +1))
            jm = (j - 1) % self.Ly
            edges.append((self.ve[(i, jm)], -1))
            edges.append((self.ve[(i, j)], +1))
        else:
            if i - 1 >= 0 and (i - 1, j) in self.he:
                edges.append((self.he[(i - 1, j)], -1))
            if (i, j) in self.he:
                edges.append((self.he[(i, j)], +1))
            if j - 1 >= 0 and (i, j - 1) in self.ve:
                edges.append((self.ve[(i, j - 1)], -1))
            if (i, j) in self.ve:
                edges.append((self.ve[(i, j)], +1))
        return edges

    def plaquette_edges(self, i, j):
        """Standard Wilson plaquette: Ux(i,j) Uy(i+1,j) Ux(i,j+1)^{-1} Uy(i,j)^{-1}."""
        if self.periodic:
            ip = (i + 1) % self.Lx
            jp = (j + 1) % self.Ly
            return [
                (self.he[(i, j)], +1),
                (self.ve[(ip, j)], +1),
                (self.he[(i, jp)], -1),
                (self.ve[(i, j)], -1),
            ]
        if i + 1 >= self.Lx or j + 1 >= self.Ly:
            return None
        return [
            (self.he[(i, j)], +1),
            (self.ve[(i + 1, j)], +1),
            (self.he[(i, j + 1)], -1),
            (self.ve[(i, j)], -1),
        ]

    # ---- group actions ----
    def vertex_action_indices(self, edges, h):
        """Standard gauge at a vertex: outgoing U->hU, incoming U->U h^{-1}."""
        N = self.N
        ih = self.inv[h]
        dim = N ** self.nE
        newflat = np.arange(dim)
        for (e, s) in edges:
            base = N ** e
            digit = (newflat // base) % N
            if s == +1:
                newd = self.table[h, digit]
            else:
                newd = self.table[digit, ih]
            newflat = newflat + (newd - digit) * base
        return newflat

    def _left_indices(self, edges, h):
        """Electric left-regular action on listed links (not a gauge transform)."""
        N = self.N
        dim = N ** self.nE
        newflat = np.arange(dim)
        for (e, _s) in edges:
            base = N ** e
            digit = (newflat // base) % N
            newflat = newflat + (self.table[h, digit] - digit) * base
        return newflat

    def gauge_projector_matvec(self, x):
        N = self.N
        y = x.copy()
        for i in range(self.Lx):
            for j in range(self.Ly):
                edges = self.vertex_edges(i, j)
                if not edges:
                    continue
                acc = np.zeros_like(y)
                for h in range(N):
                    acc[self._vertex_perm(edges, h)] += y
                y = acc / N
        return y

    def _link_perm(self, e, h):
        """Cached index permutation for link left-mult by h."""
        if not hasattr(self, "_link_perm_cache"):
            self._link_perm_cache = {}
        key = (e, h)
        if key not in self._link_perm_cache:
            self._link_perm_cache[key] = self._left_indices([(e, +1)], h)
        return self._link_perm_cache[key]

    def _vertex_perm(self, edges, h):
        if not hasattr(self, "_vertex_perm_cache"):
            self._vertex_perm_cache = {}
        key = (tuple(edges), h)
        if key not in self._vertex_perm_cache:
            self._vertex_perm_cache[key] = self.vertex_action_indices(edges, h)
        return self._vertex_perm_cache[key]

    def _link_avg_matvec(self, x, e):
        """Apply (1/|G|) sum_h L_h^(e) via cached index permutations."""
        N = self.N
        acc = np.zeros_like(x)
        for h in range(N):
            acc[self._link_perm(e, h)] += x / N
        return acc

    def electric_matvec(self, x):
        """sum_e (I - avg L_h^(e)) @ x without building sparse matrices."""
        y = np.zeros_like(x)
        for e in range(self.nE):
            y += x - self._link_avg_matvec(x, e)
        return y

    def hamiltonian_matvec_operator(self, g: float, V):
        """P H P as LinearOperator using matvec only (no dim x dim assembly)."""
        mag = self.magnetic_diagonal(V)
        ge = g * g / 2.0
        gm = 1.0 / (2.0 * g * g)
        dim = self.N ** self.nE
        P = self.gauge_projector_matvec

        def _mag_apply(x):
            return (mag * x) if x.ndim == 1 else (mag[:, np.newaxis] * x)

        def Hx(x):
            return ge * self.electric_matvec(x) + gm * _mag_apply(x)

        def HP(x):
            return P(Hx(P(x)))

        return LinearOperator(shape=(dim, dim), matvec=HP, dtype=float), mag  # type: ignore[call-arg]

    # ---- electric: link Laplacian / Casimir ----
    def elec_op_links(self):
        N = self.N
        dim = N ** self.nE
        H = None
        for e in range(self.nE):
            edges = [(e, +1)]
            A = None
            for h in range(N):
                newflat = self._left_indices(edges, h)
                col = np.arange(dim)
                blk = csr_matrix((np.ones(dim), (newflat, col)), shape=(dim, dim))
                A = blk if A is None else A + blk
            A = A / N  # type: ignore[operator]
            He_e = (sp_eye(dim, format="csr") - A)
            H = He_e if H is None else H + He_e
        return H  # sum_e (I - avg L_h^(e)); no 1/|E| division

    # ---- magnetic: Wilson plaquette weight (class function) ----
    def magnetic_diagonal(self, V):
        """Diagonal H_mag = sum_p V_R(hol(p)). V is array indexed by group index."""
        N = self.N
        dim = N ** self.nE
        mag = np.zeros(dim)
        flat = np.arange(dim)
        for i in range(self.Lx):
            for j in range(self.Ly):
                pedges = self.plaquette_edges(i, j)
                if pedges is None:
                    continue
                hol = np.zeros(dim, dtype=int)
                for (e, s) in pedges:
                    base = N ** e
                    digit = (flat // base) % N
                    if s == +1:
                        hol = self.table[hol, digit]
                    else:
                        hol = self.table[hol, self.inv[digit]]
                mag += V[hol]
        return mag

    def hamiltonian_operator(self, g: float, V):
        He = self.elec_op_links()
        Hm = diags(self.magnetic_diagonal(V), format="csr")
        H = (g * g / 2.0) * He + (1.0 / (2.0 * g * g)) * Hm  # type: ignore[operator]
        H = H.tocsr()
        Pmatvec = self.gauge_projector_matvec

        def HP(x):
            return Pmatvec(H @ Pmatvec(x))

        return LinearOperator(shape=H.shape, matvec=HP), H, He, Hm  # type: ignore[call-arg]


# -----------------------------------------------------------------
# Exact diagonalization (small lattices) on the gauge-invariant subspace
# -----------------------------------------------------------------
def _gauge_invariant_basis(lat, dim, max_dim: int = 8192):
    """Orthonormal Q spanning im(P) via incremental Gram-Schmidt.

    Memory O(dim * rank). Never stacks all P|k> (that OOMs at dim~4k via SVD).
    Refuses dim > max_dim — use orbit/tree reduction or sparse eigsh instead.
    """
    if dim > max_dim:
        raise MemoryError(
            f"gauge_invariant_basis: dim={dim} > {max_dim}; use orbit/tree/sparse path"
        )
    basis: list[np.ndarray] = []
    for kk in range(dim):
        e = np.zeros(dim)
        e[kk] = 1.0
        pk = lat.gauge_projector_matvec(e)
        for b in basis:
            pk = pk - b * float(b @ pk)
        nrm = float(np.linalg.norm(pk))
        if nrm > 1e-10:
            basis.append(pk / nrm)
    if not basis:
        return np.zeros((dim, 0))
    return np.column_stack(basis)


def _hred_on_basis(H, Q):
    """Reduced Hamiltonian Hred = Q^T H Q via sparse H matvec (no H.toarray())."""
    Hred = Q.T @ (H @ Q)
    return (Hred + Hred.T) / 2


def _gauge_invariant_hred(lat, g=G_DEFINING_KS, V=None):
    """Shared setup: sparse H from hamiltonian_operator and dense Hred on basis Q."""
    if V is None:
        _, V = wilson_weight_K4()
    dim = lat.N ** lat.nE
    _op, H, _He, _Hm = lat.hamiltonian_operator(g, V)
    Q = _gauge_invariant_basis(lat, dim)
    Hred = _hred_on_basis(H, Q)
    return Hred, Q


def gauge_invariant_spectrum(lat, g=G_DEFINING_KS, V=None):
    """True gauge-invariant spectrum via QR reduction of the projected space.

    Building P H P on the full config space and diagonalizing it is misleading:
    the orthogonal complement of the gauge-invariant subspace contributes spurious
    zero eigenvalues. Instead we collect the projected basis vectors P|k>, QR them to
    an orthonormal gauge-invariant basis Q, and diagonalize Q^T H Q (since P Q = Q).
    Returns (eigenvalues sorted, JW gap E1-E0, vacuum_multiplicity, E0). Correct for any group.
    """
    Hred, _Q = _gauge_invariant_hred(lat, g, V)
    w = np.sort(np.linalg.eigvalsh(Hred))
    e0, gap, vac = jw_gap_from_w(w)
    return w, gap, vac, e0


def gauge_invariant_reduce(lat, g=G_DEFINING_KS, V=None):
    """Gauge-invariant subspace: (eigenvalues, eigenvectors, gap, vac_mult, vacuum_energy, Q).

    Builds orthonormal Q via projected QR, diagonalizes Q^T H Q. Used by correlators
    and spectrum routines to avoid duplicate QR logic. gap is JW Delta = E1-E0.
    """
    Hred, Q = _gauge_invariant_hred(lat, g, V)
    wr, Vr = np.linalg.eigh(Hred)
    order = np.argsort(wr)
    wr = wr[order]
    Vr = Vr[:, order]
    e0, gap, vac = jw_gap_from_w(wr)
    return wr, Vr, gap, vac, e0, Q


def gauge_invariant_lowest_gap(lat, g=G_DEFINING_KS, V=None, k=4, sigma=None, return_timing=False):
    """Lowest spectral gap of P H P via matvec-only eigsh (no QR, no H assembly).

    Suitable when dim = |G|^|E| exceeds dense QR (e.g. Q8 open Lx=3, dim=8^6=262144).
    Uses shift-invert near wilson_gap(V) and SA fallback. Returns eigenvalues, gap,
    vacuum_energy; with return_timing also vacuum_mult and seconds.
    """
    if V is None:
        _, V = wilson_weight_K4()
    t0 = time.perf_counter()
    dim = lat.N ** lat.nE
    op, _mag = lat.hamiltonian_matvec_operator(g, V)
    rng = np.random.default_rng(0)
    v0 = lat.gauge_projector_matvec(rng.standard_normal(dim))
    nrm = np.linalg.norm(v0)
    v0 = v0 / nrm if nrm > 1e-15 else rng.standard_normal(dim)
    wg = wilson_gap(V)
    sig_list = [sigma] if sigma is not None else [max(0.12, wg * 0.45), max(0.2, wg * 0.7)]
    best_gap = float("nan")
    best_w = None
    for sig in sig_list:
        try:
            w, _ = eigsh(op, k=k, sigma=sig, which="LM", tol=1e-7, maxiter=800, v0=v0)  # type: ignore[arg-type]
            w = np.sort(w)
            e0 = w[0]
            pos = w[np.abs(w - e0) > 1e-5]
            gap = float(pos[0] - e0) if pos.size else float("nan")
            if pos.size and (np.isnan(best_gap) or gap < best_gap):
                best_gap = gap
                best_w = w
        except (ArpackNoConvergence, RuntimeError):
            continue
    if best_w is None:
        try:
            w, _ = eigsh(op, k=min(k, 3), which="SA", tol=1e-7, maxiter=400, v0=v0)  # type: ignore[arg-type]
            w = np.sort(w)
            e0 = w[0]
            pos = w[np.abs(w - e0) > 1e-5]
            gap = float(pos[0] - e0) if pos.size else float("nan")
            if pos.size:
                best_gap = gap
                best_w = w
        except (ArpackNoConvergence, RuntimeError):
            pass
    elapsed = time.perf_counter() - t0
    vac_e = float(best_w[0]) if best_w is not None and best_w.size else float("nan")
    vac_mult = int(np.sum(np.abs(best_w - best_w[0]) < 1e-5)) if best_w is not None else 0
    if return_timing:
        return best_w, best_gap, vac_e, vac_mult, elapsed
    return best_w, best_gap, vac_e


def jw_gap_from_spectrum(eigs, vac_tol: float = 1e-6) -> tuple[float, float, int, int]:
    """JW gap from spectrum: (E0, gap=E1-E0, vac_mult, n_distinct).

    vac_mult counts eigenvalues within vac_tol of E0.
    gap is eigs[vac_mult] - E0 after sorting. n_distinct uses round-4 uniqueness.
    """
    w = np.sort(np.asarray(eigs, dtype=float).ravel())
    if w.size == 0:
        return float("nan"), float("nan"), 0, 0
    e0 = float(w[0])
    vac = int(np.sum(np.abs(w - e0) < vac_tol))
    n_dist = int(len(np.unique(np.round(w, 4))))
    if vac >= w.size:
        return e0, float("nan"), vac, n_dist
    e1 = float(w[vac])
    return e0, e1 - e0, vac, n_dist


def jw_gap_from_w(w, vac_tol: float = 1e-6) -> tuple[float, float, int]:
    """JW mass gap on a sorted spectrum: E0, gap=E1-E0, vac_mult.

    JW: Delta = inf{<psi,H psi> : ||psi||=1, psi perp Omega} after setting E_vac=0,
    i.e. lambda_1 - lambda_0 on the gauge-invariant spectrum (unique or degenerate vac).
    """
    e0, gap, vac, _n = jw_gap_from_spectrum(w, vac_tol=vac_tol)
    return e0, gap, vac


def q8_lattice_gap_row(lat, g=G_DEFINING_KS, V=None, dim_threshold=DENSE_QR_THRESHOLD):
    """Single Q8 volume-scan row: dense QR if dim<=threshold else sparse shift-invert."""
    
    if V is None:
        _, V = wilson_weight_Q8_2d()
    dim = lat.N ** lat.nE
    if dim <= dim_threshold:
        t0 = time.perf_counter()
        w, gap, vac, e0 = gauge_invariant_spectrum(lat, g, V)
        elapsed = time.perf_counter() - t0
        method = "dense_QR"
        n_dist = len(np.unique(np.round(w, 4)))
    else:
        w, gap, e0, vac, elapsed = gauge_invariant_lowest_gap(
            lat, g, V, return_timing=True,
        )
        method = "sparse_shift_invert"
        if w is None:
            w = np.array([e0])
            gap = float("nan")
        n_dist = len(np.unique(np.round(w, 4))) if w is not None else 0
    # Open tiny spectra can be vacuum-only (n_distinct < 2) — not a physical FAIL
    is_open = not lat.periodic
    if is_open and n_dist < 2:
        status = "SKIP_small_open_trivial_spectrum"
    elif gap == gap and gap > 1e-3 and vac == 1:
        status = "ok"
    else:
        status = "fail"
    rec = gap_record("H_phys", float(gap), float(e0), int(vac), n_distinct=n_dist, status=status)
    return {
        "Lx": lat.Lx, "Ly": lat.Ly, "periodic": lat.periodic,
        "dim": dim, "links": lat.nE, "method": method,
        "vacuum_mult": int(vac), "vacuum_energy": float(e0),
        "gap": float(gap), "n_distinct": n_dist, "seconds": elapsed,
        "status": status,
        "gap_record": rec,
    }


# -----------------------------------------------------------------
# Orbit-reduction (abelian open): spanning-tree gauge fix -> |G|^{E-V+1}
# -----------------------------------------------------------------
def orbit_reduced_hamiltonian(lat, g=G_DEFINING_KS, V=None):
    """Build H on gauge orbits (abelian, open LatticeYM).

    Connected open graph: spanning tree removes V-1 edges; free links = E-V+1
    (cyclomatic number). No stub / dangling residual. dim_red = |G|^{E-V+1}.
    """
    if V is None:
        _, V = wilson_weight_K4()
    if lat.periodic:
        raise ValueError("orbit_reduced_hamiltonian: open BC only")
    if not lat._abelian:
        raise ValueError("orbit_reduced_hamiltonian: abelian G only (use tree_reduced_hamiltonian for nonabelian)")
    N = lat.N
    table, inv = lat.table, lat.inv
    Lx, Ly = lat.Lx, lat.Ly
    all_links = list(range(lat.nE))

    root: tuple[int, int] = (0, 0)
    parent: dict[tuple[int, int], tuple[int, int] | None] = {root: None}
    parent_edge: dict[tuple[int, int], tuple[int, bool]] = {}
    queue: list[tuple[int, int]] = [root]
    while queue:
        i, j = queue.pop(0)
        for di, dj, axis in ((1, 0, "h"), (-1, 0, "h"), (0, 1, "v"), (0, -1, "v")):
            ni, nj = i + di, j + dj
            if not (0 <= ni < Lx and 0 <= nj < Ly):
                continue
            if (ni, nj) in parent:
                continue
            if axis == "h":
                key = (min(i, ni), j)
                if key not in lat.he:
                    continue
                e = lat.he[key]
                fwd = di > 0
            else:
                key = (i, min(j, nj))
                if key not in lat.ve:
                    continue
                e = lat.ve[key]
                fwd = dj > 0
            parent[(ni, nj)] = (i, j)
            parent_edge[(ni, nj)] = (e, fwd)
            queue.append((ni, nj))
    if len(parent) != Lx * Ly:
        raise RuntimeError(f"orbit_reduced: lattice not connected ({len(parent)}/{Lx*Ly})")
    tree = {parent_edge[c][0] for c in parent_edge}
    free = [e for e in all_links if e not in tree]
    nfree = len(free)
    dim_red = N ** nfree
    if dim_red > 65536:
        raise MemoryError(f"orbit_reduced dim_red={dim_red} too large")
    free_pos = {e: p for p, e in enumerate(free)}
    Hred = np.zeros((dim_red, dim_red))

    order: list[tuple[int, int]] = []
    q2: list[tuple[int, int]] = [root]
    seen: set[tuple[int, int]] = {root}
    while q2:
        u = q2.pop(0)
        order.append(u)
        for v, p in parent.items():
            if p == u and v not in seen:
                seen.add(v)
                q2.append(v)

    def endpoint(e) -> tuple[tuple[int, int], tuple[int, int]]:
        for (i, j), idx in lat.he.items():
            if idx == e:
                return (i, j), (i + 1, j)
        for (i, j), idx in lat.ve.items():
            if idx == e:
                return (i, j), (i, j + 1)
        raise KeyError(e)

    def canonicalize(vals):
        g_at: dict[tuple[int, int], int] = {root: 0}
        for child in order[1:]:
            p = parent[child]
            assert p is not None
            e, fwd = parent_edge[child]
            U = vals[e]
            gp = g_at[p]
            g_at[child] = table[gp, U] if fwd else table[gp, inv[U]]
        new_vals: dict[int, int] = {}
        for e in all_links:
            u, v = endpoint(e)
            g0 = g_at[u]
            g1 = g_at[v]
            new_vals[e] = table[table[g0, vals[e]], inv[g1]]
        return tuple(new_vals[e] for e in free)

    def rep_index(vals):
        canon = canonicalize(vals)
        r = 0
        for p, val in enumerate(canon):
            r += val * (N ** p)
        return r

    for r in range(dim_red):
        vals = {e: 0 for e in all_links}
        for e, p in free_pos.items():
            vals[e] = (r // (N ** p)) % N
        mag = 0.0
        for i in range(Lx):
            for j in range(Ly):
                pedges = lat.plaquette_edges(i, j)
                if pedges is None:
                    continue
                hol = 0
                for (e, s) in pedges:
                    d = vals[e]
                    hol = table[hol, d] if s == +1 else table[hol, inv[d]]
                mag += V[hol]
        Hred[r, r] += (1.0 / (2.0 * g * g)) * mag
        for e in range(lat.nE):
            base_val = vals[e]
            Hred[r, r] += (g * g / 2.0) * (1.0 - 1.0 / N)
            for h in range(1, N):
                new_vals = dict(vals)
                new_vals[e] = table[h, base_val]
                r2 = rep_index(new_vals)
                Hred[r, r2] += (g * g / 2.0) * (-1.0 / N)
    Hred = (Hred + Hred.T) / 2
    w = np.sort(np.linalg.eigvalsh(Hred))
    _e0, gap, vac = jw_gap_from_w(w)
    return w, gap, vac


def wilson_gap(V):
    """Smallest positive Wilson plaquette weight: the magnetic contribution to the gap."""
    pos = V[V > 1e-12]
    return float(pos.min()) if pos.size else 0.0


def orbit_reduced_He_Hm(lat, V):
    """Abelian open LatticeYM: He, Hm on gauge-orbit reps (tree gauge; |G|^{E-V+1})."""
    if lat.periodic or not lat._abelian:
        raise ValueError("orbit_reduced_He_Hm: open abelian only")
    N = lat.N
    table, inv = lat.table, lat.inv
    Lx, Ly = lat.Lx, lat.Ly
    all_links = list(range(lat.nE))
    root: Site2 = (0, 0)
    parent: dict[Site2, Site2 | None] = {root: None}
    parent_edge: dict[Site2, tuple[int, bool]] = {}
    queue: list[Site2] = [root]
    while queue:
        i, j = queue.pop(0)
        for di, dj, axis in ((1, 0, "h"), (-1, 0, "h"), (0, 1, "v"), (0, -1, "v")):
            ni, nj = i + di, j + dj
            if not (0 <= ni < Lx and 0 <= nj < Ly) or (ni, nj) in parent:
                continue
            if axis == "h":
                key = (min(i, ni), j)
                if key not in lat.he:
                    continue
                e = lat.he[key]
                fwd = di > 0
            else:
                key = (i, min(j, nj))
                if key not in lat.ve:
                    continue
                e = lat.ve[key]
                fwd = dj > 0
            parent[(ni, nj)] = (i, j)
            parent_edge[(ni, nj)] = (e, fwd)
            queue.append((ni, nj))
    if len(parent) != Lx * Ly:
        raise RuntimeError(f"orbit_reduced_He_Hm: not connected ({len(parent)}/{Lx*Ly})")
    tree = {parent_edge[c][0] for c in parent_edge}
    free = [e for e in all_links if e not in tree]
    nfree = len(free)
    dim_red = N ** nfree
    if dim_red > 8192:
        raise MemoryError(f"orbit_reduced_He_Hm dim_red={dim_red}")
    free_pos = {e: p for p, e in enumerate(free)}

    order: list[Site2] = []
    q2: list[Site2] = [root]
    seen: set[Site2] = {root}
    while q2:
        u = q2.pop(0)
        order.append(u)
        for v, p in parent.items():
            if p == u and v not in seen:
                seen.add(v)
                q2.append(v)

    def endpoint(e):
        for (i, j), idx in lat.he.items():
            if idx == e:
                return (i, j), (i + 1, j)
        for (i, j), idx in lat.ve.items():
            if idx == e:
                return (i, j), (i, j + 1)
        raise KeyError(e)

    def canonicalize(vals):
        g_at: dict[Site2, int] = {root: 0}
        for child in order[1:]:
            p = parent[child]
            assert p is not None
            e, fwd = parent_edge[child]
            U = vals[e]
            gp = g_at[p]
            g_at[child] = table[gp, U] if fwd else table[gp, inv[U]]
        new_vals = {}
        for e in all_links:
            u, v = endpoint(e)
            g0 = g_at[u]
            g1 = g_at[v]
            new_vals[e] = table[table[g0, vals[e]], inv[g1]]
        return tuple(new_vals[e] for e in free)

    def rep_index(vals):
        canon = canonicalize(vals)
        r = 0
        for p, val in enumerate(canon):
            r += val * (N ** p)
        return r

    He = np.zeros((dim_red, dim_red))
    Hm = np.zeros((dim_red, dim_red))
    for r in range(dim_red):
        vals = {e: 0 for e in all_links}
        for e, p in free_pos.items():
            vals[e] = (r // (N ** p)) % N
        mag = 0.0
        for i in range(Lx):
            for j in range(Ly):
                pedges = lat.plaquette_edges(i, j)
                if pedges is None:
                    continue
                hol = 0
                for (e, s) in pedges:
                    d = vals[e]
                    hol = table[hol, d] if s == +1 else table[hol, inv[d]]
                mag += V[hol]
        Hm[r, r] += mag
        for e in range(lat.nE):
            base_val = vals[e]
            He[r, r] += 1.0 - 1.0 / N
            for h in range(1, N):
                new_vals = dict(vals)
                new_vals[e] = table[h, base_val]
                r2 = rep_index(new_vals)
                He[r, r2] += -1.0 / N
    n_plaq = sum(
        1
        for i in range(Lx)
        for j in range(Ly)
        if lat.plaquette_edges(i, j) is not None
    )
    return 0.5 * (He + He.T), 0.5 * (Hm + Hm.T), dim_red, n_plaq


def plaquette_weight_diagonal(lat: LatticeYM, V, i: int = 0, j: int = 0):
    """Full-config diagonal of V_R(hol(p_{i,j})) for one plaquette."""
    pedges = lat.plaquette_edges(i, j)
    if pedges is None:
        raise ValueError(f"plaquette_weight_diagonal: no plaquette at ({i},{j})")
    N = lat.N
    dim = N ** lat.nE
    flat = np.arange(dim)
    hol = np.zeros(dim, dtype=int)
    for (e, s) in pedges:
        base = N ** e
        digit = (flat // base) % N
        if s == +1:
            hol = lat.table[hol, digit]
        else:
            hol = lat.table[hol, lat.inv[digit]]
    return V[hol].astype(float)


def _orbit_tree_setup(lat: LatticeYM):
    """Shared spanning-tree data for abelian open orbit reduction."""
    if lat.periodic or not lat._abelian:
        raise ValueError("_orbit_tree_setup: open abelian only")
    N = lat.N
    table, inv = lat.table, lat.inv
    Lx, Ly = lat.Lx, lat.Ly
    all_links = list(range(lat.nE))
    root: Site2 = (0, 0)
    parent: dict[Site2, Site2 | None] = {root: None}
    parent_edge: dict[Site2, tuple[int, bool]] = {}
    queue: list[Site2] = [root]
    while queue:
        i, j = queue.pop(0)
        for di, dj, axis in ((1, 0, "h"), (-1, 0, "h"), (0, 1, "v"), (0, -1, "v")):
            ni, nj = i + di, j + dj
            if not (0 <= ni < Lx and 0 <= nj < Ly) or (ni, nj) in parent:
                continue
            if axis == "h":
                key = (min(i, ni), j)
                if key not in lat.he:
                    continue
                e = lat.he[key]
                fwd = di > 0
            else:
                key = (i, min(j, nj))
                if key not in lat.ve:
                    continue
                e = lat.ve[key]
                fwd = dj > 0
            parent[(ni, nj)] = (i, j)
            parent_edge[(ni, nj)] = (e, fwd)
            queue.append((ni, nj))
    if len(parent) != Lx * Ly:
        raise RuntimeError(f"_orbit_tree_setup: not connected ({len(parent)}/{Lx * Ly})")
    tree = {parent_edge[c][0] for c in parent_edge}
    free = [e for e in all_links if e not in tree]
    nfree = len(free)
    dim_red = N ** nfree
    if dim_red > 8192:
        raise MemoryError(f"_orbit_tree_setup dim_red={dim_red}")
    free_pos = {e: p for p, e in enumerate(free)}
    order: list[Site2] = []
    q2: list[Site2] = [root]
    seen: set[Site2] = {root}
    while q2:
        u = q2.pop(0)
        order.append(u)
        for v, p in parent.items():
            if p == u and v not in seen:
                seen.add(v)
                q2.append(v)

    def endpoint(e):
        for (i, j), idx in lat.he.items():
            if idx == e:
                return (i, j), (i + 1, j)
        for (i, j), idx in lat.ve.items():
            if idx == e:
                return (i, j), (i, j + 1)
        raise KeyError(e)

    def canonicalize(vals):
        g_at: dict[Site2, int] = {root: 0}
        for child in order[1:]:
            p = parent[child]
            assert p is not None
            e, fwd = parent_edge[child]
            U = vals[e]
            gp = g_at[p]
            g_at[child] = table[gp, U] if fwd else table[gp, inv[U]]
        new_vals = {}
        for e in all_links:
            u, v = endpoint(e)
            g0 = g_at[u]
            g1 = g_at[v]
            new_vals[e] = table[table[g0, vals[e]], inv[g1]]
        return tuple(new_vals[e] for e in free)

    def rep_index(vals):
        canon = canonicalize(vals)
        r = 0
        for p, val in enumerate(canon):
            r += val * (N ** p)
        return r

    return {
        "N": N,
        "table": table,
        "inv": inv,
        "all_links": all_links,
        "free": free,
        "free_pos": free_pos,
        "dim_red": dim_red,
        "rep_index": rep_index,
    }


def orbit_reduced_loop_weight_diag(lat: LatticeYM, W, edges):
    """Orbit-reduced diagonal of class-function W[hol(C)] for edge list (e,s)."""
    setup = _orbit_tree_setup(lat)
    N = setup["N"]
    table, inv = setup["table"], setup["inv"]
    all_links = setup["all_links"]
    free_pos = setup["free_pos"]
    dim_red = setup["dim_red"]
    diag = np.zeros(dim_red)
    for r in range(dim_red):
        vals = {e: 0 for e in all_links}
        for e, p in free_pos.items():
            vals[e] = (r // (N ** p)) % N
        hol = 0
        for (e, s) in edges:
            d = vals[e]
            hol = table[hol, d] if s == +1 else table[hol, inv[d]]
        diag[r] = float(W[hol])
    return diag


def orbit_reduced_plaquette_weight_diag(lat: LatticeYM, V, i0: int = 0, j0: int = 0):
    """Orbit-reduced diagonal of V_R(hol(p)) for one plaquette."""
    pedges = lat.plaquette_edges(i0, j0)
    if pedges is None:
        raise ValueError(f"orbit_reduced_plaquette_weight_diag: no plaquette at ({i0},{j0})")
    return orbit_reduced_loop_weight_diag(lat, V, pedges)


def correlator_local_mass_from_spectrum(
    wr,
    Vr,
    Od,
    *,
    t_grid: tuple[int, ...] | None = None,
    stable_tol: float = 0.02,
) -> dict:
    """Local mass from connected correlator C(t)=⟨Ω|O e^{-t(H-E0)} O|Ω⟩.

    Od is the GI-subspace matrix of a local observable. Uses spectral expansion
    (exact on the finite chart). When the spectrum is exact, m_coupled =
    min{E_n−E_0 : ⟨n|O|0⟩≠0} is the asymptotic local mass; m_lat is taken from
    a late stable −log(C(t+Δ)/C(t))/Δ window when C stays above underflow,
    otherwise m_lat := m_coupled with stable=True if n_coupled≥1.
    """
    wr = np.asarray(wr, dtype=float)
    Vr = np.asarray(Vr, dtype=float)
    Od = np.asarray(Od, dtype=float)
    Od = 0.5 * (Od + Od.T)
    Omega = Vr[:, 0]
    v0 = float(Omega @ Od @ Omega)
    Oc = Od - v0 * np.eye(len(wr))
    amps = Vr.T @ (Oc @ Omega)
    coupled = [float(wr[n] - wr[0]) for n in range(1, len(wr)) if abs(amps[n]) > 1e-8]
    m_coupled = min(coupled) if coupled else float("nan")

    if t_grid is None:
        if m_coupled > 1e-6 and not math.isnan(m_coupled):
            # ~3/m … ~12/m window (integer), at least 8 samples
            t0 = max(1, int(math.floor(2.5 / m_coupled)))
            t1 = max(t0 + 8, int(math.ceil(12.0 / m_coupled)))
            t_grid = tuple(range(t0, t1 + 1))
        else:
            t_grid = tuple(range(1, 25))

    Cs: list[float] = []
    for t in t_grid:
        s = 0.0
        for n in range(1, len(wr)):
            s += float(abs(amps[n]) ** 2) * math.exp(-(wr[n] - wr[0]) * float(t))
        Cs.append(s)

    m_eff: list[float] = []
    for i in range(len(Cs) - 1):
        dt = float(t_grid[i + 1] - t_grid[i])
        if Cs[i] > 1e-14 and Cs[i + 1] > 1e-14 and dt > 0:
            m_eff.append(-math.log(Cs[i + 1] / Cs[i]) / dt)
        else:
            m_eff.append(float("nan"))

    valid = [(i, m) for i, m in enumerate(m_eff) if not math.isnan(m)]
    if len(valid) >= 3:
        tail = [m for _, m in valid[-3:]]
        m_lat = float(sum(tail) / len(tail))
        stable = bool(max(abs(x - m_lat) for x in tail) < stable_tol)
        # Prefer exact asymptotic when available and within tol of plateau
        if not math.isnan(m_coupled) and abs(m_lat - m_coupled) < stable_tol:
            m_lat = float(m_coupled)
            stable = True
    elif not math.isnan(m_coupled):
        m_lat = float(m_coupled)
        stable = True
        tail = []
    else:
        m_lat = float("nan")
        stable = False
        tail = []

    return {
        "t_grid": list(t_grid),
        "C": Cs,
        "m_eff": m_eff,
        "m_lat": m_lat,
        "m_coupled": m_coupled,
        "stable": stable,
        "O_vac": v0,
        "n_coupled": len(coupled),
        "m_eff_tail": tail,
    }


# -----------------------------------------------------------------
# Euclidean finite-group measure (OS / transfer lemmas for _2 / _3)
# -----------------------------------------------------------------

@dataclass(frozen=True)
class FiniteGroup:
    """Integer-indexed finite group wrapping make_group tables."""

    n: int
    mul_tbl: np.ndarray
    inv_tbl: np.ndarray
    names: tuple[str, ...]
    identity: int = 0

    def mul(self, a: int, b: int) -> int:
        return int(self.mul_tbl[int(a), int(b)])

    def inv(self, a: int) -> int:
        return int(self.inv_tbl[int(a)])


def finite_group_from_tuple(group_tuple) -> FiniteGroup:
    G, gi, table, inv = group_tuple
    names = tuple(str(g) for g in G)
    return FiniteGroup(
        n=len(G),
        mul_tbl=np.asarray(table, dtype=int),
        inv_tbl=np.asarray(inv, dtype=int),
        names=names,
        identity=0,
    )


def finite_group_K4() -> FiniteGroup:
    return finite_group_from_tuple(K4())


def finite_group_Q8() -> FiniteGroup:
    return finite_group_from_tuple(Q8())


@dataclass
class Lattice2D:
    T: int
    L: int
    periodic_t: bool
    periodic_x: bool
    edges: tuple[tuple[int, int, str], ...]
    plaqs: tuple[tuple[tuple[tuple[int, int, str], bool], ...], ...]
    theta_edge: dict[tuple[int, int, str], tuple[tuple[int, int, str], bool]]
    edge_index: dict[tuple[int, int, str], int]


def build_lattice_2d(
    T: int,
    L: int,
    *,
    periodic_t: bool = False,
    periodic_x: bool = True,
) -> Lattice2D:
    """2D Euclidean lattice; open time preferred for OS half-space tests."""
    edges: list[tuple[int, int, str]] = []
    edge_index: dict[tuple[int, int, str], int] = {}

    def add_edge(t: int, x: int, dirc: str) -> None:
        k = (t, x, dirc)
        if k in edge_index:
            return
        edge_index[k] = len(edges)
        edges.append(k)

    def wrap_t(t: int) -> int:
        return t % T if periodic_t else t

    def wrap_x(x: int) -> int:
        return x % L if periodic_x else x

    for t in range(T):
        for x in range(L):
            if periodic_t or (t < T - 1):
                add_edge(t, x, "t")
            if periodic_x or (x < L - 1):
                add_edge(t, x, "x")

    plaqs: list[tuple[tuple[tuple[int, int, str], bool], ...]] = []
    t_max = T if periodic_t else T - 1
    x_max = L if periodic_x else L - 1
    for t in range(t_max):
        for x in range(x_max):
            t1 = wrap_t(t + 1)
            x1 = wrap_x(x + 1)
            loop = (
                ((t, x, "t"), False),
                ((t1, x, "x"), False),
                ((t, x1, "t"), True),
                ((t, x, "x"), True),
            )
            # skip if any edge missing (open BC corner)
            ok = True
            for ek, _inv in loop:
                if ek not in edge_index:
                    ok = False
                    break
            if ok:
                plaqs.append(loop)

    theta_edge: dict[tuple[int, int, str], tuple[tuple[int, int, str], bool]] = {}
    for t, x, dirc in edges:
        tR = T - 1 - t
        if dirc == "t":
            t_base = tR - 1
            if periodic_t:
                t_base %= T
            ekR = (t_base, x, "t")
            if ekR in edge_index:
                theta_edge[(t, x, "t")] = (ekR, True)
        else:
            ekR = (tR, x, "x")
            if ekR in edge_index:
                theta_edge[(t, x, "x")] = (ekR, False)

    return Lattice2D(
        T=T,
        L=L,
        periodic_t=periodic_t,
        periodic_x=periodic_x,
        edges=tuple(edges),
        plaqs=tuple(plaqs),
        theta_edge=theta_edge,
        edge_index=edge_index,
    )


def V_tbl_K4() -> np.ndarray:
    return np.asarray(wilson_weight_K4()[1], dtype=float)


def V_tbl_Q8() -> np.ndarray:
    return np.asarray(wilson_weight_Q8_2d()[1], dtype=float)

def plaquette_holonomy(
    G: FiniteGroup,
    cfg: np.ndarray,
    lat: Lattice2D,
    loop: tuple[tuple[tuple[int, int, str], bool], ...],
) -> int:
    h = G.identity
    for ek, inv_flag in loop:
        u = int(cfg[lat.edge_index[ek]])
        if inv_flag:
            u = G.inv(u)
        h = G.mul(h, u)
    return h


def euclid_action(
    G: FiniteGroup,
    lat: Lattice2D,
    cfg: np.ndarray,
    V_tbl: np.ndarray,
) -> float:
    s = 0.0
    for loop in lat.plaqs:
        h = plaquette_holonomy(G, cfg, lat, loop)
        s += float(V_tbl[h])
    return s


def euclid_weight(
    G: FiniteGroup,
    lat: Lattice2D,
    cfg: np.ndarray,
    beta: float,
    V_tbl: np.ndarray,
) -> float:
    return math.exp(-float(beta) * euclid_action(G, lat, cfg, V_tbl))


def enumerate_configs(G: FiniteGroup, n_edges: int):
    """Yield all group assignments on n_edges (exact; exponential)."""
    if n_edges < 0:
        raise ValueError("n_edges < 0")
    if n_edges == 0:
        yield np.zeros(0, dtype=np.int16)
        return
    cfg = np.zeros(n_edges, dtype=np.int16)
    n = G.n
    while True:
        yield cfg.copy()
        i = 0
        while i < n_edges and cfg[i] == n - 1:
            cfg[i] = 0
            i += 1
        if i == n_edges:
            break
        cfg[i] += 1


def config_count(G: FiniteGroup, n_edges: int) -> int:
    return int(G.n**n_edges)


def free_edge_indices(lat: Lattice2D, *, temporal_gauge: bool) -> list[int]:
    """Indices varied under exact sum. temporal_gauge fixes all t-links to identity (open time)."""
    if not temporal_gauge:
        return list(range(len(lat.edges)))
    if lat.periodic_t:
        raise ValueError("temporal_gauge requires open time (periodic_t=False)")
    return [i for i, ek in enumerate(lat.edges) if ek[2] == "x"]


def enumerate_lattice_configs(
    G: FiniteGroup,
    lat: Lattice2D,
    *,
    temporal_gauge: bool = True,
):
    """Exact configs; default temporal gauge (U_t=1) for open-time Wilson charts."""
    free = free_edge_indices(lat, temporal_gauge=temporal_gauge)
    nE = len(lat.edges)
    if not free:
        yield np.zeros(nE, dtype=np.int16)
        return
    for sub in enumerate_configs(G, len(free)):
        cfg = np.zeros(nE, dtype=np.int16)
        for k, ei in enumerate(free):
            cfg[ei] = int(sub[k])
        yield cfg


def lattice_config_count(
    G: FiniteGroup,
    lat: Lattice2D,
    *,
    temporal_gauge: bool = True,
) -> int:
    free = free_edge_indices(lat, temporal_gauge=temporal_gauge)
    return config_count(G, len(free))


def theta_pullback_cfg(G: FiniteGroup, lat: Lattice2D, cfg: np.ndarray) -> np.ndarray:
    """(ΘU)_e = U_{θ(e)}^{±1}."""
    cfgR = np.zeros_like(cfg)
    for ek, i in lat.edge_index.items():
        if ek not in lat.theta_edge:
            # unpaired under Θ (open-time edge near boundary): leave as identity
            cfgR[i] = G.identity
            continue
        ekR, inv_flag = lat.theta_edge[ek]
        j = lat.edge_index[ekR]
        u = int(cfg[j])
        cfgR[i] = G.inv(u) if inv_flag else u
    return cfgR


def os_reflection_apply(
    G: FiniteGroup,
    lat: Lattice2D,
    F: Callable[[np.ndarray], float],
    cfg: np.ndarray,
) -> float:
    return float(F(theta_pullback_cfg(G, lat, cfg)))


def os_gram_matrix_exact(
    G: FiniteGroup,
    lat: Lattice2D,
    beta: float,
    V_tbl: np.ndarray,
    basis_F: list[Callable[[np.ndarray], float]],
    *,
    temporal_gauge: bool = True,
    max_configs: int = 2_000_000,
) -> np.ndarray:
    """M_ij = E[(Θ F_i) F_j] under Euclidean Wilson measure."""
    ncfg = lattice_config_count(G, lat, temporal_gauge=temporal_gauge)
    if ncfg > max_configs:
        raise MemoryError(f"OS Gram exact sum too large: ncfg={ncfg}")
    m = len(basis_F)
    Z = 0.0
    M = np.zeros((m, m), dtype=float)
    for cfg in enumerate_lattice_configs(G, lat, temporal_gauge=temporal_gauge):
        w = euclid_weight(G, lat, cfg, beta, V_tbl)
        Z += w
        vals_F = np.array([float(F(cfg)) for F in basis_F], dtype=float)
        vals_Th = np.array(
            [os_reflection_apply(G, lat, F, cfg) for F in basis_F], dtype=float
        )
        M += w * np.outer(vals_Th, vals_F)
    if Z <= 0:
        return M * float("nan")
    return M / Z


def certify_os_rp_exact(
    G: FiniteGroup,
    lat: Lattice2D,
    beta: float,
    V_tbl: np.ndarray,
    basis_F: list[Callable[[np.ndarray], float]],
    *,
    tol: float = 1e-10,
    temporal_gauge: bool = True,
    max_configs: int = 2_000_000,
) -> dict:
    M = os_gram_matrix_exact(
        G, lat, beta, V_tbl, basis_F,
        temporal_gauge=temporal_gauge, max_configs=max_configs,
    )
    Ms = 0.5 * (M + M.T)
    evals = np.linalg.eigvalsh(Ms)
    return {
        "min_eig": float(evals[0]),
        "PSD": bool(evals[0] >= -tol),
        "n_basis": len(basis_F),
        "n_edges": len(lat.edges),
        "n_free": len(free_edge_indices(lat, temporal_gauge=temporal_gauge)),
        "n_plaq": len(lat.plaqs),
        "n_configs": lattice_config_count(G, lat, temporal_gauge=temporal_gauge),
        "temporal_gauge": bool(temporal_gauge),
        "beta": float(beta),
        "evals": [float(x) for x in evals],
        "pass": bool(evals[0] >= -tol),
    }


def make_basis_time_slice_V(
    G: FiniteGroup,
    lat: Lattice2D,
    V_tbl: np.ndarray,
) -> list[Callable[[np.ndarray], float]]:
    """Gauge-invariant cylinder basis: 1 and V(plaquette) on +time plaquettes."""
    basis: list[Callable[[np.ndarray], float]] = []
    basis.append(lambda cfg: 1.0)

    # Prefer plaquettes with t-support starting at t=0 (positive half for open time)
    pos_plaqs = []
    for loop in lat.plaqs:
        ts = [ek[0] for ek, _ in loop]
        if min(ts) >= 0 and max(ts) <= max(0, lat.T // 2):
            pos_plaqs.append(loop)
    if not pos_plaqs:
        pos_plaqs = list(lat.plaqs[: min(2, len(lat.plaqs))])

    for loop in pos_plaqs[:3]:
        loop_c = loop

        def F(cfg, loop=loop_c):
            h = plaquette_holonomy(G, cfg, lat, loop)
            return float(V_tbl[h])

        basis.append(F)
    return basis


def build_transfer_matrix_exact(
    G: FiniteGroup,
    lat: Lattice2D,
    beta: float,
    V_tbl: np.ndarray,
    t0: int,
    t1: int,
    *,
    temporal_gauge: bool = True,
    max_configs: int = 2_000_000,
) -> dict:
    """Joint kernel on spatial x-link slices at times t0,t1 (normalized by Z)."""
    ncfg = lattice_config_count(G, lat, temporal_gauge=temporal_gauge)
    if ncfg > max_configs:
        raise MemoryError(f"transfer exact sum too large: ncfg={ncfg}")

    slice_edges_0 = [
        (t0, x, "x")
        for x in range(lat.L)
        if (t0, x, "x") in lat.edge_index
    ]
    slice_edges_1 = [
        (t1, x, "x")
        for x in range(lat.L)
        if (t1, x, "x") in lat.edge_index
    ]
    if len(slice_edges_0) == 0 or len(slice_edges_0) != len(slice_edges_1):
        raise ValueError("slice edges unavailable or mismatched")

    slice_idx_0 = [lat.edge_index[ek] for ek in slice_edges_0]
    slice_idx_1 = [lat.edge_index[ek] for ek in slice_edges_1]
    n_slice = len(slice_idx_0)

    slice_map: dict[tuple[int, ...], int] = {}
    slice_assignments: list[tuple[int, ...]] = []
    for a in enumerate_configs(G, n_slice):
        key = tuple(int(x) for x in a)
        slice_map[key] = len(slice_assignments)
        slice_assignments.append(key)
    dim = len(slice_assignments)
    K = np.zeros((dim, dim), dtype=float)
    Z = 0.0
    for cfg in enumerate_lattice_configs(G, lat, temporal_gauge=temporal_gauge):
        w = euclid_weight(G, lat, cfg, beta, V_tbl)
        Z += w
        key0 = tuple(int(cfg[i]) for i in slice_idx_0)
        key1 = tuple(int(cfg[i]) for i in slice_idx_1)
        K[slice_map[key0], slice_map[key1]] += w
    if Z <= 0:
        raise RuntimeError("Z=0 in transfer build")
    K /= Z
    return {
        "K": K,
        "dim": dim,
        "Z": float(Z),
        "slice_edges_0": slice_edges_0,
        "slice_edges_1": slice_edges_1,
        "n_configs": ncfg,
        "temporal_gauge": bool(temporal_gauge),
        "beta": float(beta),
    }


def transfer_to_hamiltonian(
    K: np.ndarray,
    a_time: float = 1.0,
    eps: float = 1e-15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ks = 0.5 * (K + K.T)
    evals, evecs = np.linalg.eigh(Ks)
    evals = np.maximum(evals, 0.0)
    mask = evals > eps
    H_evals = np.zeros_like(evals)
    H_evals[mask] = -(1.0 / a_time) * np.log(evals[mask])
    H = (evecs * H_evals) @ evecs.T
    return H, evals, H_evals

def clustering_rate_from_2pt(dist_list, conn_vals) -> tuple[float, float, float]:
    xs = np.array(dist_list, dtype=float)
    ys = np.array([math.log(abs(c) + 1e-300) for c in conn_vals], dtype=float)
    k0 = max(0, len(xs) // 2)
    A = np.vstack([xs[k0:], np.ones(len(xs) - k0)]).T
    slope, intercept = np.linalg.lstsq(A, ys[k0:], rcond=None)[0]
    return float(-slope), float(slope), float(intercept)
