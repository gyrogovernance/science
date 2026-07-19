#!/usr/bin/env python3
"""
hqvm_cgm_trestleboard_common.py

Shared foundation for the CGM trestleboard instrument.

Holds the kernel imports, derived constants, code-atom tables, dataclasses,
ENSDF loaders, and null-model helpers used by the engine
(hqvm_cgm_trestleboard_1.py), NuclearBoard/census (hqvm_cgm_trestleboard_2.py),
and the report+CLI (hqvm_cgm_trestleboard_run.py). Split out of the original
single-file instrument for publishable organization.

Companion notes: hqvm_cgm_trestleboard_notes.txt.
"""
from __future__ import annotations
import csv
import io
import math
import random
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_PATH = _EXPERIMENTS_DIR / "hqvm_cgm_trestleboard_results.txt"
ENSDF_EV_BAND_PATH = (
    _REPO_ROOT / "data" / "catalogs" / "ensdf" / "ensdf_ev_band_levels.csv"
)
ENSDF_FIRST_EXCITED_PATH = (
    _REPO_ROOT / "data" / "catalogs" / "ensdf" / "ensdf_first_excited_actinides.csv"
)

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from gyroscopic.hQVM.api import q_word6, shell_population
from gyroscopic.hQVM.constants import (
    APERTURE_GAP,
    DELTA_BU,
    HORIZON_SIZE,
    M_A,
    OMEGA_SIZE,
    RHO,
)
from gyroscopic.hQVM.family import (
    HqvmD,
    bisect_p_c_rank_micro_ref,
    bfs_reach,
    build_hqvm_d,
    byte_from_family_micro,
    exact_root_rank_pmf,
    theta_micro_ref_exact,
    verify_d6_against_api,
    verify_f_squared_rest_d,
    verify_exact_root_rank_lock,
)

try:
    from hqvm_compact_geom_core import (
        CODE_C1,
        CODE_C2,
        CODE_C3,
        E_EW_GEV,
        M_SHELL,
        CARRIER_TRACES,
        CHANNELS,
        K4_CHANNEL_FLAGS,
        eval_law,
    )
except Exception:
    from math import comb

    E_EW_GEV = 246.22
    CODE_C1, CODE_C2, CODE_C3 = comb(6, 1), comb(6, 2), comb(6, 3)
    M_SHELL = sum(k * comb(6, k) for k in range(7))
    CHANNELS = ()
    K4_CHANNEL_FLAGS = {}
    CARRIER_TRACES = ()

    def eval_law(*a, **k):  # type: ignore
        return 0.0


# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------
DELTA = APERTURE_GAP
CHIRALITY_D = 6
OMEGA = OMEGA_SIZE
H_CARD = HORIZON_SIZE
S_P = math.pi / 2.0
U_P = 1.0 / math.sqrt(2.0)
O_P = math.pi / 4.0
S_CS = S_P / M_A
S_UNA = U_P / M_A
S_ONA = O_P / M_A
S_BU = M_A
Q_G = 4.0 * math.pi
D_SHELL = 24.0
G_KERNEL = Q_G / D_SHELL
ALPHA0 = (DELTA_BU**4) / M_A
OPTICAL_DILUTION = 1.0 / (4.0 * math.pi**2)
ZETA_G = 8.0 / (M_A * math.sqrt(3.0))
ALPHA0_ZETA = ALPHA0 * ZETA_G
ALPHA0_ZETA_TARGET = (RHO**4) / (math.pi * math.sqrt(3.0))
C1, C2, C3 = CODE_C1, CODE_C2, CODE_C3
SQRT5 = math.sqrt(5.0)
TICK_CORR_EXP = C3 * (DELTA**2)
TICK_CORR = 2.0**TICK_CORR_EXP
EV_PER_GEV = 1.0e9
V_GEV = float(E_EW_GEV)
V_EV = V_GEV * EV_PER_GEV
LOG2_INV_DELTA = math.log2(1.0 / DELTA)
LOG2_INV_RHO = math.log2(1.0 / RHO)
TICKS_PER_K = LOG2_INV_DELTA / DELTA
TICKS_PER_D = LOG2_INV_RHO / DELTA
TICKS_PER_OCTAVE = 1.0 / DELTA
TICKS_C3_DELTA = C3 * DELTA
ALPHA_EM = 1.0 / 137.035999084
M_N_MEV = 931.49410242
N_MICROREFS = 1 << CHIRALITY_D
MC_HIERARCHY_SAMPLES = 150
MC_HIERARCHY_SEED = 20260703
OCTAVE_TOL_TICKS = 2.0
W2_BYTES = (0xAA, 0xAB)
W2P_BYTES = (0x2A, 0x2B)

# Sub-tick code atoms: {C1, C2, C3, halves, differences, sums}
CODE_ATOMS_BASE = (
    C1,
    C2,
    C3,
    C1 / 2,
    C2 / 2,
    C3 / 2,
    (C2 - C1),
    (C2 - C1) / 2,  # W/Z gap is 9
    (C1 + C2),
    (C2 + C3),
    (C1 + C2 + C3),
)
CODE_ATOMS = sorted({a for x in CODE_ATOMS_BASE for a in (x, -x)}, key=abs)

# Bridge: Model 2 default — σ∝(S/E)·P_Gamow·θ(p_Δ), p_Δ=E/V_b
# (Model 1: σ∝(S/E)·θ(T), T=exp(−τ); see _p_inclusion_bridge).

# Holonomy meanings
HOLONOMY_DRESS = {
    0: "bare (no holonomy)",
    2: "Z2 two-pass spinorial (F = W2∘W2′)",
    4: "EM-depth dress (dual commutator scale)",
    5: "STF gravity bulk (5 shells, ρ^5)",
}

DRESS_OPERATOR: dict[Tuple[int, int], str] = {
    (0, 2): f"F=W2∘W2′  W2=({W2_BYTES[0]:#04x},{W2_BYTES[1]:#04x}) "
    f"W2′=({W2P_BYTES[0]:#04x},{W2P_BYTES[1]:#04x})",
    (2, 0): f"F⁻¹=F (involution)  W2=({W2_BYTES[0]:#04x},{W2_BYTES[1]:#04x})",
    (2, 4): "EM-depth (dual commutator scale)",
    (4, 2): "EM-depth⁻¹ (dual commutator scale)",
    (4, 5): "STF+1 (gravity bulk shell)",
    (5, 4): "STF-1 (gravity bulk shell)",
}

# Multi-fuel suite with empirical resonance targets
# (label, Z1, Z2, A1, A2, T_plasma_keV, E_ref_keV, Resonance_keV, Tolerance_keV)
FUEL_SUITE: Tuple[
    Tuple[str, int, int, float, float, float, float, float, float], ...
] = (
    ("D-T", 1, 1, 2.0, 3.0, 10.0, 10.0, 64.0, 25.0),  # Drexler
    ("D-D", 1, 1, 2.0, 2.0, 10.0, 10.0, 100.0, 40.0),  # broad
    ("D-3He", 1, 2, 2.0, 3.0, 50.0, 50.0, 250.0, 100.0),  # Nevins
    ("p-B11", 1, 5, 1.0, 11.0, 100.0, 100.0, 148.0, 60.0),  # Becker
)

# Primary measured energies (Zhang supersedes ENSDF Adopted for Th-229m).
# Format: (label, E_eV, tol_ticks, source, year)
NUCLEAR_OPTICAL_PRIMARY: Tuple[Tuple[str, float, float, str, int], ...] = (
    ("Th-229m (Zhang CaF2)", 8.3557335, 0.10, "Nature 633:63-70", 2024),
)
# ENSDF eV-band census: data/catalogs/ensdf/ensdf_ev_band_levels.csv
NUCLEAR_OPTICAL_TOL_TICKS = 0.10


# -----------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------
@dataclass(frozen=True)
class NuclearAuditRow:
    label: str
    E_eV: float
    tick_to_62: float
    hit_62: bool
    jp: str
    source: str
    kind: str  # "primary" | "ensdf_ev" | "superseded"
    nearest_k: int
    nearest_d: int
    nearest_label: str
    nearest_tick: float
    status: str  # "PASS" | "UNCLASSIFIED" | "SUPERSEDED" | "DIAG"


@dataclass(frozen=True)
class NuclearNullModel:
    """Predeclared (k=6,d=2) window vs ENSDF eV-band (diagnostic only)."""

    E_pred_eV: float
    tol_ticks: float
    E_lo_eV: float
    E_hi_eV: float
    band_max_eV: float
    n_census: int
    n_hits_62: int
    p_one_log: float
    p_ge1_log: float
    p_one_lin: float
    p_ge1_lin: float
    n_open: int  # census rows with |nearest_tick| > tol (unexplained)


@dataclass(frozen=True)
class ThresholdBoard:
    s_p: float
    u_p: float
    o_p: float
    m_a: float
    S_CS: float
    S_UNA: float
    S_ONA: float
    S_BU: float
    Q_G: float
    delta_BU: float
    rho: float
    Delta: float
    G_kernel: float
    alpha0: float
    optical_dilution: float
    D: float
    product_QG_ma2: float
    alpha0_zeta: float
    alpha0_zeta_target: float
    product_identity_ok: bool


@dataclass(frozen=True)
class ShellCensus:
    Omega: int
    H: int
    pops: List[int]
    holographic_ok: bool
    mean_entanglement: float


@dataclass(frozen=True)
class PercolationBoard:
    d: int
    p_c_rank: float
    p_c_span: float
    p_c_full: float
    p_c_spectrum: float
    p_c_word: float
    d6_api_ok: bool
    d6_api_note: str


@dataclass(frozen=True)
class SquareReading:
    E_eV: float
    n: float
    log2_v_over_E: float
    sector: str


@dataclass(frozen=True)
class ClosureClass:
    k: int
    d: int
    label: str
    forced: bool
    stf: bool
    tick_corr: bool


@dataclass(frozen=True)
class LevelReading:
    E_eV: float
    n: float
    cls: ClosureClass
    E_pred_eV: float
    n_pred: float
    tick_residual: float
    rel_error: float
    holonomy_note: str


@dataclass(frozen=True)
class CompassStep:
    from_k: int
    from_d: int
    to_k: int
    to_d: int
    from_E: float
    to_E: float
    move_type: str  # undress | Δ-step | dress | octave | code | offset
    note: str
    tick_gap: float
    operator: str = ""


@dataclass(frozen=True)
class PercolationState:
    r: int
    root: int
    Reach: int
    coverage: float
    parity_obstruction: bool
    note: str
    fiber_complete: bool = True  # square-root identity holds only when True


@dataclass(frozen=True)
class CrossSectionReading:
    E_keV: float
    E_MeV: float
    n: float
    p_inc: float
    r_eff: float
    theta: float
    P_gamow: float
    sigma_rel_gamow: float
    sigma_rel_cgm: float
    nearest_forced: str


@dataclass(frozen=True)
class FusionScanResult:
    V_barrier_MeV: float
    E_G_MeV: float
    E_rank_keV: float
    plasma: SquareReading
    barrier: SquareReading
    resonance: SquareReading
    rows: Tuple[CrossSectionReading, ...]
    best_gamow: CrossSectionReading
    best_cgm: CrossSectionReading
    predicted_peak_keV: float


@dataclass(frozen=True)
class ReactivityReading:
    T_keV: float
    sv_rel_gamow: float
    sv_rel_cgm: float
    enhancement: float  # ⟨σv⟩CGM / ⟨σv⟩G (absolute, not ref-normalized)


@dataclass(frozen=True)
class FusionValidation:
    """External cross-section check for the CGM channel-accessibility claim.

    Claim (parameter-free): the structural S-factor (Gamow removed)
    tracks the exact kernel coverage S_eff(E) ∝ theta(E/V_b) with
    p_Delta = E / V_b.

    Two metrics are reported:
      corr       — Pearson r between model and reference CROSS-SECTIONS
                   (sigma = S*P_Gamow/E vs model sigma). Sensitive to the
                   Gamow-footing mismatch; kept for continuity.
      corr_struct — Pearson r between model and reference STRUCTURAL
                   S-factors (model sigma*E ∝ theta; reference S_ref is
                   already Gamow-divided). This is the honest kernel-structure
                   test and is the primary number.

    status: "pending" if no reference CSV was loaded (mirrors the
    ENSDF MISSING posture — a complete, runnable gate that
    reports PENDING until data exists, not an "open" item.
    """

    status: str  # "pending" | "validated"
    fuel: str
    n_points: int
    rmse_log: float  # RMS log10(model / ref) over band
    corr: float  # Pearson r, cross-section footing (Gamow kept)
    corr_struct: float  # Pearson r, structural S-factor (Gamow stripped)
    model_peak_keV: float  # energy of max model cross-section in band
    band_keV: tuple  # (E_lo, E_hi) declared comparison band
    note: str


# Literature terrestrial Coulomb-barrier cutoff (Rider 2023 LLNL HEDS
# seminar "Is There a Better Route to Fusion?"): Z1Z2 >= 7 marked
# "Coulomb barrier is too high"; Z1Z2 >= 8 absolute. Resonance
# placement (this gate) is distinct from reactor viability — Rider's
# Pbrem/Pfus already puts p+11B at 1.19 (ouch) for equilibrium plasma.
FUSION_Z1Z2_CUTOFF = 7

# Fuel roles for resonance-map interpretation (not fitted parameters).
# power: D-T class; aneutronic: p-Li / p-B; CNO: stellar-cycle doorways.
FUSION_FUEL_ROLE = {
    "D-T": "power",
    "p-B11": "aneutronic",
    "10B-p": "aneutronic",
    "12C-p": "CNO",
    "15N-p": "CNO",
    "7Li-p": "aneutronic",
    "6Li-p": "aneutronic",
}


@dataclass(frozen=True)
class FusionResonanceRow:
    """One fuel's resonance placement against the percolation hierarchy.

    The measured resonance tick n_res is compared to the event ticks
    from the fixed kernel p_c's on the two dials (tau inversion and
    delta) plus the rank ladder and Gamow-peak landmark. nearest_event
    is the closest; offset_ticks is the residual. sub_threshold flags a
    resonance below the weakest rank-ladder p_c. role is the fuel class
    (power / aneutronic / CNO). z_cutoff is True when Z1Z2 >=
    FUSION_Z1Z2_CUTOFF (literature terrestrial-viability boundary).
    PASS if |offset| <= tol and not sub_threshold.
    """

    fuel: str
    Z1Z2: int
    role: str
    z_cutoff: bool
    E_res_keV: float
    n_res: float
    E_event_keV: float
    n_event: float
    nearest_event: str
    offset_ticks: float
    tol_ticks: float
    passed: bool
    sub_threshold: bool


@dataclass(frozen=True)
class FusionResonanceSummary:
    """Null-model significance for the resonance-map gate.

    For the declared event set (n_events), the single-resonance hit
    probability p_single is the log-uniform window (2·tol ticks) over the
    sub-barrier band [E_band_lo, V_b]. expected_hits scales with the
    number of tested resonances. P_atleast_k is the Binomial(m, p_pool)
    tail for k hits out of m tested resonances; p_bonferroni multiplies
    by n_events (number of independent declared landmarks) for the
    family-wise significance of the observed pass count.
    """

    n_events: int
    tol_ticks: float
    band_lo_eV: float
    band_hi_eV: float
    p_single: float
    expected_hits: float
    n_passed: int
    n_tested: int
    p_atleast_passed: float
    p_bonferroni: float


@dataclass(frozen=True)
class FusionBarrierCheck:
    """Barrier placement and transmission-peak coincidence result.

    The CGM Coulomb barrier V_b is a placed grammar coordinate (the
    (k=3, d=5) strong-gravity dress of v*Delta^3). The CGM transmission
    maximum argmax[P_Gamow*theta(E/V_b)/E] coincides with n(V_b) for
    non-resonant fuels. True resonances below the barrier are reported
    as measured offsets, not as a universal code-atom claim.

    Fields: ticks n_Vb, n_peak, offset d_ticks; grammar class of the
    barrier; PASS flags at_strong_gravity and peak_coincides.
    """

    fuel: str
    Z1Z2: int
    Vb_MeV: float
    EG_MeV: float
    n_Vb: float
    n_Estr: float
    n_strong_gravity: float
    n_peak: float
    d_ticks: float
    grammar_class: str
    at_strong_gravity: bool
    peak_coincides: bool
    tol_ticks: float


@dataclass(frozen=True)
class ReactivityScan:
    rows: Tuple[ReactivityReading, ...]
    T_peak_gamow: float
    T_peak_cgm: float
    peak_shifted: bool
    T_enhancement_peak: float
    enhancement_shifted: bool


# -----------------------------------------------------------------
# ENSDF loaders
# -----------------------------------------------------------------
def load_ensdf_ev_band(
    path: Path = ENSDF_EV_BAND_PATH,
    *,
    require_halflife: bool = False,
) -> List[Tuple[str, float, str, str]]:
    """Load curated ENSDF levels with 0 < E ≤ 200 eV.

    Returns (label, E_eV, jp, source_note) rows. Empty if file missing.
    If require_halflife, keep only rows with ENSDF half-life (isomer-tagged).
    """
    if not path.is_file():
        return []
    out: List[Tuple[str, float, str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                E = float(row["E_eV"])
            except (KeyError, ValueError):
                continue
            if E <= 0.0:
                continue
            has_hl = (row.get("has_halflife") or "").strip().lower() in (
                "true",
                "1",
                "yes",
            )
            if not has_hl:
                # Backward compatible: infer from half_life column.
                has_hl = bool((row.get("half_life") or "").strip())
            if require_halflife and not has_hl:
                continue
            nuclide = (row.get("nuclide") or "").strip() or "unknown"
            jp = (row.get("jp") or "").strip()
            note = (row.get("note") or "").strip()
            src = (row.get("source") or "ENSDF").strip()
            tag = "iso" if has_hl else "lvl"
            label = f"{nuclide} ENSDF/{tag}"
            if note:
                src = f"{src}; {note}"
            out.append((label, E, jp, src))
    return out


def load_ensdf_first_excited(
    path: Path = ENSDF_FIRST_EXCITED_PATH,
) -> List[Tuple[str, float, str]]:
    """Load first-excited summary (E_eV, jp) for actinide level pulls."""
    if not path.is_file():
        return []
    out: List[Tuple[str, float, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                E = float(row["E_eV"])
            except (KeyError, ValueError):
                continue
            nuclide = (row.get("nuclide") or "").strip()
            jp = (row.get("jp") or "").strip()
            if nuclide:
                out.append((nuclide, E, jp))
    return out


# -----------------------------------------------------------------
# Null-model helpers
# -----------------------------------------------------------------
def null_hit_prob_loguniform(
    E_pred: float,
    tol_ticks: float,
    band_max_eV: float,
    *,
    E_band_lo_eV: float = 0.1,
) -> float:
    """
    P(random E in [E_band_lo, band_max] lands in |tick|≤tol of E_pred)
    under log-uniform measure on that band. Tick window width is
    2·tol·Δ in log2(E).
    """
    if E_pred <= 0.0 or band_max_eV <= E_band_lo_eV or tol_ticks <= 0.0:
        return 0.0
    band_log = math.log2(band_max_eV / E_band_lo_eV)
    win_log = 2.0 * tol_ticks * DELTA
    return float(min(1.0, max(0.0, win_log / band_log)))


def null_hit_prob_linear(
    E_lo: float,
    E_hi: float,
    band_max_eV: float,
) -> float:
    """P under uniform-in-energy on (0, band_max]."""
    if band_max_eV <= 0.0:
        return 0.0
    width = max(0.0, min(E_hi, band_max_eV) - max(E_lo, 0.0))
    return float(min(1.0, width / band_max_eV))


def binom_p_ge1(p: float, n: int) -> float:
    """P(K≥1) for Binomial(n,p) = 1−(1−p)^n."""
    if n <= 0 or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return 1.0 - (1.0 - p) ** n


def binom_p_ge(k: int, p: float, n: int) -> float:
    """P(K>=k) for Binomial(n,p), exact via math.comb (n is small here)."""
    from math import comb

    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return float(
        sum(comb(n, j) * (p**j) * ((1.0 - p) ** (n - j)) for j in range(k, n + 1))
    )


# -----------------------------------------------------------------
# Percolation-board helpers
# -----------------------------------------------------------------
def _fword_bytes(m: int) -> Tuple[int, int, int, int]:
    b0, b1, b2, b3 = (byte_from_family_micro(i, m, CHIRALITY_D) for i in range(4))
    return b0, b1, b2, b3


def _event_span(eng: HqvmD, allowed: Sequence[int]) -> bool:
    _, spans, _, _ = bfs_reach(eng, allowed)
    return spans


def _event_full(eng: HqvmD, allowed: Sequence[int]) -> bool:
    _, _, _, full = bfs_reach(eng, allowed)
    return full


def _event_spectrum(_eng: HqvmD, allowed: Sequence[int]) -> bool:
    if len(allowed) < 2:
        return False
    weights: set = set()
    for i, bi in enumerate(allowed):
        qi = q_word6(bi)
        for bj in allowed[i:]:
            weights.add((qi ^ q_word6(bj)).bit_count())
    return len(weights) == 7


def _event_word(_eng: HqvmD, allowed: Sequence[int]) -> bool:
    aset = set(allowed)
    for m in range(N_MICROREFS):
        if all(b in aset for b in _fword_bytes(m)):
            return True
    return False


def _mc_event_prob(
    eng: HqvmD,
    p: float,
    event_fn: Callable[[HqvmD, Sequence[int]], bool],
    *,
    n_samples: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    hits = nz = 0
    for _ in range(n_samples):
        allowed = [b for b in range(256) if rng.random() < p]
        if not allowed:
            continue
        nz += 1
        if event_fn(eng, allowed):
            hits += 1
    return hits / max(nz, 1)


def _bisect_p_c(
    eng: HqvmD,
    event_fn: Callable[[HqvmD, Sequence[int]], bool],
    *,
    n_samples: int = MC_HIERARCHY_SAMPLES,
    seed: int = MC_HIERARCHY_SEED,
) -> float:
    lo, hi = 0.0, 1.0
    for step in range(16):
        mid = (lo + hi) / 2.0
        prob = _mc_event_prob(eng, mid, event_fn, n_samples=n_samples, seed=seed + step)
        if prob < 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@lru_cache(maxsize=1)
def _kernel_percolation_board() -> PercolationBoard:
    eng = build_hqvm_d(CHIRALITY_D)
    ok, note = verify_d6_against_api()
    return PercolationBoard(
        d=CHIRALITY_D,
        p_c_rank=bisect_p_c_rank_micro_ref(CHIRALITY_D),
        p_c_span=_bisect_p_c(eng, _event_span),
        p_c_full=_bisect_p_c(eng, _event_full),
        p_c_spectrum=_bisect_p_c(eng, _event_spectrum),
        p_c_word=_bisect_p_c(eng, _event_word),
        d6_api_ok=ok,
        d6_api_note=note,
    )


# -----------------------------------------------------------------
# Output tee
# -----------------------------------------------------------------
class _Tee:
    """Mirror stdout to multiple streams for result capture."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()
