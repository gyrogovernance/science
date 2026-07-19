#!/usr/bin/env python3
"""
hqvm_cgm_trestleboard_4.py

Structural solutions and engineering deliverables from the placed-resonance
framework. Exact kernel primitives for A–D; holdout / discriminant tests for E–H.

  A. Resonance widths from percolation susceptibility.
     chi(p) = d theta / dp at each rank-ladder landmark p_c(r) sets
     Gamma_struct of that percolation event.

  B. Beta-decay matrix elements from carrier traces C(q).
     Fermi ↔ C(0); Gamow-Teller ↔ C(1); forbidden ladder from C(3), C(5).

  C. Nuclear magic numbers from the enumerator {C1,C2,C3}.
     Shell capacities C(6,k)*64 vs nuclear magic coincidence audit.

  D. Tensor force and the pion operator from the Delta_4 gap.
     Deuteron bare + tensor split; discrete pion as Delta_shell=2 move.

  E. Width scaling law (engineering). Gamma_struct(r) ~ chi(p_c(r))^-1
     monotone across fuels; PASS/FAIL.

  F. CGM sigma(E) surrogate on predictive holdout vs Bosch–Hale CSVs.
     One-DOF scale C; RMSE(log10) and Pearson r.

  G. Rider cutoff as internal discriminant R = n(V_b) - n_cut.
     Sign separation; CGM-anomaly report (p-B11).

  H. Sparse-data prediction targets: untested [first_CSV, p_c(1)*V_b] bands.
  I. Barrier radius sensitivity: r0 sweep on V_b placement.

  Fusion gates (moved from engine _1):
  fusion_validate, fusion_barrier_gate, fusion_resonance_events/map/null.

Companion notes: hqvm_cgm_trestleboard_notes.txt.
"""
from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gyroscopic.hQVM.family import (
    theta_micro_ref_exact,
)

from hqvm_cgm_trestleboard_common import (
    C1,
    C2,
    C3,
    CHIRALITY_D,
    DELTA,
    FUSION_FUEL_ROLE,
    FUSION_Z1Z2_CUTOFF,
    H_CARD,
    M_SHELL,
    OMEGA,
    SQRT5,
    V_GEV,
    FusionBarrierCheck,
    FusionResonanceRow,
    FusionResonanceSummary,
    FusionValidation,
    binom_p_ge,
)
from hqvm_cgm_trestleboard_1 import Trestleboard

_FUSION_DIR = _REPO / "data" / "catalogs" / "fusion"

# (label, Z1, Z2, A1, A2) matching the generated reference CSVs.
HOLDOUT_FUELS: List[Tuple[str, int, int, float, float]] = [
    ("D-T", 1, 1, 2.0, 3.0),
    ("D-D", 1, 1, 2.0, 2.0),
    ("D-3He", 1, 2, 2.0, 3.0),
    ("T-T", 1, 1, 3.0, 3.0),
    ("3He-3He", 2, 2, 3.0, 3.0),
    ("p-6Li", 1, 3, 1.0, 6.0),
    ("p-B11", 1, 5, 1.0, 11.0),
]

# Fuel suite with the terrestrial-viability boundary for the discriminant.
RIDER_SUITE: List[Tuple[str, int, int, float, float]] = [
    ("D-T", 1, 1, 2.0, 3.0),
    ("D-D", 1, 1, 2.0, 2.0),
    ("D-3He", 1, 2, 2.0, 3.0),
    ("p-6Li", 1, 3, 1.0, 6.0),
    ("p-B11", 1, 5, 1.0, 11.0),
    ("10B-p", 1, 5, 10.0, 11.0),
    ("12C-p", 1, 6, 12.0, 13.0),
    ("15N-p", 1, 7, 15.0, 16.0),
]


# -----------------------------------------------------------------
# A. Resonance widths from percolation susceptibility
# -----------------------------------------------------------------
def susceptibility_chi(p: float, d: int = CHIRALITY_D, *, h: float = 1e-6) -> float:
    """Exact coverage susceptibility chi(p) = d theta / dp.

    theta_micro_ref_exact is the closed-form coverage fraction; its
    derivative with respect to the inclusion probability p is the
    susceptibility of the accessibility phase transition. Finite
    central difference on the exact function (no Monte Carlo).
    """
    return (theta_micro_ref_exact(p + h, d) - theta_micro_ref_exact(p - h, d)) / (2 * h)


def resonance_width_table() -> List[Tuple[str, float, float, float, float]]:
    """Structural width Gamma_struct for each rank-ladder landmark.

    Returns (label, p_c, chi(p_c), rank r, Gamma_struct).
    Gamma_struct is the inverse of the susceptibility-normalized width:
    a sharp transition (large chi) is a narrow structural resonance.
    The scale is set so that chi at the full-root onset (r=6) is the
    reference unit Gamma_0.

    chi(p) carries units of coverage per unit inclusion probability;
    Gamma_struct = Gamma_0 * chi_ref / chi(p) is a relative width on
    the [0,1] percolation axis, dimensionless and directly comparable
    across landmarks.
    """
    ladder = Trestleboard.rank_ladder_p_c()
    out: List[Tuple[str, float, float, float, float]] = []
    chi_at: List[Tuple[int, float, float]] = []
    for r, p in enumerate(ladder):
        chi = susceptibility_chi(p)
        chi_at.append((r, p, chi))
    chi_ref = max(c[2] for c in chi_at)
    for r, p, chi in chi_at:
        # p_c=0 (r=0 gauge doublet) has no finite width; report chi only.
        if p <= 1e-12:
            gamma = float("nan")
        else:
            gamma = chi_ref / chi
        out.append((f"r={r}", p, chi, float(r), gamma))
    return out


def fusion_resonance_widths(
    tb: Trestleboard, *, fuels: List[Tuple[str, int, int, float, float]]
) -> List[Tuple[str, float, float, str, float]]:
    """Map each fuel's per-resonance structural width.

    For a fuel with barrier V_b, the rank-ladder landmark at p_c(r)
    sits at energy E = p_c(r) * V_b. The susceptibility chi(p_c(r))
    at that inclusion sets the structural width of the event in the
    fuel's own energy units (keV). Returns
    (fuel, E_landmark_keV, chi, nearest_rank, Gamma_struct_keV).
    """
    widths = resonance_width_table()
    chi_by_rank = {int(r): chi for _, _, chi, r, _ in widths}
    gamma_by_rank = {int(r): g for _, _, _, r, g in widths if not math.isnan(g)}
    out: List[Tuple[str, float, float, str, float]] = []
    for label, Z1, Z2, A1, A2 in fuels:
        Vb = tb.coulomb_barrier_MeV(Z1, Z2, A1, A2)
        ladder = tb.rank_ladder_p_c()
        for r, p in enumerate(ladder):
            if p <= 1e-12:
                continue
            E = p * Vb * 1e3
            chi = chi_by_rank[r]
            gamma = gamma_by_rank[r]
            out.append((label, E, chi, f"r={r}", gamma))
    return out


# -----------------------------------------------------------------
# B. Beta-decay matrix elements from carrier traces C(q)
# -----------------------------------------------------------------
@dataclass(frozen=True)
class BetaMatrixResult:
    """Carrier-trace beta matrix elements.

    C[q] is the exact carrier trace Tr(M_q^2) [odd q] / Tr(M_q)
    [even q]. Fermi proxy |M_F|^2 = C[0]; Gamow-Teller proxy
    |M_GT|^2 = C[1]. forbidden_ratio[q] = C[q]/C[1] for odd q.
    """

    C: Dict[int, Fraction]
    M_F2: Fraction
    M_GT2: Fraction
    F_over_GT: Fraction
    GT_over_F: Fraction
    forbidden_ratio: Dict[int, Fraction]
    H_L_alpha: Fraction


@dataclass(frozen=True)
class MagicNumberResult:
    """Magic-number audit against structural CGM filling limits."""

    magic: List[int]
    shell_capacities: List[int]
    cumulative_capacities: List[int]
    code_gap_arithmetic: Dict[str, int]
    hits: List[Tuple[int, str]]
    n_coincide: int


def carrier_trace(q: int) -> Fraction:
    """C(q) = Tr(M_q^2) for odd q; Tr(M_q) for even q (exact api).

    The kernel shell-transition matrix M_q has entries
    P(w' | w, q) = C(w,t)C(6-w,j-t)/C(6,j) with j=q, t=(w+q-w')/2.
    For even q, Tr(M_q) is nonzero and equals 7/(q+1). For odd q the
    trace vanishes and C(q) = Tr(M_q^2) (return-trace, Krawtchouk sum).
    """
    from gyroscopic.hQVM.api import shell_transition_matrix_for_q_weight

    mat = shell_transition_matrix_for_q_weight(q)
    tr = sum(mat[w][w] for w in range(7))
    if tr != 0:
        return Fraction(tr)
    return Fraction(sum(mat[w][wp] * mat[wp][w] for w in range(7) for wp in range(7)))


def beta_matrix_elements() -> BetaMatrixResult:
    """Fermi and Gamow-Teller carrier-trace amplitudes.

    Fermi (Delta J = 0, no parity change, even parity):
      |M_F|^2 proxy = C(0) = 7  (identity carrier trace).
    Gamow-Teller (Delta J = 0, +/-1, no parity change):
      the odd-shell flip is the q=1,3,5 return trace; the canonical
      GT carrier amplitude is C(1) = 28/9 (single-dipole-flip path
      count), and the higher-odd ladder C(3), C(5) are the
      first/second-forbidden companions.

    Allowed (F + GT) strongest; first-forbidden scales by the
    C(2k+1)/C(2k) carrier-trace ratio relative to allowed.
    """
    C = {q: carrier_trace(q) for q in range(7)}
    M_F2 = C[0]  # Fermi proxy
    M_GT2 = C[1]  # Gamow-Teller proxy (allowed)
    # F/GT ratio in the carrier-trace basis (dimensionless path count).
    f_gt_ratio = Fraction(M_GT2, M_F2)
    # Forbidden ladder: odd-shell return traces vs allowed C(1).
    forbidden: Dict[int, Fraction] = {}
    for q in (3, 5):
        forbidden[q] = Fraction(C[q], C[1])
    return BetaMatrixResult(
        C=C,
        M_F2=M_F2,
        M_GT2=M_GT2,
        F_over_GT=Fraction(M_F2, M_GT2),
        GT_over_F=f_gt_ratio,
        forbidden_ratio=forbidden,
        # alpha-tunneling homology: H_L = C(2)/C(0) = 1/3 (two-dipole
        # over identity path count), the known discrete result.
        H_L_alpha=Fraction(C[2], C[0]),
    )


# -----------------------------------------------------------------
# C. Nuclear magic numbers from the enumerator {C1,C2,C3}
# -----------------------------------------------------------------
def shell_capacities() -> List[int]:
    """Carrier shell capacities: C(6,k)*64 for k=0..6."""
    return [comb(6, k) * 64 for k in range(7)]


def cumulative_capacities() -> List[int]:
    """Cumulative carrier filling limits (running sum of shells)."""
    caps = shell_capacities()
    out: List[int] = []
    acc = 0
    for c in caps:
        acc += c
        out.append(acc)
    return out


def code_gap_arithmetic() -> Dict[str, int]:
    """Structural numbers from the enumerator {C1,C2,C3}."""
    return {
        "C1": C1,
        "C2": C2,
        "C3": C3,
        "WZ_gap": C2 - C1,  # = 9 = 3^2
        "horizon": H_CARD,  # = 64 = 2^6
        "omega": OMEGA,  # = 4096 = 2^12
        "M_shell": M_SHELL,  # = 192
        "pred_P4": 3 * 2**3,  # = 48
        "pred_P7": 3 * 2**6,  # = 384
        "pred_P10": 3 * 2**9,  # = 3072
    }


def magic_number_audit() -> MagicNumberResult:
    """Compare nuclear magic numbers to structural CGM filling limits.

    Nuclear magic numbers: 2, 8, 20, 28, 50, 82, 126.
    Structural CGM limits: shell capacities C(6,k)*64, cumulative
    sums, and the code-gap arithmetic. Report per magic number whether
    it coincides with a structural limit and which one.
    """
    magic = [2, 8, 20, 28, 50, 82, 126]
    caps = shell_capacities()
    cum = cumulative_capacities()
    arith = code_gap_arithmetic()
    struct_set = set(caps) | set(cum) | set(arith.values())
    hits: List[Tuple[int, str]] = []
    for m in magic:
        if m in caps:
            hits.append((m, "shell-capacity"))
        elif m in cum:
            hits.append((m, "cumulative-capacity"))
        elif m in struct_set:
            hits.append((m, "code-gap-arithmetic"))
        else:
            hits.append((m, "no-coincidence"))
    return MagicNumberResult(
        magic=magic,
        shell_capacities=caps,
        cumulative_capacities=cum,
        code_gap_arithmetic=arith,
        hits=hits,
        n_coincide=sum(1 for _, how in hits if how != "no-coincidence"),
    )


# -----------------------------------------------------------------
# D. Tensor force and the pion operator from the Delta_4 gap
# -----------------------------------------------------------------
def deuteron_tensor_split() -> Dict[str, float]:
    """Decompose deuteron binding into bare v*Delta^3 and tensor term.

    E_D = v*Delta^3 + v*Delta^4*(2/sqrt(5)).
    The 2/sqrt(5) is the discrete trace-free quadrupole correction
    (the tensor-force fraction in the CGM carrier frame).
    """
    E_bare = V_GEV * (DELTA**3) * 1e3
    E_tensor = V_GEV * (DELTA**4) * (2.0 / SQRT5) * 1e3
    return {
        "E_bare_MeV": E_bare,
        "E_tensor_MeV": E_tensor,
        "E_total_MeV": E_bare + E_tensor,
        "tensor_fraction": E_tensor / (E_bare + E_tensor),
        "quadrupole_coeff": 2.0 / SQRT5,
    }


def pion_operator_relation() -> Dict[str, object]:
    """Discrete pion counterpart and Delta_4 / pion-mass relation.

    The pion is the Goldstone boson of chiral symmetry breaking. In the
    discrete CGM frame the trace-free quadrupole correction v*Delta^4*
    (2/sqrt(5)) comes from the Delta_shell=2 isospin-flip carrier move
    (two dipole pairs flipped in one byte step). That move is the
    minimal spin-0, isospin-1 carrier excitation — the discrete pion.

    The Delta_4 gap (the step from the bare Delta^3 to the Delta^4
    tensor term) sets the chiral-symmetry-breaking scale. We state the
    ratio Delta_4/Delta_3 = Delta and its carrier-trace content.
    """
    # Delta_4 / Delta_3 ratio in the energy exponent is exactly Delta.
    ratio = DELTA
    # Pion-mass scale proxy: the tensor correction energy over the bare.
    tensor = V_GEV * (DELTA**4) * (2.0 / SQRT5) * 1e3
    bare = V_GEV * (DELTA**3) * 1e3
    m_pion_proxy_MeV = tensor  # discrete chiral-scale energy
    return {
        "Delta_4_over_Delta_3": ratio,
        "bare_MeV": bare,
        "tensor_MeV": tensor,
        "pion_proxy_MeV": m_pion_proxy_MeV,
        "isospin_flip_operator": "Delta_shell=2 (two-dipole-flip byte step)",
        "pion_carrier_move": "q_weight=2 return trace C(2)=7/3",
        "note": "Delta_4 gap = chiral-symmetry-breaking scale in the "
        "discrete frame; the q=2 carrier move is the spin-0 "
        "isospin-1 Goldstone counterpart.",
    }


# -----------------------------------------------------------------
# E. Width scaling law (engineering)
# -----------------------------------------------------------------
@dataclass(frozen=True)
class WidthScalingResult:
    """Gamma_struct(r) = chi_ref/chi(p_c(r)) across rank rungs."""

    ladder: List[float]
    chi_ref: float
    per_fuel: Dict[str, List[Tuple[int, float, float, float, float]]]
    monotone_inverse_ok: bool


@dataclass(frozen=True)
class SurrogateResult:
    """Holdout score of the CGM sigma(E) surrogate."""

    status: str
    label: str = ""
    n_cal: int = 0
    n_hold: int = 0
    C: float = 1.0
    rmse_log: float = float("nan")
    pearson_r: float = float("nan")


@dataclass(frozen=True)
class RiderDiscriminantResult:
    """Internal CGM discriminant R = n(V_b) - n_cut at Z1Z2 cutoff."""

    cutoff: int
    n_cut_ticks: float
    rows: List[Tuple[str, int, float, float, float, bool, bool]]
    sign_separated: bool
    anomalies: List[str]


def load_fusion_csv(label: str) -> List[Tuple[float, float]]:
    path = _FUSION_DIR / f"{label}.csv"
    if not path.is_file():
        return []
    out: List[Tuple[float, float]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                E = float(row["E_keV"])
                S = float(row["S_ref"])
            except (KeyError, ValueError):
                continue
            out.append((E, S))
    return out


def width_scaling_law(tb: Trestleboard) -> WidthScalingResult:
    """Test Gamma_struct(r) ~ chi(p_c(r))^-1 across rank rungs.

    For every fuel, the landmark at rank r sits at E = p_c(r)*V_b. The
    susceptibility chi(p_c(r)) is the graph-density at that energy; the
    structural width is Gamma_struct = chi_ref / chi(p_c(r)). The scaling
    law is monotone inverse: sharper transition (larger chi) -> narrower
    width. Test the monotone ordering across r for each fuel.
    """
    ladder = tb.rank_ladder_p_c()
    chi_ref = max(susceptibility_chi(p) for p in ladder if p > 1e-12)
    per_fuel: Dict[str, List[Tuple[int, float, float, float, float]]] = {}
    monotone_ok = True
    for label, Z1, Z2, A1, A2 in HOLDOUT_FUELS:
        Vb = tb.coulomb_barrier_MeV(Z1, Z2, A1, A2)
        rows: List[Tuple[int, float, float, float, float]] = []
        prev_gamma = None
        for r, p in enumerate(ladder):
            if p <= 1e-12:
                continue
            chi = susceptibility_chi(p)
            gamma = chi_ref / chi
            E = p * Vb * 1e3
            rows.append((r, p, E, chi, gamma))
            if prev_gamma is not None and gamma > prev_gamma + 1e-9:
                monotone_ok = False
            prev_gamma = gamma
        per_fuel[label] = rows
    return WidthScalingResult(
        ladder=ladder,
        chi_ref=chi_ref,
        per_fuel=per_fuel,
        monotone_inverse_ok=monotone_ok,
    )


# -----------------------------------------------------------------
# F. CGM surrogate on holdout
# -----------------------------------------------------------------
def surrogate_holdout(
    tb: Trestleboard,
    *,
    label: str,
    Z1: int,
    Z2: int,
    A1: float,
    A2: float,
) -> SurrogateResult:
    """Calibration / holdout test of sigma_CGM(E).

    Calibration = even-indexed CSV rows; fit scale C to match the
    reference sigma. Holdout = odd-indexed rows.
    Score on holdout: RMSE(log10 sigma_CGM/sigma_ref) and Pearson r.
    """
    rows = load_fusion_csv(label)
    if len(rows) < 6:
        return SurrogateResult(status="no_data", label=label)
    EG = tb.gamow_energy_MeV(Z1, Z2, A1, A2)
    Vb = tb.coulomb_barrier_MeV(Z1, Z2, A1, A2)

    cal = rows[0::2]
    hold = rows[1::2]

    def ref_sigma(E_keV: float, S: float) -> float:
        E_MeV = E_keV / 1e3
        Pg = tb.gamow_factor(E_MeV, EG)
        return S * Pg / max(E_keV, 1e-30)

    def cgm_sigma(E_keV: float, C: float) -> float:
        E_eV = E_keV * 1e3
        E_MeV = E_keV / 1e3
        p = min(1.0, max(0.0, E_eV / (Vb * 1e6)))
        th = tb.theta_from_inclusion(p)
        Pg = tb.gamow_factor(E_MeV, EG)
        # Model 2 footing: sigma = C * (S0/E) * P_Gamow * theta(E/V_b).
        return C * (1.0 / max(E_keV, 1e-30)) * Pg * th

    cal_ref = [ref_sigma(E, S) for E, S in cal]
    cal_cgm_raw = [cgm_sigma(E, 1.0) for E, _ in cal]
    num = sum(r * w for r, w in zip(cal_ref, cal_cgm_raw))
    den = sum(w * w for w in cal_cgm_raw)
    C = num / den if den > 0 else 1.0

    if not hold:
        return SurrogateResult(status="no_holdout", label=label)
    hr = [ref_sigma(E, S) for E, S in hold]
    hc = [cgm_sigma(E, C) for E, _ in hold]
    log_res = [math.log10(max(c, 1e-300) / max(r, 1e-300)) for c, r in zip(hc, hr)]
    rmse = math.sqrt(sum(x * x for x in log_res) / len(log_res))
    n = len(hr)
    mr = sum(hr) / n
    mc = sum(hc) / n
    cov = sum((c - mc) * (r - mr) for c, r in zip(hc, hr))
    vr = sum((r - mr) ** 2 for r in hr) ** 0.5
    vc = sum((c - mc) ** 2 for c in hc) ** 0.5
    r = cov / (vr * vc) if vr > 0 and vc > 0 else float("nan")
    return SurrogateResult(
        status="ok",
        label=label,
        n_cal=len(cal),
        n_hold=len(hold),
        C=C,
        rmse_log=rmse,
        pearson_r=r,
    )


# -----------------------------------------------------------------
# G. Rider cutoff discriminant
# -----------------------------------------------------------------
def rider_discriminant(tb: Trestleboard) -> RiderDiscriminantResult:
    """Internal CGM discriminant R that flips sign near Z1Z2=7.

      R(Z1Z2) = n(V_b(Z1Z2)) - n_cut

    n_cut is the barrier tick of the Z1Z2=7 fuel (15N-p). R > 0 on the
    accessible side; R < 0 above the cutoff. Derived from barrier
    placement alone.
    """
    n_cut = None
    all_n: List[Tuple[str, int, float]] = []
    for label, Z1, Z2, A1, A2 in RIDER_SUITE:
        Vb = tb.coulomb_barrier_MeV(Z1, Z2, A1, A2)
        n_Vb = tb.n_of_E(Vb * 1e6)
        z = Z1 * Z2
        all_n.append((label, z, n_Vb))
        if z == FUSION_Z1Z2_CUTOFF:
            n_cut = n_Vb
    if n_cut is None:
        n_cut = min(n for _, _, n in all_n)
    rows: List[Tuple[str, int, float, float, float, bool, bool]] = []
    correctly_separated = True
    anomalies: List[str] = []
    for label, z, n_Vb in all_n:
        R = n_Vb - n_cut
        below = z < FUSION_Z1Z2_CUTOFF
        if z != FUSION_Z1Z2_CUTOFF:
            ok = (R > 0) == below
            if not ok:
                anomalies.append(label)
            correctly_separated &= ok
        rows.append((label, z, n_Vb, R, R, below, (R > 0) == below))
    return RiderDiscriminantResult(
        cutoff=FUSION_Z1Z2_CUTOFF,
        n_cut_ticks=n_cut,
        rows=rows,
        sign_separated=correctly_separated,
        anomalies=anomalies,
    )


# -----------------------------------------------------------------
# H. Sparse-data prediction target
# -----------------------------------------------------------------
def sparse_target(tb: Trestleboard) -> List[Tuple[str, float, float, float, str]]:
    """Fuel/energy window where CGM predicts structure but CSV is silent.

    Untested band = [first_CSV_point, p_c(1)*V_b] per holdout fuel.
    """
    out: List[Tuple[str, float, float, float, str]] = []
    for label, Z1, Z2, A1, A2 in HOLDOUT_FUELS:
        Vb = tb.coulomb_barrier_MeV(Z1, Z2, A1, A2)
        ladder = tb.rank_ladder_p_c()
        p1 = ladder[1]
        E_landmark = p1 * Vb * 1e3
        rows = load_fusion_csv(label)
        first_csv = min(E for E, _ in rows) if rows else float("nan")
        if math.isnan(first_csv):
            window = float("nan")
            note = "no CSV: entire sub-barrier band untested"
        else:
            window = max(0.0, E_landmark - first_csv)
            note = (
                f"untested band [{first_csv:.0f},{E_landmark:.0f}] keV "
                f"= {window:.0f} keV wide"
            )
        out.append((label, first_csv, E_landmark, window, note))
    return out


# -----------------------------------------------------------------
# I. Barrier radius sensitivity (falsification)
# -----------------------------------------------------------------
def _coulomb_barrier_r0(
    Z1: int, Z2: int, A1: float, A2: float, r0_fm: float
) -> float:
    """V_b (MeV) with variable nuclear-radius prefactor r0 (fm)."""
    r_fm = r0_fm * (A1 ** (1.0 / 3.0) + A2 ** (1.0 / 3.0))
    return 1.44 * Z1 * Z2 / r_fm


def barrier_radius_sensitivity(
    tb: Trestleboard,
    *,
    r0_values: Tuple[float, ...] = (1.1, 1.2, 1.3, 1.4),
    r0_ref: float = 1.2,
    tol_ticks: float = 7.0,
) -> List[Tuple[str, float, float, float, float, int, bool]]:
    """Vary r0 in V_b = 1.44 Z1 Z2 / [r0 (A1^{1/3}+A2^{1/3})].

    For each holdout fuel and r0, report (fuel, r0, Vb_MeV, n(Vb),
    dn vs r0_ref, k_class, on_strong_ladder). Prints nothing; caller
    reports. Default r0_ref=1.2 matches coulomb_barrier_MeV.
    """
    out: List[Tuple[str, float, float, float, float, int, bool]] = []
    for label, Z1, Z2, A1, A2 in HOLDOUT_FUELS:
        Vb_ref = _coulomb_barrier_r0(Z1, Z2, A1, A2, r0_ref)
        n_ref = tb.n_of_E(Vb_ref * 1e6)
        for r0 in r0_values:
            Vb = _coulomb_barrier_r0(Z1, Z2, A1, A2, r0)
            n_Vb = tb.n_of_E(Vb * 1e6)
            lr = tb.level(Vb * 1e6)
            out.append(
                (
                    label,
                    float(r0),
                    Vb,
                    n_Vb,
                    n_Vb - n_ref,
                    int(lr.cls.k),
                    lr.cls.k == 3,
                )
            )
    return out


def barrier_radius_sensitivity_summary(
    tb: Trestleboard,
    *,
    r0_values: Tuple[float, ...] = (1.1, 1.2, 1.3, 1.4),
    r0_ref: float = 1.2,
) -> dict:
    """Aggregate radius-sensitivity checks for the report."""
    rows = barrier_radius_sensitivity(tb, r0_values=r0_values, r0_ref=r0_ref)
    dn_abs = [abs(dn) for *_, dn, _k, _ok in rows]
    k3 = [ok for *_, ok in rows]
    # Per-fuel max |dn| over r0 away from ref
    per_fuel: Dict[str, float] = {}
    for label, r0, _Vb, _n, dn, _k, _ok in rows:
        if abs(r0 - r0_ref) < 1e-12:
            continue
        per_fuel[label] = max(per_fuel.get(label, 0.0), abs(dn))
    return {
        "r0_values": r0_values,
        "r0_ref": r0_ref,
        "n_rows": len(rows),
        "all_k3": all(k3),
        "max_abs_dn": max(dn_abs) if dn_abs else 0.0,
        "max_abs_dn_off_ref": max(per_fuel.values()) if per_fuel else 0.0,
        "per_fuel_max_abs_dn": per_fuel,
        "rows": rows,
    }


# -----------------------------------------------------------------
# Fusion gates (engine validation / resonance map; was on Trestleboard)
# -----------------------------------------------------------------
# ---- FUSION VALIDATION (external cross-section gate) ----
def fusion_validate(
    tb: Trestleboard,
    *,
    Z1: int,
    Z2: int,
    A1: float,
    A2: float,
    ref_csv: Optional[Path] = None,
    band_keV: Tuple[float, float] = (10.0, 200.0),
    E_ref_keV: float = 10.0,
) -> FusionValidation:
    """External check of the CGM fusion cross-section claim.

    The model cross-section shape (E_ref-normalised sigma_rel_cgm)
    is compared to the reference cross-section shape reconstructed
    from the Bosch-Hale S-factor as sigma_ref(E) = S_ref(E) *
    P_Gamow(E) / E, also normalised to its value at E_ref. Both
    are dimensionless and directly comparable.

    Metrics over the declared band: RMSE(log10) and Pearson r
    between the model and reference cross-section shapes after a
    scale-only normalisation to the reference median.

    With no reference CSV the gate returns status="pending"
    (a complete, runnable gate that reports PENDING until data
    exists — not an "open" item).
    """
    Vb = tb.coulomb_barrier_MeV(Z1, Z2, A1, A2)
    EG = tb.gamow_energy_MeV(Z1, Z2, A1, A2)
    grid = [float(E) for E in range(int(band_keV[0]), int(band_keV[1]) + 1, 1)]
    # Model cross-section shape: relative to the E_ref value
    # (sigma_rel_cgm is already E_ref-normalised and dimensionless).
    model_sig = []
    for E in grid:
        row = tb.cross_section_relative(
            E, Z1=Z1, Z2=Z2, A1=A1, A2=A2, E_ref_keV=E_ref_keV
        )
        model_sig.append(row.sigma_rel_cgm)
    model_peak_idx = max(range(len(model_sig)), key=lambda i: model_sig[i])
    model_peak_keV = grid[model_peak_idx]
    if ref_csv is None or not Path(ref_csv).is_file():
        return FusionValidation(
            status="pending",
            fuel=f"{Z1}+{Z2} A{A1:.0f}/{A2:.0f}",
            n_points=len(grid),
            rmse_log=float("nan"),
            corr=float("nan"),
            corr_struct=float("nan"),
            model_peak_keV=model_peak_keV,
            band_keV=band_keV,
            note="reference CSV not loaded; gate PENDING",
        )
    E_ref = []
    S_ref = []
    with Path(ref_csv).open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            e = float(r["E_keV"])
            if e < band_keV[0] or e > band_keV[1]:
                continue
            E_ref.append(e)
            S_ref.append(float(r["S_ref"]))

    # Reconstruct the reference cross-section sigma_ref = S(E)*P_Gamow/E,
    # then normalise to its value at E_ref so it is dimensionless and
    # directly comparable to the model's E_ref-normalised sigma.
    def ref_sigma_at(E_keV: float) -> float:
        E_MeV = E_keV / 1e3
        Pg = tb.gamow_factor(E_MeV, EG)
        return (
            S_ref[sorted(range(len(E_ref)), key=lambda i: abs(E_ref[i] - E_keV))[0]]
            * Pg
            / max(E_keV, 1e-30)
        )

    # Build ref sigma on the ref grid, normalise by value at E_ref.
    ref_sig = [ref_sigma_at(e) for e in E_ref]
    # value of ref sigma at E_ref (nearest grid point)
    iref = sorted(range(len(E_ref)), key=lambda i: abs(E_ref[i] - E_ref_keV))[0]
    ref_norm = max(ref_sig[iref], 1e-300)
    ref_sig_n = [s / ref_norm for s in ref_sig]
    # interpolate model onto the ref grid
    g = sorted(grid)
    ms = [x for _, x in sorted(zip(grid, model_sig))]

    def interp(xq, xs, ys):
        if xq <= xs[0]:
            return ys[0]
        if xq >= xs[-1]:
            return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= xq <= xs[i + 1]:
                t = (xq - xs[i]) / (xs[i + 1] - xs[i])
                return ys[i] + t * (ys[i + 1] - ys[i])
        return ys[-1]

    model_at_ref = [interp(e, g, ms) for e in E_ref]
    # scale-only normalisation to ref median (overall shape check)
    med_m = sorted(model_at_ref)[len(model_at_ref) // 2]
    med_r = sorted(ref_sig_n)[len(ref_sig_n) // 2]
    scale = med_r / max(med_m, 1e-300)
    model_norm = [x * scale for x in model_at_ref]
    log_res = [
        math.log10(max(m, 1e-300) / max(s, 1e-300))
        for m, s in zip(model_norm, ref_sig_n)
    ]
    rmse = math.sqrt(sum(x * x for x in log_res) / len(log_res))
    nm = sum(model_norm) / len(model_norm)
    ns = sum(ref_sig_n) / len(ref_sig_n)
    cov = sum((m - nm) * (s - ns) for m, s in zip(model_norm, ref_sig_n))
    vm = sum((m - nm) ** 2 for m in model_norm) ** 0.5
    vs = sum((s - ns) ** 2 for s in ref_sig_n) ** 0.5
    corr = cov / max(vm * vs, 1e-300)

    # STRUCTURAL footing: Gamow stripped from both sides.
    # Reference S_ref is already Gamow-divided by Bosch-Hale/Tentori.
    # Model structural S-equivalent = sigma_rel_cgm * E (drops the 1/E
    # and any Gamow in the model). Compare to S_ref directly.
    model_S = [m * e for m, e in zip(model_at_ref, E_ref)]
    med_mS = sorted(model_S)[len(model_S) // 2]
    med_rS = sorted(S_ref)[len(S_ref) // 2]
    scaleS = med_rS / max(med_mS, 1e-300)
    model_Sn = [x * scaleS for x in model_S]
    lrS = [math.log10(max(s, 1e-300)) for s in S_ref]
    lmS = [math.log10(max(x, 1e-300)) for x in model_Sn]
    nS = len(lrS)
    mrS = sum(lrS) / nS
    mmS = sum(lmS) / nS
    covS = sum((a - mrS) * (b - mmS) for a, b in zip(lrS, lmS))
    vrS = sum((a - mrS) ** 2 for a in lrS)
    vmS = sum((b - mmS) ** 2 for b in lmS)
    corr_struct = covS / math.sqrt(vrS * vmS) if vrS > 0 and vmS > 0 else float("nan")

    return FusionValidation(
        status="validated",
        fuel=f"{Z1}+{Z2} A{A1:.0f}/{A2:.0f}",
        n_points=len(E_ref),
        rmse_log=rmse,
        corr=corr,
        corr_struct=corr_struct,
        model_peak_keV=model_peak_keV,
        band_keV=band_keV,
        note=f"cross-section r={corr:.3f}; structural S r={corr_struct:.3f}",
    )


def fusion_barrier_gate(
    tb: Trestleboard,
    *,
    Z1: int,
    Z2: int,
    A1: float,
    A2: float,
    E_grid_MeV: Tuple[float, float] = (0.002, 2.0),
    n_steps: int = 2000,
) -> FusionBarrierCheck:
    """Barrier placement and transmission-peak coincidence gate.

    Two parameter-free grammar predictions are tested:

    1. Barrier placement. The CGM Coulomb barrier V_b (computed from
       Z1,Z2,A1,A2 with standard nuclear radii, no free parameters) is
       a placed coordinate. Its Delta-ruler tick n(V_b) lands on the
       strong-family grammar class (k=3, d), the STF-gravity dress of
       the strong bare scale v*Delta^3. The dress rank d varies with
       Z1Z2 (heavier charge products sit lower on the strong ladder).

    2. Transmission-peak coincidence. The CGM transmission maximum
       argmax [ P_Gamow(E) * theta(E/V_b) / E ] coincides with the
       barrier tick n(V_b) for non-resonant fuels (the peak is the
       maximum transmission *through* the placed barrier).

    Resonance offsets (fuels with a true nuclear resonance below the
    barrier) are reported as measured residuals, not as a universal
    code-atom claim.

    Inputs: projectile/product quantum numbers (Z1,Z2,A1,A2).
    Outputs: FusionBarrierCheck with ticks, classes, and PASS flags.
    Companion: run.py 6h) prints the gate; common.FusionBarrierCheck
    holds the result.
    """
    Vb = tb.coulomb_barrier_MeV(Z1, Z2, A1, A2)
    EG = tb.gamow_energy_MeV(Z1, Z2, A1, A2)
    n_Vb = tb.n_of_E(Vb * 1e6)

    # CGM transmission maximum on a fine E grid below the barrier.
    E_lo, E_hi = E_grid_MeV
    E_hi = min(E_hi, Vb * 0.999)
    best, best_E = 0.0, E_lo
    for i in range(1, n_steps + 1):
        E = E_lo + (E_hi - E_lo) * i / n_steps
        p = E / Vb
        th = tb.theta_from_inclusion(p)
        tau = 2.0 * math.pi * math.sqrt(EG / E) * (1.0 - math.sqrt(E / Vb))
        s = (1.0 / E) * th * math.exp(-tau)
        if s > best:
            best, best_E = s, E
    n_peak = tb.n_of_E(best_E * 1e6)
    d_ticks = n_peak - n_Vb

    # Grammar class of the barrier tick via the level reading.
    lr = tb.level(Vb * 1e6)
    gram_cls = f"({lr.cls.k},{lr.cls.d}) {lr.cls.label}"
    # Barrier is a placed coordinate on the strong-family ladder (k=3).
    on_strong_ladder = lr.cls.k == 3

    # Tolerance: fusion peak precision ~10% energy -> ~6.6 ticks.
    tol = 7.0
    peak_coincides = abs(d_ticks) <= tol

    return FusionBarrierCheck(
        fuel=f"{Z1}+{Z2} A{A1:.0f}/{A2:.0f}",
        Z1Z2=Z1 * Z2,
        Vb_MeV=Vb,
        EG_MeV=EG,
        n_Vb=n_Vb,
        n_Estr=float("nan"),
        n_strong_gravity=float("nan"),
        n_peak=n_peak,
        d_ticks=d_ticks,
        grammar_class=gram_cls,
        at_strong_gravity=on_strong_ladder,
        peak_coincides=peak_coincides,
        tol_ticks=tol,
    )


def fusion_resonance_events(
    tb: Trestleboard,
    *,
    Z1: int,
    Z2: int,
    A1: float,
    A2: float,
) -> List[Tuple[str, float]]:
    """All declared resonance-landmark energies (keV) for a fuel.

    Union of the six fixed percolation events and the rank-ladder
    events (tau and delta dials), plus the Gamow-peak landmark
    E_G/4. Every entry is predeclared by the kernel grammar; none is
    fitted to data.
    """
    pc = tb.percolation
    EG = tb.gamow_energy_MeV(Z1, Z2, A1, A2)
    Vb = tb.coulomb_barrier_MeV(Z1, Z2, A1, A2)
    ladder = tb.rank_ladder_p_c()
    ev: List[Tuple[str, float]] = [
        ("E_span", tb.E_event_for_Vb(EG, Vb, pc.p_c_span)),
        ("E_full", tb.E_event_for_Vb(EG, Vb, pc.p_c_full)),
        ("E_spec", tb.E_event_for_Vb(EG, Vb, pc.p_c_spectrum)),
        ("E_tau", tb.E_event_for_Vb(EG, Vb, pc.p_c_rank)),
        ("E_word", tb.E_event_for_Vb(EG, Vb, pc.p_c_word)),
        ("E_delta", pc.p_c_rank * Vb * 1e3),
        ("E_Gamow", EG * 1e3 / 4.0),
    ]
    for r, p in enumerate(ladder):
        ev.append((f"E_tau_r{r}", tb.E_event_for_Vb(EG, Vb, p)))
        ev.append((f"E_delta_r{r}", p * Vb * 1e3))
    return ev


def fusion_resonance_map(
    tb: Trestleboard,
    *,
    resonances: List[Tuple[str, int, int, float, float, float, float]],
    tol_ticks: float = 7.0,
) -> List[FusionResonanceRow]:
    """Map literature fusion resonances onto the percolation hierarchy.

    For each fuel with a measured resonance E_res, compute the
    declared landmark energies from the fixed kernel p_c's on the
    two dials (tau inversion and delta) plus the rank ladder and the
    Gamow-peak landmark. The resonance tick n(E_res) is compared to
    each landmark tick; the nearest landmark is the assigned
    percolation event.

    This is the falsifiable fusion claim: nuclear resonances are
    percolation events written on the Coulomb barrier, on the same
    Delta-ruler as electroweak and nuclear optical structure.

    Inputs: resonances = list of
      (label, Z1, Z2, A1, A2, E_res_keV, tol_keV_or_None).
      tol_keV None falls back to the default tick tolerance.
    Outputs: list of FusionResonanceRow (one per resonance).
    Companion: run.py 6h) prints the gate.
    """
    ladder = tb.rank_ladder_p_c()
    # sub-threshold boundary is the smallest strictly-positive rank
    # p_c (r=0 degenerates to p_c=0, the always-reachable gauge
    # doublet, so it is not a channel-closure threshold).
    pos = [p for p in ladder if p > 1e-12]
    weakest = min(pos) if pos else 0.0
    rows: List[FusionResonanceRow] = []
    for label, Z1, Z2, A1, A2, E_res, res_tol in resonances:
        Vb = tb.coulomb_barrier_MeV(Z1, Z2, A1, A2)
        n_res = tb.n_of_E(E_res * 1e3)
        # declared landmark energies (keV) for this fuel
        ev = fusion_resonance_events(tb, Z1=Z1, Z2=Z2, A1=A1, A2=A2)
        best_lab, best_E, best_n, best_off = "", 0.0, 0.0, 1e9
        for name, E_e in ev:
            n_e = tb.n_of_E(E_e * 1e3)
            off = abs(n_res - n_e)
            if off < best_off:
                best_lab, best_E, best_n, best_off = name, E_e, n_e, off
        # tolerance: from keV uncertainty if given, else default ticks
        if res_tol is not None:
            tol = abs(math.log2((E_res + res_tol) / E_res) / DELTA)
        else:
            tol = tol_ticks
        # sub-threshold: E_res sits below the weakest rank-ladder p_c
        p_delta = E_res / (Vb * 1e3)
        sub = p_delta < weakest - 1e-9
        Z1Z2 = Z1 * Z2
        role = FUSION_FUEL_ROLE.get(label, "other")
        rows.append(
            FusionResonanceRow(
                fuel=f"{label} {Z1}+{Z2} A{A1:.0f}/{A2:.0f}",
                Z1Z2=Z1Z2,
                role=role,
                z_cutoff=Z1Z2 >= FUSION_Z1Z2_CUTOFF,
                E_res_keV=E_res,
                n_res=n_res,
                E_event_keV=best_E,
                n_event=best_n,
                nearest_event=best_lab,
                offset_ticks=best_off,
                tol_ticks=tol,
                passed=(best_off <= tol) and (not sub),
                sub_threshold=sub,
            )
        )
    return rows


def fusion_resonance_null(
    tb: Trestleboard,
    *,
    resonances: List[Tuple[str, int, int, float, float, float, float]],
    rows: List[FusionResonanceRow],
    tol_ticks: float = 7.0,
) -> FusionResonanceSummary:
    """Null-model significance for the resonance-map gate.

    The declared landmark set has size n_events (six percolation
    events + Gamow peak + 2·rank_ladder). The ruler tested spans the
    tick range covered by those events (sub-threshold to barrier).
    A resonance placed uniformly on the tick ruler lands within ±tol
    of an event with window fraction p_single = 2·tol·n_events/
    tick_range (disjoint-window upper bound). p_single is averaged
    over tested fuels. expected_hits = n_tested·p_single. The
    family-wise significance of the observed n_passed is P(K>=n_passed)
    under Binomial(n_tested, p_single), Bonferroni-corrected by
    n_events.
    """
    ladder = tb.rank_ladder_p_c()
    n_events = 6 + 1 + 2 * len(ladder)
    n_tested = len(resonances)
    n_passed = sum(1 for r in rows if r.passed)
    p_vals: List[float] = []
    hi_eV = 0.0
    for (_, Z1, Z2, A1, A2, E_res, _), r in zip(resonances, rows):
        Vb = tb.coulomb_barrier_MeV(Z1, Z2, A1, A2)
        hi_eV = max(hi_eV, Vb * 1e9)
        ev = fusion_resonance_events(tb, Z1=Z1, Z2=Z2, A1=A1, A2=A2)
        ticks = [tb.n_of_E(E * 1e3) for _, E in ev]
        tr = max(ticks) - min(ticks)
        if tr > 0.0:
            p_vals.append(min(1.0, 2.0 * tol_ticks * n_events / tr))
    p_single = float(sum(p_vals) / n_tested) if n_tested else 0.0
    expected = n_tested * p_single
    p_ge = binom_p_ge(n_passed, p_single, n_tested)
    p_bonf = min(1.0, n_events * p_ge)
    return FusionResonanceSummary(
        n_events=n_events,
        tol_ticks=tol_ticks,
        band_lo_eV=hi_eV / 1e3 if hi_eV else 0.0,
        band_hi_eV=hi_eV,
        p_single=p_single,
        expected_hits=expected,
        n_passed=n_passed,
        n_tested=n_tested,
        p_atleast_passed=p_ge,
        p_bonferroni=p_bonf,
    )


# -----------------------------------------------------------------
# Driver
# -----------------------------------------------------------------
def frontier_main() -> None:
    """Structural solutions A–D and engineering deliverables E–H."""
    tb = Trestleboard(protocol="micro_ref", chirality_d=CHIRALITY_D)

    print("\nA. RESONANCE WIDTHS FROM PERCOLATION SUSCEPTIBILITY")
    print("=" * 5)
    print("  chi(p) = d theta_micro_ref_exact / dp (exact, central diff)")
    print(
        f"  {'landmark':<8}{'p_c':>12}{'chi(p_c)':>14}{'rank':>6}{'Gamma_struct':>16}"
    )
    for lab, p, chi, r, g in resonance_width_table():
        gs = "  -  " if math.isnan(g) else f"{g:16.6f}"
        print(f"  {lab:<8}{p:12.6f}{chi:14.6f}{r:6.1f}{gs}")
    print("  Gamma_struct = chi_ref / chi(p_c); sharp transition = narrow width.")

    fuels = [
        ("D-T", 1, 1, 2.0, 3.0),
        ("p-B11", 1, 5, 1.0, 11.0),
        ("10B-p", 1, 5, 1.0, 10.0),
        ("12C-p", 1, 6, 1.0, 12.0),
        ("15N-p", 1, 7, 1.0, 15.0),
    ]
    print("\n  Per-fuel landmark widths (E = p_c * V_b, keV):")
    print(
        f"  {'fuel':<8}{'E_landmark_keV':>16}{'chi':>12}{'rank':>8}{'Gamma_struct':>16}"
    )
    for label, E, chi, rk, g in fusion_resonance_widths(tb, fuels=fuels):
        print(f"  {label:<8}{E:16.4f}{chi:12.6f}{rk:>8}{g:16.6f}")

    print("\nB. BETA-DECAY MATRIX ELEMENTS FROM C(q)")
    print("=" * 5)
    b = beta_matrix_elements()
    print("  Carrier traces C(q) = Tr(M_q^2) [odd q] / Tr(M_q) [even q]:")
    for q in range(7):
        print(f"    C({q}) = {b.C[q]}")
    print(f"  Fermi proxy   |M_F|^2  = C(0) = {b.M_F2}")
    print(f"  GT proxy      |M_GT|^2 = C(1) = {b.M_GT2}")
    print(f"  F/GT carrier ratio = M_F^2 / M_GT^2 = {b.F_over_GT}")
    print(f"  GT/F carrier ratio = {b.GT_over_F}")
    print(
        f"  Forbidden ladder C(3)/C(1) = {b.forbidden_ratio[3]}  "
        f"C(5)/C(1) = {b.forbidden_ratio[5]}"
    )
    print(f"  Alpha H_L = C(2)/C(0) = {b.H_L_alpha}  (known discrete result)")

    print("\nC. MAGIC NUMBERS FROM ENUMERATOR {C1,C2,C3}")
    print("=" * 5)
    m = magic_number_audit()
    print(f"  shell capacities C(6,k)*64 = {m.shell_capacities}")
    print(f"  cumulative capacities     = {m.cumulative_capacities}")
    print(f"  code-gap arithmetic       = {m.code_gap_arithmetic}")
    print(f"  nuclear magic: {m.magic}")
    print(f"  {'magic':>6} {'structural limit':>22}")
    for mag, how in m.hits:
        print(f"    {mag:>6} {how:>22}")
    print(f"  coincidences: {m.n_coincide}/{len(m.magic)}")

    print("\nD. TENSOR FORCE AND THE PION OPERATOR FROM DELTA_4")
    print("=" * 5)
    d = deuteron_tensor_split()
    print(f"  E_bare (v*Delta^3)      = {d['E_bare_MeV']:.6f} MeV")
    print(f"  E_tensor (v*Delta^4*2/√5) = {d['E_tensor_MeV']:.6f} MeV")
    print(f"  E_total                 = {d['E_total_MeV']:.6f} MeV")
    print(f"  tensor fraction          = {d['tensor_fraction']:.6f}")
    p = pion_operator_relation()
    print(f"  Delta_4 / Delta_3 ratio = {p['Delta_4_over_Delta_3']}")
    print(f"  pion proxy (tensor energy) = {p['pion_proxy_MeV']:.6f} MeV")
    print(f"  discrete pion operator  = {p['isospin_flip_operator']}")
    print(f"  pion carrier move       = {p['pion_carrier_move']}")
    print(f"  note: {p['note']}")

    print("\nE. WIDTH SCALING LAW  Gamma_struct(r) ~ chi(p_c(r))^-1")
    print("=" * 5)
    w = width_scaling_law(tb)
    print(f"  chi_ref = {w.chi_ref:.6f}")
    print(
        f"  {'fuel':<8}{'r':>3}{'p_c':>12}{'E_keV':>12}{'chi':>12}{'Gamma_struct':>16}"
    )
    for label, rows in w.per_fuel.items():
        for r, p_c, E, chi, gamma in rows:
            print(f"  {label:<8}{r:>3}{p_c:12.6f}{E:12.4f}{chi:12.6f}{gamma:16.6f}")
    print(
        f"  monotone inverse (Gamma shrinks as r rises) ... "
        f"PASS={w.monotone_inverse_ok}"
    )

    print("\nF. CGM SURROGATE ON PREDICTIVE HOLDOUT")
    print("=" * 5)
    print("  calibration=even idx, holdout=odd idx; 1 DOF scale C")
    print(
        f"  {'fuel':<8}{'n_cal':>6}{'n_hold':>7}{'C':>12}"
        f"{'RMSE(log10)':>14}{'Pearson r':>12}"
    )
    pool_log: List[float] = []
    pool_r: List[float] = []
    for label, Z1, Z2, A1, A2 in HOLDOUT_FUELS:
        s = surrogate_holdout(tb, label=label, Z1=Z1, Z2=Z2, A1=A1, A2=A2)
        if s.status != "ok":
            print(f"  {label:<8} {s.status}")
            continue
        print(
            f"  {label:<8}{s.n_cal:>6}{s.n_hold:>7}{s.C:12.6e}"
            f"{s.rmse_log:14.4f}{s.pearson_r:12.4f}"
        )
        pool_log.extend([s.rmse_log])
        pool_r.append(s.pearson_r)
    pool_rmse = math.sqrt(sum(x * x for x in pool_log) / len(pool_log))
    pool_r_mean = sum(pool_r) / len(pool_r)
    print(
        f"  pooled RMSE(log10) = {pool_rmse:.4f}  "
        f"pooled mean Pearson r = {pool_r_mean:.4f}"
    )

    print("\nG. RIDER CUTOFF AS INTERNAL DISCRIMINANT R(Z1Z2)")
    print("=" * 5)
    print(f"  cutoff Z1Z2 = {FUSION_Z1Z2_CUTOFF}")
    print(f"  {'fuel':<8}{'Z1Z2':>5}{'n(Vb)':>10}{'gap(r5)':>10}{'R':>12}{'below':>8}")
    rid = rider_discriminant(tb)
    print(
        f"  reference tick at cutoff (Z1Z2={rid.cutoff}) = "
        f"{rid.n_cut_ticks:.2f} ticks"
    )
    for label, z, nVb, gap, R, below, ok in rid.rows:
        flag = "" if ok else "  <- CGM-ANOMALY"
        print(
            f"  {label:<8}{z:>5}{nVb:10.2f}{gap:10.2f}{R:12.2f}"
            f"{str(below):>8}{flag}"
        )
    print(f"  sign separation (strict) ... PASS={rid.sign_separated}")
    if rid.anomalies:
        print(
            f"  CGM-identified anomaly(s): {rid.anomalies} "
            f"(barrier tick below cutoff ref -> marginal, cf. Rider p-B11)"
        )

    print("\nH. SPARSE-DATA PREDICTION TARGET (CGM window vs CSV silence)")
    print("=" * 5)
    print(
        f"  {'fuel':<8}{'first_CSV_keV':>14}{'CGM_landmark_keV':>17}"
        f"{'untested_keV':>14}{'note'}"
    )
    for label, first_csv, E_landmark, window, note in sparse_target(tb):
        fc = "  -  " if math.isnan(first_csv) else f"{first_csv:14.1f}"
        wd = "  -  " if math.isnan(window) else f"{window:14.1f}"
        print(f"  {label:<8}{fc}{E_landmark:17.4f}{wd}{note}")

    print("\nI. BARRIER RADIUS SENSITIVITY (r0 falsification)")
    print("=" * 5)
    sens = barrier_radius_sensitivity_summary(tb)
    print(f"  r0 grid (fm) = {list(sens['r0_values'])}  ref = {sens['r0_ref']}")
    print(
        f"  {'fuel':<8}{'r0':>6}{'Vb_MeV':>10}{'n(Vb)':>10}"
        f"{'dn_vs_ref':>12}{'k':>4}{'k=3':>6}"
    )
    for label, r0, Vb, n_Vb, dn, k, ok in sens["rows"]:
        print(
            f"  {label:<8}{r0:6.2f}{Vb:10.4f}{n_Vb:10.2f}"
            f"{dn:12.2f}{k:4d}{str(ok):>6}"
        )
    print(f"  all barriers on k=3 .............. PASS={sens['all_k3']}")
    print(
        f"  max |dn| vs r0={sens['r0_ref']} (off-ref) "
        f"= {sens['max_abs_dn_off_ref']:.2f} ticks"
    )
    for lab, m in sorted(sens["per_fuel_max_abs_dn"].items()):
        print(f"    {lab:<8} max |dn| = {m:.2f}")


def main() -> None:
    frontier_main()


if __name__ == "__main__":
    main()
