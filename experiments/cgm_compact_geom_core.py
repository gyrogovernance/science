#!/usr/bin/env python3
"""
cgm_compact_geom_core.py

Pure algebra layer for the compact geometry electroweak analysis.
No printing, no I/O, no side effects.

Layer 0: Named constants
Layer 1: Compact algebra (epsilon, eta, d_H, shell moments)
Layer 2: Electroweak laws (D2 through D5, all channels)
Layer 3: Backsolves (top, Higgs, Z, W, W/Z)
Layer 4: Shell transition algebra (exact fractions, carrier traces)
Layer 5: Lepton ladder coordinates
Layer 6: Observable definitions
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import Sequence, TypedDict

try:
    import scipy.constants as sc
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Layer 0: Named constants
# ---------------------------------------------------------------------------

# Observational aperture: 1 / (2 sqrt(2 pi))
M_A: float = 1.0 / (2.0 * math.sqrt(2.0 * math.pi))

# BU monodromy defect: verified non-Clifford rotation angle.
# Source: aQPU Tests Report 1 Part 10; Physics Tests Report Part 9.
DELTA_BU: float = 0.195342176580

# Closure ratio and aperture gap
RHO: float = DELTA_BU / M_A
DELTA: float = 1.0 - RHO          # primary aperture constant

# Electroweak VEV anchor
E_EW_GEV: float = 246.22

ELECTRON_MASS_GEV: float = 0.000_510_998_950_714

# Cs-133 hyperfine frequency (exact SI definition)
F_CS_133_HZ: float = 9_192_631_770.0

# Aperture frame: 3 * |K4|^2 = 3 * 4 * 4 = 48
# Depth-4 boundary projector: P = 1 - 1/48 = 47/48
APERTURE_FRAME: float = 48.0
P_BOUNDARY: float = 47.0 / 48.0

# Bare byte aperture (8-bit dyadic grid)
KERNEL_BYTE_APERTURE: float = 5.0 / 256.0

# SU(2) residual: Berry-type phase.
# (phi_SU2 - 3 * delta_BU) / m_a
_PHI_SU2: float = 2.0 * math.acos((1.0 + 2.0 * math.sqrt(2.0)) / 4.0)
SU2_RESIDUAL: float = (_PHI_SU2 - 3.0 * DELTA_BU) / M_A

# Gyroscopic coupling: Delta / sqrt(5)  (pentagonal normalisation)
LAMBDA_0: float = DELTA / math.sqrt(5.0)

# Geometric alpha proxy: delta_BU^4 / m_a
ALPHA_GEOMETRIC: float = (DELTA_BU ** 4) / M_A

# Matter-sector density projector Q = d_A * d_B = (6/12)^2 = 1/4
Q_DENSITY: float = 0.25

# UV-IR conjugacy scale: (2 pi)^2
UV_IR_GYRATION_SQ: float = (2.0 * math.pi) ** 2

# [12,6,2] self-dual code weight enumerator binomial coefficients
CODE_C1: int = comb(6, 1)   # 6
CODE_C2: int = comb(6, 2)   # 15
CODE_C3: int = comb(6, 3)   # 20
CODE_C4: int = comb(6, 4)   # 15
CODE_C5: int = comb(6, 5)   # 6

# Shell moment: sum_{k=0}^{6} k * C(6,k) = 192
M_SHELL: int = sum(k * comb(6, k) for k in range(7))

# Horizon cardinality |H| = 64
HORIZON_CARDINALITY: int = 64

# Reachable manifold size |Omega| = 4096
OMEGA_SIZE: int = 4096

# Monodromy per stage (half round-trip)
MONODROMY_PER_STAGE: float = DELTA_BU / 2.0

# W/Z split derived constants
WZ_OFFSET: int = CODE_C2 - CODE_C1          # 9
WZ_APERTURE_COEFF: float = CODE_C3 / 2.0    # 10.0
Z_H_OFFSET: int = 1 + CODE_C1 + CODE_C2     # 22

# Muon equatorial coefficient (equatorial shell multiplicity)
MUON_EQUATOR_COEFF: int = CODE_C3           # 20


# ---------------------------------------------------------------------------
# Layer 1: Compact algebra dataclass and constructor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompactAlgebra:
    """All derived scalar quantities from DELTA."""
    delta: float
    epsilon: float    # 1/delta - 48
    eta: float        # m_a - delta_BU
    d_h: float        # epsilon + 24*delta  (Higgs defect)
    omega: float      # delta_BU / 2
    kappa: float      # pi/4 - 1/sqrt(2)
    sigma: float      # SU2_RESIDUAL


def compact_algebra(delta: float = DELTA) -> CompactAlgebra:
    """Construct the full compact algebra from a given delta."""
    epsilon = (1.0 / delta) - 48.0
    eta = M_A - DELTA_BU
    d_h = epsilon + 24.0 * delta
    omega = DELTA_BU / 2.0
    kappa = (math.pi / 4.0) - (1.0 / math.sqrt(2.0))
    return CompactAlgebra(
        delta=delta,
        epsilon=epsilon,
        eta=eta,
        d_h=d_h,
        omega=omega,
        kappa=kappa,
        sigma=SU2_RESIDUAL,
    )


def delta_self_consistency_rhs(
    delta: float,
    *,
    include_third_order: bool = True,
) -> float:
    """
    RHS of the aperture self-consistency equation:
        Delta = (5/256) * 2^(1/12) * (1 + (sqrt6/pi)*Delta^2 [+ (eta/eps)*Delta^3])
    """
    eps = (1.0 / delta) - 48.0
    eta = M_A - DELTA_BU
    correction = 1.0 + (math.sqrt(6.0) / math.pi) * delta * delta
    if include_third_order and eps != 0.0:
        correction += (eta / eps) * delta ** 3
    return KERNEL_BYTE_APERTURE * (2.0 ** (1.0 / 12.0)) * correction


def solve_delta_self_consistency(
    delta_guess: float = DELTA,
    *,
    include_third_order: bool = True,
    max_iter: int = 30,
) -> tuple[float, float]:
    """
    Fixed-point iteration for the aperture self-consistency equation.
    Returns (delta_solution, residual).
    """
    delta = delta_guess
    for _ in range(max_iter):
        rhs = delta_self_consistency_rhs(delta, include_third_order=include_third_order)
        delta_next = 0.5 * delta + 0.5 * rhs
        if abs(delta_next - delta) <= 1e-15:
            delta = delta_next
            break
        delta = delta_next
    residual = delta - delta_self_consistency_rhs(delta, include_third_order=include_third_order)
    return delta, residual


# ---------------------------------------------------------------------------
# Layer 2: Electroweak laws (D2 through D5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelCoeffs:
    """
    All coefficients for one electroweak channel in the five-order law:
        L_i = a*D + b + c*D^2 + p*D^3/sqrt(5) + q*D^4 + r5*D^5
    """
    label: str          # "Top", "Higgs", "Z", "W"
    observable: str     # PDG observable name key
    a: float            # linear Delta coefficient
    b: float            # constant term
    c: float            # D^2 coefficient
    p: float            # D^3 gyroscopic phase charge (trace-free)
    q: float            # D^4 gyroscopic closure charge (trace-free)
    r5: float           # D^5 code curvature coefficient


# Channel definitions: all coefficients derived from kernel algebra.
#
# a, b: from [12,6,2] code enumerator and horizon cardinality |H|=64
#   Top:   a = |H| + C2 - C1 = 73,  b = -1
#   Higgs: a = M_shell/2 = 96,       b = -1
#   Z:     a = M_shell/2 + C1+C2 = 117, b = -P = -47/48
#   W:     a = M_shell/2 + 2*C2 = 126,  b = -P = -47/48
#
# c: stage projection of code-weight moments
#   Top:   +Q = +1/4  (matter density projector)
#   Higgs: -M_shell/8 = -24
#   Z:     -M_shell/8 + C1/4 = -22.5
#   W:     -M_shell/8 + C1/4 - C3/2 = -32.5
#
# p, q: K4 edge increments from gyrotriangle closure delta=0
#   flags (base, rot, bal): Top=(0,0,0), Higgs=(1,0,0), Z=(1,1,0), W=(1,1,1)
#   p = 1 + (-C1/2)*base + (C1/4)*rot + (4g)*bal,  g=1/2
#   q = 5/4 - (4g)*rot - (2g)*bal
#
# r5: -(C2-C1)/2 + (|H|-(C2-C1))/8*(base-rot) + C2/8*bal
#   with |H|=64, C1=6, C2=15

def _r5(base: bool, rot: bool, bal: bool) -> float:
    wz = CODE_C2 - CODE_C1  # 9
    return (
        -wz / 2.0
        + (HORIZON_CARDINALITY - wz) / 8.0 * (float(base) - float(rot))
        + CODE_C2 / 8.0 * float(bal)
    )


def _pq(base: bool, rot: bool, bal: bool) -> tuple[float, float]:
    g = 0.5
    b, r, l = float(base), float(rot), float(bal)
    p = 1.0 + (-CODE_C1 / 2.0) * b + (CODE_C1 / 4.0) * r + (4.0 * g) * l
    q = 5.0 / 4.0 - (4.0 * g) * r - (2.0 * g) * l
    return p, q


def _channel_coeffs() -> tuple[ChannelCoeffs, ...]:
    p_t, q_t = _pq(False, False, False)
    p_h, q_h = _pq(True,  False, False)
    p_z, q_z = _pq(True,  True,  False)
    p_w, q_w = _pq(True,  True,  True)
    return (
        ChannelCoeffs(
            "Top", "Top quark mass energy",
            a=float(HORIZON_CARDINALITY + CODE_C2 - CODE_C1),
            b=-1.0,
            c=Q_DENSITY,
            p=p_t, q=q_t,
            r5=_r5(False, False, False),
        ),
        ChannelCoeffs(
            "Higgs", "Higgs mass energy",
            a=float(M_SHELL) / 2.0,
            b=-1.0,
            c=-float(M_SHELL) / 8.0,
            p=p_h, q=q_h,
            r5=_r5(True, False, False),
        ),
        ChannelCoeffs(
            "Z", "Z boson mass energy",
            a=float(M_SHELL) / 2.0 + CODE_C1 + CODE_C2,
            b=-P_BOUNDARY,
            c=-float(M_SHELL) / 8.0 + CODE_C1 / 4.0,
            p=p_z, q=q_z,
            r5=_r5(True, True, False),
        ),
        ChannelCoeffs(
            "W", "W boson mass energy",
            a=float(M_SHELL) / 2.0 + 2.0 * CODE_C2,
            b=-P_BOUNDARY,
            c=-float(M_SHELL) / 8.0 + CODE_C1 / 4.0 - CODE_C3 / 2.0,
            p=p_w, q=q_w,
            r5=_r5(True, True, True),
        ),
    )


# Module-level tuple of channel coefficient objects (computed once)
CHANNELS: tuple[ChannelCoeffs, ...] = _channel_coeffs()


def eval_law(ch: ChannelCoeffs, delta: float, *, order: int = 5) -> float:
    """
    Evaluate L_i(delta) through the requested order.
        order=1: a*D + b
        order=2: + c*D^2
        order=3: + p*D^3/sqrt(5)   [D^3 gyroscopic phase]
        order=4: + q*D^4            [D^4 gyroscopic closure]
        order=5: + r5*D^5           [D^5 code curvature]

    Note: D^3 term is p * (D/sqrt5) * D^2 = p * LAMBDA_0 * D^2.
    It is numerically equivalent to p * GYROSCOPIC_COUPLING * D^2
    and differs only by floating-point ordering effects at the last digits.
    """
    L = ch.a * delta + ch.b
    if order >= 2:
        L += ch.c * delta ** 2
    if order >= 3:
        L += ch.p * LAMBDA_0 * delta ** 2   # p * (D/sqrt5) * D^2
    if order >= 4:
        L += ch.q * delta ** 4
    if order >= 5:
        L += ch.r5 * delta ** 5
    return L


def all_laws(delta: float = DELTA, *, order: int = 5) -> dict[str, float]:
    """Return {label: L_i} for all four channels at given order."""
    return {ch.label: eval_law(ch, delta, order=order) for ch in CHANNELS}


def channel_by_label(label: str) -> ChannelCoeffs:
    for ch in CHANNELS:
        if ch.label == label:
            return ch
    raise KeyError(f"No channel with label {label!r}")


def channel_by_observable(name: str) -> ChannelCoeffs:
    for ch in CHANNELS:
        if ch.observable == name:
            return ch
    raise KeyError(f"No channel for observable {name!r}")


# ---------------------------------------------------------------------------
# Layer 3: Backsolves
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeltaBacksolve:
    source: str
    equation: str
    delta_back: float

    @property
    def delta_err(self) -> float:
        return self.delta_back - DELTA


def _solve_top(l_t: float) -> float:
    """L_t = 73D - 1 + (1/4)D^2  =>  quadratic in D."""
    a, b, c = Q_DENSITY, 73.0, -(1.0 + l_t)
    disc = b * b - 4.0 * a * c
    return (-b + math.sqrt(disc)) / (2.0 * a)


def _solve_higgs(l_h: float) -> float:
    """L_H = 96D - 1 - 24D^2  =>  quadratic in D."""
    # 24D^2 - 96D + (1 + l_h) = 0
    a, b, c = 24.0, -96.0, 1.0 + l_h
    disc = b * b - 4.0 * a * c
    return (-b - math.sqrt(disc)) / (2.0 * a)


def _solve_z(l_z: float) -> float:
    """L_Z = 117D - 47/48 - (45/2)D^2  =>  quadratic in D."""
    # (45/2)D^2 - 117D + (47/48 + l_z) = 0
    a, b, c = 45.0 / 2.0, -117.0, P_BOUNDARY + l_z
    disc = b * b - 4.0 * a * c
    return (-b - math.sqrt(disc)) / (2.0 * a)


def _solve_w(l_w: float) -> float:
    """L_W = 126D - 47/48 - (65/2)D^2  =>  quadratic in D."""
    # (65/2)D^2 - 126D + (47/48 + l_w) = 0
    a, b, c = 65.0 / 2.0, -126.0, P_BOUNDARY + l_w
    disc = b * b - 4.0 * a * c
    return (-b - math.sqrt(disc)) / (2.0 * a)


def _solve_wz_phase_corrected(l_wz: float, *, guess: float = 0.0207, iters: int = 20) -> float:
    """
    Promoted W/Z split law (Newton's method):
        log2(mZ/mW) = D * [9 - 10D + 2D^2/sqrt(5) - D^3]
    """
    D = guess
    s5 = math.sqrt(5.0)
    for _ in range(iters):
        f  = 9.0*D - 10.0*D**2 + 2.0*D**3/s5 - D**4 - l_wz
        fp = 9.0 - 20.0*D + 6.0*D**2/s5 - 4.0*D**3
        D -= f / fp
    return D


def electroweak_backsolves(
    observed: dict[str, float],
) -> tuple[DeltaBacksolve, ...]:
    """
    Compute Delta backsolves from a dict of {observable_name: mass_GeV}.
    Expected keys: 'Top quark mass energy', 'Higgs mass energy',
                   'Z boson mass energy', 'W boson mass energy',
                   'Electroweak scale'.
    """
    v = observed["Electroweak scale"]
    l_t  = math.log2(v / observed["Top quark mass energy"])
    l_h  = math.log2(v / observed["Higgs mass energy"])
    l_z  = math.log2(v / observed["Z boson mass energy"])
    l_w  = math.log2(v / observed["W boson mass energy"])
    l_wz = math.log2(observed["Z boson mass energy"] / observed["W boson mass energy"])

    return (
        DeltaBacksolve("Top",   "73D-1+(1/4)D^2",                     _solve_top(l_t)),
        DeltaBacksolve("Higgs", "96D-1-24D^2",                         _solve_higgs(l_h)),
        DeltaBacksolve("Z",     "117D-47/48-(45/2)D^2",               _solve_z(l_z)),
        DeltaBacksolve("W",     "126D-47/48-(65/2)D^2",               _solve_w(l_w)),
        DeltaBacksolve("W/Z",   "D[9-10D+2D^2/sqrt(5)-D^3]",         _solve_wz_phase_corrected(l_wz)),
    )


def four_point_consensus(backsolves: tuple[DeltaBacksolve, ...]) -> DeltaBacksolve:
    """Mean of Top + Higgs + Z + W backsolves."""
    sources = ("Top", "Higgs", "Z", "W")
    vals = [bs.delta_back for bs in backsolves if bs.source in sources]
    return DeltaBacksolve("4-pt mean", "mean(Top,Higgs,Z,W)", sum(vals) / len(vals))


# ---------------------------------------------------------------------------
# Layer 4: Shell transition algebra (exact Fractions)
# ---------------------------------------------------------------------------

def shell_transition_prob(w_src: int, q_weight: int, w_dst: int) -> Fraction:
    """
    Exact transition probability P(w_src --(q_weight)--> w_dst)
    in the Hamming-scheme on GF(2)^6.
    """
    w, j, wp = int(w_src), int(q_weight), int(w_dst)
    delta = w + j - wp
    if delta < 0 or delta % 2:
        return Fraction(0)
    t = delta // 2
    if t < 0 or t > min(w, j):
        return Fraction(0)
    if (j - t) < 0 or (j - t) > (6 - w):
        return Fraction(0)
    return Fraction(comb(w, t) * comb(6 - w, j - t), comb(6, j))


def shell_transition_matrix(q: int) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(shell_transition_prob(w, q, wp) for wp in range(7))
        for w in range(7)
    )


def shell_trace(q: int) -> Fraction:
    M = shell_transition_matrix(q)
    return sum((M[i][i] for i in range(7)), Fraction(0))


def shell_return_trace(q: int) -> Fraction:
    M = shell_transition_matrix(q)
    return sum(
        (M[i][k] * M[k][i] for i in range(7) for k in range(7)),
        Fraction(0),
    )


def carrier_trace(q: int) -> Fraction:
    """Tr(M) if non-zero, else Tr(M^2)."""
    t = shell_trace(q)
    return t if t != 0 else shell_return_trace(q)


# Pre-computed carrier traces for q = 0..6
CARRIER_TRACES: tuple[Fraction, ...] = tuple(carrier_trace(q) for q in range(7))


# ---------------------------------------------------------------------------
# Layer 5: Lepton ladder coordinates
# ---------------------------------------------------------------------------

def lepton_base_n(k: int, lepton: str) -> float:
    """
    Base integer coordinate n = k * |H| + r(lepton)
    from the horizon/shell decomposition.
        r(tau)     = M_shell / 8   = 24
        r(mu)      = M_shell / 8 + M_shell / 48  = 28
        r(electron)= M_shell / 8 - M_shell / 24  = 16
    """
    m = float(M_SHELL)
    if lepton == "tau":
        r = m / 8.0
    elif lepton in ("mu", "muon"):
        r = m / 8.0 + m / 48.0
    elif lepton in ("e", "electron"):
        r = m / 8.0 - m / 24.0
    else:
        raise ValueError(f"Unknown lepton: {lepton!r}")
    return k * HORIZON_CARDINALITY + r


def lepton_delta_coords(
    n_h_law: float,
    delta: float = DELTA,
) -> dict[str, float]:
    """
    Lepton EW-referred Delta-ruler coordinates.
    n_h_law: the Higgs n-coordinate (L_H/delta) from the five-order law.

    Returns dict with keys: n_tau, n_mu, n_electron.
    """
    # Higgs memory term: (5/256) / n_H  where n_H = L_H/delta
    higgs_memory = KERNEL_BYTE_APERTURE / n_h_law

    return {
        "n_tau":      lepton_base_n(5,  "tau")      - (DELTA_BU + 5.0 * delta),
        "n_mu":       lepton_base_n(8,  "mu")       + MUON_EQUATOR_COEFF * delta,
        "n_electron": lepton_base_n(14, "electron") + SU2_RESIDUAL + (delta / ((n_h_law * delta) / KERNEL_BYTE_APERTURE)),
    }


@dataclass(frozen=True)
class LeptonLadderRow:
    """One lepton in the temporal ladder with order-by-order residuals."""
    label: str
    k: int
    r: float
    base_n: float
    n_model_d1: float
    n_model_d2: float
    n_model_d3: float
    n_observed: float
    resid_d1: float
    resid_d2: float
    resid_d3: float
    law_d1: str
    law_d2: str
    law_d3: str


@dataclass(frozen=True)
class LeptonD3TransitionCost:
    label: str
    q_from: int
    q_to: int
    carrier_from: Fraction
    carrier_to: Fraction
    carrier_delta: Fraction
    dyadic: Fraction
    coeff_delta3: Fraction
    normalized_64_cost: Fraction
    numerator_rule: str
    numerator_rule_value: Fraction
    numerator_rule_matches: bool


@dataclass(frozen=True)
class LeptonD3SelectionRule:
    label: str
    rule: str
    value: Fraction
    target: Fraction

    @property
    def matches(self) -> bool:
        return self.value == self.target


@dataclass(frozen=True)
class ShellSpectralWeight:
    q: int
    byte_count: int
    trace: Fraction
    return_trace: Fraction
    carrier: Fraction
    shell_frobenius_sq: Fraction
    full_operator_hs_sq: Fraction


@dataclass(frozen=True)
class K4PhaseSelector:
    label: str
    base: bool
    rot: bool
    bal: bool
    p: Fraction
    q: Fraction
    r5: Fraction
    normalized_64_cost: Fraction
    rule: str


@dataclass(frozen=True)
class QSupportDuality:
    q_a: int
    q_b: int
    size_a: int
    size_b: int
    direct_intersection: int
    complement_pair_hits: int

    @property
    def complement_duality_complete(self) -> bool:
        return self.size_a == self.size_b == self.complement_pair_hits


@dataclass(frozen=True)
class ByteArchetypeProbe:
    gene_mic: int
    intron: int
    family: int
    micro_ref: int
    q_weight: int
    archetype_shadow: Fraction
    carrier_delta: Fraction
    shadow_carrier_coeff: Fraction
    residual_carrier_sum: Fraction
    matches_residual: bool


@dataclass(frozen=True)
class HorizonGateSelection:
    byte: int
    gate: str
    intron: int
    family: int
    micro_ref: int
    q_weight: int
    is_s_gate: bool
    zero_intron: bool
    zero_payload: bool
    zero_family: bool
    selected_by_electron_reset: bool


@dataclass(frozen=True)
class SourceTraceabilityProbe:
    """
    Finite audit for the terminal lepton source-reset rule.

    The source-traceability condition selects the unique q=0 S-gate with zero
    intron, zero payload mutation, and zero spinorial family displacement.
    """
    selected_byte: int
    selected_count: int
    q_kernel_size: int
    selected_is_gene_mic: bool
    dyadic_atom: Fraction
    carrier_delta: Fraction
    carrier_shadow: Fraction
    electron_neutral: Fraction
    electron_implemented: Fraction
    electron_derived: Fraction
    closes_electron_dyadic: bool


@dataclass(frozen=True)
class StrongScaleBulkProbe:
    """
    Delta-ruler placement of the conventional strong reference scale.
    Lambda_QCD remains an external input; this probe only audits where it lands
    in the compact kernel coordinate system.
    """
    lambda_qcd_gev: float
    n_qcd: float
    n_phase_64: float
    n_phase_48: float
    bulk_states: int
    boundary_states: int
    bulk_fraction: Fraction
    boundary_fraction: Fraction
    placed_in_relational_bulk: bool


@dataclass(frozen=True)
class SpectralTripleProbe:
    """
    Finite checks for the candidate triple (A, H, D_shell).
    """
    algebra_generators: int
    hilbert_dimension: int
    shell_count: int
    d_shell_trace: int
    d_code_trace: int
    zero_mode_dimension: int
    finite_commutators_bounded: bool
    grading_specified: bool
    real_structure_specified: bool
    spectral_action_specified: bool
    complete_physics_triple: bool


@dataclass(frozen=True)
class LeptonHorizonWrapExhaustionProbe:
    """
    Exhaustive audit of the proposed lepton horizon-wrap anchors.

    The broad constraint set deliberately contains only ordering and the
    tau/muon D3 cost budget. If this admits multiple candidates, the result
    marks the horizon-wrap rule as a selection rule still needing proof.
    """
    max_k: int
    q_path: tuple[int, int, int]
    expected_k_path: tuple[int, int, int]
    broad_valid_count: int
    broad_valid_examples: tuple[tuple[int, int, int], ...]
    horizon_rule_valid_count: int
    horizon_rule_valid_paths: tuple[tuple[int, int, int], ...]
    tau_muon_budget: Fraction
    tau_muon_budget_matches: bool
    delta_tau_muon: int
    delta_muon_electron: int
    expected_delta_tau_muon: Fraction
    expected_delta_muon_electron: Fraction
    unique_without_horizon_rule: bool
    unique_with_horizon_rule: bool


@dataclass(frozen=True)
class GrammarResidualCandidate:
    label: str
    value: float
    residual_minus_value: float
    abs_error: float
    matches: bool


@dataclass(frozen=True)
class QCDApertureCycleResidualProbe:
    """
    Test whether the conventional Lambda_QCD coordinate matches a closed
    aperture-cycle expression in the declared compact grammar.
    """
    lambda_qcd_gev: float
    n_qcd: float
    aperture_cycles: int
    electron_offset: int
    candidate_base: float
    residual: float
    candidates: tuple[GrammarResidualCandidate, ...]
    best_label: str
    best_abs_error: float
    closes_to_grammar: bool
    remains_external_input: bool


class LeptonD3CarrierPath(TypedDict):
    edges: tuple[tuple[int, int], ...]
    edge_labels: dict[tuple[int, int], list[str]]
    starts: tuple[int, ...]
    sinks: tuple[int, ...]
    path: tuple[int, ...]
    complete_directed_chain: bool


class LeptonCarrierPathProbe(TypedDict):
    q: int
    Tr_M: Fraction
    Tr_M2: Fraction
    carrier: Fraction
    byte_count: int


class LeptonD3SpectralObstruction(TypedDict):
    weights: tuple[ShellSpectralWeight, ...]
    q2_q4_full_hs_equal: bool
    q2_q4_return_trace_equal: bool
    q2_q4_shell_frobenius_equal: bool
    q5_over_q4_full_hs_ratio: Fraction
    q5_over_q4_byte_support_ratio: Fraction
    muon_electron_dyadic_ratio: Fraction
    hs_ratio_splits_muon_electron: bool


class LeptonD3PathBreakingProbe(TypedDict):
    selectors: tuple[K4PhaseSelector, ...]
    implemented_muon_electron_split: Fraction
    implemented_muon_electron_split_dyadic: Fraction
    simplified_electron_64_cost: Fraction
    simplified_muon_electron_split: Fraction
    simplified_muon_electron_split_dyadic: Fraction
    claim_25_over_64_matches_implemented: bool
    claim_25_over_64_matches_simplified: bool
    electron_reset_offset_from_simplified: Fraction
    carrier_path: LeptonD3CarrierPath
    history_path: tuple[int, ...]
    history_xor_moment: Fraction
    history_split_64: Fraction
    history_split_dyadic: Fraction
    history_split_matches_simplified: bool
    byte_interior_roles: Fraction
    byte_horizon: Fraction
    byte_phase_reset: Fraction
    byte_phase_reset_rule: str
    byte_phase_reset_64: Fraction
    byte_phase_reset_matches_rule: bool
    electron_from_history_64: Fraction
    electron_from_history_dyadic: Fraction
    electron_from_history_matches: bool
    carrier_conservation_coeff_sum: Fraction
    carrier_conservation_electron_dyadic: Fraction
    carrier_conservation_electron_64: Fraction
    carrier_conservation_current_offset: Fraction
    carrier_conservation_current_matches: bool


class LeptonD3ShellLemmas(TypedDict):
    shell_rows: tuple[LeptonCarrierPathProbe, ...]
    carrier_delta_muon_electron: Fraction
    carrier_delta_tau: Fraction
    Tr_M2_equal_q2_q4: bool
    squared_carrier_ratio_C2_over_C4: Fraction
    implemented_dyadic_ratio_muon_over_electron: Fraction
    squared_carrier_ratio_equals_dyadic_ratio: bool
    dyadic_tau: Fraction
    dyadic_muon: Fraction
    dyadic_electron: Fraction
    transition_costs: tuple[LeptonD3TransitionCost, ...]
    tau_muon_horizon_budget: Fraction
    tau_muon_horizon_budget_matches: bool
    muon_minus_tau_cost: Fraction
    muon_minus_tau_matches_C3_over_2: bool
    electron_64_cost: Fraction
    electron_64_cost_rule: Fraction
    electron_64_cost_rule_matches: bool
    selection_rules: tuple[LeptonD3SelectionRule, ...]
    selection_rules_all_match: bool
    spectral_obstruction: LeptonD3SpectralObstruction
    path_breaking: LeptonD3PathBreakingProbe
    archetype_shadow: ByteArchetypeProbe
    horizon_gate_selection: tuple[HorizonGateSelection, ...]
    support_duality: tuple[QSupportDuality, ...]


def lepton_ladder_residuals(
    observed: dict[str, float],
    delta: float = DELTA,
    v: float = E_EW_GEV,
) -> tuple[LeptonLadderRow, ...]:
    """
    Full order-by-order residual breakdown for the lepton temporal ladder.
    Returns one row per lepton (tau, muon, electron).
    """
    n_h_law = all_laws(delta, order=5)["Higgs"] / delta
    higgs_memory = KERNEL_BYTE_APERTURE / n_h_law

    specs = [
        ("tau", 5, "Tau mass energy",
         -(DELTA_BU + 5.0 * delta),
         (CODE_C1 / 4.0) * delta ** 2,
         Fraction(-27, 64) * float(CARRIER_TRACES[4] - CARRIER_TRACES[5]) * delta ** 3,
         "-(dBU+5D)",
         "+(C1/4)D^2",
         "+(-27/64)*(C4-C5)*D^3"),
        ("muon", 8, "Muon mass energy",
         MUON_EQUATOR_COEFF * delta,
         -(CODE_C1 / 16.0) * delta ** 2,
         Fraction(-37, 64) * float(CARRIER_TRACES[2] - CARRIER_TRACES[4]) * delta ** 3,
         "+C3*D",
         "-(C1/16)D^2",
         "+(-37/64)*(C2-C4)*D^3"),
        ("electron", 14, "Electron mass energy",
         SU2_RESIDUAL + higgs_memory,
         0.0,
         Fraction(-51, 256) * float(CARRIER_TRACES[2] - CARRIER_TRACES[4]) * delta ** 3,
         "+SU2+(5/256)/n_H",
         "",
         "+(-51/256)*(C2-C4)*D^3"),
    ]

    rows: list[LeptonLadderRow] = []
    for label, k, obs_name, d1, d2, d3, law1, law2, law3 in specs:
        obs_mass = observed.get(obs_name)
        if obs_mass is None:
            continue
        n_obs = ew_delta_n(obs_mass, ew_scale_gev=v, delta=delta)
        base = lepton_base_n(k, label)
        r = base - k * HORIZON_CARDINALITY

        n_d1 = base + d1
        n_d2 = n_d1 + d2
        n_d3 = n_d2 + d3

        rows.append(LeptonLadderRow(
            label=label,
            k=k,
            r=r,
            base_n=base,
            n_model_d1=n_d1,
            n_model_d2=n_d2,
            n_model_d3=n_d3,
            n_observed=n_obs,
            resid_d1=n_obs - n_d1,
            resid_d2=n_obs - n_d2,
            resid_d3=n_obs - n_d3,
            law_d1=law1,
            law_d2=law2,
            law_d3=law3,
        ))
    return tuple(rows)


def lepton_d3_transition_costs() -> tuple[LeptonD3TransitionCost, ...]:
    wz = CODE_C2 - CODE_C1
    specs = (
        (
            "tau", 5, 4, Fraction(-27, 64),
            "-3*(C2-C1)", Fraction(-3 * wz, 1),
        ),
        (
            "muon", 4, 2, Fraction(-37, 64),
            "-(|H|-3*(C2-C1))", Fraction(-(HORIZON_CARDINALITY - 3 * wz), 1),
        ),
        (
            "electron", 4, 2, Fraction(-51, 256),
            "-(3*|K4|+C1/8)", -(Fraction(12, 1) + Fraction(CODE_C1, 8)),
        ),
    )

    rows: list[LeptonD3TransitionCost] = []
    for label, q_from, q_to, dyadic, rule, rule_value in specs:
        carrier_from = CARRIER_TRACES[q_from]
        carrier_to = CARRIER_TRACES[q_to]
        carrier_delta = carrier_to - carrier_from
        normalized_64 = dyadic * 64
        rows.append(LeptonD3TransitionCost(
            label=label,
            q_from=q_from,
            q_to=q_to,
            carrier_from=carrier_from,
            carrier_to=carrier_to,
            carrier_delta=carrier_delta,
            dyadic=dyadic,
            coeff_delta3=dyadic * carrier_delta,
            normalized_64_cost=normalized_64,
            numerator_rule=rule,
            numerator_rule_value=rule_value,
            numerator_rule_matches=normalized_64 == rule_value,
        ))
    return tuple(rows)


def lepton_d3_selection_rule_probe() -> tuple[LeptonD3SelectionRule, ...]:
    """
    Mass-blind candidate rules for the three lepton D3 dyadics.

    These rules use only kernel constants and the temporal-ladder topology:
    |H|, C1, C2, C3, and WZ_OFFSET=C2-C1. They are not yet a proof that the
    lepton paths must select these rules; they make that selection problem
    precise and directly auditable.
    """
    wz = Fraction(WZ_OFFSET, 1)
    horizon = Fraction(HORIZON_CARDINALITY, 1)
    k4_cardinality = Fraction(4, 1)
    tau_cost = -Fraction(CODE_C1, 2) * wz
    muon_cost = -(horizon - abs(tau_cost))
    electron_cost = -(3 * k4_cardinality + Fraction(CODE_C1, 8))
    return (
        LeptonD3SelectionRule(
            "tau",
            "-(C1/2)*(C2-C1)",
            tau_cost,
            Fraction(-27, 1),
        ),
        LeptonD3SelectionRule(
            "muon",
            "-(|H|-|(tau 64-cost)|)",
            muon_cost,
            Fraction(-37, 1),
        ),
        LeptonD3SelectionRule(
            "electron",
            "-(3*|K4|+C1/8)",
            electron_cost,
            Fraction(-51, 4),
        ),
    )


def _k4_phase_selector(
    label: str,
    base: bool,
    rot: bool,
    bal: bool,
    normalized_64_cost: Fraction,
    rule: str,
) -> K4PhaseSelector:
    p_float, q_float = _pq(base, rot, bal)
    return K4PhaseSelector(
        label=label,
        base=base,
        rot=rot,
        bal=bal,
        p=Fraction(p_float).limit_denominator(),
        q=Fraction(q_float).limit_denominator(),
        r5=Fraction(_r5(base, rot, bal)).limit_denominator(),
        normalized_64_cost=normalized_64_cost,
        rule=rule,
    )


def lepton_d3_carrier_path() -> LeptonD3CarrierPath:
    """
    Derive the connected temporal carrier path from the D3 transition edges.

    The lepton D3 layer uses two distinct carrier edges:
      tau             q=5 -> q=4
      muon/electron   q=4 -> q=2

    With the empirical labels removed, these directed edges have one source
    endpoint and one sink endpoint, so they force the chain 5 -> 4 -> 2.
    This is the candidate canonical path used by the q-history moment below.
    """
    costs = lepton_d3_transition_costs()
    edge_labels: dict[tuple[int, int], list[str]] = {}
    for row in costs:
        edge_labels.setdefault((row.q_from, row.q_to), []).append(row.label)

    edges = tuple(edge_labels)
    sources = {src for src, _ in edges}
    targets = {dst for _, dst in edges}
    starts = sources - targets
    sinks = targets - sources

    path: list[int] = []
    complete = False
    if len(starts) == 1 and len(sinks) == 1:
        current = next(iter(starts))
        path.append(current)
        unused = set(edges)
        while unused:
            next_edges = [edge for edge in unused if edge[0] == current]
            if len(next_edges) != 1:
                break
            edge = next_edges[0]
            unused.remove(edge)
            current = edge[1]
            path.append(current)
        complete = not unused and current in sinks

    return {
        "edges": edges,
        "edge_labels": edge_labels,
        "starts": tuple(sorted(starts)),
        "sinks": tuple(sorted(sinks)),
        "path": tuple(path),
        "complete_directed_chain": complete,
    }


def lepton_d3_path_breaking_probe() -> LeptonD3PathBreakingProbe:
    """
    Candidate temporal path-breaking probe for the q=4 -> q=2 degeneracy.

    The K4 flags are tested as path labels, not as a proven transition operator.
    This records the important distinction between the simplified electron
    coefficient -3/16 and the implemented closure coefficient -51/256.
    """
    tau = _k4_phase_selector(
        "tau memory",
        True, False, False,
        Fraction(-27, 1),
        "-(C1/2)*(C2-C1)",
    )
    muon = _k4_phase_selector(
        "muon hard path",
        True, True, False,
        Fraction(-37, 1),
        "-(|H|-|tau|)",
    )
    electron = _k4_phase_selector(
        "electron reset path",
        False, False, True,
        Fraction(-51, 4),
        "-(3*|K4|+C1/8)",
    )
    simplified_electron_64 = Fraction(-12, 1)
    carrier_path = lepton_d3_carrier_path()
    history_path = carrier_path["path"]
    history_xor_moment = q_history_xor_moment(history_path)
    history_split_64 = -(history_xor_moment * WZ_OFFSET)
    byte_interior_roles = Fraction(CODE_C1, 2)
    byte_phase_reset = -byte_interior_roles / 256
    byte_phase_reset_64 = byte_phase_reset * 64
    electron_from_history_64 = (
        muon.normalized_64_cost - history_split_64 + byte_phase_reset_64
    )
    tau_coeff = Fraction(-27, 64) * (CARRIER_TRACES[4] - CARRIER_TRACES[5])
    muon_coeff = Fraction(-37, 64) * (CARRIER_TRACES[2] - CARRIER_TRACES[4])
    electron_coeff = Fraction(-51, 256) * (CARRIER_TRACES[2] - CARRIER_TRACES[4])
    carrier_delta_muon_electron = CARRIER_TRACES[2] - CARRIER_TRACES[4]
    conservation_electron_dyadic = -(tau_coeff + muon_coeff) / carrier_delta_muon_electron
    conservation_electron_64 = conservation_electron_dyadic * 64
    conservation_sum_coeff = tau_coeff + muon_coeff + electron_coeff
    return {
        "selectors": (tau, muon, electron),
        "implemented_muon_electron_split": (
            muon.normalized_64_cost - electron.normalized_64_cost
        ),
        "implemented_muon_electron_split_dyadic": (
            Fraction(-37, 64) - Fraction(-51, 256)
        ),
        "simplified_electron_64_cost": simplified_electron_64,
        "simplified_muon_electron_split": (
            muon.normalized_64_cost - simplified_electron_64
        ),
        "simplified_muon_electron_split_dyadic": (
            Fraction(-37, 64) - Fraction(-3, 16)
        ),
        "claim_25_over_64_matches_implemented": (
            Fraction(-37, 64) - Fraction(-51, 256) == Fraction(-25, 64)
        ),
        "claim_25_over_64_matches_simplified": (
            Fraction(-37, 64) - Fraction(-3, 16) == Fraction(-25, 64)
        ),
        "electron_reset_offset_from_simplified": (
            Fraction(-51, 256) - Fraction(-3, 16)
        ),
        "carrier_path": carrier_path,
        "history_path": history_path,
        "history_xor_moment": history_xor_moment,
        "history_split_64": history_split_64,
        "history_split_dyadic": history_split_64 / 64,
        "history_split_matches_simplified": history_split_64 == Fraction(-25, 1),
        "byte_interior_roles": byte_interior_roles,
        "byte_horizon": Fraction(256, 1),
        "byte_phase_reset": byte_phase_reset,
        "byte_phase_reset_rule": "-(C1/2)/256",
        "byte_phase_reset_64": byte_phase_reset_64,
        "byte_phase_reset_matches_rule": byte_phase_reset == Fraction(-3, 256),
        "electron_from_history_64": electron_from_history_64,
        "electron_from_history_dyadic": electron_from_history_64 / 64,
        "electron_from_history_matches": (
            electron_from_history_64 == electron.normalized_64_cost
        ),
        "carrier_conservation_coeff_sum": conservation_sum_coeff,
        "carrier_conservation_electron_dyadic": conservation_electron_dyadic,
        "carrier_conservation_electron_64": conservation_electron_64,
        "carrier_conservation_current_offset": (
            Fraction(-51, 256) - conservation_electron_dyadic
        ),
        "carrier_conservation_current_matches": (
            Fraction(-51, 256) == conservation_electron_dyadic
        ),
    }


def q_support_words(q: int) -> tuple[int, ...]:
    return tuple(word for word in range(64) if word.bit_count() == q)


def q_history_xor_moment(q_path: tuple[int, ...]) -> Fraction:
    supports = tuple(q_support_words(q) for q in q_path)
    total = 1
    for support in supports:
        total *= len(support)

    weighted_sum = 0

    def visit(index: int, acc: int) -> None:
        nonlocal weighted_sum
        if index == len(supports):
            weighted_sum += acc.bit_count()
            return
        for word in supports[index]:
            visit(index + 1, acc ^ word)

    visit(0, 0)
    return Fraction(weighted_sum, total)


def q_support_duality_probe() -> tuple[QSupportDuality, ...]:
    """
    Exact q-support relations inside the 6-bit chirality register.

    q=2 and q=4 have no direct set intersection, but they are perfectly paired
    by complement x -> x XOR 0x3f. This is the finite-algebra source of their
    isospectrality; a generation split therefore needs path/phase information
    beyond static q-support volume.
    """
    rows: list[QSupportDuality] = []
    for q_a, q_b in ((2, 4), (5, 4)):
        a = set(q_support_words(q_a))
        b = set(q_support_words(q_b))
        rows.append(QSupportDuality(
            q_a=q_a,
            q_b=q_b,
            size_a=len(a),
            size_b=len(b),
            direct_intersection=len(a & b),
            complement_pair_hits=sum(1 for word in a if (word ^ 0x3F) in b),
        ))
    return tuple(rows)


def byte_archetype_shadow_probe() -> ByteArchetypeProbe:
    """
    Exact byte-horizon audit for the final lepton carrier offset.

    In the aQPU byte formalism, transcription is intron = byte XOR 0xAA.
    Hence 0xAA is the unique zero-intron source. The implemented electron
    D3 coefficient differs from exact carrier neutrality by -1/256, and
    multiplying that single byte atom by the shared q=4 -> q=2 carrier delta
    gives the residual carrier coefficient exactly.
    """
    gene_mic = 0xAA
    intron = gene_mic ^ 0xAA
    family = ((intron >> 7) & 1) << 1 | (intron & 1)
    micro_ref = (intron >> 1) & 0x3F
    q_weight = 0
    archetype_shadow = Fraction(1, 256)
    carrier_delta = CARRIER_TRACES[2] - CARRIER_TRACES[4]
    shadow_carrier_coeff = -archetype_shadow * carrier_delta
    residual_carrier_sum = lepton_d3_path_breaking_probe()[
        "carrier_conservation_coeff_sum"
    ]
    return ByteArchetypeProbe(
        gene_mic=gene_mic,
        intron=intron,
        family=family,
        micro_ref=micro_ref,
        q_weight=q_weight,
        archetype_shadow=archetype_shadow,
        carrier_delta=carrier_delta,
        shadow_carrier_coeff=shadow_carrier_coeff,
        residual_carrier_sum=residual_carrier_sum,
        matches_residual=shadow_carrier_coeff == residual_carrier_sum,
    )


def horizon_gate_selection_probe() -> tuple[HorizonGateSelection, ...]:
    """
    Selection audit over the four q-kernel horizon gate bytes.

    The electron reset is tested against the source-traceability criterion:
    it must be an S-gate with no payload mutation and no nontrivial family
    phase. Among q^{-1}(0) = {0xAA, 0x54, 0xD5, 0x2B}, only 0xAA satisfies
    all three conditions.
    """
    gates = (
        (0xAA, "S"),
        (0x54, "S"),
        (0xD5, "C"),
        (0x2B, "C"),
    )
    rows: list[HorizonGateSelection] = []
    for byte, gate in gates:
        intron = byte ^ 0xAA
        family = ((intron >> 7) & 1) << 1 | (intron & 1)
        micro_ref = (intron >> 1) & 0x3F
        q_weight = 0
        is_s_gate = gate == "S"
        zero_intron = intron == 0
        zero_payload = micro_ref == 0
        zero_family = family == 0
        rows.append(HorizonGateSelection(
            byte=byte,
            gate=gate,
            intron=intron,
            family=family,
            micro_ref=micro_ref,
            q_weight=q_weight,
            is_s_gate=is_s_gate,
            zero_intron=zero_intron,
            zero_payload=zero_payload,
            zero_family=zero_family,
            selected_by_electron_reset=(
                is_s_gate and zero_intron and zero_payload and zero_family
            ),
        ))
    return tuple(rows)


def source_traceability_probe() -> SourceTraceabilityProbe:
    """
    Verify the finite source-traceability rule used by the terminal electron
    branch and its dyadic consequence.
    """
    selected = tuple(
        row for row in horizon_gate_selection_probe()
        if row.selected_by_electron_reset
    )
    carrier_delta = CARRIER_TRACES[2] - CARRIER_TRACES[4]
    dyadic_atom = Fraction(1, 256)
    electron_neutral = Fraction(-25, 128)
    electron_implemented = Fraction(-51, 256)
    electron_derived = electron_neutral - dyadic_atom
    return SourceTraceabilityProbe(
        selected_byte=selected[0].byte if selected else -1,
        selected_count=len(selected),
        q_kernel_size=len(horizon_gate_selection_probe()),
        selected_is_gene_mic=(len(selected) == 1 and selected[0].byte == 0xAA),
        dyadic_atom=dyadic_atom,
        carrier_delta=carrier_delta,
        carrier_shadow=-dyadic_atom * carrier_delta,
        electron_neutral=electron_neutral,
        electron_implemented=electron_implemented,
        electron_derived=electron_derived,
        closes_electron_dyadic=(electron_derived == electron_implemented),
    )


def lepton_horizon_wrap_exhaustion_probe(
    max_k: int = 16,
) -> LeptonHorizonWrapExhaustionProbe:
    """
    Exhaustively test the current lepton anchor constraints.

    Broad constraints:
      - k_tau < k_mu < k_e <= max_k
      - the tau/muon 64-normalised D3 budget closes to |H|

    Horizon-wrap candidate rule:
      - k_tau equals the source q of the carrier path, q=5
      - k_mu - k_tau = C1/2
      - k_e - k_mu = C1

    The broad constraints intentionally do not force a unique path. If only
    the horizon-wrap rule gives uniqueness, the probe identifies that rule as
    the missing theorem target.
    """
    q_path = (5, 4, 2)
    expected = (5, 8, 14)
    costs = {row.label: row for row in lepton_d3_transition_costs()}
    tau_mu_budget = (
        abs(costs["tau"].normalized_64_cost)
        + abs(costs["muon"].normalized_64_cost)
    )
    broad: list[tuple[int, int, int]] = []
    horizon_rule: list[tuple[int, int, int]] = []
    expected_delta_tau_mu = Fraction(CODE_C1, 2)
    expected_delta_mu_e = Fraction(CODE_C1, 1)
    for k_tau in range(max_k + 1):
        for k_mu in range(k_tau + 1, max_k + 1):
            for k_e in range(k_mu + 1, max_k + 1):
                seq = (k_tau, k_mu, k_e)
                if tau_mu_budget == HORIZON_CARDINALITY:
                    broad.append(seq)
                if (
                    k_tau == q_path[0]
                    and Fraction(k_mu - k_tau, 1) == expected_delta_tau_mu
                    and Fraction(k_e - k_mu, 1) == expected_delta_mu_e
                ):
                    horizon_rule.append(seq)

    return LeptonHorizonWrapExhaustionProbe(
        max_k=max_k,
        q_path=q_path,
        expected_k_path=expected,
        broad_valid_count=len(broad),
        broad_valid_examples=tuple(broad[:8]),
        horizon_rule_valid_count=len(horizon_rule),
        horizon_rule_valid_paths=tuple(horizon_rule),
        tau_muon_budget=tau_mu_budget,
        tau_muon_budget_matches=(tau_mu_budget == HORIZON_CARDINALITY),
        delta_tau_muon=expected[1] - expected[0],
        delta_muon_electron=expected[2] - expected[1],
        expected_delta_tau_muon=expected_delta_tau_mu,
        expected_delta_muon_electron=expected_delta_mu_e,
        unique_without_horizon_rule=(len(broad) == 1 and broad[0] == expected),
        unique_with_horizon_rule=(len(horizon_rule) == 1 and horizon_rule[0] == expected),
    )


def shell_spectral_weights() -> tuple[ShellSpectralWeight, ...]:
    """
    Exact Hilbert-Schmidt/Frobenius diagnostics for the q-weight operators.

    shell_frobenius_sq is computed from the 7x7 shell transition matrix.
    full_operator_hs_sq is the exact Frobenius norm squared of the normalized
    q-step Hamming operator on the full 64-point chirality register:
        ||T_q||_F^2 = 64 / C(6,q)
    because each row has C(6,q) entries of weight 1/C(6,q).
    """
    rows: list[ShellSpectralWeight] = []
    for q in range(7):
        M = shell_transition_matrix(q)
        shell_frob = sum(
            (M[i][j] * M[i][j] for i in range(7) for j in range(7)),
            Fraction(0),
        )
        rows.append(ShellSpectralWeight(
            q=q,
            byte_count=4 * comb(6, q),
            trace=shell_trace(q),
            return_trace=shell_return_trace(q),
            carrier=carrier_trace(q),
            shell_frobenius_sq=shell_frob,
            full_operator_hs_sq=Fraction(64, comb(6, q)),
        ))
    return tuple(rows)


def lepton_d3_spectral_obstruction() -> LeptonD3SpectralObstruction:
    """
    Test whether spectral weights alone force the lepton D3 dyadics.

    Result: q=2 and q=4 are isospectral at the full Hamming-operator level,
    so a rule depending only on q-shell Hilbert-Schmidt weights cannot split
    muon and electron, which share q=4 -> q=2 but require different dyadics.
    """
    weights = {row.q: row for row in shell_spectral_weights()}
    q2 = weights[2]
    q4 = weights[4]
    q5 = weights[5]
    dyadic_ratio = Fraction(-37, 64) / Fraction(-51, 256)
    hs_ratio_q5_q4 = q5.full_operator_hs_sq / q4.full_operator_hs_sq
    return {
        "weights": tuple(weights[q] for q in range(7)),
        "q2_q4_full_hs_equal": q2.full_operator_hs_sq == q4.full_operator_hs_sq,
        "q2_q4_return_trace_equal": q2.return_trace == q4.return_trace,
        "q2_q4_shell_frobenius_equal": q2.shell_frobenius_sq == q4.shell_frobenius_sq,
        "q5_over_q4_full_hs_ratio": hs_ratio_q5_q4,
        "q5_over_q4_byte_support_ratio": Fraction(q4.byte_count, q5.byte_count),
        "muon_electron_dyadic_ratio": dyadic_ratio,
        "hs_ratio_splits_muon_electron": dyadic_ratio == Fraction(1, 1),
    }


def lepton_d3_shell_lemmas() -> LeptonD3ShellLemmas:
    """
    Exact kernel facts for the lepton D3 carrier layer (no masses, no Delta).

    The lepton ladder uses fixed dyadic coefficients (-27/64, -37/64, -51/256)
    multiplied by carrier differences and Delta^3. This function records only
    what follows from shell_transition_matrix / carrier_trace definitions.

    Returns a plain dict suitable for inspection or documentation checks.
    """
    rows: list[LeptonCarrierPathProbe] = []
    for q in range(7):
        rows.append(
            {
                "q": q,
                "Tr_M": shell_trace(q),
                "Tr_M2": shell_return_trace(q),
                "carrier": carrier_trace(q),
                "byte_count": 4 * comb(6, q),
            }
        )

    c2 = CARRIER_TRACES[2]
    c4 = CARRIER_TRACES[4]
    c5 = CARRIER_TRACES[5]
    d_mu_e = c2 - c4
    d_tau = c4 - c5

    f_tau = Fraction(-27, 64)
    f_mu = Fraction(-37, 64)
    f_e = Fraction(-51, 256)

    ratio_sq = (c2 * c2) / (c4 * c4)
    ratio_dyadic = f_mu / f_e
    transition_costs = lepton_d3_transition_costs()
    selection_rules = lepton_d3_selection_rule_probe()
    spectral_obstruction = lepton_d3_spectral_obstruction()
    path_breaking = lepton_d3_path_breaking_probe()
    archetype_shadow = byte_archetype_shadow_probe()
    horizon_gate_selection = horizon_gate_selection_probe()
    support_duality = q_support_duality_probe()
    cost_by_label = {row.label: row for row in transition_costs}
    tau64 = abs(cost_by_label["tau"].normalized_64_cost)
    mu64 = abs(cost_by_label["muon"].normalized_64_cost)
    electron64 = abs(cost_by_label["electron"].normalized_64_cost)

    return {
        "shell_rows": tuple(rows),
        "carrier_delta_muon_electron": d_mu_e,
        "carrier_delta_tau": d_tau,
        "Tr_M2_equal_q2_q4": shell_return_trace(2) == shell_return_trace(4),
        "squared_carrier_ratio_C2_over_C4": ratio_sq,
        "implemented_dyadic_ratio_muon_over_electron": ratio_dyadic,
        "squared_carrier_ratio_equals_dyadic_ratio": ratio_sq == ratio_dyadic,
        "dyadic_tau": f_tau,
        "dyadic_muon": f_mu,
        "dyadic_electron": f_e,
        "transition_costs": transition_costs,
        "tau_muon_horizon_budget": tau64 + mu64,
        "tau_muon_horizon_budget_matches": tau64 + mu64 == HORIZON_CARDINALITY,
        "muon_minus_tau_cost": mu64 - tau64,
        "muon_minus_tau_matches_C3_over_2": mu64 - tau64 == Fraction(CODE_C3, 2),
        "electron_64_cost": electron64,
        "electron_64_cost_rule": Fraction(12, 1) + Fraction(CODE_C1, 8),
        "electron_64_cost_rule_matches": (
            electron64 == Fraction(12, 1) + Fraction(CODE_C1, 8)
        ),
        "selection_rules": selection_rules,
        "selection_rules_all_match": all(rule.matches for rule in selection_rules),
        "spectral_obstruction": spectral_obstruction,
        "path_breaking": path_breaking,
        "archetype_shadow": archetype_shadow,
        "horizon_gate_selection": horizon_gate_selection,
        "support_duality": support_duality,
    }


def quark_delta_coords(delta: float = DELTA) -> dict[str, float]:
    """
    Quark EW-referred Delta-ruler coordinates.
    Selectors sample all four vertices of the {kappa, omega} Boolean lattice.

    PDG uncertainty floor (~1% b/c, ~5% s) maps to ~0.7 tick uncertainty;
    these are compact selectors, not scale-independent mass predictions.
    """
    alg = compact_algebra(delta)
    return {
        "n_bottom":  284.0 + alg.kappa,
        "n_charm":   367.0 + alg.omega + 0.5 * delta,
        "n_strange": 548.0 - (alg.omega + alg.kappa),
    }


def strong_scale_bulk_probe(
    lambda_qcd_gev: float = 0.2,
    delta: float = DELTA,
    ew_scale_gev: float = E_EW_GEV,
) -> StrongScaleBulkProbe:
    """
    Place the conventional strong scale on the Delta ruler and audit whether
    the corresponding finite-geometric interpretation is bulk, not horizon.
    """
    n_qcd = math.log2(ew_scale_gev / lambda_qcd_gev) / delta
    boundary_states = 2 * HORIZON_CARDINALITY
    bulk_states = OMEGA_SIZE - boundary_states
    return StrongScaleBulkProbe(
        lambda_qcd_gev=lambda_qcd_gev,
        n_qcd=n_qcd,
        n_phase_64=n_qcd % HORIZON_CARDINALITY,
        n_phase_48=n_qcd % APERTURE_FRAME,
        bulk_states=bulk_states,
        boundary_states=boundary_states,
        bulk_fraction=Fraction(bulk_states, OMEGA_SIZE),
        boundary_fraction=Fraction(boundary_states, OMEGA_SIZE),
        placed_in_relational_bulk=(bulk_states > boundary_states),
    )


def qcd_aperture_cycle_residual_probe(
    lambda_qcd_gev: float = 0.2,
    delta: float = DELTA,
    ew_scale_gev: float = E_EW_GEV,
    tolerance: float = 1e-12,
) -> QCDApertureCycleResidualProbe:
    """
    Test the proposed 10*APERTURE_FRAME + r(electron) QCD coordinate.

    This is a grammar probe only. Lambda_QCD remains an external input unless
    the residual closes to an exact compact expression at the declared
    tolerance.
    """
    n_qcd = math.log2(ew_scale_gev / lambda_qcd_gev) / delta
    electron_offset = int(M_SHELL / 8 - M_SHELL / 24)
    base = 10.0 * APERTURE_FRAME + electron_offset
    residual = base - n_qcd
    alg = compact_algebra(delta)
    grammar = (
        ("3*Delta", 3.0 * delta),
        ("epsilon-eta", alg.epsilon - alg.eta),
        ("epsilon/5", alg.epsilon / 5.0),
        ("Delta/5*APERTURE_FRAME", delta * APERTURE_FRAME / 5.0),
        ("3*Delta-eta/2", 3.0 * delta - alg.eta / 2.0),
        ("3*Delta-eta/2+Delta^2/8", 3.0 * delta - alg.eta / 2.0 + delta ** 2 / 8.0),
        ("Delta/APERTURE_FRAME", delta / APERTURE_FRAME),
    )
    rows = tuple(
        GrammarResidualCandidate(
            label=label,
            value=value,
            residual_minus_value=residual - value,
            abs_error=abs(residual - value),
            matches=(abs(residual - value) <= tolerance),
        )
        for label, value in grammar
    )
    best = min(rows, key=lambda row: row.abs_error)
    closes = any(row.matches for row in rows)
    return QCDApertureCycleResidualProbe(
        lambda_qcd_gev=lambda_qcd_gev,
        n_qcd=n_qcd,
        aperture_cycles=10,
        electron_offset=electron_offset,
        candidate_base=base,
        residual=residual,
        candidates=rows,
        best_label=best.label,
        best_abs_error=best.abs_error,
        closes_to_grammar=closes,
        remains_external_input=not closes,
    )


def spectral_triple_probe() -> SpectralTripleProbe:
    """
    Finite checks for the candidate spectral triple.

    This probe establishes the finite matrix triple data. It deliberately does
    not mark the physics spectral triple as complete until grading, real
    structure, finite metric, and spectral action are specified.
    """
    return SpectralTripleProbe(
        algebra_generators=256,
        hilbert_dimension=OMEGA_SIZE,
        shell_count=7,
        d_shell_trace=HORIZON_CARDINALITY * M_SHELL,
        d_code_trace=M_SHELL,
        zero_mode_dimension=HORIZON_CARDINALITY,
        finite_commutators_bounded=True,
        grading_specified=False,
        real_structure_specified=False,
        spectral_action_specified=False,
        complete_physics_triple=False,
    )


# ---------------------------------------------------------------------------
# Layer 6: Observable definitions and coordinate machinery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observable:
    name: str
    value: float      # in declared unit
    unit: str
    dimension: str    # "energy" | "frequency" | "time" | "length"
    source: str


@dataclass(frozen=True)
class FundamentalConstants:
    c: float
    h: float
    hbar: float
    G: float
    e: float
    k_B: float


@dataclass(frozen=True)
class ScaleReference:
    time_s: float
    length_m: float
    energy_GeV: float
    frequency_Hz: float


def build_constants() -> FundamentalConstants:
    if SCIPY_AVAILABLE:
        return FundamentalConstants(
            c=sc.c, h=sc.h, hbar=sc.hbar,
            G=sc.G, e=sc.e, k_B=sc.k,
        )
    return FundamentalConstants(
        c=299_792_458.0,
        h=6.626_070_15e-34,
        hbar=1.054_571_817e-34,
        G=6.674_30e-11,
        e=1.602_176_634e-19,
        k_B=1.380_649e-23,
    )


def planck_scales(const: FundamentalConstants) -> ScaleReference:
    t_p = math.sqrt(const.hbar * const.G / const.c ** 5)
    l_p = math.sqrt(const.hbar * const.G / const.c ** 3)
    e_p_j = math.sqrt(const.hbar * const.c ** 5 / const.G)
    e_p_gev = e_p_j / (const.e * 1.0e9)
    return ScaleReference(
        time_s=t_p,
        length_m=l_p,
        energy_GeV=e_p_gev,
        frequency_Hz=1.0 / t_p,
    )


def ew_scales(const: FundamentalConstants) -> ScaleReference:
    energy_gev = E_EW_GEV
    energy_j = energy_gev * const.e * 1.0e9
    freq = energy_j / const.h
    return ScaleReference(
        time_s=1.0 / freq,
        length_m=const.c / freq,
        energy_GeV=energy_gev,
        frequency_Hz=freq,
    )


def _electron_mass_gev(const: FundamentalConstants) -> float:
    return ELECTRON_MASS_GEV


def _proton_mass_gev(const: FundamentalConstants) -> float:
    if SCIPY_AVAILABLE:
        return sc.physical_constants["proton mass energy equivalent in MeV"][0] / 1000.0
    return 0.938_272_088


def _reduced_compton_gev(const: FundamentalConstants, mass_gev: float) -> float:
    mass_kg = (mass_gev * const.e * 1.0e9) / const.c ** 2
    return const.hbar / (mass_kg * const.c)


def _bohr_radius(const: FundamentalConstants) -> float:
    if SCIPY_AVAILABLE:
        return sc.physical_constants["Bohr radius"][0]
    m_e = 9.109_383_7015e-31
    eps0 = 8.854_187_8128e-12
    return 4.0 * math.pi * eps0 * const.hbar ** 2 / (m_e * const.e ** 2)


def build_observables(const: FundamentalConstants) -> list[Observable]:
    """
    Canonical set of observables used in the compact geometry analysis.
    Values are from PDG 2024 and NIST/CODATA for particle masses and
    fundamental constants respectively.
    """
    planck = planck_scales(const)
    ew = ew_scales(const)

    e_mass  = _electron_mass_gev(const)
    p_mass  = _proton_mass_gev(const)

    # PDG 2024 electroweak masses
    m_higgs = 125.10
    m_w     = 80.379
    m_z     = 91.1876
    m_top   = 172.76
    m_bot   = 4.18
    m_charm = 1.27
    m_str   = 0.095
    m_muon  = 0.105_658_374_5
    m_tau   = 1.776_86

    h_hf_hz = 1_420_405_751.768   # hydrogen 21 cm

    obs: list[Observable] = [
        # Anchors
        Observable("Planck energy",      planck.energy_GeV, "GeV", "energy", "CODATA"),
        Observable("Electroweak scale",  ew.energy_GeV,     "GeV", "energy", "CGM anchor"),
        # EW masses
        Observable("Top quark mass energy",    m_top,   "GeV", "energy", "PDG 2024"),
        Observable("Higgs mass energy",        m_higgs, "GeV", "energy", "PDG 2024"),
        Observable("Z boson mass energy",      m_z,     "GeV", "energy", "PDG 2024"),
        Observable("W boson mass energy",      m_w,     "GeV", "energy", "PDG 2024"),
        # Quarks
        Observable("Bottom quark mass energy", m_bot,   "GeV", "energy", "PDG 2024"),
        Observable("Charm quark mass energy",  m_charm, "GeV", "energy", "PDG 2024"),
        Observable("Strange quark mass energy",m_str,   "GeV", "energy", "PDG 2024; MS-bar"),
        # Leptons
        Observable("Tau mass energy",          m_tau,   "GeV", "energy", "PDG 2024"),
        Observable("Muon mass energy",         m_muon,  "GeV", "energy", "PDG 2024"),
        Observable("Electron mass energy",     e_mass,  "GeV", "energy", "CODATA"),
        # Hadrons
        Observable("Proton mass energy",       p_mass,  "GeV", "energy", "CODATA"),
        # Frequencies
        Observable("Planck frequency",         planck.frequency_Hz, "Hz", "frequency", "CODATA"),
        Observable("Electroweak frequency",    ew.frequency_Hz,     "Hz", "frequency", "CGM anchor"),
        Observable("Cs-133 hyperfine",         F_CS_133_HZ,         "Hz", "frequency", "exact SI"),
        Observable("Hydrogen 21 cm",           h_hf_hz,             "Hz", "frequency", "astrophysics"),
        # Times
        Observable("Planck time",  planck.time_s,       "s",  "time",   "CODATA"),
        Observable("Cs-133 period",1.0 / F_CS_133_HZ,  "s",  "time",   "exact SI"),
        # Lengths
        Observable("Planck length",         planck.length_m,              "m", "length", "CODATA"),
        Observable("Electron Compton",      _reduced_compton_gev(const, e_mass), "m", "length", "CODATA"),
        Observable("Proton Compton",        _reduced_compton_gev(const, p_mass), "m", "length", "CODATA"),
        Observable("Bohr radius",           _bohr_radius(const),          "m", "length", "CODATA"),
        Observable("Hydrogen 21 cm length", const.c / h_hf_hz,           "m", "length", "astrophysics"),
    ]
    return obs


def ew_log2_separation(
    observable_mass_gev: float,
    ew_scale_gev: float = E_EW_GEV,
) -> float:
    """log2(v / m) : EW-referred log2 gap."""
    return math.log2(ew_scale_gev / observable_mass_gev)


def ew_delta_n(
    observable_mass_gev: float,
    ew_scale_gev: float = E_EW_GEV,
    delta: float = DELTA,
) -> float:
    """Delta-ruler coordinate n = log2(v/m) / delta."""
    return ew_log2_separation(observable_mass_gev, ew_scale_gev) / delta


def planck_delta_n(obs: Observable, planck: ScaleReference) -> float:
    """Planck-referred Delta-ruler coordinate."""
    dim = obs.dimension
    if dim == "energy":
        ratio = planck.energy_GeV / obs.value
    elif dim == "frequency":
        ratio = planck.frequency_Hz / obs.value
    elif dim == "time":
        ratio = obs.value / planck.time_s
    elif dim == "length":
        ratio = obs.value / planck.length_m
    else:
        raise ValueError(f"Unsupported dimension: {dim!r}")
    return math.log2(ratio) / DELTA


def optical_depth(n: float, delta: float = DELTA) -> float:
    """Beer-Lambert optical depth tau = n * delta * ln2."""
    return n * delta * math.log(2.0)


# ---------------------------------------------------------------------------
# Compound results: full electroweak coordinate set
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ElectroweakCoords:
    """All EW-referred Delta-ruler coordinates at a given order."""
    order: int
    delta: float
    n_top: float
    n_higgs: float
    n_z: float
    n_w: float
    n_tau: float
    n_mu: float
    n_electron: float
    n_bottom: float
    n_charm: float
    n_strange: float


def electroweak_coords(delta: float = DELTA, *, order: int = 5) -> ElectroweakCoords:
    """Compute all EW Delta-ruler coordinates at the requested law order."""
    laws = all_laws(delta, order=order)
    ch_top = channel_by_label("Top")
    ch_h   = channel_by_label("Higgs")

    n_top   = laws["Top"]   / delta
    n_higgs = laws["Higgs"] / delta
    n_z     = laws["Z"]     / delta
    n_w     = laws["W"]     / delta

    lep = lepton_delta_coords(n_higgs, delta)
    qrk = quark_delta_coords(delta)

    return ElectroweakCoords(
        order=order,
        delta=delta,
        n_top=n_top,
        n_higgs=n_higgs,
        n_z=n_z,
        n_w=n_w,
        n_tau=lep["n_tau"],
        n_mu=lep["n_mu"],
        n_electron=lep["n_electron"],
        n_bottom=qrk["n_bottom"],
        n_charm=qrk["n_charm"],
        n_strange=qrk["n_strange"],
    )


# ---------------------------------------------------------------------------
# Coupling parametrisation (tree-level, from model masses)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EWCouplings:
    lambda_H: float
    g: float
    g_z: float
    g_prime: float
    e: float
    alpha_ew_inv: float
    y_t: float


def ew_couplings_from_masses(
    m_t: float, m_h: float, m_z: float, m_w: float,
    v: float = E_EW_GEV,
) -> EWCouplings:
    """Tree-level couplings from mass inputs (GeV)."""
    lambda_h = m_h ** 2 / (2.0 * v ** 2)
    g        = 2.0 * m_w / v
    g_z      = 2.0 * m_z / v
    g_prime  = math.sqrt(max(g_z ** 2 - g ** 2, 0.0))
    e_charge = g * g_prime / g_z if g_z != 0.0 else float("nan")
    alpha_inv = 4.0 * math.pi / e_charge ** 2 if e_charge != 0.0 else float("inf")
    y_t      = math.sqrt(2.0) * m_t / v
    return EWCouplings(
        lambda_H=lambda_h, g=g, g_z=g_z, g_prime=g_prime,
        e=e_charge, alpha_ew_inv=alpha_inv, y_t=y_t,
    )


def model_masses(delta: float = DELTA, v: float = E_EW_GEV) -> dict[str, float]:
    """Masses predicted by the five-order law."""
    laws = all_laws(delta, order=5)
    return {
        "m_top":   v * 2.0 ** (-laws["Top"]),
        "m_higgs": v * 2.0 ** (-laws["Higgs"]),
        "m_z":     v * 2.0 ** (-laws["Z"]),
        "m_w":     v * 2.0 ** (-laws["W"]),
    }


def model_couplings(delta: float = DELTA, v: float = E_EW_GEV) -> EWCouplings:
    """Couplings from model-predicted masses (not PDG inputs)."""
    m = model_masses(delta, v)
    return ew_couplings_from_masses(m["m_top"], m["m_higgs"], m["m_z"], m["m_w"], v)


def model_couplings_d2(delta: float = DELTA, v: float = E_EW_GEV) -> EWCouplings:
    """Couplings from D2-order model masses (law without gyroscopic corrections)."""
    laws = all_laws(delta, order=2)
    m_t = v * 2.0 ** (-laws["Top"])
    m_h = v * 2.0 ** (-laws["Higgs"])
    m_z = v * 2.0 ** (-laws["Z"])
    m_w = v * 2.0 ** (-laws["W"])
    return ew_couplings_from_masses(m_t, m_h, m_z, m_w, v)


# ---------------------------------------------------------------------------
# W/Z mixing and ratio-channel functions
# ---------------------------------------------------------------------------

def wz_split(delta: float, *, promoted: bool = True) -> float:
    """
    Charged-neutral split S_WZ such that log2(mZ/mW) = delta * S_WZ.

    promoted=False: D2 backbone only:   (C2-C1) - (C3/2)*D
    promoted=True:  full D4 law:        + 2D^2/sqrt(5) - D^3
    """
    base = WZ_OFFSET - WZ_APERTURE_COEFF * delta
    if not promoted:
        return base
    return base + 2.0 * delta ** 2 / math.sqrt(5.0) - delta ** 3


def sin2_theta_w(delta: float, *, promoted: bool = True) -> float:
    """sin^2(theta_W) from compact promoted split law."""
    split = wz_split(delta, promoted=promoted)
    cos_w = 2.0 ** (-delta * split)
    return 1.0 - cos_w ** 2


def w_mass_from_z(m_z: float, delta: float) -> float:
    """Predict m_W from m_Z and delta using promoted split law."""
    split = wz_split(delta, promoted=True)
    return m_z / 2.0 ** (delta * split)


# ---------------------------------------------------------------------------
# CKM compact ansatz
# ---------------------------------------------------------------------------

def ckm_ansatz(delta: float = DELTA) -> dict[str, float]:
    """
    Compact CKM angle ansatz (empirical selector / lead).
    V_us ~ sin(delta_BU + 3D/2)
    V_cb ~ sin(2D)
    V_ub exclusive ~ sin(9D^2)
    V_ub inclusive ~ sin(9D^2 + phi_conv)  where phi_conv = eta - sin(9D^2)
    """
    alg = compact_algebra(delta)
    v_us = math.sin(DELTA_BU + 1.5 * delta)
    v_cb = math.sin(2.0 * delta)
    excl = math.sin(9.0 * delta ** 2)
    phi_conv = alg.eta - excl
    v_ub_incl = math.sin(9.0 * delta ** 2 + phi_conv)
    return {
        "V_us":       v_us,
        "V_cb":       v_cb,
        "V_ub_excl":  excl,
        "V_ub_incl":  v_ub_incl,
        "phi_conv":   phi_conv,
        "delta_CKM":  math.degrees(math.pi / 2.0 - 18.0 * delta),
    }


# ---------------------------------------------------------------------------
# H/Z/W leave-one-out
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LeaveOneOutResult:
    target: str
    delta_source: str
    delta_used: float
    predicted_mass: float
    reference_mass: float

    @property
    def relative_error(self) -> float:
        return self.predicted_mass / self.reference_mass - 1.0


def _law_for(label: str, delta: float) -> float:
    return eval_law(channel_by_label(label), delta, order=2)


def hzw_leave_one_out(
    observed: dict[str, float],
    backsolves: tuple[DeltaBacksolve, ...],
) -> tuple[LeaveOneOutResult, ...]:
    """
    For each of H, Z, W: predict mass using Delta averaged from the other two.
    """
    v = observed["Electroweak scale"]
    bs = {b.source: b.delta_back for b in backsolves}
    results: list[LeaveOneOutResult] = []
    for target in ("Higgs", "Z", "W"):
        others = [s for s in ("Higgs", "Z", "W") if s != target]
        delta_used = sum(bs[s] for s in others) / 2.0
        log_gap = _law_for(target, delta_used)
        pred = v * 2.0 ** (-log_gap)
        ref = observed[channel_by_label(target).observable]
        results.append(LeaveOneOutResult(
            target=target,
            delta_source="+".join(others),
            delta_used=delta_used,
            predicted_mass=pred,
            reference_mass=ref,
        ))
    return tuple(results)
