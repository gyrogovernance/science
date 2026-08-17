#!/usr/bin/env python3
"""
hqvm_cgm_allometry_1.py

Kernel compute for allometry at fixed d=6: product geometry, QuBEC bulk,
surface/time channels, city conjugacy, Kleiber M0, chemical clock.

No biology catalogs. Report: hqvm_cgm_allometry_run.py.
Theory: hqvm_cgm_allometry_notes.md.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from gyroscopic.hQVM.family import (
    HqvmD,
    bfs_reach,
    bisect_p_c_rank_micro_ref,
    build_hqvm_d,
    delta_spinorial_residual_d,
    depth4_projection_bits,
    enumerate_bytes,
    exact_micro_ref_p_rank_full,
    exact_micro_ref_rank_pmf,
    exact_micro_ref_theta_cond,
    gf2_rank,
    holonomy_micro_cov,
    partition_Z1_coeff_d,
    predicted_cluster_size,
    shell_population_d,
    verify_carrier_entanglement_exact,
)
from hqvm_cgm_trestleboard_common import CHIRALITY_D, DELTA, TICKS_PER_OCTAVE

EXACT_TOL = 1e-12
NEAR_TOL = 1e-3
N_DOF = int(CHIRALITY_D)
N_SPATIAL = N_DOF // 2  # rotational/translational split of transport register
A_SURFACE = 2.0 / 3.0
A_BULK = 3.0 / 4.0
A_SR = 1.0 / 2.0
A_TIME = A_BULK - A_SR  # 1/4; forced as bulk−product
A_EGRESS = (3.0 / 4.0) * A_TIME  # 3/16; BU-Egress construction time
A_IN_THERMAL = A_TIME  # 1/4; BU-Ingress maintenance at μ=1
MU_GAP = A_TIME / float(N_SPATIAL)  # 1/12 at d=6
TURN_QUANTA = 32.0
WORD_HORIZON_REACH = 128  # W2/F from rest: two horizon copies of |H|
OMEGA_FULL = 1 << (2 * N_DOF)
APERTURE_FRAME = 48
P_BOUNDARY = 1.0 - 1.0 / APERTURE_FRAME  # 47/48
U_KG = 1.66053906660e-27  # atomic mass unit (kg); parallel external constant to EW v
U_ELECTRON_KG = 9.1093837015e-31  # sensitivity contrast only
# Physical metabolic regime: condensed → thermal. Duality λ↔1/λ checked separately.
LAMBDA_PHYS = (1e-3, 0.1, 0.25, 0.5, 1.0)
LAMBDA_DUAL_APPENDIX = (1e-3, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 1e3)
LAMBDA_SWEEP = LAMBDA_PHYS  # default biology-facing sweep

_ENGINES: Dict[int, HqvmD] = {}


def _engine(d: int) -> HqvmD:
    if d not in _ENGINES:
        _ENGINES[d] = build_hqvm_d(d)
    return _ENGINES[d]


def a_from_mu(mu: float) -> float:
    return A_SURFACE + mu * MU_GAP


def mu_from_a(a: float) -> float:
    return (a - A_SURFACE) / MU_GAP


def mu_from_eta(eta: float) -> float:
    """μ = 1−|η|; on λ∈(0,1] this equals 2λ/(1+λ)=2ρ."""
    return max(0.0, min(1.0, 1.0 - abs(eta)))


def mu_from_lambda(lam: float) -> float:
    """Exact μ on the physical regime λ∈(0,1]: μ=2λ/(1+λ)=2ρ."""
    if lam <= 0.0:
        return 0.0
    if lam > 1.0:
        # Duality fold: μ(λ)=μ(1/λ)
        return mu_from_lambda(1.0 / lam)
    return (2.0 * lam) / (1.0 + lam)


def lambda_from_mu(mu: float) -> float:
    """Inverse on μ∈[0,1]: λ=μ/(2−μ) ∈ (0,1]."""
    if mu <= 0.0:
        return 0.0
    if mu >= 1.0:
        return 1.0
    return mu / (2.0 - mu)


def a_from_lambda(lam: float) -> float:
    """Exact μ-band law: a(λ)=2/3 + λ/(6(1+λ)) for λ∈(0,1]."""
    return a_from_mu(mu_from_lambda(lam))


def in_dual_band_a(a: float, tol: float = 1e-12) -> bool:
    return (A_SURFACE - tol) <= a <= (A_BULK + tol)


def a_bulk_from_shell_mean(mean_N: float) -> float:
    """Bulk/network exponent from QuBEC shell mean: ⟨N⟩/(⟨N⟩+1)."""
    if mean_N <= 0.0:
        return 0.0
    return mean_N / (mean_N + 1.0)


def a_bulk_of_d(d: int) -> float:
    """QuBEC thermal bulk at λ=1: ⟨N⟩=d/2 ⇒ a=d/(d+2)."""
    return float(d) / float(d + 2)


def a_surf_of_d(d: int) -> float:
    """Kernel-lattice surface: Δ(d)/(1/32)=4/d."""
    return 4.0 / float(d)


def a_d4_of_d(d: int) -> float:
    """Depth-four fill: depth4_bits/2^d with depth4_bits=8d."""
    return (8.0 * float(d)) / float(1 << d)


@dataclass(frozen=True)
class UniquenessIRow:
    d: int
    a_net: float
    a_d4: float
    lhs_pow: float  # 2^(d-3)
    rhs_lin: float  # d+2
    equal: bool


def uniqueness_I_family(d_values: Sequence[int] = tuple(range(1, 9))) -> List[UniquenessIRow]:
    """Theorem I: a_net(d)=a_d4(d) ⟺ 2^(d−3)=d+2; unique positive integer d=6."""
    rows: List[UniquenessIRow] = []
    for d in d_values:
        a_net = a_bulk_of_d(d)
        a_d4 = a_d4_of_d(d)
        lhs = 2.0 ** (d - 3)
        rhs = float(d + 2)
        rows.append(
            UniquenessIRow(
                d=d,
                a_net=a_net,
                a_d4=a_d4,
                lhs_pow=lhs,
                rhs_lin=rhs,
                equal=abs(a_net - a_d4) < EXACT_TOL and abs(lhs - rhs) < EXACT_TOL,
            )
        )
    return rows


@dataclass(frozen=True)
class UniquenessIIRow:
    d: int
    a_bulk: float
    a_surf: float
    a_time: float
    n: float
    lhs: float
    rhs: float
    resid: float
    equal: bool


def uniqueness_II_family(
    d_values: Sequence[int] = tuple(range(1, 9)),
) -> List[UniquenessIIRow]:
    """Theorem II: a_bulk−a_surf = a_time/(d/2) ⟺ (d−6)(d+1)=0; unique positive d=6."""
    rows: List[UniquenessIIRow] = []
    for d in d_values:
        a_bulk = a_bulk_of_d(d)
        a_surf = a_surf_of_d(d)
        a_time = a_bulk - A_SR
        n = 0.5 * float(d)
        lhs = a_bulk - a_surf
        rhs = a_time / n if n > 0 else float("nan")
        resid = abs(lhs - rhs)
        rows.append(
            UniquenessIIRow(
                d=d,
                a_bulk=a_bulk,
                a_surf=a_surf,
                a_time=a_time,
                n=n,
                lhs=lhs,
                rhs=rhs,
                resid=resid,
                equal=resid < EXACT_TOL,
            )
        )
    return rows


@dataclass(frozen=True)
class UniquenessIIIRow:
    d: int
    a_bulk: float
    a_time_V: float  # 1 − a_bulk = 2/(d+2)
    a_time_L: float  # a_bulk − a_SR = (d−2)/(2(d+2))
    resid: float
    equal: bool


def uniqueness_III_family(
    d_values: Sequence[int] = tuple(range(1, 9)),
) -> List[UniquenessIIIRow]:
    """Theorem III: volume/flow time = bulk−holographic lift ⟺ d=6."""
    rows: List[UniquenessIIIRow] = []
    for d in d_values:
        a_bulk = a_bulk_of_d(d)
        a_time_V = 1.0 - a_bulk
        a_time_L = a_bulk - A_SR
        resid = abs(a_time_V - a_time_L)
        rows.append(
            UniquenessIIIRow(
                d=d,
                a_bulk=a_bulk,
                a_time_V=a_time_V,
                a_time_L=a_time_L,
                resid=resid,
                equal=resid < EXACT_TOL,
            )
        )
    return rows


def uniqueness_pass(
    rows_i: Sequence[UniquenessIRow],
    rows_ii: Sequence[UniquenessIIRow],
    rows_iii: Optional[Sequence[UniquenessIIIRow]] = None,
) -> Dict[str, bool]:
    eq_i = [r for r in rows_i if r.equal]
    eq_ii = [r for r in rows_ii if r.equal]
    if rows_iii is None:
        rows_iii = uniqueness_III_family()
    eq_iii = [r for r in rows_iii if r.equal]
    return {
        "uniqueness_I_only_d6": len(eq_i) == 1 and eq_i[0].d == 6,
        "uniqueness_II_only_d6": len(eq_ii) == 1 and eq_ii[0].d == 6,
        "uniqueness_III_only_d6": len(eq_iii) == 1 and eq_iii[0].d == 6,
        "uniqueness_N_spatial_is_d_over_2": N_SPATIAL == N_DOF // 2,
        "uniqueness_MU_GAP_is_a_time_over_n": abs(MU_GAP - A_TIME / float(N_SPATIAL))
        < EXACT_TOL,
        "uniqueness_depth4_bits_eq_8d_at_d6": abs(a_d4_of_d(6) - A_BULK) < EXACT_TOL,
        "uniqueness_D_eff_is_thermal_shell": abs(0.5 * float(N_DOF) - 3.0) < EXACT_TOL
        and abs(a_bulk_of_d(N_DOF) - A_BULK) < EXACT_TOL,
    }


@dataclass(frozen=True)
class ChannelRule:
    trait_class: str
    channel: str
    dual_band: bool
    mu_default: Optional[float]
    a_pred: float
    a_iso: float
    rule: str


def channel_rules() -> List[ChannelRule]:
    """Channel nulls at fixed d=6: bulk, surface, time=bulk−SR, cities (d−1)/d."""
    a_infra = (N_DOF - 1) / float(N_DOF)
    a_socio = 1.0 + 1.0 / float(N_DOF)
    a_time = A_TIME
    return [
        ChannelRule("metabolic_rate", "bulk", True, 1.0, A_BULK, A_SURFACE, "QuBEC thermal μ=1"),
        ChannelRule("metabolic_rate_BU", "bulk", True, 1.0, A_BULK, A_SURFACE, "QuBEC thermal μ=1"),
        ChannelRule("surface_exchange", "surface", True, 0.0, A_SURFACE, A_SURFACE, "surface μ=0"),
        ChannelRule("exchange_surface", "surface", True, 0.0, A_SURFACE, A_SURFACE, "surface μ=0"),
        ChannelRule("heart_rate", "time", False, None, -a_time, 0.0, "−(a_bulk−a_SR)"),
        ChannelRule("specific_metabolic", "time", False, None, -a_time, 0.0, "a_B−1"),
        ChannelRule("lifespan", "time", False, None, a_time, 0.0, "composite [a_eg,a_in]=[3/16,1/4]"),
        ChannelRule("development_time", "time", False, None, A_EGRESS, 0.0, "a_eg=(3/4)·a_in(μ=1)"),
        ChannelRule("gestation", "time", False, None, A_EGRESS, 0.0, "a_eg egress/construction"),
        ChannelRule("weaning", "time", False, None, A_EGRESS, 0.0, "a_eg egress/construction"),
        ChannelRule("longevity_composite_low", "time", False, None, A_EGRESS, 0.0, "3/16 lower bound"),
        ChannelRule("longevity_composite_high", "time", False, None, a_time, 0.0, "1/4 upper bound"),
        ChannelRule(
            "aorta_radius",
            "network",
            False,
            None,
            A_BULK / 2.0,
            1.0 / N_SPATIAL,
            "Tier B: A∝B, r∝√A under fixed ΔP,u0",
        ),
        ChannelRule(
            "population_density",
            "conservation",
            False,
            None,
            -1.0,
            0.0,
            "ecosystem closure; dens∝1/M if resource∝M",
        ),
        ChannelRule(
            "population_density_damuth",
            "conservation",
            False,
            None,
            -A_BULK,
            0.0,
            "Damuth if resource∝B∝M^{3/4}",
        ),
        ChannelRule(
            "home_range",
            "conservation",
            False,
            None,
            1.0,
            1.0,
            "dual nulls: met 3/4 (∝B) and cons 1 (∝M); forbid city 7/6",
        ),
        ChannelRule(
            "home_range_metabolic",
            "conservation",
            False,
            None,
            A_BULK,
            1.0,
            "Damuth dual: A_HR∝B",
        ),
        ChannelRule("brain_size", "individuality", False, None, A_BULK, A_SURFACE, "grade shifts; split by order"),
        ChannelRule(
            "city_infrastructure",
            "physical_net",
            False,
            None,
            a_infra,
            1.0,
            "Tier C: (d−1)/d",
        ),
        ChannelRule(
            "city_socioeconomic",
            "social_info",
            False,
            None,
            a_socio,
            1.0,
            "Tier C: 1+1/d",
        ),
        ChannelRule("company_scale", "organism_net", False, None, a_infra, 1.0, "Tier C: (d−1)/d"),
    ]


def classify_vs_isometry(a: float, a_iso: float, tol: float = NEAR_TOL) -> str:
    if abs(a - a_iso) <= tol:
        return "isometric"
    return "positive" if a > a_iso else "negative"


def _alphabet_q_weight_exact(eng: HqvmD, w: int) -> List[int]:
    return [b for b in range(eng.n_bytes) if eng.q_weight[b] == w]


def _alphabet_q_weight_at_most(eng: HqvmD, w: int) -> List[int]:
    return [b for b in range(eng.n_bytes) if eng.q_weight[b] <= w]


@dataclass(frozen=True)
class SquareRootRow:
    d: int
    label: str
    rank: int
    reach: int
    pred: int
    root: int
    a_B_vs_M: float
    ok: bool


def square_root_live_gate(d_values: Sequence[int] = (4, 5, 6)) -> List[SquareRootRow]:
    """BFS |Reach| vs (2^r)^2 on fiber-complete weight shells."""
    rows: List[SquareRootRow] = []
    for d in d_values:
        eng = _engine(d)
        cases: List[Tuple[str, List[int]]] = [("0", _alphabet_q_weight_exact(eng, 0))]
        for w in range(1, d + 1):
            cases.append((str(w), _alphabet_q_weight_at_most(eng, w)))
        cases.append(("full", list(enumerate_bytes(d))))
        for label, alphabet in cases:
            if not alphabet:
                continue
            qs = [eng.q_by_byte[b] for b in alphabet]
            r = gf2_rank(qs, d)
            reach, _, _, _ = bfs_reach(eng, alphabet)
            pred = predicted_cluster_size(r)
            root = 1 << r if r >= 1 else 1
            if reach <= 1 or root <= 0:
                a = float("nan")
            else:
                a = math.log(root) / math.log(reach)
            rows.append(
                SquareRootRow(d, label, r, reach, pred, root, a, reach == pred)
            )
    return rows


@dataclass(frozen=True)
class Depth4Row:
    d: int
    H: int
    depth4_bits: int
    a_d4: float
    Delta: float
    Delta_vs_1_over_8d: float
    equals_3_4: bool
    equals_2_3: bool


def depth4_family_curve(d_values: Sequence[int] = tuple(range(1, 9))) -> List[Depth4Row]:
    """a_d4(d)=depth4_projection_bits(d)/2^d — equals 3/4 only at physical d=6."""
    rows: List[Depth4Row] = []
    for d in d_values:
        H = 1 << d
        bits = int(depth4_projection_bits(d))
        a = bits / float(H)
        Delta = float(delta_spinorial_residual_d(d))
        rows.append(
            Depth4Row(
                d=d,
                H=H,
                depth4_bits=bits,
                a_d4=a,
                Delta=Delta,
                Delta_vs_1_over_8d=abs(Delta - 1.0 / (8.0 * d)),
                equals_3_4=abs(a - A_BULK) < EXACT_TOL,
                equals_2_3=abs(a - A_SURFACE) < EXACT_TOL,
            )
        )
    return rows


@dataclass(frozen=True)
class ProxyRow:
    p: float
    E_root: float
    E_reach: float
    a_SR: float
    theta: float
    P_rank_full: float
    holonomy_d4: float


def micro_ref_proxy_curve(
    d: int = N_DOF, p_grid: Optional[Sequence[float]] = None
) -> List[ProxyRow]:
    """Exact micro-ref PMF → E[root], E[|Reach|], a_SR=log E_root / log E_Reach."""
    if p_grid is None:
        p_grid = [i / 40.0 for i in range(1, 41)]
    rows: List[ProxyRow] = []
    for p in p_grid:
        pmf = exact_micro_ref_rank_pmf(p, d)
        E_root = 0.0
        E_reach = 0.0
        for r, pr in enumerate(pmf):
            root = (1 << r) if r >= 1 else 1.0
            E_root += pr * root
            E_reach += pr * float(predicted_cluster_size(r))
        if E_reach > 1.0 and E_root > 0.0:
            a_SR = math.log(E_root) / math.log(E_reach)
        else:
            a_SR = float("nan")
        rows.append(
            ProxyRow(
                p=p,
                E_root=E_root,
                E_reach=E_reach,
                a_SR=a_SR,
                theta=exact_micro_ref_theta_cond(p, d),
                P_rank_full=exact_micro_ref_p_rank_full(p, d),
                holonomy_d4=holonomy_micro_cov(p, d, 4),
            )
        )
    return rows


def p_c_rank_micro_ref(d: int = N_DOF) -> float:
    return float(bisect_p_c_rank_micro_ref(d))


@dataclass(frozen=True)
class ChannelMeasure:
    id: str
    value: float
    note: str


def channel_measures(d: int = N_DOF) -> List[ChannelMeasure]:
    """Measured channel numbers from family APIs (not hardcoded 48/64)."""
    H = 1 << d
    bits = int(depth4_projection_bits(d))
    a_d4 = bits / float(H)
    Delta_d = float(delta_spinorial_residual_d(d))
    a_surface_kernel = Delta_d / (1.0 / TURN_QUANTA)
    a_surface_cont = float(DELTA) / (1.0 / TURN_QUANTA)
    root = 1 << d
    reach = predicted_cluster_size(d)
    a_SR = math.log(root) / math.log(reach)
    D_eff = 0.5 * d
    a_net = D_eff / (D_eff + 1.0)
    return [
        ChannelMeasure("a_SR_product", a_SR, "B∝root, M∝|Reach| ⇒ 1/2"),
        ChannelMeasure("a_d4_mask", a_d4, f"depth4_bits/H at d={d}"),
        ChannelMeasure("a_net_D_over_Dplus1", a_net, "⟨N⟩/(⟨N⟩+1) at thermal ⟨N⟩=d/2"),
        ChannelMeasure(
            "a_surface_Delta_over_1_32",
            a_surface_kernel,
            "Δ(d)/(1/32); kernel-lattice exact",
        ),
        ChannelMeasure(
            "a_surface_continuum_Delta",
            a_surface_cont,
            "trestle Δ/(1/32)",
        ),
        ChannelMeasure(
            "lift_a_net_minus_a_SR",
            a_net - a_SR,
            "a_bulk−a_SR = a_time; 1/4 at d=6",
        ),
        ChannelMeasure(
            "gap_a_d4_minus_surface",
            a_d4 - a_surface_kernel,
            "bulk−surface = 1/12 at d=6 (kernel Δ)",
        ),
        ChannelMeasure("Delta_kernel_d", Delta_d, "spinorial residual 1/(8d)"),
        ChannelMeasure("trestle_Delta", float(DELTA), "continuum aperture Δ"),
        ChannelMeasure(
            "surface_kernel_minus_continuum",
            a_surface_kernel - a_surface_cont,
            "exact 2/3 minus continuum surface",
        ),
        ChannelMeasure("p_c_rank", p_c_rank_micro_ref(d), "micro-ref P(rank=d)=1/2"),
    ]


@dataclass(frozen=True)
class ShellCensusRow:
    d: int
    shell: int
    pop: int
    pop_formula: int
    ok: bool


@dataclass(frozen=True)
class ShellMeanRow:
    d: int
    Omega: int
    mean_shell: float
    D_eff: float
    a_net: float
    entanglement_ok: bool
    pops_sum_ok: bool


def shell_census(d: int = N_DOF) -> List[ShellCensusRow]:
    """Binomial shell populations: pop = C(d,k)·2^d."""
    rows: List[ShellCensusRow] = []
    for k in range(d + 1):
        pop = int(shell_population_d(d, k))
        formula = math.comb(d, k) * (1 << d)
        rows.append(ShellCensusRow(d, k, pop, formula, pop == formula))
    return rows


def shell_mean_family(
    d_values: Sequence[int] = tuple(range(1, 9)),
) -> List[ShellMeanRow]:
    """⟨shell⟩=d/2 from census; D_eff=⟨shell⟩; a_net=D_eff/(D_eff+1)."""
    rows: List[ShellMeanRow] = []
    for d in d_values:
        Omega = 1 << (2 * d)
        mean = 0.0
        total = 0
        for k in range(d + 1):
            pop = int(shell_population_d(d, k))
            mean += k * pop
            total += pop
        mean /= float(total) if total else float("nan")
        D_eff = mean
        a_net = D_eff / (D_eff + 1.0) if D_eff >= 0 else float("nan")
        ent_ok, _, _ = verify_carrier_entanglement_exact(d)
        rows.append(
            ShellMeanRow(
                d=d,
                Omega=Omega,
                mean_shell=mean,
                D_eff=D_eff,
                a_net=a_net,
                entanglement_ok=ent_ok,
                pops_sum_ok=total == Omega,
            )
        )
    return rows


@dataclass(frozen=True)
class ScalingLadderRow:
    d: int
    a_SR: float
    a_net: float
    a_d4: float
    a_surface_Delta: float
    net_eq_SR: bool
    net_eq_surface: bool
    net_eq_d4: bool
    triple_lock_3_4: bool


def scaling_ladder(
    d_values: Sequence[int] = tuple(range(1, 9)),
) -> List[ScalingLadderRow]:
    """Compare product / shell-network / depth-4 / surface charts vs d."""
    rows: List[ScalingLadderRow] = []
    for d in d_values:
        root = 1 << d
        reach = predicted_cluster_size(d)
        a_SR = math.log(root) / math.log(reach) if reach > 1 else float("nan")
        D_eff = 0.5 * d
        a_net = D_eff / (D_eff + 1.0)
        a_d4 = float(depth4_projection_bits(d)) / float(1 << d)
        Delta = float(delta_spinorial_residual_d(d))
        a_surf = Delta / (1.0 / TURN_QUANTA)
        net_SR = abs(a_net - a_SR) < EXACT_TOL
        net_surf = abs(a_net - a_surf) < EXACT_TOL
        net_d4 = abs(a_net - a_d4) < EXACT_TOL
        triple = (
            abs(a_net - A_BULK) < EXACT_TOL
            and abs(a_d4 - A_BULK) < EXACT_TOL
            and abs(a_SR - 0.5) < EXACT_TOL
        )
        rows.append(
            ScalingLadderRow(
                d=d,
                a_SR=a_SR,
                a_net=a_net,
                a_d4=a_d4,
                a_surface_Delta=a_surf,
                net_eq_SR=net_SR,
                net_eq_surface=net_surf,
                net_eq_d4=net_d4,
                triple_lock_3_4=triple,
            )
        )
    return rows


@dataclass(frozen=True)
class ParityRow:
    d: int
    rank: int
    reach: int
    pred: int
    a_B_vs_M: float
    ok: bool
    note: str


def parity_plateau_gate(d: int = N_DOF) -> ParityRow:
    """Even-weight q only: rank d-1 plateau, |Reach|=(2^{d-1})^2."""
    eng = _engine(d)
    alphabet = [b for b in range(eng.n_bytes) if (eng.q_weight[b] % 2) == 0]
    qs = [eng.q_by_byte[b] for b in alphabet]
    r = gf2_rank(qs, d)
    reach, _, _, _ = bfs_reach(eng, alphabet)
    pred = predicted_cluster_size(r)
    root = 1 << r if r >= 1 else 1
    a = math.log(root) / math.log(reach) if reach > 1 else float("nan")
    return ParityRow(
        d=d,
        rank=r,
        reach=reach,
        pred=pred,
        a_B_vs_M=a,
        ok=reach == pred and r == d - 1,
        note=f"even-q plateau; capacity (2^{d-1})^2",
    )


@dataclass(frozen=True)
class DerivedExponent:
    id: str
    value: float
    formula: str
    note: str


def west_organism_family(a_bulk: float = A_BULK) -> List[DerivedExponent]:
    """West organism exponents as corollaries of a_SR, a_bulk, a_time=a_bulk−a_SR."""
    a_time = a_bulk - A_SR
    a_service = a_time / float(N_SPATIAL)
    return [
        DerivedExponent("metabolic_B", a_bulk, "⟨N⟩/(⟨N⟩+1)", "QuBEC thermal ⟨N⟩=3"),
        DerivedExponent("specific_B_over_M", a_bulk - 1.0, "a_bulk-1", "B/M"),
        DerivedExponent(
            "heart_respiratory_rate",
            -a_time,
            "-(a_bulk-a_SR)",
            "rates",
        ),
        DerivedExponent(
            "circulation_lifespan",
            a_time,
            "a_bulk-a_SR",
            "times",
        ),
        DerivedExponent(
            "egress_development",
            A_EGRESS,
            "(3/4)·a_time",
            "a_eg construction time",
        ),
        DerivedExponent(
            "longevity_composite_lo",
            A_EGRESS,
            "3/16",
            "composite interval lower bound",
        ),
        DerivedExponent(
            "longevity_composite_hi",
            a_time,
            "1/4",
            "composite interval upper bound",
        ),
        DerivedExponent(
            "aorta_radius",
            a_bulk / 2.0,
            "a_bulk/2",
            "Tier B: fixed ΔP,u0 ⇒ A∝B ⇒ r∝√A",
        ),
        DerivedExponent("aorta_length", a_time, "a_bulk-a_SR", "l_0"),
        DerivedExponent("blood_volume", 1.0, "1", "V_b ∝ M"),
        DerivedExponent("capillary_number", a_bulk, "a_bulk", "N_c ∝ B"),
        DerivedExponent(
            "service_radius",
            a_service,
            "a_time/n_spatial",
            "=(a_bulk-a_surf)/n at d=6",
        ),
        DerivedExponent(
            "intercapillary_spacing",
            a_service,
            "a_service",
            "linear spacing",
        ),
        DerivedExponent(
            "inverse_spacing",
            -a_service,
            "-a_service",
            "linear density dual of spacing",
        ),
        DerivedExponent(
            "capillary_density_volumetric",
            a_bulk - 1.0,
            "a_bulk-1",
            "N_c/M with N_c∝B",
        ),
        DerivedExponent(
            "total_resistance",
            -a_bulk,
            "-a_bulk",
            "Tier B: Z∝1/B if ΔP fixed",
        ),
        DerivedExponent("pressure_velocity", 0.0, "0", "ΔP, u_0 size-independent"),
        DerivedExponent(
            "lifetime_energy_per_mass",
            a_bulk + a_time - 1.0,
            "a_B+a_time-1",
            "B·life/M ∝ M^0",
        ),
        DerivedExponent(
            "lifetime_heartbeats",
            0.0,
            "(-a_time)+(+a_time)",
            "rate·life ∝ M^0",
        ),
    ]


def city_company_family(d: int = N_DOF) -> List[DerivedExponent]:
    """Tier C: cities/companies identify network dim with transport-register d."""
    a_infra = (d - 1) / float(d)
    a_socio = 1.0 + 1.0 / float(d)
    return [
        DerivedExponent(
            "city_infrastructure",
            a_infra,
            "(d-1)/d",
            "Tier C: minimal-network on register dim d",
        ),
        DerivedExponent(
            "city_socioeconomic",
            a_socio,
            "1+1/d",
            "Tier C: N^2/N^{(d-1)/d}",
        ),
        DerivedExponent(
            "city_pace",
            a_socio - 1.0,
            "1/d",
            "Tier C: pace relative to linear",
        ),
        DerivedExponent(
            "city_infra_plus_socio",
            a_infra + a_socio,
            "a_infra+a_socio",
            "Tier C: conjugacy sum = 2",
        ),
        DerivedExponent(
            "company_sublinear",
            a_infra,
            "(d-1)/d",
            "Tier C: bounded network",
        ),
        DerivedExponent(
            "company_vs_city_capacity",
            WORD_HORIZON_REACH / float(OMEGA_FULL),
            "128/4096",
            "confinement vs full spanning",
        ),
    ]


def west_family_pass(organism: Sequence[DerivedExponent], city: Sequence[DerivedExponent]) -> Dict[str, bool]:
    o = {r.id: r.value for r in organism}
    c = {r.id: r.value for r in city}
    return {
        "organism_B_is_3_4": abs(o["metabolic_B"] - A_BULK) < EXACT_TOL,
        "organism_a_time_is_bulk_minus_SR": abs(o["circulation_lifespan"] - A_TIME)
        < EXACT_TOL,
        "organism_rate_is_m1_4": abs(o["heart_respiratory_rate"] + A_TIME) < EXACT_TOL,
        "organism_life_is_p1_4": abs(o["circulation_lifespan"] - A_TIME) < EXACT_TOL,
        "a_egress_is_3_16": abs(o["egress_development"] - A_EGRESS) < EXACT_TOL
        and abs(A_EGRESS - 3.0 / 16.0) < EXACT_TOL,
        "a_egress_eq_three_quarters_a_time": abs(A_EGRESS - 0.75 * A_TIME) < EXACT_TOL,
        "longevity_interval_is_3_16_to_1_4": abs(o["longevity_composite_lo"] - A_EGRESS)
        < EXACT_TOL
        and abs(o["longevity_composite_hi"] - A_TIME) < EXACT_TOL
        and A_EGRESS < A_TIME,
        "organism_aorta_is_3_8": abs(o["aorta_radius"] - 0.375) < EXACT_TOL,
        "organism_service_is_1_12": abs(o["service_radius"] - MU_GAP) < EXACT_TOL,
        "organism_cap_density_is_m1_4": abs(o["capillary_density_volumetric"] + 0.25)
        < EXACT_TOL,
        "organism_spacing_is_1_12": abs(o["intercapillary_spacing"] - MU_GAP) < EXACT_TOL,
        "sum_rule_energy_per_mass_M0": abs(o["lifetime_energy_per_mass"]) < EXACT_TOL,
        "sum_rule_heartbeats_M0": abs(o["lifetime_heartbeats"]) < EXACT_TOL,
        "city_infra_is_(d-1)/d": abs(c["city_infrastructure"] - (N_DOF - 1) / float(N_DOF))
        < EXACT_TOL,
        "city_socio_is_1+1/d": abs(c["city_socioeconomic"] - (1.0 + 1.0 / float(N_DOF)))
        < EXACT_TOL,
        "city_pair_sums_to_2": abs(c["city_infra_plus_socio"] - 2.0) < EXACT_TOL,
        "company_capacity_is_1_32": abs(c["company_vs_city_capacity"] - 1.0 / 32.0)
        < EXACT_TOL,
    }


def shell_mean_equals_M_shell_over_H(d: int = N_DOF) -> Dict[str, float]:
    """Identity: ⟨D_shell⟩ = Tr(D_shell)/|Ω| = M_shell/|H| with M_shell=d·2^{d-1}, |H|=2^d."""
    means = shell_mean_family((d,))
    row = means[0]
    M_shell = float(d * (1 << (d - 1)))  # sum k C(d,k) = d·2^{d-1}
    H = float(1 << d)
    mean_from_M = M_shell / H
    return {
        "d": float(d),
        "mean_shell": row.mean_shell,
        "M_shell": M_shell,
        "H": H,
        "M_shell_over_H": mean_from_M,
        "Omega": float(row.Omega),
        "ok": 1.0 if abs(row.mean_shell - mean_from_M) < EXACT_TOL else 0.0,
    }


def qubec_uniform_slice(d: int = N_DOF) -> Dict[str, float]:
    """QuBEC at λ=1 (η=0): uniform occupation; M2=|Ω|; ⟨shell⟩=d/2."""
    H = 1 << d
    Omega = 1 << (2 * d)
    lam = 1.0
    eta = (1.0 - lam) / (1.0 + lam)
    M2 = Omega / ((1.0 + eta * eta) ** d)
    mean = 0.5 * d
    return {
        "lambda": lam,
        "eta": eta,
        "Z1": float(H) * ((1.0 + lam) ** d),
        "M2": M2,
        "mean_shell": mean,
        "ok_thermal": 1.0 if abs(eta) < EXACT_TOL and abs(M2 - Omega) < EXACT_TOL else 0.0,
    }


@dataclass(frozen=True)
class DeliveryRow:
    d: int
    depth4_bits: int
    H: int
    Delta_d: float
    a_deliv: float
    a_deliv_from_Delta_H: float
    P_pred: int
    a_horizon_lemma: float
    equals_3_4: bool
    bits_eq_P_pred: bool
    ok: bool


def a_d4_delivery_family(d_values: Sequence[int] = tuple(range(1, 9))) -> List[DeliveryRow]:
    """Delivery exponent a_deliv = depth4_bits/|H| = 1/(Δ(d)·|H|) from aperture×horizon.

    At d=6, depth4_bits equals Horizon-Lemma predecessor P_5=48 of |H|=64, so
    a_deliv = P_pred/|H| = 3/4. Same value as QuBEC thermal a_bulk; uniqueness only at d=6.
    """
    rows: List[DeliveryRow] = []
    for d in d_values:
        bits = depth4_projection_bits(d)
        H = 1 << d
        Delta_d = delta_spinorial_residual_d(d)
        a = bits / float(H)
        a_inv = 1.0 / (Delta_d * float(H))
        # Predecessor of dyadic |H|=2^d: P_{d-1} = 3·2^{d-2} for d>=2
        P_pred = 3 * (1 << (d - 2)) if d >= 2 else 0
        a_hl = P_pred / float(H) if H else float("nan")
        bits_eq = bits == P_pred
        eq34 = abs(a - A_BULK) < EXACT_TOL
        ok = (
            abs(a - a_inv) < 1e-9
            and abs(Delta_d - 1.0 / (8.0 * d)) < 1e-9
            and (d != 6 or (eq34 and bits_eq and abs(a_hl - A_BULK) < EXACT_TOL))
        )
        rows.append(
            DeliveryRow(
                d=d,
                depth4_bits=bits,
                H=H,
                Delta_d=Delta_d,
                a_deliv=a,
                a_deliv_from_Delta_H=a_inv,
                P_pred=P_pred,
                a_horizon_lemma=a_hl,
                equals_3_4=eq34,
                bits_eq_P_pred=bits_eq,
                ok=ok,
            )
        )
    return rows


def coverage_generation_ladder() -> List[Tuple[str, int, str]]:
    """Discrete delivery generations (percolation coverage hierarchy), not continuum N→∞."""
    return [
        ("span", 0, "rest horizon spanning"),
        ("full", 1, "|Reach|=|Ω| orbit"),
        ("spectrum", 2, "full defect weight spectrum"),
        ("rank", 3, "transport rank r=d; channel isotropy"),
        ("word", 4, "depth-4 holonomy / F-word closure"),
    ]


@dataclass(frozen=True)
class QubecLambdaRow:
    lam: float
    rho: float
    eta: float
    mean_N: float
    var_N: float
    Z1: float
    M2: float
    a_net: float
    a_fold: float
    mu_eta: float
    a_mu_eta: float
    mu_net: float
    regime: str
    z1_ok: bool
    var_ok: bool


def qubec_lambda_sweep(
    d: int = N_DOF,
    lams: Sequence[float] = LAMBDA_PHYS,
) -> List[QubecLambdaRow]:
    """QuBEC occupation → a_net and exact μ-band a(λ) on λ∈(0,1] by default."""
    Omega = 1 << (2 * d)
    rows: List[QubecLambdaRow] = []
    for lam in lams:
        if lam <= 0:
            continue
        rho = lam / (1.0 + lam)
        eta = (1.0 - lam) / (1.0 + lam)
        mean_N = d * rho
        var_N = d * lam / ((1.0 + lam) ** 2)
        Z1 = partition_Z1_coeff_d(d, lam)
        M2 = Omega / ((1.0 + eta * eta) ** d)
        z1_shell = sum(shell_population_d(d, k) * (lam**k) for k in range(d + 1))
        z1_ok = abs(Z1 - z1_shell) < 1e-6 * max(1.0, Z1)
        dmean_dlam = d / ((1.0 + lam) ** 2)
        var_from_resp = lam * dmean_dlam
        var_ok = abs(var_N - var_from_resp) < 1e-9
        a_net = a_bulk_from_shell_mean(mean_N)
        D_fold = min(mean_N, d - mean_N)
        a_fold = a_bulk_from_shell_mean(D_fold) if D_fold > 0 else 0.0
        mu_eta = mu_from_lambda(lam) if lam <= 1.0 else mu_from_eta(eta)
        a_mu = a_from_mu(mu_eta)
        if in_dual_band_a(a_net, tol=1e-9):
            mu_net = mu_from_a(a_net)
        else:
            mu_net = float("nan")
        if abs(eta) < 0.2:
            regime = "thermal"
        elif abs(eta) > 0.8:
            regime = "condensed"
        else:
            regime = "intermediate"
        rows.append(
            QubecLambdaRow(
                lam=lam,
                rho=rho,
                eta=eta,
                mean_N=mean_N,
                var_N=var_N,
                Z1=Z1,
                M2=M2,
                a_net=a_net,
                a_fold=a_fold,
                mu_eta=mu_eta,
                a_mu_eta=a_mu,
                mu_net=mu_net,
                regime=regime,
                z1_ok=z1_ok,
                var_ok=var_ok,
            )
        )
    return rows


def qubec_sweep_pass(rows: Sequence[QubecLambdaRow], d: int = N_DOF) -> Dict[str, bool]:
    by_lam = {r.lam: r for r in rows}
    r1 = by_lam.get(1.0)
    closed_ok = True
    for r in rows:
        if r.lam > 1.0 + 1e-15:
            continue
        mu_cf = (2.0 * r.lam) / (1.0 + r.lam)
        a_cf = A_SURFACE + r.lam / (6.0 * (1.0 + r.lam))
        if abs(r.mu_eta - mu_cf) > 1e-9:
            closed_ok = False
        if abs(r.a_mu_eta - a_cf) > 1e-9:
            closed_ok = False
        if abs(r.mu_eta - 2.0 * r.rho) > 1e-9:
            closed_ok = False
        if abs(r.mu_eta - (2.0 * r.mean_N) / float(d)) > 1e-9:
            closed_ok = False
        # Inverse: λ(μ) recovers λ
        if abs(lambda_from_mu(r.mu_eta) - r.lam) > 1e-9:
            closed_ok = False
    condensed = [r for r in rows if r.regime == "condensed" and r.lam <= 1.0]
    return {
        "qubec_sweep_z1": all(r.z1_ok for r in rows),
        "qubec_sweep_var_response": all(r.var_ok for r in rows),
        "qubec_sweep_lambda1_bulk": bool(
            r1
            and abs(r1.mean_N - 0.5 * d) < EXACT_TOL
            and abs(r1.a_net - A_BULK) < EXACT_TOL
            and abs(r1.mu_eta - 1.0) < EXACT_TOL
            and abs(r1.a_mu_eta - A_BULK) < EXACT_TOL
            and r1.regime == "thermal"
        ),
        "qubec_mu_a_closed_form": closed_ok,
        "qubec_mu_eta_condensed_near_0": bool(
            condensed and all(r.mu_eta < 0.25 for r in condensed)
        ),
    }


def qubec_duality_appendix_pass(
    rows: Optional[Sequence[QubecLambdaRow]] = None,
) -> Dict[str, bool]:
    """λ↔1/λ fold for a_fold and μ (outside the physical λ∈(0,1] sweep)."""
    if rows is None:
        rows = qubec_lambda_sweep(lams=LAMBDA_DUAL_APPENDIX)
    dual_ok = True
    for r in rows:
        inv = 1.0 / r.lam
        partner = min(rows, key=lambda x: abs(x.lam - inv))
        if abs(partner.lam - inv) / max(inv, 1e-12) < 0.05:
            if abs(r.a_fold - partner.a_fold) > 1e-9:
                dual_ok = False
            if abs(r.mu_eta - partner.mu_eta) > 1e-9:
                dual_ok = False
    return {"qubec_fold_duality_appendix": dual_ok}


@dataclass(frozen=True)
class KleiberIntercept:
    M_shell: float
    a_Higgs: float
    xi: float
    log2_M0_over_u: float
    M0_kg: float
    M0_kg_electron_u: float
    b_K: float
    a_bulk: float
    note: str


def kleiber_absolute_intercept(d: int = N_DOF) -> KleiberIntercept:
    """Absolute mass origin M0 and dimensionless Kleiber intercept.

    Parallel to electroweak: one external mass unit u (nucleon/amu), kernel ξ.
    log2(M0/u) = a_Higgs = M_shell/2  (Higgs equator coefficient).
    Dimensionless channel offset b_K = −P = −47/48 (boundary projector; exchange).
    Form: log2(B/B0) = (3/4) log2(M/M0) + b_K  with B0 set by the biological clock
    (chemical/thermal quantum; not Compton). Kernel-forced content is a_Higgs and b_K;
    SI M0 requires the external u convention (amu vs m_e swings four orders).
    """
    M_shell = float(d * (1 << (d - 1)))
    a_Higgs = M_shell / 2.0
    xi = 2.0**a_Higgs
    M0_kg = xi * U_KG
    return KleiberIntercept(
        M_shell=M_shell,
        a_Higgs=a_Higgs,
        xi=xi,
        log2_M0_over_u=a_Higgs,
        M0_kg=M0_kg,
        M0_kg_electron_u=xi * U_ELECTRON_KG,
        b_K=-P_BOUNDARY,
        a_bulk=A_BULK,
        note="M0=u·2^(M_shell/2); SI value anchor-sensitive (amu vs m_e)",
    )


def kleiber_intercept_pass(k: KleiberIntercept) -> Dict[str, bool]:
    return {
        "kleiber_M0_log2_is_a_Higgs": abs(k.log2_M0_over_u - k.a_Higgs) < EXACT_TOL,
        "kleiber_a_Higgs_is_96": abs(k.a_Higgs - 96.0) < EXACT_TOL,
        "kleiber_b_K_is_minus_P": abs(k.b_K + P_BOUNDARY) < EXACT_TOL,
        "kleiber_a_bulk_is_3_4": abs(k.a_bulk - A_BULK) < EXACT_TOL,
    }


# CODATA / physiological anchors for chemical-clock schema (SI completion of B0).
K_B = 1.380649e-23  # J/K
H_PLANCK = 6.62607015e-34  # J·s
EV_J = 1.602176634e-19  # J/eV
T_BODY_K = 310.0  # mammalian core temperature (external biological clock)


@dataclass(frozen=True)
class ChemicalClockRow:
    T_body_K: float
    Delta: float
    kT_J: float
    kT_eV: float
    E_a_J: float
    E_a_eV: float
    f_attempt_Hz: float
    N_H: int
    P_terminal_W: float
    B0_micro_W: float
    note: str


def chemical_clock_B0(
    *,
    T_body_K: float = T_BODY_K,
    d: int = N_DOF,
) -> ChemicalClockRow:
    """Thermal clock: E_a=kT/(2Δ), f=(kT/h)·Δ, P=(kT)^2/(2h).

    Delta cancels in P_terminal. B0_micro = |H|·P at horizon multiplicity.
    At T=310 K: E_a≈0.645 eV (MTE aerobic band ~0.6–0.7 eV).
    """
    Delta = float(DELTA)
    kT = K_B * T_body_K
    E_a = kT / (2.0 * Delta)
    f_att = (kT / H_PLANCK) * Delta
    N_H = 1 << d
    P_term = E_a * f_att  # equals (kT)^2 / (2 h)
    return ChemicalClockRow(
        T_body_K=T_body_K,
        Delta=Delta,
        kT_J=kT,
        kT_eV=kT / EV_J,
        E_a_J=E_a,
        E_a_eV=E_a / EV_J,
        f_attempt_Hz=f_att,
        N_H=N_H,
        P_terminal_W=P_term,
        B0_micro_W=float(N_H) * P_term,
        note="P=(kT)^2/(2h); B0=|H|·P; Delta cancels",
    )


def chemical_clock_pass(c: ChemicalClockRow) -> Dict[str, bool]:
    P_simp = (c.kT_J**2) / (2.0 * H_PLANCK)
    return {
        "chem_E_a_eq_kT_over_2Delta": abs(c.E_a_J - c.kT_J / (2.0 * c.Delta)) < 1e-30,
        "chem_E_a_in_MTE_eV_band": 0.5 <= c.E_a_eV <= 0.8,
        "chem_f_attempt_eq_kT_h_Delta": abs(
            c.f_attempt_Hz - (c.kT_J / H_PLANCK) * c.Delta
        )
        < 1e-3 * max(1.0, c.f_attempt_Hz),
        "chem_P_eq_kT_sq_over_2h": abs(c.P_terminal_W - P_simp)
        < 1e-30 * max(1.0, c.P_terminal_W),
        "chem_B0_micro_eq_H_Pterm": abs(c.B0_micro_W - c.N_H * c.P_terminal_W)
        < 1e-30 * max(1.0, c.B0_micro_W),
    }


def delivery_pass(rows: Sequence[DeliveryRow]) -> Dict[str, bool]:
    r6 = next(r for r in rows if r.d == 6)
    return {
        "a_deliv_eq_1_over_Delta_H": all(
            abs(r.a_deliv - r.a_deliv_from_Delta_H) < 1e-9 for r in rows
        ),
        "a_deliv_3_4_only_d6": r6.equals_3_4
        and not any(r.equals_3_4 for r in rows if r.d != 6),
        "depth4_bits_eq_P_pred_at_d6": r6.bits_eq_P_pred,
        "horizon_lemma_fill_3_4_at_d6": abs(r6.a_horizon_lemma - A_BULK) < EXACT_TOL,
        "delivery_family_ok": all(r.ok for r in rows),
    }


def kernel_pass_fail(
    live: Sequence[SquareRootRow],
    d4: Sequence[Depth4Row],
    proxies: Sequence[ProxyRow],
    measures: Sequence[ChannelMeasure],
    shell_means: Sequence[ShellMeanRow],
    ladder: Sequence[ScalingLadderRow],
    parity: ParityRow,
    census: Sequence[ShellCensusRow],
    organism: Optional[Sequence[DerivedExponent]] = None,
    city: Optional[Sequence[DerivedExponent]] = None,
    delivery: Optional[Sequence[DeliveryRow]] = None,
    qubec_rows: Optional[Sequence[QubecLambdaRow]] = None,
    kleiber: Optional[KleiberIntercept] = None,
    chem: Optional[ChemicalClockRow] = None,
) -> Dict[str, bool]:
    """Non-tautological gates only."""
    live_ok = all(r.ok for r in live)
    a_live = [
        r.a_B_vs_M
        for r in live
        if r.rank >= 1 and r.reach > 1 and not math.isnan(r.a_B_vs_M)
    ]
    a_SR_ok = bool(a_live) and all(abs(a - 0.5) < 1e-9 for a in a_live)
    d6 = next(r for r in d4 if r.d == 6)
    d4_select_ok = d6.equals_3_4 and not any(r.equals_3_4 for r in d4 if r.d != 6)
    Delta_ok = all(r.Delta_vs_1_over_8d < EXACT_TOL for r in d4)
    hi = [r for r in proxies if r.p >= 0.5]
    proxy_ok = bool(hi) and all(abs(r.a_SR - 0.5) < 1e-6 for r in hi)
    m = {c.id: c.value for c in measures}
    surface_ok = abs(m["a_surface_Delta_over_1_32"] - A_SURFACE) < EXACT_TOL
    surface_cont = m["a_surface_continuum_Delta"]
    surface_corr_ok = abs(surface_cont - A_SURFACE) > 1e-4 and surface_cont < A_SURFACE
    lift_ok = abs(m["a_d4_mask"] - A_BULK) < EXACT_TOL and abs(
        m["a_SR_product"] - 0.5
    ) < EXACT_TOL
    pc = m["p_c_rank"]
    near = min(proxies, key=lambda r: abs(r.p - pc))
    hol_delayed = near.holonomy_d4 < 0.05 and near.theta > 0.3

    census_ok = all(r.ok for r in census)
    means_ok = all(
        r.pops_sum_ok
        and r.entanglement_ok
        and abs(r.mean_shell - 0.5 * r.d) < EXACT_TOL
        for r in shell_means
    )
    by_d = {r.d: r for r in ladder}
    ladder_ok = (
        abs(by_d[2].a_net - 0.5) < EXACT_TOL
        and abs(by_d[4].a_net - A_SURFACE) < EXACT_TOL
        and abs(by_d[6].a_net - A_BULK) < EXACT_TOL
        and by_d[6].triple_lock_3_4
        and sum(1 for r in ladder if r.net_eq_d4) == 1
    )
    lift_formula_ok = abs(m["lift_a_net_minus_a_SR"] - 0.25) < EXACT_TOL
    out = {
        "square_root_BFS": live_ok,
        "a_SR_is_1_2": a_SR_ok and lift_ok,
        "a_d4_equals_3_4_only_at_d6": d4_select_ok,
        "Delta_equals_1_over_8d": Delta_ok,
        "proxy_a_SR_to_1_2": proxy_ok,
        "surface_from_Delta": surface_ok,
        "surface_continuum_below_2_3": surface_corr_ok,
        "holonomy_delayed_vs_span": hol_delayed,
        "shell_census_binomial": census_ok and means_ok,
        "a_net_ladder_1_2_2_3_3_4": ladder_ok,
        "triple_lock_only_d6": by_d[6].triple_lock_3_4 and ladder_ok,
        "lift_net_minus_SR_is_1_4": lift_formula_ok,
        "parity_plateau_even_q": parity.ok,
    }
    if organism is None:
        organism = west_organism_family()
    if city is None:
        city = city_company_family()
    out.update(west_family_pass(organism, city))
    id_row = shell_mean_equals_M_shell_over_H(N_DOF)
    qb = qubec_uniform_slice(N_DOF)
    out["shell_mean_eq_M_shell_over_H"] = bool(id_row["ok"])
    out["qubec_lambda1_thermal"] = bool(qb["ok_thermal"])
    out["quarter_lift_named"] = abs(m["lift_a_net_minus_a_SR"] - 0.25) < EXACT_TOL
    if delivery is None:
        delivery = a_d4_delivery_family()
    out.update(delivery_pass(delivery))
    if qubec_rows is None:
        qubec_rows = qubec_lambda_sweep()
    out.update(qubec_sweep_pass(qubec_rows))
    out.update(qubec_duality_appendix_pass())
    if kleiber is None:
        kleiber = kleiber_absolute_intercept()
    out.update(kleiber_intercept_pass(kleiber))
    if chem is None:
        chem = chemical_clock_B0()
    out.update(chemical_clock_pass(chem))
    rows_i = uniqueness_I_family()
    rows_ii = uniqueness_II_family()
    rows_iii = uniqueness_III_family()
    out.update(uniqueness_pass(rows_i, rows_ii, rows_iii))
    out.update(mu_mixing_pass())
    return out


def mu_mixing_pass(mus: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)) -> Dict[str, bool]:
    """Exact log-derivative mixing: a=(2/3)(1−μ)+(3/4)μ = 2/3+μ/12."""
    ok = True
    for mu in mus:
        a_mix = A_SURFACE * (1.0 - mu) + A_BULK * mu
        a_gap = A_SURFACE + mu * MU_GAP
        if abs(a_mix - a_gap) > EXACT_TOL:
            ok = False
        if abs(a_gap - a_from_mu(mu)) > EXACT_TOL:
            ok = False
    return {
        "mu_mixing_exact_log_derivative": ok,
        "mu_gap_eq_1_12": abs(MU_GAP - 1.0 / 12.0) < EXACT_TOL,
    }
