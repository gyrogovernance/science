#!/usr/bin/env python3
"""
CGM Compact Physics - Core

Data model, constants, and coordinate machinery for the
physics-facing aperture-coordinate report.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
import math
from typing import Callable, Iterable, List, Sequence, Tuple

try:
    import scipy.constants as sc

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


M_A = 1.0 / (2.0 * math.sqrt(2.0 * math.pi))
# BU monodromy defect: verified non-Clifford rotation angle from kernel tests
# (aQPU Tests Report 1, Part 10; Physics Tests Report, Part 9)
DELTA_BU = 0.195342176580
RHO = DELTA_BU / M_A
DELTA = 1.0 - RHO
E_EW_GEV = 246.22
F_CS_133_HZ = 9_192_631_770.0
APERTURE_FRAME = 48.0
KERNEL_APERTURE_Q256 = 5.0 / 256.0
KERNEL_BYTE_APERTURE = KERNEL_APERTURE_Q256

PHI_SU2 = 2.0 * math.acos((1.0 + 2.0 * math.sqrt(2.0)) / 4.0)
MONODROMY_BU = DELTA_BU
# sigma (SU2_RESIDUAL): Berry-type phase. Geometric SU(2) commutator angle minus
# three monodromy steps (dynamical), normalized by total aperture m_a.
SU2_RESIDUAL = (PHI_SU2 - 3.0 * MONODROMY_BU) / M_A
LAMBDA_0 = DELTA / math.sqrt(5.0)
GYROSCOPIC_COUPLING = LAMBDA_0
Q_G = 4.0 * math.pi
ALPHA_SURPLUS = 1.0 - RHO**4
MONODROMY_PER_STAGE = DELTA_BU / 2.0
STAGE_THRESHOLD_PI_OVER_TWO = math.pi / 2.0
STAGE_THRESHOLD_INV_SQRT2 = 1.0 / math.sqrt(2.0)
STAGE_THRESHOLD_PI_OVER_FOUR = math.pi / 4.0

CODE_C1 = comb(6, 1)
CODE_C2 = comb(6, 2)
CODE_C3 = comb(6, 3)

Z_H_OFFSET = 1 + CODE_C1 + CODE_C2
W_Z_OFFSET = CODE_C2 - CODE_C1
W_Z_APERTURE_COEFF = CODE_C3 / 2.0
MUON_EQUATOR_COEFF = CODE_C3
HORIZON_CARDINALITY = 64.0
UV_IR_GYRATION_SQUARED = (2.0 * math.pi) ** 2
ALPHA_GEOMETRIC = (DELTA_BU**4) / M_A

# CS matter channel: positive D^2 from kernel density projector (not gauge porosity).
TOP_MATTER_DENSITY_Q = 0.25

APERTURE_FRAME_LANDMARKS = (0.0, 12.0, 16.0, 32.0, 48.0)

EW_TARGET_NAMES = (
    "Top quark mass energy",
    "Z boson mass energy",
    "Bottom quark mass energy",
    "Charm quark mass energy",
    "Strange quark mass energy",
    "Higgs mass energy",
    "W boson mass energy",
    "Tau mass energy",
    "Muon mass energy",
    "Electron mass energy",
)


@dataclass(frozen=True)
class FundamentalConstants:
    c: float
    h: float
    hbar: float
    G: float
    e: float
    k_B: float


@dataclass(frozen=True)
class Observable:
    name: str
    family: str
    dimension: str
    value: float
    unit: str
    source: str
    cgm_stage: CGMStage | None = None
    sector: SectorType | None = None


@dataclass(frozen=True)
class CoordinateResult:
    name: str
    family: str
    dimension: str
    value: float
    unit: str
    source: str
    coordinate: float
    residue: float
    nearest: float
    error: float


@dataclass(frozen=True)
class ScaleReference:
    time_s: float
    length_m: float
    energy_GeV: float
    frequency_Hz: float


@dataclass(frozen=True)
class CGMStage:
    name: str
    force: str
    depth: float
    base_gyration: bool
    rotational: bool
    balance: bool


@dataclass(frozen=True)
class SectorType:
    name: str
    has_d2_correction: bool
    coordinate_law_order: int


CS = CGMStage("CS", "Strong", 1.0, False, False, False)
CS_UNA = CGMStage("CS->UNA", "VEV", 1.5, True, False, False)
UNA = CGMStage("UNA", "Weak", 2.0, True, True, False)
UNA_ONA = CGMStage("UNA->ONA", "Charged", 2.5, True, True, True)
ONA = CGMStage("ONA", "EM", 3.0, True, True, True)
BU = CGMStage("BU", "Gravity", 4.0, False, False, False)

# Stage action ratios from geometric unit analysis:
# E_UNA/E_CS = 2/(π√2), E_ONA/E_CS = 1/2, E_BU/E_CS = 2m_a/π.
STAGE_ACTION_TO_CS = {
    CS.name: 1.0,
    CS_UNA.name: 2.0 / (math.pi * STAGE_THRESHOLD_INV_SQRT2),
    UNA.name: 2.0 / (math.pi * STAGE_THRESHOLD_INV_SQRT2),
    UNA_ONA.name: 1.0 / 2.0,
    ONA.name: 1.0 / 2.0,
    BU.name: 2.0 * M_A / math.pi,
}


def stage_action_ratio_to_cs(stage: CGMStage | None) -> float:
    """Energy-ratio weight for a CGM stage relative to CS."""
    if stage is None:
        return 1.0
    return STAGE_ACTION_TO_CS.get(stage.name, 1.0)


def optical_conjugacy_seed_log2(stage: CGMStage | None) -> float:
    """Conjugacy-derived seed log2 depth before aperture transport.

    Uses the geometric relation for stage action and CGM aperture conjugacy:
      E_i^UV * E_i^IR = (E_CS * E_EW) / (4π^2)
    """
    ratio = stage_action_ratio_to_cs(stage)
    # With UV_IR_GYRATION_SQUARED = (2π)^2 = 4π^2 and E_i^UV / E_CS = ratio,
    # the conjugacy pair is fixed by
    #   E_i^UV * E_i^IR = E_CS * E_EW / (4π^2).
    # The compact seed uses the normalized geometric correction at this stage.
    return -1.0 + math.log2(1.0 / ratio)

GAUGE_SECTOR = SectorType("gauge", True, 2)
MATTER_SECTOR = SectorType("matter", False, 1)


@dataclass(frozen=True)
class DeltaBacksolve:
    source: str
    equation: str
    delta_back: float

    @property
    def delta_err(self) -> float:
        return self.delta_back - DELTA

    @property
    def err_over_delta_sq(self) -> float:
        return self.delta_err / (DELTA * DELTA)


@dataclass(frozen=True)
class CompactAlgebra:
    delta: float
    epsilon: float
    eta: float
    m_a: float
    omega: float
    kappa: float
    sigma: float
    d_h: float
    c1: int
    c2: int
    c3: int
    m1: int


@dataclass(frozen=True)
class ClosureDecomposition:
    temporal_depth: float
    spatial_mutation: float
    closure_ratio: float


def d2_coefficient_from_stage(
    stage: CGMStage | None,
    algebra: CompactAlgebra | None = None,
) -> float:
    """
    D^2 coefficient from CGM gyration projection at a stage.
    CS and BU are closure-bound and have no net quadratic term.
    """
    if stage is None:
        return 0.0
    if algebra is None:
        algebra = compact_algebra()
    base = -algebra.m1 / 8.0 if stage.base_gyration else 0.0
    rotational = algebra.c1 / 4.0 if stage.rotational else 0.0
    balance = -algebra.c3 / 2.0 if stage.balance else 0.0
    return base + rotational + balance


def k4_gyroscopic_charge(
    stage: CGMStage | None,
    code_coeff: int = CODE_C1,
    gyro_factor: float = 0.5,
) -> tuple[float, float]:
    """K4 gyroscopic charges (p, q) from K4 stage flags.

    p = 1 + (-C1 / 2) * base + (C1 / 4) * rot + (4 g) * bal
    q = 5/4 - (4 g) * rot - (2 g) * bal

    The two channels are closed by stage flags on CS->UNA->UNA->ONA.
    """
    if stage is None:
        return 0.0, 0.0
    base = float(stage.base_gyration)
    rot = float(stage.rotational)
    bal = float(stage.balance)
    g = gyro_factor
    p = 1.0 + (-code_coeff / 2.0) * base + (code_coeff / 4.0) * rot + (4.0 * g) * bal
    q = 5.0 / 4.0 - (4.0 * g) * rot - (2.0 * g) * bal
    return p, q


def gyroscopic_phase_correction(
    stage: CGMStage | None,
    delta: float = DELTA,
    gyroscopic_coupling: float = GYROSCOPIC_COUPLING,
) -> float:
    """D^3 correction to log2 depth from the gyroscopic coupling."""
    if stage is None:
        return 0.0
    p, _q = k4_gyroscopic_charge(stage)
    return p * gyroscopic_coupling * delta * delta


def gyroscopic_closure_correction(
    stage: CGMStage | None,
    delta: float = DELTA,
) -> float:
    """D^4 correction to log2 depth from K4 gyroscopic closure."""
    if stage is None:
        return 0.0
    _p, q = k4_gyroscopic_charge(stage)
    return q * delta**4


def lepton_horizon_residual_r(lepton: str, m_shell: float) -> float:
    """Horizon remainder r from M_shell code-weight ratios (tau, mu, e)."""
    if lepton == "tau":
        return m_shell / 8.0
    if lepton == "mu":
        return m_shell / 8.0 + m_shell / 48.0
    if lepton == "e":
        return m_shell / 8.0 - m_shell / 24.0
    raise ValueError(f"Unknown lepton horizon tag: {lepton!r}")


def lepton_coordinate(k: int, lepton: str, *, m_shell: float) -> float:
    """Lepton base integer n = k*|H| + r(lepton) from horizon decomposition."""
    return k * HORIZON_CARDINALITY + lepton_horizon_residual_r(lepton, m_shell)


def coordinate_law(
    stage: CGMStage | None,
    sector: SectorType | None,
    d_coeff: float,
    const: float,
    algebra: CompactAlgebra | None = None,
    *,
    matter_density_d2: float | None = None,
) -> float:
    if algebra is None:
        algebra = compact_algebra()
    delta = algebra.delta
    linear = d_coeff * delta + const
    if sector is not None and sector.has_d2_correction:
        d2_coeff = d2_coefficient_from_stage(stage, algebra)
        linear += d2_coeff * delta * delta
    if matter_density_d2 is not None:
        linear += matter_density_d2 * delta * delta
    return linear


def conversion_depth(obs: Observable, planck_ref: ScaleReference, *, delta: float = DELTA) -> float:
    if obs.dimension == "energy":
        return math.log(planck_ref.energy_GeV / obs.value, 2.0) / delta
    if obs.dimension == "frequency":
        return math.log(planck_ref.frequency_Hz / obs.value, 2.0) / delta
    if obs.dimension == "time":
        return math.log(obs.value / planck_ref.time_s, 2.0) / delta
    if obs.dimension == "length":
        return math.log(obs.value / planck_ref.length_m, 2.0) / delta
    raise ValueError(f"Unsupported dimension: {obs.dimension}")


def optical_depth_from_coordinate(
    coordinate: float,
    *,
    delta: float = DELTA,
) -> float:
    """Return Beer-Lambert optical depth tau from ruler coordinate n."""
    return coordinate * delta * math.log(2.0)


def coordinate_from_optical_depth(
    tau: float,
    *,
    delta: float = DELTA,
) -> float:
    """Return optical-depth ruler coordinate from tau."""
    return tau / (delta * math.log(2.0))


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def build_constants() -> FundamentalConstants:
    if SCIPY_AVAILABLE:
        return FundamentalConstants(
            c=sc.c,
            h=sc.h,
            hbar=sc.hbar,
            G=sc.G,
            e=sc.e,
            k_B=sc.k,
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
    t_p = math.sqrt(const.hbar * const.G / const.c**5)
    l_p = math.sqrt(const.hbar * const.G / const.c**3)
    e_p_j = math.sqrt(const.hbar * const.c**5 / const.G)
    e_p_gev = e_p_j / (const.e * 1.0e9)
    f_p = 1.0 / t_p
    return ScaleReference(time_s=t_p, length_m=l_p, energy_GeV=e_p_gev, frequency_Hz=f_p)


def ew_scales(const: FundamentalConstants) -> ScaleReference:
    energy_gev = E_EW_GEV
    energy_j = energy_gev * const.e * 1.0e9
    frequency_hz = energy_j / const.h
    time_s = 1.0 / frequency_hz
    length_m = const.c * time_s
    return ScaleReference(
        time_s=time_s,
        length_m=length_m,
        energy_GeV=energy_gev,
        frequency_Hz=frequency_hz,
    )


def electron_energy_gev(const: FundamentalConstants) -> float:
    if SCIPY_AVAILABLE:
        return sc.physical_constants["electron mass energy equivalent in MeV"][0] / 1000.0
    return 0.000_510_998_95


def proton_energy_gev(const: FundamentalConstants) -> float:
    if SCIPY_AVAILABLE:
        return sc.physical_constants["proton mass energy equivalent in MeV"][0] / 1000.0
    return 0.938_272_088


def rydberg_energy_gev() -> float:
    if SCIPY_AVAILABLE:
        return sc.physical_constants["Rydberg constant times hc in eV"][0] / 1.0e9
    return 13.605_693_122_994e-9


def electron_compton_m(const: FundamentalConstants) -> float:
    if SCIPY_AVAILABLE:
        return sc.physical_constants["Compton wavelength"][0]
    m_e = 9.109_383_7015e-31
    return const.h / (m_e * const.c)


def proton_compton_m(const: FundamentalConstants) -> float:
    if SCIPY_AVAILABLE:
        return sc.physical_constants["proton Compton wavelength"][0]
    m_p = 1.672_621_923_69e-27
    return const.h / (m_p * const.c)


def reduced_compton_from_energy_gev(
    const: FundamentalConstants,
    energy_gev: float,
) -> float:
    mass_kg = (energy_gev * const.e * 1.0e9) / (const.c**2)
    return const.hbar / (mass_kg * const.c)


def bohr_radius_m(const: FundamentalConstants) -> float:
    if SCIPY_AVAILABLE:
        return sc.physical_constants["Bohr radius"][0]
    m_e = 9.109_383_7015e-31
    epsilon0 = 8.854_187_8128e-12
    return 4.0 * math.pi * epsilon0 * const.hbar**2 / (m_e * const.e**2)


def classical_electron_radius_m(const: FundamentalConstants) -> float:
    if SCIPY_AVAILABLE:
        return sc.physical_constants["classical electron radius"][0]
    m_e = 9.109_383_7015e-31
    epsilon0 = 8.854_187_8128e-12
    return const.e**2 / (4.0 * math.pi * epsilon0 * m_e * const.c**2)


def cmb_length_m(const: FundamentalConstants, temp_k: float = 2.72548) -> float:
    return (const.hbar * const.c) / (2.0 * math.pi * const.k_B * temp_k)


def wien_length_m(const: FundamentalConstants, temp_k: float = 2.72548) -> float:
    return (const.h * const.c) / (4.965 * const.k_B * temp_k)


def cmb_thermal_energy_gev(const: FundamentalConstants) -> float:
    t_cmb = 2.7255
    e_j = const.k_B * t_cmb
    return e_j / (const.e * 1.0e9)


def build_observables(const: FundamentalConstants) -> List[Observable]:
    planck = planck_scales(const)
    ew = ew_scales(const)

    proton_energy = proton_energy_gev(const)
    electron_energy = electron_energy_gev(const)
    rydberg_energy = rydberg_energy_gev()
    electron_compton = electron_compton_m(const)
    proton_compton = proton_compton_m(const)

    h_hyperfine_hz = 1_420_405_751.768
    higgs_mass_gev = 125.10
    w_boson_mass_gev = 80.379
    z_boson_mass_gev = 91.1876
    top_quark_mass_gev = 172.76
    bottom_quark_mass_gev = 4.18
    charm_quark_mass_gev = 1.27
    strange_quark_mass_gev = 0.095
    muon_mass_gev = 0.1056583745
    tau_mass_gev = 1.77686

    lambda_bar_e = reduced_compton_from_energy_gev(const, electron_energy)
    lambda_bar_p = reduced_compton_from_energy_gev(const, proton_energy)
    lambda_bar_h = reduced_compton_from_energy_gev(const, higgs_mass_gev)
    lambda_bar_mu = reduced_compton_from_energy_gev(const, muon_mass_gev)
    lambda_bar_tau = reduced_compton_from_energy_gev(const, tau_mass_gev)

    return [
        Observable(
            "Planck energy",
            "exact-anchor",
            "energy",
            planck.energy_GeV,
            "GeV",
            "derived from CODATA constants",
            BU,
            GAUGE_SECTOR,
        ),
        Observable(
            "Electroweak scale",
            "cgm-anchor",
            "energy",
            ew.energy_GeV,
            "GeV",
            "CGM UV/IR anchor",
            BU,
            GAUGE_SECTOR,
        ),
        Observable(
            "Higgs mass energy",
            "particle-triad",
            "energy",
            higgs_mass_gev,
            "GeV",
            "reference Higgs mass",
            CS_UNA,
            GAUGE_SECTOR,
        ),
        Observable(
            "W boson mass energy",
            "gauge-boson",
            "energy",
            w_boson_mass_gev,
            "GeV",
            "reference electroweak mass",
            UNA_ONA,
            GAUGE_SECTOR,
        ),
        Observable(
            "Z boson mass energy",
            "gauge-boson",
            "energy",
            z_boson_mass_gev,
            "GeV",
            "reference electroweak mass",
            UNA,
            GAUGE_SECTOR,
        ),
        Observable(
            "Top quark mass energy",
            "quark",
            "energy",
            top_quark_mass_gev,
            "GeV",
            "reference mass; scheme-dependent for quarks",
            CS,
            MATTER_SECTOR,
        ),
        Observable(
            "Bottom quark mass energy",
            "quark",
            "energy",
            bottom_quark_mass_gev,
            "GeV",
            "reference mass; scheme-dependent for quarks",
            CS,
            MATTER_SECTOR,
        ),
        Observable(
            "Charm quark mass energy",
            "quark",
            "energy",
            charm_quark_mass_gev,
            "GeV",
            "reference mass; scheme-dependent for quarks",
            CS,
            MATTER_SECTOR,
        ),
        Observable(
            "Strange quark mass energy",
            "quark",
            "energy",
            strange_quark_mass_gev,
            "GeV",
            "MS-bar reference mass at external scale; selector only",
            CS,
            MATTER_SECTOR,
        ),
        Observable(
            "Proton mass energy",
            "particle-triad",
            "energy",
            proton_energy,
            "GeV",
            "CODATA stable quantity",
            CS,
            MATTER_SECTOR,
        ),
        Observable(
            "Electron mass energy",
            "particle-triad",
            "energy",
            electron_energy,
            "GeV",
            "CODATA stable quantity",
            ONA,
            MATTER_SECTOR,
        ),
        Observable(
            "Muon mass energy",
            "lepton-ladder",
            "energy",
            muon_mass_gev,
            "GeV",
            "PDG stable quantity",
            ONA,
            MATTER_SECTOR,
        ),
        Observable(
            "Tau mass energy",
            "lepton-ladder",
            "energy",
            tau_mass_gev,
            "GeV",
            "PDG stable quantity",
            ONA,
            MATTER_SECTOR,
        ),
        Observable("Rydberg energy", "atomic", "energy", rydberg_energy, "GeV", "atomic spectroscopy constant"),
        Observable("CMB thermal energy k_B T", "cosmic", "energy", cmb_thermal_energy_gev(const), "GeV", "T = 2.7255 K thermal scale"),
        Observable("Planck frequency", "exact-anchor", "frequency", planck.frequency_Hz, "Hz", "1 / t_P"),
        Observable("Electroweak frequency", "cgm-anchor", "frequency", ew.frequency_Hz, "Hz", "E_EW / h"),
        Observable("Cs-133 hyperfine frequency", "clock", "frequency", F_CS_133_HZ, "Hz", "exact SI second definition"),
        Observable("Hydrogen 21 cm frequency", "photon-triad", "frequency", h_hyperfine_hz, "Hz", "stable astrophysical line"),
        Observable("Planck time", "exact-anchor", "time", planck.time_s, "s", "derived from CODATA constants"),
        Observable("Cs-133 period", "clock", "time", 1.0 / F_CS_133_HZ, "s", "inverse exact SI frequency"),
        Observable("Planck length", "exact-anchor", "length", planck.length_m, "m", "derived from CODATA constants"),
        Observable("Electron Compton wavelength", "particle", "length", electron_compton, "m", "CODATA stable quantity"),
        Observable("Proton Compton wavelength", "particle", "length", proton_compton, "m", "CODATA stable quantity"),
        Observable("Hydrogen 21 cm wavelength", "photon-triad", "length", const.c / h_hyperfine_hz, "m", "c / nu_21cm"),
        Observable("Higgs reduced Compton", "particle-triad", "length", lambda_bar_h, "m", "reference reduced Compton quantity"),
        Observable("Proton reduced Compton", "particle-triad", "length", lambda_bar_p, "m", "reference reduced Compton quantity"),
        Observable("Electron reduced Compton", "particle-triad", "length", lambda_bar_e, "m", "reference reduced Compton quantity"),
        Observable("Muon reduced Compton", "lepton-ladder", "length", lambda_bar_mu, "m", "reference reduced Compton quantity"),
        Observable("Tau reduced Compton", "lepton-ladder", "length", lambda_bar_tau, "m", "reference reduced Compton quantity"),
        Observable("Bohr radius", "atomic-triad", "length", bohr_radius_m(const), "m", "reference atomic length"),
        Observable("Classical electron radius", "atomic-triad", "length", classical_electron_radius_m(const), "m", "reference atomic length"),
        Observable("CMB characteristic length", "photon-triad", "length", cmb_length_m(const), "m", "reference thermal length"),
        Observable("Wien peak length", "photon-triad", "length", wien_length_m(const), "m", "reference thermal length"),
        Observable("Green photon 550 nm", "photon-triad", "length", 550.0e-9, "m", "reference optical wavelength"),
        Observable("Ly-alpha length", "photon-triad", "length", 121.6e-9, "m", "reference hydrogen wavelength"),
        Observable("Nd:YAG length", "photon-triad", "length", 1.064e-6, "m", "reference laser wavelength"),
    ]


def aperture_delta_coordinate(obs: Observable, planck_ref: ScaleReference) -> float:
    return conversion_depth(obs, planck_ref)


def compton_threshold_coordinate(planck_ref: ScaleReference, const: FundamentalConstants) -> float:
    electron_mass_gev = electron_energy_gev(const)
    return math.log(planck_ref.energy_GeV / electron_mass_gev, 2.0) / DELTA


def one_lightyear_coordinate(planck_ref: ScaleReference, const: FundamentalConstants) -> float:
    one_ly_m = 9.461e15
    e_1ly_gev = (const.h * const.c) / (one_ly_m * const.e * 1e9)
    return math.log(planck_ref.energy_GeV / e_1ly_gev, 2.0) / DELTA


def nearest_phase_landmark(coordinate: float) -> Tuple[float, float, float]:
    cycle_index = math.floor(coordinate / APERTURE_FRAME)
    candidates: List[Tuple[float, float]] = []
    for k in range(cycle_index - 1, cycle_index + 2):
        for landmark in APERTURE_FRAME_LANDMARKS:
            target = k * APERTURE_FRAME + landmark
            candidates.append((target, abs(coordinate - target)))
    nearest, error = min(candidates, key=lambda item: item[1])
    residue = coordinate % APERTURE_FRAME
    return residue, nearest, error


def build_coordinate_table(
    observables: Iterable[Observable],
    coordinate_fn: Callable[[Observable], float],
) -> List[CoordinateResult]:
    rows: List[CoordinateResult] = []
    for obs in observables:
        coord = coordinate_fn(obs)
        residue, nearest, error = nearest_phase_landmark(coord)
        rows.append(
            CoordinateResult(
                name=obs.name,
                family=obs.family,
                dimension=obs.dimension,
                value=obs.value,
                unit=obs.unit,
                source=obs.source,
                coordinate=coord,
                residue=residue,
                nearest=nearest,
                error=error,
            )
        )
    return rows


def name_index(rows: Sequence[CoordinateResult]) -> dict[str, CoordinateResult]:
    return {row.name: row for row in rows}


def compact_algebra(delta: float = DELTA) -> CompactAlgebra:
    c1 = math.comb(6, 1)
    c2 = math.comb(6, 2)
    c3 = math.comb(6, 3)
    m1 = sum(k * math.comb(6, k) for k in range(7))
    epsilon = (1.0 / delta) - 48.0
    m_a = M_A
    eta = m_a - DELTA_BU
    # omega: single-pass monodromy defect (half of round-trip delta_BU).
    omega = DELTA_BU / 2.0
    # kappa: Fresnel-type phase residue (pi/4 geometric vs 1/sqrt(2) dynamical).
    kappa = (math.pi / 4.0) - (1.0 / math.sqrt(2.0))
    d_h = epsilon + 24.0 * delta
    return CompactAlgebra(
        delta=delta,
        epsilon=epsilon,
        eta=eta,
        m_a=m_a,
        omega=omega,
        kappa=kappa,
        sigma=SU2_RESIDUAL,
        d_h=d_h,
        c1=c1,
        c2=c2,
        c3=c3,
        m1=m1,
    )


def delta_self_consistency_rhs(delta: float, *, include_third_order: bool = True) -> float:
    """Right-hand side for the closure equation:
    D = 5/256 * 2^(1/12) * (1 + 6*pi*D^2 + (eta/epsilon)*D^3)
    """
    epsilon = (1.0 / delta) - 48.0
    eta = M_A - DELTA_BU
    correction = 1.0 + 6.0 * math.pi * delta * delta
    if include_third_order:
        correction += (eta / epsilon) * (delta * delta * delta)
    return (5.0 / 256.0) * (2.0 ** (1.0 / 12.0)) * correction


def solve_delta_self_consistency(
    delta_guess: float = DELTA,
    *,
    include_third_order: bool = True,
    max_iter: int = 20,
) -> tuple[float, float]:
    """Solve the closure equation by damped fixed-point iteration.

    Returns (delta_solution, residual).
    """
    delta = delta_guess
    for _ in range(max_iter):
        rhs = delta_self_consistency_rhs(
            delta,
            include_third_order=include_third_order,
        )
        delta_next = 0.5 * delta + 0.5 * rhs
        if abs(delta_next - delta) <= 1e-15:
            delta = delta_next
            break
        delta = delta_next

    residual = delta - delta_self_consistency_rhs(
        delta,
        include_third_order=include_third_order,
    )
    return delta, residual


def ckm_conversion_quantities(delta: float = DELTA) -> dict[str, float]:
    """Compute CKM inclusive/exclusive conversion-depth values."""
    algebra = compact_algebra(delta)
    exclusive = math.sin(9.0 * delta * delta)
    phase_corr = algebra.eta - exclusive
    inclusive_full = math.sin(9.0 * delta * delta + phase_corr)
    ratio = algebra.eta / exclusive if exclusive != 0.0 else float("inf")
    return {
        "exclusive": exclusive,
        "inclusive_density": algebra.eta,
        "inclusive_alt": inclusive_full,
        "delta_product": phase_corr,
        "ratio": ratio,
    }


def ew_log2_gap(rows: Sequence[CoordinateResult], name: str) -> float:
    by_name = name_index(rows)
    return math.log(by_name["Electroweak scale"].value / by_name[name].value, 2.0)


def ew_delta_coordinate(rows: Sequence[CoordinateResult], name: str) -> float:
    return ew_log2_gap(rows, name) / DELTA


def solve_delta_top(l_t: float) -> float:
    return (l_t + 1.0) / 73.0


def solve_delta_top_quadratic(l_t: float, zeta: float = TOP_MATTER_DENSITY_Q) -> float:
    """Back-solve Delta from L_t = 73 Delta - 1 + zeta * Delta^2.

    Default zeta = TOP_MATTER_DENSITY_Q (1/4): CS matter density projector.
    With zeta = 0: linear backbone L_t = 73D - 1 only (legacy comparison).
    """
    if zeta == 0.0:
        return solve_delta_top(l_t)
    a = zeta
    b = 73.0
    c = -(1.0 + l_t)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        raise ValueError("negative discriminant in Top quadratic back-solve")
    return (-b + math.sqrt(disc)) / (2.0 * a)


def solve_delta_higgs(l_h: float) -> float:
    return (96.0 - math.sqrt(96.0**2 - 96.0 * (1.0 + l_h))) / 48.0


def solve_delta_z(l_z: float) -> float:
    return (
        117.0
        - math.sqrt(117.0**2 - 90.0 * ((47.0 / 48.0) + l_z))
    ) / 45.0


def solve_delta_w(l_w: float) -> float:
    return (
        126.0
        - math.sqrt(126.0**2 - 130.0 * ((47.0 / 48.0) + l_w))
    ) / 65.0


def solve_delta_wz_phase_corrected(
    l_wz: float, *, guess: float = 0.0207, iterations: int = 16
) -> float:
    """Solve:

    log2(mZ/mW) = 9D - 10D^2 + 2D^3/sqrt(5) - D^4.
    """
    target = l_wz
    delta = guess

    sqrt5 = math.sqrt(5.0)
    for _ in range(iterations):
        f = 9.0 * delta - 10.0 * delta * delta + 2.0 * delta**3 / sqrt5 - delta**4 - target
        fp = 9.0 - 20.0 * delta + 6.0 * delta * delta / sqrt5 - 4.0 * delta**3
        delta -= f / fp

    return delta


def solve_delta_wz(l_wz: float) -> float:
    """Backward-compatible W/Z backsolve for the phase-closed split law."""
    return solve_delta_wz_phase_corrected(l_wz)


def electroweak_delta_backsolves(rows: Sequence[CoordinateResult]) -> Tuple[DeltaBacksolve, ...]:
    by_name = name_index(rows)
    ew = by_name["Electroweak scale"].value

    l_t = math.log(ew / by_name["Top quark mass energy"].value, 2.0)
    l_h = math.log(ew / by_name["Higgs mass energy"].value, 2.0)
    l_z = math.log(ew / by_name["Z boson mass energy"].value, 2.0)
    l_w = math.log(ew / by_name["W boson mass energy"].value, 2.0)
    l_wz = math.log(
        by_name["Z boson mass energy"].value / by_name["W boson mass energy"].value,
        2.0,
    )

    return (
        DeltaBacksolve(
            "top",
            "L_t = 73D - 1 + (1/4)D^2",
            solve_delta_top_quadratic(l_t, TOP_MATTER_DENSITY_Q),
        ),
        DeltaBacksolve("Higgs", "L_H = 96D - 1 - 24D^2", solve_delta_higgs(l_h)),
        DeltaBacksolve("Z", "L_Z = 117D - 47/48 - 45D^2/2", solve_delta_z(l_z)),
        DeltaBacksolve("W", "L_W = 126D - 47/48 - 65D^2/2", solve_delta_w(l_w)),
        DeltaBacksolve(
            "W/Z",
            "log2(mZ/mW) = D(9 - 10D + 2D^2/sqrt(5) - D^3)",
            solve_delta_wz_phase_corrected(l_wz),
        ),
    )


def hzw_delta_consensus(
    rows: Sequence[CoordinateResult],
) -> Tuple[DeltaBacksolve, DeltaBacksolve, DeltaBacksolve, DeltaBacksolve]:
    backsolves = electroweak_delta_backsolves(rows)
    h = next(item for item in backsolves if item.source == "Higgs")
    z = next(item for item in backsolves if item.source == "Z")
    w = next(item for item in backsolves if item.source == "W")
    mean = DeltaBacksolve(
        "H/Z/W mean",
        "mean(Higgs,Z,W)",
        (h.delta_back + z.delta_back + w.delta_back) / 3.0,
    )
    return h, z, w, mean


def electroweak_laws(delta: float = DELTA) -> dict[str, float]:
    algebra = compact_algebra(delta)
    zeta_h = -algebra.m1 / 8.0
    zeta_z = zeta_h + algebra.c1 / 4.0
    zeta_w = zeta_z - algebra.c3 / 2.0
    l_t = (64.0 + algebra.c2 - algebra.c1) * delta - 1.0 + TOP_MATTER_DENSITY_Q * delta * delta
    l_h = (algebra.m1 / 2.0) * delta - 1.0 + zeta_h * delta * delta
    l_z = ((algebra.m1 / 2.0) + algebra.c1 + algebra.c2) * delta - (1.0 - 1.0 / 48.0) + zeta_z * delta * delta
    l_w = ((algebra.m1 / 2.0) + 2.0 * algebra.c2) * delta - (1.0 - 1.0 / 48.0) + zeta_w * delta * delta
    return {
        "L_t": l_t,
        "L_H": l_h,
        "L_Z": l_z,
        "L_W": l_w,
        "log_y_t": 1.5 - l_t,
        "log_lambda_H": -1.0 - 2.0 * l_h,
        "log_g_Z": 95.0 / 48.0 - l_z,
        "log_g": 95.0 / 48.0 - l_w,
    }


def electroweak_laws_stage_foundation(
    delta: float = DELTA,
    *,
    include_delta_transport: bool = False,
) -> dict[str, float]:
    """Primary geometric law from stage actions and optical conjugacy.

    This form keeps stage structure explicit and optionally adds the usual Δ
    transport terms used in the compact cascade.
    """
    foundation = {
        "L_t": optical_conjugacy_seed_log2(CS),
        "L_H": optical_conjugacy_seed_log2(CS_UNA),
        "L_Z": optical_conjugacy_seed_log2(UNA),
        "L_W": optical_conjugacy_seed_log2(UNA_ONA),
    }
    if not include_delta_transport:
        return foundation

    transport = electroweak_laws(delta)
    return {
        "L_t": foundation["L_t"] + (transport["L_t"] + 1.0),
        "L_H": foundation["L_H"] + (transport["L_H"] + 1.0),
        "L_Z": foundation["L_Z"] + (transport["L_Z"] + 1.0),
        "L_W": foundation["L_W"] + (transport["L_W"] + 1.0),
        "log_y_t": transport["log_y_t"],
        "log_lambda_H": transport["log_lambda_H"],
        "log_g_Z": transport["log_g_Z"],
        "log_g": transport["log_g"],
    }


def r5_closure_coeff(
    stage: CGMStage | None,
    H: float = HORIZON_CARDINALITY,
    C1: int = CODE_C1,
    C2: int = CODE_C2,
) -> float:
    """Dyadic fifth-order closure coefficient from K4 stage flags."""
    if stage is None:
        return 0.0
    base = 1.0 if stage.base_gyration else 0.0
    rot = 1.0 if stage.rotational else 0.0
    bal = 1.0 if stage.balance else 0.0
    wz_offset = C2 - C1
    return -wz_offset / 2.0 + (H - wz_offset) / 8.0 * (base - rot) + C2 / 8.0 * bal


def electroweak_laws_full(
    delta: float = DELTA,
    *,
    order: int = 5,
) -> dict[str, float]:
    """Electroweak log laws up to the requested order.

    order=2: D^2 kernel code only.
    order=3: add D^3 gyroscopic phase.
    order=4: add D^4 gyroscopic closure.
    order=5: add D^5 code curvature.
    """
    base = electroweak_laws(delta)
    if order <= 2:
        return base
    stages = {
        "L_t": CS,
        "L_H": CS_UNA,
        "L_Z": UNA,
        "L_W": UNA_ONA,
    }
    result = dict(base)
    for key, stage in stages.items():
        if order >= 3:
            result[key] += gyroscopic_phase_correction(stage, delta)
        if order >= 4:
            result[key] += gyroscopic_closure_correction(stage, delta)
        if order >= 5:
            result[key] += r5_closure_coeff(stage) * delta**5
    return result


def electroweak_delta_coordinates_full(
    delta: float = DELTA,
    *,
    order: int = 5,
) -> dict[str, float]:
    laws = electroweak_laws_full(delta, order=order)
    algebra = compact_algebra(delta)
    ms = float(algebra.m1)
    n_t = laws["L_t"] / delta
    n_h = laws["L_H"] / delta
    n_z = laws["L_Z"] / delta
    n_w = laws["L_W"] / delta
    # Quark anchors: empirical integer shells plus kappa/omega selectors. PDG mass
    # floors (~1% b/c, ~5% s) map to ~0.7-3.5 tick uncertainty at the EW scale.
    n_bottom = 284.0 + algebra.kappa
    n_charm = 367.0 + algebra.omega + 0.5 * delta
    n_strange = 548.0 - (algebra.omega + algebra.kappa)
    n_tau = lepton_coordinate(5, "tau", m_shell=ms) - (MONODROMY_BU + 5.0 * delta)
    n_muon = lepton_coordinate(8, "mu", m_shell=ms) + float(MUON_EQUATOR_COEFF) * delta
    n_electron = (
        lepton_coordinate(14, "e", m_shell=ms)
        + SU2_RESIDUAL
        + (delta / (((n_h * delta) / KERNEL_APERTURE_Q256)))
    )
    return {
        "n_t": n_t,
        "n_h": n_h,
        "n_z": n_z,
        "n_w": n_w,
        "n_bottom": n_bottom,
        "n_charm": n_charm,
        "n_strange": n_strange,
        "n_tau": n_tau,
        "n_muon": n_muon,
        "n_electron": n_electron,
    }


def electroweak_delta_coordinates(delta: float = DELTA) -> dict[str, float]:
    """Coordinates through D^2 law."""
    return electroweak_delta_coordinates_full(delta, order=2)


def electroweak_delta_coordinates_with_phase(
    delta: float = DELTA,
) -> dict[str, float]:
    """Coordinates through D^3 law."""
    return electroweak_delta_coordinates_full(delta, order=3)


def electroweak_delta_coordinates_with_phase_and_closure(
    delta: float = DELTA,
) -> dict[str, float]:
    """Coordinates through D^4 law."""
    return electroweak_delta_coordinates_full(delta, order=4)


def higgs_bare_coordinate(delta: float = DELTA) -> float:
    n_h = electroweak_delta_coordinates(delta)["n_h"]
    return (n_h * delta) / KERNEL_APERTURE_Q256


def electron_delta_coordinate(delta: float = DELTA) -> float:
    algebra = compact_algebra(delta)
    n_h = electroweak_delta_coordinates(delta)["n_h"]
    return (
        lepton_coordinate(14, "e", m_shell=float(algebra.m1))
        + SU2_RESIDUAL
        + (delta / (((n_h * delta) / KERNEL_APERTURE_Q256)))
    )


def electroweak_masses(ew_value: float, delta: float = DELTA) -> dict[str, float]:
    """Masses from electroweak laws through five-order closure."""
    laws = electroweak_laws_full(delta, order=5)
    return {
        "m_t": ew_value * (2.0 ** (-laws["L_t"])),
        "m_h": ew_value * (2.0 ** (-laws["L_H"])),
        "m_z": ew_value * (2.0 ** (-laws["L_Z"])),
        "m_w": ew_value * (2.0 ** (-laws["L_W"])),
    }


def compact_couplings(ew_value: float, delta: float = DELTA) -> dict[str, float]:
    """Couplings derived from compact-predicted masses, not observed masses."""
    masses = electroweak_masses(ew_value, delta)
    lambda_h = (masses["m_h"] * masses["m_h"]) / (2.0 * ew_value * ew_value)
    g = (2.0 * masses["m_w"]) / ew_value
    g_z = (2.0 * masses["m_z"]) / ew_value
    g_prime = math.sqrt(max(g_z * g_z - g * g, 0.0))
    e_charge = (g * g_prime) / g_z if g_z != 0.0 else float("nan")
    alpha_ew_inv = 4.0 * math.pi / (e_charge * e_charge) if e_charge != 0.0 else float("inf")
    y_t = math.sqrt(2.0) * masses["m_t"] / ew_value
    return {
        "lambda_H": lambda_h,
        "g": g,
        "g_Z": g_z,
        "g_prime": g_prime,
        "e": e_charge,
        "alpha_EW_inv": alpha_ew_inv,
        "y_t": y_t,
    }


@dataclass(frozen=True)
class HZWLeaveOneOut:
    target: str
    delta_source: str
    delta_used: float
    predicted_mass: float
    reference_mass: float

    @property
    def relative_error(self) -> float:
        return (self.predicted_mass / self.reference_mass) - 1.0


def hzw_log_law(source: str, delta: float) -> float:
    if source == "Higgs":
        return 96.0 * delta - 1.0 - 24.0 * delta * delta
    if source == "Z":
        return 117.0 * delta - (47.0 / 48.0) - (45.0 / 2.0) * delta * delta
    if source == "W":
        return 126.0 * delta - (47.0 / 48.0) - (65.0 / 2.0) * delta * delta
    raise ValueError(f"Unsupported H/Z/W source: {source}")


def hzw_observable_name(source: str) -> str:
    if source == "Higgs":
        return "Higgs mass energy"
    if source == "Z":
        return "Z boson mass energy"
    if source == "W":
        return "W boson mass energy"
    raise ValueError(f"Unsupported H/Z/W source: {source}")


def hzw_leave_one_out_predictions(
    rows: Sequence[CoordinateResult],
) -> Tuple[HZWLeaveOneOut, ...]:
    by_name = name_index(rows)
    ew_value = by_name["Electroweak scale"].value
    h, z, w, _ = hzw_delta_consensus(rows)

    delta_by_source = {
        "Higgs": h.delta_back,
        "Z": z.delta_back,
        "W": w.delta_back,
    }

    results: list[HZWLeaveOneOut] = []
    for target in ("Higgs", "Z", "W"):
        other_sources = [source for source in ("Higgs", "Z", "W") if source != target]
        delta_used = sum(delta_by_source[source] for source in other_sources) / 2.0
        log_gap = hzw_log_law(target, delta_used)
        predicted_mass = ew_value * (2.0 ** (-log_gap))
        reference_mass = by_name[hzw_observable_name(target)].value
        results.append(
            HZWLeaveOneOut(
                target=target,
                delta_source="+".join(other_sources),
                delta_used=delta_used,
                predicted_mass=predicted_mass,
                reference_mass=reference_mass,
            )
        )

    return tuple(results)
