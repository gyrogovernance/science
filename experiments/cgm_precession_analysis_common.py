#!/usr/bin/env python3
"""
Shared types and physical-map registry for CGM precession analysis.

Companions: cgm_precession_analysis_{1,2,run}.py, cgm_precession_mixing_probes.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class OriginLayer(str, Enum):
    LOGICAL_PRIOR = "LOGICAL_PRIOR"
    EINSTEIN_REALIZATION = "EINSTEIN_REALIZATION"
    ELEMENTARY_CANONICAL = "ELEMENTARY_CANONICAL"
    COMPOSITE_CANONICAL = "COMPOSITE_CANONICAL"
    CLOSURE_RESPONSE = "CLOSURE_RESPONSE"
    COMPACT_FIBER = "COMPACT_FIBER"
    EQUAL_SPEED_CALIBRATION = "EQUAL_SPEED_CALIBRATION"
    BOOST_PROTOCOL = "BOOST_PROTOCOL"
    CHART_IMPLEMENTATION = "CHART_IMPLEMENTATION"
    DOWNSTREAM_GRAVITY = "DOWNSTREAM_GRAVITY"


class TransportLaw(str, Enum):
    NONE = "NONE"
    CANONICAL = "CANONICAL"
    LAB = "LAB"
    CHART = "CHART"
    SU2 = "SU2"


class InvariantType(str, Enum):
    PRIOR_NUMBER = "PRIOR_NUMBER"
    SPEED_IDENTIFICATION = "SPEED_IDENTIFICATION"
    CONJUGACY_ANGLE = "CONJUGACY_ANGLE"
    AXIS_TRANSPORT = "AXIS_TRANSPORT"
    SECANT_GAIN = "SECANT_GAIN"
    TANGENT_GAIN = "TANGENT_GAIN"
    LINEAR_GAIN = "LINEAR_GAIN"
    PROTOCOL_RESIDUAL = "PROTOCOL_RESIDUAL"
    CHART_RESIDUAL = "CHART_RESIDUAL"
    COMMUTATOR = "COMMUTATOR"
    COMPOSITE_CLASS = "COMPOSITE_CLASS"
    KINEMATIC_DICT = "KINEMATIC_DICT"
    PREDICTION = "PREDICTION"


class PhysicalStatus(str, Enum):
    FORCED = "FORCED"
    ELEMENTARY = "ELEMENTARY"
    COMPOSITE = "COMPOSITE"
    CALIBRATION = "CALIBRATION"
    PROTOCOL = "PROTOCOL"
    COORDINATE_READOUT = "COORDINATE_READOUT"
    CHART_IMPLEMENTATION_RESIDUAL = "CHART_IMPLEMENTATION_RESIDUAL"
    DOWNSTREAM = "DOWNSTREAM"
    UNKNOWN = "UNKNOWN"


@dataclass
class PhysicalMap:
    plain_measurement: str
    candidate_physics: str
    alternatives: list[str] = field(default_factory=list)
    status: PhysicalStatus = PhysicalStatus.UNKNOWN


@dataclass
class MetricRecord:
    name: str
    value: float
    origin_layer: OriginLayer
    transport_law: TransportLaw
    invariant_type: InvariantType
    physical_status: PhysicalStatus
    used_downstream: bool
    physical_map: PhysicalMap


GRAVITY_USED_NAMES = frozenset(
    {
        "m_a",
        "delta_BU",
        "rho",
        "rho_secant",
        "Delta",
        "Delta_secant",
    }
)


def used_downstream_for(name: str) -> bool:
    return name in GRAVITY_USED_NAMES


def physical_map_for(name: str) -> PhysicalMap:
    maps: dict[str, PhysicalMap] = {
        "u_p": PhysicalMap(
            "logical UNA threshold number",
            "stage prior; Einstein-ball beta is an extra identification",
            status=PhysicalStatus.FORCED,
        ),
        "o_p": PhysicalMap(
            "logical ONA threshold number",
            "stage prior; Einstein-ball beta is an extra identification",
            status=PhysicalStatus.FORCED,
        ),
        "m_a": PhysicalMap(
            "logical BU threshold number / aperture amplitude",
            "stage prior; gravity uses the number as BU amplitude",
            status=PhysicalStatus.FORCED,
        ),
        "theta_cs": PhysicalMap(
            "CS orientation (pi/2); not a ball speed",
            "stage prior; gauge/orientation frame",
            status=PhysicalStatus.FORCED,
        ),
        "beta_UNA": PhysicalMap(
            "Einstein-ball speed assigned to UNA",
            "realization bridge: threshold number placed as beta",
            status=PhysicalStatus.FORCED,
        ),
        "beta_ONA": PhysicalMap(
            "Einstein-ball speed assigned to ONA",
            "realization bridge: threshold number placed as beta",
            status=PhysicalStatus.FORCED,
        ),
        "beta_BU": PhysicalMap(
            "Einstein-ball speed assigned to BU",
            "realization bridge: threshold number placed as beta",
            status=PhysicalStatus.FORCED,
        ),
        "omega_UO_stage": PhysicalMap(
            "defect(origin, UNA, ONA); gyration about the complementary axis",
            "Wigner rotation of composing orthogonal boosts UNA then ONA. A carried spatial frame (gyroscope) rotates by this angle. Named coupling unknown.",
            alternatives=["Thomas rotation of those two boosts; palindrome axis-steering angle"],
            status=PhysicalStatus.ELEMENTARY,
        ),
        "omega_OB_stage": PhysicalMap(
            "defect(origin, ONA, BU+); same as omega_corner",
            "Wigner rotation of composing boosts ONA then BU+. A carried spatial frame rotates by this angle. Twice this angle is the closed dual-pole holonomy.",
            alternatives=["half of delta_BU"],
            status=PhysicalStatus.ELEMENTARY,
        ),
        "omega_UB_stage": PhysicalMap(
            "defect(origin, BU+, UNA)",
            "Wigner rotation of composing boosts BU+ then UNA. A carried spatial frame rotates by this angle. Named coupling unknown.",
            alternatives=["half of the UNA dual-pole loop"],
            status=PhysicalStatus.ELEMENTARY,
        ),
        "omega_corner": PhysicalMap(
            "Thomas-Wigner / Ungar defect of origin-ONA-BU+",
            "alias of omega_OB_stage",
            status=PhysicalStatus.ELEMENTARY,
        ),
        "delta_BU": PhysicalMap(
            "canonical holonomy of ONA-BU+-BU-; 2*omega_OB_stage",
            "closed Fermi-Walker holonomy of the dual-pole loop; path-ordered integral of the Thomas 1-form. Gravity consumes this number as its Regge deficit unit (not recomputed here).",
            alternatives=["hyperbolic area of that triangle in the velocity ball"],
            status=PhysicalStatus.FORCED,
        ),
        "delta_UNA_BU": PhysicalMap(
            "canonical holonomy of UNA-BU+-BU-; 2*omega_UB_stage",
            "closed Fermi-Walker holonomy of the UNA-rooted dual-pole triangle. Named coupling unknown.",
            status=PhysicalStatus.COMPOSITE,
        ),
        "delta_UOB": PhysicalMap(
            "canonical defect of UNA-ONA-BU+",
            "Wigner rotation of composing the three stage boosts in that order; not the scalar sum of the three pair angles. Named coupling unknown.",
            status=PhysicalStatus.COMPOSITE,
        ),
        "rho": PhysicalMap(
            "delta_BU / m_a",
            "secant closure gain of the balance channel",
            status=PhysicalStatus.FORCED,
        ),
        "rho_secant": PhysicalMap(
            "delta_BU / m_a",
            "alias of rho; total closure from 0 to m_a",
            status=PhysicalStatus.FORCED,
        ),
        "Delta": PhysicalMap(
            "1 - rho",
            "secant aperture; gravity uses this in tau_G",
            status=PhysicalStatus.FORCED,
        ),
        "Delta_secant": PhysicalMap(
            "1 - rho_secant",
            "alias of Delta",
            status=PhysicalStatus.FORCED,
        ),
        "rho0": PhysicalMap(
            "d(delta_BU)/d(m_a) at m_a=0; also 2 k(o_p)",
            "zero-BU linear closure gain",
            status=PhysicalStatus.FORCED,
        ),
        "baseline_gap": PhysicalMap(
            "1 - rho0",
            "linear-response opening of the ONA geometry with BU off",
            status=PhysicalStatus.FORCED,
        ),
        "finite_BU_closure": PhysicalMap(
            "rho - rho0",
            "closure acquired by finite BU amplitude",
            status=PhysicalStatus.FORCED,
        ),
        "rho_tangent": PhysicalMap(
            "d(delta_BU)/d(m_a) at physical m_a",
            "tangent closure gain at the operating point",
            status=PhysicalStatus.FORCED,
        ),
        "Delta_tangent": PhysicalMap(
            "1 - rho_tangent",
            "differential aperture at physical m_a",
            status=PhysicalStatus.FORCED,
        ),
        "nonlinear_closure_gain": PhysicalMap(
            "rho_tangent - rho_secant",
            "nonlinear gain between origin and physical m_a",
            status=PhysicalStatus.FORCED,
        ),
        "d_delta_d_theta": PhysicalMap(
            "d(delta_BU)/d(theta_ona) at physical thresholds",
            "ONA steering susceptibility of the balance holonomy",
            status=PhysicalStatus.FORCED,
        ),
        "elasticity_theta": PhysicalMap(
            "(theta_ona / delta_BU) * d(delta_BU)/d(theta_ona)",
            "fractional ONA elasticity of delta_BU",
            status=PhysicalStatus.FORCED,
        ),
        "elasticity_m": PhysicalMap(
            "(m_a / delta_BU) * d(delta_BU)/d(m_a)",
            "fractional BU-amplitude elasticity of delta_BU",
            status=PhysicalStatus.FORCED,
        ),
        "phi_SU2": PhysicalMap(
            "SU(2) commutator angle U V U^-1 V^-1 of orthogonal stage rotations",
            "finite rotation from a compact SU(2) commutator (Berry / solid-angle on the compact fiber). Not mass-shell Thomas holonomy. Named coupling unknown.",
            status=PhysicalStatus.UNKNOWN,
        ),
        "compact_hyperbolic_residual": PhysicalMap(
            "phi_SU2 - 3*delta_BU",
            "mismatch of the compact commutator against three copies of the hyperbolic dual-pole holonomy. Named coupling unknown.",
            status=PhysicalStatus.UNKNOWN,
        ),
        "sigma_compact": PhysicalMap(
            "(phi_SU2 - 3*delta_BU) / m_a",
            "aperture-normalized compact-hyperbolic residual. Named coupling unknown.",
            status=PhysicalStatus.UNKNOWN,
        ),
        "omega0": PhysicalMap(
            "TW(u_p, u_p; theta=o_p) equal-speed Wigner angle",
            "equal-speed UNA response calibration; not the UNA-ONA stage-pair gyration",
            status=PhysicalStatus.CALIBRATION,
        ),
        "omega_equal_speed_UNA": PhysicalMap(
            "alias of omega0",
            "equal-speed Wigner-map response",
            status=PhysicalStatus.CALIBRATION,
        ),
        "two_1_rho0": PhysicalMap(
            "2*(1-rho0)",
            "twice the linear-response opening",
            status=PhysicalStatus.COMPOSITE,
        ),
        "axis_transport": PhysicalMap(
            "acos(|axis_BU · axis_pal|)",
            "palindrome transports the BU axis by the UNA-ONA elementary angle",
            status=PhysicalStatus.ELEMENTARY,
        ),
        "can_bin_zero": PhysicalMap(
            "canonical holonomy of L=2 out-back walks",
            "identity: reverse path cancels under Fermi-Walker / gyr inversion",
            status=PhysicalStatus.COMPOSITE,
        ),
        "can_bin_ona_bu_dualpole": PhysicalMap(
            "canonical dual-pole class (same as delta_BU)",
            "see delta_BU",
            status=PhysicalStatus.COMPOSITE,
        ),
        "can_bin_una_bu_dualpole": PhysicalMap(
            "canonical defect of UNA-BU+-BU- (same as delta_UNA_BU)",
            "see delta_UNA_BU",
            status=PhysicalStatus.COMPOSITE,
        ),
        "can_bin_una_ona_bu": PhysicalMap(
            "canonical defect of UNA-ONA-BU+ (same as delta_UOB)",
            "see delta_UOB",
            status=PhysicalStatus.COMPOSITE,
        ),
        "can_bin_cyc4": PhysicalMap(
            "canonical holonomy of UNA-ONA-BU+-BU-; generators UO, OB, I, UB",
            "Fermi-Walker holonomy of that 4-cycle. Named coupling unknown.",
            status=PhysicalStatus.UNKNOWN,
        ),
        "can_bin_L4_secondary": PhysicalMap(
            "canonical holonomy of UNA-BU+-ONA-BU-; crossed dual-pole 4-cycle",
            "Fermi-Walker holonomy of the crossed 4-cycle. Named coupling unknown.",
            status=PhysicalStatus.UNKNOWN,
        ),
        "theta_lab_BU": PhysicalMap(
            "rotational part of the lab relative-boost word on the BU triangle",
            "rotation from a product of successive Lorentz boosts in one inertial frame (Thomas's original boost-composition construction). Closed only if net boost vanishes.",
            status=PhysicalStatus.PROTOCOL,
        ),
        "theta_lab_pal": PhysicalMap(
            "rotational part of the lab palindrome word",
            "same lab boost-composition construction on the palindrome word. Closed only if net boost vanishes.",
            status=PhysicalStatus.PROTOCOL,
        ),
        "F_BU": PhysicalMap(
            "theta_lab - theta_can on the BU triangle; equals lab ONA-BU+ out-back",
            "difference of lab boost-composition rotation vs Fermi-Walker holonomy on the same vertices. Protocol residual, not a new prior.",
            status=PhysicalStatus.PROTOCOL,
        ),
        "F_pal": PhysicalMap(
            "theta_lab - theta_can on the palindrome",
            "same lab-minus-Fermi-Walker residual on the palindrome. Palindrome-specific.",
            status=PhysicalStatus.PROTOCOL,
        ),
        "F_outback_ONA_BUp": PhysicalMap(
            "lab rotation of ONA-BU+-ONA (canonical 0)",
            "out-and-back residue of lab boost composition; Fermi-Walker inverts and cancels it.",
            status=PhysicalStatus.PROTOCOL,
        ),
        "F_outback_UNA_ONA": PhysicalMap(
            "lab rotation of UNA-ONA-UNA (canonical 0)",
            "out-and-back residue of lab boost composition on the UNA-ONA pair.",
            status=PhysicalStatus.PROTOCOL,
        ),
        "F_outback_UNA_BUp": PhysicalMap(
            "lab rotation of UNA-BU+-UNA (canonical 0)",
            "out-and-back residue of lab boost composition; equals F of the UNA dual-pole word.",
            status=PhysicalStatus.PROTOCOL,
        ),
        "lab_spectrum_count_L5": PhysicalMap(
            "number of distinct lab angles on closed walks L<=5",
            "path-word spectrum size; not a curvature quantum",
            status=PhysicalStatus.PROTOCOL,
        ),
        "G_z_BU": PhysicalMap(
            "spherical z-chart Pexp minus delta_BU on BU",
            "singular-coordinate readout of the same LC connection; poles at BU",
            status=PhysicalStatus.COORDINATE_READOUT,
        ),
        "G_z_pal": PhysicalMap(
            "spherical z-chart Pexp minus delta_BU on palindrome",
            "singular-coordinate readout",
            status=PhysicalStatus.COORDINATE_READOUT,
        ),
        "G_diag_BU": PhysicalMap(
            "spherical diag-chart Pexp minus delta_BU on BU",
            "singular-coordinate readout in a pole-free spherical chart",
            status=PhysicalStatus.COORDINATE_READOUT,
        ),
        "G_diag_minus_G_z": PhysicalMap(
            "G_diag - G_z on BU",
            "difference of two spherical readouts; not the connection holonomy",
            status=PhysicalStatus.COORDINATE_READOUT,
        ),
        "omega_chart_complete": PhysicalMap(
            "Cartesian Thomas Pexp (Richardson) of Palge-Pfeifer LC; regular at rest and poles",
            "path-ordered integral of the Thomas 1-form omega = (gamma^2/(gamma+1)) beta x d beta. Equals delta_BU on the dual-pole loop.",
            status=PhysicalStatus.FORCED,
        ),
    }
    return maps.get(
        name,
        PhysicalMap("unlisted metric", "unclassified", status=PhysicalStatus.UNKNOWN),
    )


def classify(name: str) -> tuple[OriginLayer, TransportLaw, InvariantType, PhysicalStatus]:
    table: dict[str, tuple[OriginLayer, TransportLaw, InvariantType, PhysicalStatus]] = {
        "u_p": (OriginLayer.LOGICAL_PRIOR, TransportLaw.NONE, InvariantType.PRIOR_NUMBER, PhysicalStatus.FORCED),
        "o_p": (OriginLayer.LOGICAL_PRIOR, TransportLaw.NONE, InvariantType.PRIOR_NUMBER, PhysicalStatus.FORCED),
        "m_a": (OriginLayer.LOGICAL_PRIOR, TransportLaw.NONE, InvariantType.PRIOR_NUMBER, PhysicalStatus.FORCED),
        "theta_cs": (OriginLayer.LOGICAL_PRIOR, TransportLaw.NONE, InvariantType.PRIOR_NUMBER, PhysicalStatus.FORCED),
        "beta_UNA": (OriginLayer.EINSTEIN_REALIZATION, TransportLaw.NONE, InvariantType.SPEED_IDENTIFICATION, PhysicalStatus.FORCED),
        "beta_ONA": (OriginLayer.EINSTEIN_REALIZATION, TransportLaw.NONE, InvariantType.SPEED_IDENTIFICATION, PhysicalStatus.FORCED),
        "beta_BU": (OriginLayer.EINSTEIN_REALIZATION, TransportLaw.NONE, InvariantType.SPEED_IDENTIFICATION, PhysicalStatus.FORCED),
        "omega_UO_stage": (OriginLayer.ELEMENTARY_CANONICAL, TransportLaw.CANONICAL, InvariantType.CONJUGACY_ANGLE, PhysicalStatus.ELEMENTARY),
        "omega_OB_stage": (OriginLayer.ELEMENTARY_CANONICAL, TransportLaw.CANONICAL, InvariantType.CONJUGACY_ANGLE, PhysicalStatus.ELEMENTARY),
        "omega_UB_stage": (OriginLayer.ELEMENTARY_CANONICAL, TransportLaw.CANONICAL, InvariantType.CONJUGACY_ANGLE, PhysicalStatus.ELEMENTARY),
        "omega_corner": (OriginLayer.ELEMENTARY_CANONICAL, TransportLaw.CANONICAL, InvariantType.CONJUGACY_ANGLE, PhysicalStatus.ELEMENTARY),
        "axis_transport": (OriginLayer.ELEMENTARY_CANONICAL, TransportLaw.CANONICAL, InvariantType.AXIS_TRANSPORT, PhysicalStatus.ELEMENTARY),
        "delta_BU": (OriginLayer.COMPOSITE_CANONICAL, TransportLaw.CANONICAL, InvariantType.CONJUGACY_ANGLE, PhysicalStatus.FORCED),
        "delta_UNA_BU": (OriginLayer.COMPOSITE_CANONICAL, TransportLaw.CANONICAL, InvariantType.COMPOSITE_CLASS, PhysicalStatus.COMPOSITE),
        "delta_UOB": (OriginLayer.COMPOSITE_CANONICAL, TransportLaw.CANONICAL, InvariantType.COMPOSITE_CLASS, PhysicalStatus.COMPOSITE),
        "rho": (OriginLayer.CLOSURE_RESPONSE, TransportLaw.CANONICAL, InvariantType.SECANT_GAIN, PhysicalStatus.FORCED),
        "rho_secant": (OriginLayer.CLOSURE_RESPONSE, TransportLaw.CANONICAL, InvariantType.SECANT_GAIN, PhysicalStatus.FORCED),
        "Delta": (OriginLayer.CLOSURE_RESPONSE, TransportLaw.CANONICAL, InvariantType.SECANT_GAIN, PhysicalStatus.FORCED),
        "Delta_secant": (OriginLayer.CLOSURE_RESPONSE, TransportLaw.CANONICAL, InvariantType.SECANT_GAIN, PhysicalStatus.FORCED),
        "rho0": (OriginLayer.CLOSURE_RESPONSE, TransportLaw.CANONICAL, InvariantType.LINEAR_GAIN, PhysicalStatus.FORCED),
        "baseline_gap": (OriginLayer.CLOSURE_RESPONSE, TransportLaw.CANONICAL, InvariantType.LINEAR_GAIN, PhysicalStatus.FORCED),
        "finite_BU_closure": (OriginLayer.CLOSURE_RESPONSE, TransportLaw.CANONICAL, InvariantType.SECANT_GAIN, PhysicalStatus.FORCED),
        "rho_tangent": (OriginLayer.CLOSURE_RESPONSE, TransportLaw.CANONICAL, InvariantType.TANGENT_GAIN, PhysicalStatus.FORCED),
        "Delta_tangent": (OriginLayer.CLOSURE_RESPONSE, TransportLaw.CANONICAL, InvariantType.TANGENT_GAIN, PhysicalStatus.FORCED),
        "nonlinear_closure_gain": (OriginLayer.CLOSURE_RESPONSE, TransportLaw.CANONICAL, InvariantType.TANGENT_GAIN, PhysicalStatus.FORCED),
        "d_delta_d_theta": (OriginLayer.CLOSURE_RESPONSE, TransportLaw.CANONICAL, InvariantType.TANGENT_GAIN, PhysicalStatus.FORCED),
        "elasticity_theta": (OriginLayer.CLOSURE_RESPONSE, TransportLaw.CANONICAL, InvariantType.TANGENT_GAIN, PhysicalStatus.FORCED),
        "elasticity_m": (OriginLayer.CLOSURE_RESPONSE, TransportLaw.CANONICAL, InvariantType.TANGENT_GAIN, PhysicalStatus.FORCED),
        "phi_SU2": (OriginLayer.COMPACT_FIBER, TransportLaw.SU2, InvariantType.COMMUTATOR, PhysicalStatus.UNKNOWN),
        "compact_hyperbolic_residual": (OriginLayer.COMPACT_FIBER, TransportLaw.SU2, InvariantType.COMMUTATOR, PhysicalStatus.UNKNOWN),
        "sigma_compact": (OriginLayer.COMPACT_FIBER, TransportLaw.SU2, InvariantType.COMMUTATOR, PhysicalStatus.UNKNOWN),
        "omega0": (OriginLayer.EQUAL_SPEED_CALIBRATION, TransportLaw.CANONICAL, InvariantType.CONJUGACY_ANGLE, PhysicalStatus.CALIBRATION),
        "omega_equal_speed_UNA": (OriginLayer.EQUAL_SPEED_CALIBRATION, TransportLaw.CANONICAL, InvariantType.CONJUGACY_ANGLE, PhysicalStatus.CALIBRATION),
        "two_1_rho0": (OriginLayer.CLOSURE_RESPONSE, TransportLaw.CANONICAL, InvariantType.LINEAR_GAIN, PhysicalStatus.COMPOSITE),
        "can_bin_zero": (OriginLayer.COMPOSITE_CANONICAL, TransportLaw.CANONICAL, InvariantType.COMPOSITE_CLASS, PhysicalStatus.COMPOSITE),
        "can_bin_ona_bu_dualpole": (OriginLayer.COMPOSITE_CANONICAL, TransportLaw.CANONICAL, InvariantType.COMPOSITE_CLASS, PhysicalStatus.COMPOSITE),
        "can_bin_una_bu_dualpole": (OriginLayer.COMPOSITE_CANONICAL, TransportLaw.CANONICAL, InvariantType.COMPOSITE_CLASS, PhysicalStatus.COMPOSITE),
        "can_bin_una_ona_bu": (OriginLayer.COMPOSITE_CANONICAL, TransportLaw.CANONICAL, InvariantType.COMPOSITE_CLASS, PhysicalStatus.COMPOSITE),
        "can_bin_cyc4": (OriginLayer.COMPOSITE_CANONICAL, TransportLaw.CANONICAL, InvariantType.COMPOSITE_CLASS, PhysicalStatus.UNKNOWN),
        "can_bin_L4_secondary": (OriginLayer.COMPOSITE_CANONICAL, TransportLaw.CANONICAL, InvariantType.COMPOSITE_CLASS, PhysicalStatus.UNKNOWN),
        "theta_lab_BU": (OriginLayer.BOOST_PROTOCOL, TransportLaw.LAB, InvariantType.PROTOCOL_RESIDUAL, PhysicalStatus.PROTOCOL),
        "theta_lab_pal": (OriginLayer.BOOST_PROTOCOL, TransportLaw.LAB, InvariantType.PROTOCOL_RESIDUAL, PhysicalStatus.PROTOCOL),
        "F_BU": (OriginLayer.BOOST_PROTOCOL, TransportLaw.LAB, InvariantType.PROTOCOL_RESIDUAL, PhysicalStatus.PROTOCOL),
        "F_pal": (OriginLayer.BOOST_PROTOCOL, TransportLaw.LAB, InvariantType.PROTOCOL_RESIDUAL, PhysicalStatus.PROTOCOL),
        "F_outback_ONA_BUp": (OriginLayer.BOOST_PROTOCOL, TransportLaw.LAB, InvariantType.PROTOCOL_RESIDUAL, PhysicalStatus.PROTOCOL),
        "F_outback_UNA_ONA": (OriginLayer.BOOST_PROTOCOL, TransportLaw.LAB, InvariantType.PROTOCOL_RESIDUAL, PhysicalStatus.PROTOCOL),
        "F_outback_UNA_BUp": (OriginLayer.BOOST_PROTOCOL, TransportLaw.LAB, InvariantType.PROTOCOL_RESIDUAL, PhysicalStatus.PROTOCOL),
        "lab_spectrum_count_L5": (OriginLayer.BOOST_PROTOCOL, TransportLaw.LAB, InvariantType.PROTOCOL_RESIDUAL, PhysicalStatus.PROTOCOL),
        "omega_chart_complete": (OriginLayer.COMPOSITE_CANONICAL, TransportLaw.CHART, InvariantType.CONJUGACY_ANGLE, PhysicalStatus.FORCED),
        "G_z_BU": (OriginLayer.CHART_IMPLEMENTATION, TransportLaw.CHART, InvariantType.CHART_RESIDUAL, PhysicalStatus.COORDINATE_READOUT),
        "G_z_pal": (OriginLayer.CHART_IMPLEMENTATION, TransportLaw.CHART, InvariantType.CHART_RESIDUAL, PhysicalStatus.COORDINATE_READOUT),
        "G_diag_BU": (OriginLayer.CHART_IMPLEMENTATION, TransportLaw.CHART, InvariantType.CHART_RESIDUAL, PhysicalStatus.COORDINATE_READOUT),
        "G_diag_minus_G_z": (OriginLayer.CHART_IMPLEMENTATION, TransportLaw.CHART, InvariantType.CHART_RESIDUAL, PhysicalStatus.COORDINATE_READOUT),
    }
    return table.get(
        name,
        (OriginLayer.COMPOSITE_CANONICAL, TransportLaw.NONE, InvariantType.COMPOSITE_CLASS, PhysicalStatus.UNKNOWN),
    )
