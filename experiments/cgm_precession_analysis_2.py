#!/usr/bin/env python3
"""
CGM precession analysis, part 2.

Measures the three stage-pair precessions, palindrome axis steering, secant vs
tangent closure, circular Thomas calibration, inertial-frame net-boost closure,
and compact vs hyperbolic holonomy.

Companions: cgm_precession_analysis_{1,run}.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, TypedDict

import mpmath as mp

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
for _p in (_EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cgm_holonomy_analysis_common import (
    _mp_su2,
    gyrotriangle_defect_mp,
    gyrotriangle_defect_triangle_vertices_mp,
    gyr_matrix_ungar_mp,
    poincare_radius_from_beta,
    radial_coordinates_mp,
    rotation_angle_atan2_mp,
    tw_angle_exact,
    tw_angle_unequal,
)
from cgm_precession_analysis_1 import (
    BU_PATH,
    NAMED_LOOPS,
    STAGES,
    EdgeCache,
    _axis_dot,
    _axis_of,
    _gate,
    _neg3,
    _norm3,
    factor_boost_word_lab_mp,
)

# CODATA 2018 (measured h; carrier masses for Compton clock)
M_E_KG = 9.1093837015e-31
M_P_KG = 1.67262192369e-27
HBAR_J_S = 1.054571817e-34
C_LIGHT = 299792458.0
GEV_TO_J = 1.602176634e-10

CYC4_PATH = ("UNA", "ONA", "BU+", "BU-", "UNA")
L4_PATH = ("UNA", "BU+", "ONA", "BU-", "UNA")
UNA_BU_DUAL_PATH = ("UNA", "BU+", "BU-", "UNA")
UOB_PATH = ("UNA", "ONA", "BU+", "UNA")
LAB_CLOSE_TOL = 1e-10


class LabWordClosure(TypedDict):
    theta: float
    net_boost: float
    eta: float
    boost_dir: tuple[float, float, float]
    status: str


class LoopRotationDecomposition(TypedDict):
    theta: float
    rv: tuple[float, float, float]
    rv_edge_sum: tuple[float, float, float]
    noncomm: tuple[float, float, float]
    noncomm_norm: float


def _fmt_axis(axis: tuple[float, float, float]) -> str:
    return f"({axis[0]:+.6f},{axis[1]:+.6f},{axis[2]:+.6f})"


def equivalent_circular_thomas(theta: float) -> dict[str, float]:
    gamma = 1.0 + theta / (2.0 * math.pi)
    beta = math.sqrt(max(0.0, 1.0 - 1.0 / (gamma * gamma)))
    eta = math.atanh(beta) if beta < 1.0 else float("inf")
    return {
        "theta": theta,
        "turn_fraction": theta / (2.0 * math.pi),
        "gamma": gamma,
        "beta": beta,
        "eta": eta,
    }


def stage_kinematics(cache: EdgeCache) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name in ("UNA", "ONA", "BU+"):
        row = radial_coordinates_mp(_norm3(cache.pts[name]))
        eta = float(row["eta"])
        out[name.replace("+", "")] = {
            "beta": float(row["beta"]),
            "gamma": float(row["gamma"]),
            "eta": eta,
            "proper_velocity": float(row["rho_over_m"]),
            "doppler": math.exp(eta),
            "poincare_r": float(row["poincare_r"]),
        }
    return out


def origin_pair_precession(cache: EdgeCache, a: str, b: str) -> dict[str, Any]:
    pa, pb = cache.pts[a], cache.pts[b]
    defect_row = gyrotriangle_defect_mp(pa, pb)
    R_origin = gyr_matrix_ungar_mp(pa, _neg3(pb))
    R_stage = cache.gyr[(a, b)]
    return {
        "a": a,
        "b": b,
        "theta": float(rotation_angle_atan2_mp(R_origin)),
        "defect": float(defect_row["defect"]),
        "theta_stage_gyr": float(rotation_angle_atan2_mp(R_stage)),
        "axis": _axis_of(R_origin),
        "axis_stage": _axis_of(R_stage),
    }


def stage_pair_precessions(cache: EdgeCache) -> dict[str, dict[str, Any]]:
    return {
        "UNA_ONA": origin_pair_precession(cache, "UNA", "ONA"),
        "ONA_BU": origin_pair_precession(cache, "ONA", "BU+"),
        "BU_UNA": origin_pair_precession(cache, "BU+", "UNA"),
    }


def edge_generator(
    cache: EdgeCache,
    a: str,
    b: str,
    omega_uo: float,
    omega_ob: float,
    omega_ub: float,
) -> dict[str, Any]:
    R = cache.gyr[(a, b)]
    theta = float(rotation_angle_atan2_mp(R))
    axis = _axis_of(R)
    if theta < 1e-12:
        name = "I"
        family = "I"
    else:
        idx = max(range(3), key=lambda i: abs(axis[i]))
        family = ("OB", "UB", "UO")[idx]
        name = family + ("+" if axis[idx] > 0 else "-")
    expected = {"UO": omega_uo, "OB": omega_ob, "UB": omega_ub, "I": 0.0}[family]
    return {"a": a, "b": b, "name": name, "family": family, "theta": theta, "axis": axis, "expected": expected}


def generator_word(cache: EdgeCache, path: tuple[str, ...], omegas: tuple[float, float, float]) -> dict[str, Any]:
    omega_uo, omega_ob, omega_ub = omegas
    bits = [edge_generator(cache, path[i], path[i + 1], omega_uo, omega_ob, omega_ub)["name"] for i in range(len(path) - 1)]
    th, _ = cache.origin_gyr(path)
    return {"path": path, "word": " ".join(bits), "theta": th}


def _k_signed(beta: Any) -> Any:
    b = mp.mpf(beta)
    if b == 0:
        return mp.mpf(0)
    return mp.sign(b) * poincare_radius_from_beta(abs(b))


def delta_formula(theta_ona: Any, m_a: Any) -> Any:
    return 4 * mp.atan(_k_signed(theta_ona) * _k_signed(m_a))


def closure_response(t: Any, basis: dict[str, float]) -> dict[str, float]:
    theta = mp.mpf(t.theta_ona)
    m_a = mp.mpf(t.m_a)
    d_m = mp.diff(lambda x: delta_formula(theta, x), m_a)
    d_theta = mp.diff(lambda x: delta_formula(x, m_a), theta)
    d2_m = mp.diff(lambda x: delta_formula(theta, x), m_a, 2)
    d_m0 = mp.diff(lambda x: delta_formula(theta, x), mp.mpf(0))
    rho_secant = basis["rho"]
    rho_tangent = float(d_m)
    rho0 = basis["rho0"]
    return {
        "rho0": rho0,
        "rho0_from_d0": float(d_m0),
        "baseline_gap": 1.0 - rho0,
        "finite_BU_closure": rho_secant - rho0,
        "rho_secant": rho_secant,
        "Delta_secant": 1.0 - rho_secant,
        "rho_tangent": rho_tangent,
        "Delta_tangent": 1.0 - rho_tangent,
        "nonlinear_closure_gain": rho_tangent - rho_secant,
        "d_delta_d_theta": float(d_theta),
        "d_delta_d_m": rho_tangent,
        "d2_delta_d_m2": float(d2_m),
        "elasticity_theta": float(theta / basis["delta_BU"] * d_theta),
        "elasticity_m": float(m_a / basis["delta_BU"] * d_m),
        "elasticity_ratio": float((theta / m_a) * (d_theta / d_m)),
        "dlogk_ona": float(mp.diff(lambda x: mp.log(poincare_radius_from_beta(x)), theta)),
        "dlogk_m": float(mp.diff(lambda x: mp.log(poincare_radius_from_beta(x)), m_a)),
    }


def equal_speed_response(t: Any) -> dict[str, float]:
    beta = mp.mpf(t.u_p)
    theta = mp.mpf(t.theta_ona)
    d_beta = mp.diff(lambda x: tw_angle_exact(x, theta), beta)
    d_theta = mp.diff(lambda x: tw_angle_exact(beta, x), theta)
    s = abs(float(d_beta)) + abs(float(d_theta))
    share_beta_closed = (12 * mp.sqrt(2) - 4) / 17
    share_theta_closed = (21 - 12 * mp.sqrt(2)) / 17
    return {
        "d_omega_d_beta": float(d_beta),
        "d_omega_d_theta": float(d_theta),
        "share_beta": abs(float(d_beta)) / s,
        "share_theta": abs(float(d_theta)) / s,
        "d_omega_d_beta_closed": float(share_beta_closed),
        "d_omega_d_theta_closed": float(share_theta_closed),
        "d_sum": float(d_beta + d_theta),
    }


def axis_transport_certificate(
    cache: EdgeCache,
    conn: dict[str, Any],
) -> tuple[dict[str, float], list[tuple[str, bool]]]:
    gates: list[tuple[str, bool]] = []
    R_bu = conn["can"]["R_BU"]
    R_pal = conn["can"]["R_pal"]
    A = conn["can"]["A"]
    axis_bu = _axis_of(R_bu)
    axis_pal = _axis_of(R_pal)
    axis_A = _axis_of(A)
    dot_bp = max(-1.0, min(1.0, abs(_axis_dot(axis_bu, axis_pal))))
    axis_transport = math.acos(dot_bp)
    theta_A = float(rotation_angle_atan2_mp(A))
    defect_UO = float(gyrotriangle_defect_mp(cache.pts["UNA"], cache.pts["ONA"])["defect"])
    perpendicular = abs(_axis_dot(axis_bu, axis_A))
    gates.append(_gate("palindrome axis turn = angle(gyr(UNA,ONA))", abs(axis_transport - theta_A) < 1e-10))
    gates.append(_gate("angle(gyr(UNA,ONA)) = defect(origin,UNA,ONA)", abs(theta_A - defect_UO) < 1e-10))
    gates.append(_gate("BU axis perpendicular to UNA-ONA steering axis", perpendicular < 1e-10))
    return {
        "axis_transport": axis_transport,
        "theta_A": theta_A,
        "defect_UNA_ONA": defect_UO,
        "axis_BU_dot_axis_A": perpendicular,
        "axis_BU_dot_axis_pal": _axis_dot(axis_bu, axis_pal),
    }, gates


def compact_hyperbolic_bridge(basis: dict[str, float], m_a: float) -> dict[str, float]:
    residual = basis["phi_SU2"] - 3.0 * basis["delta_BU"]
    return {
        "phi_SU2": basis["phi_SU2"],
        "three_delta_BU": 3.0 * basis["delta_BU"],
        "residual": residual,
        "residual_over_ma": residual / m_a,
        "relative_to_phi": residual / basis["phi_SU2"],
    }


def _cross3(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _signed_axis_turn(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    n: tuple[float, float, float],
) -> float:
    return math.atan2(_axis_dot(n, _cross3(a, b)), _axis_dot(a, b))


def su2_phi_axis(U: Any) -> tuple[float, tuple[float, float, float]]:
    tr = mp.re(U[0, 0] + U[1, 1])
    half = min(mp.mpf(1), max(mp.mpf(-1), tr / 2))
    phi = float(2 * mp.acos(half))
    s = mp.sin(mp.mpf(phi) / 2)
    if abs(s) < mp.mpf("1e-20"):
        return phi, (0.0, 0.0, 0.0)
    nx = float(-mp.im(U[0, 1]) / s)
    ny = float(-mp.re(U[0, 1]) / s)
    nz = float(-mp.im(U[0, 0]) / s)
    nrm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nrm < 1e-15:
        return phi, (0.0, 0.0, 0.0)
    return phi, (nx / nrm, ny / nrm, nz / nrm)


def su2_rel_phi(Ua: Any, Ub: Any) -> float:
    best = float("inf")
    for sign in (1, -1):
        phi, _ = su2_phi_axis(Ua.H * (sign * Ub))
        if phi < best:
            best = phi
    return best


def su2_commutator_matrix() -> Any:
    ux = _mp_su2((mp.mpf(1), mp.mpf(0), mp.mpf(0)), mp.pi / 4)
    uy = _mp_su2((mp.mpf(0), mp.mpf(1), mp.mpf(0)), mp.pi / 4)
    return ux * uy * ux.H * uy.H


def compact_hyperbolic_matrix(
    cache: EdgeCache, basis: dict[str, float]
) -> dict[str, Any]:
    c = su2_commutator_matrix()
    phi_c, axis_c = su2_phi_axis(c)
    d = mp.mpf(basis["delta_BU"])
    _, r_bu = cache.origin_gyr(("ONA", "BU+", "BU-", "ONA"))
    axis_bu = _axis_of(r_bu)
    u_bu = _mp_su2(axis_bu, d)
    u_aligned = _mp_su2(axis_c, 3 * d)
    u_bu3 = u_bu * u_bu * u_bu
    d3 = [
        _mp_su2((mp.mpf(1), mp.mpf(0), mp.mpf(0)), d),
        _mp_su2((mp.mpf(0), mp.mpf(1), mp.mpf(0)), d),
        _mp_su2((mp.mpf(0), mp.mpf(0), mp.mpf(1)), d),
    ]
    u_xyz = d3[0] * d3[1] * d3[2]
    return {
        "phi_compact": phi_c,
        "axis_compact": axis_c,
        "axis_BU": axis_bu,
        "axis_dot": _axis_dot(axis_c, axis_bu),
        "rel_aligned_3delta": su2_rel_phi(c, u_aligned),
        "rel_BU3": su2_rel_phi(c, u_bu3),
        "rel_xyz_delta": su2_rel_phi(c, u_xyz),
        "scalar_residual": abs(phi_c - 3.0 * basis["delta_BU"]),
    }


def palindrome_conjugation_chain(
    conn: dict[str, Any], omega_uo: float, n_max: int = 8
) -> dict[str, Any]:
    a = conn["can"]["A"]
    r_bu = conn["can"]["R_BU"]
    axis0 = _axis_of(r_bu)
    steer = _axis_of(a)
    rows: list[dict[str, float]] = []
    a_n = mp.eye(3)
    for n in range(0, n_max + 1):
        r = a_n ** -1 * r_bu * a_n
        axis = _axis_of(r)
        turn = _signed_axis_turn(axis0, axis, steer)
        rows.append(
            {
                "n": float(n),
                "theta": float(rotation_angle_atan2_mp(r)),
                "turn": turn,
                "n_omega_uo": n * omega_uo,
            }
        )
        a_n = a * a_n
    return {
        "rows": rows,
        "period_2pi_over_omega_uo": 2.0 * math.pi / omega_uo,
        "theta_A": float(rotation_angle_atan2_mp(a)),
    }


def integer_combo_best(
    theta: float, omegas: tuple[float, float, float], n_max: int = 2
) -> tuple[float, int, int, int, float]:
    uo, ob, ub = omegas
    best = (float("inf"), 0, 0, 0, 0.0)
    for n0 in range(-n_max, n_max + 1):
        for n1 in range(-n_max, n_max + 1):
            for n2 in range(-n_max, n_max + 1):
                pred = n0 * uo + n1 * ob + n2 * ub
                err = abs(pred - theta)
                if err < best[0]:
                    best = (err, n0, n1, n2, pred)
    return best


def poincare_pair_products(t: Any) -> dict[str, float]:
    ku = poincare_radius_from_beta(t.u_p)
    ko = poincare_radius_from_beta(t.theta_ona)
    km = poincare_radius_from_beta(t.m_a)
    return {
        "k_UNA": float(ku),
        "k_ONA": float(ko),
        "k_BU": float(km),
        "k_UNA_k_ONA": float(ku * ko),
        "k_ONA_k_BU": float(ko * km),
        "k_BU_k_UNA": float(km * ku),
        "k_UNA_over_k_ONA": float(ku / ko),
        "k_UNA_over_k_BU": float(ku / km),
        "k_ONA_over_k_BU": float(ko / km),
    }


def pair_angles_from_betas(beta_una: Any, beta_ona: Any, beta_bu: Any) -> dict[str, float]:
    w_uo = float(tw_angle_unequal(beta_una, beta_ona, mp.pi / 2))
    w_ob = float(tw_angle_unequal(beta_ona, beta_bu, mp.pi / 2))
    w_ub = float(tw_angle_unequal(beta_bu, beta_una, mp.pi / 2))
    return {
        "omega_UO": w_uo,
        "omega_OB": w_ob,
        "omega_UB": w_ub,
        "delta_BU": 2.0 * w_ob,
    }


def cgm_units_map(
    t: Any,
    basis: dict[str, float],
    omega_uo: float,
    omega_ob: float,
    omega_ub: float,
    response: dict[str, float],
) -> dict[str, float]:
    ma = float(t.m_a)
    lam = math.sqrt(2.0 * math.pi)
    qg = lam / ma
    s_cs = float(t.theta_cs) / ma
    s_una = float(t.u_p) / ma
    s_ona = float(t.theta_ona) / ma
    delta = basis["Delta"]
    d_tan = response["Delta_tangent"]
    angles = {
        "omega_UO": omega_uo,
        "omega_OB": omega_ob,
        "omega_UB": omega_ub,
        "delta_BU": basis["delta_BU"],
        "delta_UNA_BU": 2.0 * omega_ub,
        "phi_SU2": basis["phi_SU2"],
    }
    out: dict[str, float] = {
        "t_aperture": ma,
        "L_horizon": lam,
        "Q_G": qg,
        "S_CS": s_cs,
        "S_UNA": s_una,
        "S_ONA": s_ona,
        "S_BU": ma,
        "K_QG": s_ona,
        "UNA_ONA_lift": float(t.theta_ona) - float(t.u_p),
        "EM_duality_deg": math.degrees(math.atan(s_ona / s_una)),
        "alpha0": basis["delta_BU"] ** 4 / ma,
        "Q_G_m_a2": qg * ma * ma,
        "forty8_Delta": 48.0 * delta,
        "forty8_Delta_tangent": 48.0 * d_tan,
        "one_over_Delta": 1.0 / delta,
        "Delta_over_Delta_tangent": delta / d_tan,
        "two_to_Delta": 2.0 ** delta,
        "S_GUT": 1.0 / (1.0 / s_una + 1.0 / s_ona + 1.0 / s_cs),
    }
    for name, theta in angles.items():
        out[f"S_{name}"] = theta / ma
        out[f"n_{name}"] = theta / delta
    out["Omega_delta_BU"] = basis["delta_BU"] / ma
    return out


def lab_word_closure(cache: EdgeCache, path: tuple[str, ...]) -> LabWordClosure:
    ds = [cache.disp[(path[i], path[i + 1])] for i in range(len(path) - 1)]
    fac = factor_boost_word_lab_mp(ds)
    boost_residual = _norm3(fac["u_final"])
    eta = float(mp.atanh(mp.mpf(boost_residual))) if boost_residual < 1.0 - 1e-12 else float("inf")
    u = fac["u_final"]
    n = boost_residual
    direction = (
        (float(u[0]) / n, float(u[1]) / n, float(u[2]) / n) if n > 1e-15 else (0.0, 0.0, 0.0)
    )
    return {
        "theta": float(fac["theta"]),
        "net_boost": boost_residual,
        "eta": eta,
        "boost_dir": direction,
        "status": (
            "CLOSED_LAB_PRECESSION" if boost_residual < LAB_CLOSE_TOL else "OPEN_BOOST_WORD_ROTATION"
        ),
    }


def rotation_vector_mp(R: Any) -> tuple[float, float, float]:
    th = float(rotation_angle_atan2_mp(R))
    ax = _axis_of(R)
    return (th * ax[0], th * ax[1], th * ax[2])


def _vec_norm3(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _vec_add(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_sub(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def loop_rotation_decomposition(
    cache: EdgeCache, path: tuple[str, ...]
) -> LoopRotationDecomposition:
    th, R = cache.origin_gyr(path)
    rv = rotation_vector_mp(R)
    rv_edge = (0.0, 0.0, 0.0)
    for i in range(len(path) - 1):
        rv_edge = _vec_add(rv_edge, rotation_vector_mp(cache.gyr[(path[i], path[i + 1])]))
    noncomm = _vec_sub(rv, rv_edge)
    return {
        "theta": th,
        "rv": rv,
        "rv_edge_sum": rv_edge,
        "noncomm": noncomm,
        "noncomm_norm": _vec_norm3(noncomm),
    }


def g_gev_inv2_to_si(g_gev: float) -> float:
    """Natural G (GeV^-2, hbar=c=1) to SI m^3 kg^-1 s^-2 via M_Planck = 1/sqrt(G)."""
    m_planck_gev = 1.0 / math.sqrt(g_gev)
    m_planck_kg = m_planck_gev * GEV_TO_J / (C_LIGHT * C_LIGHT)
    return HBAR_J_S * C_LIGHT / (m_planck_kg * m_planck_kg)


def compton_clock_row(m_kg: float, m_a: float, rho: float, delta: float, omegas: dict[str, float]) -> dict[str, float]:
    t_c = HBAR_J_S / (m_kg * C_LIGHT * C_LIGHT)
    t_ap = m_a * t_c
    omega_c = m_kg * C_LIGHT * C_LIGHT / HBAR_J_S
    omega_bu = rho * omega_c
    e_bu = rho * m_kg * C_LIGHT * C_LIGHT
    e_ap = delta * m_kg * C_LIGHT * C_LIGHT
    row: dict[str, float] = {
        "T_C": t_c,
        "T_aperture": t_ap,
        "omega_C": omega_c,
        "Omega_BU": omega_bu,
        "f_BU_Hz": omega_bu / (2.0 * math.pi),
        "E_BU_J": e_bu,
        "E_aperture_J": e_ap,
    }
    for name, theta in omegas.items():
        omega = theta * omega_c / m_a
        row[f"Omega_{name}"] = omega
        row[f"f_{name}_Hz"] = omega / (2.0 * math.pi)
    return row


def print_physical_readings(
    cache: EdgeCache,
    kin: dict[str, dict[str, float]],
    basis: dict[str, float],
    response: dict[str, float],
    mat: dict[str, Any],
    m_a: float,
    omega_uo: float,
    omega_ob: float,
    omega_ub: float,
    t: Any | None = None,
) -> list[tuple[str, bool]]:
    gates: list[tuple[str, bool]] = []
    delta_bu = basis["delta_BU"]
    delta_una_bu = 2.0 * omega_ub
    rho = basis["rho"]
    delta = basis["Delta"]

    print()
    print("20. PHYSICAL READINGS")
    print("-" * 5)
    print("  The readings below translate the geometric angles into kinematic")
    print("  and physical quantities. beta is the Einstein velocity, gamma*beta")
    print("  is the proper velocity p/mc, and lambda_C/lambda_dB is the ratio of")
    print("  the Compton wavelength to the de Broglie wavelength.")

    print("  A. Stage momentum (beta = p/E, gamma*beta = p/mc)")
    print(f"  {'stage':<6} {'beta':>10} {'gamma*beta':>12} {'lambda_C/lambda_dB':>18}")
    for name, row in kin.items():
        gb = row["proper_velocity"]
        lam_ratio = 1.0 / gb if abs(gb) > 1e-15 else float("inf")
        print(f"  {name:<6} {row['beta']:10.6f} {gb:12.6f} {lam_ratio:18.6f}")
        if name == "UNA":
            gates.append(_gate("UNA: p/mc = 1", abs(gb - 1.0) < 1e-12))
            gates.append(_gate("UNA: lambda_dB = lambda_C", abs(lam_ratio - 1.0) < 1e-12))
    print()

    print("  B. Dual-pole channels")
    print("  The ONA-BU and UNA-BU loops are the two dual-pole channels. Their")
    print("  holonomies are delta_BU and delta_UNA_BU. The angle ratio")
    print("  delta_UNA_BU/delta_BU equals omega_UB/omega_OB. The Poincare-radius")
    print("  ratio k(UNA)/k(ONA) equals tan(omega_UB/2)/tan(omega_OB/2). Those")
    print("  two identities are distinct.")
    _, r_ona_bu = cache.origin_gyr(BU_PATH)
    _, r_una_bu = cache.origin_gyr(UNA_BU_DUAL_PATH)
    ax_ona = _axis_of(r_ona_bu)
    ax_una = _axis_of(r_una_bu)
    print(f"  {'channel':<14} {'holonomy':>14} {'axis':>28} {'pair':>10}")
    print(f"  {'ONA-BU (OB)':<14} {delta_bu:14.12f} {_fmt_axis(ax_ona):>28} {'displ-bal':>10}")
    print(f"  {'UNA-BU (UB)':<14} {delta_una_bu:14.12f} {_fmt_axis(ax_una):>28} {'rot-bal':>10}")
    print(f"  delta_UNA_BU / delta_BU     {delta_una_bu / delta_bu:.12f}")
    print(f"  omega_UB / omega_OB         {omega_ub / omega_ob:.12f}")
    tan_ratio = math.tan(0.5 * omega_ub) / math.tan(0.5 * omega_ob)
    kprod = poincare_pair_products(t) if t is not None else None
    print(f"  tan(omega_UB/2)/tan(omega_OB/2) {tan_ratio:.12f}")
    if kprod is not None:
        print(f"  k(UNA)/k(ONA)               {kprod['k_UNA_over_k_ONA']:.12f}")
    print(f"  dual-pole axis dot          {_axis_dot(ax_ona, ax_una):.12f}")
    gates.append(_gate("delta_UNA_BU = 2*omega_UB", abs(delta_una_bu - 2.0 * omega_ub) < 1e-12))
    gates.append(_gate("delta_BU = 2*omega_OB", abs(delta_bu - 2.0 * omega_ob) < 1e-12))
    gates.append(_gate("dual-pole ratio = omega_UB/omega_OB", abs((delta_una_bu / delta_bu) - (omega_ub / omega_ob)) < 1e-12))
    if kprod is not None:
        gates.append(
            _gate(
                "angle ratio distinct from k(UNA)/k(ONA)",
                abs((delta_una_bu / delta_bu) - kprod["k_UNA_over_k_ONA"]) > 1e-6,
            )
        )
    print()

    kernel_ratio = delta_una_bu / delta_bu

    print("  D. Loop rotation vector (rv = theta*axis; edge sum is non-commutative)")
    print("  The rotation of a closed loop is a rotation by theta about some axis.")
    print("  The rotation vector rv = theta*axis is its axis-weighted form. The")
    print("  column |noncomm| is the norm of the difference between the loop")
    print("  rotation and the ordered product of its edge rotations; it is zero")
    print("  only when the edges commute, and it is nonzero for every bent loop.")
    print(f"  {'loop':<10} {'theta':>12} {'rv_x':>10} {'rv_y':>10} {'rv_z':>10} {'|noncomm|':>12}")
    loops = {
        "cyc4": CYC4_PATH,
        "L4_cross": L4_PATH,
        "UOB": UOB_PATH,
        "ONA-BU": BU_PATH,
        "UNA-BU": UNA_BU_DUAL_PATH,
    }
    for label, path in loops.items():
        dec = loop_rotation_decomposition(cache, path)
        rv = dec["rv"]
        print(
            f"  {label:<10} {dec['theta']:12.8f} {rv[0]:10.6f} {rv[1]:10.6f} {rv[2]:10.6f} "
            f"{dec['noncomm_norm']:12.6e}"
        )
        if label in ("cyc4", "L4_cross", "UOB"):
            gates.append(_gate(f"{label}: noncommutative loop", float(dec["noncomm_norm"]) > 1e-6))
    print()

    print("  E. Compton clock (hbar measured; T_aperture = m_a hbar/(m c^2))")
    print("  The Compton time T_C = hbar/(m c^2) is the time attached to a")
    print("  carrier of mass m. The aperture time T_aperture = m_a T_C is the")
    print("  carrier time rescaled by the geometric aperture m_a. f_BU and")
    print("  Omega_BU are rho * omega_C in Hz and rad/s. E_BU = rho m c^2 and")
    print("  E_ap = Delta m c^2 convert the rest energy through rho and Delta.")
    omegas = {"UO": omega_uo, "OB": omega_ob, "UB": omega_ub, "BU": delta_bu}
    eV = 1.602176634e-19
    print(f"  {'carrier':<10} {'T_C':>12} {'T_aperture':>12} {'f_BU':>14} {'E_BU/eV':>14} {'E_ap/eV':>14}")
    for label, m_kg in (("electron", M_E_KG), ("proton", M_P_KG)):
        row = compton_clock_row(m_kg, m_a, rho, delta, omegas)
        print(
            f"  {label:<10} {row['T_C']:12.6e} {row['T_aperture']:12.6e} "
            f"{row['f_BU_Hz']:14.6e} {row['E_BU_J']/eV:14.6e} {row['E_aperture_J']/eV:14.6e}"
        )
    row_e = compton_clock_row(M_E_KG, m_a, rho, delta, omegas)
    from hqvm_gravity_common import g_pred_from_tau, tau_G_stf

    g_pred_gev = g_pred_from_tau(tau_G_stf)
    g_pred_si = g_gev_inv2_to_si(g_pred_gev)
    t_p_pred = math.sqrt(HBAR_J_S * g_pred_si / C_LIGHT**5)
    t_ap_cs = m_a * t_p_pred
    print(f"  G_pred (kernel, GeV^-2)      {g_pred_gev:.6e}")
    print(f"  t_P = sqrt(hbar G_pred/c^5) {t_p_pred:.6e}")
    print(f"  m_a t_P (CS tick)           {t_ap_cs:.6e}")
    print(f"  electron T_aperture         {row_e['T_aperture']:.6e}")
    print(f"  electron f_BU (Hz)          {row_e['f_BU_Hz']:.6e}")
    print(f"  electron Omega_BU (rad/s)   {row_e['Omega_BU']:.6e}")
    gates.append(_gate("electron T_aperture > m_a t_P(G_pred)", row_e["T_aperture"] > t_ap_cs))
    gates.append(
        _gate(
            "electron Omega_BU = rho * omega_C",
            abs(row_e["Omega_BU"] - rho * row_e["omega_C"]) < 1e-9 * row_e["omega_C"],
        )
    )
    print(f"  {'channel':<8} {'Omega/electron':>16} {'Omega/proton':>16}")
    row_p = compton_clock_row(M_P_KG, m_a, rho, delta, omegas)
    for ch in ("UO", "OB", "UB", "BU"):
        print(
            f"  {ch:<8} {row_e[f'Omega_{ch}']:16.6e} {row_p[f'Omega_{ch}']:16.6e}"
        )
    print("  Omega = theta / T_aperture for each channel. Values scale as 1/T_C")
    print("  and therefore linearly with carrier mass.")
    gates.append(
        _gate(
            "Omega channel scales with carrier mass",
            abs((row_p["Omega_UO"] / row_e["Omega_UO"]) - (M_P_KG / M_E_KG)) < 1e-6,
        )
    )
    print()

    chi_ch = math.degrees(math.acos(min(1.0, abs(float(mat["axis_dot"])))))
    print("  F. Compact-hyperbolic bridge")
    print("  The compact SU(2) fiber carries a commutator angle phi_SU2, and the")
    print("  hyperbolic BU loop carries delta_BU. Their difference epsilon_CH is")
    print("  the scalar residual of the compact-hyperbolic bridge, and sigma is")
    print("  that residual normalized by the aperture m_a. chi_CH is the angle")
    print("  between the compact axis n_C and the BU axis n_BU. The relative")
    print("  angles rel(C, U_BU^3) and rel(C, Ux Uy Uz delta) measure how well")
    print("  the compact fiber aligns with three successive BU rotations and with")
    print("  three orthogonal delta rotations.")
    print(f"  epsilon_CH = phi_SU2 - 3 delta_BU     {mat['scalar_residual']:.12f}")
    print(f"  sigma = epsilon_CH / m_a              {mat['scalar_residual'] / m_a:.12f}")
    print(f"  chi_CH = arccos(|n_C . n_BU|)         {chi_ch:.6f} deg")
    print(f"  rel(C, U_BU^3)                        {mat['rel_BU3']:.12f}")
    print(f"  rel(C, Ux Uy Uz delta)                {mat['rel_xyz_delta']:.12f}")
    gates.append(
        _gate(
            "chi_CH from axis_dot",
            abs(chi_ch - math.degrees(math.acos(min(1.0, abs(float(mat["axis_dot"])))))) < 1e-9,
        )
    )
    gates.append(
        _gate(
            "epsilon_CH = rel(C, U(n_C, 3 delta_BU))",
            abs(float(mat["scalar_residual"]) - float(mat["rel_aligned_3delta"])) < 1e-12,
        )
    )

    print()
    print("  G. Closure susceptibility")
    print("  rho0 is d(delta_BU)/d(m_a) at m_a = 0. rho and Delta are the secant")
    print("  closure ratio and aperture gap across the full amplitude. rho_t and")
    print("  Delta_t are the tangent values at the physical amplitude.")
    print(f"  rho0 (m=0 slope)                      {response['rho0']:.12f}")
    print(f"  rho (secant)                          {response['rho_secant']:.12f}")
    print(f"  rho_t (tangent)                       {response['rho_tangent']:.12f}")
    print(f"  Delta (secant)                        {response['Delta_secant']:.12f}")
    print(f"  Delta_t (tangent)                     {response['Delta_tangent']:.12f}")
    print(f"  d2 delta_BU / dm2                     {response['d2_delta_d_m2']:.12f}")
    print(f"  Delta / Delta_t                       {response['Delta_secant'] / response['Delta_tangent']:.12f}")
    print()

    from hqvm_gravity_common import (
        G_kernel,
        KAPPA_METRIC,
        d_BU as gravity_d_BU,
        metric_spin_deficit,
        rho as gravity_rho,
        Delta as gravity_Delta,
        tau_G_stf,
    )

    alpha0 = delta_bu**4 / m_a
    print("  H. Gravity analysis shared invariants")
    print("  d_BU, rho, and Delta are the dual-pole holonomy, the closure ratio,")
    print("  and the aperture gap. tau_G is the refractive depth of the")
    print("  gravitational coupling. G_kernel = pi/6 is the kernel flux identity.")
    print("  alpha0 = d_BU^4 / m_a is the electromagnetic coupling from the")
    print("  dual-pole holonomy. KAPPA_METRIC is the shared metric coefficient.")
    print(f"  d_BU (precession)                     {delta_bu:.12f}")
    print(f"  d_BU (gravity common)                 {gravity_d_BU:.12f}")
    print(f"  rho (precession)                      {rho:.12f}")
    print(f"  rho (gravity common)                  {gravity_rho:.12f}")
    print(f"  Delta (precession)                    {delta:.12f}")
    print(f"  Delta (gravity common)                {gravity_Delta:.12f}")
    print(f"  tau_G (STF coupling depth)            {tau_G_stf:.12f}")
    print(f"  G_kernel                              {G_kernel:.12f}")
    print(f"  alpha0 = d_BU^4 / m_a                 {alpha0:.12f}")
    print(f"  KAPPA_METRIC                          {KAPPA_METRIC:.12f}")
    print(f"  delta_UNA_BU / d_BU (kernel ratio)    {kernel_ratio:.12f}")
    h_ref = metric_spin_deficit(s=100.0, u=0.01, a_star=0.98, theta_o_deg=90.0)
    print(f"  h(s=100,a*=0.98,theta_o=90)           {h_ref:.12e}")
    gates.append(_gate("d_BU matches gravity common", abs(delta_bu - gravity_d_BU) < 1e-12))
    gates.append(_gate("rho matches gravity common", abs(rho - gravity_rho) < 1e-12))
    gates.append(_gate("Delta matches gravity common", abs(delta - gravity_Delta) < 1e-12))
    print()

    a_half = math.tan(0.5 * omega_uo)
    b_half = math.tan(0.5 * omega_ob)
    c_half = math.tan(0.5 * omega_ub)
    k_una_inv = math.sqrt(a_half * c_half / b_half)
    k_ona_inv = math.sqrt(a_half * b_half / c_half)
    k_bu_inv = math.sqrt(b_half * c_half / a_half)
    print("  I. Half-angle Poincare radius inversion")
    print("  The three pair identities tan(omega/2) = k(a) k(b) invert. Writing")
    print("  a = tan(omega_UO/2), b = tan(omega_OB/2), c = tan(omega_UB/2), the")
    print("  Poincare radii are k_UNA = sqrt(a c / b), k_ONA = sqrt(a b / c),")
    print("  k_BU = sqrt(b c / a). The threshold column is k computed from the")
    print("  Einstein speeds directly.")
    print(f"  tan(omega_UO/2) = a             {a_half:.12f}")
    print(f"  tan(omega_OB/2) = b             {b_half:.12f}")
    print(f"  tan(omega_UB/2) = c             {c_half:.12f}")
    print(f"  k_UNA = sqrt(a c / b)           {k_una_inv:.12f}")
    print(f"  k_ONA = sqrt(a b / c)           {k_ona_inv:.12f}")
    print(f"  k_BU  = sqrt(b c / a)           {k_bu_inv:.12f}")
    if t is not None:
        kref = poincare_pair_products(t)
        print(f"  k_UNA (threshold)               {kref['k_UNA']:.12f}")
        print(f"  k_ONA (threshold)               {kref['k_ONA']:.12f}")
        print(f"  k_BU (threshold)                {kref['k_BU']:.12f}")
        gates.append(_gate("half-angle inversion k_UNA", abs(k_una_inv - kref["k_UNA"]) < 1e-10))
        gates.append(_gate("half-angle inversion k_ONA", abs(k_ona_inv - kref["k_ONA"]) < 1e-10))
        gates.append(_gate("half-angle inversion k_BU", abs(k_bu_inv - kref["k_BU"]) < 1e-10))
    print()

    print()

    print("  J. Dual-pole angle partition")
    print("  Each dual-pole holonomy as a fraction of their sum.")
    gem_sum = delta_bu + delta_una_bu
    print(f"  delta_BU / (delta_BU + delta_UNA_BU)     {delta_bu / gem_sum:.12f}")
    print(f"  delta_UNA_BU / (delta_BU + delta_UNA_BU) {delta_una_bu / gem_sum:.12f}")
    print()

    print("  K. Loop Lie generator (rv = log(R) axis components; rv_x=OB, rv_y=UB, rv_z=UO)")
    print("  The Lie generator of a loop rotation R is log(R). Its components")
    print("  along the three pair axes are rv_x, rv_y, rv_z. |noncomm| is the")
    print("  residual of that generator against the sum of the edge generators.")
    print(f"  {'loop':<10} {'rv_x':>10} {'rv_y':>10} {'rv_z':>10} {'|noncomm|':>12}")
    for label, path in (("cyc4", CYC4_PATH), ("L4_cross", L4_PATH), ("UOB", UOB_PATH)):
        dec = loop_rotation_decomposition(cache, path)
        rv = dec["rv"]
        print(
            f"  {label:<10} {rv[0]:10.6f} {rv[1]:10.6f} {rv[2]:10.6f} "
            f"{dec['noncomm_norm']:12.6e}"
        )
    print()

    return gates


def run_mechanics(state: dict[str, Any] | None = None) -> list[tuple[str, bool]]:
    if state is None:
        from cgm_holonomy_analysis_common import CGMThresholds
        from cgm_precession_analysis_1 import connection_measurements, forced_basis

        mp.mp.dps = 50
        t = CGMThresholds.make()
        cache = EdgeCache(t)
        basis = forced_basis(t)
        conn = connection_measurements(t, cache)
        state = {"t": t, "cache": cache, "basis": basis, "conn": conn, "priors": {}}
    t = state["t"]
    cache: EdgeCache = state["cache"]
    basis = state["basis"]
    conn = state["conn"]
    gates: list[tuple[str, bool]] = []

    print()
    print("9. STAGE KINEMATICS")
    print("-" * 5)
    print("  Each payload threshold is an Einstein speed beta. gamma is the")
    print("  Lorentz factor, eta is the rapidity atanh(beta), gamma*beta is the")
    print("  proper velocity p/mc, Doppler is sqrt((1+beta)/(1-beta)), and")
    print("  Poincare is k(beta) = beta / (1 + sqrt(1-beta^2)).")
    print(
        f"  {'stage':<6} {'beta':>10} {'gamma':>10} {'eta':>10} "
        f"{'gamma*beta':>12} {'Doppler':>10} {'Poincare':>10}"
    )
    kin = stage_kinematics(cache)
    for name, row in kin.items():
        print(
            f"  {name:<6} {row['beta']:10.6f} {row['gamma']:10.6f} "
            f"{row['eta']:10.6f} {row['proper_velocity']:12.6f} "
            f"{row['doppler']:10.6f} {row['poincare_r']:10.6f}"
        )

    print()
    print("10. STAGE-PAIR PRECESSIONS")
    print("-" * 5)
    print("  Each row is the Wigner rotation produced by composing the two boosts")
    print("  of that payload pair from the origin. The defect column is the")
    print("  gyrotriangle defect of the origin and the two stage points, which")
    print("  equals the hyperbolic area of that triangle. The branch column is")
    print("  half the defect, the angle of one corner of the dual-pole loop.")
    pair_pre = stage_pair_precessions(cache)
    print(f"  {'pair':<10} {'defect':>14} {'branch':>14} {'gyr_origin':>14}  axis_origin")
    for name, row in pair_pre.items():
        print(
            f"  {name:<10} {row['defect']:14.12f} {0.5 * row['defect']:14.12f} "
            f"{row['theta']:14.12f}  {_fmt_axis(row['axis'])}"
        )
        gates.append(
            _gate(
                f"{name}: origin-gyr angle = gyrotriangle defect",
                abs(row["theta"] - row["defect"]) < 1e-10,
            )
        )
    axes = [pair_pre[k]["axis"] for k in ("UNA_ONA", "ONA_BU", "BU_UNA")]
    gates.append(
        _gate(
            "three origin-pair precession axes mutually orthogonal",
            all(abs(_axis_dot(axes[i], axes[j])) < 1e-10 for i in range(3) for j in range(i + 1, 3)),
        )
    )
    omega_uo = pair_pre["UNA_ONA"]["defect"]
    omega_ob = pair_pre["ONA_BU"]["defect"]
    omega_ub = pair_pre["BU_UNA"]["defect"]
    print()
    print("  omega_OB_stage is the ONA-BU corner. omega_corner is the same")
    print("  quantity from the gyrotriangle defect. omega0 is the equal-speed")
    print("  Wigner calibration TW(u_p, u_p; o_p). omega_UO_stage is the")
    print("  UNA-ONA pair angle at the actual stage speeds.")
    gates.append(_gate("omega_OB_stage = omega_corner", abs(omega_ob - basis["omega_corner"]) < 1e-12))
    gates.append(_gate("omega0 != omega_UO_stage", abs(basis["omega0"] - omega_uo) > 0.01))

    print()
    print("  Directed edges between stage points (UO/OB/UB generators or I)")
    print("  Each directed edge is a boost from one stage to another. Its")
    print("  rotation generator is one of the three pair precessions, or the")
    print("  identity, depending on whether the two stages are a payload pair.")
    print(f"  {'edge':<12} {'gen':<6} {'theta':>14}  axis")
    edge_ok = True
    for a in STAGES:
        for b in STAGES:
            if a == b:
                continue
            row = edge_generator(cache, a, b, omega_uo, omega_ob, omega_ub)
            print(
                f"  {a}->{b:<5} {row['name']:<6} {row['theta']:14.12f}  {_fmt_axis(row['axis'])}"
            )
            if abs(row["theta"] - row["expected"]) > 1e-10:
                edge_ok = False
    gates.append(_gate("every directed edge is UO, OB, UB, or I", edge_ok))

    omegas = (omega_uo, omega_ob, omega_ub)
    recon = [
        ("cyc4", ("UNA", "ONA", "BU+", "BU-", "UNA")),
        ("L4_crossed", ("UNA", "BU+", "ONA", "BU-", "UNA")),
        ("UOB", ("UNA", "ONA", "BU+", "UNA")),
        ("BU", ("ONA", "BU+", "BU-", "ONA")),
        ("UNA_dual", ("UNA", "BU+", "BU-", "UNA")),
    ]
    print()
    print("  Closed walks and their generator words")
    print("  Each walk is a closed sequence of stage points. Its rotation is a")
    print("  word in the three pair generators UO, OB, UB and the identity I.")
    print("  The word column records that word; theta is the net rotation angle.")
    print(f"  {'class':<12} {'word':<22} {'theta':>14} {'theta/2':>14}")
    for name, path in recon:
        w = generator_word(cache, path, omegas)
        print(f"  {name:<12} {w['word']:<22} {w['theta']:14.12f} {0.5 * w['theta']:14.12f}")
    l4 = generator_word(cache, recon[1][1], omegas)["theta"]
    combo = integer_combo_best(l4, omegas)
    print(
        f"  The crossed four-cycle is compared with every integer combination "
        f"n0*UO+n1*OB+n2*UB with |n|<=2. "
        f"err={combo[0]:.3e}  ({combo[1]},{combo[2]},{combo[3]}) -> {combo[4]:.12f}"
    )
    gates.append(
        _gate(
            "cyc4 = generator word UNA-ONA-BU+-BU--UNA",
            abs(generator_word(cache, recon[0][1], omegas)["theta"] - 0.4127190541) < 1e-8,
        )
    )
    gates.append(
        _gate(
            "L4_crossed = generator word UNA-BU+-ONA-BU--UNA",
            abs(l4 - 0.2567128344) < 1e-8,
        )
    )
    gates.append(_gate("L4_crossed not integer combo of pair angles |n|<=2", combo[0] > 1e-6))

    print()
    print("11. DUAL-POLE CLOSURE")
    print("-" * 5)
    print("  The three pair precessions omega_UO, omega_OB, omega_UB are the")
    print("  elementary rotations. The dual-pole loop ONA->BU+->BU-->ONA doubles")
    print("  omega_OB, and the UNA->BU+->BU-->UNA loop doubles omega_UB. The")
    print("  three-stage triangle UOB is the ordered product of the three")
    print("  elementary rotations; its defect minus their scalar sum measures")
    print("  the noncommutativity of rotations about distinct axes.")
    delta_una_bu = 2.0 * omega_ub
    delta_uob = float(
        gyrotriangle_defect_triangle_vertices_mp(
            cache.pts["UNA"], cache.pts["ONA"], cache.pts["BU+"]
        )["defect"]
    )
    omega_sum = omega_uo + omega_ob + omega_ub
    print(f"  {'omega_UO_stage':<22} {omega_uo:.12f}")
    print(f"  {'omega_OB_stage':<22} {omega_ob:.12f}")
    print(f"  {'omega_UB_stage':<22} {omega_ub:.12f}")
    print(f"  {'2*omega_OB':<22} {2.0 * omega_ob:.12f}")
    print(f"  {'delta_BU':<22} {basis['delta_BU']:.12f}")
    print(f"  {'2*omega_UB':<22} {delta_una_bu:.12f}")
    print(f"  {'delta_UOB':<22} {delta_uob:.12f}")
    print(f"  {'omega_UO+OB+UB':<22} {omega_sum:.12f}")
    print(f"  {'delta_UOB - sum':<22} {delta_uob - omega_sum:.12f}")
    gates.append(_gate("delta_BU = 2*omega_OB_stage", abs(basis["delta_BU"] - 2.0 * omega_ob) < 1e-12))
    una_bu_loop = float(
        gyrotriangle_defect_triangle_vertices_mp(
            cache.pts["UNA"], cache.pts["BU+"], cache.pts["BU-"]
        )["defect"]
    )
    print(f"  {'defect(UNA,BU+,BU-)':<22} {una_bu_loop:.12f}")
    gates.append(_gate("defect(UNA,BU+,BU-) = 2*omega_UB_stage", abs(una_bu_loop - delta_una_bu) < 1e-10))
    kprod = poincare_pair_products(t)
    tan_uo = math.tan(0.5 * omega_uo)
    tan_ob = math.tan(0.5 * omega_ob)
    tan_ub = math.tan(0.5 * omega_ub)
    print(f"  {'k_UNA':<22} {kprod['k_UNA']:.12f}")
    print(f"  {'k_ONA':<22} {kprod['k_ONA']:.12f}")
    print(f"  {'k_BU':<22} {kprod['k_BU']:.12f}")
    print(f"  {'tan(omega_UO/2)':<22} {tan_uo:.12f}")
    print(f"  {'k_UNA k_ONA':<22} {kprod['k_UNA_k_ONA']:.12f}")
    print(f"  {'tan(omega_OB/2)':<22} {tan_ob:.12f}")
    print(f"  {'k_ONA k_BU':<22} {kprod['k_ONA_k_BU']:.12f}")
    print(f"  {'tan(omega_UB/2)':<22} {tan_ub:.12f}")
    print(f"  {'k_BU k_UNA':<22} {kprod['k_BU_k_UNA']:.12f}")
    print(f"  {'tan_UB/tan_OB':<22} {tan_ub / tan_ob:.12f}")
    print(f"  {'k_UNA/k_ONA':<22} {kprod['k_UNA_over_k_ONA']:.12f}")
    gates.append(_gate("tan(omega_UO/2) = k(UNA) k(ONA)", abs(tan_uo - kprod["k_UNA_k_ONA"]) < 1e-12))
    gates.append(_gate("tan(omega_OB/2) = k(ONA) k(BU)", abs(tan_ob - kprod["k_ONA_k_BU"]) < 1e-12))
    gates.append(_gate("tan(omega_UB/2) = k(BU) k(UNA)", abs(tan_ub - kprod["k_BU_k_UNA"]) < 1e-12))
    gates.append(
        _gate(
            "tan(omega_UB/2)/tan(omega_OB/2) = k(UNA)/k(ONA)",
            abs(tan_ub / tan_ob - kprod["k_UNA_over_k_ONA"]) < 1e-12,
        )
    )

    print()
    print("12. PALINDROME AXIS STEERING")
    print("-" * 5)
    print("  Conjugating the BU dual-pole loop by the palindrome UNA->ONA->BU+")
    print("  ->UNA->BU+->ONA->UNA rotates its axis without changing the rotation")
    print("  angle. The table below tracks that axis turn as the conjugation is")
    print("  repeated n times; it advances by omega_UO per step and returns to")
    print("  the original axis every 2pi/omega_UO steps.")
    axis_data, axis_gates = axis_transport_certificate(cache, conn)
    gates.extend(axis_gates)
    for key, value in axis_data.items():
        print(f"  {key:<28} {value:.12f}")
    print(f"  axis_transport_deg            {math.degrees(axis_data['axis_transport']):.12f}")
    chain = palindrome_conjugation_chain(conn, omega_uo)
    print(f"  {'n':>3} {'theta':>14} {'axis_turn':>14} {'n*omega_UO':>14}")
    for row in chain["rows"]:
        print(
            f"  {int(row['n']):3d} {row['theta']:14.12f} {row['turn']:14.12f} "
            f"{row['n_omega_uo']:14.12f}"
        )
    print(f"  2pi/omega_UO                 {chain['period_2pi_over_omega_uo']:.12f}")
    gates.append(
        _gate(
            "conjugation n=1 |axis turn| = omega_UO",
            abs(abs(chain["rows"][1]["turn"]) - omega_uo) < 1e-10,
        )
    )
    gates.append(
        _gate(
            "conjugation n=2 |axis turn| = 2 omega_UO",
            abs(abs(chain["rows"][2]["turn"]) - 2.0 * omega_uo) < 1e-10,
        )
    )
    gates.append(
        _gate(
            "conjugated angle stays delta_BU",
            all(abs(row["theta"] - basis["delta_BU"]) < 1e-10 for row in chain["rows"]),
        )
    )

    print()
    print("13. CLOSURE RESPONSE")
    print("-" * 5)
    print("  The BU dual-pole loop is an intrinsic pure rotation for each")
    print("  admissible balance amplitude m. This section measures how the")
    print("  holonomy angle, closure ratio rho, and aperture gap Delta vary")
    print("  with m around the physical threshold m_a. rho0 is the closure")
    print("  ratio in the limit m_a -> 0, and the baseline gap 1 - rho0")
    print("  decomposes into the finite BU closure and the secant Delta.")
    response = closure_response(t, basis)
    for key, value in response.items():
        print(f"  {key:<28} {value:.12f}")
    gates.append(_gate("rho0 = d(delta_BU)/d(m_a) at m_a=0", abs(response["rho0"] - response["rho0_from_d0"]) < 1e-10))
    gates.append(
        _gate(
            "aperture decomposition 1-rho0 = (rho-rho0) + (1-rho)",
            abs(response["baseline_gap"] - response["finite_BU_closure"] - response["Delta_secant"]) < 1e-12,
        )
    )
    gates.append(
        _gate(
            "Delta_secant = Delta_tangent + nonlinear_closure_gain",
            abs(
                response["Delta_secant"]
                - response["Delta_tangent"]
                - response["nonlinear_closure_gain"]
            )
            < 1e-12,
        )
    )
    gates.append(
        _gate(
            "baseline_gap = finite_BU_closure + nonlinear_closure_gain + Delta_tangent",
            abs(
                response["baseline_gap"]
                - response["finite_BU_closure"]
                - response["nonlinear_closure_gain"]
                - response["Delta_tangent"]
            )
            < 1e-12,
        )
    )
    ratio_from_k = (float(t.theta_ona) / float(t.m_a)) * (
        response["dlogk_ona"] / response["dlogk_m"]
    )
    print(f"  {'(theta/m)*(dlogk_ona/dlogk_m)':<28} {ratio_from_k:.12f}")
    print(f"  {'Delta_secant/Delta_tangent':<28} {response['Delta_secant'] / response['Delta_tangent']:.12f}")
    gates.append(
        _gate(
            "elasticity_ratio = (theta/m)*(dlogk_ona/dlogk_m)",
            abs(response["elasticity_ratio"] - ratio_from_k) < 1e-10,
        )
    )

    print()
    print("14. EQUAL-SPEED WIGNER RESPONSE")
    print("-" * 5)
    print("  omega0 is the Wigner rotation produced by two equal-speed boosts.")
    print("  Its derivatives with respect to the common boost parameter beta")
    print("  and the directional separation angle theta are computed numerically")
    print("  and checked against their closed-form expressions. Their sum is 1.")
    eq = equal_speed_response(t)
    print(f"  {'omega0':<28} {basis['omega0']:.12f}")
    print(f"  {'omega_UO_stage':<28} {omega_uo:.12f}")
    for key, value in eq.items():
        print(f"  {key:<28} {value:.12f}")
    gates.append(
        _gate(
            "d(omega0)/d beta = (12*sqrt(2)-4)/17",
            abs(eq["d_omega_d_beta"] - eq["d_omega_d_beta_closed"]) < 1e-12,
        )
    )
    gates.append(
        _gate(
            "d(omega0)/d theta = (21-12*sqrt(2))/17",
            abs(eq["d_omega_d_theta"] - eq["d_omega_d_theta_closed"]) < 1e-12,
        )
    )
    gates.append(_gate("d(omega0)/d beta + d(omega0)/d theta = 1", abs(eq["d_sum"] - 1.0) < 1e-12))

    print()
    print("15. EQUIVALENT CIRCULAR THOMAS")
    print("-" * 5)
    print("  Every rotation angle can be replaced by a circular motion with the")
    print("  same angle, and that circular motion corresponds to a boost of some")
    print("  equivalent speed. The table below gives that equivalent speed beta")
    print("  and the fraction of a full turn the rotation represents.")
    angles = {
        "omega_UO_stage": omega_uo,
        "omega_OB_stage": omega_ob,
        "omega_UB_stage": omega_ub,
        "delta_BU": basis["delta_BU"],
        "omega_equal_speed": basis["omega0"],
        "phi_SU2": basis["phi_SU2"],
        "theta_lab_BU": conn["lab"]["relative_boost_BU"],
        "theta_lab_pal": conn["lab"]["relative_boost_pal"],
        "theta_chart_complete": conn["chart"]["omega_chart_complete_BU"],
        "theta_chart_sph_z": conn["chart"]["omega_chart_sph_BU"],
    }
    print(f"  {'name':<24} {'theta':>12} {'turn frac':>12} {'gamma_eq':>12} {'beta_eq':>12}")
    for name, angle in angles.items():
        row = equivalent_circular_thomas(float(angle))
        print(
            f"  {name:<24} {row['theta']:12.8f} {row['turn_fraction']:12.8f} "
            f"{row['gamma']:12.8f} {row['beta']:12.8f}"
        )

    print()
    print("16. INERTIAL-FRAME WORD CLOSURE")
    print("-" * 5)
    print("  A word is a closed sequence of stage points. Composing the boosts of")
    print("  that word in one inertial frame leaves a net boost and a net")
    print("  rotation. The word closes to a pure rotation when the net boost")
    print("  vanishes; the status column records that condition.")
    lab_status: dict[str, LabWordClosure] = {}
    print(f"  {'loop':<18} {'theta':>14} {'net_boost':>14} {'eta':>12}  status")
    for name, path in NAMED_LOOPS:
        row = lab_word_closure(cache, path)
        lab_status[name] = row
        print(
            f"  {name:<18} {row['theta']:14.12f} {row['net_boost']:14.6e} "
            f"{row['eta']:12.8f}  {row['status']}"
        )
    reach = conn.get("reach", {})
    n_closed = int(reach.get("n_lab_closed", -1))
    n_walks = int(reach.get("walk_count", 0))
    if n_closed >= 0:
        print(f"  Among the {n_walks} enumerated words of length two through five, "
              f"{n_closed} close with vanishing net boost:")
        for pth in reach.get("lab_closed_paths", []):
            print(f"    {pth}")
    gates.append(_gate("named BU and pal lab words are open", lab_status["BU"]["status"] == "OPEN_BOOST_WORD_ROTATION" and lab_status["pal"]["status"] == "OPEN_BOOST_WORD_ROTATION"))
    f_pal = float(lab_status["pal"]["theta"]) - basis["delta_BU"]
    f_bu = float(lab_status["BU"]["theta"]) - basis["delta_BU"]
    f_uo = float(lab_status["outback_UNA_ONA"]["theta"])
    f_ob = float(lab_status["outback_ONA_BUp"]["theta"])
    f_ub = float(lab_status["outback_UNA_BUp"]["theta"])
    hypot_f = math.sqrt(f_uo * f_uo + f_ob * f_ob + f_ub * f_ub)
    print(f"  F = theta_inert - delta_BU on named loops")
    print(f"  F_pal                                      {f_pal:.12f}")
    print(f"  F_BU                                       {f_bu:.12f}")
    print(f"  hypot(F_UNA-ONA, F_ONA-BU, F_UNA-BU)      {hypot_f:.12f}")

    print()
    print("17. COMPACT-HYPERBOLIC BRIDGE")
    print("-" * 5)
    print("  phi_SU2 is the SU(2) commutator angle at the constitutional stage")
    print("  angles. Three dual-pole holonomies give 3 delta_BU. epsilon_CH is")
    print("  their difference. The relative angles compare the compact fiber")
    print("  element C with three aligned BU rotations, with U_BU cubed, and")
    print("  with three orthogonal delta rotations.")
    bridge = compact_hyperbolic_bridge(basis, float(t.m_a))
    for key, value in bridge.items():
        print(f"  {key:<28} {value:.12f}")
    mat = compact_hyperbolic_matrix(cache, basis)
    print(f"  axis_compact                {_fmt_axis(mat['axis_compact'])}")
    print(f"  axis_BU                     {_fmt_axis(mat['axis_BU'])}")
    print(f"  {'axis_dot':<28} {mat['axis_dot']:.12f}")
    print(f"  {'rel(C, U(n_C, 3 delta_BU))':<28} {mat['rel_aligned_3delta']:.12f}")
    print(f"  {'rel(C, U_BU^3)':<28} {mat['rel_BU3']:.12f}")
    print(f"  {'rel(C, Ux Uy Uz delta)':<28} {mat['rel_xyz_delta']:.12f}")
    gates.append(
        _gate(
            "aligned 3 delta_BU relative angle = |phi_SU2 - 3 delta_BU|",
            abs(mat["rel_aligned_3delta"] - mat["scalar_residual"]) < 1e-10,
        )
    )

    print()
    print("18. REALIZATION LOCK")
    print("-" * 5)
    print("  The pair angles are recomputed from three assignments of the stage")
    print("  numbers to Einstein speeds. einstein_beta uses the thresholds as")
    print("  speeds. as_rapidity uses tanh of those numbers. UNA_angle_as_beta")
    print("  substitutes the UNA angle theta_una for the UNA speed. CS cannot")
    print("  be an Einstein speed because theta_cs = pi/2 exceeds 1.")
    maps = [
        ("einstein_beta", t.u_p, t.theta_ona, t.m_a),
        ("as_rapidity", mp.tanh(t.u_p), mp.tanh(t.theta_ona), mp.tanh(t.m_a)),
        ("UNA_angle_as_beta", t.theta_una, t.theta_ona, t.m_a),
    ]
    print(
        f"  {'map':<20} {'omega_UO':>14} {'omega_OB':>14} {'omega_UB':>14} {'delta_BU':>14}"
    )
    realized: dict[str, dict[str, float]] = {}
    for name, bu_una, bu_ona, bu_bu in maps:
        row = pair_angles_from_betas(bu_una, bu_ona, bu_bu)
        realized[name] = row
        print(
            f"  {name:<20} {row['omega_UO']:14.12f} {row['omega_OB']:14.12f} "
            f"{row['omega_UB']:14.12f} {row['delta_BU']:14.12f}"
        )
    print(f"  {'omega0 TW(u_p,u_p; o_p)':<20} {basis['omega0']:14.12f}")
    eq_orth_ona = float(tw_angle_exact(t.theta_ona, mp.pi / 2))
    print(f"  {'TW(o_p,o_p; pi/2)':<20} {eq_orth_ona:14.12f}")
    gates.append(
        _gate(
            "einstein_beta delta_BU = measured delta_BU",
            abs(realized["einstein_beta"]["delta_BU"] - basis["delta_BU"]) < 1e-12,
        )
    )
    gates.append(
        _gate(
            "as_rapidity delta_BU != einstein_beta delta_BU",
            abs(realized["as_rapidity"]["delta_BU"] - basis["delta_BU"]) > 1e-6,
        )
    )
    gates.append(
        _gate(
            "UNA_angle_as_beta makes omega_UB = omega_OB",
            abs(realized["UNA_angle_as_beta"]["omega_UB"] - realized["UNA_angle_as_beta"]["omega_OB"])
            < 1e-12,
        )
    )
    gates.append(
        _gate(
            "einstein_beta omega_UB != omega_OB",
            abs(realized["einstein_beta"]["omega_UB"] - realized["einstein_beta"]["omega_OB"]) > 1e-6,
        )
    )
    gates.append(
        _gate(
            "UNA_angle_as_beta omega_UO = TW(o_p, o_p; pi/2)",
            abs(realized["UNA_angle_as_beta"]["omega_UO"] - eq_orth_ona) < 1e-12,
        )
    )
    gates.append(
        _gate(
            "einstein_beta omega_UO != omega0",
            abs(realized["einstein_beta"]["omega_UO"] - basis["omega0"]) > 0.01,
        )
    )

    print()
    print("19. CGM UNITS")
    print("-" * 5)
    print("  Geometric time is t_aperture = m_a. The horizon length is")
    print("  L_horizon = sqrt(2 pi). Q_G = L/t is the quantum-gravity invariant.")
    print("  CS, UNA, and ONA actions are thresholds in units of m_a.")
    print("  S_BU is the aperture amplitude itself.")
    print("  S = theta / t_aperture is an angle as a rate on that clock.")
    print("  n = theta / Delta is the same angle in aperture ticks.")
    units = cgm_units_map(t, basis, omega_uo, omega_ob, omega_ub, response)
    print(f"  {'t_aperture = m_a':<28} {units['t_aperture']:.12f}")
    print(f"  {'L_horizon = sqrt(2 pi)':<28} {units['L_horizon']:.12f}")
    print(f"  {'Q_G = L/t':<28} {units['Q_G']:.12f}")
    print(f"  {'4 pi':<28} {4.0 * math.pi:.12f}")
    print(f"  {'Q_G m_a^2':<28} {units['Q_G_m_a2']:.12f}")
    print(f"  {'1/2':<28} {0.5:.12f}")
    print(f"  {'S_CS = (pi/2)/m_a':<28} {units['S_CS']:.12f}")
    print(f"  {'S_UNA = u_p/m_a':<28} {units['S_UNA']:.12f}")
    print(f"  {'S_ONA = o_p/m_a = K_QG':<28} {units['S_ONA']:.12f}")
    print(f"  {'S_BU = m_a (aperture amplitude)':<28} {units['S_BU']:.12f}")
    print(f"  {'S_GUT (eta=1)':<28} {units['S_GUT']:.12f}")
    print(f"  {'UNA-ONA lift o_p-u_p':<28} {units['UNA_ONA_lift']:.12f}")
    print(f"  {'EM duality atan(S_ONA/S_UNA) deg':<28} {units['EM_duality_deg']:.12f}")
    print(f"  {'alpha0 = delta_BU^4 / m_a':<28} {units['alpha0']:.12f}")
    print(f"  {'48 Delta':<28} {units['forty8_Delta']:.12f}")
    print(f"  {'48 Delta_tangent':<28} {units['forty8_Delta_tangent']:.12f}")
    print(f"  {'1/Delta':<28} {units['one_over_Delta']:.12f}")
    print(f"  {'Delta/Delta_tangent':<28} {units['Delta_over_Delta_tangent']:.12f}")
    print(f"  {'2^Delta':<28} {units['two_to_Delta']:.12f}")
    print(f"  {'name':<16} {'theta':>14} {'S=theta/t_ap':>14} {'n=theta/Delta':>14}")
    for name in ("omega_UO", "omega_OB", "omega_UB", "delta_BU", "delta_UNA_BU", "phi_SU2"):
        theta = {
            "omega_UO": omega_uo,
            "omega_OB": omega_ob,
            "omega_UB": omega_ub,
            "delta_BU": basis["delta_BU"],
            "delta_UNA_BU": 2.0 * omega_ub,
            "phi_SU2": basis["phi_SU2"],
        }[name]
        print(
            f"  {name:<16} {theta:14.12f} {units[f'S_{name}']:14.12f} {units[f'n_{name}']:14.12f}"
        )
    print(f"  {'Omega_delta_BU = rho':<28} {units['Omega_delta_BU']:.12f}")
    print(f"  {'rho':<28} {basis['rho']:.12f}")
    gates.append(_gate("Q_G = 4 pi", abs(units["Q_G"] - 4.0 * math.pi) < 1e-12))
    gates.append(_gate("Q_G m_a^2 = 1/2", abs(units["Q_G_m_a2"] - 0.5) < 1e-12))
    gates.append(_gate("K_QG = S_ONA", abs(units["K_QG"] - units["S_ONA"]) < 1e-15))
    gates.append(_gate("delta_BU / t_aperture = rho", abs(units["Omega_delta_BU"] - basis["rho"]) < 1e-12))
    gates.append(
        _gate(
            "UNA-ONA lift = o_p - u_p",
            abs(units["UNA_ONA_lift"] - (float(t.theta_ona) - float(t.u_p))) < 1e-15,
        )
    )

    gates.extend(
        print_physical_readings(
            cache,
            kin,
            basis,
            response,
            mat,
            float(t.m_a),
            omega_uo,
            omega_ob,
            omega_ub,
            t,
        )
    )

    print()
    print("21. CONNECTION CLASSIFICATION")
    print("-" * 5)
    print("  Each transport rule on the named loops is classified by whether")
    print("  the product is a pure rotation on the mass shell, a boost word")
    print("  with leftover velocity, a coordinate readout, or a compact fiber")
    print("  commutator.")
    print("  Fermi-Walker / geodesic     CLOSED_MASS_SHELL_HOLONOMY")
    print(f"  inertial-frame BU           {lab_status['BU']['status']}")
    print(f"  inertial-frame pal          {lab_status['pal']['status']}")
    print("  chart complete              CLOSED_MASS_SHELL_HOLONOMY")
    print("  chart spherical             SPHERICAL_COORDINATE_READOUT")
    print("  SU2                         COMPACT_FIBER_COMMUTATOR")
    print()

    state["mechanics"] = {
        "kinematics": kin,
        "pair_precessions": pair_pre,
        "delta_UNA_BU": delta_una_bu,
        "delta_UOB": delta_uob,
        "axis": axis_data,
        "response": response,
        "equal_speed": eq,
        "bridge": bridge,
        "lab_status": lab_status,
        "compact_matrix": mat,
        "pal_chain_period": chain["period_2pi_over_omega_uo"],
        "units": units,
    }
    return gates


def run_part2(state: dict[str, Any]) -> list[tuple[str, bool]]:
    return run_mechanics(state)


if __name__ == "__main__":
    failed = [name for name, ok in run_mechanics() if not ok]
    if failed:
        raise SystemExit(1)
