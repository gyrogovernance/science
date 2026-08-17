#!/usr/bin/env python3
"""
CGM precession analysis, part 2.

Measures the three-axis stage-pair basis, palindrome axis steering, secant vs
tangent closure, circular Thomas calibration, lab-word net-boost closure, and
compact vs hyperbolic holonomy. Prints the corrected metric ontology.

Companions: cgm_precession_analysis_{common,1,run}.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import mpmath as mp

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
for _p in (_EXP, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cgm_holonomy_analysis_common import (
    gyrotriangle_defect_mp,
    gyrotriangle_defect_triangle_vertices_mp,
    gyr_matrix_ungar_mp,
    poincare_radius_from_beta,
    radial_coordinates_mp,
    rotation_angle_atan2_mp,
    tw_angle_exact,
)
from cgm_precession_analysis_1 import (
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
from cgm_precession_analysis_common import (
    MetricRecord,
    PhysicalStatus,
    classify,
    physical_map_for,
    used_downstream_for,
)

LAB_CLOSE_TOL = 1e-10


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


def precession_triad(cache: EdgeCache) -> dict[str, dict[str, Any]]:
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


def _k_signed(beta: mp.mpf) -> mp.mpf:
    b = mp.mpf(beta)
    if b == 0:
        return mp.mpf(0)
    return mp.sign(b) * poincare_radius_from_beta(abs(b))


def delta_formula(theta_ona: mp.mpf, m_a: mp.mpf) -> mp.mpf:
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
    }


def equal_speed_response(t: Any) -> dict[str, float]:
    beta = mp.mpf(t.u_p)
    theta = mp.mpf(t.theta_ona)
    d_beta = mp.diff(lambda x: tw_angle_exact(x, theta), beta)
    d_theta = mp.diff(lambda x: tw_angle_exact(beta, x), theta)
    s = abs(float(d_beta)) + abs(float(d_theta))
    return {
        "d_omega_d_beta": float(d_beta),
        "d_omega_d_theta": float(d_theta),
        "share_beta": abs(float(d_beta)) / s,
        "share_theta": abs(float(d_theta)) / s,
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


def lab_word_closure(cache: EdgeCache, path: tuple[str, ...]) -> dict[str, float | str]:
    ds = [cache.disp[(path[i], path[i + 1])] for i in range(len(path) - 1)]
    fac = factor_boost_word_lab_mp(ds)
    boost_residual = _norm3(fac["u_final"])
    return {
        "theta": float(fac["theta"]),
        "net_boost": boost_residual,
        "status": (
            "CLOSED_LAB_PRECESSION" if boost_residual < LAB_CLOSE_TOL else "OPEN_BOOST_WORD_ROTATION"
        ),
    }


def name_can_bins(spec_can: list[tuple[float, int, int]], delta_bu: float) -> list[tuple[str, float]]:
    named: list[tuple[str, float]] = []
    used: set[str] = set()
    for ang, _mult, _L in spec_can:
        if abs(ang) < 1e-8:
            key = "can_bin_zero"
        elif abs(ang - delta_bu) < 1e-8:
            key = "can_bin_ona_bu_dualpole"
        elif abs(ang - 0.1668277883) < 1e-8:
            key = "can_bin_una_bu_dualpole"
        elif abs(ang - 0.4204750817) < 1e-8:
            key = "can_bin_una_ona_bu"
        elif abs(ang - 0.4127190541) < 1e-8:
            key = "can_bin_cyc4"
        elif abs(ang - 0.2567128344) < 1e-8:
            key = "can_bin_L4_secondary"
        else:
            key = f"can_bin_{ang:.6f}"
        if key in used or key == "can_bin_ona_bu_dualpole":
            continue
        used.add(key)
        named.append((key, ang))
    return named


def _rec(name: str, value: float) -> MetricRecord:
    layer, law, inv, status = classify(name)
    pmap = physical_map_for(name)
    return MetricRecord(
        name=name,
        value=float(value),
        origin_layer=layer,
        transport_law=law,
        invariant_type=inv,
        physical_status=status if pmap.status == PhysicalStatus.UNKNOWN else pmap.status,
        used_downstream=used_downstream_for(name),
        physical_map=pmap,
    )


def build_metrics(state: dict[str, Any]) -> list[MetricRecord]:
    p = state["priors"]
    b = state["basis"]
    conn = state["conn"]
    mech = state.get("mechanics", {})
    kin = mech.get("kinematics", {})
    triad = mech.get("triad", {})
    response = mech.get("response", {})
    axis = mech.get("axis", {})
    bridge = mech.get("bridge", {})
    lab_rows = {r["name"]: r for r in state.get("lab_rows", [])}
    out: list[MetricRecord] = []

    for name in ("u_p", "o_p", "m_a", "theta_cs"):
        out.append(_rec(name, p[name]))
    for stage, key in (("UNA", "beta_UNA"), ("ONA", "beta_ONA"), ("BU", "beta_BU")):
        if stage in kin:
            out.append(_rec(key, kin[stage]["beta"]))

    if triad:
        out.append(_rec("omega_UO_stage", triad["UNA_ONA"]["defect"]))
        out.append(_rec("omega_OB_stage", triad["ONA_BU"]["defect"]))
        out.append(_rec("omega_UB_stage", triad["BU_UNA"]["defect"]))
    if "delta_UNA_BU" in mech:
        out.append(_rec("delta_UNA_BU", mech["delta_UNA_BU"]))
    if "delta_UOB" in mech:
        out.append(_rec("delta_UOB", mech["delta_UOB"]))
    if axis:
        out.append(_rec("axis_transport", axis["axis_transport"]))

    for name in ("omega_corner", "delta_BU", "rho", "Delta", "phi_SU2", "omega0", "rho0", "two_1_rho0"):
        out.append(_rec(name, b[name]))
    out.append(_rec("omega_equal_speed_UNA", b["omega0"]))

    if response:
        for name in (
            "baseline_gap",
            "finite_BU_closure",
            "rho_secant",
            "Delta_secant",
            "rho_tangent",
            "Delta_tangent",
            "nonlinear_closure_gain",
            "d_delta_d_theta",
            "elasticity_theta",
            "elasticity_m",
        ):
            out.append(_rec(name, response[name]))
    if bridge:
        out.append(_rec("compact_hyperbolic_residual", bridge["residual"]))
        out.append(_rec("sigma_compact", bridge["residual_over_ma"]))

    for key, ang in name_can_bins(state.get("spec_can", []), b["delta_BU"]):
        out.append(_rec(key, ang))

    def lab_f(name: str, fallback: float) -> float:
        rec = lab_rows.get(name)
        return float(rec["F"]) if rec else fallback

    f_bu = conn["lab"]["relative_boost_BU"] - conn["can"]["origin_gyr_BU"]
    f_pal = conn["lab"]["relative_boost_pal"] - conn["can"]["origin_gyr_pal"]
    out.append(_rec("theta_lab_BU", conn["lab"]["relative_boost_BU"]))
    out.append(_rec("theta_lab_pal", conn["lab"]["relative_boost_pal"]))
    out.append(_rec("F_BU", f_bu))
    out.append(_rec("F_pal", f_pal))
    out.append(_rec("F_outback_ONA_BUp", lab_f("outback_ONA_BUp", f_bu)))
    out.append(_rec("F_outback_UNA_ONA", lab_f("outback_UNA_ONA", 0.0)))
    out.append(_rec("F_outback_UNA_BUp", lab_f("outback_UNA_BUp", 0.0)))
    out.append(_rec("lab_spectrum_count_L5", float(state.get("n_spec_lab", 0))))

    ch = conn["chart"]
    out.append(_rec("omega_chart_complete", ch["omega_chart_complete_BU"]))
    out.append(_rec("G_z_BU", ch["offset_BU"]))
    out.append(_rec("G_z_pal", ch["offset_pal"]))
    out.append(_rec("G_diag_BU", ch.get("G_diag_BU", 0.0)))
    out.append(_rec("G_diag_minus_G_z", ch.get("G_diag_BU", 0.0) - ch["offset_BU"]))
    return out


def print_ontology(metrics: list[MetricRecord]) -> None:
    print("19. METRIC ONTOLOGY")
    print("-" * 5)
    print(
        f"  {'name':<28} {'origin_layer':<26} {'transport':<10} "
        f"{'invariant':<20} {'status':<32} down"
    )
    for m in metrics:
        print(
            f"  {m.name:<28} {m.origin_layer.value:<26} {m.transport_law.value:<10} "
            f"{m.invariant_type.value:<20} {m.physical_status.value:<32} "
            f"{'yes' if m.used_downstream else 'no'}"
        )
    print()
    down = [m.name for m in metrics if m.used_downstream]
    unk = [m.name for m in metrics if m.physical_status == PhysicalStatus.UNKNOWN]
    print("  used_downstream: " + ", ".join(down))
    if unk:
        print("  unknown: " + ", ".join(unk))
    print()


def print_mapping(metrics: list[MetricRecord]) -> None:
    print("20. PHYSICAL MAPPING")
    print("-" * 5)
    for m in metrics:
        pm = m.physical_map
        print(f"  {m.name:<28} {m.value:12.8g}  {pm.status.value}")
        print(f"    measure   {pm.plain_measurement}")
        print(f"    physics   {pm.candidate_physics}")
        if pm.alternatives:
            print(f"    alt       {'; '.join(pm.alternatives)}")
    print()


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
    print("10. THREE-AXIS PRECESSION TRIAD")
    print("-" * 5)
    triad = precession_triad(cache)
    print(f"  {'pair':<10} {'defect':>14} {'gyr_origin':>14} {'gyr_stage':>14}  axis_origin")
    for name, row in triad.items():
        print(
            f"  {name:<10} {row['defect']:14.12f} {row['theta']:14.12f} "
            f"{row['theta_stage_gyr']:14.12f}  {_fmt_axis(row['axis'])}"
        )
        gates.append(
            _gate(
                f"{name}: origin-gyr angle = gyrotriangle defect",
                abs(row["theta"] - row["defect"]) < 1e-10,
            )
        )
    axes = [triad[k]["axis"] for k in ("UNA_ONA", "ONA_BU", "BU_UNA")]
    gates.append(
        _gate(
            "three origin-pair precession axes mutually orthogonal",
            all(abs(_axis_dot(axes[i], axes[j])) < 1e-10 for i in range(3) for j in range(i + 1, 3)),
        )
    )
    omega_uo = triad["UNA_ONA"]["defect"]
    omega_ob = triad["ONA_BU"]["defect"]
    omega_ub = triad["BU_UNA"]["defect"]
    gates.append(_gate("omega_OB_stage = omega_corner", abs(omega_ob - basis["omega_corner"]) < 1e-12))
    gates.append(_gate("omega0 != omega_UO_stage", abs(basis["omega0"] - omega_uo) > 0.01))

    print()
    print("  directed edges (triad generators or I)")
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
    print(f"  {'bin':<12} {'word':<22} {'theta':>14}")
    for name, path in recon:
        w = generator_word(cache, path, omegas)
        print(f"  {name:<12} {w['word']:<22} {w['theta']:14.12f}")
    gates.append(
        _gate(
            "cyc4 = generator word UNA-ONA-BU+-BU--UNA",
            abs(generator_word(cache, recon[0][1], omegas)["theta"] - 0.4127190541) < 1e-8,
        )
    )
    gates.append(
        _gate(
            "L4_crossed = generator word UNA-BU+-ONA-BU--UNA",
            abs(generator_word(cache, recon[1][1], omegas)["theta"] - 0.2567128344) < 1e-8,
        )
    )

    print()
    print("11. DUAL-POLE CLOSURE")
    print("-" * 5)
    delta_una_bu = 2.0 * omega_ub
    delta_uob = float(
        gyrotriangle_defect_triangle_vertices_mp(
            cache.pts["UNA"], cache.pts["ONA"], cache.pts["BU+"]
        )["defect"]
    )
    triad_sum = omega_uo + omega_ob + omega_ub
    print(f"  {'omega_UO_stage':<22} {omega_uo:.12f}")
    print(f"  {'omega_OB_stage':<22} {omega_ob:.12f}")
    print(f"  {'omega_UB_stage':<22} {omega_ub:.12f}")
    print(f"  {'2*omega_OB':<22} {2.0 * omega_ob:.12f}")
    print(f"  {'delta_BU':<22} {basis['delta_BU']:.12f}")
    print(f"  {'2*omega_UB':<22} {delta_una_bu:.12f}")
    print(f"  {'delta_UOB':<22} {delta_uob:.12f}")
    print(f"  {'omega_UO+OB+UB':<22} {triad_sum:.12f}")
    print(f"  {'delta_UOB - sum':<22} {delta_uob - triad_sum:.12f}")
    gates.append(_gate("delta_BU = 2*omega_OB_stage", abs(basis["delta_BU"] - 2.0 * omega_ob) < 1e-12))
    una_bu_loop = float(
        gyrotriangle_defect_triangle_vertices_mp(
            cache.pts["UNA"], cache.pts["BU+"], cache.pts["BU-"]
        )["defect"]
    )
    print(f"  {'defect(UNA,BU+,BU-)':<22} {una_bu_loop:.12f}")
    gates.append(_gate("defect(UNA,BU+,BU-) = 2*omega_UB_stage", abs(una_bu_loop - delta_una_bu) < 1e-10))

    print()
    print("12. PALINDROME AXIS STEERING")
    print("-" * 5)
    axis_data, axis_gates = axis_transport_certificate(cache, conn)
    gates.extend(axis_gates)
    for key, value in axis_data.items():
        print(f"  {key:<28} {value:.12f}")
    print(f"  axis_transport_deg            {math.degrees(axis_data['axis_transport']):.12f}")

    print()
    print("13. CLOSURE RESPONSE")
    print("-" * 5)
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

    print()
    print("14. EQUAL-SPEED WIGNER RESPONSE")
    print("-" * 5)
    eq = equal_speed_response(t)
    print(f"  {'omega0':<28} {basis['omega0']:.12f}")
    print(f"  {'omega_UO_stage':<28} {omega_uo:.12f}")
    for key, value in eq.items():
        print(f"  {key:<28} {value:.12f}")

    print()
    print("15. EQUIVALENT CIRCULAR THOMAS")
    print("-" * 5)
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
    print("16. LAB WORD CLOSURE")
    print("-" * 5)
    lab_status: dict[str, dict[str, float | str]] = {}
    print(f"  {'loop':<18} {'theta':>14} {'net_boost':>14}  status")
    for name, path in NAMED_LOOPS:
        row = lab_word_closure(cache, path)
        lab_status[name] = row
        print(f"  {name:<18} {row['theta']:14.12f} {row['net_boost']:14.6e}  {row['status']}")
    reach = conn.get("reach", {})
    n_closed = int(reach.get("n_lab_closed", -1))
    n_walks = int(reach.get("walk_count", 0))
    if n_closed >= 0:
        print(f"  census L=2..5: closed={n_closed}/{n_walks}")
        for pth in reach.get("lab_closed_paths", []):
            print(f"    {pth}")
    gates.append(_gate("named BU and pal lab words are open", lab_status["BU"]["status"] == "OPEN_BOOST_WORD_ROTATION" and lab_status["pal"]["status"] == "OPEN_BOOST_WORD_ROTATION"))

    print()
    print("17. COMPACT-HYPERBOLIC BRIDGE")
    print("-" * 5)
    bridge = compact_hyperbolic_bridge(basis, float(t.m_a))
    for key, value in bridge.items():
        print(f"  {key:<28} {value:.12f}")

    print()
    print("18. CONNECTION CLASSIFICATION")
    print("-" * 5)
    print("  canonical        CLOSED_MASS_SHELL_HOLONOMY")
    print(f"  lab BU           {lab_status['BU']['status']}")
    print(f"  lab pal          {lab_status['pal']['status']}")
    print("  chart complete   CLOSED_MASS_SHELL_HOLONOMY")
    print("  chart spherical  SPHERICAL_COORDINATE_READOUT")
    print("  SU2              COMPACT_FIBER_COMMUTATOR")
    print()

    state["mechanics"] = {
        "kinematics": kin,
        "triad": triad,
        "delta_UNA_BU": delta_una_bu,
        "delta_UOB": delta_uob,
        "axis": axis_data,
        "response": response,
        "equal_speed": eq,
        "bridge": bridge,
        "lab_status": lab_status,
    }
    return gates


def run_ontology(state: dict[str, Any]) -> None:
    metrics = build_metrics(state)
    print_ontology(metrics)
    print_mapping(metrics)
    print("21. NOTES")
    print("-" * 5)
    print("  theory: cgm_precession_analysis_theory_notes.txt")
    print()


def run_part2(state: dict[str, Any]) -> list[tuple[str, bool]]:
    gates = run_mechanics(state)
    run_ontology(state)
    return gates


if __name__ == "__main__":
    failed = [name for name, ok in run_mechanics() if not ok]
    if failed:
        raise SystemExit(1)
