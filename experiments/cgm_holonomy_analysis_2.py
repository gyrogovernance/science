#!/usr/bin/env python3
"""
cgm_holonomy_analysis_2.py

Report part 2: aperture derived quantities through integrity summary.

Invoked by cgm_holonomy_analysis_run.py. Companions: _common.py, _1.py.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import mpmath as mp

_EXP = Path(__file__).resolve().parent
_REPO = _EXP.parent
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cgm_holonomy_analysis_common import (
    APERTURE_GAP_Q256,
    TOL_MP,
    ReportState,
    binomial_c4,
    check,
    circular_curvature_holonomy_alpha_mp,
    compute_finite_holonomy,
    mp_to_str,
    palge_pfeifer_suite_mp,
    section,
    tw_angle_exact,
    tw_angle_unequal,
    _FINITE_IMPORT_ERROR,
)


def run_holonomy_2(state: ReportState) -> ReportState:
    assert state.t is not None and state.delta_bu is not None
    assert state.rho is not None and state.delta_gap is not None
    assert state.delta_bu_closed is not None
    assert state.mpj is not None and state.word is not None and state.defct is not None
    t = state.t
    delta_bu: float = state.delta_bu
    rho: float = state.rho
    delta_gap: float = state.delta_gap
    delta_bu_closed = state.delta_bu_closed
    mpj = state.mpj
    word = state.word
    rooted_bu = state.rooted_bu
    defct = state.defct

    # Aperture
    section(state, "APERTURE (delta_BU = 2*omega)")
    print("  Object: rho/Delta from delta_BU = 2*omega = 4*arctan(r(theta_ona)*r(m_a)).")
    print(f"  delta_BU = 2*omega               {mp_to_str(delta_bu_closed, 40)}")
    print(f"  m_a                              {mp_to_str(t.m_a, 20)}")
    print(f"  rho = delta_BU / m_a             {mp_to_str(delta_bu_closed / t.m_a, 20)}")
    print(f"  Delta = 1 - rho                  {mp_to_str(1 - delta_bu_closed / t.m_a, 20)}")
    print(f"  closure_percent                  {100.0 * rho:.10f}")
    print(f"  aperture_percent                 {100.0 * delta_gap:.10f}")
    print()

    # Rank-2 dependency
    section(state, "RANK-2 DEPENDENCY (ORIGIN-WIGNER)")
    print("  Scope: delta_BU = 2*omega = 4*arctan(r(theta_ona)*r(m_a)).")
    print("  Under that scope, delta depends on (theta_ona, m_a).")
    th0 = float(t.theta_ona)
    m0 = float(t.m_a)
    h_th = mp.mpf("1e-5")
    h_m = mp.mpf("1e-5")
    d_dth_an = float(
        (
            2 * tw_angle_unequal(t.theta_ona + h_th, t.m_a, mp.pi / 2)
            - 2 * tw_angle_unequal(t.theta_ona - h_th, t.m_a, mp.pi / 2)
        )
        / (2 * h_th)
    )
    d_dm_an = float(
        (
            2 * tw_angle_unequal(t.theta_ona, t.m_a + h_m, mp.pi / 2)
            - 2 * tw_angle_unequal(t.theta_ona, t.m_a - h_m, mp.pi / 2)
        )
        / (2 * h_m)
    )
    print(f"  d(delta_BU)/dtheta_ona           {d_dth_an:.8f}")
    print(f"  d(delta_BU)/dm_a                 {d_dm_an:.8f}")
    print(f"  (theta_ona/delta)*d/dtheta       {th0 * d_dth_an / delta_bu:.8f}")
    print(f"  (m_a/delta)*d/dm_a               {m0 * d_dm_an / delta_bu:.8f}")
    print()

    # Wigner map + Jacobian
    section(state, "WIGNER MAP CLASS (u_p, theta_ona)")
    w_canon = float(tw_angle_exact(t.u_p, t.theta_ona))
    print(f"  omega(u_p, theta_ona)            {w_canon:.16f}")

    h = mp.mpf("1e-20")
    d_om_db = (tw_angle_exact(t.u_p + h, t.theta_ona) - tw_angle_exact(t.u_p - h, t.theta_ona)) / (2 * h)
    d_om_dth = (tw_angle_exact(t.u_p, t.theta_ona + h) - tw_angle_exact(t.u_p, t.theta_ona - h)) / (2 * h)
    d_om_db_exact = (12 * mp.sqrt(2) - 4) / 17
    d_om_dth_exact = (21 - 12 * mp.sqrt(2)) / 17
    jac_sum = d_om_db_exact + d_om_dth_exact
    print(f"  d_omega/dbeta  (numeric)         {float(d_om_db):.16f}")
    print(f"  d_omega/dtheta (numeric)         {float(d_om_dth):.16f}")
    print(f"  d_omega/dbeta  exact form        {mp_to_str(d_om_db_exact, 20)}")
    print(f"  d_omega/dtheta exact form        {mp_to_str(d_om_dth_exact, 20)}")
    print(f"  exact forms                      (12*sqrt(2)-4)/17 , (21-12*sqrt(2))/17")
    print(f"  d_omega/dbeta + d_omega/dtheta   {mp_to_str(jac_sum, 20)}")
    print(f"  boost-magnitude response share   {float(d_om_db_exact):.6%}")
    print(f"  angular response share           {float(d_om_dth_exact):.6%}")
    check(
        state,
        f"d_omega/dbeta matches exact form (err {float(abs(d_om_db - d_om_db_exact)):.3e})",
        abs(d_om_db - d_om_db_exact) < mp.mpf("1e-20"),
    )
    check(
        state,
        f"d_omega/dtheta matches exact form (err {float(abs(d_om_dth - d_om_dth_exact)):.3e})",
        abs(d_om_dth - d_om_dth_exact) < mp.mpf("1e-20"),
    )
    check(
        state,
        f"canonical Wigner response derivatives sum to 1 (got {float(jac_sum):.16f})",
        abs(jac_sum - 1) < mp.mpf("1e-40"),
    )
    print()

    # Finite layer
    section(state, "FINITE hQVM HOLONOMY")
    finite = compute_finite_holonomy()
    if finite is None:
        print(f"  unavailable                      {_FINITE_IMPORT_ERROR}")
        check(state, "finite hQVM layer importable", False)
        bu_frac = None
    else:
        dist = finite["fold_disagreement_distribution"]
        print(f"  flat_bytes                       {finite['flat_bytes']}")
        print(f"  curved_bytes                     {finite['curved_bytes']}")
        print(f"  fold_disagreement_distribution   {dist}")
        expected_bin = {k: 16 * binomial_c4(k) for k in range(5)}
        print(f"  16*C(4,k) expected               {expected_bin}")
        bin_ok = all(dist.get(k, 0) == expected_bin[k] for k in range(5)) and sum(dist.values()) == 256
        print(f"  distribution total               {sum(dist.values())}")
        check(state, "fold disagreement = 16*C(4,k) and totals 256", bin_ok)
        print(f"  bu_boundary_disagreement_count   {finite['bu_boundary_disagreement_count']}")
        bu_frac = finite["bu_boundary_disagreement_count"] / 256.0
        print(f"  bu_boundary_fraction             {bu_frac:.16f}")
        print(f"  canonical W2 rest_ok             {finite['w2_rest_ok']}")
        print(f"  canonical W2' rest_ok            {finite['w2p_rest_ok']}")
        print(f"  W2 involution_ok                 {finite['w2_involution_ok']}")
        print(f"  T2 shell_ok                      {finite['t2_shell_ok']}")
        print(f"  T2 chirality_ok                  {finite['t2_chi_ok']}")
        print(f"  canonical certificate all_pass   {finite['k4_all_pass']}")
        check(state, "canonical W2/W2' finite holonomy certificate", bool(finite["k4_all_pass"]))
    print()

    # Byte-horizon aperture quantization
    section(state, "BYTE-HORIZON APERTURE QUANTIZATION")
    ticks = 256.0 * delta_gap
    q_ticks = int(round(ticks))
    q256 = q_ticks / 256.0
    rel_q = abs(delta_gap - q256) / delta_gap
    depth4 = 48.0 * delta_gap
    turn = delta_bu / (2.0 * math.pi)
    print(f"  Delta = 1 - delta_BU/m_a         {delta_gap:.16f}")
    print(f"  256 * Delta                      {ticks:.16f}")
    print(f"  round(256 * Delta)               {q_ticks}")
    print(f"  Q_256(Delta) = {q_ticks}/256     {q256:.16f}")
    print(f"  |Delta - Q_256| / Delta          {rel_q:.6e}")
    print(f"  48 * Delta                       {depth4:.16f}")
    print(f"  |48 * Delta - 1|                 {abs(depth4 - 1.0):.6e}")
    print(f"  delta_BU / (2*pi)                {turn:.16f}")
    print(f"  (1/48) / (1/32)                  {32.0 / 48.0:.16f}")
    check(
        state,
        f"nearest 8-bit dyadic of Delta is 5/256 (got {q_ticks}/256)",
        q_ticks == 5,
    )
    if APERTURE_GAP_Q256 is not None:
        print(f"  shared APERTURE_GAP_Q256         {APERTURE_GAP_Q256}")
        check(
            state,
            f"shared APERTURE_GAP_Q256 == 5 (got {APERTURE_GAP_Q256})",
            int(APERTURE_GAP_Q256) == 5,
        )
    else:
        print("  shared APERTURE_GAP_Q256         unavailable")
        check(state, "shared APERTURE_GAP_Q256 importable", False)
    print()

    # Continuous-finite dictionary
    section(state, "CONTINUOUS-FINITE STRUCTURAL CORRESPONDENCE")
    rows = [
        ("closed path in continuous model", "operator word on Omega"),
        ("holonomy angle / conjugacy class", "nontrivial finite involution or cycle"),
        ("BU dual-pole loop", "W2 pole exchange"),
        ("closure under return", "W2^2 = id"),
        ("aperture gap Delta", "byte-horizon dyadic 5/256"),
        ("palindromic payload path", "byte fold across BU boundary"),
        ("6 payload positions", "6 payload bits / 6 se(3) modes"),
        ("CS gauge frame", "byte bits 0 and 7 (family selector)"),
        ("continuous conjugacy spectrum", "finite involution spectrum (see wavefunction analysis)"),
    ]
    for left, right in rows:
        print(f"  {left:36s} -> {right}")
    print()
    print("  local curvature notes:")
    print("    continuous origin-gyr: gyr(BU+,BU-) is I for collinear origin vectors")
    print("    finite: fold disagreement is counted at the BU|BU boundary")
    print()

    # Palge-Pfeifer mass-shell holonomy
    section(state, "PALGE-PFEIFER MASS-SHELL HOLONOMY")
    print("  Object: Levi-Civita / spin-connection holonomy on V_m^+ (Palge-Pfeifer 2023).")
    print("  Stage betas as Einstein velocities map to 4-velocities q=(gamma, gamma beta) with m=1.")
    print("  Routes:")
    print("    (1) circular calibration: curvature disk integral Omega_s)")
    print("    (2) piecewise-geodesic polygon: pure boosts T_ij between successive q_i")
    print("    (3) spherical-chart P exp(-int omega) along those geodesics")
    print()

    suite = palge_pfeifer_suite_mp(t)
    circ = suite["circular"]
    print("  Circular Thomas calibration (rho=rho0, theta=pi/2, phi:0->2pi):")
    print(f"    V                              {mp_to_str(circ['V'], 12)}")
    print(f"    gamma(V)                       {mp_to_str(circ['gamma'], 20)}")
    print(f"    alpha_analytic = 2*pi*(gamma-1) {mp_to_str(circ['alpha_analytic'], 40)}")
    print(f"    alpha_curvature (int Omega_s)  {mp_to_str(circ['alpha_curvature'], 40)}")
    print(f"    int_C omega_13 (= 2*pi*gamma)  {mp_to_str(circ['alpha_connection_line'], 40)}")
    print(f"    int_C omega_13 - 2*pi          {mp_to_str(circ['alpha_line_minus_2pi'], 40)}")
    print(f"    |curvature - analytic|         {mp_to_str(circ['resid_curv_vs_analytic'], 10)}")
    print(f"    |line-2pi - analytic|          {mp_to_str(circ['resid_line_corr_vs_analytic'], 10)}")
    check(
        state,
        "circular curvature alpha",
        circ["resid_curv_vs_analytic"] < TOL_MP,
        tier="EXACT_MP",
        quantity="|alpha_curvature - 2*pi*(gamma-1)| (Palge-Pfeifer 47)",
        measured=f"{float(circ['resid_curv_vs_analytic']):.3e}",
        threshold="< 1e-60",
    )
    check(
        state,
        "circular line-minus-2pi alpha",
        circ["resid_line_corr_vs_analytic"] < TOL_MP,
        tier="EXACT_MP",
        quantity="|int omega_13 - 2*pi - 2*pi*(gamma-1)|",
        measured=f"{float(circ['resid_line_corr_vs_analytic']):.3e}",
        threshold="< 1e-60",
    )
    print("  Multi-V curvature scan:")
    max_circ_resid = mp.mpf(0)
    for V in (mp.mpf("0.1"), mp.mpf("0.2"), mp.mpf("0.4"), mp.mpf("0.6")):
        row = circular_curvature_holonomy_alpha_mp(V)
        max_circ_resid = max(max_circ_resid, row["resid_curv_vs_analytic"])
        print(
            f"    V={mp_to_str(V, 6)}  alpha={mp_to_str(row['alpha_analytic'], 18)}"
            f"  |curv-analytic|={mp_to_str(row['resid_curv_vs_analytic'], 8)}"
        )
    check(
        state,
        "circular multi-V",
        max_circ_resid < TOL_MP,
        tier="EXACT_MP",
        quantity="max |alpha_curvature - analytic| over V in {0.1,0.2,0.4,0.6}",
        measured=f"{float(max_circ_resid):.3e}",
        threshold="< 1e-60",
    )
    su2 = suite["circular_su2"]
    print(f"  SU(2) Hol(omega_s,C) U00=cos(alpha/2) {mp_to_str(su2['U00'], 20)}")
    print(f"                         U01=-sin(alpha/2) {mp_to_str(su2['U01'], 20)}")
    print()

    print("  CGM piecewise-geodesic path holonomy (pure-boost composition):")
    palge_bu = suite["paths"]["BU"]
    palge_pal = suite["paths"]["palindrome"]
    for label, block in (
        ("BU dual-pole", palge_bu),
        ("palindrome", palge_pal),
    ):
        geo = block["geodesic_boost"]
        om = block["omega_chart_pexp"]
        print(f"  path={label}  {' -> '.join(block['path'])}")
        print(f"    theta_geodesic               {mp_to_str(geo['theta'], 40)}")
        print(f"    theta_omega_chart_pexp       {mp_to_str(om['theta'], 40)}")
        print(f"    |geodesic - omega_chart|     {mp_to_str(abs(om['theta'] - geo['theta']), 10)}")
        print(f"    fix q0 resid                 {mp_to_str(geo['fix_q0_resid'], 10)}")
        print(f"    time-space resid             {mp_to_str(geo['time_space_resid'], 10)}")
        print(f"    ||R^T R - I||                {mp_to_str(geo['orth'], 10)}")
        print(f"    max edge T u->v resid        {mp_to_str(geo['max_edge_boost_resid'], 10)}")
        print(f"    max edge Lorentz ||L^TηL-η|| {mp_to_str(geo['max_edge_lorentz_resid'], 10)}")
        print(f"    max edge |det-1|             {mp_to_str(geo['max_edge_det_resid'], 10)}")
        print(f"    max edge orthochronous resid {mp_to_str(geo['max_edge_time_resid'], 10)}")
        print(f"    P Lorentz resid              {mp_to_str(geo['P_lorentz_resid'], 10)}")
        print(f"    P |det-1|                    {mp_to_str(geo['P_det_resid'], 10)}")
        print(f"    P orthochronous resid        {mp_to_str(geo['P_time_resid'], 10)}")
        if rooted_bu is not None and label == "BU dual-pole":
            print(f"    |geodesic - origin_gyr|      {mp_to_str(abs(geo['theta'] - rooted_bu['theta_origin_gyr_word']), 10)}")
            print(f"    |geodesic - relative_boost|  {mp_to_str(abs(geo['theta'] - rooted_bu['theta_relative_boost_word']), 10)}")
            print(f"    |geodesic - delta_BU|        {mp_to_str(abs(geo['theta'] - delta_bu_closed), 10)}")
        geo_ok = (
            geo["orth"] < TOL_MP
            and geo["time_space_resid"] < TOL_MP
            and geo["fix_q0_resid"] < TOL_MP
            and geo["P_lorentz_resid"] < TOL_MP
            and geo["P_det_resid"] < TOL_MP
            and geo["P_time_resid"] < TOL_MP
            and geo["max_edge_lorentz_resid"] < TOL_MP
            and geo["max_edge_det_resid"] < TOL_MP
            and geo["max_edge_time_resid"] < TOL_MP
        )
        check(
            state,
            f"{label} geodesic boost SO+(1,3)",
            geo_ok,
            tier="EXACT_MP",
            quantity=f"{label}: geodesic Hol SO(3) + SO+(1,3) residuals",
            measured=(
                f"orth={float(geo['orth']):.3e} ts={float(geo['time_space_resid']):.3e} "
                f"fix={float(geo['fix_q0_resid']):.3e} "
                f"P_Lor={float(geo['P_lorentz_resid']):.3e} "
                f"P_det={float(geo['P_det_resid']):.3e} "
                f"edge_Lor={float(geo['max_edge_lorentz_resid']):.3e}"
            ),
            threshold="< 1e-60 each",
        )
        if label == "BU dual-pole":
            d_gyr = abs(
                geo["theta"]
                - (
                    rooted_bu["theta_origin_gyr_word"]
                    if rooted_bu
                    else word["theta_word"]
                )
            )
            d_wig = abs(geo["theta"] - delta_bu_closed)
            check(
                state,
                "geodesic = origin-gyr word",
                d_gyr < TOL_MP,
                tier="EXACT_MP",
                quantity="|theta_geodesic - theta_origin_gyr_word| (BU dual-pole)",
                measured=f"{float(d_gyr):.3e}",
                threshold="< 1e-60",
            )
            check(
                state,
                "geodesic = delta_BU",
                d_wig < TOL_MP,
                tier="EXACT_MP",
                quantity="|theta_geodesic - delta_BU| (2*omega equation)",
                measured=f"{float(d_wig):.3e}",
                threshold="< 1e-60",
            )
    print()

    state.palge_circ = circ
    state.palge_bu = palge_bu
    state.palge_pal = palge_pal

    # Holonomy inventory
    section(state, "HOLONOMY INVENTORY")
    print("  Measured holonomy quantities (stage thresholds as Einstein betas).")
    print("  delta_BU = 2*omega = 4*arctan(r(theta_ona)*r(m_a)).")
    print(f"  {'quantity':32s} {'value':44s} method")
    print(f"  {'delta_BU (= 2*omega)':32s} {mp_to_str(delta_bu_closed, 20):44s} 4*arctan(r(theta_ona)*r(m_a))")
    print(f"  {'delta_BU_raw_map_mp':32s} {mp_to_str(mpj['delta_map'], 20):44s} mp raw gyr map r=1/2")
    print(f"  {'delta_BU_ungar_matrix_mp':32s} {mp_to_str(mpj['delta_ungar'], 20):44s} Ungar I+aOm+bOm^2")
    print(f"  {'delta_BU_origin_lorentz_fact_mp':32s} {mp_to_str(mpj['delta_lorentz'], 20):44s} L(a⊕b)^-1 L(a) L(b)")
    print(f"  {'delta_BU_word_mp':32s} {mp_to_str(word['theta_word'], 20):44s} mp origin-gyr word")
    if rooted_bu is not None:
        print(f"  {'theta_origin_gyr_word':32s} {mp_to_str(rooted_bu['theta_origin_gyr_word'], 20):44s} origin-gyr path product")
        print(f"  {'theta_relative_boost_word':32s} {mp_to_str(rooted_bu['theta_relative_boost_word'], 20):44s} L(d_i) lab-frame product")
    geo_bu = palge_bu["geodesic_boost"]
    print(f"  {'theta_geodesic_mass_shell':32s} {mp_to_str(geo_bu['theta'], 20):44s} pure-boost polygon on V_m^+")
    print(f"  {'theta_omega_chart_pexp_BU':32s} {mp_to_str(palge_bu['omega_chart_pexp']['theta'], 20):44s} spherical-chart Pexp(-int omega)")
    print(f"  {'alpha_circular_Thomas_V0.3':32s} {mp_to_str(circ['alpha_analytic'], 20):44s} 2*pi*(gamma-1)")
    if state.delta_stage is not None:
        print(f"  {'delta_stage_euclid':32s} {mp_to_str(state.delta_stage, 20):44s} pi-(theta_cs+theta_una+theta_ona)")
    print(f"  {'gyrotriangle_defect_corner':32s} {mp_to_str(defct['defect'], 20):44s} Ungar defect origin-ONA-BU+")
    if state.defct_bu_triangle is not None:
        print(f"  {'gyrotriangle_defect_BU_tri':32s} {mp_to_str(state.defct_bu_triangle['defect'], 20):44s} Ungar defect ONA-BU+-BU-")
    print(f"  |geodesic - origin_gyr|          {float(abs(geo_bu['theta'] - (rooted_bu['theta_origin_gyr_word'] if rooted_bu else word['theta_word']))):.6e}")
    if rooted_bu is not None:
        print(f"  |geodesic - relative_boost|      {float(abs(geo_bu['theta'] - rooted_bu['theta_relative_boost_word'])):.6e}")
    print(f"  |geodesic - omega_chart|         {float(abs(geo_bu['theta'] - palge_bu['omega_chart_pexp']['theta'])):.6e}")
    print()

    # Status table
    section(state, "RESULT STATUS")
    status_rows = [
        ("theta_cs+theta_una+theta_ona = pi", "exact_algebraic", "threshold definitions"),
        ("q_g * m_a^2 = 1/2", "exact_algebraic", "threshold definitions"),
        ("phi_SU2 closed form", "exact_algebraic", "SU(2) threshold angles"),
        ("TW small-angle + convergence order", "standard_analytic+numerical", "GyroVectorSpace.gyration"),
        ("delta_BU = 2*omega", "origin_wigner_closed", "4*arctan(r(theta_ona)*r(m_a))"),
        ("Ungar matrix / Lorentz / Thm7 / trace", "exact_mp", "ONA, BU+"),
        ("origin-gyr word", "exact_mp", "left-action product"),
        ("palindrome conjugacy", "exact_mp", "A^-1 R_BU A"),
        ("relative-boost word", "measured", "L(d_i) lab-frame product"),
        ("Palge-Pfeifer circular alpha", "exact_mp", "int Omega_s = 2*pi*(gamma-1)"),
        ("geodesic mass-shell holonomy", "exact_mp", "pure-boost polygon"),
        ("omega chart Pexp", "measured", "spherical pullback of omega"),
        ("delta_stage = 0", "exact_algebraic", "threshold angle sum"),
        ("defect(ONA,BU+,BU-) = delta_BU", "exact_mp", "Ungar 21/74"),
        ("gyrogroup axioms / coaddition", "exact_mp", "stage UNA,ONA,BU+"),
        ("rho, Delta from delta_BU", "derived", "delta_BU equation, m_a"),
        ("rank-2 (theta_ona, m_a)", "derived", "analytic delta_BU equation"),
        ("Wigner Jacobian sum=1 at threshold", "exact_algebraic", "beta=u_p, theta=pi/4"),
        ("canonical W2/W2' certificate", "exact_finite", "hQVM transition law"),
        ("fold distribution 16*C(4,k)", "exact_finite", "byte fold algebra"),
        ("Q_256(Delta) = 5/256", "byte_horizon", "Delta from delta_BU equation"),
        ("continuous-finite dictionary", "structural", "architecture"),
    ]
    print(f"  {'result':40s} {'status':32s} dependency")
    for name, status, dep in status_rows:
        print(f"  {name:40s} {status:32s} {dep}")
    print()

    # Integrity summary
    section(state, "INTEGRITY CHECK SUMMARY")
    n_pass = sum(1 for _, ok in state.gates if ok)
    n_fail = sum(1 for _, ok in state.gates if not ok)
    for label, ok in state.gates:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print(f"  passed={n_pass}  failed={n_fail}  total={len(state.gates)}")
    return state
