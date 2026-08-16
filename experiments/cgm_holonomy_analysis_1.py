#!/usr/bin/env python3
"""
cgm_holonomy_analysis_1.py

Report part 1: thresholds through gyrogroup axioms / coaddition / defect.

Invoked by cgm_holonomy_analysis_run.py. Companions: _common.py, _2.py.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

_EXP = Path(__file__).resolve().parent
_REPO = _EXP.parent
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cgm_holonomy_analysis_common import (
    GYR_PROBE_R,
    TOL_MP,
    TOL_TW_RESID,
    TOL_TW_SLOPE,
    CGMThresholds,
    ReportState,
    analytic_bu_holonomy,
    check,
    compute_exact_su2_holonomy,
    einstein_coaddition_mp,
    gyrogroup_axiom_residuals_mp,
    gyrotriangle_defect_mp,
    gyrotriangle_defect_triangle_vertices_mp,
    mp_stage_points,
    mp_to_str,
    mpmath_dual_pole_word,
    mpmath_ona_bu_origin_gyr_suite,
    mpmath_palindrome_word,
    mpmath_rooted_path_compare,
    poincare_radius_from_beta,
    points_from_stages,
    radial_coordinates_mp,
    run_tw_benchmark,
    section,
    so3_residuals_mp,
    stage_angle_defect_euclid_mp,
    stage_coordinates,
    tw_angle_signed_mp,
    ungar_lorentz_random_audit_mp,
)
from functions.gyrovector_ops import GyroVectorSpace


def run_holonomy_1(state: ReportState) -> ReportState:
    t = CGMThresholds.make()
    gs = GyroVectorSpace(c=1.0)

    # Thresholds
    section(state, "THRESHOLDS")
    print(f"  theta_cs                         {mp_to_str(t.theta_cs, 20)}")
    print(f"  u_p                              {mp_to_str(t.u_p, 20)}")
    print(f"  theta_una = arccos(u_p)          {mp_to_str(t.theta_una, 20)}")
    print(f"  theta_ona                        {mp_to_str(t.theta_ona, 20)}")
    print(f"  m_a                              {mp_to_str(t.m_a, 20)}")
    print(f"  q_g                              {mp_to_str(t.q_g, 20)}")
    print(f"  theta_cs+theta_una+theta_ona     {mp_to_str(t.angle_sum, 20)}")
    print(f"  q_g * m_a^2                      {mp_to_str(t.qg_ma2, 20)}")
    check(
        state,
        f"|angle_sum - pi| < 1e-15 (got {abs(float(t.angle_sum) - math.pi):.3e})",
        abs(float(t.angle_sum) - math.pi) < 1e-15,
    )
    check(
        state,
        f"|q_g*m_a^2 - 1/2| < 1e-15 (got {abs(float(t.qg_ma2) - 0.5):.3e})",
        abs(float(t.qg_ma2) - 0.5) < 1e-15,
    )
    print()

    # SU(2)
    section(state, "EXACT SU(2) COMMUTATOR HOLONOMY")
    phi_su2, closed_form, su2_err = compute_exact_su2_holonomy()
    print(f"  phi_SU2 (rad)                    {mp_to_str(phi_su2, 20)}")
    print(f"  phi_SU2 (deg)                    {float(phi_su2) * 180.0 / math.pi:.12f}")
    print(f"  closed_form                      {closed_form}")
    print(f"  |numeric - closed_form|           {mp_to_str(su2_err, 12)}")
    check(state, f"SU(2) mpmath product matches closed form (err {float(su2_err):.3e})", float(su2_err) < 1e-40)
    print()

    # TW calibration
    section(state, "THOMAS-WIGNER SMALL-ANGLE CALIBRATION")
    print("  gyration angle vs ||u x v||/(2c^2) on a fixed velocity grid")
    tw_rows = run_tw_benchmark(gs)
    print(f"  {'beta_max':>10s} {'n':>6s} {'slope':>12s} {'mean|res|':>14s} {'max|res|':>14s}")
    row_005: dict[str, float] | None = None
    for row in tw_rows:
        print(
            f"  {row['beta_max']:10.3f} {int(row['n']):6d} {row['slope']:12.6f} "
            f"{row['mean_abs_residual']:14.3e} {row['max_abs_residual']:14.3e}"
        )
        if abs(row["beta_max"] - 0.05) < 1e-12:
            row_005 = row
    assert row_005 is not None
    slope_err = abs(row_005["slope"] - 1.0)
    tw_ok = (slope_err < TOL_TW_SLOPE) and (row_005["max_abs_residual"] < TOL_TW_RESID)
    print(f"  slope_error (beta<=0.05)         {slope_err:.3e}  (tol {TOL_TW_SLOPE})")
    print(f"  max_residual (beta<=0.05)        {row_005['max_abs_residual']:.3e}  (tol {TOL_TW_RESID})")
    check(
        state,
        f"TW slope~1 and residual (beta<=0.05): |slope-1|={slope_err:.3e}, max|res|={row_005['max_abs_residual']:.3e}",
        tw_ok,
    )
    betas = np.asarray([r["beta_max"] for r in tw_rows], dtype=float)
    slope_errors = np.asarray([abs(r["slope"] - 1.0) for r in tw_rows], dtype=float)
    max_resids = np.asarray([r["max_abs_residual"] for r in tw_rows], dtype=float)
    slope_order = float(np.polyfit(np.log(betas), np.log(slope_errors), 1)[0])
    resid_order = float(np.polyfit(np.log(betas), np.log(max_resids), 1)[0])
    print(f"  slope-error convergence order   {slope_order:.6f}")
    print(f"  residual convergence order      {resid_order:.6f}")
    check(state, f"slope error scales ~ beta^2 (order {slope_order:.3f})", 1.8 < slope_order < 2.2)
    check(state, f"abs residual scales ~ beta^4 (order {resid_order:.3f})", 3.8 < resid_order < 4.2)
    print()

    # Stage coordinates + embedding registry
    section(state, "CGM STAGE COORDINATES AND EMBEDDING REGISTRY")
    print("  Object: stage points in the open Einstein ball (c=1).")
    print("  CGM threshold number taken as Einstein beta = ||v||.")
    stages = stage_coordinates(t)
    points = points_from_stages(stages)
    state.stages = stages
    state.points = points
    print(f"  name                             {stages.name}")
    print(f"  c                                {stages.c}")
    print(f"  UNA                              {stages.una_vector}")
    print(f"  ONA                              {stages.ona_vector}")
    print(f"  BU+                              {stages.bu_plus_vector}")
    print(f"  BU-                              {stages.bu_minus_vector}")
    print("  CS                               s_p=pi/2 lies outside the open ball; gauge frame only")
    domain_ok = True
    for name, vec in points.items():
        nrm = float(np.linalg.norm(np.asarray(vec, dtype=float)))
        inside = nrm < stages.c
        print(f"  ||{name}||                          {nrm:.12f}  inside_ball={inside}")
        domain_ok = domain_ok and inside
    check(
        state,
        "all payload stage vectors satisfy ||v|| < c",
        domain_ok,
        tier="STRUCTURAL",
        quantity="payload stage vectors inside open Einstein ball",
        measured=f"UNA,ONA,BU+,BU- all ||v||<c = {domain_ok}",
        threshold="||v|| < c for each payload stage",
    )
    print()
    print("  Radial charts for stage betas:")
    print(f"  {'stage':6s} {'beta':>16s} {'gamma':>16s} {'eta':>16s} {'rho/m':>16s} {'Poincare_r':>16s}")
    for name, beta in (
        ("UNA", t.u_p),
        ("ONA", t.theta_ona),
        ("BU", t.m_a),
    ):
        rc = radial_coordinates_mp(beta)
        print(
            f"  {name:6s} {mp_to_str(rc['beta'], 14):>16s} {mp_to_str(rc['gamma'], 14):>16s} "
            f"{mp_to_str(rc['eta'], 14):>16s} {mp_to_str(rc['rho_over_m'], 14):>16s} "
            f"{mp_to_str(rc['poincare_r'], 14):>16s}"
        )
    print()

    # mpmath origin-gyr suite
    section(state, "ORIGIN-GYR CROSS-CHECKS (mpmath, ONA, BU+)")
    print("  Object: origin-based gyr[ONA, BU+] with stage thresholds as Einstein betas.")
    print("  Routes:")
    print("    analytic:  delta_BU = 2*omega = 4*atan(r(theta_ona)*r(m_a)), r = Poincare radius")
    print("    raw map:   columns = gyr(r e_i)/r with r=1/2, no SVD")
    print("    Ungar:     G = I + alpha*Omega + beta*Omega^2 (eqs 48-49)")
    print("    Lorentz factorization: spatial block of L(a⊕b)^-1 L(a) L(b)")
    print("  Angle extractor: atan2(||vee(R-R^T)||/2, (tr-1)/2) on the RAW matrix.")
    print("  Trace identity (Ungar 51) on raw/Ungar matrices.")
    print("  Linearity scan: gyr(eps e_i)/eps vs eps.")
    print(f"  mp.dps                           {mp.mp.dps}")
    print(f"  probe_r                          {mp_to_str(GYR_PROBE_R, 8)}")
    mpj = mpmath_ona_bu_origin_gyr_suite(t)
    print(f"  axis (u x v)/||u x v||           ({mp_to_str(mpj['axis_u_cross_v'][0], 8)}, "
          f"{mp_to_str(mpj['axis_u_cross_v'][1], 8)}, {mp_to_str(mpj['axis_u_cross_v'][2], 8)})")
    print(f"  eps_signed (Ungar orientation)   {mp_to_str(mpj['eps_signed'], 40)}")
    print(f"  |eps_signed|                     {mp_to_str(abs(mpj['eps_signed']), 40)}")
    print(f"  delta_BU = 2*omega               {mp_to_str(mpj['delta_closed'], 50)}")
    print(f"  delta_BU_raw_map                 {mp_to_str(mpj['delta_map'], 50)}")
    print(f"  delta_BU_ungar_matrix            {mp_to_str(mpj['delta_ungar'], 50)}")
    print(f"  delta_BU_origin_lorentz_factorization {mp_to_str(mpj['delta_lorentz'], 50)}")
    print(f"  |raw - analytic|                 {mp_to_str(mpj['map_minus_closed'], 10)}")
    print(f"  |ungar - analytic|               {mp_to_str(mpj['ungar_minus_closed'], 10)}")
    print(f"  ||Ungar - raw||                  {mp_to_str(mpj['ungar_minus_raw'], 10)}")
    print(f"  ||Lorentz - raw||                {mp_to_str(mpj['lorentz_minus_raw'], 10)}")
    print(f"  raw ||R^T R - I||                {mp_to_str(mpj['orth_raw'], 10)}")
    print(f"  raw |det(R)-1|                   {mp_to_str(mpj['det_raw'], 10)}")
    print(f"  raw trace-identity residual      {mp_to_str(mpj['trace_id_raw'], 10)}")
    print(f"  Ungar ||R^T R - I||              {mp_to_str(mpj['orth_ungar'], 10)}")
    print(f"  Ungar |det(R)-1|                 {mp_to_str(mpj['det_ungar'], 10)}")
    print(f"  Ungar trace-identity residual    {mp_to_str(mpj['trace_id_ungar'], 10)}")
    print(f"  Lorentz ||R^T R - I||            {mp_to_str(mpj['orth_lorentz'], 10)}")
    print(f"  Lorentz |det(R)-1|               {mp_to_str(mpj['det_lorentz'], 10)}")
    print(f"  Lorentz trace-identity residual  {mp_to_str(mpj['trace_id_lorentz'], 10)}")
    thm7 = mpj["thm7"]
    print("  Ungar boost factorization (4x4; x'=L@x):")
    print(f"    ||B(u)B(v) - B(u⊕v)Gyr[u,v]||  {mp_to_str(thm7['resid_BuBv_vs_Buv_Gyr'], 10)}")
    print(f"    ||B(v)B(u) - B(v⊕u)Gyr[v,u]||  {mp_to_str(thm7['resid_BvBu_vs_Bvu_Gyr'], 10)}")
    print(f"    ||Gyr_Ungar - Gyr_Lorentz||    {mp_to_str(thm7['resid_ungar_vs_lorentz_gyr'], 10)}")
    print("  Linearity scan (delta from gyr(eps e_i)/eps):")
    print("  eps              delta_BU_lin                                      |lin - analytic|")
    max_lin_err = mp.mpf(0)
    for eexp, delta_jac, err in mpj["lin_rows"]:
        max_lin_err = max(max_lin_err, err)
        print(
            f"  1e-{eexp:<2d}  {mp_to_str(delta_jac, 40):<48s}  {mp_to_str(err, 10)}"
        )
    check(
        state,
        f"raw map in SO(3)",
        mpj["orth_raw"] < TOL_MP and mpj["det_raw"] < TOL_MP,
        tier="EXACT_MP",
        quantity="raw gyr map SO(3) residuals",
        measured=f"orth={float(mpj['orth_raw']):.3e}, |det-1|={float(mpj['det_raw']):.3e}",
        threshold="< 1e-60",
    )
    check(
        state,
        f"|map - closed|",
        mpj["map_minus_closed"] < TOL_MP,
        tier="EXACT_MP",
        quantity="|delta_raw_map - delta_wigner_analytic|",
        measured=f"{float(mpj['map_minus_closed']):.3e}",
        threshold="< 1e-60",
    )
    check(
        state,
        f"|ungar - closed|",
        mpj["ungar_minus_closed"] < TOL_MP,
        tier="EXACT_MP",
        quantity="|delta_ungar_matrix - delta_wigner_analytic|",
        measured=f"{float(mpj['ungar_minus_closed']):.3e}",
        threshold="< 1e-60",
    )
    check(
        state,
        f"|lorentz - raw|",
        mpj["lorentz_minus_raw"] < TOL_MP,
        tier="EXACT_MP",
        quantity="||G_lorentz - G_raw||",
        measured=f"{float(mpj['lorentz_minus_raw']):.3e}",
        threshold="< 1e-60",
    )
    check(
        state,
        "trace identity raw",
        mpj["trace_id_raw"] < TOL_MP,
        tier="EXACT_MP",
        quantity="Ungar trace identity on raw gyr matrix",
        measured=f"{float(mpj['trace_id_raw']):.3e}",
        threshold="< 1e-60",
    )
    check(
        state,
        "trace identity Ungar",
        mpj["trace_id_ungar"] < TOL_MP,
        tier="EXACT_MP",
        quantity="Ungar trace identity on Ungar matrix",
        measured=f"{float(mpj['trace_id_ungar']):.3e}",
        threshold="< 1e-60",
    )
    check(
        state,
        "Thm7 B(u)B(v)=B(u⊕v)Gyr",
        thm7["resid_BuBv_vs_Buv_Gyr"] < TOL_MP,
        tier="EXACT_MP",
        quantity="Ungar factorization ||B(u)B(v)-B(u⊕v)Gyr[u,v]||",
        measured=f"{float(thm7['resid_BuBv_vs_Buv_Gyr']):.3e}",
        threshold="< 1e-60",
    )
    check(
        state,
        "Thm7 B(v)B(u)=B(v⊕u)Gyr",
        thm7["resid_BvBu_vs_Bvu_Gyr"] < TOL_MP,
        tier="EXACT_MP",
        quantity="Ungar factorization ||B(v)B(u)-B(v⊕u)Gyr[v,u]||",
        measured=f"{float(thm7['resid_BvBu_vs_Bvu_Gyr']):.3e}",
        threshold="< 1e-60",
    )
    check(
        state,
        f"linearity |lin - analytic|",
        max_lin_err < mp.mpf("1e-40"),
        tier="EXACT_MP",
        quantity="max |delta(gyr(eps e_i)/eps) - analytic| over eps grid",
        measured=f"{float(max_lin_err):.3e}",
        threshold="< 1e-40",
    )
    print()
    print("  Ungar matrix vs Lorentz factorization on random pairs (beta<=0.3):")
    rnd = ungar_lorentz_random_audit_mp(n=24, beta_max=mp.mpf("0.3"), seed=1)
    print(f"  n_requested                      {rnd['n_requested']}")
    print(f"  n_used                           {rnd['n_used']}")
    print(f"  beta_max                         {mp_to_str(rnd['beta_max'], 8)}")
    print(f"  max ||G_ungar - G_lorentz||      {mp_to_str(rnd['max_ungar_minus_lorentz'], 10)}")
    print(f"  max trace-identity (Ungar)       {mp_to_str(rnd['max_trace_identity_ungar'], 10)}")
    print(f"  max Thm7 residual                {mp_to_str(rnd['max_thm7_resid'], 10)}")
    check(
        state,
        "random Ungar vs Lorentz",
        rnd["max_ungar_minus_lorentz"] < TOL_MP and rnd["n_used"] >= 10,
        tier="EXACT_MP",
        quantity="max ||G_ungar - G_lorentz|| over random pairs",
        measured=f"{float(rnd['max_ungar_minus_lorentz']):.3e} (n_used={rnd['n_used']})",
        threshold="< 1e-60 and n_used>=10",
    )
    check(
        state,
        "random Ungar trace identity",
        rnd["max_trace_identity_ungar"] < TOL_MP,
        tier="EXACT_MP",
        quantity="max Ungar trace-identity residual over random pairs",
        measured=f"{float(rnd['max_trace_identity_ungar']):.3e}",
        threshold="< 1e-60",
    )
    check(
        state,
        "random Thm7",
        rnd["max_thm7_resid"] < TOL_MP,
        tier="EXACT_MP",
        quantity="max boost-factorization residual over random pairs",
        measured=f"{float(rnd['max_thm7_resid']):.3e}",
        threshold="< 1e-60",
    )
    print()

    # Origin-Wigner analytic
    section(state, "ORIGIN-WIGNER ANALYTIC")
    print("  Object: BU dual-pole aperture for orthogonal boosts beta=theta_ona, beta=m_a.")
    print("  Equation:")
    print("    delta_BU = 2*omega = 4*arctan(r(theta_ona)*r(m_a))")
    print("    r(beta) = beta/(1+sqrt(1-beta^2))  (Poincare half-rapidity radius)")
    print("    omega = |eps| = 2*arctan(r(theta_ona)*r(m_a))")
    print("    eps_signed = -2*arctan(r(theta_ona)*r(m_a)) for theta=pi/2 (Ungar sign)")
    print("  Construction: origin-Wigner analytic.")
    omega_closed, delta_bu_closed = analytic_bu_holonomy(t)
    eps_signed = tw_angle_signed_mp(t.theta_ona, t.m_a, mp.pi / 2)
    k_ona = poincare_radius_from_beta(t.theta_ona)
    k_ma = poincare_radius_from_beta(t.m_a)
    rho_closed = delta_bu_closed / t.m_a
    rho_zero = 2 * k_ona
    rho_corr = rho_closed - rho_zero
    delta_gap_closed = 1 - rho_closed
    print(f"  r(theta_ona)                     {mp_to_str(k_ona, 20)}")
    print(f"  r(m_a)                           {mp_to_str(k_ma, 20)}")
    print(f"  r(theta_ona)*r(m_a)              {mp_to_str(k_ona * k_ma, 20)}")
    print(f"  eps_signed (ONA,BU+, theta=pi/2) {mp_to_str(eps_signed, 40)}")
    print(f"  omega = |eps_signed|             {mp_to_str(abs(eps_signed), 40)}")
    print(f"  omega_closed                     {mp_to_str(omega_closed, 20)}")
    print(f"  delta_BU = 2*omega               {mp_to_str(delta_bu_closed, 50)}")
    print(f"  |2*|eps| - delta_BU|             {mp_to_str(abs(2 * abs(eps_signed) - delta_bu_closed), 10)}")
    print(f"  rho = delta_BU / m_a             {mp_to_str(rho_closed, 20)}")
    print(f"  Delta = 1 - rho                  {mp_to_str(delta_gap_closed, 20)}")
    print(f"  rho(m_a -> 0) = 2*r(theta_ona)   {mp_to_str(rho_zero, 20)}")
    print(f"  finite-BU correction rho-rho0    {mp_to_str(rho_corr, 20)}")
    print(f"  baseline gap 1-rho0              {mp_to_str(1 - rho_zero, 20)}")
    check(
        state,
        "signed matches magnitude",
        abs(2 * abs(eps_signed) - delta_bu_closed) < TOL_MP,
        tier="EXACT_MP",
        quantity="|2*|eps_signed| - delta_BU|",
        measured=f"{float(abs(2 * abs(eps_signed) - delta_bu_closed)):.3e}",
        threshold="< 1e-60",
    )
    check(
        state,
        f"0 < rho < 1 (rho={float(rho_closed):.12f})",
        0 < rho_closed < 1,
        tier="EXACT_MP",
        quantity="rho = delta_BU / m_a in (0,1)",
        measured=f"{float(rho_closed):.12f}",
        threshold="0 < rho < 1",
    )
    state.delta_bu = float(delta_bu_closed)
    state.rho = float(rho_closed)
    state.delta_gap = float(delta_gap_closed)
    print()

    # mpmath full dual-pole word
    section(state, "MPMATH DUAL-POLE ORIGIN-GYR WORD")
    print("  Object: product of three origin-based gyrations on ONA, BU+, BU-.")
    print("  Convention: left action R = G_ingress G_middle G_egress.")
    print("  Construction: origin-gyr word. Primary angle = angle(R).")
    word = mpmath_dual_pole_word(t)
    print(f"  theta_egress = angle(gyr(ONA,BU+)) {mp_to_str(word['theta_egress'], 40)}")
    print(f"  theta_middle = angle(gyr(BU+,BU-)) {mp_to_str(word['theta_middle'], 40)}")
    print(f"  theta_ingress = angle(gyr(BU-,ONA)) {mp_to_str(word['theta_ingress'], 40)}")
    print(f"  theta_word = angle(R)            {mp_to_str(word['theta_word'], 50)}")
    print(f"  theta_egress + theta_ingress     {mp_to_str(word['sum_corners'], 50)}")
    print(f"  |theta_word - (egress+ingress)|  {mp_to_str(word['word_minus_egress_ingress_sum'], 10)}")
    print(f"  |theta_egress - theta_ingress|   {mp_to_str(word['egress_minus_ingress'], 10)}")
    print(f"  |theta_word - wigner_analytic|   {mp_to_str(word['word_minus_wigner'], 10)}")
    print(f"  ||R^T R - I||                    {mp_to_str(word['orth'], 10)}")
    print(f"  |det(R)-1|                       {mp_to_str(word['det'], 10)}")
    mid_orth, mid_det = so3_residuals_mp(word["G_middle"])
    print(f"  middle ||G^T G - I||             {mp_to_str(mid_orth, 10)}")
    print(f"  middle |det-1|                   {mp_to_str(mid_det, 10)}")
    print(f"  middle ||G - I||                 {mp_to_str(mp.norm(word['G_middle'] - mp.eye(3)), 10)}")
    check(
        state,
        "middle ~ I",
        mp.norm(word["G_middle"] - mp.eye(3)) < TOL_MP,
        tier="EXACT_MP",
        quantity="||gyr(BU+,BU-) - I|| (collinear origin-gyr)",
        measured=f"{float(mp.norm(word['G_middle'] - mp.eye(3))):.3e}",
        threshold="< 1e-60",
    )
    check(
        state,
        "|egress - ingress|",
        word["egress_minus_ingress"] < TOL_MP,
        tier="EXACT_MP",
        quantity="|angle(gyr(ONA,BU+)) - angle(gyr(BU-,ONA))|",
        measured=f"{float(word['egress_minus_ingress']):.3e}",
        threshold="< 1e-60",
    )
    check(
        state,
        "|theta_word - (e+i)|",
        word["word_minus_egress_ingress_sum"] < TOL_MP,
        tier="EXACT_MP",
        quantity="|angle(origin-gyr word) - (egress+ingress)|",
        measured=f"{float(word['word_minus_egress_ingress_sum']):.3e}",
        threshold="< 1e-60",
    )
    check(
        state,
        "|theta_word - wigner|",
        word["word_minus_wigner"] < TOL_MP,
        tier="EXACT_MP",
        quantity="|angle(origin-gyr word) - delta_BU|",
        measured=f"{float(word['word_minus_wigner']):.3e}",
        threshold="< 1e-60",
    )
    print()

    # mpmath palindrome
    section(state, "MPMATH PALINDROME ORIGIN-GYR WORD")
    print("  Object: origin-gyr product on UNA->ONA->BU+->BU-->ONA->UNA.")
    print("  Conjugacy under left action: R_pal = A^-1 R_BU A with A = gyr(UNA,ONA).")
    print("  Angle = conjugacy class; axis transport via residual.")
    pal_mp = mpmath_palindrome_word(t, word["R"])
    print(f"  theta_pal                        {mp_to_str(pal_mp['theta_pal'], 50)}")
    print(f"  theta_BU                         {mp_to_str(pal_mp['theta_bu'], 50)}")
    print(f"  |theta_pal - theta_BU|           {mp_to_str(abs(pal_mp['theta_pal'] - pal_mp['theta_bu']), 10)}")
    print(f"  |tr_pal - tr_BU|                 {mp_to_str(abs(pal_mp['tr_pal'] - pal_mp['tr_bu']), 10)}")
    print(f"  ||R_pal - A^-1 R_BU A||          {mp_to_str(pal_mp['resid_Ainv_R_A'], 10)}")
    print(f"  ||R_pal - A R_BU A^-1||          {mp_to_str(pal_mp['resid_A_R_Ainv'], 10)}")
    print(f"  ||R_pal^T R_pal - I||            {mp_to_str(pal_mp['orth'], 10)}")
    print(f"  |det(R_pal)-1|                   {mp_to_str(pal_mp['det'], 10)}")
    check(
        state,
        "|theta_pal - theta_BU|",
        abs(pal_mp["theta_pal"] - pal_mp["theta_bu"]) < TOL_MP,
        tier="EXACT_MP",
        quantity="|angle(palindrome word) - angle(BU word)|",
        measured=f"{float(abs(pal_mp['theta_pal'] - pal_mp['theta_bu'])):.3e}",
        threshold="< 1e-60",
    )
    check(
        state,
        "palindrome = A^-1 R_BU A",
        pal_mp["resid_Ainv_R_A"] < TOL_MP,
        tier="EXACT_MP",
        quantity="||R_pal - A^-1 R_BU A|| (left-action conjugacy)",
        measured=f"{float(pal_mp['resid_Ainv_R_A']):.3e}",
        threshold="< 1e-60",
    )
    print()

    # Path holonomies: origin-gyr word and relative-boost word
    section(state, "PATH HOLONOMIES: ORIGIN-GYR WORD AND RELATIVE-BOOST WORD")
    print("  Origin-gyr word: product of gyr(p_i, p_{i+1}).")
    print("  Relative-boost word: rotational part of product L(d_i), d_i = ⊖p_i ⊕ p_{i+1}.")
    print("  Geodesic mass-shell holonomy: section PALGE-PFEIFER MASS-SHELL HOLONOMY.")
    rooted_bu = None
    for label, path in (
        ("BU dual-pole", ("ONA", "BU+", "BU-", "ONA")),
        ("palindrome", ("UNA", "ONA", "BU+", "BU-", "ONA", "UNA")),
    ):
        cmp = mpmath_rooted_path_compare(t, path)
        if label == "BU dual-pole":
            rooted_bu = cmp
        print(f"  path={label}")
        print(f"    theta_origin_gyr_word          {mp_to_str(cmp['theta_origin_gyr_word'], 40)}")
        print(f"    theta_relative_boost_word      {mp_to_str(cmp['theta_relative_boost_word'], 40)}")
        print(f"    |origin_gyr - relative_boost|  {mp_to_str(cmp['angle_diff'], 10)}")
        print(f"    ||R_gyr - R_rel||              {mp_to_str(cmp['norm_origin_gyr_minus_relative_boost'], 10)}")
        print(f"    relative_boost time-space resid {mp_to_str(cmp['relative_boost_time_space_resid'], 10)}")
    print()

    # Gyrogroup axioms / coaddition / defect
    section(state, "GYROGROUP AXIOMS, COADDITION, GYROTRIANGLE DEFECT")
    print("  Object: Ungar gyrogroup identities on CGM stage vectors (thresholds as Einstein betas).")
    print("  Vectors: u=UNA, v=ONA, w=BU+.")
    pmp = mp_stage_points(t)
    ax = gyrogroup_axiom_residuals_mp(pmp["UNA"], pmp["ONA"], pmp["BU+"])
    print(f"  ||u⊕v - gyr[u,v](v⊕u)||          {mp_to_str(ax['gyrocommutative'], 10)}")
    print(f"  ||u⊕(v⊕w) - (u⊕v)⊕gyr[u,v]w||   {mp_to_str(ax['left_gyroassociative'], 10)}")
    print(f"  ||gyr[⊖u,⊖v] - gyr[u,v]||        {mp_to_str(ax['even_property'], 10)}")
    print(f"  ||gyr[v,u] - gyr[u,v]^-1||       {mp_to_str(ax['gyration_inverse'], 10)}")
    print(f"  ||gyr[u⊕v,v] - gyr[u,v]||        {mp_to_str(ax['left_loop'], 10)}")
    print(f"  ||gyr[u,v⊕u] - gyr[u,v]||        {mp_to_str(ax['right_loop'], 10)}")
    for key, lab in (
        ("gyrocommutative", "gyrocommutative law residual"),
        ("left_gyroassociative", "left gyroassociative residual"),
        ("even_property", "even property residual"),
        ("gyration_inverse", "gyration inverse residual"),
        ("left_loop", "left loop property residual"),
        ("right_loop", "right loop property residual"),
    ):
        check(
            state,
            key,
            ax[key] < TOL_MP,
            tier="EXACT_MP",
            quantity=lab,
            measured=f"{float(ax[key]):.3e}",
            threshold="< 1e-60",
        )
    print()
    print("  Einstein coaddition u ⊞ v = u ⊕ gyr[u,⊖v] v (commutative):")
    co_uv = einstein_coaddition_mp(pmp["ONA"], pmp["BU+"])
    co_vu = einstein_coaddition_mp(pmp["BU+"], pmp["ONA"])
    co_diff = mp.sqrt(sum((co_uv[i] - co_vu[i]) ** 2 for i in range(3)))
    print(f"  ONA ⊞ BU+                        ({mp_to_str(co_uv[0], 12)}, {mp_to_str(co_uv[1], 12)}, {mp_to_str(co_uv[2], 12)})")
    print(f"  BU+ ⊞ ONA                        ({mp_to_str(co_vu[0], 12)}, {mp_to_str(co_vu[1], 12)}, {mp_to_str(co_vu[2], 12)})")
    print(f"  ||(ONA⊞BU+) - (BU+⊞ONA)||        {mp_to_str(co_diff, 10)}")
    check(
        state,
        "coaddition commutative",
        co_diff < TOL_MP,
        tier="EXACT_MP",
        quantity="||(ONA ⊞ BU+) - (BU+ ⊞ ONA)||",
        measured=f"{float(co_diff):.3e}",
        threshold="< 1e-60",
    )
    print()
    print("  Defect inventory:")
    print("    (i) stage-angle closure: pi - (theta_cs+theta_una+theta_ona)")
    print("    (ii) Ungar gyrotriangle defects from side gammas")
    d_stage = stage_angle_defect_euclid_mp(t)
    print(f"  delta_stage (i)                  {mp_to_str(d_stage, 30)}")
    check(
        state,
        "stage-angle defect is zero",
        abs(d_stage) < mp.mpf("1e-70"),
        tier="EXACT_MP",
        quantity="|delta_stage = pi-(theta_cs+theta_una+theta_ona)|",
        measured=f"{float(abs(d_stage)):.3e}",
        threshold="< 1e-70",
    )
    print()
    print("  Corner gyrotriangle (origin, ONA, BU+) — Ungar 21/74:")
    defct = gyrotriangle_defect_mp(pmp["ONA"], pmp["BU+"])
    print(f"  gamma(ONA)                       {mp_to_str(defct['gamma_u'], 20)}")
    print(f"  gamma(BU+)                       {mp_to_str(defct['gamma_v'], 20)}")
    print(f"  gamma(⊖ONA ⊕ BU+)                {mp_to_str(defct['gamma_neg_u_plus_v'], 20)}")
    print(f"  defect(origin,ONA,BU+) = omega   {mp_to_str(defct['defect'], 40)}")
    print(f"  angle(gyr[ONA, ⊖BU+])            {mp_to_str(defct['gyr_u_neg_v_angle'], 40)}")
    print(f"  |defect - gyr angle|              {mp_to_str(defct['defect_minus_gyr_angle'], 10)}")
    print(f"  |defect - omega_wigner|           {mp_to_str(abs(defct['defect'] - omega_closed), 10)}")
    print(f"  |defect - delta_BU/2|             {mp_to_str(abs(defct['defect'] - delta_bu_closed / 2), 10)}")
    check(
        state,
        "defect = gyr[u,⊖v] angle",
        defct["defect_minus_gyr_angle"] < TOL_MP,
        tier="EXACT_MP",
        quantity="|gyrotriangle defect(origin,ONA,BU+) - angle(gyr[ONA,⊖BU+])|",
        measured=f"{float(defct['defect_minus_gyr_angle']):.3e}",
        threshold="< 1e-60",
    )
    print()
    print("  BU closed gyrotriangle (ONA, BU+, BU-) via gyrotranslation of ONA to origin:")
    def_bu_tri = gyrotriangle_defect_triangle_vertices_mp(
        pmp["ONA"], pmp["BU+"], pmp["BU-"]
    )
    print(f"  defect(ONA,BU+,BU-)              {mp_to_str(def_bu_tri['defect'], 40)}")
    print(f"  angle witness gyr[u,⊖v]          {mp_to_str(def_bu_tri['gyr_u_neg_v_angle'], 40)}")
    print(f"  |defect - gyr witness|            {mp_to_str(def_bu_tri['defect_minus_gyr_angle'], 10)}")
    print(f"  delta_BU = 2*omega               {mp_to_str(delta_bu_closed, 40)}")
    print(f"  |defect(BU tri) - delta_BU|      {mp_to_str(abs(def_bu_tri['defect'] - delta_bu_closed), 10)}")
    print(f"  |defect(BU tri) - 2*corner|      {mp_to_str(abs(def_bu_tri['defect'] - 2 * defct['defect']), 10)}")
    print("  Defect relation at CGM thresholds (stage speeds as Einstein betas):")
    print(f"    delta_stage (threshold closure)  {mp_to_str(d_stage, 20)}")
    print(f"    delta_gyro corner (Ungar)        {mp_to_str(defct['defect'], 20)}")
    print(f"    |eps_signed| (omega_corner)      {mp_to_str(abs(mpj['eps_signed']), 20)}")
    print(f"    delta_BU / 2                     {mp_to_str(delta_bu_closed / 2, 20)}")
    print(f"    delta_gyro BU triangle           {mp_to_str(def_bu_tri['defect'], 20)}")
    print(f"    delta_BU = 2*omega               {mp_to_str(delta_bu_closed, 20)}")
    check(
        state,
        "BU triangle defect = gyr witness",
        def_bu_tri["defect_minus_gyr_angle"] < TOL_MP,
        tier="EXACT_MP",
        quantity="|defect(ONA,BU+,BU-) - angle(gyr[u,⊖v])|",
        measured=f"{float(def_bu_tri['defect_minus_gyr_angle']):.3e}",
        threshold="< 1e-60",
    )
    check(
        state,
        "BU triangle defect equals delta_BU",
        abs(def_bu_tri["defect"] - delta_bu_closed) < TOL_MP,
        tier="EXACT_MP",
        quantity="|defect(ONA,BU+,BU-) - delta_BU|",
        measured=f"{float(abs(def_bu_tri['defect'] - delta_bu_closed)):.3e}",
        threshold="< 1e-60",
    )
    check(
        state,
        "BU triangle defect equals 2 * corner defect",
        abs(def_bu_tri["defect"] - 2 * defct["defect"]) < TOL_MP,
        tier="EXACT_MP",
        quantity="|defect(ONA,BU+,BU-) - 2*defect(origin,ONA,BU+)|",
        measured=f"{float(abs(def_bu_tri['defect'] - 2 * defct['defect'])):.3e}",
        threshold="< 1e-60",
    )
    print()

    state.t = t
    state.gs = gs
    state.mpj = mpj
    state.word = word
    state.pal_mp = pal_mp
    state.rooted_bu = rooted_bu
    state.defct = defct
    state.defct_bu_triangle = def_bu_tri
    state.delta_stage = d_stage
    state.omega_closed = omega_closed
    state.delta_bu_closed = delta_bu_closed
    return state

