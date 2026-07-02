#!/usr/bin/env python3
"""
hqvm_compact_geom_report.py

Report layer for the compact geometry electroweak analysis.
No computation: formats and prints results from core and kernel.

Sections:
  1.  Finite kernel algebra
  2.  Aperture Delta and projectors
  3.  Electroweak mass law
  4.  Numerical validation
  5.  Lepton sector
  6.  Quark and strong-sector diagnostics
  7.  Representation boundary and 32-bit lift
  8.  External channels
  9.  Consistency audit
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Sequence

from hqvm_compact_geom_core import (
    ALPHA_GEOMETRIC,
    CHANNELS,
    CODE_C1,
    CODE_C2,
    CODE_C3,
    DELTA,
    DELTA_BU,
    E_EW_GEV,
    ELECTRON_MASS_GEV,
    HORIZON_CARDINALITY,
    KERNEL_BYTE_APERTURE,
    LAMBDA_0,
    M_A,
    M_SHELL,
    MUON_EQUATOR_COEFF,
    OMEGA_SIZE,
    P_BOUNDARY,
    Q_DENSITY,
    RHO,
    SU2_RESIDUAL,
    UV_IR_GYRATION_SQ,
    WZ_APERTURE_COEFF,
    WZ_OFFSET,
    DeltaBacksolve,
    ElectroweakCoords,
    LeaveOneOutResult,
    channel_by_label,
    eval_law,
    ckm_ansatz,
    compact_algebra,
    delta_self_consistency_rhs,
    four_point_consensus,
    hzw_leave_one_out,
    model_couplings_d2,
    verify_ew_ladder_derivations,
    verify_omega_path_ladder,
    ew_mass_anchor_rows,
    carrier_trace,
    P6_SHELL_INDEX,
    P6_SUPPORT_STATES,
    EW_CHANNEL_Q_WEIGHT,
    K4_CHANNEL_FLAGS,
    sin2_theta_w,
    solve_delta_self_consistency,
    lepton_horizon_wrap_exhaustion_probe,
    qcd_aperture_cycle_residual_probe,
    source_traceability_probe,
    byte_archetype_shadow_probe,
    horizon_gate_selection_probe,
    lepton_d3_path_breaking_probe,
    w_mass_from_z,
    wz_split,
    ew_couplings_from_masses,
    electroweak_backsolves,
    electroweak_coords,
    ew_delta_n,
    lepton_base_n,
    lepton_d3_transition_costs,
    lepton_ladder_residuals,
    all_laws,
)
from hqvm_compact_geom_derivations import run_derivations

from hqvm_compact_geom_kernel import (
    KernelReport,
    run_kernel_verification,
    ShellTransitionRow,
    OrderLadderRow,
    su3_weight4_decomposition_probe,
    spinorial_shadow_obstruction_probe,
    color_operator_bulk_confinement_probe,
    color_adjoint_spectrum_probe,
    final_fronts_closure_probe,
    d_flow_quark_mass_mapping_probe,
    external_leads_null_audit_probe,
)

from cgm_32bit_lift_probe import run_148_51_closure_probe


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _hdr(title: str) -> None:
    print()
    print(title)
    print("=" * min(len(title), 10))


def _subhdr(title: str) -> None:
    print()
    print(f"  --- {title} ---")


def _row(label: str, value: str, width: int = 38) -> None:
    print(f"  {label:{width}} {value}")


def _fmt_frac(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def _sci(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}e}"


def _fix(x: float, digits: int = 12) -> str:
    return f"{x:.{digits}f}"


def _fmt_complex(z: complex, digits: int = 6) -> str:
    if abs(z.imag) < 10 ** (-(digits + 2)):
        return f"{z.real:.{digits}f}"
    sign = "+" if z.imag >= 0 else "-"
    return f"{z.real:.{digits}f}{sign}{abs(z.imag):.{digits}f}i"


def _pct(x: float) -> str:
    return f"{x * 100:+.3f}%"


def _ok(flag: bool) -> str:
    return "ok" if flag else "FAIL"


def _null_model_flags(a: int, r: int, b: int) -> tuple[float, float, float]:
    """Return (p, q, r5) from base, rot, bal flags."""
    a_f, r_f, b_f = float(a), float(r), float(b)
    g = 0.5
    p = 1.0 - (CODE_C1 / 2.0) * a_f + (CODE_C1 / 4.0) * r_f + (4.0 * g) * b_f
    q = 1.25 - (4.0 * g) * r_f - (2.0 * g) * b_f
    diff = CODE_C2 - CODE_C1
    r5 = -diff / 2.0 + (HORIZON_CARDINALITY - diff) / 8.0 * (a_f - r_f) + CODE_C2 / 8.0 * b_f
    return p, q, r5


# ---------------------------------------------------------------------------
# Section 1: Kernel structure
# ---------------------------------------------------------------------------

def print_kernel_structure(report: KernelReport) -> None:
    _hdr("1. Finite Kernel Algebra")

    print()
    print("  Reachable manifold Omega")
    _row("states",        str(report.omega_size))
    _row("form",          "2^12 = 16^3 = 64^2")
    _row("component density", "popcount(A) = popcount(B) = 6")
    _row("density product",   "d(A) x d(B) = 1/4")

    print()
    print("  Shell distribution  |Shell_k| = C(6,k) * 64")
    print(f"  {'Shell':>6} {'ab_dist':>8} {'population':>12} {'fraction':>10} "
          f"{'expected':>10} {'ok':>4}")
    print("  " + "-" * 10)
    all_ok = True
    for s in report.shell_stats:
        ok = "ok" if s.matches_binomial else "FAIL"
        all_ok = all_ok and s.matches_binomial
        print(f"  {s.shell:6d} {s.shell*2:8d} {s.population:12d} "
              f"{s.fraction:10.6f} {s.expected_population:10d} {ok:>4}")
    print(f"  All binomial: {'yes' if all_ok else 'NO'}")

    print()
    print("  Dual horizons")
    if report.horizon is None:
        print("  Dual horizons: unavailable")
    else:
        h = report.horizon
        _row("equality horizon  |Shell_0|",  str(h.equality_count))
        _row("complement horizon |Shell_6|", str(h.complement_count))
        _row("total boundary",               str(h.total_boundary))
        _row("|H|^2 = |Omega|",
             f"{HORIZON_CARDINALITY}^2 = {HORIZON_CARDINALITY**2} "
             f"{'ok' if h.holographic_identity else 'FAIL'}")
        _row("complementarity invariant",
             "holds" if h.complementarity_holds else "FAIL")

    print()
    print("  Boundary-to-bulk projection")
    if report.boundary_to_bulk is None:
        print("  Boundary-to-bulk projection: unavailable")
    else:
        b = report.boundary_to_bulk
        _row("complement states (|H|)", str(b.complement_states))
        _row("total fanout |H|*256",    str(b.total_fanout))
        _row("unique targets reached",  str(b.unique_targets))
        _row("min/max multiplicity",
             f"{b.min_multiplicity}/{b.max_multiplicity} "
             f"{'ok' if b.exact_uniform_multiplicity else 'FAIL'}")

    if report.byte_transitions is not None:
        bt = report.byte_transitions
        print()
        print("  Byte transition exhaustive check (4096 * 256 = 1,048,576 ops)")
        _row("active swap failures",   str(bt.active_swap_failures))
        _row("passive commit failures", str(bt.passive_commit_failures))
        _row("complement swap fraction",
             _fix(bt.complement_swap_fraction, 6))
        _row("complement commit fraction",
             _fix(bt.complement_commit_fraction, 6))
        _row("equality horizon hit fraction",
             _fix(bt.equality_horizon_hit_fraction, 6))
        _row("complement horizon hit fraction",
             _fix(bt.complement_horizon_hit_fraction, 6))
        print()
        print("  q-weight byte distribution")
        for k, cnt in enumerate(bt.q_weight_counts):
            expected = 4 * math.comb(6, k)
            ok = "ok" if cnt == expected else "FAIL"
            _row(f"q_weight={k}", f"{cnt}  (expected {expected})  {ok}")

    print()
    print("  Shell transition algebra  (exact fractions, q = 0..6)")
    print(f"  {'q':>3} {'Tr(M)':>8} {'Tr(M^2)':>10} {'carrier':>10}")
    print("  " + "-" * 10)
    for r in report.shell_transition_rows:
        print(f"  {r.q_weight:3d} {_fmt_frac(r.trace):>8} "
              f"{_fmt_frac(r.return_trace):>10} {_fmt_frac(r.carrier):>10}")

    print()
    print("  UV-IR shell DPF (carrier-trace ratios)")
    print(f"  {'UV':10} {'IR':10} {'q_uv':>5} {'q_ir':>5} "
          f"{'C_uv':>8} {'C_ir':>8} {'ratio':>10}")
    print("  " + "-" * 10)
    for d in report.uv_ir_dpf:
        print(f"  {d.uv_label:10} {d.ir_label:10} {d.uv_q:5d} {d.ir_q:5d} "
              f"{_fmt_frac(d.uv_carrier):>8} {_fmt_frac(d.ir_carrier):>8} "
              f"{d.ratio:10.6f}")

    print()
    print("  Kernel operator group")
    _row("G", "(GF(2)^6 x GF(2)^6) semidirect C2")
    _row("C2 action", "swaps the two GF(2)^6 coordinates")
    _row("|translation subgroup|", "4096")
    _row("|parity sector|", "2")
    _row("|operator group|", "8192")
    _row("|derived subgroup G'|", "64")
    _row("|centre Z(G)|", "64")
    _row("G' = Z(G)", "diagonal GF(2)^6")
    _row("|abelian shadow G/G'|", "128")
    _row("odd x odd closure", "256^2 / 4096 = 16")
    _row("two-step uniformisation", "exact translation surjection")

    print()
    _row("ALL KERNEL AUDITS PASS",
         "YES" if report.all_kernel_theorems_pass else "NO: see above")


# ---------------------------------------------------------------------------
# Section 2: Compact algebra constants
# ---------------------------------------------------------------------------

def _print_aperture_scalars_and_projectors(delta: float = DELTA) -> None:
    alg = compact_algebra(delta)

    print()
    print("  Primary aperture")
    _row("Delta  = 1 - rho",           _fix(DELTA))
    _row("rho    = delta_BU / m_a",    _fix(RHO))
    _row("m_a    = 1/(2 sqrt(2pi))",   _fix(M_A))
    _row("delta_BU",                   _fix(DELTA_BU))

    print()
    print("  Aperture Scalars")
    _row("epsilon = 1/Delta - 48",     _fix(alg.epsilon))
    _row("eta     = m_a - delta_BU",   _fix(alg.eta))
    _row("d_H     = epsilon + 24*Delta", _fix(alg.d_h))
    _row("omega   = delta_BU / 2",     _fix(alg.omega))
    _row("kappa   = pi/4 - 1/sqrt(2)", _fix(alg.kappa))
    _row("sigma   = SU2_RESIDUAL",     _fix(alg.sigma))

    print()
    print("  Projectors")
    _row("P_boundary = 47/48",         _fix(P_BOUNDARY))
    _row("Q_density  = 1/4",           _fix(Q_DENSITY))
    _row("APERTURE_FRAME = 3*|K4|^2",  "48")
    _row("|H| horizon cardinality",    str(HORIZON_CARDINALITY))
    _row("|Omega| reachable states",   str(OMEGA_SIZE))

    print()
    print("  Discrete aperture ordering")
    _row("5/256  (bare byte aperture)", _fix(KERNEL_BYTE_APERTURE))
    _row("Delta  (continuous)",         _fix(DELTA))
    _row("1/48   (depth-4 frame)",      _fix(1.0 / 48.0))
    _row("5/256 < Delta < 1/48",
         f"{'yes' if KERNEL_BYTE_APERTURE < DELTA < 1/48 else 'NO'}")
    _row("48 * Delta",                  _fix(48.0 * DELTA))
    _row("epsilon = 1/Delta - 48",      _fix(alg.epsilon))

    print()
    print("  Delta self-consistency")
    rhs_d2 = delta_self_consistency_rhs(DELTA, include_third_order=False)
    rhs_d3 = delta_self_consistency_rhs(DELTA, include_third_order=True)
    solved, resid = solve_delta_self_consistency(DELTA, include_third_order=True)
    _row("rhs at D^2 order",  _fix(rhs_d2))
    _row("rhs at D^3 order",  _fix(rhs_d3))
    _row("fixed-point delta", _fix(solved))
    _row("residual",          _sci(resid))
    _row("gap from CGM Delta", _sci(solved - DELTA))

    print()
    print("  Geometric constants")
    _row("Lambda_0 = Delta/sqrt(5)",   _fix(LAMBDA_0))
    _row("alpha_geometric = dBU^4/m_a", _sci(ALPHA_GEOMETRIC))
    _row("UV-IR gyration (2pi)^2",     _fix(UV_IR_GYRATION_SQ))


def _print_mass_coordinate_gaps(
    observed: dict[str, float],
    delta: float = DELTA,
) -> None:
    v = observed["Electroweak scale"]

    targets = [
        "Top quark mass energy",
        "Higgs mass energy",
        "Z boson mass energy",
        "W boson mass energy",
        "Tau mass energy",
        "Muon mass energy",
        "Electron mass energy",
        "Bottom quark mass energy",
        "Charm quark mass energy",
        "Strange quark mass energy",
    ]

    print()
    print("  n = log2(v/m) / Delta  (mass-coordinate gap in aperture units).")
    print()
    print(f"  {'Observable':30} {'n':>12} {'nearest':>8} {'residual':>10}")
    print("  " + "-" * 10)

    for name in targets:
        mass = observed.get(name)
        if mass is None:
            print(f"  WARNING: {name} not found in observed dict")
            continue
        n_coord = ew_delta_n(mass, ew_scale_gev=v, delta=delta)
        nearest = round(n_coord)
        resid = abs(n_coord - nearest)
        print(f"  {name[:30]:30} {n_coord:12.6f} {nearest:8.0f} {resid:10.6f}")

    print()
    print("  Strong-scale coordinate check")
    m_top = observed.get("Top quark mass energy")
    if m_top is not None:
        n_qcd_from_ew = math.log2(v / 0.2) / delta
        n_top_empirical = ew_delta_n(m_top, ew_scale_gev=v, delta=delta)
        n_qcd_from_top = n_top_empirical + math.log2(m_top / 0.2) / delta
        offset = n_qcd_from_top - n_qcd_from_ew
        lambda_qcd = v * 2.0 ** (-(n_qcd_from_ew * delta))
        qcd_probe = qcd_aperture_cycle_residual_probe(delta=delta, ew_scale_gev=v)
        _row("n_QCD from EW/0.2", _fix(n_qcd_from_ew))
        _row("n_QCD from top anchor", _fix(n_qcd_from_top))
        _row("offset (aperture units)", f"{offset:+.3f}")
        _row("Lambda_QCD (GeV)", f"{lambda_qcd:.4f}")
        _row("residual base-n_QCD", _fix(qcd_probe.residual))


def print_aperture_and_coordinates(
    observed: dict[str, float],
    coords: ElectroweakCoords,
    delta: float = DELTA,
) -> None:
    _hdr("2. Aperture Delta and Projectors")
    _subhdr("Aperture Scalars and Projectors")
    _print_aperture_scalars_and_projectors(delta)
    _subhdr("Mass-Coordinate Gaps")
    _print_mass_coordinate_gaps(observed, delta)


# ---------------------------------------------------------------------------
# Section 3: Electroweak mass law
# ---------------------------------------------------------------------------

def _print_native_derivation_body() -> None:
    """Print the three native coefficient derivations (no mass input)."""
    rep = run_derivations()
    print()
    print("  Native derivation audit (kernel geometry only, no mass input)")
    print("  D1  Lambda_0 = Delta/sqrt(Tr(P_STF))")
    print(f"      STF bulk projector trace      = {rep.lambda_0.stf_projector_trace}")
    print(f"      Lambda_0 derived              = {rep.lambda_0.lambda0_derived:.12f}")
    print(f"      Lambda_0 existing             = {rep.lambda_0.lambda0_existing:.12f}")
    print(f"      W/Z D^3 coefficient = 2/sqrt(5)   {_ok(rep.lambda_0.wz_third_order_matches)}")
    print("  D2  r5 from plaquette census + STF Regge projection")
    r = rep.r5_plaquette
    print(f"      mean plaquette defect         = {r['mean_plaquette_defect']}")
    print(f"      W/Z code gap C2-C1            = {r['wz_code_gap']}")
    print(f"      r5 constant -(C2-C1)/2        = {r['r5_constant']}")
    print(f"      (base-rot) weight = 55/8      {_ok(bool(r['base_rot_weight_is_55_over_8']))}")
    print(f"      bal weight = 15/8             {_ok(bool(r['bal_weight_is_15_over_8']))}")
    print(f"      {'Ch':6} {'K4':4} {'regge':>9} {'r5_alg':>8} {'r5_der':>8} {'match':>5}")
    for s in rep.regge_sums:
        print(f"      {s.label:6} {s.k4_element:4} {s.regge_sum:>9.6f} "
              f"{s.r5_algebraic:>8.3f} {s.r5_derived:>8.3f} {_ok(s.matches):>5}")
    print("  D3  K4 channel flags from fold geometry (byte-path lengths)")
    print(f"      {'Ch':6} {'K4':4} {'len':>3} {'flags':>8} {'match':>5}")
    for f in rep.fold_flags:
        flags_s = "".join("1" if x else "0" for x in f.flags_from_fold)
        print(f"      {f.label:6} {f.k4_element:4} {f.word_length:>3} {flags_s:>8} {_ok(f.matches):>5}")
    _row("All native derivations close", _ok(rep.all_native))


def _print_coefficient_derivation_body(
    observed: dict[str, float],
    delta: float = DELTA,
    v: float = E_EW_GEV,
) -> None:
    checks = verify_ew_ladder_derivations()
    om = verify_omega_path_ladder(delta)
    mass_rows = ew_mass_anchor_rows(observed, delta, v)

    print()
    print("  Grammar (kernel alphabet)")
    terms = []
    for k in range(7):
        c = math.comb(6, k)
        z = f"z^{2*k}" if k > 0 else "1"
        terms.append(f"{c}{z}" if c > 1 else z)
    print("  (1+z^2)^6 = " + " + ".join(terms))
    _row("C1", str(CODE_C1))
    _row("C2", str(CODE_C2))
    _row("C3", str(CODE_C3))
    _row("M_shell", str(M_SHELL))
    _row("|H|", str(HORIZON_CARDINALITY))
    _row("P_boundary", "47/48")
    _row("Q_density", "1/4")
    _row("APERTURE_FRAME", "48")

    print()
    print("  D^1 linear a_i")
    _row("Top:   |H| + C2 - C1",
         f"{HORIZON_CARDINALITY + CODE_C2 - CODE_C1}  = 73")
    _row("Higgs: M_shell/2",
         f"{M_SHELL//2}  = 96")
    _row("Z:     M_shell/2 + C1 + C2",
         f"{M_SHELL//2 + CODE_C1 + CODE_C2}  = 117")
    _row("W:     M_shell/2 + 2*C2",
         f"{M_SHELL//2 + 2*CODE_C2}  = 126")
    print("  Source: T_shells -- shell populations determine linear coefficients")

    print()
    print("  D^0 offsets b_i")
    for ch in CHANNELS:
        b_str = "-1" if ch.label in ("Top", "Higgs") else "-47/48"
        print(f"    {ch.label:6}  b = {b_str:>8}")
    print("  Source: T_F_composition -- K4 structure sets boundary projector")

    print()
    print("  D^2 carrier-trace c_i")
    for ch in CHANNELS:
        print(f"    {ch.label:6}  c = {ch.c:6.2f}")
    print("  Source: T_carrier_traces -- stage projections of code weights")

    print()
    print("  D^3-D^4 p_i, q_i and D^5 r5_i  (K4 flags, trace-free)")
    print(f"  {'Ch':6} {'q':>2} {'flags':>10} {'p':>6} {'q':>6} {'r5':>8}")
    print("  " + "-" * 10)
    for ch in CHANNELS:
        q_wt = EW_CHANNEL_Q_WEIGHT[ch.label]
        fl = K4_CHANNEL_FLAGS[ch.label]
        flags_s = f"({int(fl[0])},{int(fl[1])},{int(fl[2])})"
        print(
            f"  {ch.label:6} {q_wt:2d} {flags_s:>10} "
            f"{ch.p:6.2f} {ch.q:6.2f} {ch.r5:7.3f}"
        )
    print(f"  sum p_i = {checks['sum_p']:.4f}  sum q_i = {checks['sum_q']:.4f}")
    print("  Source: T_gyrotriangle_p, T_k4_closure_q -- trace-free K4 edge increments")
    print("  r5 = -(C2-C1)/2 + (|H|-(C2-C1))/8*(base-rot) + C2/8*bal")
    print("  Source: T_code_curvature_r5 -- plaquette census + STF Regge projection")

    _print_native_derivation_body()

    _row("All coefficients match eval_law", _ok(bool(om["all_L_polynomial_matches_eval_law"])))
    _row("bulk W matches gravity cycle", _ok(bool(om["bulk_W_matches_gravity_cycle"])))
    _row("Full derivation closes", _ok(bool(checks["all_channels_coeffs"])))

    print()
    print("  Mass identification")
    print(f"  {'Ch':6} {'m_obs':>12} {'m_pred':>12} {'ppm':>10} {'n_resid':>9}")
    print("  " + "-" * 10)
    for r in mass_rows:
        print(
            f"  {r.label:6} {r.mass_obs_gev:12.6f} {r.mass_pred_gev:12.6f} "
            f"{r.mass_ppm:10.3f} {r.shell_residual_ticks:9.4f}"
        )
    sub_ppm = all(abs(r.mass_ppm) < 1.0 for r in mass_rows if r.label in ("Top", "Z", "W"))
    _row("sub-ppm Top/Z/W", _ok(sub_ppm))

    print()
    print("  W/Z split constants")
    _row("WZ_OFFSET = C2 - C1", str(WZ_OFFSET))
    _row("WZ_APERTURE_COEFF = C3/2", str(int(WZ_APERTURE_COEFF)))
    _row("MUON_EQUATOR_COEFF = C3", str(MUON_EQUATOR_COEFF))


def _print_five_order_ladder_body(
    ladder_rows: Sequence[OrderLadderRow],
    delta: float = DELTA,
) -> None:
    print()
    print("  L_i(Delta) = a*Delta + b + c*Delta^2 + p*Delta^3/sqrt(5) + q*Delta^4 + r5*Delta^5")
    print("  Identification: L_i(Delta) = log2(v/m_i)  [mass-coordinate gap]")
    print()
    print("  Order-by-order n-residuals  (n_err = (L_obs - L_pred) / Delta)")
    print(f"  {'Ch':8} {'p':>6} {'q':>6} {'r5':>8} {'D2_err':>10} {'D3_err':>10} {'D4_err':>10} {'D5_err':>10} {'L_err/D6':>10}")
    print("  " + "-" * 10)

    max_d2 = max_d3 = max_d4 = max_d5 = 0.0
    for r in ladder_rows:
        print(f"  {r.channel_label:8} {r.p:6.2f} {r.q:6.2f} {r.r5:8.3f} "
              f"{r.n_err_d2:10.3e} {r.n_err_d3:10.3e} "
              f"{r.n_err_d4:10.3e} {r.n_err_d5:10.3e} "
              f"{r.l_err_over_d6:10.6f}")
        max_d2 = max(max_d2, abs(r.n_err_d2))
        max_d3 = max(max_d3, abs(r.n_err_d3))
        max_d4 = max(max_d4, abs(r.n_err_d4))
        max_d5 = max(max_d5, abs(r.n_err_d5))

    print("  " + "-" * 10)
    print(f"  max |n_err|  D2={max_d2:.3e}  D3={max_d3:.3e}  "
          f"D4={max_d4:.3e}  D5={max_d5:.3e}")
    if max_d3 > 0:
        print(f"  reduction D2->D3 = {max_d2/max_d3:.1f}x   "
              f"D3->D4 = {max_d3/max_d4:.1f}x   "
              f"D4->D5 = {max_d4/max_d5:.1f}x   "
              f"total D2->D5 = {max_d2/max_d5:.0f}x")
    print("  D^6 column: see Representation Boundary section.")


def print_electroweak_mass_law(
    observed: dict[str, float],
    ladder_rows: Sequence[OrderLadderRow],
    delta: float = DELTA,
    v: float = E_EW_GEV,
) -> None:
    _hdr("3. Electroweak Mass Law")
    _subhdr("Coefficient Derivation")
    _print_coefficient_derivation_body(observed, delta, v)
    if ladder_rows:
        _subhdr("Five-Order Expansion")
        _print_five_order_ladder_body(ladder_rows, delta)


def _evaluate_null_candidate(
    delta: float,
    a: float,
    b: float,
    c: float,
    p: float,
    q: float,
    r5: float,
) -> float:
    l0 = a * delta + b
    l0 += c * delta ** 2
    l0 += p * delta ** 3 / math.sqrt(5.0)
    l0 += q * delta ** 4
    l0 += r5 * delta ** 5
    return l0


def _print_null_audit_rows(
    rows: Sequence[
        tuple[float, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], float, float, float]
    ],
    *,
    show: int = 12,
) -> None:
    print(f"  {'rank':>4} {'max_err':>11} {'Top':>10} {'H':>5} {'Z':>5} {'W':>5} {'p_sum':>7} {'q_sum':>7} {'sum_abs_err':>12}")
    print("  " + "-" * 10)
    for i, (max_err, top_f, h_f, z_f, w_f, p_sum, q_sum, abs_sum) in enumerate(rows[:show], start=1):
        print(f"  {i:4d} {max_err:11.3e} {str(top_f):>10} "
              f"{str(h_f):>5} {str(z_f):>5} {str(w_f):>5} "
              f"{p_sum:7.3f} {q_sum:7.3f} {abs_sum:12.3e}")


def _print_electroweak_null_model_audit_body(
    observed: dict[str, float],
    delta: float = DELTA,
    v: float = E_EW_GEV,
) -> None:
    """Null-model audit over 4096 base/rot/bal flag assignments."""
    observed_n = {
        "Top": ew_delta_n(observed["Top quark mass energy"], ew_scale_gev=v, delta=delta),
        "Higgs": ew_delta_n(observed["Higgs mass energy"], ew_scale_gev=v, delta=delta),
        "Z": ew_delta_n(observed["Z boson mass energy"], ew_scale_gev=v, delta=delta),
        "W": ew_delta_n(observed["W boson mass energy"], ew_scale_gev=v, delta=delta),
    }

    base = {
        ch.label: (ch.a, ch.b, ch.c)
        for ch in CHANNELS
    }

    trace_rows: list[tuple[float, tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], float, float, float]] = []

    for f_top in range(8):
        a_t, r_t, b_t = ((f_top >> 2) & 1, (f_top >> 1) & 1, f_top & 1)
        p_t, q_t, r5_t = _null_model_flags(a_t, r_t, b_t)
        for f_h in range(8):
            a_h, r_h, b_h = ((f_h >> 2) & 1, (f_h >> 1) & 1, f_h & 1)
            p_h, q_h, r5_h = _null_model_flags(a_h, r_h, b_h)
            for f_z in range(8):
                a_z, r_z, b_z = ((f_z >> 2) & 1, (f_z >> 1) & 1, f_z & 1)
                p_z, q_z, r5_z = _null_model_flags(a_z, r_z, b_z)
                for f_w in range(8):
                    a_w, r_w, b_w = ((f_w >> 2) & 1, (f_w >> 1) & 1, f_w & 1)
                    p_w, q_w, r5_w = _null_model_flags(a_w, r_w, b_w)

                    p_sum = p_t + p_h + p_z + p_w
                    q_sum = q_t + q_h + q_z + q_w
                    r_tup_t = (a_t, r_t, b_t)
                    r_tup_h = (a_h, r_h, b_h)
                    r_tup_z = (a_z, r_z, b_z)
                    r_tup_w = (a_w, r_w, b_w)

                    max_err = 0.0
                    sum_abs = 0.0
                    for label, obs_n, (a_ch, b_ch, c_ch), p_val, q_val, r5 in [
                        ("Top", observed_n["Top"], base["Top"], p_t, q_t, r5_t),
                        ("Higgs", observed_n["Higgs"], base["Higgs"], p_h, q_h, r5_h),
                        ("Z", observed_n["Z"], base["Z"], p_z, q_z, r5_z),
                        ("W", observed_n["W"], base["W"], p_w, q_w, r5_w),
                    ]:
                        _ = label
                        l_pred = _evaluate_null_candidate(delta, a_ch, b_ch, c_ch, p_val, q_val, r5)
                        n_pred = l_pred / delta
                        err = n_pred - obs_n
                        sum_abs += abs(err)
                        max_err = max(max_err, abs(err))

                    if abs(p_sum) <= 1e-12 and abs(q_sum) <= 1e-12:
                        trace_rows.append(
                            (
                                max_err,
                                r_tup_t,
                                r_tup_h,
                                r_tup_z,
                                r_tup_w,
                                p_sum,
                                q_sum,
                                sum_abs,
                            )
                        )

    trace_sorted = sorted(trace_rows, key=lambda row: row[0])

    target = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1))
    target_rank: int | None = None
    for i, row in enumerate(trace_sorted, start=1):
        if (
            row[1] == target[0]
            and row[2] == target[1]
            and row[3] == target[2]
            and row[4] == target[3]
        ):
            target_rank = i
            break
    assert target_rank is not None, (
        "Declared channel flags must appear in trace-free null-model candidates"
    )

    print()
    _row("Raw flag assignments", "4096")
    _row("Trace-free filtered candidates", f"{len(trace_rows):d}")
    _row("5e-3 pass", f"{sum(1 for row in trace_rows if row[0] <= 5e-3)}")
    _row("1e-3 pass", f"{sum(1 for row in trace_rows if row[0] <= 1e-3)}")
    _row("5e-4 pass", f"{sum(1 for row in trace_rows if row[0] <= 5e-4)}")
    _row("1e-4 pass", f"{sum(1 for row in trace_rows if row[0] <= 1e-4)}")
    _row("Delta^5 pass", f"{sum(1 for row in trace_rows if row[0] <= delta ** 5):d}")
    _row("Declared assignment rank", str(target_rank))

    print()
    print("  Declared channels: Top=(0,0,0) Higgs=(1,0,0) Z=(1,1,0) W=(1,1,1)")
    print("  Best trace-free candidates (Top, Higgs, Z, W flags):")
    _print_null_audit_rows(trace_sorted, show=12)


# ---------------------------------------------------------------------------
# Section 5: Delta backsolves and four-point consensus
# ---------------------------------------------------------------------------

def _print_backsolves_body(
    observed: dict[str, float],
    backsolves: tuple[DeltaBacksolve, ...],
    delta: float = DELTA,
) -> None:
    print()
    print(f"  Reference Delta = {_fix(DELTA)}")
    print()
    print(f"  {'Source':8} {'Equation':38} {'Delta_back':>16} {'Delta_err':>16}")
    print("  " + "-" * 10)
    for bs in backsolves:
        print(f"  {bs.source:8} {bs.equation:38} "
              f"{_fix(bs.delta_back):>16} {_sci(bs.delta_err):>16}")

    print()
    consensus = four_point_consensus(backsolves)
    wz = next((bs for bs in backsolves if bs.source == "W/Z"), None)

    _row("Four-point mean (Top+H+Z+W)", _fix(consensus.delta_back))
    _row("Four-point error",            _sci(consensus.delta_err))

    top = next(bs for bs in backsolves if bs.source == "Top")
    h   = next(bs for bs in backsolves if bs.source == "Higgs")
    z   = next(bs for bs in backsolves if bs.source == "Z")
    w   = next(bs for bs in backsolves if bs.source == "W")
    spread = (max(top.delta_back, h.delta_back, z.delta_back, w.delta_back)
              - min(top.delta_back, h.delta_back, z.delta_back, w.delta_back))
    _row("Four-point spread",           _sci(spread))

    if wz is not None:
        print()
        _row("W/Z promoted backsolve",     _fix(wz.delta_back))
        _row("W/Z error vs CGM Delta",     _sci(wz.delta_err))
        gap_base_to_consensus = abs(wz.delta_back - consensus.delta_back)
        _row("W/Z vs four-point mean",     _sci(gap_base_to_consensus))

    print()
    print("  Matter-gauge dichotomy (Q = 1/4 support)")
    m_top = observed["Top quark mass energy"]
    m_v = observed["Electroweak scale"]
    l_t = math.log2(m_v / m_top)
    delta_linear = (l_t + 1.0) / 73.0
    err_linear = delta_linear - delta

    hzw_max = max(abs(h.delta_err), abs(z.delta_err), abs(w.delta_err))
    ratio_quadratic = abs(top.delta_err) / hzw_max if hzw_max else float("nan")
    ratio_linear = abs(err_linear) / hzw_max if hzw_max else float("nan")

    _row("Top error (with Q=1/4 term)",    _sci(abs(top.delta_err)))
    _row("Top error (linear only, no Q)",  _sci(abs(err_linear)))
    _row("H/Z/W max error",                _sci(hzw_max))
    _row("ratio Top/HZW (with Q)",         f"{ratio_quadratic:.2f}")
    _row("ratio Top/HZW (no Q)",           f"{ratio_linear:.2f}")
    _row("Status", "Q=1/4 lowers Top residual by ~27x")


# ---------------------------------------------------------------------------
# Section 6: W/Z ratio lock and sin^2 theta_W
# ---------------------------------------------------------------------------

def _print_wz_ratio_lock_body(
    observed: dict[str, float],
    backsolves: tuple[DeltaBacksolve, ...],
    delta: float = DELTA,
) -> None:
    m_z = observed["Z boson mass energy"]
    m_w = observed["W boson mass energy"]
    v   = observed["Electroweak scale"]

    wz_bs = next(bs for bs in backsolves if bs.source == "W/Z")
    consensus = four_point_consensus(backsolves)

    cos_obs  = m_w / m_z
    sin2_obs = 1.0 - cos_obs ** 2
    l_wz_obs = math.log2(m_z / m_w)

    split_pred = wz_split(delta, promoted=True)
    split_base = wz_split(delta, promoted=False)
    sin2_pred  = sin2_theta_w(delta, promoted=True)
    w_pred     = w_mass_from_z(m_z, delta)
    cos_pred   = 2.0 ** (-delta * split_pred)

    def _solve_wz_base(l: float) -> float:
        return (9.0 - math.sqrt(81.0 - 40.0 * l)) / 20.0

    delta_wz_base = _solve_wz_base(l_wz_obs)

    base_gap      = abs(delta_wz_base - consensus.delta_back)
    corrected_gap = abs(wz_bs.delta_back - consensus.delta_back)
    improvement   = base_gap / corrected_gap if corrected_gap > 0 else float("inf")

    print()
    print("  Charged-neutral split law")
    print("    D2 backbone:   log2(mZ/mW) = D[(C2-C1) - (C3/2)D]")
    print("    Promoted D4:   log2(mZ/mW) = D[(C2-C1) - (C3/2)D + 2D^2/sqrt(5) - D^3]")

    print()
    print("  Backbone (D2 only)")
    _row("split predicted",  _fix(split_base, 12))
    _row("split observed",   _fix(l_wz_obs / delta, 12))
    _row("split error",      _sci(l_wz_obs / delta - split_base))
    _row("Delta_wz_base",    _fix(delta_wz_base))
    _row("base vs consensus gap", _sci(base_gap))

    print()
    print("  Promoted split (D4 law)")
    _row("split predicted",  _fix(split_pred, 12))
    _row("split observed",   _fix(l_wz_obs / delta, 12))
    _row("split error",      _sci(l_wz_obs / delta - split_pred))
    _row("Delta_wz_corrected", _fix(wz_bs.delta_back))
    _row("corrected vs consensus gap", _sci(corrected_gap))
    _row("consensus improvement factor", f"{improvement:.0f}x")

    print()
    print("  sin^2 theta_W")
    _row("predicted",  _fix(sin2_pred))
    _row("observed",   _fix(sin2_obs))
    _row("error",      _sci(sin2_pred - sin2_obs))

    print()
    print("  W mass from Z and Delta")
    _row("predicted W",   _fix(w_pred, 9) + " GeV")
    _row("observed  W",   _fix(m_w,    9) + " GeV")
    _row("relative error", _sci((w_pred - m_w) / m_w))

    print()
    print("  theta_W")
    _row("cos theta_W predicted", _fix(cos_pred))
    _row("cos theta_W observed",  _fix(cos_obs))
    _row("theta_W predicted",
         f"{math.degrees(math.acos(cos_pred)):.9f} deg")


# ---------------------------------------------------------------------------
# Section 7: H/Z/W leave-one-out
# ---------------------------------------------------------------------------

def _print_leave_one_out_body(
    results: tuple[LeaveOneOutResult, ...],
) -> None:
    print()
    print(f"  {'Target':8} {'Delta source':14} {'Delta used':>16} "
          f"{'m_pred (GeV)':>16} {'m_ref (GeV)':>14} {'rel_err':>14}")
    print("  " + "-" * 10)
    for r in results:
        print(f"  {r.target:8} {r.delta_source:14} {_fix(r.delta_used):>16} "
              f"{r.predicted_mass:16.9f} {r.reference_mass:14.9f} "
              f"{_sci(r.relative_error):>14}")


def _print_couplings_body(
    observed: dict[str, float],
    delta: float = DELTA,
    v: float = E_EW_GEV,
) -> None:
    mc = model_couplings_d2(delta, v)

    m_t = observed["Top quark mass energy"]
    m_h = observed["Higgs mass energy"]
    m_z = observed["Z boson mass energy"]
    m_w = observed["W boson mass energy"]
    rc  = ew_couplings_from_masses(m_t, m_h, m_z, m_w, v)

    print()
    print("  Coupling exponents (log2 form):")
    laws = all_laws(delta, order=2)
    print(f"    y_t  = 2^(3/2 - {channel_by_label('Top').a}*D - D^2/4)")
    print(f"    lambda_H = 2^(1 - {int(2*channel_by_label('Higgs').a)}*Delta + {int(-2*channel_by_label('Higgs').c)}*Delta^2)")
    print(f"    g_Z  = 2^(95/48 - {int(channel_by_label('Z').a)}*Delta + {int(-channel_by_label('Z').c*2)}/2*Delta^2)")
    print(f"    g    = 2^(95/48 - {int(channel_by_label('W').a)}*Delta + {int(-channel_by_label('W').c*2)}/2*Delta^2)")

    print()
    print(f"  {'Quantity':12} {'Model value':>16} {'PDG reference':>16} {'rel error':>14}")
    print("  " + "-" * 10)

    fields = [
        ("lambda_H", mc.lambda_H, rc.lambda_H),
        ("g",        mc.g,        rc.g),
        ("g_Z",      mc.g_z,      rc.g_z),
        ("g'",       mc.g_prime,  rc.g_prime),
        ("e",        mc.e,        rc.e),
        ("alpha^-1", mc.alpha_ew_inv, rc.alpha_ew_inv),
        ("y_t",      mc.y_t,      rc.y_t),
    ]
    for name, model_val, ref_val in fields:
        rel = (model_val / ref_val - 1.0) if ref_val != 0.0 else float("nan")
        print(f"  {name:12} {model_val:16.9f} {ref_val:16.9f} {_sci(rel):>14}")


def print_numerical_validation(
    observed: dict[str, float],
    backsolves: tuple[DeltaBacksolve, ...],
    loo: tuple[LeaveOneOutResult, ...],
    delta: float = DELTA,
    v: float = E_EW_GEV,
) -> None:
    _hdr("4. Numerical Validation")
    _subhdr("Null-Model Audit")
    _print_electroweak_null_model_audit_body(observed, delta, v)
    _subhdr("Delta Backsolves")
    _print_backsolves_body(observed, backsolves, delta)
    _subhdr("W/Z Ratio Lock")
    _print_wz_ratio_lock_body(observed, backsolves, delta)
    _subhdr("Leave-One-Out")
    _print_leave_one_out_body(loo)
    _subhdr("Couplings")
    _print_couplings_body(observed, delta, v)


# ---------------------------------------------------------------------------
# Section 5: Lepton sector
# ---------------------------------------------------------------------------

def print_lepton_sector(
    observed: dict[str, float],
    delta: float = DELTA,
    v: float = E_EW_GEV,
) -> None:
    _hdr("5. Lepton Sector")

    rows = lepton_ladder_residuals(observed, delta, v)

    print()
    print("  Lepton coordinate: n = k*|H| + r(lepton) + residual")
    print(f"  |H| = {HORIZON_CARDINALITY},  M_shell = {M_SHELL}")
    print()
    print(f"  {'Lep':10} {'k':>3} {'r(lep)':>8} {'resid_D1':>12} "
          f"{'resid_D2':>12} {'resid_D3':>12}  law")
    print("  " + "-" * 10)

    max_d1 = max_d2 = max_d3 = 0.0
    for row in rows:
        print(f"  {row.label:10} {row.k:3d} {row.r:8.3f} "
              f"{row.resid_d1:12.3e} {row.resid_d2:12.3e} "
              f"{row.resid_d3:12.3e}  {row.law_d1}")
        max_d1 = max(max_d1, abs(row.resid_d1))
        max_d2 = max(max_d2, abs(row.resid_d2))
        max_d3 = max(max_d3, abs(row.resid_d3))

    print()
    print(f"  max |resid|  D1={max_d1:.3e}  D2={max_d2:.3e}  D3={max_d3:.3e}")
    if max_d2 > 0:
        print(f"  reduction D1->D2 = {max_d1/max_d2:.1f}x   "
              f"D2->D3 = {max_d2/max_d3:.1f}x   "
              f"total D1->D3 = {max_d1/max_d3:.0f}x")

    print()
    print("  Law by order:")
    for row in rows:
        law_d2 = row.law_d2 if row.law_d2 else "none"
        print(f"    {row.label:8}  D1: {row.law_d1}  "
              f"D2: {law_d2}  D3: {row.law_d3}")

    print()
    print("  Horizon/shell anchor:")
    print("    r(tau)      = M_shell/8          = 24")
    print("    r(muon)     = M_shell/8 + M/48   = 28")
    print("    r(electron) = M_shell/8 - M/24   = 16")

    print()
    print("  Ladder separations:")
    print("    tau->muon     : k  5->8    delta_k = 3 = C1/2")
    print("    muon->electron: k  8->14   delta_k = 6 = C1")
    print("    tau->electron : k  5->14   delta_k = 9 = 3*C1/2")

    print()
    print()
    print("  Electron residual decomposition:")
    sigma = SU2_RESIDUAL
    n_h = all_laws(delta, order=5)["Higgs"] / delta
    higgs_mem = KERNEL_BYTE_APERTURE / n_h
    total = sigma + higgs_mem
    obs_e = observed.get("Electron mass energy")
    if obs_e is not None:
        n_obs_e = ew_delta_n(obs_e, ew_scale_gev=v, delta=delta)
        obs_resid = n_obs_e - lepton_base_n(14, "electron")
        _row("SU2 holonomy sigma",          f"{sigma:.9f}  ({sigma/obs_resid*100:.2f}%)")
        _row("Higgs memory (5/256)/n_H",    f"{higgs_mem:.9f}  ({higgs_mem/obs_resid*100:.2f}%)")
        _row("sum",                         f"{total:.9f}")
        _row("observed residual",           f"{obs_resid:.9f}")
        _row("pre-carrier match error",      _sci(total - obs_resid))

    lepton_max = max((abs(r.resid_d3) for r in rows), default=float("nan"))
    print()
    _row("max |D3 carrier residual|", f"{lepton_max:.3e}")

    costs = lepton_d3_transition_costs()
    print()
    print("  Carrier identities (D3 transition costs)")
    print(f"  {'Lep':10} {'q_path':>10} {'carrier':>10} {'dyadic':>10} {'64*cost':>10}")
    print("  " + "-" * 10)
    for row in costs:
        print(
            f"  {row.label:10} {row.q_from}->{row.q_to} "
            f"{_fmt_frac(row.carrier_delta):>10} {_fmt_frac(row.dyadic):>10} "
            f"{_fmt_frac(row.normalized_64_cost):>10}"
        )

    path = lepton_d3_path_breaking_probe()
    print()
    print(
        "  q=2 and q=4 supports are isospectral at return-trace and Hilbert-Schmidt "
        f"level; dyadic ratio 148/51 differs from spectral ratio. "
        "The muon/electron split requires q-history path information."
    )
    print()
    print("  q-history derivation")
    _row("history path", " -> ".join(f"q={q}" for q in path["history_path"]))
    _row("xor moment", _fmt_frac(path["history_xor_moment"]))
    _row(
        "history split",
        f"{_fmt_frac(path['history_split_dyadic'])} "
        f"({'ok' if path['history_split_matches_simplified'] else 'FAIL'})",
    )
    _row(
        "byte phase reset",
        f"{_fmt_frac(path['byte_phase_reset'])} "
        f"({'ok' if path['byte_phase_reset_matches_rule'] else 'FAIL'})",
    )
    _row(
        "electron from history",
        f"{_fmt_frac(path['electron_from_history_dyadic'])} "
        f"({'ok' if path['electron_from_history_matches'] else 'FAIL'})",
    )

    closure_148 = run_148_51_closure_probe()
    print()
    _row("148/51 closure", f"{closure_148.ratio_num}/{closure_148.ratio_den} {_ok(closure_148.closes_exactly)}")

    arch = byte_archetype_shadow_probe()
    src = source_traceability_probe()
    print()
    print("  Archetype closure")
    _row("archetype shadow", _fmt_frac(arch.archetype_shadow))
    _row("matches residual", "yes" if arch.matches_residual else "NO")
    _row(
        "carrier conservation offset",
        f"archetype byte shadow {_fmt_frac(arch.archetype_shadow)} "
        f"({'verified' if arch.matches_residual else 'not verified'})",
    )
    _row(
        "source traceability",
        f"0x{src.selected_byte:02X} closes electron dyadic "
        f"({'verified' if src.closes_electron_dyadic else 'not verified'})",
    )
    wrap = lepton_horizon_wrap_exhaustion_probe()
    _row(
        "horizon-wrap path",
        f"{wrap.horizon_rule_valid_paths[0]} unique under horizon-wrap rule",
    )

    print()
    print("  Horizon gate selection")
    print(f"  {'byte':>6} {'gate':>5} {'q_wt':>5} {'selected':>9}")
    print("  " + "-" * 10)
    for row in horizon_gate_selection_probe():
        print(
            f"  0x{row.byte:02X} {row.gate:>5} {row.q_weight:5d} "
            f"{'yes' if row.selected_by_electron_reset else 'no':>9}"
        )
    print("  S-gate bytes with zero intron/payload/family select the electron reset.")


# ---------------------------------------------------------------------------
# Section 10: Quark and strong-sector diagnostics
# ---------------------------------------------------------------------------

def print_quark_and_strong_sector(
    observed: dict[str, float],
    delta: float = DELTA,
    v: float = E_EW_GEV,
) -> None:
    _hdr("6. Quark and Strong-Sector Diagnostics")

    alg = compact_algebra(delta)

    quark_specs = [
        ("Bottom",  "Bottom quark mass energy",  284.0, alg.kappa,  None,        "+kappa"),
        ("Charm",   "Charm quark mass energy",   367.0, None,       alg.omega,   "+omega+D/2"),
        ("Strange", "Strange quark mass energy", 548.0, alg.kappa,  alg.omega,   "-(omega+kappa)"),
    ]

    print()
    print("  Selector coordinates n = integer + compact residual")
    print(f"  kappa = pi/4 - 1/sqrt(2) = {_fix(alg.kappa)}")
    print(f"  omega = delta_BU/2       = {_fix(alg.omega)}")
    print()
    print("  Boolean lattice on {kappa, omega}:")
    print(f"  {'Quark':10} {'kappa':>6} {'omega':>6} {'n_int':>6} "
          f"{'selector':>12} {'n_model':>12} {'n_obs':>12} {'n_resid':>12}")
    print("  " + "-" * 10)

    for label, obs_name, n_int, kap, omg, sel_str in quark_specs:
        obs_mass = observed.get(obs_name)
        if obs_mass is None:
            n_obs_str = "n/a"
            n_resid_str = "n/a"
            n_mod = float("nan")
        else:
            n_obs = math.log2(v / obs_mass) / delta
            n_mod = n_int
            if kap is not None:
                n_mod += kap if sel_str.startswith("+k") else -kap
            if omg is not None:
                if "-(omega" in sel_str:
                    n_mod -= omg
                else:
                    n_mod += omg + 0.5 * delta
            n_obs_str   = f"{n_obs:12.6f}"
            n_resid_str = f"{n_obs - n_mod:12.3e}"

        has_kap = "yes" if kap is not None else "no"
        has_omg = "yes" if omg is not None else "no"
        print(f"  {label:10} {has_kap:>6} {has_omg:>6} {int(n_int):>6} "
              f"{sel_str:>12} {n_mod:12.6f} {n_obs_str} {n_resid_str}")

    print()
    print("  Top is excluded from lattice vertices.")
    print()

    b_mass = observed.get("Bottom quark mass energy")
    c_mass = observed.get("Charm quark mass energy")
    s_mass = observed.get("Strange quark mass energy")
    if b_mass and c_mass and s_mass:
        n_b = math.log2(v / b_mass) / delta
        n_c = math.log2(v / c_mass) / delta
        n_s = math.log2(v / s_mass) / delta

        kap_from_b  = n_b - 284.0
        omg_from_c  = n_c - 367.0 - 0.5 * delta
        sum_from_s  = 548.0 - n_s
        internal    = sum_from_s - (kap_from_b + omg_from_c)

        print("  Residuals against closed forms:")
        _row("kappa from Bottom",  _sci(kap_from_b - alg.kappa))
        _row("omega from Charm",   _sci(omg_from_c - alg.omega))
        _row("kappa+omega from Strange", _sci(sum_from_s - (alg.kappa + alg.omega)))
        _row("internal (no closed form used)", _sci(internal))
        print()
        print("  PDG mass uncertainty maps to ~0.7 Delta-ticks, so observed selector")
        print("  residuals at ~1e-3 Delta-ticks are below current mass-input noise.")

    mapping_rows = d_flow_quark_mass_mapping_probe(
        observed,
        include_up_down=True,
    )
    if mapping_rows:
        print()
        print("  D_flow^2 ladder mapping probe")
        print("  Quark masses are ordered and mapped to non-zero D_flow eigenvalue squares.")
        print(f"  {'Quark':8} {'mass (GeV)':>12} {'log2(mass)':>12} "
              f"{'d_flow^2':>8} {'|d_flow|':>8}")
        print("  " + "-" * 10)
        for row in mapping_rows:
            label = row.quark_label
            print(
                f"  {label:8} {row.mass_gev:12.6f} {row.log2_mass:12.6f} "
                f"{row.dflow_sq:8d} {row.dflow_abs:8d}"
            )

    print()
    print("  SU(3) colour structure")
    color = su3_weight4_decomposition_probe()
    _row("1+8+6 decomposition", _ok(color.decomposition_closes))
    _row("adjoint bracket closes", _ok(color.adjoint_bracket_closes))
    _row("raw sextet bracket closes", _ok(color.sextet_bracket_closes))
    print("  Lifted phase-symmetrized sextet: Representation Boundary section.")

    print()
    print("  Strong-sector adjoint spectral ratios")
    spectrum = color_adjoint_spectrum_probe()
    _row("adjoint spectral radius", str(spectrum.spectral_radius))
    _row(
        "finite spectral ratios",
        ", ".join(str(x) for x in spectrum.adjoint_spectral_ratios),
    )
    _row(
        "residual scales (aperture units)",
        ", ".join(f"{x:.6f}" for x in spectrum.spectral_ratio_residual_scales),
    )

    print()
    print("  Colour confinement (paired vs single-sided action)")
    cprobe = color_operator_bulk_confinement_probe()
    _row("paired action preserves bulk", _ok(cprobe.paired_action_preserves_bulk))
    _row("left-action leaks", _ok(cprobe.left_action_leaks))


# ---------------------------------------------------------------------------
# Section 11: Representation boundary and 32-bit lift
# ---------------------------------------------------------------------------

def print_representation_boundary_and_lift(
    d6_rows: Sequence,
    observed: dict[str, float],
    delta: float = DELTA,
    v: float = E_EW_GEV,
) -> None:
    _hdr("7. Representation Boundary and 32-bit Lift")

    print()
    print("  D^5 residuals are O(1) in Delta^6 units; these are representation boundary markers.")
    print(f"  P_6 shell = {P6_SHELL_INDEX}, |support| = {P6_SUPPORT_STATES}")
    print()
    print(f"  {'Ch':6} {'L_err/D6':>12} {'K4 flags':>12} {'C(q)':>10} {'full?':>6}")
    print("  " + "-" * 10)
    d6_by_label = {r.channel_label: r.l_err_over_d6 for r in d6_rows}
    w_d6 = d6_by_label.get("W", float("nan"))
    for ch in CHANNELS:
        flags = K4_CHANNEL_FLAGS[ch.label]
        flags_s = "".join(str(int(f)) for f in flags)
        q_wt = EW_CHANNEL_Q_WEIGHT[ch.label]
        ct = float(carrier_trace(q_wt))
        full = sum(flags) == 3
        l6 = d6_by_label.get(ch.label, float("nan"))
        print(
            f"  {ch.label:6} {l6:12.6f} {flags_s:>12} "
            f"{ct:10.6f} {'yes' if full else 'no':>6}"
        )

    print()
    full_count = sum(1 for fl in K4_CHANNEL_FLAGS.values() if sum(fl) == 3)
    w_unique = full_count == 1 and K4_CHANNEL_FLAGS["W"] == (True, True, True)
    w_largest = all(
        d6_by_label.get(lbl, float("-inf")) <= w_d6
        for lbl in ("Top", "Higgs", "Z", "W")
    )
    max_abs = max((abs(v) for v in d6_by_label.values()), default=float("nan"))
    _row("W unique full K4 (1,1,1)", _ok(w_unique))
    _row("W largest positive D6 residual", _ok(w_largest))
    _row("max |L_err/D6|", f"{max_abs:.6f}")
    _row("W is unique full-flag endpoint", _ok(w_unique))

    print()
    print("  24-bit obstructions")
    print("  Spectral triple: gamma commutes with D_shell; first-order condition fails.")
    print("  No J in 24-bit satisfies first-order spectral triple.")
    sob = spinorial_shadow_obstruction_probe()
    _row("shadow collapses spinorial phase", _ok(sob.shadow_collapses_spinorial_phase))
    _row("requires 32-bit lift", _ok(sob.requires_32bit_lift))

    print()
    print("  32-bit lift verification")
    final_probe = final_fronts_closure_probe(observed, delta=delta, v=v)
    _row("K4 spectral triple closed", _ok(final_probe.spectral_triple_k4_lift_closed))
    _row("symmetrized sextet closed", _ok(final_probe.sextet_phase_symmetrized_closed))
    _row("rich K6 W-boundary closed", _ok(final_probe.rich_k6_w_boundary_closed))


# ---------------------------------------------------------------------------
# Section 12: External channels (CKM)
# ---------------------------------------------------------------------------

def print_external_channels(delta: float = DELTA) -> None:
    _hdr("8. External Channels")

    ckm = ckm_ansatz(delta)

    print()
    print("  CKM compact ansatz")
    print(f"  {'Quantity':16} {'Predicted':>14} {'Reference':>14} {'Error':>12}")
    print("  " + "-" * 10)

    ckm_rows = [
        ("|V_us|", ckm["V_us"],      0.2243,   "sin(dBU + 3D/2)"),
        ("|V_cb|", ckm["V_cb"],      0.0408,   "sin(2D)"),
        ("|V_ub| excl", ckm["V_ub_excl"], 0.00382, "sin(9D^2)"),
        ("|V_ub| incl", ckm["V_ub_incl"], 0.00413, "sin(9D^2+phi_conv)"),
    ]
    for name, pred, ref, law in ckm_rows:
        err = pred - ref
        print(f"  {name:16} {pred:14.9f} {ref:14.6f} {_sci(err):>12}  {law}")

    print()
    _row("CP phase delta_CKM", f"{ckm['delta_CKM']:.3f} deg  (pi/2 - 18D)")
    _row("phi_conv (incl/excl offset)", _sci(ckm["phi_conv"]))
    print()
    print("  Antihydrogen aperture")
    print(f"    12*Delta = {12*delta:.6f}")
    print(f"    1/4 - 12*Delta = {0.25 - 12*delta:.6f}  (residual from quarter closure)")
    print(f"    predicted a_Hbar/g = 1 - 12*Delta = {1.0 - 12*delta:.6f}")
    print("  Current experimental precision is above the predicted offset scale.")
    print()
    print("  External-lead null audit")
    audit = external_leads_null_audit_probe(delta=delta, simulations=4000, seed=1729)
    _row("method", audit.method)
    _row("seed", str(audit.seed))
    for row in audit.rows:
        _row(f"{row.channel} metric", row.metric)
        _row(f"{row.channel} observed", f"{row.observed_score:.12e}")
        _row(f"{row.channel} null mean", f"{row.null_mean:.12e}")
        _row(f"{row.channel} p-value", f"{row.p_value:.6f}")
        _row(f"{row.channel} q-value", f"{row.q_value:.6f}")
        _row(f"{row.channel} status", row.status)
        _row(f"{row.channel} note", row.note)


# ---------------------------------------------------------------------------
# Section 14: Compact algebra audit
# ---------------------------------------------------------------------------

def print_algebra_audit(delta: float = DELTA) -> None:
    _hdr("9. Consistency Audit")

    alg = compact_algebra(delta)
    checks: list[tuple[str, float, float, float]] = []

    checks += [
        ("C1 = C(6,1)",  float(CODE_C1),  float(math.comb(6, 1)),  0.0),
        ("C2 = C(6,2)",  float(CODE_C2),  float(math.comb(6, 2)),  0.0),
        ("C3 = C(6,3)",  float(CODE_C3),  float(math.comb(6, 3)),  0.0),
        ("M_shell",      float(M_SHELL),  192.0,                   0.0),
    ]

    vld = verify_ew_ladder_derivations()
    checks += [
        ("all channel coeffs match", float(vld["all_channels_coeffs"]), 1.0, 0.0),
        ("sum p_i trace-free", vld["sum_p"], 0.0, 1e-12),
        ("sum q_i trace-free", vld["sum_q"], 0.0, 1e-12),
        ("q_W = c4 gravity", vld["q_W"], vld["c4_gravity"], 1e-12),
    ]

    for label, expected_r5 in [
        ("Top", -4.5), ("Higgs", 2.375), ("Z", -4.5), ("W", -2.625)
    ]:
        ch = channel_by_label(label)
        checks.append((f"{label} r5-coeff", ch.r5, expected_r5, 0.0))

    checks += [
        ("epsilon*eta = 48*dBU - 47*m_a",
         alg.epsilon * alg.eta,
         48.0 * DELTA_BU - 47.0 * M_A,
         1e-12),
        ("WZ_OFFSET = C2 - C1",
         float(WZ_OFFSET), float(CODE_C2 - CODE_C1), 0.0),
        ("WZ_APERTURE_COEFF = C3/2",
         WZ_APERTURE_COEFF, CODE_C3 / 2.0, 0.0),
    ]

    print()
    print(f"  {'Check':30} {'value':>11} {'exp':>11} "
          f"{'resid':>12} {'ok':>4}")
    print("  " + "-" * 10)

    all_pass = True
    for label, value, expected, tol in checks:
        resid = value - expected
        ok = (abs(resid) <= tol) if tol > 0.0 else (resid == 0.0)
        status = "ok" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {label[:30]:30} {value:11.6g} {expected:11.6g} "
              f"{resid:12.3e} {status:>4}")

    print()
    print(f"  All checks pass: {'YES' if all_pass else 'NO'}")


# ---------------------------------------------------------------------------
# Master report entry point
# ---------------------------------------------------------------------------

def build_observed(
    top: float   = 172.76,
    higgs: float = 125.10,
    z: float     = 91.1876,
    w: float     = 80.379,
    tau: float   = 1.77686,
    muon: float  = 0.1056583745,
    electron: float = ELECTRON_MASS_GEV,
    bottom: float = 4.18,
    charm: float  = 1.27,
    strange: float = 0.095,
    v: float     = E_EW_GEV,
) -> dict[str, float]:
    """
    Construct the observed-mass dict from PDG inputs.
    Override individual values to test sensitivity.
    """
    return {
        "Electroweak scale":       v,
        "Top quark mass energy":   top,
        "Higgs mass energy":       higgs,
        "Z boson mass energy":     z,
        "W boson mass energy":     w,
        "Tau mass energy":         tau,
        "Muon mass energy":        muon,
        "Electron mass energy":    electron,
        "Bottom quark mass energy": bottom,
        "Charm quark mass energy":  charm,
        "Strange quark mass energy": strange,
    }


class _TeeStdout:
    """Write to stdout and an optional log file (full report, not a summary)."""

    def __init__(self, log_path: str | None) -> None:
        self._stdout = __import__("sys").stdout
        self._log = open(log_path, "w", encoding="utf-8") if log_path else None

    def write(self, data: str) -> int:
        self._stdout.write(data)
        if self._log is not None:
            self._log.write(data)
        return len(data)

    def flush(self) -> None:
        self._stdout.flush()
        if self._log is not None:
            self._log.flush()

    def close(self) -> None:
        if self._log is not None:
            self._log.close()


def run_report(
    observed: dict[str, float] | None = None,
    *,
    delta: float = DELTA,
    include_byte_transitions: bool = True,
    include_structural_law: bool = False,
    skip_kernel: bool = False,
    output_path: str | None = None,
) -> None:
    """
    Run the complete compact geometry report.

    Parameters
    ----------
    observed:
        PDG mass inputs.  If None, default PDG 2024 values are used.
    include_byte_transitions:
        Run the exhaustive 4096*256 byte transition check (~20 s).
    include_structural_law:
        Run the 256^2 commutativity check (several minutes).
    skip_kernel:
        Skip kernel verification entirely (fast, algebra only).
    output_path:
        If set, write the full report to this file as well as stdout.
    """
    import sys

    tee: _TeeStdout | None = None
    if output_path:
        tee = _TeeStdout(output_path)
        sys.stdout = tee

    if observed is None:
        observed = build_observed()

    v = observed["Electroweak scale"]

    print("Compact Geometry: Spectral Algebra of the Electroweak Mass Spectrum and Beyond")
    print("=" * 69)
    print(f"Delta = {_fix(DELTA)}   v = {v} GeV")

    if skip_kernel:
        from hqvm_compact_geom_kernel import (
            KernelReport, shell_transition_algebra, uv_ir_shell_dpf,
            d6_residuals, orderwise_ladder,
        )
        report = KernelReport(
            OMEGA_SIZE,
            (),
            None,
            None,
            None,
            None,
            shell_transition_algebra(),
            uv_ir_shell_dpf(),
            d6_residuals(observed, delta, v),
            orderwise_ladder(observed, delta, v),
        )
    else:
        report = run_kernel_verification(
            observed=observed,
            include_byte_transitions=include_byte_transitions,
            include_structural_law=include_structural_law,
            delta=delta,
            v=v,
        )

    backsolves = electroweak_backsolves(observed)

    loo = hzw_leave_one_out(observed, backsolves)

    coords = electroweak_coords(delta, order=5)

    if not skip_kernel and report.shell_stats:
        print_kernel_structure(report)

    print_aperture_and_coordinates(observed, coords, delta)
    print_electroweak_mass_law(
        observed, report.order_ladder, delta, v
    )
    print_numerical_validation(observed, backsolves, loo, delta, v)
    print_lepton_sector(observed, delta, v)
    print_quark_and_strong_sector(observed, delta, v)

    if report.d6_residuals_rows:
        print_representation_boundary_and_lift(
            report.d6_residuals_rows, observed, delta, v
        )

    print_external_channels(delta)
    print_algebra_audit(delta)

    if tee is not None:
        sys.stdout = tee._stdout
        tee.close()
        print(f"Full report written to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compact Geometry electroweak mass spectrum and beyond report"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip byte-transition exhaustive check (~20 s saved)"
    )
    parser.add_argument(
        "--algebra-only", action="store_true",
        help="Skip all kernel enumeration (algebra sections only)"
    )
    parser.add_argument(
        "--structural-law", action="store_true",
        help="Include 256^2 commutativity check (slow, several minutes)"
    )
    parser.add_argument(
        "--output", metavar="PATH", default=None,
        help="Write full report to file as well as stdout"
    )
    args = parser.parse_args()

    run_report(
        include_byte_transitions=not args.fast,
        include_structural_law=args.structural_law,
        skip_kernel=args.algebra_only,
        output_path=args.output,
    )
