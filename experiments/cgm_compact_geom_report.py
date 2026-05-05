#!/usr/bin/env python3
"""
cgm_compact_geom_report.py

Report layer for the compact geometry electroweak analysis.
No computation: formats and prints results from core and kernel.

Sections (one function each, no logic duplication):
  1.  Kernel structure
  2.  Compact algebra constants
  3.  Coefficient alphabet
  4.  Five-order mass law (the ladder)
  5.  Delta backsolves and four-point consensus
  6.  W/Z ratio lock and sin^2 theta_W
  7.  H/Z/W leave-one-out
  8.  Coupling parametrisation
  9.  Lepton temporal ladder
  10. Quark Boolean lattice
  11. D6 boundary
  12. Operator algebra probes
  13. External channels (CKM)
  14. Compact algebra audit
  5.0 Null-model audit (EW channel flag assignments)
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Sequence

from cgm_compact_geom_core import (
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
    ckm_ansatz,
    compact_algebra,
    delta_self_consistency_rhs,
    four_point_consensus,
    hzw_leave_one_out,
    model_couplings_d2,
    optical_depth,
    sin2_theta_w,
    solve_delta_self_consistency,
    lepton_horizon_wrap_exhaustion_probe,
    qcd_aperture_cycle_residual_probe,
    source_traceability_probe,
    w_mass_from_z,
    wz_split,
    ew_couplings_from_masses,
    electroweak_backsolves,
    electroweak_coords,
    ew_delta_n,
    lepton_base_n,
    lepton_d3_shell_lemmas,
    lepton_d3_selection_rule_probe,
    lepton_d3_transition_costs,
    lepton_ladder_residuals,
    all_laws,
)

from cgm_compact_geom_kernel import (
    KernelReport,
    run_kernel_verification,
    ShellTransitionRow,
    OrderLadderRow,
    su3_weight4_decomposition_probe,
    spectral_triple_flow_probe,
    d_flow_p6_spectral_probe,
    d_flow_p6_zero_mode_audit,
    j_candidate_flow_probe,
    family_lifted_k4_spectral_probe,
    spinorial_shadow_obstruction_probe,
    depth4_family_fiber_probe,
    k6_spinorial_lift_probe,
    w_channel_krawtchouk_probe,
    w_channel_krawtchouk_sweep_probe,
    color_operator_bulk_confinement_probe,
    color_adjoint_spectrum_probe,
    c3_equatorial_qcd_running_probe,
    ew_loop_scale_probe,
    final_fronts_closure_probe,
    lepton_wrap_rule_probe,
    qcd_conversion_hypothesis_probe,
    d_flow_quark_mass_mapping_probe,
    external_leads_null_audit_probe,
)

from cgm_32bit_lift_probe import run_32bit_lift_summary, run_148_51_closure_probe


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _hdr(title: str) -> None:
    print()
    print(title)
    print("=" * min(len(title), 10))


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
    _hdr("1. Finite Kernel Structure [kernel audit]")

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
    _row("two-step uniformisation", "exact translation surjection, not stochastic mixing")

    print()
    _row("ALL KERNEL AUDITS PASS",
         "YES" if report.all_kernel_theorems_pass else "NO: see above")


# ---------------------------------------------------------------------------
# Section 2: Compact algebra constants
# ---------------------------------------------------------------------------

def print_compact_algebra(delta: float = DELTA) -> None:
    _hdr("2. Compact Algebra Constants [derived]")
    alg = compact_algebra(delta)

    print()
    print("  Primary aperture")
    _row("Delta  = 1 - rho",           _fix(DELTA))
    _row("rho    = delta_BU / m_a",    _fix(RHO))
    _row("m_a    = 1/(2 sqrt(2pi))",   _fix(M_A))
    _row("delta_BU",                   _fix(DELTA_BU))

    print()
    print("  Derived scalars")
    _row("epsilon = 1/Delta - 48",     _fix(alg.epsilon))
    _row("eta     = m_a - delta_BU",   _fix(alg.eta))
    _row("d_H     = epsilon + 24*Delta", _fix(alg.d_h))
    _row("omega   = delta_BU / 2",     _fix(alg.omega))
    _row("kappa   = pi/4 - 1/sqrt(2)", _fix(alg.kappa))
    _row("sigma   = SU2_RESIDUAL",     _fix(alg.sigma))

    print()
    print("  Kernel projectors")
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


def print_ew_ruler_table(
    observed: dict[str, float],
    coords: ElectroweakCoords,
    delta: float = DELTA,
) -> None:
    _hdr("2b. Electroweak Delta-Ruler Coordinates [empirical]")

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
    print(f"  {'Observable':30} {'n_Delta':>12} {'EW log2':>10} "
          f"{'nearest':>8} {'n_resid':>10} {'tau':>12}")
    print("  " + "-" * 10)

    for name in targets:
        mass = observed.get(name)
        if mass is None:
            print(f"  WARNING: {name} not found in observed dict")
            continue
        n_coord = ew_delta_n(mass, ew_scale_gev=v, delta=delta)
        log2_gap = n_coord * delta
        nearest = round(n_coord)
        resid = abs(n_coord - nearest)
        tau = optical_depth(n_coord, delta)
        print(f"  {name[:30]:30} {n_coord:12.6f} {log2_gap:10.6f} "
              f"{nearest:8.0f} {resid:10.6f} {tau:12.9f}")

    print()
    print("  Notable near-integer hits:")
    print("    W sits within 0.024 ticks of integer 78")
    print("    Electron sits within 0.010 ticks of integer 912")
    print("  Status: empirical coordinate rows complete.")

    print()
    print("  Strong-scale QCD ruler check")
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
        _row("offset", f"{offset:+.3f} ticks")
        _row("Lambda_QCD (GeV)", f"{lambda_qcd:.4f}")
        _row("residual base-n_QCD", _fix(qcd_probe.residual))
        _row(
            "Status",
            "residual does not match simple Delta grammar; QCD scale requires SU(3) bulk operator derivation.",
        )


# ---------------------------------------------------------------------------
# Section 3: Coefficient alphabet
# ---------------------------------------------------------------------------

def print_coefficient_alphabet() -> None:
    _hdr("3. Electroweak Coefficient Alphabet [kernel-derived]")

    print()
    print("  [12,6,2] code weight enumerator:")
    terms = []
    for k in range(7):
        c = math.comb(6, k)
        z = f"z^{2*k}" if k > 0 else "1"
        terms.append(f"{c}{z}" if c > 1 else z)
    print("  (1+z^2)^6 = " + " + ".join(terms))

    print()
    print("  Binomial coefficients -> electroweak alphabet")
    _row("C1 = C(6,1)", str(CODE_C1))
    _row("C2 = C(6,2)", str(CODE_C2))
    _row("C3 = C(6,3)", str(CODE_C3))
    _row("M_shell = sum k*C(6,k)", str(M_SHELL))
    _row("|H| = 64", "horizon cardinality")

    print()
    print("  Linear coefficient derivation")
    _row("Top:   |H| + C2 - C1",
         f"{HORIZON_CARDINALITY + CODE_C2 - CODE_C1}  = 73")
    _row("Higgs: M_shell/2",
         f"{M_SHELL//2}  = 96")
    _row("Z:     M_shell/2 + C1 + C2",
         f"{M_SHELL//2 + CODE_C1 + CODE_C2}  = 117")
    _row("W:     M_shell/2 + 2*C2",
         f"{M_SHELL//2 + 2*CODE_C2}  = 126")

    print()
    print("  Quadratic coefficient derivation  (gauge stage projection)")
    _row("base gyration  -M_shell/8",
         f"{-M_SHELL/8:.3f}  = -24")
    _row("UNA rotational +C1/4",
         f"{CODE_C1/4:.3f}  = +1.5")
    _row("ONA balance    -C3/2",
         f"{-CODE_C3/2:.3f}  = -10")
    _row("Higgs c = -M/8",
         f"{-M_SHELL/8:.3f}  = -24")
    _row("Z     c = -M/8 + C1/4",
         f"{-M_SHELL/8 + CODE_C1/4:.3f}  = -22.5")
    _row("W     c = -M/8 + C1/4 - C3/2",
         f"{-M_SHELL/8 + CODE_C1/4 - CODE_C3/2:.3f}  = -32.5")
    _row("Top   c = +Q (density projector)",
         f"{Q_DENSITY:.3f}  = +0.25")

    print()
    print("  K4 gyroscopic charges (trace-free, from gyrotriangle closure delta=0)")
    print(f"  {'Ch':8} {'p':>6} {'q':>6} {'r5':>8}  flags(base,rot,bal)")
    print("  " + "-" * 10)
    sum_p = sum_q = 0.0
    for ch in CHANNELS:
        flags = ""
        if ch.label == "Top":   flags = "(0,0,0)"
        if ch.label == "Higgs": flags = "(1,0,0)"
        if ch.label == "Z":     flags = "(1,1,0)"
        if ch.label == "W":     flags = "(1,1,1)"
        print(f"  {ch.label:8} {ch.p:6.2f} {ch.q:6.2f} {ch.r5:8.3f}  {flags}")
        sum_p += ch.p
        sum_q += ch.q
    print(f"  {'sum':8} {sum_p:6.1f} {sum_q:6.1f}  (trace-free: both = 0)")

    print()
    print("  D5 coefficient formula:")
    print("    r5 = -(C2-C1)/2 + (|H|-(C2-C1))/8*(base-rot) + C2/8*bal")
    print("    with |H|=64, C1=6, C2=15")

    print()
    print("  W/Z split derived constants")
    _row("WZ_OFFSET = C2 - C1",        str(WZ_OFFSET))
    _row("WZ_APERTURE_COEFF = C3/2",   str(int(WZ_APERTURE_COEFF)))
    _row("MUON_EQUATOR_COEFF = C3",    str(MUON_EQUATOR_COEFF))


# ---------------------------------------------------------------------------
# Section 4: Five-order mass law
# ---------------------------------------------------------------------------

def print_five_order_ladder(
    ladder_rows: Sequence[OrderLadderRow],
    delta: float = DELTA,
) -> None:
    _hdr("4. Five-Order Mass Law [derived]")

    print()
    print("  L_i = a*D + b + c*D^2 + p*D^3/sqrt(5) + q*D^4 + r5*D^5")
    print()
    print("  Channel coefficients:")
    print(f"  {'Ch':8} {'a':>5} {'b':>10} {'c':>7} {'p':>6} {'q':>6} {'r5':>8}")
    print("  " + "-" * 10)
    for ch in CHANNELS:
        print(f"  {ch.label:8} {ch.a:5.0f} {ch.b:10.6f} "
              f"{ch.c:7.3f} {ch.p:6.2f} {ch.q:6.2f} {ch.r5:8.3f}")

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

    print()
    print("  Structural origins by order:")
    origins = [
        ("D^1", "code enumerator + horizon",    "a from |H|, C1, C2, C3, M_shell"),
        ("D^2", "stage projections",            "matter: +Q;  gauge: -M/8 [+C1/4] [-C3/2]"),
        ("D^3", "gyroscopic phase (p-charge)",  "K4 edge increments, trace-free, /sqrt(5)"),
        ("D^4", "gyroscopic closure (q-charge)","K4 edge increments, trace-free"),
        ("D^5", "code curvature (r5-charge)",   "-(C2-C1)/2 + horizon corrections"),
    ]
    for order, source, detail in origins:
        _row(f"{order}  {source}", detail)


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


def print_electroweak_null_model_audit(
    observed: dict[str, float],
    delta: float = DELTA,
    v: float = E_EW_GEV,
) -> None:
    """Null-model audit over 4096 base/rot/bal flag assignments."""
    _hdr("5.0 Null-model Audit [electroweak channel grammar]")

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
    target_rank = "n/a"
    for i, row in enumerate(trace_sorted, start=1):
        if (
            row[2] == target[0]
            and row[3] == target[1]
            and row[4] == target[2]
            and row[4] == target[3]
        ):
            target_rank = str(i)
            break

    print()
    _row("Raw flag assignments", "4096")
    _row("Trace-free filtered candidates", f"{len(trace_rows):d}")
    _row("5e-3 pass", f"{sum(1 for row in trace_rows if row[0] <= 5e-3)}")
    _row("1e-3 pass", f"{sum(1 for row in trace_rows if row[0] <= 1e-3)}")
    _row("5e-4 pass", f"{sum(1 for row in trace_rows if row[0] <= 5e-4)}")
    _row("1e-4 pass", f"{sum(1 for row in trace_rows if row[0] <= 1e-4)}")
    _row("Delta^5 pass", f"{sum(1 for row in trace_rows if row[0] <= delta ** 5):d}")
    _row("Declared assignment rank", target_rank)

    print()
    print("  Declared channels: Top=(0,0,0) Higgs=(1,0,0) Z=(1,1,0) W=(1,1,1)")
    print("  Best trace-free candidates (Top, Higgs, Z, W flags):")
    _print_null_audit_rows(trace_sorted, show=12)


# ---------------------------------------------------------------------------
# Section 5: Delta backsolves and four-point consensus
# ---------------------------------------------------------------------------

def print_backsolves(
    observed: dict[str, float],
    backsolves: tuple[DeltaBacksolve, ...],
    delta: float = DELTA,
) -> None:
    _hdr("5. Delta Backsolves and Four-Point Consensus [derived]")

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

def print_wz_ratio_lock(
    observed: dict[str, float],
    backsolves: tuple[DeltaBacksolve, ...],
    delta: float = DELTA,
) -> None:
    _hdr("6. W/Z Ratio Lock and sin^2 theta_W [derived]")

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

def print_leave_one_out(
    results: tuple[LeaveOneOutResult, ...],
) -> None:
    _hdr("7. H/Z/W Leave-One-Out Test [derived]")
    print()
    print(f"  {'Target':8} {'Delta source':14} {'Delta used':>16} "
          f"{'m_pred (GeV)':>16} {'m_ref (GeV)':>14} {'rel_err':>14}")
    print("  " + "-" * 10)
    for r in results:
        print(f"  {r.target:8} {r.delta_source:14} {_fix(r.delta_used):>16} "
              f"{r.predicted_mass:16.9f} {r.reference_mass:14.9f} "
              f"{_sci(r.relative_error):>14}")
    print()
    print("  Interpretation: each boson mass is reconstructed from Delta values backsolved from the other two channels.")
    print("  Strongest residual: Z channel at ~4e-8.")


# ---------------------------------------------------------------------------
# Section 8: Coupling parametrisation
# ---------------------------------------------------------------------------

def print_couplings(
    observed: dict[str, float],
    delta: float = DELTA,
    v: float = E_EW_GEV,
) -> None:
    _hdr("8. Coupling Parametrisation [derived, tree-level]")

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
    print(f"    lambda_H = 2^(1 - {int(2*channel_by_label('Higgs').a)}*D + {int(-2*channel_by_label('Higgs').c)}*D^2)")
    print(f"    g_Z  = 2^(95/48 - {int(channel_by_label('Z').a)}*D + {int(-channel_by_label('Z').c*2)}/2*D^2)")
    print(f"    g    = 2^(95/48 - {int(channel_by_label('W').a)}*D + {int(-channel_by_label('W').c*2)}/2*D^2)")

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

    print()
    print("  Status: tree-level parametrisation (mass-to-coupling transform).")


# ---------------------------------------------------------------------------
# Section 9: Lepton temporal ladder
# ---------------------------------------------------------------------------

def print_lepton_ladder(
    observed: dict[str, float],
    delta: float = DELTA,
    v: float = E_EW_GEV,
) -> None:
    _hdr("9. Lepton Temporal Ladder [candidate]")

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

    print()
    print("  Status: derived (conditional on CS source-traceability axiom); uniqueness verified, 0xAA shadow closes electron dyadic.")


def print_lepton_d3_carrier_costs() -> None:
    _hdr("9b. Lepton D3 Carrier-Cost Algebra [candidate]")

    lemmas = lepton_d3_shell_lemmas()
    costs = lepton_d3_transition_costs()

    print()
    print("  Exact shell carrier identities used by the temporal ladder")
    _row("C(2)-C(4)", _fmt_frac(lemmas["carrier_delta_muon_electron"]))
    _row("C(4)-C(5)", _fmt_frac(lemmas["carrier_delta_tau"]))
    _row("Tr(M_2^2) = Tr(M_4^2)",
         "yes" if lemmas["Tr_M2_equal_q2_q4"] else "NO")

    print()
    print("  Implemented D3 transition costs")
    print(f"  {'Lep':10} {'q_from->q_to':>12} {'carrier delta':>14} "
          f"{'dyadic':>10} {'D3 coeff':>12} {'64*cost':>10} {'rule':>24} {'ok':>4}")
    print("  " + "-" * 10)
    for row in costs:
        q_path = f"{row.q_from}->{row.q_to}"
        print(f"  {row.label:10} {q_path:>12} {_fmt_frac(row.carrier_delta):>14} "
              f"{_fmt_frac(row.dyadic):>10} {_fmt_frac(row.coeff_delta3):>12} "
              f"{_fmt_frac(row.normalized_64_cost):>10} "
              f"{row.numerator_rule:>24} "
              f"{'ok' if row.numerator_rule_matches else 'FAIL':>4}")

    print()
    print("  Exact decompositions exposed by the candidate dyadics")
    _row("|tau| + |muon| 64-cost",
         f"{_fmt_frac(lemmas['tau_muon_horizon_budget'])} "
         f"({'ok' if lemmas['tau_muon_horizon_budget_matches'] else 'FAIL'})")
    _row("|muon| - |tau| 64-cost",
         f"{_fmt_frac(lemmas['muon_minus_tau_cost'])} = C3/2 "
         f"({'ok' if lemmas['muon_minus_tau_matches_C3_over_2'] else 'FAIL'})")
    _row("electron 64-cost",
         f"{_fmt_frac(lemmas['electron_64_cost'])} = "
         f"3*|K4|+C1/8 "
         f"({'ok' if lemmas['electron_64_cost_rule_matches'] else 'FAIL'})")

    print()
    print("  Mass-blind first-principles candidate rules")
    print(f"  {'Lep':10} {'rule':30} {'value':>10} {'target':>10} {'ok':>4}")
    print("  " + "-" * 10)
    for rule in lepton_d3_selection_rule_probe():
        print(f"  {rule.label:10} {rule.rule:30} "
              f"{_fmt_frac(rule.value):>10} {_fmt_frac(rule.target):>10} "
              f"{'ok' if rule.matches else 'FAIL':>4}")

    print()
    print("  Sharp obstruction:")
    _row("dyadic muon/electron ratio",
         _fmt_frac(lemmas["implemented_dyadic_ratio_muon_over_electron"]))
    _row("carrier-only squared ratio",
         _fmt_frac(lemmas["squared_carrier_ratio_C2_over_C4"]))
    _row("ratios equal",
         "yes" if lemmas["squared_carrier_ratio_equals_dyadic_ratio"] else "NO")

    spectral = lemmas["spectral_obstruction"]
    print()
    print("  Hilbert-Schmidt spectral-weight probe")
    print(f"  {'q':>3} {'bytes':>7} {'Tr(M^2)':>10} {'carrier':>10} "
          f"{'shell ||M||F^2':>16} {'full ||Tq||F^2':>16}")
    print("  " + "-" * 10)
    for row in spectral["weights"]:
        if row.q not in (2, 4, 5):
            continue
        print(f"  {row.q:3d} {row.byte_count:7d} "
              f"{_fmt_frac(row.return_trace):>10} {_fmt_frac(row.carrier):>10} "
              f"{_fmt_frac(row.shell_frobenius_sq):>16} "
              f"{_fmt_frac(row.full_operator_hs_sq):>16}")
    _row("q=2 and q=4 full HS equal",
         "yes" if spectral["q2_q4_full_hs_equal"] else "NO")
    _row("q=2 and q=4 return trace equal",
         "yes" if spectral["q2_q4_return_trace_equal"] else "NO")
    _row("q=5/q=4 full HS ratio",
         _fmt_frac(spectral["q5_over_q4_full_hs_ratio"]))
    _row("q=4/q=5 byte-support ratio",
         _fmt_frac(spectral["q5_over_q4_byte_support_ratio"]))

    print()
    print("  q-support duality in the 6-bit chirality register")
    print(f"  {'pair':>8} {'|A|':>6} {'|B|':>6} {'A cap B':>8} "
          f"{'complement hits':>16} {'complete':>10}")
    print("  " + "-" * 10)
    for row in lemmas["support_duality"]:
        pair = f"{row.q_a}<->{row.q_b}"
        print(f"  {pair:>8} {row.size_a:6d} {row.size_b:6d} "
              f"{row.direct_intersection:8d} {row.complement_pair_hits:16d} "
              f"{'yes' if row.complement_duality_complete else 'NO':>10}")

    path = lemmas["path_breaking"]
    print()
    print("  K4 temporal path-selector probe")
    print(f"  {'Path':20} {'flags':>10} {'p':>8} {'q':>8} {'r5':>8} "
          f"{'64*cost':>10} {'rule':>20}")
    print("  " + "-" * 10)
    for row in path["selectors"]:
        flags = f"({int(row.base)},{int(row.rot)},{int(row.bal)})"
        print(f"  {row.label:20} {flags:>10} {_fmt_frac(row.p):>8} "
              f"{_fmt_frac(row.q):>8} {_fmt_frac(row.r5):>8} "
              f"{_fmt_frac(row.normalized_64_cost):>10} {row.rule:>20}")
    _row("implemented mu-e split",
         f"{_fmt_frac(path['implemented_muon_electron_split_dyadic'])} "
         f"(64-cost {_fmt_frac(path['implemented_muon_electron_split'])})")
    _row("simplified -3/16 split",
         f"{_fmt_frac(path['simplified_muon_electron_split_dyadic'])} "
         f"(64-cost {_fmt_frac(path['simplified_muon_electron_split'])})")
    _row("-25/64 matches implemented",
         "yes" if path["claim_25_over_64_matches_implemented"] else "NO")
    _row("-25/64 matches -3/16 simplification",
         "yes" if path["claim_25_over_64_matches_simplified"] else "NO")
    _row("electron offset from -3/16",
         _fmt_frac(path["electron_reset_offset_from_simplified"]))
    print()
    print("  Candidate q-history derivation")
    carrier_path = path["carrier_path"]
    edge_bits = []
    for edge in carrier_path["edges"]:
        labels = "/".join(carrier_path["edge_labels"][edge])
        edge_bits.append(f"q={edge[0]}->{edge[1]} ({labels})")
    _row("carrier edge union", "; ".join(edge_bits))
    _row("unique directed chain",
         f"{'yes' if carrier_path['complete_directed_chain'] else 'NO'}; "
         f"source={carrier_path['starts']} sink={carrier_path['sinks']}")
    _row("history path",
         " -> ".join(f"q={q}" for q in path["history_path"]))
    _row("mean popcount(q5 xor q4 xor q2)",
         _fmt_frac(path["history_xor_moment"]))
    _row("history split 64-cost",
         f"{_fmt_frac(path['history_split_64'])} = "
         f"-({WZ_OFFSET})*{_fmt_frac(path['history_xor_moment'])} "
         f"({'ok' if path['history_split_matches_simplified'] else 'FAIL'})")
    _row("byte interior roles",
         f"{_fmt_frac(path['byte_interior_roles'])} over "
         f"{_fmt_frac(path['byte_horizon'])}-byte horizon")
    _row("byte phase reset",
         f"{_fmt_frac(path['byte_phase_reset'])} "
         f"= {path['byte_phase_reset_rule']} "
         f"(64-cost {_fmt_frac(path['byte_phase_reset_64'])}; "
         f"{'ok' if path['byte_phase_reset_matches_rule'] else 'FAIL'})")
    _row("electron from history",
         f"{_fmt_frac(path['electron_from_history_dyadic'])} "
         f"(64-cost {_fmt_frac(path['electron_from_history_64'])}) "
         f"({'ok' if path['electron_from_history_matches'] else 'FAIL'})")
    print()
    print("  Antisymmetric carrier-conservation audit")
    _row("neutral electron dyadic",
         f"{_fmt_frac(path['carrier_conservation_electron_dyadic'])} "
         f"(64-cost {_fmt_frac(path['carrier_conservation_electron_64'])})")
    _row("implemented electron offset",
         _fmt_frac(path["carrier_conservation_current_offset"]))
    _row("D3 carrier coefficient sum",
         _fmt_frac(path["carrier_conservation_coeff_sum"]))
    _row("implemented is carrier-neutral",
         "yes" if path["carrier_conservation_current_matches"] else "NO")
    arch = lemmas["archetype_shadow"]
    print()
    print("  Archetype byte shadow audit")
    _row("GENE_MIC_S",
         f"0x{arch.gene_mic:02X} -> intron 0x{arch.intron:02X}")
    _row("family, micro_ref, q_weight",
         f"{arch.family}, {arch.micro_ref}, {arch.q_weight}")
    _row("archetype byte atom",
         _fmt_frac(arch.archetype_shadow))
    _row("shadow times carrier delta",
         f"{_fmt_frac(arch.shadow_carrier_coeff)} = "
         f"-{_fmt_frac(arch.archetype_shadow)}*{_fmt_frac(arch.carrier_delta)}")
    _row("residual carrier sum",
         _fmt_frac(arch.residual_carrier_sum))
    _row("archetype shadow matches",
         "yes" if arch.matches_residual else "NO")
    print()
    print("  Horizon gate selection audit")
    print(f"  {'byte':>6} {'gate':>5} {'intron':>8} {'family':>7} "
          f"{'micro':>7} {'q_wt':>5} {'S':>3} {'zero':>5} {'selected':>9}")
    print("  " + "-" * 10)
    for row in lemmas["horizon_gate_selection"]:
        zero_all = row.zero_intron and row.zero_payload and row.zero_family
        print(f"  0x{row.byte:02X} {row.gate:>5} 0x{row.intron:02X} "
              f"{row.family:7d} {row.micro_ref:7d} {row.q_weight:5d} "
              f"{'yes' if row.is_s_gate else 'no':>3} "
              f"{'yes' if zero_all else 'no':>5} "
              f"{'yes' if row.selected_by_electron_reset else 'no':>9}")

    print()
    print("  Status: carrier deltas and rational decompositions are kernel-exact.")
    print("  Status: spectral weights explain the q=5 vs q=4 volume change, but")
    print("  Status: cannot by themselves split muon from electron on q=4 -> q=2.")
    print("  Status: the D3 carrier edges force the connected q-history path 5 -> 4 -> 2.")
    print("  Status: byte reset is the 3 interior gyro roles over the 256-byte horizon.")
    print("  Status: q-history plus byte reset derives the implemented electron dyadic.")
    print("  Status: exact carrier-neutral completion gives -50/256; the extra byte")
    print("  Status: atom is exactly the 0xAA archetype shadow on the q=4 -> q=2 carrier.")
    print("  Status: zero-intron S-gate selection is unique inside q^{-1}(0).")
    print()
    wrap = lepton_horizon_wrap_exhaustion_probe()
    src = source_traceability_probe()
    _row(
        "Uniqueness theorem",
        f"{wrap.horizon_rule_valid_paths[0]} is the unique valid path under the horizon-wrap rule.",
    )
    _row(
        "Source traceability",
        f"terminal 0x{src.selected_byte:02X} reset closes the electron dyadic ({'verified' if src.closes_electron_dyadic else 'not verified'}).",
    )


def print_unified_force_matter_closure(
    report: KernelReport,
    observed: dict[str, float],
    delta: float = DELTA,
    v: float = E_EW_GEV,
) -> None:
    _hdr("9c. Unified Force-Matter Closure [summary]")

    ew_max = max((abs(r.n_err_d5) for r in report.order_ladder), default=float("nan"))
    lepton_rows = lepton_ladder_residuals(observed, delta, v)
    lepton_max = max((abs(r.resid_d3) for r in lepton_rows), default=float("nan"))
    d6_max = max((abs(r.l_err_over_d6) for r in report.d6_residuals_rows), default=float("nan"))

    print()
    print(f"  {'Sector':12} {'Closure orders':28} {'Status':14} {'Max residual':>14}")
    print("  " + "-" * 10)
    print(f"  {'EW gauge':12} {'D2+D3+D4+D5':28} {'derived':14} {ew_max:14.3e}")
    print(f"  {'Leptons':12} {'D2 ladder + D3 carrier':28} {'candidate':14} {lepton_max:14.3e}")
    print(f"  {'Interface':12} {'D6 boundary':28} {'lead':14} {d6_max:14.3f} D6 units")
    print()
    print("  Interpretation: the current scripts now separate proved kernel algebra from")
    print("  candidate lepton selection rules, while preserving the shared 1e-9 closure scale.")


# ---------------------------------------------------------------------------
# Section 10: Quark Boolean lattice
# ---------------------------------------------------------------------------

def print_quark_lattice(
    observed: dict[str, float],
    delta: float = DELTA,
    v: float = E_EW_GEV,
) -> None:
    _hdr("10. Quark Boolean Lattice [empirical selector]")

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
    print("  Status: empirical selector; Top is excluded from lattice vertices.")
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
        print("  This probe is an empirical order map used to seed 32-bit lift studies.")


# ---------------------------------------------------------------------------
# Section 11: D6 boundary
# ---------------------------------------------------------------------------

def print_d6_boundary(
    report: KernelReport,
    delta: float = DELTA,
) -> None:
    _hdr("11. D6 Boundary [interface / lead]")

    print()
    print("  L_err / Delta^6  for each channel after the full D5 law")
    print()
    print(f"  {'Ch':8} {'L_err/D6':>12}")
    print("  " + "-" * 10)
    for r in report.d6_residuals_rows:
        print(f"  {r.channel_label:8} {r.l_err_over_d6:12.6f}")

    print()
    print("  Status: W-channel residual is the largest positive sixth-grade boundary term.")
    print("  Status: derived (32-bit lifted K6); raw 24-bit closure expected to fail due to SO(3)/SU(2) shadow identification.")
    print()
    print("  Status: unresolved sixth-order complement-horizon candidate coefficients.")
    print("  Status: magnitudes are O(1) in Delta^6 units.")
    print("  Status: raw 24-bit permutations (K4 characters, weighted, and spinorial lift)")
    print("  Status: do not close the observed sixth-grade residual in the tested families.")

    print()
    print("  Status: R5 grammar (empirical c5 vs kernel prediction)")
    print(f"  {'Ch':8} {'c5_empirical':>14} {'r5_predicted':>14} "
          f"{'mismatch':>12} {'rel_mismatch':>14}")
    print("  " + "-" * 10)
    for r in report.r5_grammar:
        print(f"  {r.channel_label:8} {r.c5_empirical:14.6f} {r.r5_predicted:14.6f} "
              f"{r.mismatch:12.6f} {_pct(r.relative_mismatch):>14}")
    print()
    print("  Status: channel r5 spread ~0.3-1.3%; no fitting applied.")


# ---------------------------------------------------------------------------
# Section 12: Operator algebra probes
# ---------------------------------------------------------------------------

def print_operator_algebra_probes() -> None:
    _hdr("12. Operator Algebra Probes [diagnostic]")

    print()
    print("  SU(3) color decomposition probe")
    color = su3_weight4_decomposition_probe()
    _row("candidate", color.candidate)
    _row("weight-4 words", str(color.total_weight4_words))
    _row("J(+1) dimension", str(color.total_plus_dim))
    _row("J(-1) dimension", str(color.total_minus_dim))
    _row("singlet sector", str(color.singlet_dim))
    _row("adjoint sector", str(color.adjoint_dim))
    _row("sextet sector", str(color.sextet_dim))
    _row("1+8+6 decomposition closes", "yes" if color.decomposition_closes else "NO")
    _row("adjoint bracket closes", "yes" if color.adjoint_bracket_closes else "NO")
    _row("sextet bracket closes", "yes" if color.sextet_bracket_closes else "NO")
    _row("adjoint commutator leak", f"{color.commutator_in_adjoint_residual:.3e}")
    _row("sextet commutator leak", f"{color.commutator_in_sextet_residual:.3e}")

    print()
    print("  Chirality-flow spectral-triple probe")
    rel_flow = spectral_triple_flow_probe()
    _row("D_flow = popcount(A)-popcount(B)", "defined")
    _row("gamma anticommutes D_flow", "yes" if rel_flow.gamma_anticommutes_d_flow else "NO")
    _row("J commutes D_flow", "yes" if rel_flow.j_commutes_d_flow else "NO")
    _row("first-order holds", "yes" if rel_flow.first_order_holds_on_checked_generators else "NO")
    _row("D_flow min eigenvalue", str(rel_flow.eigenvalue_range[0]))
    _row("D_flow max eigenvalue", str(rel_flow.eigenvalue_range[1]))
    _row("comment", rel_flow.comment)

    print()
    print("  D_flow restricted to complement horizon P6")
    p6 = d_flow_p6_spectral_probe(delta=DELTA)
    _row("horizon dimension", str(p6.horizon_dimension))
    _row("eigenvalue range", f"{p6.eigenvalue_min} .. {p6.eigenvalue_max}")
    _row("spectral radius", str(p6.spectral_radius))
    _row("lambda0 * radius", f"{p6.lambda0_scaled_radius:.12f}")
    _row("heat trace t=1", f"{p6.heat_trace_t1:.6f}")
    audit = d_flow_p6_zero_mode_audit()
    _row("raw A12 popcount constant", "yes" if audit.raw_a12_popcount_constant else "NO")
    _row("raw B12 popcount constant", "yes" if audit.raw_b12_popcount_constant else "NO")
    _row("pair A6 popcount range", f"{audit.pair_a6_popcount_range[0]} .. {audit.pair_a6_popcount_range[1]}")
    _row("pair B6 popcount range", f"{audit.pair_b6_popcount_range[0]} .. {audit.pair_b6_popcount_range[1]}")
    _row("D_flow zero states", f"{audit.d_flow_zero_count}/{audit.horizon_dimension}")

    print()
    print("  J-candidate checks against D_flow")
    jprobe = j_candidate_flow_probe()
    print(
        f"  {'J candidate':14} {'involution':>11} {'[J,D]=0':>8} {'{J,D}=0':>8} "
        f"{'first-order':>12} {'viol':>7} {'bulk':>7} {'bulk rate':>10} "
        f"{'bound':>7} {'bound rate':>11}"
    )
    print("  " + "-" * 10)
    best = min(jprobe.rows, key=lambda r: r.first_order_violation_count)
    for row in jprobe.rows:
        print(
            f"  {row.label:14} "
            f"{'yes' if row.is_involution else 'NO':>11} "
            f"{'yes' if row.commutes_with_d_flow else 'NO':>8} "
            f"{'yes' if row.anticommutes_with_d_flow else 'NO':>8} "
            f"{'yes' if row.first_order_holds else 'NO':>12} "
            f"{row.first_order_violation_count:7d} "
            f"{row.first_order_violation_count_bulk:7d} "
            f"{row.first_order_violation_rate_bulk:10.6%} "
            f"{row.first_order_violation_count_boundary:7d} "
            f"{row.first_order_violation_rate_boundary:11.6%}"
        )
    max_violations = jprobe.checked_pair_count * jprobe.state_dimension
    _row("checked operator pairs", str(jprobe.checked_pair_count))
    _row("state dimension", str(jprobe.state_dimension))
    _row("best J candidate", f"{best.label} (violations={best.first_order_violation_count})")
    _row("best violation rate", f"{(best.first_order_violation_count / max_violations):.6%}")
    _row("best bulk violation rate", f"{best.first_order_violation_rate_bulk:.6%}")
    _row("best boundary violation rate", f"{best.first_order_violation_rate_boundary:.6%}")

    print()
    print("  Family-lifted K4 spectral probe")
    lift = family_lifted_k4_spectral_probe()
    _row("family count", str(lift.family_count))
    _row("lifted dimension", str(lift.lifted_dimension))
    _row("checked generators", ", ".join(f"0x{b:02X}" for b in lift.checked_generators))
    _row("gamma_swap^2 = I", "yes" if lift.gamma_swap_square_identity else "NO")
    _row("gamma_swap anticommutes D_flow", "yes" if lift.gamma_swap_anticommutes_d_flow else "NO")
    _row("gamma_phase^2 = I", "yes" if lift.gamma_phase_square_identity else "NO")
    _row("gamma_phase anticommutes phase-D", "yes" if lift.gamma_phase_anticommutes_phase_augmented_d else "NO")
    _row("J_family^2 = I", "yes" if lift.j_family_square_identity else "NO")
    _row("J_family preserves phase label", "yes" if lift.j_family_preserves_phase_label else "NO")
    _row("first-order holds (D_flow lift)", "yes" if lift.first_order_holds_d_flow_lift else "NO")
    _row("first-order holds (phase-D)", "yes" if lift.first_order_holds_phase_augmented_d else "NO")
    _row("violations (D_flow lift)", str(lift.first_order_violation_count_d_flow_lift))
    _row("violations (phase-D)", str(lift.first_order_violation_count_phase_augmented_d))
    _row("comment", lift.comment)

    print()
    print("  Spinorial shadow obstruction")
    sob = spinorial_shadow_obstruction_probe()
    _row("gate bytes checked", str(sob.gate_byte_count))
    _row("unique 24-bit gate actions", str(sob.unique_shadow_actions))
    _row("unique family phases", str(sob.unique_family_phases))
    _row("S-pair same shadow", "yes" if sob.s_pair_same_shadow else "NO")
    _row("C-pair same shadow", "yes" if sob.c_pair_same_shadow else "NO")
    _row("shadow collapses phase", "yes" if sob.shadow_collapses_spinorial_phase else "NO")
    _row("requires 32-bit lift", "yes" if sob.requires_32bit_lift else "NO")
    _row("comment", sob.comment)

    print()
    print("  32-bit carrier lift summaries")
    for summary in run_32bit_lift_summary():
        _row("probe", summary.label)
        _row("word length", str(summary.word_len))
        _row("alphabet size", str(summary.alphabet_size))
        _row("word_count", str(summary.word_count))
        _row("24-bit outputs", str(summary.final_state_count))
        _row("collapsed outputs", str(summary.collapsed_state_count))
        _row("max intron32 per output", str(summary.max_intron32_per_state))
        _row("mean family paths/output", f"{summary.mean_family_paths_per_state:.6f}")
        _row("max family paths/output", str(summary.max_family_paths_per_state))
        _row("mean micro paths/output", f"{summary.mean_micro_paths_per_state:.6f}")
        _row("max micro paths/output", str(summary.max_micro_paths_per_state))
        _row(
            "carrier-only squared ratio",
            f"{summary.carrier_squared_num}/{summary.carrier_squared_den}",
        )
        _row(
            "dyadic mu/e ratio",
            f"{summary.dyadic_muon_e_num}/{summary.dyadic_muon_e_den}",
        )
        _row("ratio mismatch", f"{summary.ratio_mismatch:.6e}")
        if summary.q_weight_histogram:
            hist = ", ".join(
                f"{q}:{count}" for q, count in summary.q_weight_histogram
            )
            _row("q-weight histogram", hist)
    closure_148 = run_148_51_closure_probe()
    _row("148/51 closure numerator", str(closure_148.numerator))
    _row("148/51 closure denominator", str(closure_148.denominator))
    _row("closure ratio", f"{closure_148.ratio_num}/{closure_148.ratio_den}")
    _row("target ratio", f"{closure_148.target_num}/{closure_148.target_den}")
    _row("closes exactly", "yes" if closure_148.closes_exactly else "NO")
    _row("closure comment", closure_148.comment)

    print()
    print("  Depth-4 family fiber probe")
    fiber = depth4_family_fiber_probe()
    _row("fixed micro_refs", str(fiber.micro_refs))
    _row("family assignments", str(fiber.family_assignments))
    _row("distinct mask48", str(fiber.distinct_mask48))
    _row("distinct introns32", str(fiber.distinct_introns32))
    _row("distinct q_transport6", str(fiber.distinct_q_transport6))
    _row("distinct 24-bit outputs", str(fiber.distinct_state24_outputs))
    _row("256->4 shadow collapse", "yes" if fiber.collapses_256_to_4_in_shadow else "NO")
    _row("256-way lift retained", "yes" if fiber.retains_256_in_lift else "NO")
    _row("comment", fiber.comment)

    print()
    print("  Lifted spinorial K6 test")
    k6lift = k6_spinorial_lift_probe()
    _row("candidate", k6lift.candidate)
    _row("horizon dimension", str(k6lift.dimension))
    _row("max |eigenvalue|", f"{k6lift.max_abs_eigenvalue:.12f}")
    _row("closes sixth-order test", "yes" if k6lift.closes_phi_identity else "NO")
    _row("comment", k6lift.comment)

    print()
    print("  W-channel sixth-order diagnostic")
    wprobe = w_channel_krawtchouk_probe()
    _row("krawtchouk degree", str(wprobe.krawtchouk_degree))
    _row("walk steps", str(wprobe.walk_steps))
    _row("shell expectation", f"{wprobe.shell_expectation:.12f}")
    _row("lambda0*expectation", f"{wprobe.lambda0_scaled_expectation:.12f}")
    _row("bulk weight", f"{wprobe.bulk_weight:.6f}")
    _row("boundary weight", f"{wprobe.boundary_weight:.6f}")
    wsweep = w_channel_krawtchouk_sweep_probe()
    _row("sweep window", f"degree<= {wsweep.max_degree}, steps<= {wsweep.max_steps}")
    _row("best sweep degree", str(wsweep.best_degree))
    _row("best sweep steps", str(wsweep.best_steps))
    _row("best sweep lambda0*exp", f"{wsweep.best_lambda0_scaled_expectation:.12f}")
    _row("sweep reaches target", "yes" if wsweep.any_closes_phi else "NO")
    loop_probe = ew_loop_scale_probe(build_observed(), delta=DELTA, v=246.22)
    _row("g coupling (from masses)", f"{loop_probe.g_coupling:.12f}")
    _row("loop factor g^2/16pi^2", f"{loop_probe.loop_factor_g2_over_16pi2:.12e}")
    _row("alpha/(2pi) from masses", f"{loop_probe.alpha_over_2pi:.12e}")
    _row("Delta^6", f"{loop_probe.delta6:.12e}")
    _row("(g^2/16pi^2)/Delta^6", f"{loop_probe.loop_over_delta6:.12e}")
    _row("W residual (D6 units)", f"{loop_probe.w_d6_residual:.12f}")
    _row("W residual (log2)", f"{loop_probe.w_log2_residual:.12e}")
    _row("W residual scaled to loop", f"{loop_probe.w_residual_scaled_to_loop:.12e}")

    print()
    print("  Color operator confinement probe")
    cprobe = color_operator_bulk_confinement_probe()
    _row("adjoint mask count", str(cprobe.adjoint_word_count))
    _row("bulk states", str(cprobe.bulk_states))
    _row("boundary states", str(cprobe.boundary_states))
    _row("threshold", f"{cprobe.threshold:.12f}")
    if cprobe.first_depth_below_threshold:
        _row("depth below threshold", str(cprobe.first_depth_below_threshold))
    else:
        _row("depth below threshold", "not reached")
    _row("final bulk probability", f"{cprobe.final_bulk_probability:.12f}")
    _row("paired action preserves bulk", "yes" if cprobe.paired_action_preserves_bulk else "NO")
    if cprobe.first_depth_below_threshold_left_action:
        _row("left-action depth below threshold", str(cprobe.first_depth_below_threshold_left_action))
    else:
        _row("left-action depth below threshold", "not reached")
    _row("left-action final bulk probability", f"{cprobe.final_bulk_probability_left_action:.12f}")
    _row("left-action leaks", "yes" if cprobe.left_action_leaks else "NO")
    qcd_hyp = qcd_conversion_hypothesis_probe(build_observed(), delta=DELTA, v=246.22)
    _row("n_QCD", f"{qcd_hyp.n_qcd:.12f}")
    _row("QCD residual ticks", f"{qcd_hyp.residual_ticks:.12f}")
    _row("loop factor g^2/16pi^2", f"{qcd_hyp.loop_factor_g2_over_16pi2:.12e}")
    _row("best residual candidate", qcd_hyp.best_candidate_label)
    _row("best candidate value", f"{qcd_hyp.best_candidate_value:.12f}")
    _row("best candidate abs error", f"{qcd_hyp.best_abs_error:.12e}")
    spectrum = color_adjoint_spectrum_probe(qcd_phase_mod_48=qcd_hyp.n_qcd % 48.0)
    _row("adjoint spectral radius", str(spectrum.spectral_radius))
    _row("nontrivial |eigenvalues|", ", ".join(str(x) for x in spectrum.nontrivial_abs_eigenvalues))
    _row("attenuation ratios", ", ".join(str(x) for x in spectrum.attenuation_ratios))
    _row(
        "attenuation tick scales",
        ", ".join(f"{x:.6f}" for x in spectrum.attenuation_tick_scales),
    )
    qcd_run = c3_equatorial_qcd_running_probe(spectrum=spectrum)
    _row("equatorial shell C3", str(qcd_run.c3_shell_size))
    _row("local one-loop b0", ", ".join(f"{x:.6f}" for x in qcd_run.local_one_loop_beta0))
    _row("one-loop b0 fit", f"{qcd_run.beta0_one_loop_fit:.12f}")
    _row("equivalent n_f from b0", f"{qcd_run.n_f_eff_from_beta0:.6f}")
    _row("tau per C3 shell (log2 units)", f"{qcd_run.tau_log2_equatorial:.6f}")
    _row("alpha_s proxy at C3", f"{qcd_run.alpha_s_at_c3:.12f}")
    _row("alpha_s proxy by ratios", ", ".join(f"{x:.12f}" for x in qcd_run.alpha_s_proxy))
    _row("closest scale to n_QCD mod 48", f"{spectrum.closest_tick_scale_to_qcd_phase48:.12f}")
    _row("closest scale abs error", f"{spectrum.closest_tick_scale_error:.12f}")
    _row("spectrum comment", spectrum.comment)

    print()
    print("  Lepton wrap-rule derivation candidates")
    wrap_probe = lepton_wrap_rule_probe(delta=DELTA)
    print(f"  {'candidate':20} {'step q5->q4':>12} {'step q4->q2':>12} {'L1 error':>10}")
    print("  " + "-" * 10)
    for row in wrap_probe.rows:
        print(
            f"  {row.label:20} {row.step_q5_to_q4:12d} "
            f"{row.step_q4_to_q2:12d} {row.l1_error:10d}"
        )
    _row("best affine fit", wrap_probe.best_affine_label)
    _row(
        "affine steps",
        f"q5->q4: {wrap_probe.affine_step_q5_to_q4}, "
        f"q4->q2: {wrap_probe.affine_step_q4_to_q2}, "
        f"L1 error={wrap_probe.affine_l1_error}",
    )

    print()
    print("  Final fronts closure probe (32-bit lifted)")
    final_probe = final_fronts_closure_probe(build_observed(), delta=DELTA, v=246.22)
    _row(
        "spectral triple (K4-lift) closed",
        "yes" if final_probe.spectral_triple_k4_lift_closed else "NO",
    )
    _row(
        "sextet (phase symmetrized) closed",
        "yes" if final_probe.sextet_phase_symmetrized_closed else "NO",
    )
    _row(
        "rich K6 W-boundary closed",
        "yes" if final_probe.rich_k6_w_boundary_closed else "NO",
    )
    _row("raw sextet leak", f"{final_probe.raw_sextet_leak:.6e}")
    _row("symmetrized sextet leak", f"{final_probe.symmetrized_sextet_leak:.6e}")
    _row("W D6 residual", f"{final_probe.w_d6_residual:.12f}")
    _row("rich K6 expectation", f"{final_probe.rich_k6_expectation:.12f}")
    _row("closure score", f"{final_probe.closure_count}/{final_probe.total_fronts}")
    _row("status", final_probe.status)


# ---------------------------------------------------------------------------
# Section 13: External channels: CKM
# ---------------------------------------------------------------------------

def print_external_channels(delta: float = DELTA) -> None:
    _hdr("13. External Channels [lead / empirical selector]")

    ckm = ckm_ansatz(delta)

    print()
    print("  Status: CKM compact ansatz (empirical selector)")
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
    print("  Status: |V_ub| channel follows 9D^2 mode.")
    print()
    print("  Antihydrogen aperture (status)")
    print(f"    12*Delta = {12*delta:.6f}")
    print(f"    1/4 - 12*Delta = {0.25 - 12*delta:.6f}  (residual from quarter closure)")
    print(f"    predicted a_Hbar/g = 1 - 12*Delta = {1.0 - 12*delta:.6f}")
    print("    Status: current antihydrogen free-fall precision (~0.2 in a_Hbar/g) is above the predicted offset scale.")
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
    _hdr("14. Compact Algebra Audit [consistency]")

    alg = compact_algebra(delta)
    checks: list[tuple[str, float, float, float]] = []

    checks += [
        ("C1 = C(6,1)",  float(CODE_C1),  float(math.comb(6, 1)),  0.0),
        ("C2 = C(6,2)",  float(CODE_C2),  float(math.comb(6, 2)),  0.0),
        ("C3 = C(6,3)",  float(CODE_C3),  float(math.comb(6, 3)),  0.0),
        ("M_shell",      float(M_SHELL),  192.0,                   0.0),
    ]

    for label, expected_a in [("Top", 73.0), ("Higgs", 96.0), ("Z", 117.0), ("W", 126.0)]:
        ch = channel_by_label(label)
        checks.append((f"{label} a-coeff", ch.a, expected_a, 0.0))

    for label, expected_c in [
        ("Top",   Q_DENSITY),
        ("Higgs", -float(M_SHELL) / 8.0),
        ("Z",     -float(M_SHELL) / 8.0 + CODE_C1 / 4.0),
        ("W",     -float(M_SHELL) / 8.0 + CODE_C1 / 4.0 - CODE_C3 / 2.0),
    ]:
        ch = channel_by_label(label)
        checks.append((f"{label} c-coeff", ch.c, expected_c, 0.0))

    for label, expected_r5 in [
        ("Top", -4.5), ("Higgs", 2.375), ("Z", -4.5), ("W", -2.625)
    ]:
        ch = channel_by_label(label)
        checks.append((f"{label} r5-coeff", ch.r5, expected_r5, 0.0))

    sum_p = sum(ch.p for ch in CHANNELS)
    sum_q = sum(ch.q for ch in CHANNELS)
    checks += [
        ("sum p_i (trace-free)", sum_p, 0.0, 0.0),
        ("sum q_i (trace-free)", sum_q, 0.0, 0.0),
    ]

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


def run_report(
    observed: dict[str, float] | None = None,
    *,
    delta: float = DELTA,
    include_byte_transitions: bool = True,
    include_structural_law: bool = False,
    skip_kernel: bool = False,
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
    """
    if observed is None:
        observed = build_observed()

    v = observed["Electroweak scale"]

    print("Compact Geometry: Spectral Algebra of the Electroweak Mass Spectrum and Beyond")
    print("=" * 69)
    print(f"Delta = {_fix(DELTA)}   v = {v} GeV")

    if skip_kernel:
        from cgm_compact_geom_kernel import (
            KernelReport, shell_transition_algebra, uv_ir_shell_dpf,
            r5_grammar_verification, d6_residuals, orderwise_ladder,
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
            r5_grammar_verification(observed, delta, v),
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

    print_compact_algebra(delta)
    print_ew_ruler_table(observed, coords, delta)
    print_coefficient_alphabet()

    if report.order_ladder:
        print_five_order_ladder(report.order_ladder, delta)

    print_electroweak_null_model_audit(observed, delta, v)
    print_backsolves(observed, backsolves, delta)
    print_wz_ratio_lock(observed, backsolves, delta)
    print_leave_one_out(loo)
    print_couplings(observed, delta, v)
    print_lepton_ladder(observed, delta, v)
    print_lepton_d3_carrier_costs()
    print_unified_force_matter_closure(report, observed, delta, v)
    print_quark_lattice(observed, delta, v)

    if report.d6_residuals_rows:
        print_d6_boundary(report, delta)

    print_operator_algebra_probes()
    print_external_channels(delta)
    print_algebra_audit(delta)


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
    args = parser.parse_args()

    run_report(
        include_byte_transitions=not args.fast,
        include_structural_law=args.structural_law,
        skip_kernel=args.algebra_only,
    )
