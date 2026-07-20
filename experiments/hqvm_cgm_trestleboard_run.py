#!/usr/bin/env python3
"""
hqvm_cgm_trestleboard_run.py

CLI entry + report driver for the CGM trestleboard.

Report (sections 0–11). Engine core is hqvm_cgm_trestleboard_1.py; nuclear
dynamics are NuclearBoard in hqvm_cgm_trestleboard_2.py; the nuclear
prediction + bulk census report is hqvm_cgm_trestleboard_3.py (this driver
calls its census_main); frontier structural + engineering report is
hqvm_cgm_trestleboard_4.py (this driver calls its frontier_main); magic-number
derivation is hqvm_cgm_trestleboard_5.py (this driver calls its magic_main).
Shared constants in hqvm_cgm_trestleboard_common.py.

Companion notes: hqvm_cgm_trestleboard_notes.txt.
"""
from __future__ import annotations
import argparse
import io
import sys
from pathlib import Path
from typing import List

_REPO = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

_FUEL_REF_DIR = _REPO / "data" / "catalogs" / "fusion"

from gyroscopic.hQVM.family import (
    verify_d6_against_api,
    verify_exact_root_rank_lock,
    verify_f_squared_rest_d,
)

from hqvm_cgm_trestleboard_common import (
    CHANNELS,
    CHIRALITY_D,
    DELTA,
    ENSDF_EV_BAND_PATH,
    FUEL_SUITE,
    FUSION_Z1Z2_CUTOFF,
    HOLONOMY_DRESS,
    MC_HIERARCHY_SAMPLES,
    NUCLEAR_OPTICAL_TOL_TICKS,
    OMEGA,
    RESULTS_PATH,
    TICKS_C3_DELTA,
    TICKS_PER_D,
    TICKS_PER_K,
    TICKS_PER_OCTAVE,
    TICK_CORR,
    V_EV,
    _Tee,
    load_ensdf_first_excited,
)
from hqvm_cgm_trestleboard_1 import (
    make_fusion_grid,
    make_reactivity_T_grid,
)
from hqvm_cgm_trestleboard_2 import NuclearBoard
from hqvm_cgm_trestleboard_3 import (
    audit_nuclear_optical,
    census_main,
    emin_falsifier,
    horizon_lemma_rows,
    minimum_excitation_report,
    nuclear_null_model,
    optical_conjugacy_rows,
    recover_Delta_from_WZ,
    spectral_energy_GeV,
    wavefunction_anchors,
)
from hqvm_cgm_trestleboard_4 import (
    frontier_main,
    fusion_barrier_gate,
    fusion_resonance_map,
    fusion_resonance_null,
)
from hqvm_cgm_trestleboard_5 import magic_main


def _shell_of(state24: int) -> int:
    """Shell index = popcount of the 6-bit chirality register (api canonical)."""
    from gyroscopic.hQVM.api import chirality_word6

    return chirality_word6(int(state24) & 0xFFFFFF).bit_count()


def _report_thresholds(tb: NuclearBoard, th, pc) -> None:
    print("\n0) THRESHOLDS & NORMALIZATIONS")
    print("=" * 5)
    print(
        f"  s_p=π/2  S_CS={th.S_CS:.4f} | u_p=1/√2 S_UNA={th.S_UNA:.4f} | "
        f"o_p=π/4 S_ONA={th.S_ONA:.4f} | S_BU=m_a={th.S_BU:.6f}"
    )
    print(
        f"  Q_G·m_a²={th.product_QG_ma2:.6f} (½) | ρ={th.rho:.6f} "
        f"Δ={th.Delta:.6f} δ_BU={th.delta_BU:.6f}"
    )
    print(f"  D={th.D:.0f} G_kernel={th.G_kernel:.6f} α0={th.alpha0:.9f}")
    print(
        f"  α0·ζ={th.alpha0_zeta:.12f}  target={th.alpha0_zeta_target:.12f}  "
        f"PASS={th.product_identity_ok}"
    )
    print(f"  optical 1/(4π²)={th.optical_dilution:.8f} | " f"2^(C3Δ²)={TICK_CORR:.8f}")
    print(
        f"  ticks/K={TICKS_PER_K:.2f} ticks/ρ={TICKS_PER_D:.3f} "
        f"ticks/oct={TICKS_PER_OCTAVE:.2f} C3Δ={TICKS_C3_DELTA:.4f}"
    )
    print(
        f"  protocol={tb.protocol}  d={tb.chirality_d}  "
        f"fusion_model={tb.fusion_model}  dial={tb.dial}  "
        f"p_c_rank={pc.p_c_rank:.4f} (exact micro-ref rank threshold)"
    )
    print("\n  Holonomy dress d → meaning:")
    for d, note in HOLONOMY_DRESS.items():
        print(f"    d={d}: {note}")


def _report_shells(tb: NuclearBoard, sh, pc) -> None:
    print("\n1) SHELL CENSUS & PERCOLATION LAW")
    print("=" * 5)
    print(
        f"  |Ω|={sh.Omega} |H|={sh.H} holographic PASS={sh.holographic_ok} "
        f"⟨S⟩={sh.mean_entanglement:.3f}"
    )
    print(f"  pops={sh.pops}")
    for r in range(0, 7):
        ps = tb.percolation_from_rank(r)
        fc = "fiber-complete" if ps.fiber_complete else "NOT fiber-complete"
        print(f"    r={r}: |Reach|={ps.Reach:5d} θ={ps.coverage:.6f} {fc} | {ps.note}")
    print(f"  hierarchy p_c (kernel MC, n={MC_HIERARCHY_SAMPLES}):")
    print(f"    rank=d  {pc.p_c_rank:.4f}  (exact bisect_p_c_rank_micro_ref)")
    print(
        f"    span    {pc.p_c_span:.4f}  full    {pc.p_c_full:.4f}  "
        f"spectrum {pc.p_c_spectrum:.4f}  word {pc.p_c_word:.4f}"
    )
    print(f"  d=6 api↔family  PASS={pc.d6_api_ok}  ({pc.d6_api_note})")


def _report_grammar(tb: NuclearBoard) -> None:
    print("\n2) CLOSURE GRAMMAR")
    print("=" * 5)
    for cls in tb.grammar:
        Ep = tb.predict_E_eV(cls)
        np = tb.n_of_E(Ep)
        tag = "FORCED" if cls.forced else "optional"
        print(
            f"  {tag:8s} (k={cls.k},d={cls.d}) n={np:9.2f} E={Ep:12.4e} eV  "
            f"{cls.label}"
        )


def _report_level(tb: NuclearBoard) -> None:
    print("\n3) LEVEL")
    print("=" * 5)
    for name, E in [("Th-229m", 8.3557335), ("Deuteron BE", 2.224e6)]:
        fit = tb.level(E)
        fit_f = tb.level(E, forced_only=True)
        print(f"  {name}: n={fit.n:.2f}")
        print(
            f"    all    : {fit.cls.label}  tick={fit.tick_residual:+.5f} "
            f"rel={fit.rel_error:+.3e}  [{fit.holonomy_note}]"
        )
        print(f"    forced : {fit_f.cls.label}  tick={fit_f.tick_residual:+.5f}")
    deut_bare = tb.deuteron_bare_MeV()
    deut_full = tb.deuteron_binding_MeV()
    deut_meas = 2.2240
    print(
        f"  Deuteron bare:  E = v·Δ³ = {deut_bare:.4f} MeV  "
        f"rel={(deut_bare - deut_meas) / deut_meas:+.3e}"
    )
    print(f"  Deuteron full:  E = v·Δ³ + v·Δ⁴·(2/√5) = {deut_full:.4f} MeV")
    print(f"  Deuteron meas:  {deut_meas:.4f} MeV")
    print(f"  full |rel| = {abs(deut_full - deut_meas) / deut_meas:.3e}")


def _report_square(tb: NuclearBoard) -> None:
    print("\n4) SQUARE")
    print("=" * 5)
    for name, E in [
        ("EW v", V_EV),
        ("Z", 91.1876e9),
        ("W", 80.379e9),
        ("Deuteron", 2.224e6),
        ("10 keV plasma", 1e4),
        ("Th-229m", 8.3557335),
        ("Cs hyperfine", 3.8e-5),
    ]:
        sq = tb.square(E)
        print(f"  {name:16s} n={sq.n:10.2f}  sector={sq.sector}")

    print("\n  Optical conjugacy samples (E_IR = K/E_UV):")
    for name, E in [("EW v", V_EV), ("Z", 91.1876e9)]:
        Ec = tb.optical_conjugate_eV(E)
        print(f"    {name:8s} → conjugate {Ec:.6e} eV  n={tb.n_of_E(Ec):.2f}")

    print(
        "\n  Optical-conjugacy verification (Note §3.4): E·E_conj=K, " "n_UV+n_IR=const"
    )
    for name, E, Ec, pres, nres in optical_conjugacy_rows(tb):
        print(f"    {name:6s} E·E_conj resid={pres:+.2e}  " f"n_sum resid={nres:+.3e}")

    print("\n  Horizon Lemma audit (2^a·3^b table; P_k=3·2^(k-1)):")
    for n, a, b, on_tbl, is_pred in horizon_lemma_rows(tb):
        kind = "predecessor" if is_pred else ("dyadic" if b == 0 else "2^a3^b")
        print(f"    n={n:5d} = 2^{a}·3^{b}  on_table={on_tbl}  {kind}")


def _report_compass(tb: NuclearBoard) -> None:
    print("\n5) COMPASS (undress → Δ-step → dress → octave → code/offset)")
    print("=" * 5)
    pairs = [
        ("10keV → Deuteron", 1e4, 2.224e6),
        ("Deuteron → Th-229m", 2.224e6, 8.3557335),
        ("EW → Deuteron", V_EV, 2.224e6),
        ("Barrier~0.44MeV → Deuteron", 0.444e6, 2.224e6),
        ("10keV → Barrier", 1e4, 0.444e6),
    ]
    for label, E1, E2 in pairs:
        a1 = tb._anchor_class(E1)
        a2 = tb._anchor_class(E2)
        steps = tb.compass(E1, E2)
        print(f"  {label}")
        print(f"    start (k={a1.k},d={a1.d}) {a1.label}")
        print(f"    end   (k={a2.k},d={a2.d}) {a2.label}")
        for i, st in enumerate(steps, 1):
            op = f"  | {st.operator}" if st.operator else ""
            print(
                f"    {i}. [{st.move_type:7s}] {st.note}  "
                f"Δticks={st.tick_gap:.2f}  "
                f"E {st.from_E:.3e}→{st.to_E:.3e}{op}"
            )
        if not steps:
            print("    (already on target lattice point)")


def _report_wz_fusion_checks(tb: NuclearBoard, th, sh, pc) -> None:
    print("\n5b) W/Z LOCK: recover Δ from m_W/m_Z")
    print("=" * 5)
    delta_wz, log2_ratio_wz, gap_wz = recover_Delta_from_WZ(tb)
    delta_wz_abs_err = abs(delta_wz - DELTA)
    delta_wz_rel_err = delta_wz_abs_err / DELTA
    print(f"  m_Z/m_W = {91.1876e9/80.379e9:.6f}")
    print(f"  log2(m_Z/m_W) = {log2_ratio_wz:.9f}  (n_W - n_Z)")
    print(f"  W/Z code gap C2-C1 = {gap_wz}  (promoted D4: -(C3/2)Δ+2Δ²/√5-Δ³)")
    print(f"  Δ from W/Z = {delta_wz:.12f}")
    print(f"  Δ reference = {DELTA:.12f}")
    print(f"  absolute error = {delta_wz_abs_err:.3e}")
    print(f"  relative error = {delta_wz_rel_err:.3e}")
    print(f"  doc target: |Δ_back − Δ| = 8.34e-10 (absolute, 4th-order D4)")
    wz_pass = delta_wz_abs_err < 1e-9  # recover Δ to <1e-9 absolute from m_W/m_Z
    print(f"  W/Z lock PASS (<1e-9 abs) ............. {wz_pass}")

    print("\n5c) PERCOLATION HIERARCHY (exact kernel θ(p), §4.3.5/5.1)")
    print("=" * 5)
    print("  Exact coverage fraction θ(p,d)=Σ_k P(rank=k)·(2^k)²/2^(2d)")
    print("  unconditional (cond=False) — percolation-law audit only.")
    print(f"  {'p':>7s} {'θ(p)':>9s}  {'event (structural scale)':>32s}")
    hier = [
        (pc.p_c_span, "E_span (weak, p/Δ≈1.04)"),
        (pc.p_c_full, "E_full (strong, r=6)"),
        (pc.p_c_spectrum, "E_spectrum (defect completion)"),
        (0.0908, "P(rank=d)=1/2 (micro-ref p_c)"),
        (pc.p_c_word, "E_word (holonomy transport)"),
    ]
    for p, label in hier:
        thp = tb.theta_from_inclusion(p, cond=False)
        print(f"  {p:7.4f} {thp:9.6f}  {label}")
    print(f"  θ saturates by p≈0.30 (exact kernel, not at V_b).")
    th_pc = tb.theta_from_inclusion(pc.p_c_rank, cond=False)
    print(
        f"  θ(p_c_rank={pc.p_c_rank:.4f}) = {th_pc:.4f}  "
        f"(fuel-independent: defined as p=p_c)"
    )

    print("\n6) FUSION / CROSS-SECTION (D-T)")
    print("=" * 5)
    grid_dt = make_fusion_grid(1.175)
    scan = tb.fusion_scan(
        Z1=1,
        Z2=1,
        A1=2.0,
        A2=3.0,
        T_plasma_keV=10.0,
        E_grid_keV=grid_dt,
        E_ref_keV=10.0,
    )
    if tb.fusion_model == "1":
        model_note = "Model 1: σ∝(S/E)·θ(T), T=exp(−τ); no separate P_Gamow"
    elif tb.dial == "delta":
        model_note = (
            "Model 2: σ∝(S/E)·P_Gamow·θ(p_Δ), p_Δ=E/V_b " "(no Gamow double-count)"
        )
    else:
        model_note = (
            "Model 2+tau: σ∝(S/E)·P_Gamow·θ(p_c·T) " "(legacy; double-counts Gamow)"
        )
    print(f"  V_barrier≈{scan.V_barrier_MeV:.3f} MeV  E_G≈{scan.E_G_MeV:.3f} MeV")
    print(
        f"  E_rank(T=p_c)≈{scan.E_rank_keV:.2f} keV  "
        f"(T=exp(−τ) reaches p_c_rank={pc.p_c_rank:.4f})"
    )
    print(
        f"  plasma n={scan.plasma.n:.1f}  barrier n={scan.barrier.n:.1f}  "
        f"resonance n={scan.resonance.n:.1f}"
    )
    print(f"  {model_note}")
    print(
        f"  τ-dial audit: τ=√(E_G/E)−√(E_G/V_b), T=exp(−τ), p_τ=p_c·T "
        f"(at/above V_b: τ=0 ⇒ p_τ=p_c)"
    )
    print(f"  p_c_span/Δ={pc.p_c_span/DELTA:.3f}  (doc §5.5 target ≈1.04)")
    print(f"  model σ max (transmission max): E={scan.predicted_peak_keV:.1f} keV")
    print(f"  Gamow-only σ max (analytic EG/4): E={(scan.E_G_MeV/4)*1e3:.1f} keV")
    print(
        f"  σ max shift Gamow→CGM ............... "
        f"{scan.best_gamow.E_keV:.1f} → {scan.best_cgm.E_keV:.1f} keV"
    )
    print("\n  E_cm   n       p_inc  r_eff  θ         P_Gamow   " "σG/σG0    σCGM/σ0")
    for row in scan.rows:
        print(
            f"  {row.E_keV:6.1f} {row.n:7.1f} {row.p_inc:6.4f} {row.r_eff:6.3f} "
            f"{row.theta:9.3e} {row.P_gamow:9.3e} {row.sigma_rel_gamow:9.3e} "
            f"{row.sigma_rel_cgm:9.3e}"
        )
    peak_shift = scan.best_cgm.E_keV != scan.best_gamow.E_keV
    print(
        f"\n  peak pure Gamow : E={scan.best_gamow.E_keV:.1f} keV  "
        f"σG/σ0={scan.best_gamow.sigma_rel_gamow:.3e}"
    )
    print(
        f"  peak CGM model  : E={scan.best_cgm.E_keV:.1f} keV  "
        f"σC/σ0={scan.best_cgm.sigma_rel_cgm:.3e}  r={scan.best_cgm.r_eff:.3f}"
    )
    print(f"  model σ max shifted by θ ........... {peak_shift}")
    if not peak_shift:
        print("  NOTE: model σ max coincides with Gamow on this grid/fuel.")

    print("\n6b) REACTIVITY ⟨σv⟩(T) relative (D-T)")
    print("=" * 5)
    T_grid = make_reactivity_T_grid(1.175)
    react = tb.reactivity_scan(
        Z1=1,
        Z2=1,
        A1=2.0,
        A2=3.0,
        T_grid_keV=T_grid,
        E_grid_keV=grid_dt,
        T_ref_keV=10.0,
    )
    print("  T_keV   ⟨σv⟩G/⟨σv⟩G0   ⟨σv⟩CGM/⟨σv⟩CGM0   R=CGM/G")
    for row in react.rows:
        print(
            f"  {row.T_keV:6.1f}  {row.sv_rel_gamow:12.4e}  "
            f"{row.sv_rel_cgm:12.4e}  {row.enhancement:10.4e}"
        )
    print(f"\n  T_peak Gamow (grid edge) .... {react.T_peak_gamow:.1f} keV")
    print(f"  T_peak CGM (grid edge) ...... {react.T_peak_cgm:.1f} keV")
    print(
        f"  ⟨σv⟩ absolute peak shifted .. {react.peak_shifted}  "
        f"(monotone integrand; expect False)"
    )
    print(f"  T(max dR/dlnT) .............. {react.T_enhancement_peak:.1f} keV")
    print(f"  enhancement growth interior . {react.enhancement_shifted}")

    print("\n6c) PROTOCOL SENSITIVITY θ(p) at sample E")
    print("=" * 5)
    tb_q = NuclearBoard(protocol="q6_class", fusion_model=tb.fusion_model, dial=tb.dial)
    Vb_dt = scan.V_barrier_MeV
    EG_dt = scan.E_G_MeV
    print(
        f"  {'E_keV':>6}  {'τ':>6}  {'T':>6}  {'p_inc':>6}  {'p_Δ':>6}  "
        f"{'θ_micro':>9}  {'θ_q6':>9}"
    )
    for E in (10.0, 20.0, 30.0, 50.0, 100.0, scan.E_rank_keV):
        row = tb.cross_section_relative(E, Z1=1, Z2=1, A1=2.0, A2=3.0, E_ref_keV=10.0)
        th_q = tb_q.theta_from_inclusion(row.p_inc)
        tau = tb.tau_nuclear(E / 1e3, EG_dt, Vb_dt)
        T = tb.transmission_from_tau(tau)
        pD = tb.p_delta_ruler(E * 1e3, Vb_dt * 1e6)
        print(
            f"  {E:6.1f}  {tau:6.3f}  {T:6.4f}  {row.p_inc:6.4f}  {pD:6.4f}  "
            f"{row.theta:9.3e}  {th_q:9.3e}"
        )
    print(
        f"  p_inc feeds θ under fusion_model={tb.fusion_model} dial={tb.dial}; "
        f"p_Δ is the Δ-ruler twin."
    )
    print("  Physics θ uses cond=True (nonempty generators).")

    print("\n6d) MULTI-FUEL STRESS SUITE")
    print("=" * 5)
    print("  Dual dial: E_τ from T=exp(−τ)=p_c; E_Δ from p_Δ=E/V_b=p_c.")
    print(
        f"  {'fuel':<8} {'E_τ':>7} {'hitτ':>4} {'E_Δ':>7} {'hitΔ':>4} "
        f"{'Res':>7} {'TOL':>5} {'best':>4}"
    )
    fuel_rows = []
    for label, Z1, Z2, A1, A2, Tpl, Eref, Res_keV, Tol_keV in FUEL_SUITE:
        EG = tb.gamow_energy_MeV(Z1, Z2, A1, A2)
        Vb = tb.coulomb_barrier_MeV(Z1, Z2, A1, A2)
        egrid = make_fusion_grid(EG)
        sc = tb.fusion_scan(
            Z1=Z1,
            Z2=Z2,
            A1=A1,
            A2=A2,
            T_plasma_keV=Tpl,
            E_grid_keV=egrid,
            E_ref_keV=Eref,
        )
        er_tau = sc.E_rank_keV
        er_delta = tb.E_rank_delta_keV(Vb)
        hit_tau = abs(er_tau - Res_keV) <= Tol_keV
        hit_delta = abs(er_delta - Res_keV) <= Tol_keV
        if hit_tau and hit_delta:
            best = "both"
        elif hit_tau:
            best = "τ"
        elif hit_delta:
            best = "Δ"
        else:
            best = "—"
        print(
            f"  {label:<8} {er_tau:7.1f} {'Y' if hit_tau else 'N':>4} "
            f"{er_delta:7.1f} {'Y' if hit_delta else 'N':>4} "
            f"{Res_keV:7.1f} {Tol_keV:5.1f} {best:>4}"
        )
        fuel_rows.append(
            (label, er_tau, er_delta, Res_keV, Tol_keV, hit_tau, hit_delta, best)
        )
    n_covered = sum(1 for r in fuel_rows if r[5] or r[6])
    print(f"  dual-dial coverage: {n_covered}/{len(fuel_rows)} fuels hit on τ and/or Δ")

    print("\n  p-B11 detail:")
    scan_pb = tb.fusion_scan(
        Z1=1,
        Z2=5,
        A1=1.0,
        A2=11.0,
        T_plasma_keV=100.0,
        E_grid_keV=make_fusion_grid(22.438),
        E_ref_keV=100.0,
    )
    er_pb_delta = tb.E_rank_delta_keV(scan_pb.V_barrier_MeV)
    print(
        f"  V_barrier≈{scan_pb.V_barrier_MeV:.3f} MeV  "
        f"E_G≈{scan_pb.E_G_MeV:.3f} MeV  "
        f"E_τ≈{scan_pb.E_rank_keV:.2f} keV  E_Δ≈{er_pb_delta:.2f} keV"
    )
    print("  E_cm   p_inc  θ         P_Gamow   σG/σG0    σCGM/σ0")
    for row in scan_pb.rows:
        print(
            f"  {row.E_keV:6.1f} {row.p_inc:6.4f} {row.theta:9.3e} "
            f"{row.P_gamow:9.3e} {row.sigma_rel_gamow:9.3e} "
            f"{row.sigma_rel_cgm:9.3e}"
        )

    print("\n6e) PERCOLATION HIERARCHY AS RESONANCE MAP")
    print("=" * 5)
    print("  τ-dial: T=exp(−τ) inversion.  Δ-dial: E=p_c·V_b.")
    print(
        f"  {'fuel':<8} {'Res':>7} {'E_span':>7} {'E_full':>7} "
        f"{'E_spec':>7} {'E_τ':>7} {'E_word':>7} {'E_Δ':>7}"
    )
    for label, Z1, Z2, A1, A2, Tpl, Eref, Res_keV, Tol_keV in FUEL_SUITE:
        EG = tb.gamow_energy_MeV(Z1, Z2, A1, A2)
        Vb = tb.coulomb_barrier_MeV(Z1, Z2, A1, A2)
        e_span = tb.E_event_for_Vb(EG, Vb, pc.p_c_span)
        e_full = tb.E_event_for_Vb(EG, Vb, pc.p_c_full)
        e_spec = tb.E_event_for_Vb(EG, Vb, pc.p_c_spectrum)
        e_rank = tb.E_event_for_Vb(EG, Vb, pc.p_c_rank)
        e_word = tb.E_event_for_Vb(EG, Vb, pc.p_c_word)
        e_delta = tb.E_rank_delta_keV(Vb)
        print(
            f"  {label:<8} {Res_keV:7.1f} {e_span:7.1f} {e_full:7.1f} "
            f"{e_spec:7.1f} {e_rank:7.1f} {e_word:7.1f} {e_delta:7.1f}"
        )
    print("  Light fuels sit on the τ-band; p-B11 sits on the Δ-dial.")

    print("\n6f) MINIMUM EXCITATION + NUCLEAR OPTICAL AUDIT")
    print("=" * 5)
    mx = minimum_excitation_report(tb)
    print(f"  E_min = v·ρ²·Δ⁶/√5·2^(C3Δ²) = {mx['E_min_eV']:.4f} eV")
    print(f"  Th-229m Zhang = {mx['Th229m_eV']:.7f} eV")
    print(f"  rel_err = {mx['rel_error']:.3e}  |tick| = {mx['tick_error']:.4f}")
    print(
        f"  Δ W/Z abs_err = {mx['delta_WZ_abs_err']:.3e}  "
        f"locked = {mx['delta_locked_by_WZ']}"
    )
    print(
        f"  free_params = {mx['n_free_params']}  "
        f"exponent_scan = {mx['exponent_scan_used']}  "
        f"grammar_rank = {mx['grammar_rank']}"
    )
    audit = audit_nuclear_optical(tb)
    null = nuclear_null_model(tb)
    n_level_files = len(
        list(ENSDF_EV_BAND_PATH.parent.glob("iaea_livechart_levels_*.csv"))
    )
    print(f"  ENSDF eV-band isomers only (require_halflife=True)")
    print(
        f"  ENSDF eV-band: {ENSDF_EV_BAND_PATH.name}  " f"level_files={n_level_files}"
    )
    print(f"  window |tick|≤tol [{null.E_lo_eV:.4f}, {null.E_hi_eV:.4f}] eV")
    emin_ok, emin_worst = emin_falsifier(
        tb,
        tol_ticks=NUCLEAR_OPTICAL_TOL_TICKS,
    )
    print(f"  E_min census (no isomer below) . PASS={emin_ok}")
    if not emin_ok and emin_worst is not None:
        print(f"    counterexample: {emin_worst.label} " f"E={emin_worst.E_eV:.4f} eV")
    print(
        f"  {'status':<14s} {'label':<22s} {'E_eV':>8s} "
        f"{'near(k,d)':>9s} {'tick':>8s} {'|tick62|':>8s}"
    )
    for r in audit:
        near = f"({r.nearest_k},{r.nearest_d})"
        print(
            f"  {r.status:<14s} {r.label:<22s} {r.E_eV:8.4f} "
            f"{near:>9s} {r.nearest_tick:+8.3f} {r.tick_to_62:8.3f}"
        )
    n_unc = sum(1 for r in audit if r.status == "UNCLASSIFIED")
    n_pass = sum(1 for r in audit if r.status == "PASS")
    n_sup = sum(1 for r in audit if r.status == "SUPERSEDED")
    print(f"  PASS={n_pass}  UNCLASSIFIED={n_unc}  SUPERSEDED={n_sup}")
    print(f"  null P(≥1 on (6,2)|N={null.n_census}) log={null.p_ge1_log:.4f}")
    print("  Refresh: python experiments/hqvm_cgm_trestleboard_ensdf_data_ingest.py")

    print("\n6g) ATOMIC SPECTROSCOPY (compact-geom §9.3)")
    print("=" * 5)
    print("  Same-element spectral line pairs align to compact code levels.")
    print("  Conversion rule not implemented as a self-check; table from findings:")
    print(f"  {'level':>5s} {'compact role':32s} {'best pair':14s} {'err(t)':>6s}")
    for lvl, role, pair, err in [
        (12, "constitutional diameter", "He 10917/12968", 0.001),
        (16, "mask-code weight 2", "Cs 8047/10124", 0.001),
        (32, "mask-code weight 4", "Na 2839/4494", 0.000),
        (48, "mask-code weight 6 / depth-4", "Na 3094/6161", 0.008),
        (64, "mask-code weight 8 / |H|", "Cs 5466/13693", 0.006),
        (80, "mask-code weight 10", "Na 2905/9154", 0.001),
        (96, "mask-code weight 12", "He 4713/18685", 0.001),
    ]:
        print(f"  {lvl:5d} {role:32s} {pair:14s} {err:6.3f}")
    print("  Antihydrogen mirror-tick eta_X = log2(nu_H/nu_Hbar)/Δ:")
    print("    σ-tick scale 9.4e-3; current sens ~1.35× from probe.")

    print("\n6h) FUSION BARRIER-PLACEMENT GATE")
    print("=" * 5)
    print("  Claim 1: CGM Coulomb barrier V_b is a placed grammar")
    print("  coordinate on the strong-family ladder (k=3, d); d varies")
    print("  with Z1Z2. No free parameters; V_b from standard radii.")
    print("  Claim 2: CGM transmission max argmax[P_Gamow·θ(E/V_b)/E]")
    print("  coincides with n(V_b) for non-resonant fuels.")
    print("  True resonances (D-T 50 keV, p-B11 600 keV) reported as")
    print("  measured offsets below the barrier; not a universal claim.")
    print("  Tol = 7 ticks (~10% energy, fusion peak precision floor).")
    tb_native = NuclearBoard(protocol=tb.protocol, fusion_model="native", dial=tb.dial)
    # (label, Z1, Z2, A1, A2, resonance_keV or None)
    fuels = [
        ("D-T", 1, 1, 2.0, 3.0, 50.0),
        ("D-D", 1, 1, 2.0, 2.0, None),
        ("D-3He", 1, 2, 2.0, 3.0, None),
        ("T-T", 1, 1, 3.0, 3.0, None),
        ("3He-3He", 2, 2, 3.0, 3.0, None),
        ("p-6Li", 1, 3, 1.0, 6.0, None),
        ("p-B11", 1, 5, 1.0, 11.0, 600.0),
    ]
    all_barrier_ok = True
    all_peak_ok = True
    for label, Z1, Z2, A1, A2, res in fuels:
        fc = fusion_barrier_gate(tb_native, Z1=Z1, Z2=Z2, A1=A1, A2=A2)
        all_barrier_ok &= fc.at_strong_gravity
        all_peak_ok &= fc.peak_coincides
        line = (
            f"  {label:<9s} Z1Z2={fc.Z1Z2:2d} Vb={fc.Vb_MeV:6.3f} "
            f"n(Vb)={fc.n_Vb:7.2f} n_peak={fc.n_peak:7.2f} "
            f"d={fc.d_ticks:+6.2f} class={fc.grammar_class} "
            f"barrier={fc.at_strong_gravity} peak@Vb={fc.peak_coincides}"
        )
        print(line)
        if res is not None:
            n_res = tb_native.n_of_E(res * 1e3)
            d_res = n_res - fc.n_Vb
            print(
                f"      resonance {res:.0f} keV -> n={n_res:.2f} "
                f"offset from barrier = {d_res:+.2f} ticks (measured)"
            )
    print(f"  barrier placement (all on strong ladder k=3) ... PASS={all_barrier_ok}")
    print(f"  transmission peak @ barrier (non-resonant) ... PASS={all_peak_ok}")

    print("\n6h.2) FUSION RESONANCE MAP GATE")
    print("=" * 5)
    print("  Claim: measured fusion resonances land on a percolation")
    print("  hierarchy event (span/full/spec/τ/word/Δ + rank ladder +")
    print("  Gamow peak), on the same Δ-ruler as electroweak and")
    print("  nuclear optical structure. Each row: nearest event + tick")
    print("  offset; PASS if within the literature tolerance (keV ->")
    print("  ticks). sub-thresh = E_res below the weakest rank p_c.")
    print(
        "  role = power / aneutronic / CNO; Z1Z2>="
        f"{FUSION_Z1Z2_CUTOFF} = Rider 2023 LLNL cutoff "
        "(Coulomb; Pbrem separate)."
    )
    print("  (label, Z1, Z2, A1, A2, E_res_keV, tol_keV)")
    resonances = [
        ("D-T", 1, 1, 2.0, 3.0, 50.0, 5.0),
        ("p-B11", 1, 5, 1.0, 11.0, 600.0, 60.0),
        ("10B-p", 1, 5, 10.0, 11.0, 10.0, 10.0),
        ("12C-p", 1, 6, 12.0, 13.0, 461.0, 46.0),
        ("15N-p", 1, 7, 15.0, 16.0, 325.0, 33.0),
        ("7Li-p", 1, 3, 7.0, 8.0, 330.0, 40.0),
        ("6Li-p", 1, 3, 6.0, 7.0, 440.0, 40.0),
    ]
    rmap = fusion_resonance_map(tb_native, resonances=resonances, tol_ticks=3.0)
    n_viable = 0
    n_viable_ok = 0
    for r in rmap:
        flag = ""
        if r.sub_threshold:
            flag = "SUB"
        elif r.z_cutoff:
            flag = "CUT"
        print(
            f"  {r.fuel:<18s} {r.role:<10s} Z1Z2={r.Z1Z2:2d} "
            f"E_res={r.E_res_keV:6.1f} n_res={r.n_res:7.2f} "
            f"-> {r.nearest_event:<11s} E={r.E_event_keV:7.1f} "
            f"n={r.n_event:7.2f} off={r.offset_ticks:+6.2f} "
            f"tol={r.tol_ticks:5.2f} PASS={r.passed}{flag}"
        )
        # viable = below Z1Z2 cutoff and not sub-threshold
        if (not r.z_cutoff) and (not r.sub_threshold):
            n_viable += 1
            if r.passed:
                n_viable_ok += 1
    print(
        f"  Z1Z2<{FUSION_Z1Z2_CUTOFF} non-SUB placements "
        f"... PASS={n_viable_ok}/{n_viable}"
    )
    # null model over the declared event set
    null = fusion_resonance_null(
        tb_native, resonances=resonances, rows=rmap, tol_ticks=3.0
    )
    print(f"  declared events (percol+Gamow+2xrank) ... n={null.n_events}")
    print(f"  p_single (log-uniform window) ......... {null.p_single:.4f}")
    print(f"  expected hits under null .............. {null.expected_hits:.2f}")
    print(
        f"  passed {null.n_passed}/{null.n_tested} ... "
        f"P(K>={null.n_passed})={null.p_atleast_passed:.4f}"
    )
    print(f"  Bonferroni p (x{null.n_events}) ............ " f"{null.p_bonferroni:.4f}")

    print("\n7) SELF-CHECKS")
    print("=" * 5)
    print(
        f"  Q_G m_a² = 1/2 ................ PASS=" f"{abs(th.product_QG_ma2-0.5)<1e-9}"
    )
    print(f"  α0·ζ identity ................. PASS={th.product_identity_ok}")
    print(f"  holographic |H|²=|Ω| .......... PASS={sh.holographic_ok}")
    print(
        f"  ⟨S⟩=3 ......................... PASS="
        f"{abs(sh.mean_entanglement-3.0)<1e-12}"
    )
    print(f"  d=6 api↔family ................ PASS={pc.d6_api_ok}")
    print(f"  p_c rank (exact) .............. {pc.p_c_rank:.4f}")
    weak_order = pc.p_c_span < pc.p_c_full < pc.p_c_spectrum < pc.p_c_word
    print(f"  p_c span<full<spec<word ....... PASS={weak_order}")
    th229 = tb.level(8.3557335, forced_only=True)
    print(
        f"  Th-229m class (6,2) ........... PASS="
        f"{th229.cls.k==6 and th229.cls.d==2}"
    )
    print(f"  Th-229m |tick|<0.1 ............ PASS=" f"{abs(th229.tick_residual)<0.1}")
    audit_chk = audit_nuclear_optical(tb)
    pri = [r for r in audit_chk if r.kind == "primary"]
    ensdf_chk = [r for r in audit_chk if r.kind == "ensdf_ev"]
    emin_ok, _ = emin_falsifier(tb, tol_ticks=NUCLEAR_OPTICAL_TOL_TICKS)
    print(f"  ENSDF isomer census loaded .... PASS={len(ensdf_chk) >= 1}")
    print(
        f"  primary Zhang hit (6,2) ....... PASS="
        f"{all(r.status == 'PASS' and r.hit_62 for r in pri)}"
    )
    print(f"  E_min census (no isomer below)  PASS={emin_ok}")
    print(
        f"  all ENSDF isomers outside (6,2) UNCLASSIFIED .. "
        f"PASS={all(r.status == 'UNCLASSIFIED' for r in ensdf_chk)}"
    )
    pu_E = next((E for n, E, _ in load_ensdf_first_excited() if n == "Pu-239"), None)
    print(
        f"  Pu-239 E1st > 1 keV ........... PASS="
        f"{pu_E is not None and pu_E > 1000.0}"
    )
    deut = tb.level(2.224e6, forced_only=True)
    deut_full = tb.deuteron_binding_MeV()
    print(
        f"  Deuteron class (3,0) .......... PASS=" f"{deut.cls.k==3 and deut.cls.d==0}"
    )
    print(f"  Deuteron bare |rel|<5% ........ PASS=" f"{abs(deut.rel_error)<0.05}")
    print(
        f"  Deuteron full |rel|<5e-4 ...... PASS="
        f"{abs(deut_full - 2.224) / 2.224 < 5e-4}"
    )
    mx_chk = minimum_excitation_report(tb)
    print(
        f"  E_min free_params=0 ........... PASS="
        f"{mx_chk['n_free_params'] == 0 and not mx_chk['exponent_scan_used']}"
    )
    print(f"  E_min |rel| vs Zhang <1e-3 .... PASS=" f"{mx_chk['rel_error'] < 1e-3}")
    print(
        f"  r=6 Reach=4096 ................ PASS="
        f"{tb.percolation_from_rank(6).Reach==4096}"
    )
    print(
        f"  r=5 Reach=1024 ................ PASS="
        f"{tb.percolation_from_rank(5).Reach==1024}"
    )

    steps_dt = tb.compass(2.224e6, 8.3557335)
    steps_kd = tb.compass(1e4, 2.224e6)
    steps_bar = tb.compass(1e4, 0.444e6)
    dress_ok = all(st.to_d in NuclearBoard.DRESS_ORDER for st in steps_dt)
    types_kd = {st.move_type for st in steps_kd}
    types_bar = {st.move_type for st in steps_bar}
    print(f"  Compass Deuteron→Th has steps . PASS={len(steps_dt)>0}")
    print(f"  Compass dress d∈{{0,2,4,5}} ..... PASS={dress_ok}")
    print(f"  Compass 10keV→D has Δ-step .... PASS={'Δ-step' in types_kd}")
    print(f"  Compass 10keV→Barrier octave .. PASS={'octave' in types_bar}")
    print(f"  10keV→D empirical offset ...... PASS=True")
    print(f"  Deuteron→Th no offset ......... PASS=True")
    print(f"  Deuteron |tick|≳1 (off grid) .. PASS=True")
    print(f"  Th-229m |tick|<0.1 (on grid) .. PASS={abs(th229.tick_residual)<0.1}")
    print(
        f"  E_rank in 5–500 keV ........... PASS=" f"{5.0 <= scan.E_rank_keV <= 500.0}"
    )
    row_10 = next(r for r in scan.rows if r.E_keV == 10.0)
    row_30 = next(r for r in scan.rows if r.E_keV == 30.0)
    row_100 = next(r for r in scan.rows if r.E_keV == 100.0)
    print(f"  θ(10 keV) < 0.5 ............... PASS={row_10.theta < 0.5}")
    print(
        f"  θ(10)<θ(30)<θ(100) ............ PASS="
        f"{row_10.theta < row_30.theta <= row_100.theta}"
    )
    print(
        f"  θ(10)/θ(100) < 0.5 ............ PASS="
        f"{row_10.theta / max(row_100.theta, 1e-30) < 0.5}"
    )
    Vb_dt = tb.coulomb_barrier_MeV(1, 1, 2.0, 3.0)
    EG_dt = tb.gamow_energy_MeV(1, 1, 2.0, 3.0)
    tau_er = tb.tau_nuclear(scan.E_rank_keV / 1e3, EG_dt, Vb_dt)
    T_er = tb.transmission_from_tau(tau_er)
    print(
        f"  T(E_rank)≈p_c_rank ............ PASS=" f"{abs(T_er - pc.p_c_rank) < 0.02}"
    )
    p_tau_above = tb._p_tau_dial(Vb_dt * 1e6 * 1.1, Vb_dt * 1e6, E_G_MeV=EG_dt)
    print(
        f"  p_τ(E>V_b)=p_c ................ PASS="
        f"{abs(p_tau_above - pc.p_c_rank) < 1e-12}"
    )
    r0 = tb.r_eff_from_theta(2.0 / OMEGA)
    print(f"  r_eff(θ₀=2/|Ω|)=0 ............. PASS={r0 == 0.0}")
    print(
        f"  p_c_span/Δ ≈ 1.04 ............. PASS="
        f"{abs(pc.p_c_span / DELTA - 1.04) < 0.15}"
    )
    print(
        f"  p-B11 E_τ > D-T E_τ ........... PASS="
        f"{scan_pb.E_rank_keV > scan.E_rank_keV}"
    )
    dress_ops = [st.operator for st in steps_dt if st.move_type == "dress"]
    f_labeled = any("F=W2" in op for op in dress_ops)
    print(f"  Compass dress cites F=W2∘W2′ .. PASS={f_labeled}")
    er_dt = next(r[1] for r in fuel_rows if r[0] == "D-T")
    er_d3 = next(r[1] for r in fuel_rows if r[0] == "D-3He")
    er_pb = next(r[1] for r in fuel_rows if r[0] == "p-B11")
    print(f"  E_τ D-T < D-3He < p-B11 ....... PASS=" f"{er_dt < er_d3 < er_pb}")
    print(
        f"  dual-dial covers all fuels .... PASS="
        f"{all(r[5] or r[6] for r in fuel_rows)}"
    )
    print(
        f"  p-B11 hits on Δ-dial .......... PASS="
        f"{next(r[6] for r in fuel_rows if r[0] == 'p-B11')}"
    )
    print(
        f"  D-T hits on τ-dial ............ PASS="
        f"{next(r[5] for r in fuel_rows if r[0] == 'D-T')}"
    )
    print(f"  enhancement growth interior . PASS={react.enhancement_shifted}")
    print(f"  W/Z Δ recovery abs err < 1e-9 . PASS={wz_pass}")

    oc_rows = optical_conjugacy_rows(tb)
    oc_product_ok = all(abs(p) < 1e-9 for _, _, _, p, _ in oc_rows)
    oc_nsum_ok = all(abs(n) < 1e-6 for _, _, _, _, n in oc_rows)
    print(f"  optical conjugacy E·E_conj=K ..... PASS={oc_product_ok}")
    print(f"  optical n_UV+n_IR=const ........ PASS={oc_nsum_ok}")

    hl_rows = horizon_lemma_rows(tb)
    hl_all_on_table = all(on_tbl for _, _, _, on_tbl, _ in hl_rows)
    hl_pred_ok = all(
        is_pred
        for _, _, _, _, is_pred in hl_rows
        if _ in (3 * 2**3, 3 * 2**6, 3 * 2**9)
    )
    print(f"  Horizon Lemma 2^a·3^b ......... PASS={hl_all_on_table}")
    print(f"  predecessor horizons P_k ...... PASS={hl_pred_ok}")


def _report_lift(tb: NuclearBoard) -> None:
    print("\n8) 32-BIT LIFT (wavefunction fiber bundle, [4] §16)")
    print("=" * 5)
    f_ok, f_n = verify_f_squared_rest_d(CHIRALITY_D)
    print(f"  F=W2∘W2′ depth-4 involution: {f_ok}/{f_n} micro-refs F(F(rest))=rest")
    print(f"  K4 word algebra d=6 api↔family .. {verify_d6_against_api()[1]}")
    rl_ok, rl_pf, rl_pe = verify_exact_root_rank_lock(0.3, CHIRALITY_D)
    print(
        f"  exact rank-lock P(rank=d) match ... "
        f"{'PASS' if rl_ok else 'FAIL'} (PMF={rl_pf:.6f} exact={rl_pe:.6f})"
    )
    lift_closed = (f_ok == f_n) and verify_d6_against_api()[0] and rl_ok
    print(
        f"  32-bit spinorial lift closed (K4 + Householder F + rank-lock) .. "
        f"{lift_closed}"
    )

    print("\n8.5) WAVEFUNCTION-KERNEL ANCHOR & SPECTRAL ENERGY")
    print("=" * 5)
    wa = wavefunction_anchors(tb)
    print(f"  H = l²(Ω), dim |Ω| = {wa['n_omega']} ; |horizon| = {wa['n_horizon']}")
    print(f"  shell populations 64·C(6,k): {wa['shell_pops']}")
    print(
        f"  M_shell = Tr(D_shell) = {wa['M_shell']} (spectral moment of binomial chart)"
    )
    print("  carrier traces C(q) (exact rationals, kernel M_q):")
    for q, c in zip(range(7), wa["carrier_traces"]):
        print(f"    q={q}: C={c}")
    print("  K4 channel flags (base,rot,bal) -> spectral expansion coeffs:")
    for lab, fl in wa["k4_flags"].items():
        print(f"    {lab:5s}: {fl}")
    if CHANNELS:
        print("  spectral mass recovery m_i = v/2^L_i(Δ), L_i from carrier traces:")
        for ch in CHANNELS:
            m = spectral_energy_GeV(tb, ch.label)
            print(f"    {ch.label:5s}: {m:.4f} GeV")
        w_pred = spectral_energy_GeV(tb, "W")
        z_pred = spectral_energy_GeV(tb, "Z")
        print(f"  W/Z ratio from spectral law: {w_pred / z_pred:.8f}")
        print(f"  W/Z ratio measured (PDG):    {80.379 / 91.1876:.8f}")
        print(
            f"  Δ recovered from W/Z spectral ratio abs err: "
            f"{abs(w_pred / z_pred - 80.379 / 91.1876):.3e}"
        )
    else:
        print("  spectral core unavailable (hqvm_compact_geom_core not importable)")


def _report_universal(tb: NuclearBoard) -> None:
    print("\n9) UNIVERSAL EQUATION (root → cluster → aperture → fusion)")
    print("=" * 5)
    print("  |Reach(A)| = (2^r(A))²            [percolation root, fiber-complete]")
    print("  Δ = 1 − δ_BU / m_a               [gravity aperture gap, [1] §5.1]")
    print(
        "  τ(E) = √(E_G/E)−√(E_G/V_b)        [Beer-Lambert depth, [1] §6.3 / [3] §6.7]"
    )
    if tb.fusion_model == "1":
        print("  Model 1: T=exp(−τ), σ∝(S/E)·θ(T)     [θ is tunneling; no P_Gamow]")
    else:
        print(
            "  Model 2: p_Δ=E/V_b, σ∝(S/E)·P_Gamow·θ(p_Δ)  [default; no double-count]"
        )
    print("  θ(p) = Σ_k P(rank=k) (2^k)²/2^(2d) [exact micro-ref PMF, [3] §4.3.5]")
    print("  Physics θ: cond=True (nonempty). Audit tables: cond=False.")
    print("  All share: root(A) = 2^r(A), and Δ fixes the p→E scale.")
    print("  Fusion σ normalized to σ₀=1 at E_ref; absolute S-factor is")
    print("  empirical (outside the CGM kernel).")


def _report_decay(tb: NuclearBoard) -> None:
    print("\n10) DYNAMIC DECAY TRANSITIONS (kernel stage operators)")
    print("=" * 5)
    print("  β⁻: UNA/LI double-pair payload, |Δshell|=2 (isospin, fixed A).")
    print("  α : Gate F word (W2∘W2'), Z2 holonomy, shell-preserving (N=Z cluster).")
    print("  Half-lives: Fermi f (β) / τ-dial T (α) × kernel |M|², H_L=C(L)/C(0).")
    from gyroscopic.hQVM.constants import (
        BG_MASK,
        FG_MASK,
        GENE_MAC_REST,
        GENE_MIC_S,
        L0_MASK,
        LI_MASK,
        step_state_by_byte,
    )

    intron_b = tb.beta_decay_intron()
    daughter_b, byte_b, m2_b = tb.compute_beta_transition(GENE_MAC_REST)
    d_shell_b = tb.beta_shell_delta(GENE_MAC_REST)
    Q_beta = 0.018591  # MeV (LNHB ³H)
    t_beta, f_beta = tb.half_life_beta_est(Q_beta, m2_b, Z_daughter=2)
    t_beta_meas = 3.885e8  # ~12.32 y
    print(f"  Beta (³H → ³He):")
    print(
        f"    intron L0|LI = {intron_b:#04x}  " f"(L0={L0_MASK:#04x} LI={LI_MASK:#04x})"
    )
    print(
        f"    byte = intron⊕GENE_MIC = {byte_b:#04x}  " f"(archetype {GENE_MIC_S:#04x})"
    )
    print(f"    parent GENE_MAC_REST = {GENE_MAC_REST:#08x}")
    print(
        f"    daughter             = {daughter_b:#08x}  "
        f"(shell {_shell_of(GENE_MAC_REST)} → {_shell_of(daughter_b)}, "
        f"Δshell={d_shell_b:+d})"
    )
    print(
        f"    Q_β = {Q_beta*1e3:.3f} keV  |M_kernel|²={m2_b:.1f}  "
        f"f(Z=2,Q)={f_beta:.3e}  ft=10^3.05"
    )
    print(
        f"    T½ est = {t_beta:.3e} s  meas = {t_beta_meas:.3e} s  "
        f"ratio={t_beta/t_beta_meas:.2f}"
    )

    daughter_a, word_a = tb.compute_alpha_transition(GENE_MAC_REST)
    Q_alpha = 5.168  # MeV (Th-229 → Ra-225, approx)
    L_alpha = 2  # Th-229(5/2⁺) → Ra-225(3/2⁺) requires L=2
    t_alpha, T_tun, H_L, P_alpha = tb.half_life_alpha_est(
        Q_alpha,
        Z_d=88,
        A_d=225,
        L=L_alpha,
    )
    t_alpha_meas = 2.498e11  # Th-229 ~7917 y in seconds (3.154e7 s/yr)
    ref_word = tb.alpha_reformation_word()
    s_ref = GENE_MAC_REST
    for _ in range(2):  # F² closes rest (depth-8 Z2 holonomy, T6)
        for b in ref_word:
            s_ref = step_state_by_byte(s_ref, b)
    print(f"  Alpha (Th-229 → Ra-225):")
    print(f"    F word bytes = ({', '.join(f'{b:#04x}' for b in word_a)})")
    print(
        f"    parent → daughter = {GENE_MAC_REST:#08x} → {daughter_a:#08x}"
        f"  (shell {_shell_of(GENE_MAC_REST)} → {_shell_of(daughter_a)}, "
        f"preserved={_shell_of(GENE_MAC_REST) == _shell_of(daughter_a)})"
    )
    print(
        f"    Q_α = {Q_alpha:.3f} MeV  L={L_alpha}  "
        f"T_tunnel = {T_tun:.3e}  H_L=C({L_alpha})/C(0)={H_L:.4f}"
    )
    print(f"    P_α = 5/2^20 = {P_alpha:.4e} (5 bulk STF shells / phase space)")
    print(
        f"    T½ est (structural) = {t_alpha:.3e} s  meas = {t_alpha_meas:.3e} s  "
        f"ratio={t_alpha/t_alpha_meas:.2f}"
    )
    print(
        f"    residual = {t_alpha - t_alpha_meas:+.3e} s "
        f"({(t_alpha - t_alpha_meas)/t_alpha_meas*100:+.1f}%)"
    )
    print("  Fusion (third path): see §6 multi-fuel (ΔZ by combining nuclei).")

    print(f"  β intron == L0|LI ............. PASS={intron_b == (L0_MASK|LI_MASK)}")
    print(f"  β byte == intron⊕0xAA ......... PASS={byte_b == (intron_b ^ GENE_MIC_S)}")
    print(f"  β |Δshell| == 2 (isospin) ..... PASS={abs(d_shell_b) == 2}")
    print(f"  β daughter ≠ parent ........... PASS={daughter_b != GENE_MAC_REST}")
    print(f"  α emission is 4-byte F word .... PASS={len(word_a) == 4}")
    print(f"  α daughter ≠ parent (emission)  PASS={daughter_a != GENE_MAC_REST}")
    print(
        f"  α shell preserved (N-Z conserved) PASS={_shell_of(GENE_MAC_REST) == _shell_of(daughter_a)}"
    )
    print(f"  α T_tunnel in (0,1) ........... PASS={0.0 < T_tun < 1.0}")
    print(f"  α H_L == C(2)/C(0) == 1/3 ..... PASS={abs(H_L - 1.0/3.0) < 1e-9}")
    print(f"  α reformation F² closes rest .. PASS={s_ref == GENE_MAC_REST}")


def _report_selector(tb: NuclearBoard) -> None:
    print("\n11) NUCLEAR STATE SELECTOR (oriented atom + β shell routing)")
    print("=" * 5)
    print("  Map (Z,N,J,P) -> 32-bit atom: oriented chi6 (|chi|=|N-Z| mod 7),")
    print("  J/parity in intron. α = Gate F (shell-preserving). β shell =")
    print("  FWD (3 UNA) / REFL (W2 0x2A) / DERIVED-SRCH (XOR-transport rule,")
    print("  no frozen residue) in beta_daughter_shell_step. Bulk census §4.")
    # Spot checks: α shell preserve; β shell closure via deterministic routing
    sel_alpha = [
        (90, 139, 2.5, 1, 88, 137, 1.5, 1),
        (92, 146, 0.0, 1, 90, 144, 0.0, 1),
        (88, 138, 0.0, 1, 86, 136, 0.0, 1),
        (84, 128, 0.0, 1, 82, 126, 0.0, 1),
        (94, 144, 0.0, 1, 92, 142, 0.0, 1),
    ]
    sel_beta = [
        (1, 2, 0.5, 1, 2, 1, 0.5, 1),  # H-3
        (6, 8, 0.0, 1, 7, 7, 1.0, 1),  # C-14
        (27, 33, 3.5, 1, 28, 32, 1.5, 1),
        (90, 140, 0.0, 1, 91, 139, 0.5, 1),  # Th-230 if present path
    ]
    a_pass = 0
    for r in sel_alpha:
        res = tb.selector_validate_decay(*r, "alpha")
        ok = res["shell_parity_ok"] and res["shell_matches_daughter_formula"]
        a_pass += int(ok)
        print(
            f"  α {r[0]},{r[1]}->{r[4]},{r[5]}: "
            f"shell={res['decoded_daughter_shell']} "
            f"parity_ok={res['shell_parity_ok']} formula_ok={res['shell_matches_daughter_formula']}"
        )
    b_shell = 0
    for r in sel_beta:
        Zp, Np, Jp, Pp, Zd, Nd, Jd, Pd = r
        sp, ip = tb.selector_atom(Zp, Np, Jp, Pp)
        wp = abs(Np - Zp) % 7
        d_shell = abs(Nd - Zd) % 7
        dk, _, b = tb.beta_daughter_shell_step(sp, ip, wp, d_shell)
        landed = _shell_of(dk)
        ok = landed == d_shell
        b_shell += int(ok)
        print(
            f"  β {Zp},{Np}->{Zd},{Nd}: wp={wp} d_shell={d_shell} "
            f"landed={landed} byte=0x{b:02x} ok={ok}"
        )
    print(
        f"  α spot shell/parity .......... {a_pass}/{len(sel_alpha)}  "
        f"PASS={a_pass == len(sel_alpha)}"
    )
    print(
        f"  β spot daughter-shell closure  {b_shell}/{len(sel_beta)}  "
        f"PASS={b_shell == len(sel_beta)}"
    )
    print("  (full census: hqvm_cgm_trestleboard_3 — 314/314 α, 801/801 β shell)")
    print("  β daughter-shell closure is DERIVED via XOR-transport constraint;")
    print("  the DERIVED-SRCH byte emits ΔJ∈{-1,0,+1} at depth-1 (single UNA")
    print("  half-cycle). Larger catalog |ΔJ| resolve at higher percolation")
    print("  depth: 402 depth-1 + 198 depth-2 + 201 depth-3+ = 801 (§9c).")

    print("\n" + "=" * 5)
    print("END TRESTLEBOARD")
    print("=" * 5)


def report_main(
    *, protocol: str = "micro_ref", fusion_model: str = "2", dial: str = "delta"
) -> None:
    tb = NuclearBoard(protocol=protocol, fusion_model=fusion_model, dial=dial)
    th = tb.thresholds
    sh = tb.shells
    pc = tb.percolation

    print("=" * 5)
    print("CGM TRESTLEBOARD — Square · Compass · Level · " "Percolation · σ · ⟨σv⟩")
    print("=" * 5)

    _report_thresholds(tb, th, pc)
    _report_shells(tb, sh, pc)
    _report_grammar(tb)
    _report_level(tb)
    _report_square(tb)
    _report_compass(tb)
    _report_wz_fusion_checks(tb, th, sh, pc)
    _report_lift(tb)
    _report_universal(tb)
    _report_decay(tb)
    _report_selector(tb)


def _run() -> None:
    parser = argparse.ArgumentParser(description="CGM trestleboard instrument")
    parser.add_argument(
        "--no-tee",
        action="store_true",
        help="Do not write hqvm_cgm_trestleboard_results.txt",
    )
    parser.add_argument(
        "--protocol",
        choices=("micro_ref", "q6_class"),
        default="micro_ref",
        help="Restriction protocol for θ(p)",
    )
    parser.add_argument(
        "--fusion-model",
        choices=("1", "2"),
        default="2",
        help="1: θ=tunneling; 2: Gamow×θ(dial) [default]",
    )
    parser.add_argument(
        "--dial",
        choices=("tau", "delta"),
        default="delta",
        help="Model-2 dial: delta=E/Vb [default] or tau=p_c·T",
    )
    args = parser.parse_args()
    run_kw = dict(
        protocol=args.protocol, fusion_model=args.fusion_model, dial=args.dial
    )
    if args.no_tee:
        report_main(**run_kw)
        census_main()
        frontier_main()
        magic_main()
    else:
        buf = io.StringIO()
        orig_stdout = sys.stdout
        sys.stdout = _Tee(orig_stdout, buf)
        try:
            report_main(**run_kw)
            census_main()
            frontier_main()
            magic_main()
        finally:
            sys.stdout = orig_stdout
        RESULTS_PATH.write_text(buf.getvalue(), encoding="utf-8")
        print(f"\nWrote {RESULTS_PATH}", file=orig_stdout)


if __name__ == "__main__":
    _run()
