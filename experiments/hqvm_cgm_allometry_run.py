#!/usr/bin/env python3
"""
hqvm_cgm_allometry_run.py

CLI report: Script 1 (kernel scaling) + Script 2 (catalog fits)
+ Script 3 (fiber-complete alphabet composition).

Writes experiments/hqvm_cgm_allometry_results.txt.
Theory: hqvm_cgm_allometry_notes.md.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

RESULTS_PATH = _EXP / "hqvm_cgm_allometry_results.txt"

from hqvm_cgm_allometry_1 import (
    A_BULK,
    A_EGRESS,
    A_SURFACE,
    N_DOF,
    N_SPATIAL,
    U_ELECTRON_KG,
    U_KG,
    a_d4_delivery_family,
    channel_measures,
    chemical_clock_B0,
    city_company_family,
    coverage_generation_ladder,
    depth4_family_curve,
    kernel_pass_fail,
    kleiber_absolute_intercept,
    micro_ref_proxy_curve,
    p_c_rank_micro_ref,
    parity_plateau_gate,
    qubec_lambda_sweep,
    qubec_uniform_slice,
    scaling_ladder,
    shell_census,
    shell_mean_equals_M_shell_over_H,
    shell_mean_family,
    square_root_live_gate,
    uniqueness_I_family,
    uniqueness_II_family,
    uniqueness_III_family,
    west_organism_family,
)
from hqvm_cgm_allometry_3 import (
    composition_pass,
    cumulative_weight_ladder,
    interface_rule_eval,
    parity_shell_unions,
    product_geometry_identity,
    quotient_census,
    scope_boundary_single_q,
    weight_shell_unions,
)
from hqvm_cgm_trestleboard_common import _Tee


def _print_section(title: str) -> None:
    print(title)
    print("=" * 5)


def _print_union_table(rows) -> None:
    print(
        f"  {'label':>22} {'rA':>3} {'rB':>3} {'rU':>3} "
        f"{'R_A':>6} {'R_B':>6} {'R_U':>6} {'a_U':>8}  ok"
    )
    print("  " + "-" * 5)
    for r in rows:
        ok = (
            r.fiber_U
            and r.srct_U
            and r.rank_mono
            and r.reach_mono
            and r.root_mono
            and (r.r_U < 1 or r.a_is_half)
        )
        a_s = f"{r.a_U:8.6f}" if r.a_U == r.a_U else f"{'nan':>8s}"
        print(
            f"  {r.label:>22} {r.r_A:3d} {r.r_B:3d} {r.r_U:3d} "
            f"{r.reach_A:6d} {r.reach_B:6d} {r.reach_U:6d} {a_s}  "
            f"{'PASS' if ok else 'FAIL'}"
        )


def report() -> int:
    live = square_root_live_gate((4, 5, 6))
    d4 = depth4_family_curve(tuple(range(1, 9)))
    proxies = micro_ref_proxy_curve(N_DOF)
    measures = channel_measures(N_DOF)
    census = shell_census(N_DOF)
    means = shell_mean_family(tuple(range(1, 9)))
    ladder = scaling_ladder(tuple(range(1, 9)))
    parity = parity_plateau_gate(N_DOF)
    organism = west_organism_family()
    city = city_company_family()
    id_shell = shell_mean_equals_M_shell_over_H(N_DOF)
    qb = qubec_uniform_slice(N_DOF)
    delivery = a_d4_delivery_family(tuple(range(1, 9)))
    gens = coverage_generation_ladder()
    qubec_rows = qubec_lambda_sweep(N_DOF)
    kleiber = kleiber_absolute_intercept(N_DOF)
    chem = chemical_clock_B0()
    uniq_i = uniqueness_I_family()
    uniq_ii = uniqueness_II_family()
    uniq_iii = uniqueness_III_family()
    gates = kernel_pass_fail(
        live,
        d4,
        proxies,
        measures,
        means,
        ladder,
        parity,
        census,
        organism,
        city,
        delivery,
        qubec_rows,
        kleiber,
        chem,
    )

    _print_section("1. SQUARE-ROOT LIVE GATE (BFS vs (2^r)^2)")
    print(f"  {'d':>3} {'w':>5} {'r':>3} {'Reach':>8} {'pred':>8} {'root':>6} {'a_B/M':>10}  ok")
    print("  " + "-" * 5)
    for r in live:
        print(
            f"  {r.d:3d} {r.label:>5} {r.rank:3d} {r.reach:8d} {r.pred:8d} "
            f"{r.root:6d} {r.a_B_vs_M:10.6f}  {'PASS' if r.ok else 'FAIL'}"
        )
    print(f"  aggregate {sum(1 for r in live if r.ok)}/{len(live)}")
    print()

    weight_u = weight_shell_unions(N_DOF)
    cum_u = cumulative_weight_ladder(N_DOF)
    parity_u = parity_shell_unions(N_DOF)
    prod = product_geometry_identity()
    q_census = quotient_census(N_DOF)
    iface = interface_rule_eval()
    scope_b = scope_boundary_single_q(N_DOF)
    comp_gates = composition_pass(
        weight_u, cum_u, parity_u, prod, q_census, iface, scope_b
    )

    _print_section("1b. ALPHABET COMPOSITION (fiber-complete union; real hQVM)")
    print("  scope: kernel composition only; no biological unit map")
    print("  SRCT proof: percolation_analysis_4 / §1 live gate (not repeated)")
    print("  structured families: weight shells, even/odd q (same as percolation_4)")
    print("  QuBEC multi-cell GF(2)^{6B}: QuBEC_Theory §22 (not simulated)")
    print()
    print("  weight-shell pairwise unions")
    _print_union_table(weight_u)
    print(f"  n={len(weight_u)}")
    print()
    print("  cumulative weight ladder")
    _print_union_table(cum_u)
    print(f"  n={len(cum_u)}")
    print()
    print("  parity-shell unions")
    _print_union_table(parity_u)
    print(f"  n={len(parity_u)}")
    print()
    print("  scope boundary: single-q fiber (FC, not structured SRCT domain)")
    print(f"  {'label':>10} {'n':>4} {'r':>3} {'reach':>6} {'pred':>6} {'fc':>3}  srct")
    print("  " + "-" * 5)
    for r in scope_b:
        print(
            f"  {r.label:>10} {r.n:4d} {r.rank:3d} {r.reach:6d} {r.pred:6d} "
            f"{'Y' if r.fiber_ok else 'N':>3}  {'Y' if r.srct_ok else 'N'}"
        )
    print()
    print("  product-geometry identity |Ω1|·|Ω2|=(2^{r1+r2})^2")
    print(f"  {'r1':>3} {'r2':>3} {'R1':>8} {'R2':>8} {'R1*R2':>10} {'R(r1+r2)':>10}  ok")
    print("  " + "-" * 5)
    for r in prod:
        print(
            f"  {r.r1:3d} {r.r2:3d} {r.reach1:8d} {r.reach2:8d} "
            f"{r.reach_prod:10d} {r.reach_sum_rank:10d}  "
            f"{'PASS' if r.equal else 'FAIL'}"
        )
    print()
    print(
        f"  quotient census d={q_census.d}  "
        f"bytes={q_census.n_bytes} even_q={q_census.n_even_q_bytes} "
        f"H={q_census.n_horizon} shells={q_census.n_shells}  "
        f"{'PASS' if q_census.chain_ok else 'FAIL'}"
    )
    print("  interface a(D)=D/(D+1)")
    for r in iface:
        print(
            f"  D={r.D} a={r.a:.6f} target={r.target:.6f}  "
            f"{'PASS' if r.ok else 'FAIL'}"
        )
    for name, ok in comp_gates.items():
        print(f"  {name:42s} {'PASS' if ok else 'FAIL'}")
    print()

    _print_section("2. SHELL CENSUS → D_eff=⟨shell⟩ → a_net=D/(D+1)")
    print(f"  d={N_DOF} shell populations")
    print(f"  {'k':>3} {'pop':>8} {'C(d,k)*2^d':>12}  ok")
    print("  " + "-" * 5)
    for r in census:
        print(
            f"  {r.shell:3d} {r.pop:8d} {r.pop_formula:12d}  "
            f"{'PASS' if r.ok else 'FAIL'}"
        )
    print()
    print(f"  {'d':>3} {'Omega':>8} {'⟨shell⟩':>10} {'D_eff':>8} {'a_net':>10}  ent  sum")
    print("  " + "-" * 5)
    for r in means:
        print(
            f"  {r.d:3d} {r.Omega:8d} {r.mean_shell:10.6f} {r.D_eff:8.4f} "
            f"{r.a_net:10.6f}  {'Y' if r.entanglement_ok else 'N'}    "
            f"{'Y' if r.pops_sum_ok else 'N'}"
        )
    print()
    print(
        f"  identity ⟨shell⟩=M_shell/|H|  mean={id_shell['mean_shell']:.6f} "
        f"M_shell={id_shell['M_shell']:.0f} H={id_shell['H']:.0f} "
        f"M/H={id_shell['M_shell_over_H']:.6f}  "
        f"{'PASS' if id_shell['ok'] else 'FAIL'}"
    )
    print(
        f"  QuBEC λ=1  η={qb['eta']:.6f} Z1={qb['Z1']:.0f} M2={qb['M2']:.0f} "
        f"⟨shell⟩={qb['mean_shell']:.6f}  "
        f"{'PASS' if qb['ok_thermal'] else 'FAIL'}"
    )
    print()

    _print_section("3. SCALING LADDER (product / network / depth-4 / surface)")
    print(
        f"  {'d':>3} {'a_SR':>8} {'a_net':>8} {'a_d4':>8} {'a_surf':>8} "
        f"{'net=SR':>6} {'net=2/3':>7} {'net=d4':>6} {'3/4lock':>7}"
    )
    print("  " + "-" * 5)
    for r in ladder:
        print(
            f"  {r.d:3d} {r.a_SR:8.5f} {r.a_net:8.5f} {r.a_d4:8.5f} "
            f"{r.a_surface_Delta:8.5f} "
            f"{str(r.net_eq_SR):>6} {str(r.net_eq_surface):>7} "
            f"{str(r.net_eq_d4):>6} {str(r.triple_lock_3_4):>7}"
        )
    print("  landmarks: d=2→1/2  d=4→2/3  d=6→3/4; net=d4 only at d=6")
    print()

    _print_section("3b. DIMENSION-SIX CONSISTENCY (cross-check; not a derivation of d)")
    print(f"  N_SPATIAL = N_DOF//2 = {N_SPATIAL}")
    print("  Rel I: a_net=a_d4 ⟺ 2^(d-3)=d+2")
    print(f"  {'d':>3} {'a_net':>8} {'a_d4':>8} {'2^(d-3)':>10} {'d+2':>6}  eq")
    print("  " + "-" * 5)
    for r in uniq_i:
        print(
            f"  {r.d:3d} {r.a_net:8.5f} {r.a_d4:8.5f} {r.lhs_pow:10.5f} "
            f"{r.rhs_lin:6.1f}  {'Y' if r.equal else 'N'}"
        )
    print("  Rel II: a_bulk−a_surf = a_time/(d/2) ⟺ (d−6)(d+1)=0")
    print(
        f"  {'d':>3} {'a_bulk':>8} {'a_surf':>8} {'a_time':>8} {'n=d/2':>7} "
        f"{'lhs':>8} {'rhs':>8} {'resid':>10}  eq"
    )
    print("  " + "-" * 5)
    for r in uniq_ii:
        print(
            f"  {r.d:3d} {r.a_bulk:8.5f} {r.a_surf:8.5f} {r.a_time:8.5f} "
            f"{r.n:7.2f} {r.lhs:8.5f} {r.rhs:8.5f} {r.resid:10.2e}  "
            f"{'Y' if r.equal else 'N'}"
        )
    print("  Rel III: 1−a_bulk = a_bulk−a_SR ⟺ d=6")
    print(
        f"  {'d':>3} {'a_bulk':>8} {'a_time_V':>9} {'a_time_L':>9} "
        f"{'resid':>10}  eq"
    )
    print("  " + "-" * 5)
    for r in uniq_iii:
        print(
            f"  {r.d:3d} {r.a_bulk:8.5f} {r.a_time_V:9.5f} {r.a_time_L:9.5f} "
            f"{r.resid:10.2e}  {'Y' if r.equal else 'N'}"
        )
    print()

    _print_section("4. DEPTH-4 FAMILY a_d4(d)=depth4_bits/2^d")
    print(f"  {'d':>3} {'H':>5} {'bits':>5} {'a_d4':>10} {'Delta':>12} {'=3/4':>5}")
    print("  " + "-" * 5)
    for r in d4:
        print(
            f"  {r.d:3d} {r.H:5d} {r.depth4_bits:5d} {r.a_d4:10.6f} "
            f"{r.Delta:12.8f} {str(r.equals_3_4):>5}"
        )
    print()

    _print_section("5. CHANNEL MEASURES + PARITY PLATEAU")
    print(f"  p_c_rank(d={N_DOF}) = {p_c_rank_micro_ref(N_DOF):.12g}")
    print(f"  {'id':32s} {'value':>14s}  note")
    print("  " + "-" * 5)
    for m in measures:
        print(f"  {m.id:32s} {m.value:14.12g}  {m.note}")
    print()
    print(
        f"  parity_even_q  d={parity.d} r={parity.rank} Reach={parity.reach} "
        f"pred={parity.pred} a={parity.a_B_vs_M:.6f}  "
        f"{'PASS' if parity.ok else 'FAIL'}  {parity.note}"
    )
    print()

    _print_section("6. MICRO-REF PROXY CURVE (exact PMF)")
    print(
        f"  {'p':>6} {'E_root':>10} {'E_Reach':>10} {'a_SR':>10} "
        f"{'theta':>10} {'P_full':>10} {'hol4':>10}"
    )
    print("  " + "-" * 5)
    show = [
        r
        for r in proxies
        if abs(40 * r.p - round(40 * r.p)) < 1e-9 and int(round(40 * r.p)) % 2 == 0
    ]
    for r in show:
        print(
            f"  {r.p:6.3f} {r.E_root:10.4f} {r.E_reach:10.1f} {r.a_SR:10.6f} "
            f"{r.theta:10.6f} {r.P_rank_full:10.6f} {r.holonomy_d4:10.6f}"
        )
    print()

    _print_section("7. A_DELIV DELIVERY (aperture×horizon; Horizon Lemma)")
    print(
        f"  {'d':>3} {'bits':>5} {'H':>5} {'Delta':>10} {'a_deliv':>8} "
        f"{'1/(ΔH)':>8} {'P_pred':>6} {'P/H':>8} {'=3/4':>5} ok"
    )
    print("  " + "-" * 5)
    for r in delivery:
        print(
            f"  {r.d:3d} {r.depth4_bits:5d} {r.H:5d} {r.Delta_d:10.8f} "
            f"{r.a_deliv:8.5f} {r.a_deliv_from_Delta_H:8.5f} {r.P_pred:6d} "
            f"{r.a_horizon_lemma:8.5f} {str(r.equals_3_4):>5} "
            f"{'PASS' if r.ok else 'FAIL'}"
        )
    print("  coverage generations (discrete branching depth)")
    for name, gen, note in gens:
        print(f"  gen={gen}  {name:10s}  {note}")
    print()

    _print_section("8. QUBEC λ SWEEP (physical λ∈(0,1]; a(λ)=2/3+λ/(6(1+λ)))")
    print(
        f"  {'λ':>8} {'η':>8} {'⟨N⟩':>7} {'M2':>8} {'a_net':>8} {'a_fold':>8} "
        f"{'μ':>6} {'a(μ)':>8} {'regime':12s} z1"
    )
    print("  " + "-" * 5)
    for r in qubec_rows:
        print(
            f"  {r.lam:8.3g} {r.eta:8.4f} {r.mean_N:7.4f} {r.M2:8.1f} "
            f"{r.a_net:8.5f} {r.a_fold:8.5f} {r.mu_eta:6.3f} {r.a_mu_eta:8.5f} "
            f"{r.regime:12s} {'Y' if r.z1_ok else 'N'}"
        )
    print()

    _print_section("9. KLEIBER ABSOLUTE INTERCEPT (M0)")
    print(f"  M_shell           {kleiber.M_shell:.0f}")
    print(f"  a_Higgs=M_shell/2 {kleiber.a_Higgs:.0f}")
    print(f"  log2(M0/u)        {kleiber.log2_M0_over_u:.0f}")
    print(f"  xi=2^a_Higgs      {kleiber.xi:.6e}")
    print(f"  M0_kg (u=1 amu)   {kleiber.M0_kg:.6e}")
    print(f"  M0_kg (u=m_e)     {kleiber.M0_kg_electron_u:.6e}")
    print(f"  u_amu_kg          {U_KG:.6e}")
    print(f"  u_electron_kg     {U_ELECTRON_KG:.6e}")
    print(f"  b_K=−P            {kleiber.b_K:.12g}")
    print(f"  a_bulk            {kleiber.a_bulk:.6f}")
    print(f"  note              {kleiber.note}")
    print()

    _print_section("10. WEST ORGANISM FAMILY (corollaries of a_SR, a_bulk, a_time)")
    print(f"  {'id':28s} {'a':>10s} {'formula':18s}  note")
    print("  " + "-" * 5)
    for r in organism:
        print(f"  {r.id:28s} {r.value:10.6f} {r.formula:18s}  {r.note}")
    print()

    _print_section("11. CITY / COMPANY Tier C ((d−1)/d and 1+1/d; not kernel-forced)")
    print(f"  {'id':28s} {'a':>10s} {'formula':18s}  note")
    print("  " + "-" * 5)
    for r in city:
        print(f"  {r.id:28s} {r.value:10.6f} {r.formula:18s}  {r.note}")
    print()

    _print_section("12. CHEMICAL CLOCK (E_a=kT/(2Δ); MTE band)")
    print(f"  T_body_K          {chem.T_body_K:.1f}")
    print(f"  Delta             {chem.Delta:.8f}")
    print(f"  kT_eV             {chem.kT_eV:.6f}")
    print(f"  E_a_eV            {chem.E_a_eV:.6f}")
    print(f"  f_attempt_Hz      {chem.f_attempt_Hz:.6e}")
    print(f"  N_H               {chem.N_H}")
    print(f"  P_terminal_W      {chem.P_terminal_W:.6e}")
    print(f"  B0_micro_W        {chem.B0_micro_W:.6e}")
    print(f"  note              {chem.note}")
    print()

    _print_section("13. KERNEL PASS/FAIL")
    all_ok = True
    for name, ok in gates.items():
        print(f"  {name:36s} {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    for name, ok in comp_gates.items():
        print(f"  {name:36s} {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    print()
    print(f"  results_path  {RESULTS_PATH}")
    print()
    return 0 if all_ok else 1


def report_data() -> int:
    from hqvm_cgm_allometry_2 import (
        ALLOMETRY_CATALOG,
        activity_regime_audit,
        activity_regime_pass,
        ci_contains,
        default_catalog_suite,
        default_synthetic_suite,
        info_conjugacy_audit,
        kleiber_m0_audit,
        run_data_battery,
        run_model_battery,
        run_null_battery,
        tick_residual,
    )

    code = 0
    _print_section("D0. SYNTHETIC SELF-CHECK")
    series = default_synthetic_suite()
    fits, _comps = run_data_battery(series, n_boot=200)
    print(
        f"  {'name':22s} {'n':>4s} {'a_pri':>9s} {'est':>4s} {'mu':>8s} "
        f"{'null':16s} status"
    )
    print("  " + "-" * 5)
    for f in fits:
        mu_s = f"{f.mu_primary:8.4f}" if f.mu_in_band else f"{f.mu_primary:8.4f}*"
        print(
            f"  {f.name:22s} {f.n:4d} {f.a_primary:9.6f} {f.estimator:>4s} "
            f"{mu_s:>9s} {f.nearest_null:16s} {f.status}"
        )
    print()
    clean = [f for f in fits if "noisy" not in f.name]
    clean_ok = all(f.status in ("EXACT", "NEAR") for f in clean)
    noisy = [f for f in fits if "noisy" in f.name]
    noisy_ok = (
        all(f.status in ("EXACT", "NEAR", "SCAN") for f in noisy) if noisy else True
    )
    print(f"  synthetic_clean_recover  {'PASS' if clean_ok else 'FAIL'}")
    print(f"  synthetic_noisy_accounted {'PASS' if noisy_ok else 'FAIL'}")
    print()
    if not (clean_ok and noisy_ok):
        code = 1

    _print_section("D0b. SYNTHETIC MODEL COMPARE (AIC)")
    for row in run_model_battery(series):
        print(
            f"  {row.series:22s} {row.model:12s} a={row.a:9.6f} "
            f"AIC={row.aic:.3f} dAIC={row.delta_aic:.3f}"
        )
    print()

    _print_section("D0c. SYNTHETIC NULL SHUFFLE")
    nulls = run_null_battery(series, n_perm=100)
    for nrow in nulls:
        print(
            f"  {nrow.series:22s} real={nrow.real_status:5s} "
            f"frac_NEAR/EXACT={nrow.frac_near_or_exact:.3f} "
            f"{'PASS' if nrow.pass_null else 'FAIL'}"
        )
    null_ok = all(n.pass_null for n in nulls)
    print(f"  synthetic_null_audit  {'PASS' if null_ok else 'FAIL'}")
    print()
    if not null_ok:
        code = 1

    _print_section("D1. CATALOG FITS (OLS primary for organism; RMA for city)")
    try:
        series = default_catalog_suite()
    except FileNotFoundError as e:
        print(f"  FAIL  {e}")
        return 1
    fits, comps = run_data_battery(series, n_boot=400)
    print(
        f"  {'name':32s} {'n':>5s} {'est':>4s} {'a_pri':>9s} {'OLS':>9s} "
        f"{'RMA':>9s} {'lo':>9s} {'hi':>9s} status"
    )
    print("  " + "-" * 5)
    for s, f, c in zip(series, fits, comps):
        print(
            f"  {f.name:32s} {f.n:5d} {f.estimator:>4s} {f.a_primary:9.6f} "
            f"{f.a_OLS:9.6f} {f.a_RMA:9.6f} "
            f"{f.a_lo:9.6f} {f.a_hi:9.6f} {f.status}"
        )
        has23 = ci_contains(f.a_lo, f.a_hi, A_SURFACE)
        has34 = ci_contains(f.a_lo, f.a_hi, A_BULK)
        print(
            f"    nearest={f.nearest_null}  CI_2/3={has23}  CI_3/4={has34}  "
            f"tick23={tick_residual(f.a_primary, A_SURFACE):+.3f}  "
            f"tick34={tick_residual(f.a_primary, A_BULK):+.3f}"
        )
        print(f"    source={s.source}")
    print()

    _print_section("D1b. CATALOG MODEL COMPARE (AIC)")
    for row in run_model_battery(series):
        print(
            f"  {row.series:32s} {row.model:12s} a={row.a:9.6f} "
            f"dAIC={row.delta_aic:.3f}"
        )
    print()

    _print_section("D1c. CATALOG NULL SHUFFLE")
    nulls = run_null_battery(series, n_perm=200)
    for nrow in nulls:
        print(
            f"  {nrow.series:32s} real={nrow.real_status:5s} "
            f"{'PASS' if nrow.pass_null else 'FAIL'}"
        )
    null_ok = all(n.pass_null for n in nulls)
    print(f"  catalog_null_audit  {'PASS' if null_ok else 'FAIL'}")
    print()
    if not null_ok:
        code = 1

    _print_section("D3. KLEIBER M0 INTERCEPT AUDIT")
    metabolic_names = (
        "pantheria_bmr",
        "anage_metabolic",
        "animaltraits_metabolic_Mammalia",
    )
    by_spec = {s.name: s for s in series}
    m0_rows = []
    for name in metabolic_names:
        sp = by_spec.get(name)
        if sp is None:
            continue
        row = kleiber_m0_audit(sp)
        m0_rows.append(row)
        print(
            f"  {row.series:32s} n={row.n:4d} K={row.K:.6g} "
            f"B0_emp={row.B0_emp:.6g} M_gmean/M0={row.M_over_M0:.4f} "
            f"resid_std={row.resid_std_log10:.4f}"
        )
        print(
            f"    c_M0={row.c_M0:.6f}  log2_B0_emp={row.log2_B0_emp:.6f}  "
            f"log10_K={row.log10_K:.6f}"
        )
    if len(m0_rows) >= 2:
        logs = [r.log2_B0_emp for r in m0_rows]
        spread = max(logs) - min(logs)
        print(f"  log2_B0_spread_octaves  {spread:.4f}")
    print()

    _print_section("D4. METABOLIC SERIES vs μ")
    act = activity_regime_audit(fits)
    print(
        f"  {'series':32s} {'a_pri':>9s} {'μ':>7s} {'regime':12s} "
        f"{'a_hyp':>9s} {'resid_hyp':>9s} nearer"
    )
    print("  " + "-" * 5)
    for r in act:
        a_h = f"{r.a_hyp:9.6f}" if r.a_hyp == r.a_hyp else f"{'nan':>9s}"
        rh = f"{r.resid_hyp:9.6f}" if r.resid_hyp == r.resid_hyp else f"{'nan':>9s}"
        print(
            f"  {r.series:32s} {r.a_primary:9.6f} {r.mu_primary:7.4f} "
            f"{r.regime_hyp:12s} {a_h} {rh} "
            f"{'Y' if r.nearer_hyp_than_alt else 'N'}"
        )
    print(
        f"  {'development_time_3_16':32s} {A_EGRESS:9.6f} {'—':>7s} "
        f"{'fixed':12s} {A_EGRESS:9.6f} {0.0:9.6f} Y"
    )
    act_gates = activity_regime_pass(act)
    for name, ok in act_gates.items():
        print(f"  {name:42s} {'PASS' if ok else 'FAIL'}")
        if not ok:
            code = 1
    print()

    _print_section("D5. INFO CONJUGACY / SUM RULES")
    for row in info_conjugacy_audit(fits):
        ar = f"{row.a_right:9.6f}" if row.a_right == row.a_right else f"{'—':>9s}"
        print(
            f"  {row.id:28s} aL={row.a_left:9.6f} aR={ar} "
            f"sum={row.a_sum:9.6f} tgt={row.target:9.6f} "
            f"resid={row.resid:9.6f} {row.status}"
        )
    print()

    _print_section("D6. CATALOG")
    print(f"  n_series={len(fits)}  catalog_dir={ALLOMETRY_CATALOG}")
    print()
    return code


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CGM allometry report (compute + data)")
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args(argv)

    def _go() -> int:
        code = report()
        code2 = report_data()
        return code if code else code2

    if args.no_write:
        return _go()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = _Tee(old, buf)  # type: ignore[assignment]
    try:
        code = _go()
    finally:
        sys.stdout = old
    RESULTS_PATH.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
