#!/usr/bin/env python3
"""
hqvm_cgm_octaves_run.py

CLI report for the CGM-hQVM octaves program (scripts 1-3).

Writes experiments/hqvm_cgm_octaves_results.txt.
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

from hqvm_cgm_octaves_common import (
    C6,
    CHIRALITY_SPACE,
    DELTA,
    DELTA_CONT,
    DELTA_DEPTH4,
    DELTA_DYADIC_8,
    OCTAVE_CONSTANTS,
    RESULTS_PATH,
    TICKS_PER_OCTAVE,
    Tee,
    fmt,
)
from hqvm_cgm_octaves_1 import run_octaves_1
from hqvm_cgm_octaves_2 import run_octaves_2
from hqvm_cgm_octaves_3 import run_octaves_3


def _section(title: str) -> None:
    print(title)
    print("=" * 5)


def _gate_line(name: str, ok: bool) -> None:
    print(f"  {name:48s} {'PASS' if ok else 'FAIL'}")


def _report_1(c1) -> None:
    _section("0. KERNEL GATES")
    for name, ok in c1.gates:
        _gate_line(name, ok)
    print(f"  APERTURE_GAP=Delta_cont={DELTA_CONT}")
    print(f"  Delta_depth4={DELTA_DEPTH4} Delta_dyadic_8={DELTA_DYADIC_8}")
    print(f"  ticks_per_octave={TICKS_PER_OCTAVE:.6f}")
    print(f"  chirality_space={CHIRALITY_SPACE}")
    print(f"  C6={C6}")
    print()

    _section("0b. FROZEN CONSTANTS")
    for rec in OCTAVE_CONSTANTS:
        print(f"  {rec.name:20s} {rec.value:.12g}  [{rec.derivation}]  {rec.source}")
    print()

    _section("0c. OCTAVE RESOLUTIONS / APERTURE RESIDUE")
    ar = c1.aperture_residues
    print(f"  Delta_dyadic_8={ar['Delta_dyadic_8']}")
    print(f"  Delta_cont={ar['Delta_cont']}")
    print(f"  Delta_depth4={ar['Delta_depth4']}")
    print(f"  ordering_5/256 < Delta_cont < 1/48: {ar['ordering_dyadic_lt_cont_lt_depth4']}")
    print(f"  ticks_per_octave_cont={ar['ticks_per_octave_cont']}")
    print(f"  epsilon_oct = 1/Delta - 48 = {ar['epsilon_oct_ticks']}")
    print(f"  48*Delta={ar['forty_eight_Delta']}  1-48*Delta={ar['one_minus_48_Delta']}")
    print(f"  log2(Q_G/(2pi))={ar['log2_Q_G_over_2pi']}  (exact one octave)")
    print(f"  chirality_space=(1/48)/(1/32)={ar['chirality_space_2_3']}")
    print()

    _section("0d. WAVEFUNCTION KERNEL -> OCTAVE PRIMITIVES")
    wf = c1.wavefunction_octave_primitives
    print(f"  source={wf['source']}")
    print(f"  |Omega|=|H|^2: {wf['carrier_is_horizon_squared']}  log2(|Omega|/|H|)={wf['log2_Omega_over_H']}")
    print(f"  half_frame_phases={wf['half_frame_phases']} full_frame_bits={wf['full_frame_bits']}")
    print(f"  flat_bytes={wf['flat_bytes']} curved={wf['curved_bytes']} fold_hist={wf['fold_disagreement_hist']}")
    print(f"  local_dual_aperture={wf['local_dual_reading_aperture']} global_Delta={wf['global_Delta_cont']}")
    print(f"  compression_50_to_Delta={fmt(float(wf['compression_ratio_50_to_Delta']),4)}")
    print("  holographic levels (|Space|=|Subspace|^2):")
    for h in wf["holographic_levels"]:
        print(
            f"    {h['name']:10s} dof={h['dof']} sub={h['subspace']} "
            f"space={h['space']} dim={h['dimension']} log2(space/sub)={h['log2_space_over_subspace']}"
        )
    wp = wf["word_periods"]
    print(
        f"  word_periods: W2_len={wp['W2_len']} Wfull_len={wp['Wfull_len']} F2_len={wp['F2_len']} "
        f"W2^2=id={wp['W2_sq_is_id_rest']} F2=id={wp['F2_is_id_rest']}"
    )
    print(f"  k4_w2_import_pass={wf['k4_w2_all_pass']}")
    print("  predecessor ladder P_k=3*2^(k-1):")
    for r in c1.predecessor_ladder:
        print(
            f"    k={r['k']} P={r['P_k']} dyadic={r['is_dyadic']} is_48={r['is_48']}"
        )
    print()

    _section("1. DYADIC ATLAS")
    print(f"  {'name':22s} {'kind':8s} {'size':>8s} {'log2':>10s} {'role'}")
    for n in c1.atlas:
        lg = f"{n.log2_card:10.6f}" if n.log2_card == n.log2_card else f"{'nan':>10s}"
        print(f"  {n.name:22s} {n.kind:8s} {n.cardinality:8d} {lg} {n.role}")
    print()
    print("  doubling edges (same kind, size -> 2*size)")
    for e in c1.doubling_edges:
        same = "same_role" if e["same_role"] else "cross_role"
        print(
            f"  [{e['kind']}] {e['from']:22s} -> {e['to']:22s}  "
            f"{e['from_card']}->{e['to_card']}  {same}"
        )
    print(f"  n_edges={len(c1.doubling_edges)}")
    print()

    _section("2. WORD PROBES")
    print(
        f"  {'name':14s} {'L':>2s} {'par':>3s} {'tu':>3s} {'tv':>3s} "
        f"{'shell':>5s} {'id':>3s} bytes"
    )
    for r in c1.word_probes:
        print(
            f"  {r['name']:14s} {r['length']:2d} {r['parity']:3d} "
            f"{r['tau_u6']:3d} {r['tau_v6']:3d} {r['shell_tau']:5d} "
            f"{int(r['is_identity']):3d} {r['bytes_hex']}"
        )
    print()

    _section("3. SIGNATURE MONOID HOMOMORPHISM")
    print(f"  random summary: {c1.random_defect_summary}")
    print(
        f"  {'name':16s} {'fam':9s} {'L':>2s} {'ham_lift':>8s} "
        f"{'ham_id':>6s} {'compose'}"
    )
    for r in c1.doubling_defects:
        if r["family"] != "canonical":
            continue
        print(
            f"  {r['name']:16s} {r['family']:9s} {r['length']:2d} "
            f"{r['hamming_to_lift']:8d} {r['hamming_to_id']:6d} "
            f"{r['compose_exact']}"
        )
    n_rand_exact = sum(
        1 for r in c1.doubling_defects if r["family"] == "random" and r["compose_exact"]
    )
    n_rand = sum(1 for r in c1.doubling_defects if r["family"] == "random")
    print(f"  random compose_exact: {n_rand_exact}/{n_rand}")
    print()

    _section("3b. OMEGA12 WORD vs SIGNATURE DISAGREE")
    print(f"  {'name':16s} {'L':>2s} {'disagree':>8s} {'agree_frac':>10s} exact")
    for r in c1.omega_sig_disagree:
        print(
            f"  {r['name']:16s} {r['length']:2d} {r['disagree']:8d} "
            f"{fmt(float(r['agree_frac']), 6):>10s} {r['exact']}"
        )
    print()

    _section("4. PROJECTION EQUIVOCATION H(X|Y)")
    for r in c1.projection_entropy:
        extra = ""
        if "fiber_sizes" in r:
            extra = f" sizes={r['fiber_sizes']}"
        if "n_histories" in r:
            extra += f" n_hist={r['n_histories']}"
        print(
            f"  {r['chart']:36s} H={fmt(float(r['H_proj_bits']), 4)}  "
            f"fib=[{r['fiber_min']},{r['fiber_max']}]  {r['status']}{extra}"
        )
    s = c1.one_step_shadow
    print(
        f"  {'byte->state24_next(rest)':36s} "
        f"n_unique={s['n_unique_next']} fib=[{s['fiber_min']},{s['fiber_max']}] "
        f"H(byte|next)={fmt(float(s['H_byte_given_next']),4)} "
        f"fib2={s['all_fibres_size_2']} {s['status']}"
    )
    print()

    _section("5. COMMA vs APERTURE (HYPOTHESIS-GENERATING)")
    print(
        f"  {'comma':8s} {'aperture':18s} {'d_oct':>10s} {'d_cents':>10s} {'d_ticks':>10s}"
    )
    for r in c1.comma_rows:
        print(
            f"  {r['comma']:8s} {r['aperture']:18s} "
            f"{fmt(r['abs_diff_octaves'], 6):>10s} "
            f"{fmt(r['abs_diff_cents'], 4):>10s} "
            f"{fmt(r['abs_diff_ticks'], 4):>10s}"
        )
    print("  best per comma / dyadic certificate:")
    for r in c1.comma_best:
        if r.get("comma") == "Delta_cont_best_dyadic_256":
            print(
                f"  Delta_cont best k/256: k={r['best_k']} "
                f"approx={r['best_approx']} err={fmt(float(r['abs_err']), 8)} "
                f"is_5={r['is_k_equals_5']} {r['status']}"
            )
        else:
            print(
                f"  {r['comma']:8s} -> {r['nearest_aperture']:18s} "
                f"d_oct={fmt(float(r['abs_diff_octaves']), 6)} "
                f"reject_hol={r['sanity_rejected_holonomy']} {r['status']}"
            )
    print()

    _section("6. SHELL PAIR-RATIOS vs JUST INTERVALS")
    print(
        f"  {'tag':12s} {'ratio':8s} {'cents':>10s} {'nearest':22s} {'res_cents':>10s}"
    )
    for r in c1.shell_ratio_hits:
        tag = f"C{r['i']}/C{r['j']}" if int(r["j"]) >= 0 else str(r.get("note", r["ratio"]))
        print(
            f"  {tag:12s} {str(r['ratio']):8s} "
            f"{fmt(float(r['cents']), 4):>10s} {str(r['nearest']):22s} "
            f"{fmt(float(r['residual_cents']), 4):>10s}"
        )
    print()

    _section("7. BYTE PALINDROME / NEWLANDS")
    p = c1.palindrome
    print(f"  frame={p['frame']}")
    print(f"  fold_zero_bytes={p['fold_zero_bytes']}")
    print(f"  consonant_even_fold0={p['consonant_even_fold0']}")
    print(f"  dissonant_odd_fold_gt0={p['dissonant_odd_fold_gt0']}")
    print(f"  ratio={fmt(float(p['ratio_consonant_dissonant']), 4)}")
    print(f"  newlands_CS_hits={p['newlands_CS_hits']}/{p['newlands_n']}")
    for h in p["newlands_nobles"]:
        print(
            f"  Z={h['Z']:3d} pos={h['pos_mod8']} stage={h['stage']} on_CS={h['on_CS']}"
        )
    print()

    _section("8. FIFTHS / 12-vs-19 / 2/3")
    f = c1.fifths_fingerprint
    t = c1.twelve_vs_nineteen
    print(f"  log2(3/2)={fmt(float(f['log2_fifth']), 8)}")
    print(f"  12*fifth-7={fmt(float(f['12_fifths_minus_7_oct']), 8)}")
    print(f"  PC={fmt(float(f['PC_log2']), 8)} abs_err={fmt(float(f['abs_err']), 12)}")
    print(f"  fifth_ticks={fmt(float(f['log2_fifth_over_Delta_ticks']), 4)}")
    print(f"  chirality_space={f['chirality_space']} status={f['status']}")
    print(f"  12*log2(3)-19={fmt(float(t['12_log2_3_minus_19']), 8)}")
    print(f"  12*log2(3)-18={fmt(float(t['12_log2_3_minus_18']), 8)}")
    print(f"  |PC-Delta_cont|={fmt(float(t['abs_PC_minus_Delta']), 8)}")
    print(f"  |PC-Delta_dyadic_8|={fmt(float(t['abs_PC_minus_Delta_dyadic_8']), 8)}")
    print(f"  |PC-Delta_depth4|={fmt(float(t['abs_PC_minus_Delta_depth4']), 8)}")
    print(f"  status={t['status']}")
    print()


def _report_2(c2) -> None:
    _section("10. WALSH DYADIC BANDS")
    print(f"  default_perm sizes(bands0..3)={c2.walsh_band_sizes['default_perm']}")
    print(f"  unique size tuples under perms={c2.walsh_band_sizes.get('perm_size_unique')}")
    print("  energy flow (isotropic_bits, selected ensembles t0/t1/t2):")
    for r in c2.walsh_energy_rows:
        if r["transport"] != "isotropic_bits":
            continue
        if r["ensemble"] not in ("uniform", "equator", "independent_bit"):
            continue
        print(
            f"  {r['ensemble']:16s} {r['t']:3s}  "
            f"E=({fmt(float(r['E0']),4)},{fmt(float(r['E1']),4)},"
            f"{fmt(float(r['E2']),4)},{fmt(float(r['E3']),4)})"
        )
    print()

    _section("10b. WALSH–KRAWTCHOUK phi(u)=1-2 wt(u)/6")
    for r in c2.walsh_phi_audit:
        if r.get("check") == "phi_closed_form_vs_WHT":
            print(f"  phi_closed_form_vs_WHT match={r['match']} {r['status']}")
        elif r.get("check") == "damping":
            print(
                f"  {r['ensemble']:16s} t={r['t']} max_err={float(r['max_abs_err']):.3e} "
                f"exact={r['exact']} {r['status']}"
            )
    print()

    _section("10c. WAVE DISPERSION (phi_r=1-r/3)")
    print(f"  {'r':>2s} {'phi':>8s} {'lam':>8s} {'w_cont':>8s} {'w_disc':>8s}")
    for r in c2.wave_dispersion:
        print(
            f"  {r['r']:2d} {fmt(float(r['phi_r']),4):>8s} {fmt(float(r['lambda_r']),4):>8s} "
            f"{fmt(float(r['omega_cont_c1']),4):>8s} {fmt(float(r['omega_disc']),4):>8s}"
        )
    vel = c2.radial_velocity
    print(f"  damping-channel vp=vg: {vel['all_vp_eq_vg']} eta={vel['eta_probe']}")
    print()

    _section("10d. PROJECTED chi COMMUTATOR (identity on GF(2)^6)")
    cc = c2.chi_commutator
    print(f"  projected_identity={cc['projected_commutator_is_identity']}")
    print(f"  chi_algebraic_bad={cc['chi_algebraic_bad']} chi_rest_fail={cc['chi_rest_path_fail']}")
    print(f"  omega_rest_moved_frac={fmt(float(cc['omega_rest_moved_frac']),4)}")
    print(f"  byte_orders={cc['byte_order_census']}")
    print(f"  note={cc['d_q_note']}")
    print(f"  mask_code={cc['mask_code_note']}")
    print()

    _section("10e. K4 INTERFERENCE MAP (abelian holonomy)")
    for r in c2.k4_interference:
        if r["gate"] == "note":
            print(f"  note={r['interference']}")
        else:
            print(f"  {r['gate']:4s} {r['interference']}")
    print()

    _section("11. KRAWTCHOUK / COMPATIBILITY")
    for r in c2.krawtchouk_rows:
        if r["mode_r"] == -1:
            print(f"  corr_l1_vs_harmonic={fmt(float(r['corr_l1_vs_harmonic']), 4)}")
        else:
            print(f"  r={r['mode_r']} l1={r['l1']} row={r['row']}")
    print("  Walsh-band x shell mass (>0.05):")
    for r in c2.compatibility:
        if float(r["mass"]) > 0.05:
            print(f"  band={r['band']} shell={r['shell']} mass={fmt(float(r['mass']),4)}")
    print()

    _section("11c. STANDING-WAVE SHELL AMPLITUDES")
    sw = c2.standing_wave
    print(f"  {'k':>2s} {'C6':>4s} {'pop':>6s} {'pi_k':>8s} {'cos2':>8s} node/anti")
    for r in sw["rows"]:
        tag = "node" if r["is_horizon_node"] else ("anti" if r["is_equator_antinode"] else "")
        print(
            f"  {r['k']:2d} {r['C6']:4d} {r['pop']:6d} {fmt(float(r['pi_k']),4):>8s} "
            f"{fmt(float(r['cos2_envelope']),4):>8s} {tag}"
        )
    print(f"  corr_pop_vs_cos2={fmt(float(sw['corr_pop_vs_cos2']),4)}")
    print(f"  bulk_sum={sw['bulk_sum']} bulk_is_3968={sw['bulk_is_3968']}")
    print()

    _section("12. QUBEC OCTAVE RG DEFECT (pi K vs K pi)")
    print(f"  {'ensemble':16s} {'lvl':>3s} {'TV':>10s} {'KL':>10s} exact")
    for r in c2.rg_rows:
        print(
            f"  {r['ensemble']:16s} {r['level']:3d} "
            f"{fmt(float(r['TV']),6):>10s} {fmt(float(r['KL']),6):>10s} "
            f"{r['exact_zero']}"
        )
    print()

    _section("13. PERCOLATION DYADIC")
    board = c2.percolation["board"]
    for k, v in board.items():
        print(f"  {k}={v}")
    print("  thresholds:")
    for r in c2.percolation["thresholds"]:
        print(
            f"  {r['name']:14s} p={fmt(float(r['p']),6)} z={fmt(float(r['z_p']),4)} "
            f"oct_from_Delta={fmt(float(r['octaves_from_Delta']),4)} "
            f"near={r['nearest_ref']} d_oct={fmt(float(r['nearest_ref_oct']),4)}"
        )
    print("  pairwise log2(pi/pj) vs {0,±1,±1/2}:")
    for r in c2.dyadic_pairs:
        print(
            f"  {r['i']:12s}/{r['j']:12s}  "
            f"log2={fmt(float(r['log2_pi_pj']),4)} "
            f"tgt={r['nearest_target']} err={fmt(float(r['abs_err']),4)} "
            f"hit={r['hit_tol_0.05']}"
        )
    print()

    _section("13b. SQUARE-ROOT STRUCTURED SHELL AUDIT")
    print(f"  {'case':22s} {'r':>3s} {'n_b':>4s} {'reach':>6s} {'pred':>6s} fc match")
    for r in c2.square_root_audit:
        if r["case"] == "ranks_covered":
            print(
                f"  ranks_seen={r['ranks_seen']} covers_0_1_5_6={r['covers_0_1_5_6']} "
                f"plateau_r5_1024={r['plateau_r5_1024']} all_match={r['all_structured_match']}"
            )
            continue
        print(
            f"  {r['case']:22s} {r['r']:3d} {r['n_bytes']:4d} "
            f"{r['n_reach']:6d} {r['predicted']:6d} {r['fiber_complete']} {r['match']}"
        )
    print()

    _section("14. DYADIC SCALE RESPONSE R_2(p)=theta(min(2p,1))-theta(p)")
    for r in c2.octave_response:
        print(
            f"  p={fmt(float(r['p']),4)} th={fmt(float(r['theta_p']),4)} "
            f"th2={fmt(float(r['theta_2p']),4)} "
            f"R_double={fmt(float(r['R_double']),4)} "
            f"R_half={fmt(float(r['R_half']),4)}"
        )
    print()

    _section("15. WORD COMPLETION LAW")
    print(f"  {'L':>2s} {'p':>8s} {'P_word':>10s} {'P_L4':>10s} {'ratio':>8s}")
    for r in c2.word_law:
        if r["L"] not in (1, 2, 4, 8):
            continue
        if abs(float(r["p"]) - float(board["p_c_rank"])) > 1e-12 and float(r["p"]) not in (
            0.1,
            0.3,
            0.5,
        ):
            continue
        print(
            f"  {r['L']:2d} {fmt(float(r['p']),4):>8s} "
            f"{fmt(float(r['P_word']),6):>10s} "
            f"{fmt(float(r['P_word_L4']),6):>10s} "
            f"{fmt(float(r['ratio_to_L4']),4):>8s}"
        )
    print()

    _section("16. EVEN-INDEX ATLAS E32 + DYADIC SPINE O5")
    eh = c2.even_harmonics
    print(f"  note={eh.get('atlas_note','')}")
    print(f"  |E32|={eh['n_H']} O5={eh['D']} nondyadic={eh['n_nondyadic']} 48_in_E32={eh['predecessor_48_in_H']}")
    print(f"  span_octaves_2_to_64={eh['span_octaves_2_to_64']}")
    print(f"  bulk_sum={eh['bulk_sum']} bulk_is_3968={eh.get('bulk_is_3968')}")
    print(f"  node: even_midpoint={eh['even_modes_node_at_midpoint']} odd_antinode={eh['odd_modes_antinode_at_midpoint']}")
    print(f"  fold: flat={eh['fold']['flat']} curved={eh['fold']['curved']}")
    print(f"  weight: even_q={eh['even_q_count']} A_even={eh['A_even_n']} A_odd={eh['A_odd_n']}")
    print(f"  reach_even_formula={eh['reach_even_formula']} bfs={eh['reach_even_bfs']}")
    print(f"  cover: shadow={eh['byte_shadow_128']} hologram={eh['hologram_4096']}")
    print(f"  shells: bulk_sum={eh['bulk_sum']} horizon_sum={eh['horizon_sum']} rho5={fmt(float(eh['rho5']),6)}")
    print(f"  F_two_cycles={eh['F_two_cycles']} cycle_index_n={eh['cycle_index_n']} all_f_xor_eps={eh['all_f_xor_epsilon']}")
    print("  cycle_index head (harmonic_n, even_q6, f_cycle_lo/hi):")
    for r in eh["cycle_index_head"]:
        print(
            f"    n={r['harmonic_n']:2d} q={r['even_q6']:2d} wt={r['wt']} "
            f"pair=({r['f_cycle_lo']},{r['f_cycle_hi']}) "
            f"dyadic={r['is_dyadic']} pred48={r['is_predecessor_48']}"
        )
    if eh["cycle_index_48"]:
        r = eh["cycle_index_48"][0]
        print(
            f"  predecessor_48 row: q={r['even_q6']} pair=({r['f_cycle_lo']},{r['f_cycle_hi']})"
        )
    print("  scale_map:")
    for k, v in eh["scale_map"].items():
        print(f"    {k}: {v}")
    print(f"  status={eh['status']}")
    print()

    _section("17. SCRIPT2 GATES")
    for name, ok in c2.gates:
        kind = c2.gate_kinds.get(name, "internal_kernel_identity")
        print(f"  {name:48s} {'PASS' if ok else 'FAIL'}  [{kind}]")
    print()


def _report_3(c3) -> None:
    _section("20. APERTURE AS COMMA")
    a = c3.aperture_cents
    for k in (
        "Delta_cont_cents",
        "PC_cents",
        "SC_cents",
        "Delta_depth4_cents",
        "Delta_dyadic_8_cents",
        "cgm_over_PC",
        "semitone_ratio_2_1_12",
        "48_Delta",
        "48_Delta_residue_octaves",
        "48_Delta_residue_cents",
        "ticks_per_octave",
        "best_dyadic_k_for_Delta_cont",
        "best_dyadic_err",
        "Delta_dyadic_8_is_best",
    ):
        print(f"  {k}={a[k]}")
    print(f"  status={a['status']}")
    print()

    _section("20b. FOUNDATION LOCKS (wave_normalization axiom)")
    L = c3.foundation_locks
    print(f"  phi_SU2={L.get('phi_SU2')}")
    for k in (
        "Q_G_m_a2",
        "s_p_over_m_a2",
        "four_pi2",
        "gyro_sum",
        "48_Delta",
        "48_Delta_residue",
        "chirality_space",
        "alpha0",
        "zeta",
        "alpha0_zeta",
        "rho4_over_pi_sqrt3",
        "rho5",
        "m_gap_RouteA_GeV",
        "E_grade2_GeV",
        "M_shell",
        "tau_G_leading",
    ):
        print(f"  {k}={L[k]}")
    for name, ok in L["checks"].items():
        kind = L.get("claim_kinds", {}).get(name, "")
        print(f"  {name:48s} {'PASS' if ok else 'FAIL'}  [{kind}]")
    print(f"  status={L['status']}")
    print()

    _section("20c. YM SHADOW / MASS-GAP LOCKS")
    y = c3.ym_shadow_locks
    for k in (
        "E_grade2_GeV",
        "m_gap_RouteA_GeV",
        "C2",
        "Delta_W_256",
        "Delta_W_128_over_255",
        "Delta_W_limit",
        "glueball_0pp_window_GeV",
        "m_gap_in_glueball_window_anno",
    ):
        print(f"  {k}={y[k]}")
    for name, ok in y["checks"].items():
        kind = y.get("claim_kinds", {}).get(name, "")
        print(f"  {name:48s} {'PASS' if ok else 'FAIL'}  [{kind}]")
    print(f"  status={y['status']}")
    print()

    _section("20d. CURVATURE MANIFESTATIONS")
    ca = c3.curvature_audit
    for r in ca["rows"]:
        print(f"  {r['name']:32s} {r['value']}  [{r['claim']}]")
    ot = ca["one_third_identity"]
    print(
        f"  |delta_BU - phi_SU2/3|={ot['abs_delta_BU_minus_phi_over_3']} "
        f"within_5e-4={ot['within_5e-4']} within_2e-3={ot['within_2e-3']}"
    )
    print(f"  fold_counts={ca['fold_disagreement_counts']}")
    print()

    _section("20e. STF QUADRUPOLE LAYER")
    st = c3.stf_layer
    for r in st["rows"]:
        print(f"  {r['tag']:24s} {r['value']}  ({r['note']}) [{r['claim']}]")
    print(f"  bulk_3968={st['all_five_shell_count']} dyadic_num5={st['dyadic_numerator_is_5']}")
    print()

    _section("21. LANDMARKS (octave phase)")
    print(
        f"  {'name':22s} {'sector':12s} {'oct':>8s} {'ticks':>10s} "
        f"{'phase':>8s} {'d_oct':>7s} {'d_half':>7s} {'d_dy8':>7s}"
    )
    for r in c3.landmarks:
        print(
            f"  {r['name']:22s} {r['sector']:12s} "
            f"{fmt(float(r['octaves_from_v']),4):>8s} "
            f"{fmt(float(r['ticks']),4):>10s} "
            f"{fmt(float(r['octave_phase_ticks']),4):>8s} "
            f"{fmt(float(r['dist_octave_boundary_ticks']),3):>7s} "
            f"{fmt(float(r['dist_half_octave_ticks']),3):>7s} "
            f"{fmt(float(r['dist_dyadic_8_ticks']),3):>7s}"
        )
    print()

    _section("22. CLUSTERING vs UNIFORM NULL (AUDIT-ONLY)")
    c = c3.clustering
    for k, v in c.items():
        if k == "status":
            continue
        print(f"  {k}={v}")
    print(f"  status={c['status']}")
    print()

    _section("23. EW MASSES")
    for r in c3.ew_masses:
        print(
            f"  {r['name']:10s} GeV={r.get('GeV')} "
            f"oct={fmt(float(r['octaves_below_v']),4)} "
            f"ticks={fmt(float(r['ticks']),4)} "
            f"res48={fmt(float(r.get('residual_48', float('nan'))),4)}"
        )
    print()

    _section("24. NUCLEAR")
    for r in c3.nuclear:
        keys = [k for k in r.keys() if k not in ("status",)]
        bits = " ".join(f"{k}={r[k]}" for k in keys)
        print(f"  {bits}  [{r['status']}]")
    print()

    _section("25. GRAVITY / APERTURE")
    g = c3.gravity
    for k, v in g.items():
        if k == "status":
            continue
        print(f"  {k}={v}")
    print(f"  status={g['status']}")
    print()

    _section("26. ALLOMETRY OCTAVE CONVERSION")
    for r in c3.allometry:
        print(
            f"  {r['name']:24s} a={fmt(float(r['exponent']),4)} "
            f"ticks/mass_oct={fmt(float(r['ticks_out_per_mass_octave']),4)}"
        )
    print()

    _section("27. CMB ell=37*n (HYPOTHESIS-GENERATING)")
    for r in c3.cmb_harmonics:
        print(
            f"  n={r['n']} ell={r['ell']} "
            f"oct={fmt(float(r['octaves_from_fundamental']),4)} "
            f"int_oct={r['is_integer_octave']} "
            f"stf_n5={r.get('is_stf_index_n5', False)}"
        )
    print()

    _section("28. FORTY-EIGHT INVENTORY")
    for r in c3.forty_eight:
        print(f"  {r['name']:20s} {r['value']}  ({r['note']})")
    print()

    _section("29. CROSS-CHART S_oct (AUDIT-ONLY)")
    x = c3.cross_chart
    for k in (
        "n_objects",
        "charts",
        "tol_ticks",
        "cross_pairs",
        "hits",
        "S_oct",
        "null_mean_S",
        "p_hat_S_ge_obs",
        "status",
    ):
        print(f"  {k}={x[k]}")
    print()

    _section("29b. AUDIT-ONLY (not kernel identities)")
    for name, status, val in c3.audits:
        print(f"  {name:32s} {status}  metric={val}")
    print()

    _section("30. SCRIPT3 GATES")
    for name, ok in c3.gates:
        kind = c3.gate_kinds.get(name, "")
        print(f"  {name:48s} {'PASS' if ok else 'FAIL'}  [{kind}]")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="CGM-hQVM octaves report")
    parser.add_argument("--only-1", action="store_true")
    parser.add_argument("--only-2", action="store_true")
    parser.add_argument("--only-3", action="store_true")
    parser.add_argument("--skip-2", action="store_true")
    parser.add_argument("--skip-3", action="store_true")
    args = parser.parse_args()

    run1 = not (args.only_2 or args.only_3)
    run2 = not (args.only_1 or args.only_3 or args.skip_2)
    run3 = not (args.only_1 or args.only_2 or args.skip_3)
    if args.only_2:
        run2 = True
    if args.only_3:
        run3 = True

    c1 = run_octaves_1() if run1 else None
    c2 = run_octaves_2() if run2 else None
    c3 = run_octaves_3() if run3 else None

    if c1 is not None:
        _report_1(c1)
    if c2 is not None:
        _report_2(c2)
    if c3 is not None:
        _report_3(c3)

    all_gates = []
    if c1 is not None:
        all_gates.extend(c1.gates)
    if c2 is not None:
        all_gates.extend(c2.gates)
    if c3 is not None:
        all_gates.extend(c3.gates)
    n_fail = sum(1 for _, ok in all_gates if not ok)
    _section("GATES SUMMARY")
    print(f"  exact_gates_fail={n_fail}/{len(all_gates)}")
    print(f"  results={RESULTS_PATH}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = Tee(old, buf)
    try:
        rc = main()
    finally:
        sys.stdout = old
    RESULTS_PATH.write_text(buf.getvalue(), encoding="utf-8")
    try:
        sys.stderr.write(f"wrote {RESULTS_PATH}\n")
    except Exception:
        pass
    raise SystemExit(rc)
