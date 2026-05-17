"""
CGM gravity: rho^5 structure, kernel transport, and G prediction.
"""
import math
from math import comb
import numpy as np
from cgm_gravity_common import (
    AF, C4_REF, Delta, FA_STF, G_kernel, G_meas, H_size, Omega_size,
    Q_G, TR_SIGMA_SHELL, V_EW_PDG, alpha_G_meas, binom_shell,
    build_joint_table, c4_from_anchors, chi6_step_bit_stats,
    configure_stdout_utf8, f_ordered, g_pred_from_tau,
    k4_pq_charges, kappa_binom_step, kappa_pi_step,
    measure_defect_invariant, pi_norm_shell, poly_mul,
    print_joint_table_condensed, require_compact_geom,
    require_monodromy, rho, stf_k4_coupling_matrix, tau_G_formula,
    tau_cycle_from_defect, tau_cycle_weighted, tau_g_with_c4,
    tau_path_binom, tau_path_kappa, tau_req_meas, tau_required,
    v_EW, weights_pop, d_BU, m_a,
)

configure_stdout_utf8()

print("=" * 10)
print("CGM gravity: rho^5, kernel transport, G prediction")
print("=" * 10)

# ============================================================
# Section 1: The 6 → 1+5 Decomposition
# ============================================================
print("\n" + "=" * 10)
print("Section 1: The 6 DoF → 1 Trace + 5 Trace-Free")
print("=" * 10)

l_quad = 2
n_STF = 2 * l_quad + 1
print(f"Quadrupole (l={l_quad}): {n_STF} STF components")
print(f"Monopole (l=0): 1 trace component")
print(f"Total: {n_STF + 1} = 6 (matches SE(3) DoF)")
print()
print("Trace (1 DoF): symmetric tensor monopole l=0")
print("STF (5 DoF): symmetric tensor quadrupole l=2")
print()

E_trace = np.eye(3) / np.sqrt(3)
E_stf = np.zeros((5, 3, 3))
E_stf[0] = np.diag([1, -1, 0]) / np.sqrt(2)
E_stf[1] = np.diag([1, 1, -2]) / np.sqrt(6)
E_stf[2] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
E_stf[3] = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
E_stf[4] = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])

print("STF basis verification:")
all_ok = True
for i in range(5):
    tr = np.trace(E_stf[i])
    norm = np.sum(E_stf[i] ** 2)
    ortho_trace = abs(np.sum(E_stf[i] * E_trace))
    if abs(tr) > 1e-10 or ortho_trace > 1e-10:
        all_ok = False
    print(f"  E_stf[{i}]: trace={tr:+.1e}, ||·||²={norm:.4f}, "
          f"⊥trace={ortho_trace:.1e}")
print(f"  All trace-free and orthogonal to trace: {all_ok}")

gram = np.zeros((6, 6))
all_basis = np.concatenate([[E_trace], E_stf])
for i in range(6):
    for j in range(6):
        gram[i, j] = np.sum(all_basis[i] * all_basis[j])
print(f"  Gram matrix diagonal: {np.diag(gram)}")
print(f"  Off-diagonal max: {np.max(np.abs(gram - np.diag(np.diag(gram)))):.1e}")

# ============================================================
# Section 2: Kernel Payload → STF Mapping
# ============================================================
print("\n" + "=" * 10)
print("Section 2: Kernel 6-Bit Payload → 1+5 Decomposition")
print("=" * 10)

print("\nSTF decomposition by payload population:")
print(f"{'pop':>4} {'Tr(σ)':>8} {'p_A':>8} {'||π||':>8} "
      f"{'STF amps (5)':>50} {'STF>0':>10}")
print("-" * 100)

for pop in range(7):
    tr_sigma = [0, 0.416667, 0.666667, 0.75, 0.666667, 0.416667, 0][pop]
    p_A = tr_sigma / 3
    if 1 <= pop <= 5:
        fa = np.sqrt(6) / 9
        sigma_norm = tr_sigma
        pi_norm = fa * sigma_norm
    else:
        pi_norm = 0.0
    if pi_norm > 0:
        stf_amps = [pi_norm / np.sqrt(5)] * 5
        stf_flag = "YES"
    else:
        stf_amps = [0.0] * 5
        stf_flag = "NO"
    amps_str = "  ".join(f"{a:.4f}" for a in stf_amps)
    print(f"{pop:>4} {tr_sigma:>8.4f} {p_A:>8.4f} {pi_norm:>8.4f} "
          f"{amps_str:>50} {stf_flag:>10}")

print()
print("Populations 1-5: nonzero ||pi||; populations 0 and 6: ||pi|| = 0")

# ============================================================
# Section 3: Quadrupole cross-reference
# ============================================================
print("\n" + "=" * 10)
print("Section 3: Quadrupole cross-reference")
print("=" * 10)
print(f"  Einstein quadrupole radiation: L_GW ∝ 1/dim(STF(3)) = 1/{n_STF}")
print(f"  CGM gravitational attenuation: ρ^5 = (1-Δ)^5, exponent = dim(STF(3)) = {n_STF}")

# ============================================================
# Section 4: STF attenuation factor
# ============================================================
print("\n" + "=" * 10)
print("Section 4: STF attenuation rho^5")
print("=" * 10)
tau_per_channel = -np.log(rho)
print(f"  dim(STF(3)) = {n_STF};  rho^5 = {rho**5:.10f}")
print(f"  per-channel tau_1 = -ln(rho) = {tau_per_channel:.6f}")

tau_naive = Omega_size * Delta
tau_with_stf = Omega_size * Delta * rho**5
tau_with_correction = Omega_size * Delta * rho**5 * (1 - 4*rho*Delta**2)
print()
print("  Optical depth comparison:")
print(f"    Naive (|Omega|*Delta):              {tau_naive:.6f}")
print(f"    With rho^5:                         {tau_with_stf:.6f}")
print(f"    With K4 (1-4*rho*Delta^2):          {tau_with_correction:.6f}")
print(f"    Required (2 ln(E_CS/v)):            {tau_required:.6f}")

# ============================================================
# Section 5: Kernel Shell Structure
# ============================================================
print("\n" + "=" * 10)
print("Section 5: Kernel Shells → 5 Bulk + 2 Horizons")
print("=" * 10)

shell_pops = [64 * comb(6, k) for k in range(7)]
bulk_shells = []
horizon_shells = []

print(f"\n{'Shell':>6} {'Population':>10} {'Fraction':>10} {'Tr(σ)':>8} "
      f"{'||π||':>8} {'Type':>15}")
print("-" * 70)
for k in range(7):
    frac = shell_pops[k] / Omega_size
    tr_sigma = [0, 0.416667, 0.666667, 0.75, 0.666667, 0.416667, 0][k]
    fa = np.sqrt(6)/9 if 1 <= k <= 5 else 0
    pi_norm = fa * tr_sigma if 1 <= k <= 5 else 0
    if k == 0 or k == 6:
        stype = "HORIZON"
        horizon_shells.append(k)
    else:
        stype = "BULK"
        bulk_shells.append(k)
    print(f"{k:>6} {shell_pops[k]:>10} {frac:>10.4f} {tr_sigma:>8.4f} "
          f"{pi_norm:>8.4f} {stype:>15}")

print()
print(f"Horizon shells (||pi||=0): {horizon_shells}")
print(f"Bulk shells (||pi||>0): {bulk_shells}")
print(f"Bulk shell count: {len(bulk_shells)}")
print(f"STF component count (Section 1): {n_STF}")
print(f"Both counts equal 5: {len(bulk_shells) == n_STF}")

# ============================================================
# Section 6: K4 correction factor
# ============================================================
print("\n" + "=" * 10)
print("Section 6: K4 correction factor (1 - 4 rho Delta^2)")
print("=" * 10)
tau_no_k4 = Omega_size * Delta * rho**5
tau_k4 = Omega_size * Delta * rho**5 * (1 - 4 * rho * Delta**2)
print(f"  tau without K4: {tau_no_k4:.6f}")
print(f"  tau with K4:    {tau_k4:.6f}")
print(f"  4*rho*D^2:      {4 * rho * Delta**2:.6e}")

# ============================================================
# Section 7: Kernel transport
# ============================================================
print("\n" + "=" * 10)
print("Section 7: Kernel transport")
print("=" * 10)

mon = require_monodromy()
joint_table = build_joint_table(mon) if mon is not None else []

print()
print("7.1  chi6 bit stats (per step)")
if joint_table:
    chi6_step_bit_stats(joint_table)
else:
    print("  (monodromy unavailable)")

print()
print("7.2  Path tau: 512-row sum vs binom cycle")
if not joint_table:
    print("  (monodromy unavailable)")
else:
    tau_512_cf = 0.0
    for rec in joint_table:
        w_step = rec["weight"] / 8.0
        tau_512_cf += w_step * kappa_pi_step(rec["shell"], rec["chi6"])
    print(f"  sum w*kappa (512 rows):          {tau_512_cf:.9f}")
    print(f"  tau_G closed:                    {tau_G_formula:.9f}")
    if mon is not None:
        w_sum = sum(weights_pop[m] for m in range(64))
        tau_binom_cycle = sum(
            weights_pop[m] * tau_path_binom(mon, m) for m in range(64)
        ) / w_sum
        tau_512_binom = sum(
            rec["weight"] / 8.0
            * kappa_binom_step(rec["shell"], rec["chi6"])
            for rec in joint_table
        )
        print(f"  tau_cycle binom (64 avg):        {tau_binom_cycle:.9f}")
        print(f"  8*tau_step_binom/w_sum:          {tau_512_binom * 8.0 / w_sum:.9f}")
        print(f"  N = tau_G/tau_binom:             {tau_G_formula / tau_binom_cycle:.4f}")

print()
print("7.3  Defect weight (kernel invariant)")
if mon is None:
    print("  (monodromy unavailable)")
else:
    uniq_w, mean_w = measure_defect_invariant(mon)
    tau_cycle_def = tau_cycle_from_defect(mean_w)
    tau_48 = 48.0 * Delta**2 / rho**3
    print(f"  defect weights (unique):         {uniq_w}")
    print(f"  mean weight/cycle:               {mean_w:.0f}")
    print(f"  tau_cycle(defect):               {tau_cycle_def:.12f}")
    print(f"  48*D^2/rho^3:                    {tau_48:.12f}")
    print(f"  ratio:                           {tau_cycle_def / tau_48:.12f}")

print()
print("7.4  Trace||pi|| along depth-8 (64 micro avg)")
if mon is None:
    print("  (monodromy unavailable)")
else:
    nariai = FA_STF
    for step_idx in range(1, 9):
        tr_vals, pi_vals, ratio_vals = [], [], []
        for m in range(64):
            word = mon.family_word_for_micro(m) * 2
            rows = [r for r in mon.trace_word(word)[1:] if r.byte is not None]
            if step_idx > len(rows):
                continue
            sh = rows[step_idx - 1].obs.arch_shell
            tr = TR_SIGMA_SHELL[sh] if 0 <= sh <= 6 else 0.0
            pn = pi_norm_shell(sh)
            tr_vals.append(tr)
            pi_vals.append(pn)
            if tr > 1e-12:
                ratio_vals.append(pn / tr)
        tr_m = sum(tr_vals) / len(tr_vals) if tr_vals else 0.0
        pi_m = sum(pi_vals) / len(pi_vals) if pi_vals else 0.0
        r_m = sum(ratio_vals) / len(ratio_vals) if ratio_vals else 0.0
        print(
            f"  step {step_idx}: mean Tr={tr_m:.4f}  ||pi||={pi_m:.4f}  "
            f"||pi||/Tr={r_m:.4f}  (Nariai={nariai:.4f})"
        )

print()
print("7.5  Per-family tau (depth-4 word)")
if mon is None:
    print("  (monodromy unavailable)")
else:
    tau_fam = []
    for fam in range(4):
        tau_sum = w_sum_f = 0.0
        for m in range(64):
            w = weights_pop[m]
            byte = mon.byte_from_family_and_micro(fam, m)
            for row in mon.trace_word([byte] * 4)[1:]:
                if row.byte is None:
                    continue
                tau_sum += w * kappa_pi_step(row.obs.arch_shell, row.obs.chi6)
                w_sum_f += w
        tau_word = tau_sum / w_sum_f if w_sum_f > 0 else 0.0
        tau_fam.append(tau_word)
        print(f"  family {fam}: tau_word={tau_word:.9f}  tau_step={tau_word / 4:.9f}")
    mean_tw = sum(tau_fam) / 4.0
    print(f"  mean tau_word: {mean_tw:.9f}  var: {sum((t - mean_tw)**2 for t in tau_fam):.6e}")

print()
print("7.6  CS fixed-point micro 63")
micro_63 = 0x3F
print(f"  pop={bin(micro_63).count('1')}  weight 12 even: {bin(micro_63).count('1') * 2 == 12}")
if mon is not None:
    qxor_vals = []
    chi_ok = True
    for row in mon.trace_word(mon.family_word_for_micro(63) * 2)[1:]:
        if row.byte is None:
            continue
        qxor_vals.append(row.qxor)
        if mon.byte_chirality_increment(row.byte) != 63:
            chi_ok = False
    print(f"  chi increment=63 all steps: {chi_ok}")
    print(f"  qxor path: {qxor_vals}")

# ============================================================
# Section 8: tau_cycle and structural tests
# ============================================================
print("\n" + "=" * 10)
print("Section 8: tau_cycle and structural tests")
print("=" * 10)

print(f"tau_required (2 ln(E_CS/v_EW)):  {tau_required:.6f}")
print(f"tau_req (-ln(alpha_G/G_kernel)): {tau_req_meas:.6f}")
print(f"tau_G formula:                   {tau_G_formula:.6f}")
print(f"tau_G - tau_required:            {tau_G_formula - tau_required:.2e}")
print()

tau_cycle = None
tau_cycle_pred_48 = AF * Delta**2 / rho**3

if mon is not None:
    tau_binom = tau_cycle_weighted(mon, weights_pop, binom_shell)
    tau_cycle = tau_binom

    print("8.1  Two-lemma factorization")
    n_cycles_pred = (Omega_size / AF) * (rho**8 / Delta)
    n_cycles_pred_k4 = n_cycles_pred * f_ordered
    tau_g_lemma = n_cycles_pred_k4 * tau_cycle_pred_48
    rho_exp = 8 - 3
    print(f"  Lemma A: tau_cycle = 48*Delta^2/rho^3 = {tau_cycle_pred_48:.9f}")
    print(f"  Lemma B: N_cycles = (|Omega|/48)*(rho^8/Delta)*(1-4*rho*Delta^2) = {n_cycles_pred_k4:.6f}")
    print(f"  Product: N * tau_cycle = {tau_g_lemma:.9f}")
    print(f"  |Omega|*Delta*rho^5*K4 = {tau_G_formula:.9f}")
    print(f"  product - tau_G: {tau_g_lemma - tau_G_formula:.6e}")
    print(f"  product - tau_required: {tau_g_lemma - tau_required:.6e}")
    print(f"  rho exponent: 8-3 = {rho_exp}")

    print()
    print("8.2  tau_cycle (binom shell weights)")
    print(f"  tau_binom: {tau_binom:.9f}  /pred_48={tau_binom/tau_cycle_pred_48:.6f}")
    per_pop = {}
    for micro in range(64):
        p = bin(micro).count("1")
        per_pop.setdefault(p, []).append(tau_path_kappa(mon, micro, binom_shell))
    pop_parts = [
        f"p{p}={sum(v)/len(v):.6f}(n={len(v)})"
        for p, v in sorted(per_pop.items())
    ]
    print(f"  per-pop tau: {' '.join(pop_parts)}")
    print(
        f"  Lemma A: tau/pred_48={tau_binom/tau_cycle_pred_48:.6f}  "
        f"Lemma B: N_meas/pred_k4={tau_G_formula/tau_binom/n_cycles_pred_k4:.6f}"
    )
else:
    print("8.1  Two-lemma factorization: (monodromy unavailable)")

if tau_cycle is not None and tau_cycle > 0:
    n_cycles_meas = tau_G_formula / tau_cycle

    print()
    print("8.3  Effective rho exponent (data-driven)")
    denom_a = Omega_size * Delta * f_ordered
    a_fit = np.log(tau_required / denom_a) / np.log(rho)
    print(f"  a_fit: {a_fit:.6f}")
    print(f"  rho^5: {rho**5:.12f}")
    print(f"  rho^(8-3): {rho**8 / rho**3:.12f}")

    print()
    print("8.4  Binomial generating functions")
    binom_rows = []
    for k in range(7):
        b5 = comb(5, k) if k <= 5 else 0
        b6 = comb(6, k)
        b5km1 = comb(5, k - 1) if k >= 1 else 0
        binom_rows.append(f"k{k}:{b5}/{b6}/{b6-b5}/{b5km1}")
    print(f"  b5/b6/diff/b5k-1: {' | '.join(binom_rows)}")
    rho5_coeffs = [(-1) ** k * comb(5, k) for k in range(6)]
    gen6_coeffs = [comb(6, k) for k in range(7)]
    print(f"  (1-D)^5 coeffs: {rho5_coeffs}")
    print(f"  (1+D)^6 coeffs: {gen6_coeffs}")
    rho5_binom = sum((-1) ** k * comb(5, k) * Delta**k for k in range(6))
    print(f"  rho^5 numeric: {rho**5:.12f}")
    print(f"  (1-D)^5 binom: {rho5_binom:.12f}")

    residual_tau = tau_G_formula - tau_required
    ln_rho = -np.log(rho)

    print()
    print("8.5  Trace excess from a_fit")
    a_fit_all = np.log(tau_required / (Omega_size * Delta * f_ordered)) / np.log(rho)
    delta_tau_afit = tau_G_formula * (a_fit_all - 5.0) * ln_rho
    print(f"  a_fit: {a_fit_all:.9f}")
    print(f"  a_fit - 5: {a_fit_all - 5.0:.9e}")
    print(f"  tau_G * (a_fit-5) * ln(rho): {delta_tau_afit:.6e}")
    print(f"  tau_G - tau_required: {residual_tau:.6e}")
    print(f"  ratio: {delta_tau_afit / residual_tau:.6f}")

    print()
    print("8.6  Residual scaled by |Omega|*Delta^n")
    base_od = Omega_size * Delta
    scale_parts = [
        f"D^{n}={residual_tau / (base_od * Delta ** (n - 1)):.6f}"
        for n in (3, 4, 5)
    ]
    print(f"  {' '.join(scale_parts)}")
    ratio_d4 = residual_tau / (tau_G_formula * Delta**4)
    print(f"  residual / (tau_G * Delta^4): {ratio_d4:.6f}")
    print(f"  7/4 reference: {7.0 / 4.0:.6f}")
    tau_d4_test = tau_G_formula * (1.0 + (7.0 / 4.0) * Delta**4)
    print(f"  tau_G * (1 + 7/4 Delta^4): {tau_d4_test:.9f}")
    print(f"  minus tau_required: {tau_d4_test - tau_required:.6e}")

    channels, h_card = require_compact_geom()
    if channels is not None:
        p_vals = [ch.p for ch in channels]
        q_vals = [ch.q for ch in channels]
        r5_vals = [ch.r5 for ch in channels]
        print()
        print("8.7  EW Delta-expansion coeffs (compact geometry)")
        ch_parts = [
            f"{ch.label}(p,q,r5,c)={ch.p:.4f},{ch.q:.4f},{ch.r5:.4f},{ch.c:.4f}"
            for ch in channels
        ]
        print(f"  {'; '.join(ch_parts)}")
        print(
            f"  sum(p,q,r5)=({sum(p_vals):.6f},{sum(q_vals):.6f},{sum(r5_vals):.6f})"
        )

    print()
    print("Depth-8 cycle counting:")
    print(f"  N cycles = tau_G / tau_cycle: {n_cycles_meas:.1f}")
    print(f"  tau_G / tau_required: {tau_G_formula / tau_required:.6f}")
    print(f"  K4 factor: {f_ordered:.12f}")

# ============================================================
# Section 9: Joint table and Delta expansion
# ============================================================
print("\n" + "=" * 10)
print("Section 9: Joint table and Delta expansion")
print("=" * 10)

if not joint_table and mon is not None:
    joint_table = build_joint_table(mon)

print()
print("9.1  Joint table (64 micros x 8 steps)")
if not joint_table:
    print("  (monodromy unavailable)")
else:
    print_joint_table_condensed(joint_table)

print()
print("9.2  Perturbation series in Delta")
p_rho5 = [comb(5, k) * ((-1) ** k) for k in range(6)]
p_corr = [1.0, 0.0, -4.0, 4.0]
p_prod = poly_mul(p_rho5, p_corr)
tau_coeffs = [0.0] + [Omega_size * c for c in p_prod]
tau_exact = Omega_size * Delta * rho**5 * f_ordered
tau_partial = 0.0
partials = []
terms = []
for n, c in enumerate(tau_coeffs):
    term = c * Delta**n
    tau_partial += term
    partials.append(f"{tau_partial:.6f}")
    terms.append(f"{term:.6f}")
print(f"  partial_tau n=0..{len(tau_coeffs)-1}: {', '.join(partials)}")
print(f"  terms n=0..{len(tau_coeffs)-1}:       {', '.join(terms)}")
print(f"  exact={tau_exact:.10f} sum={tau_partial:.10f} diff={tau_exact - tau_partial:.6e}")
cn_over = [float(c / Omega_size) for c in tau_coeffs]
print(f"  c_n/|Omega| n=0..{len(tau_coeffs)-1}: {cn_over}")
conv_parts = []
for n in range(len(tau_coeffs) - 1):
    if tau_coeffs[n] != 0:
        conv_parts.append(
            f"{n}->{n+1}:{abs(tau_coeffs[n + 1] * Delta / tau_coeffs[n]):.6f}"
        )
print(f"  |c_{{n+1}}*D/c_n|: {' '.join(conv_parts)}")

cn_ref = np.array([float(c) for c in p_prod], dtype=float)
cn_meas = np.array(
    [tau_coeffs[n + 1] / Omega_size for n in range(len(p_prod))],
    dtype=float,
)
cn_diff = float(np.max(np.abs(cn_ref - cn_meas))) if len(cn_ref) else 0.0
tau_poly = sum(tau_coeffs[n] * Delta**n for n in range(len(tau_coeffs)))
print()
print("9.3  Series = Delta*(1-D)^5*(1-4(1-D)D^2)")
print(f"  max |coeff_ref - coeff_meas|: {cn_diff:.6e}")
print(f"  max degree: {len(tau_coeffs) - 1}")
print(f"  poly - closed form: {tau_poly - tau_exact:.6e}")
if cn_diff < 1e-12 and abs(tau_poly - tau_exact) < 1e-6:
    print("  THEOREM: series matches closed tau_G.")

print()
print("9.4  Residual and c4")
residual_tau = tau_G_formula - tau_required
k_needed = tau_required / (Omega_size * Delta * rho**5)
x_exact = k_needed - f_ordered
c4_needed = x_exact / Delta**4 if Delta > 0 else float("nan")
g_pred_mf = g_pred_from_tau(tau_G_formula)
ppm_g = (g_pred_mf / G_meas - 1.0) * 1e6 if G_meas > 0 else float("nan")
print(f"  tau_G - tau_req: {residual_tau:.6e} ({residual_tau/tau_G_formula*1e6:.2f} ppm on tau)")
print(f"  G_pred/G_meas: {ppm_g:.1f} ppm")
print(f"  c4 = x/D^4: {c4_needed:.6f}  ref -7/4 = {C4_REF:.6f}")
tr_eq = TR_SIGMA_SHELL[3]
mono = (1.0 + tr_eq) * Delta**4 * tau_G_formula
print(f"  (1+Tr_eq)*D^4*tau_G: {mono:.6e}  /residual = {mono/residual_tau:.6f}")
tau_ext = tau_g_with_c4(C4_REF)
g_ext = g_pred_from_tau(tau_ext)
print(f"  with c4=-7/4: tau residual {tau_ext - tau_required:.3e}  G ppm {(g_ext/G_meas-1)*1e6:.2f}")

try:
    from scipy import constants as sc_const
    g_row = sc_const.physical_constants[
        "Newtonian constant of gravitation over h-bar c"
    ]
    g_codata, g_sigma = float(g_row[0]), float(g_row[2])
    v_pdg, v_sigma = V_EW_PDG
    c4_c = c4_from_anchors(g_codata, v_pdg)
    c4_gh = c4_from_anchors(g_codata + g_sigma, v_pdg)
    c4_gl = c4_from_anchors(g_codata - g_sigma, v_pdg)
    c4_vh = c4_from_anchors(g_codata, v_pdg + v_sigma)
    c4_vl = c4_from_anchors(g_codata, v_pdg - v_sigma)
    sig_c4 = math.sqrt(
        ((c4_gh - c4_gl) / 2) ** 2 + ((c4_vh - c4_vl) / 2) ** 2
    )
    print(f"  c4 (CODATA+PDG): {c4_c:.4f} +/- {sig_c4:.4f}  "
          f"within 1-sig of -7/4: {abs(c4_c - C4_REF) < sig_c4}")
except ImportError:
    pass

_pq, q_sum = k4_pq_charges()
q_w = next(q for n, _p, q in _pq if n == "w")
print(f"  K4 sum(q_i)={q_sum:.4f}  q_W={q_w:.4f}")

# ============================================================
# Section 10: Three routes to exponent 5
# ============================================================
print("\n" + "=" * 10)
print("Section 10: Three routes to exponent 5")
print("=" * 10)
print(f"""
Route A (representation theory):
  Symmetric 3x3 tensor: 6 = 1 trace + 5 STF
  dim(STF(3)) = {n_STF}

Route B (kernel shells):
  Horizon shells (||pi||=0): {horizon_shells}
  Bulk shells (||pi||>0): {bulk_shells}
  Bulk count = {len(bulk_shells)}

Route C (cycle geometry):
  rho exponent: 8-3 = 5

All three yield exponent 5 without reference to G.
""")

# ============================================================
# Section 11: Coupling summary
# ============================================================
print("=" * 10)
print("Section 11: Coupling summary")
print("=" * 10)

tau_G = tau_G_formula
G_pred = g_pred_from_tau(tau_G)
alpha_G_pred = G_kernel * math.exp(-tau_G)

print(f"CGM Invariants:")
print(f"  Q_G = {Q_G:.10f}")
print(f"  m_a = {m_a:.12f}")
print(f"  δ_BU = {d_BU:.12f}")
print(f"  ρ = {rho:.12f}")
print(f"  Δ = {Delta:.12f}")
print()
print(f"Kernel Invariants:")
print(f"  |Ω| = {Omega_size}")
print(f"  |H| = {H_size}")
print(f"  Shell displacement = 24")
print(f"  G_kernel = π/6 = {G_kernel:.12f}")
print()
print(f"Optical Depth Formula:")
print(f"  τ_G = |Ω|·Δ·ρ⁵·(1-4ρΔ²)")
print(f"      = {Omega_size} × {Delta:.12f} × {rho:.12f}⁵ × {1-4*rho*Delta**2:.12f}")
print(f"      = {tau_G:.12f}")
print()
print(f"Gravitational Coupling:")
print(f"  α_G(v) = G_kernel × exp(-τ_G) = {alpha_G_pred:.6e}")
print(f"  α_G(v) measured = {alpha_G_meas:.6e}")
print()
print(f"  G_pred = {G_pred:.6e} GeV⁻²")
print(f"  G_meas = {G_meas:.6e} GeV⁻²")
print(f"  G_pred/G_meas - 1 = {(G_pred/G_meas - 1):.6e}")
print()
print(f"  Residual tau_G - tau_required: {tau_G - tau_required:.6e}")

# ============================================================
# Section 12: STF-K4 coupling matrix
# ============================================================
print("\n" + "=" * 10)
print("Section 12: STF-K4 coupling matrix")
print("=" * 10)

canonical_path = {
    "shell": [0, 1, 6, 1, 0, 1, 6, 1, 0],
    "fsum": [0, 0, 1, 3, 2, 2, 3, 1, 0],
    "H": [True, False, False, False, True, False, False, False, True],
    "E_flag": [False, False, True, False, False, False, True, False, False],
}
frac_stf = [0.3, 0.3, 0.2, 0.1, 0.1]
m_couple, bulk_steps = stf_k4_coupling_matrix(canonical_path, frac_stf)

print()
print("12.1  M[a,g] (pop=1 path)")
print(f"  bulk step indices: {bulk_steps}")
for a in range(5):
    row = " ".join(f"{m_couple[a, g]:.6f}" for g in range(4))
    print(f"    a={a}: {row}")
stf_l2 = float(np.sum(m_couple ** 2))
decomp_79 = -1 + comb(5, 2) * 4 + comb(5, 3) * 4
print(f"  ||M||_F^2={stf_l2:.6f}")
print(f"  c6 series: -1 + C(5,2)*4 + C(5,3)*4 = {decomp_79:.0f}")

print()
print("12.2  Two-lemma vs tau_G formula")
if mon is not None and tau_cycle is not None and tau_cycle > 0:
    n_k4 = (Omega_size / AF) * (rho**8 / Delta) * f_ordered
    tau_lemma = n_k4 * tau_cycle_pred_48
    print(f"  N_cycles_k4: {n_k4:.6f}")
    print(f"  tau_cycle (binom): {tau_cycle:.9f}")
    print(f"  N * tau_binom: {n_k4 * tau_cycle:.9f}")
    print(f"  tau_G formula: {tau_G_formula:.9f}")
    print(f"  N*tau_48 (Lemma A*B): {tau_lemma:.9f}")
else:
    print("  (monodromy unavailable)")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 10)
print("SUMMARY")
print("=" * 10)
print(f"  rho = {rho:.12f}")
print(f"  Delta = {Delta:.12f}")
print(f"  STF dim = {n_STF}, bulk shells = {len(bulk_shells)}, |K4| = 4")
print(f"  tau_G = |Omega| Delta rho^5 (1-4 rho Delta^2) = {tau_G:.6f}")
print(f"  G_pred/G_meas - 1 = {(G_pred/G_meas - 1)*1e6:.1f} ppm")
print(f"  Residual: tau_G - tau_required = {tau_G - tau_required:.6e}")
print(f"  c4 = {c4_needed:.6f} (ref -7/4 = {C4_REF:.6f})")
print(f"  (1+Tr_eq)*D^4*tau_G / residual = {mono/residual_tau:.6f}")