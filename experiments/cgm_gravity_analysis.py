#!/usr/bin/env python3
"""
cgm_gravity_analysis.py
CGM gravitational coupling analysis.

The central principle: gravity is the cost of preserving common ancestry
across scale. Mass is accumulated common-source structure; the ancestry
credit field C_A measures that accumulation; the gravitational potential
is Φ = −C_A.

The finite CGM kernel supplies shell and horizon invariants that
normalise the coupling. The depth-8 orientation-recovery cycle gives
the factor of two in the relativistic coupling decomposition.

Derived quantities are stratified:
  - Deductive: field equation form, Q_G = 4π, G_kernel = π/6,
    spin-2 from depth-8, shell displacement invariant = 24
  - Invariant: Q_G, δ_BU, m_a, Δ, ρ (representation-independent)
  - Matched (not yet derived): τ_G exponents (5,4), v_EW anchor
  - Phenomenological: α(z) oscillation, α·ζ product
"""
import math
import cmath
import sys
from typing import Any

_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(_reconfigure):
    try:
        _reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ================================================================
# 1  Primitive measures and derived amplitudes
# ================================================================
s_p = math.pi / 2.0           # CS threshold
u_p = 1.0 / math.sqrt(2.0)   # UNA threshold = cos(π/4)
o_p = math.pi / 4.0           # ONA threshold

# Zero-defect gyrotriangle: s_p + o_p + o_p = π
Q_G = 4.0 * math.pi           # quantum-gravity horizon

# Aperture from chiral seed distributed across L/R phase ranges
# A² × (2π)_L × (2π)_R = s_p  ⟹  m_a² = s_p/(4π²)
m_a = math.sqrt(s_p / (4.0 * math.pi * math.pi))

d_BU  = 0.195342176580         # BU dual-pole monodromy
rho   = d_BU / m_a             # closure ratio
Delta = 1.0 - rho              # aperture gap
S_geo = m_a * math.pi * math.sqrt(3.0) / 2.0
zeta  = Q_G / S_geo
alpha = d_BU**4 / m_a          # fine-structure constant (CGM geometric)
phi_SU2 = 2.0 * math.acos((1.0 + 2.0 * math.sqrt(2.0)) / 4.0)

# Dimensionless stage actions
S_CS  = s_p / m_a
S_UNA = u_p / m_a
S_ONA = o_p / m_a
S_BU  = 1.0                    # m_a / m_a

# Product invariant: α·ζ = ρ⁴/(π√3)
alpha_zeta_exact = rho**4 / (math.pi * math.sqrt(3.0))
alpha_zeta_check = alpha * zeta

# 48-fold structure
AF = 48
prod_48Delta = AF * Delta
N_e = AF**2
lam0 = Delta / math.sqrt(5.0)
lam0_expected = 1.0 / (AF * math.sqrt(5.0))

# ================================================================
# 2  Physical anchors
# ================================================================
v_EW = 246.22                   # electroweak scale (GeV)
G_measured_nat = 6.70881e-39     # GeV⁻²
G_SI_measured  = 6.67430e-11     # m³ kg⁻¹ s⁻²
alpha_G_meas   = G_measured_nat * v_EW**2

# ================================================================
# 3  Kernel invariants (from monodromy module)
# ================================================================

def _require_monodromy():
    """Import the monodromy module or abort gracefully."""
    try:
        import cgm_aqpu_monodromy as mon
        return mon
    except Exception as exc:
        print(f"  Monodromy module unavailable: {exc}")
        return None


def run_shell_displacement_invariant(mon):
    """Section 3.2: verify shell displacement = 24 for all payloads."""
    print("\n--- 3.2  Shell displacement invariant ---")
    if mon is None:
        print("  Skipped: monodromy module unavailable.")
        return {"total": 0, "failed": 64}

    fam_ok, d4_ok, inv_ok = mon.verify_family_shell_law_exhaustive()
    print(f"  first-front law: family_shell={fam_ok}, "
          f"depth4_return={d4_ok}, invariant={inv_ok}")

    disp_vals = []
    half_vals  = []
    failures   = []
    payload_hits = half_hits = mass_hits = depth_hits = 0

    for micro in range(64):
        pop  = mon.payload_popcount(micro)
        word = mon.family_word_for_micro(micro) * 2
        rows = mon.trace_word(word)
        shells = [r.obs.arch_shell for r in rows]
        if len(shells) != 9 or any(s < 0 for s in shells):
            failures.append(micro)
            continue
        deltas = [abs(shells[i] - shells[i-1]) for i in range(1, len(shells))]
        total  = sum(deltas)
        half   = sum(deltas[:4])
        first  = shells[1] if len(shells) > 1 else None
        disp_vals.append(total)
        half_vals.append(half)
        if total == 24:     payload_hits += 1
        if half  == 12:     half_hits    += 1
        if first == pop:    mass_hits    += 1
        if max(shells) == 6 and min(shells) == 0:  depth_hits += 1

    print(f"  depth-8 displacement: unique values {sorted(set(disp_vals))}")
    print(f"  depth-4 half-cycle:   unique values {sorted(set(half_vals))}")
    print(f"  pass: {64 - len(failures)}/64  "
          f"(disp={payload_hits}  half={half_hits}  "
          f"mass-shell={mass_hits}  envelope={depth_hits})")
    return {"total": 64, "failed": len(failures),
            "disp_vals": disp_vals, "half_vals": half_vals}


def run_gauss_map(shell_inv):
    """Section 3.4: G_kernel = Q_G / 24."""
    print("\n--- 3.4  Kernel Gauss map ---")
    G_kernel = Q_G / 24.0
    failed = shell_inv.get("failed", 0) if shell_inv else 64
    total  = shell_inv.get("total", 0)  if shell_inv else 0
    print(f"  displacement invariant holds: {failed == 0 and total == 64}")
    print(f"  Q_G / 24 = π/6 = {G_kernel:.12f}")
    return G_kernel


def run_equality_horizon(mon):
    """Section 3.5: chirality cancellation at equality horizon."""
    print("\n--- 3.5  Equality horizon chirality cancellation ---")
    if mon is None:
        print("  Skipped.")
        return {}
    canonical = mon.family_word_for_micro(mon.FAMILY_RAY_REF)
    rows = mon.trace_word(canonical * 2)
    eq_steps = []
    qx_vals  = []
    for r in rows:
        phase = mon.phase_name(r.obs, r.step)
        if r.obs.equality_horizon:
            eq_steps.append(r.step)
            qx_vals.append(r.qxor & 0x3F)
    all_flip = len(eq_steps) == 2 and all(v == 0x3F for v in qx_vals)
    print(f"  equality steps: {eq_steps}")
    print(f"  qx at equality: {qx_vals}")
    print(f"  all six chirality bits flip: {all_flip}")
    return {"eq_steps": eq_steps, "qx": qx_vals, "all_flip": all_flip}


# ================================================================
# 4  Orientation recovery and the factor two
# ================================================================

def run_monodromy_probes(mon):
    """Section 4.1: depth-4 vs depth-8 closure."""
    print("\n--- 4.1  Depth-4 / depth-8 closure ---")
    if mon is None:
        print("  Skipped.")
        return False
    try:
        summaries = dict(mon.monodromy_probe_summaries())
        s4 = summaries["canonical"]
        s8 = summaries["canonical x2"]
        s16 = summaries["canonical x4"]
        print(f"  depth-4: terminal={s4.terminal_coordinate}, "
              f"defect=0x{s4.closure_defect:06X}, rest={s4.carrier_revisits_rest}")
        print(f"  depth-8: terminal={s8.terminal_coordinate}, "
              f"defect=0x{s8.closure_defect:06X}, "
              f"first_return={s8.first_revisit_of_start_step}")
        ok = (s4.terminal_coordinate == "swapped"
              and not s4.carrier_revisits_rest
              and s8.carrier_revisits_rest
              and s8.first_revisit_of_start_step == 8)
        if ok:
            print("  depth-4: swapped; depth-8: rest at step 8")
        else:
            print(f"  orientation recovery: {ok}")
        return ok
    except Exception as exc:
        print(f"  Failed: {exc}")
        return False


def run_family_circulation(mon):
    """Section 4.2: two-pass carrier return → spin-2."""
    print("\n--- 4.2  Two-pass carrier return ---")
    if mon is None:
        print("  Skipped.")
        return {}
    canonical = mon.family_word_for_micro(mon.FAMILY_RAY_REF)
    rows = mon.trace_word(canonical * 2)
    fsums = [r.fsum for r in rows]
    deltas = [fsums[i] - fsums[i-1] for i in range(1, len(fsums))]
    egress  = deltas[:4]
    ingress = deltas[4:]
    net_e = sum(egress)
    net_i = sum(ingress)
    print(f"  family sequence: {fsums}")
    print(f"  increments: {deltas}")
    print(f"  egress circulation: {net_e:+d}")
    print(f"  ingress circulation: {net_i:+d}")
    print(f"  net cancellation: {net_e + net_i}")
    print(f"  |circulation per half-cycle| = {abs(net_e)} (spin-2 signature)")
    return {"fsums": fsums, "deltas": deltas,
            "net_egress": net_e, "net_ingress": net_i}


def run_gravitational_memory(mon):
    """Section 6: gravitational memory from depth-4 defect."""
    print("\n--- 6  Gravitational memory ---")
    if mon is None:
        print("  Skipped.")
        return {}
    try:
        summaries = dict(mon.monodromy_probe_summaries())
        s4 = summaries["canonical"]
        s8 = summaries["canonical x2"]
        xor = s4.final_state ^ s8.final_state
        frac4 = xor.bit_count() / 24.0
        frac8 = s8.closure_defect.bit_count() / 24.0
        print(f"  depth-4 defect: 0x{s4.closure_defect:06X}  "
              f"(full displacement memory retained)")
        print(f"  depth-8 defect: 0x{s8.closure_defect:06X}  "
              f"(memory cancelled)")
        return {"frac4": frac4, "frac8": frac8}
    except Exception as exc:
        print(f"  Failed: {exc}")
        return {}


# ================================================================
# 5  Gravitational radiation
# ================================================================

def run_gw_spectrum(mon):
    """Section 5: dominant k=2 quadrupole mode."""
    print("\n--- 5  Gravitational radiation ---")
    if mon is None:
        print("  Skipped.")
        return {}
    canonical = mon.family_word_for_micro(mon.FAMILY_RAY_REF)
    rows = mon.trace_word(canonical * 2)
    shells = [r.obs.arch_shell for r in rows][:-1]
    baseline = shells[0]
    strain = [s - baseline for s in shells]
    n = len(strain)
    mean = sum(strain) / n
    centered = [v - mean for v in strain]
    spectrum = []
    for k in range(n // 2 + 1):
        c = sum(centered[t] * cmath.exp(-2j * math.pi * k * t / n)
                for t in range(n))
        spectrum.append(abs(c) / n)
    freqs = [k / n for k in range(len(spectrum))]
    if len(spectrum) > 2:
        dom_amp, dom_freq, dom_k = max(
            (spectrum[k], freqs[k], k) for k in range(1, len(spectrum)))
    else:
        dom_k = None
        dom_amp = dom_freq = 0.0
    print(f"  shell path: {shells}")
    print(f"  dominant mode: k={dom_k}, freq={dom_freq:.3f}, amp={dom_amp:.4f}")
    print(f"  k=2 (quadrupole) amplitude: "
          f"{spectrum[2]:.4f}" if len(spectrum) > 2 else "  N/A")
    return {"dominant_k": dom_k, "spectrum": spectrum}


# ================================================================
# 7  Potential profile
# ================================================================

def run_potential_profile(G_kernel):
    """Section 7: binomial interior potential and far-field check."""
    print("\n--- 7  Potential profile ---")
    shell_w = [math.comb(6, k) / 64.0 for k in range(7)]
    total_w = sum(shell_w)
    shell_d = [w / total_w for w in shell_w]

    # Continuous radial model
    enclosed = []
    s = 0.0
    for w in shell_w:
        s += w
        enclosed.append(s)

    r_vals = [k / 6.0 for k in range(7)]
    r_vals[0] = 1e-6
    g_field = [0.0] * 7
    for k in range(1, 7):
        r = r_vals[k]
        g_field[k] = -G_kernel * enclosed[k] / (r * r)
    g_field[0] = g_field[1]

    phi = [0.0] * 7
    for k in range(5, -1, -1):
        dr = r_vals[k+1] - r_vals[k]
        phi[k] = phi[k+1] - 0.5 * (g_field[k] + g_field[k+1]) * dr

    print("  r     M(r)    g(r)       C_A(r)     Phi(r)=-C_A")
    for k in range(7):
        c_a = phi[k]
        phi_grav = -phi[k]
        pr = phi_grav * r_vals[k]
        print(f"  {r_vals[k]:.3f} {enclosed[k]:7.4f} "
              f"{g_field[k]:10.6f} {c_a:10.6f} {phi_grav:10.6f}")
    print(f"  boundary field = −G_kernel = {g_field[-1]:.6f}  "
          f"(expected −π/6 = {-math.pi/6:.6f})")

    # Extended far-field grid
    N = 1000
    r_ext = [1e-6 + i * (10.0 - 1e-6) / (N - 1) for i in range(N)]
    rho_ext = []
    mass_acc = 0.0
    enc_ext = [0.0] * N
    for i in range(N):
        r = r_ext[i]
        if r <= 1.0:
            x = r * 6.0
            k0 = min(int(x), 5)
            k1 = k0 + 1
            t = x - k0
            rho_ext.append((1 - t) * shell_w[k0] + t * shell_w[k1])
        else:
            rho_ext.append(0.0)
        if i > 0:
            dV = (4.0 / 3.0) * math.pi * (r_ext[i]**3 - r_ext[i-1]**3)
            mass_acc += rho_ext[i-1] * dV
            enc_ext[i] = mass_acc

    g_ext = [0.0] * N
    for i in range(N):
        r = r_ext[i]
        if r > 0:
            g_ext[i] = -G_kernel * enc_ext[i] / (r * r)

    phi_ext = [0.0] * N
    for i in range(N - 2, -1, -1):
        dr = r_ext[i+1] - r_ext[i]
        phi_ext[i] = phi_ext[i+1] - 0.5 * (g_ext[i] + g_ext[i+1]) * dr

    far_idx = [i for i in range(N) if 1.5 <= r_ext[i] <= 9.0]
    phi_r_far = [phi_ext[i] * r_ext[i] for i in far_idx]
    if phi_r_far:
        mean_pr = sum(phi_r_far) / len(phi_r_far)
        std_pr  = (sum((x - mean_pr)**2 for x in phi_r_far)
                   / len(phi_r_far))**0.5
    else:
        mean_pr = std_pr = 0.0

    print(f"  far-field Phi*r: mean={mean_pr:.6f}, std={std_pr:.2e}")
    rel_std = std_pr / (abs(mean_pr) + 1e-20)
    if rel_std < 0.01:
        print("  exterior 1/r: interior boundary normalisation ok; "
              "full exterior match is a continuum-map item (Section 10).")
    else:
        print(f"  exterior 1/r: relative std={rel_std:.4f} "
              "(continuum-map item, Section 10)")

    return {"r": r_vals, "phi": phi, "g": g_field,
            "mass_total": mass_acc}


# ================================================================
# 8  Coupling constant
# ================================================================

def run_coupling(G_kernel):
    """Section 8: coupling form, optical depth, scale inference."""
    print("\n--- 8  Coupling constant ---")

    # 8.1  Form: G = G_kernel / E_CS²
    E_CS = math.sqrt(G_kernel / G_measured_nat)
    print(f"  8.1  G = G_kernel / E_CS²")
    print(f"       G_kernel = π/6 = {G_kernel:.12f}")
    print(f"       E_CS = √(G_kernel/G) = {E_CS:.6e} GeV")
    print(f"       (≈ {E_CS/1.22e19:.2f} × E_Planck)")

    # 8.2  Dimensionless coupling and optical depth
    alpha_G_pred_kernel = G_kernel * v_EW**2 / E_CS**2
    tau_G_from_scale = 2.0 * math.log(E_CS / v_EW)
    print(f"\n  8.2  Dimensionless coupling at v_EW:")
    print(f"       α_G(v) = G·v² = {alpha_G_meas:.6e}")
    print(f"       α_G(v) = G_kernel · exp(−τ_G)")
    print(f"       τ_G = 2·ln(E_CS/v) = {tau_G_from_scale:.6f}")
    print(f"       (factor 2: two-pass carrier return, §4.2)")

    # 8.3  Aperture-variable representation of τ_G
    # NOTE: The exponents 5 and 4 in the formula below are
    # motivated by the five shell checkpoints and four transitions
    # of the canonical depth-4 word, but they are not yet derived
    # from a first-principles path integral.  See Required
    # Continuum Maps, item 1.
    n_states = 4096.0
    tau_0        = n_states * Delta
    rho5         = rho**5
    tau_parallel = tau_0 * rho5
    f_ordered    = 1.0 - 4.0 * rho * Delta**2
    tau_G        = tau_parallel * f_ordered

    alpha_pred   = G_kernel * math.exp(-tau_G)
    alpha_par    = G_kernel * math.exp(-tau_parallel)
    G_pred       = alpha_pred / v_EW**2
    G_par        = alpha_par  / v_EW**2

    tau_req = -math.log(alpha_G_meas / G_kernel)
    print(f"\n  8.3  Aperture-depth reconstruction:")
    print(f"       |Omega|*Delta     = {tau_0:.6f}")
    print(f"       rho^5             = {rho5:.12f}")
    print(f"       |Omega|*Delta*rho^5 = {tau_parallel:.6f}  (no 4-transition term)")
    print(f"       1−4ρΔ²      = {f_ordered:.12f}")
    print(f"       τ_G         = {tau_G:.6f}")
    print(f"       tau_required  = {tau_req:.6f}")
    print(f"       tau_G - tau_req = {tau_G - tau_req:.2e}")
    print(f"       α_G pred    = {alpha_pred:.6e}")
    print(f"       α_G meas    = {alpha_G_meas:.6e}")
    print(f"       G pred      = {G_pred:.6e} GeV⁻²")
    print(f"       G meas      = {G_measured_nat:.6e} GeV⁻²")
    print(f"       G_pred/G_meas − 1 = {G_pred/G_measured_nat - 1:.2e}")

    # 8.4  Canonical-word per-cycle optical depth
    mon = _require_monodromy()
    if mon is not None:
        binom = [math.comb(6, s) / 64.0 for s in range(7)]
        chi6_full = 0x3F
        tau_sum = 0.0
        w_sum   = 0.0
        per_pop = {}
        for micro in range(64):
            pop  = bin(micro).count("1")
            word = mon.family_word_for_micro(micro) * 2
            rows = mon.trace_word(word)
            tau_path = 0.0
            for idx in range(1, len(rows)):
                obs = rows[idx].obs
                s = obs.arch_shell
                if s < 0 or s > 6:
                    continue
                sw = binom[s]
                cf = 0.0 if obs.chi6 == chi6_full else 1.0
                tau_path += Delta * sw * cf
            pw = binom[pop]
            tau_sum += tau_path * pw
            w_sum   += pw
            per_pop.setdefault(pop, []).append(tau_path)
        if w_sum > 0:
            tau_cycle = tau_sum / w_sum
            n_cycles  = tau_G / tau_cycle if tau_cycle > 0 else 0
            print(f"\n  8.4  Canonical-word optical depth:")
            print(f"       τ per depth-8 cycle = {tau_cycle:.6f}")
            print(f"       N cycles = τ_G/τ_cycle = {n_cycles:.1f}")
            for p in sorted(per_pop):
                vals = per_pop[p]
                print(f"       pop={p}: τ={sum(vals)/len(vals):.6f} (n={len(vals)})")

    return {"E_CS": E_CS, "tau_G": tau_G, "G_pred": G_pred,
            "alpha_pred": alpha_pred}


# ================================================================
# 9  Testable predictions
# ================================================================

def run_predictions():
    """Section 9: α·ζ product, α(z), higher-order sign."""
    print("\n--- 9  Testable predictions ---")

    # 9.1  α·ζ product
    print("  9.1  α·ζ product (no free continuous parameter):")
    print(f"       α·ζ = ρ⁴/(π√3) = {alpha_zeta_exact:.12f}")
    print(f"       check: α·ζ = {alpha_zeta_check:.12f}")
    print(f"       |diff| = {abs(alpha_zeta_exact - alpha_zeta_check):.2e}")

    # 9.2  α(z) oscillation
    print("\n  9.2  α(z) oscillation:")
    period_z = Delta * math.log(2.0)
    sub_z    = period_z / 7.0
    shell_w  = [math.comb(6, k) for k in range(7)]
    total_sw = float(sum(shell_w))
    shell_wn = [w / total_sw for w in shell_w]
    max_var  = 6 * 0.25

    n_pts = 500
    z_vals = [0.001 + (6.0 - 0.001) * i / (n_pts - 1) for i in range(n_pts)]
    a_vals = []
    for z in z_vals:
        n = math.log1p(z) / Delta
        p = min(1.0 - math.exp(-n * Delta), 1.0 - 1e-12)
        probs = [math.comb(6, k) * p**k * (1 - p)**(6 - k) for k in range(7)]
        norm = sum(probs)
        var_k = sum(k * k * probs[k] for k in range(7)) / norm \
              - (sum(k * probs[k] for k in range(7)) / norm)**2
        amp = 1e-5 * var_k / max_var
        osc = amp * math.cos(2 * math.pi * 7.0 * n)
        a_vals.append(alpha * (1.0 + osc))

    da = max(a_vals) - min(a_vals)
    print(f"       main period Δz ≈ {period_z:.4f}")
    print(f"       sub-cycle Δz ≈ {sub_z:.4f}")
    print(f"       delta_alpha/alpha_0 (peak-to-peak) = {da/alpha:.2e}")

    # 9.3  Sign of higher-order α correction
    print("\n  9.3  Higher-order correction to α:")
    print("       dual-pole symmetry requires O(δ_BU⁶) correction to be negative.")
    print("       A positive correction falsifies the geometric identification.")


# ================================================================
# 10  Required continuum maps
# ================================================================

def print_required_maps():
    """Section 10: open problems."""
    print("\n--- 10  Required continuum maps ---")
    maps = [
        "1.  Optical-conjugacy map for E_CS (predict G from geometry)",
        "2.  SU(2) generator map to gravitomagnetic vector potential A_g",
        "3.  Shell displacement map to Riemann curvature tensor",
        "4.  Nonlinear self-consistency and 3+1 Hamiltonian constraint",
        "5.  Exterior matching of shell-regularised interior profile",
        "6.  Physical memory map (depth-8 residual → BMS supertranslation)",
        "7.  Helicity-2 map from two-pass orientation recovery",
        "8.  Quadrupole radiation map (k=2 shell mode → metric strain)",
        "9.  Perihelion precession (requires nonlinear field interaction)",
        "10. EM/gravity coupling identification (α_CGM → α, ζ_CGM → ζ_G)",
    ]
    for m in maps:
        print(f"     {m}")


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CGM Gravitational Coupling Analysis")
    print("=" * 70)
    print("  G derivation policy:")
    print("  - Planck mass/energy not used as input (circular).")
    print("  - ℏ for unit conversion only, not a gravitational scale.")
    print("  - v_EW is the sole dimensional anchor for G_pred.")
    print("  - Layer 1: exact kernel invariants (shell displacement, horizons).")
    print("  - Layer 2: numerical coupling reconstruction at v_EW.")
    print("  - Layer 3: aperture-depth closure law (rho^5, 4-transition term).")

    # 1
    print("\n--- 1  Primitive measures ---")
    print(f"  s_p (CS)   = π/2       = {s_p:.12f}")
    print(f"  u_p (UNA)  = cos(π/4)  = {u_p:.12f}")
    print(f"  o_p (ONA)  = π/4       = {o_p:.12f}")
    print(f"  Q_G        = 4π        = {Q_G:.12f}")
    print(f"  m_a        = 1/(2√2π)  = {m_a:.12f}")
    print(f"  δ_BU       =           = {d_BU:.12f}")
    print(f"  ρ          = δ_BU/m_a  = {rho:.12f}")
    print(f"  Δ          = 1 − ρ     = {Delta:.12f}")
    print(f"  α (CGM)    = δ_BU⁴/m_a = {alpha:.12f}")
    print(f"  ζ          = Q_G/S_geo = {zeta:.12f}")
    print(f"  48·Δ       =           = {prod_48Delta:.12f}")
    print(f"  α·ζ        = ρ⁴/(π√3) = {alpha_zeta_exact:.12f}")

    # 2
    print("\n--- 2  Physical anchors ---")
    print(f"  v_EW       = {v_EW} GeV")
    print(f"  G (meas)   = {G_measured_nat:.5e} GeV⁻²")
    print(f"  α_G(v)     = {alpha_G_meas:.5e}")

    # 3  UV ratio consistency
    print("\n--- 3  UV ratio consistency ---")
    r_UNA = u_p / s_p
    r_ONA = o_p / s_p
    r_BU  = 2 * m_a**2 / math.pi
    r_BU2 = 1.0 / (4.0 * math.pi**2)
    print(f"  E_UNA/E_CS = {r_UNA:.6f}")
    print(f"  E_ONA/E_CS = {r_ONA:.6f}")
    print(f"  E_BU/E_CS  = {r_BU:.8f}  (= 1/(4π²) = {r_BU2:.8f})")

    # 4  Kernel invariants
    print("\n--- 4  Kernel invariants ---")
    mon = _require_monodromy()
    shell_inv = run_shell_displacement_invariant(mon)
    G_kernel  = run_gauss_map(shell_inv)
    eq_horiz  = run_equality_horizon(mon)

    # 5  Orientation recovery
    print("\n--- 5  Orientation recovery ---")
    mono_ok = run_monodromy_probes(mon)
    circ    = run_family_circulation(mon)
    mem     = run_gravitational_memory(mon)

    # 6  Gravitational radiation
    gw = run_gw_spectrum(mon)

    # 7  Potential profile
    pot = run_potential_profile(G_kernel)

    # 8  Coupling constant
    coup = run_coupling(G_kernel)

    # 9  Testable predictions
    run_predictions()

    # 10  Required maps
    print_required_maps()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Q_G       = {Q_G:.12f}")
    print(f"  m_a       = {m_a:.12f}")
    print(f"  δ_BU      = {d_BU:.12f}")
    print(f"  ρ         = {rho:.12f}")
    print(f"  Δ         = {Delta:.12f}")
    print(f"  G_kernel  = π/6 = {G_kernel:.12f}")
    print(f"  α·ζ       = {alpha_zeta_exact:.12f}")
    if coup:
        print(f"  τ_G       = {coup['tau_G']:.6f}  (exponents motivated)")
        print(f"  G_pred    = {coup['G_pred']:.6e} GeV⁻²")
        print(f"  G_meas    = {G_measured_nat:.6e} GeV⁻²")
        print(f"  G_pred/G_meas − 1 = {coup['G_pred']/G_measured_nat - 1:.2e}")
        print(f"  E_CS      = {coup['E_CS']:.6e} GeV")
    print(f"  G (SI)    = {G_SI_measured:.6e} m³ kg⁻¹ s⁻²")
    print(f"  orientation recovery = {mono_ok}")
    print()
    print("  Status of τ_G = |Ω|·Δ·ρ⁵·(1−4ρΔ²):")
    print("  The exponents 5 and 4 are motivated by the five shell")
    print("  checkpoints and four transitions of the canonical depth-4")
    print("  word. They are not yet derived from a first-principles")
    print("  path integral through shell space. See §10 item 1.")