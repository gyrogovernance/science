#!/usr/bin/env python3
"""
cgm_gravity_analysis_1.py
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
  - Matched: τ_G closure law, v_EW anchor
  - Phenomenological: α(z) oscillation, α·ζ product
"""
import math
import cmath
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_rp = str(_REPO_ROOT)
if _rp not in sys.path:
    sys.path.insert(0, _rp)

from gyroscopic.aQPU.constants import (
    APERTURE_GAP,
    CHIRALITY_MASK_6,
    DELTA_BU,
    M_A,
    RHO,
    popcount,
)
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

# Aperture ladder (canonical values: gyroscopic/aQPU/constants.py)
m_a   = M_A                   # aperture scale; satisfies m_a² = s_p/(4π²)
d_BU  = DELTA_BU              # BU dual-pole monodromy δ_BU
rho   = RHO                   # closure ratio δ_BU/m_a
Delta = APERTURE_GAP          # aperture gap 1 − ρ
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

    r_grid = np.asarray(r_ext, dtype=float)
    g_vals = np.asarray(g_ext, dtype=float)

    # Far-field inverse-square verification
    rf = r_grid[r_grid > 1.0]
    gf = g_vals[r_grid > 1.0]

    gr2 = np.abs(gf) * rf ** 2
    gr2_mean = float(np.mean(gr2))
    gr2_rel_std = float(np.std(gr2)) / gr2_mean

    log_rf = np.log(rf)
    log_g = np.log(np.abs(gf))
    exponent = float(np.polyfit(log_rf, log_g, 1)[0])

    flux_vals = 4.0 * np.pi * rf ** 2 * gf
    flux_mean = float(np.mean(flux_vals))
    flux_expected = (-Q_G * G_kernel) * mass_acc

    print("  Far-field inverse-square check:")
    print(f"    g*r^2 relative std: {gr2_rel_std:.6e}")
    print(f"    Power-law exponent: {exponent:.6f} (expected -2.000000)")
    print(f"    Exponent error: {abs(exponent - (-2.0)):.2e}")
    print(f"    Continuum integrated mass M = {mass_acc:.6f}")
    print(f"    Exterior flux: {flux_mean:.6f} "
          f"(expected {flux_expected:.6f} = M * (-Q_G * G_kernel))")
    print(f"    Flux ratio: {flux_mean / flux_expected:.6f}")

    return {"r": r_vals, "phi": phi, "g": g_field,
            "mass_total": mass_acc}


# ================================================================
# 7b  Gauss law bridge: shell displacement -> Poisson equation
# ================================================================

def run_gauss_law_bridge(G_kernel):
    """
    Bridge: shell displacement invariant -> continuous Poisson equation.

    Shows:

      1. Shell displacement invariant D = 24 is a discrete Gauss law
         with total flux Q_G = 4*pi and G_kernel = Q_G / D = pi/6.

      2. Radial embedding of the 7 shells gives a discrete field
         g(r) = -G_kernel M(r) / r^2 whose boundary flux equals
         -Q_G * G_kernel.

      3. The divergence theorem then yields div g = -Q_G G rho_A
         in the continuum limit.
    """
    print("\n" + "=" * 10)
    print("GAUSS LAW BRIDGE: Shell Displacement -> Poisson Equation")
    print("=" * 10)

    n_shells = 7
    binom = [math.comb(6, k) for k in range(n_shells)]
    total_binom = float(sum(binom))

    # Step 1: discrete Gauss law (D = 24, G_kernel = Q_G / 24)
    print("\nStep 1: Discrete Gauss Law")
    print("  Shell displacement invariant: D = 24 (any closed cycle, any mass)")
    print(f"  Kernel Gauss map: G_kernel = Q_G / D = 4*pi / 24 = pi/6 = {G_kernel:.12f}")
    print("  Total flux per cycle: D * G_kernel = Q_G = 4*pi")

    mon = _require_monodromy()
    if mon is not None:
        displacements = []
        for micro in range(64):
            word = mon.family_word_for_micro(micro) * 2  # depth-8
            rows = mon.trace_word(word)
            shells = [r.obs.arch_shell for r in rows]
            if len(shells) == 9:
                deltas = [abs(shells[i] - shells[i - 1])
                          for i in range(1, len(shells))]
                displacements.append(sum(deltas))
        unique_d = sorted(set(displacements))
        n_pass = sum(1 for d in displacements if d == 24)
        print(f"  Monodromy check: D values = {unique_d},  D=24 count = {n_pass}/64")

    # Step 2: radial embedding and boundary flux
    print("\nStep 2: Radial Embedding and Boundary Flux")

    r_inner = [max(0.0, (k - 0.5) / 6.0) for k in range(n_shells)]
    r_outer = [(k + 0.5) / 6.0 for k in range(n_shells)]
    r_center = [k / 6.0 for k in range(n_shells)]

    shell_vol = [(4.0 / 3.0) * math.pi * (r_outer[k]**3 - r_inner[k]**3)
                 for k in range(n_shells)]
    rho_shell = [binom[k] / (total_binom * shell_vol[k]) for k in range(n_shells)]

    M_enc = [0.0] * n_shells
    for k in range(n_shells):
        M_enc[k] = sum(binom[j] / total_binom for j in range(k + 1))

    g_field = [0.0] * n_shells
    for k in range(n_shells):
        r = r_outer[k]
        if r > 1e-10:
            g_field[k] = -G_kernel * M_enc[k] / (r * r)

    flux = [g_field[k] * Q_G * r_outer[k]**2 for k in range(n_shells)]

    print("  k   r_k    rho(r_k)     M(r_k)   g(r_k)    Flux(r_k)")
    print("  --- ------ ---------- -------- --------- -----------")
    for k in range(n_shells):
        print(f"  {k}   {r_center[k]:.4f} {rho_shell[k]:.4f}     "
              f"{M_enc[k]:.4f}   {g_field[k]:.6f} {flux[k]:.6f}")

    print(f"\n  Flux at outer boundary: {flux[-1]:.6f}")
    print(f"  Expected -Q_G * G_kernel: {-Q_G * G_kernel:.6f}")
    print(f"  Ratio: {flux[-1] / (-Q_G * G_kernel):.6f}")

    # Step 3: continuum divergence law (analytic)
    print("\nStep 3: Continuum Limit (analytic)")
    print("  In spherical symmetry: div g = (1/r^2) d(r^2 g)/dr")
    print("  From discrete Gauss law: r^2 g(r) = -G_kernel M(r)")
    print("  With dM/dr = Q_G r^2 rho(r), we obtain:")
    print("    div g = -Q_G * G_kernel * rho")
    print("  With G = G_kernel / E_CS^2 and rho_A the ancestry density:")
    print("    div g = -Q_G * G * rho_A")
    print("\n  This is the CGM field equation in differential form.")

    return {
        "M_enc": M_enc,
        "g_field": g_field,
        "flux_boundary": flux[-1] if flux else None,
    }


# ================================================================
# 7c  Ancestry stress from 6-bit payload
# ================================================================


def _mat_zero() -> list[list[float]]:
    return [[0.0] * 3 for _ in range(3)]


def _vec_zero() -> list[float]:
    return [0.0, 0.0, 0.0]


def _vec_add_weighted(acc: list[float], w: float, v: tuple[float, float, float]) -> None:
    acc[0] += w * v[0]
    acc[1] += w * v[1]
    acc[2] += w * v[2]


def _pc6(micro: int) -> int:
    return popcount(int(micro) & CHIRALITY_MASK_6)


def translational_activation(micro: int) -> tuple[int, int, int]:
    """(P_x, P_y, P_z) bits 6,5,4 of the 6-bit payload, returned as 0/1."""
    bx = (micro >> 5) & 1
    by = (micro >> 4) & 1
    bz = (micro >> 3) & 1
    return (bx, by, bz)


def translational_vector(micro: int) -> tuple[float, float, float]:
    """
    Absolute translational activation from bits 6,5,4 of the 6-bit payload.
    bit=0 -> 0 (no activation)
    bit=1 -> 1 (generator active)
    This is an occupancy pattern of the three translational generators, not a
    metric displacement in space. The anisotropy ratio (trace-free Frobenius
    norm over full Frobenius norm) is invariant under affine recodings of bit
    polarity (for example 0/1 versus plus/minus 1): covariance is unchanged by
    constant shifts and scales uniformly under independent sign flips.
    """
    bx, by, bz = translational_activation(micro)
    return (float(bx), float(by), float(bz))


def translational_popcount(micro: int) -> int:
    """Number of activated translational bits (0 through 3)."""
    bx, by, bz = translational_activation(micro)
    return bx + by + bz


# Rotational payload bits do not enter sigma_A (translational covariance only).
# Kept for axis mapping consistency with the paper and for future J_A coupling.
def rotational_activation(micro: int) -> tuple[int, int, int]:
    """(J_x, J_y, J_z) from bits 3,2,1."""
    return ((micro >> 2) & 1, (micro >> 1) & 1, micro & 1)


def ancestry_stress_weighted(weights: dict[int, float]) -> list[list[float]]:
    """
    Translational ancestry stress as a centered second moment (covariance):
      sigma_ij = <v_i v_j> - <v_i><v_j>
    where v is the absolute translational activation vector (0/1 per
    component) extracted from payload bits 6,5,4.

    Continuum scaling: sigma_A^ij(physical) = sigma_A^ij(kernel) * E_CS^4
    * (geometric correction).
    """
    sumw = sum(float(weights.get(m, 0.0)) for m in range(64))
    if sumw <= 0.0:
        return _mat_zero()

    mu = _vec_zero()
    for micro in range(64):
        w = float(weights.get(micro, 0.0)) / sumw
        _vec_add_weighted(mu, w, translational_vector(micro))

    acc = _mat_zero()
    for micro in range(64):
        w = float(weights.get(micro, 0.0)) / sumw
        vx, vy, vz = translational_vector(micro)
        dx, dy, dz = (vx - mu[0], vy - mu[1], vz - mu[2])
        for i, di in enumerate((dx, dy, dz)):
            for j, dj in enumerate((dx, dy, dz)):
                acc[i][j] += w * di * dj
    return acc


def _trace(s: list[list[float]]) -> float:
    return s[0][0] + s[1][1] + s[2][2]


def _fro_sq(s: list[list[float]]) -> float:
    t = 0.0
    for row in s:
        for v in row:
            t += v * v
    return t


def _tracefree_norm(s: list[list[float]]) -> float:
    tr = _trace(s) / 3.0
    u = [[s[i][j] - (tr if i == j else 0.0) for j in range(3)] for i in range(3)]
    return math.sqrt(_fro_sq(u))


def _fro_norm(s: list[list[float]]) -> float:
    return math.sqrt(_fro_sq(s))


def _alpha_z_oscillation_params() -> tuple[float, float, float]:
    """Return (period_ln1z, amplitude_A, peak_to_peak) from binomial shell weights."""
    period_ln1z = APERTURE_GAP * math.log(2)
    binom_weights = [math.comb(6, k) / 64.0 for k in range(7)]
    mean_bw = sum(binom_weights) / 7.0
    var_bw = sum((w - mean_bw) ** 2 for w in binom_weights) / 7.0
    rel_var = var_bw / mean_bw ** 2
    amp_a = rel_var * APERTURE_GAP ** 2
    return period_ln1z, amp_a, 2.0 * amp_a


def _print_sigma(label: str, s: list[list[float]]) -> None:
    print(f"{label}")
    print("  sigma = [")
    for i in range(3):
        print(f"           [{s[i][0]:.6f}, {s[i][1]:.6f}, {s[i][2]:.6f}],")
    print("          ]")


def run_ancestry_stress():
    """
    Translational stress as centered second moment of absolute activation;
    sanity checks, shell decomposition, Hamiltonian constraint source.

    Bit-to-axis mapping (reference implementation):
      payload bit 6 -> P_x  (shift 5)
      payload bit 5 -> P_y  (shift 4)
      payload bit 4 -> P_z  (shift 3)
      payload bit 3 -> J_x  (shift 2)
      payload bit 2 -> J_y  (shift 1)
      payload bit 1 -> J_z  (shift 0)

    The ancestry stress sigma_A captures only the symmetric translational
    contribution. The antisymmetric rotational contribution (frame-dragging)
    appears in the gravitomagnetic sector through B_g = curl(A_g).
    The full source structure (rho_A, J_A, sigma_A) is consistent with
    the ADM decomposition of T_mu_nu into three 3D objects.
    """

    print("\n--- Ancestry stress tensor ---")

    print("\nMapping: bits 1-3 -> rotational (J_x, J_y, J_z) payload order")
    print("         bits 4-6 -> translational (P_x, P_y, P_z) payload order")
    print("Convention: bit=0 -> 0 (no activation), bit=1 -> 1 (activated)")
    print("Anisotropy ratio is invariant under affine recodings of bit polarity")
    print("(0/1 vs +/-1): covariance is shift-invariant; sign flips scale it uniformly.")

    # Test 1: isotropic ensemble.
    iso = ancestry_stress_weighted({m: 1.0 for m in range(64)})
    tr_i = _trace(iso)
    tf_i = _tracefree_norm(iso)
    _print_sigma("Isotropic case (uniform micro weights):", iso)
    print(f"  Trace: Tr(sigma) = {tr_i:.6f}")
    print(f"  Trace-free Frobenius norm ||sigma tilde|| = {tf_i:.6e}")
    print("  Expected: pure trace (trace-free norm = 0)")
    print(f"  Analytic: each bit on with p=1/2, Var = 1/4, Tr = 3/4 = {3.0/4.0:.6f}")
    print("  Kernel: isotropic Tr(sigma) = 3/4; Tr(sigma)/3 = 1/4 in these units.")
    print("  Physical stress: sigma_physical = sigma_kernel * E_CS^4 * (geometric factor).")

    # --- Ancestry current J_A (mean signed translational activation) ---
    print("\nAncestry current J_A (mean signed translational activation, +/-1 map):")
    print("  J_A,i = <v_i> where v_i = +1 if bit on, -1 if bit off")

    def translational_vector_pm1(micro: int) -> tuple[float, float, float]:
        bx, by, bz = translational_activation(micro)
        return (2.0 * bx - 1.0, 2.0 * by - 1.0, 2.0 * bz - 1.0)

    j_iso = _vec_zero()
    for micro in range(64):
        vx, vy, vz = translational_vector_pm1(micro)
        j_iso[0] += vx / 64.0
        j_iso[1] += vy / 64.0
        j_iso[2] += vz / 64.0
    print(f"  Isotropic J_A = ({j_iso[0]:.6f}, {j_iso[1]:.6f}, {j_iso[2]:.6f})")
    print("  Expected: (0, 0, 0) (each bit symmetric about zero)")

    print("  Shell  J_A,x    J_A,y    J_A,z    |J_A|")
    print("  -----  ------   ------   ------   ------")
    for k in range(7):
        inds = [m for m in range(64) if _pc6(m) == k]
        if not inds:
            continue
        j_k = _vec_zero()
        nk = len(inds)
        for m in inds:
            vx, vy, vz = translational_vector_pm1(m)
            j_k[0] += vx / nk
            j_k[1] += vy / nk
            j_k[2] += vz / nk
        jmag = math.sqrt(j_k[0] ** 2 + j_k[1] ** 2 + j_k[2] ** 2)
        print(f"  {k}      {j_k[0]:.6f}  {j_k[1]:.6f}  {j_k[2]:.6f}  {jmag:.6f}")

    # Monodromy-visit weights: count payload micro refs along canonical depth-8 word.
    mon = _require_monodromy()
    sigma_mono = None
    if mon is not None:
        ref = mon.FAMILY_RAY_REF
        word = mon.family_word_for_micro(ref) * 2
        rows = mon.trace_word(word)
        visit_w: dict[int, float] = {}
        for row in rows[1:]:
            if row.byte is None:
                continue
            intron = mon.byte_to_intron(row.byte)
            m = mon.intron_micro_ref(intron)
            visit_w[m] = visit_w.get(m, 0.0) + 1.0
        sigma_mono = ancestry_stress_weighted(visit_w)
        tr_m = _trace(sigma_mono)
        fn_m = _fro_norm(sigma_mono)
        tf_m = _tracefree_norm(sigma_mono)
        ar_m = (tf_m / fn_m) if fn_m > 1e-30 else 0.0
        n_steps = int(sum(visit_w.values()))
        n_distinct = len(visit_w)
        print(f"\nMonodromy-visit weighted stress (canonical depth-8 word, ref micro={ref}):")
        print(f"  Visit counts along trace: {n_steps} steps; {n_distinct} distinct payload micro(s).")
        if n_distinct <= 1:
            print("  Note: one payload ref repeated every step; centered covariance is zero.")
        _print_sigma("  sigma (visit-frequency weights, then normalized):", sigma_mono)
        print(f"  Tr(sigma) = {tr_m:.6f}  ||sigma~||/||sigma|| = {ar_m:.6f}")

        pooled: dict[int, float] = {}
        total_pooled_steps = 0
        for m in range(64):
            wdm = mon.family_word_for_micro(m) * 2
            rdm = mon.trace_word(wdm)
            for row in rdm[1:]:
                if row.byte is None:
                    continue
                intron_p = mon.byte_to_intron(row.byte)
                mp = mon.intron_micro_ref(intron_p)
                pooled[mp] = pooled.get(mp, 0.0) + 1.0
                total_pooled_steps += 1
        sigma_pooled = ancestry_stress_weighted(pooled)
        tr_p = _trace(sigma_pooled)
        fn_p = _fro_norm(sigma_pooled)
        tf_p = _tracefree_norm(sigma_pooled)
        ar_p = (tf_p / fn_p) if fn_p > 1e-30 else 0.0
        print("\nPooled monodromy visit weights (depth-8 word for each micro m=0..63):")
        print(f"  Total step visits = {total_pooled_steps} (64 rays x 8 steps).")
        _print_sigma("  sigma:", sigma_pooled)
        print(f"  Tr(sigma) = {tr_p:.6f}  ||sigma~||/||sigma|| = {ar_p:.6f}")
    else:
        print("\nMonodromy-visit weighted stress: skipped (monodromy module unavailable).")
        sigma_pooled = None

    # Test 2: translational bx=1 constrained ensemble (32 micros).
    beam = {m: 1.0 for m in range(64) if ((m >> 5) & 1) == 1}
    bs = ancestry_stress_weighted(beam)
    _print_sigma("\nTranslational constrained ensemble (all micros with bit 6 = 1):", bs)
    print("  Expected: sigma_xx = 0 (pinned), sigma_yy = sigma_zz = 1/4")

    # Test 3: single microstate => delta distribution => zero covariance.
    all_trans_active_micro = (1 << 3) | (1 << 4) | (1 << 5)
    ss = ancestry_stress_weighted({all_trans_active_micro: 1.0})
    _print_sigma("\nSingle-vector configuration (bits 4,5,6 all on):", ss)
    print("  Expected: zero (covariance of a single point)")

    # Test 4: vacuum micro (all translational bits off).
    vac_micro = 0
    sv = ancestry_stress_weighted({vac_micro: 1.0})
    _print_sigma("\nVacuum configuration (all translational bits off):", sv)
    print("  Expected: zero (single point, no activation)")

    # --- Anisotropy spectrum by translational popcount ---
    print("\nAnisotropy spectrum (by translational popcount tpop):")
    print("  tpop  ||sigma~||/||sigma||  Tr(sigma)  Physical analog")
    print("  ----  ----------------------  ----------  -------------------------")
    analogs = {
        0: "tpop=0: v fixed at 0 (zero variance)",
        1: "heuristic: jet-like anisotropy",
        2: "heuristic: complement shell anisotropy",
        3: "tpop=3: v fixed at 1 (zero variance)",
    }
    for tpop in range(4):
        inds = [m for m in range(64) if translational_popcount(m) == tpop]
        if not inds:
            continue
        weights_t = {m: 1.0 for m in inds}
        s_t = ancestry_stress_weighted(weights_t)
        fn = _fro_norm(s_t)
        tr_t = _trace(s_t)
        if fn <= 1e-30:
            ratio = 0.0
        else:
            ratio = _tracefree_norm(s_t) / fn
        print(f"  {tpop}     {ratio:.6f}             {tr_t:.6f}    "
              f"{analogs.get(tpop, '')}")

    # --- Anisotropy spectrum by full 6-bit popcount ---
    print("\nAnisotropy spectrum (by full 6-bit popcount):")
    print("  pop   ||sigma~||/||sigma||  Tr(sigma)")
    print("  ----  ----------------------  ----------")
    for pop in range(7):
        inds = [m for m in range(64) if _pc6(m) == pop]
        if not inds:
            continue
        weights_pop = {m: 1.0 for m in inds}
        s_pop = ancestry_stress_weighted(weights_pop)
        fn = _fro_norm(s_pop)
        tr_pop = _trace(s_pop)
        if fn <= 1e-30:
            ratio = 0.0
        else:
            ratio = _tracefree_norm(s_pop) / fn
        print(f"  {pop}     {ratio:.6f}             {tr_pop:.6f}")

    # --- Shell-weighted stress (per mass shell) ---
    print("\nShell-weighted stress (per payload popcount shell):")
    print("  shell  weight      Tr(sigma)  ||sigma~||  anisotropy")
    print("  -----  ----------  ---------  ----------  ----------")
    binom_shell = [math.comb(6, k) / 64.0 for k in range(7)]
    total_binom = sum(binom_shell)
    for k in range(7):
        inds = [m for m in range(64) if _pc6(m) == k]
        if not inds:
            continue
        w_shell = binom_shell[k] / total_binom
        weights_s = {m: 1.0 for m in inds}
        s_k = ancestry_stress_weighted(weights_s)
        tr_k = _trace(s_k)
        tf_k = _tracefree_norm(s_k)
        fn_k = _fro_norm(s_k)
        aniso_k = tf_k / fn_k if fn_k > 1e-30 else 0.0
        print(f"  {k}      {w_shell:.6f}    {tr_k:.6f}    "
              f"{tf_k:.6f}    {aniso_k:.6f}")
        if 1 <= k <= 5:
            print(f"    off-diag: sigma_xy={s_k[0][1]:.6f}, "
                  f"sigma_xz={s_k[0][2]:.6f}, sigma_yz={s_k[1][2]:.6f}")

    # --- Mass-weighted average stress ---
    # Each micro m with popcount k gets weight binom_shell[k] / C(6,k) = 1/64.
    # So the mass-weighted ensemble is uniform over all 64 micros, giving
    # the isotropic result. This is a consistency check.
    mass_weights = {}
    for m in range(64):
        pop = _pc6(m)
        c6k = math.comb(6, pop)
        mass_weights[m] = binom_shell[pop] / c6k if c6k > 0 else 0.0
    s_mass = ancestry_stress_weighted(mass_weights)
    tr_mass = _trace(s_mass)
    tf_mass = _tracefree_norm(s_mass)
    _print_sigma("\nMass-weighted average stress:", s_mass)
    print(f"  Tr(sigma) = {tr_mass:.6f}")
    print(f"  ||sigma tilde|| = {tf_mass:.6e}")
    print(f"  Consistency check: mass-weighted == isotropic? "
          f"{abs(tr_mass - tr_i) < 1e-10 and tf_mass < 1e-10}")

    print("\nShell mixture source (kernel units): source_k = rho_k + Tr(sigma_k)/3")
    print("  rho_k = C(6,k)/64; sigma_k uniform within shell k")
    print("  shell  rho_k       Tr(sigma)/3  source_k")
    print("  -----  ---------------  -----------  --------------------")
    ham_shell_mix = 0.0
    for k in range(7):
        inds = [m for m in range(64) if _pc6(m) == k]
        if not inds:
            continue
        rho_k = binom_shell[k] / total_binom
        weights_s = {m: 1.0 for m in inds}
        s_k = ancestry_stress_weighted(weights_s)
        tr_third = _trace(s_k) / 3.0
        source_k = rho_k + tr_third
        ham_shell_mix += rho_k * source_k
        print(f"  {k}      {rho_k:.6f}         {tr_third:.6f}      {source_k:.6f}")
    print(f"  sum_k rho_k * source_k = {ham_shell_mix:.6f}")

    # --- Continuum scaling ---
    G_kernel_local = Q_G / 24.0
    E_CS_local = math.sqrt(G_kernel_local / G_measured_nat)
    print("\nContinuum scaling:")
    print(f"  sigma_A^ij(physical) = sigma_A^ij(kernel) * E_CS^4 * (geometric factor)")
    print(f"  E_CS = {E_CS_local:.6e} GeV")
    print(f"  E_CS^4 = {E_CS_local**4:.6e} GeV^4")
    print(f"  Isotropic Tr(sigma_kernel) = {tr_i:.6f}")
    print(f"  Isotropic Tr(sigma_physical) = {tr_i:.6f} * {E_CS_local**4:.6e} * (correction)")

    out = {"isotropic_sigma": iso,
           "beam_ensemble_sigma": bs,
           "all_trans_active_micro": all_trans_active_micro,
           "mass_weighted_sigma": s_mass}
    if sigma_mono is not None:
        out["monodromy_ref_visit_sigma"] = sigma_mono
    if mon is not None and sigma_pooled is not None:
        out["monodromy_pooled_visit_sigma"] = sigma_pooled
    return out


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
    print(f"       (factor 2: two-pass carrier return, Section 4.2)")

    # 8.3  Aperture-variable representation of τ_G
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
    period_ln1z, amp_a, peak_to_peak = _alpha_z_oscillation_params()
    sub_z = period_ln1z / 7.0
    print(f"       main period Delta z (small z) ~ {period_ln1z:.4f}")
    print(f"       sub-cycle Delta z ~ {sub_z:.4f}")
    print(f"       delta_alpha/alpha_0 (peak-to-peak) ~ {peak_to_peak:.2e} "
          f"(2 x A, A = {amp_a:.2e})")

    # 9.3  Sign of higher-order α correction
    print("\n  9.3  Higher-order correction to α:")
    print("       dual-pole symmetry requires O(δ_BU⁶) correction to be negative.")
    print("       A positive correction falsifies the geometric identification.")


def run_alpha_z_oscillation_params():
    """Explicit alpha(z) period and amplitude from aperture and shell weights."""
    print("\n--- Alpha(z) oscillation parameters ---")

    period_ln1z, amp_a, peak_to_peak = _alpha_z_oscillation_params()
    print(f"  Period in ln(1+z): {period_ln1z:.6f}")
    print(f"  Period in z (small z): ~ {period_ln1z:.6f}")

    sub_period_ln1z = period_ln1z / 7.0
    print(f"  Sub-cycle period in ln(1+z): {sub_period_ln1z:.6f}")

    binom_weights = [math.comb(6, k) / 64.0 for k in range(7)]
    mean_bw = sum(binom_weights) / 7.0
    var_bw = sum((w - mean_bw) ** 2 for w in binom_weights) / 7.0
    rel_var = var_bw / mean_bw ** 2
    print(f"  Binomial shell weight variance: {var_bw:.6e}")
    print(f"  Relative variance (var/mean^2): {rel_var:.6f}")
    print(f"  Predicted fractional amplitude A ~ {amp_a:.2e} (leading order in Delta^2)")
    print(f"  Peak-to-peak Delta(alpha)/alpha_0 ~ {peak_to_peak:.2e} (2 x A)")

    print("  Prediction: alpha(z)/alpha_0 = 1 + A * sin(2*pi * ln(1+z) / P)")
    print(f"    P = {period_ln1z:.6f}")
    print(f"    A ~ {amp_a:.2e} (leading order in Delta^2)")
    print("  Falsification: absence of oscillation with period P at 3-sigma")
    print(f"    confidence in a survey spanning ln(1+z) range >= {period_ln1z:.2f}.")


def run_holographic_entropy():
    """Bulk vs boundary entropy from |Omega| and |H|."""
    print("\n--- Holographic entropy relation ---")

    s_bulk = math.log(4096)
    s_boundary = math.log(64)
    ratio_s = s_bulk / s_boundary

    print("  |Omega| = 4096, |H| = 64")
    print(f"  S_bulk = ln(|Omega|) = {s_bulk:.6f} = 12 * ln(2)")
    print(f"  S_boundary = ln(|H|) = {s_boundary:.6f} = 6 * ln(2)")
    print(f"  S_bulk / S_boundary = {ratio_s:.6f}")
    print("  Relation: S_bulk = 2 * S_boundary")
    print("  The factor 2 is the two-pass carrier-return count (Section 7.2).")
    print("  In continuum terms: S = 2 * A / (4 * ln(2)), where A = ln|H|.")
    print("  This identifies the holographic entropy constant with the spin-2")
    print("  structure of the gravitational sector.")


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":

    print("=" * 10)
    print("CGM Gravitational Coupling Analysis")
    print("=" * 10)
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

    # 7b  Gauss law bridge
    gauss = run_gauss_law_bridge(G_kernel)

    run_ancestry_stress()

    # 8  Coupling constant
    coup = run_coupling(G_kernel)

    # 9  Testable predictions
    run_predictions()
    run_alpha_z_oscillation_params()
    run_holographic_entropy()

    # Summary
    print("\n" + "=" * 10)
    print("SUMMARY")
    print("=" * 10)
    print(f"  Q_G       = {Q_G:.12f}")
    print(f"  m_a       = {m_a:.12f}")
    print(f"  δ_BU      = {d_BU:.12f}")
    print(f"  ρ         = {rho:.12f}")
    print(f"  Δ         = {Delta:.12f}")
    print(f"  G_kernel  = π/6 = {G_kernel:.12f}")
    print(f"  α·ζ       = {alpha_zeta_exact:.12f}")
    if coup:
        print(f"  τ_G       = {coup['tau_G']:.6f}")
        print(f"  G_pred    = {coup['G_pred']:.6e} GeV⁻²")
        print(f"  G_meas    = {G_measured_nat:.6e} GeV⁻²")
        print(f"  G_pred/G_meas − 1 = {coup['G_pred']/G_measured_nat - 1:.2e}")
        print(f"  E_CS      = {coup['E_CS']:.6e} GeV")
    print(f"  G (SI)    = {G_SI_measured:.6e} m³ kg⁻¹ s⁻²")
    print(f"  orientation recovery = {mono_ok}")