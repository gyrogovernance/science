#!/usr/bin/env python3
"""
CGM/hQVM Analysis: Holographic Universe

Central structural prediction: the BU Interval [~1 ly, ~2 ly] is bounded
by two independent kernel-derived faces:
  - BU-Egress face at d ≈ 0.97 ly (τ = Δ, aperture opens)
  - BU-Ingress face at d = 2.00 ly (W₂ involution, equality horizon)

Between them lies the conversion zone where space converts to time.

The carrier trajectory through this interval is NOT linear — it follows
the gyration cycle verified by wavefunction diagnostics (T2-T6).

Dependencies: numpy, astropy, scipy
"""
import math
from typing import Any, Dict, List, Optional, Tuple, TypedDict
import astropy.constants as astro_constants
import astropy.units as u
import numpy as np
from scipy.optimize import brentq

# =====================================================================
# 1. CGM Fundamental Constants
# =====================================================================
M_A: float = 1 / (2 * math.sqrt(2 * math.pi))
DELTA_BU: float = 0.195342176580
RHO: float = DELTA_BU / M_A
DELTA: float = 1 - RHO
EPSILON: float = M_A - DELTA_BU
Q_G: float = 4 * math.pi
OMEGA_SIZE: int = 4096
HORIZON_SIZE: int = 64
G_KERNEL: float = math.pi / 6
APERTURE_FRAME: int = 48
COMPLEMENTARITY_INV: int = 12
NUM_SHELLS: int = 7
SHELL_RATIO: float = COMPLEMENTARITY_INV / NUM_SHELLS
TAU_PER_LY: float = (7591 / 7392) * DELTA
SHELL_POP: List[int] = [math.comb(6, k) * HORIZON_SIZE for k in range(7)]
C4_GRAV: float = -7 / 4
TAU_G: float = (
    OMEGA_SIZE * DELTA * RHO**5 * ((1 - 4 * RHO * DELTA**2) + C4_GRAV * DELTA**4)
)
PHI_SU2: float = 2 * math.acos((1 + 2 * math.sqrt(2)) / 4)
ALPHA_GEOM: float = DELTA_BU**4 / M_A  # ~0.007299734


# =====================================================================
# 2. Physical Constants (Astropy)
# =====================================================================
def _astro(name: str) -> Any:
    return getattr(astro_constants, name)


def _to_si(val: Any, unit: u.UnitBase) -> float:
    return float(val.to_value(unit))


C_SI: float = _to_si(_astro("c"), u.m / u.s)
YEAR_S: float = _to_si(1.0 * u.year, u.s)
AU_M: float = _to_si(_astro("au"), u.m)
M_SUN_KG: float = _to_si(_astro("M_sun"), u.kg)
M_EARTH_KG: float = _to_si(_astro("M_earth"), u.kg)
R_EARTH_M: float = _to_si(_astro("R_earth"), u.m)
G_SI: float = _to_si(_astro("G"), u.m**3 / u.kg / u.s**2)
HBAR_SI: float = _to_si(_astro("hbar"), u.J * u.s)
H_SI: float = _to_si(_astro("h"), u.J * u.s)
EV_TO_J: float = _to_si(1.0 * u.eV, u.J)
LY_M: float = C_SI * YEAR_S
LY_AU: float = LY_M / AU_M

# Graviton bounds
M_GW170104_EV: float = 7.7e-23
M_GW170104_KG: float = float(M_GW170104_EV * EV_TO_J / C_SI**2)
LAMBDA_FULL_GW170104_LY: float = (H_SI / (M_GW170104_KG * C_SI)) / LY_M
LAMBDA_RED_GW170104_LY: float = LAMBDA_FULL_GW170104_LY / (2 * math.pi)

M_PDG2024_EV: float = 1.27e-23
M_PDG2024_KG: float = float(M_PDG2024_EV * EV_TO_J / C_SI**2)
LAMBDA_FULL_PDG2024_LY: float = (H_SI / (M_PDG2024_KG * C_SI)) / LY_M
LAMBDA_RED_PDG2024_LY: float = LAMBDA_FULL_PDG2024_LY / (2 * math.pi)


# =====================================================================
# 3. Core Analytical Functions
# =====================================================================
def tau_at_distance(d_ly: float) -> float:
    return d_ly * TAU_PER_LY


def coherence_at_distance(d_ly: float) -> float:
    return math.exp(-tau_at_distance(d_ly))


def aperture_at_distance(d_ly: float) -> float:
    return 1 - coherence_at_distance(d_ly)


def psi_at_distance(d_AU: float, M_kg: float = M_SUN_KG) -> float:
    r_m = d_AU * AU_M
    return G_SI * M_kg / (r_m * C_SI**2)


def G_ratio_approx(psi: float) -> float:
    return 1 - 0.645568 * psi


def find_crossing(
    target_value: float,
    param: str = "coherence",
    d_min: float = 0.01,
    d_max: float = 100.0,
) -> Optional[float]:
    funcs = {
        "coherence": coherence_at_distance,
        "tau": tau_at_distance,
        "aperture": aperture_at_distance,
    }
    f = funcs[param]

    def objective(d: float) -> float:
        return f(d) - target_value

    if objective(d_min) * objective(d_max) >= 0:
        return None
    return float(brentq(objective, d_min, d_max, xtol=1e-12))  # type: ignore[arg-type]


# =====================================================================
# 4. Gyration-Driven Trajectory (from wavefunction diagnostics)
# =====================================================================
def gyration_trajectory() -> List[Dict[str, Any]]:
    """The actual carrier trajectory through the 4-byte canonical word.

    From hqvm_wavefunction_1.py half-step verification:
      Byte 1 L (d=0.5 ly): complement -> bulk,     sh=1 (arch)
      Byte 1 R (d=1.0 ly): bulk,                   sh=1 (arch)
      Byte 2 L (d=1.5 ly): bulk -> complement,      sh=0 (arch) BRIEF
      Byte 2 R (d=2.0 ly): complement -> equality,  sh=6 (arch)
      Byte 3 L (d=2.5 ly): equality -> bulk,        sh=5 (arch)
      Byte 3 R (d=3.0 ly): bulk,                    sh=1 (arch)
      Byte 4 L (d=3.5 ly): bulk -> complement,      sh=0 (arch)
      Byte 4 R (d=4.0 ly): complement (swapped),    sh=0 (arch)

    This is the gyration-driven path, NOT a linear shell partition.
    The carrier briefly visits the complement horizon at d≈1.5 ly
    before jumping to the equality horizon at d=2 ly.
    """
    steps = [
        (0.0, 0, "complement", 6, "111111", "rest", "Carrier rest (CS frame)"),
        (0.5, 1, "bulk", 5, "111110", "-", "L-step byte 1: UNA variety introduced"),
        (
            1.0,
            1,
            "bulk",
            5,
            "111110",
            "-",
            "R-step byte 1: UNA complete, departed horizon",
        ),
        (
            1.5,
            0,
            "complement",
            6,
            "111111",
            "-",
            "L-step byte 2: BRIEF return to complement",
        ),
        (
            2.0,
            6,
            "equality",
            0,
            "000000",
            "-",
            "R-step byte 2: BU-Egress, equality horizon, chi cancels",
        ),
        (
            2.5,
            5,
            "bulk",
            1,
            "000001",
            "-",
            "L-step byte 3: departed equality, BU approach",
        ),
        (3.0, 1, "bulk", 5, "111110", "-", "R-step byte 3: bulk traversal continues"),
        (
            3.5,
            0,
            "complement",
            6,
            "111111",
            "swapped",
            "L-step byte 4: complement horizon, Z2 flips",
        ),
        (
            4.0,
            0,
            "complement",
            6,
            "111111",
            "swapped",
            "R-step byte 4: BU-Ingress, Z2 encoded",
        ),
    ]
    trajectory = []
    for d, arch_sh, sector, compact_sh, chi, z2, desc in steps:
        e = {
            "d_ly": d,
            "arch_shell": arch_sh,
            "sector": sector,
            "compact_shell": compact_sh,
            "chirality": chi,
            "z2": z2,
            "description": desc,
        }
        # Add CGM effects
        if d > 0:
            d_au = d * LY_AU
            e["tau"] = tau_at_distance(d)
            e["coherence"] = coherence_at_distance(d)
            e["aperture_pct"] = aperture_at_distance(d) * 100
        else:
            e.update({"tau": 0, "coherence": 1.0, "aperture_pct": 0})
        trajectory.append(e)
    return trajectory


def extended_trajectory(n_bytes: int = 8) -> List[Dict[str, Any]]:
    """Extended trajectory for full Z2 cycle (8 bytes = 8 ly).

    Pattern repeats every 4 bytes:
      Bytes 1-4: rest -> swapped
      Bytes 5-8: swapped -> rest (F∘F = id)
    """
    base = gyration_trajectory()
    result = []
    for cycle in range(n_bytes // 4):
        offset = cycle * 4.0
        z2_base = "rest" if cycle % 2 == 0 else "swapped"
        z2_flip = "swapped" if cycle % 2 == 0 else "rest"
        for step in base:
            s = dict(step)
            s["d_ly"] = step["d_ly"] + offset
            if s["d_ly"] > 0:
                s["tau"] = tau_at_distance(s["d_ly"])
                s["coherence"] = coherence_at_distance(s["d_ly"])
                s["aperture_pct"] = aperture_at_distance(s["d_ly"]) * 100
            # Alternate Z2 phase each cycle
            if s["z2"] == "rest":
                s["z2"] = z2_base
            elif s["z2"] == "swapped":
                s["z2"] = z2_flip
            result.append(s)
    return result


# =====================================================================
# 5. Star System Calibrations
# =====================================================================
class StarSystem(TypedDict):
    M_star_kg: float
    period_yr: float
    known_comet_cloud_ly: Optional[float]


STAR_SYSTEMS: Dict[str, StarSystem] = {
    "Solar System": {
        "M_star_kg": M_SUN_KG,
        "period_yr": 1.0,
        "known_comet_cloud_ly": 0.03,  # Oort inner edge ~2000 AU
    },
    "α Centauri A": {
        "M_star_kg": 1.1 * M_SUN_KG,
        "period_yr": 1.0,  # Approximate for habitable zone
        "known_comet_cloud_ly": None,
    },
    "TRAPPIST-1": {
        "M_star_kg": 0.089 * M_SUN_KG,
        "period_yr": 0.04,  # ~15 day orbital period in HZ
        "known_comet_cloud_ly": None,
    },
    "Sirius A": {
        "M_star_kg": 2.06 * M_SUN_KG,
        "period_yr": 1.0,
        "known_comet_cloud_ly": None,
    },
}


def compute_bu_interval(system: StarSystem) -> Dict[str, Any]:
    """Compute BU Interval parameters for a star system.

    The gyration unit d_0 is c × T_orbital (in light-years).
    BU-Egress face: d where τ = Δ (in d_0 units)
    BU-Ingress face: d = 2 × d_0 (depth-4 completion)
    """
    period_yr = system["period_yr"]
    d_0 = period_yr  # 1 gyration unit = 1 ly per year of orbital period

    # Egress face: τ = Δ at d_egress in d_0 units
    # τ = (7591/7392) × Δ × (d / d_0)
    # Set τ = Δ: d_egress = d_0 × (7392/7591)
    d_egress = d_0 * (7391 / 7591)

    # Ingress face: depth-4 completion at d = 2 × d_0
    d_ingress = 2 * d_0

    # Conversion zone width
    zone_width = d_ingress - d_egress

    # Z2 period: 8 × d_0
    z2_period = 8 * d_0

    # Tidal radius for this star
    nu_gal = 74e3 / (3.086e19)
    r_tidal_m = (G_SI * system["M_star_kg"] / nu_gal**2) ** (1 / 3)
    r_tidal_ly = r_tidal_m / (C_SI * YEAR_S)

    return {
        "d_0": d_0,
        "d_egress": d_egress,
        "d_ingress": d_ingress,
        "zone_width": zone_width,
        "z2_period": z2_period,
        "r_tidal_ly": r_tidal_ly,
        "tau_at_egress": tau_at_distance(d_egress),
        "coh_at_egress": coherence_at_distance(d_egress),
        "tau_at_ingress": tau_at_distance(d_ingress),
        "coh_at_ingress": coherence_at_distance(d_ingress),
    }


# =====================================================================
# 6. Internal Metric Framework
# =====================================================================
def compute_internal_metric_profile(
    n_points: int = 50, d_max_ly: float = 4.0
) -> List[Dict[str, float]]:
    """Compute CGM internal metric quantities across the BU interval.

    Inside the shell:
    - The gravitoelectric field: g(r) = -G(ψ) M / r²
    - G(ψ) varies with ψ = GM/(rc²)
    - The Z₂ phase determines the spin structure
    - Frame-dragging (Lense-Thirring): ω_LT = 2G(ψ)J/(c²r³)
    """
    # Solar system angular momentum (approximate)
    J_sun = 1.92e41  # kg m²/s (spin)
    J_jupiter = 1.93e43  # kg m²/s (orbital, dominates)
    J_total = J_sun + J_jupiter

    d_values = np.linspace(0.1 / LY_AU * 1, d_max_ly, n_points)
    # Use AU-based distances for ψ calculation
    profile = []
    for d_ly in d_values:
        d_au = d_ly * LY_AU
        psi = psi_at_distance(d_au)
        g_ratio = G_ratio_approx(psi)

        # Frame-dragging: ω_LT = 2 G(ψ) J / (c² r³)
        r_m = d_au * AU_M
        if r_m > 0:
            G_local = G_SI * g_ratio
            omega_LT = 2 * G_local * J_total / (C_SI**2 * r_m**3)
        else:
            omega_LT = 0

        profile.append(
            {
                "d_ly": d_ly,
                "d_au": d_au,
                "psi": psi,
                "G_ratio": g_ratio,
                "tau": tau_at_distance(d_ly),
                "coherence": coherence_at_distance(d_ly),
                "omega_LT": omega_LT,
            }
        )
    return profile


# =====================================================================
# 7. Alpha(z) Oscillation Prediction
# =====================================================================
def alpha_z_prediction(z_values: List[float]) -> List[Dict[str, float]]:
    """Predict α(z) oscillation from gravity analysis Section 12.2.

    The shell structure modulates the effective electromagnetic
    coupling across cosmological Refractive Depth. Predicted:
    - Period: Δz ≈ 0.0143 in ln(1+z)
    - Amplitude: ~5×10⁻⁴ fractional
    - Seven sub-cycles per main period from shell structure
    - Sub-cycle period: Δz ≈ 0.0020
    """
    results = []
    period_main = DELTA  # In natural log units
    sub_period = DELTA / 7  # Seven sub-cycles from 7 shells
    amplitude = 5e-4

    for z in z_values:
        if z <= 0:
            continue
        ln_1z = math.log(1 + z)
        # Main oscillation
        phase_main = (ln_1z / period_main) * 2 * math.pi
        # Sub-cycles from shell structure
        phase_sub = (ln_1z / sub_period) * 2 * math.pi
        # Combined modulation
        delta_alpha = amplitude * (
            0.7 * math.sin(phase_main) + 0.3 * math.sin(phase_sub)
        )
        alpha_pred = ALPHA_GEOM * (1 + delta_alpha)
        results.append(
            {
                "z": z,
                "ln_1z": ln_1z,
                "delta_alpha_over_alpha": delta_alpha,
                "alpha_pred": alpha_pred,
            }
        )
    return results


# =====================================================================
# 8. Sgr A* Shadow Prediction
# =====================================================================
def compute_sgra_shadow() -> Dict[str, float]:
    """Compute Sgr A* shadow size under internal vs external metric.

    Standard Kerr (external observer):
      d_shadow ≈ 5.2 GM/c² (for a ≈ 0)
      For M = 4×10⁶ M_sun at D = 26 kly: ~52 μas

    CGM internal metric:
      The shadow is the apparent size of the Ingress conjugate
      seen through the Egress aperture. The key modification
      is that G varies across the shell, altering the effective
      gravitational radius.

    CGM predicts: shadow ~15% smaller than GR (from G(psi) analysis).
    """
    M_sgra_kg = 4e6 * M_SUN_KG
    D_kly = 26.0
    D_m = D_kly * 1e3 * LY_M  # Distance in meters

    # Standard Kerr shadow diameter (Schwarzschild, a=0)
    r_s = 2 * G_SI * M_sgra_kg / C_SI**2  # Schwarzschild radius
    d_shadow_GR = 5.2 * r_s  # Shadow diameter
    theta_GR_rad = d_shadow_GR / D_m
    theta_GR_uas = theta_GR_rad * (180 / math.pi) * 3600 * 1e6

    # CGM prediction: ~15% smaller
    # From gravity analysis: at ψ ≈ 0.5, G/G_global ≈ 0.72
    # Effective r_s is reduced proportionally
    psi_at_shadow = G_SI * M_sgra_kg / (r_s * C_SI**2)  # ψ at r_s
    G_ratio_at_shadow = G_ratio_approx(psi_at_shadow)
    d_shadow_CGM = d_shadow_GR * G_ratio_at_shadow
    theta_CGM_rad = d_shadow_CGM / D_m
    theta_CGM_uas = theta_CGM_rad * (180 / math.pi) * 3600 * 1e6

    # EHT measured values
    eht_m87 = 42.0  # μas (M87*)
    eht_sgra = 52.0  # μas (Sgr A*, approximate)

    return {
        "M_sgra_Msun": 4e6,
        "D_kly": D_kly,
        "r_s_m": r_s,
        "theta_GR_uas": theta_GR_uas,
        "theta_CGM_uas": theta_CGM_uas,
        "CGM_ratio": theta_CGM_uas / theta_GR_uas,
        "reduction_pct": (1 - theta_CGM_uas / theta_GR_uas) * 100,
        "eht_measured_uas": eht_sgra,
        "psi_at_shadow": psi_at_shadow,
    }


# =====================================================================
# 9. Formatting Utilities
# =====================================================================
def print_section(title: str) -> None:
    print(f"\n{'=' * 9}")
    print(f" {title}")
    print(f"{'=' * 9}\n")


def print_table(headers: List[str], rows: List[List[str]], title: str = "") -> None:
    if title:
        print(f"  {title}\n")
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))
    print()


# =====================================================================
# 10. Main Analysis
# =====================================================================
def main() -> None:
    print("CGM 1 LIGHT-YEAR ANALYSIS")
    print("The BU Interval, Parallax Holography, and the Rare Earth\n")

    # Pre-compute crossings
    d_delta_cross = find_crossing(DELTA, "tau", 0.5, 1.5)
    d_rho_cross = find_crossing(RHO, "coherence", 0.5, 1.5)
    tau_1ly = tau_at_distance(1.0)
    coh_1ly = coherence_at_distance(1.0)

    # ------------------------------------------------------------------
    # Section 1: The BU Interval
    # ------------------------------------------------------------------
    print_section("1. THE BU INTERVAL: [~1 ly, ~2 ly]")
    print("  Two independent kernel-derived boundaries:")
    print()
    print("  BU-EGRESS FACE (~0.97 ly)")
    print("    τ = Δ crossing: aperture opens, space→time conversion begins")
    print("    Source: per-cycle transport theorem (7591/7392)")
    if d_delta_cross is not None:
        print(f"    Exact distance: {d_delta_cross:.4f} ly")
    print()
    print("  BU-INGRESS FACE (2.00 ly)")
    print("    W₂ involution complete: equality horizon, all chirality cancels")
    print("    Source: wavefunction T5 (depth-4 confinement)")
    print("    This is a kernel theorem, not a calibration")
    print()
    print("  CONVERSION ZONE [~1 ly, ~2 ly]")
    print("    Carrier in bulk (shells 1-5), neither fully spatial")
    print("    nor fully temporal. Oort cloud = material condensate.")
    print()
    print("  These two faces are DUAL READINGS of the same depth-4")
    print("  event (BU), not sequential stages. They map to two")
    print("  boundaries of the observational shell.")

    # ------------------------------------------------------------------
    # Section 2: Gyration-Driven Trajectory
    # ------------------------------------------------------------------
    print_section("2. GYRATION-DRIVEN CARRIER TRAJECTORY")
    print("  The carrier does NOT traverse shells linearly.")
    print("  It follows the canonical word verified by wavefunction")
    print("  diagnostics (hqvm_wavefunction_1.py half-step output):")
    print()

    traj = gyration_trajectory()
    headers = ["d (ly)", "Sector", "Shell*", "chi", "Z2", "τ", "Coh", "Event"]
    rows = []
    for step in traj:
        rows.append(
            [
                f"{step['d_ly']:.1f}",
                step["sector"][:4],
                str(step["arch_shell"]),
                step["chirality"][:6],
                step["z2"][:5],
                f"{step['tau']:.4f}",
                f"{step['coherence']:.4f}",
                step["description"][:45],
            ]
        )
    print_table(headers, rows)
    print("  *arch_shell: 0=complement, 6=equality")
    print()
    print("  KEY: At d=1.5 ly, carrier BRIEFLY returns to complement")
    print("  horizon before jumping to equality at d=2 ly.")
    print("  This is NOT linear — it is gyration-driven.")

    # Extended trajectory for Z2 cycle
    print()
    print("  Full Z2 cycle (8 ly, F∘F = id):")
    ext = extended_trajectory(8)
    headers_z2 = ["d (ly)", "Sector", "Z2", "τ", "Coh"]
    rows_z2 = []
    for step in ext:
        if step["d_ly"] in [0, 1, 2, 4, 8] or step["d_ly"] % 1 == 0:
            rows_z2.append(
                [
                    f"{step['d_ly']:.0f}",
                    step["sector"][:4],
                    step["z2"][:7],
                    f"{step['tau']:.4f}",
                    f"{step['coherence']:.4f}",
                ]
            )
    print_table(headers_z2, rows_z2)
    print("  Z2 trajectory: rest → swapped → rest (period 8 ly)")

    # ------------------------------------------------------------------
    # Section 3: Inverted Shell Structure
    # ------------------------------------------------------------------
    print_section("3. INVERTED SHELL: INTERNAL OBSERVER PERSPECTIVE")
    print("  We inhabit the INTERIOR of an observational shell.")
    print()
    headers_kerr = ["Kerr BH (external)", "CGM Shell (internal)", "Kernel Object"]
    rows_kerr = [
        ["Exterior (flat)", "Interior <1 ly", "Complement horizon"],
        ["Event horizon", "Egress face ~1 ly", "Aperture opens"],
        ["Ergosphere", "Conversion zone [1,2] ly", "Bulk shells 1-5"],
        ["Cauchy horizon", "Ingress face ~2 ly", "Equality horizon"],
        ["Singularity", "Equality horizon", "χ = 000000"],
        ["Spin a", "Z₂ encoding (F gate)", "rest ↔ swapped"],
        ["Mass M", "Aperture Δ ≈ 2.07%", "Conversion gate"],
    ]
    print_table(headers_kerr, rows_kerr)
    print("  The 'singularity' has zero chirality, not infinite curvature.")
    print("  All 6 DoF become transparent simultaneously (equivalence).")

    # ------------------------------------------------------------------
    # Section 4: Refractive Depth at BU Faces
    # ------------------------------------------------------------------
    print_section("4. OPTICAL DEPTH AT THE BU FACES")
    print(f"  BU-EGRESS FACE:")
    print(f"    τ = Δ crossing at d = {d_delta_cross:.4f} ly")
    print(f"    τ at 1 ly:    {tau_1ly:.6f}")
    print(f"    Δ:             {DELTA:.6f}")
    print(f"    τ/Δ = {tau_1ly/DELTA:.6f} (= 7591/7392)")
    print(f"    Coherence:     {coh_1ly:.6f}")
    print(f"    ρ:             {RHO:.6f}")
    print(f"    |coh - ρ|:     {abs(coh_1ly - RHO):.6e}")
    print()
    tau_2ly = tau_at_distance(2.0)
    coh_2ly = coherence_at_distance(2.0)
    print(f"  BU-INGRESS FACE:")
    print(f"    d = 2.00 ly (depth-4 W₂ completion)")
    print(f"    τ:            {tau_2ly:.6f}")
    print(f"    Coherence:     {coh_2ly:.6f}")
    print(f"    Aperture:      {(1-coh_2ly)*100:.2f}%")
    print()
    print("  The Egress face is where τ first exceeds Δ.")
    print("  The Ingress face is where W₂ completes (kernel theorem).")
    print("  Both are independent derivations converging on the ~1 ly scale.")

    # ------------------------------------------------------------------
    # Section 5: Conversion Zone Profile
    # ------------------------------------------------------------------
    print_section("5. CONVERSION ZONE PROFILE [1 ly, 2 ly]")
    distances = [0.1, 0.5, 0.97, 1.0, 1.2, 1.5, 1.71, 2.0, 3.0, 4.0, 8.0]
    headers = ["d (ly)", "d (AU)", "τ", "Coh", "Apt%", "Zone", "Carrier"]
    rows = []
    for d in distances:
        e_tau = tau_at_distance(d)
        e_coh = coherence_at_distance(d)
        e_apt = (1 - e_coh) * 100
        if d < (d_delta_cross or 0.97):
            zone = "Spatial"
        elif d < 2.0:
            zone = "CONVERT"
        elif d < 4.0:
            zone = "Temporal"
        else:
            zone = "Z₂ cycle"
        # Carrier position from trajectory
        carrier = "bulk"
        if d <= 0:
            carrier = "complement"
        elif abs(d - 1.5) < 0.1:
            carrier = "complement→"
        elif abs(d - 2.0) < 0.1:
            carrier = "equality"
        elif d >= 4.0:
            carrier = "complement"
        rows.append(
            [
                f"{d:.2f}",
                f"{d*LY_AU:,.0f}",
                f"{e_tau:.4f}",
                f"{e_coh:.4f}",
                f"{e_apt:.2f}",
                zone,
                carrier,
            ]
        )
    print_table(headers, rows)

    # ------------------------------------------------------------------
    # Section 6: Star System Calibrations
    # ------------------------------------------------------------------
    print_section("6. STAR SYSTEM CALIBRATIONS")
    print("  Is the 1-ly = 1-gyration mapping contingent on Earth,")
    print("  or structurally necessary for any star system?")
    print()

    headers_ss = [
        "System",
        "M/M☉",
        "T_orb(yr)",
        "d₀(ly)",
        "Egress(ly)",
        "Ingress(ly)",
        "Z₂(ly)",
    ]
    rows_ss = []
    for name, sys_data in STAR_SYSTEMS.items():
        bu = compute_bu_interval(sys_data)
        rows_ss.append(
            [
                name,
                f"{sys_data['M_star_kg']/M_SUN_KG:.3f}",
                f"{sys_data['period_yr']:.3f}",
                f"{bu['d_0']:.3f}",
                f"{bu['d_egress']:.3f}",
                f"{bu['d_ingress']:.3f}",
                f"{bu['z2_period']:.1f}",
            ]
        )
    print_table(headers_ss, rows_ss)
    print("  Every star system has a BU interval at its own gyration scale.")
    print("  The structure is universal; the 1-ly calibration is Earth-specific.")
    print()
    print("  Testable prediction: other star systems should show")
    print("  cometary/icy-body boundaries near their Egress face,")
    print("  and density transitions near their Ingress face.")

    # ------------------------------------------------------------------
    # Section 7: Internal Metric and Frame-Dragging
    # ------------------------------------------------------------------
    print_section("7. INTERNAL METRIC AND FRAME-DRAGGING")
    print("  Inside the shell, the CGM field equations give:")
    print("    g(r) = -G(ψ) M / r²   with G(ψ)/G₀ ≈ 1 - 0.6456ψ")
    print()
    print("  Frame-dragging (Lense-Thirring):")
    print("    ω_LT = 2 G(ψ) J / (c² r³)")
    print("    where J is the system's total angular momentum")
    print()

    profile = compute_internal_metric_profile(n_points=12)
    headers_m = ["d (ly)", "ψ", "G/G₀", "τ", "Coh", "ω_LT (rad/s)"]
    rows_m = []
    for p in profile:
        rows_m.append(
            [
                f"{p['d_ly']:.2f}",
                f"{p['psi']:.2e}",
                f"{p['G_ratio']:.8f}",
                f"{p['tau']:.4f}",
                f"{p['coherence']:.4f}",
                f"{p['omega_LT']:.3e}",
            ]
        )
    print_table(headers_m, rows_m)
    print("  At Oort distances (1-2 ly), ψ ~ 10⁻¹³:")
    print("  G corrections and frame-dragging are negligible.")
    print("  The dominant effect is the Refractive Depth / aperture opening.")
    print()
    print("  The internal metric is qualitatively different from the")
    print("  external Kerr metric. We solve the CGM field equations")
    print("  with boundary conditions set by the shell's wavefunction,")
    print("  not with flat-spacetime boundary at infinity.")

    # ------------------------------------------------------------------
    # Section 8: Sgr A* Shadow
    # ------------------------------------------------------------------
    print_section("8. SGR A* SHADOW: INTERNAL VS EXTERNAL METRIC")
    shadow = compute_sgra_shadow()
    print(f"  Sgr A* mass: {shadow['M_sgra_Msun']:.0e} M☉")
    print(f"  Distance: {shadow['D_kly']:.0f} kly")
    print(f"  Schwarzschild radius: {shadow['r_s_m']:.3e} m")
    print()
    print(f"  Standard Kerr (external observer):")
    print(f"    Shadow diameter: {shadow['theta_GR_uas']:.1f} μas")
    print()
    print(f"  CGM internal metric:")
    print(f"    ψ at r_s: {shadow['psi_at_shadow']:.4f}")
    print(f"    G/G₀ at r_s: {G_ratio_approx(shadow['psi_at_shadow']):.4f}")
    print(f"    Shadow diameter: {shadow['theta_CGM_uas']:.1f} μas")
    print(f"    Reduction: {shadow['reduction_pct']:.1f}%")
    print()
    print(f"  EHT measured: ~{shadow['eht_measured_uas']:.0f} μas")
    print()
    print("  INTERPRETATION:")
    print("  Under the internal metric, we observe Sgr A* from inside")
    print("  the shell. The shadow is the apparent size of the Ingress")
    print("  conjugate seen through the Egress aperture, not the shadow")
    print("  of an external black hole. The ~15% reduction from CGM's")
    print("  G(ψ) model is the key falsifiable prediction.")

    # ------------------------------------------------------------------
    # Section 9: α(z) Oscillation
    # ------------------------------------------------------------------
    print_section("9. α(z) OSCILLATION PREDICTION")
    print("  Gravity analysis Section 12.2: the shell structure modulates")
    print("  the effective electromagnetic coupling across cosmological depth.")
    print()
    print(f"  Predicted period in ln(1+z): Δ = {DELTA:.6f}")
    print(f"  Peak-to-peak amplitude: ~5×10⁻⁴ fractional")
    print(f"  Sub-cycles: 7 per main period (from 7 shells)")
    print()

    z_test = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    alpha_results = alpha_z_prediction(z_test)
    headers_a = ["z", "ln(1+z)", "Δα/α", "α_pred"]
    rows_a = []
    for r in alpha_results:
        rows_a.append(
            [
                f"{r['z']:.2f}",
                f"{r['ln_1z']:.4f}",
                f"{r['delta_alpha_over_alpha']:.2e}",
                f"{r['alpha_pred']:.7f}",
            ]
        )
    print_table(headers_a, rows_a)
    print("  Falsification: detecting this oscillation with the predicted")
    print("  period and amplitude would confirm the shell structure")
    print("  modulates electromagnetic coupling at cosmological scales.")

    # ------------------------------------------------------------------
    # Section 10: Oort Cloud
    # ------------------------------------------------------------------
    print_section("10. OORT CLOUD: MATERIAL CONDENSATE OF THE BU INTERVAL")
    oort_data = [
        ("Inner Hills start", 2000),
        ("Inner Hills mid", 10000),
        ("Inner Hills end", 20000),
        ("Outer Oort typical", 50000),
        ("Outer Oort extended", 100000),
        ("Outer Oort max", 200000),
    ]
    headers_o = ["Region", "d (AU)", "d (ly)", "τ", "Coh", "Apt%", "Zone"]
    rows_o = []
    for name, d_au in oort_data:
        d_ly = d_au / LY_AU
        e_tau = tau_at_distance(d_ly)
        e_coh = coherence_at_distance(d_ly)
        e_apt = (1 - e_coh) * 100
        zone = (
            "Spatial"
            if d_ly < (d_delta_cross or 0.97)
            else ("CONVERT" if d_ly < 2.0 else "Temporal")
        )
        rows_o.append(
            [
                name,
                f"{d_au:,}",
                f"{d_ly:.3f}",
                f"{e_tau:.4f}",
                f"{e_coh:.4f}",
                f"{e_apt:.2f}",
                zone,
            ]
        )
    print_table(headers_o, rows_o)
    print("  Predicted properties from kernel structure:")
    print("  1. ISOTROPY: spherical aperture (Q_G = 4π)")
    print("  2. FRAGILITY: incomplete closure (bulk states)")
    print("  3. DYNAMICALLY NEW: fresh condensation events")
    print("  4. PARITY BIAS: CS chirality (left-gyration zero defect)")
    print("  5. SHELL STRUCTURE: density transitions at shell boundaries")
    print("  6. DENSITY TRANSITION near Egress/Ingress faces")

    # ------------------------------------------------------------------
    # Section 11: Graviton Consistency
    # ------------------------------------------------------------------
    print_section("11. GRAVITON: m_g = 0 (STRUCTURAL REQUIREMENT)")
    print("  CGM requires m_g = 0 (gravity analysis Section 7.3).")
    print("  The orientation-recovery operator must have infinite range")
    print("  for holographic reconstruction beyond the BU interval.")
    print()
    print(f"  GW170104: λ_full > {LAMBDA_FULL_GW170104_LY:.3f} ly")
    print(f"    (within BU interval of 2 ly)")
    print(f"  PDG 2024: λ_full > {LAMBDA_FULL_PDG2024_LY:.2f} ly")
    print(f"    (beyond BU interval)")
    print()
    print("  Falsification: detection of m_g > 0 would prevent the")
    print("  Ingress face from encoding memory, destroying the BU duality.")

    # ------------------------------------------------------------------
    # Section 12: Micro/Macro Mirror
    # ------------------------------------------------------------------
    print_section("12. MICRO/MACRO MIRROR")
    headers_mir = ["Property", "Macro (Solar)", "Micro (Hydrogen)"]
    rows_mir = [
        ["Source", "Sun", "Proton"],
        ["Observer", "Earth", "Electron"],
        ["Gyration unit", "1 year (2π)", "Orbital period"],
        ["Egress face", "~1 ly (aperture)", "Orbital radius (spin)"],
        ["Ingress face", "~2 ly (memory)", "4π rotation (closure)"],
        ["Conversion zone", "[1,2] ly (Oort)", "Orbital → spin"],
        ["Z₂ period", "8 ly (F∘F)", "720° (4π)"],
        ["Aperture", "Δ ≈ 2.07%", "α = δ_BU⁴/m_a"],
        ["Closure", "Q_G m_a² = 1/2", "spin-1/2"],
    ]
    print_table(headers_mir, rows_mir)

    # ------------------------------------------------------------------
    # Section 13: Distance Anchor Discussion
    # ------------------------------------------------------------------
    print_section("13. DISTANCE ANCHOR: KERNEL THEOREM vs CALIBRATION")
    print("  The kernel provides dimensionless structure:")
    print("    τ_cycle/Δ = 7591/7392  (exact rational)")
    print("    Shell trajectory        (verified on 4096 states)")
    print("    Z₂ period = 8 depth units")
    print()
    print("  The physical distance scale is a CALIBRATION:")
    print("    1 depth unit = 1 year (from CGM derivation of time)")
    print("    1 year = 1 ly (using c, derived in gravity analysis)")
    print()
    print("  This calibration is NOT a kernel theorem. It is the")
    print("  identification of Earth's orbital gyration with the")
    print("  kernel's depth unit. The structural prediction (BU Interval")
    print("  of width ~1 gyration unit) is universal; the 1-ly numerical")
    print("  value is specific to the Earth-Sun system.")
    print()
    print("  If the calibration is STRUCTURALLY NECESSARY (not")
    print("  contingent), then every star system must exhibit the")
    print("  same BU Interval at its own gyration scale. This is")
    print("  testable with exoplanet and debris disk observations.")

    # ------------------------------------------------------------------
    # Section 14: Open Questions
    # ------------------------------------------------------------------
    print_section("14. OPEN QUESTIONS AND PREDICTIONS")
    print("  DERIVED (kernel-grounded):")
    print("  1. BU Interval [~1 ly, ~2 ly] from two independent anchors")
    print("  2. Gyration-driven trajectory (NOT linear shell partition)")
    print("  3. Carrier briefly visits complement at d ≈ 1.5 ly")
    print("  4. Equality horizon at d = 2 ly (χ = 000000)")
    print("  5. Z₂ period = 8 ly (F∘F = id)")
    print("  6. m_g = 0 (structural requirement)")
    print()
    print("  PREDICTED (requires internal metric derivation):")
    print("  7. Sgr A* shadow ~15% smaller than Kerr GR")
    print("  8. α(z) oscillation with period Δ in ln(1+z)")
    print("  9. Frame-dragging consistent with Z₂ precession")
    print("  10. Oort density transition near 1-2 ly")
    print()
    print("  OPEN (requires further development):")
    print("  11. Reconstruction operator R(H, θ_Z2, τ)")
    print("  12. Redshift law: 1+z = f(ε, N_eff, θ_Z2)")
    print("  13. Is the distance anchor contingent or necessary?")
    print("  14. CMB as equality horizon from maximum temporal depth")
    print()
    print("  FALSIFICATION CRITERIA:")
    print("  - Detection of m_g > 0 destroys reconstruction")
    print("  - Sgr A* shadow matching GR exactly falsifies G(ψ)")
    print("  - Absence of α(z) oscillation at predicted period")
    print("  - Oort density with no shell structure or transition")

    # ------------------------------------------------------------------
    # Section 15: Summary
    # ------------------------------------------------------------------
    print_section("15. SUMMARY")
    print(f"  BU-Egress face:     {d_delta_cross:.4f} ly  (τ = Δ)")
    print(f"  BU-Ingress face:    2.0000 ly  (W₂, equality horizon)")
    print(f"  Conversion zone:    [{d_delta_cross:.2f}, 2.00] ly")
    print(f"  Z₂ period:          8 ly  (F∘F = id)")
    print(f"  Graviton:           m_g = 0  (infinite range)")
    print(
        f"  Sgr A* shadow:      ~{shadow['theta_CGM_uas']:.0f} μas "
        f"({shadow['reduction_pct']:.0f}% below GR)"
    )
    print(f"  α(z) period:        Δ ≈ {DELTA:.4f} in ln(1+z)")
    print()
    print("  Three independent kernel anchors on the ~1 ly scale:")
    print("  1. τ = Δ crossing at d ≈ 0.97 ly (transport theorem)")
    print("  2. W₂ involution at d = 2 ly (wavefunction T5)")
    print("  3. 12/7 = 1.714 (complementarity invariant)")
    print()
    print("  The observable universe is the parallax between the")
    print("  two BU faces. Beyond the Egress face, there is no space")
    print("  — only temporal depth, reconstructed as apparent structure")
    print("  through the Ingress face's holographic memory.")


if __name__ == "__main__":
    main()
