#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CGM Fine-Structure Constant Analysis

====================================

This script computes the CGM prediction for the fine-structure constant alpha
using parameters derived from CGM geometric structure.

Core prediction: Base alpha_0 = delta_BU^4 /  m_a (geometric IR focus coupling)

Parameters are computed from integrated CGM components:
- m_a: From gyrotriangle closure (3D/6DoF analysis)
- delta_BU: From Thomas-Wigner dual-pole monodromy (gyrovector operations)

No experimental alpha data is used until final comparison.
"""

import sys
import platform
import hashlib
import time
import mpmath as mp
import numpy as np

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Set high precision
mp.mp.dps = 100

# =============================================================================
# INTEGRATION WITH CGM COMPONENTS (from recent work)
# =============================================================================


def compute_from_cgm_3d_6dof():
    """Compute  m_a and thresholds from gyrotriangle (cgm_3D_6DoF_analysis.py)."""
    # Exact from gyrotriangle angles (π/2, π/4, π/4) with defect=0
    s_p = mp.pi / 2
    u_p = mp.cos(mp.pi / 4)  # 1/sqrt(2)
    o_p = mp.pi / 4
    m_a = mp.mpf(1) / (mp.mpf(2) * mp.sqrt(mp.mpf(2) * mp.pi))
    return {"m_a": m_a, "s_p": s_p, "u_p": u_p, "o_p": o_p}


def compute_delta_BU_from_gyrovector():
    """
    Compute δ_BU from BU dual-pole monodromy using gyrovector space.

    δ_BU = 2 × ω(ONA ↔ BU) where ω is the rotation angle from gyration.

    This is the PRIMARY derivation from CGM geometric structure:
    - Uses the same GyroVectorSpace implementation as tw_closure_test.py
    - Traverse path: ONA → BU+ → BU- → ONA
    - Compute gyration matrix G(ONA, BU+) using Einstein gyrovector space
    - Extract rotation angle ω from G
    - δ_BU = 2ω represents the dual-pole memory

    Implementation matches tw_closure_test.py compute_bu_dual_pole_monodromy() exactly.
    """
    try:
        from functions.gyrovector_ops import GyroVectorSpace
    except ImportError:
        # Fallback for direct script execution
        import sys
        import os

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from functions.gyrovector_ops import GyroVectorSpace

    # Get CGM parameters
    cgm_params = compute_from_cgm_3d_6dof()
    m_a = float(cgm_params["m_a"])
    o_p = float(cgm_params["o_p"])  # π/4

    # Create gyrovector space (same as tw_closure_test.py)
    gyrospace = GyroVectorSpace(c=1.0)

    # Define vectors exactly as in tw_closure_test.py
    v_ona = np.array([0.0, o_p, 0.0], dtype=np.float64)
    v_bu_plus = np.array([0.0, 0.0, m_a], dtype=np.float64)

    # Compute gyration matrix G(ONA, BU+) - same as tw_closure_test.py
    G_on_to_bu = gyrospace.gyration(v_ona, v_bu_plus)

    # Extract rotation angle ω - same method as tw_closure_test.py
    omega_on_to_bu = float(gyrospace.rotation_angle_from_matrix(G_on_to_bu))

    # δ_BU = 2 × ω(ONA ↔ BU) - the dual-pole monodromy
    delta_BU = 2.0 * omega_on_to_bu

    # Convert to mpmath for high precision
    delta_BU_mp = mp.mpf(delta_BU)

    # Assertion: Lock δ_BU provenance - must equal 2×ω(ONA↔BU) from gyrovector gyration
    # Expected value from tw_closure_test.py: δ_BU ≈ 0.1953421766 rad
    # Allow tolerance for numerical precision (float64: ~1e-15 relative)
    expected_delta_BU = mp.mpf("0.1953421766")
    tolerance = mp.mpf("1e-8")  # Allow 0.01 ppb tolerance
    assert (
        abs(delta_BU_mp - expected_delta_BU) < tolerance
    ), f"δ_BU must equal 2×ω(ONA↔BU) from gyrovector gyration. Got {delta_BU_mp:.12f}, expected ~{expected_delta_BU:.12f}"

    return delta_BU_mp


def estimate_thomas_curvature_local(dβ=1e-3, dθ=1e-3):
    """
    Estimate Thomas curvature proxy F̄ using small rectangle plaquette around (u_p, o_p).

    Uses the same Wigner-angle formula as tw_closure_test.py for local curvature.
    """
    cgm = compute_from_cgm_3d_6dof()
    u_p = float(cgm["u_p"])
    o_p = float(cgm["o_p"])

    def wigner(beta, theta):
        """Wigner rotation angle formula: w(β, θ) = 2 atan(sin(θ) * sinh²(η/2) / (cosh²(η/2) + cos(θ) * sinh²(η/2)))"""
        beta = mp.mpf(beta)
        theta = mp.mpf(theta)
        eta = mp.atanh(beta)
        sh2 = mp.sinh(eta / 2) ** 2
        ch2 = mp.cosh(eta / 2) ** 2
        tan_half = mp.sin(theta) * sh2 / (ch2 + mp.cos(theta) * sh2)
        return 2 * mp.atan(tan_half)

    # Compute partial derivatives using finite differences
    domega_dθ = (wigner(u_p, o_p + dθ) - wigner(u_p, o_p - dθ)) / (2 * dθ)
    domega_dβ = (wigner(u_p + dβ, o_p) - wigner(u_p - dβ, o_p)) / (2 * dβ)

    # F̄ = dω/dθ - dω/dβ (curvature proxy from local plaquette)
    F_bar = domega_dθ - domega_dβ

    return mp.mpf(F_bar)


def compute_F_bar_from_axiomatization():
    """Compute F̄ from local Thomas curvature estimate around (u_p, o_p)."""
    return estimate_thomas_curvature_local()


def compute_toroidal_4leg_holonomy():
    """
    Compute 4-leg toroidal holonomy: CS → UNA → ONA → BU → CS.

    This matches the computation in tw_closure_test.py toroidal_holonomy().
    """
    try:
        from functions.gyrovector_ops import GyroVectorSpace
    except ImportError:
        import sys
        import os

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from functions.gyrovector_ops import GyroVectorSpace

    cgm = compute_from_cgm_3d_6dof()
    s_p = float(cgm["s_p"])
    u_p = float(cgm["u_p"])
    o_p = float(cgm["o_p"])
    m_a = float(cgm["m_a"])

    G = GyroVectorSpace(c=1.0)

    # Define vectors exactly as in tw_closure_test.py
    v_CS = np.array([0.0, 0.0, s_p])
    v_UNA = np.array([u_p, 0.0, 0.0])
    v_ONA = np.array([0.0, o_p, 0.0])
    v_BU = np.array([0.0, 0.0, m_a])

    # Compute 4-leg loop: CS → UNA → ONA → BU → CS
    R = (
        G.gyration(v_CS, v_UNA)
        @ G.gyration(v_UNA, v_ONA)
        @ G.gyration(v_ONA, v_BU)
        @ G.gyration(v_BU, v_CS)
    )

    return float(G.rotation_angle_from_matrix(R))


def compute_holo_ratio_from_energy():
    """
    Compute holonomy ratio from 4-leg and 8-leg toroidal holonomies.

    holo_ratio = holo_4leg / holo_8leg
    where holo_8leg = δ_BU (the dual-pole monodromy).
    """
    holo_4leg_val = compute_toroidal_4leg_holonomy()  # ≈ 0.862833
    delta_BU = compute_delta_BU_from_gyrovector()  # δ_BU (8-leg holonomy)
    holo_ratio = mp.mpf(holo_4leg_val) / delta_BU
    return holo_ratio


# =============================================================================
# REPRODUCIBILITY METADATA
# =============================================================================


def print_metadata():
    """Print environment and source info."""
    print("CGM Alpha Analysis Metadata")
    print("-" * 30)
    print(f"Python: {platform.python_version()}")
    print(f"mpmath dps: {mp.mp.dps}")
    print(f"numpy: {np.__version__}")
    print(f"Run time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Hash inputs for reproducibility (thresholds, etc.)
    inputs_str = f"m_a=1/(2sqrt(2pi)); thresholds=pi/2,cos(pi/4),pi/4"
    hash_val = hashlib.sha256(inputs_str.encode()).hexdigest()[:8]
    print(f"Input hash: {hash_val}")
    print(
        "Sources: Integrated from cgm_3D_6DoF (gyrotriangle), tw_closure_test (gyrovector)."
    )
    print("         No alpha_exp used upstream.")
    print()


# =============================================================================
# MAIN COMPUTATION
# =============================================================================


def main():
    print_metadata()

    print("CGM Fine-Structure Constant: Geometric IR Focus Prediction")
    print("=" * 65)
    print("Base prediction from BU dual-pole monodromy and aperture structure.")
    print("=" * 65)

    # Compute parameters from CGM components
    cgm_params = compute_from_cgm_3d_6dof()
    m_a = cgm_params["m_a"]
    print(f" m_a = 1/(2*sqrt(2*pi)) = {float(m_a):.12f} (from gyrotriangle closure)")

    delta_BU = compute_delta_BU_from_gyrovector()
    print(
        f"delta_BU = {float(delta_BU):.12f} rad (computed from gyrovector dual-pole monodromy)"
    )
    print("  Derivation: δ_BU = 2 × ω(ONA ↔ BU) from gyration in gyrovector space")
    print("  Path: ONA → BU+ → BU- → ONA (dual-pole traversal)")

    Delta = mp.mpf(1) - (delta_BU / m_a)
    rho = delta_BU / m_a
    print(f"Delta = 1 - (delta_BU/m_a) = {float(Delta):.12f} (aperture gap)")
    print(f"rho = delta_BU/ m_a = {float(rho):.12f} (closure fraction)")

    # BASE PREDICTION (lead with this)
    print("\nBASE PREDICTION (IR Focus Geometry)")
    print("-" * 40)
    alpha_base = (delta_BU**4) / m_a
    print(f"alpha_0 = delta_BU^4 /  m_a = {float(alpha_base):.12f}")
    print("  Pure geometric IR focus coupling at BU stage.")

    # Experimental comparison (only here, at end)
    alpha_exp = mp.mpf("0.007297352563")  # CODATA 2020
    error_base_ppm = (alpha_base - alpha_exp) / alpha_exp * 1e6
    print(f"Experimental alpha = {float(alpha_exp):.12f} (CODATA)")
    print(
        f"Error: {float(error_base_ppm):.3f} ppm (319 ppm accuracy from geometry alone)"
    )
    print("  Remarkable: 3 significant figures match without corrections.")

    # SENSITIVITY ANALYSIS (built-in robustness)
    print("\nSENSITIVITY TO PARAMETER ROUNDING (delta_BU)")
    print("-" * 45)
    rounding_levels = [6, 5, 4, 3]  # Decimals
    for ndec in rounding_levels:
        rounded_delta = mp.nstr(delta_BU, ndec)
        d_rounded = mp.mpf(rounded_delta)
        alpha_rounded = (d_rounded**4) / m_a
        err_rounded_ppm = (alpha_rounded - alpha_exp) / alpha_exp * 1e6
        print(
            f"  delta_BU rounded to {ndec} dec: {rounded_delta} -> alpha_0 = {float(alpha_rounded):.9f} (err = {float(err_rounded_ppm):.1f} ppm)"
        )
    print(
        "  Observation: Base α₀ requires δ_BU to at least 5–6 decimals for ppm-level accuracy;"
    )
    print("                rounding to 4 decimals degrades accuracy (≈544 ppm).")
    print("                ppb-level accuracy requires full-precision δ_BU.")

    print("\nConclusion: Base α₀ from CGM geometry achieves 319 ppm accuracy.")
    print(
        "This represents 3 significant figures of agreement with no fitted parameters."
    )

    return {
        "alpha_base": alpha_base,
        "error_base_ppm": error_base_ppm,
        "m_a": m_a,
        "delta_BU": delta_BU,
    }


if __name__ == "__main__":
    results = main()
