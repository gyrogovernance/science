#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CGM Fine-Structure Constant Analysis

Computes α₀ = δ_BU⁴ / m_a and the three UV-IR transport corrections.
δ_BU is the closed form 4 · arctan(k(π/4) · k(m_a)), matching Analysis_Holonomy.md
and Analysis_Fine_Structure.md. No experimental α enters until the final comparison.
"""

import hashlib
import platform
import sys
import time

import mpmath as mp

if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

mp.mp.dps = 80

# Measured geometric ratio (not fitted to α); same value as Analysis_CGM_Constants.md
R_CURVATURE = mp.mpf("0.993434896272")
CODATA_2018_INV = mp.mpf("137.035999084")


def k_half_rapidity(beta):
    """k(β) = β / (1 + √(1 − β²)) = tanh(atanh(β)/2)."""
    return beta / (1 + mp.sqrt(1 - beta**2))


def compute_m_a():
    return mp.mpf(1) / (2 * mp.sqrt(2 * mp.pi))


def compute_delta_BU(m_a):
    """δ_BU = 4 · arctan(k(π/4) · k(m_a))."""
    return 4 * mp.atan(k_half_rapidity(mp.pi / 4) * k_half_rapidity(m_a))


def compute_phi_SU2():
    """φ_SU2 = 2 · arccos((1 + 2√2) / 4)."""
    return 2 * mp.acos((1 + 2 * mp.sqrt(2)) / 4)


def alpha_correction_chain(delta_BU, m_a, phi_SU2):
    """Return α₀…α₃ and factors from Analysis_Fine_Structure.md equations (3)–(6)."""
    rho = delta_BU / m_a
    Delta = 1 - rho
    diff = phi_SU2 - 3 * delta_BU
    alpha0 = (delta_BU**4) / m_a
    C_AB = 1 - (mp.mpf(3) / 4) * R_CURVATURE * Delta**2
    C_HC = 1 - (mp.mpf(5) / 6) * (
        (phi_SU2 / (3 * delta_BU) - 1)
        * Delta**2
        / (4 * mp.pi * mp.sqrt(3))
    )
    C_IDE = 1 + (1 / rho) * diff * Delta**4
    alpha1 = alpha0 * C_AB
    alpha2 = alpha1 * C_HC
    alpha3 = alpha2 * C_IDE
    return {
        "rho": rho,
        "Delta": Delta,
        "diff": diff,
        "C_AB": C_AB,
        "C_HC": C_HC,
        "C_IDE": C_IDE,
        "alpha0": alpha0,
        "alpha1": alpha1,
        "alpha2": alpha2,
        "alpha3": alpha3,
    }


def relative_error(pred, ref, scale):
    return (pred - ref) / ref * scale


def print_metadata():
    print("CGM Alpha Analysis Metadata")
    print("-" * 5)
    print(f"Python: {platform.python_version()}")
    print(f"mpmath dps: {mp.mp.dps}")
    print(f"Run time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    inputs_str = "delta_BU=4*atan(k(pi/4)*k(m_a)); m_a=1/(2*sqrt(2*pi))"
    print(f"Input hash: {hashlib.sha256(inputs_str.encode()).hexdigest()[:8]}")
    print("Sources: closed-form δ_BU (Analysis_Holonomy.md); chain (Analysis_Fine_Structure.md).")
    print()


def main():
    print_metadata()

    print("CGM Fine-Structure Constant: IR Focus and UV-IR Corrections")
    print("=" * 5)

    m_a = compute_m_a()
    delta_BU = compute_delta_BU(m_a)
    phi_SU2 = compute_phi_SU2()
    chain = alpha_correction_chain(delta_BU, m_a, phi_SU2)
    alpha_ref = 1 / CODATA_2018_INV

    print(f"m_a = 1/(2*sqrt(2*pi)) = {float(m_a):.12f}")
    print(f"delta_BU = 4*atan(k(pi/4)*k(m_a)) = {float(delta_BU):.12f} rad")
    print(f"phi_SU2 = 2*acos((1+2*sqrt(2))/4) = {float(phi_SU2):.12f} rad")
    print(f"rho = delta_BU/m_a = {float(chain['rho']):.12f}")
    print(f"Delta = 1 - rho = {float(chain['Delta']):.12f}")
    print(f"diff = phi_SU2 - 3*delta_BU = {float(chain['diff']):.12f}")

    print("\nBASE")
    print("-" * 5)
    print(f"alpha_0 = delta_BU^4 / m_a = {float(chain['alpha0']):.12f}")
    print(
        f"residual vs CODATA 2018: {float(relative_error(chain['alpha0'], alpha_ref, 1e6)):.3f} ppm"
    )

    print("\nCORRECTIONS")
    print("-" * 5)
    print(f"C_AB  = {float(chain['C_AB']):.12f}")
    print(f"C_HC  = {float(chain['C_HC']):.12f}")
    print(f"C_IDE = {float(chain['C_IDE']):.12f}")
    print(f"alpha_1 = {float(chain['alpha1']):.12f}  "
          f"({float(relative_error(chain['alpha1'], alpha_ref, 1e6)):.6f} ppm)")
    print(f"alpha_2 = {float(chain['alpha2']):.12f}  "
          f"({float(relative_error(chain['alpha2'], alpha_ref, 1e6)):.6f} ppm)")
    print(f"alpha_3 = {float(chain['alpha3']):.12f}  "
          f"({float(relative_error(chain['alpha3'], alpha_ref, 1e9)):.3f} ppb)")

    print("\nREFERENCE")
    print("-" * 5)
    print(f"CODATA 2018 alpha = 1/{float(CODATA_2018_INV):.9f} = {float(alpha_ref):.12f}")

    return {
        "m_a": m_a,
        "delta_BU": delta_BU,
        "alpha_ref": alpha_ref,
        **chain,
    }


if __name__ == "__main__":
    main()
