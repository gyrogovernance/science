#!/usr/bin/env python3
"""
CGM SU(3) Hadron Analysis

SU(3)_flavor octet/decuplet checks for CGM geometric patterns in hadron masses.
This analysis looks for CGM invariants in hadron mass relations as a probe of
the Higgs mechanism through Yukawa couplings.
"""

import numpy as np
from math import sqrt, pi


def su3_hadron_checks():
    """SU(3)_flavor octet/decuplet checks for CGM geometric patterns in hadron masses."""
    print("\nCGM SU(3) HADRON ANALYSIS")
    print("=" * 50)
    print(
        "SU(3)_flavor octet/decuplet checks for CGM geometric patterns in hadron masses."
    )
    print(
        "This analysis looks for CGM invariants in hadron mass relations as a probe of"
    )
    print("the Higgs mechanism through Yukawa couplings.")
    print("-" * 70)

    # PDG central values [GeV]
    # Mesons (use isospin averages for simplicity)
    m_pi = 0.1380
    m_K = 0.4956
    m_eta = 0.5479

    # Baryon octet (averages for GMO relations)
    M_N = 0.9389  # average p/n
    M_Sigma = 1.1926  # average Σ-, Σ0, Σ+
    M_Lambda = 1.1157
    M_Xi = 1.3149  # average Ξ0, Ξ-

    # Individual isospin doublets for Coleman-Glashow
    M_Sigma_minus = 1.1974  # Σ⁻
    M_Sigma_plus = 1.1894  # Σ⁺
    M_Xi_minus = 1.3213  # Ξ⁻
    M_Xi_zero = 1.3149  # Ξ⁰

    # Baryon decuplet
    M_Delta = 1.2320
    M_SigS = 1.3846  # Σ*
    M_XiS = 1.5334  # Ξ*
    M_Omega = 1.6724  # Ω-

    # CGM invariants to compare
    m_p = 1 / (2 * sqrt(2 * pi))
    rho = 0.195342176580 / m_p
    Delta = 1 - rho  # ~ 0.0206995539

    # 1) Baryon GMO: 2N + 2Ξ ?= 3Λ + Σ
    lhs = 2 * M_N + 2 * M_Xi
    rhs = 3 * M_Lambda + M_Sigma
    avg = 0.5 * (lhs + rhs)
    D_baryon = (lhs - rhs) / avg
    print(f"GMO (baryon octet): 2N+2Ξ vs 3Λ+Σ")
    print(f"  defect = {D_baryon*100:.3f}%   (CGM Δ = {Delta*100:.3f}%)")

    # 2) Meson GMO (squared masses): 3 η^2 + π^2 ?= 4 K^2
    lhs_m = 3 * m_eta**2 + m_pi**2
    rhs_m = 4 * m_K**2
    D_meson = (lhs_m - rhs_m) / rhs_m
    print(f"GMO (pseudoscalar octet, mass^2): 3η^2 + π^2 vs 4K^2")
    print(f"  defect = {D_meson*100:.3f}%   (CGM Δ = {Delta*100:.3f}%)")

    # 3) Decuplet equal spacing
    spacings = [M_SigS - M_Delta, M_XiS - M_SigS, M_Omega - M_XiS]
    mean_spacing = np.mean(spacings)
    frac_devs = [(s - mean_spacing) / mean_spacing for s in spacings]
    rms_dev = sqrt(np.mean([d**2 for d in frac_devs]))
    print(f"Decuplet equal-spacing (Δ, Σ*, Ξ*, Ω)")
    print(
        f"  spacings: {[f'{s:.3f}' for s in spacings]} GeV, mean={mean_spacing:.3f} GeV"
    )
    print(
        f"  frac devs: {[f'{d*100:.2f}%' for d in frac_devs]}   (Δ = {Delta*100:.2f}%)"
    )
    print(f"  RMS |dev| ≈ {rms_dev*100:.2f}% vs Δ = {Delta*100:.2f}%")

    # 4) GMOR-like ratio (Yukawa hierarchy proxy)
    R = (m_K**2 - m_pi**2) / m_pi**2
    print(f"GMOR-like ratio R = (K^2 − π^2)/π^2 ≈ {R:.2f}  (Yukawa hierarchy proxy)")

    # 5) Coleman-Glashow isospin breaking
    Delta_Xi = M_Xi_minus - M_Xi_zero
    Delta_Sigma = M_Sigma_minus - M_Sigma_plus
    print(
        f"Coleman-Glashow isospin: (Ξ⁻ − Ξ⁰) = {Delta_Xi*1000:.1f} MeV, (Σ⁻ − Σ⁺) = {Delta_Sigma*1000:.1f} MeV"
    )
    print(
        f"  Fractional defect: {((Delta_Xi + Delta_Sigma)/2/0.001 - 1)*100:.3f}%  (EM/isospin breaking ~1 MeV)"
    )

    print("\nNotes:")
    print("  • Baryon GMO: SU(3)_flavor breaking by Yukawas and EM effects")
    print("  • Meson GMO: Uses mass^2, sensitive to η–η′ mixing")
    print("  • Decuplet: Equal-spacing rule deviations from SU(3) breaking")
    print("  • Coleman-Glashow: Tests isospin symmetry (perfect at tree level)")


if __name__ == "__main__":
    su3_hadron_checks()
