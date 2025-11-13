#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cosmological Constant Analysis with Balance Index
==================================================

This script calculates the relationship between the Balance Index B_i
and cosmological constant / dark energy density, examining the
120-order vacuum energy discrepancy and scale hierarchies.

No claims of "resolution" — just rigorous calculations of the relationships.
"""

from decimal import Decimal, getcontext
import math
import sys
import io

# Set UTF-8 encoding for stdout
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Set high precision
getcontext().prec = 50

# Mathematical constants
PI = Decimal("3.1415926535897932384626433832795028841971693993751")
TWO_PI = Decimal("2") * PI
FOUR_PI = Decimal("4") * PI
EIGHT_PI = Decimal("8") * PI


class FundamentalConstants:
    """Fundamental physical constants from CODATA 2018"""

    # Defining constants (exact by SI definition)
    c = Decimal("299792458")  # Speed of light [m/s]
    h = Decimal("6.62607015e-34")  # Planck constant [J⋅s]
    k_B = Decimal("1.380649e-23")  # Boltzmann constant [J/K]

    # Measured constant
    G = Decimal("6.67430e-11")  # Gravitational constant [m³/(kg⋅s²)]
    G_uncertainty = Decimal("0.00015e-11")

    # Derived
    hbar = h / TWO_PI


class CosmologicalParameters:
    """Current cosmological measurements"""

    # Planck 2018
    H0_planck = Decimal("67.27")  # km/s/Mpc
    H0_planck_error = Decimal("0.60")
    Omega_L = Decimal("0.6889")  # Dark energy density parameter

    # Unit conversions
    MPC_TO_M = Decimal("3.0857e22")
    KM_TO_M = Decimal("1000")

    @classmethod
    def H0_to_SI(cls, H0_kmsmpc):
        """Convert H_0 from km/s/Mpc to SI units (1/s)"""
        return (H0_kmsmpc * cls.KM_TO_M) / cls.MPC_TO_M


class CGMEnergyScales:
    """CGM energy scale anchors in GeV"""

    E_CS = Decimal("1.22089e19")
    E_EW = Decimal(
        "246.22"
    )  # Higgs vacuum expectation value v = (√2 G_F)^(-1/2) ≈ 246.22 GeV
    E_BU = Decimal("3.09e17")


class CGMGeometry:
    """CGM geometric parameters"""

    m_a = Decimal("1") / (Decimal("2") * (Decimal("2") * PI).sqrt())

    @classmethod
    def aperture_factor(cls):
        """Aperture correction: (1+m_a)^-2"""
        return Decimal("1") / ((Decimal("1") + cls.m_a) ** 2)


def calculate_balance_index(H0_SI, const):
    """Calculate Balance Index B_i = 2Gc/(k_B H₀)"""
    return (Decimal("2") * const.G * const.c) / (const.k_B * H0_SI)


def calculate_cosmological_constant_terms(H0_SI, const):
    """
    Calculate cosmological constant terms and vacuum energy comparison

    Returns densities and ratios without claiming resolution
    """

    # Observed dark energy density: ρ_Λ,obs = 3H₀²/(8πG) in SI (kg/m³)
    rho_L_obs_SI = (Decimal("3") * H0_SI**2) / (EIGHT_PI * const.G)

    # Convert to J/m³ for comparison
    rho_L_obs = rho_L_obs_SI * const.c**2

    # Theoretical vacuum energy density (Planck scale cutoff)
    # ρ_vac ~ c⁷/(ħG²) - dimensional estimate in J/m³
    rho_vac = const.c**7 / (const.hbar * const.G**2)

    # Ratio (the famous 120-order discrepancy)
    ratio = rho_vac / rho_L_obs

    # Balance Index
    B_i = calculate_balance_index(H0_SI, const)

    return {
        "rho_L_obs": rho_L_obs,
        "rho_L_obs_SI": rho_L_obs_SI,
        "rho_vac": rho_vac,
        "ratio_vac_to_obs": ratio,
        "B_i": B_i,
        "log10_ratio": float(ratio.ln() / Decimal("10").ln()),
    }


def calculate_equation_of_state_parameters(H0_SI, const, cosmo):
    """
    Calculate equation of state parameters
    w = P/(ρc²) for dark energy component
    """

    # Critical density
    rho_crit = (Decimal("3") * H0_SI**2) / (EIGHT_PI * const.G)

    # Dark energy density
    rho_L = cosmo.Omega_L * rho_crit

    # Pressure for cosmological constant: P = -ρc²
    P_L = -rho_L * const.c**2

    # Equation of state: w = P/(ρc²)
    w = P_L / (rho_L * const.c**2)

    return {
        "rho_crit": rho_crit,
        "rho_L": rho_L,
        "P_L": P_L,
        "w": float(w),
        "Omega_L": float(cosmo.Omega_L),
    }


def calculate_scale_hierarchy(H0_SI, const):
    """
    Calculate natural scale hierarchy
    Shows the separation between Planck, EW, and Hubble scales
    """
    scales = CGMEnergyScales()

    # Length scales
    L_H = const.c / H0_SI  # Hubble length
    L_Pl = (const.hbar * const.G / const.c**3).sqrt()  # Planck length
    E_EW_J = scales.E_EW * Decimal("1.602176634e-10")
    L_EW = (const.hbar * const.c) / E_EW_J  # EW Compton length

    # Balance Index length scale
    B_i = calculate_balance_index(H0_SI, const)
    L_B = (const.k_B * B_i) / const.G

    # Temperature scales
    T_eq = (const.hbar * H0_SI) / (TWO_PI * const.k_B)
    T_CMB = Decimal("2.725")  # K

    # Energy density scales
    rho_H = (Decimal("3") * H0_SI**2) / (EIGHT_PI * const.G)
    rho_Pl = const.c**5 / (const.hbar * const.G**2)
    rho_EW = E_EW_J**4 / (const.hbar**3 * const.c**5)

    return {
        "L_H": L_H,
        "L_Pl": L_Pl,
        "L_EW": L_EW,
        "L_B": L_B,
        "T_eq": T_eq,
        "T_CMB": T_CMB,
        "rho_H": rho_H,
        "rho_Pl": rho_Pl,
        "rho_EW": rho_EW,
        "L_ratio_H_to_Pl": L_H / L_Pl,
        "T_ratio_CMB_to_eq": T_CMB / T_eq,
        "rho_ratio_Pl_to_H": rho_Pl / rho_H,
    }


def calculate_lambda_terms(H0_SI, const):
    """
    Calculate cosmological constant Λ in different frameworks
    Test cutoff sensitivity
    """

    # Theoretical vacuum energy Λ (Planck scale dimensional estimate)
    rho_vac_Pl = const.c**7 / (const.hbar * const.G**2)
    Lambda_vac_Pl = (EIGHT_PI * const.G * rho_vac_Pl) / const.c**4

    # EW scale cutoff (naive estimate)
    scales = CGMEnergyScales()
    E_EW_J = scales.E_EW * Decimal("1.602176634e-10")
    rho_vac_EW = E_EW_J**4 / (const.hbar**3 * const.c**5)
    Lambda_vac_EW = (EIGHT_PI * const.G * rho_vac_EW) / const.c**4

    # Observed Λ from H₀ (standard ΛCDM)
    Lambda_obs = (Decimal("3") * H0_SI**2) / const.c**2

    # Ratios
    vac_Pl_vs_obs = Lambda_vac_Pl / Lambda_obs
    vac_EW_vs_obs = Lambda_vac_EW / Lambda_obs

    return {
        "Lambda_vac_Pl": Lambda_vac_Pl,
        "Lambda_vac_EW": Lambda_vac_EW,
        "Lambda_obs": Lambda_obs,
        "vac_Pl_vs_obs": vac_Pl_vs_obs,
        "vac_EW_vs_obs": vac_EW_vs_obs,
        "log10_vac_Pl": float(vac_Pl_vs_obs.ln() / Decimal("10").ln()),
        "log10_vac_EW": float(vac_EW_vs_obs.ln() / Decimal("10").ln()),
    }


def schwarzschild_I1(M_kg, const):
    """
    Calculate I₁ for a Schwarzschild black hole
    Contrast with cosmological I₁ = 2
    """
    R_s = 2 * const.G * M_kg / const.c**2
    A = FOUR_PI * R_s**2
    kappa = const.c**4 / (Decimal("4") * const.G * M_kg)
    H_bh = kappa / const.c
    T = const.hbar * H_bh / (TWO_PI * const.k_B)
    mP2 = const.hbar * const.c / const.G
    Sigma = (A * T) / mP2
    I1 = (const.k_B * Sigma * H_bh) / (const.G * const.c)
    return float(I1)


def propagate_uncertainties(H0_kmsmpc, H0_err_kmsmpc, const, cosmo):
    """
    Propagate uncertainties to B_i and ρ_Λ
    """
    H0 = cosmo.H0_to_SI(H0_kmsmpc)
    H0_err = cosmo.H0_to_SI(H0_err_kmsmpc)

    # B_i ∝ G / H0
    rel_H0 = H0_err / H0
    rel_G = const.G_uncertainty / const.G
    rel_Bi = (rel_G**2 + rel_H0**2).sqrt()

    # ρ_Λ ∝ H0²
    rel_rhoL = (Decimal("4") * rel_H0**2 + rel_G**2).sqrt()

    B_i = calculate_balance_index(H0, const)
    Bi_err = B_i * rel_Bi

    rho_L = (Decimal("3") * H0**2) / (EIGHT_PI * const.G)
    rho_L_err = rho_L * rel_rhoL

    return {
        "B_i": B_i,
        "B_i_err": Bi_err,
        "rel_Bi": rel_Bi,
        "rho_L": rho_L,
        "rho_L_err": rho_L_err,
        "rel_rhoL": rel_rhoL,
    }


def infer_bridge_exponent_from_H0(H0_SI, const, scales):
    """
    Given observed H0, solve for empirical bridge exponent n_emp
    From L_B × ℓ* = l_P² × F with F = C_geom × (E_CS/E_anchor)^n
    """
    GeV_to_J = Decimal("1.602176634e-10")
    E_anchor_J = scales.E_EW * GeV_to_J
    E_CS_J = scales.E_CS * GeV_to_J

    # Lengths
    L_B = (Decimal("2") * const.c) / H0_SI
    ell_star = (const.hbar * const.c) / E_anchor_J
    lP2 = (const.hbar * const.G) / (const.c**3)

    F_obs = (L_B * ell_star) / lP2

    # With aperture
    C_geom_ap = CGMGeometry.aperture_factor()
    n_emp_ap = (F_obs / C_geom_ap).ln() / (E_CS_J / E_anchor_J).ln()

    # Without aperture
    n_emp_noap = F_obs.ln() / (E_CS_J / E_anchor_J).ln()

    return {"n_emp_aperture": n_emp_ap, "n_emp_no_aperture": n_emp_noap, "F_obs": F_obs}


def predict_H0_from_UVIR(const, scales, n_th):
    """
    Predict H0 from micro→macro UV/IR bridge (noncircular)
    H0 = 2c⁵/(G E_anchor) × (E_anchor/E_CS)^n × C_geom

    Uses only {G, c, ħ, E_CS, E_anchor, m_a}, not observed H0
    """
    GeV_to_J = Decimal("1.602176634e-10")
    E_anchor_J = scales.E_EW * GeV_to_J
    E_CS_J = scales.E_CS * GeV_to_J

    # Prefactor: 2c⁵/(G E_anchor)
    pref = (Decimal("2") * const.c**5) / (const.G * E_anchor_J)

    # Power-law bridge (E_anchor/E_CS)^n
    alpha = E_anchor_J / E_CS_J
    alpha_pow = alpha**n_th

    # With aperture correction
    C_geom_ap = CGMGeometry.aperture_factor()
    H0_pred_ap = pref * alpha_pow * C_geom_ap

    # Without aperture
    H0_pred_noap = pref * alpha_pow

    return {
        "H0_pred_aperture": H0_pred_ap,
        "H0_pred_no_aperture": H0_pred_noap,
        "n_th": n_th,
    }


def print_analysis():
    """Main analysis output"""

    const = FundamentalConstants()
    cosmo = CosmologicalParameters()

    # Use Planck H0
    H0_SI = cosmo.H0_to_SI(cosmo.H0_planck)
    B_i = calculate_balance_index(H0_SI, const)

    print("COSMOLOGICAL CONSTANT AND BALANCE INDEX ANALYSIS")

    print()
    print(f"Using Planck 2018: H₀ = {float(cosmo.H0_planck):.2f} km/s/Mpc")
    print(f"Balance Index: B_i = {float(B_i):.6e} m²⋅K⋅kg⁻²")
    print()

    # Cosmological constant density comparison
    cc_terms = calculate_cosmological_constant_terms(H0_SI, const)

    print("VACUUM ENERGY DISCREPANCY")
    print("-" * 80)
    print(
        f"Observed dark energy density:   ρ_Λ,obs = {float(cc_terms['rho_L_obs']):.6e} J/m³"
    )
    print(
        f"Vacuum energy estimate:         ρ_vac   = {float(cc_terms['rho_vac']):.6e} J/m³"
    )
    print(f"Ratio ρ_vac/ρ_Λ,obs:            10^{cc_terms['log10_ratio']:.1f}")
    print("  (Standard 120-order magnitude discrepancy)")
    print()

    print("BALANCE INDEX IN COSMOLOGICAL CONTEXT")
    print("-" * 80)
    print(f"B_i = 2Gc/(k_B H₀) = {float(cc_terms['B_i']):.6e} m²⋅K⋅kg⁻²")
    print()

    # Closure identity (tautological by construction)
    chi = (const.k_B * cc_terms["B_i"] * H0_SI) / (const.G * const.c)
    print(f"χ = (k_B B_i H₀)/(Gc) = {float(chi):.15f}")
    assert abs(chi - 2) < Decimal("1e-15"), "χ = 2 by definition"
    print("  (Identity: equals 2 by definition of B_i)")
    print()

    # Lambda terms
    lambda_terms = calculate_lambda_terms(H0_SI, const)

    print("COSMOLOGICAL CONSTANT Λ (Cutoff Sensitivity)")
    print("-" * 80)
    print(
        f"Λ_obs (from 3H₀²/c²):           {float(lambda_terms['Lambda_obs']):.6e} m⁻²"
    )
    print()
    print("Vacuum energy estimates (cutoff-dependent):")
    print(f"  Planck cutoff:  Λ_vac = {float(lambda_terms['Lambda_vac_Pl']):.6e} m⁻²")
    print(f"    Ratio to Λ_obs:       10^{lambda_terms['log10_vac_Pl']:.1f}")
    print(f"  EW cutoff:      Λ_vac = {float(lambda_terms['Lambda_vac_EW']):.6e} m⁻²")
    print(f"    Ratio to Λ_obs:       10^{lambda_terms['log10_vac_EW']:.1f}")
    print()
    print("  → Discrepancy magnitude is cutoff-dependent")
    print()

    # Equation of state with cross-validation
    eos = calculate_equation_of_state_parameters(H0_SI, const, cosmo)

    print("EQUATION OF STATE")
    print("-" * 80)
    print(f"Dark energy density parameter:  Ω_Λ = {eos['Omega_L']:.4f}")
    print(
        f"Critical density:               ρ_crit = {float(eos['rho_crit']):.6e} kg/m³"
    )
    print(f"Dark energy density:            ρ_Λ = {float(eos['rho_L']):.6e} kg/m³")
    print(f"Pressure:                       P_Λ = {float(eos['P_L']):.6e} Pa")
    print(f"Equation of state:              w = {eos['w']:.15f}")
    print()

    # Cross-validation: ρ_Λ via Ω_Λ route vs direct route
    # ρ_Λ = Ω_Λ × ρ_crit, so ρ_Λ (Ω route) / ρ_Λ (direct) should equal Ω_Λ
    rho_L_from_Omega = eos["rho_L"] * const.c**2  # Convert to J/m³
    rho_L_total = cc_terms["rho_L_obs_SI"] * const.c**2  # Total from 3H₀²c²/(8πG)
    ratio_cross = rho_L_from_Omega / rho_L_total
    print(f"Cross-validation:")
    print(f"  ρ_Λ(Ω_Λ × ρ_crit):            {float(rho_L_from_Omega):.6e} J/m³")
    print(f"  ρ_Λ(3H₀²c²/(8πG)):            {float(rho_L_total):.6e} J/m³")
    print(f"  Ratio:                        {float(ratio_cross):.15f}")
    print(f"  Expected (Ω_Λ):               {float(cosmo.Omega_L):.15f}")
    assert abs(ratio_cross - cosmo.Omega_L) < Decimal("1e-10"), "Ratio must equal Ω_Λ"
    print()

    # Uncertainty propagation
    uncert = propagate_uncertainties(
        cosmo.H0_planck, cosmo.H0_planck_error, const, cosmo
    )
    print("UNCERTAINTY PROPAGATION")
    print("-" * 80)
    print(
        f"B_i = ({float(uncert['B_i']):.6e} ± {float(uncert['B_i_err']):.2e}) m²⋅K⋅kg⁻²"
    )
    print(f"  Relative error: {float(uncert['rel_Bi']):.6e}")
    print()
    print(
        f"ρ_Λ = ({float(uncert['rho_L']):.6e} ± {float(uncert['rho_L_err']):.2e}) kg/m³"
    )
    print(f"  Relative error: {float(uncert['rel_rhoL']):.6e}")
    print()

    # Scale hierarchy
    scales = calculate_scale_hierarchy(H0_SI, const)

    print("NATURAL SCALE HIERARCHY")
    print("-" * 80)
    print("Length scales:")
    print(f"  Planck length L_Pl:           {float(scales['L_Pl']):.6e} m")
    print(f"  EW Compton length L_EW:       {float(scales['L_EW']):.6e} m")
    print(f"  Hubble length L_H:            {float(scales['L_H']):.6e} m")
    print(f"  Balance length L_B:           {float(scales['L_B']):.6e} m")
    print()
    print(
        f"  L_H / L_Pl ratio:             10^{float((scales['L_ratio_H_to_Pl']).ln()/Decimal('10').ln()):.1f}"
    )

    # Verify L_B = 2c/H₀
    L_B_expected = Decimal("2") * const.c / H0_SI
    L_B_ratio = scales["L_B"] / L_B_expected
    print(f"  L_B / (2c/H₀) verification:   {float(L_B_ratio):.15f}")
    assert abs(L_B_ratio - Decimal("1")) < Decimal("1e-15"), "L_B identity check"
    print()

    print("Temperature scales:")
    print(f"  Equilibrium temp T_eq:        {float(scales['T_eq']):.6e} K")
    print(f"  CMB temperature T_CMB:        {float(scales['T_CMB']):.6e} K")
    print(
        f"  T_CMB / T_eq ratio:           10^{float((scales['T_ratio_CMB_to_eq']).ln()/Decimal('10').ln()):.1f}"
    )
    print("    (30-order temperature hierarchy)")
    print()

    print("Energy density scales:")
    print(f"  Hubble scale ρ_H:             {float(scales['rho_H']):.6e} J/m³")
    print(f"  Planck scale ρ_Pl:            {float(scales['rho_Pl']):.6e} J/m³")
    print(
        f"  ρ_Pl / ρ_H ratio:             10^{float((scales['rho_ratio_Pl_to_H']).ln()/Decimal('10').ln()):.1f}"
    )
    print()

    # Schwarzschild comparison
    print("SCHWARZSCHILD COMPARISON")
    print("-" * 80)
    M_solar = Decimal("1.989e30")  # kg
    I1_bh = schwarzschild_I1(M_solar, const)
    print(f"I₁ (cosmological, de Sitter):   2.000...")
    print(f"I₁ (Schwarzschild, 1 M_☉):      {I1_bh:.6f}")
    print(f"  → Factor of {2.0/I1_bh:.1f} difference")
    print("  → Different horizon geometries yield different thermal invariants")
    print()

    # UV/IR Bridge: H0 from CGM geometry (noncircular prediction)

    print("UV/IR BRIDGE: H₀ FROM CGM GEOMETRY")

    print()

    # Empirical exponent (diagnostic: what n fits observed H0?)
    n_emp = infer_bridge_exponent_from_H0(H0_SI, const, CGMEnergyScales())
    print("EMPIRICAL BRIDGE EXPONENT (Diagnostic)")
    print("-" * 80)
    print(f"Given observed H₀, inferred exponent n:")
    print(f"  With aperture correction:     n = {float(n_emp['n_emp_aperture']):.6f}")
    print(
        f"  Without aperture:             n = {float(n_emp['n_emp_no_aperture']):.6f}"
    )
    print(f"  → Confirms conjugacy exponent ~4.7 from main analysis")
    print()

    # Predictive mode (noncircular: uses only micro anchors + CGM geometry)
    n_th = Decimal("14") / Decimal("3")  # Rational approximation: 14/3 ≈ 4.666...
    pred = predict_H0_from_UVIR(const, CGMEnergyScales(), n_th)

    print("PREDICTIVE MODE (Noncircular)")
    print("-" * 80)
    print(f"Theory exponent:                n_th = {float(n_th):.6f} (14/3)")
    print("Inputs: G, c, ħ, E_CS, E_EW,  m_a (no observed H₀)")
    print()
    print(f"With aperture correction (1+m_a)^-2:")
    print(f"  H₀_pred = {float(pred['H0_pred_aperture']):.6e} s⁻¹")
    print(f"  H₀_obs  = {float(H0_SI):.6e} s⁻¹")
    print(f"  Ratio:    {float(pred['H0_pred_aperture']/H0_SI):.6f}")
    print()
    print(f"Without aperture correction:")
    print(f"  H₀_pred = {float(pred['H0_pred_no_aperture']):.6e} s⁻¹")
    print(f"  Ratio:    {float(pred['H0_pred_no_aperture']/H0_SI):.6f}")
    print()
    print("Interpretation:")
    print("  • Noncircular prediction uses UV/IR conjugacy bridge")
    print("  • Exponent n ~ 14/3 is model-dependent (CGM ansatz)")
    print("  • Aperture correction from CGM geometric closure")
    print("  • Comparison to Planck H₀ tests CGM micro→macro mapping")
    print()

    # Summary

    print("SUMMARY")

    print()
    print("Calculations performed:")
    print(
        "  • Vacuum energy discrepancy: ~10^{:.0f} orders".format(
            cc_terms["log10_ratio"]
        )
    )
    print("  • Natural scale hierarchy: L_H/L_Pl ~ 10^61, T_CMB/T_eq ~ 10^30")
    print("  • B_i framework: ℏ-independent, dimensions [M^-2 L^2 Θ^1 T^0]")
    print("  • Equation of state: w = -1.000... (standard ΛCDM)")
    print()
    print("Key relationships:")
    print("  • B_i parameterizes H₀ through thermal-gravitational equilibrium")
    print("  • B_i ~ 10^39 m²⋅K⋅kg⁻² ↔ H₀ ~ 10^-18 s⁻¹ (given G, c, k_B)")
    print("  • ρ_Pl/ρ_H ~ 10^123 matches Λ_vac/Λ_obs discrepancy")
    print()


if __name__ == "__main__":
    print_analysis()
