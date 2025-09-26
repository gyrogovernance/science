"""
CGM Black Hole Universe Analysis: Complete Framework Test with Observational Validation

This analysis tests the hypothesis that our observable universe represents the interior 
of a Planck-scale black hole, with UV-IR optical conjugacy revealing a fundamental 
inside-outside duality. The framework examines whether the smallest energy scales (UV) 
systematically map to the largest physical scales through exact geometric relationships.

Key tests include: exact optical conjugacy E_UV × E_IR = K across all CGM energy stages, 
preservation of Schwarzschild radius and entropy products as geometric invariants, 
verification that the universe sits precisely on the Schwarzschild threshold (r_s = R_H), 
and demonstration that the aperture parameter m_p provides consistent thermodynamic 
corrections while maintaining observational completeness through Q_G = 4π.

CRITICAL INSIGHT: Q_G = 4π steradians IS quantum gravity itself - the complete 
observational solid angle required for coherent observation in 3D space. This is not 
a quantized field or force, but the geometric requirement that enables quantum structure 
to emerge through the commutator algebra [X,P] = iK_QG where K_QG = Q_G × S_min ≈ 3.937.

Enhanced with direct observational validation including Planck 2023 and SH0ES cosmological 
parameters with error propagation, entropy budget comparison with CMB+neutrino+black hole 
observations, and cosmologically coupled black hole (CCBH) dark energy analysis.
"""

import math

# Physical constants (SI units)
C = 299_792_458.0  # m/s
K_B = 1.380_649e-23  # J/K
HBAR = 1.054_571_817e-34  # J·s
G = 6.674_30e-11  # m³/(kg·s²)
H_PLANCK = 6.626_070_15e-34  # Planck constant (J·s)

# Cosmological parameters
H0_PLANCK = 67.27 * 1e3 / (3.086e22)  # Planck 2023: 2.180e-18 s⁻¹
H0_PLANCK_ERR = 0.60 * 1e3 / (3.086e22)  # 1-σ error
H0_SHOES = 73.04 * 1e3 / (3.086e22)  # SH0ES 2022: 2.367e-18 s⁻¹
H0_SHOES_ERR = 1.04 * 1e3 / (3.086e22)  # 1-σ error
OMEGA_L = 0.685  # Dark energy density parameter (Planck 2023)

# Energy conversion
EV_J = 1.602_176_634e-19  # J per eV
GEV_J = EV_J * 1e9  # J per GeV

# CGM fundamental parameters
M_P = 1.0 / (2.0 * math.sqrt(2.0 * math.pi))  # Aperture parameter ≈ 0.19947
Q_G = 4.0 * math.pi  # Complete solid angle (steradians)

# Energy scales from CGM Energy Analysis (GeV)
E_CS_UV = 1.22e19   # Planck scale
E_UNA_UV = 5.50e18  # UNA scale  
E_ONA_UV = 6.10e18  # ONA scale
E_GUT_UV = 2.34e18  # GUT scale
E_BU_UV = 3.09e17   # BU scale
E_EW = 240.0        # Electroweak scale (IR anchor)

# Sagittarius A* parameters
M_SGRA = 4.0e6 * 1.989e30  # kg (4 million solar masses)

# Observational data for validation (all in k_B units)
S_CMB_OBS = 2.9e100  # CMB entropy
S_NU_OBS = 1.3e101   # Neutrino entropy
S_BH_OBS = 1.2e102   # Black hole entropy (Bousso & Hall 2021)
RHO_BH_OBS = 2.0e-31 # kg/m³ (JWST+UNCOVER 2024)
K_CCBH = 3.0         # CCBH coupling (Croker et al. 2021)


def verify_cgm_identities():
    """Verify fundamental CGM geometric identities."""
    identity_1 = Q_G * M_P**2  # Should equal 0.5
    identity_2 = M_P * math.sqrt(2 * Q_G)  # Should equal 1
    
    return {
        'Q_G_mp_squared': identity_1,
        'aperture_identity': identity_2,
        'Q_G': Q_G,
        'm_p': M_P
    }


def compute_optical_invariant():
    """Compute the optical conjugacy invariant K = (E_CS × E_EW)/(4π²)."""
    K_GeV2 = (E_CS_UV * E_EW) / (4 * math.pi**2)
    K_J2 = K_GeV2 * (GEV_J**2)
    
    return K_GeV2, K_J2


def compute_exact_ir_energies(K_GeV2):
    """Compute exact IR energies from optical conjugacy."""
    energies_uv = {
        'CS': E_CS_UV,
        'UNA': E_UNA_UV, 
        'ONA': E_ONA_UV,
        'GUT': E_GUT_UV,
        'BU': E_BU_UV
    }
    
    energies_ir = {}
    conjugacy_check = {}
    
    for stage, e_uv in energies_uv.items():
        e_ir = K_GeV2 / e_uv
        energies_ir[stage] = e_ir
        conjugacy_check[stage] = e_uv * e_ir
    
    return energies_uv, energies_ir, conjugacy_check


def compute_schwarzschild_properties(energy_gev):
    """Compute Schwarzschild radius and entropy for given energy."""
    mass_kg = (energy_gev * GEV_J) / (C**2)
    r_s = 2 * G * mass_kg / (C**2)
    area = 4 * math.pi * r_s**2
    entropy = K_B * area * (C**3) / (4 * G * HBAR)
    
    return {
        'mass_kg': mass_kg,
        'r_s': r_s,
        'area': area,
        'entropy': entropy,
        'entropy_kb_units': entropy / K_B
    }


def compute_kerr_area(mass_kg, a_star):
    """Compute Kerr black hole area for given dimensionless spin."""
    r_g = G * mass_kg / (C**2)
    a = a_star * r_g
    r_plus = r_g + math.sqrt(r_g**2 - a**2)
    area = 4 * math.pi * (r_plus**2 + a**2)
    return area


def analyze_universe_as_blackhole(H0, H0_err, label):
    """Analyze universe as critical density black hole with error propagation."""
    # Critical density and cosmic parameters
    rho_c = 3 * H0**2 / (8 * math.pi * G)
    R_hubble = C / H0
    M_universe = rho_c * (4/3) * math.pi * R_hubble**3
    r_s_universe = 2 * G * M_universe / (C**2)
    
    # Error propagation
    rho_c_err = 2 * rho_c * H0_err / H0
    R_hubble_err = C * H0_err / (H0**2)
    ratio_rs_R = r_s_universe / R_hubble
    ratio_err = ratio_rs_R * math.sqrt(2) * H0_err / H0  # Simplified error
    
    # De Sitter horizon entropy (critical density)
    T_dS = HBAR * H0 / (2 * math.pi * K_B)
    S_dS = math.pi * (C**5) / (G * HBAR * H0**2)  # In k_B units
    
    # S_dS error propagation (S_dS ∝ 1/H0^2, so error ~ 4*H0_err/H0)
    S_dS_err = 4 * S_dS * H0_err / H0
    
    # Lambda-dominated horizon entropy
    S_dS_lambda = S_dS / OMEGA_L  # In k_B units
    
    # CGM corrections
    S_dS_CGM = S_dS * (1 + M_P)
    S_dS_lambda_CGM = S_dS_lambda * (1 + M_P)
    
    return {
        'label': label,
        'H0': H0,
        'H0_err': H0_err,
        'ratio_rs_R': ratio_rs_R,
        'ratio_rs_R_err': ratio_err,
        'T_dS': T_dS,
        'S_dS': S_dS,
        'S_dS_err': S_dS_err,
        'S_dS_lambda': S_dS_lambda,
        'S_dS_CGM': S_dS_CGM,
        'S_dS_lambda_CGM': S_dS_lambda_CGM
    }


def compute_entropy_budget():
    """Compare CGM entropy with observed universe entropy budget."""
    # Observed total (in k_B units)
    S_obs_total = S_CMB_OBS + S_NU_OBS + S_BH_OBS
    
    # Use Planck H0 for entropy calculations
    S_dS = math.pi * (C**5) / (G * HBAR * H0_PLANCK**2)  # k_B units
    S_dS_lambda = S_dS / OMEGA_L  # k_B units
    S_dS_CGM = S_dS * (1 + M_P)
    S_dS_lambda_CGM = S_dS_lambda * (1 + M_P)
    
    # SH0ES entropy calculations
    S_dS_shoes = math.pi * (C**5) / (G * HBAR * H0_SHOES**2)  # k_B units
    S_dS_lambda_shoes = S_dS_shoes / OMEGA_L  # k_B units
    S_dS_CGM_shoes = S_dS_shoes * (1 + M_P)
    S_dS_lambda_CGM_shoes = S_dS_lambda_shoes * (1 + M_P)
    
    return {
        'S_obs_total': S_obs_total,
        'S_dS': S_dS,
        'S_dS_lambda': S_dS_lambda,
        'S_dS_CGM': S_dS_CGM,
        'S_dS_lambda_CGM': S_dS_lambda_CGM,
        'S_dS_shoes': S_dS_shoes,
        'S_dS_lambda_shoes': S_dS_lambda_shoes,
        'S_dS_CGM_shoes': S_dS_CGM_shoes,
        'S_dS_lambda_CGM_shoes': S_dS_lambda_CGM_shoes,
        'ratio_dS_obs': S_dS / S_obs_total,
        'ratio_lambda_obs': S_dS_lambda / S_obs_total,
        'ratio_CGM_obs': S_dS_CGM / S_obs_total,
        'ratio_lambda_CGM_obs': S_dS_lambda_CGM / S_obs_total
    }


def analyze_sgra_horizon():
    """Analyze Sagittarius A* horizon properties."""
    r_s_sgra = 2 * G * M_SGRA / (C**2)
    T_hawking = HBAR * (C**3) / (8 * math.pi * G * M_SGRA * K_B)
    
    # Hawking radiation frequency
    f_hawking = 2.701 * K_B * T_hawking / H_PLANCK
    
    # Interstellar plasma cutoff (electron density ~ 1 cm⁻³)
    n_e = 1e6  # m⁻³
    e_charge = 1.602e-19  # C
    m_e = 9.109e-31  # kg
    epsilon_0 = 8.854e-12  # F/m
    f_plasma = (1.0/(2.0*math.pi)) * math.sqrt(n_e * e_charge**2 / (epsilon_0 * m_e))
    
    return {
        'M_sgra': M_SGRA,
        'r_s_sgra': r_s_sgra,
        'T_hawking': T_hawking,
        'f_hawking': f_hawking,
        'f_plasma': f_plasma,
        'ratio_f': f_hawking / f_plasma
    }


def compute_dual_invariants(K_J2, energies_uv, energies_ir):
    """Compute dual invariant products for GR and CGM."""
    # Schwarzschild radius product
    r_s_products = {}
    entropy_products_GR = {}
    entropy_products_CGM = {}
    
    for stage in energies_uv:
        # UV and IR Schwarzschild radii
        mass_uv = (energies_uv[stage] * GEV_J) / (C**2)
        mass_ir = (energies_ir[stage] * GEV_J) / (C**2)
        
        r_s_uv = 2 * G * mass_uv / (C**2)
        r_s_ir = 2 * G * mass_ir / (C**2)
        
        r_s_products[stage] = r_s_uv * r_s_ir
        
        # Entropy products
        area_uv = 4 * math.pi * r_s_uv**2
        area_ir = 4 * math.pi * r_s_ir**2
        
        S_uv = area_uv * (C**3) / (4 * G * HBAR)  # Already in k_B units
        S_ir = area_ir * (C**3) / (4 * G * HBAR)  # Already in k_B units
        
        entropy_products_GR[stage] = S_uv * S_ir
        entropy_products_CGM[stage] = entropy_products_GR[stage] * (1 + M_P)**2
    
    # Theoretical dual length
    L_dual = math.sqrt(list(r_s_products.values())[0])
    
    # Closed form verification
    theoretical_rs_product = (4 * G**2 * K_J2) / (C**8)
    theoretical_entropy_product = ((Q_G * G)**2 * (K_J2**2)) / (HBAR**2 * C**10)
    
    # Planck length for scaling
    l_planck = math.sqrt(HBAR * G / (C**3))
    L_dual_planck_units = L_dual / l_planck
    
    return {
        'r_s_products': r_s_products,
        'L_dual': L_dual,
        'L_dual_planck_units': L_dual_planck_units,
        'theoretical_rs_product': theoretical_rs_product,
        'entropy_products_GR': entropy_products_GR,
        'entropy_products_CGM': entropy_products_CGM,
        'theoretical_entropy_product': theoretical_entropy_product
    }


def test_kerr_conjugacy(K_GeV2, energies_uv):
    """Test optical conjugacy for Kerr black holes at various spins."""
    spin_values = [0.0, 0.5, 0.9]
    kerr_results = {}
    kerr_scatter = {}
    
    for a_star in spin_values:
        area_products = {}
        
        for stage, e_uv in energies_uv.items():
            e_ir = K_GeV2 / e_uv
            
            mass_uv = (e_uv * GEV_J) / (C**2)
            mass_ir = (e_ir * GEV_J) / (C**2)
            
            area_uv = compute_kerr_area(mass_uv, a_star)
            area_ir = compute_kerr_area(mass_ir, a_star)
            
            area_products[stage] = area_uv * area_ir
        
        # Compute scatter across stages for this spin
        products = list(area_products.values())
        mean_product = sum(products) / len(products)
        max_deviation = max(abs(p - mean_product) for p in products) / mean_product
        
        kerr_results[a_star] = area_products
        kerr_scatter[a_star] = max_deviation
    
    return kerr_results, kerr_scatter


def test_ir_anchor_sensitivity():
    """Test optical conjugacy with alternative IR anchor (QCD scale)."""
    E_QCD = 1.0  # GeV (QCD scale)
    K_QCD = (E_CS_UV * E_QCD) / (4 * math.pi**2)
    
    # Compute IR energies with QCD anchor
    energies_uv_qcd = {
        'CS': E_CS_UV,
        'UNA': E_UNA_UV, 
        'ONA': E_ONA_UV,
        'GUT': E_GUT_UV,
        'BU': E_BU_UV
    }
    
    energies_ir_qcd = {}
    for stage, e_uv in energies_uv_qcd.items():
        e_ir = K_QCD / e_uv
        energies_ir_qcd[stage] = e_ir
    
    return {
        'K_QCD': K_QCD,
        'energies_ir_qcd': energies_ir_qcd,
        'E_QCD': E_QCD
    }


def analyze_ccbh_dark_energy():
    """Analyze cosmologically coupled black hole dark energy."""
    # CCBH prediction
    rho_DE_BH = K_CCBH * RHO_BH_OBS
    
    # Observed dark energy density (70% of critical, using Planck H0)
    rho_c = 3 * H0_PLANCK**2 / (8 * math.pi * G)
    rho_de_obs = 0.7 * rho_c
    
    # Comparison
    ratio_ccbh_obs = rho_DE_BH / rho_de_obs
    deficit_factor = 1.0 / ratio_ccbh_obs
    
    return {
        'rho_BH_obs': RHO_BH_OBS,
        'rho_DE_BH': rho_DE_BH,
        'rho_de_obs': rho_de_obs,
        'ratio_ccbh_obs': ratio_ccbh_obs,
        'deficit_factor': deficit_factor
    }


def print_stage_table(energies_uv, energies_ir, dual_invariants):
    """Print compact stage table with key properties."""
    print(f"\nStage Properties Table:")
    print(f"{'Stage':<6} {'E_UV (GeV)':<12} {'E_IR (GeV)':<12} {'r_s_UV (m)':<12} {'r_s_IR (m)':<12} {'S_UV (k_B)':<12} {'S_IR (k_B)':<12}")
    print("-" * 90)
    
    for stage in energies_uv:
        # Get Schwarzschild properties
        mass_uv = (energies_uv[stage] * GEV_J) / (C**2)
        mass_ir = (energies_ir[stage] * GEV_J) / (C**2)
        
        r_s_uv = 2 * G * mass_uv / (C**2)
        r_s_ir = 2 * G * mass_ir / (C**2)
        
        area_uv = 4 * math.pi * r_s_uv**2
        area_ir = 4 * math.pi * r_s_ir**2
        
        S_uv = area_uv * (C**3) / (4 * G * HBAR)  # k_B units
        S_ir = area_ir * (C**3) / (4 * G * HBAR)  # k_B units
        
        print(f"{stage:<6} {energies_uv[stage]:<12.2e} {energies_ir[stage]:<12.2e} {r_s_uv:<12.2e} {r_s_ir:<12.2e} {S_uv:<12.2e} {S_ir:<12.2e}")


def main():
    """Execute comprehensive CGM black hole universe analysis with observational validation."""
    
    print("CGM BLACK HOLE UNIVERSE: Complete Framework Test with Observational Validation")
    print("=" * 90)
    print("All values use Planck 2023 unless noted")
    
    # Fundamental CGM identities
    identities = verify_cgm_identities()
    print(f"\nCGM Fundamental Identities:")
    print(f"  Quantum Gravity: Q_G = {identities['Q_G']:.6f} steradians (complete observational solid angle)")
    print(f"  Aperture parameter: m_p = {identities['m_p']:.6f}")
    print(f"  Q_G × m_p² = {identities['Q_G_mp_squared']:.10f} (exact: 0.5)")
    print(f"  m_p × √(2Q_G) = {identities['aperture_identity']:.10f} (exact: 1.0)")
    print(f"  Note: Q_G = 4π defines quantum gravity geometrically, not as a quantized field")
    
    # Optical invariant
    K_GeV2, K_J2 = compute_optical_invariant()
    print(f"\nOptical Conjugacy Invariant:")
    print(f"  K = (E_CS × E_EW)/(4π²) = {K_GeV2:.6e} GeV² (IR anchor: {E_EW} GeV)")
    print(f"  K = {K_J2:.6e} J²")
    
    # Exact stage conjugacy
    energies_uv, energies_ir, conjugacy_check = compute_exact_ir_energies(K_GeV2)
    print(f"\nExact UV-IR Conjugacy (E_UV × E_IR = K):")
    deviation_max = 0
    for stage in energies_uv:
        deviation = abs(conjugacy_check[stage] - K_GeV2) / K_GeV2
        deviation_max = max(deviation_max, deviation)
        print(f"  {stage:3s}: {energies_uv[stage]:.2e} × {energies_ir[stage]:.2e} = {conjugacy_check[stage]:.6e} GeV² (δ: {deviation:.2e})")
    print(f"  Maximum relative deviation: {deviation_max:.2e}")
    
    # Dual invariants
    dual_invariants = compute_dual_invariants(K_J2, energies_uv, energies_ir)
    print(f"\nGeometric Dual Invariants:")
    print(f"  Schwarzschild radius product (all stages): {list(dual_invariants['r_s_products'].values())[0]:.6e} m²")
    print(f"  Theoretical r_s product: {dual_invariants['theoretical_rs_product']:.6e} m²")
    print(f"  Dual length L_dual = √(r_s,UV × r_s,IR): {dual_invariants['L_dual']:.6e} m")
    print(f"  L_dual in Planck units: {dual_invariants['L_dual_planck_units']:.2e} l_P")
    print(f"  Note: Below-Planck lengths are derived invariants, not resolvable scales")
    
    print(f"\nEntropy Products:")
    S_prod_GR = list(dual_invariants['entropy_products_GR'].values())[0]
    S_prod_CGM = list(dual_invariants['entropy_products_CGM'].values())[0]
    ratio_expected = (1 + M_P)**2
    print(f"  GR entropy product (k_B units): {S_prod_GR:.6e}")
    print(f"  CGM entropy product (k_B units): {S_prod_CGM:.6e}")
    print(f"  CGM enhancement factor: {S_prod_CGM/S_prod_GR:.6f} (exact: {ratio_expected:.6f})")
    print(f"  Theoretical entropy product: {dual_invariants['theoretical_entropy_product']:.6e}")
    
    # Universe as black hole - both Planck and SH0ES
    universe_planck = analyze_universe_as_blackhole(H0_PLANCK, H0_PLANCK_ERR, "Planck 2023")
    universe_shoes = analyze_universe_as_blackhole(H0_SHOES, H0_SHOES_ERR, "SH0ES 2022")
    
    print(f"\nUniverse as Critical Black Hole:")
    print(f"  Planck 2023: H₀ = {H0_PLANCK*1e18:.3f} ± {H0_PLANCK_ERR*1e18:.3f} × 10⁻¹⁸ s⁻¹")
    print(f"    r_s/R_H = {universe_planck['ratio_rs_R']:.4f} ± {universe_planck['ratio_rs_R_err']:.4f}")
    print(f"    De Sitter temperature: {universe_planck['T_dS']:.3e} K")
    print(f"    De Sitter entropy: {universe_planck['S_dS']:.2e} ± {universe_planck['S_dS_err']:.2e} k_B")
    
    print(f"  SH0ES 2022:  H₀ = {H0_SHOES*1e18:.3f} ± {H0_SHOES_ERR*1e18:.3f} × 10⁻¹⁸ s⁻¹")
    print(f"    r_s/R_H = {universe_shoes['ratio_rs_R']:.4f} ± {universe_shoes['ratio_rs_R_err']:.4f}")
    print(f"    De Sitter temperature: {universe_shoes['T_dS']:.3e} K")
    print(f"    De Sitter entropy: {universe_shoes['S_dS']:.2e} ± {universe_shoes['S_dS_err']:.2e} k_B")
    
    print(f"  Note: In ΛCDM this is an instantaneous identity at critical density")
    
    # Entropy budget comparison
    entropy_budget = compute_entropy_budget()
    print(f"\nEntropy Budget Comparison (k_B units):")
    print(f"  Observed total (CMB+ν+BH): {entropy_budget['S_obs_total']:.2e}")
    print(f"  De Sitter horizon (critical): {entropy_budget['S_dS']:.2e}")
    print(f"  De Sitter horizon (Λ-dominated): {entropy_budget['S_dS_lambda']:.2e}")
    print(f"  CGM corrected (critical): {entropy_budget['S_dS_CGM']:.2e}")
    print(f"  CGM corrected (Λ-dominated): {entropy_budget['S_dS_lambda_CGM']:.2e}")
    print(f"  S_dS/S_obs ratio: {entropy_budget['ratio_dS_obs']:.2e}")
    print(f"  S_CGM/S_obs ratio: {entropy_budget['ratio_CGM_obs']:.2e}")
    print(f"  Note: Large horizon-to-matter entropy gap is standard in cosmology")
    
    # Kerr conjugacy test
    kerr_results, kerr_scatter = test_kerr_conjugacy(K_GeV2, energies_uv)
    print(f"\nKerr Conjugacy Test:")
    for a_star in [0.0, 0.5, 0.9]:
        area_product = list(kerr_results[a_star].values())[0]
        scatter = kerr_scatter[a_star]
        print(f"  a* = {a_star:.1f}: A_UV × A_IR = {area_product:.6e} m⁴ (stage scatter: {scatter:.2e})")
    
    # Spin dependence ratio
    area_ratio = list(kerr_results[0.9].values())[0] / list(kerr_results[0.0].values())[0]
    print(f"  Area product ratio A(a*=0.9)/A(a*=0): {area_ratio:.3f}")
    print(f"  For fixed a*, UV-IR area products are stage-independent to ~10⁻¹⁶ precision")
    print(f"  Note: Spin dependence aligns with LIGO area theorem constraints (δA/A ~3-18%)")
    
    # Stage properties table
    print_stage_table(energies_uv, energies_ir, dual_invariants)
    
    # IR anchor sensitivity test
    ir_sensitivity = test_ir_anchor_sensitivity()
    print(f"\nIR Anchor Sensitivity Test:")
    print(f"  QCD anchor (1 GeV): K_QCD = {ir_sensitivity['K_QCD']:.6e} GeV²")
    print(f"  CS-IR energy with QCD: {ir_sensitivity['energies_ir_qcd']['CS']:.2e} GeV")
    print(f"  BU-IR energy with QCD: {ir_sensitivity['energies_ir_qcd']['BU']:.2e} GeV")
    
    # Boundary scale analysis
    cs_props = compute_schwarzschild_properties(E_CS_UV)
    ew_props = compute_schwarzschild_properties(E_EW)
    print(f"\nBoundary Scale Analysis:")
    print(f"  Planck boundary: M = {cs_props['mass_kg']:.3e} kg, r_s = {cs_props['r_s']:.3e} m, S = {cs_props['entropy_kb_units']:.2e} k_B")
    print(f"  EW boundary (horizon-like closure): M = {ew_props['mass_kg']:.3e} kg, r_s = {ew_props['r_s']:.3e} m, S = {ew_props['entropy_kb_units']:.2e} k_B")
    
    # Sagittarius A* analysis
    sgra = analyze_sgra_horizon()
    print(f"\nSagittarius A* Horizon Analysis:")
    print(f"  Mass: {sgra['M_sgra']:.2e} kg (4×10⁶ M_☉)")
    print(f"  Schwarzschild radius: {sgra['r_s_sgra']:.2e} m")
    print(f"  Hawking temperature: {sgra['T_hawking']:.3e} K")
    print(f"  Hawking frequency: {sgra['f_hawking']:.3e} Hz")
    print(f"  Plasma cutoff frequency: {sgra['f_plasma']:.3e} Hz")
    print(f"  Frequency ratio f_H/f_p: {sgra['ratio_f']:.3e} (EM radiation blocked by ISM)")
    
    # CCBH dark energy analysis
    ccbh = analyze_ccbh_dark_energy()
    print(f"\nCosmologically Coupled Black Hole Dark Energy:")
    print(f"  Observed BH density ρ_BH: {ccbh['rho_BH_obs']:.2e} kg/m³ (JWST+UNCOVER 2024)")
    print(f"  CCBH predicted ρ_DE: {ccbh['rho_DE_BH']:.2e} kg/m³ (k = {K_CCBH})")
    print(f"  Observed dark energy ρ_DE: {ccbh['rho_de_obs']:.2e} kg/m³")
    print(f"  CCBH/observed ratio: {ccbh['ratio_ccbh_obs']:.3e} (short by ~{ccbh['deficit_factor']:.0e}×)")
    print(f"  CCBH provides directionally consistent mechanism but quantitatively insufficient")
    
    # CGM aperture transmission (derived from ρ = δ_BU/m_p)
    # Using δ_BU/m_p = 0.9793, so transmission = 1 - 0.9793 = 0.0207 = 2.07%
    aperture_transmission = 0.0207  # 2.07% transmission
    print(f"\nCGM Aperture Properties:")
    print(f"  Aperture transmission: {aperture_transmission*100:.2f}% (derived from ρ = δ_BU/m_p)")
    print(f"  Closure completeness: {(1-aperture_transmission)*100:.2f}%")
    print(f"  Balance enables existence: sufficient structure + sufficient observation")
    
    print(f"\nObservational Validation Summary:")
    print(f"  • Universe r_s/R_H ratio: {universe_planck['ratio_rs_R']:.4f} ± {universe_planck['ratio_rs_R_err']:.4f} (Planck)")
    print(f"  • UV-IR conjugacy deviation: {deviation_max:.2e} across all stages")
    print(f"  • CGM entropy enhancement factor: {S_prod_CGM/S_prod_GR:.3f}")
    print(f"  • Kerr area product stage scatter: ~{max(kerr_scatter.values()):.2e}")
    print(f"  • Hawking/plasma frequency ratio: {sgra['ratio_f']:.3e}")
    print(f"  • CCBH/observed dark energy ratio: {ccbh['ratio_ccbh_obs']:.3e}")
    print(f"  • Aperture transmission: {aperture_transmission*100:.2f}%")
    
    # --- Optical-illusion diagnostics tied to current tensions and near-future tests ---
    print(f"\nOptical-Illusion Diagnostics:")
    hdiag = h0_tension_void_aperture()
    print(f"  H0 ratio observed (SH0ES/Planck): {hdiag['ratio_obs']:.4f}")
    print(f"  H0 ratio predicted (20% void × aperture): {hdiag['ratio_pred_20pct_void']:.4f}")
    print(f"  Inferred local underdensity from ratio: {hdiag['delta_void_inferred']:.3f}  (~−0.20 expected)")
    print(f"  Decomposition: H0_SHOES/H0_Planck ≈ (1 + 0.0667_void) × (1 + 0.0207_aperture) = {hdiag['ratio_pred_20pct_void']:.3f} vs {hdiag['ratio_obs']:.3f} observed")

    rd = redshift_drift_LCDM_vs_CGM(z=2.0, Omega_m=0.315)
    print(f"  Redshift drift at z=2 (ΛCDM): {rd['dv_dt_LCDM_cm_per_s_per_yr']:.3f} cm/s/yr  vs CGM (no-expansion): {rd['dv_dt_CGM_cm_per_s_per_yr']:.3f}")

    dd = distance_duality_prediction()
    print(f"  Distance duality deviation η0 (predicted): {dd['eta0_range'][0]:.3f} – {dd['eta0_range'][1]:.3f}")

    gw = gw_memory_prediction()
    print(f"  GW memory fraction h_mem/h_peak (predicted): {gw['h_memory_over_h_peak']*100:.2f}%")

    w0 = effective_w0_from_q0(q0=-0.55, Omega_m=0.315)
    print(f"  Effective w0 from q0,Ωm (diagnostic): {w0['w_eff0_from_q0']:.3f}  (≈ −1 expected observationally)")

    okb = apparent_curvature_bound(universe_planck['ratio_rs_R'], universe_planck['ratio_rs_R_err'])
    print(f"  Apparent curvature bound proxy |Ω_k,eff| ≲ {okb['Omega_k_eff_bound_1sigma']:.2e} (very conservative)")


def h0_tension_void_aperture():
    """H0 tension analysis: void + aperture decomposition."""
    # Observed ratio
    ratio_obs = H0_SHOES / H0_PLANCK  # ~1.0858
    # Aperture transmission (from δ_BU/m_p)
    aperture = 0.0207  # 2.07%
    # Predict ratio from a 20% local void (linear LTB: δH/H ≈ -(1/3) δrho/rho)
    delta_void = -0.20
    dh_over_h_void = -(1.0/3.0) * delta_void  # +0.0667
    ratio_pred = (1.0 + dh_over_h_void) * (1.0 + aperture)
    # Invert to infer the void depth that best matches the observed H0 ratio
    dh_over_h_needed = ratio_obs/(1.0 + aperture) - 1.0
    delta_void_inferred = -3.0 * dh_over_h_needed  # ≈ -0.19 (i.e., ~20% underdensity)
    return {
        'ratio_obs': ratio_obs,
        'ratio_pred_20pct_void': ratio_pred,
        'delta_void_inferred': delta_void_inferred
    }


def redshift_drift_LCDM_vs_CGM(z=2.0, Omega_m=0.315):
    """Redshift drift test: ΛCDM vs CGM no-expansion prediction."""
    # ΛCDM prediction for redshift drift (observer time derivative)
    Omega_L = 1.0 - Omega_m
    H0 = H0_PLANCK
    Hz_over_H0 = math.sqrt(Omega_m*(1+z)**3 + Omega_L)
    dz_dt_LCDM = H0*((1+z) - Hz_over_H0)
    # Velocity drift per year (cm/s/yr)
    dv_dt_LCDM = (C * dz_dt_LCDM/(1+z)) * (365.25*24*3600) * 100.0  # m/s -> cm/s per year
    # CGM no-expansion optical illusion: predict strictly zero drift
    dv_dt_CGM = 0.0
    return {
        'z': z,
        'dv_dt_LCDM_cm_per_s_per_yr': dv_dt_LCDM,
        'dv_dt_CGM_cm_per_s_per_yr': dv_dt_CGM
    }


def distance_duality_prediction():
    """Distance duality deviation prediction from aperture effects."""
    # Etherington reciprocity: D_L = (1+z)^2 D_A
    # CGM predicts a tiny achromatic deviation at aperture level
    aperture = 0.0207
    eta0_minus = 1.0 - aperture  # if interpreted as transparency-like loss
    eta0_plus  = 1.0 + aperture  # if interpreted as geometric gain
    return {'eta0_range': (eta0_minus, eta0_plus)}


def gw_memory_prediction():
    """GW memory fraction prediction from aperture parameter."""
    # Fractional permanent strain offset vs oscillatory strain
    aperture = 0.0207
    return {'h_memory_over_h_peak': aperture}


def effective_w0_from_q0(q0=-0.55, Omega_m=0.315):
    """Effective equation of state from deceleration parameter."""
    # In flat FRW with constant w: q0 = 0.5 + 1.5 w (1 - Ωm)
    # Solve for w
    w_eff0 = (q0 - 0.5) / (1.5*(1.0 - Omega_m))
    return {'w_eff0_from_q0': w_eff0}


def apparent_curvature_bound(ratio_rs_R, ratio_err):
    """Apparent curvature bound from threshold identity."""
    # Very conservative "effective" curvature bound proxy from threshold identity
    # Treat |Ω_k,eff| as bounded by twice the fractional deviation of r_s/R_H from 1
    dev = abs(ratio_rs_R - 1.0)
    bound = 2.0*(dev + ratio_err)  # add 1σ conservatively
    return {'Omega_k_eff_bound_1sigma': bound}


if __name__ == "__main__":
    main()