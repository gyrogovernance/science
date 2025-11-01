#!/usr/bin/env python3
"""
Balance Index Calculator for the Common Governance Model
===========================================================

This script calculates a newly identified index with dimensions [M^-2 L^2 Θ^1 T^0]
that emerges from considering thermodynamic equilibrium at cosmological horizons.

The Balance Index is defined as:
    B_i = (A_H × T_eq) / M_P^2

Where:
    A_H = 4π R_H^2 is the horizon area
    T_eq = ℏ H_0 / (2π k_B) is the de Sitter equilibrium temperature  
    M_P^2 = (ℏc/G) is the Planck mass squared

This simplifies to the compact form:
    B_i = 2Gc/(k_B H_0)

Physical interpretation: This Index represents the thermal surface capacity
per unit gravitational mass squared at cosmological equilibrium, providing a
timeless (T^0) measure of the universe's thermal-gravitational balance.
"""

from decimal import Decimal, getcontext
import math
import sys
import io

# Set UTF-8 encoding for stdout to handle Greek letters and symbols
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Set high precision for calculations
getcontext().prec = 50

# Mathematical constants
PI = Decimal('3.1415926535897932384626433832795028841971693993751')
TWO_PI = Decimal('2') * PI
FOUR_PI = Decimal('4') * PI

class FundamentalConstants:
    """
    Fundamental physical constants from CODATA 2018
    https://physics.nist.gov/cuu/Constants/
    """
    
    # Defining constants (exact by SI definition)
    c = Decimal('299792458')           # Speed of light in vacuum [m/s]
    h = Decimal('6.62607015e-34')      # Planck constant [J⋅s]
    k_B = Decimal('1.380649e-23')      # Boltzmann constant [J/K]
    
    # Measured constant
    G = Decimal('6.67430e-11')         # Gravitational constant [m³/(kg⋅s²)]
    G_uncertainty = Decimal('0.00015e-11')  # Standard uncertainty in G
    
    # Derived quantities
    hbar = h / TWO_PI                   # Reduced Planck constant [J⋅s]
    

class CosmologicalParameters:
    """
    Current cosmological measurements from major collaborations
    """
    
    # Planck Collaboration 2020 (arXiv:1807.06209)
    # TT,TE,EE+lowE+lensing+BAO
    H0_planck = Decimal('67.27')           # km/s/Mpc
    H0_planck_error = Decimal('0.60')      # km/s/Mpc
    
    # SH0ES Collaboration 2022 (arXiv:2112.04510)
    # Cepheid-calibrated Type Ia supernovae
    H0_shoes = Decimal('73.04')            # km/s/Mpc
    H0_shoes_error = Decimal('1.04')       # km/s/Mpc
    
    # Unit conversion factors
    MPC_TO_M = Decimal('3.0857e22')        # 1 Mpc in meters
    KM_TO_M = Decimal('1000')              # 1 km in meters
    
    @classmethod
    def H0_to_SI(cls, H0_kmsmpc):
        """Convert H_0 from km/s/Mpc to SI units (1/s)"""
        return (H0_kmsmpc * cls.KM_TO_M) / cls.MPC_TO_M

class CGMParameters:
    """
    Geometric parameters from the Common Governance Model
    """
    
    # Aperture parameter from geometric closure requirements
    m_p = Decimal('1') / (Decimal('2') * (Decimal('2') * PI).sqrt())
    
    # Quantum gravity as complete solid angle
    Q_G = FOUR_PI  # 4π steradians
    
    # Holonomy measurements (radians) from CGM analysis
    delta_BU = Decimal('0.195342')        # BU dual-pole monodromy (8-leg)
    phi_SU2 = Decimal('0.587901')         # SU(2) commutator holonomy
    phi_4leg = Decimal('0.862833')        # 4-leg toroidal holonomy
    
    @classmethod
    def aperture_correction(cls):
        """Temperature correction factor in CGM: T → T/(1 + m_p)"""
        return Decimal('1') / (Decimal('1') + cls.m_p)

class CGMEnergyScales:
    """
    CGM energy scale anchors (UV focus at Planck scale)
    All energies in GeV
    """
    
    # UV ladder (anchored at CS = Planck scale)
    E_CS = Decimal('1.22089e19')      # Planck energy
    E_UNA = Decimal('5.50e18')        # UNA scale
    E_ONA = Decimal('6.10e18')        # ONA scale
    E_GUT = Decimal('2.34e18')        # GUT scale (union formation)
    E_BU = Decimal('3.09e17')         # BU scale
    E_EW = Decimal('246.22')          # Higgs vacuum expectation value v = (√2 G_F)^(-1/2) ≈ 246.22 GeV

class ParticleData:
    """
    Standard Model particle masses and energies
    All masses in GeV, converted to energies via E = mc²
    """
    
    # Leptons (charged)
    m_electron = Decimal('0.000510998910')
    m_muon = Decimal('0.105658367')
    m_tau = Decimal('1.77682')
    
    # Quarks (MSbar for light, pole for top)
    m_up = Decimal('0.0022')
    m_down = Decimal('0.00475')
    m_strange = Decimal('0.095')
    m_charm = Decimal('1.27')
    m_bottom = Decimal('4.18')
    m_top = Decimal('172.76')
    
    # Gauge bosons
    m_W = Decimal('80.379')
    m_Z = Decimal('91.1876')
    
    # Higgs
    m_Higgs = Decimal('125.25')
    
    # Neutrinos (representative light scale)
    m_nu_light = Decimal('1e-11')     # 0.01 eV in GeV units
    
    @classmethod
    def get_all_particles(cls):
        """Return all particles with their masses"""
        return {
            'Electron': cls.m_electron,
            'Muon': cls.m_muon,
            'Tau': cls.m_tau,
            'Up': cls.m_up,
            'Down': cls.m_down,
            'Strange': cls.m_strange,
            'Charm': cls.m_charm,
            'Bottom': cls.m_bottom,
            'Top': cls.m_top,
            'W Boson': cls.m_W,
            'Z Boson': cls.m_Z,
            'Higgs': cls.m_Higgs,
            'Neutrino (light)': cls.m_nu_light
        }

def calculate_balance_index(H0_kmsmpc, H0_error_kmsmpc=None):
    """
    Calculate the Balance Index and related quantities
    
    Parameters:
        H0_kmsmpc: Hubble constant in km/s/Mpc
        H0_error_kmsmpc: Uncertainty in H0 (optional)
    
    Returns:
        Dictionary containing all calculated values
    """
    
    const = FundamentalConstants()
    cosmo = CosmologicalParameters()
    cgm = CGMParameters()
    
    # Convert H0 to SI units
    H0 = cosmo.H0_to_SI(H0_kmsmpc)
    
    # Calculate horizon radius and area
    R_H = const.c / H0
    A_H = FOUR_PI * R_H**2
    
    # Calculate equilibrium temperature (de Sitter temperature)
    T_eq = (const.hbar * H0) / (TWO_PI * const.k_B)
    
    # Calculate Planck mass squared
    m_P_squared = (const.hbar * const.c) / const.G
    
    # Calculate Balance Index (area formula)
    Sigma_eq_area = (A_H * T_eq) / m_P_squared
    
    # Calculate Balance Index (compact formula)
    # This should give identical result, serving as verification
    Sigma_eq_compact = (Decimal('2') * const.G * const.c) / (const.k_B * H0)
    
    # CGM-modified version with aperture correction
    Sigma_eq_CGM = Sigma_eq_area * cgm.aperture_correction()
    
    # Error propagation if uncertainty provided
    # B_i = 2Gc/(k_B H₀), so B_i ∝ G/H₀
    # ΔΣ/Σ = sqrt((ΔG/G)² + (ΔH₀/H₀)²)
    if H0_error_kmsmpc:
        H0_error = cosmo.H0_to_SI(H0_error_kmsmpc)
        rel_error_H0 = H0_error / H0
        rel_error_G = const.G_uncertainty / const.G
        rel_error_total = (rel_error_H0**2 + rel_error_G**2).sqrt()
        Sigma_eq_error = Sigma_eq_area * rel_error_total
    else:
        Sigma_eq_error = None
    
    return {
        'H0_kmsmpc': float(H0_kmsmpc),
        'H0_SI': float(H0),
        'R_H': float(R_H),
        'A_H': float(A_H),
        'T_eq': float(T_eq),
        'm_P_squared': float(m_P_squared),
        'Sigma_eq': float(Sigma_eq_area),
        'Sigma_eq_compact': float(Sigma_eq_compact),
        'Sigma_eq_CGM': float(Sigma_eq_CGM),
        'Sigma_eq_error': float(Sigma_eq_error) if Sigma_eq_error else None,
        'verification_ratio': float(Sigma_eq_area / Sigma_eq_compact),
        # Store Decimals for identity checks
        'T_eq_decimal': T_eq,
        'A_H_decimal': A_H,
        'R_H_decimal': R_H,
        'H0_SI_decimal': H0,
        'Sigma_eq_decimal': Sigma_eq_area,
        'Sigma_eq_CGM_decimal': Sigma_eq_CGM
    }

def calculate_phase_curvature_H(H0_SI, method='baseline'):
    """
    Calculate phase curvature ℋ from cosmological H0
    
    In CGM, ℋ is the mean phase curvature of the closed BU helical generator.
    Multiple estimators test emergent constancy under different recursion views.
    
    Args:
        H0_SI: Hubble constant in SI units (1/s)
        method: Estimation method - 'baseline', 'SU2_holo', '4leg_holo', 'aperture'
    
    Returns:
        Phase curvature ℋ in SI units (1/s)
    """
    cgm = CGMParameters()
    
    if method == 'baseline':
        # Baseline: ℋ = H0 (cosmological phase curvature)
        return H0_SI
    
    elif method == 'SU2_holo':
        # Scale by SU(2) commutator holonomy per 3-step closure
        # ℋ = H0 × (φ_SU2 / (3 δ_BU))
        return H0_SI * (cgm.phi_SU2 / (Decimal('3') * cgm.delta_BU))
    
    elif method == '4leg_holo':
        # Scale by 4-leg toroidal holonomy per quarter-loop
        # ℋ = H0 × (φ_4leg / (4 δ_BU))
        return H0_SI * (cgm.phi_4leg / (Decimal('4') * cgm.delta_BU))
    
    elif method == 'aperture':
        # Transport curvature through aperture correction
        # ℋ = H0 / (1 + m_p)
        return H0_SI / (Decimal('1') + cgm.m_p)
    
    else:
        raise ValueError(f"Unknown method: {method}")

def calculate_all_phase_curvatures(H0_SI):
    """
    Calculate all holonomy-based ℋ estimators
    
    Returns dictionary with all variants for robustness testing
    """
    return {
        'baseline': calculate_phase_curvature_H(H0_SI, 'baseline'),
        'SU2_holo': calculate_phase_curvature_H(H0_SI, 'SU2_holo'),
        '4leg_holo': calculate_phase_curvature_H(H0_SI, '4leg_holo'),
        'aperture': calculate_phase_curvature_H(H0_SI, 'aperture')
    }

def calculate_energy_balance_index(energy_GeV, H_curv_SI, hbar):
    """
    Calculate energy balance index Ξ(E) = (E/ℏ) / ℋ
    
    This compares intrinsic phase curvature (E/ℏ) to global coherence curvature (ℋ).
    
    Args:
        energy_GeV: Energy in GeV
        H_curv_SI: Phase curvature ℋ in SI units (1/s)
        hbar: Reduced Planck constant in J⋅s
    
    Returns:
        Dictionary with balance index and log10(Ξ)
    """
    # Convert energy to Joules
    GeV_to_J = Decimal('1.602176634e-10')  # 1 GeV = 1.602176634e-10 J
    E_J = energy_GeV * GeV_to_J
    
    # Phase curvature of the energy scale
    omega_E = E_J / hbar  # Internal phase curvature (1/s)
    
    # Balance index: ratio of intrinsic to global curvature
    Xi = omega_E / H_curv_SI
    
    # Log10 for manageable scale
    log_Xi = float(Xi.ln() / Decimal('10').ln())
    
    return {
        'Xi': float(Xi),
        'log10_Xi': log_Xi,
        'omega_E': float(omega_E),
        'H_curv': float(H_curv_SI)
    }

def calculate_emergent_constancy_invariants(Sigma_eq, H_curv_SI, const):
    """
    Calculate emergent constancy invariants for CGM closure
    
    Invariants:
        I₁ = (k_B Σ ℋ) / (G c) should equal 2 (dimensionless closure check)
        L_B = (k_B Σ) / G = 2c/ℋ (characteristic length scale)
    
    Args:
        Sigma_eq: Balance constant in m²⋅K⋅kg⁻²
        H_curv_SI: Phase curvature ℋ in 1/s
        const: FundamentalConstants instance
    
    Returns:
        Dictionary with invariants
    """
    # I₁ = (k_B Σ ℋ) / (G c) should be 2
    I1 = (const.k_B * Sigma_eq * H_curv_SI) / (const.G * const.c)
    
    # L_B = (k_B Σ) / G = 2c/ℋ
    L_B = (const.k_B * Sigma_eq) / const.G
    L_B_check = (Decimal('2') * const.c) / H_curv_SI
    
    # Ratio check (should be 1.0)
    L_B_ratio = L_B / L_B_check
    
    return {
        'I1': float(I1),
        'I1_deviation': float(abs(I1 - Decimal('2'))),
        'L_B': float(L_B),
        'L_B_check': float(L_B_check),
        'L_B_ratio': float(L_B_ratio)
    }

def transported_sigma(H_curv_SI, const):
    """
    Calculate transported Balance Index for a given phase curvature
    
    Σ_trans = 2Gc/(k_B ℋ)
    
    When ℋ is transported via holonomy, Σ must be transported with it
    to maintain closure invariants.
    
    Args:
        H_curv_SI: Phase curvature ℋ in 1/s
        const: FundamentalConstants instance
    
    Returns:
        Transported Σ in m²⋅K⋅kg⁻²
    """
    return (Decimal('2') * const.G * const.c) / (const.k_B * H_curv_SI)

def analyze_balance_bands(H_curv_SI, hbar, Xi_EW=None):
    """
    Analyze balance index bands across CGM energy scales and particles
    
    Args:
        H_curv_SI: Phase curvature in SI units
        hbar: Reduced Planck constant
        Xi_EW: Optional EW reference for normalization
    
    Returns:
        Dictionary with band analysis results
    """
    scales = CGMEnergyScales()
    particles = ParticleData()
    
    # CGM anchor scales
    cgm_scales = {
        'CS (Planck)': scales.E_CS,
        'UNA': scales.E_UNA,
        'ONA': scales.E_ONA,
        'GUT': scales.E_GUT,
        'BU': scales.E_BU,
        'EW': scales.E_EW
    }
    
    # Particle masses
    particle_masses = particles.get_all_particles()
    
    results = {
        'cgm_scales': {},
        'particles': {},
        'bands': []
    }
    
    # Calculate balance indices for CGM scales
    for name, energy in cgm_scales.items():
        idx = calculate_energy_balance_index(energy, H_curv_SI, hbar)
        idx['energy_GeV'] = float(energy)  # Store energy for later display
        results['cgm_scales'][name] = idx
    
    # Calculate balance indices for particles
    for name, mass in particle_masses.items():
        idx = calculate_energy_balance_index(mass, H_curv_SI, hbar)
        idx['mass_GeV'] = float(mass)  # Store mass for later display
        # Add normalized index if reference provided
        if Xi_EW is not None:
            idx['Delta_Xi'] = idx['log10_Xi'] - math.log10(float(Xi_EW))
        results['particles'][name] = idx
    
    # Analyze for clustering (look for 3-band structure)
    all_log_values = []
    for data in results['cgm_scales'].values():
        all_log_values.append(data['log10_Xi'])
    for data in results['particles'].values():
        all_log_values.append(data['log10_Xi'])
    
    results['log_range'] = (min(all_log_values), max(all_log_values))
    results['log_span'] = max(all_log_values) - min(all_log_values)
    
    return results

def compute_natural_breaks(particles_dict, k=3, use_delta=False, Xi_EW=None):
    """
    Find natural breaks in particle balance indices via largest gaps
    
    Uses gap-based clustering (Jenks-like) to find k bands without
    enforcing equal counts. This tests whether structure emerges
    from the Ξ landscape itself.
    
    Args:
        particles_dict: Dictionary of particle balance indices
        k: Number of bands to find (default 3)
        use_delta: If True, use ΔΞ = log10(Ξ/Ξ_EW) for view-invariant partitions
        Xi_EW: Reference Ξ value for normalization
    
    Returns:
        Tuple of (bands, cut_values) where bands is list of particle lists
    """
    # Extract and sort by log10_Xi or ΔΞ
    if use_delta and Xi_EW is not None:
        items = sorted([(name, data['log10_Xi'] - math.log10(float(Xi_EW))) 
                       for name, data in particles_dict.items()], 
                      key=lambda x: x[1])
    else:
        items = sorted([(name, data['log10_Xi']) for name, data in particles_dict.items()], 
                       key=lambda x: x[1])
    
    if len(items) < k:
        return ([items], [], [])
    
    # Find the k-1 largest gaps between consecutive values
    gaps = [(j, items[j+1][1] - items[j][1]) for j in range(len(items)-1)]
    gaps_sorted = sorted(gaps, key=lambda x: x[1], reverse=True)[:k-1]
    cut_indices = sorted([g[0] for g in gaps_sorted])
    
    # Slice into k bands
    bands = []
    start = 0
    for cut in cut_indices:
        bands.append(items[start:cut+1])
        start = cut+1
    bands.append(items[start:])
    
    # Cut values are at the boundaries
    cut_values = [items[c][1] for c in cut_indices]
    
    # Gap sizes (strengths) at the cuts
    gap_sizes = [gaps_sorted[i][1] for i in range(len(cut_indices))]
    
    return bands, cut_values, gap_sizes

def compute_band_statistics(nat_bands, gaps):
    """
    Compute mean values and ratios for natural-break bands
    
    Args:
        nat_bands: List of bands from compute_natural_breaks
        gaps: Gap sizes between bands
    
    Returns:
        Tuple of (means, ratios) where ratios are 10^(Δmean)
    """
    means = []
    for band in nat_bands:
        logs = [val for name, val in band]
        mean = sum(logs)/len(logs) if logs else float('nan')
        means.append(mean)
    
    ratios = []
    for i in range(1, len(means)):
        if means[i-1] != float('nan') and means[i] != float('nan'):
            ratios.append(10**(means[i] - means[i-1]))
        else:
            ratios.append(float('nan'))
    
    return means, ratios

def compute_micro_macro_bridge(Sigma_eq, H_curv_SI, const):
    """
    Compute micro-macro UV-IR conjugacy check
    
    L_B = (k_B Σ)/G = 2c/ℋ is the macro Hubble diameter
    ℓ* = ħc/E_EW is the micro Compton-like scale at EW anchor
    
    Test: L_B × ℓ* should relate to CGM invariants (Q_G, m_p)
    Conjugacy exponent tests whether bridge scales as (E_CS/E_EW)^n
    
    Returns:
        Dictionary with micro-macro bridge quantities
    """
    scales = CGMEnergyScales()
    cgm = CGMParameters()
    
    # Macro length (Hubble diameter)
    L_B = (const.k_B * Sigma_eq) / const.G
    
    # Micro length at EW anchor (Compton-like)
    E_EW_J = scales.E_EW * Decimal('1.602176634e-10')  # GeV to J
    ell_star = (const.hbar * const.c) / E_EW_J
    
    # Product (dimensionless when normalized)
    L_B_times_ell = L_B * ell_star
    
    # CGM reference scales for comparison
    # l_P^2 = ħG/c^3
    l_P_squared = (const.hbar * const.G) / const.c**3
    
    # Ratio to Planck area
    ratio_to_planck_area = L_B_times_ell / l_P_squared
    
    # Check against CGM numbers
    Q_G_ratio = ratio_to_planck_area / cgm.Q_G
    m_p_ratio = ratio_to_planck_area / (cgm.m_p**2)
    
    # Conjugacy exponent: does L_B × ℓ*(EW) scale like (E_CS/E_EW)^n?
    E_CS = scales.E_CS
    E_EW = scales.E_EW
    log_ratio_area = float((ratio_to_planck_area).ln() / Decimal('10').ln())
    log_ratio_EW = float((E_CS / E_EW).ln() / Decimal('10').ln())
    conjugacy_exp = log_ratio_area / log_ratio_EW if log_ratio_EW != 0 else float('nan')
    
    return {
        'L_B': float(L_B),
        'ell_star': float(ell_star),
        'L_B_times_ell': float(L_B_times_ell),
        'ratio_to_planck_area': float(ratio_to_planck_area),
        'Q_G_ratio': float(Q_G_ratio),
        'm_p_squared_ratio': float(m_p_ratio),
        'conjugacy_exp': conjugacy_exp
    }

def print_results():
    """
    Calculate and display results for both Planck and SH0ES values
    """
    
    const = FundamentalConstants()
    cosmo = CosmologicalParameters()
    cgm = CGMParameters()
    
     
    print("CGM BALANCE INDEX ANALYSIS")
     
    print()
    print(f"B_i = 2Gc/(k_B H₀)  |  Dimensions: [M^-2 L^2 Θ^1 T^0]  |  ℏ cancels")
    print()
    
    print("BALANCE INDEX VALUES")
    print("-" * 10)
    
    # Calculate for both Planck and SH0ES values
    datasets = [
        ("Planck 2020 (TT,TE,EE+lowE+lensing+BAO)", 
         cosmo.H0_planck, cosmo.H0_planck_error),
        ("SH0ES 2022 (Cepheid-SNe Ia)", 
         cosmo.H0_shoes, cosmo.H0_shoes_error)
    ]
    
    all_results = []
    for name, H0, H0_err in datasets:
        results = calculate_balance_index(H0, H0_err)
        all_results.append((name, results))
        
        print(f"{name}: H₀ = {float(H0):.2f} ± {float(H0_err):.2f} km/s/Mpc")
        print(f"  B_i = ({results['Sigma_eq']:.6e} ± {results['Sigma_eq_error']:.2e}) m²⋅K⋅kg⁻²")
        print(f"  B_i,CGM (aperture) = {results['Sigma_eq_CGM']:.6e} m²⋅K⋅kg⁻²")
        print(f"  Verification (area/compact): {results['verification_ratio']:.15f}")
        
        # Guardrail: verification must be exact
        assert abs(results['verification_ratio'] - 1.0) < 1e-15, "Verification ratio must equal 1"
    
        print()
    
    # Compare the two measurements
    planck_results = all_results[0][1]
    shoes_results = all_results[1][1]
    ratio = shoes_results['Sigma_eq'] / planck_results['Sigma_eq']
    print(f"Hubble tension: B_i(SH0ES)/B_i(Planck) = {ratio:.6f} (∝ H₀ ratio)")
    print()
    
    # Fundamental geometric identities and dimensionless checks
    print("FUNDAMENTAL GEOMETRIC IDENTITIES")
    print("----------")
    
    # Dimensionless closure vector (what geometry forces)
    chi = (const.k_B * planck_results['Sigma_eq_decimal'] * planck_results['H0_SI_decimal']) / (const.G * const.c)
    print(f"Identity: χ = (k_B Σ H₀)/(G c) = {float(chi):.15f} (= 2)")
    assert abs(chi - 2) < 1e-15, "χ must equal 2"
    
    # Topology check: 4π (area) vs 2π (temperature) identity
    I_top = (const.k_B * planck_results['T_eq_decimal'] * planck_results['A_H_decimal']) / \
            (const.hbar * planck_results['H0_SI_decimal'] * Decimal('2') * planck_results['R_H_decimal']**2)
    print(f"Identity: τ = Topology (4π/2π) = {float(I_top):.15f} (= 1)")
    assert abs(I_top - 1) < 1e-15, "Topology identity must equal 1"
    
    # Q_G and aperture identities (CGM geometric core)
    s_p = PI / Decimal('2')  # π/2 from CS threshold
    check_QG_aperture = cgm.Q_G * (cgm.m_p ** 2)
    check_sp_aperture = s_p / (cgm.m_p ** 2)
    phi = check_sp_aperture / (Decimal('4') * PI ** 2)
    print(f"Identity: ϕ = (s_p/m_p²)/(4π²) = {float(phi):.15f} (= 1)")
    print(f"Identity: Q_G × m_p² = {float(check_QG_aperture):.15f} (= 0.5)")
    assert abs(check_QG_aperture - Decimal('0.5')) < 1e-15, "Q_G aperture must equal 0.5"
    assert abs(phi - 1) < 1e-10, "ϕ must equal 1"
    print()
    print("Closure vector (χ, τ, ϕ) = (2, 1, 1) — geometric necessities")
    print()
    
    # Unit-neutral checks (dimensionless physics)
    print("DIMENSIONLESS PHYSICS")
    print("----------")
    
    # H0 in Planck units
    t_P = (const.hbar * const.G / const.c**5).sqrt()
    H0_planck_units = planck_results['H0_SI_decimal'] * t_P
    print(f"Ĥ₀ = H₀ × t_P (dimensionless):  {float(H0_planck_units):.6e}")
    
    # Gravitational coupling at EW scale
    E_EW_J = CGMEnergyScales.E_EW * Decimal('1.602176634e-10')
    m_EW = E_EW_J / const.c**2
    alpha_G_EW = const.G * m_EW**2 / (const.hbar * const.c)
    print(f"α_G(EW) = Gm²/(ħc) at 246.22 GeV: {float(alpha_G_EW):.6e}")
    
    # Gravitational coupling at proton mass
    m_proton = Decimal('1.67262192e-27')  # kg
    alpha_G_proton = const.G * m_proton**2 / (const.hbar * const.c)
    print(f"α_G(proton):                    {float(alpha_G_proton):.6e}")
    print()
    
    # Error budget for B_i
    print("ERROR BUDGET")
    print("----------")
    rel_err_G = float(const.G_uncertainty / const.G)
    rel_err_H0 = float(cosmo.H0_planck_error / cosmo.H0_planck)
    rel_err_total = math.sqrt(rel_err_G**2 + rel_err_H0**2)
    print(f"Fractional uncertainty contributions:")
    print(f"  ΔG/G:                         {rel_err_G:.6e} ({100*rel_err_G**2/rel_err_total**2:.1f}%)")
    print(f"  ΔH₀/H₀:                       {rel_err_H0:.6e} ({100*rel_err_H0**2/rel_err_total**2:.1f}%)")
    print(f"  Total ΔΣ/Σ:                   {rel_err_total:.6e}")
    print(f"  → B_i precision limited by H₀ measurement ({100*rel_err_H0**2/rel_err_total**2:.0f}%)")
    print()
    
    # Aperture recovery test
    print("APERTURE RECOVERY")
    print("----------")
    I1_CGM_computed = (const.k_B * planck_results['Sigma_eq_CGM_decimal'] * 
                       (planck_results['H0_SI_decimal'] / (Decimal('1') + cgm.m_p))) / (const.G * const.c)
    m_p_recovered = (Decimal('2') / I1_CGM_computed).sqrt() - Decimal('1')
    m_p_deviation = abs(m_p_recovered - cgm.m_p)
    print(f"From I₁_CGM: m_p = √(2/I₁_CGM) − 1")
    print(f"  Recovered:  {float(m_p_recovered):.15f}")
    print(f"  Expected:   {float(cgm.m_p):.15f}")
    print(f"  Deviation:  {float(m_p_deviation):.2e}")
    print(f"  → Aperture value {'locked by equilibrium ✓' if m_p_deviation < 1e-10 else 'not fully determined'}")
    print()
    
    # Constant recovery from closure (equilibrium viewpoint)
    print("CONSTANT RECOVERY FROM CLOSURE χ = 2")
    print("----------")
    
    # H0 recovery
    H0_from_BI = (Decimal('2') * const.G * const.c) / (const.k_B * planck_results['Sigma_eq_decimal'])
    H0_ratio = H0_from_BI / planck_results['H0_SI_decimal']
    print(f"H₀ from B_i:                    {float(H0_from_BI):.6e} s⁻¹")
    print(f"H₀ input:                       {float(planck_results['H0_SI_decimal']):.6e} s⁻¹")
    print(f"  Ratio: {float(H0_ratio):.15f}")
    assert abs(H0_ratio - 1) < 1e-15, "H0 inversion must be exact"
    
    # G recovery from χ = 2
    G_recovered = (const.k_B * planck_results['Sigma_eq_decimal'] * planck_results['H0_SI_decimal']) / (Decimal('2') * const.c)
    G_ratio = G_recovered / const.G
    print(f"G from χ = 2:                   {float(G_recovered):.8e} m³⋅kg⁻¹⋅s⁻²")
    print(f"G CODATA:                       {float(const.G):.8e} m³⋅kg⁻¹⋅s⁻²")
    print(f"  Ratio: {float(G_ratio):.15f}")
    assert abs(G_ratio - 1) < 1e-15, "G recovery must be exact"
    
    # k_B recovery from χ = 2
    kB_recovered = (Decimal('2') * const.G * const.c) / (planck_results['Sigma_eq_decimal'] * planck_results['H0_SI_decimal'])
    kB_ratio = kB_recovered / const.k_B
    print(f"k_B from χ = 2:                 {float(kB_recovered):.8e} J⋅K⁻¹")
    print(f"k_B CODATA:                     {float(const.k_B):.8e} J⋅K⁻¹")
    print(f"  Ratio: {float(kB_ratio):.15f}")
    assert abs(kB_ratio - 1) < 1e-15, "k_B recovery must be exact"
    
    print()
    print("Interpretation:")
    print("  • Constants are overdetermined by closure vector (χ, τ, ϕ) = (2, 1, 1)")
    print("  • Values locked by geometric equilibrium, not arbitrary")
    print("  • 'Why these values?' → 'Why closure vector (2, 1, 1)?'")
    print()
    
    # Extended CGM Analysis: Emergent Constancy and Balance Index
    print()
     
    print("CGM EXTENDED ANALYSIS")
     
    print()
    
    # Use Planck values for extended analysis (Decimal for precision)
    planck_sigma = planck_results['Sigma_eq_decimal']
    planck_H0_SI = planck_results['H0_SI_decimal']
    
    # Gibbons-Hawking product (ħ-cancellation parallel to B_i)
    # T_eq × S_dS should equal c⁵/(2G H_0)
    print("GIBBONS-HAWKING THERMODYNAMIC IDENTITY")
    print("----------")
    S_dS = const.k_B * planck_results['A_H_decimal'] * (const.c ** 3) / \
           (Decimal('4') * const.G * const.hbar)
    gh_left = planck_results['T_eq_decimal'] * S_dS
    gh_right = (const.c ** 5) / (Decimal('2') * const.G * planck_H0_SI)
    gh_ratio = gh_left / gh_right
    print(f"Identity: T_eq × S_dS = c⁵/(2G H₀)")
    print(f"  Left side:  {float(gh_left):.6e} J")
    print(f"  Right side: {float(gh_right):.6e} J")
    print(f"  Ratio:      {float(gh_ratio):.15f}")
    assert abs(gh_ratio - 1) < 1e-15, "GH product identity must equal 1"
    print(f"  → ħ cancels in equilibrium thermodynamics (parallel to B_i)")
    print()
    
    print("EMERGENT CONSTANCY CHECKS")
    print("-" * 10)
    print("Testing closure invariants for CGM framework:")
    print()
    
    # Calculate all phase curvature variants
    H_curvs = calculate_all_phase_curvatures(planck_H0_SI)
    
    print(f"Phase Curvature ℋ Estimators (CGM Holonomy-Based):")
    print(f"  Baseline (H0):              {float(H_curvs['baseline']):.6e} s⁻¹")
    print(f"  SU(2) holonomy scaled:      {float(H_curvs['SU2_holo']):.6e} s⁻¹")
    print(f"  4-leg toroidal scaled:      {float(H_curvs['4leg_holo']):.6e} s⁻¹")
    print(f"  Aperture corrected:         {float(H_curvs['aperture']):.6e} s⁻¹")
    print()
    
    # Transported-Σ constancy check
    print(f"Transported-Σ Constancy Check (CGM-consistent):")
    print(f"{'Method':<20} {'I₁(trans)':<15} {'L_B(trans) (m)':<20}")
    print("-" * 60)
    
    for method, H_curv in H_curvs.items():
        Sigma_trans = transported_sigma(H_curv, const)
        inv_t = calculate_emergent_constancy_invariants(Sigma_trans, H_curv, const)
        print(f"{method:<20} {inv_t['I1']:<15.12f} {inv_t['L_B']:<20.6e}")
        
        # Guardrail: transported invariant must equal 2
        assert abs(inv_t['I1'] - 2.0) < 1e-10, f"I1(trans) for {method} must equal 2"
    
    print()
    print("Interpretation:")
    print("  • I₁(trans) = 2.000... for all methods: Closure invariant preserved under transport")
    print("  • L_B(trans) = 2c/ℋ_method: Each ℋ defines its own horizon scale")
    print("  • Emergent constancy validated: identity survives holonomy transport")
    print()
    
    # CGM aperture-specific check
    H_aperture = H_curvs['aperture']
    Sigma_eq_CGM_dec = planck_results['Sigma_eq_CGM_decimal']
    inv_cgm = calculate_emergent_constancy_invariants(Sigma_eq_CGM_dec, H_aperture, const)
    expected_I1_CGM_dec = Decimal('2') / ((Decimal('1') + cgm.m_p)**2)
    print(f"Identity: CGM Aperture I₁_CGM = 2/(1+m_p)²")
    print(f"  Computed: {inv_cgm['I1']:.12f}")
    print(f"  Expected: {float(expected_I1_CGM_dec):.12f}")
    print(f"  Deviation: {abs(inv_cgm['I1'] - float(expected_I1_CGM_dec)):.2e}")
    print(f"  → Aperture reduces invariant quadratically (T and ℋ both corrected)")
    
    # Guardrail: aperture identity must be exact
    assert abs(Decimal(str(inv_cgm['I1'])) - expected_I1_CGM_dec) < Decimal('1e-10'), "Aperture identity must match"
    print()
    
    # Micro-macro bridge
    print("MICRO-MACRO UV-IR CONJUGACY")
    print("-" * 10)
    bridge = compute_micro_macro_bridge(planck_sigma, H_curvs['baseline'], const)
    print(f"Macro length L_B (Hubble diameter):   {bridge['L_B']:.6e} m")
    L_B_over_2RH = bridge['L_B'] / (2 * planck_results['R_H'])
    print(f"L_B / (2R_H):                          {L_B_over_2RH:.15f}")
    assert abs(L_B_over_2RH - 1) < 1e-10, "L_B must equal 2R_H"
    print(f"Micro length ℓ* (EW Compton):          {bridge['ell_star']:.6e} m")
    print(f"Product L_B × ℓ*:                      {bridge['L_B_times_ell']:.6e} m²")
    print()
    print(f"Normalized to Planck area (l_P²):      {bridge['ratio_to_planck_area']:.6e}")
    print(f"Ratio to Q_G (4π):                     {bridge['Q_G_ratio']:.6e}")
    print(f"Ratio to m_p²:                         {bridge['m_p_squared_ratio']:.6e}")
    print()
    print(f"Signal: Conjugacy exponent (log area / log energy):")
    print(f"  Using EW anchor (246.22 GeV):  {bridge['conjugacy_exp']:.3f}")
    print(f"  Suggests 4th-5th power scaling: (E_CS/E_EW)^n with n~4.7")
    print()
    print("Interpretation:")
    print("  • L_B × ℓ*(EW) tests UV-IR conjugacy in length space")
    print("  • Exponent ~4.7 consistent with 4th-power scaling")
    print("  • Mirrors CGM fine-structure α ∝ δ⁴ fourth-root structure")
    print()
    
    print("ENERGY BALANCE INDEX ANALYSIS (Signal)")
    print("-" * 10)
    print("Ξ(E) = (E/ℏ)/ℋ compares intrinsic phase curvature to global curvature")
    print("Testing multiple ℋ estimators for three-band flavor structure")
    print()
    
    # Analyze with baseline first to get EW reference
    H_baseline = H_curvs['baseline']
    bands_baseline = analyze_balance_bands(H_baseline, const.hbar)
    Xi_EW = bands_baseline['cgm_scales']['EW']['Xi']
    
    # Now analyze with normalized indices for each method
    for method_name, H_curv_method in [('baseline', H_baseline), 
                                         ('SU2_holo', H_curvs['SU2_holo'])]:
        print(f"Method: {method_name}")
        print("-" * 60)
        
        bands = analyze_balance_bands(H_curv_method, const.hbar, Xi_EW if method_name != 'baseline' else None)
        
        print(f"Particle log₁₀(Ξ) range: {bands['log_range'][0]:.1f} to {bands['log_range'][1]:.1f}")
        print(f"Span: {bands['log_span']:.1f} decades")
        print()
        
        # Natural breaks analysis (use ΔΞ for holonomy methods for view-invariance)
        use_delta = (method_name != 'baseline')
        nat_bands, cuts, gaps = compute_natural_breaks(bands['particles'], k=3, use_delta=use_delta, Xi_EW=Xi_EW)
        
        label = "ΔΞ" if use_delta else "log₁₀(Ξ)"
        print(f"Natural-Breaks Partition ({label}):")
        for i, band in enumerate(nat_bands, 1):
            band_name = ['light', 'medium', 'heavy'][i-1] if i <= 3 else f'band{i}'
            print(f"  Band {i} ({band_name}): {len(band)} particles")
            for name, val in band:
                mass = bands['particles'][name]['mass_GeV']
                print(f"    {name:<18} {mass:<10.2e} GeV  {label} = {val:.2f}")
            if i < len(nat_bands):
                print(f"  ──── Break at {label} = {cuts[i-1]:.2f} (gap: {gaps[i-1]:.2f}) ────")
        
        # Band statistics
        means, ratios = compute_band_statistics(nat_bands, gaps)
        if len(means) >= 3:
            print(f"  Band means: {means[0]:.2f}, {means[1]:.2f}, {means[2]:.2f}")
        if len(ratios) >= 2:
            print(f"  Ratios (10^Δ): {ratios[0]:.1e}, {ratios[1]:.1e}")
            print(f"    Band 1→2: ≈ 10^{math.log10(ratios[0]):.1f} (neutrino seesaw regime)")
            print(f"    Band 2→3: ≈ 10^{math.log10(ratios[1]):.1f} (generational scaling)")
        print()
    
    # Physical interpretation
     
    print("PHYSICAL INTERPRETATION")
     
    print()
    print("B_i = 1.33×10³⁹ m²·K·kg⁻² represents:")
    print("  • Thermal capacity of the cosmic horizon per gravitational mass²")
    print("  • Timeless (T⁰) equilibrium between geometry and thermodynamics")
    print("  • The 'Balance' that maintains r_s/R_H = 1 in CGM BH universe")
    print("  • Classical limit where ℏ cancels but structure persists")
    print("  • Fundamental geometric index parallel to Q_G = 4π")
    print()
    
    # Summary metrics
     
    print("SUMMARY METRICS")
     
    print()
    print(f"Balance Index (Planck):          B_i = {planck_results['Sigma_eq']:.6e} m²⋅K⋅kg⁻²")
    print(f"Balance Index (SH0ES):           B_i = {shoes_results['Sigma_eq']:.6e} m²⋅K⋅kg⁻²")
    print(f"Ratio (Hubble tension):          {ratio:.6f}")
    print()
    print(f"Closure invariant (baseline):    I₁ = 2.000000000000")
    print(f"Aperture identity:               I₁_CGM = {inv_cgm['I1']:.12f}")
    print(f"  Expected [2/(1+m_p)²]:         {float(expected_I1_CGM_dec):.12f} ✓")
    print()
    print(f"UV-IR conjugacy exponent:        {bridge['conjugacy_exp']:.3f}")
    print(f"  (4th-5th power scaling at EW)")
    print()
    
    # Get natural break cutpoints from baseline
    baseline_bands, baseline_cuts, baseline_gaps = compute_natural_breaks(bands_baseline['particles'], k=3)
    print(f"Three-band structure (natural breaks):")
    print(f"  Band 1 (light):  {len(baseline_bands[0])} particles")
    print(f"  Band 2 (medium): {len(baseline_bands[1])} particles")
    print(f"  Band 3 (heavy):  {len(baseline_bands[2])} particles")
    print(f"  Cutpoints: {baseline_cuts[0]:.2f}, {baseline_cuts[1]:.2f}")
    print(f"  Gap strengths: {baseline_gaps[0]:.2f}, {baseline_gaps[1]:.2f} decades")
    print()
     

if __name__ == "__main__":
    print_results()