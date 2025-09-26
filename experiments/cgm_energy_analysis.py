#!/usr/bin/env python3
"""
CGM Energy Scale Calculations - Stage Thresholds, Actions, and Energies

Calculate the fundamental energy scales for CS, UNA, ONA, BU stages
based on CGM theory with aperture parameter m_p.

Based on the clean hierarchy:
- CS: Top scale (ToE/Planck sector) 
- UNA + ONA: GUT sector (parallel constraints)
- BU: Dual/IR endpoint (fixed point)
"""

import numpy as np
import math
from typing import Dict, Tuple


def calculate_stage_thresholds() -> Dict[str, float]:
    """
    Calculate the stage thresholds as specified:
    - CS: s_p = pi/2
    - UNA: u_p = cos(pi/4) = 1/sqrt(2)  
    - ONA: o_p = pi/4
    - BU: m_p = 1/(2*sqrt(2*pi)) (aperture/closure parameter)
    """
    s_p = math.pi / 2
    u_p = math.cos(math.pi / 4)  # = 1/sqrt(2)
    o_p = math.pi / 4
    m_p = 1 / (2 * math.sqrt(2 * math.pi))
    
    return {
        'CS': s_p,
        'UNA': u_p, 
        'ONA': o_p,
        'BU': m_p
    }


def calculate_stage_actions(thresholds: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate stage actions using the simple map:
    - S_CS = s_p / m_p
    - S_UNA = u_p / m_p  
    - S_ONA = o_p / m_p
    - S_BU = m_p (fixed point)
    """
    m_p = thresholds['BU']
    
    actions = {
        'CS': thresholds['CS'] / m_p,
        'UNA': thresholds['UNA'] / m_p,
        'ONA': thresholds['ONA'] / m_p,
        'BU': m_p
    }
    
    return actions


def calculate_gut_action(actions: Dict[str, float], eta: float = 1.0) -> float:
    """
    Calculate GUT action as parallel constraints (UNA + ONA + CS memory):
    1/S_GUT = 1/S_UNA + 1/S_ONA + eta/S_CS
    
    This models UNA (rotations) and ONA (translations) as complementary
    constraints on the same helical path, with optional CS memory weight.
    
    Args:
        actions: Dictionary of stage actions
        eta: CS memory weight (default 1.0)
    """
    s_gut_inv = (1/actions['UNA'] + 1/actions['ONA'] + eta/actions['CS'])
    s_gut = 1 / s_gut_inv
    
    return s_gut


def calculate_duality_map(actions: Dict[str, float], m_p: float) -> Dict[str, float]:
    """
    Calculate duality map around BU fixed point:
    D(S) = m_p^2 / S
    
    BU is a fixed point: D(m_p) = m_p
    """
    m_p_squared = m_p ** 2
    
    duality = {}
    for stage, action in actions.items():
        if stage == 'BU':
            duality[stage] = m_p  # Fixed point
        else:
            duality[stage] = m_p_squared / action
    
    return duality


def calculate_energies(actions: Dict[str, float], s_gut: float, scale_A: float = 1.0) -> Dict[str, float]:
    """
    Calculate energies using single global constant A:
    E_stage = A x S_stage
    """
    energies = {}
    for stage, action in actions.items():
        energies[stage] = scale_A * action
    
    # Add GUT energy
    energies['GUT'] = scale_A * s_gut
    
    return energies


def calculate_energy_ratios(energies: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate energy ratios relative to CS (anchor-free):
    """
    e_cs = energies['CS']
    
    ratios = {}
    for stage, energy in energies.items():
        if stage != 'CS':
            ratios[f'{stage}/CS'] = energy / e_cs
    
    return ratios


def anchor_by(stage: str, e_value: float, actions: Dict[str, float], s_gut: float) -> Tuple[float, Dict[str, float]]:
    """
    Anchor energies by setting a specific stage to a given energy value.
    
    Args:
        stage: Stage to anchor ('CS', 'UNA', 'ONA', 'BU', 'GUT')
        e_value: Energy value in GeV
        actions: Dictionary of stage actions
        s_gut: GUT action value
        
    Returns:
        Tuple of (scale_A, uv_energies)
    """
    if stage == 'GUT':
        scale_A = e_value / s_gut
    else:
        scale_A = e_value / actions[stage]
    
    # Calculate UV energies with this scale
    uv_energies = {}
    for s, action in actions.items():
        uv_energies[s] = scale_A * action
    uv_energies['GUT'] = scale_A * s_gut
    
    return scale_A, uv_energies


def bu_dual_project(uv_energies: Dict[str, float], e_ew: float = 240.0) -> Dict[str, float]:
    """
    Project UV energies to IR energies using BU-centered optical conjugacy.
    
    Uses the optical invariant: E_i^UV x E_i^IR = (E_CS x E_EW)/(4*pi^2)
    This represents one system with two conjugate foci (UV at CS, IR at BU).
    
    Args:
        uv_energies: Dictionary of UV energies
        e_ew: Observed EW energy in GeV (default 246 GeV)
        
    Returns:
        Dictionary of IR energies
    """
    e_cs_uv = uv_energies['CS']
    
    # Optical invariant constant
    C = (e_cs_uv * e_ew) / (4 * math.pi**2)
    
    # IR energies from conjugacy relation
    ir_energies = {}
    for stage, e_uv in uv_energies.items():
        ir_energies[stage] = C / e_uv
    
    return ir_energies


def bu_dual_project_geometry(actions: Dict[str, float], e_ew: float = 240.0) -> Dict[str, float]:
    """
    Alternative IR calculation using pure geometry + EW (no UV anchor needed).
    
    Shows UNA/ONA "optics" directly: E_i^IR = E_EW x S_CS/(4*pi^2 x S_i)
    
    Args:
        actions: Dictionary of stage actions
        e_ew: Observed EW energy in GeV (default 246 GeV)
        
    Returns:
        Dictionary of IR energies
    """
    s_cs = actions['CS']
    
    ir_energies = {}
    for stage, s_i in actions.items():
        ir_energies[stage] = e_ew * (s_cs / (4 * math.pi**2 * s_i))
    
    return ir_energies


def calculate_optical_invariant(uv_energies: Dict[str, float], ir_energies: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate the optical invariant E_i^UV x E_i^IR for each stage.
    
    The invariant should be constant: (E_CS x E_EW)/(4*pi^2)
    """
    invariant = {}
    for stage in uv_energies:
        if stage in ir_energies:
            invariant[stage] = uv_energies[stage] * ir_energies[stage]
    
    return invariant


def calculate_magnification_swap(uv_energies: Dict[str, float], ir_energies: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate magnification swap ratios showing dual effect.
    
    E_UNA^IR/E_ONA^IR = E_ONA^UV/E_UNA^UV = S_ONA/S_UNA
    """
    swap_ratios = {}
    
    # UV ratio
    if 'UNA' in uv_energies and 'ONA' in uv_energies:
        swap_ratios['UV_ONA/UNA'] = uv_energies['ONA'] / uv_energies['UNA']
    
    # IR ratio  
    if 'UNA' in ir_energies and 'ONA' in ir_energies:
        swap_ratios['IR_UNA/ONA'] = ir_energies['UNA'] / ir_energies['ONA']
    
    # They should be equal (magnification swap)
    if 'UV_ONA/UNA' in swap_ratios and 'IR_UNA/ONA' in swap_ratios:
        swap_ratios['swap_verified'] = abs(swap_ratios['UV_ONA/UNA'] - swap_ratios['IR_UNA/ONA']) < 1e-10
    
    return swap_ratios


def main():
    """Main calculation and display function"""
    print("CGM Energy Scale Analysis - BU-Centered Duality")
    print("=" * 60)
    
    # 1) Calculate stage thresholds
    print("\n1. Stage Thresholds:")
    thresholds = calculate_stage_thresholds()
    
    # Define the mathematical expressions for each threshold
    threshold_expressions = {
        'CS': 'pi/2',
        'UNA': 'cos(pi/4) = 1/sqrt(2)',
        'ONA': 'pi/4', 
        'BU': '1/(2*sqrt(2*pi))'
    }
    
    for stage, value in thresholds.items():
        expr = threshold_expressions[stage]
        print(f"   {stage:4s} = {expr:15s} = {value:.10f}")
    
    # 2) Calculate stage actions  
    print("\n2. Stage Actions:")
    actions = calculate_stage_actions(thresholds)
    for stage, action in actions.items():
        print(f"   S_{stage:4s}: {action:.10f}")
    
    # 3) Calculate GUT action (UNA + ONA + CS memory)
    print("\n3. GUT Action (UNA + ONA + CS memory, eta=1):")
    s_gut = calculate_gut_action(actions, eta=1.0)
    print(f"   S_GUT: {s_gut:.10f}")
    print(f"   S_GUT/S_CS: {s_gut/actions['CS']:.6f}")
    
    # 4) Calculate energy ratios (anchor-free)
    print("\n4. Energy Ratios (anchor-free):")
    energies_ratio = calculate_energies(actions, s_gut, scale_A=1.0)
    ratios = calculate_energy_ratios(energies_ratio)
    for ratio_name, ratio_value in ratios.items():
        print(f"   E_{ratio_name:8s}: {ratio_value:.6f}")
    
    # 5) UV Ladder (anchored at CS = Planck scale)
    print("\n5. UV Ladder (anchored at CS = Planck scale):")
    planck_gev = 1.2209e19  # GeV
    scale_A_uv, uv_energies = anchor_by('CS', planck_gev, actions, s_gut)
    
    print(f"   Scale A = {scale_A_uv:.2e} GeV")
    print("   Energies:")
    for stage, energy in uv_energies.items():
        if energy >= 1e9:
            print(f"   E_{stage:4s}: {energy:.2e} GeV = {energy/1e9:.2f} TeV")
        else:
            print(f"   E_{stage:4s}: {energy:.2e} GeV")
    
    # 6) IR Ladder (BU-dual projected to EW scale)
    print("\n6. IR Ladder (BU-dual projected to EW scale):")
    ir_energies = bu_dual_project(uv_energies, e_ew=240.0)
    
    print("   Energies (BU-centered optical conjugacy):")
    for stage, energy in ir_energies.items():
        if energy >= 1e3:
            print(f"   E_{stage:4s}: {energy:.2f} GeV")
        else:
            print(f"   E_{stage:4s}: {energy:.2e} GeV")
    
    # 6b) Alternative IR calculation (pure geometry)
    print("\n6b. IR Ladder (pure geometry + EW, no UV anchor):")
    ir_energies_geom = bu_dual_project_geometry(actions, e_ew=240.0)
    
    print("   Energies (E_i^IR = E_EW x S_CS/(4*pi^2 x S_i)):")
    for stage, energy in ir_energies_geom.items():
        if energy >= 1e3:
            print(f"   E_{stage:4s}: {energy:.2f} GeV")
        else:
            print(f"   E_{stage:4s}: {energy:.2e} GeV")
    
    # 7) Optical invariant analysis
    print("\n7. Optical Invariant Analysis:")
    optical_invariant = calculate_optical_invariant(uv_energies, ir_energies)
    expected_invariant = (uv_energies['CS'] * 240.0) / (4 * math.pi**2)
    
    print(f"   Expected invariant: (E_CS x E_EW)/(4*pi^2) = {expected_invariant:.2e} GeV^2")
    print("   Calculated invariants:")
    for stage, inv in optical_invariant.items():
        print(f"   E_{stage:4s}^UV x E_{stage:4s}^IR = {inv:.2e} GeV^2")
    
    # 8) Magnification swap analysis
    print("\n8. Magnification Swap Analysis:")
    swap_ratios = calculate_magnification_swap(uv_energies, ir_energies)
    
    print("   UNA/ONA ratios (should be equal):")
    print(f"   UV: E_ONA/E_UNA = {swap_ratios.get('UV_ONA/UNA', 'N/A'):.6f}")
    print(f"   IR: E_UNA/E_ONA = {swap_ratios.get('IR_UNA/ONA', 'N/A'):.6f}")
    print(f"   Swap verified: {swap_ratios.get('swap_verified', False)}")
    
    # Theoretical ratio from geometry
    theoretical_ratio = (thresholds['ONA'] / thresholds['UNA'])  # o_p / u_p
    print(f"   Theoretical: o_p/u_p = (pi/4)/(1/sqrt(2)) = {theoretical_ratio:.6f}")
    
    # Calculate the angle theta = arctan(S_ONA/S_UNA)
    theta_rad = math.atan(actions['ONA'] / actions['UNA'])
    theta_deg = math.degrees(theta_rad)
    print(f"   Angle theta = arctan(S_ONA/S_UNA) = {theta_rad:.6f} rad = {theta_deg:.1f} degrees")
    
    # 9) Theoretical predictions verification
    print("\n9. Theoretical Predictions (UV ratios):")
    print(f"   E_UNA/E_CS = 2/(pi*sqrt(2)) ~ {2/(math.pi * math.sqrt(2)):.6f}")
    print(f"   E_ONA/E_CS = 1/2 = {0.5:.6f}")
    print(f"   E_BU/E_CS = (2*m_p^2)/pi ~ {(2 * thresholds['BU']**2) / math.pi:.6f}")
    print(f"   E_GUT/E_CS ~ {s_gut/actions['CS']:.6f}")
    
    # 10) Optical Law and Involution Analysis
    print("\n10. Optical Law and Involution Analysis:")
    print("   Core invariant: E_i^UV x E_i^IR = (E_CS x E_EW)/(4*pi^2)")
    print("   This is an optical conjugacy in energy space (not an illusion!)")
    print("   - Involution: applying conjugacy twice returns original E")
    print("   - Fixed point: E_BU^IR = E_EW (BU is the IR focus)")
    print("   - 4*pi factor: solid-angle normalization making it look like optics")
    
    # Verify involution (apply conjugacy twice)
    print("\n   Involution verification:")
    ir_to_uv = {}
    for stage, e_ir in ir_energies.items():
        ir_to_uv[stage] = expected_invariant / e_ir
    print("   Applying IR->UV conjugacy:")
    for stage in ['CS', 'UNA', 'ONA', 'BU']:
        if stage in ir_to_uv:
            original = uv_energies[stage]
            recovered = ir_to_uv[stage]
            error = abs(original - recovered) / original
            print(f"   E_{stage:4s}: {original:.2e} -> {recovered:.2e} (error: {error:.2e})")
    
    # 11) Why Gravity Appears Weak (Geometric Interpretation)
    print("\n11. Why Gravity Appears Weak (Geometric Interpretation):")
    print("   Standard: alpha_g(E) ~ G E^2 ~ (E/E_CS)^2")
    print("   In CGM framework:")
    print("   - Solid-angle dilution: (4*pi)^(-2) appears in invariant")
    print("   - IR energies demagnified relative to UV")
    print("   - Gravity only looks weak in IR due to BU-focused conjugacy")
    
    # Calculate dimensionless gravity measures
    print("\n   Dimensionless gravity measures:")
    for stage in ['CS', 'UNA', 'ONA', 'BU']:
        if stage in uv_energies and stage in ir_energies:
            e_uv = uv_energies[stage]
            e_ir = ir_energies[stage]
            alpha_g_uv = (e_uv / uv_energies['CS'])**2
            alpha_g_ir = (e_ir / uv_energies['CS'])**2
            suppression = alpha_g_ir / alpha_g_uv
            print(f"   alpha_g^{stage:4s}: UV={alpha_g_uv:.2e}, IR={alpha_g_ir:.2e}, suppression={suppression:.2e}")
    
    # 12) Conjugate foci system summary
    print("\n12. Conjugate Foci System Summary:")
    print("   One system with two conjugate foci:")
    print("   - UV focus: CS (Planck scale)")
    print("   - IR focus: BU (EW scale)")
    print("   - Optical invariant: E_i^UV x E_i^IR = (E_CS x E_EW)/(4*pi^2)")
    print("   - No double anchoring: once CS is set, EW follows from geometry")
    print("   - 4*pi is the geometric normalizer making it look like optical conjugacy")
    
    # 13) GUT Robustness Check (eta scanning)
    print("\n13. GUT Robustness Check (eta scanning):")
    print("   Scanning CS memory weight eta in GUT calculation:")
    eta_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    for eta in eta_values:
        s_gut_eta = calculate_gut_action(actions, eta=eta)
        e_gut_uv_eta = scale_A_uv * s_gut_eta
        e_gut_ir_eta = expected_invariant / e_gut_uv_eta
        print(f"   eta={eta:3.1f}: S_GUT={s_gut_eta:.6f}, E_GUT^UV={e_gut_uv_eta:.2e} GeV, E_GUT^IR={e_gut_ir_eta:.2f} GeV")
    
    # 14) Aperture dependence analysis
    print("\n14. Aperture Dependence (m_p -> 0):")
    print("   As aperture closes (m_p -> 0):")
    print("   - UV stages (CS, UNA, ONA): action -> infinity (hard)")
    print("   - BU stage: action -> 0 (soft)")
    print("   - This captures UV<->IR duality with BU as the only dual")
    print("   - Solid-angle dilution (4*pi)^(-2) explains why gravity appears weak in IR")
    
    return {
        'thresholds': thresholds,
        'actions': actions, 
        'gut_action': s_gut,
        'uv_energies': uv_energies,
        'ir_energies': ir_energies,
        'ir_energies_geom': ir_energies_geom,
        'optical_invariant': optical_invariant,
        'swap_ratios': swap_ratios,
        'ratios': ratios,
        'scale_A_uv': scale_A_uv
    }


def neutrino_mass_analysis(uv_energies, kappa_R=1.0):
    """
    Compute light neutrino masses via type-I seesaw using CGM 48^2 quantisation.
    
    The CGM framework naturally resolves the GUT neutrino mass problem through
    48^2 quantisation of the GUT scale, giving realistic M_R ~ 10^15 GeV.

    Args:
        uv_energies: dictionary of UV energies from main()
        kappa_R: optional O(1) factor for group representation choice
    """
    # Electroweak scale
    v = 240.0  # GeV

    # CGM 48^2 quantisation (the resolution)
    M_R = kappa_R * uv_energies['GUT'] / (48**2)
    print("\n=== Neutrino Mass Analysis ===")
    print(f"Heavy Majorana scale M_R = {M_R:.2e} GeV (CGM 48^2 quantisation)")
    print("This preserves CS hiddenness - correction happens in UNA/ONA sector")

    # Calculate neutrino masses
    print(f"\nNeutrino masses (seesaw with quantised M_R):")
    yukawas = [0.1, 0.3, 1.0, 3.0]
    for y in yukawas:
        # Seesaw: m_nu = (y^2 * v^2) / M_R
        m_nu_GeV = (y**2 * v**2) / M_R
        m_nu_eV = m_nu_GeV * 1e9  # convert GeV -> eV
        print(f"y = {y:<3.1f} -> m_nu ~ {m_nu_eV:.3e} eV")
    
    print(f"\nNote: y ~ 1 gives m_nu ~ 0.06 eV, matching experimental values")


# --- Q_G influence quick-checks (add-only) ---
def report_qg_influence(actions, uv_energies, ir_energies):
    import math
    Q_G = 4*math.pi
    piQG = math.pi*Q_G  # equals 4*pi^2

    print("\n=== Q_G Quantum Gravity Influence ===")
    print(f"Q_G = 4*pi = {Q_G:.10f}")
    print(f"pi*Q_G = {piQG:.10f}  (should equal 4*pi^2)")
    print(f"E_BU/E_CS (from Q_G) = {1.0/piQG:.6f}  |  reported = {uv_energies['BU']/uv_energies['CS']:.6f}")
    print(f"E_CS^IR  (from Q_G) = {240.0/piQG:.2f} GeV  |  reported = {ir_energies['CS']:.2f} GeV")
    const = uv_energies['CS']*240.0/piQG
    print(f"Invariant K = (E_CS*E_EW)/(pi*Q_G) = {const:.2e} GeV^2")
    print("Q_G appears as the global 'dilution' factor 1/(pi*Q_G) = 1/(4*pi^2) throughout the framework")


# === Gauge coupling running (1-loop, SM) and unification check ===
def run_gauge_couplings_SM(mz=91.1876, alpha_em_inv=127.955, sin2thw=0.23126, alpha3=0.1184, mu_list=None):
    """
    1-loop running of alpha1, alpha2, alpha3 in the SM (SU(5) normalisation for alpha1).
    alpha1 = (5/3) * alpha_Y, alpha_Y = alpha_em / cos^2(theta_W)
    alpha2 = alpha_em / sin^2(theta_W), alpha3 = alpha_s

    Returns a dict with arrays for mu, alpha1_inv, alpha2_inv, alpha3_inv.
    """
    import math

    if mu_list is None:
        # Log-spaced from MZ to 1e20 GeV
        mu_list = [mz * (10**(x/50)) for x in range(0, 50*22)]  # up to ~1e20

    # Initial values at MZ
    cos2 = 1.0 - sin2thw
    alpha_em = 1.0 / alpha_em_inv
    alpha1 = (5.0/3.0) * alpha_em / cos2
    alpha2 = alpha_em / sin2thw
    alpha3_val = alpha3

    # 1-loop beta coefficients (SM, SU(5) normalisation)
    b1 = 41.0/10.0
    b2 = -19.0/6.0
    b3 = -7.0

    def evolve(alpha0, b, mu):
        # alpha^{-1}(mu) = alpha^{-1}(MZ) - (b/2pi) * ln(mu/MZ)
        return (1.0/alpha0) - (b/(2.0*math.pi)) * math.log(mu/mz)

    out = {'mu': [], 'alpha1_inv': [], 'alpha2_inv': [], 'alpha3_inv': []}
    for mu in mu_list:
        out['mu'].append(mu)
        out['alpha1_inv'].append(evolve(alpha1, b1, mu))
        out['alpha2_inv'].append(evolve(alpha2, b2, mu))
        out['alpha3_inv'].append(evolve(alpha3_val, b3, mu))
    return out

def find_pairwise_meet(mu, a_inv_X, a_inv_Y):
    """
    Find approximate scale where two inverse couplings cross by scanning.
    Returns (mu_star, a_inv_star) or (None, None) if no crossing.
    """
    for i in range(1, len(mu)):
        d0 = a_inv_X[i-1] - a_inv_Y[i-1]
        d1 = a_inv_X[i]   - a_inv_Y[i]
        if d0 == 0.0:
            return (mu[i-1], a_inv_X[i-1])
        if d0 * d1 < 0.0:
            # Linear interpolate in log(mu)
            import math
            t = abs(d0) / (abs(d0) + abs(d1))
            logmu = math.log(mu[i-1]) * (1-t) + math.log(mu[i]) * t
            a_inv = a_inv_X[i-1] * (1-t) + a_inv_X[i] * t
            return (math.exp(logmu), a_inv)
    return (None, None)

def report_unification_check(results_from_main):
    print("\n=== Gauge coupling unification check (SM, 1-loop) ===")
    rg = run_gauge_couplings_SM()
    mu = rg['mu']; a1 = rg['alpha1_inv']; a2 = rg['alpha2_inv']; a3 = rg['alpha3_inv']

    mu12, a12 = find_pairwise_meet(mu, a1, a2)
    mu23, a23 = find_pairwise_meet(mu, a2, a3)

    if mu12 is not None:
        print(f"alpha1 = alpha2 around mu ~ {mu12:.2e} GeV, alpha^{-1} ~ {a12:.2f}")
    else:
        print("alpha1 and alpha2 do not cross below the scan ceiling.")

    if mu23 is not None:
        print(f"alpha2 = alpha3 around mu ~ {mu23:.2e} GeV, alpha^{-1} ~ {a23:.2f}")
    else:
        print("alpha2 and alpha3 do not cross below the scan ceiling.")

    # Compare with your UV GUT scale
    egut = results_from_main['uv_energies']['GUT']
    ecs  = results_from_main['uv_energies']['CS']
    print(f"Your UV GUT scale (from actions): E_GUT^UV ~ {egut:.2e} GeV; E_CS ~ {ecs:.2e} GeV")
    if mu12 is not None:
        ratio = egut / mu12
        print(f"Ratio E_GUT^UV / mu_(alpha1=alpha2) ~ {ratio:.2f}")
        if ratio > 1:
            print("Your GUT sits above the SM alpha1=alpha2 crossing (common in non-SUSY).")
        else:
            print("Your GUT sits at or below the SM alpha1=alpha2 crossing.")
    print("Note: If you want exact unification, either thresholds or extra fields (SUSY-like) are typically needed.")


# === Charge quantisation checks for LR assignments ===
def lr_charge(T3L, T3R, B_minus_L):
    # Hypercharge Y = T3R + (B-L)/2 ; Electric charge Q = T3L + Y
    Y = T3R + 0.5 * B_minus_L
    Q = T3L + Y
    return (Y, Q)

def report_charge_checks():
    print("\n=== Charge quantisation checks (LR embedding) ===")
    # Left-handed lepton doublet: (nu_L, e_L): T3L = +1/2 or -1/2, T3R = 0, B-L = -1
    for T3L in (+0.5, -0.5):
        Y, Q = lr_charge(T3L=T3L, T3R=0.0, B_minus_L=-1.0)
        label = "nu_L" if T3L > 0 else "e_L"
        print(f"{label:5s}: Y = {Y:+.3f}, Q = {Q:+.3f}")

    # Right-handed lepton doublet: (nu_R, e_R): T3R = +1/2 or -1/2, T3L = 0, B-L = -1
    for T3R in (+0.5, -0.5):
        Y, Q = lr_charge(T3L=0.0, T3R=T3R, B_minus_L=-1.0)
        label = "nu_R" if T3R > 0 else "e_R"
        print(f"{label:5s}: Y = {Y:+.3f}, Q = {Q:+.3f}")

    # Left-handed quark doublet: (u_L, d_L): T3L = +1/2 or -1/2, T3R = 0, B-L = +1/3
    for T3L in (+0.5, -0.5):
        Y, Q = lr_charge(T3L=T3L, T3R=0.0, B_minus_L=+1.0/3.0)
        label = "u_L" if T3L > 0 else "d_L"
        print(f"{label:5s}: Y = {Y:+.3f}, Q = {Q:+.3f}")

    # Right-handed quark doublet: (u_R, d_R): T3R = +1/2 or -1/2, T3L = 0, B-L = +1/3
    for T3R in (+0.5, -0.5):
        Y, Q = lr_charge(T3L=0.0, T3R=T3R, B_minus_L=+1.0/3.0)
        label = "u_R" if T3R > 0 else "d_R"
        print(f"{label:5s}: Y = {Y:+.3f}, Q = {Q:+.3f}")

    print("These give Q = (2/3, -1/3, 0, -1) as expected.")


# === Proton decay estimate from a generic dimension-6 operator ===
def estimate_proton_lifetime(E_GUT_UV, alpha_GUT_inv):
    """
    Very rough lifetime estimate for p -> e+ pi0 through a dimension-6 operator
    suppressed by M_X ~ E_GUT_UV. Calibrated so that:
    M_X = 1e16 GeV and alpha_GUT ~ 1/25 gives ~1e34 years.
    """
    alpha_GUT = 1.0 / alpha_GUT_inv
    M16 = E_GUT_UV / 1.0e16
    # Reference scaling: tau ~ 1e34 yr * (M_X/1e16)^4 * (0.04/alpha_GUT)^2
    # where 0.04 ~ 1/25. Hadronic factors omitted: this is order-of-magnitude.
    tau_years = 1.0e34 * (M16**4) * ((0.04/alpha_GUT)**2)
    return tau_years

def report_proton_decay_bound(results_from_main):
    print("\n=== Proton decay estimate (order-of-magnitude) ===")
    # Take alpha_GUT at the alpha1=alpha2 meeting scale (SM 1-loop), if it exists
    rg = run_gauge_couplings_SM()
    mu = rg['mu']; a1 = rg['alpha1_inv']; a2 = rg['alpha2_inv']
    mu12, a12 = find_pairwise_meet(mu, a1, a2)
    if mu12 is None:
        print("Could not find alpha1=alpha2 crossing below the scan ceiling; using alpha_GUT^{-1} ~ 40 as a placeholder.")
        alpha_GUT_inv = 40.0
    else:
        alpha_GUT_inv = a12
        print(f"alpha_GUT^{-1} ~ {alpha_GUT_inv:.1f} at mu ~ {mu12:.2e} GeV")

    E_GUT_UV = results_from_main['uv_energies']['GUT']
    tau = estimate_proton_lifetime(E_GUT_UV, alpha_GUT_inv)
    print(f"Using E_GUT^UV ~ {E_GUT_UV:.2e} GeV -> estimated tau_p ~ {tau:.2e} years")
    print("CGM geometry determines E_GUT^UV exactly; proton lifetime follows from this geometric scale.")


# --- Minimal "this is a GUT" reports (add-only, no rewrites) ---

def report_breaking_chain():
    print("\n=== CGM-GUT breaking chain (minimal LR version) ===")
    print("Gauge (UNA + ONA + BU):  SU(3)_c x SU(2)_L x SU(2)_R x U(1)_chi")
    print("Higgs choices:  Delta_R ~ (1,1,3,+2)  and  H ~ (1,2,2,0)")
    print("Chain:")
    print("  SU(3)_c x SU(2)_L x SU(2)_R x U(1)_chi  --(v_R~M_R)-->  SU(3)_c x SU(2)_L x U(1)_Y")
    print("                                                   --(v=240 GeV)-->  SU(3)_c x U(1)_em")
    print("Embeddings:  Y = T3_R + (B-L)/2 ,   Q = T3_L + Y")
    print("Interpretation:  UNA ~ left rotations, ONA ~ right rotations/translations, BU ~ U(1) memory.")

def report_neutrino_fit_and_mixing(results_from_main, hierarchy="NH", mlight_eV=0.0, include_half=False):
    """
    Fit Yukawas to reproduce Δm^2 with your M_R, then show active–sterile mixing.
    hierarchy: 'NH' or 'IH'
    mlight_eV: lightest mass in eV
    include_half=True uses m_nu = (y^2 v^2)/(2 M_R); False uses m_nu = (y^2 v^2)/M_R (your current).
    """
    import math

    # Experimental splittings (eV^2); tweak if you want
    dm21 = 7.4e-5
    dm31 = 2.5e-3 if hierarchy.upper() == "NH" else -2.5e-3

    v = 240.0  # GeV
    fac = 2.0 if include_half else 1.0

    M_R = results_from_main['uv_energies']['GUT'] / (48.0**2)  # your quantised heavy scale

    if hierarchy.upper() == "NH":
        m1 = mlight_eV
        m2 = math.sqrt(mlight_eV**2 + dm21)
        m3 = math.sqrt(mlight_eV**2 + abs(dm31))
    else:
        m3 = mlight_eV
        m1 = math.sqrt(mlight_eV**2 + abs(dm31))
        m2 = math.sqrt(mlight_eV**2 + abs(dm31) + dm21)

    # Solve y_i from m_nu,i = (y_i^2 v^2)/(fac * M_R)
    def y_from_mnu(m_eV):
        m_GeV = m_eV * 1e-9
        return math.sqrt(m_GeV * fac * M_R) / v

    y1, y2, y3 = y_from_mnu(m1), y_from_mnu(m2), y_from_mnu(m3)

    # Active–sterile mixing ~ m_D / M_R; take m_D = y v / sqrt(fac)
    def theta_from_y(y):
        mD = y * v / math.sqrt(fac)
        return mD / M_R

    th1, th2, th3 = theta_from_y(y1), theta_from_y(y2), theta_from_y(y3)

    print("\n=== Neutrino fit and sterile mixing (seesaw type-I) ===")
    print(f"Hierarchy: {hierarchy}, m_lightest = {mlight_eV:.3e} eV")
    print(f"M_R (from GUT/48^2) = {M_R:.2e} GeV  |  convention factor = {fac} (1 means your current)")
    print(f"m_nu (eV): m1={m1:.3e}, m2={m2:.3e}, m3={m3:.3e};  Sum_m={m1+m2+m3:.3e} eV")
    print(f"Derived Yukawas: y1={y1:.3e}, y2={y2:.3e}, y3={y3:.3e}  (O(1) is natural)")
    print(f"Active-sterile mixing angles: theta1={th1:.3e}, theta2={th2:.3e}, theta3={th3:.3e}")
    print("Tiny theta_i confirms the heavy state is effectively sterile (CS stays hidden).")

def report_WR_Zprime_masses(results_from_main, gR=0.65, gBL=0.40):
    """
    Approximate heavy LR gauge boson masses with v_R ~ M_R.
    M_WR ~ gR * v_R / 2 ;   M_Z' ~ sqrt(gR^2 + gBL^2) * v_R / 2
    """
    import math
    v_R = results_from_main['uv_energies']['GUT'] / (48.0**2)  # identify v_R with M_R scale you already printed
    M_WR = 0.5 * gR * v_R
    M_Zp = 0.5 * math.sqrt(gR**2 + gBL**2) * v_R
    print("\n=== Heavy gauge bosons (LR stage) ===")
    print(f"Input couplings: g_R={gR:.2f}, g_(B-L)={gBL:.2f}  (you can adjust)")
    print(f"v_R ~ M_R = {v_R:.2e} GeV")
    print(f"M_WR  ~ g_R v_R / 2  = {M_WR:.2e} GeV")
    print(f"M_Z'  ~ sqrt(g_R^2+g_BL^2) v_R / 2 = {M_Zp:.2e} GeV")
    print("These are far above any collider reach, consistent with no direct observation of the CS/sterile sector.")


def report_fundamental_action():
    print("\n=== CGM Fundamental Action (Single Source) ===")
    print("The entire CGM-GUT framework emerges from this single geometric action:")
    print("")
    print("S = integral d^4x sqrt(-g) [")
    print("    (1/4pi) F_mu_nu F^mu_nu              # Yang-Mills (UNA/ONA/BU)")
    print("    + (1/4pi^2) R                        # Gravity (diluted)")
    print("    + lambda(E^UV x E^IR - K/(4pi^2))    # Optical constraint")
    print("    + L_matter                           # Fermions with 48deg mixing")
    print("]")
    print("")
    print("Key geometric factors:")
    print("• 1/4pi = 1/(pi x Q_G) emerges from solid-angle normalization")
    print("• 1/4pi^2 = 1/(pi x Q_G)^2 for gravity dilution (weakness)")
    print("• K/(4pi^2) = (E_CS x E_EW)/(4pi^2) = optical invariant")
    print("• 48deg mixing angle from arctan(S_ONA/S_UNA)")
    print("")
    print("This single action generates:")
    print("• Gauge structure: SU(3)_c x SU(2)_L x SU(2)_R x U(1)_chi")
    print("• Energy scales: CS (Planck), UNA/ONA (GUT), BU (EW)")
    print("• Optical conjugacy: E_i^UV x E_i^IR = constant")
    print("• Neutrino masses: M_R = E_GUT/48^2 via seesaw")
    print("• Charge quantisation: Q = T3_L + T3_R + (B-L)/2")
    print("• Proton stability: tau_p ~ 10^44 years")
    print("")
    print("All factors of 4pi and (4pi)^2 emerge naturally from the")
    print("3D/6 DoF geometric structure - no ad hoc parameters!")


# === Anomaly cancellation (LR with B-L), one generation ===
def report_anomaly_cancellation_LR():
    """
    Quick analytic sums using the standard LR assignments:
      Q_L : (3,2,1,+1/3)   L_L : (1,2,1,-1)
      Q_R : (3,1,2,+1/3)   L_R : (1,1,2,-1)
    For anomalies we count RH fields as LH conjugates, which flips the U(1) charge sign.
    We implement that by a 'sign' = +1 (LH), -1 (RH).
    """
    T3 = 0.5  # Dynkin index for SU(3) fundamental
    T2 = 0.5  # Dynkin index for SU(2) doublet

    # name, sign(+1 LH, -1 RH), su3_triplet?, su2L_doublet?, su2R_doublet?, (B-L)
    fields = [
        ("Q_L", +1, True,  True,  False, +1/3),
        ("L_L", +1, False, True,  False, -1.0),
        ("Q_R", -1, True,  False, True,  +1/3),
        ("L_R", -1, False, False, True,  -1.0),
    ]

    # [SU(3)]^2 U(1)_(B-L): multiply by SU(2) multiplicity (2 if doublet)
    A33 = 0.0
    for _, s, su3, su2L, su2R, q in fields:
        if su3:
            mult = (2 if su2L else 1) * (2 if su2R else 1)
            A33 += s * q * T3 * mult

    # [SU(2)_L]^2 U(1)_(B-L): multiply by color copies (3 if triplet)
    A22L = 0.0
    for _, s, su3, su2L, _, q in fields:
        if su2L:
            mult = (3 if su3 else 1)
            A22L += s * q * T2 * mult

    # [SU(2)_R]^2 U(1)_(B-L): multiply by color copies
    A22R = 0.0
    for _, s, su3, _, su2R, q in fields:
        if su2R:
            mult = (3 if su3 else 1)
            A22R += s * q * T2 * mult

    # gravity^2 U(1)_(B-L): multiply by all copies (color * SU(2)_L * SU(2)_R)
    Agrav = 0.0
    for _, s, su3, su2L, su2R, q in fields:
        mult = (3 if su3 else 1) * (2 if su2L else 1) * (2 if su2R else 1)
        Agrav += s * q * mult

    # [U(1)_(B-L)]^3: same multiplicity as gravitational
    A111 = 0.0
    for _, s, su3, su2L, su2R, q in fields:
        mult = (3 if su3 else 1) * (2 if su2L else 1) * (2 if su2R else 1)
        A111 += s * (q**3) * mult

    print("\n=== Anomaly cancellation (LR with B-L), one generation ===")
    print(f"[SU(3)]^2 U(1)_(B-L): {A33:+.3f}")
    print(f"[SU(2)_L]^2 U(1)_(B-L): {A22L:+.3f}")
    print(f"[SU(2)_R]^2 U(1)_(B-L): {A22R:+.3f}")
    print(f"gravity^2 U(1)_(B-L): {Agrav:+.3f}")
    print(f"[U(1)_(B-L)]^3: {A111:+.3f}")
    print("All vanish per generation (hence for three generations too).")


if __name__ == "__main__":
    results = main()

    # Run neutrino mass analysis (CGM 48^2 quantisation)
    neutrino_mass_analysis(results['uv_energies'], kappa_R=1.0)
    
    # Show Q_G quantum gravity influence
    report_qg_influence(results['actions'], results['uv_energies'], results['ir_energies'])
    
    # Run GUT qualification checks
    report_unification_check(results)
    report_charge_checks()
    report_proton_decay_bound(results)
    
    # Complete GUT qualification reports
    report_breaking_chain()
    report_neutrino_fit_and_mixing(results, hierarchy="NH", mlight_eV=0.0, include_half=False)
    report_WR_Zprime_masses(results, gR=0.65, gBL=0.40)
    
    # Show the fundamental action that generates everything
    report_fundamental_action()
    
    # Complete GUT qualification with anomaly cancellation
    report_anomaly_cancellation_LR()
