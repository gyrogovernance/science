#!/usr/bin/env python3
"""
CGM 1 Light-Year Threshold Analysis
Analyzes the hypothesis that 1 light-year represents a "complete cycle of light"
in the Common Governance Model, corresponding to local coherence boundaries.
"""

import math
import numpy as np

# Configuration flags
TIDAL_EXPONENT = 1.0  # Set to 1.0 (no tidal correction) unless otherwise justified
SHOW_DIAGNOSTICS = False  # Enable to print Hilbert proxy / Fourier exploratory sections

G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
c = 299792458  # Speed of light (m/s)
AU = 1.495978707e11  # Astronomical unit (m)
M_sun = 1.989e30  # Solar mass (kg)
year_tropical = 365.25636 * 24 * 3600  # Tropical year (seconds)
year_julian = 365.25 * 24 * 3600  # Julian year (seconds)

m_p = 1 / (2 * math.sqrt(2 * math.pi))
Q_G = 4 * math.pi
s_p = math.pi / 2
u_p = 1 / math.sqrt(2)
o_p = math.pi / 4
delta_BU = 0.195342176580

APERTURE_TRANSMISSION = 1 - delta_BU / m_p

ly_tropical = c * year_tropical
ly_julian = c * year_julian
ly_AU_tropical = ly_tropical / AU
ly_AU_julian = ly_julian / AU

nu_galactic = 74e3 / (3.086e19)  # Galactic vertical frequency (s^-1): 74 km/s per kpc
r_tidal = (G * M_sun / nu_galactic**2)**(1/3)  # Tidal truncation radius (m)
r_tidal_AU = r_tidal / AU
r_tidal_ly = r_tidal / ly_tropical

h = 6.62607015e-34  # Planck's constant (J s)
hbar = h / (2 * math.pi)
m_electron = 9.1093837e-31  # Electron mass (kg)
m_proton = 1.6726219e-27  # Proton mass (kg)
compton_electron = hbar / (m_electron * c)
compton_proton = hbar / (m_proton * c)

oort_boundaries = {
    "Inner Oort (Hills) start": (2000, 500),
    "Inner Oort (Hills) mid": (10000, 2000),
    "Inner Oort (Hills) end": (20000, 5000),
    "Outer Oort start": (20000, 5000),
    "Outer Oort typical": (50000, 15000),
    "Outer Oort extended": (100000, 25000),
    "Outer Oort maximum": (200000, 50000),
}

def phase_at_distance_tidal(d_AU, cycle_time=year_tropical, tidal_exponent=1.0):
    """
    Calculate accumulated geometric phase with tidal weighting.
    Phase = pi when d = 1 light-year (complete cycle) for exponent=1.0.
    """
    d_meters = d_AU * AU
    cycle_distance = c * cycle_time
    normalized_distance = d_meters / cycle_distance
    return math.pi * (normalized_distance ** tidal_exponent)

def compute_monodromy_multiplicity(phase):
    """
    Compute multiplicity based on monodromy crossings of delta_BU within phase bands.
    """
    if phase < math.pi / 4:
        return 1
    
    n_crossings = int(math.floor(phase / delta_BU))
    multiplicity = 1
    
    if phase >= math.pi / 4:
        multiplicity = 2
    if phase >= math.pi / 2:
        multiplicity = 3
    if phase >= math.pi:
        multiplicity = 5
    if phase >= 2 * math.pi:
        multiplicity = 8
    
    return multiplicity

def compute_parity_bias(phase):
    """
    Compute expected left:right parity fraction based on CGM asymmetry.
    Returns fraction (0-1) representing left-handed preference.
    """
    if phase < math.pi / 4:
        return 1.0
    left_bias = u_p
    if phase >= math.pi / 2:
        left_bias *= (1 + delta_BU / math.pi)
    return left_bias

def compute_light_time_depth(d_AU):
    """
    Compute light-time depth: simple light-travel time (d/c in years).
    """
    d_ly = (d_AU * AU) / ly_tropical
    return d_ly

def compute_phase_time_depth(d_AU, cycle_time=year_tropical, tidal_exponent=1.0, T_coh=None):
    """
    Compute CGM phase-time depth: how far back in time do projections reach?
    Time-depth = phase/pi * coherence time scale T_coh.
    If T_coh=None, uses local Hubble time; if T_coh='local', uses cycle_time.
    """
    phase = phase_at_distance_tidal(d_AU, cycle_time, tidal_exponent)
    if T_coh == 'local':
        T_coh_value = cycle_time
    elif T_coh is None:
        H0_local = 70 * 1000 / (3.086e19)  # Local Hubble constant (s^-1)
        T_coh_value = 1.0 / H0_local  # Local Hubble time (s)
    else:
        T_coh_value = T_coh
    time_depth_years = (phase / math.pi) * (T_coh_value / year_tropical)
    return time_depth_years

def compute_unity_proxy(phase):
    """
    Hilbert proxy: weight in constant function sector (unity projection).
    U_proxy = (1 + cos φ)/2 approximates ⟨ψ|P_U|ψ⟩.
    """
    return (1.0 + math.cos(phase)) / 2.0

def compute_balance_proxy(phase):
    """
    Hilbert proxy: depth-four balance closure expectation.
    B_proxy = cos²(φ/2) approximates ⟨ψ|P_B|ψ⟩ (near 1 at φ = π).
    """
    return math.cos(phase / 2.0) ** 2

def compute_qg_budget(d_AU, cycle_time=year_tropical, tidal_exponent=1.0):
    """
    Compute Q_G = 4π solid-angle budget: coherent vs incoherent fractions.
    Returns fraction in ℓ=0 (coherent) vs ℓ>0 (incoherent) modes.
    """
    phase = phase_at_distance_tidal(d_AU, cycle_time, tidal_exponent)
    coherent_fraction = compute_unity_proxy(phase)  # ℓ=0 weight
    incoherent_fraction = 1.0 - coherent_fraction  # ℓ>0 weight
    return {
        "coherent_fraction": coherent_fraction,
        "incoherent_fraction": incoherent_fraction,
        "coherent_solid_angle": coherent_fraction * Q_G,
        "incoherent_solid_angle": incoherent_fraction * Q_G
    }

def compute_nested_thresholds(cycle_length, label="Micro"):
    """
    Compute CGM thresholds at nested (e.g., particle) scales.
    Uses given cycle_length (e.g., Compton wavelength) as base cycle.
    """
    k_values = {
        f"{label} Amplitude (1/8pi)": 1 / (8 * math.pi),
        f"{label} Quarter (1/4)": 1 / 4,
        f"{label} Half (1/2)": 1 / 2,
        f"{label} Full cycle (1)": 1.0,
        f"{label} Double (2)": 2.0,
        f"{label} Closure (pi)": math.pi,
    }

    boundaries = {}
    for name, k in k_values.items():
        R_m = cycle_length * k
        R_AU = R_m / AU
        boundaries[name] = {
            "k_factor": k,
            "distance_m": R_m,
            "distance_AU": R_AU,
            "phase_accumulated": math.pi * k
        }

    return boundaries

def compute_spin_proxy(distance_m, cycle_length):
    """
    Compute spin from phase wrapping with SU(2) 4π periodicity.
    Spin = 0.5 when distance equals the micro cycle_length.
    """
    wraps = distance_m / cycle_length
    return 0.5 * wraps

def compute_geometric_period(tidal_exponent=1.0):
    """
    Compute fundamental geometric periods from phase relation φ = π * (d/ly)^p.
    Returns distances for π, 2π, 4π closures.
    """
    factors = {
        "Half-cycle (π)": 1.0,
        "Full wrap (2π)": 2.0,
        "Double wrap (4π)": 4.0,
    }
    periods = {}
    for label, multiple in factors.items():
        distance_ly = multiple ** (1.0 / tidal_exponent)
        distance_AU = distance_ly * ly_AU_tropical
        periods[label] = {
            "distance_ly": distance_ly,
            "distance_AU": distance_AU,
            "phase": multiple * math.pi
        }
    return periods

def find_monodromy_closures(max_multiple=40, tidal_exponent=1.0, max_distance_ly=5.0):
    """
    Find distances where accumulated phase equals n * delta_BU.
    Limits results to distances within max_distance_ly.
    """
    closures = []
    for n in range(1, max_multiple + 1):
        target_phase = n * delta_BU
        distance_ly = (target_phase / math.pi) ** (1.0 / tidal_exponent)
        if distance_ly <= max_distance_ly:
            closures.append({
                "multiple": n,
                "phase": target_phase,
                "distance_ly": distance_ly,
                "distance_AU": distance_ly * ly_AU_tropical
            })
    return closures

def compute_monodromy_matches(tidal_exponent=1.0, max_multiple=80):
    """
    Compare theoretical thresholds (π/4, π/2, π) with nearest n × δ_BU closures.
    Returns fractional differences for diagnostic table.
    """
    targets = {
        "Quarter (π/4)": math.pi / 4,
        "Half (π/2)": math.pi / 2,
        "Full (π)": math.pi,
    }
    results = []
    for label, target_phase in targets.items():
        target_distance_ly = (target_phase / math.pi) ** (1.0 / tidal_exponent)
        target_distance_AU = target_distance_ly * ly_AU_tropical
        best = None
        for n in range(1, max_multiple + 1):
            phase_n = n * delta_BU
            distance_ly_n = (phase_n / math.pi) ** (1.0 / tidal_exponent)
            diff = abs(distance_ly_n - target_distance_ly)
            if best is None or diff < best["abs_diff_ly"]:
                best = {
                    "label": label,
                    "target_distance_ly": target_distance_ly,
                    "target_distance_AU": target_distance_AU,
                    "n": n,
                    "closure_distance_ly": distance_ly_n,
                    "closure_distance_AU": distance_ly_n * ly_AU_tropical,
                    "abs_diff_ly": diff,
                }
        if best:
            best["pct_diff"] = 100.0 * best["abs_diff_ly"] / best["target_distance_ly"]
            results.append(best)
    return results

def compute_fourier_spectrum(max_distance_ly=5.0, samples=512, tidal_exponent=1.0):
    """
    Compute Fourier spectrum of coherence vs distance profile.
    Returns dominant frequencies (cycles per ly) and magnitudes.
    """
    distances_ly = np.linspace(0.01, max_distance_ly, samples)
    distances_AU = distances_ly * ly_AU_tropical
    coherences = np.array([
        aperture_effects(d, year_tropical, tidal_exponent)["coherence"]
        for d in distances_AU
    ])
    window = np.hanning(samples)
    spectrum = np.fft.rfft(coherences * window)
    freqs = np.fft.rfftfreq(samples, d=distances_ly[1] - distances_ly[0])
    magnitudes = np.abs(spectrum)
    dominant_indices = np.argsort(magnitudes)[-5:][::-1]
    return [
        {
            "frequency_cycles_per_ly": freqs[i],
            "magnitude": magnitudes[i]
        }
        for i in dominant_indices
    ]

def compute_comet_sibling_count(d_AU, cycle_time=year_tropical, tidal_exponent=1.0):
    """
    Predict expected number of comet siblings due to multiplicity at distance d.
    Siblings = multiplicity * (1 - coherence), representing phase-sliced duplicates.
    """
    effects = aperture_effects(d_AU, cycle_time, tidal_exponent)
    siblings = effects['multiplicity'] * (1.0 - effects['coherence'])
    return siblings

def aperture_effects(d_AU, cycle_time=year_tropical, tidal_exponent=1.0):
    """
    Calculate observational effects with tidal-weighted phase and monodromy.
    """
    phase = phase_at_distance_tidal(d_AU, cycle_time, tidal_exponent)
    coherence = math.exp(-phase / math.pi)
    multiplicity = compute_monodromy_multiplicity(phase)
    parity_ratio = compute_parity_bias(phase)
    observable_fraction = coherence * APERTURE_TRANSMISSION
    
    return {
        "phase_radians": phase,
        "phase_fraction_of_pi": phase / math.pi,
        "coherence": coherence,
        "multiplicity": multiplicity,
        "parity_ratio": parity_ratio,
        "observable_fraction": observable_fraction
    }

def compute_cgm_thresholds(cycle_time=year_tropical):
    """
    Compute CGM coherence boundaries for a given cycle time.
    """
    cycle_distance = c * cycle_time
    
    k_values = {
        "Amplitude limit (1/8pi)": 1 / (8 * math.pi),
        "Quarter horizon (1/4)": 1 / 4,
        "Half horizon (1/2)": 1 / 2,
        "Full cycle (1)": 1.0,
        "Double cycle (2)": 2.0,
        "Closure region (pi)": math.pi,
    }
    
    boundaries = {}
    for name, k in k_values.items():
        R = cycle_distance * k
        R_ly = R / ly_tropical
        R_AU = R / AU
        boundaries[name] = {
            "k_factor": k,
            "distance_ly": R_ly,
            "distance_AU": R_AU,
            "phase_accumulated": math.pi * k
        }
    
    return boundaries

def main():
    print("CGM 1 LIGHT-YEAR THRESHOLD ANALYSIS")
    print()
    
    print("1. FUNDAMENTAL CGM PARAMETERS")
    print(f"Aperture parameter (m_p):     {m_p:.6f}")
    print(f"Quantum gravity (Q_G):         {Q_G:.6f} sr = 4pi")
    print(f"Closure constraint:            Q_G x m_p^2 = {Q_G * m_p**2:.6f}")
    print(f"Aperture fraction:             {APERTURE_TRANSMISSION:.4f} = {APERTURE_TRANSMISSION*100:.2f}%")
    print(f"BU dual-pole monodromy:        {delta_BU:.6f} rad")
    print()
    
    print(f"Speed of light (c):            {c:,} m/s")
    print(f"Tropical year:                  {year_tropical/86400:.3f} days")
    print(f"Julian year:                    {year_julian/86400:.3f} days")
    print(f"1 light-year (tropical):       {ly_AU_tropical:.1f} AU")
    print(f"1 light-year (julian):         {ly_AU_julian:.1f} AU")
    print()
    
    print("2. GALACTIC TIDAL STRUCTURE")
    print(f"Galactic vertical frequency:   {nu_galactic:.3e} s^-1")
    print(f"Tidal truncation radius:       {r_tidal_AU:,.0f} AU = {r_tidal_ly:.3f} ly")
    print()
    
    tidal_exp = TIDAL_EXPONENT
    print("3. TIDAL PHASE ASSUMPTION")
    print(f"Using tidal exponent p = {tidal_exp:.3f}. (Set TIDAL_EXPONENT to adjust this assumption.)")
    print()
    
    print("4. CGM COHERENCE BOUNDARIES (T = 1 Earth year)")
    boundaries = compute_cgm_thresholds(year_tropical)
    
    print(f"{'Threshold':<25} {'k-factor':<12} {'Distance (AU)':<15} {'Distance (ly)':<15} {'Phase/pi':<10}")
    for name, data in boundaries.items():
        print(f"{name:<25} {data['k_factor']:<12.6f} {data['distance_AU']:<15.0f} {data['distance_ly']:<15.4f} {data['phase_accumulated']/math.pi:<10.3f}")
    print()
    
    print("5. OORT CLOUD STRUCTURE WITH CONFIDENCE BANDS")
    print(f"{'Oort Region':<30} {'Distance (AU)':<25} {'Distance (ly)':<15} {'Phase/pi':<10} {'Multiplicity':<12} {'Left Bias (%)':<13}")
    for region, (distance_AU, uncertainty_AU) in oort_boundaries.items():
        distance_ly = distance_AU / ly_AU_tropical
        effects = aperture_effects(distance_AU, year_tropical, tidal_exp)
        dist_range = f"{distance_AU - uncertainty_AU:,} - {distance_AU + uncertainty_AU:,}"
        left_bias_pct = effects['parity_ratio'] * 100
        print(f"{region:<30} {dist_range:<25} {distance_ly:<15.4f} {effects['phase_fraction_of_pi']:<10.3f} {effects['multiplicity']:<12} {left_bias_pct:<12.2f}%")
    print()
    
    print("6. CRITICAL PHASE THRESHOLDS")
    critical_phases = {
        "pi/4 (path splitting begins)": math.pi/4,
        "pi/2 (decoherence onset)": math.pi/2,
        "pi (complete cycle)": math.pi,
        "2pi (double cycle)": 2*math.pi
    }
    
    for description, phase_target in critical_phases.items():
        d_AU = (phase_target / math.pi) * ly_AU_tropical
        d_ly = d_AU / ly_AU_tropical
        print(f"{description:<30} occurs at {d_AU:>10,.0f} AU = {d_ly:>7.4f} ly")
    print()
    
    print("7. 1 LIGHT-YEAR COMPLETE CYCLE ANALYSIS")
    effects_1ly = aperture_effects(ly_AU_tropical, year_tropical, tidal_exp)
    print(f"At exactly 1 light-year ({ly_AU_tropical:.0f} AU):")
    print(f"  Accumulated phase:        {effects_1ly['phase_radians']:.6f} rad = pi")
    print(f"  Phase/pi:                  {effects_1ly['phase_fraction_of_pi']:.6f}")
    print(f"  Coherence factor:         {effects_1ly['coherence']:.4f}")
    print(f"  Multiplicity:             {effects_1ly['multiplicity']}")
    left_bias_pct = effects_1ly['parity_ratio'] * 100
    print(f"  Left bias:                {left_bias_pct:.2f}%")
    print(f"  Observable fraction:      {effects_1ly['observable_fraction']:.4%}")
    print()
    
    print("8. CGM PREDICTIONS VS OORT CLOUD OBSERVATIONS")
    matches = [
        ("Amplitude limit (1/8pi)", 2517, "Inner Hills cloud start", 2000, 500),
        ("Quarter horizon (1/4)", 15810, "Inner Hills cloud middle", 10000, 2000),
        ("Half horizon (1/2)", 31621, "Outer Oort start", 20000, 5000),
        ("Full cycle (1)", 63241, "Outer Oort typical", 50000, 15000),
        ("Double cycle (2)", 126483, "Outer Oort extended", 100000, 25000),
    ]
    
    print(f"{'CGM Threshold':<25} {'Predicted (AU)':<15} {'Oort Feature':<25} {'Observed (AU)':<25}")
    for cgm_name, cgm_au, oort_name, oort_au, oort_unc in matches:
        lower = oort_au - oort_unc
        upper = oort_au + oort_unc
        oort_str = f"{lower:,} - {upper:,}"
        if lower <= cgm_au <= upper:
            deviation = "in range"
        else:
            pct_diff = abs(cgm_au - oort_au) / oort_au * 100
            deviation = f"{pct_diff:.1f}% diff"
        print(f"{cgm_name:<25} {cgm_au:<15,} {oort_name:<25} {oort_str:<25} ({deviation})")
    print()
    
    print("9. PHASE-WEIGHTED COHERENCE AT OORT DISTANCES")
    oort_test_distances = [2000, 10000, 20000, 50000, 100000, 200000]
    print(f"{'Distance (AU)':<15} {'Phase/pi':<10} {'Coherence':<12} {'Multiplicity':<12} {'Left Bias (%)':<13}")
    for d in oort_test_distances:
        effects = aperture_effects(d, year_tropical, tidal_exp)
        left_bias_pct = effects['parity_ratio'] * 100
        print(f"{d:<15,} {effects['phase_fraction_of_pi']:<10.3f} {effects['coherence']:<12.4f} {effects['multiplicity']:<12} {left_bias_pct:<12.2f}")
    print()
    
    print("10. FALSIFICATION METRICS: COMET SIBLING PREDICTIONS")
    print("Expected number of phase-sliced comet siblings at each Oort distance:")
    print(f"{'Distance (AU)':<15} {'Siblings per comet':<18} {'Light-time (yr)':<15} {'Phase-time (yr)':<18}")
    for d in oort_test_distances:
        siblings = compute_comet_sibling_count(d, year_tropical, tidal_exp)
        light_time = compute_light_time_depth(d)
        phase_time = compute_phase_time_depth(d, year_tropical, tidal_exp, T_coh='local')
        multiplicity_dominant = "***" if siblings > 1.0 else ""
        print(f"{d:<15,} {siblings:<18.2f} {light_time:<15.4f} {phase_time:<18,.0f} {multiplicity_dominant}")
    print()
    print("Note: Distances marked *** are multiplicity-dominant (siblings > 1).")
    print("Gaia parallax/proper motions robust to >100 ly; illusions here are")
    print("phase-sliced projections of source classes, not literal Solar duplicates.")
    print()
    
    print("11. GALACTIC CENTER AS TEMPORAL PROJECTION")
    galactic_center_distance_ly = 26000  # Approximate distance to Sagittarius A*
    galactic_center_distance_AU = galactic_center_distance_ly * ly_AU_tropical
    effects_gc = aperture_effects(galactic_center_distance_AU, year_tropical, tidal_exp)
    light_time_gc = compute_light_time_depth(galactic_center_distance_AU)
    phase_time_gc_hubble = compute_phase_time_depth(galactic_center_distance_AU, year_tropical, tidal_exp, T_coh=None)
    phase_time_gc_local = compute_phase_time_depth(galactic_center_distance_AU, year_tropical, tidal_exp, T_coh='local')
    print(f"Galactic center distance:           {galactic_center_distance_ly:,} ly = {galactic_center_distance_AU:,.0f} AU")
    print(f"Light-time depth:                   {light_time_gc:,.0f} years")
    print(f"Phase at galactic center:          {effects_gc['phase_fraction_of_pi']:.3f} pi")
    print(f"CGM phase-time (T_coh = 1/H0):      {phase_time_gc_hubble:,.0f} years")
    print(f"CGM phase-time (T_coh = 1 year):   {phase_time_gc_local:,.0f} years")
    print(f"Multiplicity:                       {effects_gc['multiplicity']}")
    print(f"Expected delay (if Sun's past):     {phase_time_gc_local / 26000:.1f}x present age")
    print()
    
    if SHOW_DIAGNOSTICS:
        print("12. HILBERT PROXY METRICS (diagnostic)")
        print("Unity and balance proxies approximate L²(S²) expectations. Interpret qualitatively.")
        print(f"{'Distance (AU)':<15} {'Phase/pi':<10} {'U_proxy':<12} {'B_proxy':<12} {'U_proxy vs e^-φ':<18}")
        for d in oort_test_distances:
            phase = phase_at_distance_tidal(d, year_tropical, tidal_exp)
            u_proxy = compute_unity_proxy(phase)
            b_proxy = compute_balance_proxy(phase)
            coherence = math.exp(-phase / math.pi)
            comparison = f"{u_proxy:.4f} vs {coherence:.4f}"
            print(f"{d:<15,} {phase/math.pi:<10.3f} {u_proxy:<12.4f} {b_proxy:<12.4f} {comparison:<18}")
        print()

        print("13. Q_G = 4π SOLID-ANGLE BUDGET (diagnostic)")
        print("Fraction of 4π steradians in coherent (ℓ=0) vs incoherent (ℓ>0) modes.")
        print(f"{'Distance (AU)':<15} {'Coherent (ℓ=0)':<18} {'Incoherent (ℓ>0)':<20} {'Coherent (sr)':<15} {'Incoherent (sr)':<15}")
        for d in oort_test_distances:
            budget = compute_qg_budget(d, year_tropical, tidal_exp)
            print(f"{d:<15,} {budget['coherent_fraction']:<18.4f} {budget['incoherent_fraction']:<20.4f} {budget['coherent_solid_angle']:<15.4f} {budget['incoherent_solid_angle']:<15.4f}")
        print()

        spectrum = compute_fourier_spectrum(max_distance_ly=5.0, samples=512, tidal_exponent=tidal_exp)
        print("14. FOURIER ANALYSIS OF COHERENCE (diagnostic)")
        print(f"{'Frequency (1/ly)':<20} {'Magnitude':<15}")
        for component in spectrum:
            print(f"{component['frequency_cycles_per_ly']:<20.6f} {component['magnitude']:<15.6f}")
        print("(Exponential profile dominated by low-frequency components; interpret cautiously.)")
        print()

    print("12. LOCAL HORIZON ANALYSIS")
    print("Beyond 1 ly, objects become phase-sliced illusions:")
    test_distances_ly = [0.5, 1.0, 2.0, 10.0, 100.0]
    print(f"{'Distance (ly)':<15} {'Distance (AU)':<15} {'Phase/pi':<10} {'Coherence':<12} {'Siblings':<10} {'Light-time (yr)':<15} {'Phase-time (yr)':<15}")
    for d_ly in test_distances_ly:
        d_AU = d_ly * ly_AU_tropical
        effects = aperture_effects(d_AU, year_tropical, tidal_exp)
        siblings = compute_comet_sibling_count(d_AU, year_tropical, tidal_exp)
        light_time = compute_light_time_depth(d_AU)
        phase_time = compute_phase_time_depth(d_AU, year_tropical, tidal_exp, T_coh='local')
        print(f"{d_ly:<15.1f} {d_AU:<15,.0f} {effects['phase_fraction_of_pi']:<10.3f} {effects['coherence']:<12.4f} {siblings:<10.2f} {light_time:<15.4f} {phase_time:<15,.0f}")

    print()
    print("13. GEOMETRIC PERIODS FROM S² WRAPPING")
    periods = compute_geometric_period(tidal_exp)
    print(f"{'Closure':<20} {'Distance (ly)':<15} {'Distance (AU)':<15} {'Phase (rad)':<12}")
    for label, data in periods.items():
        print(f"{label:<20} {data['distance_ly']:<15.4f} {data['distance_AU']:<15.0f} {data['phase']:<12.3f}")

    closures = find_monodromy_closures(max_multiple=40, tidal_exponent=tidal_exp, max_distance_ly=5.0)
    print()
    print("14. MONODROMY CLOSURE LADDER (n × δ_BU)")
    print(f"{'n':<5} {'Phase (rad)':<15} {'Distance (ly)':<15} {'Distance (AU)':<15}")
    for entry in closures:
        print(f"{entry['multiple']:<5} {entry['phase']:<15.6f} {entry['distance_ly']:<15.6f} {entry['distance_AU']:<15.0f}")

    matches = compute_monodromy_matches(tidal_exp)
    print()
    print("15. MONODROMY ↔ PHASE THRESHOLD ALIGNMENT")
    print(f"{'Threshold':<20} {'n':<5} {'Closure (ly)':<15} {'Target (ly)':<15} {'% diff':<10}")
    for m in matches:
        print(f"{m['label']:<20} {m['n']:<5} {m['closure_distance_ly']:<15.6f} {m['target_distance_ly']:<15.6f} {m['pct_diff']:<10.3f}")

    print()
    print("16. MICROSCOPIC ANALOGUE: PARTICLE SPIN WRAPPING")
    print("Electron Compton wavelength as micro 'cycle distance'")
    micro_e = compute_nested_thresholds(compton_electron, "Electron")
    print(f"{'Threshold':<30} {'k-factor':<12} {'Distance (m)':<15} {'Distance (AU)':<15} {'Phase/(4π)':<12} {'Spin proxy':<12}")
    for name, data in micro_e.items():
        spin = compute_spin_proxy(data['distance_m'], compton_electron)
        phase_micro = 4 * math.pi * (data['distance_m'] / compton_electron)
        print(f"{name:<30} {data['k_factor']:<12.6f} {data['distance_m']:<15.3e} {data['distance_AU']:<15.3e} {phase_micro/(4*math.pi):<12.3f} {spin:<12.3f}")
    print()

    print("Proton Compton wavelength (composite spin structure)")
    micro_p = compute_nested_thresholds(compton_proton, "Proton")
    print(f"{'Threshold':<30} {'k-factor':<12} {'Distance (m)':<15} {'Distance (AU)':<15} {'Phase/(4π)':<12} {'Spin proxy':<12}")
    for name, data in micro_p.items():
        spin = compute_spin_proxy(data['distance_m'], compton_proton)
        phase_micro = 4 * math.pi * (data['distance_m'] / compton_proton)
        print(f"{name:<30} {data['k_factor']:<12.6f} {data['distance_m']:<15.3e} {data['distance_AU']:<15.3e} {phase_micro/(4*math.pi):<12.3f} {spin:<12.3f}")
    print()

    print("Spin proxy hits 0.5 at one Compton cycle (4π wrapping), matching spin-1/2.")
    print("Nested CGM cycles tie microscopic SU(2) behavior to macroscopic phase closure.")

if __name__ == "__main__":
    main()
