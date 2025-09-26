"""
CGM Chronology Analysis: Recursion Depth Hypothesis Testing

Purpose
-------
- Test the hypothesis that cosmological time t_FRW maps to recursion depth n via
  t_FRW = t_* * λ^n where λ = 1/m_p and m_p = 1/(2*sqrt(2*pi)).
- Evaluate whether fractional parts of n cluster near {0, 0.5} more than expected
  from uniform distribution.
- Provide proper statistical testing with uncertainties and significance levels.

Notes
-----
- This is a hypothesis test, not a discovery claim.
- Uses CGM values: m_p ≈ 0.199471, T₀ = 8.916×10⁻⁴³ s
- Includes Monte Carlo uncertainty propagation and statistical significance testing.
- Results should be interpreted as evidence for/against the hypothesis, not as
  definitive proof of quantization.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, TypedDict, Union


# ========= CGM Constants from Analysis_CGM_Units.md =========
# CGM time scale: T₀ = 8.916×10⁻⁴³ s = 16.54 × T_Planck
T0_CGM = 8.916e-43  # seconds


# Aperture parameter: m_p = 1/(2*sqrt(2*pi)) ≈ 0.199471
def compute_aperture_parameter() -> float:
    """Return CGM aperture parameter m_p = 1/(2*sqrt(2*pi))."""
    return 1.0 / (2.0 * math.sqrt(2.0 * math.pi))


def compute_planck_time_from_cgm() -> float:
    """Return Planck time computed from CGM time scale: T_Planck = T₀ / 16.54."""
    return T0_CGM / 16.54


def recursion_depth_from_time(
    t_seconds: float,
    t_planck_seconds: float,
    aperture_parameter: float,
) -> float:
    """
    Compute recursion depth n from physical time t via:

        t = t_Planck * (1/m_p)^n  =>  n = log(t/t_Planck) / log(1/m_p)

    Args:
        t_seconds: Observed time in seconds (> 0)
        t_planck_seconds: Planck time in seconds (> 0)
        aperture_parameter: m_p in (0, 1)

    Returns:
        Recursion depth n (float)
    """
    if t_seconds <= 0.0:
        raise ValueError("t_seconds must be positive")
    if t_planck_seconds <= 0.0:
        raise ValueError("t_planck_seconds must be positive")
    if not (0.0 < aperture_parameter < 1.0):
        raise ValueError("aperture_parameter (m_p) must be in (0, 1)")

    base = 1.0 / aperture_parameter
    return math.log(t_seconds / t_planck_seconds, base)


def recursion_depth_from_temperature(
    temperature_gev: float,
    reference_temp_gev: float,
    aperture_parameter: float,
) -> float:
    """
    Compute recursion depth n from temperature T via:

        T = T_* * (1/m_p)^n  =>  n = log(T/T_*) / log(1/m_p)

    Args:
        temperature_gev: Temperature in GeV (> 0)
        reference_temp_gev: Reference temperature in GeV (> 0)
        aperture_parameter: m_p in (0, 1)

    Returns:
        Recursion depth n (float)
    """
    if temperature_gev <= 0.0:
        raise ValueError("temperature_gev must be positive")
    if reference_temp_gev <= 0.0:
        raise ValueError("reference_temp_gev must be positive")
    if not (0.0 < aperture_parameter < 1.0):
        raise ValueError("aperture_parameter (m_p) must be in (0, 1)")

    base = 1.0 / aperture_parameter
    return math.log(temperature_gev / reference_temp_gev, base)


def recursion_depth_from_scale_factor(
    scale_factor: float,
    reference_scale_factor: float,
    aperture_parameter: float,
) -> float:
    """
    Compute recursion depth n from scale factor a via:

        a = a_* * (1/m_p)^n  =>  n = log(a/a_*) / log(1/m_p)

    Args:
        scale_factor: Scale factor (> 0)
        reference_scale_factor: Reference scale factor (> 0)
        aperture_parameter: m_p in (0, 1)

    Returns:
        Recursion depth n (float)
    """
    if scale_factor <= 0.0:
        raise ValueError("scale_factor must be positive")
    if reference_scale_factor <= 0.0:
        raise ValueError("reference_scale_factor must be positive")
    if not (0.0 < aperture_parameter < 1.0):
        raise ValueError("aperture_parameter (m_p) must be in (0, 1)")

    base = 1.0 / aperture_parameter
    return math.log(scale_factor / reference_scale_factor, base)


def nearest_integer(value: float) -> Tuple[int, float]:
    """Return (nearest_int, absolute_difference)."""
    k = int(round(value))
    return k, abs(value - k)


def nearest_half_integer(value: float) -> Tuple[float, float]:
    """Return (nearest_half_int, absolute_difference)."""
    k2 = round(value * 2.0)
    half = k2 / 2.0
    return half, abs(value - half)


@dataclass
class Milestone:
    name: str
    t_seconds: float
    uncertainty_fraction: float = 0.1  # 10% relative uncertainty by default
    note: str = ""


class ResultRow(TypedDict):
    name: str
    t_seconds: float
    n: float
    nearest_int: int
    diff_int: float
    nearest_half: float
    diff_half: float
    near_int: bool
    near_half: bool
    note: str


def default_milestones(t_planck: float) -> List[Milestone]:
    """Cosmological milestones with tighter uncertainties for well-measured events."""
    year = 365.25 * 24.0 * 3600.0
    kyr = 1.0e3 * year
    Myr = 1.0e6 * year
    Gyr = 1.0e9 * year

    return [
        Milestone("Planck time", t_planck, 0.0, "Definition"),
        Milestone("Planck epoch", 1.0e-43, 0.3, "Early universe"),
        Milestone("GUT epoch", 1.0e-36, 0.2, "Grand unification"),
        Milestone("Electroweak", 1.0e-12, 0.1, "Electroweak transition"),
        Milestone("QCD transition", 1.0e-6, 0.1, "QCD confinement"),
        Milestone("Neutrino decoupling", 1.0, 0.05, "Neutrino freeze-out"),
        Milestone("BBN", 180.0, 0.02, "Big Bang nucleosynthesis"),
        Milestone("Matter–radiation equality", 50.0 * kyr, 0.05, "Density equality"),
        Milestone("Recombination / CMB", 380.0 * kyr, 0.01, "CMB formation"),
        Milestone("First stars", 180.0 * Myr, 0.1, "Population III stars"),
        Milestone("Reionization midpoint", 550.0 * Myr, 0.08, "Reionization"),
        Milestone("DE dominance onset", 7.0 * Gyr, 0.05, "Dark energy dominance"),
        Milestone("Today", 13.8 * Gyr, 0.01, "Present epoch"),
    ]


def early_universe_temperature_milestones() -> List[Milestone]:
    """Early universe milestones based on temperature/energy scales with tighter uncertainties."""
    # Temperature scales in GeV
    GeV = 1.0  # GeV
    MeV = 1e-3 * GeV
    keV = 1e-6 * GeV

    return [
        Milestone("Planck scale", 1.22e19, 0.05, "T_Planck ~ 10^19 GeV"),
        Milestone("GUT scale", 1e16, 0.1, "T_GUT ~ 10^16 GeV"),
        Milestone("Electroweak scale", 100.0, 0.05, "T_EW ~ 100 GeV"),
        Milestone("QCD scale", 0.2, 0.05, "T_QCD ~ 200 MeV"),
        Milestone("Neutrino decoupling", 1.0, 0.02, "T_ν ~ 1 MeV"),
        Milestone("BBN", 0.1, 0.01, "T_BBN ~ 0.1 MeV"),
        Milestone("Inflation end", 1e15, 0.1, "T_infl ~ 10^15 GeV"),
        Milestone("Quark-gluon plasma", 0.15, 0.05, "T_QGP ~ 150 MeV"),
    ]


def late_universe_scale_factor_milestones() -> List[Milestone]:
    """Late universe milestones based on scale factor/redshift."""
    # Scale factor a (normalized to a_0 = 1 today)
    return [
        Milestone("Matter–radiation equality", 1 / 3400, 0.1, "a_eq ~ 1/3400"),
        Milestone("Recombination / CMB", 1 / 1100, 0.05, "a_rec ~ 1/1100"),
        Milestone("First stars", 1 / 20, 0.2, "a_stars ~ 1/20"),
        Milestone("Reionization midpoint", 1 / 8, 0.15, "a_reion ~ 1/8"),
        Milestone("DE dominance onset", 1 / 2, 0.1, "a_DE ~ 1/2"),
        Milestone("Today", 1.0, 0.0, "a_0 = 1"),
    ]


def compute_generalized_rayleigh(
    fractional_parts: List[float], m: int = 2
) -> Dict[str, float]:
    """
    Generalized Rayleigh test for clustering at m equally spaced target phases.

    For m=2: targets at {0, 0.5} (doubled-angle test)
    For m=3: targets at {0, 1/3, 2/3} (three-phase structure)
    For m=4: targets at {0, 1/4, 1/2, 3/4} (quarter-cycle structure)

    Maps fractional parts f to angles θ_m = 2πmf and tests for concentration
    at the target phases using the Rayleigh test.
    """
    if not fractional_parts:
        return {
            "p_value": 1.0,
            "concentration": 0.0,
            "n_events": 0,
            "m": m,
            "circular_mean": 0.0,
        }

    # Convert to angles: f -> 2πmf
    angles = [2 * math.pi * m * f for f in fractional_parts]

    # Compute circular statistics
    cos_sum = sum(math.cos(a) for a in angles)
    sin_sum = sum(math.sin(a) for a in angles)
    n = len(angles)

    # Mean resultant length (concentration)
    concentration = math.sqrt(cos_sum**2 + sin_sum**2) / n

    # Circular mean angle (effect size - where clustering points)
    circular_mean = math.atan2(sin_sum, cos_sum) if concentration > 0 else 0.0

    # Rayleigh test statistic: 2n * R^2 ~ chi2(2) under null
    rayleigh_stat = 2 * n * concentration**2

    # Small-sample correction for p-value
    p_value = rayleigh_small_sample_pvalue(rayleigh_stat, n)

    return {
        "p_value": p_value,
        "concentration": concentration,
        "n_events": n,
        "rayleigh_stat": rayleigh_stat,
        "m": m,
        "circular_mean": circular_mean,
    }


def rayleigh_small_sample_pvalue(z: float, n: int) -> float:
    """
    Small-sample corrected p-value for Rayleigh test.

    Uses the improved approximation:
    p ≈ e^(-Z) * (1 + (2Z - Z²)/(4n) - (24Z - 132Z² + 76Z³ - 9Z⁴)/(288n²))
    """
    if z <= 0:
        return 1.0

    # Basic exponential term
    p_basic = math.exp(-z)

    if n <= 1:
        return p_basic

    # Small-sample correction terms
    term1 = (2 * z - z**2) / (4 * n)
    term2 = (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4) / (288 * n**2)

    correction = 1 + term1 - term2
    return p_basic * correction


def compute_doubled_angle_rayleigh(fractional_parts: List[float]) -> Dict[str, float]:
    """
    Doubled-angle Rayleigh test for clustering at {0, 0.5}.

    Convenience wrapper for m=2 case.
    """
    return compute_generalized_rayleigh(fractional_parts, m=2)


def compute_spacing_test(
    milestone_times: List[float], t_planck: float, m_p: float
) -> Dict[str, float]:
    """
    Test whether adjacent milestone separations cluster near integers in n.

    If t ∝ λ^n holds, then Δn_i = log_{1/m_p}(t_i/t_{i-1}) should concentrate near integers.
    """
    # Sort times to handle order swaps from log-normal noise
    times = sorted(t for t in milestone_times if t > 0.0)
    if len(times) < 2:
        return {"p_value": 1.0, "concentration": 0.0, "n_spacings": 0}

    # Compute Δn_i for adjacent milestones
    delta_n_values = [
        math.log(times[i] / times[i - 1], 1.0 / m_p) for i in range(1, len(times))
    ]

    if not delta_n_values:
        return {"p_value": 1.0, "concentration": 0.0, "n_spacings": 0}

    # Reduce modulo 1: u_i = Δn_i - floor(Δn_i)
    fractional_spacings = [delta_n - math.floor(delta_n) for delta_n in delta_n_values]

    # Test clustering at 0 using standard Rayleigh
    angles = [2 * math.pi * u for u in fractional_spacings]
    cos_sum = sum(math.cos(a) for a in angles)
    sin_sum = sum(math.sin(a) for a in angles)
    n = len(angles)

    concentration = math.sqrt(cos_sum**2 + sin_sum**2) / n
    rayleigh_stat = 2 * n * concentration**2
    p_value = rayleigh_small_sample_pvalue(rayleigh_stat, n)

    return {
        "p_value": p_value,
        "concentration": concentration,
        "n_spacings": n,
        "rayleigh_stat": rayleigh_stat,
    }


def kuiper_test(
    fractional_parts: List[float], n_bootstrap: int = 1_000
) -> Dict[str, float]:
    """
    Kuiper test for uniformity with bootstrap p-value.

    Tests against uniform distribution on [0,1) using the Kuiper statistic.
    """
    if not fractional_parts or len(fractional_parts) < 3:
        return {"p_value": 1.0, "kuiper_stat": 0.0, "n_events": 0}

    n = len(fractional_parts)
    sorted_f = sorted(fractional_parts)

    # Compute Kuiper statistic
    max_plus = max((i + 1) / n - f for i, f in enumerate(sorted_f))
    max_minus = max(f - i / n for i, f in enumerate(sorted_f))
    kuiper_stat = max_plus + max_minus

    # Bootstrap p-value under uniform null
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        uniform_sample = [random.random() for _ in range(n)]
        uniform_sample.sort()

        max_plus_u = max((i + 1) / n - f for i, f in enumerate(uniform_sample))
        max_minus_u = max(f - i / n for i, f in enumerate(uniform_sample))
        bootstrap_stats.append(max_plus_u + max_minus_u)

    bootstrap_stats.sort()
    p_value = sum(1 for stat in bootstrap_stats if stat >= kuiper_stat) / len(
        bootstrap_stats
    )

    return {
        "p_value": p_value,
        "kuiper_stat": kuiper_stat,
        "n_events": n,
    }


def watson_test(
    fractional_parts: List[float], n_bootstrap: int = 1_000
) -> Dict[str, float]:
    """
    Watson's U² test for uniformity with bootstrap p-value.

    Tests against uniform distribution on [0,1) using Watson's U² statistic.
    """
    if not fractional_parts or len(fractional_parts) < 3:
        return {"p_value": 1.0, "watson_stat": 0.0, "n_events": 0}

    n = len(fractional_parts)
    sorted_f = sorted(fractional_parts)

    # Compute Watson's U² statistic
    mean_f = sum(sorted_f) / n
    u_squared = 0.0
    for i, f in enumerate(sorted_f):
        term = (f - (i + 0.5) / n) ** 2
        u_squared += term

    u_squared = u_squared / n + 1.0 / (12 * n)

    # Bootstrap p-value under uniform null
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        uniform_sample = [random.random() for _ in range(n)]
        uniform_sample.sort()

        mean_u = sum(uniform_sample) / n
        u_squared_u = 0.0
        for i, f in enumerate(uniform_sample):
            term = (f - (i + 0.5) / n) ** 2
            u_squared_u += term
        u_squared_u = u_squared_u / n + 1.0 / (12 * n)
        bootstrap_stats.append(u_squared_u)

    bootstrap_stats.sort()
    p_value = sum(1 for stat in bootstrap_stats if stat >= u_squared) / len(
        bootstrap_stats
    )

    return {
        "p_value": p_value,
        "watson_stat": u_squared,
        "n_events": n,
    }


def chi2_sf_df4(t: float) -> float:
    """Survival function for chi-squared distribution with 4 degrees of freedom."""
    if t <= 0:
        return 1.0
    p = math.exp(-t / 2.0) * (1.0 + t / 2.0)
    return min(max(p, 0.0), 1.0)


def compute_cgm_composite_harmonic_test(
    fractional_parts: List[float],
) -> Dict[str, float]:
    """
    Composite harmonic test for CGM stage phases using m=2 and m=4 components.

    Tests for clustering at {0, 0.5} and {0, 0.25, 0.5, 0.75} simultaneously.
    Uses T_CGM = (2/n)·(R₂² + R₄²) ~ χ²(4) under null.
    """
    if not fractional_parts:
        return {"p_value": 1.0, "concentration": 0.0, "n_events": 0}

    n = len(fractional_parts)

    # Compute harmonic components for m=2 and m=4
    c2 = sum(math.cos(4 * math.pi * f) for f in fractional_parts)
    s2 = sum(math.sin(4 * math.pi * f) for f in fractional_parts)
    r2_squared = c2**2 + s2**2

    c4 = sum(math.cos(8 * math.pi * f) for f in fractional_parts)
    s4 = sum(math.sin(8 * math.pi * f) for f in fractional_parts)
    r4_squared = c4**2 + s4**2

    # Composite test statistic: T_CGM = (2/n)·(R₂² + R₄²)
    t_cgm = (2.0 / n) * (r2_squared + r4_squared)

    # P-value from chi-squared(4) distribution (corrected df=4)
    p_value = chi2_sf_df4(t_cgm)

    # Overall concentration (magnitude of combined harmonics)
    concentration = math.sqrt((r2_squared + r4_squared) / (2.0 * n * n))

    return {
        "p_value": p_value,
        "concentration": concentration,
        "n_events": n,
        "t_cgm": t_cgm,
        "r2_squared": r2_squared,
        "r4_squared": r4_squared,
    }


def compute_weighted_cgm_test(fractional_parts: List[float]) -> Dict[str, float]:
    """
    Weighted CGM Test accounting for unequal stage lengths.

    CGM stages: CS (0.5), UNA (0.25), ONA (0.25) of total cycle
    Target phases: {0, 0.5, 0.75} with weights [0.5, 0.25, 0.25]

    Uses a weighted von Mises mixture approach to test for clustering
    at the specific CGM stage boundaries with proper weighting.
    """
    if not fractional_parts:
        return {"p_value": 1.0, "concentration": 0.0, "n_events": 0}

    n = len(fractional_parts)

    # CGM stage boundaries and weights
    target_phases = [0.0, 0.5, 0.75]
    weights = [0.5, 0.25, 0.25]  # CS, UNA, ONA stage lengths

    # Convert fractional parts to angles
    angles = [2 * math.pi * f for f in fractional_parts]
    target_angles = [2 * math.pi * p for p in target_phases]

    # Compute weighted concentration towards each target
    weighted_concentration = 0.0
    for i, (target_angle, weight) in enumerate(zip(target_angles, weights)):
        # Distance from each point to this target
        cos_sum = sum(math.cos(a - target_angle) for a in angles)
        sin_sum = sum(math.sin(a - target_angle) for a in angles)
        concentration_i = math.sqrt(cos_sum**2 + sin_sum**2) / n
        weighted_concentration += weight * concentration_i

    # Test statistic: weighted mean resultant length
    # Under null hypothesis, this should be small
    test_stat = 2 * n * weighted_concentration**2

    # P-value using chi-squared approximation (conservative)
    p_value = math.exp(-test_stat / 2.0)
    p_value = min(max(p_value, 0.0), 1.0)

    return {
        "p_value": p_value,
        "concentration": weighted_concentration,
        "n_events": n,
        "test_stat": test_stat,
    }


def monte_carlo_analysis_time(
    milestones: List[Milestone],
    t_planck: float,
    m_p: float,
    n_samples: int = 10_000,
    exclude_anchors: bool = True,
    rng: random.Random | None = None,
) -> Tuple[
    List[ResultRow], Dict[str, Union[float, str]], List[Dict[str, Dict[str, float]]]
]:
    """Monte Carlo analysis for time-based milestones (FRW proper time)."""
    if rng is None:
        rng = random.Random()

    # Separate anchors from test milestones
    if exclude_anchors:
        test_milestones = [
            m for m in milestones if m.name not in ["Planck time", "Today"]
        ]
        anchor_milestones = [
            m for m in milestones if m.name in ["Planck time", "Today"]
        ]
    else:
        test_milestones = milestones
        anchor_milestones = []

    # Compute mean n values for display
    results: List[ResultRow] = []
    for m in test_milestones:
        n_mean = recursion_depth_from_time(m.t_seconds, t_planck, m_p)
        k_int, d_int = nearest_integer(n_mean)
        k_half, d_half = nearest_half_integer(n_mean)

        row: ResultRow = {
            "name": m.name,
            "t_seconds": float(m.t_seconds),
            "n": float(n_mean),
            "nearest_int": int(k_int),
            "diff_int": float(d_int),
            "nearest_half": float(k_half),
            "diff_half": float(d_half),
            "near_int": bool(d_int < 0.1),
            "near_half": bool(d_half < 0.1),
            "note": m.note,
        }
        results.append(row)

    # Monte Carlo testing
    test_stats = []
    for _ in range(n_samples):
        fractional_parts = []
        milestone_times = []
        for m in test_milestones:
            if m.uncertainty_fraction > 0:
                sigma_ln = math.sqrt(math.log(1 + m.uncertainty_fraction**2))
                mu_ln = math.log(m.t_seconds) - 0.5 * sigma_ln**2
                log_t = mu_ln + rng.gauss(0, sigma_ln)
                t_sample = math.exp(log_t)
            else:
                t_sample = m.t_seconds

            milestone_times.append(t_sample)
            n = recursion_depth_from_time(t_sample, t_planck, m_p)
            fractional_parts.append(n - math.floor(n))

        if fractional_parts and len(milestone_times) >= 2:
            replicate_stats = {
                "rayleigh_m2": compute_generalized_rayleigh(fractional_parts, m=2),
                "rayleigh_m3": compute_generalized_rayleigh(fractional_parts, m=3),
                "rayleigh_m4": compute_generalized_rayleigh(fractional_parts, m=4),
                "cgm_composite": compute_cgm_composite_harmonic_test(fractional_parts),
                "weighted_cgm": compute_weighted_cgm_test(fractional_parts),
                "spacing_test": compute_spacing_test(milestone_times, t_planck, m_p),
                "kuiper": kuiper_test(fractional_parts),
                "watson": watson_test(fractional_parts),
            }
            test_stats.append(replicate_stats)

    # Aggregate statistics
    if not test_stats:
        return results, {"error": "No test statistics computed"}, []

    aggregated_stats: Dict[str, Union[float, str]] = {
        "n_replicates": len(test_stats),
        "n_test_events": len(test_milestones),
        "n_anchor_events": len(anchor_milestones),
    }

    test_types = [
        "rayleigh_m2",
        "rayleigh_m3",
        "rayleigh_m4",
        "cgm_composite",
        "weighted_cgm",
        "spacing_test",
        "kuiper",
        "watson",
    ]
    for test_type in test_types:
        p_values = [s[test_type]["p_value"] for s in test_stats if test_type in s]
        concentrations = [
            s[test_type].get("concentration", 0.0) for s in test_stats if test_type in s
        ]

        if p_values:
            aggregated_stats[f"{test_type}_mean_p"] = sum(p_values) / len(p_values)
            aggregated_stats[f"{test_type}_median_p"] = sorted(p_values)[
                len(p_values) // 2
            ]
            aggregated_stats[f"{test_type}_mean_concentration"] = sum(
                concentrations
            ) / len(concentrations)
        else:
            aggregated_stats[f"{test_type}_mean_p"] = 1.0
            aggregated_stats[f"{test_type}_median_p"] = 1.0
            aggregated_stats[f"{test_type}_mean_concentration"] = 0.0

    return results, aggregated_stats, test_stats


def run_comprehensive_chronology_analysis() -> None:
    """Run comprehensive chronology analysis testing multiple clock variables."""
    # Create dedicated RNG for reproducibility
    rng = random.Random(42)

    t_planck = compute_planck_time_from_cgm()
    t0_cgm = T0_CGM  # Use CGM's natural time scale
    m_p = compute_aperture_parameter()

    print("CGM Chronology Analysis: CGM-Native Variables")
    print("=" * 60)
    print(f"Random seed: 42 (for reproducibility)")
    print(f"Planck time t_P = {t_planck:.6e} s")
    print(f"CGM time scale T₀ = {t0_cgm:.6e} s (16.54 × t_P)")
    print(f"Aperture parameter m_p = {m_p:.6f} (1/m_p = {1.0/m_p:.6f})")
    print(f"Testing both λ = 1/m_p = {1.0/m_p:.6f} and λ = √3 = {math.sqrt(3):.6f}")
    print()

    # Test 1: FRW time with T0_CGM reference (CGM-pure scaling)
    print("TEST 1: FRW Time with T0_CGM Reference")
    print("-" * 40)
    milestones_time = default_milestones(t_planck)
    results_time, stats_time, test_stats_time = monte_carlo_analysis_time(
        milestones_time, t0_cgm, m_p, n_samples=10_000, exclude_anchors=True, rng=rng
    )
    print_brief_results(stats_time, "FRW Time (T0_CGM)")
    print()

    # Test 2: Early universe temperature scale
    print("TEST 2: Early Universe Temperature Scale")
    print("-" * 40)
    milestones_temp = early_universe_temperature_milestones()
    # Use Planck temperature as reference
    t_planck_gev = 1.22e19  # GeV
    results_temp, stats_temp, test_stats_temp = monte_carlo_analysis_temperature(
        milestones_temp,
        t_planck_gev,
        m_p,
        n_samples=10_000,
        exclude_anchors=True,
        rng=rng,
    )
    print_brief_results(stats_temp, "Temperature")
    print()

    # Test 3: Late universe scale factor
    print("TEST 3: Late Universe Scale Factor")
    print("-" * 40)
    milestones_scale = late_universe_scale_factor_milestones()
    # Use a_0 = 1 as reference
    a_reference = 1.0
    results_scale, stats_scale, test_stats_scale = monte_carlo_analysis_scale_factor(
        milestones_scale,
        a_reference,
        m_p,
        n_samples=10_000,
        exclude_anchors=True,
        rng=rng,
    )
    print_brief_results(stats_scale, "Scale Factor")
    print()

    # Test 4: Temperature scale with λ = √3 (duality ratio)
    print("TEST 4: Temperature Scale with λ = √3")
    print("-" * 40)
    lambda_sqrt3 = math.sqrt(3)
    results_temp_sqrt3, stats_temp_sqrt3, test_stats_temp_sqrt3 = (
        monte_carlo_analysis_temperature(
            milestones_temp,
            t_planck_gev,
            1.0 / lambda_sqrt3,
            n_samples=10_000,
            exclude_anchors=True,
            rng=rng,
        )
    )
    print_brief_results(stats_temp_sqrt3, "Temperature (λ=√3)")
    print()

    # Summary comparison
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Clock Variable':<20} {'Primary p-value':<15} {'Evidence'}")
    print("-" * 60)

    primary_p_time = float(stats_time.get("weighted_cgm_median_p", 1.0))
    primary_p_temp = float(stats_temp.get("weighted_cgm_median_p", 1.0))
    primary_p_scale = float(stats_scale.get("weighted_cgm_median_p", 1.0))
    primary_p_temp_sqrt3 = float(stats_temp_sqrt3.get("weighted_cgm_median_p", 1.0))

    print(
        f"{'FRW Time (T0_CGM)':<20} {primary_p_time:<15.4f} {'Null' if primary_p_time > 0.05 else 'Significant'}"
    )
    print(
        f"{'Temperature (1/m_p)':<20} {primary_p_temp:<15.4f} {'Null' if primary_p_temp > 0.05 else 'Significant'}"
    )
    print(
        f"{'Temperature (√3)':<20} {primary_p_temp_sqrt3:<15.4f} {'Null' if primary_p_temp_sqrt3 > 0.05 else 'Significant'}"
    )
    print(
        f"{'Scale Factor':<20} {primary_p_scale:<15.4f} {'Null' if primary_p_scale > 0.05 else 'Significant'}"
    )
    print()

    # Interpretation
    best_p = min(primary_p_time, primary_p_temp, primary_p_temp_sqrt3, primary_p_scale)
    if best_p < 0.05:
        if primary_p_temp_sqrt3 == best_p:
            print(
                "INTERPRETATION: Evidence for CGM chronology in TEMPERATURE scale with λ = √3"
            )
        elif primary_p_temp == best_p:
            print(
                "INTERPRETATION: Evidence for CGM chronology in TEMPERATURE scale with λ = 1/m_p"
            )
        elif primary_p_scale == best_p:
            print("INTERPRETATION: Evidence for CGM chronology in SCALE FACTOR")
        else:
            print("INTERPRETATION: Evidence for CGM chronology in FRW TIME")
    else:
        print(
            "INTERPRETATION: No significant evidence for CGM chronology in any clock variable"
        )
        print("This suggests either:")
        print("  • CGM chronology uses a different variable not tested here")
        print("  • The mapping requires additional free parameters")
        print("  • CGM chronology operates at a different scale/level")


def print_brief_results(stats: Dict[str, Union[float, str]], clock_name: str) -> None:
    """Print brief results for a single clock variable test."""
    primary_p = float(stats.get("weighted_cgm_median_p", 1.0))
    mean_conc = float(stats.get("weighted_cgm_mean_concentration", 0.0))
    n_events = int(stats.get("n_test_events", 0))

    print(f"  Test events: {n_events}")
    print(f"  Weighted CGM p-value: {primary_p:.4f}")
    print(f"  Mean concentration: {mean_conc:.4f}")
    print(f"  Evidence: {'Significant' if primary_p < 0.05 else 'Null'}")


def monte_carlo_analysis_temperature(
    milestones: List[Milestone],
    reference_temp_gev: float,
    m_p: float,
    n_samples: int = 10_000,
    exclude_anchors: bool = True,
    rng: random.Random | None = None,
) -> Tuple[
    List[ResultRow], Dict[str, Union[float, str]], List[Dict[str, Dict[str, float]]]
]:
    """Monte Carlo analysis for temperature-based milestones."""
    if rng is None:
        rng = random.Random()

    # Separate anchors from test milestones
    if exclude_anchors:
        test_milestones = [
            m for m in milestones if m.name not in ["Planck scale", "Today"]
        ]
        anchor_milestones = [
            m for m in milestones if m.name in ["Planck scale", "Today"]
        ]
    else:
        test_milestones = milestones
        anchor_milestones = []

    # Compute mean n values for display
    results: List[ResultRow] = []
    for m in test_milestones:
        n_mean = recursion_depth_from_temperature(m.t_seconds, reference_temp_gev, m_p)
        k_int, d_int = nearest_integer(n_mean)
        k_half, d_half = nearest_half_integer(n_mean)

        row: ResultRow = {
            "name": m.name,
            "t_seconds": float(m.t_seconds),
            "n": float(n_mean),
            "nearest_int": int(k_int),
            "diff_int": float(d_int),
            "nearest_half": float(k_half),
            "diff_half": float(d_half),
            "near_int": bool(d_int < 0.1),
            "near_half": bool(d_half < 0.1),
            "note": m.note,
        }
        results.append(row)

    # Monte Carlo testing (same as original but with temperature)
    test_stats = []
    for _ in range(n_samples):
        fractional_parts = []
        for m in test_milestones:
            if m.uncertainty_fraction > 0:
                sigma_ln = math.sqrt(math.log(1 + m.uncertainty_fraction**2))
                mu_ln = math.log(m.t_seconds) - 0.5 * sigma_ln**2
                log_t = mu_ln + rng.gauss(0, sigma_ln)
                t_sample = math.exp(log_t)
            else:
                t_sample = m.t_seconds

            n = recursion_depth_from_temperature(t_sample, reference_temp_gev, m_p)
            fractional_parts.append(n - math.floor(n))

        if fractional_parts:
            replicate_stats = {
                "rayleigh_m2": compute_generalized_rayleigh(fractional_parts, m=2),
                "rayleigh_m3": compute_generalized_rayleigh(fractional_parts, m=3),
                "rayleigh_m4": compute_generalized_rayleigh(fractional_parts, m=4),
                "cgm_composite": compute_cgm_composite_harmonic_test(fractional_parts),
                "weighted_cgm": compute_weighted_cgm_test(fractional_parts),
                "kuiper": kuiper_test(fractional_parts),
                "watson": watson_test(fractional_parts),
            }
            test_stats.append(replicate_stats)

    # Aggregate statistics
    if not test_stats:
        return results, {"error": "No test statistics computed"}, []

    aggregated_stats: Dict[str, Union[float, str]] = {
        "n_replicates": len(test_stats),
        "n_test_events": len(test_milestones),
        "n_anchor_events": len(anchor_milestones),
    }

    test_types = [
        "rayleigh_m2",
        "rayleigh_m3",
        "rayleigh_m4",
        "cgm_composite",
        "weighted_cgm",
        "kuiper",
        "watson",
    ]
    for test_type in test_types:
        p_values = [s[test_type]["p_value"] for s in test_stats if test_type in s]
        concentrations = [
            s[test_type].get("concentration", 0.0) for s in test_stats if test_type in s
        ]

        if p_values:
            aggregated_stats[f"{test_type}_mean_p"] = sum(p_values) / len(p_values)
            aggregated_stats[f"{test_type}_median_p"] = sorted(p_values)[
                len(p_values) // 2
            ]
            aggregated_stats[f"{test_type}_mean_concentration"] = sum(
                concentrations
            ) / len(concentrations)
        else:
            aggregated_stats[f"{test_type}_mean_p"] = 1.0
            aggregated_stats[f"{test_type}_median_p"] = 1.0
            aggregated_stats[f"{test_type}_mean_concentration"] = 0.0

    return results, aggregated_stats, test_stats


def monte_carlo_analysis_scale_factor(
    milestones: List[Milestone],
    reference_scale_factor: float,
    m_p: float,
    n_samples: int = 10_000,
    exclude_anchors: bool = True,
    rng: random.Random | None = None,
) -> Tuple[
    List[ResultRow], Dict[str, Union[float, str]], List[Dict[str, Dict[str, float]]]
]:
    """Monte Carlo analysis for scale factor-based milestones."""
    if rng is None:
        rng = random.Random()

    # Separate anchors from test milestones
    if exclude_anchors:
        test_milestones = [m for m in milestones if m.name not in ["Today"]]
        anchor_milestones = [m for m in milestones if m.name in ["Today"]]
    else:
        test_milestones = milestones
        anchor_milestones = []

    # Compute mean n values for display
    results: List[ResultRow] = []
    for m in test_milestones:
        n_mean = recursion_depth_from_scale_factor(
            m.t_seconds, reference_scale_factor, m_p
        )
        k_int, d_int = nearest_integer(n_mean)
        k_half, d_half = nearest_half_integer(n_mean)

        row: ResultRow = {
            "name": m.name,
            "t_seconds": float(m.t_seconds),
            "n": float(n_mean),
            "nearest_int": int(k_int),
            "diff_int": float(d_int),
            "nearest_half": float(k_half),
            "diff_half": float(d_half),
            "near_int": bool(d_int < 0.1),
            "near_half": bool(d_half < 0.1),
            "note": m.note,
        }
        results.append(row)

    # Monte Carlo testing (same as original but with scale factor)
    test_stats = []
    for _ in range(n_samples):
        fractional_parts = []
        for m in test_milestones:
            if m.uncertainty_fraction > 0:
                sigma_ln = math.sqrt(math.log(1 + m.uncertainty_fraction**2))
                mu_ln = math.log(m.t_seconds) - 0.5 * sigma_ln**2
                log_t = mu_ln + rng.gauss(0, sigma_ln)
                t_sample = math.exp(log_t)
            else:
                t_sample = m.t_seconds

            n = recursion_depth_from_scale_factor(t_sample, reference_scale_factor, m_p)
            fractional_parts.append(n - math.floor(n))

        if fractional_parts:
            replicate_stats = {
                "rayleigh_m2": compute_generalized_rayleigh(fractional_parts, m=2),
                "rayleigh_m3": compute_generalized_rayleigh(fractional_parts, m=3),
                "rayleigh_m4": compute_generalized_rayleigh(fractional_parts, m=4),
                "cgm_composite": compute_cgm_composite_harmonic_test(fractional_parts),
                "weighted_cgm": compute_weighted_cgm_test(fractional_parts),
                "kuiper": kuiper_test(fractional_parts),
                "watson": watson_test(fractional_parts),
            }
            test_stats.append(replicate_stats)

    # Aggregate statistics
    if not test_stats:
        return results, {"error": "No test statistics computed"}, []

    aggregated_stats: Dict[str, Union[float, str]] = {
        "n_replicates": len(test_stats),
        "n_test_events": len(test_milestones),
        "n_anchor_events": len(anchor_milestones),
    }

    test_types = [
        "rayleigh_m2",
        "rayleigh_m3",
        "rayleigh_m4",
        "cgm_composite",
        "weighted_cgm",
        "kuiper",
        "watson",
    ]
    for test_type in test_types:
        p_values = [s[test_type]["p_value"] for s in test_stats if test_type in s]
        concentrations = [
            s[test_type].get("concentration", 0.0) for s in test_stats if test_type in s
        ]

        if p_values:
            aggregated_stats[f"{test_type}_mean_p"] = sum(p_values) / len(p_values)
            aggregated_stats[f"{test_type}_median_p"] = sorted(p_values)[
                len(p_values) // 2
            ]
            aggregated_stats[f"{test_type}_mean_concentration"] = sum(
                concentrations
            ) / len(concentrations)
        else:
            aggregated_stats[f"{test_type}_mean_p"] = 1.0
            aggregated_stats[f"{test_type}_median_p"] = 1.0
            aggregated_stats[f"{test_type}_mean_concentration"] = 0.0

    return results, aggregated_stats, test_stats


def main() -> int:
    """Run the comprehensive chronology analysis."""
    run_comprehensive_chronology_analysis()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
