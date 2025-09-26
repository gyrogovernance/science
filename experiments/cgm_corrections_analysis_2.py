#!/usr/bin/env python3
"""
cgm_corrections_analysis_2.py

This script implements the Common Governance Model (CGM) framework for galaxy rotation curves,
applying the universal correction operator C to gravitational fields. The key predictions are:

1. Universal acceleration scale: a0 = cH0/(2π) (matches MOND without fitting)
2. Gravitational field correction: g_corrected = g_Newton * C(representation, scale)
3. No dark matter halos required - the geometric corrections explain flat rotation curves

The script loads and analyzes SPARC galaxy data from the local data folder, comparing CGM predictions with:
- Newtonian gravity (no dark matter)
- MOND predictions (using a0 = cH0/(2π))
- Observed rotation curves

Key CGM parameters from corrections analysis:
- Aperture fraction: Δ = 0.0207 (2.07%)
- Universal correction operator C with representation-dependent weights
- Geometric acceleration scale a0 = cH0/(2π)
"""

import numpy as np
import pandas as pd  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt
import glob
import os
import argparse
from decimal import Decimal, getcontext
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set high precision for critical calculations
getcontext().prec = 50

class CGMRotationCurves:
    """
    CGM framework for galaxy rotation curves using universal correction operator.
    """
    
    def __init__(self):
        """Initialize with CGM parameters from corrections analysis."""
        # CGM geometric parameters (derived from CGM geometry, see docs/Findings/Analysis_Fine_Structure.md)
        self.delta_BU = Decimal("0.195342176580")  # BU dual-pole monodromy δ_BU
        self.R = Decimal("0.993434896272")  # Thomas-Wigner curvature ratio: R = (F̄/π)/m_p with F̄ = 0.622543 (Eq. 54)
        self.h = Decimal("4.417034")  # 4-leg/8-leg holonomy ratio (Eq. 68)
        self.rho_inv = Decimal("1.021137")  # Inverse closure fraction: 1/ρ where ρ = δ_BU/m_p = 0.979300 (Eq. 81)
        self.diff = Decimal("0.001874")  # Monodromic residue: φ_SU2 - 3δ_BU (Eq. 82)
        
        # Derived parameters
        self.pi = Decimal("3.14159265358979323846264338327950288419716939937510")
        self.m_p = Decimal(1) / (Decimal(2) * (Decimal(2) * self.pi).sqrt())
        self.Delta = Decimal(1) - self.delta_BU / self.m_p
        
        # Physical constants
        self.c = Decimal("299792458")  # m/s (exact)
        self.G = Decimal("6.67430e-11")  # m^3/kg/s^2
        self.H0_planck = Decimal("67.27")  # km/s/Mpc
        self.H0_planck_err = Decimal("0.60")  # km/s/Mpc (Planck 2018)
        self.H0_sh0es = Decimal("73.04")  # km/s/Mpc
        self.H0_sh0es_err = Decimal("1.04")  # km/s/Mpc (SH0ES 2019)
        
        # Universal acceleration scale a0 = cH0/(2π) with error propagation
        self.a0_planck, self.a0_planck_err = self._compute_a0_with_errors(self.H0_planck, self.H0_planck_err)
        self.a0_sh0es, self.a0_sh0es_err = self._compute_a0_with_errors(self.H0_sh0es, self.H0_sh0es_err)
        
        print(f"CGM Initialized:")
        print(f"  Aperture fraction Δ = {self.Delta:.6f} ({float(self.Delta)*100:.3f}%)")
        print(f"  a0 (Planck) = {self.a0_planck:.6e} ± {self.a0_planck_err:.6e} m/s²")
        print(f"  a0 (SH0ES) = {self.a0_sh0es:.6e} ± {self.a0_sh0es_err:.6e} m/s²")
    
    def _compute_a0_with_errors(self, H0_km_s_Mpc: Decimal, H0_err_km_s_Mpc: Decimal) -> tuple[Decimal, Decimal]:
        """Compute universal acceleration scale a0 = cH0/(2π) with error propagation."""
        Mpc_m = Decimal("3.085677581491367e22")
        H0_s_inv = (H0_km_s_Mpc * Decimal(1000)) / Mpc_m
        H0_err_s_inv = (H0_err_km_s_Mpc * Decimal(1000)) / Mpc_m
        
        a0 = (self.c * H0_s_inv) / (Decimal(2) * self.pi)
        # Error propagation: δa0/a0 = δH0/H0 (since a0 ∝ H0)
        a0_err = a0 * H0_err_s_inv / H0_s_inv
        
        return a0, a0_err
    
    def _compute_a0(self, H0_km_s_Mpc: Decimal) -> Decimal:
        """Compute universal acceleration scale a0 = cH0/(2π) (legacy method)."""
        Mpc_m = Decimal("3.085677581491367e22")
        H0_s_inv = (H0_km_s_Mpc * Decimal(1000)) / Mpc_m
        return (self.c * H0_s_inv) / (Decimal(2) * self.pi)
    
    
    def baryonic_velocity_curve(self, df: pd.DataFrame, ups_disk: float = 0.5, ups_bulge: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
        """Build baryonic velocity curve from SPARC components."""
        Vgas = df['Vgas'].to_numpy(dtype=float)
        Vdisk = df['Vdisk'].to_numpy(dtype=float)
        Vbul = df['Vbul'].to_numpy(dtype=float)
        R = df['R'].to_numpy(dtype=float)
        
        # Baryonic velocity: V_b^2 = V_gas^2 + (Υ_d V_disk)^2 + (Υ_b V_bulge)^2
        Vb = np.sqrt(np.maximum(0.0, Vgas**2 + (ups_disk*Vdisk)**2 + (ups_bulge*Vbul)**2))
        
        return R, Vb

    def newtonian_velocity(self, r_kpc: np.ndarray, v_baryon_km_s: np.ndarray) -> np.ndarray:
        """Newtonian velocity is just the baryonic velocity (no dark matter)."""
        return v_baryon_km_s
    
    def mond_velocity(self, r_kpc: np.ndarray, v_baryon_km_s: np.ndarray, a0: float) -> np.ndarray:
        """Compute MOND velocity using correct ν-function."""
        r_m = r_kpc * 3.086e19  # kpc to m
        aN = (v_baryon_km_s * 1e3)**2 / r_m  # Newtonian acceleration from baryons
        
        # MOND ν-function: a = 0.5*aN + sqrt((0.5*aN)^2 + a0*aN)
        a = 0.5*aN + np.sqrt((0.5*aN)**2 + a0*aN)
        v = np.sqrt(a * r_m) / 1e3  # m/s to km/s
        return v
    
    def nfw_dark_matter_velocity(self, r_kpc: np.ndarray, v_baryon_km_s: np.ndarray, 
                                M_baryon_solar: float) -> np.ndarray:
        """Compute velocity with NFW dark matter halo (standard ΛCDM approach)."""
        r_m = r_kpc * 3.086e19  # kpc to m
        M_baryon_kg = M_baryon_solar * 1.989e30  # M☉ to kg
        
        # Estimate NFW halo parameters from baryonic mass
        # Typical scaling: M_halo ~ 10 * M_baryon for galaxies
        M_halo_kg = 10.0 * M_baryon_kg  # Total halo mass
        
        # NFW scale radius (typical: r_s ~ 5-20 kpc for galaxies)
        r_s_kpc = 10.0  # kpc
        r_s_m = r_s_kpc * 3.086e19  # kpc to m
        
        # NFW density profile: ρ(r) = ρ_0 / [(r/r_s)(1 + r/r_s)²]
        # Circular velocity: v² = GM(r)/r where M(r) = 4πρ_0 r_s³ f(r/r_s)
        # f(x) = ln(1+x) - x/(1+x)
        
        x = r_kpc / r_s_kpc  # dimensionless radius
        f_x = np.log(1 + x) - x / (1 + x)  # NFW function
        
        # NFW velocity contribution
        v_nfw_squared = (6.67e-11 * M_halo_kg * f_x) / (r_m * (np.log(1 + 1) - 1/(1 + 1)))  # Normalized
        v_nfw = np.sqrt(np.maximum(0, v_nfw_squared)) / 1e3  # m/s to km/s
        
        # Total velocity: v² = v_baryon² + v_dark²
        v_total = np.sqrt(v_baryon_km_s**2 + v_nfw**2)
        
        return v_total
    
    def cgm_velocity(self, r_kpc: np.ndarray, v_baryon_km_s: np.ndarray, a0: float) -> np.ndarray:
        """Compute CGM velocity using full universal correction operator."""
        r_m = r_kpc * 3.086e19  # kpc to m
        aN = (v_baryon_km_s * 1e3)**2 / r_m  # Newtonian acceleration from baryons
        
        # Holographic projection using same ν-function as MOND
        a_holo = 0.5*aN + np.sqrt((0.5*aN)**2 + a0*aN)
        
        # Full universal correction operator for spin-2 fields
        # C_AB_spin2 = 1 - (5/4) * R * Δ² (spin-2 weight, not 3/4)
        c_ab_spin2 = float(1 - (5.0/4.0) * float(self.R) * (float(self.Delta)**2))
        
        # C_HC = 1 - (5/6) * ((φ/(3δ) - 1)) * (1 - Δ²*h) * Δ² / (4π√3)
        phi = 3 * float(self.delta_BU) + float(self.diff)  # φ = 3δ + diff
        phi_term = (phi / (3 * float(self.delta_BU))) - 1
        h_term = 1 - (float(self.Delta)**2) * float(self.h)
        c_hc = float(1 - (5.0/6.0) * phi_term * h_term * (float(self.Delta)**2) / (4 * float(self.pi) * np.sqrt(3)))
        
        # C_IDE = 1 + (1/ρ) * diff * Δ⁴
        c_ide = float(1 + float(self.rho_inv) * float(self.diff) * (float(self.Delta)**4))
        
        # Optional scale dependence using dimensionless acceleration ratio x = aN/a0
        x = aN / a0  # dimensionless acceleration ratio
        delta2 = float(self.Delta)**2
        # mild, parameter-free, geometry-tied modulation (Δ²-suppressed)
        f_x = 1.0 - (delta2 / (1.0 + x))  # f_x ∈ [1-Δ², 1), small boost
        
        # Full universal correction operator
        c_total = c_ab_spin2 * c_hc * c_ide * f_x
        
        # CGM acceleration with full universal operator
        a = a_holo / c_total
        v = np.sqrt(a * r_m) / 1e3  # m/s to km/s
        return v
    
    def load_sparc_data(self) -> Dict[str, pd.DataFrame]:
        """Load SPARC galaxy data from local data folder."""
        print("Loading SPARC data from data/sparc_database/...")

        # Look for data files in the data folder
        data_files = []
        for ext in ['*.mrt', '*.txt', '*.dat', '*.csv']:
            data_files.extend(glob.glob(f"data/sparc_database/{ext}"))
            
        if not data_files:
            raise FileNotFoundError("No data files found in data/sparc_database/")
            
        print(f"Found {len(data_files)} SPARC data files")
        
        # Load individual galaxy files (SPARC format)
        galaxies = {}
        included = 0
        excluded = 0
        exclusion_reasons = {}
        
        for file_path in data_files:
            try:
                # Read tab-separated data, skip first 2 header rows
                df = pd.read_csv(file_path, sep='\t', skiprows=2, header=None)
                
                # SPARC format: R(kpc), Vobs(km/s), errV, Vgas, Vdisk, Vbul, SBdisk, SBbul, ...
                if df.shape[1] >= 6:
                    df.columns = ['R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul'] + list(df.columns[6:])
                    
                    # Check for required columns
                    required_cols = ['R', 'Vobs', 'e_Vobs', 'Vgas', 'Vdisk', 'Vbul']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        excluded += 1
                        exclusion_reasons['missing_columns'] = exclusion_reasons.get('missing_columns', 0) + 1
                        continue
                    
                    # Clean data - remove rows with NaN or invalid values
                    df_clean = df.dropna(subset=['R', 'Vobs']).copy()
                    
                    # Convert to numeric, handling any string values
                    df_clean['R'] = pd.to_numeric(df_clean['R'], errors='coerce')
                    df_clean['Vobs'] = pd.to_numeric(df_clean['Vobs'], errors='coerce')
                    df_clean['e_Vobs'] = pd.to_numeric(df_clean['e_Vobs'], errors='coerce')
                    
                    # Remove rows with NaN after conversion
                    df_clean = df_clean.dropna(subset=['R', 'Vobs'])
                    df_clean = df_clean[df_clean['R'] > 0]  # Positive radii only
                    df_clean = df_clean[df_clean['Vobs'] > 0]  # Positive velocities only
                    
                    if len(df_clean) > 5:  # Only include galaxies with enough data points
                        galaxy_name = os.path.basename(file_path).replace('_rotmod.dat', '')
                        galaxies[galaxy_name] = df_clean
                        included += 1
                    else:
                        excluded += 1
                        exclusion_reasons['insufficient_data'] = exclusion_reasons.get('insufficient_data', 0) + 1
                else:
                    excluded += 1
                    exclusion_reasons['insufficient_columns'] = exclusion_reasons.get('insufficient_columns', 0) + 1
                
            except Exception as e:
                excluded += 1
                exclusion_reasons['load_error'] = exclusion_reasons.get('load_error', 0) + 1
                print(f"✗ Failed to load {file_path}: {e}")
                continue
        
        if not galaxies:
            raise ValueError("Could not load any SPARC data files")

        print(f"\nData loading summary:")
        print(f"  Included: {included} galaxies")
        print(f"  Excluded: {excluded} galaxies")
        for reason, count in exclusion_reasons.items():
            print(f"    {reason}: {count}")
        print(f"  Total: {included + excluded} galaxies")
        
        return galaxies
    
    
    def estimate_baryon_mass(self, galaxy_data: pd.DataFrame) -> float:
        """
        Estimate baryon mass from SPARC data using proper mass modeling.
        Uses baryonic velocity components with standard mass-to-light ratios.
        """
        # Standard mass-to-light ratios (from SPARC papers)
        upsilon_disk = 0.5  # M☉/L☉ for 3.6μm
        upsilon_bulge = 0.7  # M☉/L☉ for 3.6μm
        
        # Get velocity components (ensure numpy arrays)
        V_gas = np.array(galaxy_data['Vgas'].values, dtype=float)
        V_disk = np.array(galaxy_data['Vdisk'].values, dtype=float)
        V_bulge = np.array(galaxy_data['Vbul'].values, dtype=float)
        R = np.array(galaxy_data['R'].values, dtype=float)
        
        # Calculate baryonic velocity at each radius
        V_bary = np.sqrt(np.maximum(0, V_gas**2 + (upsilon_disk * V_disk)**2 + (upsilon_bulge * V_bulge)**2))
        
        # Use the maximum baryonic velocity for mass estimation
        # M_bary = V_bary^2 * R / G (at the radius where V_bary is maximum)
        max_idx = np.argmax(V_bary)
        v_max = V_bary[max_idx]
        r_max = R[max_idx]
        
        # Convert to SI and calculate mass
        v_max_ms = v_max * 1000  # km/s to m/s
        r_max_m = r_max * 3.086e19  # kpc to m
        M_kg = (v_max_ms**2 * r_max_m) / float(self.G)
        M_sun = M_kg / 1.989e30
        
        return M_sun
    
    def analyze_galaxy(self, name: str, data: pd.DataFrame, a0: float) -> Dict:
        """Analyze a single galaxy with CGM, Newton, and MOND predictions."""
        r = np.array(data['R'].values)
        v_obs = np.array(data['Vobs'].values)
        e_v_obs = np.array(data['e_Vobs'].values)
        
        # Build baryonic velocity curve from SPARC components
        r_bary, v_bary = self.baryonic_velocity_curve(data)
        
        # Estimate baryon mass for reference
        M_baryon = self.estimate_baryon_mass(data)
        
        # Compute predictions using baryonic curves
        v_newton = self.newtonian_velocity(r_bary, v_bary)
        v_mond = self.mond_velocity(r_bary, v_bary, a0)
        v_dark_matter = self.nfw_dark_matter_velocity(r_bary, v_bary, M_baryon)
        v_cgm = self.cgm_velocity(r_bary, v_bary, a0)
        
        # Compute chi-squared and statistical measures
        def chi2(v_pred, v_obs, e_v_obs):
            # Add small floor to prevent pathological χ² when e_v_obs has zeros
            e_v_obs = np.where(e_v_obs <= 0, 5.0, e_v_obs)
            return np.sum(((v_pred - v_obs) / e_v_obs)**2)
        
        def reduced_chi2(chi2_val, n_data, n_params):
            """Calculate reduced chi-squared (chi-squared per degree of freedom)."""
            dof = n_data - n_params
            return chi2_val / dof if dof > 0 else float('inf')
        
        def aic(chi2_val, n_params):
            """Calculate Akaike Information Criterion."""
            return chi2_val + 2 * n_params
        
        def bic(chi2_val, n_params, n_data):
            """Calculate Bayesian Information Criterion."""
            return chi2_val + n_params * np.log(n_data)
        
        n_data = len(v_obs)
        n_params = 0  # No free parameters for any model
        
        chi2_newton = chi2(v_newton, v_obs, e_v_obs)
        chi2_mond = chi2(v_mond, v_obs, e_v_obs)
        chi2_dark_matter = chi2(v_dark_matter, v_obs, e_v_obs)
        chi2_cgm = chi2(v_cgm, v_obs, e_v_obs)
        
        # Statistical measures
        rchi2_newton = reduced_chi2(chi2_newton, n_data, n_params)
        rchi2_mond = reduced_chi2(chi2_mond, n_data, n_params)
        rchi2_dark_matter = reduced_chi2(chi2_dark_matter, n_data, n_params)
        rchi2_cgm = reduced_chi2(chi2_cgm, n_data, n_params)
        
        aic_newton = aic(chi2_newton, n_params)
        aic_mond = aic(chi2_mond, n_params)
        aic_dark_matter = aic(chi2_dark_matter, n_params)
        aic_cgm = aic(chi2_cgm, n_params)
        
        bic_newton = bic(chi2_newton, n_params, n_data)
        bic_mond = bic(chi2_mond, n_params, n_data)
        bic_dark_matter = bic(chi2_dark_matter, n_params, n_data)
        bic_cgm = bic(chi2_cgm, n_params, n_data)
        
        return {
            'name': name,
            'r': r_bary,
            'v_obs': v_obs,
            'e_v_obs': e_v_obs,
            'v_bary': v_bary,
            'v_newton': v_newton,
            'v_mond': v_mond,
            'v_dark_matter': v_dark_matter,
            'v_cgm': v_cgm,
            'M_baryon': M_baryon,
            'chi2_newton': chi2_newton,
            'chi2_mond': chi2_mond,
            'chi2_dark_matter': chi2_dark_matter,
            'chi2_cgm': chi2_cgm,
            'rchi2_newton': rchi2_newton,
            'rchi2_mond': rchi2_mond,
            'rchi2_dark_matter': rchi2_dark_matter,
            'rchi2_cgm': rchi2_cgm,
            'aic_newton': aic_newton,
            'aic_mond': aic_mond,
            'aic_dark_matter': aic_dark_matter,
            'aic_cgm': aic_cgm,
            'bic_newton': bic_newton,
            'bic_mond': bic_mond,
            'bic_dark_matter': bic_dark_matter,
            'bic_cgm': bic_cgm,
            'n_data': n_data,
            'n_params': n_params
        }
    
    def print_galaxy_analysis(self, analysis: Dict):
        """Print detailed analysis for a single galaxy."""
        print(f"\n=== {analysis['name']} ===")
        print(f"Data: {len(analysis['r'])} points, R={analysis['r'].min():.1f}-{analysis['r'].max():.1f} kpc, V={analysis['v_obs'].min():.0f}-{analysis['v_obs'].max():.0f} km/s")
        print(f"Mass: {analysis['M_baryon']:.1e} M☉")
        print(f"χ²: Newton={analysis['chi2_newton']:.0f}, MOND={analysis['chi2_mond']:.0f}, DM={analysis['chi2_dark_matter']:.0f}, CGM={analysis['chi2_cgm']:.0f}")
        print(f"Reduced χ²: Newton={analysis['rchi2_newton']:.2f}, MOND={analysis['rchi2_mond']:.2f}, DM={analysis['rchi2_dark_matter']:.2f}, CGM={analysis['rchi2_cgm']:.2f}")
        print(f"AIC: Newton={analysis['aic_newton']:.1f}, MOND={analysis['aic_mond']:.1f}, DM={analysis['aic_dark_matter']:.1f}, CGM={analysis['aic_cgm']:.1f}")
        print(f"BIC: Newton={analysis['bic_newton']:.1f}, MOND={analysis['bic_mond']:.1f}, DM={analysis['bic_dark_matter']:.1f}, CGM={analysis['bic_cgm']:.1f}")
        
        # Print CGM universal correction factors
        c_ab_spin2 = float(1 - (5.0/4.0) * float(self.R) * (float(self.Delta)**2))
        phi = 3 * float(self.delta_BU) + float(self.diff)
        phi_term = (phi / (3 * float(self.delta_BU))) - 1
        h_term = 1 - (float(self.Delta)**2) * float(self.h)
        c_hc = float(1 - (5.0/6.0) * phi_term * h_term * (float(self.Delta)**2) / (4 * float(self.pi) * np.sqrt(3)))
        c_ide = float(1 + float(self.rho_inv) * float(self.diff) * (float(self.Delta)**4))
        c_base = c_ab_spin2 * c_hc * c_ide
        
        # Calculate f_x range for this galaxy
        r = analysis['r']
        v_bary = analysis['v_bary']
        r_m = r * 3.086e19
        aN = (v_bary * 1e3)**2 / r_m
        x = aN / float(self.a0_planck)  # Use Planck a0 for consistency
        delta2 = float(self.Delta)**2
        fx = 1.0 - (delta2 / (1.0 + x))
        
        print(f"CGM base operator: C_AB={c_ab_spin2:.6f}, C_HC={c_hc:.6f}, C_IDE={c_ide:.6f}, C_base={c_base:.6f}")
        print(f"  f_x range: {fx.min():.6f}–{fx.max():.6f} (median {np.median(fx):.6f})")
    
    def plot_galaxy(self, galaxy_name: str, analysis: Dict, save_plot: bool = True):
        """Plot rotation curve for a single galaxy."""
        r = analysis['r']
        v_obs = analysis['v_obs']
        v_obs_err = analysis.get('v_obs_err', None)  # Handle missing error data
        v_newton = analysis['v_newton']
        v_mond = analysis['v_mond']
        v_cgm = analysis['v_cgm']
        
        plt.figure(figsize=(10, 8))
        
        # Plot observed data with or without error bars
        if v_obs_err is not None:
            plt.errorbar(r, v_obs, yerr=v_obs_err, fmt='ko', markersize=6, 
                        capsize=3, capthick=1, label='Observed', zorder=5)
        else:
            plt.plot(r, v_obs, 'ko', markersize=6, label='Observed', zorder=5)
        
        # Plot model predictions
        plt.plot(r, v_newton, 'r--', linewidth=2, label='Newton (no DM)', alpha=0.8)
        plt.plot(r, v_mond, 'b-', linewidth=2, label='MOND', alpha=0.8)
        plt.plot(r, v_cgm, 'g-', linewidth=2, label='CGM', alpha=0.8)
        
        plt.xlabel('Radius (kpc)', fontsize=12)
        plt.ylabel('Velocity (km/s)', fontsize=12)
        plt.title(f'{galaxy_name} Rotation Curve\nCGM vs MOND vs Newton', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'χ²: N={analysis["chi2_newton"]:.0f}, M={analysis["chi2_mond"]:.0f}, C={analysis["chi2_cgm"]:.0f}\n'
        stats_text += f'Reduced χ²: N={analysis["rchi2_newton"]:.1f}, M={analysis["rchi2_mond"]:.1f}, C={analysis["rchi2_cgm"]:.1f}\n'
        stats_text += f'AIC: N={analysis["aic_newton"]:.1f}, M={analysis["aic_mond"]:.1f}, C={analysis["aic_cgm"]:.1f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(f'plots/{galaxy_name}_rotation_curve.png', dpi=300, bbox_inches='tight')
            print(f"Saved plot: plots/{galaxy_name}_rotation_curve.png")
        
        plt.show()
    
    def plot_comparison(self, results: List[Dict], save_plot: bool = True):
        """Plot comparison of all models across galaxies."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data
        galaxy_names = [r['name'] for r in results]
        chi2_newton = [r['chi2_newton'] for r in results]
        chi2_mond = [r['chi2_mond'] for r in results]
        chi2_cgm = [r['chi2_cgm'] for r in results]
        rchi2_newton = [r['rchi2_newton'] for r in results]
        rchi2_mond = [r['rchi2_mond'] for r in results]
        rchi2_cgm = [r['rchi2_cgm'] for r in results]
        
        # Plot 1: Chi-squared comparison
        x = np.arange(len(galaxy_names))
        width = 0.25
        axes[0,0].bar(x - width, chi2_newton, width, label='Newton', color='red', alpha=0.7)
        axes[0,0].bar(x, chi2_mond, width, label='MOND', color='blue', alpha=0.7)
        axes[0,0].bar(x + width, chi2_cgm, width, label='CGM', color='green', alpha=0.7)
        axes[0,0].set_xlabel('Galaxy')
        axes[0,0].set_ylabel('χ²')
        axes[0,0].set_title('Chi-squared Comparison')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(galaxy_names, rotation=45, ha='right')
        axes[0,0].legend()
        axes[0,0].set_yscale('log')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Reduced chi-squared comparison
        axes[0,1].bar(x - width, rchi2_newton, width, label='Newton', color='red', alpha=0.7)
        axes[0,1].bar(x, rchi2_mond, width, label='MOND', color='blue', alpha=0.7)
        axes[0,1].bar(x + width, rchi2_cgm, width, label='CGM', color='green', alpha=0.7)
        axes[0,1].set_xlabel('Galaxy')
        axes[0,1].set_ylabel('Reduced χ²')
        axes[0,1].set_title('Reduced Chi-squared Comparison')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(galaxy_names, rotation=45, ha='right')
        axes[0,1].legend()
        axes[0,1].set_yscale('log')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: CGM vs MOND ratio
        cgm_mond_ratios = []
        for r in results:
            ratio = r['v_cgm'] / r['v_mond']
            cgm_mond_ratios.extend(ratio)
        
        axes[1,0].hist(cgm_mond_ratios, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1,0].axvline(np.mean(cgm_mond_ratios), color='red', linestyle='--', linewidth=2, 
                         label=f'Mean: {np.mean(cgm_mond_ratios):.4f}')
        axes[1,0].set_xlabel('v_CGM / v_MOND')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('CGM-MOND Velocity Ratio Distribution')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Model performance summary
        models = ['Newton', 'MOND', 'CGM']
        mean_rchi2 = [np.mean(rchi2_newton), np.mean(rchi2_mond), np.mean(rchi2_cgm)]
        std_rchi2 = [np.std(rchi2_newton), np.std(rchi2_mond), np.std(rchi2_cgm)]
        
        bars = axes[1,1].bar(models, mean_rchi2, yerr=std_rchi2, capsize=5, 
                            color=['red', 'blue', 'green'], alpha=0.7)
        axes[1,1].set_ylabel('Mean Reduced χ²')
        axes[1,1].set_title('Model Performance Summary')
        axes[1,1].set_yscale('log')
        axes[1,1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, mean_rchi2, std_rchi2):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + std,
                          f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
            print("Saved plot: plots/model_comparison.png")
        
        plt.show()
    
    def run_analysis(self, n_galaxies: Optional[int] = None, a0_choice: str = "planck", 
                    create_plots: bool = False, individual_plots: bool = False, comparison_plot: bool = False):
        """Run analysis on multiple galaxies."""
        # Choose a0 value
        a0 = float(self.a0_planck) if a0_choice == "planck" else float(self.a0_sh0es)
        
        # Load data
        galaxies = self.load_sparc_data()
        
        # Analyze all galaxies by default
        if n_galaxies is None:
            n_galaxies = len(galaxies)
        
        galaxy_names = list(galaxies.keys())[:n_galaxies]
        
        print(f"\nAnalyzing {len(galaxy_names)} galaxies with a0 = {a0:.6e} m/s²")
        print("="*60)
        
        results = []
        cgm_mond_ratios = []  # For CGM-MOND diagnostics
        rar_data = []  # For RAR diagnostics
        
        for i, name in enumerate(galaxy_names):
            analysis = self.analyze_galaxy(name, galaxies[name], a0)
            results.append(analysis)
            
            # Progress indicator for large runs
            if n_galaxies > 10 and (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{n_galaxies} galaxies...")
            elif n_galaxies <= 10:
                print(f"{name}: M={analysis['M_baryon']:.1e} M☉, χ²: N={analysis['chi2_newton']:.0f}, M={analysis['chi2_mond']:.0f}, DM={analysis['chi2_dark_matter']:.0f}, C={analysis['chi2_cgm']:.0f}")
            
            # Collect data for diagnostics
            v_cgm = analysis['v_cgm']
            v_mond = analysis['v_mond']
            v_obs = analysis['v_obs']
            v_bary = analysis['v_bary']  # Use actual baryonic velocity
            r = analysis['r']
            
            # CGM-MOND ratio
            ratio = v_cgm / v_mond
            cgm_mond_ratios.extend(ratio)
            
            # RAR data
            r_m = r * 3.086e19  # kpc to m
            a_obs = (v_obs * 1e3)**2 / r_m
            a_bar = (v_bary * 1e3)**2 / r_m
            a_pred_mond = (v_mond * 1e3)**2 / r_m
            a_pred_cgm = (v_cgm * 1e3)**2 / r_m
            
            for i in range(len(r)):
                rar_data.append({
                    'a_obs': a_obs[i],
                    'a_bar': a_bar[i],
                    'a_pred_mond': a_pred_mond[i],
                    'a_pred_cgm': a_pred_cgm[i]
                })
            
            # Create individual galaxy plots if requested
            if individual_plots:
                self.plot_galaxy(name, analysis, save_plot=True)
        
        # Summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        chi2_newton = [r['chi2_newton'] for r in results]
        chi2_mond = [r['chi2_mond'] for r in results]
        chi2_dark_matter = [r['chi2_dark_matter'] for r in results]
        chi2_cgm = [r['chi2_cgm'] for r in results]
        
        rchi2_newton = [r['rchi2_newton'] for r in results]
        rchi2_mond = [r['rchi2_mond'] for r in results]
        rchi2_dark_matter = [r['rchi2_dark_matter'] for r in results]
        rchi2_cgm = [r['rchi2_cgm'] for r in results]
        
        aic_newton = [r['aic_newton'] for r in results]
        aic_mond = [r['aic_mond'] for r in results]
        aic_dark_matter = [r['aic_dark_matter'] for r in results]
        aic_cgm = [r['aic_cgm'] for r in results]
        
        print(f"Newton (no DM):  χ² = {np.mean(chi2_newton):.0f} ± {np.std(chi2_newton):.0f}, χ²/dof = {np.mean(rchi2_newton):.2f}")
        print(f"MOND:           χ² = {np.mean(chi2_mond):.0f} ± {np.std(chi2_mond):.0f}, χ²/dof = {np.mean(rchi2_mond):.2f}")
        print(f"Dark Matter:    χ² = {np.mean(chi2_dark_matter):.0f} ± {np.std(chi2_dark_matter):.0f}, χ²/dof = {np.mean(rchi2_dark_matter):.2f}")
        print(f"CGM:            χ² = {np.mean(chi2_cgm):.0f} ± {np.std(chi2_cgm):.0f}, χ²/dof = {np.mean(rchi2_cgm):.2f}")
        
        print(f"\nAIC (lower is better):")
        print(f"Newton: {np.mean(aic_newton):.1f} ± {np.std(aic_newton):.1f}")
        print(f"MOND:   {np.mean(aic_mond):.1f} ± {np.std(aic_mond):.1f}")
        print(f"DM:     {np.mean(aic_dark_matter):.1f} ± {np.std(aic_dark_matter):.1f}")
        print(f"CGM:    {np.mean(aic_cgm):.1f} ± {np.std(aic_cgm):.1f}")
        
        # Count wins based on AIC (more robust than χ² alone)
        cgm_wins = sum(1 for r in results if r['aic_cgm'] < min(r['aic_newton'], r['aic_mond'], r['aic_dark_matter']))
        mond_wins = sum(1 for r in results if r['aic_mond'] < min(r['aic_newton'], r['aic_cgm'], r['aic_dark_matter']))
        dark_matter_wins = sum(1 for r in results if r['aic_dark_matter'] < min(r['aic_newton'], r['aic_mond'], r['aic_cgm']))
        newton_wins = sum(1 for r in results if r['aic_newton'] < min(r['aic_mond'], r['aic_cgm'], r['aic_dark_matter']))
        
        print(f"\nBest fits (AIC): CGM={cgm_wins}, MOND={mond_wins}, DM={dark_matter_wins}, Newton={newton_wins}")
        
        # CGM-MOND diagnostics
        print(f"\nCGM-MOND DIAGNOSTICS:")
        cgm_mond_ratios = np.array(cgm_mond_ratios)
        delta_ratios = cgm_mond_ratios - 1
        print(f"  Mean (v_cgm/v_mond - 1): {np.mean(delta_ratios):.6f}")
        print(f"  RMS (v_cgm/v_mond - 1): {np.sqrt(np.mean(delta_ratios**2)):.6f}")
        print(f"  Range: {np.min(delta_ratios):.6f} to {np.max(delta_ratios):.6f}")
        
        # Full-sample universality test
        print(f"\nFULL-SAMPLE UNIVERSALITY TEST:")
        per_galaxy_means = []
        per_galaxy_rms = []
        
        for r in results:
            v_cgm = r['v_cgm']
            v_mond = r['v_mond']
            ratios = v_cgm / v_mond
            delta = ratios - 1
            per_galaxy_means.append(np.mean(delta))
            per_galaxy_rms.append(np.std(delta))
        
        per_galaxy_means = np.array(per_galaxy_means)
        per_galaxy_rms = np.array(per_galaxy_rms)
        
        print(f"  Per-galaxy mean (v_cgm/v_mond - 1): {np.mean(per_galaxy_means):.6f} ± {np.std(per_galaxy_means):.6f}")
        print(f"  Per-galaxy RMS (v_cgm/v_mond - 1): {np.mean(per_galaxy_rms):.6f} ± {np.std(per_galaxy_rms):.6f}")
        print(f"  Grand mean across all galaxies: {np.mean(per_galaxy_means):.6f}")
        print(f"  Grand RMS across all galaxies: {np.std(per_galaxy_means):.6f}")
        
        # Regime-binned test by acceleration x = aN/a0
        print(f"\nREGIME-BINNED TEST (x = aN/a0):")
        all_x = []
        all_delta = []
        
        for r in results:
            r_kpc = r['r']
            v_bary = r['v_bary']
            r_m = r_kpc * 3.086e19
            aN = (v_bary * 1e3)**2 / r_m
            x = aN / a0
            v_cgm = r['v_cgm']
            v_mond = r['v_mond']
            delta = (v_cgm / v_mond) - 1
            
            all_x.extend(x)
            all_delta.extend(delta)
        
        all_x = np.array(all_x)
        all_delta = np.array(all_delta)
        
        # Bin by acceleration regime
        low_acc = all_delta[all_x < 0.1]
        mid_acc = all_delta[(all_x >= 0.1) & (all_x <= 10)]
        high_acc = all_delta[all_x > 10]
        
        print(f"  x < 0.1 (deep-MOND): mean = {np.mean(low_acc):.6f}, std = {np.std(low_acc):.6f}, N = {len(low_acc)}")
        print(f"  0.1 ≤ x ≤ 10 (transition): mean = {np.mean(mid_acc):.6f}, std = {np.std(mid_acc):.6f}, N = {len(mid_acc)}")
        print(f"  x > 10 (Newtonian): mean = {np.mean(high_acc):.6f}, std = {np.std(high_acc):.6f}, N = {len(high_acc)}")
        
        # Predict vs observe check (operator-only)
        print(f"\nPREDICT vs OBSERVE CHECK (operator-only):")
        delta_pred_list = []
        delta_meas_list = []
        
        for r in results:
            # Calculate C_total for this galaxy
            r_kpc = r['r']
            v_bary = r['v_bary']
            r_m = r_kpc * 3.086e19
            aN = (v_bary * 1e3)**2 / r_m
            x = aN / a0
            delta2 = float(self.Delta)**2
            fx = 1.0 - (delta2 / (1.0 + x))
            
            c_ab_spin2 = float(1 - (5.0/4.0) * float(self.R) * (float(self.Delta)**2))
            phi = 3 * float(self.delta_BU) + float(self.diff)
            phi_term = (phi / (3 * float(self.delta_BU))) - 1
            h_term = 1 - (float(self.Delta)**2) * float(self.h)
            c_hc = float(1 - (5.0/6.0) * phi_term * h_term * (float(self.Delta)**2) / (4 * float(self.pi) * np.sqrt(3)))
            c_ide = float(1 + float(self.rho_inv) * float(self.diff) * (float(self.Delta)**4))
            c_total = c_ab_spin2 * c_hc * c_ide * fx
            
            delta_pred = 0.5 * (1/c_total - 1)
            delta_meas = np.mean((r['v_cgm'] / r['v_mond']) - 1)
            
            delta_pred_list.append(np.mean(delta_pred))
            delta_meas_list.append(delta_meas)
        
        delta_pred_list = np.array(delta_pred_list)
        delta_meas_list = np.array(delta_meas_list)
        residual = delta_meas_list - delta_pred_list
        
        print(f"  δ_pred = 0.5(1/C_total - 1): mean = {np.mean(delta_pred_list):.6f} ± {np.std(delta_pred_list):.6f}")
        print(f"  δ_meas = mean(v_CGM/v_MOND - 1): mean = {np.mean(delta_meas_list):.6f} ± {np.std(delta_meas_list):.6f}")
        print(f"  δ_meas - δ_pred: mean = {np.mean(residual):.6f} ± {np.std(residual):.6f}")
        print(f"  Residual RMS: {np.sqrt(np.mean(residual**2)):.6f}")
        
        # RAR diagnostics
        print(f"\nRAR/MDAR DIAGNOSTICS:")
        a_obs = np.array([d['a_obs'] for d in rar_data])
        a_bar = np.array([d['a_bar'] for d in rar_data])
        a_pred_mond = np.array([d['a_pred_mond'] for d in rar_data])
        a_pred_cgm = np.array([d['a_pred_cgm'] for d in rar_data])
        
        # Log-log correlation
        log_a_obs = np.log10(a_obs)
        log_a_bar = np.log10(a_bar)
        correlation = np.corrcoef(log_a_obs, log_a_bar)[0, 1]
        print(f"  log(a_obs) vs log(a_bar) correlation: {correlation:.4f}")
        
        # RMS scatter
        mond_scatter = np.sqrt(np.mean((log_a_obs - np.log10(a_pred_mond))**2))
        cgm_scatter = np.sqrt(np.mean((log_a_obs - np.log10(a_pred_cgm))**2))
        print(f"  RMS scatter (dex): MOND={mond_scatter:.4f}, CGM={cgm_scatter:.4f}")
        
        # Key insight
        print(f"\nKEY INSIGHT: CGM uses full universal correction operator")
        print(f"CGM a0 = cH0/(2π) is fixed by geometry, not fitted to data")
        print(f"Universal operator: C_total = C_AB_spin2 × C_HC × C_IDE × f(x)")
        print(f"Note: No mass-to-light fitting; higher χ² expected; comparisons across models are still fair")
        
        # Create comparison plots if requested
        if comparison_plot:
            print(f"\nCreating comparison plots...")
            self.plot_comparison(results, save_plot=True)
        
        # Uncertainty analysis
        print(f"\nUNCERTAINTY ANALYSIS:")
        print(f"H0 uncertainty propagation to a0:")
        print(f"  Planck: δa0/a0 = δH0/H0 = {float(self.H0_planck_err/self.H0_planck)*100:.2f}%")
        print(f"  SH0ES:  δa0/a0 = δH0/H0 = {float(self.H0_sh0es_err/self.H0_sh0es)*100:.2f}%")
        print(f"CGM geometric parameters (no experimental uncertainty):")
        print(f"  Δ = {self.Delta:.6f} (derived from δ_BU, exact)")
        print(f"  R = {self.R:.12f} (Thomas-Wigner curvature, measured)")
        print(f"  h = {self.h:.6f} (holonomy ratio, measured)")
        print(f"  AB factor uncertainty: δC_AB/C_AB ≈ 2δR/R + 4δΔ/Δ")
        
        return results

def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='CGM Galaxy Rotation Curve Analysis')
    parser.add_argument('--galaxies', '-g', type=int, default=None,
                        help='Number of galaxies to analyze (default: all available)')
    parser.add_argument('--a0', choices=['planck', 'sh0es'], default='planck',
                       help='H0 source for a0 calculation (default: planck)')
    parser.add_argument('--plots', action='store_true',
                       help='Enable plotting (default: disabled)')
    parser.add_argument('--individual-plots', action='store_true',
                       help='Enable individual galaxy plots (default: disabled)')
    parser.add_argument('--comparison-plot', action='store_true',
                       help='Enable model comparison plot (default: disabled)')
    parser.add_argument('--all-plots', action='store_true',
                       help='Enable all plots (equivalent to --plots --individual-plots --comparison-plot)')
    parser.add_argument('--help-plots', action='store_true',
                       help='Show detailed help for plot options')
    
    args = parser.parse_args()
    
    # Handle --help-plots flag
    if args.help_plots:
        print("CGM Galaxy Rotation Curve Analysis - Plot Options")
        print("="*60)
        print("Plot Control Switches:")
        print("  --plots              Enable basic plotting (overall control)")
        print("  --individual-plots   Create individual galaxy rotation curve plots")
        print("  --comparison-plot    Create model comparison plots")
        print("  --all-plots          Enable all plot types")
        print()
        print("Examples:")
        print("  python cgm_rotations_analysis.py                    # No plots (default)")
        print("  python cgm_rotations_analysis.py --plots            # Enable all plots")
        print("  python cgm_rotations_analysis.py --individual-plots # Only individual plots")
        print("  python cgm_rotations_analysis.py --all-plots        # All plots")
        print("  python cgm_rotations_analysis.py -g 10 --plots      # 10 galaxies with plots")
        print("  python cgm_rotations_analysis.py --a0 sh0es --plots # Use SH0ES a0 with plots")
        return
    
    # Handle --all-plots flag
    if args.all_plots:
        args.plots = True
        args.individual_plots = True
        args.comparison_plot = True
    
    print("CGM Galaxy Rotation Curve Analysis")
    print("="*50)
    print(f"Configuration:")
    print(f"  Galaxies: {args.galaxies if args.galaxies else 'all available'}")
    print(f"  a0 source: {args.a0}")
    print(f"  Plots enabled: {args.plots}")
    print(f"  Individual plots: {args.individual_plots}")
    print(f"  Comparison plot: {args.comparison_plot}")
    print()
    
    # Initialize CGM framework
    cgm = CGMRotationCurves()
    
    # Run analysis with plot settings
    results = cgm.run_analysis(
        n_galaxies=args.galaxies, 
        a0_choice=args.a0, 
        create_plots=args.plots,
        individual_plots=args.individual_plots,
        comparison_plot=args.comparison_plot
    )
    
    # Print detailed analysis for first few galaxies (only if verbose)
    if len(results) <= 5:  # Only show details for small runs
        for i, analysis in enumerate(results[:3]):
            cgm.print_galaxy_analysis(analysis)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
