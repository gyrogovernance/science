import numpy as np
import math

# Import AliCE core function from main script
import sys
sys.path.append('.')
from cgm_AliCE import compute_chi

print("\nPHYSICS VALIDATION: Fundamental Constant Identities")
print("Testing exact algebraic identities using fundamental constants")
print()

print("Experiment A: Balance Index Identities")

# Constants (SI)
G = 6.67430e-11        # m^3 kg^-1 s^-2
c = 299792458.0        # m/s
k_B = 1.380649e-23     # J/K
pi = math.pi

# Two H0 values (s^-1)
H0_planck = 67.27 * 1000 / (3.085677581491367e22)   # Planck 2020: km/s/Mpc -> s^-1
H0_sh0es  = 73.04 * 1000 / (3.085677581491367e22)   # SH0ES 2022

def balance_index(H0):
    return 2*G*c/(k_B*H0)  # m^2 K kg^-2

def identities(H0):
    Bi = balance_index(H0)
    I1 = (k_B*Bi*H0)/(G*c)         # should be 2
    LB = (k_B*Bi)/G                # should be 2c/H0
    return Bi, I1, LB

for name, H0 in [("Planck", H0_planck), ("SH0ES", H0_sh0es)]:
    Bi, I1, LB = identities(H0)
    print(f"  {name}: B_i = {Bi:.6e} m^2·K·kg^-2")
    print(f"         I1 = {I1:.12f}")
    print(f"         L_B = {LB:.6e} m")
    print(f"         2c/H0 = {(2*c/H0):.6e} m")
    print(f"         |L_B - 2c/H0| / 2c/H0 = {abs(LB - 2*c/H0)/(2*c/H0):.6e}")

# Aperture-corrected invariant
 m_a = 1.0/(2.0*math.sqrt(2.0*pi))
I1_cgm = 2.0/((1.0+m_a)**2)
print(f"\n  Aperture-corrected I1_CGM = {I1_cgm:.12f}")

 
print("Experiment B: Universe Schwarzschild Threshold")
  

def universe_ratio(H0):
    rho_c = 3*H0**2/(8*pi*G)      # kg/m^3
    R_H   = c/H0                  # m
    V_H   = (4.0/3.0)*pi*R_H**3   # m^3
    M     = rho_c * V_H           # kg
    r_s   = 2*G*M/c**2            # m
    return r_s/R_H

for name, H0 in [("Planck", H0_planck), ("SH0ES", H0_sh0es)]:
    ratio = universe_ratio(H0)
    print(f"  {name}: r_s/R_H = {ratio:.6f}")

 
print("Experiment C: de Sitter Thermodynamics Identity")
  

hbar = 1.054571817e-34  # J·s

def de_sitter_identity(H0):
    R_H = c/H0
    A_H = 4*pi*R_H**2
    T_eq = hbar*H0/(2*pi*k_B)
    S_dS = k_B * A_H * c**3 / (4*G*hbar)
    lhs = T_eq * S_dS
    rhs = c**5/(2*G*H0)
    return lhs, rhs

for name, H0 in [("Planck", H0_planck), ("SH0ES", H0_sh0es)]:
    lhs, rhs = de_sitter_identity(H0)
    rel_err = abs(lhs - rhs)/rhs
    print(f"  {name}: T_eq*S_dS = {lhs:.6e} J")
    print(f"         c^5/(2GH0) = {rhs:.6e} J")
    print(f"         |LHS - RHS| / RHS = {rel_err:.6e}")
    print()

  
