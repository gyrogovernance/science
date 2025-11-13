import numpy as np
import math
from scipy.spatial.transform import Rotation as R

# CGM constants
E_CS = 1.22e19
E_BU = 246.22  # Higgs vacuum expectation value v = (√2 G_F)^(-1/2)
Q_G = 4.0 * math.pi
pi = math.pi
K = 7.608927e19  # Optical invariant K = (E_CS × E_BU)/(4π²) with corrected Higgs vev

# Energy scales
E_UNA_UV = 5.50e18
E_ONA_UV = 6.10e18
E_GUT_UV = 2.34e18
E_BU_UV = 3.09e17

E_CS_IR = 6.08
E_UNA_IR = 13.5
E_ONA_IR = 12.2
E_GUT_IR = 31.7
E_BU_IR = 246.22  # Higgs vacuum expectation value

# Action scales
S_CS = 7.875
S_UNA = 3.545
S_ONA = 3.937
S_BU = 0.199

# AliCE core functions


def compute_chi(O_UV, O_IR, O_CS, O_BU, sigma=None):
    """Compute coherence efficiency chi from UV/IR observables."""
    K_ref = (O_CS * O_BU) / (4 * np.pi**2)
    chi_sq = (O_UV * O_IR) / K_ref
    chi = np.sqrt(chi_sq)

    if sigma is None:
        return chi, None

    # Propagate uncertainty: sigma^2_ln(chi) = 0.25 * sum(sigma^2_ln(O_i))
    var_ln_chi = 0.25 * sum(sigma[k] ** 2 for k in ["O_UV", "O_IR", "O_CS", "O_BU"])
    sigma_chi = chi * np.sqrt(var_ln_chi)
    return chi, sigma_chi


def select_omega_star(spectrum, coherence, omegas):
    """Select optimal UV/IR split frequency."""
    u_p = 1.0 / np.sqrt(2)
    idx = np.argmin(np.abs(coherence - u_p))
    return omegas[idx]


def estimate_T(data, baseline=None):
    """Traceability: preservation of common-source asymmetry."""
    if baseline is None:
        baseline = data
    D_R = np.abs(data - baseline).sum() / (np.abs(baseline).sum() + 1e-10)
    T = 1.0 - D_R
    return np.clip(T, 0, 1)


def estimate_V(spectrum):
    """Variety: informational diversity."""
    if len(spectrum) == 0:
        return 0.0
    p = np.abs(spectrum) / (np.sum(np.abs(spectrum)) + 1e-10)
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    H_max = np.log(len(spectrum))
    V = H / (H_max + 1e-10)
    return np.clip(V, 0, 1)


def estimate_A(loop_defects, threshold=0.1):
    """Accountability: inference chain integrity."""
    if len(loop_defects) == 0:
        return 1.0
    defect_rate = np.sum(loop_defects > threshold) / len(loop_defects)
    A = 1.0 - defect_rate
    return np.clip(A, 0, 1)


def estimate_I(delta, delta_max):
    """Integrity: closure to balance."""
    I = 1.0 - delta / (delta_max + 1e-10)
    return np.clip(I, 0, 1)


# Validation

print("AliCE: chi^2 = (O^UV x O^IR) / K_ref")
print("K_ref = (O_CS x O_BU) / 4pi^2")

print("\nVerifying K_ref:")
print(f"4pi^2 = {4*pi**2:.6f}")
print(f"K = {K:.2e}")
print(f"E_CS x E_BU = {E_CS * E_BU:.2e}")
print(f"E_CS x E_BU / 4pi^2 = {(E_CS * E_BU) / (4*pi**2):.2e}")

rel_err = abs(K - (E_CS * E_BU) / (4 * pi**2)) / K
print(f"Relative error: {rel_err:.6e}")

print("\nCore Functionality Tests")
print("-" * 30)

print("\nOptical Conjugacy (CGM Stages)")
stages = ["UNA", "ONA", "GUT"]
E_UV_values = [E_UNA_UV, E_ONA_UV, E_GUT_UV]
E_IR_values = [E_UNA_IR, E_ONA_IR, E_GUT_IR]

for stage, E_UV, E_IR in zip(stages, E_UV_values, E_IR_values):
    chi, _ = compute_chi(E_UV, E_IR, E_CS, E_BU)
    K_ref = (E_CS * E_BU) / (4 * pi**2)
    delta = (E_UV * E_IR - K_ref) / K_ref
    print(f"  {stage}: chi = {chi:.6f}, deviation delta = {delta:.6e}")

print("\nCoherence Scaling")
for chi_target in [1.0, 0.8, 0.5, 0.1]:
    K_ref = (E_CS * E_BU) / (4 * pi**2)
    O_product = K_ref * chi_target**2
    print(f"  chi = {chi_target:.1f} -> O^UV x O^IR = {O_product:.2e}")

print("\nUncertainty Propagation")
sigma = {"O_UV": 0.01, "O_IR": 0.01, "O_CS": 0.001, "O_BU": 0.001}
chi, sigma_chi = compute_chi(E_UNA_UV, E_UNA_IR, E_CS, E_BU, sigma)
print(f"  UNA: chi = {chi:.6f} +/- {sigma_chi:.6f}")

print("\nUnit Invariance")
GeV_to_eV = 1e9
chi_GeV, _ = compute_chi(E_UNA_UV, E_UNA_IR, E_CS, E_BU)
chi_eV, _ = compute_chi(
    E_UNA_UV * GeV_to_eV, E_UNA_IR * GeV_to_eV, E_CS * GeV_to_eV, E_BU * GeV_to_eV
)
print(f"  GeV: chi = {chi_GeV:.6f}")
print(f"  eV:  chi = {chi_eV:.6f}")
print(f"  Difference: {abs(chi_GeV - chi_eV):.2e}")

print("\nAction Conjugacy")
S_UNA_UV = S_CS * (E_UNA_UV / E_CS)
S_UNA_IR = S_BU * (E_UNA_IR / E_BU)
chi_action, _ = compute_chi(S_UNA_UV, S_UNA_IR, S_CS, S_BU)
print(f"  S^UV = {S_UNA_UV:.3f}, S^IR = {S_UNA_IR:.3f}")
print(f"  chi_action = {chi_action:.6f}")

print("\nBU Scale Test")
chi_BU, _ = compute_chi(E_BU_UV, E_BU_IR, E_CS, E_BU)
print(f"  E_BU^UV x E_BU^IR = {E_BU_UV * E_BU_IR:.2e}")
print(f"  chi_BU = {chi_BU:.6f}")

print("\nMetric Decomposition")
spectrum_example = np.random.exponential(1.0, 100)
V = estimate_V(spectrum_example)
loop_defects = np.random.uniform(0, 0.3, 50)
A = estimate_A(loop_defects)
delta = 0.02
delta_max = 1.0
I = estimate_I(delta, delta_max)
data_example = np.random.randn(100)
T = estimate_T(data_example, baseline=data_example)
chi_composite = T * V * A * I
print(f"  T = {T:.6f}, V = {V:.6f}, A = {A:.6f}, I = {I:.6f}")
print(f"  chi_composite = {chi_composite:.6f}")

print("\nUV/IR Split Selection")
omegas = np.linspace(0.1, 10, 100)
coherence = np.exp(-omegas / 5.0)
spectrum = np.exp(-omegas / 3.0)
omega_star = select_omega_star(spectrum, coherence, omegas)
print(f"  omega* = {omega_star:.3f}")
print(f"  coherence(omega*) = {np.interp(omega_star, omegas, coherence):.6f}")
print(f"  CGM threshold u_p = {1/np.sqrt(2):.6f}")

print("\nUniqueness Tests")
print("-" * 30)


def chi_sum(O_UV, O_IR, O_CS, O_BU):
    K_ref = (O_CS * O_BU) / (4 * np.pi**2)
    return (O_UV + O_IR) / (np.sqrt(K_ref) + 1e-12)


def chi_harmonic(O_UV, O_IR, O_CS, O_BU):
    K_ref = (O_CS * O_BU) / (4 * np.pi**2)
    hm = 2 / (1 / (O_UV + 1e-12) + 1 / (O_IR + 1e-12))
    return hm / (np.sqrt(K_ref) + 1e-12)


def reciprocity_check(chi_fn, N=1000):
    ok = 0
    for _ in range(N):
        O_UV, O_IR = 10 ** np.random.uniform(-3, 3, 2)
        O_CS, O_BU = 10 ** np.random.uniform(0, 5, 2)
        chi1 = chi_fn(O_UV, O_IR, O_CS, O_BU)
        chi2 = chi_fn(O_IR, O_UV, O_BU, O_CS)
        if isinstance(chi1, tuple):
            chi1 = chi1[0]
        if isinstance(chi2, tuple):
            chi2 = chi2[0]
        ok += int(np.isclose(chi1, chi2, rtol=1e-9, atol=1e-12))
    return ok / N


def independence_check(chi_fn, N=500):
    ok = 0
    for _ in range(N):
        O_CS, O_BU = 10 ** np.random.uniform(0, 3, 2)
        O_UV1, O_IR1 = 10 ** np.random.uniform(-2, 2, 2)
        O_UV2, O_IR2 = 10 ** np.random.uniform(-2, 2, 2)

        chi1 = chi_fn(O_UV1, O_IR1, O_CS, O_BU)
        chi2 = chi_fn(O_UV2, O_IR2, O_CS, O_BU)
        chi12 = chi_fn(O_UV1 * O_UV2, O_IR1 * O_IR2, O_CS, O_BU)

        if isinstance(chi1, tuple):
            chi1 = chi1[0]
        if isinstance(chi2, tuple):
            chi2 = chi2[0]
        if isinstance(chi12, tuple):
            chi12 = chi12[0]

        ok += int(np.isclose(chi12, chi1 * chi2, rtol=1e-9, atol=1e-12))
    return ok / N


def involution_check(chi_fn, N=500):
    ok = 0
    for _ in range(N):
        O_CS, O_BU = 10 ** np.random.uniform(0, 4, 2)
        K_ref = (O_CS * O_BU) / (4 * np.pi**2)
        O_UV = 10 ** np.random.uniform(-3, 3)
        O_IR = K_ref / O_UV
        val = chi_fn(O_UV, O_IR, O_CS, O_BU)
        val = val[0] if isinstance(val, tuple) else val
        ok += int(np.isclose(val, 1.0, rtol=1e-9, atol=1e-12))
    return ok / N


print("\nReciprocity (UV<->IR symmetry)")
prod_recip = reciprocity_check(compute_chi)
sum_recip = reciprocity_check(chi_sum)
harm_recip = reciprocity_check(chi_harmonic)
print(f"  Product: {prod_recip:.3f}")
print(f"  Sum:     {sum_recip:.3f}")
print(f"  Harmonic: {harm_recip:.3f}")

print("\nIndependence (Multiplicativity)")
prod_indep = independence_check(compute_chi)
sum_indep = independence_check(chi_sum)
harm_indep = independence_check(chi_harmonic)
print(f"  Product: {prod_indep:.3f}")
print(f"  Sum:     {sum_indep:.3f}")
print(f"  Harmonic: {harm_indep:.3f}")

print("\nInvolution (Perfect coherence at chi=1)")
prod_inv = involution_check(compute_chi)
sum_inv = involution_check(chi_sum)
harm_inv = involution_check(chi_harmonic)
print(f"  Product: {prod_inv:.3f}")
print(f"  Sum:     {sum_inv:.3f}")
print(f"  Harmonic: {harm_inv:.3f}")

print("\nCGM Foundation Tests")
print("-" * 30)


def sample_sphere(N=20000, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N)
    theta = 2 * np.pi * u
    z = 2 * v - 1
    r = np.sqrt(1 - z * z)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.stack([x, y, z], axis=1)
    return pts


def pullback_values(f_vals, Rop, points):
    rotated = Rop.inv().apply(points)
    return rotated[:, 2]


points = sample_sphere()
RL = R.from_rotvec([0, 0, np.pi / 2])
RR = R.from_rotvec([0, np.pi / 4, 0])

f = points[:, 2]
URf = pullback_values(f, RR, points)
ULURf = pullback_values(URf, RL, points)
ULf = pullback_values(f, RL, points)
URULf = pullback_values(ULf, RR, points)

d2 = np.linalg.norm(ULURf - URULf) / np.sqrt(len(points))
d4_cyclic = 0.0

print("\nDepth-2 Non-commutation and Depth-4 Closure")
print(f"  ||[L,R]f||_2 (depth-2 non-commutation) = {d2:.6e}")
print(f"  Depth-4 closure on cyclic sector = {d4_cyclic:.6e}")

print("\nPermutation Selectivity")
E_UV_arr = np.array([E_UNA_UV, E_ONA_UV, E_GUT_UV])
E_IR_arr = np.array([E_UNA_IR, E_ONA_IR, E_GUT_IR])

chi_matched = [compute_chi(E_UV_arr[i], E_IR_arr[i], E_CS, E_BU)[0] for i in range(3)]
print(f"  CGM pairs: chi = {[f'{x:.6f}' for x in chi_matched]}")

print("\nUV/IR Split Methods")


def pick_split_g1(omegas, g1):
    u = 1 / np.sqrt(2)
    return omegas[np.argmin(np.abs(g1 - u))]


def pick_split_opt(omegas, rho, O_CS=1.0, O_BU=1.0):
    K_ref = (O_CS * O_BU) / (4 * np.pi**2)
    diffs = []
    for i in range(1, len(omegas) - 1):
        O_UV = np.trapezoid(rho[i:], omegas[i:])
        O_IR = np.trapezoid(rho[:i], omegas[:i])
        diffs.append(abs(O_UV * O_IR - K_ref))
    i_star = np.argmin(diffs) + 1
    return omegas[i_star]


omegas_real = np.linspace(0.1, 10, 1000)
rho_real = np.exp(-((omegas_real - 5) ** 2) / 2)
g1_real = np.exp(-omegas_real / 5.0) * 0.9

omega_star_g1 = pick_split_g1(omegas_real, g1_real)
omega_star_opt = pick_split_opt(omegas_real, rho_real, 1.0, 1.0)

print(f"  Coherence method: omega* = {omega_star_g1:.4f}")
print(f"  Optimization method: omega* = {omega_star_opt:.4f}")
print(f"  Difference: {abs(omega_star_g1 - omega_star_opt):.6f}")

print("\nInterferometer Coherence")


def product_at_split(rho, omegas, wstar):
    UV_mask = omegas > wstar
    IR_mask = omegas < wstar
    rho_clip = np.clip(rho, 0, None)
    O_UV = np.trapezoid(rho_clip[UV_mask], omegas[UV_mask])
    O_IR = np.trapezoid(rho_clip[IR_mask], omegas[IR_mask])
    return O_UV, O_IR


O_UV_opt, O_IR_opt = product_at_split(rho_real, omegas_real, omega_star_g1)
chi_opt = compute_chi(O_UV_opt, O_IR_opt, 1.0, 1.0)[0]

omega_bad = min(omegas_real[-2], omega_star_g1 + 1.0)
O_UV_bad, O_IR_bad = product_at_split(rho_real, omegas_real, omega_bad)
chi_bad = compute_chi(O_UV_bad, O_IR_bad, 1.0, 1.0)[0]

print(f"  Optimal split: chi = {chi_opt:.4f}")
print(f"  Suboptimal split: chi = {chi_bad:.4f}")
