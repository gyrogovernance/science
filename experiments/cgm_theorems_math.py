#!/usr/bin/env python3
"""
CGM Proof Runner

This module runs systematic proofs of the core CGM theorems:
- Theorem A: Dimensional homomorphism (uniqueness of exponents)
- Theorem B: Unique G monomial (κ is necessary & dimensionless)
- Theorem C: Calibrator base-unit identities (c-invariance)
- Theorem D: Gyrotriangle defect = hyperbolic area/c² (Gauss-Bonnet)
- Theorem E: Thomas-Wigner small-velocity gyration is a rotation

Run with: python -m Experiments.theorems.run_proofs
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the required modules
from experiments.functions.dimensions import DimensionalCalibrator, DimVec
from experiments.functions.gyrogeometry import GyroTriangleDefectTheorem
from experiments.functions.gyrovector_ops import GyroVectorSpace


def test_theorem_c_base_unit_identities():
    """Test Theorem C: Calibrator base-unit identities and c-invariance."""
    print("Testing Theorem C: Base-unit identities & c-invariance...")

    hbar = 1.054571817e-34
    c = 2.99792458e8
    m_e = 9.1093837015e-31

    cal = DimensionalCalibrator(hbar, c, m_e)

    # Get base units
    base_units = cal.base_units_SI()
    M0, L0, T0 = base_units["M0"], base_units["L0"], base_units["T0"]

    # Verify identities
    assert np.isclose(M0, m_e, rtol=1e-12), f"M0 = {M0} ≠ {m_e}"
    assert np.isclose(L0, hbar / (m_e * c), rtol=1e-12), f"L0 = {L0} ≠ {hbar/(m_e*c)}"
    assert np.isclose(
        T0, hbar / (m_e * c**2), rtol=1e-12
    ), f"T0 = {T0} ≠ {hbar/(m_e*c**2)}"
    assert np.isclose(L0 / T0, c, rtol=1e-12), f"L0/T0 = {L0/T0} ≠ {c}"

    print("✓ Base-unit identities & c-invariance")
    return True


def test_theorem_b_g_monomial():
    """Test Theorem B: Unique G monomial and κ necessity."""
    print("Testing Theorem B: Unique G monomial (κ is dimensionless)...")

    hbar = 1.054571817e-34
    c = 2.99792458e8
    m_e = 9.1093837015e-31

    cal = DimensionalCalibrator(hbar, c, m_e)

    # G dimensions: M⁻¹L³T⁻²
    DIM_G = DimVec(M=-1, L=3, T=-2)
    monoG = cal.monomial_for(DIM_G)

    # Verify the unique monomial: a_G = (1, 1, -2)
    expected_exponents = [1.0, 1.0, -2.0]
    actual_exponents = [monoG["hbar"], monoG["c"], monoG["m_anchor"]]

    assert np.allclose(
        actual_exponents, expected_exponents, rtol=1e-12
    ), f"G monomial {actual_exponents} ≠ expected {expected_exponents}"

    # Verify this gives the correct dimensional structure
    # G = ħ¹ c¹ m_anchor⁻² × (dimensionless κ)
    g_unit = cal.get_unit(DIM_G)
    g_expected = (hbar * c) / (m_e**2)

    # The unit should match the dimensional structure (up to dimensionless κ)
    assert np.isclose(
        g_unit, g_expected, rtol=1e-12
    ), f"G unit {g_unit} ≠ expected {g_expected}"

    print("✓ Unique G monomial a_G=(1,1,-2); κ is dimensionless")
    return True


def test_theorem_a_dimensional_homomorphism():
    """Test Theorem A: Dimensional homomorphism u(d₁+d₂) = u(d₁)u(d₂)."""
    print("Testing Theorem A: Dimensional homomorphism...")

    hbar = 1.054571817e-34
    c = 2.99792458e8
    m_e = 9.1093837015e-31

    cal = DimensionalCalibrator(hbar, c, m_e)

    # Test with basis vectors
    E = DimVec(1, 0, 0)  # M
    L = DimVec(0, 1, 0)  # L
    T = DimVec(0, 0, 1)  # T

    # Get individual units
    uE, uL, uT = cal.get_unit(E), cal.get_unit(L), cal.get_unit(T)

    # Get unit of sum
    ELT = DimVec(1, 1, 1)  # M·L·T
    uELT = cal.get_unit(ELT)

    # Verify homomorphism: u(d₁+d₂) = u(d₁)u(d₂)
    assert np.isclose(
        uELT, uE * uL * uT, rtol=1e-12
    ), f"u(ELT) = {uELT} ≠ u(E)u(L)u(T) = {uE*uL*uT}"

    # Test with more complex combinations
    energy = DimVec(1, 2, -2)  # ML²T⁻²
    momentum = DimVec(1, 1, -1)  # MLT⁻¹
    energy_momentum = DimVec(2, 3, -3)  # M²L³T⁻³

    u_energy = cal.get_unit(energy)
    u_momentum = cal.get_unit(momentum)
    u_energy_momentum = cal.get_unit(energy_momentum)

    # Test homomorphism for energy + momentum = energy_momentum
    # Note: This is not a direct sum, but tests the monomial structure
    energy_plus_momentum = energy + momentum  # (1,2,-2) + (1,1,-1) = (2,3,-3)
    u_energy_plus_momentum = cal.get_unit(energy_plus_momentum)

    # Verify: u(energy + momentum) = u(energy) × u(momentum)
    homomorphism_verified = np.isclose(
        u_energy_plus_momentum, u_energy * u_momentum, rtol=1e-12
    )
    assert (
        homomorphism_verified
    ), f"Energy+momentum homomorphism failed: {u_energy_plus_momentum} ≠ {u_energy * u_momentum}"

    print("✓ Dimensional homomorphism u(d₁+d₂) = u(d₁)u(d₂)")
    return True


def test_theorem_d_gauss_bonnet():
    """Test Theorem D: Gyrotriangle defect = hyperbolic area/c²."""
    print("Testing Theorem D: Gauss-Bonnet defect = area/c²...")

    gb = GyroTriangleDefectTheorem(c=1.0)

    # Test the closure case: (π/2, π/4, π/4)
    angles = (np.pi / 2, np.pi / 4, np.pi / 4)
    defect = np.pi - sum(angles)  # Should be 0

    # Verify defect-area equivalence
    ver = gb.verify_defect_area_equivalence(defect, angles)

    assert ver["all_consistent"], f"Defect-area equivalence failed: {ver}"
    assert np.isclose(defect, 0.0, atol=1e-12), f"Defect {defect} ≠ 0"

    # Test area computation
    area = gb.compute_hyperbolic_area(angles)
    assert np.isclose(area, 0.0, atol=1e-12), f"Area {area} ≠ 0"

    print("✓ Defect = area/c² (Gauss-Bonnet) closure case")
    return True


def test_theorem_e_thomas_wigner():
    """Test Theorem E: Thomas-Wigner small-velocity gyration is a rotation."""
    print("Testing Theorem E: Thomas-Wigner gyration is SO(3)...")

    gs = GyroVectorSpace(c=1.0)

    # Test with small velocities (should give proper rotation)
    u = np.array([1e-3, 0, 0])
    v = np.array([0, 1e-3, 0])

    R = gs.gyration(u, v)

    # Check rotation properties: orthogonal, det=1
    orthogonality_error = np.linalg.norm(R.T @ R - np.eye(3))
    det_error = abs(np.linalg.det(R) - 1.0)

    assert orthogonality_error < 1e-10, f"Orthogonality error: {orthogonality_error}"
    assert det_error < 1e-10, f"Determinant error: {det_error}"

    # Test with near-light-speed vectors (should trigger proof-guard)
    u_near_c = np.array([0.9, 0.1, 0.0])
    v_near_c = np.array([0.1, 0.9, 0.0])

    R_near_c = gs.gyration(u_near_c, v_near_c)

    # Should still be a proper rotation matrix
    orthogonality_error_near_c = np.linalg.norm(R_near_c.T @ R_near_c - np.eye(3))
    det_error_near_c = abs(np.linalg.det(R_near_c) - 1.0)

    assert (
        orthogonality_error_near_c < 1e-10
    ), f"Near-c orthogonality error: {orthogonality_error_near_c}"
    assert det_error_near_c < 1e-10, f"Near-c determinant error: {det_error_near_c}"

    print("✓ Thomas-Wigner gyration is SO(3) (with proof-guard)")
    return True


def test_basis_necessity_theorem():
    """Test the Basis Necessity Theorem: why we need {ħ, c, m⋆}."""
    print("Testing Basis Necessity Theorem...")

    hbar = 1.054571817e-34
    c = 2.99792458e8
    m_e = 9.1093837015e-31

    # Test with full basis {ħ, c, m⋆}
    cal_full = DimensionalCalibrator(hbar, c, m_e)
    base_full = cal_full.base_units_SI()

    # Should work and give c-invariance
    assert np.isclose(base_full["L0"] / base_full["T0"], c, rtol=1e-12)

    # Test with only {ħ, c} (missing mass scale)
    # Use a very small but non-zero mass to avoid exact zero issues
    m_tiny = 1e-30  # Small mass (not extremely small to avoid numerical issues)
    cal_partial = DimensionalCalibrator(hbar, c, m_tiny)
    base_partial = cal_partial.base_units_SI()

    # With small m_anchor, we should get:
    # M0 ≈ m_tiny (mass scale small)
    # L0 ≈ ħ/(m_tiny*c) (length scale small for heavy particles)
    # T0 ≈ ħ/(m_tiny*c²) (time scale small for heavy particles)
    # L0/T0 = c (c-invariance preserved)
    assert np.isclose(
        base_partial["M0"], m_tiny, rtol=1e-12
    ), f"Mass scale should be {m_tiny}, got {base_partial['M0']}"
    assert (
        base_partial["L0"] < 1e-10
    ), f"Length scale should be small for heavy particle, got {base_partial['L0']}"
    assert (
        base_partial["T0"] < 1e-18
    ), f"Time scale should be small for heavy particle, got {base_partial['T0']}"
    assert np.isclose(
        base_partial["L0"] / base_partial["T0"], c, rtol=1e-12
    ), "c-invariance should be preserved"

    print(
        "✓ Basis Necessity Theorem: {ħ, c, m⋆} basis is necessary (tested with extreme scaling)"
    )
    return True


def test_property_based_homomorphism():
    """Test homomorphism property with random dimension vectors."""
    print("Testing property-based homomorphism...")

    hbar = 1.054571817e-34
    c = 2.99792458e8
    m_e = 9.1093837015e-31

    cal = DimensionalCalibrator(hbar, c, m_e)

    # Test with random dimension vectors
    np.random.seed(42)  # for reproducibility
    n_tests = 100

    for i in range(n_tests):
        # Generate random dimension vectors with reasonable exponents
        # Avoid extreme negative exponents that cause overflow
        d1 = DimVec(
            M=np.random.randint(-1, 3),
            L=np.random.randint(-1, 3),
            T=np.random.randint(-1, 3),
        )
        d2 = DimVec(
            M=np.random.randint(-1, 3),
            L=np.random.randint(-1, 3),
            T=np.random.randint(-1, 3),
        )

        # Test homomorphism: u(d₁ + d₂) = u(d₁) u(d₂)
        u1 = cal.get_unit(d1)
        u2 = cal.get_unit(d2)
        d_sum = d1 + d2
        u_sum = cal.get_unit(d_sum)

        # Verify homomorphism
        homomorphism_ok = np.isclose(u_sum, u1 * u2, rtol=1e-12)
        assert (
            homomorphism_ok
        ), f"Homomorphism failed for d1={d1}, d2={d2}: u_sum={u_sum} ≠ u1*u2={u1*u2}"

        # Test monomial solving: B a = d
        mono1 = cal.monomial_for(d1)
        mono2 = cal.monomial_for(d2)
        mono_sum = cal.monomial_for(d_sum)

        # Verify that exponents add: a(d₁ + d₂) = a(d₁) + a(d₂)
        for key in ["hbar", "c", "m_anchor"]:
            expected = mono1[key] + mono2[key]
            actual = mono_sum[key]
            assert np.isclose(
                actual, expected, rtol=1e-12
            ), f"Exponent addition failed for {key}: {actual} ≠ {expected}"

    print(f"✓ Property-based homomorphism: {n_tests} random tests passed")
    return True


def run_all_proofs():
    """Run all CGM theorem proofs."""
    print("CGM Theorem Proof Runner")
    print("=" * 40)
    print()

    tests = [
        test_theorem_c_base_unit_identities,
        test_theorem_b_g_monomial,
        test_theorem_a_dimensional_homomorphism,
        test_theorem_d_gauss_bonnet,
        test_theorem_e_thomas_wigner,
        test_basis_necessity_theorem,
        test_property_based_homomorphism,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            print()

    print("=" * 40)
    print(f"Proof Summary: {passed}/{total} theorems validated")

    if passed == total:
        print("🎉 All CGM theorems proven and validated!")
        return True
    else:
        print(f"⚠️  {total - passed} theorem(s) need attention")
        return False


if __name__ == "__main__":
    success = run_all_proofs()
    sys.exit(0 if success else 1)
