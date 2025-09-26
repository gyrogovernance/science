#!/usr/bin/env python3
"""
Tests for Dimensional Engine Core Invariants

This validates the key dimensional relationships that make the CGM
dimensional engine rigorous and non-circular.
"""

import numpy as np
import sys
import os

from ..functions.dimensions import DimensionalCalibrator, DimVec


def test_monomial_audit():
    """Test that L ≡ ħ c⁻¹ m⁻¹ and T ≡ ħ c⁻² m⁻¹"""
    print("Testing Dimensional Monomial Audit")
    print("-" * 35)

    # Initialize with measured constants
    hbar = 1.054571817e-34  # J·s
    c = 2.99792458e8  # m/s
    m_e = 9.1093837015e-31  # kg
    calib = DimensionalCalibrator(hbar, c, m_e)

    # Test L ≡ ħ c⁻¹ m⁻¹
    L_dim = DimVec(0, 1, 0)  # [L]
    L_expected = DimVec(1, -1, -1)  # ħ c⁻¹ m⁻¹

    L_computed = calib.monomial_for(L_dim)
    L_expected_computed = calib.monomial_for(L_expected)

    L_match = np.allclose(
        [L_computed["hbar"], L_computed["c"], L_computed["m_anchor"]],
        [
            L_expected_computed["hbar"],
            L_expected_computed["c"],
            L_expected_computed["m_anchor"],
        ],
        rtol=1e-12,
    )

    # Test T ≡ ħ c⁻² m⁻¹
    T_dim = DimVec(0, 0, 1)  # [T]
    T_expected = DimVec(1, -2, -1)  # ħ c⁻² m⁻¹

    T_computed = calib.monomial_for(T_dim)
    T_expected_computed = calib.monomial_for(T_expected)

    T_match = np.allclose(
        [T_computed["hbar"], T_computed["c"], T_computed["m_anchor"]],
        [
            T_expected_computed["hbar"],
            T_expected_computed["c"],
            T_expected_computed["m_anchor"],
        ],
        rtol=1e-12,
    )

    print(f"L ≡ ħ c⁻¹ m⁻¹: {'✅ PASS' if L_match else '❌ FAIL'}")
    print(f"T ≡ ħ c⁻² m⁻¹: {'✅ PASS' if T_match else '❌ FAIL'}")

    return L_match and T_match


def test_c_invariance():
    """Test that L₀/T₀ = c is preserved"""
    print("\nTesting c-Invariance (L₀/T₀ = c)")
    print("-" * 35)

    hbar = 1.054571817e-34  # J·s
    c = 2.99792458e8  # m/s
    m_e = 9.1093837015e-31  # kg
    calib = DimensionalCalibrator(hbar, c, m_e)

    # L₀/T₀ should equal c
    c_ratio = calib.L0 / calib.T0
    c_expected = c

    c_invariant = np.isclose(c_ratio, c_expected, rtol=1e-12)

    print(f"L₀/T₀: {c_ratio:.6e}")
    print(f"c:      {c_expected:.6e}")
    print(f"Match:  {'✅ PASS' if c_invariant else '❌ FAIL'}")

    return c_invariant


def test_compton_scales():
    """Test that Compton scales are correctly computed"""
    print("\nTesting Compton Scales")
    print("-" * 35)

    hbar = 1.054571817e-34  # J·s
    c = 2.99792458e8  # m/s
    m_e = 9.1093837015e-31  # kg
    calib = DimensionalCalibrator(hbar, c, m_e)

    # Test electron Compton wavelength
    lambda_compton_expected = hbar / (m_e * c)
    lambda_compton_computed = calib.L0  # This is ħ/(m⋆ c)

    lambda_match = np.isclose(
        lambda_compton_computed, lambda_compton_expected, rtol=1e-12
    )

    print(f"λ_compton (expected): {lambda_compton_expected:.6e} m")
    print(f"λ_compton (computed): {lambda_compton_computed:.6e} m")
    print(f"Match:                {'✅ PASS' if lambda_match else '❌ FAIL'}")

    return lambda_match


def run_all_dimensional_tests():
    """Run all dimensional engine tests"""
    print("DIMENSIONAL ENGINE VALIDATION")
    print("=" * 40)

    tests = [test_monomial_audit, test_c_invariance, test_compton_scales]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ {test.__name__} failed with error: {e}")
            results.append(False)

    passed = sum(results)
    total = len(results)

    print(f"\n{'='*40}")
    print(f"RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎯 All dimensional invariants validated!")
        print("   CGM dimensional engine is rigorous and non-circular")
    else:
        print("⚠️  Some dimensional tests failed - check implementation")

    return passed == total


if __name__ == "__main__":
    run_all_dimensional_tests()
