#!/usr/bin/env python3
"""
Tests for Thomas-Wigner Precession Experiment

This module tests the Thomas-Wigner rotation ↔ hyperbolic triangle defect
equivalence that provides a clean, falsifiable result for CGM.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
from ..tw_precession import batch_check


def test_tw_precession_small_angle():
    """Test that small-velocity Thomas-Wigner precession works correctly."""
    stats = batch_check(n=500, beta_max=0.05, seed=123, c=1.0)

    # Expect tiny errors in the linear regime
    assert stats["max_abs_residual"] < 5e-6, f"Residual too large: {stats}"

    print("✅ Thomas-Wigner precession small angle test passed")


def test_tw_precession_consistency():
    """Test that results are consistent across different random seeds."""
    stats1 = batch_check(n=100, beta_max=0.1, seed=42, c=1.0)
    stats2 = batch_check(n=100, beta_max=0.1, seed=123, c=1.0)

    # Results should be similar (same order of magnitude)
    residual1 = stats1["max_abs_residual"]
    residual2 = stats2["max_abs_residual"]

    # Within factor of 10
    ratio = max(residual1, residual2) / min(residual1, residual2)
    assert ratio < 10.0, f"Results too inconsistent: ratio = {ratio}"

    print("✅ Thomas-Wigner precession consistency test passed")


def test_tw_precession_velocity_scaling():
    """Test that smaller velocities give smaller residuals."""
    stats_small = batch_check(n=100, beta_max=0.01, seed=42, c=1.0)
    stats_large = batch_check(n=100, beta_max=0.1, seed=42, c=1.0)

    # Smaller velocities should give smaller residuals (better linear approximation)
    assert (
        stats_small["max_abs_residual"] < stats_large["max_abs_residual"]
    ), "Smaller velocities should give smaller residuals"

    print("✅ Thomas-Wigner precession velocity scaling test passed")


def run_all_tw_tests():
    """Run all Thomas-Wigner precession tests."""
    print("Running Thomas-Wigner Precession Tests")
    print("=" * 40)

    try:
        test_tw_precession_small_angle()
        test_tw_precession_consistency()
        test_tw_precession_velocity_scaling()

        print("\n🎉 All Thomas-Wigner tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tw_tests()
    sys.exit(0 if success else 1)
