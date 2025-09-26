"""
Tests for the CGM experiments package

This module contains test functions for validating the CGM framework:

Dimensional Tests:
    test_monomial_audit: Test monomial dimensional consistency
    test_c_invariance: Test speed of light invariance
    test_compton_scales: Test Compton wavelength calculations
    run_all_dimensional_tests: Run complete dimensional validation

Thomas-Wigner Tests:
    test_tw_precession_small_angle: Test TW precession in small angle limit
    test_tw_precession_consistency: Test TW precession consistency
    test_tw_precession_velocity_scaling: Test velocity scaling properties
    run_all_tw_tests: Run complete TW validation
"""

# Import test functions
from .test_dimensions import (
    test_monomial_audit,
    test_c_invariance,
    test_compton_scales,
    run_all_dimensional_tests,
)

from .test_tw_precession import (
    test_tw_precession_small_angle,
    test_tw_precession_consistency,
    test_tw_precession_velocity_scaling,
    run_all_tw_tests,
)

__all__ = [
    # Dimensional tests
    "test_monomial_audit",
    "test_c_invariance",
    "test_compton_scales",
    "run_all_dimensional_tests",
    # Thomas-Wigner tests
    "test_tw_precession_small_angle",
    "test_tw_precession_consistency",
    "test_tw_precession_velocity_scaling",
    "run_all_tw_tests",
]
