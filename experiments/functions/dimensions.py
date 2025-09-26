"""
Dimensional Calibration Engine for CGM

This module implements the dimensional calibration engine that maps dimension vectors
to physical units using the basis {ħ, c, m⋆}. The engine provides a group homomorphism
from (ℝ³, +) to (ℝ₊, ×) and enables non-circular dimensional analysis.

Key Features:
    - Dimensional matrix B with {ħ, c, m⋆} basis
    - Unique exponent solutions via matrix inversion
    - Base unit computation (M₀, L₀, T₀)
    - c-invariance preservation (L₀/T₀ = c)
    - Monomial solving for arbitrary dimensions
    - Homomorphism property verification

Classes:
    DimVec: Dimension vector with M, L, T exponents
    DimensionalCalibrator: Main calibration engine
    DimensionalEngineHomomorphism: Mathematical proof of homomorphism property
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional


# Exponents order: [M, L, T]
@dataclass(frozen=True)
class DimVec:
    M: float  # Mass exponent
    L: float  # Length exponent
    T: float  # Time exponent

    def as_array(self) -> np.ndarray:
        return np.array([self.M, self.L, self.T], dtype=float)

    def __add__(self, other: "DimVec") -> "DimVec":
        """Add dimension vectors component-wise."""
        return DimVec(self.M + other.M, self.L + other.L, self.T + other.T)


# Dimensions of common constants in SI
DIM_HBAR = DimVec(M=1, L=2, T=-1)  # ħ: M·L²·T⁻¹
DIM_C = DimVec(M=0, L=1, T=-1)  # c: L·T⁻¹
DIM_MASS = DimVec(M=1, L=0, T=0)  # m⋆: M (anchor mass)


class DimensionalCalibrator:
    """
    Solve B * a = d, where columns of B are the basis dims (ħ, c, m⋆),
    'a' are exponents, and d is the target dimension vector.

    This provides a clean mapping from target dimensions to SI units
    using measured constants as the basis.
    """

    def __init__(self, hbar: float, c: float, m_anchor: float):
        """
        Initialize with measured constants that form a complete basis.

        Args:
            hbar: Planck constant in J·s
            c: Speed of light in m/s
            m_anchor: Anchor mass in kg (e.g., electron mass)
        """
        self.values = np.array([hbar, c, m_anchor], dtype=float)

        # Build the dimensional basis matrix B
        # Columns are [ħ, c, m⋆] dimensions in [M, L, T] order
        self.B = np.column_stack(
            [
                DIM_HBAR.as_array(),  # [1, 2, -1]
                DIM_C.as_array(),  # [0, 1, -1]
                DIM_MASS.as_array(),  # [1, 0, 0]
            ]
        )  # shape (3,3), must be invertible

        # Verify the basis is invertible (linearly independent)
        if np.linalg.det(self.B) == 0:
            raise ValueError("Basis constants have linearly dependent dimensions")

        # Precompute physical base units (SI scalings)
        self.M0 = self._unit_from_dim(DIM_MASS)  # should return m_anchor
        self.L0 = self._unit_from_dim(DimVec(0, 1, 0))  # ħ/(m⋆ c)
        self.T0 = self._unit_from_dim(DimVec(0, 0, 1))  # ħ/(m⋆ c²)

        # Validate the calibration
        self._validate_calibration()

    def _exponents_for(self, d: DimVec) -> np.ndarray:
        """Solve B * a = d for the exponents a."""
        return np.linalg.solve(self.B, d.as_array())

    def _unit_from_dim(self, d: DimVec) -> float:
        """Convert a dimension vector to SI units using the basis."""
        a = self._exponents_for(d)
        return float(np.prod(self.values**a))

    def _validate_calibration(self):
        """Verify that the calibration produces expected results."""
        # M0 should equal the anchor mass
        if not np.isclose(self.M0, self.values[2], rtol=1e-10):
            raise ValueError(f"Mass calibration failed: {self.M0} != {self.values[2]}")

        # L0 should equal ħ/(m⋆ c) (Compton length)
        expected_L0 = self.values[0] / (self.values[2] * self.values[1])
        if not np.isclose(self.L0, expected_L0, rtol=1e-10):
            raise ValueError(f"Length calibration failed: {self.L0} != {expected_L0}")

        # T0 should equal ħ/(m⋆ c²)
        expected_T0 = self.values[0] / (self.values[2] * self.values[1] ** 2)
        if not np.isclose(self.T0, expected_T0, rtol=1e-10):
            raise ValueError(f"Time calibration failed: {self.T0} != {expected_T0}")

    def base_units_SI(self) -> Dict[str, float]:
        """Get the base SI units from calibration."""
        return {
            "M0": self.M0,  # kg
            "L0": self.L0,  # m
            "T0": self.T0,  # s
        }

    def get_unit(self, d: DimVec) -> float:
        """Get SI units for any dimension vector."""
        return self._unit_from_dim(d)

    def monomial_for(self, d: DimVec) -> Dict[str, float]:
        """Return the unique exponents (a_hbar, a_c, a_m) with B a = d."""
        a = self._exponents_for(d)
        return {"hbar": float(a[0]), "c": float(a[1]), "m_anchor": float(a[2])}

    def audit_dimensions(self, expression_name: str, d: DimVec) -> Dict[str, Any]:
        """Audit the dimensional analysis of an expression."""
        exponents = self._exponents_for(d)
        si_unit = self._unit_from_dim(d)

        return {
            "expression": expression_name,
            "dimensions": {"M": d.M, "L": d.L, "T": d.T},
            "basis_exponents": {
                "hbar": exponents[0],
                "c": exponents[1],
                "m_anchor": exponents[2],
            },
            "si_unit": si_unit,
            "si_unit_description": self._describe_unit(d),
        }

    def _describe_unit(self, d: DimVec) -> str:
        """Generate a human-readable description of the SI unit."""
        parts = []
        if d.M != 0:
            parts.append(f"kg^{d.M}")
        if d.L != 0:
            parts.append(f"m^{d.L}")
        if d.T != 0:
            parts.append(f"s^{d.T}")

        if not parts:
            return "dimensionless"

        return "·".join(parts)


def audit_expression(
    calibrator: DimensionalCalibrator,
    expression_name: str,
    dimensions: DimVec,
    value: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Audit the dimensional analysis of an expression.

    Args:
        calibrator: The dimensional calibrator instance
        expression_name: Name of the expression being audited
        dimensions: Dimensional exponents [M, L, T]
        value: Optional numerical value for additional validation

    Returns:
        Dictionary with dimensional analysis results
    """
    audit = calibrator.audit_dimensions(expression_name, dimensions)

    if value is not None:
        # Check if the value has reasonable magnitude for the units
        si_unit = audit["si_unit"]
        if si_unit > 0:
            # For length: should be reasonable (nm to km range)
            if dimensions.L == 1 and dimensions.M == 0 and dimensions.T == 0:
                if si_unit < 1e-12 or si_unit > 1e6:  # nm to km
                    audit["warning"] = (
                        f"Length {si_unit:.2e} m may be outside reasonable range"
                    )
            # For time: should be reasonable (fs to years range)
            elif dimensions.T == 1 and dimensions.M == 0 and dimensions.L == 0:
                if si_unit < 1e-18 or si_unit > 1e9:  # fs to years
                    audit["warning"] = (
                        f"Time {si_unit:.2e} s may be outside reasonable range"
                    )
            # For mass: should be reasonable (ng to kg range)
            elif dimensions.M == 1 and dimensions.L == 0 and dimensions.T == 0:
                if si_unit < 1e-15 or si_unit > 1e3:  # ng to kg
                    audit["warning"] = (
                        f"Mass {si_unit:.2e} kg may be outside reasonable range"
                    )

    return audit


def validate_dimensions(
    calibrator: DimensionalCalibrator,
    expected: DimVec,
    actual: DimVec,
    tolerance: float = 1e-10,
) -> bool:
    """
    Validate that two dimension vectors match within tolerance.

    Args:
        calibrator: The dimensional calibrator instance
        expected: Expected dimensions
        actual: Actual dimensions
        tolerance: Numerical tolerance for comparison

    Returns:
        True if dimensions match, False otherwise
    """
    # Compare the dimension *vectors* (or the basis exponents), not the
    # numerical unit values which might coincide by accident.
    exp_expected = calibrator._exponents_for(expected)
    exp_actual = calibrator._exponents_for(actual)
    return bool(np.allclose(exp_expected, exp_actual, rtol=0, atol=tolerance))


class DimensionalEngineHomomorphism:
    """
    Theorem: The DimensionalCalibrator implements a group homomorphism.

    Let B = [dim(ħ) dim(c) dim(m⋆)] with rows (M,L,T).
    We solve Ba = d and map d ↦ u(d) = ∏ᵢ vᵢ^aᵢ where v = (ħ,c,m⋆).

    Because B is invertible, the exponent vector a(d) = B⁻¹d is unique,
    hence the unit map is well-defined and:

        u(d₁ + d₂) = ∏ v^(a(d₁) + a(d₂)) = u(d₁) u(d₂)

    So the calibrator implements a group homomorphism ℝ³ → ℝ₊
    (additive exponents → multiplicative units).

    This class is a thin wrapper around DimensionalCalibrator
    that proves the homomorphism property.
    """

    def __init__(self, hbar: float, c: float, m_anchor: float):
        """
        Initialize with measured constants.

        Args:
            hbar: Planck constant in J·s
            c: Speed of light in m/s
            m_anchor: Anchor mass in kg
        """
        self.calibrator = DimensionalCalibrator(hbar, c, m_anchor)
        self.values = np.array([hbar, c, m_anchor], dtype=float)

    def verify_homomorphism(self, d1: DimVec, d2: DimVec) -> Dict[str, Any]:
        """
        Verify the homomorphism property: u(d₁ + d₂) = u(d₁) u(d₂)

        Args:
            d1, d2: Dimension vectors to test

        Returns:
            Verification results
        """
        # Use the core implementation to compute units
        u1 = self.calibrator.get_unit(d1)
        u2 = self.calibrator.get_unit(d2)

        # Compute unit of sum
        d_sum = d1 + d2
        u_sum = self.calibrator.get_unit(d_sum)

        # Verify homomorphism: u(d₁ + d₂) = u(d₁) u(d₂)
        homomorphism_satisfied = np.isclose(u_sum, u1 * u2, rtol=1e-10)

        return {
            "d1": d1,
            "d2": d2,
            "d_sum": d_sum,
            "u1": u1,
            "u2": u2,
            "u_sum": u_sum,
            "u1_times_u2": u1 * u2,
            "homomorphism_satisfied": homomorphism_satisfied,
            "note": "u(d₁ + d₂) = u(d₁) u(d₂) using DimensionalCalibrator",
        }

    def test_base_units(self) -> Dict[str, Any]:
        """
        Test that base units are correctly computed.

        Returns:
            Verification of L₀, T₀, M₀ calculations
        """
        base = self.calibrator.base_units_SI()

        # Verify c-invariance: L₀/T₀ = c
        c_invariant = base["L0"] / base["T0"]
        c_correct = self.values[1]
        c_invariance_satisfied = np.isclose(c_invariant, c_correct, rtol=1e-10)

        return {
            "M0_computed": base["M0"],
            "M0_correct": self.values[2],
            "M0_consistent": np.isclose(base["M0"], self.values[2], rtol=1e-10),
            "L0_computed": base["L0"],
            "T0_computed": base["T0"],
            "c_invariant": c_invariant,
            "c_correct": c_correct,
            "c_invariance_satisfied": c_invariance_satisfied,
            "all_consistent": c_invariance_satisfied,
            "note": "Base units from DimensionalCalibrator",
        }
