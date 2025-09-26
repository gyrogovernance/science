"""
Gyrogeometry Theorems for CGM

This module contains mathematical proofs related to gyrovector geometry,
including the Gauss-Bonnet theorem for constant curvature spaces.
"""

import numpy as np
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod


class GyroTriangleDefectTheorem:
    """
    Theorem: In the Einstein-Ungar ball model (velocity space),
    the gyrotriangle defect equals the hyperbolic area.

    Proof: The metric has constant sectional curvature -1/c².
    For a geodesic triangle with angles α, β, γ:

        Area = (π - (α + β + γ)) / (1/c²) = c² δ

    where δ = π - (α + β + γ) is the defect.

    Reference: Gauss-Bonnet theorem on constant curvature manifolds.
    """

    def __init__(self, c: float = 1.0):
        self.c = c
        self.curvature = -1.0 / (c**2)

    def compute_hyperbolic_area(self, angles: Tuple[float, float, float]) -> float:
        """
        Compute hyperbolic area from angles using Gauss-Bonnet.

        Args:
            angles: Tuple of (α, β, γ) in radians

        Returns:
            Hyperbolic area = c² × defect
        """
        alpha, beta, gamma = angles
        defect = np.pi - (alpha + beta + gamma)
        area = defect / abs(self.curvature)  # = c² × defect
        return float(area)

    def verify_defect_area_equivalence(
        self, computed_defect: float, angles: Tuple[float, float, float]
    ) -> Dict[str, Any]:
        """
        Verify that the computed defect equals the hyperbolic area.

        Args:
            computed_defect: Defect computed by gyrotriangle logic
            angles: Tuple of (α, β, γ) in radians

        Returns:
            Verification results
        """
        hyperbolic_area = self.compute_hyperbolic_area(angles)
        defect_from_area = hyperbolic_area / (self.c**2)

        # The defect should equal π - (α + β + γ)
        theoretical_defect = np.pi - sum(angles)

        # Verify all three are consistent
        defect_consistency = np.isclose(computed_defect, defect_from_area, rtol=1e-10)
        area_consistency = np.isclose(computed_defect, theoretical_defect, rtol=1e-10)

        return {
            "computed_defect": computed_defect,
            "hyperbolic_area": hyperbolic_area,
            "defect_from_area": defect_from_area,
            "theoretical_defect": theoretical_defect,
            "defect_area_consistent": defect_consistency,
            "theoretical_consistent": area_consistency,
            "all_consistent": defect_consistency and area_consistency,
            "note": "Gauss-Bonnet: defect = area/c² for constant curvature -1/c²",
        }

    def test_closure_case(self) -> Dict[str, Any]:
        """
        Test the special closure case: α = π/2, β = γ = π/4.
        This should give zero defect (degenerate triangle).
        """
        angles = (np.pi / 2, np.pi / 4, np.pi / 4)  # 90°, 45°, 45°
        defect = np.pi - sum(angles)
        area = self.compute_hyperbolic_area(angles)

        return {
            "angles_deg": [np.degrees(a) for a in angles],
            "defect": defect,
            "hyperbolic_area": area,
            "is_zero_defect": np.isclose(defect, 0.0, atol=1e-12),
            "note": "Zero defect ⇒ zero hyperbolic area (Euclidean-limit case)",
        }


class ThomasWignerRotation:
    """
    Theorem: The gyration gyr[a,b] for small velocities is a pure rotation.

    Proof: In the Lorentz algebra so(3,1), two small boosts compose via BCH:

        e^(εu·K) e^(εv·K) = e^(ε(u+v)·K) e^(-ε²/2 (u×v)·J) + O(ε³)

    The noncommutativity yields a pure rotation about u×v with angle
    ε²/2 ||u×v||/c². Hence the Jacobian of gyr[a,b] at the origin
    lies in SO(3) with generator proportional to (u×v)/c².
    """

    def __init__(self, c: float = 1.0):
        self.c = c

    def compute_bch_rotation(
        self, u: np.ndarray, v: np.ndarray, eps: float
    ) -> np.ndarray:
        """
        Compute the BCH rotation matrix for small velocities.

        Args:
            u, v: Velocity vectors
            eps: Small parameter for expansion

        Returns:
            Rotation matrix R ≈ I + ε²(u×v)/c²
        """
        u_cross_v = np.cross(u, v)
        cross_norm = np.linalg.norm(u_cross_v)

        if cross_norm < 1e-12:
            return np.eye(3)

        # BCH rotation angle: ε²/2 ||u×v||/c²
        angle = (eps**2 / 2) * cross_norm / (self.c**2)

        if angle < 1e-12:
            return np.eye(3)

        # Rotation axis: u×v / ||u×v||
        axis = u_cross_v / cross_norm

        # Rodrigues' rotation formula for small angles
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )

        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R

    def verify_rotation_properties(self, R: np.ndarray) -> Dict[str, Any]:
        """
        Verify that R is a proper rotation matrix.

        Args:
            R: Matrix to verify

        Returns:
            Verification results
        """
        # Check orthogonality: R^T R = I
        orthogonality_error = np.linalg.norm(R.T @ R - np.eye(3))

        # Check determinant: det(R) = 1
        determinant = np.linalg.det(R)
        det_error = abs(determinant - 1.0)

        # Check trace: tr(R) = 1 + 2cos(θ)
        trace = np.trace(R)
        cos_theta = (trace - 1.0) / 2.0
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        return {
            "is_orthogonal": orthogonality_error < 1e-10,
            "orthogonality_error": orthogonality_error,
            "is_rotation": det_error < 1e-10,
            "determinant": determinant,
            "trace": trace,
            "rotation_angle": angle,
            "note": "R should be orthogonal with det=1 and tr=1+2cos(θ)",
        }


class BUAmplitudeIdentity:
    """
    Theorem: The BU amplitude threshold A satisfies A² × (2π)² = π/2.

    Proof: Given the closure condition A² × (2π)_L × (2π)_R = π/2
    with (2π)_L = (2π)_R = 2π, we have:

        A² × (2π)² = π/2
        A² = π/2 / (2π)² = 1/(8π)
        A = 1/(2√(2π))

    This is exactly what the code returns.
    """

    def __init__(self):
        self.theoretical_amplitude = 1.0 / (2.0 * np.sqrt(2.0 * np.pi))

    def verify_identity(self, computed_amplitude: float) -> Dict[str, Any]:
        """
        Verify that the computed amplitude satisfies the identity.

        Args:
            computed_amplitude: Amplitude from BU stage

        Returns:
            Verification results
        """
        # Check the identity: A² × (2π)² = π/2
        lhs = computed_amplitude**2 * (2 * np.pi) ** 2
        rhs = np.pi / 2

        identity_satisfied = np.isclose(lhs, rhs, rtol=1e-10)

        return {
            "computed_amplitude": computed_amplitude,
            "theoretical_amplitude": self.theoretical_amplitude,
            "lhs": lhs,
            "rhs": rhs,
            "identity_satisfied": identity_satisfied,
            "relative_error": abs(lhs - rhs) / rhs,
            "note": "A² × (2π)² = π/2 should hold exactly",
        }
