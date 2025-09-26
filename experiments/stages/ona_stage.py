"""
ONA Stage (Opposition Non-Absolute) - Opposition is Non-Absolute

Full differentiation is achieved with both gyrations maximally non-identity.
The system reaches peak non-associativity, generating complete structural framework.
"""

import sys
import os

import numpy as np
from typing import Dict, Any, Tuple, List
from ..functions.gyrovector_ops import GyroVectorSpace


class ONAStage:
    """
    Opposition Non-Absolute (ONA) Stage Implementation

    Characteristics:
    - Right gyration: non-identity (maximal)
    - Left gyration: non-identity (maximal)
    - Degrees of freedom: 6 (3 rotational + 3 translational)
    - Threshold: γ = π/4, ratio oₚ = π/4
    """

    def __init__(self, gyrospace: GyroVectorSpace):
        self.gyrospace = gyrospace
        self.stage_name = "ONA"
        self.angle = np.pi / 4
        self.threshold_ratio = np.pi / 4

        # ONA properties
        self.left_gyration_active = True
        self.right_gyration_active = True
        self.degrees_of_freedom = 6

    def get_stage_properties(self) -> Dict[str, Any]:
        """Get the defining properties of ONA stage"""
        return {
            "stage": self.stage_name,
            "angle": self.angle,
            "threshold_ratio": self.threshold_ratio,
            "left_gyration": (
                "non-identity" if self.left_gyration_active else "identity"
            ),
            "right_gyration": (
                "non-identity" if self.right_gyration_active else "identity"
            ),
            "dof": self.degrees_of_freedom,
            "operation": "Bi-gyroassociativity",
            "governing_law": "Full gyrogroup operations with maximal non-associativity",
        }

    def bi_gyroassociativity_check(
        self, a: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> Tuple[float, float]:
        """
        Check both left and right gyroassociativity at ONA

        Left: a ⊕ (b ⊕ c) = (a ⊕ b) ⊕ gyr[a,b]c
        Right: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ gyr[b,a]c)

        Args:
            a, b, c: Test vectors

        Returns:
            Tuple of (left_defect, right_defect)
        """
        # Left associativity
        left_assoc, left_defect = self.gyrospace.gyroassociativity_check(a, b, c)

        # Right associativity (reverse order)
        right_assoc, right_defect = self.gyrospace.gyroassociativity_check(c, b, a)

        return left_defect, right_defect

    def translational_dof_activation(self) -> np.ndarray:
        """
        Generate the three translational degrees of freedom activated at ONA

        Returns:
            3x3 matrix of translational basis vectors
        """
        # Translational DoF emerge from the second π/4 rotation
        # These are perpendicular to the rotational axes from UNA
        cos_angle = np.cos(self.angle)
        sin_angle = np.sin(self.angle)

        # Translational basis (diagonal tilt from UNA plane)
        trans_basis = np.array(
            [
                [cos_angle, 0, sin_angle],  # x-translation with tilt
                [0, cos_angle, 0],  # y-translation
                [-sin_angle, 0, cos_angle],  # z-translation with tilt
            ]
        )

        return trans_basis

    def full_su2_activation(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate both SU(2)_L and SU(2)_R groups at full activation

        Returns:
            Tuple of (left_su2_generators, right_su2_generators)
        """
        # Left SU(2) (from CS through UNA)
        left_sigma_x = np.array([[0, 1], [1, 0]])
        left_sigma_y = np.array([[0, -1j], [1j, 0]])
        left_sigma_z = np.array([[1, 0], [0, -1]])

        # Right SU(2) (activated at UNA, full at ONA)
        right_sigma_x = np.array([[0, 1], [1, 0]])
        right_sigma_y = np.array([[0, -1j], [1j, 0]])
        right_sigma_z = np.array([[1, 0], [0, -1]])

        left_generators = [left_sigma_x, left_sigma_y, left_sigma_z]
        right_generators = [right_sigma_x, right_sigma_y, right_sigma_z]

        return left_generators, right_generators

    def su2_frame_generation(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate the SU(2)_L and SU(2)_R frames for rotational degrees of freedom

        Returns:
            Tuple of (left_generators, right_generators)
        """
        # Pauli matrices for SU(2) representation
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])

        left_generators = [sigma_x, sigma_y, sigma_z]
        right_generators = [sigma_x, sigma_y, sigma_z]

        return left_generators, right_generators

    def non_associativity_peak(self, test_vectors: list) -> float:
        """
        Measure the peak non-associativity at ONA

        Args:
            test_vectors: List of vectors to test

        Returns:
            Peak non-associativity measure
        """
        peak_associativity = 0.0

        for a in test_vectors:
            for b in test_vectors:
                for c in test_vectors:
                    if (
                        np.linalg.norm(a) > 0
                        and np.linalg.norm(b) > 0
                        and np.linalg.norm(c) > 0
                    ):
                        left_defect, right_defect = self.bi_gyroassociativity_check(
                            a, b, c
                        )
                        peak_associativity += (left_defect + right_defect) / 2

        return peak_associativity / len(test_vectors) ** 3

    def opposition_non_absolute_measure(self, opposing_vectors: list) -> float:
        """
        Measure the non-absoluteness of opposition

        Args:
            opposing_vectors: List of vector pairs representing oppositions

        Returns:
            Non-absoluteness measure (0 = absolute opposition, higher = more non-absolute)
        """
        non_absoluteness = 0.0

        for pair in opposing_vectors:
            u, v = pair

            # Test various compositions
            u_plus_v = self.gyrospace.gyroaddition(u, v)
            v_plus_u = self.gyrospace.gyroaddition(v, u)

            # Opposition should not be absolute (some residual connection)
            den = np.linalg.norm(u_plus_v) * np.linalg.norm(v_plus_u)
            if den < 1e-12:
                residual_connection = 0.0
            else:
                residual_connection = np.dot(u_plus_v, v_plus_u) / den

            non_absoluteness += 1.0 - abs(residual_connection)

        return non_absoluteness / len(opposing_vectors)

    def diagonal_tilt_angle(self) -> float:
        """
        Compute the diagonal tilt angle required for 3D translation

        Returns:
            Diagonal tilt angle γ = π/4
        """
        return self.angle

    def transition_to_bu(self):
        """
        Transition from ONA to BU stage

        Returns:
            BUStage instance
        """
        from bu_stage import BUStage

        return BUStage(self.gyrospace)

    def get_closure_constraint(self) -> float:
        """
        Get the closure constraint contribution from ONA

        Returns:
            ONA contribution to global closure: γ = π/4
        """
        return float(self.angle)

    def monodromy_measure(self, loop_points: list) -> float:
        """
        Measure the monodromy around a loop at ONA

        Args:
            loop_points: Points defining a closed loop

        Returns:
            Monodromy magnitude
        """
        if len(loop_points) < 3:
            return 0.0

        # Compute ordered product of gyrations around the loop
        monodromy = np.eye(3)

        for i in range(len(loop_points) - 1):
            p1 = loop_points[i]
            p2 = loop_points[i + 1]

            # Gyration between consecutive points
            gyr = self.gyrospace.gyration(p1, p2)

            # Ensure gyr is a proper 3x3 matrix
            if not (hasattr(gyr, "shape") and gyr.shape == (3, 3)):
                raise TypeError("gyration must be a 3x3 matrix")

            monodromy = monodromy @ gyr

        # Close the loop
        p_last = loop_points[-1]
        p_first = loop_points[0]
        final_gyr = self.gyrospace.gyration(p_last, p_first)

        # Ensure final_gyr is a proper 3x3 matrix
        if not (hasattr(final_gyr, "shape") and final_gyr.shape == (3, 3)):
            raise TypeError("gyration must be a 3x3 matrix")

        monodromy = monodromy @ final_gyr

        return float(np.linalg.norm(monodromy - np.eye(3)))
