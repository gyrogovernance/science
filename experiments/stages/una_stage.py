"""
UNA Stage (Unity Non-Absolute) - Unity is Non-Absolute

Observable structure emerges when right gyration activates while left gyration persists.
This creates the minimal asymmetry required for observation while preserving left-bias.
"""

import sys
import os

import numpy as np
from typing import Dict, Any, Tuple, List

from ..functions.gyrovector_ops import GyroVectorSpace


class UNAStage:
    """
    Unity Non-Absolute (UNA) Stage Implementation

    Characteristics:
    - Right gyration: non-identity (newly activated)
    - Left gyration: non-identity (persisting from CS)
    - Degrees of freedom: 3 (rotational)
    - Threshold: β = π/4, ratio uₚ = 1/√2
    """

    def __init__(self, gyrospace: GyroVectorSpace):
        self.gyrospace = gyrospace
        self.stage_name = "UNA"
        self.angle = np.pi / 4
        self.threshold_ratio = 1.0 / np.sqrt(2)

        # UNA properties
        self.left_gyration_active = True
        self.right_gyration_active = True  # Newly activated!
        self.degrees_of_freedom = 3

    def get_stage_properties(self) -> Dict[str, Any]:
        """Get the defining properties of UNA stage"""
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
            "operation": "Gyrocommutativity",
            "governing_law": "a ⊕ b = gyr[a,b](b ⊕ a)",
        }

    def gyrocommutativity_check(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Check gyrocommutativity: u ⊕ v ?= gyr[u,v](v ⊕ u)

        Args:
            u, v: Test vectors

        Returns:
            Defect magnitude (0 = perfect commutativity)
        """
        _, defect = self.gyrospace.gyrocommutativity_check(u, v)
        return defect

    def orthogonal_spin_axes(self) -> np.ndarray:
        """
        Generate the three orthogonal spin axes that emerge at UNA

        Returns:
            3x3 matrix of orthogonal axes
        """
        # The three axes emerge from cos(π/4) = 1/√2 relationships
        cos_angle = np.cos(self.angle)
        sin_angle = np.sin(self.angle)

        axes = np.array(
            [
                [1, 0, 0],  # x-axis
                [0, cos_angle, -sin_angle],  # y-axis rotated
                [0, sin_angle, cos_angle],  # z-axis rotated
            ]
        )

        return axes

    def su2_frame_generation(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate the SU(2)_L frame for rotational degrees of freedom

        Returns:
            Tuple of (left_generators, right_generators)
        """
        # Pauli matrices for SU(2) representation
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])

        left_generators = [sigma_x, sigma_y, sigma_z]

        # Right generators (activated at UNA)
        right_generators = [sigma_x.copy(), sigma_y.copy(), sigma_z.copy()]

        return left_generators, right_generators

    def observable_distinction_measure(self, test_vectors: List[np.ndarray]) -> float:
        """
        Measure the observable distinction that emerges at UNA

        Args:
            test_vectors: List of vectors to test

        Returns:
            Distinction measure (higher = more observable structure)
        """
        distinction_sum = 0.0

        for i, u in enumerate(test_vectors):
            for j, v in enumerate(test_vectors):
                if i != j:
                    # Compare u⊕v vs v⊕u (now shows observable difference)
                    u_plus_v = self.gyrospace.gyroaddition(u, v)
                    v_plus_u = self.gyrospace.gyroaddition(v, u)
                    gyr_v = self.gyrospace.gyr_apply(u, v, v_plus_u)
                    distinction_sum += np.linalg.norm(
                        u_plus_v - v_plus_u
                    ) + np.linalg.norm(u_plus_v - gyr_v)

        return float(distinction_sum / len(test_vectors) ** 2)

    def chiral_memory_preservation(self) -> float:
        """
        Measure how well the primordial left-bias is preserved

        Returns:
            Preservation measure (1.0 = perfectly preserved)
        """
        # Test vectors with different chiralities
        left_vector = np.array([1, 0, 0])
        right_vector = np.array([-1, 0, 0])

        # Apply operations and check if left-bias is maintained
        left_result = self.gyrospace.gyroaddition(left_vector, right_vector)
        right_result = self.gyrospace.gyroaddition(right_vector, left_vector)

        # The difference should be non-zero (distinction) but asymmetric (left-bias)
        den = np.linalg.norm(left_result - right_result)
        if den < 1e-12:
            return 1.0
        bias_measure = np.linalg.norm(left_result + right_result) / den

        return float(1.0 / (1.0 + bias_measure))  # Normalize to [0,1]

    def transition_to_ona(self):
        """
        Transition from UNA to ONA stage

        Returns:
            ONAStage instance
        """
        from ona_stage import ONAStage

        return ONAStage(self.gyrospace)

    def get_closure_constraint(self) -> float:
        """
        Get the closure constraint contribution from UNA

        Returns:
            UNA contribution to global closure: β = π/4
        """
        return float(self.angle)
