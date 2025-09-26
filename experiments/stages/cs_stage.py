"""
CS Stage (Common Source) - The Source is Common

The originating condition containing inherent chirality and directionality.
CS is freedom itself - not mere potential but the active source of parity violation.
"""

import sys
import os

import numpy as np
from typing import Dict, Any
from ..functions.gyrovector_ops import GyroVectorSpace


class CSStage:
    """
    Common Source (CS) Stage Implementation

    Characteristics:
    - Right gyration: identity (rgyr = id)
    - Left gyration: non-identity (lgyr ≠ id)
    - Degrees of freedom: 1 (chiral seed)
    - Threshold: α = π/2
    """

    def __init__(self, gyrospace: GyroVectorSpace):
        self.gyrospace = gyrospace
        self.stage_name = "CS"
        self.angle = np.pi / 2
        self.threshold_ratio = np.pi / 2

        # CS properties
        self.left_gyration_active = True
        self.right_gyration_active = False
        self.degrees_of_freedom = 1

    def get_stage_properties(self) -> Dict[str, Any]:
        """Get the defining properties of CS stage"""
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
            "operation": "Left gyroassociativity",
            "governing_law": "a ⊕ (b ⊕ c) = (a ⊕ b) ⊕ gyr[a,b]c",
        }

    def left_gyroassociativity_check(
        self, a: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> float:
        """
        Check left gyroassociativity: a ⊕ (b ⊕ c) = (a ⊕ b) ⊕ gyr[a,b]c

        Args:
            a, b, c: Test vectors

        Returns:
            Defect magnitude (0 = perfect associativity)
        """
        is_assoc, defect = self.gyrospace.gyroassociativity_check(a, b, c)
        return defect

    def chiral_asymmetry_measure(self, test_vectors: list) -> float:
        """
        Measure the inherent chiral asymmetry of CS

        Args:
            test_vectors: List of vectors to test

        Returns:
            Asymmetry measure (higher = more asymmetric)
        """
        asymmetry_sum = 0.0

        for i, u in enumerate(test_vectors):
            for j, v in enumerate(test_vectors):
                if i != j:
                    # Compare u⊕v vs v⊕u (should be different due to left bias)
                    u_plus_v = self.gyrospace.gyroaddition(u, v)
                    v_plus_u = self.gyrospace.gyroaddition(v, u)
                    asymmetry_sum += np.linalg.norm(u_plus_v - v_plus_u)

        return float(asymmetry_sum / len(test_vectors) ** 2)

    def primordial_gyration_field(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the primordial gyration field at CS

        Args:
            x: Position vector

        Returns:
            Local gyration matrix
        """
        # At CS, only left gyration is active
        # Simplified model: rotation around z-axis with angle proportional to |x|
        angle = self.angle * np.linalg.norm(x)
        return np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )

    def transition_to_una(self):
        """
        Transition from CS to UNA stage

        Returns:
            UNAStage instance
        """
        from una_stage import UNAStage

        return UNAStage(self.gyrospace)

    def is_unobservable_directly(self) -> bool:
        """
        CS is necessarily unobservable directly by logical necessity

        Returns:
            True (CS cannot be observed directly)
        """
        return True

    def get_closure_constraint(self) -> float:
        """
        Get the closure constraint contribution from CS

        Returns:
            CS contribution to global closure: α = π/2
        """
        return float(self.angle)
