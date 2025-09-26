"""
BU Stage (Balance Universal) - Balance is Universal

After differentiation, the system reaches self-consistent configuration where
non-associativity cancels globally while recursive memory is preserved.
"""

import sys
import os

import numpy as np
from typing import Dict, Any, Tuple
from ..functions.gyrovector_ops import GyroVectorSpace, RecursivePath


class BUStage:
    """
    Balance Universal (BU) Stage Implementation

    Characteristics:
    - Right gyration: identity (rgyr = id)
    - Left gyration: identity (lgyr = id)
    - Degrees of freedom: 6 (stabilized)
    - Threshold: δ = 0, amplitude mₚ = 1/(2√(2π))
    """

    def __init__(self, gyrospace: GyroVectorSpace):
        self.gyrospace = gyrospace
        self.stage_name = "BU"
        self.defect = 0.0
        self.amplitude_threshold = 1.0 / (2.0 * np.sqrt(2.0 * np.pi))

        # BU properties
        self.left_gyration_active = False  # Returns to identity
        self.right_gyration_active = False  # Returns to identity
        self.degrees_of_freedom = 6  # Stabilized

    def get_stage_properties(self) -> Dict[str, Any]:
        """Get the defining properties of BU stage"""
        return {
            "stage": self.stage_name,
            "defect": self.defect,
            "amplitude_threshold": self.amplitude_threshold,
            "left_gyration": "identity",
            "right_gyration": "identity",
            "dof": self.degrees_of_freedom,
            "operation": "Coaddition",
            "governing_law": "a ⊞ b = a ⊕ gyr[a,⊖b]b (commutative & associative)",
        }

    def coaddition_check(self, u: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
        """
        Check coaddition properties at BU stage

        Args:
            u, v: Test vectors

        Returns:
            Tuple of (commutativity_defect, associativity_defect)
        """
        # Test commutativity: u ⊞ v ?= v ⊞ u
        left_comm = self.gyrospace.coaddition(u, v)
        right_comm = self.gyrospace.coaddition(v, u)
        commutativity_defect = np.linalg.norm(left_comm - right_comm)

        # Test associativity: (u ⊞ v) ⊞ w ?= u ⊞ (v ⊞ w)
        # Use a fixed small probe to make runs deterministic
        w = np.array([1e-2, -2e-2, 1.5e-2])  # Fixed small test vector
        left_assoc = self.gyrospace.coaddition(self.gyrospace.coaddition(u, v), w)
        right_assoc = self.gyrospace.coaddition(u, self.gyrospace.coaddition(v, w))

        associativity_defect = np.linalg.norm(left_assoc - right_assoc)

        return float(commutativity_defect), float(associativity_defect)

    def recursive_memory_preservation(self, recursive_path: RecursivePath) -> float:
        """
        Measure how well recursive memory is preserved at BU

        Args:
            recursive_path: Path with accumulated memory

        Returns:
            Memory preservation measure
        """
        # At BU, memory should be preserved even as gyrations become identity
        memory_matrix = recursive_path.get_recursive_memory()

        # Memory preservation is the trace of the memory matrix
        # (should be non-zero even when gyrations are identity)
        preservation = np.trace(memory_matrix) / memory_matrix.shape[0]

        return preservation

    def global_closure_verification(self) -> bool:
        """
        Verify that BU achieves global closure

        Returns:
            True if closure conditions are satisfied
        """
        # Check amplitude constraint: A² × (2π)_L × (2π)_R = π/2
        left_angle_range = 2 * np.pi  # (2π)_L
        right_angle_range = 2 * np.pi  # (2π)_R

        lhs = (self.amplitude_threshold**2) * left_angle_range * right_angle_range
        rhs = np.pi / 2

        return abs(lhs - rhs) < 1e-10

    def algebraic_memory_ranges(self) -> Tuple[float, float]:
        """
        Get the latent angular memory ranges preserved at BU

        Returns:
            Tuple of (left_memory_range, right_memory_range)
        """
        return (2.0 * np.pi, 2.0 * np.pi)  # (2π)_L, (2π)_R

    def topological_realization(self) -> str:
        """
        Get the global topology realized at BU

        Returns:
            Topology description: S³ × S³
        """
        return "S³ × S³ (rotational and translational memories)"

    def gravitational_field_computation(
        self, x: np.ndarray, recursive_paths: list
    ) -> np.ndarray:
        """
        Compute gravitational field from residual coherence failure:
        G(x) = ∇ arg[∏(i∈N(x)) gyr[a_i,b_i]]

        Args:
            x: Position vector
            recursive_paths: List of recursive paths in neighborhood

        Returns:
            Gravitational field vector
        """
        if not recursive_paths:
            return np.zeros_like(x)

        total_gyration = np.eye(3)
        for path in recursive_paths:
            total_gyration = total_gyration @ path.get_recursive_memory()

        # rotation angle from trace: tr(R) = 1 + 2 cos θ
        tr = np.trace(total_gyration)
        cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
        angle = np.arccos(cos_theta)

        # radial falloff – avoid division by zero at origin
        r = np.linalg.norm(x)
        if r < 1e-12:
            return np.zeros_like(x)

        # simple radial gradient proxy
        return (angle / r) * (x / r)

    def infinity_as_phase_flattening(
        self, recursive_field: np.ndarray, saturation_depth: float
    ) -> bool:
        """
        Check for recursive infinity (loss of resolution due to phase flattening)

        Args:
            recursive_field: Coherence field ψ_rec
            saturation_depth: Saturation depth ℓ*

        Returns:
            True if infinity condition is met
        """
        # lim ℓ→ℓ* ||∇arg(ψ_rec)|| → 0
        gradient_norm = np.linalg.norm(np.gradient(recursive_field))

        return bool(gradient_norm < 1e-6)  # Threshold for "flattening"

    def singularity_as_memory_failure(self, loop_monodromy: np.ndarray) -> bool:
        """
        Check for recursive singularity (memory cannot store torsion)

        Args:
            loop_monodromy: Monodromy around a loop

        Returns:
            True if singularity condition is met
        """
        # ||μ(M_ℓ)|| → ∞ but ψ_rec(ℓ) → 0
        monodromy_norm = np.linalg.norm(loop_monodromy)
        coherence_field = np.exp(-monodromy_norm)  # Simplified model

        return bool(monodromy_norm > 1e6 and coherence_field < 1e-6)

    def verify_universal_balance(self, test_configurations: list) -> float:
        """
        Verify that balance is universal across configurations

        Args:
            test_configurations: List of test configurations

        Returns:
            Universal balance measure (1.0 = perfectly balanced)
        """
        balance_measures = []

        for config in test_configurations:
            # Test coaddition properties
            vectors = config.get("vectors", [np.random.rand(3) for _ in range(3)])

            total_commutativity = 0.0
            total_associativity = 0.0

            for i, u in enumerate(vectors):
                for j, v in enumerate(vectors):
                    if i != j:
                        comm_def, assoc_def = self.coaddition_check(u, v)
                        total_commutativity += comm_def
                        total_associativity += assoc_def

            n_pairs = len(vectors) * (len(vectors) - 1)
            config_balance = 1.0 / (
                1.0 + (total_commutativity + total_associativity) / n_pairs
            )
            balance_measures.append(config_balance)

        return float(np.mean(balance_measures))

    def get_closure_constraint(self) -> float:
        """
        Get the closure constraint contribution from BU

        Returns:
            BU contribution to global closure: δ = 0
        """
        return float(self.defect)
