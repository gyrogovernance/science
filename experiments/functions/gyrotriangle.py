"""
Gyrotriangle implementation for CGM

Implements the gyrotriangle defect calculations and closure conditions
that are fundamental to the CGM stage theorems.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Union

try:
    from .gyrovector_ops import GyroVectorSpace, RecursivePath
except ImportError:
    # Fallback for direct execution
    from gyrovector_ops import GyroVectorSpace, RecursivePath
from numpy.typing import NDArray


class GyroTriangle:
    """
    Represents a gyrotriangle in gyrovector space with defect calculations

    The gyrotriangle defect δ = π - (α + β + γ) is central to CGM closure.
    """

    def __init__(self, gyrospace: GyroVectorSpace):
        self.gyrospace = gyrospace

    def compute_defect(self, alpha: float, beta: float, gamma: float) -> float:
        """
        Compute the gyrotriangle defect: δ = π - (α + β + γ)

        Args:
            alpha: CS angle (π/2)
            beta: UNA angle (π/4)
            gamma: ONA angle (π/4)

        Returns:
            Defect δ in radians
        """
        return np.pi - (alpha + beta + gamma)

    def is_closed(
        self, alpha: float, beta: float, gamma: float, tolerance: float = 1e-10
    ) -> bool:
        """
        Check if gyrotriangle closes (δ = 0)

        Args:
            alpha, beta, gamma: Triangle angles
            tolerance: Numerical tolerance

        Returns:
            True if defect is within tolerance
        """
        defect = self.compute_defect(alpha, beta, gamma)
        return abs(defect) < tolerance

    def cgm_standard_angles(self) -> Tuple[float, float, float]:
        """
        Return the standard CGM angles: (π/2, π/4, π/4)

        Returns:
            Tuple of (alpha, beta, gamma)
        """
        return (np.pi / 2, np.pi / 4, np.pi / 4)

    def check_cgm_closure(self) -> Tuple[bool, float]:
        """
        Check if standard CGM angles achieve closure

        Returns:
            Tuple of (is_closed, defect)
        """
        alpha, beta, gamma = self.cgm_standard_angles()
        defect = self.compute_defect(alpha, beta, gamma)
        return abs(defect) < 1e-10, defect

    def side_parameters_from_angles(
        self, alpha: float, beta: float, gamma: float, a_s: float, b_s: float
    ) -> float:
        """
        Compute third side parameter using defect formula:
        tan(δ/2) = (a_s * b_s * sin(γ)) / (1 - a_s * b_s * cos(γ))

        Args:
            alpha, beta, gamma: Triangle angles
            a_s, b_s: Two side parameters

        Returns:
            Third side parameter c_s
        """
        defect = self.compute_defect(alpha, beta, gamma)
        tan_half_defect = np.tan(defect / 2)
        if abs(tan_half_defect) < 1e-12:
            return float("inf")  # or np.nan, depending on how we want to flag closure

        numerator = a_s * b_s * np.sin(gamma)
        denominator = 1 - a_s * b_s * np.cos(gamma)

        if abs(denominator) < 1e-10:
            return float("inf")  # Degenerate case

        c_s = (numerator / denominator) / tan_half_defect
        return c_s

    def angles_from_sides(
        self, a_s: float, b_s: float, c_s: float
    ) -> Tuple[float, float, float]:
        """
        Compute angles from side parameters using inverse defect formula

        Args:
            a_s, b_s, c_s: Side parameters

        Returns:
            Tuple of (alpha, beta, gamma)
        """
        # Use law of cosines in gyrovector space
        # α = acos((b_s² + c_s² - a_s²) / (2 b_s c_s))
        cos_alpha = (b_s**2 + c_s**2 - a_s**2) / (2 * b_s * c_s)
        cos_beta = (a_s**2 + c_s**2 - b_s**2) / (2 * a_s * c_s)
        cos_gamma = (a_s**2 + b_s**2 - c_s**2) / (2 * a_s * b_s)

        # Clamp to avoid numerical issues
        cos_alpha = np.clip(cos_alpha, -1, 1)
        cos_beta = np.clip(cos_beta, -1, 1)
        cos_gamma = np.clip(cos_gamma, -1, 1)

        alpha = np.arccos(cos_alpha)
        beta = np.arccos(cos_beta)
        gamma = np.arccos(cos_gamma)

        return alpha, beta, gamma

    def recursive_closure_amplitude(self) -> float:
        """
        Compute the BU amplitude threshold from closure condition:
        A² × (2π)_L × (2π)_R = π/2 ⇒ A = 1/(2√(2π))

        Returns:
            Amplitude threshold m_p = 1/(2√(2π))
        """
        return 1.0 / (2.0 * np.sqrt(2.0 * np.pi))

    def verify_defect_asymmetry(self, sign_factor: float = 1.0) -> Tuple[float, float]:
        """
        Test defect asymmetry: positive vs negative angle sequences

        Args:
            sign_factor: +1 for positive sequence, -1 for negative

        Returns:
            Tuple of (positive_defect, negative_defect)
        """
        # Positive sequence: (+π/2, +π/4, +π/4) - left-initiated path
        pos_alpha, pos_beta, pos_gamma = (
            sign_factor * np.pi / 2,
            sign_factor * np.pi / 4,
            sign_factor * np.pi / 4,
        )
        pos_defect = self.compute_defect(pos_alpha, pos_beta, pos_gamma)

        # Negative sequence: (-π/2, -π/4, -π/4) - right-initiated path
        neg_alpha, neg_beta, neg_gamma = (
            -sign_factor * np.pi / 2,
            -sign_factor * np.pi / 4,
            -sign_factor * np.pi / 4,
        )
        neg_defect = self.compute_defect(neg_alpha, neg_beta, neg_gamma)

        return pos_defect, neg_defect

    def compute_enhanced_defect_asymmetry(self) -> Dict[str, float]:
        """
        Compute enhanced defect asymmetry analysis

        Returns:
            Dictionary with detailed asymmetry analysis
        """
        # Test multiple sequences to understand asymmetry
        sequences = {
            "left_initiated": (np.pi / 2, np.pi / 4, np.pi / 4),  # CS → UNA → ONA → BU
            "right_initiated": (
                -np.pi / 2,
                -np.pi / 4,
                -np.pi / 4,
            ),  # Attempted right-precedence
            "mixed_positive": (np.pi / 2, -np.pi / 4, np.pi / 4),  # Mixed signs
            "mixed_negative": (-np.pi / 2, np.pi / 4, -np.pi / 4),  # Mixed signs
        }

        results = {}
        for name, (alpha, beta, gamma) in sequences.items():
            defect = self.compute_defect(alpha, beta, gamma)
            results[name] = defect

        # Compute asymmetry measures
        results["asymmetry_measure"] = (
            results["right_initiated"] - results["left_initiated"]
        )
        results["positive_vs_negative"] = results["left_initiated"] - abs(
            results["right_initiated"]
        )

        return results

    def find_closure_thresholds(
        self, tolerance: float = 1e-10, search_range: float = 0.1
    ) -> List[Tuple[float, float, float]]:
        """
        Search for angle combinations that achieve closure δ = 0

        Args:
            tolerance: Defect tolerance
            search_range: Search range around CGM values

        Returns:
            List of (alpha, beta, gamma) tuples that achieve closure
        """
        closed_triangles = []

        # Search around CGM standard values
        alpha_0, beta_0, gamma_0 = self.cgm_standard_angles()

        # Grid search
        n_points = 50
        alpha_range = np.linspace(
            alpha_0 - search_range, alpha_0 + search_range, n_points
        )
        beta_range = np.linspace(beta_0 - search_range, beta_0 + search_range, n_points)
        gamma_range = np.linspace(
            gamma_0 - search_range, gamma_0 + search_range, n_points
        )

        for alpha in alpha_range:
            for beta in beta_range:
                for gamma in gamma_range:
                    # Check ordering constraint: α ≥ β ≥ γ
                    if alpha >= beta >= gamma:
                        defect = self.compute_defect(alpha, beta, gamma)
                        if abs(defect) < tolerance:
                            closed_triangles.append((alpha, beta, gamma))

        return closed_triangles

    def compute_recursive_memory_defect(self, path: RecursivePath) -> float:
        """
        Compute defect from recursive path memory

        Args:
            path: RecursivePath object with accumulated memory

        Returns:
            Memory-based defect measure
        """
        memory_matrix = path.get_recursive_memory()

        # Defect is related to deviation from identity
        identity = np.eye(memory_matrix.shape[0])
        defect = np.linalg.norm(memory_matrix - identity)

        return float(defect)

    def stage_transition_matrix(self, from_stage: str, to_stage: str) -> np.ndarray:
        """
        Compute gyration transition matrix between CGM stages

        Args:
            from_stage: Starting stage ('CS', 'UNA', 'ONA', 'BU')
            to_stage: Ending stage ('CS', 'UNA', 'ONA', 'BU')

        Returns:
            Transition gyration matrix
        """
        stage_angles = {"CS": np.pi / 2, "UNA": np.pi / 4, "ONA": np.pi / 4, "BU": 0.0}

        if from_stage not in stage_angles or to_stage not in stage_angles:
            raise ValueError(f"Unknown stage: {from_stage} or {to_stage}")

        # Transition gyration based on angle difference
        angle_diff = stage_angles[to_stage] - stage_angles[from_stage]

        # Simple rotation matrix for demonstration (3D case)
        rotation = np.array(
            [
                [np.cos(angle_diff), -np.sin(angle_diff), 0],
                [np.sin(angle_diff), np.cos(angle_diff), 0],
                [0, 0, 1],
            ]
        )

        return rotation
