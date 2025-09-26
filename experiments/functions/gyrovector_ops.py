"""
Core gyrovector operations for CGM

This module implements the fundamental gyrovector space operations based on
Einstein-Ungar formalism and the recursive principles of CGM.

The gyrovector space provides a non-Euclidean geometry where:
- Addition is non-commutative and non-associative
- Gyration matrices encode the non-associativity
- Small velocities approach classical vector addition
- Large velocities exhibit relativistic effects

Based on the mathematical formalism presented in the CGM framework.

Classes:
    GyroVectorSpace: Main gyrovector space implementation
"""

import numpy as np
import warnings
from typing import Tuple, Optional, Union, List, Callable, Any


try:
    from scipy.linalg import polar

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    polar = None


class GyroVectorSpace:
    """
    Einstein-Ungar gyrovector space implementation for CGM

    A gyrovector space (V, ⊕, ⊖, gyr) with speed parameter c.
    """

    def __init__(self, c: float = 1.0):
        """
        Initialize gyrovector space with speed parameter c

        Args:
            c: Speed parameter (typically c = speed of light in natural units)
        """
        self.c = c

    def gamma(self, v: np.ndarray) -> float:
        """
        Lorentz factor γ(v) = 1 / sqrt(1 - |v|^2 / c^2).
        Clamps speeds to strictly subluminal for numerical safety.
        """
        v = np.asarray(v, dtype=float)
        v2 = float(np.dot(v, v))
        beta2 = v2 / (self.c**2)
        if beta2 >= 1.0:
            beta2 = 1.0 - 1e-12
        return float(1.0 / np.sqrt(1.0 - beta2))

    def _ensure_subluminal(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        speed = np.linalg.norm(v)
        if speed >= self.c:
            v = v * ((self.c - 1e-12) / (speed + 1e-12))
        return v

    def gyroaddition(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Einstein–Ungar gyroaddition via parallel/perpendicular decomposition.

        Let v = v_∥ + v_⊥ with respect to u. Then
          (u ⊕ v)_∥ = (u + v_∥) / (1 + u·v/c²)
          (u ⊕ v)_⊥ = (v_⊥/γ_u) / (1 + u·v/c²)
        and u ⊕ v = (u ⊕ v)_∥ + (u ⊕ v)_⊥.
        """
        u = self._ensure_subluminal(np.asarray(u, dtype=float))
        v = self._ensure_subluminal(np.asarray(v, dtype=float))
        if np.allclose(u, 0):
            return v.copy()
        if np.allclose(v, 0):
            return u.copy()

        u2 = float(np.dot(u, u))
        if u2 < 1e-18:
            return v.copy()

        γu = self.gamma(u)
        uv_over_c2 = float(np.dot(u, v)) / (self.c**2)
        denom = 1.0 + uv_over_c2
        if abs(denom) < 1e-15:
            denom = 1e-15 if denom == 0 else np.sign(denom) * 1e-15

        # Decompose v into components parallel and perpendicular to u
        v_para = (np.dot(v, u) / u2) * u
        v_perp = v - v_para

        w_para = (u + v_para) / denom
        w_perp = (v_perp / γu) / denom
        w = w_para + w_perp
        return self._ensure_subluminal(w)

    def gyrosubtraction(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Gyrosubtraction: u ⊖ v = u ⊕ (⊖v)

        Args:
            u, v: Vectors

        Returns:
            Gyrosubtraction u ⊖ v
        """
        return self.gyroaddition(u, -v)

    def _gyr_apply(self, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Apply the gyro-automorphism to w via the defining identity:
            gyr[a,b] w = (⊖(a ⊕ b)) ⊕ (a ⊕ (b ⊕ w))
        In Einstein gyrogroup, ⊖x = -x.
        """
        return self.gyroaddition(
            -self.gyroaddition(a, b), self.gyroaddition(a, self.gyroaddition(b, w))
        )

    def gyr_apply(self, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Apply gyr[a,b] directly to vector w using the defining identity
        without constructing/orthogonalizing a matrix.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        w = np.asarray(w, dtype=float)
        return self._gyr_apply(a, b, w)

    def gyration(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Return a 3×3 linear approximation (Jacobian) of gyr[a,b] at the origin.
        We apply gyr[a,b] to ε e_i and divide by ε to form columns.

        Implements Thomas-Wigner rotation for small velocities via BCH formula.
        """
        # Adaptive epsilon based on vector magnitudes for better numerical stability
        eps = max(
            1e-8, 1e-8 * (float(np.linalg.norm(a)) + float(np.linalg.norm(b)) + 1)
        )
        E = np.eye(3)
        cols = []
        for i in range(3):
            col = self._gyr_apply(a, b, eps * E[:, i]) / eps
            cols.append(col)
        G = np.column_stack(cols)

        # Project to nearest rotation matrix via polar decomposition for better orthogonality
        U, _, Vt = np.linalg.svd(G)
        R = U @ Vt
        if np.linalg.det(R) < 0:  # fix reflection
            U[:, -1] *= -1
            R = U @ Vt

        # Proof-guard: check orthogonality and fall back to BCH if needed
        orthogonality_error = np.linalg.norm(R.T @ R - np.eye(3))
        if orthogonality_error > 1e-10:
            # BCH rotation: θ ≈ ||u×v||/(2c²), independent of eps
            u_cross_v = np.cross(a, b)
            cross = float(np.linalg.norm(u_cross_v))
            if cross < 1e-15:
                R = np.eye(3)
            else:
                angle = cross / (2.0 * self.c**2)
                angle = float(np.clip(angle, 0.0, np.pi))  # keep in-range
                axis = u_cross_v / cross
                K = np.array(
                    [
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0],
                    ]
                )
                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        return R

    def coaddition(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Einstein coaddition:
            u ⊞ v = u ⊕ gyr[u, ⊖v] v
        Implemented via direct gyration application for numerical stability.
        """
        v_tilt = self.gyr_apply(u, -np.asarray(v, dtype=float), v)
        return self.gyroaddition(u, v_tilt)

    def gyroassociativity_check(
        self, u: np.ndarray, v: np.ndarray, w: np.ndarray, tolerance: float = 1e-10
    ) -> Tuple[bool, float]:
        """
        Check gyroassociativity: (u ⊕ v) ⊕ w ?= u ⊕ (v ⊕ w)

        Args:
            u, v, w: Test vectors
            tolerance: Numerical tolerance

        Returns:
            Tuple of (is_associative, defect_magnitude)
        """
        # Gyroassociativity law: u ⊕ (v ⊕ w) = (u ⊕ v) ⊕ gyr[u,v] w
        left_side = self.gyroaddition(u, self.gyroaddition(v, w))
        gyr_w = self.gyr_apply(u, v, w)
        right_side = self.gyroaddition(self.gyroaddition(u, v), gyr_w)

        # Compute defect
        defect = np.linalg.norm(left_side - right_side)
        is_associative = defect < tolerance

        return bool(is_associative), float(defect)

    def gyrocommutativity_check(
        self, u: np.ndarray, v: np.ndarray, tolerance: float = 1e-10
    ) -> Tuple[bool, float]:
        """
        Check gyrocommutativity: u ⊕ v ?= gyr[u,v](v ⊕ u)

        Args:
            u, v: Test vectors
            tolerance: Numerical tolerance

        Returns:
            Tuple of (is_commutative, defect_magnitude)
        """
        left_side = self.gyroaddition(u, v)
        v_plus_u = self.gyroaddition(v, u)
        right_side = self.gyr_apply(u, v, v_plus_u)

        defect = np.linalg.norm(left_side - right_side)
        is_commutative = defect < tolerance

        return bool(is_commutative), float(defect)

    def commutativity_defect(self, u: np.ndarray, v: np.ndarray) -> float:
        """|| (u ⊕ v) - (v ⊕ u) ||"""
        return float(np.linalg.norm(self.gyroaddition(u, v) - self.gyroaddition(v, u)))

    def associator_defect(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
        """|| (u ⊕ v) ⊕ w - u ⊕ (v ⊕ w) ||"""
        left = self.gyroaddition(self.gyroaddition(u, v), w)
        right = self.gyroaddition(u, self.gyroaddition(v, w))
        return float(np.linalg.norm(left - right))

    def holonomy_triangle(
        self, u: np.ndarray, v: np.ndarray, w: np.ndarray | None = None
    ):
        """
        Rotation holonomy around a closed triangle in velocity space.
        If w is None, close with w = -(u ⊕ v). Returns (angle, matrix).
        """
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        if w is None:
            w = -self.gyroaddition(u, v)

        R1 = self.gyration(u, v)
        R2 = self.gyration(v, w)
        R3 = self.gyration(w, u)
        R = R3 @ R2 @ R1

        tr = float(np.trace(R))
        tr = np.clip(tr, -1.0, 3.0)
        angle = float(np.arccos((tr - 1.0) / 2.0))
        return angle, R

    def rotation_angle_from_matrix(self, R: np.ndarray) -> float:
        """Principal rotation angle from an SO(3) matrix."""
        tr = float(np.clip(np.trace(R), -1.0, 3.0))
        return float(np.arccos((tr - 1.0) / 2.0))


class RecursivePath:
    """
    Represents a recursive path with memory accumulation
    """

    def __init__(self, gyrospace: GyroVectorSpace):
        self.gyrospace = gyrospace
        self.path_points: List[np.ndarray] = []
        self.gyration_memory: List[np.ndarray] = []
        self.coherence_field: List[complex] = []

    def add_step(self, point: np.ndarray):
        """Add a point to the recursive path"""
        self.path_points.append(point)

        if len(self.path_points) >= 2:
            # Compute gyration between consecutive points
            prev = self.path_points[-2]
            curr = self.path_points[-1]
            # Use consecutive "velocities" directly as (a,b)
            R = self.gyrospace.gyration(prev, curr)
            self.gyration_memory.append(R)

            # Extract rotation angle from SO(3)-like map and build complex coherence
            I = np.eye(3)
            tr = float(np.trace(R))
            cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
            theta = float(np.arccos(cos_theta))
            amp = float(np.exp(-np.linalg.norm(R - I)))
            coherence = amp * np.exp(1j * theta)
            self.coherence_field.append(coherence)

    def get_recursive_memory(self) -> np.ndarray:
        """
        Compute the accumulated recursive memory (monodromy)

        Returns:
            Product of gyrations along the path
        """
        if not self.gyration_memory:
            return np.eye(3)  # Identity for empty path

        # Compute ordered product of gyrations
        memory = np.eye(3)  # Start with identity

        for gyr in self.gyration_memory:
            # Ensure gyr is a proper 3x3 matrix
            if not (hasattr(gyr, "shape") and gyr.shape == (3, 3)):
                raise TypeError("gyration memory element must be a 3x3 matrix")

            memory = memory @ gyr

        return memory

    def temporal_measure(self) -> float:
        """
        Compute temporal measure: τ_obs = ∫ ∇arg(ψ_rec) dl

        Returns:
            Temporal measure of the recursive path
        """
        if len(self.coherence_field) < 2:
            return 0.0

        # Simplified numerical integration
        total_time = 0.0
        for i in range(1, len(self.coherence_field)):
            # Gradient of argument (phase)
            phase_gradient = np.angle(self.coherence_field[i]) - np.angle(
                self.coherence_field[i - 1]
            )
            # Path length element
            dl = np.linalg.norm(self.path_points[i] - self.path_points[i - 1])
            total_time += phase_gradient * dl

        return float(total_time)
