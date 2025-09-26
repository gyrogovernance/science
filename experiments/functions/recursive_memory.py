#!/usr/bin/env python3
"""
Recursive Memory Structure for CGM

This module implements the recursive memory structure that connects
abstract gyrogroup geometry to physical observables like κ, α_EM, and
gravitational fields.

Key components:
1. Coherence fields ψ_rec that accumulate along recursive paths
2. Monodromy residue μ(M) computation around closed loops
3. Phase gradients ∇arg(ψ_rec) for temporal measure
4. Stage transition observables (SU(2) spin, SO(3) translation)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings

try:
    from .gyrovector_ops import GyroVectorSpace
except ImportError:
    # Fallback for direct execution
    from gyrovector_ops import GyroVectorSpace


@dataclass
class MemoryPath:
    """A path through recursive memory space."""

    points: List[np.ndarray]  # 3D points in gyrovector space
    velocities: List[np.ndarray]  # Velocity vectors at each point
    phases: List[complex]  # Complex phases accumulated along path
    stage_indicators: List[str]  # CGM stage at each point (CS, UNA, ONA, BU)

    def __post_init__(self):
        """Validate path consistency."""
        if not (
            len(self.points)
            == len(self.velocities)
            == len(self.phases)
            == len(self.stage_indicators)
        ):
            raise ValueError("All path components must have same length")

    def add_step(
        self, point: np.ndarray, velocity: np.ndarray, phase: complex, stage: str
    ):
        """Add a step to the memory path."""
        self.points.append(point)
        self.velocities.append(velocity)
        self.phases.append(phase)
        self.stage_indicators.append(stage)

    def get_coherence_field(self) -> complex:
        """Get the total coherence field accumulated along the path."""
        if not self.phases:
            return 1.0 + 0j

        # Accumulate phases multiplicatively (geometric phase)
        total_phase = 1.0 + 0j
        for phase in self.phases:
            total_phase *= phase

        return total_phase

    def get_phase_gradient(self) -> np.ndarray:
        """Compute the phase gradient ∇arg(ψ_rec) along the path."""
        if len(self.points) < 2:
            return np.zeros(3)

        # Compute phase differences between consecutive points
        phase_diffs = []
        point_diffs = []

        for i in range(1, len(self.points)):
            phase_diff = np.angle(self.phases[i] / self.phases[i - 1])
            point_diff = self.points[i] - self.points[i - 1]

            phase_diffs.append(phase_diff)
            point_diffs.append(point_diff)

        if not phase_diffs:
            return np.zeros(3)

        # Compute gradient as weighted average of phase differences
        # weighted by the inverse of point differences
        total_weight = 0.0
        weighted_gradient = np.zeros(3)

        for phase_diff, point_diff in zip(phase_diffs, point_diffs):
            distance = np.linalg.norm(point_diff)
            if distance > 1e-12:  # Avoid division by zero
                weight = 1.0 / distance
                total_weight += weight
                weighted_gradient += weight * phase_diff * point_diff / distance

        if total_weight > 0:
            return weighted_gradient / total_weight
        else:
            return np.zeros(3)


class RecursiveMemory:
    """
    Implements recursive memory structure for CGM.

    This class builds on the proven mathematical foundation to implement:
    1. Coherence field accumulation along recursive paths
    2. Monodromy residue computation around closed loops
    3. Phase gradient computation for temporal measure
    4. Stage transition observables
    """

    def __init__(
        self,
        gyrospace: GyroVectorSpace,
        memory_depth: int = 100,
        coherence_threshold: float = 1e-6,
    ):
        """
        Initialize recursive memory system.

        Args:
            gyrospace: GyroVectorSpace instance for geometric operations
            memory_depth: Maximum depth of recursive memory paths
            coherence_threshold: Threshold for coherence field significance
        """
        self.gyrospace = gyrospace
        self.memory_depth = memory_depth
        self.coherence_threshold = coherence_threshold

        # Initialize memory storage
        self.memory_paths: List[MemoryPath] = []
        self.coherence_fields: List[complex] = []
        self.monodromy_residues: List[complex] = []

        # CGM stage parameters (from proven foundations)
        self.stage_angles = {
            "CS": 0.0,  # Chiral Source
            "UNA": np.pi / 4,  # Unity Non-Absolute (π/4)
            "ONA": np.pi / 2,  # Opposition Non-Absolute (π/2)
            "BU": np.pi,  # Balance Universal (π)
        }

        # Stage transition thresholds
        self.transition_thresholds = {
            "CS_to_UNA": 0.1,  # Velocity threshold for UNA emergence
            "UNA_to_ONA": 0.5,  # Velocity threshold for ONA activation
            "ONA_to_BU": 0.8,  # Velocity threshold for BU closure
        }

    def determine_stage(
        self, velocity: np.ndarray, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Determine CGM stage based on velocity and context.

        Args:
            velocity: Velocity vector in gyrovector space
            context: Optional context for stage determination

        Returns:
            CGM stage identifier (CS, UNA, ONA, BU)
        """
        speed = np.linalg.norm(velocity)

        # Basic stage determination based on velocity magnitude
        if speed < self.transition_thresholds["CS_to_UNA"]:
            return "CS"
        elif speed < self.transition_thresholds["UNA_to_ONA"]:
            return "UNA"
        elif speed < self.transition_thresholds["ONA_to_BU"]:
            return "ONA"
        else:
            return "BU"

    def compute_stage_phase(
        self, stage: str, velocity: np.ndarray, context: Optional[Dict[str, Any]] = None
    ) -> complex:
        """
        Compute the phase contribution for a given CGM stage.

        Args:
            stage: CGM stage identifier
            velocity: Velocity vector
            context: Optional context for phase computation

        Returns:
            Complex phase contribution
        """
        base_angle = self.stage_angles[stage]
        speed = np.linalg.norm(velocity)

        # Stage-specific phase computations
        if stage == "CS":
            # Chiral source: minimal phase accumulation
            phase = np.exp(1j * base_angle * speed)

        elif stage == "UNA":
            # Unity Non-Absolute: spin emergence (SU(2))
            # Phase accumulates as exp(iβσ₃/2) where β = π/4
            spin_phase = np.exp(1j * base_angle * speed)
            phase = spin_phase

        elif stage == "ONA":
            # Opposition Non-Absolute: translation activation (SO(3))
            # Phase accumulates as exp(iγσ₂/2) where γ = π/2
            trans_phase = np.exp(1j * base_angle * speed)
            phase = trans_phase

        elif stage == "BU":
            # Balance Universal: global closure
            # Phase accumulates as exp(iπ) = -1 for closure
            closure_phase = np.exp(1j * np.pi * speed)
            phase = closure_phase

        else:
            # Unknown stage: default to identity
            phase = 1.0 + 0j

        return phase

    def build_recursive_path(
        self,
        initial_point: np.ndarray,
        initial_velocity: np.ndarray,
        n_steps: int,
        step_function: Optional[Callable] = None,
    ) -> MemoryPath:
        """
        Build a recursive memory path.

        Args:
            initial_point: Starting point in gyrovector space
            initial_velocity: Initial velocity vector
            n_steps: Number of steps in the path
            step_function: Optional function to determine next step

        Returns:
            MemoryPath object representing the recursive path
        """
        if n_steps > self.memory_depth:
            warnings.warn(
                f"Path length {n_steps} exceeds memory depth {self.memory_depth}"
            )
            n_steps = self.memory_depth

        # Initialize path
        path = MemoryPath(
            points=[initial_point.copy()],
            velocities=[initial_velocity.copy()],
            phases=[1.0 + 0j],
            stage_indicators=["CS"],
        )

        current_point = initial_point.copy()
        current_velocity = initial_velocity.copy()

        try:
            for step in range(1, n_steps):
                # Determine current stage
                current_stage = self.determine_stage(current_velocity)

                # Compute stage phase
                stage_phase = self.compute_stage_phase(current_stage, current_velocity)

                # Determine next step
                if step_function is not None:
                    next_point, next_velocity = step_function(
                        current_point, current_velocity, step
                    )
                else:
                    # Default: gyrovector addition with some randomness
                    random_direction = np.random.normal(0, 1, 3)
                    random_direction = random_direction / np.linalg.norm(
                        random_direction
                    )
                    step_size = 0.1 * (1.0 - np.linalg.norm(current_velocity))

                    # Ensure step_size is reasonable
                    step_size = max(0.01, min(0.5, float(step_size)))

                    try:
                        next_point = self.gyrospace.gyroaddition(
                            current_point, step_size * random_direction
                        )
                        next_velocity = self.gyrospace.gyroaddition(
                            current_velocity, step_size * random_direction
                        )
                    except Exception as e:
                        # Fallback to simple addition if gyroaddition fails
                        warnings.warn(
                            f"Gyroaddition failed at step {step}, using fallback: {e}"
                        )
                        next_point = current_point + step_size * random_direction
                        next_velocity = current_velocity + step_size * random_direction

                # Add step to path
                path.add_step(next_point, next_velocity, stage_phase, current_stage)

                # Update current state
                current_point = next_point.copy()
                current_velocity = next_velocity.copy()

        except Exception as e:
            warnings.warn(f"Error building recursive path: {e}")
            # Return partial path if there's an error
            pass

        return path

    def compute_monodromy_residue(self, path: MemoryPath) -> complex:
        """
        Compute monodromy residue μ(M) around a closed loop.

        Args:
            path: MemoryPath representing the loop

        Returns:
            Complex monodromy residue
        """
        if len(path.points) < 3:
            return 1.0 + 0j

        # Check if path is approximately closed
        start_point = path.points[0]
        end_point = path.points[-1]
        closure_error = np.linalg.norm(end_point - start_point)

        if closure_error > 0.1:  # Path not closed
            warnings.warn("Path is not closed, monodromy may not be meaningful")

        # Compute monodromy as product of all phase factors
        monodromy = 1.0 + 0j
        for phase in path.phases:
            monodromy *= phase

        return monodromy

    def compute_coherence_field(self, paths: List[MemoryPath]) -> complex:
        """
        Compute total coherence field from multiple memory paths.

        Args:
            paths: List of MemoryPath objects

        Returns:
            Total coherence field ψ_rec
        """
        if not paths:
            return 1.0 + 0j

        # Accumulate coherence from all paths
        total_coherence = 1.0 + 0j
        for path in paths:
            path_coherence = path.get_coherence_field()
            total_coherence *= path_coherence

        return total_coherence

    def compute_phase_gradient(self, paths: List[MemoryPath]) -> np.ndarray:
        """
        Compute phase gradient ∇arg(ψ_rec) from multiple memory paths.

        Args:
            paths: List of MemoryPath objects

        Returns:
            Phase gradient vector
        """
        if not paths:
            return np.zeros(3)

        # Compute weighted average of phase gradients from all paths
        total_weight = 0.0
        weighted_gradient = np.zeros(3)

        for path in paths:
            path_gradient = path.get_phase_gradient()
            path_weight = np.linalg.norm(path_gradient)

            if path_weight > self.coherence_threshold:
                total_weight += path_weight
                weighted_gradient += path_weight * path_gradient

        if total_weight > 0:
            return weighted_gradient / total_weight
        else:
            return np.zeros(3)

    def estimate_kappa_from_geometry(
        self, n_paths: int = 10, path_length: int = 50
    ) -> Dict[str, Any]:
        """
        Estimate κ from CGM geometry using recursive memory.

        This is the key function that should connect CGM structure
        to the gravitational coupling κ.

        Args:
            n_paths: Number of memory paths to generate
            path_length: Length of each memory path

        Returns:
            Dictionary with κ estimate and related quantities
        """
        # Generate multiple recursive memory paths
        paths = []
        for i in range(n_paths):
            # Random initial conditions
            initial_point = np.random.normal(0, 0.1, 3)
            initial_velocity = np.random.normal(0, 0.1, 3)

            path = self.build_recursive_path(
                initial_point, initial_velocity, path_length
            )
            paths.append(path)

        # Compute coherence field
        total_coherence = self.compute_coherence_field(paths)
        coherence_magnitude = abs(total_coherence)

        # Compute phase gradient
        phase_gradient = self.compute_phase_gradient(paths)
        gradient_magnitude = np.linalg.norm(phase_gradient)

        # Compute monodromy residues
        monodromy_residues = []
        for path in paths:
            residue = self.compute_monodromy_residue(path)
            monodromy_residues.append(residue)

        # Average monodromy residue
        avg_monodromy = np.mean(monodromy_residues)
        monodromy_magnitude = abs(avg_monodromy)

        # Estimate κ from geometric quantities
        # This is the key relationship to be developed:
        # κ ∝ (coherence_magnitude × gradient_magnitude) / monodromy_magnitude

        if monodromy_magnitude > self.coherence_threshold:
            kappa_estimate = (
                coherence_magnitude * gradient_magnitude
            ) / monodromy_magnitude
        else:
            kappa_estimate = np.inf

        return {
            "kappa_estimate": kappa_estimate,
            "coherence_magnitude": coherence_magnitude,
            "gradient_magnitude": gradient_magnitude,
            "monodromy_magnitude": monodromy_magnitude,
            "total_coherence": total_coherence,
            "phase_gradient": phase_gradient,
            "monodromy_residues": monodromy_residues,
            "n_paths": n_paths,
            "path_length": path_length,
            "note": "κ estimate from CGM geometry via recursive memory",
        }

    def run_memory_experiment(
        self, n_paths: int = 20, path_length: int = 100
    ) -> Dict[str, Any]:
        """
        Run a comprehensive memory experiment.

        Args:
            n_paths: Number of memory paths
            path_length: Length of each path

        Returns:
            Comprehensive results from memory experiment
        """
        print("CGM Recursive Memory Experiment")
        print("=" * 40)

        # Generate memory paths
        print(f"Generating {n_paths} memory paths of length {path_length}...")
        paths = []
        for i in range(n_paths):
            initial_point = np.random.normal(0, 0.1, 3)
            initial_velocity = np.random.normal(0, 0.1, 3)

            path = self.build_recursive_path(
                initial_point, initial_velocity, path_length
            )
            paths.append(path)

            if (i + 1) % 5 == 0:
                print(f"  Generated {i + 1}/{n_paths} paths")

        # Compute key quantities
        print("\nComputing coherence fields and monodromy...")
        total_coherence = self.compute_coherence_field(paths)
        phase_gradient = self.compute_phase_gradient(paths)

        monodromy_residues = []
        for path in paths:
            residue = self.compute_monodromy_residue(path)
            monodromy_residues.append(residue)

        # Statistical analysis
        coherence_magnitude = abs(total_coherence)
        gradient_magnitude = np.linalg.norm(phase_gradient)
        monodromy_magnitudes = [abs(r) for r in monodromy_residues]

        avg_monodromy = np.mean(monodromy_magnitudes)
        std_monodromy = np.std(monodromy_magnitudes)

        # κ estimation
        if avg_monodromy > self.coherence_threshold:
            kappa_estimate = (coherence_magnitude * gradient_magnitude) / avg_monodromy
        else:
            kappa_estimate = np.inf

        print(f"\nResults:")
        print(f"  Total coherence magnitude: {coherence_magnitude:.6f}")
        print(f"  Phase gradient magnitude: {gradient_magnitude:.6f}")
        print(
            f"  Average monodromy magnitude: {avg_monodromy:.6f} ± {std_monodromy:.6f}"
        )
        print(f"  κ estimate from geometry: {kappa_estimate:.6f}")

        # Store results
        self.memory_paths = paths
        self.coherence_fields = [path.get_coherence_field() for path in paths]
        self.monodromy_residues = monodromy_residues

        return {
            "paths": paths,
            "total_coherence": total_coherence,
            "phase_gradient": phase_gradient,
            "monodromy_residues": monodromy_residues,
            "kappa_estimate": kappa_estimate,
            "statistics": {
                "coherence_magnitude": coherence_magnitude,
                "gradient_magnitude": gradient_magnitude,
                "avg_monodromy": avg_monodromy,
                "std_monodromy": std_monodromy,
            },
        }
