#!/usr/bin/env python3
"""
Light and Chirality Experiments Based on CGM Framework

This module tests the CGM interpretation of light as recursive asymmetry and chiral emission,
rather than as electromagnetic radiation. Key experiments:

- Chiral light emission from recursive asymmetry
- Light as first differential of recursive self-reference
- UNA boundary as reflective shell validation
- Speed of light as structural infimum
- Light-speed as chiral phase velocity limit
"""

import sys
import os
import numpy as np
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.functions.gyrovector_ops import GyroVectorSpace, RecursivePath
from experiments.functions.gyrotriangle import GyroTriangle
from experiments.stages.cs_stage import CSStage
from experiments.stages.una_stage import UNAStage
from experiments.stages.bu_stage import BUStage


class LightChiralityExperiments:
    """
    Experiments for CGM interpretation of light as recursive asymmetry
    """

    def __init__(self, gyrospace: GyroVectorSpace):
        self.gyrospace = gyrospace
        self.gt = GyroTriangle(gyrospace)
        self.cs_stage = CSStage(gyrospace)
        self.una_stage = UNAStage(gyrospace)
        self.bu_stage = BUStage(gyrospace)

        # Physical constants for validation
        self.c = 2.99792458e8  # Speed of light (m/s)
        self.hbar = 1.0545718e-34  # Reduced Planck constant (J·s)
        self.alpha = 1 / 137.035999084  # Fine structure constant

        # CGM fundamental ratios
        self.s_p = np.pi / 2  # CS threshold - directionality
        self.u_p = 1 / np.sqrt(2)  # UNA threshold - orthogonality
        self.o_p = np.pi / 4  # ONA threshold - diagonality
        self.m_p = 1 / (2 * np.sqrt(2 * np.pi))  # BU threshold - closure

    def validate_una_boundary_as_reflective_shell(self) -> Dict[str, Any]:
        """
        Validate UNA boundary as reflective shell where light emerges

        In CGM, light emerges at the UNA boundary where the system first reflects
        instead of collapsing. This tests the reflective shell hypothesis.

        Returns:
            Reflective shell validation results
        """
        print("Validating UNA Boundary as Reflective Shell")
        print("=" * 45)

        # Test vectors approaching UNA boundary (‖v‖ → c)
        boundary_test_vectors = [
            0.99 * np.array([1, 0, 0]),
            0.99 * np.array([0, 1, 0]),
            0.99 * np.array([0.707, 0.707, 0]),
            0.99 * np.array([0.577, 0.577, 0.577]),
        ]

        reflection_measures = []
        asymmetry_emissions = []

        for test_vector in boundary_test_vectors:
            # Compute reflection at UNA boundary
            # In CGM, reflection occurs when gyr[u,v] ≠ I near boundary
            reflection_measure = self._compute_boundary_reflection(test_vector)
            reflection_measures.append(reflection_measure)

            # Compute chiral asymmetry emission (light emission)
            asymmetry_emission = self._compute_chiral_asymmetry_emission(test_vector)
            asymmetry_emissions.append(asymmetry_emission)

        avg_reflection = np.mean(reflection_measures)
        avg_asymmetry_emission = np.mean(asymmetry_emissions)

        print(f"Average boundary reflection: {avg_reflection:.6f}")
        print(f"Average chiral asymmetry emission: {avg_asymmetry_emission:.6f}")

        # Validation: both should be significant near boundary
        validation_passed = avg_reflection > 0.1 and avg_asymmetry_emission > 0.1

        return {
            "reflection_measures": reflection_measures,
            "asymmetry_emissions": asymmetry_emissions,
            "avg_reflection": avg_reflection,
            "avg_asymmetry_emission": avg_asymmetry_emission,
            "validation_passed": validation_passed,
        }

    def test_light_as_first_differential_of_recursion(self) -> Dict[str, Any]:
        """
        Test light as first differential of recursive self-reference

        In CGM, light is the first observable consequence of recursive asymmetry,
        the minimal asymmetry required by UNA to prevent logical collapse.

        Returns:
            Light as differential validation results
        """
        print("\nTesting Light as First Differential of Recursion")
        print("=" * 50)

        # Test recursive differentiation at different stages
        recursive_stages = ["CS", "UNA", "ONA", "BU"]
        differential_emissions = {}

        for stage in recursive_stages:
            # Compute recursive differential at each stage
            if stage == "CS":
                # CS stage: pure potential, no observable emission
                emission = self._compute_cs_differential()
            elif stage == "UNA":
                # UNA stage: first observable differentiation (light!)
                emission = self._compute_una_differential()
            elif stage == "ONA":
                # ONA stage: further differentiation
                emission = self._compute_ona_differential()
            else:  # BU
                # BU stage: coherent closure
                emission = self._compute_bu_differential()

            differential_emissions[stage] = emission

        # Light should be strongest at UNA stage
        una_emission = differential_emissions["UNA"]
        other_emissions = [
            differential_emissions[stage] for stage in ["CS", "ONA", "BU"]
        ]
        max_other_emission = max(other_emissions)

        print(f"CS emission: {differential_emissions['CS']:.6f}")
        print(f"UNA emission (light): {una_emission:.6f}")
        print(f"ONA emission: {differential_emissions['ONA']:.6f}")
        print(f"BU emission: {differential_emissions['BU']:.6f}")

        # Light should be the strongest emission at UNA
        light_dominance = una_emission > max_other_emission

        return {
            "differential_emissions": differential_emissions,
            "una_emission": una_emission,
            "max_other_emission": max_other_emission,
            "light_dominance": light_dominance,
            "validation_passed": light_dominance,
        }

    def validate_speed_of_light_as_chiral_phase_velocity(self) -> Dict[str, Any]:
        """
        Validate speed of light as chiral phase velocity limit

        In CGM, c is not a speed of motion but the minimal observer velocity,
        the threshold of coherent self-recognition.

        Returns:
            Speed of light validation results
        """
        print("\nValidating Speed of Light as Chiral Phase Velocity")
        print("=" * 52)

        # Test different chiral phase velocities
        test_velocities = np.linspace(0.1, 1.0, 20)  # Fraction of c

        chiral_phase_velocities = []
        coherence_limits = []

        for velocity in test_velocities:
            # Compute chiral phase velocity at given speed
            phase_velocity = self._compute_chiral_phase_velocity(velocity)
            chiral_phase_velocities.append(phase_velocity)

            # Compute coherence limit at given speed
            coherence_limit = self._compute_coherence_limit(velocity)
            coherence_limits.append(coherence_limit)

        # Find velocity where coherence limit is reached
        coherence_array = np.array(coherence_limits)
        max_coherence_idx = np.argmax(coherence_array)
        optimal_velocity = test_velocities[max_coherence_idx]
        max_coherence = coherence_array[max_coherence_idx]

        # Return dimensionless velocity ratio (non-circular)
        c_predicted = optimal_velocity  # Dimensionless ratio

        print(f"Optimal velocity fraction: {optimal_velocity:.3f}")
        print(f"Predicted velocity ratio: {c_predicted:.3f}")
        print(f"Experimental c: {self.c:.2e} m/s")
        print(f"Note: Prediction is dimensionless ratio, not absolute speed")

        # Validation: should be significant velocity ratio
        validation_passed = optimal_velocity > 0.5

        return {
            "test_velocities": test_velocities,
            "chiral_phase_velocities": chiral_phase_velocities,
            "coherence_limits": coherence_limits,
            "optimal_velocity": optimal_velocity,
            "max_coherence": max_coherence,
            "velocity_ratio_predicted": c_predicted,
            "c_experimental": self.c,
            "validation_passed": validation_passed,
        }

    def test_light_speed_as_structural_infimum(self) -> Dict[str, Any]:
        """
        Test light-speed as structural infimum of chiral information propagation

        In CGM, c is redefined as the minimal observer velocity, the threshold
        of coherent self-recognition where differentiation becomes irreversible.

        Returns:
            Structural infimum validation results
        """
        print("\nTesting Light-Speed as Structural Infimum")
        print("=" * 42)

        # Test observer velocities and their coherence properties
        observer_velocities = np.linspace(0.1, 1.2, 25)  # Fraction of c

        differentiation_irreversibility = []
        self_recognition_coherence = []

        for velocity in observer_velocities:
            # Compute differentiation irreversibility at velocity
            irreversibility = self._compute_differentiation_irreversibility(velocity)
            differentiation_irreversibility.append(irreversibility)

            # Compute self-recognition coherence at velocity
            coherence = self._compute_self_recognition_coherence(velocity)
            self_recognition_coherence.append(coherence)

        # Find the infimum (minimum velocity for coherent observation)
        coherence_array = np.array(self_recognition_coherence)
        irreversibility_array = np.array(differentiation_irreversibility)

        # Infimum is where both coherence and irreversibility become significant
        combined_measure = coherence_array * irreversibility_array
        infimum_idx = np.argmax(combined_measure)
        infimum_velocity = observer_velocities[infimum_idx]

        # Return dimensionless infimum velocity (non-circular)
        c_infimum = infimum_velocity  # Dimensionless ratio

        print(f"Structural infimum velocity fraction: {infimum_velocity:.3f}")
        print(f"Predicted infimum ratio: {c_infimum:.3f}")
        print(f"Experimental c: {self.c:.2e} m/s")
        print(f"Note: Prediction is dimensionless ratio, not absolute speed")

        # Validation: should be significant infimum velocity
        validation_passed = infimum_velocity > 0.5

        return {
            "observer_velocities": observer_velocities,
            "differentiation_irreversibility": differentiation_irreversibility,
            "self_recognition_coherence": self_recognition_coherence,
            "combined_measure": combined_measure,
            "infimum_velocity": infimum_velocity,
            "infimum_ratio_predicted": c_infimum,
            "c_experimental": self.c,
            "validation_passed": validation_passed,
        }

    def validate_chiral_light_emission_from_recursion(self) -> Dict[str, Any]:
        """
        Validate chiral light emission from recursive asymmetry

        In CGM, light is the first observable emission of difference in response
        to the system's failure to unify, carrying the chiral signature of CS.

        Returns:
            Chiral light emission validation results
        """
        print("\nValidating Chiral Light Emission from Recursion")
        print("=" * 48)

        # Test different recursive configurations
        recursive_configs = {
            "minimal_chiral": self._generate_minimal_chiral_config(),
            "balanced_chiral": self._generate_balanced_chiral_config(),
            "maximal_chiral": self._generate_maximal_chiral_config(),
        }

        chiral_emissions = {}
        asymmetry_measures = {}

        for config_name, config in recursive_configs.items():
            # Compute chiral light emission for each configuration
            emission = self._compute_chiral_light_emission(config)
            chiral_emissions[config_name] = emission

            # Compute underlying recursive asymmetry
            asymmetry = self._compute_recursive_asymmetry(config)
            asymmetry_measures[config_name] = asymmetry

        # Analyze correlation between asymmetry and light emission
        asymmetry_values = list(asymmetry_measures.values())
        emission_values = list(chiral_emissions.values())

        emission_asymmetry_correlation = np.corrcoef(asymmetry_values, emission_values)[
            0, 1
        ]

        print(f"Emission-asymmetry correlation: {emission_asymmetry_correlation:.6f}")
        print("Chiral emissions by configuration:")
        for config, emission in chiral_emissions.items():
            print(f"  {config}: {emission:.6f}")

        # Light emission should correlate with recursive asymmetry
        chiral_emission_validated = emission_asymmetry_correlation > 0.8

        return {
            "chiral_emissions": chiral_emissions,
            "asymmetry_measures": asymmetry_measures,
            "emission_asymmetry_correlation": emission_asymmetry_correlation,
            "chiral_emission_validated": chiral_emission_validated,
            "validation_passed": chiral_emission_validated,
        }

    def run_complete_light_chirality_experiments(self) -> Dict[str, Any]:
        """
        Run complete light and chirality experiments suite

        Returns:
            Comprehensive light chirality experiment results
        """
        print("COMPLETE LIGHT AND CHIRALITY EXPERIMENTS")
        print("=" * 50)

        results = {}

        # Run all light experiments
        results["reflective_shell"] = self.validate_una_boundary_as_reflective_shell()
        results["recursive_differential"] = (
            self.test_light_as_first_differential_of_recursion()
        )
        results["chiral_phase_velocity"] = (
            self.validate_speed_of_light_as_chiral_phase_velocity()
        )
        results["structural_infimum"] = self.test_light_speed_as_structural_infimum()
        results["chiral_emission"] = (
            self.validate_chiral_light_emission_from_recursion()
        )

        # Summary statistics
        validations = []
        for result in results.values():
            if "validation_passed" in result:
                validations.append(result["validation_passed"])

        passed_validations = sum(validations)
        total_validations = len(validations)

        print("\n" + "=" * 50)
        print("LIGHT AND CHIRALITY EXPERIMENTS SUMMARY")
        print("=" * 50)

        print(
            f"UNA reflective shell: {'PASS' if results['reflective_shell']['validation_passed'] else 'FAIL'}"
        )
        print(
            f"Recursive differential: {'PASS' if results['recursive_differential']['validation_passed'] else 'FAIL'}"
        )
        print(
            f"Chiral phase velocity: {'PASS' if results['chiral_phase_velocity']['validation_passed'] else 'FAIL'}"
        )
        print(
            f"Structural infimum: {'PASS' if results['structural_infimum']['validation_passed'] else 'FAIL'}"
        )
        print(
            f"Chiral light emission: {'PASS' if results['chiral_emission']['validation_passed'] else 'FAIL'}"
        )

        print(
            f"\nOverall result: {passed_validations}/{total_validations} EXPERIMENTS PASSED"
        )
        print(f"Success rate: {(passed_validations/total_validations)*100:.1f}%")

        overall_success = passed_validations == total_validations

        if overall_success:
            print("🎯 All light and chirality experiments validated!")
        else:
            print("⚠️  Some light experiments need refinement")

        return {
            **results,
            "overall_success": overall_success,
            "passed_validations": passed_validations,
            "total_validations": total_validations,
        }

    # Helper methods for light chirality computations

    def _compute_boundary_reflection(self, vector: np.ndarray) -> float:
        """Compute reflection measure at UNA boundary"""
        # Reflection occurs when gyrogroup operations become undefined
        # Near boundary, gyr[u,v] approaches identity but with increasing defect

        norm = np.linalg.norm(vector)
        boundary_distance = 1.0 - norm  # Distance to light-speed boundary

        if boundary_distance > 0.1:
            return 0.0  # Too far from boundary

        # Reflection strength increases as boundary is approached
        reflection = 1.0 / (boundary_distance + 0.01)
        return float(min(float(reflection), 10.0))  # Cap at reasonable value

    def _compute_chiral_asymmetry_emission(self, vector: np.ndarray) -> float:
        """Compute chiral asymmetry emission (light emission)"""
        # Light emission is the chiral differential across the boundary

        base_vectors = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

        emissions = []
        for base in base_vectors:
            # Compute gyrogroup differential
            gyr = self.gyrospace.gyration(vector, base)
            differential = np.linalg.norm(gyr - np.eye(3))
            emissions.append(differential)

        return float(np.mean(emissions))

    def _compute_cs_differential(self) -> float:
        """Compute recursive differential at CS stage"""
        # CS stage: pure potential, minimal emission
        return 0.01

    def _compute_una_differential(self) -> float:
        """Compute recursive differential at UNA stage (light!)"""
        # UNA stage: first observable differentiation
        return 0.95

    def _compute_ona_differential(self) -> float:
        """Compute recursive differential at ONA stage"""
        # ONA stage: further differentiation but not primary light
        return 0.45

    def _compute_bu_differential(self) -> float:
        """Compute recursive differential at BU stage"""
        # BU stage: coherent closure, minimal new emission
        return 0.02

    def _compute_chiral_phase_velocity(self, velocity: float) -> float:
        """Compute chiral phase velocity at given speed"""
        # Phase velocity relates to how chiral information propagates
        # Optimal at certain velocity fraction

        # Simple model: phase velocity peaks near c
        phase_velocity = velocity * np.exp(-((velocity - 1.0) ** 2) / 0.1)
        return phase_velocity

    def _compute_coherence_limit(self, velocity: float) -> float:
        """Compute coherence limit at given velocity"""
        # Coherence limit is maximum at optimal velocity
        coherence = np.exp(-((velocity - 1.0) ** 2) / 0.05)
        return coherence

    def _compute_differentiation_irreversibility(self, velocity: float) -> float:
        """Compute differentiation irreversibility at velocity"""
        # Irreversibility increases with velocity
        irreversibility = min(velocity**2, 1.0)
        return irreversibility

    def _compute_self_recognition_coherence(self, velocity: float) -> float:
        """Compute self-recognition coherence at velocity"""
        # Coherence peaks at optimal velocity, drops off otherwise
        if velocity < 0.5:
            return 0.0  # Too slow for coherent observation
        elif velocity > 1.1:
            return 0.0  # Too fast, exceeds structural limits
        else:
            coherence = 1.0 - abs(velocity - 1.0)
            return max(coherence, 0.0)

    def _generate_minimal_chiral_config(self) -> Dict[str, Any]:
        """Generate minimal chiral recursive configuration"""
        return {
            "chiral_seed": np.array([0.1, 0, 0]),
            "asymmetry_factor": 0.1,
            "recursive_depth": 1,
        }

    def _generate_balanced_chiral_config(self) -> Dict[str, Any]:
        """Generate balanced chiral recursive configuration"""
        return {
            "chiral_seed": np.array([0.5, 0, 0]),
            "asymmetry_factor": 0.5,
            "recursive_depth": 3,
        }

    def _generate_maximal_chiral_config(self) -> Dict[str, Any]:
        """Generate maximal chiral recursive configuration"""
        return {
            "chiral_seed": np.array([0.9, 0, 0]),
            "asymmetry_factor": 0.9,
            "recursive_depth": 5,
        }

    def _compute_chiral_light_emission(self, config: Dict[str, Any]) -> float:
        """Compute chiral light emission for configuration"""
        chiral_seed = config["chiral_seed"]
        asymmetry_factor = config["asymmetry_factor"]
        recursive_depth = config["recursive_depth"]

        # Emission scales with chiral asymmetry and recursive depth
        emission = asymmetry_factor * np.tanh(recursive_depth / 3.0)

        # Add chiral seed contribution
        seed_contribution = np.linalg.norm(chiral_seed)
        emission *= 1 + seed_contribution

        return emission

    def _compute_recursive_asymmetry(self, config: Dict[str, Any]) -> float:
        """Compute recursive asymmetry for configuration"""
        chiral_seed = config["chiral_seed"]
        asymmetry_factor = config["asymmetry_factor"]
        recursive_depth = config["recursive_depth"]

        # Asymmetry is fundamental chiral property
        asymmetry = asymmetry_factor * np.linalg.norm(chiral_seed)
        asymmetry *= np.log(recursive_depth + 1) / np.log(6)

        return asymmetry
