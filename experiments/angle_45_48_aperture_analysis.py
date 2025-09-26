#!/usr/bin/env python3
"""
45° to 48° Aperture Analysis
============================

This script analyzes the geometric relationship between 45° (perfect closure) 
and 48° (aperture closure) in the context of CGM aperture principles.

The key insight: 45° represents perfect geometric closure, while 48° introduces
a 3.33% aperture that may be related to CGM's 2.07% aperture principle.

Author: CGM Analysis
Date: 2025
"""

import math
from typing import Dict, List, Tuple
from fractions import Fraction

class Angle45_48ApertureAnalyzer:
    """Analyzer for 45° to 48° aperture transition."""
    
    def __init__(self):
        self.angle_45 = 45.0  # degrees
        self.angle_48 = 48.0  # degrees
        self.angle_45_rad = math.radians(self.angle_45)
        self.angle_48_rad = math.radians(self.angle_48)
        
        # CGM aperture parameters
        self.m_p = 1 / (2 * math.sqrt(2 * math.pi))
        self.delta_BU = 0.195342176580
        self.cgm_aperture = 1 - self.delta_BU / self.m_p
        
    def analyze_aperture_transition(self) -> Dict:
        """Analyze the geometric transition from 45° to 48°."""
        
        # Basic angular relationships
        angular_difference = self.angle_48 - self.angle_45
        percentage_increase = (angular_difference / self.angle_45) * 100
        
        # Closure analysis
        closure_45 = self.angle_45 / 90  # 45° as fraction of right angle
        closure_48 = self.angle_48 / 90  # 48° as fraction of right angle
        aperture_45_48 = (angular_difference / 90) * 100  # Aperture as % of right angle
        
        # Radian relationships
        angle_45_rad = math.radians(self.angle_45)
        angle_48_rad = math.radians(self.angle_48)
        radian_difference = angle_48_rad - angle_45_rad
        
        # Exact fractional relationships
        ratio_48_45 = self.angle_48 / self.angle_45
        ratio_45_48 = self.angle_45 / self.angle_48
        
        # Musical interval analysis
        musical_interval_cents = 1200 * math.log2(ratio_48_45)
        
        return {
            'angular_analysis': {
                'angle_45_deg': self.angle_45,
                'angle_48_deg': self.angle_48,
                'difference_deg': angular_difference,
                'percentage_increase': percentage_increase,
                'angle_45_rad': angle_45_rad,
                'angle_48_rad': angle_48_rad,
                'difference_rad': radian_difference
            },
            'closure_analysis': {
                'closure_45': closure_45,
                'closure_48': closure_48,
                'aperture_percentage': aperture_45_48,
                'closure_45_percent': closure_45 * 100,
                'closure_48_percent': closure_48 * 100
            },
            'fractional_relationships': {
                'ratio_48_over_45': ratio_48_45,
                'ratio_45_over_48': ratio_45_48,
                'fraction_48_over_45': Fraction(int(round(self.angle_48)), int(round(self.angle_45))),
                'fraction_45_over_48': Fraction(int(round(self.angle_45)), int(round(self.angle_48))),
                'exact_ratio': '16/15'
            },
            'musical_analysis': {
                'interval_cents': musical_interval_cents,
                'interval_name': 'Major Second (Semitone)',
                'frequency_ratio': ratio_48_45,
                'just_intonation': '16:15'
            }
        }
    
    def compare_with_cgm_aperture(self) -> Dict:
        """Compare 45°→48° aperture with CGM aperture principles."""
        
        # 45°→48° aperture
        aperture_45_48 = (self.angle_48 - self.angle_45) / 90 * 100
        
        # CGM aperture
        cgm_aperture_percent = self.cgm_aperture * 100
        cgm_closure_percent = (1 - self.cgm_aperture) * 100
        
        # Ratios and relationships
        aperture_ratio = cgm_aperture_percent / aperture_45_48
        closure_ratio = (1 - aperture_45_48/100) / (1 - self.cgm_aperture)
        
        # Geometric significance
        # 45° = π/4 (perfect square)
        # 48° = 4π/15 (aperture square)
        pi_4 = math.pi / 4
        four_pi_15 = 4 * math.pi / 15
        difference_rad = four_pi_15 - pi_4
        
        return {
            'aperture_comparison': {
                'cgm_aperture_percent': cgm_aperture_percent,
                'cgm_closure_percent': cgm_closure_percent,
                'angle_45_48_aperture_percent': aperture_45_48,
                'angle_45_48_closure_percent': 100 - aperture_45_48,
                'aperture_ratio_cgm_to_angle': aperture_ratio,
                'closure_ratio_angle_to_cgm': closure_ratio
            },
            'geometric_significance': {
                'angle_45_as_pi_fraction': 'π/4',
                'angle_48_as_pi_fraction': '4π/15',
                'difference_as_pi_fraction': f'{difference_rad/math.pi:.6f}π',
                'perfect_closure_45': '4 × 45° = 180° (perfect square)',
                'aperture_closure_48': '4 × 48° = 192° (aperture square)',
                'aperture_gap': '192° - 180° = 12° total aperture'
            },
            'theoretical_connection': {
                'both_represent_closure_with_aperture': True,
                'cgm_aperture_purpose': 'Allows observation while maintaining structure',
                'angle_aperture_purpose': 'Allows geometric flexibility while maintaining closure',
                'common_principle': 'Incomplete closure enables dynamic observation'
            }
        }
    
    def analyze_geometric_closure_principles(self) -> Dict:
        """Analyze the underlying geometric closure principles."""
        
        # Perfect closure (45°)
        perfect_closure_angles = [45, 90, 135, 180, 225, 270, 315, 360]
        perfect_closure_analysis = {}
        
        for angle in perfect_closure_angles:
            if angle % 45 == 0:
                perfect_closure_analysis[f"{angle}°"] = {
                    'is_perfect_closure': True,
                    'closure_type': 'Perfect geometric closure',
                    'aperture_percentage': 0.0
                }
        
        # Aperture closure (48°)
        aperture_closure_angles = [48, 96, 144, 192, 240, 288, 336]
        aperture_closure_analysis = {}
        
        for angle in aperture_closure_angles:
            if angle % 48 == 0:
                aperture_percent = (angle % 360) / 360 * 100
                aperture_closure_analysis[f"{angle}°"] = {
                    'is_aperture_closure': True,
                    'closure_type': 'Aperture geometric closure',
                    'aperture_percentage': aperture_percent,
                    'closure_percentage': 100 - aperture_percent
                }
        
        # CGM closure analysis
        cgm_closure_analysis = {
            'cgm_closure_percent': (1 - self.cgm_aperture) * 100,
            'cgm_aperture_percent': self.cgm_aperture * 100,
            'closure_type': 'CGM structural closure',
            'aperture_purpose': 'Enables observation while maintaining structure'
        }
        
        return {
            'perfect_closure': perfect_closure_analysis,
            'aperture_closure': aperture_closure_analysis,
            'cgm_closure': cgm_closure_analysis,
            'closure_principles': {
                'perfect_closure': 'Complete geometric closure (45°, 90°, etc.)',
                'aperture_closure': 'Incomplete closure with intentional gap (48°, etc.)',
                'cgm_closure': 'Structural closure with observational aperture (2.07%)',
                'common_theme': 'All represent different levels of geometric completeness'
            }
        }
    
    def generate_comprehensive_analysis(self) -> str:
        """Generate a comprehensive analysis report."""
        
        aperture_data = self.analyze_aperture_transition()
        cgm_comparison = self.compare_with_cgm_aperture()
        closure_principles = self.analyze_geometric_closure_principles()
        
        output = []
        output.append("=" * 80)
        output.append("45° TO 48° APERTURE TRANSITION ANALYSIS")
        output.append("=" * 80)
        output.append("")
        
        # Angular analysis
        output.append("ANGULAR TRANSITION ANALYSIS:")
        output.append("-" * 40)
        ang = aperture_data['angular_analysis']
        output.append(f"45° = {ang['angle_45_deg']}° = {ang['angle_45_rad']:.6f} rad")
        output.append(f"48° = {ang['angle_48_deg']}° = {ang['angle_48_rad']:.6f} rad")
        output.append(f"Difference = {ang['difference_deg']}° = {ang['difference_rad']:.6f} rad")
        output.append(f"Percentage increase = {ang['percentage_increase']:.2f}%")
        output.append("")
        
        # Closure analysis
        output.append("CLOSURE ANALYSIS:")
        output.append("-" * 20)
        closure = aperture_data['closure_analysis']
        output.append(f"45° closure = {closure['closure_45_percent']:.2f}% (perfect square)")
        output.append(f"48° closure = {closure['closure_48_percent']:.2f}% (aperture square)")
        output.append(f"Aperture = {closure['aperture_percentage']:.2f}%")
        output.append("")
        
        # Fractional relationships
        output.append("FRACTIONAL RELATIONSHIPS:")
        output.append("-" * 30)
        frac = aperture_data['fractional_relationships']
        output.append(f"48°/45° = {frac['ratio_48_over_45']:.6f}")
        output.append(f"Exact fraction = {frac['exact_ratio']}")
        output.append(f"Musical interval = {aperture_data['musical_analysis']['interval_name']}")
        output.append(f"Frequency ratio = {frac['ratio_48_over_45']:.6f}")
        output.append("")
        
        # CGM comparison
        output.append("CGM APERTURE COMPARISON:")
        output.append("-" * 30)
        cgm = cgm_comparison['aperture_comparison']
        output.append(f"CGM aperture = {cgm['cgm_aperture_percent']:.2f}%")
        output.append(f"CGM closure = {cgm['cgm_closure_percent']:.2f}%")
        output.append(f"45°→48° aperture = {cgm['angle_45_48_aperture_percent']:.2f}%")
        output.append(f"45°→48° closure = {cgm['angle_45_48_closure_percent']:.2f}%")
        output.append(f"Aperture ratio (CGM/45°→48°) = {cgm['aperture_ratio_cgm_to_angle']:.3f}")
        output.append("")
        
        # Geometric significance
        output.append("GEOMETRIC SIGNIFICANCE:")
        output.append("-" * 25)
        geom = cgm_comparison['geometric_significance']
        output.append(f"45° = {geom['angle_45_as_pi_fraction']} (perfect square)")
        output.append(f"48° = {geom['angle_48_as_pi_fraction']} (aperture square)")
        output.append(f"Difference = {geom['difference_as_pi_fraction']}")
        output.append(f"Perfect closure: {geom['perfect_closure_45']}")
        output.append(f"Aperture closure: {geom['aperture_closure_48']}")
        output.append(f"Total aperture gap: {geom['aperture_gap']}")
        output.append("")
        
        # Theoretical connection
        output.append("THEORETICAL CONNECTION:")
        output.append("-" * 25)
        theory = cgm_comparison['theoretical_connection']
        output.append(f"Both represent closure with aperture: {theory['both_represent_closure_with_aperture']}")
        output.append(f"CGM aperture purpose: {theory['cgm_aperture_purpose']}")
        output.append(f"Angle aperture purpose: {theory['angle_aperture_purpose']}")
        output.append(f"Common principle: {theory['common_principle']}")
        output.append("")
        
        # Closure principles
        output.append("CLOSURE PRINCIPLES:")
        output.append("-" * 20)
        principles = closure_principles['closure_principles']
        for key, value in principles.items():
            output.append(f"{key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(output)
    
    def run_analysis(self):
        """Run the complete analysis."""
        print("Starting 45° to 48° aperture analysis...")
        
        analysis = self.generate_comprehensive_analysis()
        print(analysis)
        
        # Save results
        with open('experiments/results_45_48_aperture_analysis.txt', 'w', encoding='utf-8') as f:
            f.write(analysis)
        
        print(f"\nAnalysis complete! Results saved to 'experiments/results_45_48_aperture_analysis.txt'")

def main():
    """Main function to run the analysis."""
    analyzer = Angle45_48ApertureAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
