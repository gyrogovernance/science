#!/usr/bin/env python3
"""
Angle 48° Analysis Script
=========================

This script analyzes the relationships between 48 degrees and other important angles,
calculating percentages of differences, ratios, and other mathematical relationships.

Author: CGM Analysis
Date: 2025
"""

import math
from typing import Dict, List, Tuple
from fractions import Fraction

class Angle48Analyzer:
    """Analyzer for 48-degree angle relationships."""
    
    def __init__(self):
        self.base_angle = 48.0  # degrees
        self.base_angle_rad = math.radians(self.base_angle)
        
        # Key angles to analyze
        self.key_angles_deg = [12, 24, 30, 36, 45, 60, 72, 90, 120, 135, 150, 180, 270, 360]
        self.pi_angles_rad = [math.pi/6, math.pi/4, math.pi/3, math.pi/2, math.pi, 2*math.pi, 4*math.pi]
        self.pi_angles_deg = [math.degrees(angle) for angle in self.pi_angles_rad]
        
    def calculate_percentage_difference(self, angle1: float, angle2: float) -> float:
        """Calculate percentage difference between two angles."""
        return abs(angle1 - angle2) / max(angle1, angle2) * 100
    
    def calculate_ratio(self, angle1: float, angle2: float) -> float:
        """Calculate ratio of angle1 to angle2."""
        return angle1 / angle2 if angle2 != 0 else float('inf')
    
    def find_common_factors(self, angle1: float, angle2: float) -> List[int]:
        """Find common factors between two angles (as integers)."""
        int1, int2 = int(round(angle1)), int(round(angle2))
        factors1 = [i for i in range(1, int1 + 1) if int1 % i == 0]
        factors2 = [i for i in range(1, int2 + 1) if int2 % i == 0]
        return list(set(factors1) & set(factors2))
    
    def analyze_angle_relationships(self) -> Dict:
        """Analyze relationships between 48° and other angles."""
        results = {
            'base_angle_deg': self.base_angle,
            'base_angle_rad': self.base_angle_rad,
            'key_angle_analysis': {},
            'pi_angle_analysis': {},
            'special_relationships': {}
        }
        
        # Analyze key angles
        for angle in self.key_angles_deg:
            ratio = self.calculate_ratio(self.base_angle, angle)
            pct_diff = self.calculate_percentage_difference(self.base_angle, angle)
            common_factors = self.find_common_factors(self.base_angle, angle)
            
            results['key_angle_analysis'][f"{angle}°"] = {
                'angle_deg': angle,
                'angle_rad': math.radians(angle),
                'ratio_48_to_angle': ratio,
                'ratio_angle_to_48': 1/ratio if ratio != 0 else float('inf'),
                'percentage_difference': pct_diff,
                'common_factors': common_factors,
                'fraction_48_over_angle': Fraction(int(round(self.base_angle)), int(round(angle))),
                'fraction_angle_over_48': Fraction(int(round(angle)), int(round(self.base_angle)))
            }
        
        # Analyze pi-related angles
        for i, angle_rad in enumerate(self.pi_angles_rad):
            angle_deg = self.pi_angles_deg[i]
            ratio = self.calculate_ratio(self.base_angle, angle_deg)
            pct_diff = self.calculate_percentage_difference(self.base_angle, angle_deg)
            
            results['pi_angle_analysis'][f"{angle_rad/math.pi:.2f}π"] = {
                'angle_deg': angle_deg,
                'angle_rad': angle_rad,
                'ratio_48_to_angle': ratio,
                'ratio_angle_to_48': 1/ratio if ratio != 0 else float('inf'),
                'percentage_difference': pct_diff,
                'fraction_48_over_angle': Fraction(int(round(self.base_angle)), int(round(angle_deg))),
                'fraction_angle_over_48': Fraction(int(round(angle_deg)), int(round(self.base_angle)))
            }
        
        # Special relationships
        results['special_relationships'] = {
            '48_as_fraction_of_360': self.base_angle / 360,
            '48_as_fraction_of_180': self.base_angle / 180,
            '48_as_fraction_of_90': self.base_angle / 90,
            '48_as_fraction_of_45': self.base_angle / 45,
            '48_as_fraction_of_24': self.base_angle / 24,
            '48_as_fraction_of_12': self.base_angle / 12,
            'complement_to_90': 90 - self.base_angle,
            'complement_to_180': 180 - self.base_angle,
            'supplement_to_360': 360 - self.base_angle,
            '48_in_radians': self.base_angle_rad,
            '48_as_fraction_of_2pi': self.base_angle_rad / (2 * math.pi),
            '48_as_fraction_of_pi': self.base_angle_rad / math.pi,
            '48_as_fraction_of_pi_over_2': self.base_angle_rad / (math.pi / 2)
        }
        
        return results
    
    def find_harmonic_relationships(self) -> Dict:
        """Find harmonic and musical relationships."""
        harmonics = {}
        
        # Common musical intervals (in cents, where 1200 cents = octave)
        intervals = {
            'unison': 0,
            'minor_second': 100,
            'major_second': 200,
            'minor_third': 300,
            'major_third': 400,
            'perfect_fourth': 500,
            'tritone': 600,
            'perfect_fifth': 700,
            'minor_sixth': 800,
            'major_sixth': 900,
            'minor_seventh': 1000,
            'major_seventh': 1100,
            'octave': 1200
        }
        
        # Convert 48° to cents (degrees * 1200 / 360)
        angle_48_cents = self.base_angle * 1200 / 360
        
        for interval_name, cents in intervals.items():
            # Find how many of this interval fit in 48°
            ratio = angle_48_cents / cents if cents != 0 else float('inf')
            remainder = angle_48_cents % cents if cents != 0 else 0
            
            harmonics[interval_name] = {
                'cents': cents,
                'ratio_in_48_degrees': ratio,
                'remainder_cents': remainder,
                'closest_whole_number': round(ratio) if ratio != float('inf') else 0
            }
        
        return harmonics
    
    def calculate_geometric_properties(self) -> Dict:
        """Calculate geometric properties related to 48°."""
        return {
            'regular_polygon_sides': 360 / self.base_angle,  # How many sides for regular polygon with 48° interior angle
            'exterior_angle': 180 - self.base_angle,  # Exterior angle
            'interior_angle_sum': (360 / self.base_angle - 2) * 180,  # Sum of interior angles
            'central_angle': self.base_angle,  # Central angle in circle
            'arc_length_ratio': self.base_angle / 360,  # Arc length as fraction of circumference
            'sector_area_ratio': self.base_angle / 360,  # Sector area as fraction of circle
            'chord_length_ratio': 2 * math.sin(self.base_angle_rad / 2),  # Chord length as fraction of diameter
            'apothem_ratio': math.cos(self.base_angle_rad / 2),  # Apothem as fraction of radius
        }
    
    def generate_summary_table(self, results: Dict) -> str:
        """Generate a formatted summary table."""
        output = []
        output.append("=" * 80)
        output.append("48° ANGLE RELATIONSHIP ANALYSIS")
        output.append("=" * 80)
        output.append(f"Base angle: {self.base_angle}° = {self.base_angle_rad:.6f} radians")
        output.append("")
        
        # Key angles analysis
        output.append("KEY ANGLES ANALYSIS:")
        output.append("-" * 50)
        output.append(f"{'Angle':<8} {'Ratio 48°/A':<12} {'Ratio A/48°':<12} {'% Diff':<10} {'Fraction':<15}")
        output.append("-" * 50)
        
        for angle_name, data in results['key_angle_analysis'].items():
            ratio_48_to_a = f"{data['ratio_48_to_angle']:.3f}"
            ratio_a_to_48 = f"{data['ratio_angle_to_48']:.3f}"
            pct_diff = f"{data['percentage_difference']:.2f}%"
            fraction = f"{data['fraction_48_over_angle']}"
            
            output.append(f"{angle_name:<8} {ratio_48_to_a:<12} {ratio_a_to_48:<12} {pct_diff:<10} {fraction:<15}")
        
        output.append("")
        
        # Pi angles analysis
        output.append("PI-RELATED ANGLES ANALYSIS:")
        output.append("-" * 50)
        output.append(f"{'Angle':<10} {'Degrees':<10} {'Ratio 48°/A':<12} {'Ratio A/48°':<12} {'% Diff':<10}")
        output.append("-" * 50)
        
        for angle_name, data in results['pi_angle_analysis'].items():
            degrees = f"{data['angle_deg']:.2f}°"
            ratio_48_to_a = f"{data['ratio_48_to_angle']:.3f}"
            ratio_a_to_48 = f"{data['ratio_angle_to_48']:.3f}"
            pct_diff = f"{data['percentage_difference']:.2f}%"
            
            output.append(f"{angle_name:<10} {degrees:<10} {ratio_48_to_a:<12} {ratio_a_to_48:<12} {pct_diff:<10}")
        
        output.append("")
        
        # Special relationships
        output.append("SPECIAL RELATIONSHIPS:")
        output.append("-" * 30)
        for key, value in results['special_relationships'].items():
            if isinstance(value, float):
                output.append(f"{key:<25}: {value:.6f}")
            else:
                output.append(f"{key:<25}: {value}")
        
        return "\n".join(output)
    
    
    def run_complete_analysis(self):
        """Run the complete analysis and generate all outputs."""
        print("Starting 48° angle analysis...")
        
        # Perform analysis
        results = self.analyze_angle_relationships()
        harmonics = self.find_harmonic_relationships()
        geometry = self.calculate_geometric_properties()
        
        # Generate summary
        summary = self.generate_summary_table(results)
        print(summary)
        
        # Print harmonic relationships
        print("\n" + "=" * 50)
        print("HARMONIC RELATIONSHIPS (Musical Intervals)")
        print("=" * 50)
        for interval, data in harmonics.items():
            if data['closest_whole_number'] > 0:
                print(f"{interval:<15}: {data['closest_whole_number']} intervals fit in 48° "
                      f"(remainder: {data['remainder_cents']:.1f} cents)")
        
        # Print geometric properties
        print("\n" + "=" * 50)
        print("GEOMETRIC PROPERTIES")
        print("=" * 50)
        for key, value in geometry.items():
            print(f"{key:<25}: {value:.6f}")
        
        return results, harmonics, geometry

def main():
    """Main function to run the analysis."""
    analyzer = Angle48Analyzer()
    results, harmonics, geometry = analyzer.run_complete_analysis()
    
    # Save detailed results to file
    with open('results_angle_48_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("48° ANGLE ANALYSIS - DETAILED RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        # Write all results
        f.write("KEY ANGLES ANALYSIS:\n")
        for angle, data in results['key_angle_analysis'].items():
            f.write(f"\n{angle}:\n")
            for key, value in data.items():
                f.write(f"  {key}: {value}\n")
        
        f.write("\n\nPI-RELATED ANGLES ANALYSIS:\n")
        for angle, data in results['pi_angle_analysis'].items():
            f.write(f"\n{angle}:\n")
            for key, value in data.items():
                f.write(f"  {key}: {value}\n")
        
        f.write("\n\nSPECIAL RELATIONSHIPS:\n")
        for key, value in results['special_relationships'].items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\n\nHARMONIC RELATIONSHIPS:\n")
        for interval, data in harmonics.items():
            f.write(f"  {interval}: {data}\n")
        
        f.write("\n\nGEOMETRIC PROPERTIES:\n")
        for key, value in geometry.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"\nAnalysis complete! Results saved to 'results_angle_48_analysis.txt'")

if __name__ == "__main__":
    main()
