"""
CGM Geometric Coherence Analysis
===============================

This analysis explores how the Common Governance Model stages manifest in fundamental geometry:
- CS (Common Source): Angular momentum as the chirality generator
- UNA (Unity Non-Absolute): Rotational coherence through circle/sphere geometry  
- ONA (Opposition Non-Absolute): Axial structure through square/cube geometry
- BU (Balance Universal): Aperture balance enabling observation

The story: CS seeds chirality through L, UNA captures it with rotational symmetry,
ONA converts it to axial/tileable structure at calculable cost, BU caps it with aperture m_p.
"""

import math
from typing import Dict, Tuple, List, Any

class CGMGeometricCoherence:
    """
    Analyzes the geometric relationships between rotational (UNA) and axial (ONA) structures
    in the context of angular momentum (CS) and observational aperture (BU).
    """
    
    def __init__(self):
        """Initialize fundamental geometric constants and CGM thresholds."""
        
        # ===== CGM Stage Thresholds =====
        self.cs_threshold = math.pi / 2              # π/2 - chirality seed
        self.una_threshold = math.cos(math.pi / 4)   # 1/√2 - orthogonal projection  
        self.ona_threshold = math.pi / 4             # π/4 - diagonal angle
        self.bu_threshold = 1 / (2 * math.sqrt(2 * math.pi))  # m_p aperture parameter
        self.bu_monodromy = 0.19534217658            # δ_BU monodromy angle
        
        # Derived CGM constants
        self.quantum_geometric_constant = self.ona_threshold / self.bu_threshold  # K_QG ≈ 3.937
        self.una_ona_lift = self.ona_threshold - self.una_threshold  # π/4 - 1/√2 ≈ 0.0783
        
        # ===== Fundamental Shapes (unit scale) =====
        # Circle and square with same "footprint" (inscribed circle in unit square)
        self.square_side = 1.0
        self.circle_radius = 0.5  # inscribed in unit square
        
        # 3D extensions
        self.sphere_radius = 0.5  # inscribed in unit cube
        self.cube_side = 1.0
        
        # ===== Golden ratio for reference =====
        self.golden_ratio = (1 + math.sqrt(5)) / 2
    
    def calculate_una_geometry(self) -> Dict[str, float]:
        """
        UNA Geometry: Rotational Coherence
        Circle and sphere represent maximum rotational symmetry.
        """
        
        # 2D: Circle inscribed in unit square
        circle_area = math.pi * (self.circle_radius ** 2)  # π/4
        circle_circumference = 2 * math.pi * self.circle_radius  # π
        
        # 3D: Sphere inscribed in unit cube  
        sphere_volume = (4/3) * math.pi * (self.sphere_radius ** 3)  # π/6
        sphere_surface = 4 * math.pi * (self.sphere_radius ** 2)  # π
        
        # Isoperimetric quotients - measure of rotational perfection
        q2_circle = self.isoperimetric_quotient_2d(circle_area, circle_circumference)  # = 1.0
        q3_sphere = self.isoperimetric_quotient_3d(sphere_volume, sphere_surface)      # = 1.0
        
        # Angular momentum storage (rotational inertia per unit mass)
        inertia_disk_2d = 0.5 * (self.circle_radius ** 2)  # I/M = r²/2
        inertia_sphere_3d = 0.4 * (self.sphere_radius ** 2)  # I/M = 2r²/5
        
        # Polar radius of gyration
        k_disk = self.circle_radius / math.sqrt(2)  # r/√2
        
        return {
            'circle_area': circle_area,
            'circle_circumference': circle_circumference, 
            'sphere_volume': sphere_volume,
            'sphere_surface': sphere_surface,
            'q2_perfect_rotation': q2_circle,
            'q3_perfect_rotation': q3_sphere,
            'angular_storage_2d': inertia_disk_2d,
            'angular_storage_3d': inertia_sphere_3d,
            'polar_radius_gyration': k_disk
        }
    
    def calculate_ona_geometry(self) -> Dict[str, float]:
        """
        ONA Geometry: Axial Structure  
        Square and cube represent axial organization.
        """
        
        # 2D: Unit square
        square_area = self.square_side ** 2  # = 1.0
        square_perimeter = 4 * self.square_side  # = 4.0
        
        # 3D: Unit cube
        cube_volume = self.cube_side ** 3  # = 1.0  
        cube_surface = 6 * (self.cube_side ** 2)  # = 6.0
        
        # Isoperimetric quotients - deviation from rotational perfection
        q2_square = self.isoperimetric_quotient_2d(square_area, square_perimeter)  # π/4
        q3_cube = self.isoperimetric_quotient_3d(cube_volume, cube_surface)        # π/6
        
        # Rotational deficits (cost of axial structure)
        rotational_deficit_2d = 1.0 - q2_square  # 1 - π/4
        rotational_deficit_3d = 1.0 - q3_cube    # 1 - π/6
        
        # Angular momentum storage (less efficient due to edge/corner concentration)
        inertia_square_2d = (1/6) * (self.square_side ** 2)  # I/M = a²/6
        inertia_cube_3d = (1/6) * (self.cube_side ** 2)      # I/M = a²/6 (about one axis)
        
        # Polar radius of gyration
        k_square = 1 / math.sqrt(6)  # 1/√6
        
        # Diagonal relationships
        diagonal_length = math.sqrt(2)  # diagonal of unit square
        
        return {
            'square_area': square_area,
            'square_perimeter': square_perimeter,
            'cube_volume': cube_volume, 
            'cube_surface': cube_surface,
            'q2_axial_structure': q2_square,
            'q3_axial_structure': q3_cube,
            'rotational_deficit_2d': rotational_deficit_2d,
            'rotational_deficit_3d': rotational_deficit_3d,
            'angular_storage_2d': inertia_square_2d,
            'angular_storage_3d': inertia_cube_3d,
            'polar_radius_gyration': k_square,
            'diagonal_length': diagonal_length
        }
    
    def calculate_cs_bridge(self) -> Dict[str, float]:
        """
        CS Bridge: Angular Momentum Relationships
        """
        
        una = self.calculate_una_geometry()
        ona = self.calculate_ona_geometry()
        
        # Angular momentum ratios
        angular_ratio_2d = ona['angular_storage_2d'] / una['angular_storage_2d']  # (1/6)/(1/8) = 4/3
        angular_ratio_3d = ona['angular_storage_3d'] / una['angular_storage_3d']  # (1/6)/(1/10) = 5/3
        
        # Polar radii of gyration ratios
        k_ratio = ona['polar_radius_gyration'] / una['polar_radius_gyration']  # 2/√3
        
        # Geometric efficiency ratios  
        area_efficiency = una['circle_area'] / ona['square_area']  # π/4
        volume_efficiency = una['sphere_volume'] / ona['cube_volume']  # π/6
        
        # The key CS signature: how these ratios relate to CGM thresholds
        area_to_una_threshold = area_efficiency / self.una_threshold  # (π/4)/(1/√2) = π√2/4
        area_to_ona_threshold = area_efficiency / self.ona_threshold  # (π/4)/(π/4) = 1.0
        
        return {
            'angular_ratio_2d': angular_ratio_2d,
            'angular_ratio_3d': angular_ratio_3d,
            'gyration_ratio': k_ratio,
            'area_efficiency_una_over_ona': area_efficiency,
            'volume_efficiency_una_over_ona': volume_efficiency,
            'area_una_threshold_ratio': area_to_una_threshold,
            'area_ona_threshold_ratio': area_to_ona_threshold,
            'cs_to_una_threshold_ratio': self.cs_threshold / self.una_threshold,
            'cs_to_ona_threshold_ratio': self.cs_threshold / self.ona_threshold
        }
    
    def calculate_triangles(self) -> Dict[str, Dict]:
        """Analyze fundamental triangles and CGM gyrotriangle."""
        
        # 45-45-90 triangle (square's diagonal - ONA representation)
        triangle_45_45_90 = {
            'angles': [math.pi/4, math.pi/4, math.pi/2],
            'sides_ratio': [1, 1, math.sqrt(2)],
            'area_ratio': 0.5,  # Area = (1*1)/2 = 0.5 relative to square
            'diagonal_side_ratio': math.sqrt(2)
        }
        
        # 30-60-90 triangle (UNA representation - rotational symmetry)
        triangle_30_60_90 = {
            'angles': [math.pi/6, math.pi/3, math.pi/2],
            'sides_ratio': [1, math.sqrt(3), 2],
            'area_ratio': math.sqrt(3)/2,  # ≈ 0.866
            'height_base_ratio': math.sqrt(3)
        }
        
        # CGM gyrotriangle with angles π/2, π/4, π/4
        angles = [math.pi/2, math.pi/4, math.pi/4]
        defect = math.pi - sum(angles)  # Should be 0 for closure
        
        # Side ratios using law of sines
        side_a = math.sin(angles[0])  # sin(π/2) = 1
        side_b = math.sin(angles[1])  # sin(π/4) = √2/2 ≈ 0.7071
        side_c = math.sin(angles[2])  # sin(π/4) = √2/2 ≈ 0.7071
        
        area = 0.5 * side_b * side_c * math.sin(angles[0])  # = 0.25
        
        cgm_gyrotriangle = {
            'angles': angles,
            'defect': defect,
            'side_ratios': [side_a, side_b, side_c],
            'area': area
        }
        
        return {
            'triangle_45_45_90': triangle_45_45_90,
            'triangle_30_60_90': triangle_30_60_90,
            'cgm_gyrotriangle': cgm_gyrotriangle
        }
    
    def calculate_aperture_pair(self) -> Dict[str, Any]:
        """Analyze the 45°-48° aperture pair (auxiliary construction)."""
        
        # Angles in radians
        angle_45 = math.pi/4  # 45°
        angle_48 = 4*math.pi/15  # 48° exactly
        
        # Beat (difference)
        beat_rad = angle_48 - angle_45  # π/60 ≈ 0.05236 rad = 3°
        beat_deg = math.degrees(beat_rad)
        
        # Reduced fractions
        r_45 = angle_45 / (2 * math.pi)  # 1/8
        r_48 = angle_48 / (2 * math.pi)  # 2/15
        r_beat = r_48 - r_45  # 1/120
        
        # Minimal common closure: m * 45° = n * 48° = common (minimal 720°)
        m, n = 16, 15  # From 45m = 48n
        common_deg = 45 * m  # 720°
        
        # Relation to golden ratio
        ratio_16_15 = 16/15
        to_golden = self.golden_ratio / ratio_16_15
        
        # Duality angle from ONA/UNA actions
        tan_48 = math.tan(angle_48)  # Should be close to ONA/UNA action ratio
        
        return {
            'angle_45_rad': angle_45,
            'angle_48_rad': angle_48,
            'beat_rad': beat_rad,
            'beat_deg': beat_deg,
            'r_45': r_45,
            'r_48': r_48,
            'r_beat': r_beat,
            'common_steps_45': m,
            'common_steps_48': n,
            'common_deg': common_deg,
            'ratio_16_15': ratio_16_15,
            'to_golden': to_golden,
            'tan_48': tan_48
        }
    
    def continued_fraction(self, x: float, max_terms: int = 12) -> List[int]:
        """Compute continued fraction for x."""
        a = []
        y = x
        for _ in range(max_terms):
            ai = math.floor(y)
            a.append(ai)
            frac = y - ai
            if frac < 1e-14:
                break
            y = 1 / frac
        return a
    
    def convergents(self, cf: List[int]) -> List[tuple]:
        """Compute convergents (p/q) from continued fraction."""
        if not cf:
            return []
        conv = []
        p0, q0 = 0, 1
        p1, q1 = 1, 0
        for a in cf:
            p2 = a * p1 + p0
            q2 = a * q1 + q0
            conv.append((p2, q2))
            p0, q0 = p1, q1
            p1, q1 = p2, q2
        return conv
    
    def calculate_normalizations(self) -> Dict[str, Dict]:
        """Calculate different normalization modes."""
        
        # Equal perimeter normalization (P=4)
        P = 4.0
        s = P/4  # Square side
        r = P/(2*math.pi)  # Circle radius
        equal_perim = {
            'square_area': s*s,
            'circle_area': math.pi * r*r,
            'area_ratio_circle_over_square': (math.pi * r*r)/(s*s),  # = 4/π
            'perimeter': P
        }
        
        # Equal area normalization (A=1)
        A = 1.0
        P_sq = 4*math.sqrt(A)
        P_circ = 2*math.sqrt(math.pi*A)
        equal_area = {
            'square_perimeter': P_sq,
            'circle_perimeter': P_circ,
            'perimeter_ratio_square_over_circle': P_sq/P_circ,  # = 2/√π
            'area': A
        }
        
        # Equal diameter normalization (d=1)
        d = 1.0
        A_circ = math.pi * (d*d) / 4
        A_sq = (d*d)/2  # Square with diagonal = d
        equal_diam = {
            'square_area': A_sq,
            'circle_area': A_circ,
            'area_ratio_circle_over_square': A_circ/A_sq,  # = π/2
            'diameter': d
        }
        
        return {
            'equal_perimeter': equal_perim,
            'equal_area': equal_area,
            'equal_diameter': equal_diam
        }
    
    def calculate_packing(self) -> Dict[str, float]:
        """Calculate circle packing densities."""
        
        phi_square = math.pi/4  # Square lattice
        phi_triangular = math.pi/(2*math.sqrt(3))  # Triangular/hexagonal lattice
        improvement = phi_triangular / phi_square  # = 2/√3
        
        return {
            'square_packing': phi_square,
            'triangular_packing': phi_triangular,
            'improvement_factor': improvement
        }
    
    def calculate_dimensional_transitions(self) -> Dict[str, float]:
        """Dimensional Transitions: 2D → 3D Scaling"""
        
        una = self.calculate_una_geometry()
        ona = self.calculate_ona_geometry()
        
        # 2D ratios
        ratio_2d_area = una['circle_area'] / ona['square_area']  # π/4
        ratio_2d_perimeter = una['circle_circumference'] / ona['square_perimeter']  # π/4
        
        # 3D ratios  
        ratio_3d_volume = una['sphere_volume'] / ona['cube_volume']  # π/6
        ratio_3d_surface = una['sphere_surface'] / ona['cube_surface']  # π/6
        
        # Dimensional scaling factors
        volume_to_area_scaling = ratio_3d_volume / ratio_2d_area  # (π/6)/(π/4) = 2/3
        surface_to_perimeter_scaling = ratio_3d_surface / ratio_2d_perimeter  # 2/3
        
        # Angular momentum scaling
        # UNA: 0.1/0.125 = 0.8 = 4/5 (since I/M = 0.4 r^2 with r=0.5 → 0.1)
        angular_3d_to_2d_una = una['angular_storage_3d'] / una['angular_storage_2d']  # 0.8
        angular_3d_to_2d_ona = ona['angular_storage_3d'] / ona['angular_storage_2d']  # 1.0
        
        return {
            'una_over_ona_2d_area': ratio_2d_area,
            'una_over_ona_2d_perimeter': ratio_2d_perimeter,
            'una_over_ona_3d_volume': ratio_3d_volume,
            'una_over_ona_3d_surface': ratio_3d_surface,
            'dimensional_scaling_volume': volume_to_area_scaling,
            'dimensional_scaling_surface': surface_to_perimeter_scaling,
            'angular_scaling_una': angular_3d_to_2d_una,
            'angular_scaling_ona': angular_3d_to_2d_ona
        }
    
    def calculate_aperture_balance(self) -> Dict[str, float]:
        """BU Aperture Balance: The 97.93%/2.07% Split"""
        
        # The fundamental balance: Q_G × m_p² = 1/2
        q_g = 4 * math.pi  # Complete solid angle
        balance_check = q_g * (self.bu_threshold ** 2)  # Should equal 0.5
        
        # Correct calculation: The aperture creates 2.07% opening
        # From CGM theory: m_p² = 1/(8π), so 8π × m_p² = 1
        # The closure/aperture split comes from the specific geometric construction
        aperture_fraction = 0.0207  # Model constant from CGM (aperture split); not derived in this function
        closure_fraction = 1 - aperture_fraction  # 97.93%
        
        # Relationships to other thresholds
        quantum_geometric = self.ona_threshold / self.bu_threshold  # K_QG
        cs_to_bu_ratio = self.cs_threshold / self.bu_threshold
        
        return {
            'quantum_gravity_qg': q_g,
            'aperture_parameter_mp': self.bu_threshold,
            'balance_verification': balance_check,
            'structural_closure_percent': closure_fraction * 100,
            'dynamic_aperture_percent': aperture_fraction * 100,
            'quantum_geometric_constant': quantum_geometric,
            'cs_amplification_factor': cs_to_bu_ratio
        }
    
    def calculate_curvature_invariants(self) -> Dict[str, float]:
        """Calculate curvature and turning invariants."""
        
        # Total turning for closed convex curves
        total_turning = 2 * math.pi  # Invariant for all closed convex curves
        
        # Mean width from Cauchy formula
        square_perimeter = 4.0
        circle_circumference = math.pi
        mean_width_square = square_perimeter / math.pi  # 4/π
        mean_width_circle = circle_circumference / math.pi  # 1.0
        
        # Gaussian curvature integral
        sphere_gauss_bonnet = 4 * math.pi  # Total Gaussian curvature for sphere
        
        return {
            'total_turning_2d': total_turning,
            'sphere_total_gaussian': sphere_gauss_bonnet,
            'mean_width_square': mean_width_square,
            'mean_width_circle': mean_width_circle
        }
    
    def isoperimetric_quotient_2d(self, area: float, perimeter: float) -> float:
        """2D isoperimetric quotient Q2 = 4πA/P² (1.0 for circle, <1.0 for others)"""
        return 4 * math.pi * area / (perimeter ** 2)
    
    def isoperimetric_quotient_3d(self, volume: float, surface: float) -> float:
        """3D isoperimetric quotient Q3 = 36πV²/S³ (1.0 for sphere, <1.0 for others)"""
        return 36 * math.pi * (volume ** 2) / (surface ** 3)
    
    def print_coherence_analysis(self):
        """Print the complete geometric coherence analysis as a coherent story."""
        
        print("=" * 80)
        print("CGM GEOMETRIC COHERENCE ANALYSIS")
        print("Circle-Sphere (UNA) ↔ Square-Cube (ONA) ↔ Angular Momentum (CS)")
        print("=" * 80)
        
        # ===== Stage 1: CGM Thresholds =====
        print("\n1. CGM STAGE THRESHOLDS")
        print("-" * 40)
        print(f"   CS (Common Source - Chirality Seed):     {self.cs_threshold:.6f} = π/2")
        print(f"   UNA (Unity Non-Absolute - Projection):   {self.una_threshold:.6f} = 1/√2") 
        print(f"   ONA (Opposition Non-Absolute - Angle):   {self.ona_threshold:.6f} = π/4")
        print(f"   BU (Balance Universal - Aperture):       {self.bu_threshold:.6f} = 1/(2√(2π))")
        print(f"   BU Monodromy δ_BU:                      {self.bu_monodromy:.6f} rad")
        print(f"\n   UNA→ONA Lift:                           {self.una_ona_lift:.6f} = π/4 - 1/√2")
        print(f"   Quantum Geometric Constant K_QG:         {self.quantum_geometric_constant:.6f} = (π/4)/m_p")
        
        # ===== Stage 2: UNA Analysis =====
        una = self.calculate_una_geometry()
        print("\n2. UNA GEOMETRY: ROTATIONAL COHERENCE")
        print("-" * 40)
        print("   Circle/Sphere = Maximum rotational symmetry")
        print(f"\n   2D Circle (inscribed, r=0.5):")
        print(f"   • Area:                      {una['circle_area']:.6f} = π/4")
        print(f"   • Circumference:             {una['circle_circumference']:.6f} = π") 
        print(f"   • Isoperimetric Q2:          {una['q2_perfect_rotation']:.6f} = 1.0 (perfect)")
        print(f"   • Angular storage I/M:       {una['angular_storage_2d']:.6f} = r²/2")
        print(f"   • Polar radius of gyration:  {una['polar_radius_gyration']:.6f} = r/√2")
        
        print(f"\n   3D Sphere (inscribed, r=0.5):")
        print(f"   • Volume:                    {una['sphere_volume']:.6f} = π/6")
        print(f"   • Surface area:              {una['sphere_surface']:.6f} = π")
        print(f"   • Isoperimetric Q3:          {una['q3_perfect_rotation']:.6f} = 1.0 (perfect)")
        print(f"   • Angular storage I/M:       {una['angular_storage_3d']:.6f} = 2r²/5")
        
        # ===== Stage 3: ONA Analysis =====
        ona = self.calculate_ona_geometry()
        print("\n3. ONA GEOMETRY: AXIAL STRUCTURE")
        print("-" * 40)
        print("   Square/Cube = Discrete axial organization")  
        print(f"\n   2D Square (unit side):")
        print(f"   • Area:                      {ona['square_area']:.6f} = 1.0")
        print(f"   • Perimeter:                 {ona['square_perimeter']:.6f} = 4.0")
        print(f"   • Isoperimetric Q2:          {ona['q2_axial_structure']:.6f} = π/4")
        print(f"   • Rotational deficit:        {ona['rotational_deficit_2d']:.6f} = 1 - π/4")
        print(f"   • Angular storage I/M:       {ona['angular_storage_2d']:.6f} = a²/6")
        print(f"   • Polar radius of gyration:  {ona['polar_radius_gyration']:.6f} = 1/√6")
        print(f"   • Diagonal length:           {ona['diagonal_length']:.6f} = √2")
        
        print(f"\n   3D Cube (unit side):")
        print(f"   • Volume:                    {ona['cube_volume']:.6f} = 1.0")
        print(f"   • Surface area:              {ona['cube_surface']:.6f} = 6.0") 
        print(f"   • Isoperimetric Q3:          {ona['q3_axial_structure']:.6f} = π/6")
        print(f"   • Rotational deficit:        {ona['rotational_deficit_3d']:.6f} = 1 - π/6")
        print(f"   • Angular storage I/M:       {ona['angular_storage_3d']:.6f} = a²/6")
        
        # ===== Stage 4: Triangles =====
        triangles = self.calculate_triangles()
        print("\n4. TRIANGLE ANALYSIS")
        print("-" * 40)
        
        print(f"   45-45-90 Triangle (ONA diagonal):")
        t45 = triangles['triangle_45_45_90']
        print(f"   • Angles: {[f'{a:.4f}' for a in t45['angles']]}")
        print(f"   • Side ratios: {[f'{s:.4f}' for s in t45['sides_ratio']]}")
        print(f"   • Area ratio to square: {t45['area_ratio']:.6f}")
        print(f"   • Diagonal/Side: {t45['diagonal_side_ratio']:.6f} = √2")
        
        print(f"\n   30-60-90 Triangle (UNA rotational):")
        t30 = triangles['triangle_30_60_90']
        print(f"   • Angles: {[f'{a:.4f}' for a in t30['angles']]}")
        print(f"   • Side ratios: {[f'{s:.4f}' for s in t30['sides_ratio']]}")
        print(f"   • Area ratio: {t30['area_ratio']:.6f}")
        print(f"   • Height/Base: {t30['height_base_ratio']:.6f} = √3")
        
        print(f"\n   CGM Gyrotriangle (π/2, π/4, π/4):")
        cgm_tri = triangles['cgm_gyrotriangle']
        print(f"   • Angles: {[f'{a:.4f}' for a in cgm_tri['angles']]}")
        print(f"   • Defect: {cgm_tri['defect']:.6f} (closure)")
        print(f"   • Side ratios: {[f'{s:.4f}' for s in cgm_tri['side_ratios']]}")
        print(f"   • Area: {cgm_tri['area']:.6f}")
        
        # ===== Stage 5: CS Bridge =====
        cs = self.calculate_cs_bridge()
        print("\n5. CS BRIDGE: ANGULAR MOMENTUM RELATIONSHIPS")
        print("-" * 40)
        print(f"   Angular Storage Ratios (ONA/UNA):")
        print(f"   • 2D ratio:                  {cs['angular_ratio_2d']:.6f} = 4/3")
        print(f"   • 3D ratio:                  {cs['angular_ratio_3d']:.6f} = 5/3") 
        print(f"   • Gyration ratio k_sq/k_disk: {cs['gyration_ratio']:.6f} = 2/√3")
        
        print(f"\n   Geometric Efficiency (UNA/ONA):")
        print(f"   • 2D area efficiency:        {cs['area_efficiency_una_over_ona']:.6f} = π/4")
        print(f"   • 3D volume efficiency:      {cs['volume_efficiency_una_over_ona']:.6f} = π/6")
        
        print(f"\n   Threshold Relationships:")
        print(f"   • Area/UNA threshold:        {cs['area_una_threshold_ratio']:.6f} = π√2/4")
        print(f"   • Area/ONA threshold:        {cs['area_ona_threshold_ratio']:.6f} = 1.0 (exact)")
        print(f"   • CS/UNA threshold:          {cs['cs_to_una_threshold_ratio']:.6f} = π√2/2") 
        print(f"   • CS/ONA threshold:          {cs['cs_to_ona_threshold_ratio']:.6f} = 2.0 (exact)")
        
        # ===== Stage 6: Dimensional Transitions =====
        dim = self.calculate_dimensional_transitions()
        print("\n6. DIMENSIONAL TRANSITIONS: 2D → 3D SCALING")
        print("-" * 40)
        print(f"   2D Ratios (UNA/ONA):")
        print(f"   • Area ratio:                {dim['una_over_ona_2d_area']:.6f} = π/4")
        print(f"   • Perimeter ratio:           {dim['una_over_ona_2d_perimeter']:.6f} = π/4")
        
        print(f"\n   3D Ratios (UNA/ONA):")  
        print(f"   • Volume ratio:              {dim['una_over_ona_3d_volume']:.6f} = π/6")
        print(f"   • Surface ratio:             {dim['una_over_ona_3d_surface']:.6f} = π/6")
        
        print(f"\n   Dimensional Scaling (3D/2D):")
        print(f"   • Volume/Area scaling:       {dim['dimensional_scaling_volume']:.6f} = 2/3")
        print(f"   • Surface/Perimeter scaling: {dim['dimensional_scaling_surface']:.6f} = 2/3")
        print(f"   • UNA angular scaling:       {dim['angular_scaling_una']:.6f}")
        print(f"   • ONA angular scaling:       {dim['angular_scaling_ona']:.6f} = 1.0")
        
        # ===== Stage 7: Aperture Pair =====
        aperture_pair = self.calculate_aperture_pair()
        print("\n7. APERTURE PAIR ANALYSIS (45°-48°)")
        print("-" * 40)
        print("   Auxiliary construction showing beat frequencies")
        print(f"   • 45° angle:                 {aperture_pair['angle_45_rad']:.6f} rad = π/4")
        print(f"   • 48° angle:                 {aperture_pair['angle_48_rad']:.6f} rad = 4π/15")
        print(f"   • Beat frequency:            {aperture_pair['beat_rad']:.6f} rad = {aperture_pair['beat_deg']:.1f}°")
        print(f"   • Common closure:            {aperture_pair['common_steps_45']}×45° = {aperture_pair['common_steps_48']}×48° = {aperture_pair['common_deg']}°")
        print(f"   • 16/15 ratio:               {aperture_pair['ratio_16_15']:.6f}")
        print(f"   • To golden ratio:           {aperture_pair['to_golden']:.6f}")
        print(f"   • tan(48°):                  {aperture_pair['tan_48']:.6f}")
        
        # ===== Stage 8: Continued Fractions =====
        print("\n8. CONTINUED FRACTION ANALYSIS")
        print("-" * 40)
        rho_bu = self.bu_monodromy / (2 * math.pi)
        cf = self.continued_fraction(rho_bu)
        convs = self.convergents(cf)
        print(f"   BU monodromy ratio ρ_BU = δ_BU/(2π) = {rho_bu:.10f}")
        print(f"   Continued fraction: {cf[:8]}...")
        if len(convs) >= 3:
            print(f"   Key convergents: {convs[0]}, {convs[1]}, {convs[2]}...")
            print(f"   Final convergent: {convs[-1]} (near-closure after {convs[-1][1]} steps)")
        
        # ===== Stage 9: Normalizations =====
        norms = self.calculate_normalizations()
        print("\n9. NORMALIZATION MODES")
        print("-" * 40)
        
        eq_p = norms['equal_perimeter']
        print(f"   Equal Perimeter (P=4):")
        print(f"   • Square area:               {eq_p['square_area']:.6f}")
        print(f"   • Circle area:               {eq_p['circle_area']:.6f}")
        print(f"   • Area ratio (circle/square): {eq_p['area_ratio_circle_over_square']:.6f} = 4/π")
        
        eq_a = norms['equal_area']
        print(f"\n   Equal Area (A=1):")
        print(f"   • Square perimeter:          {eq_a['square_perimeter']:.6f}")
        print(f"   • Circle perimeter:          {eq_a['circle_perimeter']:.6f}")
        print(f"   • Perimeter ratio (sq/circ): {eq_a['perimeter_ratio_square_over_circle']:.6f} = 2/√π")
        
        eq_d = norms['equal_diameter']
        print(f"\n   Equal Diameter (d=1):")
        print(f"   • Square area:               {eq_d['square_area']:.6f}")
        print(f"   • Circle area:               {eq_d['circle_area']:.6f}")
        print(f"   • Area ratio (circle/square): {eq_d['area_ratio_circle_over_square']:.6f} = π/2")
        
        # ===== Stage 10: Packing =====
        packing = self.calculate_packing()
        print("\n10. CIRCLE PACKING DENSITIES")
        print("-" * 40)
        print(f"   Square lattice:              {packing['square_packing']:.6f} = π/4")
        print(f"   Triangular lattice:          {packing['triangular_packing']:.6f} = π/(2√3)")
        print(f"   Improvement factor:          {packing['improvement_factor']:.6f} = 2/√3")
        
        # ===== Stage 11: Curvature =====
        curv = self.calculate_curvature_invariants()
        print("\n11. CURVATURE AND TURNING INVARIANTS")
        print("-" * 40)
        print(f"   Total turning (2D closed):   {curv['total_turning_2d']:.6f} = 2π")
        print(f"   Sphere Gaussian curvature:   {curv['sphere_total_gaussian']:.6f} = 4π")
        print(f"   Mean width square:           {curv['mean_width_square']:.6f} = 4/π")
        print(f"   Mean width circle:           {curv['mean_width_circle']:.6f} = 1.0")
        
        # ===== Stage 12: Aperture Balance =====
        aperture = self.calculate_aperture_balance()
        print("\n12. BU APERTURE BALANCE: THE 97.93%/2.07% SPLIT")
        print("-" * 40)
        print(f"   Quantum Gravity Constant:    Q_G = {aperture['quantum_gravity_qg']:.6f} = 4π")
        print(f"   Aperture Parameter:          m_p = {aperture['aperture_parameter_mp']:.6f}")
        print(f"   Balance Verification:        Q_G × m_p² = {aperture['balance_verification']:.6f} = 1/2")
        
        print(f"\n   Universal Balance Split:")
        print(f"   • Structural closure:        {aperture['structural_closure_percent']:.2f}%")
        print(f"   • Dynamic aperture:          {aperture['dynamic_aperture_percent']:.2f}%")
        
        print(f"\n   Amplification Factors:")
        print(f"   • Quantum geometric K_QG:    {aperture['quantum_geometric_constant']:.6f}")
        print(f"   • CS amplification:          {aperture['cs_amplification_factor']:.6f}")
        
        # ===== Final Summary =====
        print("\n13. KEY INSIGHTS: THE GEOMETRIC STORY")
        print("-" * 40)
        print("   CS seeds chirality through angular momentum L")
        print("   UNA captures it with perfect rotational coherence (circles/spheres)")
        print("   ONA converts it to discrete axial structure (squares/cubes)")
        print("   BU balances structure vs openness with m_p aperture")
        
        print(f"\n   The π/4 signature appears as:")
        print(f"   • ONA threshold = π/4")
        print(f"   • Circle/Square area ratio = π/4")
        print(f"   • Square isoperimetric Q2 = π/4")
        
        print(f"\n   The 2/√3 factor connects:")
        print(f"   • Polar radii of gyration ratio")
        print(f"   • Triangular vs square packing improvement")
        
        print(f"\n   Angular momentum costs:")
        print(f"   • ONA needs 4/3 more angular storage than UNA in 2D")
        print(f"   • ONA needs 5/3 more angular storage than UNA in 3D")
        print(f"   • This quantifies the cost of discrete structure")
        
        # ===== Stage 13: Quantum Gravity Q_G = 4π Connections =====
        qg_conn = self.calculate_qg_connections()
        print("\n14. QUANTUM GRAVITY Q_G = 4π CONNECTIONS")
        print("-" * 40)
        print("   How the quantum gravity invariant Q_G = 4π appears everywhere:")
        print(f"   • Q_G value:                    {qg_conn['q_g_value']:.6f} = 4π")
        print(f"   • Sphere surface factor:        {qg_conn['sphere_surface_factor']:.6f} = 4π")
        print(f"   • Complete solid angle:         {qg_conn['solid_angle_complete']:.6f} = 4π")
        print(f"   • Gauss-Bonnet sphere:          {qg_conn['gauss_bonnet_sphere']:.6f} = 4π")

        print(f"\n   Geometric content analysis:")
        print(f"   • Circle area contains:         {qg_conn['circle_pi_content']:.6f} = π")
        print(f"   • Sphere surface contains:      {qg_conn['sphere_4pi_content']:.6f} = 4π")
        print(f"   • Hemisphere solid angle:       {qg_conn['solid_angle_2d']:.6f} = 2π")
        print(f"   • Solid angle 3D (full):        {qg_conn['solid_angle_3d']:.6f} = 4π")
        print(f"   • Solid angle scaling:          {qg_conn['solid_angle_scaling']:.6f} = 1/2")

        print(f"\n   Quantum gravity commutator:")
        print(f"   • K_QG theoretical:             {qg_conn['k_qg_theoretical']:.6f} = π²/√(2π)")
        print(f"   • K_QG empirical:               {qg_conn['k_qg_empirical']:.6f} = (π/4)/m_p")
        print(f"   • π² as Q_G × (π/4):            {qg_conn['pi_squared_relation']:.6f} = 4π × π/4")

        print(f"\n   Optical conjugacy and gravity dilution:")
        print(f"   • Optical conjugacy factor:     {qg_conn['optical_conjugacy_factor']:.6f} = 4π²")
        print(f"   • Gravity dilution:             {qg_conn['gravity_dilution_factor']:.6f} = 1/(4π²)")

        print(f"\n   Key insight: π/4 = 1/16 of Q_G")
        print(f"   • π/4 as fraction of Q_G:       {qg_conn['pi_over_4_as_fraction_of_qg']:.6f} = 1/16")
        print(f"   • 2/√3 as fraction of Q_G:      {qg_conn['two_over_root3_as_fraction_of_qg']:.6f}")

        print(f"\n   This means:")
        print(f"   • The ONA threshold π/4 is exactly 1/16 of the quantum gravity invariant")
        print(f"   • All our geometric ratios are fractions of the complete solid angle Q_G")
        print(f"   • Gravity's apparent weakness comes from 4π² geometric dilution")
        print(f"   • The sphere's 4π surface is the physical manifestation of Q_G")
        
        print("\n" + "=" * 80)
    
    def calculate_qg_connections(self) -> Dict[str, Any]:
        """Show how Q_G = 4π connects to all the geometric relationships."""
        
        q_g = 4 * math.pi
        
        # ===== Direct connections to 4π =====
        
        # 1. Sphere surface area: 4πr² - the 4π comes directly from Q_G
        sphere_surface_factor = 4 * math.pi  # Q_G itself
        
        # 2. Solid angle coverage: complete sphere = 4π steradians
        solid_angle_complete = q_g
        
        # 3. Gaussian curvature integral for sphere: ∫K dA = 4π (Gauss-Bonnet)
        gauss_bonnet_sphere = q_g
        
        # 4. Relationship to aperture parameter: Q_G × m_p² = 1/2
        aperture_balance = q_g * (self.bu_threshold ** 2)
        
        # ===== How Q_G appears in the geometric ratios =====
        
        una = self.calculate_una_geometry()
        ona = self.calculate_ona_geometry()
        
        # Circle area contains π, sphere surface contains 4π
        circle_area_contains_pi = una['circle_area'] / (self.circle_radius ** 2)  # = π
        sphere_surface_contains_4pi = una['sphere_surface'] / (self.sphere_radius ** 2)  # = 4π
        
        # The 2/3 dimensional scaling relates to solid angle fractions
        # 2D: covers π steradians (hemisphere), 3D: covers 4π steradians (full sphere)
        solid_angle_2d = 2 * math.pi  # hemisphere
        solid_angle_3d = q_g          # full sphere
        solid_angle_scaling = solid_angle_2d / solid_angle_3d  # = 1/2
        
        # But our dimensional scaling was 2/3, not 1/2
        # This suggests the remaining factor comes from volume vs surface considerations
        
        # ===== Quantum gravity commutator connection =====
        
        # From CGM: [X,P] = iK_QG, where K_QG = π²/√(2π)
        # But K_QG also = S_CS/2 = (π/4)/m_p ≈ 3.9374
        k_qg_theoretical = (math.pi ** 2) / math.sqrt(2 * math.pi)
        k_qg_empirical = self.quantum_geometric_constant
        
        # The factor π² connects to 4π through: π² = (4π) × (π/4)
        pi_squared_relation = q_g * (math.pi / 4)
        
        # ===== Optical conjugacy and 4π dilution =====
        
        # The optical conjugacy relation: E^UV × E^IR = const/(4π²)
        # The 4π² factor = (4π) × π = Q_G × π
        optical_conjugacy_factor = q_g * math.pi
        
        # This explains why gravity appears weak: geometric dilution by 4π²
        gravity_dilution_factor = 1 / (optical_conjugacy_factor)  # ≈ 1/39.48
        
        return {
            'q_g_value': q_g,
            'sphere_surface_factor': sphere_surface_factor,
            'solid_angle_complete': solid_angle_complete,
            'gauss_bonnet_sphere': gauss_bonnet_sphere,
            'aperture_balance_check': aperture_balance,
            
            'circle_pi_content': circle_area_contains_pi,
            'sphere_4pi_content': sphere_surface_contains_4pi,
            'solid_angle_2d': solid_angle_2d,
            'solid_angle_3d': solid_angle_3d,
            'solid_angle_scaling': solid_angle_scaling,
            
            'k_qg_theoretical': k_qg_theoretical,
            'k_qg_empirical': k_qg_empirical,
            'pi_squared_relation': pi_squared_relation,
            
            'optical_conjugacy_factor': optical_conjugacy_factor,
            'gravity_dilution_factor': gravity_dilution_factor,
            
            # Key insight: All the π/4 and 2/√3 factors are fractions of 4π
            'pi_over_4_as_fraction_of_qg': (math.pi / 4) / q_g,  # = 1/16
            'two_over_root3_as_fraction_of_qg': (2/math.sqrt(3)) / q_g
        }

def main():
    """Run the complete CGM geometric coherence analysis."""
    
    analysis = CGMGeometricCoherence()
    analysis.print_coherence_analysis()
    
    return analysis

if __name__ == "__main__":
    analysis = main()