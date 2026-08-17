# Precession analysis — AI worknotes

Organizing principle: do not treat every produced angle as a candidate physical constant. The holonomy run exposes a three-axis precession triad of origin-pair stage rotations, plus composites, protocol residuals, and spherical-coordinate readouts of the same mass-shell connection.

Universality here means the same transport geometry on the velocity manifold, with different connections and projections. It does not mean every angle equals delta_BU. Canonical holonomy is intrinsic curvature. Lab holonomy is external-frame boost stitching. Complete chart (Cartesian Thomas Pexp) recovers the same conjugacy angle as geodesic transport. Spherical (theta, phi) Pexp is a singular-coordinate readout of that connection.

Precession is frame-transport holonomy in curved velocity space. Spin is a detector of the transported frame, not the cause of the rotation. Pair defects are finite Wigner rotations (no duration). Closed-loop delta_BU is the integral of the Thomas 1-form. A rate Omega = theta / T needs a duration assigned outside this program.

## Elementary triad

Stage vectors are orthogonal in the Einstein-ball realization: UNA along x, ONA along y, BU along z. Pairwise origin-gyrations complete an so(3) frame: UNA-ONA about z, ONA-BU about x, BU-UNA about y.

Elementary magnitudes are origin-pair defects, not equal-speed Wigner evaluations:

- omega_UO = defect(origin, UNA, ONA)
- omega_OB = defect(origin, ONA, BU+) = omega_corner = 2 atan(r(ONA) r(BU)) for the orthogonal pair
- omega_UB = defect(origin, BU+, UNA)

omega0 = TW(u_p, u_p; separation o_p) is an equal-speed response calibration. It is not the UNA-ONA stage-pair rotation. omega0 / omega_corner is not a universal constant; the two use different speed pairs.

u_p, o_p, m_a as numbers are logical priors. Placing them as Einstein beta is a realization bridge. Rapidities eta = atanh(beta) and Poincare r = k(beta) are alternative radial coordinates on the same manifold, not extra constants. CS is orientation, not a ball vertex.

## Composites

delta_BU = 2 omega_OB = defect(ONA, BU+, BU-). The middle BU+-BU- edge is collinear, hence I. For this planar gyrotriangle the conjugacy angle equals the hyperbolic area. Gravity uses that area as the Regge deficit unit.

delta_UNA_BU = 2 omega_UB is the UNA-based dual-pole triangle, a different polygon, not a transported copy of delta_BU.

delta_UOB = defect(UNA, ONA, BU+) is the noncommutative composite of the three elementary rotations (not their scalar sum). Every directed stage edge is UO, OB, UB, or I. Leftover bins are generator words: cyc4 = UNA-ONA-BU+-BU- and L4_crossed = UNA-BU+-ONA-BU-. Minimum closed canonical word with nonzero area has length 3 (out-backs cancel).

Do not identify bins with g_s, sin^2 theta_W, dark-sector scales, or pentagonal pi/5 numerology. phi_SU2 is numerically near delta_UOB + delta_UNA_BU at about 0.1 percent; that neighbor is not a derivation.

## Palindrome

R_pal = A^{-1} R_BU A with A = gyr(UNA, ONA). Angle is preserved (delta_BU). Axis is transported by omega_UO. ONA-BU sets balance magnitude; UNA-ONA steers the axis. Matrix conjugacy is the identity, not a scalar-axis comparison.

## Closure

rho0 is the zero-BU tangent d(delta_BU)/d(m_a)|_{m=0} = 2 k(o_p). Aperture splits as (1-rho0) = (rho-rho0) + (1-rho): linear opening, finite-BU closure, remaining secant aperture. rho_tangent is d(delta_BU)/d(m_a) at physical m_a. Delta_tangent = 1 - rho_tangent. nonlinear_closure_gain = rho_tangent - rho_secant.

## Connections

Canonical: geodesic, origin-gyr, Ungar defect, dual-pole word. Refactorization-invariant. Out-back = 0. This is the gravity connection.

Lab: relative-boost word in one inertial frame. Out-back nonzero because stitching does not invert. On L<=5 only collinear BU+-BU- out-backs close (theta=0). Every bent word is OPEN_BOOST_WORD_ROTATION. The group element is R_F = R_can^{-1} R_lab; scalar F = theta_lab - theta_can is a report of angles. On out-backs they agree. On the BU loop, theta(R_F) equals the sum of the two angles, not the scalar difference. F_BU equals the ONA-BU+ out-back regardless. F_pal / sqrt(3) and F_BU / pi are word-specific ratios, not derived constants. Lab bins are observer path classes, not curvature quanta. Circular Thomas alpha = 2 pi (gamma-1) is already gated in cgm_holonomy_analysis.

Chart complete: Palge-Pfeifer LC in Cartesian velocity coordinates, omega = (gamma^2/(gamma+1)) beta x d beta, Richardson Pexp. Regular at rest and at spherical poles. Equals delta_BU. Spherical G_* is a singular-coordinate readout, not a third holonomy and not a Dirac-string theorem.

phi_SU2 is a compact-fiber commutator. compact_hyperbolic_residual = phi_SU2 - 3 delta_BU.

Gravity uses delta_BU, rho, Delta, m_a, Q_G, kernel invariants. Canonical connection does not by itself make a quantity gravitational.

## Files

Core: experiments/cgm_precession_analysis_{1,2,common,run}.py. Theory: experiments/cgm_precession_analysis_theory_notes.txt. Circular Thomas multi-V calibration: experiments/cgm_holonomy_analysis_2.py. Gravity's tau_G, T_Z2, N_cycles, g1, and Mercury live in hqvm_gravity_analysis_*; this run does not recompute them.

## Scope

Holonomy: do algebraic routes agree on a defect.

Precession (this program): what is the defect as a physical process. A gyroscope following the stage boosts rotates. Pair defects are finite Wigner rotations (no duration). Closed-loop delta_BU is the path-ordered integral of the Thomas 1-form omega = (gamma^2/(gamma+1)) beta x d beta. Canonical transport is Fermi-Walker / geodesic. Lab transport is Thomas's original product of successive Lorentz boosts in one inertial frame.

Gravity: consumes m_a, delta_BU, rho, Delta and builds clocks, optical depth, and perihelion from them. Not this program.

## What the unexposed angles are

omega_UO is the Wigner rotation of composing UNA then ONA. A carried frame rotates by that angle. Named coupling unknown.

omega_OB is the same for ONA then BU+. Twice that angle is the closed dual-pole holonomy. Gravity takes that holonomy as its Regge unit; this program measures the pair rotation and the loop integral.

omega_UB is the Wigner rotation of composing BU+ then UNA. Named coupling unknown.

delta_UNA_BU, delta_UOB, cyc4, L4_crossed are Fermi-Walker holonomies of other stage polygons. Named couplings unknown.

phi_SU2 is a compact SU(2) commutator (Berry / solid angle on the compact fiber), not mass-shell Thomas holonomy. Named coupling unknown.

Lab F is the residual of boost-composition versus Fermi-Walker on the same vertices.

Spherical G is a singular-coordinate readout of the same Thomas connection.

used_downstream marks names gravity already consumes. It is not a license to reprint tau_G or Mercury here.

## Not this program

Kernel 8-byte T, tau_G, N_cycles, T_Z2, GW memory, g1, Mercury orbital period. Those are already in the gravity scripts.
