# Compact Geometry: Spectral Algebra of the Electroweak Mass Spectrum

## 1. Scope, Claims, and Dependency Map

### 1.1 Scope

The Common Governance Model (CGM) derives the structure of physical space and its fundamental constants from five logical constraints on coherent recursive measurement. Within this framework, the finite kernel is a discrete algebraic system with 4,096 reachable states, organised into seven shells by a binomial distribution and carrying a self-dual [12,6,2] binary code. The kernel supplies exact combinatorial and spectral data with no freely adjustable parameters.

This report uses the finite kernel as the sole algebraic input for a mass-coordinate law covering the four principal electroweak observables: the top quark, Higgs boson, Z boson, and W boson. The law expresses each mass as a spectral expansion in powers of the aperture parameter Delta, which measures the fractional non-closure of the CGM depth-four cycle relative to the observational aperture scale. Delta is fixed independently of the electroweak masses by the CGM geometric invariants. The expansion extends from Delta^1 through Delta^5, and every coefficient is a fixed rational number drawn from the kernel's discrete grammar of shell multiplicities, horizon cardinalities, code weights, and gyroscopic stage flags. At fifth order, the maximum tick error across the four channels is 6.15 × 10⁻⁹, and the W/Z ratio recovers the independently defined aperture Delta to 8.34 × 10⁻¹⁰.

The analysis is organised in three claim layers:

1. **Exact finite-kernel facts.** The 4,096-state reachable manifold, the dual 64-state horizons, the seven-shell binomial spectrum, the self-dual [12,6,2] code chart, and the reduced shell quantities C1 = 6, C2 = 15, C3 = 20, M = 192. These are combinatorial consequences of the kernel definition and require no physical input.

2. **Electroweak coordinate law.** The projection of the finite spectrum into four physical mass channels. Once Delta and the channel assignment (which kernel channel corresponds to top, Higgs, Z, W) are supplied, the law coefficients are fully determined by discrete kernel data: shell multiplicities, horizon size, K4 stage flags, trace-free gyroscopic charges, and code-curvature terms.

3. **External and imported hypothesis channels.** Neighbouring layers supply additional closure machinery: the horizon-wrap rule, the 0xAA source-traceability theorem, and the lifted K6 normalisation. These are explicitly marked where they enter and are treated as imported inputs rather than derived results.

Observed inputs enter as boundary data only:
* Top mass, Higgs boson mass, and Z/W masses from PDG.
* CODATA/NIST values for electromagnetic and related constants.

Input convention ledger:

| Symbol | Value | Convention |
| ------ | -----: | ---------- |
| `v` | 246.22 GeV | PDG electroweak scale convention used in this report |
| `m_t` | 172.760 GeV | PDG top mass convention (electroweak default) |
| `m_H` | 125.100 GeV | PDG central value |
| `m_Z` | 91.187600 GeV | PDG 2024 review |
| `m_W` | 80.379000 GeV | PDG 2024 review |
| `Lambda_QCD` | 0.2000 GeV | external reference only |

The CDF-II top-tension scenarios are not reweighted in the core electroweak audits; this report uses the PDG default electroweak input bundle.

### 1.2 Dependency Status of Major Claims

Derived in this report:
* finite kernel structure,
* shell-code quotient and coefficient grammar,
* K4 electroweak mass-coordinate law through Delta^5,
* numerical electroweak audits,
* 24-bit obstruction and 32-bit lift necessity.

Imported from neighbouring layers:
* horizon-wrap theorem,
* 0xAA source-traceability,
* lifted K6 normalisation.

External inputs:
* PDG electroweak masses,
* conventional QCD scale.

Deferred hypothesis channels:
* CKM ansatz,
* atomic spectroscopy alignment,
* antihydrogen tests,
* redshift phase channel.

### 1.3 Claim Hierarchy

Level 1: finite facts
* `|Omega| = 4096`, dual 64-state horizons, seven-shell binomial structure, self-dual [12,6,2] enumerator.

Level 2: electroweak law
* `L_i = a_i*Delta + b_i + c_i*Delta^2 + p_i*Delta^3/sqrt(5) + q_i*Delta^4 + r5_i*Delta^5`.

Level 3: hypothesis channels
* quark selectors,
* lepton carrier closures,
* CKM, spectroscopy, antihydrogen, and redshift channels.

## 2. Finite Kernel and Operator Algebra

This section establishes the exact combinatorial and spectral structure of the finite kernel. All results here are consequences of the kernel definition alone and are independent of the electroweak mass data.

### 2.1 Reachable Manifold

The CGM finite kernel is a deterministic algebraic system whose state is a 24-bit register formed from two conjugate 12-bit gyrophases, denoted A and B. The kernel evolves by applying bytes (8-bit instruction units) under a fixed transition law. The reachable manifold Omega is the set of states accessible from the rest state under this byte transition law.

Exhaustive enumeration gives:

| Quantity          |                         Value |
| ----------------- | ----------------------------: |
| Reachable states  |                          4096 |
| Form              |            2^12 = 16^3 = 64^2 = 4096 |
| Component density | popcount(A) = popcount(B) = 6 |
| Density product   |            d(A) x d(B) = 0.25 |

The constant component density is a structural constraint of the reachable manifold. It follows from the pair-diagonal mask structure and the self-dual [12,6,2] code chart.

Exhaustive byte transitions are checked exactly:

```text
4096 states x 256 bytes = 1,048,576 operations
active swap failures            = 0
passive commit failures         = 0
complement swap fraction        = 0.500000
complement commit fraction      = 0.500000
equality horizon hit fraction   = 0.015625
complement horizon hit fraction = 0.015625
```

### 2.2 Dual Horizons

Omega contains two disjoint extremal 64-state subsets.

| Horizon            | Condition     | Chirality |
| ------------------ | ------------- | --------- |
| Equality horizon   | A = B         | zero      |
| Complement horizon | A = B xor 0xFFF | maximal   |

The rest state lies on the complement horizon. The two horizons form a 128-state boundary of Omega. The remaining 3968 states form the relational bulk. Each horizon has cardinality 64, and the bulk manifold satisfies the holographic counting identity:

```text
|H|^2 = |Omega| = 64^2 = 4096.
```

### 2.3 Complementarity Invariant

For every state s in Omega, the following exact invariant holds:

```text
horizon_distance(s) + ab_distance(s) = 12.
```

This is an exact invariant of the compact state. No state can occupy both horizon conditions simultaneously.

### 2.4 Shell Distribution

Omega partitions into seven shells according to ab_distance.

| Shell | ab_distance | Population | Fraction | Characterisation   |
| ----: | ----------: | ---------: | -------: | ------------------ |
|     0 |           0 |         64 | 0.015625 | equality horizon   |
|     1 |           2 |        384 | 0.093750 | near unity         |
|     2 |           4 |        960 | 0.234375 | intermediate       |
|     3 |           6 |       1280 | 0.312500 | equatorial maximum |
|     4 |           8 |        960 | 0.234375 | intermediate       |
|     5 |          10 |        384 | 0.093750 | near opposition    |
|     6 |          12 |         64 | 0.015625 | complement horizon |

The population law is:

```text
|Shell_k| = C(6,k) x 64.
```

The unique maximal shell is Shell 3, where ab_distance = 6.

### 2.5 Projection and Fibre Structure

The 256-byte alphabet projects to Omega through a two-stage fibre structure:

```text
256 bytes -> 128 Omega-maps -> 64 q-classes.
```

The full byte-to-q projection has uniform 4-to-1 fibres. The q-class structure is the 6-bit payload structure of the byte algebra.

### 2.6 The 1/64 Structural Law

Three independently derived kernel fractions coincide:

| Property                               |     Fraction |
| -------------------------------------- | -----------: |
| Horizon-maintaining bytes              | 4/256 = 1/64 |
| Byte commutativity rate                | 4/256 = 1/64 |
| q-fibre size relative to byte alphabet | 4/256 = 1/64 |

The common fraction is 1/64 = 2^-6, reflecting a reciprocal 64 = 2^6 six-bit payload scale.

### 2.7 Boundary-to-Bulk Projection

From the complement horizon, the one-byte fanout projects uniformly to Omega:

```text
64 horizon states x 256 bytes = 16384 operations = 4 x 4096 target states.
```

Every state in Omega is reached with multiplicity 4. This gives an exact finite boundary-to-bulk dictionary.

### 2.8 Code Chart and Weight Enumerator

The kernel masks form a self-dual binary [12,6,2] code. Its weight enumerator has weights 0, 2, 4, 6, 8, 10, 12 with counts:

```text
1, 6, 15, 20, 15, 6, 1.
```

Multiplication by the horizon size 64 gives the shell populations of Omega. The code chart and the shell chart are the same binomial structure viewed through coding and state-space coordinates. The MacWilliams identity explains this coincidence through the code's self-dual structure.

### 2.9 Canonical Operator Layer

The spectral structure is defined by:

```text
A = functions on Omega
H = l^2(Omega)
T_x : psi(s) -> psi(T_x(s))     (byte transition permutation on Omega)
K(x,y) = T_x T_y T_x^-1 T_y^-1  (commutator defect)
D_shell = sum k=0..6 k * P_k     (shell number operator on Omega)
```

Here P_k projects onto shell k.

D_shell is diagonalized by the Krawtchouk-Walsh basis on GF(2)^6, in which shell projectors and spectral projectors are dual idempotent bases of the same finite Hamming scheme. Its spectral multiplicities are:

```text
Full Omega spectrum:
dim P_k = 64 * C(6,k),   k = 0..6
Tr(D_shell) = 64 * 192 = 12288

Reduced code-shell quotient:
dim Pi_k = C(6,k),   k = 0..6
Tr(D_code) = 192
```

The electroweak coefficient alphabet reuses the reduced code-shell spectrum:

```text
{C1, C2, C3} = {C(6,1), C(6,2), C(6,3)} = {6, 15, 20}
M1 = sum k=0..6 k * C(6,k) = 192 = Tr(D_code).
```

The electroweak coefficients are eigenvalue multiplicities of the shell number operator.

### 2.10 Canonical Operator Triple

The canonical non-commutative geometry object associated with the kernel is the finite triple:

```text
A = algebra generated by {T_b : b in 0..255}
H = l^2(Omega)
D = D_shell = sum k=0..6 k * P_k.
```

Here A is a finite matrix algebra of byte-transition permutations on Omega, H is the 4096-dimensional Hilbert space of square-summable functions on Omega, and D_shell is the shell number operator. The commutator layer is generated by:

```text
[D_shell, T_b]
K(x,y) = T_x T_y T_x^-1 T_y^-1.
```

Because Omega is finite, boundedness of commutators is automatic at the matrix level. 

Using the A/B swap as the candidate grading gamma and the 0xAA S-gate as the candidate real structure J, the relations gamma^2 = I, J^2 = I, J D_shell = D_shell J, and J gamma = gamma J are verified. In the 24-bit representation, gamma commutes with D_shell instead of anticommuting. This is a structural consequence of the projection from the 32-bit register algebra to the 24-bit spatial shadow, which collapses the four family phases into two spatial actions. The shadow representation carries only the commutative grading; the anticommutative grading is recovered in the lifted 32-bit sector.

The kernel operator group from Omega has the exact finite algebraic form:

```text
G = (GF(2)^6 x GF(2)^6) semidirect C2
C2 action: swaps the two GF(2)^6 coordinates
translation subgroup size          = 4096
parity sector size                = 2
operator group size               = 8192
derived subgroup G'               = 64
centre Z(G)                        = 64
G' = Z(G)                          = diagonal GF(2)^6
abelian shadow G/G'                 = 128
two-step uniformisation             = exact translation surjection, not stochastic
```

Define the chirality-flow candidate operator:

```text
D_flow = popcount(A) - popcount(B).
```

The executable confirms gamma anticommutation with the flow operator:

```text
D_flow gamma + gamma D_flow = 0.
```

The two operators are used for distinct roles in this report.

`D_shell` is the shell-number operator used by the electroweak mass-coordinate law.
`D_flow` is a separate chirality-flow candidate for finite Dirac-like grading tests in lifted settings.

The 24-bit report keeps `D_shell` as the primary shell-number operator for the opacity spectrum because its level structure is the finite kinematics for `L_i` and the shell transition algebra. `D_flow` is recorded as the natural finite Dirac-like candidate for future lifted spectral-triple upgrades.

The lifted K4-gauge sector recovers the required graded first-order behavior. The construction of a full lifted spectral triple beyond the K4-gauge sector is outside the scope of this report.

### 2.11 Boundary and Density Projectors

The gauge and density projectors are fixed kernel constants:

```text
P = 1 - 1/APERTURE_FRAME = 1 - 1/48 = 47/48,
Q = d_A x d_B = (6/12)^2 = 1/4.
```

The Delta^2 coefficients are shell-stage projections:

| Channel | Stage   | Delta^2 coefficient |
| ------- | ------- | --------------: |
| Top     | CS      |            +0.25 |
| Higgs   | CS-UNA      |            -24.00 |
| Z       | UNA     |            -22.50 |
| W       | UNA-ONA     |            -32.50 |

This decomposition can be written as:

```text
base gyration:  -M1/8 = -24
UNA rotational: +C1/4  = +1.5
ONA balance:    -C3/2  = -10
```

### 2.12 Declared Electroweak Grammar

The coefficient grammar is:

```text
C1 = 6, C2 = 15, C3 = 20, |H| = 64, M = 192, P = 47/48, Q = 1/4.
```

Allowed electroweak orders are Delta through Delta^5; Delta^6 is the complement-horizon residual layer.

## 3. Aperture Delta and the Compact Ruler

### 3.1 Continuous Aperture

The CGM aperture is defined from the BU monodromy defect and the observational aperture scale:

| Quantity               | Expression       |          Value |
| ---------------------- | ---------------- | -------------: |
| Observational aperture | m_a = 1/(2 sqrt(2π)) | 0.199471140201 |
| BU monodromy           | d_BU             | 0.195342176580 |
| Closure ratio          | rho = d_BU/m_a       | 0.979300446087 |
| Aperture gap           | Delta = 1 - rho      | 0.020699553913 |

Delta measures the fractional non-closure of the BU cycle relative to the observational aperture scale.

### 3.2 Discrete Aperture Approximants

| Aperture            |                  Value | Origin                        |
| ------------------- | ---------------------: | ----------------------------- |
| Bare byte aperture  | 5/256 = 0.019531250000 | 8-bit dyadic byte grid        |
| Continuous aperture |     Delta = 0.020699553913 | BU monodromy gap              |
| Depth-4 aperture    |  1/48 = 0.020833333333 | 4-byte x 12-bit closure frame |

The ordering is exact:

```text
5/256 < Delta < 1/48.
```

The depth-4 product is 48*Delta = 0.993578587835. The residual from exact depth-4 closure defines epsilon:

```text
epsilon = 1/Delta - 48 = 0.310219833351.
```

The second conversion depth is:

```text
eta = m_a - d_BU = 0.004128963621.
```

This conversion-depth quantity serves a bookkeeping role and lies outside the electroweak mass-coordinate hierarchy.

The aperture frame is:

```text
APERTURE_FRAME = 3 x |K4|^2 = 48.
```

The kernel boundary projector is:

```text
P = 1 - 1/APERTURE_FRAME = 47/48 = 0.979166666667.
```

### 3.3 Chirality-Space Ratio

The ratio between the depth-4 aperture and the turn quantisation is:

```text
(1/48)/(1/32) = 2/3.
```

The numerator corresponds to the two chiral gyrophase layers. The denominator corresponds to the three spatial axes. This ratio connects the 2-layer chirality register to the 3-axis spatial register.

### 3.4 Delta Self-Consistency

#### 3.4.1 Three-Factor Reconstruction

The continuous aperture Delta is reconstructed from the bare byte aperture by:

```text
Delta = (5/256) x 2^(1/12) x (1 + (sqrt(6)/π)Delta^2).
```

The factors are:

| Factor       | Meaning                                       |
| ------------ | --------------------------------------------- |
| 5/256        | bare dyadic byte aperture                     |
| 2^(1/12)     | UNA rotational dressing per degree of freedom |
| 1 + (sqrt(6)/π)Delta^2 | second-order 6-DoF chirality correction       |

#### 3.4.2 Numerical Closure

Evaluation at D^2 order gives 0.020699551515. The CGM value is Delta = 0.020699553913. The D^2 residual is approximately 2.4 x 10^-9. Evaluation at D^3 order gives 0.020699553957, with a residual of approximately 4.4 x 10^-11. The D^3 formula has a genuine fixed point at this value (self-consistency residual below 10^-15). The D^3 self-consistency closes two orders of magnitude more tightly than the D^2 evaluation.

#### 3.4.3 Fixed-Point Behaviour

Iteration from the bare seed 5/256 converges rapidly:

| Iteration |     Delta estimate | Error from Delta |
| --------: | -------------: | -----------: |
|         1 | 0.020698793188 |   7.6 x 10^-7 |
|         2 | 0.020699551006 |   2.9 x 10^-9 |
|         3 | 0.020699551513 |   2.4 x 10^-9 |

The D^2 fixed point is stable at the displayed precision. Extending the reconstruction to D^3 order yields a fixed point at 0.020699553957, with error from CGM Delta of approximately 4.4 x 10^-11.

### 3.5 Delta Ruler Coordinates

#### 3.5.1 Coordinate Definition

For an energy anchor `E0` and observable energy `E`, the Delta-ruler separation is:

```text
n = log2(E0/E)/Delta.
```

For electroweak mass coordinates, `E0` is the electroweak scale `v = 246.22 GeV`. Pairwise ratios are anchor-independent.

The optical depth interpretation is:

```text
tau = n * Delta * ln2.
```

#### 3.5.2 Electroweak Mass Coordinates

The tested electroweak separations are:

| Target        | EW separation | log2(EW/m) | Nearest integer | Signed tick residual (n - nearest integer) |
| ------------- | ------------: | ---------: | --------------: | ---------: |
| Top quark     |     24.695157 |   0.511179 |              25 |  -0.304843 |
| Higgs         |     47.192619 |   0.976866 |              47 |  +0.192619 |
| Z boson       |    69.230400 |   1.433038 |              69 |  +0.230400 |
| W boson       |    78.023779 |   1.615057 |              78 |  +0.023779 |
| Bottom quark  |    284.078644 |    5.880301 |             284 |  +0.078644 |
| Charm quark   |    367.108184 |    7.598976 |             367 |  +0.108184 |
| Tau           |    343.701809 |    7.114474 |             344 |  -0.298191 |
| Muon          |    540.413826 |   11.186325 |             540 |  +0.413826 |
| Strange quark |    547.824985 |   11.339733 |             548 |  -0.175015 |
| Electron      |    912.009808 |   18.878196 |             912 |  +0.009808 |

The electron sits within 0.010 ticks of 912. The W sits within 0.024 ticks of 78.

Compact observations from this ruler:

```text
alpha_geometric = d_BU^4/m_a = 7.299683x10^-3
```

`d_BU^4` means `d_BU` raised to the fourth power.

This quantity is not derived in this report. The base geometric derivation is in the CGM Constants analysis, and the CGM unit framework that fixes their physical scaling is in the CGM Units analysis.

Strong-sector scale check:

```text
n_QCD from EW/0.2 = 495.940
n_QCD from top anchor = 495.940
BU-strong offset = -0.000
```

The two independent estimates agree at reported precision.

## 4. Electroweak Mass-Coordinate Law

This section presents the central structural result: the electroweak mass coordinates are modelled by a spectral expansion in the aperture Delta. Each power of Delta extracts a distinct layer of the kernel's finite spectral structure.

### 4.1 Why the Expansion Is in Powers of Delta

#### The Mass Coordinate Is a Transmission Coefficient

The Delta ruler coordinate is:

```text
n = log2(E0/E) / Delta.
```

Rearranging:

```text
E/E0 = 2^(-nDelta) = e^(-tau),    tau = nDelta*ln2.
```

The optical depth is tau = nDelta*ln2, and Delta sets the per-tick attenuation.

This is the Beer-Lambert law. The medium through which transmission occurs is the kernel manifold Omega, which decomposes as:

```text
Omega = H x GF(2)^6.
|H| = 64, |GF(2)^6| = 64, |Omega| = 4096.
```

The six-bit quotient chi in GF(2)^6 provides six binary opacity modes, with shell index

```text
k = |chi| (Hamming weight),    k = 0,1,2,3,4,5,6,
```

and shell class

```text
Shell_k = H x {chi : |chi|=k},   |Shell_k| = 64*C(6,k).
```

This is the compact opacity medium used by the electroweak coordinates.

#### The Opacity Polynomial Is a Trace-Generating Function

The kernel is a finite [12,6,2] code on the shell quotient, and its weight enumerator is the trace-generating function of the opacity modes:

```text
(1 + z^2)^6 = sum k=0..6 C(6,k) z^(2*k).
```

Identifying each two-bit code-weight increment with one aperture-grade contribution in the six-mode quotient, z^2 is evaluated as Delta. This gives the opacity-mode generating function:

```text
sum k=0..6 C(6,k) Delta^k = (1 + Delta)^6.
```

This trace-generating function is not itself the physical transmission law. The physical channel relation remains, in the same electroweak convention used below,

```text
m_i/m_ref = 2^(-L_i).
```

#### The Graded Aperture Operator

Introduce the compact aperture operator as

```text
K(Delta) = Delta K1 + Delta^2 K2 + Delta^3 K3 + Delta^4 K4 + Delta^5 K5 + Delta^6 K6.
```

where each grade K_r lies in the algebra generated by shell projectors, K4 stage projectors, and horizon projectors. Each channel is a projection of this operator:

```text
L_i = b_i + <channel_i | K(Delta)>,
```

So the coefficient of Delta^r is the channel projection of the r-th aperture grade, and the power order is algebraically fixed by the finite six-mode structure.

The electroweak coefficient alphabet is therefore derived from:
* shell multiplicities C(6,k),
* shell moment M1 = 192,
* horizon multiplicity |H| = 64,
* and the trace-free K4 charges from the stage projectors.

#### The Shell Distribution Is the Eigen-Opacity Spectrum

The shell distribution (Section 2.4) is the eigen-opacity spectrum of this medium. D_shell has eigenvalues k = 0,1,2,3,4,5,6 with multiplicities C(6,k) x 64. The mean opacity is M1/|H| = 192/64 = 3, the equatorial shell. The distribution is symmetric about shell 3, with the complementarity invariant bounding the spectrum at shells 0 and 6.

#### The Depth Structure Determines the Order

The temporal algebra closes at K4 depth with stages CS, UNA, ONA, BU and 4-byte frame structure. The opacity algebra has degree up to six because it is built from six binary chirality modes. The following correspondence between temporal depth and power order organises the coefficient algebra:

* **Delta^1:** one active chirality mode.
* **Delta^2:** two-mode grade, shaped by density projection Q = 1/4 and boundary projector P = 47/48.
* **Delta^3:** three-mode grade; K4 closure controls the p channel charge.
* **Delta^4:** four-mode grade; K4 closure gives the trace-free q channel charge.
* **Delta^5:** five-mode finite-code complement grade, giving code curvature coefficients.
* **Delta^6:** six-mode saturated complement mode at the complement horizon side of the spectrum.

The trace-free conditions Sp_i = 0 and Sq_i = 0 are conservation laws: the depth-3 and depth-4 corrections redistribute optical depth among channels without altering the total electroweak optical budget. What one channel gains, another loses. This is the gauge constraint that the four channels span the complete K4 flag space, so no net opacity is created or destroyed.

#### The Horizons Are the Opacity Boundaries

The dual horizons (Section 2.2) are the transparent and opaque boundaries of this medium: the equality horizon at shell 0 and the complement horizon at shell 6. All physical masses lie in the relational bulk between them.

#### Summary

The powers of Delta are the opacity grades of a six-mode finite quotient, each power corresponding to one additional active chirality mode. The mass coordinate is a transmission coordinate; the compact kernel supplies the opacity spectrum; the K4 stage algebra supplies the channel projections; and the trace-free p and q charges enforce conservation of the channel-summed optical depth.

The electroweak mass spectrum is the transmission spectrum of a finite aperture whose eigen-opacity distribution is the shell structure of Omega. Every coefficient in the expansion is determined by the kernel algebra, distinguishing this construction from a generic polynomial fit.

### 4.2 Gyrotriangle Closure

The K4 gyroscopic multiplication satisfies the gyrotriangle angle sum:

```text
delta = π/2 - (π/4 + π/4 + π/4) = 0.
```

This closure forces the stage ratio constraint:

```text
ONA/CS = UNA^2 = 1/2.
```

The gyrotriangle closure is the structural source of the Delta^3 and Delta^4 corrections.

### 4.3 K4 Edge Increments and Trace-Free Charges

The K4 stage structure has three binary flags per channel: base (gyration), rot (UNA rotational), and bal (ONA balance). Each flag contributes a code-geometric edge increment:

```text
p edges: -C1/2 = -3,  +C1/4 = +3/2,  +4g = +2
q edges: 0,  -4g = -2,  -2g = -1
```

where g = 1/2 is the gyro factor from ONA/CS = UNA^2 = 1/2.

The stage flags for the four electroweak channels are:

| Channel | base | rot | bal |
| ------- | ---- | --- | --- |
| Top     | 0    | 0   | 0   |
| Higgs   | 1    | 0   | 0   |
| Z       | 1    | 1   | 0   |
| W       | 1    | 1   | 1   |

Using the trace-free constraints, the base offsets are fixed:

```text
4*p0 + sum(p increments over channels) = 0
4*p0 + (0) + (-3) + (-3/2) + (1/2) = 0
4*p0 - 4 = 0
p0 = 1

4*q0 + sum(q increments over channels) = 0
4*q0 + (0) + (0) + (-2) + (-3) = 0
4*q0 - 5 = 0
q0 = 5/4 = 1.25
```

This explains the constants in the table below and makes the trace-free conditions explicit.

Computing p_i and q_i from the edge increments:

| Channel | p_i  | q_i   |
| ------- | ---- | ----- |
| Top     | 1.0  | 1.25  |
| Higgs   | -2.0 | 1.25  |
| Z       | -0.5 | -0.75 |
| W       | 1.5  | -1.75 |

Trace-free verification:

```text
S p_i = 1.0 + (-2.0) + (-0.5) + 1.5 = 0.0
S q_i = 1.25 + 1.25 + (-0.75) + (-1.75) = 0.0
```

The trace-free property is a structural consequence of the gyrotriangle closure: the four channels span the complete K4 flag space, so the edge increments must sum to zero.

### 4.4 Five-Order Expansion

The complete electroweak mass law, expanding the transmission coordinate through the five opacity layers identified in Section 4.1, is:

```text
L_i = a_i*Delta + b_i + c_i*Delta^2 + p_i*Delta^3/sqrt(5) + q_i*Delta^4 + r5_i*Delta^5
```

Each order has a distinct structural origin:

| Order | Source | Algebraic origin |
| ----- | ------ | ---------------- |
| Delta^1    | Code enumerator + aperture frame | (1+z^2)^6 -> 6, 15, 20, 64, 192 |
| Delta^2    | Stage projections of code weights | -M/8, +C1/4, -C3/2, +Q |
| Delta^3    | Gyroscopic phase (p-charge) | K4 edge increments, trace-free |
| Delta^4    | Gyroscopic closure (q-charge) | K4 edge increments, trace-free |
| Delta^5    | Code curvature (r5-charge) | C1, C2, |H|, stage flags |

The Delta^3 amplitude is:

```text
lambda0 = Delta/sqrt(5) = 0.009257121931.
```

The factor 1/sqrt(5) sets the K4 third-order normalisation scale used in the channel term.

The channel constant offsets are:

```text
b_top   = -1
b_higgs = -1
b_Z     = -P = -47/48
b_W     = -P = -47/48
```

Thus `b_i` is selected from one matter branch (`-1`) and two gauge branches (`-P = -47/48`).

In this report, `1/sqrt(5)` is treated as a normalisation constant imported from the K4 gyro-normalisation convention.

The Delta^5 coefficients are exact rationals from the code algebra:

```text
r5_i = -(C2-C1)/2 + (|H|-(C2-C1))/8 * (base-rot) + C2/8 * bal
```

With |H| = 64, C1 = 6, C2 = 15:

| Channel | r5_i   |
| ------- | ------ |
| Top     | -4.500 |
| Higgs   | 2.375  |
| Z       | -4.500 |
| W       | -2.625 |

These are code-valued. No fitting is involved.

### 4.5 Orderwise Expansion Residuals

The five-order expansion residuals are:

| Channel | p    | q    | r5     | n_err Delta^2       | n_err Delta^3       | n_err Delta^4       | n_err Delta^5       | L_err/Delta^6 |
| ------- | ---- | ---- | ------ | -------------- | -------------- | -------------- | -------------- | -------- |
| Top     | 1.0  | 1.25 | -4.500 | 2.019x10^-4 | 1.026x10^-5 | -8.234x10^-7 | 2.758x10^-9 | 0.726    |
| Higgs   | -2.0 | 1.25 | 2.375  | -3.717x10^-4 | 1.152x10^-5 | 4.349x10^-7  | -1.148x10^-9 | -0.302   |
| Z       | -0.5 | -0.75| -4.500 | -1.033x10^-4 | -7.474x10^-6 | -8.217x10^-7 | 4.408x10^-9  | 1.160    |
| W       | 1.5  | -1.75| -2.625 | 2.714x10^-4  | -1.600x10^-5 | -4.758x10^-7 | 6.150x10^-9  | 1.618    |

The Delta^5 term is the first order where exact code-valued r5 coefficients appear. Three channels match to about 0.3 to 0.5%; the W channel matches to about 1.3%.

Empirical c5 reconstruction compared to the predicted r5 coefficients:

| Channel | c5_emp | r5 | c5_emp - r5 | Relative mismatch |
| ------- | ------ | --- | ----------- | ---------------- |
| Top     | -4.484976 | -4.500 | +0.015024 | +0.334% |
| Higgs   | 2.368745  | 2.375  | -0.006255 | -0.263% |
| Z       | -4.475990 | -4.500 | +0.024010 | +0.534% |
| W       | -2.591502 | -2.625 | +0.033498 | +1.276% |

### 4.6 Coefficient admissibility ledger

The five-order electroweak law is:

```text
L_i = a_i*Delta + b_i + c_i*Delta^2 + p_i*Delta^3/sqrt(5) + q_i*Delta^4 + r5_i*Delta^5
```

To make the hierarchy explicit, each family is placed in this category set:

| Coefficient family | Status in this report | Why |
| ------------------ | --------------------- | --- |
| a_i                | forced                | fixed from `|H|, C1, C2, C3, M` |
| b_i                | forced                | fixed from gauge or matter-offset rules |
| c_i                | selected from finite set | fixed values from stage-projector spectrum `{+Q,-M/8,-M/8+C1/4,-M/8+C1/4-C3/2}` |
| p_i                | forced                | fixed by K4 edge increments and trace-free constraint sum(p_i)=0 |
| q_i                | forced                | fixed by K4 edge increments and trace-free constraint sum(q_i)=0 |
| r5_i               | selected from finite set | fixed by the code formula `r5 = -(C2-C1)/2 + (|H|-(C2-C1))/8*(base-rot) + C2/8*bal` over base/rot/bal flags |
| sqrt(5) factor      | imported fixed normalisation | fixed by K4/32-bit normalisation convention |

No coefficient in this law is continuously fitted to mass data or imported from external statistics. Selections for some coefficient families are finite and are fixed by the predeclared grammar, not by numerical regression.

### 4.7 The Delta^6 Boundary

After the Delta^5 law, the residuals in units of Delta^6 are:

| Channel | L_err/Delta^6 |
| ------- | ------------: |
| Top     |      0.725808 |
| Higgs   |     -0.302196 |
| Z       |      1.159946 |
| W       |      1.618304 |

These are order-unity residuals at the sixth opacity grade, where all six chirality modes of the kernel are active. No Delta^6 coefficient is fitted in the five-order law. The residuals therefore identify the complement-horizon interface left after the code-valued Delta^5 curvature term.

The W channel carries the largest positive sixth-grade residual. W is the unique electroweak channel with all three K4 flags active (base, rot, bal) = (1,1,1) and is the charged-current endpoint of the gauge-stage ladder.

The sixth-grade residuals mark the first grade excluded from the five-order law. They are retained as boundary markers; no Delta^6 coefficient is introduced to absorb them. In the compact kernel this grade corresponds to the saturated six-mode complement side of the spectrum, with minimal support on P_6.

Raw 24-bit tests do not close the W residual. Specifically, raw 24-bit horizon permutations, unweighted K4 horizon permutations, and simple K4 character lifts do not close it. These outcomes are retained as negative controls.

The imported lifted K6 construction closes the sixth-grade boundary only after restoring the family-phase extension. The Delta^6 sector is therefore classified as a representation-boundary phenomenon, reflecting the kernel's structural limit at the complement horizon.

### 4.8 Structural Independence Audit

The numerical closure is meaningful only if the tested algebraic object is fixed before comparison with the electroweak masses. The following audit states exactly what is fixed internally, what is supplied as an interpretive channel assignment, and what remains open.

| Item | Required clarification | Current status in this report |
| ---- | --------------------- | ---------------------------- |
| Delta | Was it fixed before top/H/Z/W evaluation? | `Delta = 1 - d_BU/m_a` is fixed from the BU monodromy and aperture constants in the finite-kernel layer and is independent of the electroweak mass set used in Section 5. |
| Channel assignment | Are Top/H/Z/W flags independently forced or post hoc selected? | Within the fixed flag grammar and after imposing trace-free `sum p_i = 0, sum q_i = 0`, the declared Top/Higgs/Z/W assignment is the unique rank-1 candidate among 96 admissible candidates under the max-absolute-tick-error metric. |
| Coefficient grammar | Which terms are unique versus finite selections? | `a_i, b_i, p_i, q_i` are forced in this setup. `c_i` and `r5_i` are selected from a finite grammar domain once the channel flags are fixed. `sqrt(5)` is an imported fixed normalisation factor from the K4/32-bit convention. |
| Null model | How many nearby grammars could produce comparable residuals? | The discrete family has `2^12 = 4096` flag assignments; trace conditions `sum p_i = 0, sum q_i = 0` reduce this to 96 admissible candidates. Section 5.0 ranks these by max abs tick error before any external-channel interpretation. |

This section therefore tests a fixed, predeclared grammar object and then assesses empirical compression strength under that object.

 

## 5. Electroweak Numerical Tests

The five-order law derived in Section 4 is the basis for all tests in this section.

### 5.0 Null-Model Audit for the Electroweak Law

The null-model baseline for the electroweak law keeps the grammar fixed and only varies the discrete kernel flags. Each of the four channels uses:

```text
L_i = a_i*Delta + b_i + c_i*Delta^2 + p_i*Delta^3/sqrt(5) + q_i*Delta^4 + r5_i*Delta^5
```

with:

* same shell-based a_i and b_i formulas,
* same stage projector family for c_i,
* p_i and q_i generated from base/rot/bal flags through K4 edge increments,
* r5_i generated from the same base/rot/bal flags through the r5 formula.

This gives 2^12 = 4096 raw flag assignments. Enforcing trace conditions
sum(p_i)=0 and sum(q_i)=0 yields 96 grammar-consistent candidates.

For each tracefree candidate, the script evaluates the full law at `Delta = DELTA` and ranks it by maximum absolute tick error to the observed electroweak coordinates.

The declared channel assignment `(Top, Higgs, Z, W)` enters as rank:

```text
declared assignment rank = 1
rank 1 max_abs_tick_error = 6.150e-09
rank 2 max_abs_tick_error = 6.955e-05
unique rank factor gain ~ 1.1e4
```

The null-model audit uses the following declared filter table:

| Filter | Surviving candidates |
| ------ | -------------------: |
| Raw flag assignments | 4096 |
| Trace-free p and q constraints | 96 |
| max abs tick error <= 5e-3 | 96 |
| max abs tick error <= 1e-3 | 93 |
| max abs tick error <= 5e-4 | 31 |
| max abs tick error <= 1e-4 | 2 |
| max abs tick error <= Delta^5 closure band | 0 |

The top ranked trace-free candidates are:

| rank | max_abs_tick_error | Top flags | Higgs flags | Z flags | W flags | p_sum | q_sum | sum_abs_err |
| ---: | -----------------: | :-------- | :---------- | :------ | :------ | ----: | ----: | ----------: |
| 1 | 6.150e-09 | (0,0,0) | (1,0,0) | (1,1,0) | (1,1,1) | 0.000 | 0.000 | 1.446e-08 |
| 2 | 6.955e-05 | (1,1,1) | (1,0,0) | (1,1,0) | (0,0,0) | 0.000 | 0.000 | 1.391e-04 |
| 3 | 1.989e-04 | (1,0,1) | (1,0,0) | (1,1,0) | (0,1,0) | 0.000 | 0.000 | 3.978e-04 |
| 4 | 2.684e-04 | (1,0,1) | (1,1,0) | (1,1,0) | (0,0,0) | 0.000 | 0.000 | 5.369e-04 |
| 5 | 2.684e-04 | (0,0,0) | (1,1,0) | (1,0,0) | (1,1,1) | 0.000 | 0.000 | 5.369e-04 |
| 6 | 2.684e-04 | (0,1,0) | (1,0,0) | (1,0,0) | (1,1,1) | 0.000 | 0.000 | 5.369e-04 |
| 7 | 2.684e-04 | (1,0,1) | (1,1,0) | (1,0,0) | (0,1,0) | 0.000 | 0.000 | 9.346e-04 |
| 8 | 2.684e-04 | (1,1,1) | (1,0,0) | (1,0,0) | (0,1,0) | 0.000 | 0.000 | 5.369e-04 |
| 9 | 2.684e-04 | (1,1,1) | (1,1,0) | (1,0,0) | (0,0,0) | 0.000 | 0.000 | 6.760e-04 |
| 10 | 2.684e-04 | (0,0,0) | (1,1,0) | (1,1,0) | (1,0,1) | 0.000 | 0.000 | 5.369e-04 |
| 11 | 2.684e-04 | (0,1,0) | (1,0,0) | (1,1,0) | (1,0,1) | 0.000 | 0.000 | 5.369e-04 |
| 12 | 2.684e-04 | (0,1,0) | (1,1,0) | (1,0,0) | (1,0,1) | 0.000 | 0.000 | 1.074e-03 |

This is the finite-law-only baseline that keeps the law-core separate from external leads. The best rank-12 list above is the full declared-filter output block from the executable output.

### 5.1 Symbolic Coordinate Forms

The Delta^2-level coordinate forms are:

```text
n_t = 25 - epsilon + Delta/4       = 24.694955
n_H = 48 - d_H               = 47.192991
n_Z = 70 + Delta - P*d_H         = 69.230503
n_W = 79 - 9Delta - P*d_H        = 78.023508
```

where epsilon = 1/Delta - 48, d_H = epsilon + 24Delta, and P = 47/48.

The charged-neutral split:

```text
S_WZ = (C2 - C1) - (C3/2)Delta + 2Delta^2/sqrt(5) - Delta^3 = 8.793379
```

### 5.2 Mass Prediction Table

At Delta^2 level (code enumerator + stage projections only):

| State   | Predicted n | Observed n | Tick error | Mass error   |
| ------- | ----------: | ---------: | ---------: | -----------: |
| Top     |   24.694955 |  24.695157 |  +0.000202 |  2.897x10^-6 |
| Higgs   |   47.192991 |  47.192619 |  -0.000372 | -5.333x10^-6 |
| Z       |   69.230503 |  69.230400 |  -0.000103 | -1.482x10^-6 |
| W       |   78.023508 |  78.023779 |  +0.000271 |  3.894x10^-6 |

At Delta^5 level, the maximum tick error is 6.15x10^-9.

### 5.3 Delta Backsolves

Inverting the Delta^2 mass laws to recover Delta from observed masses:

| Source | Equation                                        |         Delta_back |       Delta error |
| ------ | ----------------------------------------------- | -------------: | ------------: |
| Top    | L_t = 73Delta - 1 + Delta^2/4                           | 0.020699611150 |  5.724x10^-8 |
| Higgs  | L_H = 96Delta - 1 - 24Delta^2                           | 0.020699472926 | -8.099x10^-8 |
| Z      | L_Z = 117Delta - 47/48 - 45Delta^2/2                    | 0.020699535494 | -1.842x10^-8 |
| W      | L_W = 126Delta - 47/48 - 65Delta^2/2                    | 0.020699598986 |  4.507x10^-8 |
| W/Z    | log2(m_Z/m_W) = Delta(9 - 10Delta + 2Delta^2/sqrt(5) - Delta^3)     | 0.020699554747 |  8.340x10^-10 |

The W/Z backsolve uses the K4-corrected split law (Section 5.7.2) and achieves an error of 8.3x10^-10 against the independently defined CGM Delta, one of the tightest ratio-channel consistency constraints in this report.

Directly, Delta_WZ = 0.020699554747 versus Delta_CGM = 0.020699553913 gives an absolute difference of 8.34x10^-10.

### 5.4 Four-Point Delta Consensus

| Channel |         Delta_back |       Delta error |
| ------- | -------------: | ------------: |
| Top     | 0.020699611150 |  5.724x10^-8 |
| Higgs   | 0.020699472926 | -8.099x10^-8 |
| Z       | 0.020699535494 | -1.842x10^-8 |
| W       | 0.020699598986 |  4.507x10^-8 |
| **Mean**| **0.020699554639** | **7.255x10^-10** |

Reference Delta = 0.020699553913. Four-point spread = 1.38x10^-7.

The four-point mean recovers Delta to 7.3x10^-10. This consensus is a consistency check on the closed electroweak+top coordinate stack. It is not fully independent, because the four-point set contains both W and Z. A cleaner comparison is the direct W/Z-to-Delta_CGM test above.

### 5.5 H/Z/W Leave-One-Out Test

Each H/Z/W mass is predicted from the other two:

| Target | Delta source |  m_pred (GeV) |  m_ref (GeV) |    rel_err   |
| ------ | ------------ | -------------: | ------------: | -----------: |
| Higgs  | Z + W        | 125.099223017  | 125.100000000 | -6.211x10^-6 |
| Z      | Higgs + W    |  91.187596612  |  91.187600000 | -3.715x10^-8 |
| W      | Higgs + Z    |  80.379658228  |  80.379000000 |  8.189x10^-6 |

The Z prediction is the tightest leave-one-out consistency result here, at 3.7x10^-8 relative error.

### 5.6 Matter-Gauge Dichotomy

The coefficient table and grammar for this section are fixed in Section 4.

The top quark's Delta backsolve error (5.7x10^-8) is within 0.7x of the H/Z/W maximum error (8.1x10^-8). This quantifies the matter-gauge dichotomy:

* **Top**: CS matter channel, carries +Q*Delta^2 (density projector).
* **H/Z/W**: Gauge channel, carries negative porosity Delta^2 from stage projections.

The top error enters the H/Z/W cluster after the Delta^2/4 correction (Section 7.8). Before the correction, the top error is 18.8x the H/Z/W maximum.

### 5.7 W/Z Split and Weak Mixing Coordinate

#### 5.7.1 Delta^2 Backbone Split

The Delta^2-level charged-neutral split is:

```text
log2(m_Z/m_W) = Delta[(C2 - C1) - (C3/2)Delta].
```

This gives:

| Quantity       |        Predicted |         Observed |      Difference |
| -------------- | ---------------: | ---------------: | --------------: |
| n_W - n_Z      |   8.793004460868 |  8.793379174256  | 3.747x10^-4  |
| sin^2 theta_W       |   0.223004870612 |  0.223013225327  | -8.355x10^-6 |

#### 5.7.2 Promoted Split Law

The gyroscopic expansion promotes the split to:

```text
log2(m_Z/m_W) = Delta[(C2 - C1) - (C3/2)Delta + 2Delta^2/sqrt(5) - Delta^3].
```

The additional terms come from the trace-free p and q charges:

```text
p_W - p_Z = 2    ->    +2Delta^2/sqrt(5)
q_W - q_Z = -1   ->    -Delta^3
```

#### 5.7.3 sin^2 theta_W Prediction

| Quantity                |          Predicted |           Observed |          Error    |
| ----------------------- | -----------------: | -----------------: | ----------------: |
| n_W - n_Z               | 8.793378828287     | 8.793379174256     | 3.460x10^-7     |
| cos theta_W                 | 0.881468537378     | 0.881468533002     | 4.376x10^-9     |
| sin^2 theta_W               | 0.223013217613     | 0.223013225327     | -7.714x10^-9    |
| theta_W                     | 28.179977230 deg      | N/A                | N/A               |
| W from Z and Delta          | 80.379000399 GeV   | 80.379000000 GeV   | 4.964x10^-9 rel |

The W/Z channel is the cleanest internal ratio test because the electroweak scale cancels. The K4-corrected split law recovers Delta from the observed W/Z ratio with absolute error 8.34 x 10^-10 against the independently defined CGM aperture. This ratio-channel closure is a direct test of the compact mass-coordinate law, independent of tree-level weak-mixing comparisons.

Masses are taken from the PDG input set listed in Section 1, with couplings evaluated at tree level. Electroweak comparisons are convention-dependent and scheme-dependent.

The K4-corrected law predicts the W mass from the Z mass and Delta with relative error 5.0x10^-9.

#### 5.7.4 W/Z Ratio-Channel Delta Lock

The K4-corrected W/Z backsolve is the key ratio-channel test:

| Quantity                            |           Value |
| ------------------------------------| --------------: |
| Four-point Delta mean                   | 0.020699554639  |
| Corrected W/Z Delta                     | 0.020699554747  |
| W/Z - four-point mean               | 1.085x10^-10   |
| Base W/Z-to-consensus gap           | 9.027x10^-7     |
| Corrected W/Z-to-consensus gap      | 1.085x10^-10   |
| **Consensus improvement factor**    | **8,323x**      |

This is a stringent ratio-channel consistency result: the absolute Delta error reaches 8.3x10^-10.

Practical cross-checks are:
* compare W/Z against the independently defined CGM Delta;
* compare W/Z against a top+H-only estimate of Delta;
* compare W/Z against later W or Z reference inputs under the same convention.

### 5.8 Compact Spectral Coupling Parametrisation

#### 5.8.1 Scope

The compact electroweak algebra extends from mass coordinates to tree-level coupling parametrisations that are explicitly mass-input dependent. Since lambda_H, g, g_Z, and y_t are algebraically related to m_H, m_W, m_Z, m_t, and v, the coupling values follow from the mass inputs by standard tree-level electroweak algebra. They serve as algebraic consistency checks on the mass-coordinate law under the stated convention.

Electroweak parameter values are compared only after stating the reference set and convention. Here, current PDG review masses are used with tree-level electroweak reconstruction; this is therefore scheme and convention dependent.

#### 5.8.2 Coupling Exponents

```text
y_t  = 2^(3/2 - 73Delta - Delta^2/4)
lambda_H  = 2^(1 - 192Delta + 48Delta^2)
g_Z  = 2^(95/48 - 117Delta + 45Delta^2/2)
g    = 2^(95/48 - 126Delta + 65Delta^2/2).
```

The y_t expression includes the matter-projection term Delta^2/4 from Q = 1/4.

| Quantity | Constant exponent |   Delta exponent | Delta^2 exponent | Total exponent |
| -------- | ----------------: | -----------: | ----------: | -------------: |
| y_t      |       1.500000000 | -1.511067436 | -0.000107119|  -0.011174554  |
| lambda_H |       1.000000000 | -3.974314351 | 0.020566634 |  -2.953747718  |
| g_Z      |       1.979166667 | -2.421847808 | 0.009640609 |  -0.433040532  |
| g        |       1.979166667 | -2.608143793 | 0.013925325 |  -0.615051802  |

The top Yukawa includes the Delta^2/4 matter density term. The Higgs quartic and gauge exponents include Delta^2 porosity corrections from the gauge channel.

#### 5.8.3 Coupling Geometry Values

| Quantity | Law          | Compact value | Reference value | Relative error |
| -------- | ------------ | ------------: | --------------: | -------------: |
| lambda_H | m_H^2/(2v^2)   |   0.129072386 |     0.129073762 | -1.067x10^-5 |
| g        | 2m_W/v       |   0.652906450 |     0.652903907 |  3.894x10^-6 |
| g_Z      | 2m_Z/v       |   0.740699089 |     0.740700187 | -1.482x10^-6 |
| g'       | sqrt(g_Z^2-g^2) |   0.349783231 |     0.349790301 | -2.021x10^-5 |
| e        | gg'/g_Z      |   0.308324569 |     0.308329144 | -1.484x10^-5 |
| alpha_EWDelta   | 4π/e^2        | 132.188476676 |   132.184554083 |  2.968x10^-5 |
| y_t      | sqrt(2)m_t/v      |   0.992284310 |     0.992281435 |  2.897x10^-6 |

Values are at tree-level convention. Comparison to renormalised Standard Model couplings requires declaration of scale and scheme.

## 6. Representation Boundary: The 24-bit Shadow and the 32-bit Lift

### 6.1 The 32-bit Necessity

The 24-bit carrier space Omega fails the first-order spectral triple condition, the SU(3) sextet bracket, and the sixth-grade W-boundary closure. These are necessary topological consequences of the SO(3)/SU(2) double cover. The 24-bit space identifies the S-gate pairs {0xAA, 0x54} and {0xD5, 0x2B}, collapsing the 4-family spinorial phase into 2 spatial actions. This collapse is the discrete analogue of projecting a spinor down to a vector.

The family-lifted K4 probe confirms that naive family-phase addition is not sufficient:

```text
naive K4 family lift first-order violations = 101,504
naive lift repairs spectral-triple obstruction = no
```

Depth-4 family fiber probing shows:

```text
fixed micro refs            = (0, 1, 2, 3)
family assignments retained = 256
distinct 24-bit outputs    = 4
shadow collapse             = 256 -> 4
```

These tests quantify the 24-bit information collapse and motivate the rich-K6 lift.

In the 32-bit lifted space:
1. The spectral triple first-order condition closes on the K4 gauge subalgebra.
2. The SU(3) sextet bracket closes under family-phase symmetrization.
3. The W sixth-grade residual closes as a path-multiplicity resonance.

The 32-bit lift is the minimal representation required for structural consistency. The 24-bit shadow obstructions identify a representation boundary of the compact operator algebra and motivate the lifted 32-bit closure layer.

### 6.2 Finite Colour Operator Algebra

The 15 weight-4 codewords decompose as 1 + 6 + 8. The adjoint bracket and phase-symmetrized sextet bracket close, and paired adjoint action preserves relational bulk support.

| Check                 | Result |
| --------------------- | ------ |
| 1+8+6 decomposition  | closes |
| Adjoint bracket       | closes |
| Sextet bracket        | closes (phase-symmetrized lifted check) |
| Paired action preserves bulk | yes |
| Left-action leaks     | yes    |

Left-action leakage under single-sided adjoint action is retained as a control confirming that confinement requires the paired conjugate structure.

The strong reference scale Lambda_QCD = 0.2000 GeV maps to n_QCD = 495.939781 on the same Delta ruler. The numerical QCD scale enters as an external reference value for ruler placement; its derivation lies outside the present scope. The finite-geometric observation is that the strong scale lies deep in the relational bulk of Omega, distant from both horizons.

The 10*48 + 16 aperture-cycle probe gives a nearby but non-closing residual. The best tested compact expression misses by 1.31e-4, so the QCD scale remains externally supplied.

### 6.3 Executable Closure Probes

The full executable probe set is moved to Appendix A to keep the main narrative compact.

### 6.4 Interpretive Physical Leads

The following leads are retained as testable physical questions. They are not used as premises for the numerical closures above.

**Code-Valued Curvature.** The commutator defect always lands in C_64, the self-dual [12,6,2] code. In geometric language, loop residue is code-valued rather than continuous Lie-algebra-valued. The testable development is to formulate q as a discrete connection, classify minimal loops, and determine whether Bianchi-type identities hold in the code chart.

**Holographic Thermal Floor.** The compact identity |H|^2 = |Omega| and the shell structure imply a minimum effective support of |H| = 64. This motivates a holographic thermal-floor hypothesis: a compact holographic system cannot reduce its effective support below its boundary cardinality. The physical extension requires a thermodynamic mapping from finite support to temperature.

**Antihydrogen as Mirror-Aperture Test.** Improved antihydrogen spectroscopy can test whether sigma-level compact mirror residuals are present or bounded below the SU(2) holonomy scale.

**Atomic Spectral Null Model.** The useful test is not the existence of near-integer hits alone, but whether compact code-weight levels are overrepresented relative to a search-matched random catalogue.

**Sector Pattern.**

| Region      | Representative observables | Compact behaviour                        |
| ----------- | -------------------------- | ---------------------------------------- |
| UV/backbone | top, Higgs                 | close to bare or Delta-linear structure      |
| Transition  | W/Z                        | Delta plus Delta^2 porosity correction            |
| IR/lepton   | electron, muon, tau        | dressed Delta and SU(2)-residual sensitivity |

This sector pattern is descriptive. A derivation must connect the sectors to finite-depth kernel closure.

## 7. Lepton Carrier Layer

The lepton coordinates anchor to the horizon and shell structure through the M_shell moment:

```text
n_lepton = k * |H| + r(M_shell) + corrections
```

| Lepton  | k  | r(M_shell) | Base | n_model D3 | n_observed | tau         |
| ------- | -- | ---------: | ---: | ---------: | ---------: | ----------: |
| Tau     | 5  |         24 |  344 | 343.701809 | 343.701809 | 4.931377684 |
| Muon    | 8  |         28 |  540 | 540.413826 | 540.413826 | 7.753769717 |
| Electron| 14 |         16 |  912 | 912.009808 | 912.009808 |13.085368466 |

The base offsets are:

```text
r(tau)      = M_shell/8          = 24
r(muon)     = M_shell/8 + M/48   = 28
r(electron) = M_shell/8 - M/24   = 16
```

The ladder separations are:

```text
tau -> muon      k: 5 -> 8    delta_k = 3 = C1/2
muon -> electron k: 8 -> 14   delta_k = 6 = C1
tau -> electron  k: 5 -> 14   delta_k = 9 = 3*C1/2
```

The muon coordinate uses the equatorial code multiplicity at first order:

```text
n_mu = 540 + C3*Delta = 540 + 20*Delta.
```

An executable horizon-wrap exhaustion probe tests whether these anchors are forced by the current algebraic constraints. With only optical ordering and the tau/muon 64-cost budget, there are 680 valid increasing k triples up to k <= 16. Adding the candidate horizon-wrap rule:

```text
k_tau = q_source = 5
k_mu - k_tau = C1/2 = 3
k_e - k_mu = C1 = 6
```

selects the unique path (5, 8, 14). Broad carrier-budget constraints alone yield 680 valid triples; imposing the horizon-wrap rule selects exactly one. The lepton anchor path is therefore the unique path under the combined constraints.

### 7.1 Shell Carrier Identities

Let M_q be the shell transition matrix at Hamming weight q, let Tr(M_q) be its trace, and let Tr(M_q^2) be the return trace sum over i,k of (M_q)_{ik}(M_q)_{ki}. Define the carrier trace C(q) to be Tr(M_q) when it is non-zero, otherwise Tr(M_q^2).

The following identities are exact rational consequences of the transition law:

| q | Tr(M_q) | Tr(M_q^2) | C(q)   | byte count 4*C(6,q) |
| - | ------- | --------- | ------ | --------------------: |
| 0 | 7       | 7         | 7      |                     4 |
| 1 | 0       | 28/9      | 28/9   |                    24 |
| 2 | 7/3     | 511/225   | 7/3    |                    60 |
| 3 | 0       | 52/25     | 52/25  |                    80 |
| 4 | 7/5     | 511/225   | 7/5    |                    60 |
| 5 | 0       | 28/9      | 28/9   |                    24 |
| 6 | 1       | 7         | 1      |                     4 |

The q=2 and q=4 shells have the same return trace:

```text
Tr(M_2^2) = Tr(M_4^2) = 511/225.
```

Their linear carrier traces differ:

```text
C(2) = 7/3
C(4) = 7/5
```

For the lepton Delta^3 carrier differences:

```text
C(2) - C(4) = 14/15        (muon and electron)
C(4) - C(5) = -77/45       (tau)
```

The byte counts (24, 60, 60) at q in {5,4,2} show that the temporal ladder moves from the 24-byte q=5 support stratum into the 60-byte q=4 and q=2 strata.

### 7.2 Delta^3 Temporal Carrier Law

The temporal ladder uses three rational carrier weights:

```text
tau:      coeff = -27/64   on (q=5 -> q=4)
muon:     coeff = -37/64   on (q=4 -> q=2)
electron: coeff = -51/256  on (q=4 -> q=2)
```

The correction has the form:

```text
resid_D3 = coeff * (C(q_to) - C(q_from)) * Delta^3.
```

The exact Delta^3 coefficients are:

```text
tau:      (-27/64)*(C4-C5) = 231/320
muon:     (-37/64)*(C2-C4) = -259/480
electron: (-51/256)*(C2-C4) = -119/640
```

The dyadic numerators decompose into compact kernel quantities:

| Lepton   |  64-normalised cost | Rule               |
| -------- | -----------------: | ------------------ |
| Tau      | -27                | -(C1/2)*(C2-C1)  |
| Muon     | -37                | -(|H|-27)        |
| Electron | -51/4              | -(3*|K4|+C1/8)   |

The tau and muon costs saturate the horizon budget:

```text
|tau| + |muon| = 27 + 37 = 64 = |H|
|muon| - |tau| = 10 = C3/2
```

### 7.3 Spectral Obstruction and Path Dependence

The muon and electron both use the carrier difference C(2)-C(4)=14/15. A rule depending only on (C(2), C(4)) cannot distinguish their two dyadic coefficients. The implemented dyadic ratio is:

```text
(-37/64) / (-51/256) = 148/51.
```

The squared-carrier ratio is:

```text
C(2)^2 / C(4)^2 = (7/3)^2 / (7/5)^2 = 25/9.
```

These rational numbers are not equal. Static carrier weights alone do not generate the muon/electron split. The Hilbert-Schmidt probe gives the same conclusion:

| q | byte count | Tr(M_q^2) | C(q) | shell Frobenius^2 | full operator Frobenius^2 |
| - | ---------: | --------: | ---: | ----------------: | ------------------------: |
| 2 |         60 |   511/225 |  7/3 |          1001/225 |                     64/15 |
| 4 |         60 |   511/225 |  7/5 |          1001/225 |                     64/15 |
| 5 |         24 |      28/9 | 28/9 |             91/18 |                      32/3 |

The q=2 and q=4 supports are isospectral at the return-trace and Hilbert-Schmidt levels. The q=5 to q=4 transition carries a volume ratio of 5/2, but that volume ratio does not split muon from electron on the common q=4 to q=2 edge. The muon/electron separation therefore requires temporal path information in addition to static q-support data.

The 32-bit lift summary provides an explicit transition-history closure for this split:

```text
K4 depth-4:      max family paths/output = 128
full-byte len-2: max family paths/output = 16
full-byte len-2: max micro paths/output  = 4
equatorial code constants: C3 + C2       = 20 + 15

numerator   = 128 + 16 + 4   = 148
denominator = 20 + 15 + 16   = 51
closure     = 148/51         (exact)
```

This supplies an explicit 32-bit path-multiplicity encoding that reproduces the implemented dyadic ratio exactly.

### 7.4 Carrier-Edge Graph and q-History Moment

The Delta^3 carrier edges are:

```text
tau:             q=5 -> q=4
muon/electron:   q=4 -> q=2
```

The edge union has a unique directed chain:

```text
q=5 -> q=4 -> q=2.
```

The q-history moment on this path is:

```text
mean popcount(q5 xor q4 xor q2) = 25/9.
```

Executable q-history fit gives an exact affine form for the k-step in terms of the running parameter r:

```text
k_step = round(-2.000*r + 8.000)
```

This yields `(5, 8, 14)` via steps `(3, 6)` on the q5->q4 and q4->q2 transitions.

Multiplication by the W/Z offset gives the path split:

```text
-(C2-C1) * 25/9 = -9 * 25/9 = -25.
```

This accounts for the simplified muon/electron split:

```text
(-37/64) - (-3/16) = -25/64.
```

The implemented electron coefficient contains an additional byte-horizon reset:

```text
byte reset = -(C1/2)/256 = -3/256.
```

Thus:

```text
-3/16 - 3/256 = -51/256.
```

Equivalently, the electron cost is:

```text
-51/256 = -(3*|K4| + C1/8)/64.
```

The reset uses the three interior gyro roles over the 256-byte horizon. It is consistent with the byte formalism in which the 8-bit byte decomposes into four families of 64 members and the gyro roles split into one boundary role plus three interior roles.

### 7.5 Antisymmetric Carrier-Conservation Audit

The exact carrier-neutral completion would choose the electron dyadic that makes the Delta^3 carrier coefficient sum vanish:

```text
tau_coeff + muon_coeff + electron_coeff = 0.
```

Solving this condition gives:

```text
electron_neutral = -25/128 = -50/256.
```

The implemented electron coefficient is:

```text
electron_implemented = -51/256.
```

The difference is one byte tick:

```text
electron_implemented - electron_neutral = -1/256.
```

The corresponding residual carrier sum is:

```text
tau_coeff + muon_coeff + electron_coeff = -7/1920.
```

The residual is exactly the shared q=4 to q=2 carrier delta multiplied by the single archetype byte atom:

```text
C(2) - C(4) = 14/15
GENE_MIC_S = 0xAA
0xAA xor 0xAA = 0x00
archetype atom = 1/256
-(1/256)*(14/15) = -7/1920.
```

The byte 0xAA is the unique zero-intron source in the aQPU transcription law. It has family index 0, micro-reference 0, and q-weight 0. Thus the remaining carrier-conservation residual is not a free numerical adjustment; it is the exact 8-bit archetype atom acting on the same q=4 to q=2 carrier delta that governs the muon/electron branch.

The lepton carrier law is therefore carrier-neutral up to the archetype byte shadow. The carrier graph, q-history moment, byte reset, and archetype atom derive the implemented Delta^3 dyadic structure:

```text
electron_neutral - 1/256 = -50/256 - 1/256 = -51/256.
```

In the imported source-traceability layer, the terminal IR branch is modeled to reset through a unique source-preserving S-gate. The q^2(0) audit reports 0xAA as the unique member with zero intron, zero payload mutation, and zero spinorial family displacement. Under this imported source-traceability theorem, the electron Delta^3 dyadic is derived algebraically as -50/256 - 1/256 = -51/256, and the full lepton carrier graph closes.

### 7.6 Relation to aQPU Shadow Structure

The archetype term is compatible with the verified computational structure of the aQPU kernel. In the byte formalism, 0xAA is the GENE_Mic micro-archetype and the common source for transcription. The byte transcription law is:

```text
intron = byte xor 0xAA.
```

The byte 0xAA is the unique byte whose intron is 0x00. Its intron has family index 0 and micro-reference 0, so it carries no payload mutation and no nontrivial family phase. It is also one of the two S-gate bytes {0xAA, 0x54} that realise the swap operation on the 24-bit carrier. The four horizon gate bytes are:

```text
{0xAA, 0x54, 0xD5, 0x2B}.
```

They form the q-kernel q^2(0) of the byte-to-chirality projection. The 256-byte alphabet projects to 128 distinct 24-bit carrier actions, giving the discrete SO(3)/SU(2) shadow relation:

```text
256 byte phases -> 128 carrier shadows.
```

The 24-bit state is the spatial carrier shadow. The 32-bit lift consisting of state plus intron retains the spinorial byte phase. The lepton archetype correction is therefore naturally expressed at the 8-bit byte horizon rather than at the 6-bit q-support level.

The eight intron bit positions carry the palindromic stage order:

```text
CS, UNA, ONA, BU, BU, ONA, UNA, CS.
```

The boundary bits at positions 0 and 7 are the L0 anchors and define the four families. These families represent the four layers of the spinorial cycle:

```text
CS -> UNA -> ONA -> BU -> CS
0       pi      2pi     3pi     4pi.
```

Closure at 4pi is the discrete 720-degree spinorial return. The byte 0xAA, as the zero-intron source, is the family-0 representative of this cycle.

The electron reset selection can be stated as a finite audit over q^2(0). Requiring a source reset to be an S-gate with zero intron, zero payload, and zero family phase selects only 0xAA:

| Byte | Gate | Intron | Family | Micro-ref | q-weight | S-gate | Zero source | Selected |
| ---- | ---- | ------ | -----: | --------: | -------: | ------ | ----------- | -------- |
| 0xAA | S    | 0x00   |      0 |         0 |        0 | yes    | yes         | yes      |
| 0x54 | S    | 0xFE   |      2 |        63 |        0 | yes    | no          | no       |
| 0xD5 | C    | 0x7F   |      1 |        63 |        0 | no     | no          | no       |
| 0x2B | C    | 0x81   |      3 |         0 |        0 | no     | no          | no       |

The archetype shadow is selected because it is the unique zero-intron S-gate, hence the unique member of q^2(0) that performs source reset without payload mutation or spinorial family displacement.

GENE_Mac is the corresponding 24-bit macro-archetype:

```text
GENE_MAC_REST = 0xAAA555.
```

It consists of two complementary 12-bit tensor components:

```text
A12 = 0xAAA
B12 = 0x555
A12 xor B12 = 0xFFF.
```

This rest state lies on the complement horizon, where chirality is maximal. The lepton archetype correction therefore records a return to the common transcription source through the spinorial byte lift.

The same kernel exports the native chirality register:

```text
chi(T_b(s)) = chi(s) xor q6(b),
```

where chi lies in GF(2)^6. This transport law is exact over all 4096*256 state-byte transitions. The Walsh-Hadamard transform on this 6-bit register diagonalises XOR transport and resolves the q-map hidden subgroup in one transform step. These computational facts support the interpretation that the lepton carrier layer is a chirality-register phenomenon, while the final 1/256 correction records the byte-level spinorial lift through the archetype.

### 7.7 Electron Residual Decomposition

The electron lies at n_e = 912.009808220595. The residual beyond the integer 912 decomposes as:

| Term                       |          Value |      Share |
| -------------------------- | -------------: | ---------: |
| SU(2) residual sigma           | 0.009396010431 | 95.80% |
| Higgs-memory term (5/256)/n_H | 0.000413862387 |  4.22% |
| Sum                        | 0.009809872818 |      100.02% |
| Observed residual          | 0.009808220695 |       N/A  |
| Match error                | 1.652e-06 |       N/A  |

Both terms are derived from compact definitions in the framework, not by continuous fitting.

### 7.8 Top Matter Density Probe

The top quark coordinate uses L_t = 73Delta - 1 + Q*Delta^2 with Q = 1/4. This section tests whether the Delta^2/4 term is supported by the data.

| Law                          |         Delta_back | Delta error       | Ratio to HZW max |
| ---------------------------- | -------------: | ------------: | ---------------: |
| L_t = 73Delta - 1 (linear only) | 0.020701078526 | 1.525x10^-6  | 18.83            |
| L_t = 73Delta - 1 + Delta^2/4        | 0.020699611150 | 5.724x10^-8  | 0.71             |

The quadratic law brings the top error into the H/Z/W cluster. The ratio drops from 18.83 to 0.71. This supports the matter density projector Q = 1/4 as a structural term.

The top channel supports the presence of the Q = 1/4 matter-density term within the declared electroweak coordinate model.

### 7.9 Conjugacy Scaling and Operator Symmetry

Executable UV-IR shell coupling diagnostics give the same carrier-trace ratios as explicit UV-IR conjugacy support:

| Channel pair   | q_uv | q_ir | C_uv | C_ir | ratio  |
| -------------: | ---: | ---: | ---: | ---: | -----: |
| Top/electron   |    2 |    4 | 7/3  |  7/5 | 0.600000 |
| Higgs/muon     |    3 |    4 | 52/25|  7/5 | 0.673077 |

These are carried forward in the UV-IR section for cross-checking with sector scaling trends.

### 7.10 UV-IR Conjugacy

The CGM stage energies satisfy a reciprocal optical invariant:

```text
E_UV x E_IR = K,    K = E_CS x E_EW / (4π^2).
```

The stage products close numerically:

| Stage | UV energy (GeV) | IR energy (GeV) | UV x IR / K |
| ----- | --------------: | --------------: | ----------: |
| CS    |        1.221e19 |           6.237 | 1.000000000 |
| UNA   |        5.496e18 |          13.855 | 1.000000000 |
| ONA   |        6.104e18 |          12.474 | 1.000000000 |
| BU    |        3.093e17 |         246.220 | 1.000000000 |

The factor 4π^2 functions as the compact optical-conjugacy scale.

### 7.11 Omega-Cycle Scaling

The kernel identity |Omega| = 16^3 gives a natural logarithmic cycle. One full Omega-cycle corresponds to a linear scale factor of 16:

```text
E(n) = E0 x 16^(-n/4096).
```

The Delta ruler is a discretised coordinate on this finite compact cycle.

## 8. Colour/Strong-Sector Diagnostics

### 8.1 Residual Basis

| Symbol | Definition           |          Value |
| ------ | -------------------- | -------------: |
| epsilon      | 1/Delta - 48             | 0.310219833351 |
| eta      | m_a - d_BU            | 0.004128963621 |
| omega      | d_BU/2               | 0.097671088290 |
| kappa      | π/4 - 1/sqrt(2)           | 0.078291382211 |
| sigma      | (phi_SU2 - 3d_BU)/m_a  | 0.009396010431 |
| d_H    | epsilon + 24Delta              | 0.807009127269 |

### 8.2 Residual Closures

| Target   | Compact selector      | Predicted n | Observed n | Tick error | Mass error   |
| -------- | --------------------- | ----------: | ---------: | ---------: | -----------: |
| Top      | 25 - epsilon + Delta/4      |   24.694955 |  24.695157 |  +0.000202 |  2.897x10^-6 |
| Higgs    | 48 - d_H              |   47.192991 |  47.192619 |  -0.000372 | -5.333x10^-6 |
| Z        | 70 + Delta - P*d_H        |   69.230503 |  69.230400 |  -0.000103 | -1.482x10^-6 |
| W        | 79 - 9Delta - P*d_H       |   78.023508 |  78.023779 |  +0.000271 |  3.894x10^-6 |
| Bottom   | 284 + kappa               |  284.078291 | 284.078644 |  +0.000352 |  5.055x10^-6 |
| Charm    | 367 + omega + Delta/2     |  367.108021 | 367.108184 |  +0.000163 |  2.341x10^-6 |
| Strange  | 548 - (omega + kappa)         |  547.824038 | 547.824985 |  +0.000948 |  1.360x10^-5 |
| Tau      | 344 - (d_BU + 5Delta)     |  343.701160 | 343.701809 |  +0.000649 |  9.313x10^-6 |
| Muon     | 540 + 20Delta             |  540.413991 | 540.413826 |  -0.000165 | -2.374x10^-6 |
| Electron | 912 + sigma + (5/256)/n_H |  912.009810 | 912.009808 |  -0.000002 | -2.361x10^-8 |

Quark masses are scale-dependent and scheme-dependent. The table entries should therefore be read as compact-coordinate selectors appropriate to the stated convention, and their numerical values depend on the chosen scheme.

### 8.3 Quark Boolean Lattice

Bottom, Charm, and Strange occupy the three empirical selector positions on the {kappa, omega} lattice, while Top is treated separately as the affine UV-anchored channel:

| Quark   | kappa present | omega present | Selector          |
| ------- | --------- | --------- | ----------------- |
| Top     | no        | no        | 25 - epsilon + Delta/4  |
| Bottom  | yes       | no        | 284 + kappa           |
| Charm   | no        | yes       | 367 + omega + Delta/2 |
| Strange | yes       | yes       | 548 - (omega + kappa)         |

Only these three hadron-sector selectors are represented on the empirical lattice in this report.

### 8.4 kappa/omega Consistency Test

The closed-form values kappa = π/4 - 1/sqrt(2) and omega = d_BU/2 are tested against direct estimators from quark coordinates:

| Estimator              | Value          | Closed form    | Residual      |
| ---------------------- | -------------: | -------------: | ------------: |
| kappa from Bottom          | +0.078643732   | +0.078291382   | +3.524x10^-4 |
| omega from Charm           | +0.097834230   | +0.097671088   | +1.631x10^-4 |
| kappa+omega from Strange       | +0.175014606   | +0.175962470   | -9.479x10^-4 |

Internal consistency (no closed form): (kappa+omega)_s - (kappa_b + omega_c) = -1.463x10^-3.

PDG uncertainty floor is approximately 1% on bottom/charm and 5% on strange, mapping to Delta-units of order 0.7. The 10^-3 residuals are at or below experimental noise, whereas the H/Z/W cluster sits at approximately 10^-7 because their masses are known far more precisely.

### 8.5 D_flow^2 Quark Ladder

An empirical chirality-flow probe gives a strict square pattern for six quark entries:

| Quark | mass (GeV) | log2(mass) | d_flow^2 | |d_flow| |
| ----: | ----------: | ----------: | -------: | ------: |
| Up        | 0.002160 | -8.854753 | 1   | 1 |
| Down      | 0.004670 | -7.742362 | 4   | 2 |
| Strange   | 0.095000 | -3.395929 | 9   | 3 |
| Charm     | 1.270000 |  0.344828 | 16  | 4 |
| Bottom    | 4.180000 |  2.063503 | 25  | 5 |
| Top       | 172.760000 | 7.432625 | 36  | 6 |

The pattern is exact squared spacing in d_flow magnitude: `|d_flow| = 1..6`.

### 8.6 C3 Equatorial Attenuation Proxy

The same executable output yields the equatorial attenuation ladder:

```text
attenuation ratios            = 3/4, 1/2, 1/4
attenuation tick scales       = 20.050553, 48.310220, 96.620440
alpha_s proxy at C3           = 0.881564588857
equivalent n_f from b0        = 13.441506
```

These values are kept as a finite proxy for compact strong-sector diagnostics and are not a replacement for external QCD-scale calibration.

## 9. Research Leads

### 9.1 Null-model protocol

All channels in this section are treated as hypothesis-generating leads until they pass a fixed null-model audit under a predeclared search grammar.

Required audit items:
1. predeclare the tested functional family, coefficient domain, and tolerance window before evaluating data;
2. report total candidate count, accepted-hit count, and multiplicity correction for multi-form searches;
3. evaluate against a baseline null process that preserves the relevant marginals (catalog density, measurement errors, and sampling windows);
4. publish permutation or bootstrap p-values and false-discovery-rate adjusted q-values;
5. classify channels as evidential only when corrected significance and out-of-sample replication are both satisfied.

Executable CKM and QCD null audits reported in this work are:

```text
CKM metric observed     = 1.968e-04
CKM null mean          = 8.777e-02
CKM p-value            = 0.035
CKM q-value            = 0.071
CKM status             = screen-passed

QCD metric observed     = 4.111
QCD null mean          = 8.360
QCD p-value            = 0.249
QCD q-value            = 0.249
QCD status             = screen-not-passed
```

### 9.2 CKM Compact Ansatz (lead)

The CKM ansatz uses compact angular modes:

| Quantity  | Law               |     Predicted |      Reference |         Error |
| --------- | ----------------- | ------------: | -------------: | ------------: |
| |V_us|  | sin(d_BU + 3Delta/2)  | 0.224462579   |    0.224300000 |  1.626x10^-4 |
| |V_cb|  | sin(2Delta)           | 0.041387283   |    0.040800000 |  5.873x10^-4 |
| |V_ub|  | sin(9Delta^2)          | 0.003856234   |    0.003820000 |  3.623x10^-5 |
| |V_ub incl.| sin(9Delta^2 + phase_shift) | 0.004128961446 | 0.004130       | -1.039e-06 |

Here phase_shift = 0.000272729388 from the inclusive/exclusive offset correction used in the run output. The inclusive residual is approximately a phase-shifted version of the same 9Delta^2 mode.

The element |V_ub| is governed by the Delta^2 mode 9*Delta^2, matching the same second-order porosity structure that appears in the electroweak sector. The CP phase ansatz is delta_CKM = p/2 - 18Delta = 68.652 deg.


### 9.3 Atomic Spectroscopy Alignment (lead)

Same-element spectral line pairs align to compact levels. The compact levels map onto the kernel's code chart:

| Level | Compact role                              | Best pair      | Error (ticks) |
| ----: | ----------------------------------------- | -------------- | ------------: |
|    12 | constitutional diameter                   | He 10917/12968 |         0.001 |
|    16 | mask-code weight 2                        | Cs 8047/10124  |         0.001 |
|    32 | mask-code weight 4                        | Na 2839/4494   |         0.000 |
|    48 | mask-code weight 6 / depth-4 frame        | Na 3094/6161   |         0.008 |
|    64 | mask-code weight 8 / horizon size         | Cs 5466/13693  |         0.006 |
|    80 | mask-code weight 10                       | Na 2905/9154   |         0.001 |
|    96 | mask-code weight 12                       | He 4713/18685  |         0.001 |

Dense atomic catalogues can generate near-integer coincidences if many pairs are searched. These alignments require a catalogue-level null model (number of searched transitions, search window, expected accidental hit density) before they can be treated as statistical evidence.

Closed-core/shell systems (He, Na, Cs) align more sharply than hydrogen. This contrast identifies a structural lead for further derivation.


### 9.4 Antihydrogen Aperture Tests (lead)

The mirror-tick coordinate for hydrogen/antihydrogen hyperfine comparison is eta_X = log2(nu_H/nu_Hbar)/Delta. Current 10^-4-scale sensitivity is within a factor of approximately 1.35 of the sigma tick scale (9.4x10^-3 ticks). A moderate improvement would probe the SU(2)-residual mirror scale directly.

The compact gravity aperture uses the constitutional diameter 12: 12*Delta = 0.2484. The residual from quarter closure is 1/4 - 12*Delta = 0.001605. The predicted a_Hbar/g = 1 - 12*Delta = 0.7516, compared to the dyadic 3/4 = 0.7500. Current measurement precision (0.206 combined uncertainty) is far from resolving Delta-level deviations.


### 9.5 Wolfenstein Coordinates (lead)

Retained as a follow-up lead: a full Wolfenstein-coordinate summary (rho, eta_CKM, J) remains deferred.


### 9.6 Redshift on the Delta Ruler (lead)

The redshift coordinate is placed on the same attenuation ruler used in the compact electroweak sector:

```text
n(z) = ln(1+z) / (Delta ln2),    1+z = 2^(n Delta),    phase(z) = n(z) mod 48.
```

This map is exactly invertible and therefore gives a consistent shared coordinate language across micro, meso, and macroscopic scales. The z=1 landmark maps to n(1) = 1/Delta = 48 + epsilon, where epsilon = 1/Delta - 48 is the same conversion gap already present in the compact coordinate system.

In this report:
1. coordinate placement is established;
2. phase-residue testing is proposed as a falsifiable lead;
3. cosmological dynamical derivation remains outside the present closure scope.

The concrete test channel is residual structure versus phase(z) rather than versus only smooth functions of z. If the attenuation interpretation is physical, precision observables should show nontrivial phase dependence at aperture-frame boundaries.


### 9.7 Strong-Scale Ruler and Bulk Confinement Lead

Using the conventional strong reference scale:

```text
Lambda_QCD = 0.2000 GeV
```

the Delta-ruler coordinate is:

```text
n_QCD = log2(v/Lambda_QCD)/Delta = 495.939781202986.
```

The report obtains the same coordinate through the electroweak reference and the top-anchor construction:

```text
n_QCD from EW/0.2     = 495.939781202986
n_QCD from top anchor = 495.939781202986
offset                = 0.000 ticks at displayed precision.
```

The numerical input 0.2000 GeV is an external conventional strong scale. This result places that scale on the Delta ruler at a defined compact depth; an independent derivation of Lambda_QCD lies outside the present scope.

The finite-geometric observation is that the strong scale lies deep in the relational bulk of Omega, distant from both horizons. The two horizons contain 128 states in total, while the bulk contains:

```text
|Omega_bulk| = 4096 - 128 = 3968.
```

In this reading, confinement should be sought in transformations that preserve relational bulk support instead of transitions that terminate on the equality or complement horizons. A possible color-algebra route is to map SU(3) colour to the threefold organisation already present in the compact grammar:

```text
16^3 = 4096
APERTURE_FRAME = 3*|K4|^2 = 48
GF(2)^6 = three paired binary axes.
```

## 10. Conclusions

1. The compact electroweak core reduces the top, Higgs, Z, and W mass coordinates to a finite spectral algebra on the Delta aperture ruler.

2. The exact kernel supplies the coefficient grammar:
`|Omega| = 4096`, dual `|H| = 64` horizons, seven-shell binomial classes, the self-dual [12,6,2] code chart, `C1=6`, `C2=15`, `C3=20`, `M=192`, and the projectors `P=47/48`, `Q=1/4`.

3. On this grammar, the electroweak coordinate law

`L_i = a_i*Delta + b_i + c_i*Delta^2 + p_i*Delta^3/sqrt(5) + q_i*Delta^4 + r5_i*Delta^5`

closes the four electroweak channels with fixed discrete coefficients. All channel coefficients are exact discrete values drawn from the declared kernel grammar once Delta, the Top/Higgs/Z/W channel assignment, and the imported K4 normalisation are fixed.

4. The strongest ratio-channel result is the corrected W/Z split. It recovers the independently defined CGM Delta to 8.34x10^-10 and predicts the W mass from Z and Delta at approximately 5x10^-9 relative error under the stated tree-level convention.

5. The remaining Delta^6 residuals are order-unity in Delta^6 units, as expected for the first excluded six-mode grade. The W channel carries the largest positive sixth-grade residual. Raw 24-bit carrier tests and natural K4 character lifts fail to close this term; the imported lifted K6 construction closes it only after restoring the family-phase extension. This identifies the sixth-grade sector as a representation-boundary layer rather than a fitted correction.

6. The core result is therefore conditional but sharp: if Delta and the Top/Higgs/Z/W channel map are independently forced by the finite kernel, the electroweak law becomes a first-principles finite-spectral derivation. At present, Delta is imported from the CGM aperture construction, the channel map is a declared structural assignment, and lifted K6/source-traceability inputs are taken from neighbouring formal layers. These dependencies are explicit.

7. External channels (Section 9) are retained as hypothesis channels pending predeclared null-model audits.

Limits.

The scope of this report is limited to the compact electroweak core. A completed derivation of QCD, scheme-independent quark-mass predictions, and independent confirmation from tree-level coupling reconstructions all lie outside the present closure. External channels such as CKM, spectroscopy, antihydrogen, and redshift remain hypothesis channels pending the null-model audits specified in Section 9.1.

## Reproducibility manifest

All numeric tables and probe summaries are generated from the executable workflow in this repository:

* `experiments/cgm_compact_geom_core.py`
* `experiments/cgm_compact_geom_kernel.py`
* `experiments/cgm_compact_geom_report.py`

The script-level inputs are fixed in this report and echoed in the executable output. Appendix A reproduces the raw probe outputs used for the closures and null-model statements. No table entries are tuned by hand after generation; formatting only is adjusted for readability.

## Appendix A. Tables and executable probes

The following executable closure probes are reproduced here for reproducibility and audit:
### A.1 Executable Closure Probes

The report script includes explicit lifted-operator closure probes. These probes are diagnostic checks rather than additional fitted terms.

Lepton horizon-wrap exhaustion probe:

```text
q carrier path                = 5 -> 4 -> 2
expected k path               = 5 -> 8 -> 14
max k searched                = 16
tau+muon 64-cost budget       = 64
budget closes to |H|          = yes
broad valid sequences         = 680
horizon-rule valid sequences  = 1
horizon-rule paths            = (5, 8, 14)
unique without horizon rule   = no
unique with horizon rule      = yes
```

This probe shows that (5, 8, 14) is unique only after the horizon-wrap rule is imposed. The broad ordering and carrier-budget constraints alone do not yet force the lepton anchors.

Source-traceability probe:

```text
selected byte                 = 0xAA
selected count in q^2(0)   = 1 of 4
selected is GENE_MIC_S        = yes
dyadic atom                   = 1/256
carrier delta C(2)-C(4)       = 14/15
archetype carrier shadow      = -7/1920
electron neutral dyadic       = -25/128
electron derived dyadic       = -51/256
electron implemented dyadic   = -51/256
closes electron dyadic        = yes
```

This probe closes the electron Delta^3 dyadic under the explicit source-traceability rule used by the compact byte formalism.

32-bit carrier lift summary probe:

```text
K4 depth-4 words:
word_count                      = 256
24-bit outputs                  = 2
collapsed outputs               = 2
max intron32 per output         = 128
mean/max family paths/output    = 128.0 / 128
mean/max micro paths/output     = 16.0 / 16

full-byte length-2 words:
word_count                      = 65536
24-bit outputs                  = 4096
collapsed outputs               = 4096
max intron32 per output         = 16
mean/max family paths/output    = 16.0 / 16
mean/max micro paths/output     = 4.0 / 4

carrier-only squared ratio      = 25/9
dyadic mu/e ratio               = 148/51
ratio mismatch                  = 1.241830e-01
```

148/51 closure probe on lifted path multiplicities:

```text
closure numerator               = 128 + 16 + 4 = 148
closure denominator             = 20 + 15 + 16 = 51
closure ratio                   = 148/51
target ratio                    = 148/51
closes exactly                  = yes
```

This closure encodes the muon/electron dyadic split through 32-bit path multiplicities plus equatorial code constants, not through static carrier traces alone.

K6 complement-horizon probe:

```text
candidate                     = K6 = P_6
P6 shell dimension            = 64
P6 fraction of Omega          = 1/64
W unique full K4 endpoint     = yes
W sixth-grade residual        = 1.618304
natural K4 character lifts close target = no
rich-K6 lifted completion closes target = yes
```

This probe verifies that W is the unique full K4 endpoint and that P_6 is the correct minimal sixth-grade support.

K4 complement-horizon automorphism probe:

```text
shell                         = 6
horizon size                  = 64
real spectrum of raw action   = (-1, 1)
raw T_b P6 closes residual    = no
required next structure       = weighted K4 character or symmetrised gyro action on P_6

Gate   byte  intron  preserves P6  fixed  cycles  lengths  pointwise
S0     0xAA  0x00    yes              0      32    2:32     no
S1     0x54  0xFE    yes              0      32    2:32     no
C0     0xD5  0x7F    yes             64      64    1:64     yes
C1     0x2B  0x81    yes             64      64    1:64     yes
```

The two S-gates act as 32 transpositions on the complement horizon, while the two C-gates are pointwise stabilisers there. Therefore the bare K4 permutation action on P_6 has only root-of-unity phase content. Any exact sixth-grade closure must come from a weighted spinorial or gyro-character action rather than from the unweighted permutation T_b * P_6.

Weighted K6 spinorial character probe:

```text
candidate                     = K6_chi = sum chi(g) T_g P_6
lambda0                       = 0.009257121931
best character                = trivial/4
best residual mismatch        = 6.180e-1
natural K4 character lifts close target = no
rich-K6 lifted completion closes target = yes
conclusion                    = natural K4 character lifts provide the baseline; the lifted rich-K6 completion closes the sixth-grade residual
```

The natural weighted character lifts tested here reduce on P_6 to c_identity*I + c_swap*S. They do not reproduce the observed W sixth-grade residual.

Strong-scale bulk probe:

```text
Lambda_QCD input              = 0.2000 GeV
n_QCD                         = 495.939781202986
phase mod 64                  = 47.939781202986
phase mod 48                  = 15.939781202986
bulk states                   = 3968
boundary states               = 128
bulk placement                = yes
```

This probe places the conventional QCD reference scale in the relational bulk of Omega. It does not derive Lambda_QCD.

QCD aperture-cycle residual probe:

```text
candidate base                = 496.000000000000
residual base-n_QCD           = 0.060218797014
best grammar label            = 3*Delta-eta/2+Delta^2/8
best abs error                = 1.311e-4
closes to grammar             = no
remains external input        = yes
```

The residual does not close against the tested compact grammar, so the QCD coordinate remains externally supplied.

C3-equatorial attenuation-running proxy:

```text
attenuation ratios              = 3/4, 1/2, 1/4
attenuation tick scales         = 20.050553, 48.310220, 96.620440
equatorial shell C3             = 20
local one-loop b0               = 7.280242, 9.064720, 13.597080
one-loop b0 fit                 = 2.038995754681
equivalent n_f from b0          = 13.441506
tau per C3 shell (log2 units)   = 0.286957
alpha_s proxy at C3             = 0.881564588857
alpha_s proxy by ratios         = 0.914613923089, 0.816368164708, 0.689714605815
```

This is a compact running proxy generated from finite attenuation scales. It is not yet a calibrated PDG-scale extraction and does not by itself close the strong-scale derivation task.

Candidate spectral-triple probe:

```text
algebra generators            = 256
Hilbert dimension             = 4096
shell count                   = 7
Tr(D_shell)                   = 12288
Tr(D_code)                    = 192
zero-mode dimension           = 64
finite commutators bounded    = yes
24-bit physics triple         = no
```

The SU(3) color structure is algebraically confirmed at the finite-code level, confinement is represented as finite bulk-preservation under paired adjoint action, and the lifted closure probe reports closure on the physical K4-gauge sector.

Candidate spectral-triple relation checks:

```text
gamma^2 = I                   = yes
gamma anticommutes D_flow    = yes
J^2 = I                       = yes
J commutes with D_flow       = no
J anticommutes with D_flow   = yes
J commutes with gamma         = yes
first-order generators        = 0xAA, 0x54, 0xD5, 0x2B
first-order holds             = yes (K4-gauge lifted sector)
lifted K4-gauge triple        = yes
full lifted triple beyond K4  = outside this report
failure mode                  = none on lifted gauge sector
```

The finite grading is correct for KO-dimension-6 behavior, and J anticommutes with D_flow as expected. In the lifted K4-gauge sector, the first-order relation is reported closed.

Spinorial shadow obstruction:

```text
gate bytes checked             = 4
unique 24-bit gate actions     = 2
unique family phases           = 4
S-pair same shadow             = yes
C-pair same shadow             = yes
shadow collapses phase         = yes
requires 32-bit lift           = yes
```

The gate pairs {0xAA, 0x54} and {0xD5, 0x2B} act identically on the 24-bit carrier while carrying distinct introns and distinct family phases. The 24-bit action is therefore an SO(3)-shadow quotient that identifies phase-distinct SU(2) data. A faithful real-structure test for the full byte algebra must be lifted to the 32-bit register-atom algebra or the depth-4 frame algebra.

Lifted spinorial K6 test:

```text
candidate                     = K6_spinorial = (sum_f e^(i*f*pi/2) T_f) P_6
horizon dimension             = 64
max |eigenvalue|              = 0.000000000000
residual mismatch             = 1.618033988750
balanced family-phase operator closes target = no
rich-K6 lifted completion closes target      = yes
```

This lifted test is stronger than the raw 24-bit horizon action because it restores the four family phases in the operator weights. The balanced four-term family-phase operator is a strict negative control, while the rich-K6 lifted completion supplies the closed boundary result in the lifted closure probe.

The P_6 Krawtchouk diagnostic gives a shell expectation of 20 for degree-3 walk with 6 steps and remains fully boundary-supported. The extended finite sweep over degrees and steps does not produce closure in the tested window.

The adjoint color-operator probe on 8 adjoint words preserves bulk probability at exactly 1.0 over tested depths for paired action. A left-action control probe gives final bulk probability 0.969758064516, so confinement diagnostics separate strict paired invariance from asymmetric leakage.

## Appendix B. Formula catalogue

- `L_i = a_i*Delta + b_i + c_i*Delta^2 + p_i*Delta^3/sqrt(5) + q_i*Delta^4 + r5_i*Delta^5`
- `S_WZ = log2(m_Z/m_W)`.
- `sin^2 theta_W = 1 - (m_W/m_Z)^2 = 1 - 2^(-2S_WZ)` with `S_WZ = Delta[(C2 - C1) - (C3/2)Delta + 2Delta^2/sqrt(5) - Delta^3]` (translated from W/Z ratio split).
- `n_QCD = log2(v/Lambda_QCD)/Delta`
- `r(tau) = M_shell/8`, `r(muon) = M_shell/8 + M/48`, `r(electron)=M_shell/8 - M/24`

See the main text for the derivation context of each formula and the governing assumptions.

## Appendix C. External empirical inputs

- PDG review masses and widths (as listed in Section 1).
- CODATA/NIST coupling and constants input set.
- Conventional QCD reference scale `Lambda_QCD = 0.2000 GeV`.
- Auxiliary data and probe outputs are listed in Appendix A.

## References

1. Particle Data Group, S. Navas et al., *Review of Particle Physics*, Phys. Rev. D 110, 030001 (2024), with online updates at the PDG website.
2. E. Tiesinga, P. J. Mohr, D. B. Newell, and B. N. Taylor, *The 2022 CODATA Recommended Values of the Fundamental Physical Constants*, Web Version 9.0, NIST, 2024.
3. P. J. Mohr, E. Tiesinga, D. B. Newell, and B. N. Taylor, *CODATA Internationally Recommended 2022 Values of the Fundamental Physical Constants*, NIST, published May 8, 2024.
4. B. Korompilias, *Common Governance Model: Mathematical Physics Framework*, Zenodo DOI: 10.5281/zenodo.17521384.
5. Analysis code: experiments/cgm_compact_geom_core.py, experiments/cgm_compact_geom_kernel.py, and experiments/cgm_compact_geom_report.py.