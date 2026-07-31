# Analysis: Allometry and Ancestry Preservation in the Common Governance Model

**Citation:** Korompilias, B. (2025). Common Governance Model: Mathematical Physics Framework. Zenodo. https://doi.org/10.5281/zenodo.17521384. Repository: https://github.com/gyrogovernance/science

**Reproducibility:** `experiments/hqvm_cgm_allometry_1.py`, `experiments/hqvm_cgm_allometry_2.py`, `experiments/hqvm_cgm_allometry_3.py`, `experiments/hqvm_cgm_allometry_run.py`. Combined output: `experiments/hqvm_cgm_allometry_results.txt`. External catalogs: `data/catalogs/allometry/` (PanTHERIA, AnAge, AnimalTraits, city wage and road-length series; provenance in the catalog directory).

**Subject classes (arXiv-style):** q-bio.PE; q-bio.QM; physics.bio-ph; math-ph

**Keywords:** Common Governance Model, allometry, Kleiber relation, holographic constraint, source-accessibility exponent, West-Brown-Enquist network theory, metabolic scaling, holonomic quantum virtual machine, QuBEC thermodynamics, ancestry preservation, aperture gap, transition state theory, Damuth relation

## Abstract

Allometry is the empirical study of power-law relations between organism size and physiological or morphological traits. This analysis explains allometric scaling as a consequence of the Common Governance Model (CGM), an axiomatic framework for fundamental physics and information science. CGM rests on a single axiom: every distinguishable state in a coherent system must trace to a common source. This analysis uses two consequences of that requirement: (i) the transport register has a fixed chirality dimension `d = 6`, and (ii) its equilibrium occupation fixes a canonical network delivery exponent. A concrete computational realization used here is the Holonomic Quantum Virtual Machine (hQVM): a 4096-state finite carrier with a six-bit chirality register `χ ∈ GF(2)^6`, one bit per dipole mode (six degrees of freedom).

The central result identifies empirical allometric exponents with evaluations of a source-accessibility exponent. Define the source-accessibility exponent `a := (d ln C) / (d ln M)`, where `M` is bulk organism size and `C` is source-connected delivery capacity. On the hQVM carrier, under the identification `C ∝ |root(A)|` and `M ∝ |Reach(A)|`, the Square-Root Cluster Theorem fixes `a = 1/2`. Under three-dimensional geometric similarity at constant density, surface-to-volume scaling fixes `a = 2/3`. Under the West-Brown-Enquist network assumptions, identifying the effective network dimension with the QuBEC thermal mean `⟨N⟩` (which equals `3` at `d = 6`) gives delivery exponent `a = ⟨N⟩/(⟨N⟩+1) = 3/4`, the Kleiber relation. A continuous parameter `μ`, defined as the fraction of total metabolic flux carried by the network channel at fixed dimensionality, interpolates between the surface and network endpoints. As a modeling step, `μ` is identified with the excitation coordinate of the same chirality register whose aperture gap `Δ` enters the CGM gravity construction [15].

Three combinatorial constructions on the family of carriers `hQVM(d)`, indexed by chirality dimension `d`, each select the physical value `d = 6` as a consistency condition. Organismal scaling is therefore bounded by chirality dimension six. A chemical activation energy of 0.645 electron volts at mammalian core temperature follows from the aperture gap with body temperature as the sole biological input, and falls inside the independently measured 0.6--0.7 electron-volt band of the Metabolic Theory of Ecology. Development and life history are reported as a Tier C structural reading of progressive coverage of the ancestry-preserving transport network. City and company scaling exponents follow from a distinct Tier C structural hypothesis and are reported separately at reduced confidence relative to the organism channel basis.

## 1. Scope and Tiers of Claim

### 1.1 Scope

This analysis explains allometric scaling as a consequence of the CGM requirement that ancestry be preserved under recursive operational transformation. Quantities used in the channel basis are drawn either from the hQVM carrier at the physical chirality dimension `d = 6` (finite carrier layer) or from the continuum monodromy invariants of the CGM aperture gap `Δ` (continuum layer). Biological bridge identifications that map carrier quantities to organismal observables are stated with each claim. The empirical exponents of metabolic and morphological scaling are read as evaluations of the source-accessibility exponent defined in Section 4.

### 1.2 Tiers of Claim

Three tiers are used throughout, matching the convention of the CGM research programme.

**Tier A.** Channel-basis exponents forced by the hQVM constructions of Sections 3 through 5, together with the biological bridge that identifies bulk capacity and source-accessible capacity with organism mass and metabolic delivery. Metabolic rate, mass-specific metabolic rate, circulatory time, and the two dimensionless sum rules of Section 7 belong to this tier as channel arithmetic under that bridge. Heart and respiratory rate, blood volume, capillary number, service radius, and volumetric capillary density require the additional physiological identifications listed with each trait in Section 7.

**Tier B.** Forced given one auxiliary physiological assumption external to the CGM channel basis, specifically the West-Brown-Enquist assumption that the pressure drop across the circulatory network and the terminal blood velocity are independent of body size. Aortic radius and total vascular resistance belong to this tier.

**Tier C.** A structural hypothesis tested against external data and open relative to the CGM dimensional theorem. Development as percolation coverage, the chemical activation mapping of Section 9, the absolute mass intercept of Section 10, brain-mass placement, and the identification of urban and corporate infrastructure networks with the six-dimensional chirality-transport register belong to this tier.

### 1.3 External Data

Empirical audits use the PanTHERIA mammalian trait database, the AnAge longevity database, the AnimalTraits metabolic and brain-mass database, and city wage and road-length series, all catalogued in `data/catalogs/allometry/` with source provenance recorded there. These audits evaluate the closed-form channel basis of Section 4 under ordinary least squares and reduced major axis regression protocols defined in Section 14.

### 1.4 Self-Containment

Quantities drawn from the wider CGM framework are restated at the point of use. Companion analyses and kernel specifications are listed in the References.

## 2. Ancestry Preservation as the Physical Content of an Organism

CGM defines ancestry preservation as the joint satisfiability of identity and individuality under recursive transformation. Identity is the requirement that a reference remain recoverable from public transition data. Individuality is the requirement that transitions produce distinguishable outcomes. Depth-four balance is the requirement that recursive composition closes after four stages while retaining a nonzero aperture through which distinguishable operation remains possible.

In the continuum sector, the same requirement manifests as gravity. Gravity, in CGM, is the emergent balance establishing preservation of ancestry through freedom of identity and individuality [15]. Identity requires that local configurations remain recoverable as part of one source-linked history. Individuality requires that configurations permit real displacement and differentiation. Balance requires that those displacements close into a coherent operational cycle that retains both distinguishability and source relation. Directionally unbiased accountability on a closed finite domain fixes the angular normalization `Q_G = 4π` steradians, since complete angular coverage in three dimensions is `4π` and the sphere is the unique closed geometry with no privileged direction. In this reading, mass-energy is accumulated source-linked structure required to sustain spherical alignment under displacement. The gravitational refractive depth `τ_G` and the position-dependent coupling `G(ψ)` are continuum readouts of the same depth-four closure and aperture gap `Δ` realized on the finite carrier [15].

An organism is a bounded source-to-bulk transport system realizing the same ancestry-preserving closure under directed delivery. Physiological homeostasis supplies recoverability of reference. Metabolic differentiation supplies distinguishable outcomes across parts and times. Circulatory and respiratory closure supplies depth-four balance with sustained throughput. Metabolism is the power required to maintain this source-to-bulk organization under continual exchange.

The computational realization used here is the hQVM carrier. Its reachable manifold `Ω` represents bulk operational capacity. Its horizons, each of cardinality `|H|`, represent source-connected boundary roots. The product identity `|Ω| = |H|^2` separates bulk capacity from root-accessible delivery and yields the absolute transport floor `a_SR = 1/2` derived in Section 4.1. The aperture gap `Δ` that enters the gravitational sector also enters the chemical-activation construction of Section 9. The chirality dimension `d = 6` that closes the organism channel basis is established in Section 3.1 and is the same integer used in the gravitational sector. The remainder of this analysis evaluates the source-accessibility exponent across computational, spatial, and network scales at fixed dimensionality.

## 3. Kernel and QuBEC

### 3.1 Carrier, Horizons, and Shell Census

The hQVM realizes the CGM ancestry-preservation conditions as a finite reversible transducer. Its carrier state is a 24-bit register split into two 12-bit components, `A12` (active gyrophase) and `B12` (passive gyrophase). One transition step is selected by an 8-bit input byte. A word is a finite sequence of successive byte transitions. A depth-four word is a four-byte sequence that implements the minimal closure pattern required by CGM balance while retaining a nonzero aperture. The reachable state set from the rest configuration, denoted `Omega`, contains 4096 states. This set factorizes as a product of two 64-element boundary sets (the horizons),

```text
|Omega| = |H|^2 = 4096,  |H| = 64
```

`|Omega|` counts the reachable states; `|H|` is the horizon cardinality per face. The horizon size equals `2^d` with `d = 6`, the chirality dimension: the number of independent binary transport channels on the six-bit register `χ`, obtained by collapsing the pair-diagonal difference `A12 XOR B12` to one bit per dipole mode.

CGM fixes `d = 6` by deriving three spatial dimensions with six degrees of freedom [15]. Non-absolute unity forces three rotational degrees of freedom through the minimal non-abelian compact Lie group SU(2). Non-absolute opposition forces three further translational degrees of freedom through the semidirect extension to SE(3). The total is six degrees of freedom, and the chirality register carries one binary channel for each. This dimensional result is an input to the channel basis of the present analysis.

The reachable state set partitions into seven shells by the Hamming weight of the chirality register, running from shell 0 through shell 6. Shell `k` contains `64 × C(6,k)` states, where `C(6,k)` denotes the binomial coefficient, giving the census 64, 384, 960, 1280, 960, 384, 64. This sums to 4096 and is symmetric about shell 3, the equatorial maximum. The census is verified by exhaustive enumeration on the hQVM carrier.

A four-byte closure word carries a projected information content of `8d` bits at general chirality dimension `d`. At `d = 6` this gives 48 bits.

The aperture gap of the hQVM carrier at general chirality dimension `d` is

```text
Delta_kernel(d) = 1 / (8d)
```

`Delta_kernel` is the discrete byte-lattice aperture. At `d = 6` this gives `1/48 ≈ 0.020833`. The continuum monodromy version, obtained from the ratio of the dual-pole holonomy defect `delta_BU` to the aperture reference scale `m_a`, is

```text
Delta = 1 - delta_BU / m_a = 0.020699553913
```

Here `delta_BU ≈ 0.195342` rad is the geometric phase accumulated by a closed loop traversing both constitutional poles of the carrier, and `m_a ≈ 0.199471` is the aperture reference scale of the CGM continuous formalism. The two values differ by approximately six parts in ten thousand, a resolution-scale distinction between the discrete byte lattice and the continuum limit. Both are used below: the kernel value for rational channel statements and the continuum value for the chemical clock of Section 9. The same aperture gap enters the gravitational refractive depth and the coupling `G(ψ)` in the CGM gravitational theory (see [15]).

### 3.2 QuBEC Partition Function and Shell Occupation

The QuBEC (Quantum Bose-Einstein Computational Condensate) formalism assigns to the hQVM carrier a partition function that weights states by the shell index `N = wt(χ)`, the Hamming weight of the six-bit chirality register, using a shell-weight coordinate `lambda`:

```text
Z_1(lambda) = |H| × (1 + lambda)^d
```

`Z_1` is the canonical partition function, built from horizon degeneracy `|H| = 64` and chirality dimension `d = 6`. At the physical dimension, `Z_1(lambda) = 64 (1+lambda)^6`. This is the partition function of `d` independent, non-interacting two-level systems, each carrying a Boltzmann weight ratio `lambda` between an excited and a ground state, multiplied by the fixed boundary degeneracy.

The mean shell occupation follows directly:

```text
⟨N⟩ = d × lambda / (1 + lambda)
```

`⟨N⟩` is the mean shell index: the average number of excited transport modes at activity coordinate `lambda`. Under a thermal mapping, `lambda = exp(-epsilon_bit/(kT_eff))` defines an effective conjugate temperature `T_eff` for the register, with `epsilon_bit` the energy cost of one chirality-mode excitation. The coordinate `lambda` is a kernel occupation parameter before any biological reading. Section 6.2 identifies it with metabolic excitation as a modeling step, with cardiac output as a fraction of maximum output proposed as the primary physiological proxy for falsification.

### 3.3 Thermal Point and the West Network Exponent

On the positive-temperature two-level domain `0 < lambda ≤ 1`, the occupation reaches its maximum at `lambda = 1`:

```text
⟨N⟩|_(lambda=1) = d/2 = 3
```

For unrestricted `lambda > 1` the occupation continues above `d/2`. The physical metabolic band used in this analysis is `lambda in (0,1]`. This thermal-point identity is a closed-form consequence of the partition function. The reproducibility output confirms the equivalent identity `⟨N⟩ = M_shell / |H|` with shell moment `M_shell = d × 2^(d-1) = 192` at `d = 6`.

The West-Brown-Enquist (WBE) theory of allometric scaling derives the relation `B ∝ M^(D/(D+1))` from three structural assumptions on a resource-supply network: the network fills the organism volume through a fixed number of branching generations, terminal delivery units have properties independent of organism size, and the network minimizes the energy dissipated in circulation [4]. The effective dimension `D` in that relation is fixed within the WBE framework by geometric assumptions about branching ratios.

This analysis identifies `D` with the thermal shell occupation mean of the chirality register:

```text
D_eff := ⟨N⟩|_(lambda=1) = d/2
```

`D_eff` measures the mean number of simultaneously active transport modes at thermal occupation. At the physical chirality dimension `d = 6`, `D_eff = 3` and

```text
a_bulk := D_eff / (D_eff + 1) = d / (d+2) = 3/4
```

`a_bulk` is the bulk network delivery exponent. This reproduces the Kleiber relation [2]. The same formula evaluated at other even chirality dimensions gives

```text
a_bulk(2) = 1/2,  a_bulk(4) = 2/3,  a_bulk(6) = 3/4
```

These are algebraic parallels on the chirality-dimension family. The physical organism remains at `d = 6`. At chirality dimension two, `a_bulk(2) = 1/2` coincides with the product-structure throughput exponent implied by `|Omega| = |H|^2`. At chirality dimension six, the computational, spatial, and network evaluations separate.

## 4. The Channel Basis

Ancestry preservation requires that every distinguishable part of a coherent system remain reconstructable from a common origin. Define the source-accessibility exponent

```text
a := (d ln C) / (d ln M)
```

where `M` is bulk size and `C` is source-connected delivery capacity. Three organizational scales supply three evaluations of `a` under separately stated assumptions, and the time and service exponents follow from these values at chirality dimension six.

An organism realizes these evaluations as a macroscopic transport system. Its circulatory and respiratory networks maintain continuous supply from a common metabolic source across the tissue volume. As size and activity vary, delivery shifts between the spatial surface channel and the internal network channel at fixed dimensionality, while the absolute transport floor set by the carrier product remains `a_SR = 1/2`.

### 4.1 Computational Scale: Holographic Throughput, `a_SR = 1/2`

As established in Section 3.1, each transition is an 8-bit byte operator, and the map `q6` assigns to each byte its six-bit transport class on the chirality register. Define `q6(b)` as the six-bit transport class assigned to byte `b` by the hQVM transition rule. The fiber over a transport class `q` is the set of four bytes sharing that class, one per family. A generator set `A` is fiber-complete when it contains all four family variants for every included transport class. The transport rank `r(A)` is the GF(2) rank of the span of `q6(b)` over `b` in `A`.

Under fiber-complete restriction, the reachable set factorizes as `U_R × V_R` with `|U_R| = |V_R| = 2^{r(A)}`. The boundary root cardinality is proportional to `2^{r(A)}`, while total capacity is proportional to `(2^{r(A)})^2`. The Square-Root Cluster Theorem states

```text
|Reach(A)| = (2^r(A))^2
```

`|Reach(A)|` counts states reachable under the restricted generator set `A`; `r(A)` is that set's transport rank. The theorem holds for `r(A) >= 1` and is verified by breadth-first search on the hQVM carrier across all structured generator restrictions (52 of 52 cases across chirality dimensions one through eight).

The scale-free structural identification of this section sets organism size `M` proportional to reachable capacity and delivery capacity `B` proportional to the surviving boundary root. This transfers the carrier log-log slope into the organism channel basis. Absolute metabolic capacity is anchored later by the chemical-clock construction of Section 9. Under this identification, with `C` the root and `M` the capacity, the log-log slope is

```text
a_SR = 1/2
```

`a_SR` is the source-accessibility exponent at the computational scale. The conjugate product structure of the carrier forces delivery capacity to scale as the square root of bulk capacity. This exponent `1/2` is the absolute transport floor of the hQVM carrier.

### 4.2 Spatial Scale: Surface Exchange, `a_surf = 2/3`

Under geometric similarity in three spatial dimensions at constant density, the external exchange area is a two-dimensional surface serving a three-dimensional volume:

```text
a_surf := (n - 1) / n = 2/3
```

With spatial dimension `n = 3`, `a_surf` is the surface exchange exponent. The square-cube relation of classical scaling is this spatial evaluation of source-accessibility: a coherent volume exchanges with its exterior through a surface whose area grows more slowly than the volume it serves. The spatial dimension used here is `n = 3`, the same dimension derived in CGM's spacetime characterization; the surface exponent is the geometric-similarity readout of source-accessibility at that dimension.

The hQVM carrier supplies a discrete readout that coincides with the surface exponent `2/3` at `d = 6`. The byte intron carries a palindromic phase structure with a fold at the depth-four balance boundary. Phase is quantized on a 256-tick turn scale, and the fold unit used here is one tick aggregated into `1/32` of a full turn. The carrier surface exponent is the ratio of the aperture gap to this fold unit:

```text
a_surf,kernel(d) := Delta_kernel(d) / (1/32) = 4/d
```

`a_surf,kernel` is the discrete carrier analogue of the surface exponent, with carrier aperture gap `Delta_kernel(d) = 1/(8d)`. At `d = 6` this equals `2/3`. A spatial-boundary generalization at half-dimension `n = d/2` gives

```text
a_surf,spatial(d) := (n - 1) / n = 1 - 2/d
```

These two expressions agree only at `d = 6`, where `4/d = 1 - 2/d`. Section 5.2 records that coincidence as a consistency condition. Organism-level nulls in this analysis use the common value `2/3` at physical `d = 6`.

The continuum monodromy version of the fold construction is

```text
a_surf,continuum := Delta / (1/32) = 0.662386
```

`Delta = 0.020699553913` is the continuum aperture gap entering this fold ratio. This value carries a correction analogous to the monodromy dressing applied to the fine-structure constant and to Newton's constant in the CGM framework.

### 4.3 Network Scale: Bulk Network Throughput, `a_bulk = 3/4`

An internal delivery network folds the common origin into the bulk, so that exchange proceeds through the tissue volume as well as through the external surface. As derived in Section 3.3, the QuBEC thermal shell mean supplies the effective network dimension `D_eff = d/2 = 3` under the WBE assumptions, and

```text
a_bulk := D_eff / (D_eff + 1) = 3/4
```

`a_bulk` is again the bulk network delivery exponent, now evaluated at thermal shell mean `D_eff = 3` for chirality dimension six. The resulting Kleiber exponent `3/4` is the empirical metabolic scaling relation first documented by Kleiber (1932) [2]. Relative to the surface evaluation of Section 4.2, the network channel raises delivery from `2/3` to `3/4` by distributing origin access throughout the tissue volume. Relative to the computational floor of Section 4.1, the channel remains bounded below by `a_SR = 1/2`.

### 4.4 Circulatory Time, `a_time = 1/4`

At `d = 6`, the volume-flow construction (`a_time = 1 - a_bulk`) and the channel-lift construction (`a_time = a_bulk - a_SR`) yield the same value:

```text
a_time := 1 - a_bulk = a_bulk - a_SR = 1/4
```

`a_time` governs the characteristic transport time as a function of body mass. Section 5.3 derives the coincidence of these two constructions at `d = 6` as a consistency condition. Physiological rates scaling with the inverse of a characteristic circulation time carry exponent `-1/4`, and physiological times carry exponent `+1/4`.

### 4.5 Service Radius, `a_service = 1/12`

With `n = d/2 = 3` denoting the spatial dimension corresponding to the six-mode split,

```text
a_service := a_time / n = (1/4) / 3 = 1/12
```

`a_service` scales the radius of tissue served by one terminal delivery unit. Inverse service radius scales as `-1/12`. Volumetric capillary density is a separate quantity treated in Section 7.

## 5. Dimension-Six Consistency Relations

The chirality dimension used throughout is `d = 6`, established in Section 3.1. This section reports three combinatorial or thermodynamic constructions on the family of carriers `hQVM(d)`, indexed by chirality dimension `d`. Each construction equates independently defined expressions on the channel basis and selects `d = 6` as a consistency condition, matching the chirality dimension fixed independently in Section 3.1.

### 5.1 Consistency Relation I: Thermal Bulk Exponent Equals Depth-Four Projection Fraction

A four-byte closure word on the carrier projects `8d` bits at chirality dimension `d`. The horizon cardinality per face is `2^d`. The depth-four projection fraction is the ratio

```text
a_d4(d) := 8d / 2^d
```

`a_d4` is the fraction of horizon degrees that a closed word can address per unit closure. Setting this equal to the thermal bulk exponent `a_bulk(d) = d/(d+2)` of Section 3.3 gives the condition

```text
d / (d+2) = 8d / 2^d  <=>  2^(d-3) = d + 2
```

Evaluated over positive integers, `2^(d-3) - (d+2)` is negative for `d` from one through five and positive for `d` at least seven, with equality only at `d = 6`, where `2^3 = 8 = 6+2`. This is verified in Section 3b of `hqvm_cgm_allometry_results.txt`.

### 5.2 Consistency Relation II: Bulk-Surface Gap Equals Service Exponent

Section 4.2 introduced two candidate surface generalizations,

```text
a_surf,kernel(d) = 4/d,     a_surf,spatial(d) = 1 - 2/d
```

which agree only at `d = 6`. With the fold generalization `a_surf(d) = 4/d` and `a_time(d) = a_bulk(d) - 1/2`, the requirement that the bulk-surface gap equal the service exponent,

```text
a_bulk(d) - a_surf(d) = a_time(d) / (d/2)
```

reduces, after substitution and simplification, to the quadratic condition

```text
(d - 6)(d + 1) = 0
```

whose only positive integer root is `d = 6`. This is verified in Section 3b of `hqvm_cgm_allometry_results.txt`.

### 5.3 Consistency Relation III: Two Constructions of Circulatory Time Coincide

The first construction follows the WBE volume-flow argument: a characteristic circulation time equals a transported volume (proportional to mass under isometric density) divided by a flow rate (proportional to the bulk delivery exponent),

```text
a_time^(i)(d) = 1 - a_bulk(d) = 2 / (d+2)
```

The second construction is the channel-lift gap between the bulk network channel and the holographic throughput channel:

```text
a_time^(ii)(d) = a_bulk(d) - a_SR = d/(d+2) - 1/2 = (d-2) / (2(d+2))
```

Equating the two,

```text
2/(d+2) = (d-2) / (2(d+2))
```

reduces to the linear condition in `d`,

```text
4 = d - 2
```

giving `d = 6` as the unique positive integer root. At `d = 6` both constructions give `a_time = 1/4`, confirmed numerically to sixteen decimal digits in Section 3b of `hqvm_cgm_allometry_results.txt`.

Relations I, II, and III are algebraically independent: an exponential-linear crossing, a quadratic condition, and a linear condition respectively. Each locks the channel basis to chirality dimension six, confirming that organismal timing, surface exchange, and bulk delivery close under the six degrees of freedom of Section 3.1.

## 6. The Continuous Mu-Parameter

### 6.1 Exact Flux-Fraction Identity

Total metabolic throughput is written as the sum of two channels at fixed dimensionality `d = 6`:

```text
B(M,x)  = B_s(M,x) + B_b(M,x)
B_s(M,x)= k_s(x) M^(2/3)
B_b(M,x)= k_b(x) M^(3/4)
```

Here `M` is organism mass and `x` denotes physiological state (activity level, taxon, temperature, and related controls). The coefficients `k_s(x)` and `k_b(x)` are state-dependent prefactors.

Define the network flux fraction

```text
μ(M, x) := B_b(M, x) / B(M, x)
```

`μ` is the share of total throughput carried by the network channel at size `M` and state `x`. The local scaling exponent follows by direct differentiation,

```text
a(M, x) = (2/3)(1 - μ) + (3/4) μ = 2/3 + μ/12
```

`a` is the log-log slope of total throughput against mass. This holds for any positive coefficients, valid at every organism size. The endpoints are `μ = 0` (pure surface exchange, `a = 2/3`) and `μ = 1` (pure network delivery, `a = 3/4`). Variation of empirical exponents between these endpoints is variation of `μ` across samples.

For coefficients independent of `x`,

```text
μ(M) = k_b M^(1/12) / (k_s + k_b M^(1/12))
```

so `μ` increases with body mass even at fixed activity. Size dependence and activity dependence are distinct. Inverting on the physical band at fixed `x`,

```text
μ = 12(a - 2/3),   μ in [0, 1]
```

### 6.2 Identification of Mu with the Metabolic Excitation Parameter

Section 3.2 established that the chirality register admits an occupation coordinate `lambda` with

```text
μ_QuBEC(lambda) = 2 lambda / (1 + lambda)
```

`μ_QuBEC` is the occupation fraction read from the chirality-register measure at shell-weight coordinate `lambda`. On the physical band `lambda in (0,1]`, this ranges from near 0 (as `lambda → 0`) to 1 (at `lambda = 1`).

The flux fraction `μ(M, x)` is defined by throughput decomposition (Section 6.1). The QuBEC occupation `μ_QuBEC(lambda)` is defined by the chirality-register measure (Section 3.2). The identification `μ(M, x) = μ_QuBEC(lambda)` is a modeling step of this analysis. It equates the network's geometric share of throughput with the thermal occupation measure of the register that supplies the aperture gap, and recovers `lambda` from `μ` by `lambda = μ / (2 - μ)`. The primary physiological proxy proposed for falsification is cardiac output as a fraction of maximum output. Capillary perfusion fraction and oxygen extraction ratio are alternative candidates, each measuring a different process and requiring its own test protocol.

With this identification,

```text
a(lambda) = 2/3 + lambda / (6(1+lambda))
```

The local scaling exponent `a` is now written as a function of the register occupation coordinate `lambda`. A resting or basal metabolic state corresponds to small `lambda`: the bulk network channel remains near its ground state, oxygen supply relies on surface diffusive exchange, and the exponent sits near `2/3`. An active or field metabolic state corresponds to `lambda` near one: the circulatory network thermalizes, oxygen supply shifts to perfusion-limited bulk delivery, and the exponent sits near `3/4`. The endpoints are fixed by the hQVM carrier structure. Intermediate values are fixed by the excitation state of the same chirality register that supplies the aperture gap governing gravitational coupling.

### 6.3 Dual Time Channels: Egress and Ingress

CGM balance is dual (BU-Egress and BU-Ingress). A living ancestry-preserving system therefore has two distinct time scaling relations.

**Ingress / maintenance time.** This is the closed-loop time that scales with the maintenance of the existing network (BU-Ingress sector). It is the time required to circulate resources through the existing transport volume. Its exponent is the volume-flow exponent:

```text
a_in(μ) := 1 - a(μ) = 1 - (2/3 + μ/12) = 1/3 - μ/12
```

At the fully thermalized endpoint μ=1, `a_in = 1/4`, recovering the West circulatory time exponent. At lower μ, `a_in > 1/4`.

**Egress / construction time.** This is the forward build time required to construct the network up to functional capacity (BU-Egress sector). Return closure is not required for this forward build. In the CGM 4-stage cycle, one stage is pure balance/commit (BU) and three stages change structure (CS, UNA, ONA). Under BU-Egress uniformity, the four stages have equal weight, so the egress time exponent is the 3/4 fraction of the fully thermalized ingress time exponent:

```text
a_eg := (3/4) * a_in(μ=1) = (3/4) * (1/4) = 3/16 = 0.1875
```

This value 3/16 is forced by the CGM stage structure. It is the predicted exponent for developmental times (gestation, weaning) that are dominated by network construction rather than network maintenance.

**Composite time relation.** A measured biological time `t` (such as longevity) that includes both developmental and maintenance contributions is a sum of two power laws:

```text
t(M) = A * M^(3/16) + B * M^(1/4)
```

The local log-log slope of this sum is:

```text
(d ln t) / (d ln M) = (3/16)(1 - ν) + (1/4)ν = 3/16 + ν/16
```

where `ν := B M^(1/4) / (A M^(3/16) + B M^(1/4))` is the maintenance-fraction of total time. This predicts that observed longevity slopes fall in the closed interval [3/16, 1/4] and drift upward with body mass and with adult-dominance fraction ν.

## 7. The West Organism Family

Every exponent in this section is an algebraic composition of the five-quantity channel basis of Section 4. Tier labels follow the definitions of Section 1.2. Bridge assumptions name the physiological identification required beyond channel arithmetic.

| Trait | Exponent | Composition | Tier | Bridge |
|---|---|---|---|---|
| Metabolic rate | 2/3 + μ/12 | `a(μ)` | A | Two-channel flux-fraction relation of Section 6.1 |
| Bulk-thermalized metabolic rate (maximal network delivery) | 3/4 | μ=1 endpoint | A | Full bulk thermalization at λ=1 |
| Mass-specific metabolic rate | -1/4 | `a_bulk - 1` | A | Same bridge as metabolic rate |
| Heart and respiratory rate | -1/4 | `-a_time` | A | Cardiac and respiratory periods identified with circulatory time |
| Circulatory period | +1/4 | `+a_time` | A | Characteristic transport time |
| Developmental time (gestation, weaning) | 3/16 | `a_eg` | A | Egress/construction time: 3/4 of thermalized ingress time |
| Longevity (composite) | [3/16, 1/4] | `3/16 + ν/16` | A | Sum of egress and ingress components; ν is maintenance fraction |
| Aortic length | 1/4 | `a_time` | A | Length scales with the characteristic transport time at fixed velocity class |
| Blood volume | 1 | volume isometry | A | Size-independent tissue density and blood-volume fraction |
| Capillary number | 3/4 | proportional to `B` | A | Invariant terminal-unit delivery capacity |
| Service radius | 1/12 | `a_time / n` | A | Three-dimensional service-volume geometry |
| Intercapillary spacing | +1/12 | `a_service` | A | Same service geometry |
| Inverse spacing | -1/12 | `-a_service` | A | Linear density dual of spacing |
| Volumetric capillary density | -1/4 | `a_bulk - 1` | A | `N_c / M` with `N_c ∝ B` and volume ∝ `M` |
| Lifetime energy per unit mass | 0 | `a_bulk + a_time - 1` | A | Lifespan identified with characteristic transport time |
| Lifetime heartbeat count | 0 | `(-a_time) + (+a_time)` | A | Same lifespan identification |
| Aortic radius | 3/8 | `a_bulk / 2` | B | Fixed `ΔP`, `u0` |
| Total vascular resistance | -3/4 | `-a_bulk` | B | Fixed `ΔP` |
| Pressure drop and terminal velocity | 0 | size-independent | B | WBE auxiliaries |

The two Tier B entries require, in addition to the Tier A channel basis, the WBE auxiliary assumption that the pressure drop across the vascular network and the terminal blood velocity are independent of organism size. Under that assumption the cross-sectional area of the aorta scales with metabolic rate, giving a radius scaling as the square root of `a_bulk`, and total resistance scales as the inverse of metabolic rate at fixed pressure drop. These two exponents are corollaries of the Tier A basis combined with an auxiliary hemodynamic assumption external to the CGM channel construction.

If capillary number scales as `N_c ∝ M^(3/4)` and tissue volume scales as `M`, volumetric capillary density scales as `N_c / M ∝ M^(-1/4)`. Intercapillary service radius scales as `M^(1/12)`, and inverse spacing as `M^(-1/12)`. Those are distinct geometric quantities.

The lifetime invariants follow from exponent arithmetic:

```text
B ∝ M^(a_bulk),      t ∝ M^(a_time)
B t / M ∝ M^(a_bulk + a_time − 1) = M^0
f ∝ M^(−a_time),     f t ∝ M^0
```

Metabolic throughput `B`, characteristic time `t` (adult maintenance lifespan under matched activity regime), body mass `M`, and heart rate `f` enter as labeled. The first line states that the product of field or regime-matched throughput and adult lifespan divided by mass is mass-independent. The second line states that the total number of heartbeats in a lifetime is mass-independent. Catalog tests that multiply basal metabolic rate by maximum longevity mix a low-μ rate with an egress-contaminated time and do not instantiate either identity.

## 8. Development and Life History as the Percolation Coverage Hierarchy

Organismal growth is the progressive coverage of an ancestry-preserving transport root. The hQVM carrier admits a hierarchy of five events of increasing structural demand on the same rank-six transport root, each turning on at a distinct inclusion fraction `p` of the byte generator alphabet. Define `parity(q) = popcount(q) mod 2`. Even-parity increments preserve shell parity, so full reachability from the shell-6 anchor requires odd-parity access as a structural condition on the generator set.

| Generation | Event | Structural condition |
|---|---|---|
| 0 | Horizon spanning | A single path connects the two constitutional horizons |
| 1 | Full reachability | The transport rank reaches six with odd-parity access, recovering all 4096 states |
| 2 | Defect-spectrum completion | All seven transport-defect weights are represented |
| 3 | Channel isotropy and exact rank threshold | The transport root is symmetrically populated |
| 4 | Holonomy transport | Depth-four closure words become available, completing the return path |

These five generations order developmental and life-history milestones. Horizon spanning is the initial supply connection. Full reachability is the closed circulatory orbit. Spectrum completion is metabolic flexibility across substrate and pathway types. Channel isotropy is adult homeostatic capacity. Holonomy transport is heritable return, reproduction and the reconstruction of ancestry in the next generation. The numerical thresholds `p_c(span) ≈ 0.022`, `p_c(full) ≈ 0.029`, and `p_c(word) ≈ 0.309` are properties of the hQVM carrier.

This developmental correspondence is a Tier C structural interpretation (Section 1.2). A direct falsification handle is the timing of spectrum completion. Metabolic flexibility across substrate and pathway types is predicted to become available only after the transport root reaches full defect-spectrum coverage, and that onset should track developmental stage and body mass under a matched protocol.

The WBE derivation employs an asymptotic branching-depth limit. The hQVM percolation hierarchy provides a finite coverage ladder with separate thresholds for spanning, full reachability, spectrum completion, isotropy, and return closure. The bulk-channel regime and the return-closure regime occur at distinct finite coverage thresholds, each computable from the percolation hierarchy.

## 9. Chemical Activation Energy on the Delta-Ruler

This section proposes a thermal reading of the aperture gap on a source-rooted delivery channel. The continuum aperture `Delta` sets an activation barrier relative to thermal energy. The thermal attempt rate `kT/h`, weighted by `Delta`, sets an attempt frequency. Their product is an elementary power quantum independent of `Delta`. The horizon multiplicity `|H|` counts available source-root channels. The biological identifications that convert the resulting power into sustained capillary output remain Tier C relative to a full reaction-rate derivation.

### 9.1 The Activation Energy

The continuum aperture gap `Delta` sets a dimensionless ratio between an activation barrier and thermal energy:

```text
E_a := kT / (2 Delta)
```

`E_a` is the activation barrier scale; `k` is Boltzmann's constant, `T` absolute temperature, and `Delta = 0.020699553913` the continuum aperture gap. At mammalian core temperature `T = 310` kelvin, `kT = 0.026714` electron volts, giving

```text
E_a = 0.026714 / (2 × 0.020700) = 0.645 electron volts
```

This lies inside the aerobic activation-energy band of 0.6 to 0.7 electron volts established independently by the Metabolic Theory of Ecology from cross-taxon respiration-rate data [10, 11]. The dual-pole holonomy defect `delta_BU` measures the geometric phase accumulated by a closed loop traversing both constitutional poles of the carrier. A chemical reaction crossing a transition-state barrier is a single-direction process, one pole of the dual loop, so the factor of two in `E_a/(kT) = 1/(2 Delta)` matches the factor that separates dual-pole from single-pass holonomy in the CGM gravitational construction.

### 9.2 Placement on the Shared Energy Ruler

The Delta-ruler places any energy `E` at a tick coordinate

```text
n_tick(E) = log2(v/E) / Delta
```

The tick coordinate `n_tick` places energy `E` relative to electroweak vacuum expectation value `v = 246.22` giga-electron-volts at the origin, with tick spacing set by aperture gap `Delta`. The chemical activation energy `E_a` occupies a specific coordinate on this same ruler that also carries the electroweak masses, the strong-interaction bare scale, and the Th-229m nuclear isomer excitation [15]. Metabolic chemistry is therefore placed on the same logarithmic energy coordinate as nuclear and electroweak observables.

### 9.3 Attempt Frequency and the Elementary Power Quantum

Standard transition-state theory gives an attempt frequency `kT/h` for barrier crossing, independent of barrier height. Depth-four balance requires a bounded oscillation about the closed configuration, and the aperture gap `Delta` is the amplitude of this residual vibration. The attempt frequency for barrier crossing scales as the fraction `Delta` of the thermal attempt rate:

```text
f_attempt := (kT/h) × Delta
```

`f_attempt` is the sustained closure-frequency scale, from Boltzmann constant `k`, Planck's constant `h`, absolute temperature `T`, and aperture gap `Delta`. The single-event power quantum is the product of barrier and attempt frequency,

```text
P_terminal := E_a × f_attempt = (kT)^2 / (2h)
```

`P_terminal` is the elementary power per barrier-crossing event. The aperture `Delta` cancels, so the power quantum is fixed by thermal energy and Planck's constant alone. Multiplying by the horizon size `|H| = 64`,

```text
B0_micro := |H| × P_terminal = 8.847 × 10^{-7} watts
```

at `T = 310` K. `B0_micro` is the elementary terminal power quantum; `|H| = 64` is the fixed boundary multiplicity of the hQVM carrier.

In this construction, delivery is counted on a source-connected boundary root, whose cardinality is `|H|`. The same product geometry of Section 3.1 that separates bulk tissue capacity from root delivery forces `a_SR = 1/2` in Section 4.1.

The numerical range `10^{-8}` to `10^{-7}` watts falls in the range estimated for the metabolic power supported by a single capillary bed in mammalian tissue, obtained by dividing total basal metabolic power by total capillary count. The identification of `|H| = 64` with the elementary delivery-unit multiplicity remains part of the Tier C mapping of this section.

The macroscopic Kleiber prefactor `K` in `B = K M^{3/4}` factorizes into the terminal power quantum `B0_micro` and a terminal-density prefactor. The power quantum is fixed by the carrier geometry and the core temperature. The density prefactor remains a measured empirical quantity governed by macroscopic tissue volume, treated in Section 10.

## 10. The Absolute Mass Origin

The hQVM carrier fixes a dimensionless intercept for the Kleiber relation through the shell moment `M_shell = d × 2^(d-1) = 192` at `d = 6`:

```text
log2(M0/u) = M_shell / 2 = 96
```

`M0` is a reference mass scale in external mass unit `u`, with shell moment `M_shell = 192` at chirality dimension six. The exponent 96 and the associated logarithmic offset `b_K = -47/48` (a carrier boundary-projector invariant) are fixed quantities independent of any unit convention. The numerical value of `M0` in kilograms depends on the choice of `u`: with `u` as one atomic mass unit, `M0 ≈ 132` kg; with `u` as one electron mass, `M0 ≈ 0.072` kg. The four-order-of-magnitude difference arises entirely from the unit convention.

The macroscopic Kleiber prefactor `K` in `B = K M^{3/4}` (with `B` in watts and `M` in kilograms) remains a measured empirical quantity. PanTHERIA, AnAge, and AnimalTraits Mammalia give `K` in the range 2.9 to 3.4 watts per kilogram to the three-quarter power, with logarithmic intercepts agreeing to within a quarter of one Delta-ruler octave across the three catalogs (Section D3 of `hqvm_cgm_allometry_results.txt`).

## 11. Conservation Constraints Distinguished from Transport Channels

Population density scaling is a conservation constraint on ecosystem resource allocation, distinct from the transport exponents of the channel basis in Section 4, and carries two distinct nulls depending on which resource-demand model is assumed. If per-capita resource requirement is proportional to body mass, population density at fixed total resource supply scales as `M^{-1}`. If per-capita resource requirement is proportional to metabolic rate, following the Damuth relation [12], population density scales as `M^{-a_bulk} = M^{-3/4}`.

On the PanTHERIA mammalian dataset, ordinary least squares regression, treating mass as the covariate under a directed error model, recovers a slope of `-0.741`, near the Damuth conservation null. Reduced major axis regression on the same data recovers a slope of `-0.980`, near the pure mass-proportional conservation null. Because the two estimators encode different assumptions about measurement error in mass and density, the primary estimator for a directed ecological constraint with mass as explanatory covariate is ordinary least squares. Phylogenetic generalized least squares is outside the scope of the present catalogs, so cross-species residuals retain shared-descent correlation.

Home range area is likewise a conservation-adjacent trait, outside the Tier A transport basis, and carries two distinct nulls. Under a metabolic demand model, in which an organism ranges over an area proportional to its resource requirement and resource requirement is proportional to metabolic rate, home range scales as `M^{3/4}`. Under a mass-proportional demand model, in which resource requirement is proportional to body mass directly, home range scales as `M^1`. The PanTHERIA ordinary least squares estimate of `1.06` sits nearer the mass-proportional conservation null than the metabolic null. This is reported as a dual-null conservation trait, matching the treatment of population density. Phylogenetic correction is likewise absent for this series.

## 12. Brain Mass

Brain mass scaling against body mass, pooled across mammalian orders, gives an ordinary least squares slope of `0.877` on the AnimalTraits catalog. This pooled value is inflated by grade shifts, systematic offsets in the brain-to-body relation between taxonomic orders that produce an artificially steep slope when data from multiple orders are combined in one regression, a well-documented effect in comparative brain-mass studies. Within single orders the slope varies: Rodentia gives `0.668`, Carnivora gives `0.623`, and Primates gives `0.887`. The metabolic interval `[2/3, 3/4]` of Section 6 applies to metabolic throughput under the two-channel model. Brain mass is a morphological trait. Placement of brain-mass scaling inside that interval requires a separate derivation. The pooled cross-order slope is reported as a diagnostic of taxonomic heterogeneity. Within-order slopes are the relevant morphological comparison. Phylogenetic generalized least squares is outside the present catalogs.

## 13. Cities and Companies: A Structural Hypothesis

City and corporate infrastructure scaling exponents follow from a hypothesis distinct from the organism channel basis of Section 4, and are reported at Tier C throughout this analysis.

An organism is a coherent volume embedded in three-dimensional Euclidean space. Its delivery network must serve that volume, so the network exponent closes at `a_bulk = 3/4` under the channel basis of Section 4. Cities and companies are socioeconomic information networks. Under the Tier C hypothesis they route exchange on the six-dimensional chirality-transport register directly, so the effective network dimension is the relational register dimension.

For a near-minimal Euclidean network connecting `N` terminals embedded in an effective dimension `d_net`, standard results for minimum spanning trees, traveling-salesman tours, and related subadditive network functionals give a total network length scaling as `N^{(d_net - 1)/d_net}`. Under the Tier C identification of `d_net` with the chirality-transport register dimension `d = 6`,

```text
a_infra = (d-1)/d = 5/6,  a_socio = 1 + 1/d = 7/6,  a_infra + a_socio = 2
```

Infrastructure exponent `a_infra` and socioeconomic exponent `a_socio` form a conjugate pair summing to 2. If infrastructure exponents across multiple cities remain consistent with `d_net = 2` or `d_net = 3` under a matched definition of terminals and a matched metric for infrastructure length, the identification `d_net = 6` fails.

The infrastructure exponent `5/6` is numerically close to commonly cited empirical urban infrastructure exponents near `0.85` [14]. This numerical proximity is reported as a motivation for the hypothesis. Corporate scaling uses the same sublinear exponent `5/6` together with the word-regime confinement fraction `128/4096 = 1/32`, the fraction of reachable carrier states accessible under depth-four closure operators alone.

City wage and road-length series in `data/catalogs/allometry/` are reported as external checks on this hypothesis, using reduced major axis regression as the primary estimator, appropriate for a conjugate pair of exponents summing to a fixed target. These checks are reported in Section 14 at the same procedural standard as the organism catalogs, and stand apart from the Tier A organism results in any pass or fail summary.

## 14. Empirical Protocol

Directed organism traits, in which body mass is the explanatory covariate and a physiological trait is the response, use ordinary least squares as the primary estimator, with reduced major axis regression reported alongside for comparison. City conjugacy checks, in which two exponents are compared against a fixed target sum, use reduced major axis regression as primary. Bootstrap confidence intervals, Akaike information criterion model comparison among fixed-exponent, free-exponent, and mu-family models, and label-shuffle null controls accompany every fit reported in `hqvm_cgm_allometry_results.txt`. Species-tree phylogenetic correlation structure is outside the scope of this analysis, so every cross-species slope inherits residual shared-descent correlation.

| Series | n | OLS | RMA | Nearest reference |
|---|---|---|---|---|
| PanTHERIA basal metabolic rate | 573 | 0.717 | 0.743 | mu-band (resting; mu≈0.60) |
| PanTHERIA mass-specific rate | 573 | -0.283 | -0.344 | -1/4 (endpoint; follows a_B−1) |
| PanTHERIA longevity | 1000 | 0.198 | 0.274 | composite [3/16, 1/4] |
| PanTHERIA gestation | 1335 | 0.189 | 0.280 | egress 3/16 |
| PanTHERIA population density | 947 | -0.741 | -0.980 | Damuth -3/4 and conservation -1 |
| PanTHERIA home range | 700 | 1.061 | 1.288 | conservation dual nulls 3/4 and 1 |
| AnimalTraits Mammalia metabolic rate | 177 | 0.671 | 0.703 | metabolic mu-band, CI contains 2/3 |
| AnimalTraits pooled brain mass | 1639 | 0.877 | 0.901 | morphological, outside metabolic band |
| AnimalTraits brain mass, Rodentia | 83 | 0.668 | 0.693 | morphological within-order |
| AnAge metabolic rate | 627 | 0.713 | 0.788 | mu-band (resting; mu≈0.55) |
| AnAge longevity | 3210 | 0.138 | 0.242 | max longevity; egress-failure below 3/16 |
| City wages | 382 | 1.101 | 1.122 | Tier C socioeconomic 7/6 |
| City road length | 312 | 0.959 | 1.051 | Tier C infrastructure 5/6 |

The AnimalTraits pooled metabolic series across all classes gives an ordinary least squares slope of `1.03`, outside the metabolic band of `[2/3, 3/4]` under the two-channel model of Section 6. This pooled series aggregates basal, field, and maximal rates across taxa under a single regression; the high-mass end is dominated by elevated activity regimes (`μ > 1` in the Section 6 coordinate), so the pooled slope is a mixed-regime diagnostic rather than a Tier A basal-metabolic test. Phylogenetic correction is absent for all series in this table.

Basal metabolic series are classified against the μ-band `[2/3, 3/4]`, not against the μ=1 endpoint `3/4`. Maximum-longevity catalogs that fall below `3/16` are classified as egress-failure mixtures (infant and early mortality pulling the ordinary least squares slope below the adult maintenance interval), not as failures of the composite longevity null. The Tier A lifetime energy sum rule of Section 7 identifies field throughput with adult maintenance time; pairing basal rate with maximum longevity is a mixed-regime proxy and is not scored as a Tier A catalog test. The heartbeat sum rule `f t ∝ M^0` is the matched catalog form when heart-rate series are available.

## 15. Falsifiable Predictions

**P1. Ordering of resting and active metabolic exponents.** Basal or resting metabolic rate measurements, corresponding to low `μ` under the identification of Section 6.2, place systematically closer to the surface endpoint `2/3` than field or active metabolic rate measurements, corresponding to high `μ`. After regime separation (activity state and taxon), mammalian metabolic series are predicted to fall inside the closed interval `[2/3, 3/4]` under the two-channel model of Section 6.

**P2. Temperature dependence of metabolic rate at fixed mass.** The chemical activation scale `E_a` defined by `E_a/(kT) = 1/(2 Delta)` sets a barrier ratio for single-direction crossing. Temperature-dependent metabolic-rate series at fixed mass are predicted to cluster around a single dimensionless barrier ratio `1/(2 Delta)` when fitted in Arrhenius form.

**P3. Ontogenetic curvature.** Within a single species' growth series, if the network flux fraction `μ(M, x)` varies across developmental stage, the instantaneous log-log slope of metabolic rate against mass shows measurable curvature, remaining bounded within `[2/3, 3/4]` at every point along the series.

**P3b. Developmental time exponent.** Mammalian gestation and weaning periods are predicted to scale with body mass with exponent `3/16 = 0.1875`, the egress/construction time exponent forced by the CGM 4-stage cycle. Mammalian adult longevity is predicted to scale with an exponent in the interval `[3/16, 1/4]`, reflecting a mixture of developmental and maintenance times. Maximum-longevity catalogs that include early mortality are predicted to fall at or below the lower endpoint `3/16` as egress-failure mixtures.

**P4. Chemical activation-energy clustering.** Cross-taxon compilations of metabolic activation energies from temperature-dependence studies should cluster near the value `E_a ≈ 0.645` electron volts derived in Section 9.1 from the aperture gap and core body temperature alone.

**P5. Replication of the Damuth dual-null pattern.** Independent mammalian trait compilations should reproduce the dual-null pattern of Section 11 under a matched error model: ordinary least squares of population density against mass near the Damuth exponent `-3/4`, and reduced major axis near the pure conservation exponent `-1`, with phylogenetic structure acknowledged as uncorrected in the present catalogs.

**P6. Lifetime invariants within homogeneous metabolic classes.** Within a taxonomically and physiologically homogeneous class, such as endothermic mammals under a matched activity regime, the product of regime-matched metabolic rate and adult lifespan divided by mass, and the product of heart rate and lifespan, remain independent of body mass, following the two channel-basis sum rules of Section 7.

## 16. Falsification Criteria

This analysis is falsified by any of the following outcomes. A regime-controlled mammalian metabolic exponent falling outside the closed interval `[2/3, 3/4]` after activity state and taxon are properly separated under the two-channel model. A measured activation energy for aerobic metabolic rate lying substantially outside the 0.5 to 0.8 electron-volt band implied by the aperture gap at physiological temperature. Failure of the two lifetime sum rules of Section 7 within a homogeneous metabolic class beyond the scatter expected from measurement uncertainty. Failure of any of the three chirality-dimension consistency relations of Section 5 under independent recomputation of the hQVM(d) family falsifies the structural channel basis used in Sections 3 through 7. Systematic failure of the ordinary least squares versus reduced major axis dual-null pattern for the Damuth conservation trait under independent trait compilations and a matched error model.

## 17. Conclusion

Allometric scaling in mammals is explained here as a consequence of the Common Governance Model (CGM): coherent systems preserve ancestry, meaning that distinguishable states remain traceable to a common source under recursive transformation. Gravity is the continuum balance readout of this requirement under displacement. An organism is a bounded source-to-bulk transport realization of the same closure. The dimensional and thermal structure used in the channel basis is fixed at chirality dimension `d = 6` (six degrees of freedom) [15].

The source-accessibility exponent evaluates to `1/2` on the hQVM carrier under the Square-Root Cluster Theorem, to `2/3` under three-dimensional spatial isometry, and to `3/4` under the West-Brown-Enquist network assumptions with `D_eff = ⟨N⟩ = 3` at chirality dimension six. The elementary metabolic power quantum is `B0_micro = |H| (kT)^2/(2h)`, in which the aperture cancels. The QuBEC thermal shell mean fixes the network delivery exponent at physical chirality dimension six. Organismal scaling inherits that six-degree-of-freedom kinematic freedom and is cross-checked by three dimension-six consistency relations on the family of carriers `hQVM(d)`. A continuous flux-fraction parameter `μ(M, x)` interpolates between the surface and network endpoints at fixed dimensionality and is identified, as a modeling step, with the occupation measure of the same thermal register that governs the aperture gap.

Development is a Tier C structural reading of progressive coverage of the ancestry-preserving transport network. City and corporate scaling follow from a Tier C hypothesis in which socioeconomic networks operate on the six-dimensional relational register. The organism-level channel arithmetic is closed once the biological bridge identifications of Sections 4 and 7 are adopted. Measured Kleiber prefactors, body temperature, and physiological state controls remain external inputs.

## References

1. B. Korompilias, *Common Governance Model: Mathematical Physics Framework*, Zenodo (2025), https://doi.org/10.5281/zenodo.17521384. Repository: https://github.com/gyrogovernance/science.
2. M. Kleiber, *Body size and metabolism*, Hilgardia 6, 315-353 (1932).
3. M. Rubner, *Ueber den Einfluss der Koerpergroesse auf Stoff- und Kraftwechsel*, Zeitschrift fur Biologie 19, 535-562 (1883).
4. G.B. West, J.H. Brown, and B.J. Enquist, *A general model for the origin of allometric scaling laws in biology*, Science 276, 122-126 (1997).
5. G.B. West, *Scale: The Universal Laws of Growth, Innovation, Sustainability, and the Pace of Life in Organisms, Cities, Economies, and Companies*, Penguin Press (2017).
6. C.P. White and R.S. Seymour, *Mammalian basal metabolic rate is proportional to body mass^(2/3)*, Proceedings of the National Academy of Sciences 100, 4046-4049 (2003).
7. P.S. Dodds, D.H. Rothman, and J.S. Weitz, *Re-examination of the "3/4-law" of metabolism*, Journal of Theoretical Biology 209, 9-27 (2001).
8. D.S. Glazier, *Beyond the "3/4-power law": variation in the intra- and interspecific scaling of metabolic rate in animals*, Biological Reviews 80, 611-662 (2005).
9. D.S. Glazier, *A unifying explanation for diverse metabolic scaling in animals and plants*, Biological Reviews 85, 111-138 (2010).
10. J.F. Gillooly, J.H. Brown, G.B. West, V.M. Savage, and E.L. Charnov, *Effects of size and temperature on metabolic rate*, Science 293, 2248-2251 (2001).
11. J.H. Brown, J.F. Gillooly, A.P. Allen, V.M. Savage, and G.B. West, *Toward a metabolic theory of ecology*, Ecology 85, 1771-1789 (2004).
12. J. Damuth, *Population density and body size in mammals*, Nature 290, 699-700 (1981).
13. J.R. Speakman, *Body size, energy metabolism and lifespan*, Journal of Experimental Biology 208, 1717-1730 (2005).
14. L.M.A. Bettencourt, J. Lobo, D. Helbing, C. Kuhnert, and G.B. West, *Growth, innovation, scaling, and the pace of life in cities*, Proceedings of the National Academy of Sciences 104, 7301-7306 (2007).
15. B. Korompilias, CGM companion analyses (repository paths): `docs/CGM_Logic.md`, `docs/CGM_Program.md`, `docs/Findings/Analysis_3D_6DOF_Proof.md`, `docs/Findings/Analysis_CGM_Units.md`, `docs/Findings/Analysis_hQVM_Percolation.md`, `docs/Findings/Analysis_hQVM_CGM_Trestleboard.md`, `docs/Findings/Analysis_Gravity.md`, `docs/Findings/Analysis_Gravity_Note.md`; kernel specifications: `docs/Gyroscopic_Computational_Theory/hQVM_Specs_Formalism.md`, `docs/Gyroscopic_Computational_Theory/hQVM_QuBEC_Theory.md`, `docs/Gyroscopic_Computational_Theory/hQVM_SDK_Quantum_Computing.md`. https://github.com/gyrogovernance/science (2025).
16. Data catalogs: `data/catalogs/allometry/` (source files in the catalog directory).

## Appendix A. Reproducibility

`experiments/hqvm_cgm_allometry_1.py` computes the closed channel basis, the three dimension-six consistency relations, the QuBEC excitation sweep, the chemical clock, and the Kleiber intercept construction at fixed chirality dimension six, using the family of carriers `hQVM(d)` for chirality dimensions one through eight as the consistency cross-check of Section 5. `experiments/hqvm_cgm_allometry_2.py` audits the external catalogs of Section 14 under ordinary least squares and reduced major axis regression, including the brain-mass taxonomic split of Section 12 and the Damuth density dual-null pattern of Section 11, with bootstrap confidence intervals, Akaike information criterion model comparison, and label-shuffle null controls. `experiments/hqvm_cgm_allometry_3.py` checks fiber-complete alphabet composition on the hQVM carrier within the structured families of the percolation Square-Root Cluster Theorem (weight shells, even/odd transport), records the single-q fiber scope boundary, and measures the QuBEC quotient census. Combined output: `experiments/hqvm_cgm_allometry_results.txt`. Run with `python experiments/hqvm_cgm_allometry_run.py`.