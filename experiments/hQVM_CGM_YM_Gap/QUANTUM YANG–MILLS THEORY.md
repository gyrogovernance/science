# QUANTUM YANG–MILLS THEORY
**Arthur Jaffe and Edward Witten**

## 1. The Physics of Gauge Theory

Since the early part of the 20th century, it has been understood that the description of nature at the subatomic scale requires quantum mechanics. In quantum mechanics, the position and velocity of a particle are noncommuting operators acting on a Hilbert space, and classical notions such as “the trajectory of a particle” do not apply.

But quantum mechanics of particles is not the whole story. In 19th and early 20th century physics, many aspects of nature were described in terms of fields—the electric and magnetic fields that enter in Maxwell’s equations, and the gravitational field governed by Einstein’s equations. Since fields interact with particles, it became clear by the late 1920s that an internally coherent account of nature must incorporate quantum concepts for fields as well as for particles.

After doing this, quantities such as the components of the electric field at different points in space-time become non-commuting operators. When one attempts to construct a Hilbert space on which these operators act, one finds many surprises. The distinction between fields and particles breaks down, since the Hilbert space of a quantum field is constructed in terms of particle-like excitations. Conventional particles, such as electrons, are reinterpreted as states of the quantized field. In the process, one finds the prediction of “antimatter”; for every particle, there must be a corresponding antiparticle, with the same mass and opposite electric charge. Soon after P.A.M. Dirac predicted this on the basis of quantum field theory, the “positron” or oppositely charged antiparticle of the electron was discovered in cosmic rays.

The most important Quantum Field Theories (QFTs) for describing elementary particle physics are gauge theories. The classical example of a gauge theory is Maxwell’s theory of electromagnetism. For electromagnetism the gauge symmetry group is the abelian group U(1). If A denotes the U(1) gauge connection, locally a one-form on space-time, then the curvature or electromagnetic field tensor is the two-form F = dA, and Maxwell’s equations in the absence of charges and currents read 0 = dF = d∗F. Here ∗ denotes the Hodge duality operator; indeed, Hodge introduced his celebrated theory of harmonic forms as a generalization of the solutions to Maxwell’s equations. Maxwell’s equations describe large-scale electric and magnetic fields and also—as Maxwell discovered—the propagation of light waves, at a characteristic velocity, the speed of light.

The idea of a gauge theory evolved from the work of Hermann Weyl. One can find in [34] an interesting discussion of the history of gauge symmetry and the discovery of Yang–Mills theory [50], also known as “non-abelian gauge theory.” At the classical level one replaces the gauge group U(1) of electromagnetism by a compact gauge group G. The definition of the curvature arising from the connection must be modified to F = dA + A ∧ A, and Maxwell’s equations are replaced by the Yang–Mills equations, 0 = d_AF = d_A ∗ F, where d_A is the gauge-covariant extension of the exterior derivative.

These classical equations can be derived as variational equations from the Yang–Mills Lagrangian

> **(1)** ℒ = (1 / 4g²) Tr F ∧ ∗F,

where Tr denotes an invariant quadratic form on the Lie algebra of G. The Yang–Mills equations are nonlinear—in contrast to the Maxwell equations. Like the Einstein equations for the gravitational field, only a few exact solutions of the classical equation are known. But the Yang–Mills equations have certain properties in common with the Maxwell equations: In particular they provide the classical description of massless waves that travel at the speed of light.

By the 1950s, when Yang–Mills theory was discovered, it was already known that the quantum version of Maxwell theory—known as Quantum Electrodynamics or QED—gives an extremely accurate account of electromagnetic fields and forces. In fact, QED improved the accuracy for certain earlier quantum theory predictions by several orders of magnitude, as well as predicting new splittings of energy levels. So it was natural to inquire whether non-abelian gauge theory described other forces in nature, notably the weak force (responsible among other things for certain forms of radioactivity) and the strong or nuclear force (responsible among other things for the binding of protons and neutrons into nuclei). The massless nature of classical Yang–Mills waves was a serious obstacle to applying Yang–Mills theory to the other forces, for the weak and nuclear forces are short range and many of the particles are massive. Hence these phenomena did not appear to be associated with long-range fields describing massless particles.

In the 1960s and 1970s, physicists overcame these obstacles to the physical interpretation of non-abelian gauge theory. In the case of the weak force, this was accomplished by the Glashow–Salam–Weinberg electroweak theory [47, 40] with gauge group H = SU(2) × U(1). By elaborating the theory with an additional “Higgs field,” one avoided the massless nature of classical Yang–Mills waves. The Higgs field transforms in a two-dimensional representation of H; its non-zero and approximately constant value in the vacuum state reduces the structure group from H to a U(1) subgroup (diagonally embedded in SU(2) × U(1)). This theory describes both the electromagnetic and weak forces, in a more or less unified way; because of the reduction of the structure group to U(1), the long-range fields are those of electromagnetism only, in accord with what we see in nature.

The solution to the problem of massless Yang–Mills fields for the strong interactions has a completely different nature. That solution did not come from adding fields to Yang–Mills theory, but by discovering a remarkable property of the quantum Yang–Mills theory itself, that is, of the quantum theory whose classical Lagrangian has been given in (1). This property is called “asymptotic freedom” [21, 38]. Roughly this means that at short distances the field displays quantum behavior very similar to its classical behavior; yet at long distances the classical theory is no longer a good guide to the quantum behavior of the field.

Asymptotic freedom, together with other experimental and theoretical discoveries made in the 1960s and 1970s, made it possible to describe the nuclear force by a non-abelian gauge theory in which the gauge group is G = SU(3). The additional fields describe, at the classical level, “quarks,” which are spin 1/2 objects somewhat analogous to the electron, but transforming in the fundamental representation of SU(3). The non-abelian gauge theory of the strong force is called Quantum Chromodynamics (QCD).

The use of QCD to describe the strong force was motivated by a whole series of experimental and theoretical discoveries made in the 1960s and 1970s, involving the symmetries and high-energy behavior of the strong interactions. But classical non-abelian gauge theory is very different from the observed world of strong interactions; for QCD to describe the strong force successfully, it must have at the quantum level the following three properties, each of which is dramatically different from the behavior of the classical theory:

1. It must have a “mass gap;” namely there must be some constant Δ > 0 such that every excitation of the vacuum has energy at least Δ.
2. It must have “quark confinement,” that is, even though the theory is described in terms of elementary fields, such as the quark fields, that transform non-trivially under SU(3), the physical particle states—such as the proton, neutron, and pion—are SU(3)-invariant.
3. It must have “chiral symmetry breaking,” which means that the vacuum is potentially invariant (in the limit, that the quark-bare masses vanish) only under a certain subgroup of the full symmetry group that acts on the quark fields.

The first point is necessary to explain why the nuclear force is strong but short ranged; the second is needed to explain why we never see individual quarks; and the third is needed to account for the “current algebra” theory of soft pions that was developed in the 1960s.

Both experiment—since QCD has numerous successes in confrontation with experiment—and computer simulations, see for example [8], carried out since the late 1970s, have given strong encouragement that QCD does have the properties cited above. These properties can be seen, to some extent, in theoretical calculations carried out in a variety of highly oversimplified models (like strongly coupled lattice gauge theory, see, for example, [48]). But they are not fully understood theoretically; there does not exist a convincing, whether or not mathematically complete, theoretical computation demonstrating any of the three properties in QCD, as opposed to a severely simplified truncation of it.

## 2. Quest for Mathematical Understanding

In surveying the physics of gauge theories in the last section, we considered both classical properties—such as the Higgs mechanism for the electroweak theory—and quantum properties that do not have classical analogs—like the mass gap and confinement for QCD. Classical properties of gauge theory are within the reach of established mathematical methods, and indeed classical non-abelian gauge theory has played a very important role in mathematics in the last twenty years, especially in the study of three- and four-dimensional manifolds. On the other hand, one does not yet have a mathematically complete example of a quantum gauge theory in four-dimensional space-time, nor even a precise definition of quantum gauge theory in four dimensions. Will this change in the 21st century? We hope so!

At times, mathematical structures of importance have first appeared in physics before their mathematical importance was fully recognized. This happened with the discovery of calculus, which was needed to develop Newtonian mechanics, with functional analysis and group representation theory, topics whose importance became clearer with quantum mechanics, and even with the study of Riemannian geometry, whose development was greatly intensified once it became clear, through Einstein’s invention of General Relativity to describe gravity, that this subject plays a role in the description of nature. These areas of mathematics became generally accessible only after a considerable time, over which the ideas were digested, simplified, and integrated into the general mathematical culture.

Quantum Field Theory (QFT) became increasingly central in physics throughout the 20th century. There are reasons to believe that it may be important in 21st century mathematics. Indeed, many mathematical subjects that have been actively studied in the last few decades appear to have natural formulations—at least at a heuristic level—in terms of QFT. New structures spanning probability, analysis, algebra, and geometry have emerged, for which a general mathematical framework is still in its infancy.

On the analytic side, a byproduct of the existence proofs and mathematical construction of certain quantum field theories was the construction of new sorts of measures, in particular non-Gaussian, Euclidean-invariant measures on spaces of generalized functionals. Dirac fields and gauge fields require measures on spaces of functions taking values in a Grassmann algebra and on spaces of functions into other target geometries.

Renormalization theory arises from the physics of quantum field theory and provides a basis for the mathematical investigation of local singularities (ultraviolet regularity) and of global decay (infra-red regularity) in quantum field theories. Asymptotic freedom ensures a decisive regularity in the case when classical Sobolev inequalities are borderline. Surprisingly, the ideas from renormalization theory also apply in other areas of mathematics, including classic work on the convergence of Fourier series and recent progress on classical dynamical systems.

On the algebraic side, investigations of soluble models of quantum field theory and statistical mechanics have led to many new discoveries involving topics such as Yang–Baxter equations, quantum groups, Bose–Fermi equivalence in two dimensions, and rational conformal field theory.

Geometry abounds with new mathematical structures rooted in quantum field theory, many of them actively studied in the last twenty years. Examples include Donaldson theory of 4-manifolds, the Jones polynomial of knots and its generalizations, mirror symmetry of complex manifolds, elliptic cohomology, and SL(2, ℤ) symmetry in the theory of affine Kac–Moody algebras.

QFT has in certain cases suggested new perspectives on mathematical problems, while in other cases its mathematical value up to the present time is motivational. In the case of the geometric examples cited above, a mathematical definition of the relevant QFTs (or one in which the relevant physical techniques can be justified) is not yet at hand. Existence theorems that put QFTs on a solid mathematical footing are needed to make the geometrical applications of QFT into a full-fledged part of mathematics. Regardless of the future role of QFT in pure mathematics, it is a great challenge for mathematicians to understand the physical principles that have been so important and productive throughout the twentieth century.

Finally, QFT is the jumping-off point for a quest that may prove central in 21st century physics—the effort to unify gravity and quantum mechanics, perhaps in string theory. For mathematicians to participate in this quest, or even to understand the possible results, QFT must be developed further as a branch of mathematics. It is important not only to understand the solution of specific problems arising from physics, but also to set such results within a new mathematical framework. One hopes that this framework will provide a unified development of several fields of mathematics and physics, and that it will also provide an arena for the development of new mathematics and physics.

For these reasons the Scientific Advisory Board of CMI has chosen a Millennium problem about quantum gauge theories. Solution of the problem requires both understanding one of the deep unsolved physics mysteries, the existence of a mass gap, and also producing a mathematically complete example of quantum gauge field theory in four-dimensional space-time.

## 3. Quantum Fields

A quantum field, or local quantum field operator, is an operator-valued generalized function on space-time obeying certain axioms. The properties required of the quantum fields are described at a physical level of precision in many textbooks, see, for example, [27]. Gårding and Wightman gave mathematically precise axioms for quantum field theories on ℝ⁴ with a Minkowski signature, see [45], and Haag and Kastler introduced a related scheme for local functions of the field, see [24]. Basically, one requires that the Hilbert space ℋ of the quantum field carry a representation of the Poincaré group (or inhomogeneous Lorentz group). The Hamiltonian H and momentum 𝐏 are the self-adjoint elements of the Lie algebra of the group that generate translations in time and space. A vacuum vector is an element of ℋ that is invariant under the (representation of the) Poincaré group. One assumes that the representation has positive energy, 0 ≤ H, and a vacuum vector Ω ∈ ℋ that is unique up to a phase. Gauge-invariant functions of the quantum fields also act as linear transformations on ℋ and transform covariantly under the Poincaré group. Quantum fields in space-time regions that cannot be connected by a light signal should be independent; Gårding and Wightman formulate independence as the commuting of the field operators (anti-commuting for two fermionic fields).

One of the achievements of 20th century axiomatic quantum field theory was the discovery of how to convert a Euclidean-invariant field theory on a Euclidean space-time to a Lorentz-invariant field theory on Minkowski space-time, and vice-versa. Wightman used positive energy to establish analytic continuation of expectations of Minkowski field theories to Euclidean space. Kurt Symanzik interpreted the Euclidean expectations as a statistical mechanical ensemble of classical Markov fields [46], with a probability density proportional to exp(−S), where S denotes the Euclidean action functional. E. Nelson reformulated Symanzik’s picture and showed that one can also construct a Hilbert space and a quantum-mechanical field from a Markov field [33]. Osterwalder and Schrader then discovered the elementary “reflection-positivity” condition to replace the Markov property. This gave rise to a general theory establishing equivalence between Lorentzian and Euclidean axiom schemes [35]. See also [13].

One hopes that the continued mathematical exploration of quantum field theory will lead to refinements of the axiom sets that have been in use up to now, perhaps to incorporate properties considered important by physicists such as the existence of an operator product expansion or of a local stress-energy tensor.

## 4. The Problem

To establish existence of four-dimensional quantum gauge theory with gauge group G, one should define a quantum field theory (in the above sense) with local quantum field operators in correspondence with the gauge-invariant local polynomials in the curvature F and its covariant derivatives, such as Tr Fᵢⱼ Fₖₗ(x).¹ Correlation functions of the quantum field operators should agree at short distances with the predictions of asymptotic freedom and perturbative renormalization theory, as described in textbooks. Those predictions include among other things the existence of a stress tensor and an operator product expansion, having prescribed local singularities predicted by asymptotic freedom.

Since the vacuum vector Ω is Poincaré invariant, it is an eigenstate with zero energy, namely HΩ = 0. The positive energy axiom asserts that in any quantum field theory, the spectrum of H is supported in the region [0, ∞). A quantum field theory has a mass gap Δ if H has no spectrum in the interval (0, Δ) for some Δ > 0. The supremum of such Δ is the mass m, and we require m < ∞.

**Yang–Mills Existence and Mass Gap.** Prove that for any compact simple gauge group G, a non-trivial quantum Yang–Mills theory exists on ℝ⁴ and has a mass gap Δ > 0. Existence includes establishing axiomatic properties at least as strong as those cited in [45, 35].

## 5. Comments

An important consequence of the existence of a mass gap is clustering: Let 𝐱 ∈ ℝ³ denote a point in space. We let H and 𝐏 denote the energy and momentum, generators of time and space translation. For any positive constant C < Δ and for any local quantum field operator O(𝐱) = e^(−i𝐏·𝐱) O e^(i𝐏·𝐱) such that ⟨Ω, OΩ⟩ = 0, one has

> **(2)** |⟨Ω, O(𝐱) O(𝐲) Ω⟩| ≤ exp(−C|𝐱 − 𝐲|),

as long as |𝐱 − 𝐲| is sufficiently large. Clustering is a locality property that, roughly speaking, may make it possible to apply mathematical results established on ℝ⁴ to any 4-manifold, as argued at a heuristic level (for a supersymmetric extension of four-dimensional gauge theory) in [49]. Thus the mass gap not only has a physical significance (as explained in the introduction), but it may also be important in mathematical applications of four-dimensional quantum gauge theories to geometry.

In addition the existence of a uniform gap for finite-volume approximations may play a fundamental role in the proof of existence of the infinite-volume limit.

There are many natural extensions of the Millennium problem. Among other things, one would like to prove the existence of an isolated one-particle state (an upper gap, in addition to the mass gap), to prove confinement, to prove existence of other four-dimensional gauge theories (incorporating additional fields that preserve asymptotic freedom), to understand dynamical questions (such as the possible mass gap, confinement, and chiral symmetry breaking) in these more general theories, and to extend the existence theorems from ℝ⁴ to an arbitrary 4-manifold.

But a solution of the existence and mass gap problem as stated above would be a turning point in the mathematical understanding of quantum field theory, with a chance of opening new horizons for its applications.

## 6. Mathematical Perspective

Wightman and others have questioned for approximately fifty years whether mathematically well-defined examples of relativistic, nonlinear quantum field theories exist. We now have a partial answer: Extensive results on the existence and physical properties of nonlinear QFTs have been proved through the emergence of the body of work known as “constructive quantum field theory” (CQFT).

The answers are partial, for in most of these field theories one replaces the Minkowski space-time M⁴ by a lower-dimensional space-time M² or M³, or by a compact approximation such as a torus. (Equivalently in the Euclidean formulation one replaces Euclidean space-time ℝ⁴ by ℝ² or ℝ³.) Some results are known for Yang–Mills theory on a 4-torus T⁴ approximating ℝ⁴, and, while the construction is not complete, there is ample indication that known methods could be extended to construct Yang–Mills theory on T⁴.

In fact, at present we do not know any non-trivial relativistic field theory that satisfies the Wightman (or any other reasonable) axioms in four dimensions. So even having a detailed mathematical construction of Yang–Mills theory on a compact space would represent a major breakthrough. Yet, even if this were accomplished, no present ideas point the direction to establish the existence of a mass gap that is uniform in the volume. Nor do present methods suggest how to obtain the existence of the infinite volume limit T⁴ → ℝ⁴.

### 6.1. Methods

Since the inception of quantum field theory, two central methods have emerged to show the existence of quantum fields on non-compact configuration space (such as Minkowski space). These known methods are:
1. Find an exact solution in closed form;
2. Solve a sequence of approximate problems, and establish convergence of these solutions to the desired limit.

Exact solutions may be available for nonlinear fields for special values of the coupling which yields extra symmetries or integrable models. They might be achieved after clever changes of variables. In the case of four-dimensional Yang–Mills theory, an exact solution appears unlikely, though there might some day be an asymptotic solution in a large N limit.

The second method is to use mathematical approximations to show the convergence of approximate solutions to exact solutions of the nonlinear problems, known as constructive quantum field theory, or CQFT. Two principle approaches—studying quantum theory on Hilbert space, and studying classical functional integrals—are related by the Osterwalder–Schrader construction. Establishing uniform a priori estimates is central to CQFT, and three schemes for establishing estimates are
(a) correlation inequalities,
(b) symmetries of the interaction,
(c) convergent expansions.

The correlation inequality methods have an advantage; they are general. But correlation inequalities rely on special properties of the interaction that often apply only for scalar bosons or abelian gauge theories. The use of symmetry also applies only to special values of the couplings and is generally combined with another method to obtain analytic control. In most known examples, perturbation series, i.e., power series in the coupling constant, are divergent expansions; even Borel and other resummation methods have limited applicability.

This led to development of expansion methods, generally known as cluster expansions. Each term in a cluster expansion sum depends on the coupling constants in a complicated fashion; they often arise as functional integrals. One requires sufficient quantitative knowledge of the properties of each term in an expansion to ensure the convergence of the sum and to establish its qualitative properties. Refined estimates yield the rate of exponential decay of Green’s functions, magnitude of masses, the existence of symmetry breaking (or its preservation), etc.

Over the past thirty years, a panoply of expansion methods have emerged as a central tool for establishing mathematical results in CQFT. In their various incarnations, these expansions encapsulate ideas of the asymptotic nature of perturbation theory, of space-time localization, of phase-space localization, of renormalization theory, of semi-classical approximations (including “non-perturbative” effects), and of symmetry breaking. One can find an introduction to many of these methods and references in [18], and a more recent review of results in [28]. These expansion methods can be complicated and the literature is huge, so we can only hope to introduce the reader to a few ideas; we apologize in advance for important omissions.

### 6.2. The First Examples: Scalar Fields

Since the 1940s the quantum Klein-Gordon field ϕ provided an example of a linear, scalar, mass-m field theory (arising from a quadratic potential). This field, and the related free spinor Dirac field, served as models for formulating the first axiom schemes in the 1950s [45].

Moments of a Euclidean-invariant, reflection-positive, ergodic, Borel measure dμ on the space S′(ℝᵈ) of tempered distributions may satisfy the Osterwalder–Schrader axioms. The Gaussian measure dμ with mean zero and covariance C = (−∆ + m₀²)⁻¹ yields the free, mass-m₀ field; but one requires non-Gaussian dμ to obtain nonlinear fields. (For the Gaussian measure, reflection positivity is equivalent to positivity of the transformation ΘC, restricted to L²(ℝᵈ₊) ⊂ L²(ℝᵈ). Here Θ : t → −t denotes the time-reflection operator, and ℝᵈ₊ = {(t, 𝐱) : t ≥ 0} is the positive-time subspace.)

The first proof that nonlinear fields satisfy the Wightman axioms and the first construction of such non-Gaussian measures only emerged in the 1970s. The initial examples comprised fields with small, polynomial nonlinearities on ℝ²: first in finite volume, and then in the infinite volume limit [15, 19, 22]. These field theories obey the Wightman axioms (as well as all other axiomatic formulations), the fields describe particles of a definite mass, and the fields produce multi-particle states with non-trivial scattering [19]. The scattering matrix can be expanded as an asymptotic series in the coupling constants, and the results agree term-by-term with the standard description of scattering in perturbation theory that one finds in physics texts [37].

A quartic Wightman QFT on ℝ³ also exists, obtained by constructing a remarkable non-Gaussian measure dμ on S′(ℝ³) [16, 10]. This merits further study.

We now focus on some properties of the simplest perturbation to the action density of the free field, namely, the even quartic polynomial

> **(3)** λϕ⁴ + ½σϕ² + c.

The coefficients 0 < λ and σ, c ∈ ℝ are real parameters, all zero for the free field. For 0 < λ ≪ 1, one can choose σ(λ), c(λ) so the field theory described by (3) exists, is unique, and has a mass equal to m such that |m − m₀| is small.

Because of the local singularity of the nonlinear field, one must first cut off the interaction. The simplest method is to truncate the Fourier expansion of the field ϕ in (3) to ϕ_κ(x) = ∫_{|k|≤κ} ϕ̃(k)e^(−ikx)dk and to compactify the spatial volume of the perturbation to V. One obtains the desired quantum field theory as a limit of the truncated approximations. The constants σ, c have the form σ = αλ + βλ² and c = γλ + δλ² + ϵλ³. One desires that the expectations of products of fields have a limit as κ → ∞. One chooses α, γ (in dimension 2), and one chooses all the coefficients α, β, γ, δ, ϵ (in dimension 3), to depend on κ in the way that perturbation theory suggests. One then proves that the expectations converge as κ → ∞, even though the specified constants α, … diverge. These constants provide the required renormalization of the interaction. In the three-dimensional case one also needs to normalize vectors in the Fock space a constant that diverges with κ; one calls this constant a wave-function renormalization constant.

The “mass” operator in natural units is M = √(H² − 𝐏²) ≥ 0, and the vacuum vector Ω is a null vector, MΩ = 0. Massive single particle states are eigenvectors of an eigenvalue m > 0. If m is an isolated eigenvalue of M, then one infers from the Wightman axioms and Haag–Ruelle scattering theory that asymptotic scattering states of an arbitrary number of particles exist, see [24, 18].

The fundamental problem of asymptotic completeness is the question whether these asymptotic states (including possible bound states) span ℋ. Over the past thirty years, several new methods have emerged, yielding proofs of asymptotic completeness in scattering theory for non-relativistic quantum mechanics. This gives some hope that one can now attack the open problem of asymptotic completeness for any known example of nonlinear quantum field theory.

In contrast to the existence of quantum fields with a ϕ⁴ nonlinearity in dimensions 2 and 3, the question of extending these results to four dimensions is problematic. It is known that self-interacting scalar fields with a quartic nonlinearity do not exist in dimension 5 or greater [12, 1]. (The proofs apply to field theories with a single, scalar field.) Analysis of the borderline dimension 4 (between existence and non-existence) is more subtle; if one makes some reasonable (but not entirely proved) assumptions, one also can conclude triviality for the quartic coupling in four dimensions. Not only is this persuasive evidence, but furthermore the quartic coupling does not have the property of asymptotic freedom in four dimensions. Thus all insights from random walks, perturbation theory, and renormalization analysis point toward triviality of the quartic interaction in four dimensions.

Other polynomial interactions in four dimensions are even more troublesome: The classical energy of the cubic interaction is unbounded from below, so it appears an unlikely candidate for a quantum theory where positivity of the energy is an axiom. And polynomial interactions of degree greater than quartic are more singular in perturbation theory than the quartic interaction.

All these reasons complement the physical and geometric importance of Yang–Mills theory and highlight it as the natural candidate for four-dimensional CQFT.

### 6.3. Large Coupling Constant

In two dimensions, the field theory with energy density (3) exists for all positive λ. For 0 ≤ λ ≪ 1 the solution is unique under a variety of conditions; but for λ ≫ 1 two different solutions exist, each characterized by its ground state or “phase.” The solution in each phase satisfies the Osterwalder–Schrader and Wightman axioms with a non-zero mass gap and a unique, Poincaré invariant vacuum state. The distinct solutions appear as a bifurcation of a unique approximating solution with finite volume V as V → ∞.

One exhibits this behavior by reordering and scaling the λϕ⁴ interaction (3) with λ ≫ 1 to obtain an equivalent double-well potential of the form

> **(4)** λ(ϕ² − 1/λ)² + ½σϕ² + c.

Here λ ≪ 1 is a new coupling constant and the renormalization constants σ and c are somewhat different from those above. The two solutions for a given λ are related by the broken ϕ → −ϕ symmetry of the interaction (4). The proof of these facts relies upon developing a convergent cluster expansion about each minimum of the potential arising from (4) and proving the probability of tunneling between the two solutions is small [20].

A critical value λ_c of λ in (3) provides a boundary between the uniqueness of the solution (for λ < λ_c) and the existence of a phase transition λ > λ_c. As λ increases to λ_c, the mass gap m = m(λ) decreases monotonically and continuously to zero [23, 17, 32].

The detailed behavior of the field theory (or the mass) in the neighborhood of λ = λ_c is extraordinarily difficult to analyze; practically nothing has been proved. Physicists have a qualitative picture based on the assumed fractional power-law behavior m(λ) ∼ |λ_c − λ|ν above or below the critical point, where the exponent ν depends on the dimension. One also expects that the critical coupling λ_c corresponds to the greatest physical force between particles, and that these critical theories are close to scaling limits of Ising-type modes in statistical physics. One expects that further understanding of these ideas will result in new computational tools for quantum fields and for statistical physics.

There is some partial understanding of a more general multi-phase case. One can find an arbitrary number n of phases by making a good choice of a polynomial energy density P_n(ϕ) with n minima. It is interesting to study the perturbation of a fixed such polynomial P_n by polynomials Q of lower degree and with small coefficients. Among these perturbations one can find families of polynomials Q(ϕ) that yield field theories with exactly n′ ≤ n phases [26].

### 6.4. Yukawa Interactions and Abelian Gauge Theory

The existence of boson-fermion interactions is also known in two dimensions, and partial results exist in three dimensions. In two dimensions Yukawa interactions of the form ψψϕ exist with appropriate renormalization, as well as their generalizations of the form P(ϕ) + ψψQ′′(ϕ), see [42, 18]. The supersymmetric case P = |Q′|² requires extra care in dealing with cancellations of divergences, see [28] for references.

A continuum two-dimensional Higgs model describes an abelian gauge field interacting with a charged scalar field. Brydges, Fröhlich, and Seiler constructed this theory and showed that it satisfies the Osterwalder–Schrader axioms [7], providing the only complete example of an interacting gauge theory satisfying the axioms. A mass gap exists in this model [4]. Extending all these conclusions to a non-abelian Higgs model, even in two dimensions, would represent a qualitative advance.

Partial results on the three-dimensional ψψϕ interaction have been established, see [30], as well as for other more singular interactions [14].

### 6.5. Yang–Mills Theory

Much of the mathematical progress reviewed above results from understanding functional integration and using those methods to construct Euclidean field theories. Functional integration for gauge theories raises new technical problems revolving around the rich group of symmetries, especially gauge symmetry. Both the choice of gauge and the transformation between different choices complicate the mathematical structure; yet gauge symmetry provides the possibility of asymptotic freedom. Certain insights and proposals in the physics literature [9, 5] have led to an extensive framework; yet the implications of these ideas for a mathematical construction of Yang–Mills theory need further understanding.

Wilson suggested a different approach based on approximating continuum space-time by a lattice, on which he defined a gauge-invariant action [48]. With a compact gauge group and a compactified space-time, the lattice approximation reduces the functional integration to a finite-dimensional integral. One must then verify the existence of limits of appropriate expectations of gauge-invariant observables as the lattice spacing tends to zero and as the volume tends to infinity.

Reflection positivity holds for the Wilson approximation [36], a major advantage; few methods exist to recover reflection positivity in case it is lost through regularization—such as with dimensional regularization, Pauli–Villars regularization, and many other methods. Establishing a quantum mechanical Hilbert space is part of the solution to this Millennium problem.

Balaban studied this program in a three-dimensional lattice with periodic boundary conditions, approximating a space-time torus [2]. He studied renormalization transformations (integration of large-momentum degrees of freedom followed by rescaling) and established many interesting properties of the effective action they produce. These estimates are uniform in the lattice spacing, as the spacing tends to zero. The choices of gauges are central to this work, as well as the use of Sobolev space norms to capture an analysis of geometric effects.

One defines these gauges in phase cells: The choices vary locally in space-time, as well as on different length scales. The choices evolve inductively as the renormalization transformations proceed, from gauges suitable for local regularity (ultraviolet gauges) to those suitable for macroscopic distances (infrared gauges). This is an important step toward establishing the existence of the continuum limit on a compactified space-time. These results need to be extended to the study of expectations of gauge-invariant functions of the fields.

While this work in three dimensions is important in its own right, a qualitative breakthrough came with Balaban’s extension of this analysis to four dimensions [3]. This includes an analysis of asymptotic freedom to control the renormalization group flow as well as obtaining quantitative estimates on effects arising from large values of the gauge field.

Extensive work has also been done on a continuum regularization of the Yang–Mills interaction, and it has the potential for further understanding [39, 29].

These steps toward understanding quantum Yang–Mills theory lead to a vision of extending the present methods to establish a complete construction of the Yang–Mills quantum field theory on a compact, four-dimensional space-time. One presumably needs to revisit known results at a deep level, simplify the methods, and extend them.

New ideas are needed to prove the existence of a mass gap that is uniform in the volume of space-time. Such a result presumably would enable the study of the limit as T⁴ → ℝ⁴.²

### 6.6. Further Remarks

Because four-dimensional gauge theory is a theory in which the mass gap is not classically visible, to demonstrate it may require a non-classical change of variables or “duality transformation.” For example, duality has been used to establish a mass gap in the statistical mechanics problem of a Coulomb gas, where the phenomenon is known as Debye screening: Macroscopic test charges in a neutral Coulomb gas experience a mutual force that decays exponentially with the distance. The mathematical proof of this screening phenomenon proceeds through the identity of the partition function of the Coulomb gas to that of a cos(λϕ) (sine-Gordon) field theory, and the approximate parabolic potential near a minimum of this potential, see [6].

One view of the mass gap in Yang–Mills theory suggests that it could arise from the quartic potential (A ∧ A)² in the action, where F = dA + gA ∧ A, see [11], and may be tied to curvature in the space of connections, see [44]. Although the Yang–Mills action has flat directions, certain quantum mechanics problems with potentials involving flat directions (directions for which the potential remains bounded as |x| → ∞) do lead to bound states [43].

A prominent speculation about a duality that might shed light on dynamical properties of four-dimensional gauge theory involves the 1/N expansion [25]. It is suspected that four-dimensional quantum gauge theory with gauge group SU(N) (or SO(N), or Sp(N)) may be equivalent to a string theory with 1/N as the string coupling constant. Such a description might give a clear-cut explanation of the mass gap and confinement and perhaps a good starting point for a rigorous proof (for sufficiently large N). There has been surprising progress along these lines for certain strongly coupled four-dimensional gauge systems with matter [31], but as of yet there is no effective approach to the gauge theory without fermions. Investigations of supersymmetric theories and string theories have uncovered a variety of other approaches to understanding the mass gap in certain four-dimensional gauge theories with matter fields; for example, see [41].

---

### Footnotes

¹ A natural 1–1 correspondence between such classical ‘differential polynomials’ and quantized operators does not exist, since the correspondence has some standard subtleties involving renormalization [27]. One expects that the space of classical differential polynomials of dimension ≤ d does correspond to the space of local quantum operators of dimension ≤ d.

² We specifically exclude weak-existence (compactness) as the solution to the existence part of the Millennium problem, unless one also uses other techniques to establish properties of the limit (such as the existence of a mass gap and the axioms).

---

### References

[1] M. Aizenman, Geometric analysis of ϕ⁴ fields and Ising models, Comm. Math. Phys. 86 (1982), 1–48.

[2] T. Balaban, Ultraviolet stability of three-dimensional lattice pure gauge field theories, Comm. Math. Phys. 102 (1985), 255–275.

[3] T. Balaban, Renormalization group approach to lattice gauge field theories. I: generation of effective actions in a small field approximation and a coupling constant renormalization in 4D, Comm. Math. Phys. 109 (1987), 249–301.

[4] T. Balaban, D. Brydges, J. Imbrie, and A. Jaffe, The mass gap for Higgs models on a unit lattice, Ann. Physics 158 (1984), 281–319.

[5] C. Becchi, A. Rouet, and R. Stora, Renormalization of gauge theories, Ann. Phys. 98 (1976), 287–321.

[6] D. Brydges and P. Federbush, Debye screening, Comm. Math. Phys. 73 (1980), 197–246.

[7] D. Brydges, J. Fröhlich, and E. Seiler, On the construction of quantized gauge fields, I. Ann. Phys. 121 (1979), 227–284, II. Comm. Math. Phys. 71 (1980), 159–205, and III. Comm. Math. Phys. 79 (1981), 353–399.

[8] M. Creutz, Monte carlo study of quantized SU(2) gauge theory, Phys. Rev. D21 (1980), 2308–2315.

[9] L. D. Faddeev and V. N. Popov, Feynman diagrams for the Yang–Mills fields, Phys. Lett. B25 (1967), 29–30.

[10] J. Feldman and K. Osterwalder, The Wightman axioms and the mass gap for weakly coupled ϕ₃⁴ quantum field theories, Ann. Physics 97 (1976), 80–135.

[11] R. Feynman, The quantitative behavior of Yang–Mills theory in 2+1 dimensions, Nucl. Phys. B188 (1981), 479–512.

[12] J. Fröhlich, On the triviality of λϕ_d⁴ theories and the approach to the critical point, Nucl. Phys. 200 (1982), 281–296.

[13] J. Fröhlich, K. Osterwalder, and E. Seiler, On virtual representations of symmetric spaces and theory analytic continuation, Ann. Math. 118 (1983), 461–489.

[14] K. Gawedzki, Renormalization of a non-renormalizable quantum field theory, Nucl. Phys. B262 (1985), 33–48.

[15] J. Glimm and A. Jaffe, The λϕ₂⁴ quantum field theory without cut-offs, I. Phys. Rev. 176 (1968), 1945–1951, II. Ann. Math. 91 (1970), 362–401, III. Acta. Math. 125 (1970), 203–267, and IV. J. Math. Phys. 13 (1972), 1568–1584.

[16] J. Glimm and A. Jaffe, Positivity of the ϕ₃⁴ Hamiltonian, Fortschr. Phys. 21 (1973), 327–376.

[17] J. Glimm and A. Jaffe, ϕ₂⁴ quantum field model in the single phase region: differentiability of the mass and bounds on critical exponents, Phys. Rev. 10 (1974), 536–539.

[18] J. Glimm and A. Jaffe, Quantum Physics, Second Edition, Springer Verlag, 1987, and Selected Papers, Volumes I and II, Birkhäuser Boston, 1985. (Volume II includes reprints of [15, 16, 19, 20].)

[19] J. Glimm, A. Jaffe, and T. Spencer, The Wightman axioms and particle structure in the P(ϕ)₂ quantum field model, Ann. Math. 100 (1974), 585–632.

[20] J. Glimm, A. Jaffe, and T. Spencer, A convergent expansion about mean field theory, I. Ann. Phys. 101 (1976), 610–630, and II. Ann. Phys. 101 (1976), 631–669.

[21] D. J. Gross and F. Wilczek, Ultraviolet behavior of non-abelian gauge theories, Phys. Rev. Lett. 30 (1973), 1343–1346.

[22] F. Guerra, L. Rosen, and B. Simon, The P(ϕ)₂ Euclidean quantum field theory as classical statistical mechanics, Ann. Math. 101 (1975), 111–259.

[23] F. Guerra, L. Rosen, and B. Simon, Correlation inequalities and the mass gap in P(ϕ)₂, III. Mass gap for a class of strongly coupled theories with non-zero external field, Comm. Math. Phys. 41 (1975), 19–32.

[24] R. Haag, Local Quantum Physics, Springer Verlag, 1992.

[25] G. ’t Hooft, A planar diagram theory for strong interactions, Nucl. Phys. B72 (1974), 461–473.

[26] J. Imbrie, Phase diagrams and cluster expansions for low temperature P(ϕ)₂ models, Comm. Math. Phys. 82 (1981), 261–304 and 305–343.

[27] C. Itzykson and J.-B. Zuber, Quantum Field Theory, McGraw-Hill, New York, 1980.

[28] A. Jaffe, Constructive quantum field theory, in Mathematical Physics, T. Kibble, ed., World Scientific, Singapore, 2000.

[29] J. Magnen, Vincent Rivasseau, and Roland Sénéor, Construction of YM₄ with an infrared cutoff, Comm. Math. Phys. 155 (1993), 325–383.

[30] J. Magnen and R. Sénéor, Yukawa quantum field theory in three dimensions, in Third International Conference on Collective Phenomena, J. Lebowitz et. al. eds., New York Academy of Sciences, 1980.

[31] J. Maldacena, The large N limit of superconformal field theories and supergravity, Adv. Theor. Math. Phys. 2 (1998), 231–252.

[32] O. McBryan and J. Rosen, Existence of the critical point in ϕ⁴ field theory, Comm. Math. Phys. 51 (1976), 97–105.

[33] E. Nelson, Quantum fields and Markoff fields, in Proc. Sympos. Pure Math. XXIII 1971, AMS, Providence, R.I., 1973, 413–420.

[34] L. O’Raifeartaigh, The Dawning of Gauge Theory, Princeton University Press, 1997.

[35] K. Osterwalder and R. Schrader, Axioms for Euclidean Green’s functions, Comm. Math. Phys. 31 (1973), 83–112, and Comm. Math. Phys. 42 (1975), 281–305.

[36] K. Osterwalder and E. Seiler, Gauge theories on the lattice, Ann. Phys. 110 (1978), 440–471.

[37] K. Osterwalder and R. Sénéor, A nontrivial scattering matrix for weakly coupled P(ϕ)₂ models, Helv. Phys. Acta 49 (1976), 525–535.

[38] H. D. Politzer, Reliable perturbative results for strong interactions? Phys. Rev. Lett. 30 (1973), 1346–1349.

[39] V. Rivasseau, From Perturbative to Constructive Renormalization, Princeton Univ. Press, Princeton, NJ, 1991.

[40] A. Salam, Weak and electromagnetic interactions, in Svartholm: Elementary Particle Theory, Proceedings of The Nobel Symposium held in 1968 at Lerum, Sweden, Stockholm, 1968, 367–377.

[41] N. Seiberg and E. Witten, Electric-magnetic duality, monopole condensation, and confinement in N=2 supersymmetric Yang–Mills theory, Nucl. Phys. B426 (1994), 19–52.

[42] E. Seiler, Schwinger functions for the Yukawa model in two dimensions, Comm. Math. Phys. 42 (1975), 163–182.

[43] B. Simon, Some quantum operators with discrete spectrum but classically continuous spectrum, Ann. Phys. 146 (1983), 209–220.

[44] I. M. Singer, The geometry of the orbit space for nonabelian gauge theories, Phys. Scripta 24 (1981), 817–820.

[45] R. Streater and A. Wightman, PCT, Spin and Statistics and all That, W. A. Benjamin, New York, 1964.

[46] K. Symanzik, Euclidean quantum field theory, in Local Quantum Theory, R. Jost, ed., Academic Press, New York, 1969, 152–226.

[47] S. Weinberg, A model of leptons, Phys. Rev. Lett. 19 (1967), 1264–1266.

[48] K. G. Wilson, Quarks and strings on a lattice, in New Phenomena In Subnuclear Physics, Proceedings of the 1975 Erice School, A. Zichichi, ed., Plenum Press, New York, 1977.

[49] E. Witten, Supersymmetric Yang–Mills theory on a four-manifold, J. Math. Phys. 35 (1994), 5101–5135.

[50] C. N. Yang and R. L. Mills, Conservation of isotopic spin and isotopic gauge invariance, Phys. Rev. 96 (1954), 191–195.