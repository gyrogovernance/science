{{Short description|Unitary matrix containing information on the weak interaction}}
{{Flavour quantum numbers}}

In the [[Standard Model]] of [[particle physics]], the '''Cabibbo–Kobayashi–Maskawa matrix''', '''CKM matrix''', '''quark mixing matrix''', or '''KM matrix''' is a [[unitary matrix]] that contains information on the strength of the [[Flavour (particle physics)|flavour]]-changing [[weak interaction]]. Technically, it specifies the mismatch of [[quantum state]]s of [[quark]]s when they propagate freely and when they take part in the [[weak interaction]]s. It is important in the understanding of [[CP violation]]. This matrix was introduced for three generations of quarks by [[Makoto Kobayashi (physicist)|Makoto Kobayashi]] and [[Toshihide Maskawa]], adding one [[generation (particle physics)|generation]] to the matrix previously introduced by [[Nicola Cabibbo]]. This matrix is also an extension of the [[GIM mechanism]], which only includes two of the three current families of quarks.

==Description==
===Predecessor: Cabibbo matrix===
[[Image:Cabibbo angle.svg|thumb|270px|right|The Cabibbo angle represents the rotation of the mass eigenstate vector space formed by the mass eigenstates
<math> | d \rangle , \,| s \rangle </math>
into the weak eigenstate vector space formed by the weak eigenstates
<math> | d' \rangle \,, ~ | \, s' \rangle ~.</math> {{nowrap|{{mvar|θ}}{{sub|c}} {{=}} 13.02°  .}} ]]
In 1963, [[Nicola Cabibbo]] introduced the '''Cabibbo angle''' ({{mvar|θ}}{{sub|c}}) to preserve the universality of the [[weak interaction]].<ref name="Cabibbo">
{{cite journal
 |first=N. |last=Cabibbo
 |year=1963
 |title=Unitary Symmetry and Leptonic Decays
 |journal=[[Physical Review Letters]]
 |volume=10  |issue=12  |pages=531–533
 |doi=10.1103/PhysRevLett.10.531  |doi-access=free
 |bibcode=1963PhRvL..10..531C 
}}
</ref> 
Cabibbo was inspired by previous work by [[Murray Gell-Mann]] and  Maurice Lévy,<ref>
{{cite journal
 |first1=M. |last1=Gell-Mann |author1-link=Murray Gell-Mann
 |first2=M. |last2=Lévy
 |year=1960
 |title= The Axial Vector Current in Beta Decay
 |journal=[[Il Nuovo Cimento]]
 |volume=16 |issue=4 |pages=705–726
 |doi=10.1007/BF02859738
 |bibcode=1960NCim...16..705G
 |s2cid=122945049
}}
</ref>
on the effectively rotated nonstrange and strange vector and axial weak currents, which he references.<ref>
{{cite journal
 |first=L. |last=Maiani
 |year=2009
 |title=Sul premio Nobel per la fisica 2008
 |trans-title=On the Nobel prize in Physics for 2008
 |journal=Il Nuovo Saggiatore
 |volume=25  |issue=1–2  |pages=78
 |url=http://prometeo.sif.it:8080/papers/online/sag/025/01-02/pdf/78_opinioni.pdf
 |url-status=dead  |access-date=30 November 2010
 |archive-url=https://web.archive.org/web/20110722053046/http://prometeo.sif.it:8080/papers/online/sag/025/01-02/pdf/78_opinioni.pdf
 |archive-date=22 July 2011  |df=dmy-all
}}
</ref>

In light of current concepts (quarks had not yet been proposed), the Cabibbo angle is related to the relative probability that [[down quark|down]] and [[strange quark]]s decay into [[up quark]]s  (&nbsp;|{{mvar|V}}{{sub|ud}}|{{sup|2}} &nbsp; and &nbsp; |{{mvar|V}}{{sub|us}}|{{sup|2}}&nbsp;, respectively). In particle physics terminology, the object that couples to the up quark via charged-current weak interaction is a superposition of down-type quarks, here denoted by {{mvar|d′}}.<ref name="Hughes">
{{cite book
 |first=I.S. |last=Hughes
 |year=1991
 |chapter=Chapter 11.1 – Cabibbo Mixing
 |title=Elementary Particles
 |edition=3rd  |pages=242–243
 |publisher=[[Cambridge University Press]]
 |isbn=978-0-521-40402-0
 |chapter-url=https://books.google.com/books?id=JN6qlZlGUG4C&q=cabbibo+angle&pg=PA242
}}
</ref>
Mathematically this is:

:<math> d' = V_\mathrm{ud} \; d  ~~ + ~~ V_\mathrm{us} \; s ~,</math>

or using the Cabibbo angle:

:<math> d' = \cos \theta_\mathrm{c} \; d  ~~ + ~~ \sin \theta_\mathrm{c} \; s ~.</math>

Using the currently accepted values for &nbsp; |{{mvar|V}}{{sub|ud}}| &nbsp; and &nbsp; |{{mvar|V}}{{sub|us}}| &nbsp; (see below), the Cabibbo angle can be calculated using

:<math> \tan\theta_\mathrm{c} = \frac{\, |V_\mathrm{us}| \,}{|V_\mathrm{ud}|} = \frac{0.22534}{0.97427} \quad \Rightarrow \quad \theta_\mathrm{c}= ~13.02^\circ ~.</math>

When the [[charm quark]] was discovered in 1974, it was noticed that the down and strange quark could transition into either the up or charm quark, leading to two sets of equations:

:<math> d' = V_\mathrm{ud} \; d ~~ + ~~ V_\mathrm{us} \; s ~,</math>
:<math> s' = V_\mathrm{cd} \; d ~~ + ~~ V_\mathrm{cs} \; s ~;</math>

or using the Cabibbo angle:

:<math> d' = ~~~ \cos{\theta_\mathrm{c}} \; d  ~~+~~ \sin{\theta_\mathrm{c}} \; s ~,</math>
:<math> s' =  -  \sin{\theta_\mathrm{c}} \; d  ~~+~~ \cos{\theta_\mathrm{c}} \; s ~.</math>

This can also be written in [[matrix (mathematics)|matrix notation]] as:

:<math>
\begin{bmatrix}  d'  \\  s'  \end{bmatrix} =
\begin{bmatrix} V_\mathrm{ud} & V_\mathrm{us} \\ V_{cd} & V_{cs} \\ \end{bmatrix}
\begin{bmatrix}  d  \\  s  \end{bmatrix} ~,
</math>

or using the Cabibbo angle

:<math>
\begin{bmatrix} d'  \\  s'  \end{bmatrix} =
\begin{bmatrix} ~~\cos{ \theta_\mathrm{c} } & \sin{ \theta_\mathrm{c} } \\  -\sin{\theta_\mathrm{c}} & \cos{\theta_\mathrm{c}}\\ \end{bmatrix}
\begin{bmatrix}  d  \\  s  \end{bmatrix}~,
</math>

where the various |{{mvar|V{{sub|ij}}}}|{{sup|2}} represent the probability that the quark of flavor {{mvar|j}} decays into a quark of flavor {{mvar|i}}. This 2×2&nbsp;[[rotation matrix]] is called the "Cabibbo matrix", and was subsequently expanded to the 3×3 CKM matrix.

[[Image:Quark weak interactions.svg|thumb|270px|right|A pictorial representation of the six quarks' decay modes, with mass increasing from left to right.]]

===CKM matrix===
[[Image:Weak_Decay_(flipped).svg|thumb|270px|right|A diagram depicting the decay routes due to the charged weak interaction and some indication of their likelihood. The intensity of the lines is given by the CKM parameters]]
In 1973, observing that [[CP-violation]] could not be explained in a four-quark model, Kobayashi and Maskawa generalized the Cabibbo matrix into the Cabibbo–Kobayashi–Maskawa matrix (or CKM matrix) to keep track of the weak decays of three generations of quarks:<ref name="KM">
{{cite journal
 |first1=M. |last1=Kobayashi
 |first2=T. |last2=Maskawa
 |year=1973
 |title=CP-violation in the renormalizable theory of weak interaction
 |journal=[[Progress of Theoretical Physics]]
 |volume=49 |issue=2 |pages=652–657
 |doi=10.1143/PTP.49.652  |doi-access=free 
 |bibcode=1973PThPh..49..652K
|hdl=2433/66179|hdl-access=free}}
</ref>

:<math>\begin{bmatrix}  d'  \\  s'  \\  b'  \end{bmatrix} = \begin{bmatrix} V_\mathrm{ud} & V_\mathrm{us} & V_\mathrm{ub} \\ V_\mathrm{cd} & V_\mathrm{cs} & V_\mathrm{cb} \\ V_\mathrm{td} & V_\mathrm{ts} & V_\mathrm{tb} \end{bmatrix} \begin{bmatrix}  d  \\  s  \\  b  \end{bmatrix}~.</math>

On the left are the [[weak interaction]] doublet partners of down-type quarks, and on the right is the CKM matrix, along with a vector of mass eigenstates of down-type quarks. The CKM matrix describes the probability of a transition from one flavour {{mvar|j}} quark to another flavour {{mvar|i}} quark. These transitions are proportional to |{{mvar|V{{sub|ij}}}}|{{sup|2}}.

As of 2023, the best determination of the individual [[absolute value|magnitude]]s of the CKM matrix elements was:<ref name="PDG2023">{{cite journal |last1=R.L. Workman et al. (Particle Data Group) |title=Review of Particle Physics (and 2023 update) |journal=Progress of Theoretical and Experimental Physics |date=August 2022 |volume=2022 |issue=8 |pages=083C01 |doi=10.1093/ptep/ptac097 |url=https://pdg.lbl.gov/ |access-date=12 September 2023 |ref=PDG2023|doi-access=free |hdl=20.500.11850/571164 |hdl-access=free }}</ref>
:<math>
\begin{bmatrix}
|V_{ud}| & |V_{us}| & |V_{ub}| \\
|V_{cd}| & |V_{cs}| & |V_{cb}| \\
|V_{td}| & |V_{ts}| & |V_{tb}|
\end{bmatrix} = \begin{bmatrix}
0.97435 \pm 0.00016 & 0.22500 \pm 0.00067 & 0.00369\pm 0.00011\\
0.22486 \pm 0.00067 & 0.97349 \pm 0.00016 & 0.04182^{+0.00085}_{-0.00074} \\
0.00857_{-0.00018}^{+0.00020} & 0.04110^{+0.00083}_{-0.00072}  & 0.999118^{+0.000031}_{-0.000036}
\end{bmatrix}.
</math>

Using those values, one can check the unitarity of the CKM matrix. In particular, we find that the first-row matrix elements give: <math> |V_\mathrm{ud}|^2 + |V_\mathrm{us}|^2 + |V_\mathrm{ub}|^2 = 0.999997 \pm 0.0007</math>

making the experimental results in line with the theoretical value of 1.

The choice of usage of down-type quarks in the definition is a convention, and does not represent a physically preferred asymmetry between up-type and down-type quarks. Other conventions are equally valid: The mass eigenstates {{math|u}}, {{math|c}}, and {{math|t}} of the up-type quarks can equivalently define the matrix in terms of ''their'' weak interaction partners {{math|u′}}, {{math|c′}}, and {{math|t′}}. Since the CKM matrix is unitary, its inverse is the same as its [[conjugate transpose]], which the alternate choices use; it appears as the same matrix, in a slightly altered form.

==General case construction==
To generalize the matrix, count the number of physically important parameters in this matrix {{mvar|V}} which appear in experiments. If there are {{sc|{{mvar|N}}}} generations of quarks (2{{sc|{{mvar|N}}}} [[flavour (particle physics)|flavour]]s) then

* An {{sc|{{mvar|N}}}}&nbsp;×&nbsp;{{sc|{{mvar|N}}}} unitary matrix (that is, a matrix {{mvar|V}} such that {{mvar|V{{sup|†}}V&nbsp;{{=}}&nbsp;I}}, where {{mvar|V{{sup|†}}}} is the [[conjugate transpose]] of {{mvar|V}} and {{mvar|I}} is the identity matrix) requires {{sc|{{mvar|N}}}}<sup>2</sup> real parameters to be specified.
* 2{{sc|{{mvar|N}}}}&nbsp;−&nbsp;1 of these parameters are not physically significant, because one phase can be absorbed into each quark field (both of the mass eigenstates, and of the weak eigenstates), but the matrix is independent of a common phase. Hence, the total number of free variables independent of the choice of the phases of basis vectors is {{sc|{{mvar|N}}}}<sup>2</sup>&nbsp;−&nbsp;(2{{sc|{{mvar|N}}}}&nbsp;−&nbsp;1) = ({{sc|{{mvar|N}}}}&nbsp;−&nbsp;1)<sup>2</sup>.
** Of these, {{sfrac|1|2}}{{sc|{{mvar|N}}}}({{sc|{{mvar|N}}}}&nbsp;−&nbsp;1) are rotation angles called ''quark mixing angles''.
** The remaining {{sfrac|1|2}}({{sc|{{mvar|N}}}}&nbsp;−&nbsp;1)({{sc|{{mvar|N}}}}&nbsp;−&nbsp;2) are complex phases, which cause [[CP violation]].

=== {{sc|{{mvar|N}}}} = 2 ===
For the case {{sc|{{mvar|N}}}}&nbsp;=&nbsp;2, there is only one parameter, which is a mixing angle between two generations of quarks. Historically, this was the first version of CKM matrix when only two generations were known. It is called the '''Cabibbo angle''' after its inventor [[Nicola Cabibbo]].

=== {{sc|{{mvar|N}}}} = 3 ===
For the [[Standard Model]] case ({{sc|{{mvar|N}}}}&nbsp;=&nbsp;3), there are three mixing angles and one CP-violating complex phase.<ref>
{{cite web
 |first=J.C. |last=Baez
 |date=4 April 2011
 |title=Neutrinos and the mysterious Pontecorvo-Maki-Nakagawa-Sakata matrix
 |url=http://math.ucr.edu/home/baez/neutrinos.html |access-date=2016-02-13
 |df=dmy-all
 |quote=In fact, the [[Pontecorvo–Maki–Nakagawa–Sakata matrix]] actually affects the behavior of all leptons, not just neutrinos. Furthermore, a similar trick works for quarks – but then the matrix ''U'' is called the Cabibbo–Kobayashi–Maskawa matrix.
}}
</ref>

==Observations and predictions==
Cabibbo's idea originated from a need to explain two observed phenomena:
#the transitions {{nowrap| {{math|u ↔ d}}, }} {{nowrap| {{math|e ↔ ν<sub>e</sub>}} ,}} and {{nowrap| {{math|μ ↔ ν<sub>μ</sub>}} }} had similar amplitudes.
#the transitions with change in strangeness {{nowrap| {{math|ΔS {{=}} 1}} }} had amplitudes equal to {{sfrac| 1 |4}} of those with {{nowrap| {{math|ΔS {{=}} 0}} .}}
Cabibbo's solution consisted of postulating ''weak universality'' (see below) to resolve the first issue, along with a mixing angle {{math|''θ''<sub>c</sub>}}, now called the ''Cabibbo angle'', between the {{math|d}} and {{math|s}} quarks to resolve the second.

For two generations of quarks, there can be no CP violating phases, as shown by the counting of the previous section. Since CP violations ''had'' already been seen in 1964, in neutral [[kaon]] decays, the [[standard model (basic details)|Standard Model]] that emerged soon after clearly indicated the existence of a third generation of quarks, as Kobayashi and Maskawa pointed out in 1973. The discovery of the [[bottom quark]] at [[Fermilab]] (by [[Leon Lederman]]'s group) in 1976 therefore immediately started off the search for the [[top quark]], the missing third-generation quark.

Note, however, that the specific values that the angles take on are ''not'' a prediction of the standard model: They are [[free parameter]]s. At present, there is no generally-accepted theory that explains why the angles should have the values that are measured in experiments.

==Weak universality==
The constraints of unitarity of the CKM-matrix on the diagonal terms can be written as

::<math>\sum_k |V_{jk}|^2 = \sum_k |V_{kj}|^2 = 1</math>

separately for each generation {{mvar|j}}. This implies that the sum of all couplings of any ''one'' of the up-type quarks to ''all'' the down-type quarks is the same for all generations. This relation is called ''weak universality'' and was first pointed out by [[Nicola Cabibbo]] in 1967. Theoretically it is a consequence of the fact that all [[SU(2)]] doublets couple with the same strength to the [[vector boson]]s of weak interactions. It has been subjected to continuing experimental tests.

==Unitary triangles==
The remaining constraints of unitarity of the CKM-matrix can be written in the form

:<math>\sum_k V_{ik}V^*_{jk} = 0 ~.</math>

For any fixed and different {{mvar|i}} and {{mvar|j}}, this is a constraint on three complex numbers, one for each {{mvar|k}}, which says that these numbers form the sides of a triangle in the [[complex plane]]. There are six choices of {{mvar|i}} and {{mvar|j}} (three independent), and hence six such triangles, each of which is called a ''unitary triangle''. Their shapes can be very different, but they all have the same area, which can be related to the [[CP violation|CP violating]] phase. The area vanishes for the specific parameters in the Standard Model for which there would be no [[CP violation]]. The orientation of the triangles depend on the phases of the quark fields.

A popular quantity amounting to twice the area of the unitarity triangle is the '''Jarlskog invariant''' (introduced by [[Cecilia Jarlskog]] in 1985), 
:<math> J = c_{12}c_{13}^2 c_{23}s_{12}s_{13}s_{23}\sin \delta \approx 3\cdot10^{-5} ~.</math> 
For Greek indices denoting up quarks and Latin ones down quarks, the 4-tensor <math>\;(\alpha,\beta;i,j)\equiv \operatorname{Im} (V_{\alpha i} V_{\beta j} V^*_{\alpha j} V_{\beta i}^{*}) \;</math> is doubly antisymmetric,
:<math>(\beta,\alpha;i,j) = -(\alpha,\beta;i,j)=(\alpha,\beta;j,i) ~.</math>
Up to antisymmetry, it only has {{nowrap| 9 {{=}} 3 × 3 }} non-vanishing components, which, remarkably, from the unitarity of {{mvar|V}}, can be shown to be ''all identical in magnitude'', that is,
:<math>
(\alpha,\beta;i,j)= J   ~  \begin{bmatrix} \;~~0 & \;~~1 & -1 \\ -1 & \;~~0 & \;~~1 \\ \;~~1 & -1 & \;~~0 \end{bmatrix}_{\alpha \beta} \otimes \begin{bmatrix} \;~~0 & \;~~1 & -1 \\ -1 & \;~~0 & \;~~1 \\ \;~~1 & -1 & \;~~0 \end{bmatrix}_{ij} \;,
</math>
so that 
:<math>J = (u,c;s,b) = (u,c;d,s) = (u,c;b,d) = (c,t;s,b) = (c,t;d,s) = (c,t;b,d)
 = (t,u;s,b) = (t,u;b,d) = (t,u;d,s) ~.</math>

Since the three sides of the triangles are open to direct experiment, as are the three angles, a class of tests of the Standard Model is to check that the triangle closes. This is the purpose of a modern series of experiments under way at the Japanese [[Belle experiment|BELLE]] and the American [[BaBar experiment|BaBar]] experiments, as well as at [[LHCb]] in CERN, Switzerland.

==Parameterizations==
Four independent parameters are required to fully define the CKM matrix.  Many parameterizations have been proposed, and three of the most common ones are shown below.

===KM parameters===
The original parameterization of Kobayashi and Maskawa used three angles ({{thin space}}{{mvar|θ}}{{sub|1}}, {{mvar|θ}}{{sub|2}}, {{mvar|θ}}{{sub|3}}{{thin space}}) and a CP-violating phase angle ({{thin space}}{{mvar|δ}}{{thin space}}).<ref name="KM"/> {{mvar|θ}}{{sub|1}} is the Cabibbo angle. For brevity, the cosines and sines of the angles {{mvar|θ}}{{sub|k}} are denoted {{mvar|c}}{{sub|k}} and {{mvar|s}}{{sub|k}}, for {{nowrap|k {{=}} 1,{{thin space}}2,{{thin space}}3}} respectively. 
::<math>\begin{bmatrix} c_1 & s_1 c_3 & s_1 s_3 \\
 -s_1 c_2 & c_1 c_2 c_3 - s_2 s_3 e^{i\delta} &  c_1 c_2 s_3 + s_2 c_3 e^{i\delta}\\
 -s_1 s_2 & c_1 s_2 c_3 + c_2 s_3 e^{i\delta} &  c_1 s_2 s_3 - c_2 c_3 e^{i\delta} \end{bmatrix}. </math>

==="Standard" CK parameters===
A "standard" Chau-Keung<ref>{{cite journal | last1=Kuznetsov | first1=V. E. | last2=Naumov | first2=V. A. | title=Relationship between the Kobayashi-Maskawa and Chau-Keung presentation of the quark mixing matrix | journal=Il Nuovo Cimento A | date=1995 | volume=108 | issue=12 | pages=1451–1456 | doi=10.1007/BF02821061 | arxiv=hep-ph/9605211 | bibcode=1995NCimA.108.1451K }}</ref> parameterization of the CKM matrix uses three [[Euler angles]] ({{thin space}}{{mvar|θ}}{{sub|12}}, {{mvar|θ}}{{sub|23}}, {{mvar|θ}}{{sub|13}}{{thin space}}) and one CP-violating phase ({{thin space}}{{mvar|δ}}{{sub|13}}{{thin space}}).<ref>{{cite journal |first1=L.L. |last1=Chau |first2=W.-Y. |last2=Keung |year=1984 |title=Comments on the Parametrization of the Kobayashi-Maskawa Matrix |journal=[[Physical Review Letters]] |volume=53 |pages=1802–1805 |doi=10.1103/PhysRevLett.53.1802 |bibcode=1984PhRvL..53.1802C |issue=19}}</ref> {{mvar|θ}}{{sub|12}} is the Cabibbo angle. This is the convention advocated by the [[Particle Data Group]]. Couplings between quark generations {{math|j}} and {{math|k}} vanish if {{nowrap|{{mvar|θ}}{{sub|jk}} {{=}} 0 }}. Cosines and sines of the angles are denoted {{mvar|c}}{{sub|jk}} and {{mvar|s}}{{sub|jk}}, respectively.
::<math> \begin{align} & \begin{bmatrix} 1 & 0 & 0 \\ 0 & c_{23} & s_{23} \\ 0 & -s_{23} & c_{23} \end{bmatrix}
\begin{bmatrix}
            1&0&0\\
            0&1&0\\
            0&0&e^{i\delta}
        \end{bmatrix}
		\begin{bmatrix}
				c_{13} &0& s_{13} \\
				0&1&0\\
				-s_{13} &0&c_{13} \\
		\end{bmatrix} 
        \begin{bmatrix}
            1&0&0\\
            0&1&0\\
            0&0&e^{-i\delta}
        \end{bmatrix}
 \begin{bmatrix} c_{12} & s_{12} & 0 \\ -s_{12} & c_{12} & 0 \\ 0 & 0 & 1 \end{bmatrix} \\
 & = \begin{bmatrix} c_{12}c_{13} & s_{12} c_{13} & s_{13}e^{-i\delta_{13}} \\
 -s_{12}c_{23} - c_{12}s_{23}s_{13}e^{i\delta_{13}} & c_{12}c_{23} - s_{12}s_{23}s_{13}e^{i\delta_{13}} & s_{23}c_{13}\\
 s_{12}s_{23} - c_{12}c_{23}s_{13}e^{i\delta_{13}} & -c_{12}s_{23} - s_{12}c_{23}s_{13}e^{i\delta_{13}} & c_{23}c_{13} \end{bmatrix}. \end{align} </math>

The 2008 values for the standard parameters were:<ref>Values obtained from values of Wolfenstein parameters in the 2008 ''[[Review of Particle Physics]]''.</ref>
:{{mvar|θ}}{{sub|12}} = {{val|13.04|0.05|u=°}}, {{mvar|θ}}{{sub|13}} = {{val|0.201|0.011|u=°}},  {{mvar|θ}}{{sub|23}} = {{val|2.38|0.06|u=°}}
and
:{{mvar|δ}}{{sub|13}} = {{val|1.20|0.08}}&nbsp;radians = {{val|68.8|4.5|u=°}}.

===Wolfenstein parameters===
A third parameterization of the CKM matrix was introduced by [[Lincoln Wolfenstein]] with the four real parameters {{mvar|λ}}, {{mvar|A}}, {{mvar|ρ}}, and {{mvar|η}}, which would all 'vanish' (would be zero) if there were no coupling.<ref>
{{cite journal
 |first=L. |last=Wolfenstein  |author-link=Lincoln Wolfenstein
 |year=1983
 |title=Parametrization of the Kobayashi-Maskawa Matrix
 |journal=[[Physical Review Letters]]
 |volume=51  |pages=1945–1947  |issue=21
 |doi=10.1103/PhysRevLett.51.1945  |bibcode=1983PhRvL..51.1945W
}}
</ref> The four Wolfenstein parameters have the property that all are of order 1 and are related to the 'standard' parameterization:
:{|
|-
| <math> \lambda = s_{12} ~, </math>
| <math> \lambda = s_{12} ~,</math>
|-
| <math> A \lambda^2 = s_{23} ~, </math>
| <math> A = \frac{s_{23} }{\; s_{12}^2 \;} ~,</math>
|-
| <math> A \lambda^3 ( \rho  - i \eta ) = s_{13} e^{-i\delta} ~, \quad </math>
| <math>  \rho = \operatorname\mathcal{R_e} \left\{ \frac{\; s_{13} \, e^{-i\delta} \;}{ s_{12} \, s_{23} } \right\} ~, \quad  \eta  = - \operatorname\mathcal{I_m} \left\{ \frac{\; s_{13} \, e^{-i\delta} \;}{ s_{12} \, s_{23} } \right\} ~. </math>
|}

Although the Wolfenstein parameterization of the CKM matrix can be as exact as desired when carried to high order, it is mainly used for generating convenient approximations to the standard parameterization. The approximation to order {{mvar|λ}}{{sup|3}}, good to better than 0.3% accuracy, is:
::<math>\begin{bmatrix} 1 - \tfrac{1}{2}\lambda^2 & \lambda & A\lambda^3(\rho-i\eta) \\
 -\lambda & 1-\tfrac{1}{2}\lambda^2 & A\lambda^2 \\
 A\lambda^3(1-\rho-i\eta) & -A\lambda^2 & 1  \end{bmatrix} + O(\lambda^4) ~. </math>

Rates of [[CP violation]] correspond to the parameter {{mvar|η}}.

Using the values of the previous section for the CKM matrix, as of 2008 the best determination of the Wolfenstein parameter values is:<ref name="PDG2023">{{cite journal |last1=R.L. Workman et al. (Particle Data Group) |title=Review of Particle Physics (and 2023 update) |journal=Progress of Theoretical and Experimental Physics |date=August 2022 |volume=2022 |issue=8 |pages=083C01 |doi=10.1093/ptep/ptac097 |url=https://pdg.lbl.gov/ |access-date=12 September 2023 |ref=PDG2023|doi-access=free |hdl=20.500.11850/571164 |hdl-access=free }}</ref>
:{{mvar|λ}} =.22500 ± 0.0067, &nbsp; {{mvar|A}} = {{val|0.826|+0.018|-0.015}}, &nbsp; {{mvar|ρ}} = 0.159±0.010, &nbsp; and &nbsp; {{mvar|η}} = 0.348±0.010.

==Nobel Prize==
In 2008, Kobayashi and Maskawa shared one half of the [[Nobel Prize in Physics]] "for the discovery of the origin of the broken symmetry which predicts the existence of at least three families of quarks in nature".<ref>
{{cite press release |publisher=[[The Nobel Foundation]] |date=7 October 2008 |title=The Nobel Prize in Physics 2008 |url=http://nobelprize.org/nobel_prizes/physics/laureates/2008/press.html |access-date=2009-11-24 |df=dmy-all}}</ref> Some physicists were reported to harbor bitter feelings about the fact that the Nobel Prize committee failed to reward the work of [[Nicola Cabibbo|Cabibbo]], whose prior work was closely related to that of Kobayashi and Maskawa.<ref>{{cite web |first=V. |last=Jamieson |date=7 October 2008 |title=Physics Nobel Snubs key Researcher |url=https://www.newscientist.com/article/dn14885-physics-nobel-snubs-key-researcher.html?DCMP=ILC-hmts&nsref=news8_head_dn14885 |work=[[New Scientist]] |access-date=2009-11-24 |df=dmy-all}}</ref> Asked for a reaction on the prize, Cabibbo preferred to give no comment.<ref>{{cite web |date=7 October 2008 |title=Nobel, l'amarezza dei fisici italiani |url=http://www.corriere.it/scienze_e_tecnologie/08_ottobre_07/nobel_fisica_italiani_traditi_d9993120-946d-11dd-a0d8-00144f02aabc.shtml |work=[[Corriere della Sera]] |language=it |access-date=2009-11-24 |df=dmy-all}}</ref>

{{Short description|Angle characterizing electroweak symmetry breaking}}
[[File:Weinberg angle (relation between coupling constants).svg|300px|thumb|Weinberg angle {{math|''θ''}}{{sub|W}}, and relation between couplings {{math|''g''}}, {{math|''{{prime|g}}''}}, and {{math|1= ''e'' = ''g'' sin ''θ''}}{{sub|W}}. Adapted from Lee (1981).<ref>{{cite journal |first=T.D. |last=Lee |year=1981 |title=Particle Physics and Introduction to Field Theory |journal=Physics Today |volume=34 |issue=12 |page=55 |doi=10.1063/1.2914386 |bibcode=1981PhT....34l..55L }}</ref>]]
[[File:Electroweak.svg|300px|right|thumb|The pattern of [[weak isospin]], {{math|''T''{{sub|3}}}}, and [[weak hypercharge]], {{math|''Y''}}{{sub|W}}, of the known elementary particles, showing electric charge, {{math|''Q''}},{{efn|The electric charge {{math|''Q''}} is distinct from the similar-appearing symbol occasionally used for momentum-transfer {{math|∆''Q''}}. This article uses {{math|∆''q''}}, but upper case is common and may occur in some graphs.}} along the Weinberg angle. The neutral Higgs field (upper left, circled) breaks the electroweak symmetry and interacts with other particles to give them mass. Three components of the Higgs field become part of the massive [[W and Z bosons]].]]

The '''weak mixing angle''' or '''Weinberg angle'''<ref>{{cite journal |last1=Glashow |first1=Sheldon |date=February 1961 |title=Partial-symmetries of weak interactions |journal=Nuclear Physics |volume=22 |issue=4 |pages=579–588 |doi=10.1016/0029-5582(61)90469-2|bibcode=1961NucPh..22..579G }}</ref> is a parameter in the Weinberg–Salam theory (by [[Steven Weinberg]] and [[Abdus Salam]]) of the [[electroweak interaction]], part of the [[Standard Model]] of particle physics, and is usually denoted as {{math|''θ''}}{{sub|W}}. It is the angle by which [[spontaneous symmetry breaking]] [[rotation matrix|rotates]] the original {{SubatomicParticle|W boson0}} and {{SubatomicParticle|B boson0}} [[vector boson]] plane, producing as a result the {{SubatomicParticle|Z boson0|link}}&nbsp;boson, and the [[photon]].<ref name=Cheng-Li-2006>{{cite book |first1=T.P. |last1=Cheng |author2-link=Ling-Fong Li |first2=L.F. |last2=Li |year=2006 |title=Gauge Theory of Elementary Particle Physics |pages=349–355 |publisher=[[Oxford University Press]] |isbn=0-19-851961-3}}</ref> Its measured value is slightly below 30°, but also varies, very slightly increasing, depending on how high the relative momentum of the particles involved in the interaction is that the angle is used for.<ref name=wein/>

== Details ==
The algebraic formula for the combination of the {{SubatomicParticle|W boson0}} and {{SubatomicParticle|B boson0}} [[vector boson]]s (i.e. 'mixing') that simultaneously produces the massive {{nobr|{{SubatomicParticle|Z boson0|link}} boson}} and the massless [[photon]] ({{math|{{SubatomicParticle|photon|link}}}}) is expressed by the formula

{{in5}}<math> \begin{pmatrix}
\gamma~ \\
\textsf{Z}^0 \end{pmatrix} = \begin{pmatrix}
\quad \cos \theta_\textsf{w} & \sin \theta_\textsf{w} \\
-\sin \theta_\textsf{w} & \cos \theta_\textsf{w} \end{pmatrix} \begin{pmatrix}
\textsf{B}^0 \\
\textsf{W}^0 \end{pmatrix} .</math><ref name=Cheng-Li-2006/>

The ''weak mixing angle'' also gives the relationship between the masses of the [[W and Z bosons]] (denoted as {{math|''m''}}{{sub|W}} and {{math|''m''}}{{sub|Z}}),

{{in5}}<math> m_\textsf{Z} = \frac{m_\textsf{W}}{\,\cos\theta_\textsf{w}} \,.</math>

The angle can be expressed in terms of the {{math|SU(2){{sub|''L''}}}} and {{math|U(1){{sub|''Y''}}}} [[coupling constant|couplings]] ([[weak isospin]] {{math|''g''}} and [[weak hypercharge]] {{math|''{{prime|g}}''}}, respectively),

{{in5}}<math>\cos \theta_\textsf{w} = \frac{\quad g ~}{\ \sqrt{ g^2 + g'^{\ 2} ~}\ } \quad </math> and <math> \quad \sin \theta_\textsf{w} = \frac{\quad g' ~}{\ \sqrt{ g^2 + g'^{\ 2} ~}\ } ~.</math>

The electric charge is then expressible in terms of it, {{math|''e'' {{=}} ''g'' sin ''θ''}}{{sub|w}}{{math|&nbsp;{{=}} ''{{prime|g}}'' cos ''θ''}}{{sub|w}} (refer to the figure).

Because the value of the mixing angle is currently determined empirically, in the absence of any superseding theoretical derivation it is mathematically defined as

{{in5}}<math>\cos \theta_\textsf{w} = \frac{\ m_\textsf{W}\ }{ m_\textsf{Z} } ~.</math><ref>
{{cite book
 |last=Okun |first=L.B.
 |year=1982
 |title=Leptons and Quarks
 |page=214
 |publisher=[[North-Holland Publishing|North-Holland Physics Publishing]]
 |isbn=0-444-86924-7
}}
</ref>

The value of {{math|''θ''}}{{sub|w}} varies as a function of the [[momentum transfer]], {{math|∆''q''}}, at which it is measured. This variation, or '[[renormalization group|running]]', is a key prediction of the electroweak theory. The most precise measurements have been carried out in electron–positron collider experiments at a value of {{nowrap|{{math|∆''q''}} {{=}} 91.2 GeV''/c''}}, corresponding to the mass of the {{SubatomicParticle|Z boson0|link}}&nbsp;boson, {{math|''m''}}{{sub|Z}}.

In practice, the quantity {{math|sin{{sup|2}} ''θ''}}{{sub|w}} is more frequently used. The 2004 best estimate of {{math|sin{{sup|2}} ''θ''}}{{sub|w}}, at {{nowrap|{{math|∆''q''}} {{=}} 91.2 GeV/''c''}}, in the [[minimal subtraction scheme|{{overline|MS}} scheme]] is {{val|0.23120|0.00015}}, which is an average over measurements made in different processes, at different detectors. Atomic [[parity violation]] experiments yield values for {{math|sin{{sup|2}} ''θ''}}{{sub|w}} at smaller values of {{math|∆''q''}}, below 0.01&nbsp;GeV/''c'', but with much lower precision. In 2005 results were published from a study of [[parity violation]] in [[Møller scattering]] in which a value of {{nowrap|{{math|sin{{sup|2}} ''θ''}}{{sub|w}} {{=}} {{val|0.2397|0.0013}}}} was obtained at {{nowrap|{{math|∆''q''}} {{=}} 0.16 GeV/''c''}}, establishing experimentally the so-called 'running' of the weak mixing angle. These values correspond to a Weinberg angle varying between 28.7° and {{nowrap|29.3° ≈ 30°}}. [[LHCb]] measured in 7 and 8&nbsp;TeV proton–proton collisions an effective angle of {{nowrap|{{math|sin{{sup|2}} ''θ''}}{{su|lh=1|b=w|p=eff}} {{=}} 0.23142}},<ref>
{{cite journal
 |last1=Aaij      |first1=R.     |last2=Adeva    |first2=B.
 |last3=Adinolfi  |first3=M.     |last4=Affolder |first4=A.
 |last5=Ajaltouni |first5=Z.     |last6=Akar     |first6=S.
 |last7=Albrecht  |first7=J.     |last8=Alessio  |first8=F.
 |last9=Alexander |first9=M.     |last10=Ali     |first10=S.
 |last11=Alkhazov |first11=G.
 |display-authors=6
 |date=2015-11-27
 |title=Measurement of the forward-backward asymmetry in {{nobr|Z/{{math|γ}}<sup>∗</sup> → μ<sup>+</sup>μ<sup>−</sup>}} decays and determination of the effective weak mixing angle
 |journal=Journal of High Energy Physics
 |volume=2015  |issue=11  |page=190
 |doi=10.1007/JHEP11(2015)190
 |issn=1029-8479 |s2cid=118478870
 |hdl=1721.1/116170 |hdl-access=free
|arxiv=1509.07645}}
</ref>
though the value of {{math|∆''q''}} for this measurement is determined by the partonic collision energy, which is close to the Z&nbsp;boson mass.

CODATA 2022<ref name=wein>
{{cite web
 |title=Weak mixing angle
 |date=30 May 2024
 |series=2022 CODATA value
 |website=The NIST reference on constants, units, and uncertainty
 |publisher=[[National Institute of Standards and Technology]]
 |url=http://physics.nist.gov/cgi-bin/cuu/Value?sin2th
 |access-date=2024-05-30
}}
</ref>
gives the value

{{in5}}<math>\sin^2 \theta _\textsf{w} = 1 - \left( \frac{\ m_\textsf{W}\ }{ m_\textsf{Z} }\right)^2 = 0.22305(23) ~.</math>{{efn|
Note that at present, there is no generally accepted theory that explains ''why'' the measured value {{nowrap|{{math|''θ''}}{{sub|w}} ≈ 29°}} should be what it is. The specific value is ''not'' predicted by the [[Standard Model]]: The Weinberg angle {{math|''θ''}}{{sub|w}} is an open, free parameter, although it is constrained and predicted through other measurements of [[Standard Model]] quantities.
}}

The massless photon ({{math|{{SubatomicParticle|photon|link=yes}}}}) couples to the unbroken electric charge, {{nowrap|{{math| ''Q'' {{=}} ''T''{{sub|3}} + {{small| {{sfrac| 1 | 2 }} }}''Y''}}{{sub|w}}}}, while  the {{SubatomicParticle|Z boson0|link}}&nbsp;boson couples to the broken charge {{nowrap|{{math|''T''{{sub|3}} − ''Q'' sin{{sup|2}} ''θ''}}{{sub|w}}}}.

== Footnotes ==
{{notelist}}

== References ==
{{reflist|22em}}
 
* {{cite report |first1=J. |last1=Erler |first2=A. |last2=Freitas |year=2019 |collaboration=[[Particle Data Group]] (PDG) |title=Review of the Standard Model |orig-year=revised March 2018 |url=https://pdg.lbl.gov/2019/reviews/rpp2019-rev-standard-model.pdf}}
* {{cite report |title=E158: A precision measurement of the weak mixing angle in Møller scattering |publisher=[[Stanford University]] |department=[[Stanford Linear Accelerator]] (SLAC) |url=https://www.slac.stanford.edu/exp/e158/ }}
* {{cite report |title=Q-weak: A precision test of the Standard Model and determination of the weak charges of the quarks through parity-violating electron scattering |publisher=U.S. [[Department of Energy]] |department=[[Thomas Jefferson National Accelerator Facility|Jefferson National Accelerator Lab.]] |url=https://www.jlab.org/qweak/}}

===

{{Short description|Model of neutrino oscillation}}
{{Flavour quantum numbers}}
In [[particle physics]], the '''Pontecorvo–Maki–Nakagawa–Sakata matrix''' ('''PMNS matrix'''), '''Maki–Nakagawa–Sakata matrix''' ('''MNS matrix'''), '''[[lepton]] mixing matrix''', or '''[[neutrino]] mixing matrix''' is a [[unitary matrix|unitary]]{{efn|
Note however, that the PMNS matrix is ''not'' unitary in the [[seesaw mechanism|seesaw model]].
}}
[[mixing angle|mixing matrix]] that contains information on the mismatch of [[quantum state]]s of [[neutrino]]s when they propagate freely and when they take part in [[weak interaction]]s. It is a model of [[neutrino oscillation]]. This matrix was introduced in 1962 by [[Ziro Maki]], [[Masami Nakagawa]], and [[Shoichi Sakata]],<ref>
{{cite journal
 |first1=Z. |last1=Maki 
 |first2=M. |last2=Nakagawa
 |first3=S. |last3=Sakata
 |year=1962
 |title=Remarks on the unified model of elementary particles
 |journal=[[Progress of Theoretical Physics]]
 |volume=28 |issue=5 |page=870
 |bibcode=1962PThPh..28..870M
 |doi=10.1143/PTP.28.870 |doi-access=free
}}
</ref>
to explain the neutrino oscillations predicted by [[Bruno Pontecorvo]].<ref>
{{cite journal
 |last=Pontecorvo |first=B.
 |year=1957
 |title=Inverse beta processes and nonconservation of lepton charge
 |journal=[[Zhurnal Éksperimental'noĭ i Teoreticheskoĭ Fiziki]]
 |volume=34 |page=247
}}
reproduced and translated in
{{cite journal
 |last=Pontecorvo |first=B.
 |year=1958
 |title=[no title cited]
 |journal=[[Soviet Physics JETP]]
 |volume=7 |page=172
}}
</ref>

== PMNS matrix ==
The [[Standard Model]] of particle physics contains three [[generation (particle physics)|generations]] or "[[Flavour (physics)|flavors]]" of neutrinos, {{tmath|1= \nu_\mathrm{e} }}, {{tmath|1= \nu_\mu }}, and {{tmath|1= \nu_\tau }}, each labeled with a subscript showing the charged [[lepton]] that it partners with in the [[W boson exchange|charged-current weak interaction]]. These three [[eigenstates]] of the weak interaction form a complete, [[orthonormal basis]] for the Standard Model neutrino. Similarly, one can construct an [[eigenbasis]] out of three neutrino states of definite mass, {{tmath|1= \nu_1 }}, {{tmath|1= \nu_2 }}, and {{tmath|1= \nu_3 }}, which diagonalize the neutrino's free-particle [[Hamiltonian (quantum mechanics)|Hamiltonian]]. Observations of neutrino oscillation established experimentally that for neutrinos, as for [[quarks]], these two eigenbases are different – they are 'rotated' relative to each other.

Consequently, each flavor eigenstate can be written as a combination of mass eigenstates, called a "[[Quantum superposition|superposition]]", and vice versa. The PMNS matrix, with components <math>U_{\alpha\,i}</math> corresponding to the amplitude of mass eigenstate {{nowrap|<math>i = </math> 1, 2, 3}} in terms of flavor <math> \alpha = </math> "{{math|e}}", "{{math|μ}}", "{{math|τ}}"; parameterizes the unitary transformation between the two bases:
: <math>\begin{bmatrix} ~ \nu_\mathrm{e} \\ ~ \nu_\mu \\ ~ \nu_\tau ~ \end{bmatrix} 
= \begin{bmatrix} ~ U_{\mathrm{e} 1} ~ & ~ U_{\mathrm{e} 2} ~ & ~ U_{\mathrm{e} 3} \\ ~ U_{\mu 1} & ~ U_{\mu 2} ~ & ~ U_{\mu 3} \\ ~ U_{\tau 1} ~ & ~ U_{\tau 2} ~ & ~ U_{\tau 3} \end{bmatrix} \begin{bmatrix} ~ \nu_1 \\ ~ \nu_2 \\ ~ \nu_3 ~ \end{bmatrix} ~.</math>

The vector on the left represents a generic neutrino expressed in the flavor-eigenstate basis, and on the right is the PMNS matrix multiplied by a vector representing that same neutrino in the mass-eigenstate basis. A neutrino of a given flavor <math>\alpha</math> is thus a "blend" of neutrinos with distinct masses: If one could measure directly that neutrino's mass, it would be found to have mass <math>m_i</math> with probability {{tmath|1= \left\vert U_{\alpha\,i}\right\vert^2 }}.

The PMNS matrix for [[antineutrino]]s is identical to the matrix for neutrinos under [[CPT symmetry]].

Due to the difficulties of [[neutrino detector|detecting neutrino]]s, it is much more difficult to determine the individual coefficients than for the equivalent matrix for the quarks (the [[Cabibbo–Kobayashi–Maskawa matrix|CKM matrix]]).

=== Assumptions ===

==== Standard Model ====
In the Standard Model, the PMNS matrix is [[unitarity (physics)|unitary]]. This implies that the sum of the squares of the values in each row and in each column, which represent the probabilities of different possible events given the same starting point, add up to 100%.

In the simplest case, the Standard Model posits three generations of neutrinos with Dirac mass that oscillate between three neutrino mass eigenvalues, an assumption that is made when best fit values for its parameters are calculated.

==== Other models ====
In other models the PMNS matrix is not necessarily unitary, and additional parameters are necessary to describe all possible neutrino mixing parameters in other models of neutrino oscillation and mass generation, such as the see-saw model, and in general, in the case of neutrinos that have [[Majorana fermion|Majorana mass]] rather than [[Dirac fermion|Dirac mass]].

There are also additional mass parameters and mixing angles in a simple extension of the PMNS matrix in which there are more than three flavors of neutrinos, regardless of the character of neutrino mass. As of July&nbsp;2014, scientists studying neutrino oscillation are actively considering fits of the experimental neutrino oscillation data to an extended PMNS matrix with a fourth, light "sterile" neutrino and four mass eigenvalues, although the current experimental data tends to disfavor that possibility.<ref>
{{cite journal
 |first=Boris |last=Kayser
 |title=Are there sterile neutrinos?
 |journal=Dark Matter
 |pages=201–203
 |date=February 13, 2014
 |arxiv=1402.3028 |doi=10.1063/1.4883431
 |series=AIP Conference Proceedings
 |volume=1604
 |issue=1
 |bibcode=2014AIPC.1604..201K
 |s2cid=119182490
 |df=dmy-all
}}
</ref><ref>
{{cite journal
 |first1=Arman  |last1=Esmaili     |first2=Ernesto |last2=Kemp
 |first3=O.L.G. |last3=Peres       |first4=Zahra   |last4=Tabrizi
 |date=30 Oct 2013
 |title=Probing light sterile neutrinos in medium baseline reactor experiments
 |journal=[[Physical Review D]]
 |volume=88 |issue=7 |article-number=073012
 |arxiv=1308.6218 |doi=10.1103/PhysRevD.88.073012
 |bibcode=2013PhRvD..88g3012E
|s2cid=119208413 }}
</ref><ref>
{{cite journal
 |first1=F.P. |last1=An
 |display-authors=etal
 |collaboration=Daya Bay collaboration
 |date=27 July 2014
 |title=Search for a light sterile neutrino at Daya Bay
 |journal=Physical Review Letters
 |volume=113 |issue=14 |article-number=141802
 |arxiv=1407.7259 |doi=10.1103/PhysRevLett.113.141802
 |pmid=25325631
 |bibcode=2014PhRvL.113n1802A  |s2cid=10500157
 |df=dmy-all
}}
</ref>

=== Parameterization ===
In general, there are nine degrees of freedom in any unitary three by three matrix. However, in the case of the PMNS matrix, five of those real parameters can be absorbed as phases of the lepton fields and thus the PMNS matrix can be fully described by four free parameters.<ref>
{{cite journal
 |last=Valle  |first=J.W.F.
 |year=2006
 |title=Neutrino physics overview
 |journal=[[Journal of Physics: Conference Series]]
 |volume=53 |issue=1 |pages=473–505
 |arxiv=hep-ph/0608101 |bibcode=2006JPhCS..53..473V
 |doi=10.1088/1742-6596/53/1/031
 |s2cid=2094005
}}
</ref>
The PMNS matrix is most commonly parameterized by three mixing angles ({{tmath|1= \theta_{12} }}, {{tmath|1= \theta_{23} }}, and {{tmath|1= \theta_{13} }}) and a single phase angle called <math>\delta_{\mathrm{CP}}</math> related to [[CP violation|charge–parity violations]] (i.e. differences in the rates of oscillation between two states with opposite starting points, which makes the order in time in which events take place necessary to predict their oscillation rates), in which case the matrix can be written as:
: <math> \begin{align} & \begin{bmatrix} 1 & 0 & 0 \\ 0 & c_{23} & s_{23} \\ 0 & -s_{23} & c_{23} \end{bmatrix}
 \begin{bmatrix} c_{13} & 0 & s_{13}e^{-i\delta_\mathrm{CP}} \\ 0 & 1 & 0 \\ -s_{13}e^{i\delta_\mathrm{CP}} & 0 & c_{13} \end{bmatrix}
 \begin{bmatrix} c_{12} & s_{12} & 0 \\ -s_{12} & c_{12} & 0 \\ 0 & 0 & 1 \end{bmatrix} \\
 & = \begin{bmatrix} c_{12}c_{13} & s_{12} c_{13} & s_{13}e^{-i\delta_\mathrm{CP}} \\
 -s_{12}c_{23} - c_{12}s_{23}s_{13}e^{i\delta_\mathrm{CP}} & c_{12}c_{23} - s_{12}s_{23}s_{13}e^{i\delta_\mathrm{CP}} & s_{23}c_{13}\\
 s_{12}s_{23} - c_{12}c_{23}s_{13}e^{i\delta_\mathrm{CP}} & -c_{12}s_{23} - s_{12}c_{23}s_{13}e^{i\delta_\mathrm{CP}} & c_{23}c_{13} \end{bmatrix}, \end{align} </math>
where <math>s_{ij}</math> and <math>c_{ij}</math> are used to denote <math>\sin\theta_{ij}</math> and <math>\cos\theta_{ij}</math> respectively. In the case of Majorana neutrinos, two extra complex phases are needed, as the phase of Majorana fields cannot be freely redefined due to the condition {{tmath|1= \nu = \nu^c }}. An infinite number of possible parameterizations exist; one other common example being the [[Cabibbo–Kobayashi–Maskawa matrix#Wolfenstein parameters|Wolfenstein parameterization]].

The mixing angles have been measured by a variety of experiments (see [[neutrino mixing]] for a description). The CP-violating phase <math>\delta_\mathrm{CP}</math> has not been measured directly, but estimates can be obtained by fits using the other measurements.

The absolute values of the elements of the PMNS matrix can be reconstructed from the neutrino mass matrix using an [[Eigenvalues and eigenvectors|eigenvector-eigenvalue identity]]. The PMNS matrix admits an algebraic representation in terms of the neutrino mass matrix and its [[Frobenius covariant]]. The neutrino-related amplitudes can be expressed using either the mixing matrix or the mass matrix.

===Experimentally measured parameter values===
As of November&nbsp;2022, the current best-fit values from Nu-FIT.org, from direct and indirect measurements, using normal ordering, are:<ref name=NuFIT>{{cite web |first1=Ivan |last1=Esteban |first2=Concha |last2=Gonzalez Garcia |first3=Michele |last3=Maltoni |first4=Thomas |last4=Schwetz |first5=Zhou|last5=Albert |edition=NuFIT 5.2 |website=NuFIT.org |url=http://www.nu-fit.org/?q=node/256 |series=Three-neutrino fit |title=Parameter ranges |date=November 2022 |access-date=2023-03-29 |df=dmy-all}}</ref>
For September&nbsp;2024 data, see NuFIT6<ref name=NuFIT6>{{cite web |first1=Ivan |last1=Esteban |first2=Concha |last2=Gonzalez Garcia |first3=Michele |last3=Maltoni |first4=Ivan |last4=Martinez-Soler |first5=João Paulo|last5=Pinheiro |first6=Thomas |last6=Schwetz |edition=NuFIT 6.0 |website=NuFIT.org |url=http://www.nu-fit.org/?q=node/294 |series=Three-neutrino fit |title=Parameter ranges |date=September 2024 |access-date=2024-12-10 |df=dmy-all}}</ref>
: <math>
\begin{align}
\theta_{12} & =  {33.41^\circ}^{+0.75^\circ}_{-0.72^\circ} \\
\theta_{23} & =  {49.1^\circ}^{+1.0^\circ}_{-1.3^\circ}\\
\theta_{13} & =  {8.54^\circ}^{+0.11^\circ}_{-0.12^\circ}  \\
\delta_{\textrm{CP}} & = {197^\circ}^{+42^\circ}_{-25^\circ} \\
\end{align}
</math>

As of November&nbsp;2022, the 3&nbsp;{{mvar|&sigma;}} ranges (99.7% confidence) for the magnitudes of the elements of the matrix were:<ref name=NuFIT/>

: <math>
|U| = \begin{bmatrix}
  ~ |U_{\mathrm{e} 1}|    ~ & |U_{\mathrm{e} 2}|    ~ & |U_{\mathrm{e} 3}|    \\
  ~ |U_{\mu 1}|  ~ & |U_{\mu 2}|  ~ & |U_{\mu 3}|  \\
  ~ |U_{\tau 1}| ~ & |U_{\tau 2}| ~ & |U_{\tau 3}|  ~ 
\end{bmatrix} = \left[\begin{array}{rrr}
   ~ 0.803 \sim 0.845 ~~ &  0.514 \sim 0.578 ~~ &  0.142 \sim 0.155 ~ \\
   ~ 0.233 \sim 0.505 ~~ &  0.460 \sim 0.693 ~~ &  0.630 \sim 0.779 ~ \\
   ~ 0.262 \sim 0.525 ~~ &  0.473 \sim 0.702 ~~ &  0.610 \sim 0.762 ~
\end{array}\right]
</math>

<!-- ———————————————— preserved 2014 values ———————————————— 
<ref>
{{cite web
 |last1=Gonzalez-Garcia |first1=M. C. |last2=Maltoni |first2=M.
 |last3=Salvado |first3=J. |last4=Schwetz |first4=T.
 |title=NuFit 1.3
 |date=June 2014
 |url=http://www.nu-fit.org/?q=node/75
 |access-date=2014-07-09
}}
</ref>
:<math>
\begin{align}
\theta_{12} [^\circ] & = 33.36^{+0.81}_{-0.78} \\
\theta_{23} [^\circ] & = 40.0^{+2.1}_{-1.5}~\textrm{or}~50.4^{+1.3}_{-1.3} \\
\theta_{13} [^\circ] & = 8.66^{+0.44}_{-0.46}  \\
\delta_{\textrm{CP}} [^\circ] & = -60^{+66}_{-138} \\
\end{align}
</math>

So the current matrix will be:
<math>
U = \begin{bmatrix}
  |U_{\mathrm{e} 1}|    & |U_{\mathrm{e} 2}|    & |U_{\mathrm{e} 3}| \\
  |U_{\mu 1}|  & |U_{\mu 2}|  & |U_{\mu 3}| \\
  |U_{\tau 1}| & |U_{\tau 2}| & |U_{\tau 3}|
\end{bmatrix} = \left[\begin{array}{rrr}
   0.82 \pm 0.01 &  0.54 \pm 0.02 & -0.15 \pm 0.03 \\
  -0.35 \pm 0.06 &  0.70 \pm 0.06 &  0.62 \pm 0.06 \\
   0.44 \pm 0.06 & -0.45 \pm 0.06 &  0.77 \pm 0.06
\end{array}\right]
 </math>
 ———————————————— end old 2014 values ————————————————  -->

; Notes regarding the best fit parameter values :
* These best fit values imply that there is much more neutrino mixing than there is mixing between the quark flavors in the CKM matrix (in the CKM matrix, the corresponding mixing angles are {{nobr|<math>\theta_{12} = </math> {{val|13.04|0.05|u=deg}} ,}} {{nobr|<math>\theta_{23} = </math> {{val|2.38|0.06|u=deg}}}}, {{nobr|<math>\theta_{13} = </math> {{val|0.201|0.011|u=deg}} ).}}
* These values are inconsistent with [[tribimaximal mixing|tribimaximal neutrino mixing]] (i.e. {{nobr|<math> \theta_{12} \approx </math> 35.3° ,}} {{nobr|<math> \theta_{23} = </math> 45° ,}} {{nobr|<math> \theta_{13} = </math> 0° )}} at a statistical significance of more than five standard deviations. Tribimaximal neutrino mixing was a common assumption in theoretical physics papers analyzing neutrino oscillation before more precise measurements were available.
* The value of {{nobr|<math> \delta_{\textrm{CP} } = </math> {{val|197|42|25|u=deg}} }} is very difficult to measure, and is the object of ongoing research; however the current constraint {{nobr| 169° <math> \le \delta_{\textrm{CP} } \le </math> 246° ,}} closer to 180° and farther from 0° (or 360°), shows a clear bias towards [[CP violation|charge-parity violation]].
