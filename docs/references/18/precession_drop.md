{{short description|Periodic change in the direction of a rotation axis}}
{{Other uses}}
{{redir-dist|Precission|Precision (disambiguation){{!}}Precision|Recission (disambiguation){{!}}Recission}}
[[File:Gyroscope precession.gif|thumb|Precession of a [[gyroscope]]{{clarify|reason=This image, while it moves, does not clarify which aspect of the gyroscope is demonstrating precession! This could be improved with a caption but would be better improved by a label in the image.|date=November 2022}}]]
[[File:Praezession.svg|thumb|{{legend-line|green solid 2px|[[Rotation]]}} {{legend-line|blue solid 2px|Precession}}{{legend-line|red solid 2px|[[Nutation]] in [[obliquity]] of a planet|inline=yes}}]] 

'''Precession''' is a change in the [[orientation (geometry)|orientation]] of the rotational{{nbsp}}axis of a [[rotating]]{{nbsp}}body. If the axis of rotation of a body is itself rotating about a second axis, that body is said to be precessing about the second{{nbsp}}axis. In an appropriate [[reference frame|reference{{nbsp}}frame]] it can be defined as a change in the first [[Euler angle|Euler{{nbsp}}angle]], whereas the third Euler{{nbsp}}angle defines the [[rotation around a fixed axis|rotation itself]]. A motion in which the second Euler{{nbsp}}angle changes is called ''[[nutation]]''. In [[physics]], there are two types of precession: {{nowr|[[torque]]-free}} and {{nowr|torque-induced}}.

In astronomy, '''precession''' refers to any of several slow changes in an astronomical{{nbsp}}body's rotational or orbital parameters. An important example is the steady change in the orientation of the axis of [[rotation of the Earth|rotation{{nbsp}}of the Earth]], known as the [[precession of the equinoxes|precession{{nbsp}}of the equinoxes]].

==Torque-free or torque neglected==
Torque-free precession implies that no external moment ([[torque]]) is applied to the body. In {{nowr|torque-free}} precession, the [[angular momentum]] is a constant, but the [[angular velocity]] [[vector (physics)|vector]] changes orientation with time. What makes this possible is a {{nowr|time-varying}} [[moment of inertia|moment{{nbsp}}of inertia]], or more precisely, a {{nowr|time-varying}} [[inertia matrix]]. The inertia matrix is composed of the moments{{nbsp}}of inertia of a body calculated with respect to separate [[Basis (linear algebra)|coordinate{{nbsp}}axes]] (e.g.{{nbsp}}{{math|''x''}}, {{math|''y''}}, {{math|''z''}}). If an object is asymmetric about its principal axis{{nbsp}}of rotation, the moment{{nbsp}}of inertia with respect to each coordinate direction will change with time, while preserving angular momentum. The result is that the [[Vector component#Decomposition|component]] of the angular velocities of the body about each axis will vary inversely with each axis' moment{{nbsp}}of inertia.

The torque-free precession rate of an object with an axis of symmetry, such as a disk, spinning about an axis not aligned with that axis{{nbsp}}of symmetry can be calculated as follows:<ref>{{Citation|author1-link=Hanspeter Schaub| last1=Schaub| first1=Hanspeter |last2=Junkins |first2=John L. |author2-link=John L. Junkins |editor-last=Schetz |editor-first=Joseph A. |year=2003 | title=Analytical Mechanics of Space Systems | publisher=[[American Institute of Aeronautics and Astronautics]] | publication-place=[[Reston, VA]] |series=AIAA Education Series |isbn=978-1-60086-027-0 |oclc=637318824 | pages=149–150 | url=https://books.google.com/books?id=qXvESNWrfpUC}}</ref>
<math display="block">\boldsymbol\omega_\mathrm{p} = \frac{\boldsymbol I_\mathrm{s} \boldsymbol\omega_\mathrm{s} } {\boldsymbol I_\mathrm{p} \cos(\boldsymbol \alpha)}</math>
where {{math|'''''ω'''''<sub>p</sub>}} is the precession rate, {{math|'''''ω'''''<sub>s</sub>}} is the spin rate about the axis of symmetry, {{math|'''''I'''''<sub>s</sub>}} is the moment of inertia about the axis of symmetry, {{math|'''''I'''''<sub>p</sub>}} is moment of inertia about either of the other two equal perpendicular principal axes, and {{mvar|'''α'''}} is the angle between the moment of inertia direction and the symmetry axis.<ref>{{cite web| url = https://www.sfu.ca/~boal/211lecs/211lec26.pdf| title = Lecture 26 – Torque-free rotation – body-fixed axes| first = David| last = Boal| year = 2001 | access-date = 2008-09-17}}</ref>

When an object is not perfectly [[Rigid body dynamics|rigid]], inelastic dissipation will tend to damp {{nowr|torque-free}} precession,<ref>{{cite journal |doi=10.1111/j.1365-2966.2005.08864.x |title=Nutational damping times in solids of revolution |journal=Monthly Notices of the Royal Astronomical Society |volume=359 |issue=1 |page=79 |year=2005 |last1=Sharma |first1=Ishan |last2=Burns |first2=Joseph A. |last3=Hui |first3=C.-H. |doi-access=free |bibcode=2005MNRAS.359...79S }}</ref> and the rotation axis will align itself with one of the inertia axes of the body.

For a generic solid object without any axis of symmetry, the evolution of the object's orientation, represented (for example) by a rotation matrix {{mvar|'''R'''}} that transforms internal to external coordinates, may be numerically simulated. Given the object's fixed internal [[moment of inertia tensor]] {{math|'''''I'''''<sub>0</sub>}} and fixed external angular momentum {{mvar|'''L'''}}, the instantaneous angular velocity is
<math display="block">\boldsymbol\omega\left(\boldsymbol R\right) = \boldsymbol R \boldsymbol I_0^{-1} \boldsymbol R ^T \boldsymbol L</math>
Precession occurs by repeatedly recalculating {{mvar|'''ω'''}} and applying a small [[Rotation representation (mathematics)#Euler axis and angle (rotation vector)|rotation vector]] {{math|'''''ω''' dt''}} for the short time {{math|''dt''}}; e.g.:
<math display="block">\boldsymbol R_\text{new} = \exp\left(\left[\boldsymbol\omega\left(\boldsymbol R_\text{old}\right)\right]_{\times} dt\right) \boldsymbol R_\text{old}</math>
for the [[Cross product#Conversion to matrix multiplication|skew-symmetric matrix]] {{math|['''''ω''''']<sub>×</sub>}}. The errors induced by finite time steps tend to increase the rotational kinetic energy:
<math display="block">E\left(\boldsymbol R\right) = \boldsymbol \omega\left(\boldsymbol R\right) \cdot \frac{\boldsymbol L }{ 2}</math>
this unphysical tendency can be counteracted by repeatedly applying a small rotation vector {{mvar|'''v'''}} perpendicular to both {{mvar|'''ω'''}} and {{mvar|'''L'''}}, noting that
<math display="block">E\left(\exp\left(\left[\boldsymbol v\right]_{\times}\right) \boldsymbol R\right) \approx E\left(\boldsymbol R\right) + \left(\boldsymbol \omega\left(\boldsymbol R\right) \times \boldsymbol L\right) \cdot \boldsymbol v</math>

==Torque-induced==
Torque-induced precession ('''gyroscopic precession''') is the phenomenon in which the [[axis of rotation|axis]] of a spinning object (e.g., a [[gyroscope]]) describes a [[Cone (geometry)|cone]] in space when an external [[torque]] is applied to it. The phenomenon is commonly seen in a [[spinning top|spinning toy top]], but all rotating objects can undergo precession. If the [[speed]] of the rotation and the [[Magnitude (mathematics)|magnitude]] of the external torque are constant, the spin axis will move at [[right angle]]s to the [[Direction (geometry, geography)|direction]] that would intuitively result from the external torque. In the case of a toy top, its weight is acting downwards from its [[center of mass|center{{nbsp}}of mass]] and the [[normal force|normal{{nbsp}}force]] (reaction) of the ground is pushing up on it at the point of contact with the support. These two opposite forces produce a torque which causes the top to precess.

[[Image:Gyroscopic precession 256x256.png|frame|right|The response of a rotating system to an applied torque. When the device swivels, and some roll is added, the wheel tends to pitch.]]
The device depicted on the right is [[gimbal]]-mounted. From inside to outside there are three axes of rotation: the hub of the wheel, the gimbal axis, and the vertical pivot.

To distinguish between the two horizontal axes, rotation around the wheel hub will be called ''spinning'', and rotation around the gimbal axis will be called ''pitching''. Rotation around the vertical pivot axis is called ''rotation''.

First, imagine that the entire device is rotating around the (vertical) pivot axis. Then, spinning of the wheel (around the wheelhub) is added. Imagine the gimbal axis to be locked, so that the wheel cannot pitch. The gimbal axis has sensors, that measure whether there is a [[torque]] around the gimbal axis.

In the picture, a section of the wheel has been named {{math|''dm''<sub>1</sub>}}. At the depicted moment in time, section {{math|''dm''<sub>1</sub>}} is at the [[perimeter]] of the rotating motion around the (vertical) pivot axis. Section {{math|''dm''<sub>1</sub>}}, therefore, has a lot of angular rotating [[velocity]] with respect to the rotation around the pivot axis, and as {{math|''dm''<sub>1</sub>}} is forced closer to the pivot axis of the rotation (by the wheel spinning further), because of the [[Coriolis effect]], with respect to the vertical pivot axis, {{math|''dm''<sub>1</sub>}} tends to move in the direction of the top-left arrow in the diagram (shown at 45°) in the direction of rotation around the pivot axis.<ref name="Teodorescu">{{cite book | last = Teodorescu | first = Petre P | title = Mechanical Systems, Classical Models: Volume II: Mechanics of Discrete and Continuous Systems | publisher = Springer Science & Business Media | date = 2002 | page = 420 | url = https://books.google.com/books?id=aXCBlHOtO3kC&pg=PA396 | isbn = 978-1-4020-8988-6}}</ref>  Section {{math|''dm''<sub>2</sub>}} of the wheel is moving away from the pivot axis, and so a force (again, a Coriolis force) acts in the same direction as in the case of {{math|''dm''<sub>1</sub>}}. Note that both arrows point in the same direction.

The same reasoning applies for the bottom half of the wheel, but there the arrows point in the opposite direction to that of the top arrows. Combined over the entire wheel, there is a torque around the gimbal axis when some spinning is added to rotation around a vertical axis.

The torque around the gimbal axis arises without any delay; the response is instantaneous.

In the discussion above, the setup was kept unchanging by preventing pitching around the gimbal axis. In the case of a spinning toy top, when the spinning top starts tilting, gravity exerts a torque. However, instead of rolling over, the spinning top just pitches a little. This pitching motion reorients the spinning top with respect to the torque that is being exerted. The result is that the torque exerted by gravity{{snd}} via the pitching motion{{snd}} elicits gyroscopic precession (which in turn yields a counter torque against the gravity torque) rather than causing the spinning top to fall to its side.

Precession or gyroscopic considerations have an effect on [[bicycle]] performance at high speed. Precession is also the mechanism behind [[gyrocompass]]es.

===Classical (Newtonian)===
[[File:PrecessionOfATop.svg|thumb|right|256px|The [[torque]] caused by the normal force – {{math|'''F'''<sub>g</sub>}} and the weight of the top causes a change in the [[angular momentum]] {{math|'''L'''}} in the direction of that torque. This causes the top to precess.]]

Precession is the change of [[angular velocity]] and [[angular momentum]] produced by a torque. The general equation that relates the torque to the rate of change of angular momentum is:
<math display="block">\boldsymbol{\tau} = \frac{\mathrm{d}\mathbf{L}}{\mathrm{d}t}</math>
where <math>\boldsymbol{\tau}</math> and <math>\mathbf{L}</math> are the torque and angular momentum vectors respectively.

Due to the way the torque vectors are defined, it is a vector that is perpendicular to the plane of the forces that create it. Thus it may be seen that the angular momentum vector will change perpendicular to those forces. Depending on how the forces are created, they will often rotate with the angular momentum vector, and then circular precession is created.

Under these circumstances the angular velocity of precession is given by: <ref>{{cite book |last1=Moebs |first1=William |last2=Ling |first2=Samuel J. |last3=Sanny |first3=Jeff |title=11.4 Precession of a Gyroscope - University Physics Volume 1 {{!}} OpenStax |date=Sep 19, 2016 |location=Houston, Texas |url=https://openstax.org/books/university-physics-volume-1/pages/11-4-precession-of-a-gyroscope |access-date=23 October 2020 |language=en}}</ref>

:<math>\boldsymbol\omega_\mathrm{p} = \frac{\ mgr}{I_\mathrm{s}\boldsymbol\omega_\mathrm{s}} = \frac{ \tau}{I_\mathrm{s}\boldsymbol\omega_\mathrm{s}\sin(\theta)}</math>

where {{math|''I''<sub>s</sub>}} is the [[moment of inertia]], {{math|'''''ω'''''<sub>s</sub>}} is the angular velocity of spin about the spin axis, {{mvar|m}} is the mass, {{math|''g''}} is the acceleration due to gravity, {{mvar|θ}} is the angle between the spin axis and the axis of precession and {{math|''r''}} is the distance between the center of mass and the pivot.  The torque vector originates at the center of mass. Using {{math|1='''''ω''''' = {{sfrac|2π|''T''}}}}, we find that the [[Frequency|period]] of precession is given by:<ref>{{cite book |last1=Moebs |first1=William |last2=Ling |first2=Samuel J. |last3=Sanny |first3=Jeff |title=11.4 Precession of a Gyroscope - University Physics Volume 1 {{!}} OpenStax |date=Sep 19, 2016 |location=Houston, Texas |url=https://openstax.org/books/university-physics-volume-1/pages/11-4-precession-of-a-gyroscope |access-date=23 October 2020 |language=en}}</ref>
<math display="block">T_\mathrm{p} = \frac{4\pi^2 I_\mathrm{s}}{\ mgrT_\mathrm{s}} = \frac{4\pi^2 I_\mathrm{s}\sin(\theta)}{\ \tau T_\mathrm{s}}</math>

Where {{math|''I''<sub>s</sub>}} is the [[moment of inertia]], {{math|''T''<sub>s</sub>}} is the period of spin about the spin axis, and {{mvar|'''τ'''}} is the [[torque]]<!-- Torque is not introduced -->. In general, the problem is more complicated than this, however.

===Relativistic (Einsteinian) ===
The special and general theories of [[Theory of relativity|relativity]] give three types of corrections to the Newtonian precession, of a gyroscope near a large mass such as Earth, described above. They are:
* [[Thomas precession]], a special-relativistic correction accounting for an object (such as a gyroscope) being accelerated along a curved path.
* [[Geodetic effect|de Sitter precession]], a general-relativistic correction accounting for the {{langr|de|Schwarzschild}} metric of curved space near a large non-rotating mass.
* [[Lense–Thirring precession]], a general-relativistic correction accounting for the frame dragging by the Kerr{{nbsp}}metric of curved space near a large rotating mass.

The [[Schwarzschild geodesics|{{langr|de|Schwarzschild|cat=no}} geodesics]] (sometimes {{langr|de|Schwarzschild}} precession) are used in the prediction of the [[anomalous perihelion precession]] of the planets, most notably for the accurate prediction of the [[apsidal precession]] of Mercury.

== Astronomy ==
In astronomy, precession refers to any of several gravity-induced, slow and continuous changes in an astronomical body's rotational axis or orbital path. Precession{{nbsp}}of the equinoxes, perihelion precession, changes in the [[tilt of Earth's axis|tilt{{nbsp}}of Earth's axis]] to its orbit, and the [[Orbital eccentricity|eccentricity]] of its orbit over tens{{nbsp}}of thousands of years are all important parts of the astronomical theory of [[ice age|ice{{nbsp}}age]]s. {{xref|(See [[Milankovitch cycles|{{langr|sr-Latn|Milankovitch|cat=no}}{{nbsp}}cycles]].)}}

=== Axial precession (precession of the equinoxes) ===
{{Main|Axial precession}}

Axial precession is the movement of the rotational axis of an astronomical body, whereby the axis slowly traces out a cone. In the case of Earth, this type of precession is also known as the ''precession{{nbsp}}of the equinoxes'', ''lunisolar precession'', or ''precession{{nbsp}}of the equator''. Earth goes through one such complete precessional cycle in a period of approximately 26,000{{nbsp}}years or 1° every 72{{nbsp}}years, during which the positions of stars will slowly change in both [[equatorial coordinates]] and [[ecliptic longitude]]. Over this cycle, [[Earth's north axial pole]] moves from where it is now, within 1° of [[Polaris]], in a circle around the [[ecliptic pole|ecliptic{{nbsp}}pole]], with an angular radius of about{{nbsp}}23.5°.

The [[Greek astronomy|ancient Greek astronomer]] {{langr|grc-Latn|[[Hipparchus]]}}{{nbsp}}({{c.|190–120{{nbsp}}BC}}) is generally accepted to be the earliest known astronomer to recognize and assess the precession{{nbsp}}of the equinoxes at about 1° per{{nbsp}}century (which is not far from the actual value for antiquity, 1.38°),<ref>{{cite book |last=Barbieri |first=Cesare |title=Fundamentals of Astronomy |year=2007 |publisher=Taylor and Francis Group |location=New York |isbn=978-0-7503-0886-1 |page=71 }}</ref> although there is some minor dispute about whether he was.<ref>{{cite book |last = Swerdlow |first = Noel |title = On the cosmical mysteries of Mithras |publisher = Classical Philology, 86, (1991), 48–63 |date = 1991 |page = 59}}</ref> In [[ancient China]], the [[Jin dynasty (265–420)|Jin dynasty]] scholar-official {{langr|zh-Latn|[[Yu Xi]]}}{{nbsp}}({{fl.|307–345{{nbsp}}AD}}) made a similar discovery centuries later, noting that the position of the Sun during the [[winter solstice]] had drifted roughly one degree over the course of fifty years relative to the position of the stars.<ref>Sun, Kwok. (2017). ''Our Place in the Universe: Understanding Fundamental Astronomy from Ancient Discoveries'', second edition. Cham, Switzerland: Springer. {{ISBN|978-3-319-54171-6}}, p. 120; see also Needham, Joseph; Wang, Ling. (1995) [1959]. ''Science and Civilization in China: Mathematics and the Sciences of the Heavens and the Earth'', vol. 3, reprint edition. Cambridge: Cambridge University Press. {{ISBN|0-521-05801-5}}, p. 220.</ref> The precession of Earth's axis was later explained by [[classical mechanics|Newtonian physics]]. Being an [[oblate spheroid]], Earth has a {{nowr|non-spherical}} shape, bulging outward at the equator. The gravitational [[tidal force]]s of the [[Moon]] and [[Sun]] apply torque to the equator, attempting to pull the [[equatorial bulge]] into the plane of the [[ecliptic]], but instead causing it to precess. The torque exerted by the planets, particularly [[Jupiter]], also plays a role.<ref name="Bradt">{{cite book |last = Bradt |first = Hale |title = Astronomy Methods |publisher = [[Cambridge University Press]] |date = 2007 |pages = 66 |isbn = 978-0-521-53551-9}}</ref>

{{multiple image
 |direction = horizontal
 |align= center
 |width1= 158
 |width2= 308
 |width3= 180
 |alt3=Small white disks representing the northern stars on a black background, overlaid by a circle showing the position of the north pole over time 
 |image1=Earth precession.svg
 |image2=Equinox path.png
 |image3=Precession N.gif
 |footer=Precessional movement of the axis (left), precession of the equinox in relation to the distant stars (middle), and the path of the north celestial pole among the stars due to the precession. Vega is the bright star near the bottom (right).
}}

===Apsidal precession===
[[File:Precessing Kepler orbit 280frames e0.6 smaller.gif|thumb|upright=1.25|[[Apsidal precession]]—the orbit rotates gradually over time.]]
{{main|Apsidal precession}}{{See also|Anomalous perihelion precession}}
The [[orbit]]s of planets around the [[Sun]] do not really follow an identical ellipse each time, but actually trace out a flower-petal shape because the major axis of each planet's elliptical orbit also precesses within its orbital plane, partly in response to perturbations in the form of the changing gravitational forces exerted by other planets. This is called perihelion precession or [[apsidal precession]].

In the adjunct image, Earth's apsidal precession is illustrated. As the Earth travels around the Sun, its elliptical orbit rotates gradually over time. The eccentricity of its ellipse and the precession rate of its orbit are exaggerated for visualization. Most orbits in the Solar System have a much smaller eccentricity and precess at a much slower rate, making them nearly circular and nearly stationary.

Discrepancies between the observed perihelion precession rate of the planet [[Mercury (planet)|Mercury]] and that predicted by [[classical mechanics]] were prominent among the forms of experimental evidence leading to the acceptance of [[Albert Einstein|Einstein]]'s [[Theory of Relativity]] (in particular, his [[General relativity|General Theory of Relativity]]), which accurately predicted the anomalies.<ref>[[Max Born]] (1924), ''Einstein's Theory of Relativity'' (The 1962 Dover edition, page 348 lists a table documenting the observed and calculated values for the precession of the perihelion of Mercury, Venus, and Earth.)</ref><ref>{{cite web| url = http://www.dailygalaxy.com/my_weblog/2008/03/18-billion-suns.html| title = An even larger value for a precession has been found, for a black hole in orbit around a much more massive black hole, amounting to 39 degrees each orbit.| date = 18 March 2008| access-date = 2023-11-15| archive-date = 2018-08-07| archive-url = https://web.archive.org/web/20180807131603/http://www.dailygalaxy.com/my_weblog/2008/03/18-billion-suns.html| url-status = bot: unknown}}</ref> Deviating from Newton's law, Einstein's theory of gravitation predicts an extra term of {{math|{{sfrac|''A''|''r''<sup>4</sup>}}}}, which accurately gives the observed excess turning rate of {{val|43|u=[[arcsecond]]s}} per{{nbsp}}century.

===Nodal precession===
{{main|Nodal precession}}
[[Orbital node]]s also [[nodal precession|precess]] over time.

{{for|the precession of the Moon's orbit|lunar precession}}

==See also==
* {{anl|Larmor precession}}
* {{anl|Nutation}}
* {{anl|Polar motion}}
* {{anl|Precession (mechanical)}}
* {{anl|Precession as a form of parallel transport}}
