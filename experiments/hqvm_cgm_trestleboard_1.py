#!/usr/bin/env python3
"""
hqvm_cgm_trestleboard_1.py

CGM trestleboard engine: the Trestleboard class.

Holds Square · Compass · Level · Percolation · θ(p) · fusion cross-section /
reactivity · barrier dials · deuteron strong anchor. Pure computation; no printing.

Fusion validation / resonance map: hqvm_cgm_trestleboard_4.py.
Optical / E_min / spectral audits: hqvm_cgm_trestleboard_3.py.
Nuclear selector / α/β / half-lives: NuclearBoard in hqvm_cgm_trestleboard_2.py.
Report + CLI: hqvm_cgm_trestleboard_run.py. Shared: hqvm_cgm_trestleboard_common.py.

Companion notes: hqvm_cgm_trestleboard_notes.txt.
"""
from __future__ import annotations
import math
from typing import List, Optional, Sequence, Tuple

from gyroscopic.hQVM.family import (
    bisect_p_c_rank_micro_ref,
    exact_root_rank_pmf,
    predicted_cluster_size,
    theta_micro_ref_exact,
)

from hqvm_cgm_trestleboard_common import (
    ALPHA0,
    ALPHA0_ZETA_TARGET,
    ALPHA_EM,
    C3,
    CHIRALITY_D,
    CODE_ATOMS,
    DELTA,
    DELTA_BU,
    DRESS_OPERATOR,
    EV_PER_GEV,
    D_SHELL,
    G_KERNEL,
    H_CARD,
    HOLONOMY_DRESS,
    M_A,
    M_N_MEV,
    OCTAVE_TOL_TICKS,
    OMEGA,
    OPTICAL_DILUTION,
    Q_G,
    RHO,
    S_BU,
    S_CS,
    S_ONA,
    S_P,
    S_UNA,
    SQRT5,
    U_P,
    O_P,
    TICKS_PER_K,
    TICKS_PER_OCTAVE,
    TICK_CORR,
    V_EV,
    V_GEV,
    ZETA_G,
    ClosureClass,
    CompassStep,
    CrossSectionReading,
    FusionScanResult,
    LevelReading,
    PercolationState,
    ReactivityReading,
    ReactivityScan,
    ShellCensus,
    SquareReading,
    ThresholdBoard,
    _kernel_percolation_board,
)


# -----------------------------------------------------------------
# Theta / grid helpers (exact kernel coverage, fusion grids)
# -----------------------------------------------------------------
def theta_q_class_exact(p: float, d: int, *, cond: bool = False) -> float:
    dist = exact_root_rank_pmf(p, d)
    n_omega = 1 << (2 * d)
    total = 0.0
    for r, pr in enumerate(dist):
        if r == 0:
            frac = 2.0 / n_omega  # r=0 gauge doublet: |Reach|=2
        else:
            frac = float((1 << r) ** 2) / n_omega
        total += pr * frac
    if not cond:
        return total
    p_nz = 1.0 - (1.0 - p) ** (1 << d)
    if p_nz <= 0.0:
        return 0.0
    return total / p_nz


def theta_for_protocol(p: float, d: int, protocol: str, *, cond: bool = False) -> float:
    if protocol == "micro_ref":
        return theta_micro_ref_exact(p, d, cond=cond)
    if protocol == "q6_class":
        return theta_q_class_exact(p, d, cond=cond)
    raise ValueError(f"unknown protocol: {protocol}")


def make_fusion_grid(E_G_MeV: float) -> List[float]:
    gamow_peak_keV = E_G_MeV * 1e3 / 4.0
    max_E = max(gamow_peak_keV * 1.5, 1000.0)
    points = [
        1,
        2,
        5,
        10,
        20,
        30,
        50,
        64,
        75,
        100,
        120,
        150,
        200,
        250,
        300,
        400,
        500,
        600,
        800,
        1000,
    ]
    E = 1000
    while E < max_E:
        E = int(E * 1.5)
        points.append(E)
    return sorted(set(points))


def make_reactivity_T_grid(E_G_MeV: float) -> List[float]:
    # Cover past the Gamow temperature scale so the enhancement ratio
    # can peak and fall (∫f e^{-E/T}dE is monotone in T; the falsifiable
    # gate is argmax_T of ⟨σv⟩CGM/⟨σv⟩G, not the absolute ⟨σv⟩ peak).
    max_T = max(E_G_MeV * 1e3 / 2.0, 500.0)
    points = [
        1,
        2,
        5,
        10,
        20,
        30,
        50,
        64,
        75,
        100,
        120,
        150,
        200,
        250,
        300,
        400,
        500,
        600,
        800,
        1000,
    ]
    T = 1000
    while T < max_T:
        T = int(T * 1.5)
        points.append(T)
    return sorted({p for p in points if p <= max_T})


# -----------------------------------------------------------------
# Core instrument
# -----------------------------------------------------------------
class Trestleboard:
    def __init__(
        self,
        grammar: Optional[List[ClosureClass]] = None,
        *,
        protocol: str = "micro_ref",
        chirality_d: int = CHIRALITY_D,
        fusion_model: str = "2",
        dial: str = "delta",
    ):
        """
        fusion_model:
          "1" — θ is the accessibility/tunneling factor; σ∝(S/E)·θ(p),
                p=T=exp(−τ); no separate Gamow factor (no double-count).
          "2" — standard Gamow in σ; θ is an extra factor from a non-Gamow
                dial (default p_Δ=E/V_b). σ∝(S/E)·P_Gamow·θ(p_Δ).
          "native" — pure kernel model: σ∝(1/E)·θ(p_Δ), p_Δ=E/V_b from the
                per-fuel Coulomb barrier, NO Gamow factor. Tests whether the
                exact coverage θ(p) reproduces the measured cross-section on
                its own (the proposed native CGM fusion hypothesis).
        dial: "tau" | "delta" — which dial feeds θ in Model 2 (ignored in Model 1
              and native, which fix their own p mapping).
        """
        if protocol not in ("micro_ref", "q6_class"):
            raise ValueError(f"protocol must be micro_ref|q6_class, got {protocol}")
        if fusion_model not in ("1", "2", "native"):
            raise ValueError(f"fusion_model must be 1|2|native, got {fusion_model}")
        if dial not in ("tau", "delta"):
            raise ValueError(f"dial must be tau|delta, got {dial}")
        self.protocol = protocol
        self.chirality_d = int(chirality_d)
        self.fusion_model = fusion_model
        self.dial = dial
        self.thresholds = self._thresholds()
        self.shells = self._shell_census()
        self.percolation = _kernel_percolation_board()
        self.grammar = grammar if grammar is not None else default_grammar()

    @staticmethod
    def _thresholds() -> ThresholdBoard:
        prod = Q_G * (M_A**2)
        az = ALPHA0 * ZETA_G
        tgt = ALPHA0_ZETA_TARGET
        return ThresholdBoard(
            s_p=S_P,
            u_p=U_P,
            o_p=O_P,
            m_a=M_A,
            S_CS=S_CS,
            S_UNA=S_UNA,
            S_ONA=S_ONA,
            S_BU=S_BU,
            Q_G=Q_G,
            delta_BU=DELTA_BU,
            rho=RHO,
            Delta=DELTA,
            G_kernel=G_KERNEL,
            alpha0=ALPHA0,
            optical_dilution=OPTICAL_DILUTION,
            D=D_SHELL,
            product_QG_ma2=prod,
            alpha0_zeta=az,
            alpha0_zeta_target=tgt,
            product_identity_ok=abs(az - tgt) < 1e-9,
        )

    @staticmethod
    def _shell_census() -> ShellCensus:
        from gyroscopic.hQVM.api import shell_population

        pops = [shell_population(k) for k in range(7)]
        holo = (H_CARD**2 == OMEGA) and (sum(pops) == OMEGA)
        mean_S = sum(k * pops[k] for k in range(7)) / OMEGA
        return ShellCensus(OMEGA, H_CARD, pops, holo, mean_S)

    def _find_grammar_class(
        self, k: int, d: int, *, forced_only: bool = False
    ) -> Optional[ClosureClass]:
        for cls in self.grammar:
            if cls.k == k and cls.d == d:
                if forced_only and not cls.forced:
                    continue
                return cls
        return None

    def _anchor_class(self, E_eV: float) -> ClosureClass:
        """
        Anchor routing: prefer Level when close, else snap to the
        nearest bare k on the integer grid.
        """
        lv = self.level(E_eV)
        if abs(lv.tick_residual) < 0.5 * TICKS_PER_K:
            return lv.cls
        n = self.n_of_E(E_eV)
        k = int(round(n / TICKS_PER_K))
        k = max(0, min(12, k))
        found = self._find_grammar_class(k, 0)
        if found is not None:
            return found
        return ClosureClass(k, 0, f"k={k} bare (anchor)", False, False, False)

    def predict_E_eV(self, cls: ClosureClass) -> float:
        E = V_GEV * (DELTA**cls.k) * (RHO**cls.d)
        if cls.stf:
            E /= SQRT5
        if cls.tick_corr:
            E *= TICK_CORR
        return E * EV_PER_GEV

    def n_of_E(self, E_eV: float) -> float:
        return math.log2(V_EV / E_eV) / DELTA

    def E_of_n(self, n: float) -> float:
        return V_EV / (2.0 ** (n * DELTA))

    # ---- SQUARE ----
    def square(self, E_eV: float) -> SquareReading:
        if E_eV <= 0:
            raise ValueError("Energy must be positive")
        log2_voe = math.log2(V_EV / E_eV)
        n = log2_voe / DELTA
        if n < 0:
            sector = "Planck/CS"
        elif n < 200:
            sector = "EW/UV"
        elif n < 900:
            sector = "Strong/IR"
        elif n < 1200:
            sector = "keV/Plasma"
        elif n < 1900:
            sector = "Nuclear/Boundary"
        else:
            sector = "Atomic/Deep IR"
        return SquareReading(E_eV, n, log2_voe, sector)

    # ---- LEVEL ----
    def level(self, E_eV: float, *, forced_only: bool = False) -> LevelReading:
        n = self.n_of_E(E_eV)
        best: Optional[LevelReading] = None
        for cls in self.grammar:
            if forced_only and not cls.forced:
                continue
            Ep = self.predict_E_eV(cls)
            np = self.n_of_E(Ep)
            tres = n - np
            rel = (E_eV - Ep) / Ep
            hnote = HOLONOMY_DRESS.get(cls.d, f"d={cls.d}")
            fit = LevelReading(E_eV, n, cls, Ep, np, tres, rel, hnote)
            if best is None or abs(fit.tick_residual) < abs(best.tick_residual):
                best = fit
        assert best is not None
        return best

    # ---- COMPASS ----
    DRESS_ORDER: Tuple[int, ...] = (0, 2, 4, 5)

    def compass(self, E1_eV: float, E2_eV: float) -> List[CompassStep]:
        c1 = self._anchor_class(E1_eV)
        c2 = self._anchor_class(E2_eV)
        k1, d1 = c1.k, c1.d
        k2, d2 = c2.k, c2.d
        steps: List[CompassStep] = []
        ck, cd = k1, d1
        cE = self.predict_E_eV(c1)

        def gap(to_E: float) -> float:
            return abs(self.n_of_E(to_E) - self.n_of_E(cE))

        def dress_op(d_from: int, d_to: int) -> str:
            return DRESS_OPERATOR.get((d_from, d_to), HOLONOMY_DRESS.get(d_to, ""))

        # Connection 1-form chain (hQVM Specs Formalism §16.9): the seven
        # constitutional phase boundaries traversed by each move type.
        def boundary_label(mt: str) -> str:
            return {
                "undress": "ONA|BU→BU|BU→BU|ONA",
                "Δ-step": "CS|UNA",
                "dress": "ONA|BU→BU|BU→BU|ONA",
                "octave": "UNA|CS",
                "code": "BU|ONA",
                "offset": "BU|ONA",
            }.get(mt, "")

        # 1) Undress
        while cd != 0:
            idx = self.DRESS_ORDER.index(cd)
            nd = self.DRESS_ORDER[idx - 1]
            ncls = self._find_grammar_class(ck, nd)
            if ncls is None:
                ncls = ClosureClass(
                    ck, nd, f"k={ck} d={nd} (temp)", False, nd > 0, False
                )
            nE = self.predict_E_eV(ncls)
            steps.append(
                CompassStep(
                    from_k=ck,
                    from_d=cd,
                    to_k=ck,
                    to_d=nd,
                    from_E=cE,
                    to_E=nE,
                    move_type="undress",
                    note=f"ρ^-1: d={cd}→d={nd} [{boundary_label('undress')}]",
                    tick_gap=gap(nE),
                    operator=dress_op(cd, nd),
                )
            )
            cd, cE = nd, nE

        # 2) Δ-step
        while ck != k2:
            dk = 1 if k2 > ck else -1
            nk = ck + dk
            ncls = self._find_grammar_class(nk, 0)
            if ncls is None:
                ncls = ClosureClass(nk, 0, f"k={nk} bare (temp)", False, False, False)
            nE = self.predict_E_eV(ncls)
            steps.append(
                CompassStep(
                    from_k=ck,
                    from_d=0,
                    to_k=nk,
                    to_d=0,
                    from_E=cE,
                    to_E=nE,
                    move_type="Δ-step",
                    note=f"Δ^{dk:+d}: k={ck}→k={nk} [{boundary_label('Δ-step')}]",
                    tick_gap=gap(nE),
                    operator=f"aperture grade Δ^{dk:+d}",
                )
            )
            ck, cE = nk, nE

        # 3) Dress
        while cd != d2:
            idx = self.DRESS_ORDER.index(cd)
            nd = self.DRESS_ORDER[idx + 1]
            ncls = self._find_grammar_class(ck, nd)
            if ncls is None:
                ncls = ClosureClass(
                    ck,
                    nd,
                    f"k={ck} d={nd} (temp)",
                    False,
                    nd > 0,
                    ck == 6 and nd == 2,
                )
            nE = self.predict_E_eV(ncls)
            steps.append(
                CompassStep(
                    from_k=ck,
                    from_d=cd,
                    to_k=ck,
                    to_d=nd,
                    from_E=cE,
                    to_E=nE,
                    move_type="dress",
                    note=f"ρ^{nd-cd:+d}: d={cd}→d={nd} [{boundary_label('dress')}]",
                    tick_gap=gap(nE),
                    operator=dress_op(cd, nd),
                )
            )
            cd, cE = nd, nE

        # 4) Octave
        while True:
            rem = self.n_of_E(E2_eV) - self.n_of_E(cE)
            if abs(abs(rem) - TICKS_PER_OCTAVE) > OCTAVE_TOL_TICKS:
                break
            if rem > 0:
                nE = cE / 2.0
                note = "octave: E→E/2 (dyadic IR)"
            else:
                nE = cE * 2.0
                note = "octave: E→2E (dyadic UV)"
            steps.append(
                CompassStep(
                    from_k=ck,
                    from_d=cd,
                    to_k=ck,
                    to_d=cd,
                    from_E=cE,
                    to_E=nE,
                    move_type="octave",
                    note=f"{note} [{boundary_label('octave')}]",
                    tick_gap=gap(nE),
                    operator="Horizon Lemma (dyadic)",
                )
            )
            cE = nE

        # 5) Code / offset
        code_ticks = self.n_of_E(E2_eV) - self.n_of_E(cE)
        if abs(code_ticks) > 0.01:
            best_code = min(CODE_ATOMS, key=lambda c: abs(abs(code_ticks) - c))
            if abs(abs(code_ticks) - best_code) < 0.5:
                steps.append(
                    CompassStep(
                        from_k=ck,
                        from_d=cd,
                        to_k=ck,
                        to_d=cd,
                        from_E=cE,
                        to_E=E2_eV,
                        move_type="code",
                        note=f"code {code_ticks:+.3f} ticks ({best_code:g}-tick atom) [{boundary_label('code')}]",
                        tick_gap=abs(code_ticks),
                        operator=f"code atom ~{best_code:g}",
                    )
                )
            else:
                steps.append(
                    CompassStep(
                        from_k=ck,
                        from_d=cd,
                        to_k=ck,
                        to_d=cd,
                        from_E=cE,
                        to_E=E2_eV,
                        move_type="offset",
                        note=f"empirical offset {code_ticks:+.3f} ticks (no code atom match) [{boundary_label('offset')}]",
                        tick_gap=abs(code_ticks),
                        operator="empirical bound state",
                    )
                )
        return steps

    # ---- PERCOLATION ----
    @staticmethod
    def percolation_from_rank(r: int) -> PercolationState:
        if r < 0 or r > CHIRALITY_D:
            raise ValueError(f"rank must be in 0..{CHIRALITY_D}")
        if r == 0:
            return PercolationState(
                0,
                2,
                2,
                2 / OMEGA,
                False,
                "r=0 gauge doublet",
                fiber_complete=False,
            )
        root = 2**r
        Reach = predicted_cluster_size(r)
        cov = Reach / OMEGA
        parity = r == 5
        # r=5 is the even-weight plateau: fiber-complete but parity-
        # obstructed (no odd-shell transport), so the full product form
        # holds yet full Omega is unreachable. r=0 doublet is not fiber-
        # complete. All other r are fiber-complete.
        fc = r != 0
        note = f"|Reach|=(2^{r})^2={Reach}; θ={cov:.6f}"
        if parity:
            note += " | even-weight plateau (parity-obstructed)"
        if not fc:
            note += " | not fiber-complete"
        return PercolationState(r, root, Reach, cov, parity, note, fc)

    @staticmethod
    def theta_from_rank(r: float) -> float:
        r = min(float(CHIRALITY_D), max(0.0, r))
        if r <= 0.0:
            return 2.0 / OMEGA
        return (2.0**r / float(H_CARD)) ** 2

    def theta_from_inclusion(self, p: float, *, cond: bool = True) -> float:
        """
        Exact coverage fraction θ(p). Default cond=True for physical-access
        mapping (condition on nonempty generator set). Use cond=False only
        in percolation-law audit tables (doc §5 unconditional formula).
        """
        return theta_for_protocol(p, self.chirality_d, self.protocol, cond=cond)

    def r_eff_from_theta(self, theta: float) -> float:
        # Exact inverse of θ(r)=(2^r/|H|)^2 for r>=1, with r=0 → θ=2/|Ω|.
        theta0 = 2.0 / OMEGA
        if theta <= theta0:
            return 0.0
        t = min(1.0, max(theta, theta0))
        r = float(self.chirality_d) + 0.5 * math.log2(t)
        return float(min(float(self.chirality_d), max(0.0, r)))

    # ---- BRIDGE: Beer-Lambert τ → transmission → inclusion dial ----
    @staticmethod
    def tau_nuclear(E_MeV: float, E_G_MeV: float, V_barrier_MeV: float) -> float:
        """
        Barrier-normalized Gamow optical depth (Beer-Lambert).
        Nuclear analogue of Gravity §6.3 τ(ψ)=τ_G(1−ψ): zero at the
        Coulomb barrier, positive below it. Percolation §6.7 tracks the
        same −ln T spanning-transmission depth on the generator graph.
        """
        if E_MeV <= 0.0 or V_barrier_MeV <= 0.0 or E_G_MeV <= 0.0:
            return float("inf")
        if E_MeV >= V_barrier_MeV:
            return 0.0
        return math.sqrt(E_G_MeV / E_MeV) - math.sqrt(E_G_MeV / V_barrier_MeV)

    @staticmethod
    def transmission_from_tau(tau: float) -> float:
        """T = exp(−τ). At the barrier τ=0 ⇒ T=1; deep IR ⇒ T→0."""
        if tau <= 0.0:
            return 1.0
        if tau > 500.0:
            return 0.0
        return math.exp(-tau)

    def _p_tau_dial(self, E_eV: float, E_barrier_eV: float, *, E_G_MeV: float) -> float:
        """
        τ-dial: p = p_c · T, T=exp(−τ). At/above barrier τ=0 ⇒ T=1 ⇒ p=p_c
        (NOT p=1 — that contradicted the stated bridge).
        """
        pc = self.percolation.p_c_rank
        if E_eV <= 0.0:
            return 0.0
        if E_barrier_eV <= 0.0:
            return pc
        E_MeV = E_eV / 1e6
        Vb_MeV = E_barrier_eV / 1e6
        tau = self.tau_nuclear(E_MeV, E_G_MeV, Vb_MeV)
        T = self.transmission_from_tau(tau)
        return min(1.0, max(0.0, pc * T))

    def p_delta_ruler(self, E_eV: float, E_barrier_eV: float) -> float:
        """
        Δ-ruler dial (Gravity §6.3 Route A): p_Δ = E/V_b.
        Beer-Lambert with τ_Δ = log2(V_b/E) gives transmission 2^{−τ_Δ}=E/V_b.
        """
        if E_eV <= 0.0 or E_barrier_eV <= 0.0:
            return 0.0
        return min(1.0, max(0.0, E_eV / E_barrier_eV))

    def _p_inclusion_bridge(
        self,
        E_eV: float,
        E_barrier_eV: float,
        *,
        E_G_MeV: float,
        E_ir_eV: Optional[float] = None,
    ) -> float:
        """
        Inclusion dial that feeds θ in the fusion σ formula.
        Model 1: bare T=exp(−τ) (θ is the tunneling factor).
        Model 2 + dial=delta: p_Δ=E/V_b (non-Gamow; no double-count).
        Model 2 + dial=tau: p_c·T (legacy; double-counts with P_Gamow).
        native: p = E/E_ir, E_ir = E_str (BU strong anchor, the nuclear-frame
                IR conjugate of the BU merge) — no electrostatic barrier, no
                Gamow. Defaults to E_barrier_eV when E_ir_eV is None.
        """
        if self.fusion_model == "native":
            scale = E_ir_eV if E_ir_eV and E_ir_eV > 0.0 else E_barrier_eV
            if E_eV <= 0.0 or scale <= 0.0:
                return 0.0
            return min(1.0, max(0.0, E_eV / scale))
        if self.fusion_model == "1":
            if E_eV <= 0.0:
                return 0.0
            if E_barrier_eV <= 0.0:
                return 1.0
            tau = self.tau_nuclear(E_eV / 1e6, E_G_MeV, E_barrier_eV / 1e6)
            return self.transmission_from_tau(tau)
        if self.dial == "delta":
            return self.p_delta_ruler(E_eV, E_barrier_eV)
        return self._p_tau_dial(E_eV, E_barrier_eV, E_G_MeV=E_G_MeV)

    def r_eff_from_energy(
        self, E_eV: float, E_barrier_eV: float, E_res_eV: float, *, E_G_MeV: float
    ) -> float:
        p = self._p_inclusion_bridge(E_eV, E_barrier_eV, E_G_MeV=E_G_MeV)
        theta = self.theta_from_inclusion(p)
        return self.r_eff_from_theta(theta)

    def E_event_for_Vb(
        self, E_G_MeV: float, V_barrier_MeV: float, p_target: float
    ) -> float:
        """
        Energy (keV) at which bare transmission T=exp(−τ) equals p_target.
        Hierarchy map and E_rank invert the transmission dial (not p_c·T),
        so every p_target ∈ (0,1) is reachable below the barrier:

            T = exp(√(E_G/V_b) − √(E_G/E)) = p_target
            √(E_G/E) = √(E_G/V_b) − ln(p_target)
        """
        if p_target <= 0.0 or p_target >= 1.0:
            return float("inf")
        rhs = math.sqrt(E_G_MeV / V_barrier_MeV) - math.log(p_target)
        if rhs <= 0:
            return float("inf")
        return (E_G_MeV / (rhs**2)) * 1e3

    def E_rank_keV(
        self, E_G_MeV: float, V_barrier_MeV: float, *, p_target: Optional[float] = None
    ) -> float:
        """
        E_rank: energy where bare transmission T(E) reaches the exact
        rank threshold p_c_rank (P(rank=d)=1/2). This is the structural
        onset of full transport root on the Beer-Lambert dial.
        """
        if p_target is None:
            p_target = self.percolation.p_c_rank
        return self.E_event_for_Vb(E_G_MeV, V_barrier_MeV, p_target)

    def E_rank_delta_keV(
        self, V_barrier_MeV: float, *, p_target: Optional[float] = None
    ) -> float:
        """
        Δ-ruler twin of E_rank (Gravity §6.3 Route A): energy where
        p_Δ = E/V_b equals p_target (default p_c_rank).
          E = p_target · V_b
        """
        if p_target is None:
            p_target = self.percolation.p_c_rank
        if p_target <= 0.0 or p_target >= 1.0:
            return float("inf")
        return p_target * V_barrier_MeV * 1e3

    # ---- CROSS-SECTION / FUSION ----
    @staticmethod
    def reduced_mass_MeV(A1: float, A2: float) -> float:
        return (A1 * A2) / (A1 + A2) * M_N_MEV

    @staticmethod
    def gamow_energy_MeV(Z1: int, Z2: int, A1: float, A2: float) -> float:
        mu = Trestleboard.reduced_mass_MeV(A1, A2)
        return 2.0 * mu * (math.pi * ALPHA_EM * Z1 * Z2) ** 2

    @staticmethod
    def gamow_factor(E_MeV: float, E_G_MeV: float) -> float:
        if E_MeV <= 0:
            return 0.0
        return math.exp(-math.sqrt(E_G_MeV / E_MeV))

    @staticmethod
    def coulomb_barrier_MeV(Z1: int, Z2: int, A1: float, A2: float) -> float:
        r_fm = 1.2 * (A1 ** (1.0 / 3.0) + A2 ** (1.0 / 3.0))
        return 1.44 * Z1 * Z2 / r_fm

    def cross_section_relative(
        self,
        E_keV: float,
        *,
        Z1: int,
        Z2: int,
        A1: float,
        A2: float,
        E_ref_keV: float,
        S0: float = 1.0,
    ) -> CrossSectionReading:
        E_MeV = E_keV / 1e3
        E_eV = E_keV * 1e3
        E_G = self.gamow_energy_MeV(Z1, Z2, A1, A2)
        Vb = self.coulomb_barrier_MeV(Z1, Z2, A1, A2)

        p_inc = self._p_inclusion_bridge(E_eV, Vb * 1e6, E_G_MeV=E_G)
        theta = self.theta_from_inclusion(p_inc)
        r_eff = self.r_eff_from_theta(theta)
        Pg = self.gamow_factor(E_MeV, E_G)
        # Model 2: Gamow × θ(dial). Model 1 and native: θ only (no Pg).
        if self.fusion_model == "2":
            sig_c = S0 / max(E_MeV, 1e-30) * Pg * theta
        else:
            sig_c = S0 / max(E_MeV, 1e-30) * theta
        sig_g = S0 / max(E_MeV, 1e-30) * Pg

        Eref_MeV = E_ref_keV / 1e3
        Eref_eV = E_ref_keV * 1e3
        p_ref = self._p_inclusion_bridge(Eref_eV, Vb * 1e6, E_G_MeV=E_G)
        th_ref = self.theta_from_inclusion(p_ref)
        Pg_ref = self.gamow_factor(Eref_MeV, E_G)
        if self.fusion_model == "2":
            sig_c_ref = S0 / max(Eref_MeV, 1e-30) * Pg_ref * th_ref
        else:
            sig_c_ref = S0 / max(Eref_MeV, 1e-30) * th_ref
        sig_g_ref = S0 / max(Eref_MeV, 1e-30) * Pg_ref

        lv = self.level(E_eV, forced_only=True)
        return CrossSectionReading(
            E_keV=E_keV,
            E_MeV=E_MeV,
            n=self.n_of_E(E_eV),
            p_inc=p_inc,
            r_eff=r_eff,
            theta=theta,
            P_gamow=Pg,
            sigma_rel_gamow=sig_g / max(sig_g_ref, 1e-300),
            sigma_rel_cgm=sig_c / max(sig_c_ref, 1e-300),
            nearest_forced=lv.cls.label,
        )

    def _predicted_peak_keV(
        self, E_grid_keV: Sequence[float], E_G_MeV: float, V_barrier_MeV: float
    ) -> float:
        """Grid argmax of the Model-1/2 σ integrand (relative shape)."""
        best = -1.0
        best_E = 0.0
        for E in E_grid_keV:
            E_MeV = E / 1e3
            p = self._p_inclusion_bridge(E * 1e3, V_barrier_MeV * 1e6, E_G_MeV=E_G_MeV)
            th = self.theta_from_inclusion(p)
            if self.fusion_model == "2":
                Pg = self.gamow_factor(E_MeV, E_G_MeV)
                sig = (1.0 / max(E_MeV, 1e-30)) * Pg * th
            else:  # "1" and "native"
                sig = (1.0 / max(E_MeV, 1e-30)) * th
            if sig > best:
                best = sig
                best_E = E
        return best_E

    def fusion_scan(
        self,
        *,
        Z1: int,
        Z2: int,
        A1: float,
        A2: float,
        T_plasma_keV: float,
        E_grid_keV: Sequence[float],
        E_ref_keV: float,
    ) -> FusionScanResult:
        Vb = self.coulomb_barrier_MeV(Z1, Z2, A1, A2)
        EG = self.gamow_energy_MeV(Z1, Z2, A1, A2)
        rows = tuple(
            self.cross_section_relative(
                E, Z1=Z1, Z2=Z2, A1=A1, A2=A2, E_ref_keV=E_ref_keV
            )
            for E in E_grid_keV
        )
        best_g = max(rows, key=lambda r: r.sigma_rel_gamow)
        best_c = max(rows, key=lambda r: r.sigma_rel_cgm)
        plasma = self.square(T_plasma_keV * 1e3)
        barrier = self.square(Vb * 1e6)
        E_str = self.predict_E_eV(ClosureClass(3, 0, "strong", True, False, False))
        res = self.square(E_str)
        E_rank = self.E_rank_keV(EG, Vb)
        predicted_peak = self._predicted_peak_keV(E_grid_keV, EG, Vb)
        return FusionScanResult(
            Vb,
            EG,
            E_rank,
            plasma,
            barrier,
            res,
            rows,
            best_g,
            best_c,
            predicted_peak,
        )

    def reactivity_scan(
        self,
        *,
        Z1: int,
        Z2: int,
        A1: float,
        A2: float,
        T_grid_keV: Sequence[float],
        E_grid_keV: Sequence[float],
        T_ref_keV: float = 10.0,
    ) -> ReactivityScan:
        E_G = self.gamow_energy_MeV(Z1, Z2, A1, A2)
        Vb = self.coulomb_barrier_MeV(Z1, Z2, A1, A2)
        E_arr = [float(E) for E in E_grid_keV]

        def integrand(E_keV: float, use_theta: bool) -> float:
            E_MeV = E_keV / 1e3
            Pg = self.gamow_factor(E_MeV, E_G)
            if not use_theta:
                return Pg
            p = self._p_inclusion_bridge(E_keV * 1e3, Vb * 1e6, E_G_MeV=E_G)
            th = self.theta_from_inclusion(p)
            if self.fusion_model == "2":
                return Pg * th
            return th  # "1" and "native"

        def integrate(use_theta: bool) -> List[float]:
            vals = [integrand(E, use_theta) for E in E_arr]
            out = []
            for T in T_grid_keV:
                T = max(float(T), 1e-12)
                w = [v * math.exp(-E / T) for E, v in zip(E_arr, vals)]
                acc = 0.0
                for i in range(len(E_arr) - 1):
                    dE = E_arr[i + 1] - E_arr[i]
                    acc += 0.5 * (w[i] + w[i + 1]) * dE
                out.append(acc)
            return out

        g_raw = integrate(False)
        c_raw = integrate(True)
        T_list = [float(T) for T in T_grid_keV]
        i_ref = min(range(len(T_list)), key=lambda i: abs(T_list[i] - T_ref_keV))
        g_ref = max(g_raw[i_ref], 1e-300)
        c_ref = max(c_raw[i_ref], 1e-300)
        enh = [c / max(g, 1e-300) for g, c in zip(g_raw, c_raw)]
        rows = tuple(
            ReactivityReading(T, g / g_ref, c / c_ref, e)
            for T, g, c, e in zip(T_list, g_raw, c_raw, enh)
        )
        best_g = max(rows, key=lambda r: r.sv_rel_gamow)
        best_c = max(rows, key=lambda r: r.sv_rel_cgm)
        # R(T)=⟨σv⟩CGM/⟨σv⟩G is monotone→1 as T→∞ (θ-weighted average).
        # The falsifiable interior signal is argmax of dR/dlnT — the
        # temperature where θ most rapidly reshapes the Maxwellian window.
        growth: List[Tuple[float, float]] = []
        for i in range(1, len(T_list)):
            dlnT = math.log(T_list[i] / T_list[i - 1])
            if dlnT <= 0:
                continue
            dR = (enh[i] - enh[i - 1]) / dlnT
            growth.append((T_list[i], dR))
        if growth:
            T_grow, _ = max(growth, key=lambda x: x[1])
        else:
            T_grow = T_list[-1]
        enh_shifted = T_grow != T_list[0] and T_grow != T_list[-1]
        return ReactivityScan(
            rows=rows,
            T_peak_gamow=best_g.T_keV,
            T_peak_cgm=best_c.T_keV,
            peak_shifted=best_g.T_keV != best_c.T_keV,
            T_enhancement_peak=T_grow,
            enhancement_shifted=enh_shifted,
        )

    # Square-root cluster theorem: one predeclared p_c per integer rank r.
    # bisect_p_c_rank_micro_ref(r) is the inclusion p where the reachable
    # set reaches rank r (P(rank=r)=1/2). Distinct values only; small r are
    # degenerate (r=1,2,3 share p_c) so we dedupe.
    _RANK_LADDER_PC: Optional[List[float]] = None

    @classmethod
    def rank_ladder_p_c(cls) -> List[float]:
        """Distinct rank-ladder inclusion thresholds p_c(r), r=0..6."""
        if cls._RANK_LADDER_PC is None:
            seen: List[float] = []
            for r in range(7):
                p = bisect_p_c_rank_micro_ref(r)
                if not seen or abs(p - seen[-1]) > 1e-9:
                    seen.append(p)
            cls._RANK_LADDER_PC = seen
        return cls._RANK_LADDER_PC

    def optical_conjugate_eV(self, E_eV: float, E_CS_eV: float = 1.22e28) -> float:
        K = E_CS_eV * V_EV * OPTICAL_DILUTION
        return K / max(E_eV, 1e-300)

    def deuteron_bare_MeV(self) -> float:
        """Strong bare: v·Δ³ (MeV)."""
        return V_GEV * (DELTA**3) * 1e3

    def deuteron_binding_MeV(self) -> float:
        """E_D = v·Δ³ + v·Δ⁴·(2/√5); 2/√5 = (p_W−p_Z)/√5 (EW expansion)."""
        E_bare = V_GEV * (DELTA**3)
        E_tensor = V_GEV * (DELTA**4) * (2.0 / SQRT5)
        return (E_bare + E_tensor) * 1e3


def default_grammar() -> List[ClosureClass]:
    forced = [
        ClosureClass(3, 0, "Strong bare (v·Δ³)", True, False, False),
        ClosureClass(6, 2, "Nuclear spinorial (v·ρ²·Δ⁶/√5·2^(C3Δ²))", True, True, True),
    ]
    optional = [
        ClosureClass(6, 0, "Boundary bare (v·Δ⁶)", False, False, False),
        ClosureClass(6, 4, "Boundary EM (v·ρ⁴·Δ⁶/√5)", False, True, False),
        ClosureClass(6, 5, "Boundary gravity (v·ρ⁵·Δ⁶/√5)", False, True, False),
        ClosureClass(4, 0, "keV bare (v·Δ⁴)", False, False, False),
        ClosureClass(4, 2, "keV spinorial (v·ρ²·Δ⁴/√5)", False, True, False),
        ClosureClass(3, 2, "Strong spinorial (v·ρ²·Δ³/√5)", False, True, False),
        ClosureClass(3, 5, "Strong gravity (v·ρ⁵·Δ³/√5)", False, True, False),
    ]
    return forced + optional
