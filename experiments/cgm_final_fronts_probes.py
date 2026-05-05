#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

from cgm_32bit_lift_probe import run_148_51_closure_probe, run_32bit_lift_summary
from cgm_compact_geom_core import DELTA
from cgm_compact_geom_kernel import (
    build_observed,
    color_adjoint_spectrum_probe,
    ew_loop_scale_probe,
    family_lifted_k4_spectral_probe,
    k6_spinorial_lift_probe,
    su3_weight4_decomposition_probe,
)


@dataclass(frozen=True)
class FinalFrontsProbe:
    spectral_gauge_sector_ready: bool
    spectral_first_order_closed: bool
    sextet_bracket_closed_raw: bool
    sextet_bracket_conditional_32bit: bool
    rich_k6_conditional_closed: bool
    rich_k6_max_abs_eigenvalue: float
    w_residual_d6: float
    closure_148_51_exact: bool
    closure_148_51_ratio: str


def run_final_fronts_probe() -> FinalFrontsProbe:
    lift = family_lifted_k4_spectral_probe()
    color = su3_weight4_decomposition_probe()
    k6 = k6_spinorial_lift_probe()
    closure = run_148_51_closure_probe(run_32bit_lift_summary())
    loop = ew_loop_scale_probe(build_observed(), delta=DELTA, v=246.22)
    spectrum = color_adjoint_spectrum_probe()

    spectral_gauge_sector_ready = (
        lift.gamma_swap_square_identity
        and lift.gamma_swap_anticommutes_d_flow
        and lift.j_family_square_identity
        and lift.j_family_preserves_phase_label
    )
    spectral_first_order_closed = lift.first_order_holds_d_flow_lift
    sextet_conditional = color.sextet_bracket_closes or (
        (not color.sextet_bracket_closes)
        and closure.closes_exactly
        and spectrum.spectral_radius == 8
        and bool(spectrum.attenuation_ratios)
    )
    rich_k6_conditional = (
        closure.closes_exactly
        and spectral_gauge_sector_ready
        and k6.dimension == 64
        and k6.max_abs_eigenvalue >= 0.0
    )

    return FinalFrontsProbe(
        spectral_gauge_sector_ready=spectral_gauge_sector_ready,
        spectral_first_order_closed=spectral_first_order_closed,
        sextet_bracket_closed_raw=color.sextet_bracket_closes,
        sextet_bracket_conditional_32bit=sextet_conditional,
        rich_k6_conditional_closed=rich_k6_conditional,
        rich_k6_max_abs_eigenvalue=k6.max_abs_eigenvalue,
        w_residual_d6=loop.w_d6_residual,
        closure_148_51_exact=closure.closes_exactly,
        closure_148_51_ratio=f"{closure.ratio_num}/{closure.ratio_den}",
    )


def main() -> None:
    p = run_final_fronts_probe()
    print("CGM Final Fronts Probe")
    print("======================")
    print(f"spectral_gauge_sector_ready      {p.spectral_gauge_sector_ready}")
    print(f"spectral_first_order_closed      {p.spectral_first_order_closed}")
    print(f"sextet_bracket_closed_raw        {p.sextet_bracket_closed_raw}")
    print(f"sextet_conditional_32bit         {p.sextet_bracket_conditional_32bit}")
    print(f"rich_k6_conditional_closed       {p.rich_k6_conditional_closed}")
    print(f"rich_k6_max_abs_eigenvalue       {p.rich_k6_max_abs_eigenvalue:.12f}")
    print(f"w_residual_d6                    {p.w_residual_d6:.12f}")
    print(f"closure_148_51_exact             {p.closure_148_51_exact}")
    print(f"closure_148_51_ratio             {p.closure_148_51_ratio}")


if __name__ == "__main__":
    main()
