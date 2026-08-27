#!/usr/bin/env python3
"""
hqvm_cgm_genomics_run.py

CLI report for the CGM-hQVM genomics suite (common + scripts 1-8).
Writes experiments/hqvm_cgm_genomics_results.txt and hqvm_cgm_genomics_gates.json.

Use --only SECTION --patch to re-run one census and splice it into results.txt
without a full suite pass.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

_REPO = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from hqvm_cgm_genomics_common import Tee, configure_stdio_utf8, kernel_manifest, report_check, report_section
from hqvm_cgm_genomics_1 import (
    chemical_k4_on_reference,
    conjugacy_and_quotient_census,
    cycle_j2_basis_census,
    print_chart_census,
    print_cycle_j2_basis_census,
)
from hqvm_cgm_genomics_2 import (
    family_mu_census,
    genealogy_ingress_census,
    kernel_layer_census,
    pole_census,
    print_family_mu_census,
    print_genealogy_ingress_census,
    print_kernel_layer_census,
    print_pole_census,
    print_sequence_census,
    sequence_census,
)
from hqvm_cgm_genomics_3 import (
    polarity_fiber_census,
    print_polarity_fiber_census,
    print_spectral_bundle_census,
    spectral_bundle_census,
)
from hqvm_cgm_genomics_4 import (
    algebra_completion_census,
    dynamics_census,
    identity_axes_census,
    print_algebra_completion_census,
    print_dynamics_census,
    print_identity_axes_census,
    print_qubec_order_census,
    print_stage_palindrome_census,
    print_wall_transport_census,
    qubec_order_census,
    stage_palindrome_census,
    wall_transport_census,
)
from hqvm_cgm_genomics_5 import (
    mirror_skew_census,
    moduli_census,
    omega_walk_census,
    print_mirror_skew_census,
    print_moduli_census,
    print_omega_walk_census,
    print_sequence_probes_census,
    print_surgery_moduli_census,
    print_theta_law_census,
    print_word_geometry_census,
    sequence_probes_census,
    surgery_moduli_census,
    theta_law_census,
    word_geometry_census,
)
from hqvm_cgm_genomics_6 import (
    defect_census,
    depth4_closure_census,
    flat_byte_census,
    print_defect_census,
    print_depth4_census,
    print_flat_byte_census,
    print_rebase_census,
    print_replichore_path_census,
    print_skew_census,
    print_signature_census,
    print_synonymous_recode_census,
    print_uniprot_loc_census,
    rebase_parity_census,
    replichore_path_census,
    signature_census,
    skew_channel_census,
    synonymous_recode_census,
    uniprot_location_census,
)
from hqvm_cgm_genomics_7 import (
    aut_reconciliation_census,
    bu_curvature_census,
    constitutional_fiber_census,
    fold_weyl_census,
    genome_wall_census,
    kernel_percolation_census,
    print_aut_reconcile,
    print_bu_curvature,
    print_constitutional,
    print_fold_weyl,
    print_genome_wall,
    print_kernel_percolation,
    print_trna_census,
    print_u8_aut,
    print_u8_local,
    print_u8_ncbi,
    print_u8_slice,
    print_u8_standard,
    print_u8_two_move,
    print_wall_direct_sum,
    trna_fold_census,
    u8_aut_quotient_census,
    u8_local_moduli_census,
    u8_ncbi_placement_census,
    u8_standard_census,
    u8_structured_slice_census,
    u8_two_move_census,
    wall_direct_sum_census,
)

RESULTS_PATH = _EXP / "hqvm_cgm_genomics_results.txt"
GATES_PATH = _EXP / "hqvm_cgm_genomics_gates.json"

from hqvm_cgm_genomics_8 import (
    aff_orbit_census,
    boundary_moduli_census,
    codon_pair_radial_census,
    compile_print_census,
    print_aff_orbit_census,
    print_boundary_moduli,
    print_codon_pair_radial,
    print_compile_print_census,
    print_s6_covariance,
    print_ser_synthetase,
    print_singular_sector,
    print_wall_breach_census,
    s6_covariance_census,
    ser_synthetase_census,
    singular_sector_census,
    wall_breach_census,
)


ONLY_HELP = (
    "ingress | family_mu | poles | moduli | replichore | "
    "kernel2 | script2 | script5 | script6 | script7 | script8 | "
    "s6cov | boundary | breach | compile | orbit | j2 | singular | ser_synth | cpb"
)


def _progress(script: int) -> None:
    print(f"[hqvm_cgm_genomics] running script {script}...", file=sys.stderr, flush=True)


def _load_gates() -> Dict[str, bool]:
    if not GATES_PATH.exists():
        return {}
    data = json.loads(GATES_PATH.read_text(encoding="utf-8"))
    return {str(k): bool(v) for k, v in data.items()}


def _save_gates(gates: Dict[str, bool]) -> None:
    GATES_PATH.write_text(
        json.dumps(dict(sorted(gates.items())), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _patch_section(results_text: str, section_prefix: str, new_body: str) -> str:
    """Replace a numbered section through the next section header."""
    if not section_prefix:
        return results_text
    body = new_body.strip() + "\n\n"
    pat = re.compile(
        rf"(?ms)^({re.escape(section_prefix)}[^\n]*\n.*?)(?=^\d+[a-z]?\.\s|\Z)",
    )
    if pat.search(results_text):
        return pat.sub(body, results_text, count=1)
    tally = re.search(r"(?m)^15\. CHECK TALLY\n", results_text)
    if tally:
        return results_text[: tally.start()] + body + results_text[tally.start() :]
    return results_text.rstrip() + "\n\n" + body


def _patch_tally(results_text: str, gates: Dict[str, bool]) -> str:
    n_pass = sum(1 for v in gates.values() if v)
    n_fail = sum(1 for v in gates.values() if not v)
    fails = "\n".join(f"  FAIL {name}" for name, ok in gates.items() if not ok)
    block = (
        "15. CHECK TALLY\n"
        "=====\n"
        f"  pass={n_pass} fail={n_fail} total={len(gates)}\n"
    )
    if fails:
        block += fails + "\n"
    block += "\n"
    if re.search(r"(?m)^15\. CHECK TALLY\n", results_text):
        return re.sub(r"(?ms)^15\. CHECK TALLY\n.*", block.rstrip() + "\n", results_text, count=1)
    return results_text.rstrip() + "\n\n" + block


def _run_script2(all_gates: Dict[str, bool], *, sequence: bool) -> None:
    k = kernel_layer_census()
    print_kernel_layer_census(k)
    all_gates.update(k.gates)
    if sequence:
        seq = sequence_census()
        print_sequence_census(seq)
        all_gates.update(seq.gates)
        mu = family_mu_census()
        print_family_mu_census(mu)
        all_gates.update(mu.gates)
        ingress = genealogy_ingress_census()
        print_genealogy_ingress_census(ingress)
        all_gates.update(ingress.gates)
        poles = pole_census()
        print_pole_census(poles)
        all_gates.update(poles.gates)


def _run_script3(all_gates: Dict[str, bool], *, spectral: bool) -> None:
    pol = polarity_fiber_census()
    print_polarity_fiber_census(pol)
    all_gates.update(pol.gates)
    if spectral:
        hard = spectral_bundle_census()
        print_spectral_bundle_census(hard)
        all_gates.update(hard.gates)


def _run_script4(all_gates: Dict[str, bool]) -> None:
    pal = stage_palindrome_census()
    print_stage_palindrome_census(pal)
    all_gates.update(pal.gates)
    wall = wall_transport_census()
    print_wall_transport_census(wall)
    all_gates.update(wall.gates)
    axes = identity_axes_census()
    print_identity_axes_census(axes)
    all_gates.update(axes.gates)
    alg = algebra_completion_census()
    print_algebra_completion_census(alg)
    all_gates.update(alg.gates)
    dyn = dynamics_census()
    print_dynamics_census(dyn)
    all_gates.update(dyn.gates)
    qub = qubec_order_census()
    print_qubec_order_census(qub)
    all_gates.update(qub.gates)


def _run_script5(all_gates: Dict[str, bool]) -> None:
    geo = word_geometry_census()
    print_word_geometry_census(geo)
    all_gates.update(geo.gates)
    seqp = sequence_probes_census()
    print_sequence_probes_census(seqp)
    all_gates.update(seqp.gates)
    mod = moduli_census()
    print_moduli_census(mod)
    all_gates.update(mod.gates)
    theta = theta_law_census()
    print_theta_law_census(theta)
    all_gates.update(theta.gates)
    walk = omega_walk_census()
    print_omega_walk_census(walk)
    all_gates.update(walk.gates)
    ms = mirror_skew_census()
    print_mirror_skew_census(ms)
    all_gates.update(ms.gates)
    sm = surgery_moduli_census()
    print_surgery_moduli_census(sm)
    all_gates.update(sm.gates)


def _run_script6(all_gates: Dict[str, bool]) -> None:
    sig = signature_census()
    print_signature_census(sig)
    all_gates.update(sig.gates)
    defc = defect_census()
    print_defect_census(defc)
    all_gates.update(defc.gates)
    flat = flat_byte_census()
    print_flat_byte_census(flat)
    all_gates.update(flat.gates)
    d4 = depth4_closure_census()
    print_depth4_census(d4)
    all_gates.update(d4.gates)
    rec_rows = synonymous_recode_census()
    all_gates.update(print_synonymous_recode_census(rec_rows))
    skew_rows = skew_channel_census()
    all_gates.update(print_skew_census(skew_rows))
    rph = replichore_path_census()
    print_replichore_path_census(rph)
    all_gates.update(rph.gates)
    n_pal, n_m2, n_m0, n_total, rb_gates = rebase_parity_census()
    print_rebase_census(n_pal, n_m2, n_m0, n_total, rb_gates)
    all_gates.update(rb_gates)
    loc_rows = uniprot_location_census()
    all_gates.update(print_uniprot_loc_census(loc_rows))


def _run_script7(all_gates: Dict[str, bool]) -> None:
    wall = wall_direct_sum_census()
    print_wall_direct_sum(wall)
    all_gates.update(wall.gates)
    perc = kernel_percolation_census()
    print_kernel_percolation(perc)
    all_gates.update(perc.gates)
    buc = bu_curvature_census()
    print_bu_curvature(buc)
    all_gates.update(buc.gates)
    weyl = fold_weyl_census()
    print_fold_weyl(weyl)
    all_gates.update(weyl.gates)
    gw = genome_wall_census()
    print_genome_wall(gw)
    all_gates.update(gw.gates)

    u8s = u8_standard_census()
    print_u8_standard(u8s)
    all_gates.update(u8s.gates)
    u8l = u8_local_moduli_census()
    print_u8_local(u8l)
    all_gates.update(u8l.gates)
    u8sl = u8_structured_slice_census()
    print_u8_slice(u8sl)
    all_gates.update(u8sl.gates)
    trna = trna_fold_census()
    print_trna_census(trna)
    all_gates.update(trna.gates)
    u8a = u8_aut_quotient_census(u8l.survivors)
    print_u8_aut(u8a)
    all_gates.update(u8a.gates)
    u8t = u8_two_move_census(u8l.survivors)
    print_u8_two_move(u8t)
    all_gates.update(u8t.gates)
    u8n = u8_ncbi_placement_census()
    print_u8_ncbi(u8n)
    all_gates.update(u8n.gates)
    recon = aut_reconciliation_census(u8l.survivors, u8a)
    print_aut_reconcile(recon)
    all_gates.update(recon.gates)
    const = constitutional_fiber_census()
    print_constitutional(const)
    all_gates.update(const.gates)


def _kernel_header(all_gates: Dict[str, bool]) -> None:
    man = kernel_manifest()
    ok = bool(man["d6_api_ok"])
    all_gates["11_kernel_api"] = ok
    report_section("0. KERNEL DEPENDENCY")
    report_check(
        "d=6 API binding",
        ok,
        str(man["d6_api_note"]),
        "q_word, step_uv, gates match api at d=6",
    )
    print()


def _run_script8(all_gates: Dict[str, bool]) -> None:
    c29 = s6_covariance_census()
    print_s6_covariance(c29)
    all_gates.update(c29.gates)
    c30 = boundary_moduli_census()
    print_boundary_moduli(c30)
    all_gates.update(c30.gates)
    c31 = wall_breach_census()
    print_wall_breach_census(c31)
    all_gates.update(c31.gates)
    g32 = compile_print_census()
    print_compile_print_census(g32)
    all_gates.update(g32)
    c33 = aff_orbit_census()
    print_aff_orbit_census(c33)
    all_gates.update(c33.gates)
    c40 = singular_sector_census()
    print_singular_sector(c40)
    all_gates.update(c40.gates)
    c41 = ser_synthetase_census()
    print_ser_synthetase(c41)
    all_gates.update(c41.gates)
    c42 = codon_pair_radial_census()
    print_codon_pair_radial(c42)
    all_gates.update(c42.gates)


def _run_only(name: str, all_gates: Dict[str, bool]) -> str:
    key = name.strip().lower().replace("-", "_")
    if key in ("ingress", "genealogy_ingress", "20b"):
        c = genealogy_ingress_census()
        print_genealogy_ingress_census(c)
        all_gates.update(c.gates)
        return "20b."
    if key in ("family_mu", "mu", "20"):
        c = family_mu_census()
        print_family_mu_census(c)
        all_gates.update(c.gates)
        return "20."
    if key in ("poles", "pole", "21"):
        c = pole_census()
        print_pole_census(c)
        all_gates.update(c.gates)
        return "21."
    if key in ("moduli", "ncbi_moduli", "13"):
        c = moduli_census()
        print_moduli_census(c)
        all_gates.update(c.gates)
        return "13."
    if key in ("replichore", "replichore_path", "28c", "ori_ter"):
        c = replichore_path_census()
        print_replichore_path_census(c)
        all_gates.update(c.gates)
        return "28c."
    if key in ("kernel2",):
        k = kernel_layer_census()
        print_kernel_layer_census(k)
        all_gates.update(k.gates)
        return "2."
    if key == "script2":
        _run_script2(all_gates, sequence=True)
        return ""
    if key == "script5":
        _run_script5(all_gates)
        return ""
    if key == "script6":
        _run_script6(all_gates)
        return ""
    if key == "script7":
        _run_script7(all_gates)
        return ""
    if key in ("script8", "script9", "script10"):
        _run_script8(all_gates)
        return ""
    if key in ("s6cov", "29"):
        c = s6_covariance_census()
        print_s6_covariance(c)
        all_gates.update(c.gates)
        return "29."
    if key in ("boundary", "30", "stop_boundary"):
        c = boundary_moduli_census()
        print_boundary_moduli(c)
        all_gates.update(c.gates)
        return "30."
    if key in ("breach", "breaches", "31"):
        c = wall_breach_census()
        print_wall_breach_census(c)
        all_gates.update(c.gates)
        return "31."
    if key in ("compile", "compile_print", "32"):
        g32 = compile_print_census()
        print_compile_print_census(g32)
        all_gates.update(g32)
        return "32."
    if key in ("orbit", "aff_orbit", "33"):
        c33 = aff_orbit_census()
        print_aff_orbit_census(c33)
        all_gates.update(c33.gates)
        return "33."
    if key in ("j2", "cycle_j2", "19"):
        j2 = cycle_j2_basis_census()
        print_cycle_j2_basis_census(j2)
        all_gates.update(j2.gates)
        return "19."
    if key in ("singular", "singular_sector", "40"):
        c40 = singular_sector_census()
        print_singular_sector(c40)
        all_gates.update(c40.gates)
        return "40."
    if key in ("ser_synth", "serine_synth", "41"):
        c41 = ser_synthetase_census()
        print_ser_synthetase(c41)
        all_gates.update(c41.gates)
        return "41."
    if key in ("cpb", "codon_pair", "42"):
        c42 = codon_pair_radial_census()
        print_codon_pair_radial(c42)
        all_gates.update(c42.gates)
        return "42."
    raise SystemExit(f"unknown --only {name!r}; choose from: {ONLY_HELP}")


def main() -> Tuple[int, Dict[str, bool], str, Optional[str], bool]:
    parser = argparse.ArgumentParser(description="CGM-hQVM genomics gate report")
    parser.add_argument("--from", dest="from_script", type=int, choices=(3, 4, 5, 6, 7, 8), metavar="N")
    parser.add_argument("--only", dest="only", metavar="SECTION", help=f"run one section ({ONLY_HELP})")
    parser.add_argument("--patch", action="store_true", help="with --only: splice into results.txt + merge gates.json")
    parser.add_argument("--no-sequence", action="store_true")
    parser.add_argument("--no-spectral", action="store_true")
    args = parser.parse_args()

    configure_stdio_utf8()
    man = kernel_manifest()
    all_gates: Dict[str, bool] = {}
    section_prefix = ""
    had_prior_gates = False

    if args.only:
        # Always merge into prior gates so --only cannot wipe the certificate file.
        prior = _load_gates()
        had_prior_gates = len(prior) > 0
        all_gates.update(prior)
        section_prefix = _run_only(args.only, all_gates)
    elif args.from_script == 8:
        _kernel_header(all_gates)
        _progress(8)
        _run_script8(all_gates)
    elif args.from_script == 7:
        _kernel_header(all_gates)
        _progress(7)
        _run_script7(all_gates)
        _progress(8)
        _run_script8(all_gates)
    elif args.from_script == 6:
        _kernel_header(all_gates)
        _progress(6)
        _run_script6(all_gates)
        _progress(7)
        _run_script7(all_gates)
        _progress(8)
        _run_script8(all_gates)
    elif args.from_script == 5:
        _kernel_header(all_gates)
        _progress(5)
        _run_script5(all_gates)
        _progress(6)
        _run_script6(all_gates)
        _progress(7)
        _run_script7(all_gates)
        _progress(8)
        _run_script8(all_gates)
    elif args.from_script == 4:
        _kernel_header(all_gates)
        _progress(4)
        _run_script4(all_gates)
        _progress(5)
        _run_script5(all_gates)
        _progress(6)
        _run_script6(all_gates)
        _progress(7)
        _run_script7(all_gates)
        _progress(8)
        _run_script8(all_gates)
    elif args.from_script == 3:
        _kernel_header(all_gates)
        _progress(3)
        _run_script3(all_gates, spectral=not args.no_spectral)
        _progress(4)
        _run_script4(all_gates)
        _progress(5)
        _run_script5(all_gates)
        _progress(6)
        _run_script6(all_gates)
        _progress(7)
        _run_script7(all_gates)
        _progress(8)
        _run_script8(all_gates)
    else:
        _progress(1)
        chart = conjugacy_and_quotient_census()
        ref_k4 = chemical_k4_on_reference()
        _kernel_header(all_gates)
        print("  objects")
        print("    d=6 API bind; aperture constants; sha256 pins")
        print()
        print(f"  APERTURE_GAP={man['APERTURE_GAP']} RHO={man['RHO']} M_A={man['M_A']}")
        for rel, digest in man["hashes"].items():
            print(f"  sha256 {rel} {digest}")
        print()
        print_chart_census(chart, ref_k4)
        all_gates.update(chart.gates)
        j2 = cycle_j2_basis_census()
        print_cycle_j2_basis_census(j2)
        all_gates.update(j2.gates)
        _progress(2)
        _run_script2(all_gates, sequence=not args.no_sequence)
        _progress(3)
        _run_script3(all_gates, spectral=not args.no_spectral)
        _progress(4)
        _run_script4(all_gates)
        _progress(5)
        _run_script5(all_gates)
        _progress(6)
        _run_script6(all_gates)
        _progress(7)
        _run_script7(all_gates)
        _progress(8)
        _run_script8(all_gates)

    if not args.only:
        report_section("15. CHECK TALLY")
        n_pass = sum(1 for v in all_gates.values() if v)
        n_fail = sum(1 for v in all_gates.values() if not v)
        print(f"  pass={n_pass} fail={n_fail} total={len(all_gates)}")
        for name, ok in all_gates.items():
            if not ok:
                print(f"  FAIL {name}")
        print()
    else:
        n_pass = sum(1 for v in all_gates.values() if v)
        n_fail = sum(1 for v in all_gates.values() if not v)
        print(
            f"[only] gates={len(all_gates)} pass={n_pass} fail={n_fail}",
            file=sys.stderr,
        )

    rc = 0 if all(all_gates.values()) else 1
    only = args.only
    patch = bool(args.patch and args.only)
    return rc, all_gates, section_prefix, ("patch" if patch else None), had_prior_gates


if __name__ == "__main__":
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = Tee(old, buf)
    try:
        rc, gates, section_prefix, mode, had_prior_gates = main()
        out = buf.getvalue()
    finally:
        sys.stdout = old

    _save_gates(gates)

    only_mode = "--only" in sys.argv
    if mode == "patch":
        prev = RESULTS_PATH.read_text(encoding="utf-8") if RESULTS_PATH.exists() else ""
        if section_prefix:
            prev = _patch_section(prev, section_prefix, out)
        else:
            tally = re.search(r"(?m)^15\. CHECK TALLY\n", prev)
            body = out.strip() + "\n\n"
            if tally:
                prev = prev[: tally.start()] + body + prev[tally.start() :]
            else:
                prev = prev.rstrip() + "\n\n" + body
        RESULTS_PATH.write_text(prev, encoding="utf-8")
        print(
            f"patched {RESULTS_PATH} section={section_prefix or 'append'} "
            f"gates={GATES_PATH} (tally left for hand-edit)"
        )
    elif not only_mode:
        RESULTS_PATH.write_text(out, encoding="utf-8")
        print(f"wrote {RESULTS_PATH}")
    else:
        print(f"updated {GATES_PATH} (results.txt unchanged; pass --patch to splice)")

    raise SystemExit(rc)
