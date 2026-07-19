#!/usr/bin/env python3
"""
hqvm_cgm_trestleboard_ensdf_data_ingest.py

Pull IAEA LiveChart ENSDF level schemes and rebuild the eV-band census
tables under data/catalogs/ensdf/.

Inputs:
  data/catalogs/ensdf/iaea_livechart_ground_states.csv  (nuclide list)
  optional: existing iaea_livechart_levels_*.csv

Outputs:
  iaea_livechart_levels_<A><el>.csv   (one per nuclide, energies in keV)
  ensdf_ev_band_levels.csv            (0 < E <= E_BAND_EV)
  ensdf_first_excited_actinides.csv   (first excited per pulled file)
  SOURCE.txt                          (refresh stamp)

API: https://nds.iaea.org/relnsd/v1/data?fields=levels&nuclides=<A><el>
Requires User-Agent (IAEA returns 403 without it). One nuclide per request.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict, cast

_REPO = Path(__file__).resolve().parents[1]
ENSDF_DIR = _REPO / "data" / "catalogs" / "ensdf"
GS_PATH = ENSDF_DIR / "iaea_livechart_ground_states.csv"
EV_BAND_PATH = ENSDF_DIR / "ensdf_ev_band_levels.csv"
FIRST_EX_PATH = ENSDF_DIR / "ensdf_first_excited_actinides.csv"
SOURCE_PATH = ENSDF_DIR / "SOURCE.txt"

API = "https://nds.iaea.org/relnsd/v1/data?fields=levels&nuclides={tag}"
UA = "Livechart/1.0"
E_BAND_EV = 200.0  # eV upper cut for optical-band census
Z_LO, Z_HI = 88, 98
A_LO, A_HI = 220, 250
SLEEP_S = 0.75


class LevelRow(TypedDict):
    nuclide: str
    z: int
    n: int
    A: int
    symbol: str
    E_keV: float
    E_eV: float
    unc_e_keV: str
    jp: str
    half_life: str
    unit_hl: str
    half_life_sec: str
    idx: str
    ENSDF_cut_off: str
    Extraction_date: str
    source_file: str
    source: str
    note: str
    has_halflife: bool


def _nuclide_tag(A: int, symbol: str) -> str:
    return f"{A}{symbol.lower()}"


def candidates_from_ground_states(
    path: Path = GS_PATH,
    *,
    z_lo: int = Z_LO,
    z_hi: int = Z_HI,
    a_lo: int = A_LO,
    a_hi: int = A_HI,
) -> List[Tuple[int, int, str, str]]:
    """Return (z, A, symbol, tag) for actinide GS rows in the A/Z window."""
    out: List[Tuple[int, int, str, str]] = []
    seen = set()
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                z = int(row["z"])
                n = int(row["n"])
            except (KeyError, ValueError):
                continue
            A = z + n
            sym = (row.get("symbol") or "").strip()
            if not sym:
                continue
            if not (z_lo <= z <= z_hi and a_lo <= A <= a_hi):
                continue
            tag = _nuclide_tag(A, sym)
            if tag in seen:
                continue
            seen.add(tag)
            out.append((z, A, sym, tag))
    return sorted(out, key=lambda t: (t[0], t[1]))


def fetch_levels(tag: str, *, timeout: float = 60.0) -> Optional[str]:
    url = API.format(tag=tag)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"  FAIL {tag}: {e}", file=sys.stderr)
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    # Numeric error codes when empty (API guide).
    stripped = text.strip()
    if not stripped or stripped.isdigit():
        return None
    if "z," not in stripped.split("\n", 1)[0].lower() and "z," not in stripped[:80]:
        # Not a CSV header — treat as empty / error payload.
        if len(stripped) < 40:
            return None
    return text


def pull_missing(
    cands: Sequence[Tuple[int, int, str, str]],
    *,
    force: bool = False,
    sleep_s: float = SLEEP_S,
) -> Tuple[int, int, int]:
    """Download missing level CSVs. Returns (ok, skip, fail)."""
    ENSDF_DIR.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_fail = 0
    for _z, _A, _sym, tag in cands:
        dest = ENSDF_DIR / f"iaea_livechart_levels_{tag}.csv"
        if dest.is_file() and dest.stat().st_size > 0 and not force:
            n_skip += 1
            continue
        text = fetch_levels(tag)
        time.sleep(sleep_s)
        if text is None:
            n_fail += 1
            continue
        dest.write_text(text, encoding="utf-8")
        n_ok += 1
        print(f"  wrote {dest.name} ({len(text)} bytes)")
    return n_ok, n_skip, n_fail


def _parse_level_row(row: Dict[str, str], source_file: str) -> Optional[LevelRow]:
    e = (row.get("energy") or "").strip()
    try:
        ek = float(e)
    except ValueError:
        return None
    if ek < 0.0:
        return None
    # ENSDF X+E placements: need RIPL absolute shift to place on the axis.
    es = (row.get("energy_shift") or "").strip()
    rs = (row.get("ripl_shift") or "").strip()
    if es:
        if not rs:
            return None  # unplaceable
        try:
            ek = ek + float(rs)
        except ValueError:
            return None
    if ek <= 0.0:
        return None
    try:
        z = int(float(row["z"]))
        n = int(float(row["n"]))
    except (KeyError, ValueError):
        return None
    A = z + n
    sym = (row.get("symbol") or "").strip()
    hl = (row.get("half_life") or "").strip()
    unit = (row.get("unit_hl") or "").strip()
    note = ""
    if not hl:
        note = "no half-life in ENSDF (level, not tagged isomer)"
    return {
        "nuclide": f"{sym}-{A}",
        "z": z,
        "n": n,
        "A": A,
        "symbol": sym,
        "E_keV": ek,
        "E_eV": ek * 1000.0,
        "unc_e_keV": (row.get("unc_e") or "").strip(),
        "jp": (row.get("jp") or "").strip(),
        "half_life": hl,
        "unit_hl": unit,
        "half_life_sec": (row.get("half_life_sec") or "").strip(),
        "idx": (row.get("idx") or "").strip(),
        "ENSDF_cut_off": (
            row.get("ENSDFpublicationcut-off")
            or row.get("ENSDF_publication_cut-off")
            or ""
        ).strip(),
        "Extraction_date": (row.get("Extraction_date") or "").strip(),
        "source_file": source_file,
        "source": "IAEA LiveChart API levels (ENSDF snapshot)",
        "note": note,
        "has_halflife": bool(hl),
    }


def rebuild_tables(*, e_band_ev: float = E_BAND_EV) -> Tuple[int, int, int]:
    """Rebuild eV-band and first-excited CSVs from all level files."""
    notes_extra = {
        ("Th", 229): (
            "ENSDF Adopted still lists ~7.6 eV; "
            "Zhang et al. Nature 2024 measures 8.3557335(24) eV"
        ),
        ("U", 235): ("U-235m isomer; ~76 eV (NOT 7.6 eV). Half-life ~26 min."),
    }
    fields = [
        "nuclide",
        "z",
        "n",
        "A",
        "symbol",
        "E_keV",
        "E_eV",
        "unc_e_keV",
        "jp",
        "half_life",
        "unit_hl",
        "half_life_sec",
        "idx",
        "ENSDF_cut_off",
        "Extraction_date",
        "source_file",
        "source",
        "note",
        "has_halflife",
    ]
    band: List[LevelRow] = []
    firsts: List[LevelRow] = []
    n_files = 0
    for path in sorted(ENSDF_DIR.glob("iaea_livechart_levels_*.csv")):
        n_files += 1
        excited: List[LevelRow] = []
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                item = _parse_level_row(row, path.name)
                if item is None:
                    continue
                excited.append(item)
                if item["E_eV"] <= e_band_ev:
                    extra = notes_extra.get((item["symbol"], item["A"]), "")
                    row = item
                    if extra:
                        note = f"{item['note']}; {extra}" if item["note"] else extra
                        row = cast(LevelRow, {**item, "note": note})
                    band.append(row)
        if excited:
            first = min(excited, key=lambda x: x["E_keV"])
            out_first: LevelRow = first
            if first["E_eV"] > e_band_ev:
                out_first = cast(
                    LevelRow,
                    {
                        **first,
                        "note": "first excited is above eV band (keV-scale)",
                    },
                )
            else:
                extra = notes_extra.get((first["symbol"], first["A"]), "")
                if extra:
                    note = f"{first['note']}; {extra}" if first["note"] else extra
                    out_first = cast(LevelRow, {**first, "note": note})
            firsts.append(out_first)

    band.sort(key=lambda r: (r["z"], r["A"], r["E_eV"]))
    firsts.sort(key=lambda r: (r["z"], r["A"]))

    with EV_BAND_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in band:
            w.writerow(r)

    with FIRST_EX_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in firsts:
            w.writerow(r)

    return n_files, len(band), len(firsts)


def write_source_stamp(
    *,
    n_files: int,
    n_band: int,
    n_first: int,
    e_band_ev: float,
) -> None:
    text = f"""IAEA LiveChart / ENSDF nuclear structure snapshots
==================================================

Source
  IAEA Nuclear Data Section LiveChart Data Download API
  https://nds.iaea.org/relnsd/v1/data
  API guide: https://nds.iaea.org/relnsd/vcharthtml/api_v0_guide.html
  Underlying evaluations: ENSDF (Evaluated Nuclear Structure Data File).

Files
  iaea_livechart_ground_states.csv
    GET fields=ground_states&nuclides=all
  iaea_livechart_levels_<A><el>.csv
    GET fields=levels&nuclides=<A><el>  (energies in keV)
  ensdf_ev_band_levels.csv
    Filter 0 < E <= {e_band_ev:.0f} eV from all level files in this folder
    ({n_band} rows from {n_files} level files in this rebuild).
  ensdf_first_excited_actinides.csv
    First excited state per level file ({n_first} nuclides).

Refresh
  python experiments/hqvm_cgm_trestleboard_ensdf_data_ingest.py
  python experiments/hqvm_cgm_trestleboard_ensdf_data_ingest.py --force
  python experiments/hqvm_cgm_trestleboard_ensdf_data_ingest.py --rebuild-only

Notes for the optical-isomer audit
  - ENSDF Adopted for Th-229 still lists the isomer near 7.6 eV;
    Zhang et al., Nature 633, 63 (2024) measures 8.3557335(24) eV.
  - U-235m is ~76 eV (T1/2 ~26 min), not 7.6 eV.
  - Pu-239 first excited is ~7.861 keV, not an eV-band isomer.
  - LiveChart ground_states=all does not emit separate metastable rows;
    use levels= for excited states.
  - Default pull window: Z={Z_LO}-{Z_HI}, A={A_LO}-{A_HI}.

Rebuild stamp: see ensdf_ev_band_levels.csv Extraction_date / this run.
"""
    SOURCE_PATH.write_text(text, encoding="utf-8")


def print_census(e_band_ev: float = E_BAND_EV) -> None:
    print("=" * 5)
    print("ENSDF eV-BAND CENSUS")
    print("=" * 5)
    if not EV_BAND_PATH.is_file():
        print("  (no ensdf_ev_band_levels.csv)")
        return
    rows = list(csv.DictReader(EV_BAND_PATH.open(encoding="utf-8", newline="")))
    n_iso = sum(
        1 for r in rows if (r.get("has_halflife") or "").lower() in ("true", "1", "yes")
    )
    print(f"  band cut: 0 < E <= {e_band_ev:.0f} eV")
    print(f"  levels in band: {len(rows)}  (with half-life: {n_iso})")
    print(f"  {'nuclide':<10s} {'E_eV':>10s} {'Jπ':>8s} {'T1/2':>12s} {'iso':>3s}")
    for r in rows:
        hl = f"{r.get('half_life','').strip()} {r.get('unit_hl','').strip()}".strip()
        iso = (
            "Y"
            if (r.get("has_halflife") or "").lower() in ("true", "1", "yes")
            else "N"
        )
        print(
            f"  {r['nuclide']:<10s} {float(r['E_eV']):10.4f} "
            f"{(r.get('jp') or '—'):>8s} {hl:>12s} {iso:>3s}"
        )
    if FIRST_EX_PATH.is_file():
        firsts = list(csv.DictReader(FIRST_EX_PATH.open(encoding="utf-8", newline="")))
        n_ev = sum(1 for r in firsts if float(r["E_eV"]) <= e_band_ev)
        n_keV = len(firsts) - n_ev
        print(
            f"  first-excited: {n_ev} in eV band, {n_keV} keV-scale "
            f"(of {len(firsts)} nuclides)"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="ENSDF eV-band ingest / rebuild")
    ap.add_argument(
        "--force", action="store_true", help="Re-download levels even if CSV exists"
    )
    ap.add_argument(
        "--rebuild-only",
        action="store_true",
        help="Skip downloads; rebuild tables from existing CSVs",
    )
    ap.add_argument(
        "--e-band-ev",
        type=float,
        default=E_BAND_EV,
        help="Upper energy cut in eV (default 200)",
    )
    ap.add_argument(
        "--sleep", type=float, default=SLEEP_S, help="Seconds between API calls"
    )
    args = ap.parse_args()

    if not GS_PATH.is_file():
        print(f"missing ground-states catalog: {GS_PATH}", file=sys.stderr)
        sys.exit(1)

    cands = candidates_from_ground_states()
    print(f"candidates Z={Z_LO}-{Z_HI} A={A_LO}-{A_HI}: {len(cands)}")

    if not args.rebuild_only:
        n_ok, n_skip, n_fail = pull_missing(
            cands,
            force=args.force,
            sleep_s=args.sleep,
        )
        print(f"download: ok={n_ok} skip={n_skip} fail/empty={n_fail}")

    n_files, n_band, n_first = rebuild_tables(e_band_ev=args.e_band_ev)
    write_source_stamp(
        n_files=n_files,
        n_band=n_band,
        n_first=n_first,
        e_band_ev=args.e_band_ev,
    )
    print(f"rebuild: level_files={n_files} eV_band={n_band} first_ex={n_first}")
    print(f"wrote {EV_BAND_PATH}")
    print(f"wrote {FIRST_EX_PATH}")
    print_census(args.e_band_ev)


if __name__ == "__main__":
    main()
