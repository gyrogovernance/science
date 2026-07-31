#!/usr/bin/env python3
"""
hqvm_cgm_allometry_data_ingest.py

Download / refresh allometry catalogs under data/catalogs/allometry/ and
emit derived analysis CSVs for hqvm_cgm_allometry_2.py.

Derived outputs (UTF-8 CSV):
  pantheria_bmr.csv
  pantheria_longevity.csv
  pantheria_gestation.csv
  pantheria_weaning.csv
  pantheria_pop_density.csv
  pantheria_home_range.csv
  animaltraits_metabolic.csv
  animaltraits_specific_mr.csv
  animaltraits_brain.csv
  anage_metabolic.csv
  anage_longevity.csv
  city_wages.csv
  city_road_length.csv

Companion: hqvm_cgm_allometry_notes.md, data/catalogs/allometry/SOURCE.txt
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.request import Request, urlopen

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "data" / "catalogs" / "allometry"

URLS = {
    "PanTHERIA_1-0_WR05_Aug2008.txt": "https://esapubs.org/archive/ecol/E090/184/PanTHERIA_1-0_WR05_Aug2008.txt",
    "PanTHERIA_1-0_WR93_Aug2008.txt": "https://esapubs.org/archive/ecol/E090/184/PanTHERIA_1-0_WR93_Aug2008.txt",
    "E090-184-metadata.htm": "https://esapubs.org/archive/ecol/E090/184/metadata.htm",
    "AnimalTraits_observations.csv": "https://animaltraits.org/observations.csv",
    "AnimalTraits_column-documentation.csv": "https://animaltraits.org/column-documentation.csv",
    "anage_dataset.zip": "https://genomics.senescence.info/species/dataset.zip",
    "bea_msa_wages.csv": "https://raw.githubusercontent.com/mansueto-institute/Urban-Growth-Emergent-Statistics/master/wages.csv",
    "bea_msa_population.csv": "https://raw.githubusercontent.com/mansueto-institute/Urban-Growth-Emergent-Statistics/master/population.csv",
    "us_city_ssp2_infra.csv": "https://raw.githubusercontent.com/usutradhar/quantify-infrastructure/main/df_ssp2_clean.csv",
}


def _fpos(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    if not s or s in {"NA", "na", "NaN", "-999", "-999.0", "-999.00", "(NA)"}:
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    if not math.isfinite(x) or x <= 0:
        return None
    return x


def _download_one(name: str, url: str, force: bool = False) -> Path:
    dest = _OUT / name
    if dest.exists() and not force:
        print(f"  skip {dest.name} ({dest.stat().st_size} bytes)")
        return dest
    print(f"  GET {url}")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (CGM-allometry-ingest)"})
    data = urlopen(req, timeout=300).read()
    dest.write_bytes(data)
    print(f"  wrote {dest} ({len(data)} bytes)")
    return dest


def download(force: bool = False) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        _download_one(name, url, force=force)


def _write_csv(dest: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> Path:
    with dest.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  built {dest.name} n={len(rows)}")
    return dest


def _pantheria_rows() -> Iterable[Dict[str, str]]:
    src = _OUT / "PanTHERIA_1-0_WR05_Aug2008.txt"
    with src.open(encoding="utf-8", errors="replace", newline="") as fh:
        yield from csv.DictReader(fh, delimiter="\t")


# Energetic equivalent of O2 (frozen): 20.1 kJ / L O2 → W from mL O2 / hr
O2_KJ_PER_L = 20.1
ML_O2_HR_TO_W = O2_KJ_PER_L / 3600.0  # = 0.00558333... W per (mL O2/hr)


def build_pantheria_bmr() -> Path:
    dest = _OUT / "pantheria_bmr.csv"
    rows_out: List[Dict[str, str]] = []
    for r in _pantheria_rows():
        bmr = _fpos(r.get("18-1_BasalMetRate_mLO2hr"))
        mass = _fpos(r.get("5-2_BasalMetRateMass_g")) or _fpos(r.get("5-1_AdultBodyMass_g"))
        if bmr is None or mass is None:
            continue
        mass_kg = mass / 1000.0
        bmr_W = bmr * ML_O2_HR_TO_W
        rows_out.append(
            {
                "species": (r.get("MSW05_Binomial") or r.get("MSW93_Binomial") or "").strip(),
                "order": (r.get("MSW05_Order") or r.get("MSW93_Order") or "").strip(),
                "family": (r.get("MSW05_Family") or r.get("MSW93_Family") or "").strip(),
                "mass_g": f"{mass:.8g}",
                "bmr_mLO2_hr": f"{bmr:.8g}",
                "mass_kg": f"{mass_kg:.8g}",
                "bmr_W": f"{bmr_W:.8g}",
                "specific_bmr_mLO2_hr_kg": f"{(bmr / mass_kg):.8g}",
                "specific_bmr_W_kg": f"{(bmr_W / mass_kg):.8g}",
                "source": "PanTHERIA_WR05",
            }
        )
    return _write_csv(
        dest,
        [
            "species",
            "order",
            "family",
            "mass_g",
            "mass_kg",
            "bmr_mLO2_hr",
            "bmr_W",
            "specific_bmr_mLO2_hr_kg",
            "specific_bmr_W_kg",
            "source",
        ],
        rows_out,
    )


def build_pantheria_longevity() -> Path:
    dest = _OUT / "pantheria_longevity.csv"
    rows_out: List[Dict[str, str]] = []
    for r in _pantheria_rows():
        mass = _fpos(r.get("5-1_AdultBodyMass_g"))
        life_m = _fpos(r.get("17-1_MaxLongevity_m"))
        if mass is None or life_m is None:
            continue
        rows_out.append(
            {
                "species": (r.get("MSW05_Binomial") or "").strip(),
                "order": (r.get("MSW05_Order") or "").strip(),
                "family": (r.get("MSW05_Family") or "").strip(),
                "mass_g": f"{mass:.8g}",
                "mass_kg": f"{mass/1000.0:.8g}",
                "max_longevity_months": f"{life_m:.8g}",
                "max_longevity_years": f"{life_m/12.0:.8g}",
                "source": "PanTHERIA_WR05",
            }
        )
    return _write_csv(
        dest,
        [
            "species",
            "order",
            "family",
            "mass_g",
            "mass_kg",
            "max_longevity_months",
            "max_longevity_years",
            "source",
        ],
        rows_out,
    )


def _build_pantheria_pair(
    dest_name: str,
    y_col_src: str,
    y_out: str,
    y_scale: float = 1.0,
) -> Path:
    dest = _OUT / dest_name
    rows_out: List[Dict[str, str]] = []
    for r in _pantheria_rows():
        mass = _fpos(r.get("5-1_AdultBodyMass_g"))
        y = _fpos(r.get(y_col_src))
        if mass is None or y is None:
            continue
        yv = y * y_scale
        rows_out.append(
            {
                "species": (r.get("MSW05_Binomial") or "").strip(),
                "order": (r.get("MSW05_Order") or "").strip(),
                "family": (r.get("MSW05_Family") or "").strip(),
                "mass_g": f"{mass:.8g}",
                "mass_kg": f"{mass/1000.0:.8g}",
                y_out: f"{yv:.8g}",
                "source": "PanTHERIA_WR05",
            }
        )
    return _write_csv(
        dest,
        ["species", "order", "family", "mass_g", "mass_kg", y_out, "source"],
        rows_out,
    )


def build_pantheria_gestation() -> Path:
    return _build_pantheria_pair(
        "pantheria_gestation.csv", "9-1_GestationLen_d", "gestation_days"
    )


def build_pantheria_weaning() -> Path:
    return _build_pantheria_pair(
        "pantheria_weaning.csv", "25-1_WeaningAge_d", "weaning_days"
    )


def build_pantheria_pop_density() -> Path:
    return _build_pantheria_pair(
        "pantheria_pop_density.csv",
        "21-1_PopulationDensity_n/km2",
        "pop_density_per_km2",
    )


def build_pantheria_home_range() -> Path:
    return _build_pantheria_pair(
        "pantheria_home_range.csv", "22-1_HomeRange_km2", "home_range_km2"
    )


def _to_kg(mass: float, units: str) -> Optional[float]:
    u = (units or "").strip().lower()
    if u in {"kg", "kilogram", "kilograms"}:
        return mass
    if u in {"g", "gram", "grams"}:
        return mass / 1000.0
    if u in {"mg"}:
        return mass / 1e6
    if u in {"lb", "lbs", "pound", "pounds"}:
        return mass * 0.45359237
    return None


def build_animaltraits_metabolic() -> Path:
    src = _OUT / "AnimalTraits_observations.csv"
    dest = _OUT / "animaltraits_metabolic.csv"
    rows_out: List[Dict[str, str]] = []
    with src.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            mass = _fpos(r.get("body mass"))
            mr = _fpos(r.get("metabolic rate"))
            if mass is None or mr is None:
                continue
            mass_kg = _to_kg(mass, r.get("body mass - units") or "")
            if mass_kg is None:
                continue
            species = f"{(r.get('genus') or '').strip()} {(r.get('specificEpithet') or r.get('species') or '').strip()}".strip()
            rows_out.append(
                {
                    "species": species,
                    "class": (r.get("class") or "").strip(),
                    "order": (r.get("order") or "").strip(),
                    "family": (r.get("family") or "").strip(),
                    "mass_kg": f"{mass_kg:.8g}",
                    "metabolic_rate": f"{mr:.8g}",
                    "metabolic_rate_units": (r.get("metabolic rate - units") or "").strip(),
                    "source": "AnimalTraits",
                }
            )
    return _write_csv(
        dest,
        [
            "species",
            "class",
            "order",
            "family",
            "mass_kg",
            "metabolic_rate",
            "metabolic_rate_units",
            "source",
        ],
        rows_out,
    )


def build_animaltraits_specific_mr() -> Path:
    src = _OUT / "AnimalTraits_observations.csv"
    dest = _OUT / "animaltraits_specific_mr.csv"
    rows_out: List[Dict[str, str]] = []
    with src.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            mass = _fpos(r.get("body mass"))
            smr = _fpos(r.get("mass-specific metabolic rate"))
            if mass is None or smr is None:
                continue
            mass_kg = _to_kg(mass, r.get("body mass - units") or "")
            if mass_kg is None:
                continue
            species = f"{(r.get('genus') or '').strip()} {(r.get('specificEpithet') or r.get('species') or '').strip()}".strip()
            rows_out.append(
                {
                    "species": species,
                    "class": (r.get("class") or "").strip(),
                    "order": (r.get("order") or "").strip(),
                    "family": (r.get("family") or "").strip(),
                    "mass_kg": f"{mass_kg:.8g}",
                    "specific_mr": f"{smr:.8g}",
                    "specific_mr_units": (
                        r.get("mass-specific metabolic rate - units") or ""
                    ).strip(),
                    "source": "AnimalTraits",
                }
            )
    return _write_csv(
        dest,
        [
            "species",
            "class",
            "order",
            "family",
            "mass_kg",
            "specific_mr",
            "specific_mr_units",
            "source",
        ],
        rows_out,
    )


def build_animaltraits_brain() -> Path:
    src = _OUT / "AnimalTraits_observations.csv"
    dest = _OUT / "animaltraits_brain.csv"
    rows_out: List[Dict[str, str]] = []
    with src.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            mass = _fpos(r.get("body mass"))
            brain = _fpos(r.get("brain size"))
            if mass is None or brain is None:
                continue
            mass_kg = _to_kg(mass, r.get("body mass - units") or "")
            if mass_kg is None:
                continue
            species = f"{(r.get('genus') or '').strip()} {(r.get('specificEpithet') or r.get('species') or '').strip()}".strip()
            rows_out.append(
                {
                    "species": species,
                    "class": (r.get("class") or "").strip(),
                    "order": (r.get("order") or "").strip(),
                    "family": (r.get("family") or "").strip(),
                    "mass_kg": f"{mass_kg:.8g}",
                    "brain_size": f"{brain:.8g}",
                    "brain_units": (r.get("brain size - units") or "").strip(),
                    "source": "AnimalTraits",
                }
            )
    return _write_csv(
        dest,
        [
            "species",
            "class",
            "order",
            "family",
            "mass_kg",
            "brain_size",
            "brain_units",
            "source",
        ],
        rows_out,
    )


def _extract_anage_txt() -> Path:
    zpath = _OUT / "anage_dataset.zip"
    dest = _OUT / "anage_data.txt"
    if dest.exists() and dest.stat().st_mtime >= zpath.stat().st_mtime:
        return dest
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        member = next((n for n in names if n.endswith("anage_data.txt")), names[0])
        dest.write_bytes(zf.read(member))
    print(f"  extracted {dest.name} ({dest.stat().st_size} bytes)")
    return dest


def build_anage_metabolic() -> Path:
    src = _extract_anage_txt()
    dest = _OUT / "anage_metabolic.csv"
    rows_out: List[Dict[str, str]] = []
    with src.open(encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            mass = _fpos(r.get("Body mass (g)")) or _fpos(r.get("Adult weight (g)"))
            mr = _fpos(r.get("Metabolic rate (W)"))
            if mass is None or mr is None:
                continue
            genus = (r.get("Genus") or "").strip()
            species = (r.get("Species") or "").strip()
            rows_out.append(
                {
                    "species": f"{genus} {species}".strip(),
                    "class": (r.get("Class") or "").strip(),
                    "order": (r.get("Order") or "").strip(),
                    "family": (r.get("Family") or "").strip(),
                    "mass_g": f"{mass:.8g}",
                    "mass_kg": f"{mass/1000.0:.8g}",
                    "metabolic_rate_W": f"{mr:.8g}",
                    "source": "AnAge",
                }
            )
    return _write_csv(
        dest,
        [
            "species",
            "class",
            "order",
            "family",
            "mass_g",
            "mass_kg",
            "metabolic_rate_W",
            "source",
        ],
        rows_out,
    )


def build_anage_longevity() -> Path:
    src = _extract_anage_txt()
    dest = _OUT / "anage_longevity.csv"
    rows_out: List[Dict[str, str]] = []
    with src.open(encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            mass = _fpos(r.get("Body mass (g)")) or _fpos(r.get("Adult weight (g)"))
            life = _fpos(r.get("Maximum longevity (yrs)"))
            if mass is None or life is None:
                continue
            genus = (r.get("Genus") or "").strip()
            species = (r.get("Species") or "").strip()
            rows_out.append(
                {
                    "species": f"{genus} {species}".strip(),
                    "class": (r.get("Class") or "").strip(),
                    "order": (r.get("Order") or "").strip(),
                    "family": (r.get("Family") or "").strip(),
                    "mass_g": f"{mass:.8g}",
                    "mass_kg": f"{mass/1000.0:.8g}",
                    "max_longevity_years": f"{life:.8g}",
                    "source": "AnAge",
                }
            )
    return _write_csv(
        dest,
        [
            "species",
            "class",
            "order",
            "family",
            "mass_g",
            "mass_kg",
            "max_longevity_years",
            "source",
        ],
        rows_out,
    )


def _bea_year_map(path: Path, year: str = "2010") -> Dict[str, Tuple[str, float]]:
    """Map GeoFips -> (GeoName, value) for a BEA wide table."""
    text = path.read_text(encoding="utf-8", errors="replace")
    # Skip title rows until GeoFips header.
    lines = text.splitlines()
    header_i = next(i for i, ln in enumerate(lines) if ln.startswith("GeoFips,"))
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_i:])))
    out: Dict[str, Tuple[str, float]] = {}
    for r in reader:
        fips = (r.get("GeoFips") or "").strip()
        if not fips or fips == "998":
            continue
        name = (r.get("GeoName") or "").strip()
        val = _fpos(r.get(year))
        if val is None:
            continue
        out[fips] = (name, val)
    return out


def build_city_wages(year: str = "2010") -> Path:
    pop = _bea_year_map(_OUT / "bea_msa_population.csv", year=year)
    wages = _bea_year_map(_OUT / "bea_msa_wages.csv", year=year)
    dest = _OUT / "city_wages.csv"
    rows_out: List[Dict[str, str]] = []
    for fips, (name, n) in pop.items():
        if fips not in wages:
            continue
        _, w = wages[fips]
        rows_out.append(
            {
                "geofips": fips,
                "name": name,
                "population": f"{n:.8g}",
                "wages_thousands_usd": f"{w:.8g}",
                "year": year,
                "source": "BEA_CA30_via_Bettencourt2020",
            }
        )
    return _write_csv(
        dest,
        ["geofips", "name", "population", "wages_thousands_usd", "year", "source"],
        rows_out,
    )


def build_city_road_length() -> Path:
    src = _OUT / "us_city_ssp2_infra.csv"
    dest = _OUT / "city_road_length.csv"
    rows_out: List[Dict[str, str]] = []
    for r in csv.DictReader(src.open(encoding="utf-8", errors="replace", newline="")):
        if (r.get("city type") or "").strip().lower() != "urban":
            continue
        pop = _fpos(r.get("CensusPop_20"))
        per_cap = _fpos(r.get("length_m_perCap_2020_ssp2"))
        if pop is None or per_cap is None:
            continue
        length = per_cap * pop
        if length <= 0:
            continue
        rows_out.append(
            {
                "geoid": (r.get("GEOID") or "").strip(),
                "name": (r.get("NAMELSAD") or "").strip(),
                "city_type": (r.get("city type") or "").strip(),
                "population": f"{pop:.8g}",
                "road_length_m": f"{length:.8g}",
                "road_length_m_per_cap": f"{per_cap:.8g}",
                "year": "2020",
                "source": "quantify-infrastructure_ssp2",
            }
        )
    return _write_csv(
        dest,
        [
            "geoid",
            "name",
            "city_type",
            "population",
            "road_length_m",
            "road_length_m_per_cap",
            "year",
            "source",
        ],
        rows_out,
    )


def write_source() -> None:
    text = """Allometry life-history / metabolic / city catalogs
==========================================

Source
  PanTHERIA (Jones et al. 2009, Ecology 90:2648)
    ESA Ecological Archives E090-184
    https://esapubs.org/archive/ecol/E090/184/
  AnimalTraits (Herberstein et al. 2022, Scientific Data 9:265)
    https://animaltraits.org/
    https://doi.org/10.1038/s41597-022-01364-9
  AnAge (Tacutu et al.; Human Ageing Genomic Resources)
    https://genomics.senescence.info/species/
  BEA MSA wages/population via Bettencourt 2020 companion
    https://github.com/mansueto-institute/Urban-Growth-Emergent-Statistics
  US city roadway length (2020) via quantify-infrastructure
    https://github.com/usutradhar/quantify-infrastructure

Unit freeze (metabolic intercepts)
  O2 energetic equivalent: 20.1 kJ / L O2
  1 mL O2 / hr → 20.1/3600 W ≈ 0.00558333 W
  PanTHERIA bmr_W and specific_bmr_W_kg use this conversion
  AnAge and AnimalTraits metabolic rates are already in Watts

Raw files
  PanTHERIA_1-0_WR05_Aug2008.txt
  PanTHERIA_1-0_WR93_Aug2008.txt
  E090-184-metadata.htm
  AnimalTraits_observations.csv
  AnimalTraits_column-documentation.csv
  anage_dataset.zip / anage_data.txt
  bea_msa_wages.csv
  bea_msa_population.csv
  us_city_ssp2_infra.csv

Derived analysis tables (UTF-8 CSV)
  pantheria_bmr.csv
  pantheria_longevity.csv
  pantheria_gestation.csv
  pantheria_weaning.csv
  pantheria_pop_density.csv
  pantheria_home_range.csv
  animaltraits_metabolic.csv
  animaltraits_specific_mr.csv
  animaltraits_brain.csv
  anage_metabolic.csv
  anage_longevity.csv
  city_wages.csv
  city_road_length.csv

Refresh
  python experiments/hqvm_cgm_allometry_data_ingest.py
  python experiments/hqvm_cgm_allometry_data_ingest.py --force
  python experiments/hqvm_cgm_allometry_data_ingest.py --rebuild-only

Used by
  experiments/hqvm_cgm_allometry_2.py
  experiments/hqvm_cgm_allometry_run.py
"""
    (_OUT / "SOURCE.txt").write_text(text, encoding="utf-8")
    print("  wrote SOURCE.txt")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Ingest allometry catalogs")
    p.add_argument("--force", action="store_true", help="re-download raw files")
    p.add_argument("--rebuild-only", action="store_true", help="skip download")
    args = p.parse_args(argv)
    print("ALLOMETRY CATALOG INGEST")
    print("=" * 5)
    if not args.rebuild_only:
        download(force=args.force)
    build_pantheria_bmr()
    build_pantheria_longevity()
    build_pantheria_gestation()
    build_pantheria_weaning()
    build_pantheria_pop_density()
    build_pantheria_home_range()
    build_animaltraits_metabolic()
    build_animaltraits_specific_mr()
    build_animaltraits_brain()
    build_anage_metabolic()
    build_anage_longevity()
    build_city_wages()
    build_city_road_length()
    write_source()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
