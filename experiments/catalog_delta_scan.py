#!/usr/bin/env python3
"""
Catalog delta scan

Downloads seed catalogs into data/catalogs when missing, normalizes the
machine-readable ones, and scans catalog observables against the same Delta
grid used by experiments/cgm_compact_geometry_core.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from html import unescape
from itertools import combinations, product
import re
from pathlib import Path
from typing import Iterable, List, Sequence
from urllib.request import Request, urlopen

import cgm_compact_geometry_core as compact

try:
    from pypdf import PdfReader

    PYPDF_AVAILABLE = True
except ImportError:
    PdfReader = None
    PYPDF_AVAILABLE = False


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = REPO_ROOT / "data" / "catalogs"
PDG_LISTING_DIR = CATALOG_DIR / "pdg_listings"

CATEGORY_NIST_STRONG_LINES = {
    "hydrogen": "https://www.physics.nist.gov/PhysRefData/Handbook/Tables/hydrogentable2_a.htm",
    "helium": "https://physics.nist.gov/PhysRefData/Handbook/Tables/heliumtable2_a.htm",
    "sodium": "https://www.physics.nist.gov/PhysRefData/Handbook/Tables/sodiumtable2_a.htm",
    "cesium": "https://physics.nist.gov/PhysRefData/Handbook/Tables/cesiumtable2_a.htm",
}

CATALOG_URLS = {
    "nist_hydrogen_strong_lines.html": "https://www.physics.nist.gov/PhysRefData/Handbook/Tables/hydrogentable2.htm",
    "nist_hydrogen_strong_lines_ascii.txt": CATEGORY_NIST_STRONG_LINES["hydrogen"],
    "pdg_particle_properties_2024.html": "https://pdg.lbl.gov/2024/listings/particle_properties.html",
    "pdg_booklet_2024.pdf": "https://pdg.lbl.gov/2024/download/db2024.pdf",
}

NIST_HYDROGEN_ASCII = CATALOG_DIR / "nist_hydrogen_strong_lines_ascii.txt"
PDG_INDEX_HTML = CATALOG_DIR / "pdg_particle_properties_2024.html"
NIST_LINES_CSV = CATALOG_DIR / "nist_hydrogen_lines.csv"
NIST_COMBINED_LINES_CSV = CATALOG_DIR / "nist_selected_strong_lines.csv"
PDG_INDEX_CSV = CATALOG_DIR / "pdg_listing_index.csv"
PDG_MASS_CSV = CATALOG_DIR / "pdg_selected_masses.csv"
MATCHES_CSV = CATALOG_DIR / "catalog_delta_matches.csv"
EXCLUDED_CORE_LENGTHS = frozenset({"Ly-alpha length"})

PDG_SELECTED_LISTINGS = {
    "electron": (
        "https://pdg.lbl.gov/2024/listings/rpp2024-list-electron.pdf",
        "MeV",
    ),
    "muon": (
        "https://pdg.lbl.gov/2024/listings/rpp2024-list-muon.pdf",
        "MeV",
    ),
    "tau": (
        "https://pdg.lbl.gov/2024/listings/rpp2024-list-tau.pdf",
        "MeV",
    ),
    "W boson": (
        "https://pdg.lbl.gov/2024/listings/rpp2024-list-w-boson.pdf",
        "GeV",
    ),
}

KNOWN_HYDROGEN_LABELS = {
    972.5367: "Ly-gamma",
    1025.7222: "Ly-beta",
    1215.66824: "Ly-alpha",
    1215.67364: "Ly-alpha (component)",
    4340.462: "H-gamma",
    4861.3615: "H-beta",
    6562.8518: "H-alpha",
    12818.07: "Paschen-beta",
    18751.01: "Paschen-alpha",
}


@dataclass(frozen=True)
class HydrogenLine:
    name: str
    medium: str
    intensity: int
    wavelength_angstrom: float
    wavelength_m: float
    frequency_hz: float
    source: str


@dataclass(frozen=True)
class StrongLine:
    element: str
    name: str
    medium: str
    intensity: int
    wavelength_angstrom: float
    wavelength_m: float
    frequency_hz: float
    source: str


@dataclass(frozen=True)
class PdgListing:
    section: str
    title: str
    pdf_url: str


@dataclass(frozen=True)
class PdgMass:
    name: str
    value_gev: float
    source: str
    pdf_file: str


@dataclass(frozen=True)
class MatchRow:
    scope: str
    lhs: str
    rhs: str
    delta_tick: float
    nearest_horizon: float
    horizon_error: float
    nearest_landmark: float
    landmark_error: float


def ensure_catalogs() -> None:
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in CATALOG_URLS.items():
        target = CATALOG_DIR / filename
        if target.exists():
            continue
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request) as response:
            target.write_bytes(response.read())
    for element, url in CATEGORY_NIST_STRONG_LINES.items():
        target = CATALOG_DIR / f"nist_{element}_strong_lines_ascii.txt"
        if target.exists():
            continue
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request) as response:
            target.write_bytes(response.read())


def ensure_selected_pdg_pdfs() -> None:
    PDG_LISTING_DIR.mkdir(parents=True, exist_ok=True)
    for name, (url, _) in PDG_SELECTED_LISTINGS.items():
        filename = name.lower().replace(" ", "_") + ".pdf"
        target = PDG_LISTING_DIR / filename
        if target.exists():
            continue
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request) as response:
            target.write_bytes(response.read())


def match_known_hydrogen_label(wavelength_angstrom: float) -> str:
    nearest_value = min(
        KNOWN_HYDROGEN_LABELS,
        key=lambda known_value: abs(wavelength_angstrom - known_value),
    )
    if abs(wavelength_angstrom - nearest_value) < 0.02:
        return KNOWN_HYDROGEN_LABELS[nearest_value]
    return f"H line {wavelength_angstrom:.4f} A"


def parse_nist_hydrogen_lines(path: Path) -> List[HydrogenLine]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    pre_match = re.search(r"<pre>(.*?)</pre>", text, re.IGNORECASE | re.DOTALL)
    if pre_match is None:
        raise RuntimeError(f"Could not find <pre> block in {path}")

    pre = unescape(pre_match.group(1))
    const = compact.build_constants()
    current_medium = "unknown"
    lines: List[HydrogenLine] = []

    row_pattern = re.compile(
        r"^\s*(\d+)\s+(?:P(?:,c)?\s+)?([0-9]+(?:\.[0-9]+)?)\s+H I\b"
    )

    for raw_line in pre.splitlines():
        line = re.sub(r"<[^>]+>", " ", raw_line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if line == "Vacuum":
            current_medium = "vacuum"
            continue
        if line == "Air":
            current_medium = "air"
            continue
        if "Wavelength" in line or line.startswith("Intensity"):
            continue

        match = row_pattern.match(line)
        if match is None:
            continue

        intensity = int(match.group(1))
        wavelength_angstrom = float(match.group(2))
        wavelength_m = wavelength_angstrom * 1.0e-10
        frequency_hz = const.c / wavelength_m
        label = match_known_hydrogen_label(wavelength_angstrom)
        if current_medium != "unknown":
            name = f"{label} [{current_medium}]"
        else:
            name = label

        lines.append(
            HydrogenLine(
                name=name,
                medium=current_medium,
                intensity=intensity,
                wavelength_angstrom=wavelength_angstrom,
                wavelength_m=wavelength_m,
                frequency_hz=frequency_hz,
                source="NIST hydrogen strong lines",
            )
        )

    return lines


def parse_nist_strong_lines(path: Path, element: str) -> List[StrongLine]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    pre_match = re.search(r"<pre>(.*?)</pre>", text, re.IGNORECASE | re.DOTALL)
    if pre_match is None:
        raise RuntimeError(f"Could not find <pre> block in {path}")

    pre = unescape(pre_match.group(1))
    const = compact.build_constants()
    current_medium = "unknown"
    lines: List[StrongLine] = []
    row_pattern = re.compile(r"^\s*(\d+)\s+(?:P|w|c|P,c)?\s*([0-9]+(?:\.[0-9]+)?)\s+([A-Za-z]+\s+[IVX]+)\b")

    for raw_line in pre.splitlines():
        line = re.sub(r"<[^>]+>", " ", raw_line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if line == "Vacuum":
            current_medium = "vacuum"
            continue
        if line == "Air":
            current_medium = "air"
            continue
        if "Wavelength" in line or line.startswith("Intensity"):
            continue
        match = row_pattern.match(line)
        if match is None:
            continue
        intensity = int(match.group(1))
        wavelength_angstrom = float(match.group(2))
        spectrum = match.group(3)
        wavelength_m = wavelength_angstrom * 1.0e-10
        frequency_hz = const.c / wavelength_m
        name = f"{element.title()} {spectrum} {wavelength_angstrom:.4f} A [{current_medium}]"
        lines.append(
            StrongLine(
                element=element,
                name=name,
                medium=current_medium,
                intensity=intensity,
                wavelength_angstrom=wavelength_angstrom,
                wavelength_m=wavelength_m,
                frequency_hz=frequency_hz,
                source=f"NIST {element.title()} strong lines",
            )
        )
    return lines


def parse_pdg_listing_index(path: Path) -> List[PdgListing]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    section = "unknown"
    listings: List[PdgListing] = []

    section_pattern = re.compile(
        r'data-toggle="collapse"\s+href="#[^"]+"[^>]*>([^<]+)</a>',
        re.IGNORECASE,
    )
    item_pattern = re.compile(
        r'<div class="list-group-item"><a class="iframe" href="\.\./web/viewer\.html\?file=(\.\./[^"]+\.pdf)"\s*>([^<]+)</a>',
        re.IGNORECASE,
    )

    for raw_line in text.splitlines():
        section_match = section_pattern.search(raw_line)
        if section_match is not None:
            section = unescape(section_match.group(1)).strip()

        item_match = item_pattern.search(raw_line)
        if item_match is None:
            continue

        relative_pdf = item_match.group(1).replace("../", "")
        title = unescape(item_match.group(2)).strip()
        pdf_url = f"https://pdg.lbl.gov/2024/{relative_pdf}"
        listings.append(PdgListing(section=section, title=title, pdf_url=pdf_url))

    return listings


def parse_first_mass_value(pdf_path: Path, unit: str) -> float:
    if not PYPDF_AVAILABLE:
        raise RuntimeError("pypdf is not available")

    reader = PdfReader(str(pdf_path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages[:3])
    safe = text.encode("ascii", "ignore").decode("ascii")
    pattern = rf"VALUE \({unit}\).*?\n\s*([0-9][0-9\.\s]+)"
    match = re.search(pattern, safe, re.IGNORECASE | re.DOTALL)
    if match is None:
        raise RuntimeError(f"Could not find VALUE ({unit}) block in {pdf_path.name}")

    raw = match.group(1)
    number_match = re.search(r"([0-9]+(?:\s*\.\s*[0-9]+)?)", raw)
    if number_match is None:
        raise RuntimeError(f"Could not parse first mass value from {pdf_path.name}")

    value = float(number_match.group(1).replace(" ", ""))
    if unit == "MeV":
        return value / 1000.0
    return value


def build_pdg_mass_rows() -> List[PdgMass]:
    if not PYPDF_AVAILABLE:
        return []

    ensure_selected_pdg_pdfs()
    rows: List[PdgMass] = []
    for name, (_, unit) in PDG_SELECTED_LISTINGS.items():
        filename = name.lower().replace(" ", "_") + ".pdf"
        pdf_path = PDG_LISTING_DIR / filename
        value_gev = parse_first_mass_value(pdf_path, unit)
        rows.append(
            PdgMass(
                name=name,
                value_gev=value_gev,
                source="PDG 2024 listing PDF",
                pdf_file=pdf_path.name,
            )
        )
    return rows


def read_pdg_mass_rows(path: Path) -> List[PdgMass]:
    if not path.exists():
        return []
    rows: List[PdgMass] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                PdgMass(
                    name=row["name"],
                    value_gev=float(row["value_gev"]),
                    source=row["source"],
                    pdf_file=row["pdf_file"],
                )
            )
    return rows


def write_csv(path: Path, rows: Sequence[object]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    first = rows[0]
    fieldnames = list(asdict(first).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_horizon_levels(max_delta: float, include_zero: bool = False) -> List[int]:
    landmarks = sorted({float(v) for v in compact.APERTURE_FRAME_LANDMARKS if v >= 0.0})
    if not landmarks:
        return [0] if include_zero else []
    unique: List[int] = []
    frame = compact.APERTURE_FRAME
    if frame <= 0.0:
        return []
    max_cycle = int(max_delta // frame) + 3
    for cycle in range(max_cycle + 1):
        base = cycle * frame
        for landmark in landmarks:
            level = int(base + landmark)
            if include_zero or level > 0:
                unique.append(level)
    return sorted(set(unique))


def make_hydrogen_observables(lines: Iterable[HydrogenLine]) -> List[compact.Observable]:
    observables: List[compact.Observable] = []
    for line in lines:
        observables.append(
            compact.Observable(
                name=line.name,
                family="catalog-hydrogen",
                dimension="length",
                value=line.wavelength_m,
                unit="m",
                source=line.source,
            )
        )
    return observables


def make_strong_line_observables(lines: Iterable[StrongLine]) -> List[compact.Observable]:
    return [
        compact.Observable(
            name=line.name,
            family=f"catalog-{line.element}",
            dimension="length",
            value=line.wavelength_m,
            unit="m",
            source=line.source,
        )
        for line in lines
    ]


def canonical_hydrogen_lines(lines: Sequence[HydrogenLine]) -> List[HydrogenLine]:
    best_by_name: dict[str, HydrogenLine] = {}
    for line in lines:
        if line.name.startswith("H line "):
            continue
        base_name = line.name.replace(" (component)", "")
        current = best_by_name.get(base_name)
        candidate = HydrogenLine(
            name=base_name,
            medium=line.medium,
            intensity=line.intensity,
            wavelength_angstrom=line.wavelength_angstrom,
            wavelength_m=line.wavelength_m,
            frequency_hz=line.frequency_hz,
            source=line.source,
        )
        if current is None or candidate.intensity > current.intensity:
            best_by_name[base_name] = candidate
    return sorted(best_by_name.values(), key=lambda row: row.wavelength_angstrom)


def make_pdg_mass_observables(rows: Iterable[PdgMass]) -> List[compact.Observable]:
    return [
        compact.Observable(
            name=row.name,
            family="catalog-pdg",
            dimension="energy",
            value=row.value_gev,
            unit="GeV",
            source=row.source,
        )
        for row in rows
    ]


def build_rows(
    observables: Sequence[compact.Observable],
    planck: compact.ScaleReference,
) -> List[compact.CoordinateResult]:
    return compact.build_coordinate_table(
        observables=observables,
        coordinate_fn=lambda obs: compact.aperture_delta_coordinate(obs, planck),
    )


def match_pair(
    scope: str,
    lhs: compact.CoordinateResult,
    rhs: compact.CoordinateResult,
) -> MatchRow:
    delta_tick = abs(lhs.coordinate - rhs.coordinate)
    residue, nearest_landmark, landmark_error = compact.nearest_phase_landmark(delta_tick)
    nearest_horizon = nearest_landmark
    horizon_levels = build_horizon_levels(max(0.0, delta_tick), include_zero=True)
    if horizon_levels:
        nearest_horizon = min(horizon_levels, key=lambda level: abs(delta_tick - level))
        horizon_error = abs(delta_tick - nearest_horizon)
    else:
        nearest_horizon = delta_tick
        horizon_error = residue
    return MatchRow(
        scope=scope,
        lhs=lhs.name,
        rhs=rhs.name,
        delta_tick=delta_tick,
        nearest_horizon=float(nearest_horizon),
        horizon_error=horizon_error,
        nearest_landmark=nearest_landmark,
        landmark_error=landmark_error,
    )


def scan_self_pairs(
    scope: str,
    rows: Sequence[compact.CoordinateResult],
) -> List[MatchRow]:
    return [match_pair(scope, lhs, rhs) for lhs, rhs in combinations(rows, 2)]


def scan_cross_pairs(
    scope: str,
    lhs_rows: Sequence[compact.CoordinateResult],
    rhs_rows: Sequence[compact.CoordinateResult],
) -> List[MatchRow]:
    return [match_pair(scope, lhs, rhs) for lhs, rhs in product(lhs_rows, rhs_rows)]


def scan_strong_line_pairs_by_element(
    scope: str,
    lines: Sequence[StrongLine],
    rows: Sequence[compact.CoordinateResult],
) -> List[MatchRow]:
    by_name = {row.name: row for row in rows}
    matches: List[MatchRow] = []
    for lhs, rhs in combinations(lines, 2):
        if lhs.element != rhs.element:
            continue
        matches.append(match_pair(scope, by_name[lhs.name], by_name[rhs.name]))
    return matches


def core_length_observables(
    observables: Sequence[compact.Observable],
) -> List[compact.Observable]:
    return [
        obs
        for obs in observables
        if (
            obs.dimension == "length"
            and not compact.is_reference_observable(obs.name)
            and obs.name not in EXCLUDED_CORE_LENGTHS
        )
    ]


def named_hydrogen_observables(
    observables: Sequence[compact.Observable],
) -> List[compact.Observable]:
    return [obs for obs in observables if not obs.name.startswith("H line ")]


def canonical_name(name: str) -> str:
    return name.replace(" (component)", "")


def dedupe_matches(matches: Sequence[MatchRow]) -> List[MatchRow]:
    seen: set[tuple[str, str, float, float]] = set()
    unique: List[MatchRow] = []
    for row in matches:
        lhs = canonical_name(row.lhs)
        rhs = canonical_name(row.rhs)
        pair = tuple(sorted((lhs, rhs)))
        key = (
            pair[0],
            pair[1],
            round(row.nearest_horizon, 6),
            round(row.nearest_landmark, 6),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            MatchRow(
                scope=row.scope,
                lhs=pair[0],
                rhs=pair[1],
                delta_tick=row.delta_tick,
                nearest_horizon=row.nearest_horizon,
                horizon_error=row.horizon_error,
                nearest_landmark=row.nearest_landmark,
                landmark_error=row.landmark_error,
            )
        )
    return unique


def print_header(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def print_match_table(title: str, matches: Sequence[MatchRow], key: str, limit: int = 10) -> None:
    print_header(title)
    deduped = dedupe_matches(matches)
    if not deduped:
        print("No matches available.")
        return
    print(
        f"{'Pair':58} {'Delta':>10} {'Nearest':>10} {'Error':>10}"
    )
    print("-" * 92)
    if key == "horizon":
        ordered = sorted(deduped, key=lambda row: (row.horizon_error, row.delta_tick))
        for row in ordered[:limit]:
            label = f"{row.lhs} <-> {row.rhs}"
            print(
                f"{label[:58]:58} {row.delta_tick:10.3f} "
                f"{row.nearest_horizon:10.0f} {row.horizon_error:10.3f}"
            )
    else:
        ordered = sorted(deduped, key=lambda row: (row.landmark_error, row.delta_tick))
        for row in ordered[:limit]:
            label = f"{row.lhs} <-> {row.rhs}"
            print(
                f"{label[:58]:58} {row.delta_tick:10.3f} "
                f"{row.nearest_landmark:10.3f} {row.landmark_error:10.3f}"
            )


def print_best_horizon_hits_by_level(
    title: str,
    matches: Sequence[MatchRow],
) -> None:
    print_header(title)
    deduped = dedupe_matches(matches)
    if not deduped:
        print("No matches available.")
        return
    print(f"{'Level':>8} {'BestDelta':>12} {'Error':>10} {'Pair':52}")
    print("-" * 88)
    horizon_levels = build_horizon_levels(
        max((row.nearest_horizon for row in deduped), default=0.0),
        include_zero=False,
    )
    for level in horizon_levels:
        best = min(deduped, key=lambda row: abs(row.delta_tick - level))
        label = f"{best.lhs} <-> {best.rhs}"
        print(
            f"{level:8d} {best.delta_tick:12.3f} {abs(best.delta_tick - level):10.3f} "
            f"{label[:52]:52}"
        )


def print_unfilled_horizon_levels(
    title: str,
    matches: Sequence[MatchRow],
    threshold: float,
) -> None:
    print_header(title)
    deduped = dedupe_matches(matches)
    if not deduped:
        print("No matches available.")
        return
    missing: List[int] = []
    horizon_levels = build_horizon_levels(
        max((row.nearest_horizon for row in deduped), default=0.0),
        include_zero=False,
    )
    for level in horizon_levels:
        best_error = min(abs(row.delta_tick - level) for row in deduped)
        if best_error > threshold:
            missing.append(level)
    if missing:
        print(", ".join(str(level) for level in missing))
    else:
        print("None")


def print_best_landmark_identities(
    title: str,
    matches: Sequence[MatchRow],
    landmarks: Sequence[int],
) -> None:
    print_header(title)
    deduped = [row for row in dedupe_matches(matches) if row.delta_tick >= 8.0]
    if not deduped:
        print("No matches available.")
        return
    print(f"{'Landmark':>10} {'BestDelta':>12} {'Error':>10} {'Pair':58}")
    print("-" * 96)
    for target in landmarks:
        candidates = [
            row for row in deduped
            if abs(row.nearest_landmark - target) < 1.0e-9
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda row: row.landmark_error)
        label = f"{best.lhs} <-> {best.rhs}"
        print(
            f"{target:10d} {best.delta_tick:12.3f} {best.landmark_error:10.3f} "
            f"{label[:58]:58}"
        )


def print_catalog_status(pdg_masses: Sequence[PdgMass]) -> None:
    print_header("Catalog Delta Scan")
    print(f"catalog_dir = {CATALOG_DIR}")
    print(f"hydrogen csv = {NIST_LINES_CSV}")
    print(f"nist combined csv = {NIST_COMBINED_LINES_CSV}")
    print(f"pdg index csv = {PDG_INDEX_CSV}")
    print(f"pdg mass csv = {PDG_MASS_CSV}")
    print(f"match csv = {MATCHES_CSV}")
    print(f"hydrogen lines loaded = {sum(1 for _ in NIST_LINES_CSV.open('r', encoding='utf-8')) - 1}")
    print(f"pdg listings indexed = {sum(1 for _ in PDG_INDEX_CSV.open('r', encoding='utf-8')) - 1}")
    print(f"pdg masses loaded = {len(pdg_masses)}")
    if not PYPDF_AVAILABLE and pdg_masses:
        print("pdg pdf extraction = cached csv")
    elif PYPDF_AVAILABLE:
        print("pdg pdf extraction = live pdf parse")
    else:
        print("pdg pdf extraction = unavailable")


def main() -> None:
    ensure_catalogs()

    const = compact.build_constants()
    planck = compact.planck_scales(const)
    repo_observables = compact.build_observables(const)
    core_lengths = core_length_observables(repo_observables)

    hydrogen_lines = parse_nist_hydrogen_lines(NIST_HYDROGEN_ASCII)
    combined_nist_lines: List[StrongLine] = []
    for element in CATEGORY_NIST_STRONG_LINES:
        path = CATALOG_DIR / f"nist_{element}_strong_lines_ascii.txt"
        combined_nist_lines.extend(parse_nist_strong_lines(path, element))
    pdg_listings = parse_pdg_listing_index(PDG_INDEX_HTML)

    pdg_mass_rows = build_pdg_mass_rows()
    if pdg_mass_rows:
        write_csv(PDG_MASS_CSV, pdg_mass_rows)
    else:
        pdg_mass_rows = read_pdg_mass_rows(PDG_MASS_CSV)

    write_csv(NIST_LINES_CSV, hydrogen_lines)
    write_csv(NIST_COMBINED_LINES_CSV, combined_nist_lines)
    write_csv(PDG_INDEX_CSV, pdg_listings)

    hydrogen_observables = make_hydrogen_observables(hydrogen_lines)
    combined_nist_observables = make_strong_line_observables(combined_nist_lines)
    pdg_mass_observables = make_pdg_mass_observables(pdg_mass_rows)
    named_hydrogen = make_hydrogen_observables(canonical_hydrogen_lines(hydrogen_lines))
    hydrogen_rows = build_rows(hydrogen_observables, planck)
    combined_nist_rows = build_rows(combined_nist_observables, planck)
    named_hydrogen_rows = build_rows(named_hydrogen, planck)
    core_rows = build_rows(core_lengths, planck)
    pdg_mass_rows_delta = build_rows(pdg_mass_observables, planck)

    hydrogen_self = scan_self_pairs("hydrogen-self", hydrogen_rows)
    combined_nist_self = scan_self_pairs("nist-selected-self", combined_nist_rows)
    combined_nist_same_element = scan_strong_line_pairs_by_element(
        "nist-selected-same-element",
        combined_nist_lines,
        combined_nist_rows,
    )
    named_hydrogen_self = scan_self_pairs("hydrogen-self-named", named_hydrogen_rows)
    pdg_mass_pairs = scan_self_pairs("pdg-mass-self", pdg_mass_rows_delta)
    pdg_vs_core = scan_cross_pairs("pdg-vs-core-energy", pdg_mass_rows_delta, build_rows(
        [obs for obs in repo_observables if obs.dimension == "energy"],
        planck,
    ))
    all_matches = sorted(
        [
            *hydrogen_self,
            *named_hydrogen_self,
            *combined_nist_self,
            *combined_nist_same_element,
            *pdg_mass_pairs,
            *pdg_vs_core,
        ],
        key=lambda row: (row.horizon_error, row.landmark_error, row.delta_tick),
    )
    write_csv(MATCHES_CSV, all_matches)

    print_catalog_status(pdg_mass_rows)

    print_best_horizon_hits_by_level(
        "Hydrogen Self-Pairs vs Horizon Ladder",
        hydrogen_self,
    )
    print_best_horizon_hits_by_level(
        "Named Hydrogen Pairs vs Horizon Ladder",
        named_hydrogen_self,
    )
    print_best_horizon_hits_by_level(
        "Selected NIST Strong-Line Pairs vs Horizon Ladder",
        combined_nist_self,
    )
    print_best_horizon_hits_by_level(
        "Selected NIST Same-Element Pairs vs Horizon Ladder",
        combined_nist_same_element,
    )
    print_best_horizon_hits_by_level(
        "PDG Mass Pairs vs Horizon Ladder",
        pdg_mass_pairs,
    )
    print_match_table(
        "Hydrogen Self-Pairs: Best Horizon Matches",
        hydrogen_self,
        key="horizon",
        limit=12,
    )
    print_match_table(
        "Hydrogen Self-Pairs: Best Architectural Matches",
        [row for row in hydrogen_self if row.delta_tick >= 8.0],
        key="landmark",
        limit=12,
    )
    print_match_table(
        "Named Hydrogen Pairs: Best Horizon Matches",
        named_hydrogen_self,
        key="horizon",
        limit=12,
    )
    print_match_table(
        "Named Hydrogen Pairs: Best Architectural Matches",
        [row for row in named_hydrogen_self if row.delta_tick >= 8.0],
        key="landmark",
        limit=12,
    )
    print_unfilled_horizon_levels(
        "Named Hydrogen: Unfilled Horizon Levels (> 1.5 ticks)",
        named_hydrogen_self,
        threshold=1.5,
    )
    print_match_table(
        "Selected NIST Strong-Line Pairs: Best Horizon Matches",
        combined_nist_self,
        key="horizon",
        limit=16,
    )
    print_match_table(
        "Selected NIST Strong-Line Pairs: Best Architectural Matches",
        [row for row in combined_nist_self if row.delta_tick >= 8.0],
        key="landmark",
        limit=16,
    )
    print_unfilled_horizon_levels(
        "Selected NIST Strong Lines: Unfilled Horizon Levels (> 1.5 ticks)",
        combined_nist_self,
        threshold=1.5,
    )
    print_match_table(
        "Selected NIST Same-Element Pairs: Best Horizon Matches",
        combined_nist_same_element,
        key="horizon",
        limit=16,
    )
    print_match_table(
        "Selected NIST Same-Element Pairs: Best Architectural Matches",
        [row for row in combined_nist_same_element if row.delta_tick >= 8.0],
        key="landmark",
        limit=16,
    )
    print_best_landmark_identities(
        "Selected NIST Same-Element: Landmark Identities",
        combined_nist_same_element,
        [12, 16, 32, 48, 64, 96, 128, 144, 192, 256],
    )
    print_unfilled_horizon_levels(
        "Selected NIST Same-Element: Unfilled Horizon Levels (> 1.5 ticks)",
        combined_nist_same_element,
        threshold=1.5,
    )
    print_match_table(
        "PDG Mass Pairs: Best Architectural Matches",
        pdg_mass_pairs,
        key="landmark",
        limit=12,
    )
    print_match_table(
        "PDG Mass vs Core Energies: Best Horizon Matches",
        pdg_vs_core,
        key="horizon",
        limit=12,
    )
    print_match_table(
        "PDG Mass vs Core Energies: Best Architectural Matches",
        pdg_vs_core,
        key="landmark",
        limit=12,
    )
    print_unfilled_horizon_levels(
        "PDG Mass vs Core Energies: Unfilled Horizon Levels (> 1.5 ticks)",
        pdg_vs_core,
        threshold=1.5,
    )


if __name__ == "__main__":
    main()
