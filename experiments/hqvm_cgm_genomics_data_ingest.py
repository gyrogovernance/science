#!/usr/bin/env python3
"""
hqvm_cgm_genomics_data_ingest.py

Download and freeze the genomic catalogs that hqvm_cgm_genomics_*.py actually
load under data/catalogs/genomics/, then write SOURCE.txt and MANIFEST.sha256.

NCBI translation tables are always emitted from the frozen map in
hqvm_cgm_genomics_common.py. Network objects are optional; --skip-network
records present files without downloading.

Companion: hqvm_cgm_genomics_run.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_REPO = Path(__file__).resolve().parents[1]
_EXP = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from hqvm_cgm_genomics_common import (
    CODE_NAMES,
    CODE_OVERRIDES,
    CODONS,
    DATA_DIR,
    NCBI_TABLE_IDS,
    STANDARD_CODE,
    sha256_file,
    translation_table,
)

USER_AGENT = "Mozilla/5.0 (CGM-hQVM-genomics-ingest)"
TIMEOUT_S = 300
REGULONDB_TIMEOUT_S = 600

# Files loaded by the genomics suite (scripts 1-8 + common).
URLS: Dict[str, str] = {
    "ecoli_k12_cds.fna.gz": (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/"
        "GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_cds_from_genomic.fna.gz"
    ),
    "ecoli_k12_full.fna.gz": (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/"
        "GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz"
    ),
    "yeast_s288c_cds.fna.gz": (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/146/045/"
        "GCF_000146045.2_R64/GCF_000146045.2_R64_cds_from_genomic.fna.gz"
    ),
    "sars_cov2.fna.gz": (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/009/858/895/"
        "GCF_009858895.2_ASM985889v3/GCF_009858895.2_ASM985889v3_genomic.fna.gz"
    ),
    "sars_cov2_cds.fna.gz": (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/009/858/895/"
        "GCF_009858895.2_ASM985889v3/GCF_009858895.2_ASM985889v3_cds_from_genomic.fna.gz"
    ),
    "chr22.fa.gz": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz",
    "gencode.v47.annotation.gtf.gz": (
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_47/"
        "gencode.v47.annotation.gtf.gz"
    ),
    "rebase_withrefm.txt": "https://ftp.neb.com/pub/rebase/withrefm.txt",
}

UNIPROT_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?query=(organism_id:83333)+AND+reviewed:true&format=txt"
)
UNIPROT_NAME = "ecoli_k12_uniprot.txt"

REGULONDB_GRAPHQL = (
    "https://regulondb.ccg.unam.mx/graphql",
    "https://regulondb.ccg.unam.mx/api/graphql",
)
REGULONDB_NAME = "regulondb_promoter_set.txt"
REGULONDB_QUERY = """
{
  getDataOfFile(fileName: "PromoterSet") {
    content
    version
    rdbVersion
    creationDate
    citation
  }
}
"""

# Expected consumers (documentation for SOURCE.txt).
CONSUMERS: Dict[str, str] = {
    "ncbi_genetic_codes.json": "common.translation_table",
    "ecoli_k12_cds.fna.gz": "scripts 2-8 load_named_fasta(ecoli_k12)",
    "ecoli_k12_full.fna.gz": "script 2 genealogy; script 6 replichore; script 8 compile",
    "yeast_s288c_cds.fna.gz": "scripts 2-8 load_named_fasta(yeast_s288c)",
    "sars_cov2.fna.gz": "scripts 2,7 load_named_fasta fallback",
    "sars_cov2_cds.fna.gz": "scripts 2,7 load_named_fasta(sars_cov2_cds)",
    "chr22.fa.gz": "common.load_chr22_sequence; scripts 2,5,6 splice/CDS",
    "gencode.v47.annotation.gtf.gz": "common.extract_chr22_cds; scripts 2,5,6",
    "rebase_withrefm.txt": "script 6 rebase_parity_census",
    "ecoli_k12_uniprot.txt": "script 6 uniprot_location_census",
    "regulondb_promoter_set.txt": "script 2 genealogy; script 8 compile",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    return ctx


def _download(url: str, dest: Path, force: bool = False) -> Tuple[str, Optional[str]]:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return "skip", sha256_file(dest)
    urls = [url]
    if url.startswith("https://hgdownload.soe.ucsc.edu/"):
        urls.append("http://" + url[len("https://") :])
    if url.startswith("https://ftp.neb.com/"):
        urls.append("ftp://" + url[len("https://") :])
    last_err = None
    for attempt in urls:
        req = Request(attempt, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(req, timeout=TIMEOUT_S, context=_ssl_context()) as resp:
                data = resp.read()
        except Exception as exc:
            last_err = exc
            continue
        if not data:
            last_err = RuntimeError("empty body")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return "wrote", _sha256_bytes(data)
    return f"FAIL {last_err}", None


def write_ncbi_tables() -> Path:
    tables = {}
    for tid in NCBI_TABLE_IDS:
        code = dict(STANDARD_CODE)
        code.update(CODE_OVERRIDES[tid])
        tables[str(tid)] = {
            "id": tid,
            "name": CODE_NAMES.get(tid, str(tid)),
            "aa": "".join(code[c] for c in CODONS),
        }
    payload = {
        "source": "NCBI transl_table overrides frozen in hqvm_cgm_genomics_common.py",
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "codon_order": "itertools.product(ACGT, repeat=3)",
        "tables": tables,
    }
    dest = DATA_DIR / "ncbi_genetic_codes.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _ = translation_table(1)
    return dest


def download_uniprot(force: bool = False) -> Tuple[str, str, Optional[str]]:
    dest = DATA_DIR / UNIPROT_NAME
    status, digest = _download(UNIPROT_URL, dest, force=force)
    return UNIPROT_NAME, status, digest


def download_regulondb(force: bool = False) -> Tuple[str, str, Optional[str], str]:
    """Fetch RegulonDB PromoterSet via GraphQL getDataOfFile."""
    dest = DATA_DIR / REGULONDB_NAME
    meta_note = "via=RegulonDB GraphQL getDataOfFile(fileName:PromoterSet)"
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return REGULONDB_NAME, "skip", sha256_file(dest), meta_note
    last_err: Optional[BaseException] = None
    body = json.dumps({"query": REGULONDB_QUERY}).encode("utf-8")
    for endpoint in REGULONDB_GRAPHQL:
        req = Request(
            endpoint,
            data=body,
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=REGULONDB_TIMEOUT_S, context=_ssl_context()) as resp:
                raw = resp.read()
            payload = json.loads(raw.decode("utf-8", "replace"))
            block = (payload.get("data") or {}).get("getDataOfFile") or {}
            content = block.get("content")
            if not content or not isinstance(content, str):
                raise RuntimeError(f"empty PromoterSet content from {endpoint}")
            text = content if content.endswith("\n") else content + "\n"
            dest.write_text(text, encoding="utf-8")
            digest = sha256_file(dest)
            meta_note = (
                f"via={endpoint} getDataOfFile(fileName:PromoterSet) "
                f"rdbVersion={block.get('rdbVersion')} "
                f"creationDate={block.get('creationDate')} "
                f"citation={block.get('citation')}"
            )
            return REGULONDB_NAME, "wrote", digest, meta_note
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
            last_err = exc
            continue
    return REGULONDB_NAME, f"FAIL {last_err}", None, meta_note


def write_source_txt(rows: List[str]) -> Path:
    dest = DATA_DIR / "SOURCE.txt"
    dest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return dest


def write_manifest(entries: List[Tuple[str, str, Optional[str]]]) -> Path:
    dest = DATA_DIR / "MANIFEST.sha256"
    lines = ["# file\tstatus\tsha256"]
    for name, status, digest in entries:
        lines.append(f"{name}\t{status}\t{digest or '-'}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def _record_present(name: str, entries: List[Tuple[str, str, Optional[str]]], notes: List[str]) -> None:
    dest = DATA_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        digest = sha256_file(dest)
        entries.append((name, "present", digest))
        notes.append(f"{name} present sha256={digest} consumer={CONSUMERS.get(name, '?')}")
        print(f"  present {name}")
    else:
        entries.append((name, "SKIP", None))
        notes.append(f"{name} SKIP consumer={CONSUMERS.get(name, '?')}")
        print(f"  SKIP {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="CGM-hQVM genomics catalog ingest")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="do not download; hash whatever is already on disk",
    )
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    source_rows = [
        f"ingest_utc={date}",
        (
            "assembly_notes="
            "E.coli GCF_000005845.2 (CDS + full replicon); "
            "yeast GCF_000146045.2 CDS; "
            "SARS-CoV-2 GCF_009858895.2; "
            "human hg38 chr22 + GENCODE v47; "
            "REBASE withrefm; UniProt reviewed taxid 83333; "
            "RegulonDB PromoterSet"
        ),
        "scope=only catalogs loaded by hqvm_cgm_genomics_*.py",
    ]
    entries: List[Tuple[str, str, Optional[str]]] = []

    ncbi = write_ncbi_tables()
    digest = sha256_file(ncbi)
    entries.append((ncbi.name, "wrote", digest))
    source_rows.append(
        f"{ncbi.name} wrote sha256={digest} consumer={CONSUMERS[ncbi.name]}"
    )
    print(f"  wrote {ncbi.name}")

    if args.skip_network:
        print("  skip-network")
        for name in URLS:
            _record_present(name, entries, source_rows)
        _record_present(UNIPROT_NAME, entries, source_rows)
        _record_present(REGULONDB_NAME, entries, source_rows)
    else:
        for name, url in URLS.items():
            dest = DATA_DIR / name
            print(f"  GET {name}")
            status, digest = _download(url, dest, force=args.force)
            print(f"  {status} {name}")
            entries.append((name, status, digest))
            source_rows.append(
                f"{name} {status} sha256={digest} url={url} "
                f"consumer={CONSUMERS.get(name, '?')}"
            )

        print(f"  GET {UNIPROT_NAME}")
        name, status, digest = download_uniprot(force=args.force)
        print(f"  {status} {name}")
        entries.append((name, status, digest))
        source_rows.append(
            f"{name} {status} sha256={digest} url={UNIPROT_URL} "
            f"consumer={CONSUMERS[name]}"
        )

        print(f"  GET {REGULONDB_NAME}")
        name, status, digest, meta = download_regulondb(force=args.force)
        print(f"  {status} {name}")
        entries.append((name, status, digest))
        source_rows.append(
            f"{name} {status} sha256={digest} {meta} consumer={CONSUMERS[name]}"
        )

    write_source_txt(source_rows)
    man = write_manifest(entries)
    print(f"  wrote {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
