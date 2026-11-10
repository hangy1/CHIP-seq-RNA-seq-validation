"""Find and download processed GRCh38 files for ENCODE experiments via the REST API."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.encodeproject.org"
FIELDS = [
    "accession", "href", "output_type", "file_format", "assembly", "status",
    "biological_replicates", "genome_annotation", "md5sum", "file_size",
    "date_created", "preferred_default",
]
# What we want from each assay type
WANT = {
    "h3k4me3": {"output_type": "fold change over control", "file_format": "bigWig"},
    "rna": {"output_type": "gene quantifications", "file_format": "tsv"},
}


def list_files(accession: str, session: requests.Session | None = None) -> pd.DataFrame:
    s = session or requests.Session()
    params = [("type", "File"), ("dataset", f"/experiments/{accession}/"),
              ("assembly", "GRCh38"), ("status", "released"),
              ("format", "json"), ("limit", "all")]
    params += [("field", f) for f in FIELDS]
    r = s.get(f"{BASE}/search/", params=params, headers={"Accept": "application/json"}, timeout=60)
    if r.status_code == 404:  # ENCODE returns 404 when a search has no hits
        return pd.DataFrame(columns=FIELDS)
    r.raise_for_status()
    df = pd.DataFrame(r.json().get("@graph", []))
    for f in FIELDS:
        if f not in df:
            df[f] = None
    return df[FIELDS]


def select_files(files: pd.DataFrame, assay: str) -> pd.DataFrame:
    """ChIP: one pooled fold-change bigWig. RNA: all replicate gene quantifications."""
    want = WANT[assay]
    hits = files[(files.output_type == want["output_type"]) &
                 (files.file_format == want["file_format"])].copy()
    if hits.empty:
        return hits
    if assay == "h3k4me3":
        hits["n_reps"] = hits.biological_replicates.apply(lambda x: len(x) if isinstance(x, list) else 0)
        hits["pref"] = hits.preferred_default.fillna(False).astype(bool)
        hits = hits.sort_values(["pref", "n_reps", "date_created"], ascending=False).head(1)
        return hits.drop(columns=["n_reps", "pref"])
    return hits  # RNA: keep each replicate; they are averaged later


def build_manifest(samples: pd.DataFrame) -> pd.DataFrame:
    s = requests.Session()
    rows = []
    for rec in samples.itertuples():
        chosen = select_files(list_files(rec.accession, s), rec.assay)
        if chosen.empty:
            print(f"WARNING: no GRCh38 {WANT[rec.assay]['output_type']} for {rec.sample} ({rec.accession})")
            continue
        for f in chosen.itertuples():
            rows.append({"sample": rec.sample, "cell": rec.cell, "assay": rec.assay,
                         "role": rec.role, "experiment": rec.accession,
                         "file": f.accession, "url": BASE + f.href,
                         "format": f.file_format, "annotation": f.genome_annotation,
                         "md5": f.md5sum, "size_mb": round((f.file_size or 0) / 1e6, 1)})
    return pd.DataFrame(rows)


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, expected_md5: str | None = None) -> Path:
    if dest.exists() and (not expected_md5 or md5(dest) == expected_md5):
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for block in r.iter_content(1 << 20):
                fh.write(block)
    if expected_md5 and md5(tmp) != expected_md5:
        tmp.unlink()
        raise IOError(f"md5 mismatch for {url}")
    tmp.rename(dest)
    return dest
