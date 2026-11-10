#!/usr/bin/env python3
"""Resolve ENCODE experiments to GRCh38 files, write a manifest, optionally download.

    python scripts/01_fetch_encode.py                 # manifest only (check it first)
    python scripts/01_fetch_encode.py --download      # then fetch files + GENCODE GTF
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfsim.encode import build_manifest, download  # noqa: E402

GTF_URL = ("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_29/"
           "gencode.v29.primary_assembly.annotation.gtf.gz")

ap = argparse.ArgumentParser()
ap.add_argument("--samples", default="config/samples.tsv")
ap.add_argument("--out", default="data/raw")
ap.add_argument("--download", action="store_true")
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
manifest_path = out / "manifest.tsv"

if manifest_path.exists():
    manifest = pd.read_csv(manifest_path, sep="\t")
else:
    manifest = build_manifest(pd.read_csv(args.samples, sep="\t"))
    manifest.to_csv(manifest_path, sep="\t", index=False)

print(manifest[["sample", "file", "format", "annotation", "size_mb"]].to_string(index=False))
print(f"\nTotal: {manifest.size_mb.sum() / 1000:.1f} GB in {len(manifest)} files -> {manifest_path}")
if manifest.annotation.nunique() > 1:
    print("NOTE: RNA files use different GENCODE versions; genes are joined on unversioned IDs.")

if args.download:
    for rec in manifest.itertuples():
        dest = out / f"{rec.file}.{'bigWig' if rec.format == 'bigWig' else 'tsv'}"
        print(f"downloading {rec.sample} {rec.file} ({rec.size_mb} MB)")
        download(rec.url, dest, rec.md5)
    gtf = Path("data/ref") / Path(GTF_URL).name
    gtf.parent.mkdir(parents=True, exist_ok=True)
    print("downloading GENCODE v29 GTF")
    download(GTF_URL, gtf)
    print("done")
