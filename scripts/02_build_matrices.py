#!/usr/bin/env python3
"""Promoter H3K4me3 matrix + RNA TPM matrix on a shared set of protein-coding genes.

Outputs (data/processed/):
  promoters.tsv   gene windows (TSS +/- flank)
  chip_fc.tsv     genes x cells, mean fold change over control in each promoter
  rna_tpm.tsv     genes x tumor cells, mean TPM across replicates
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfsim.features import promoter_signal_bigwig, promoters_from_gtf, rna_tpm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw")
ap.add_argument("--gtf", default="data/ref/gencode.v29.primary_assembly.annotation.gtf.gz")
ap.add_argument("--out", default="data/processed")
ap.add_argument("--flank", type=int, default=1000)
args = ap.parse_args()

raw, out = Path(args.raw), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
manifest = pd.read_csv(raw / "manifest.tsv", sep="\t")

prom = promoters_from_gtf(args.gtf, args.flank)
prom.to_csv(out / "promoters.tsv", sep="\t", index=False)
print(f"{len(prom)} protein-coding promoters")

chip = pd.DataFrame(index=prom.gene_id)
for rec in manifest[manifest.assay == "h3k4me3"].itertuples():
    print(f"promoter signal: {rec.cell}")
    chip[rec.cell] = promoter_signal_bigwig(raw / f"{rec.file}.bigWig", prom)

rna = pd.DataFrame(index=prom.gene_id)
for cell, grp in manifest[manifest.assay == "rna"].groupby("cell"):
    rna[cell] = rna_tpm([raw / f"{f}.tsv" for f in grp.file]).reindex(prom.gene_id)

missing = rna.isna().any(axis=1)
print(f"dropping {missing.sum()} genes absent from an RNA annotation")
chip, rna = chip[~missing], rna[~missing]
chip.insert(0, "gene_name", prom.set_index("gene_id").gene_name.reindex(chip.index))
rna.insert(0, "gene_name", chip.gene_name)
chip.to_csv(out / "chip_fc.tsv", sep="\t")
rna.to_csv(out / "rna_tpm.tsv", sep="\t")
print(f"wrote {len(chip)} genes; chip cells={list(chip.columns[1:])}; rna cells={list(rna.columns[1:])}")
