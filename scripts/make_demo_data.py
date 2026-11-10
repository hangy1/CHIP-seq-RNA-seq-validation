#!/usr/bin/env python3
"""Synthetic stand-in for the ENCODE matrices, so the pipeline can be run end to end
without downloads (for testing only; results are not biological)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/demo/processed")
ap.add_argument("--genes", type=int, default=8000)
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

G = args.genes
tumors = ["MCF-7", "A549", "HepG2", "K562"]
blood = ["neutrophil", "CD14_monocyte"]
housekeeping = rng.random(G) < 0.35          # on everywhere
programs = {c: rng.random(G) < 0.06 for c in tumors + ["blood"]}  # lineage-specific genes

expr = {}
for c in tumors + blood:
    on = housekeeping | programs["blood" if c in blood else c]
    on = on ^ (rng.random(G) < 0.03)          # some cell-specific noise
    level = np.where(on, rng.lognormal(3.0, 1.2, G), rng.lognormal(-1.5, 1.0, G))
    expr[c] = level

chip = {c: 1 + 6 * np.log1p(expr[c]) / np.log1p(expr[c]).max() * rng.lognormal(0, 0.35, G)
        + rng.gamma(2, 0.1, G) for c in tumors + blood}

ids = [f"ENSG{i:011d}" for i in range(G)]
names = [f"GENE{i}" for i in range(G)]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"gene_name": names, **chip}, index=pd.Index(ids, name="gene_id")).to_csv(out / "chip_fc.tsv", sep="\t")
pd.DataFrame({"gene_name": names, **{c: expr[c] for c in tumors}},
             index=pd.Index(ids, name="gene_id")).to_csv(out / "rna_tpm.tsv", sep="\t")
print(f"demo matrices -> {out}")
