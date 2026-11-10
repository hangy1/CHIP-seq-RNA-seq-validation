"""Promoter windows, promoter H3K4me3 signal, and RNA TPM matrices."""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

MAIN_CHROMS = {f"chr{i}" for i in list(range(1, 23)) + ["X"]}  # chrY dropped: MCF-7/HepG2/K562 lack it


def strip_version(gene_id: pd.Series) -> pd.Series:
    return gene_id.str.replace(r"\.\d+(_PAR_Y)?$", "", regex=True)


def promoters_from_gtf(gtf: str | Path, flank: int = 1000) -> pd.DataFrame:
    """One window per protein-coding gene: annotated gene start (strand-aware) +/- flank."""
    opener = gzip.open if str(gtf).endswith(".gz") else open
    rows = []
    attr = re.compile(r'(gene_id|gene_name|gene_type) "([^"]+)"')
    with opener(gtf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] != "gene" or f[0] not in MAIN_CHROMS:
                continue
            a = dict(attr.findall(f[8]))
            if a.get("gene_type") != "protein_coding":
                continue
            tss = int(f[3]) - 1 if f[6] == "+" else int(f[4]) - 1  # 0-based
            rows.append((a["gene_id"], a["gene_name"], f[0], max(0, tss - flank), tss + flank + 1, f[6]))
    df = pd.DataFrame(rows, columns=["gene_id", "gene_name", "chrom", "start", "end", "strand"])
    df["gene_id"] = strip_version(df.gene_id)
    return df.drop_duplicates("gene_id").reset_index(drop=True)


def promoter_signal_bigwig(bw_path: str | Path, promoters: pd.DataFrame) -> np.ndarray:
    """Mean fold-change-over-control in each promoter window (0 where no coverage)."""
    import pyBigWig  # imported lazily so the rest of the package works without it

    out = np.zeros(len(promoters))
    with pyBigWig.open(str(bw_path)) as bw:
        chroms = bw.chroms()
        for i, (c, s, e) in enumerate(zip(promoters.chrom, promoters.start, promoters.end)):
            if c not in chroms:
                continue
            v = bw.stats(c, int(s), min(int(e), chroms[c]), type="mean")[0]
            out[i] = v if v is not None and np.isfinite(v) else 0.0
    return out


def rna_tpm(paths: list[str | Path]) -> pd.Series:
    """Average TPM across ENCODE RSEM gene-quantification replicates, keyed by unversioned gene_id."""
    series = []
    for p in paths:
        df = pd.read_csv(p, sep="\t", usecols=["gene_id", "TPM"])
        df = df[df.gene_id.str.startswith("ENSG") & ~df.gene_id.str.endswith("_PAR_Y")]
        series.append(df.assign(gene_id=strip_version(df.gene_id)).groupby("gene_id").TPM.sum())
    return pd.concat(series, axis=1).mean(axis=1)
