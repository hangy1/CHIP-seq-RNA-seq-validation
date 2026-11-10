#!/usr/bin/env python3
"""Figure 1: does promoter H3K4me3 track expression in pure cell lines?"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfsim import plotstyle  # noqa: E402
from cfsim.calls import labels  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--processed", default="data/processed")
ap.add_argument("--figures", default="figures")
ap.add_argument("--results", default="results")
args = ap.parse_args()
plotstyle.apply()

chip = pd.read_csv(Path(args.processed) / "chip_fc.tsv", sep="\t", index_col=0)
rna = pd.read_csv(Path(args.processed) / "rna_tpm.tsv", sep="\t", index_col=0)
lines = [c for c in rna.columns if c != "gene_name"]

fig, axes = plt.subplots(1, len(lines), figsize=(3 * len(lines), 3), sharey=True, sharex=True)
rows = []
for ax, line in zip(np.atleast_1d(axes), lines):
    x = np.log2(chip[line].clip(lower=0) + 1)
    y = np.log10(rna[line] + 1)
    rho = spearmanr(x, y).statistic
    lab = labels(rna[line].to_numpy())
    rows.append({"line": line, "spearman": rho, "n_genes": len(x),
                 "median_fc_on": chip[line][lab == 1].median(),
                 "median_fc_off": chip[line][lab == 0].median()})
    ax.hexbin(x, y, gridsize=40, cmap=plotstyle.BLUES, mincnt=1, bins="log", linewidths=0)
    ax.set_title(f"{line}   ρ = {rho:.2f}")
    ax.set_xlabel("log2(promoter H3K4me3 FC + 1)")
np.atleast_1d(axes)[0].set_ylabel("log10(TPM + 1)")
fig.suptitle("Promoter H3K4me3 vs RNA expression, pure cell lines", x=0.01, y=1.04, ha="left")
Path(args.figures).mkdir(exist_ok=True)
fig.savefig(Path(args.figures) / "fig1_baseline.png")

Path(args.results).mkdir(exist_ok=True)
table = pd.DataFrame(rows)
table.to_csv(Path(args.results) / "baseline.tsv", sep="\t", index=False, float_format="%.3f")
print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
