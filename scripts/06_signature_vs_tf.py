#!/usr/bin/env python3
"""Fig 6: does each line's top Hallmark signature survive dilution into plasma? (10M reads)"""
import sys
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfsim import plotstyle  # noqa: E402
from cfsim.simulate import (TumorFractionEstimator, blood_profile, recover_tumor,  # noqa: E402
                            simulate_counts, to_props)

TFS = [0.02, 0.05, 0.1, 0.2, 0.3]
DEPTH_M, PROM_FRAC, REPS, EPS = 10, 0.25, 3, 1e-5
plotstyle.apply()

chip = pd.read_csv("data/processed/chip_fc.tsv", sep="\t", index_col=0)
rna = pd.read_csv("data/processed/rna_tpm.tsv", sep="\t", index_col=0)
lines = [c for c in rna.columns if c != "gene_name"]
blood_w = {"neutrophil": 0.7, "CD14_monocyte": 0.3}
p_blood = blood_profile({k: chip[k].to_numpy() for k in blood_w}, blood_w)
props = {c: to_props(chip[c].to_numpy()) for c in lines + list(blood_w)}
names = chip.gene_name.to_numpy()
hallmark = gp.get_library("MSigDB_Hallmark_2020", organism="Human")


def gsea(profile, line):
    """Preranked GSEA on tumor-specific promoter score vs all other cell types."""
    others = np.max([props[c] for c in props if c != line], axis=0)
    score = np.log2((profile + EPS) / (others + EPS))
    rnk = pd.Series(score, index=names).groupby(level=0).max().sort_values(ascending=False)
    res = gp.prerank(rnk=rnk, gene_sets=hallmark, permutation_num=200, seed=0, threads=4,
                     min_size=15, max_size=500, outdir=None, verbose=False).res2d
    return res.set_index("Term")[["NES", "FDR q-val"]].astype(float)


rng = np.random.default_rng(0)
rows = []
for line in lines:
    pure = gsea(props[line], line)
    term = pure.NES.idxmax()                      # signature defined on the pure profile
    rows.append({"line": line, "term": term, "tf": "pure",
                 "NES": pure.at[term, "NES"], "FDR": pure.at[term, "FDR q-val"]})
    est = TumorFractionEstimator(p_blood, [props[l] for l in lines if l != line])
    for tf in TFS:
        counts = simulate_counts(props[line], p_blood, tf, DEPTH_M * 1e6 * PROM_FRAC, REPS, rng)
        for rec in recover_tumor(counts, p_blood, est(counts)):
            if np.isnan(rec).any():
                continue
            res = gsea(rec, line)
            rows.append({"line": line, "term": term, "tf": tf,
                         "NES": res.NES.get(term, np.nan), "FDR": res["FDR q-val"].get(term, np.nan)})
    print(f"{line}: {term}")

df = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/signature_vs_tf.tsv", sep="\t", index=False, float_format="%.4g")

# ---- plot ----
xpos = {tf: i for i, tf in enumerate(TFS + ["pure"])}
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for (line, g), color in zip(df.groupby("line", sort=False), plotstyle.CATEGORICAL):
    s = g.groupby("tf", sort=False).agg(NES=("NES", "mean"), sd=("NES", "std"), FDR=("FDR", "median"))
    s = s.reindex(TFS + ["pure"])
    x = [xpos[t] for t in s.index]
    ax.errorbar(x, s.NES, yerr=s.sd.fillna(0), color=color, lw=2, capsize=2,
                label=f"{line}: {g.term.iloc[0]}")
    sig = s.FDR < 0.05
    ax.scatter(np.array(x)[sig], s.NES[sig], color=color, s=40, zorder=3)
    ax.scatter(np.array(x)[~sig], s.NES[~sig], facecolor="white", edgecolor=color, s=40, lw=1.5, zorder=3)
ax.set_xticks(list(xpos.values()), [f"{t:g}" if t != "pure" else "pure" for t in xpos])
ax.set_xlabel("tumor fraction (10M reads)")
ax.set_ylabel("NES of line's top Hallmark signature")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
ax.set_title("Pathway signature vs dilution   (filled = FDR < 0.05)", loc="left")
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/fig6_signature_vs_tf.png")
print("\n" + "="*80)
print(df.groupby(["line", "tf"], sort=False)[["NES", "FDR"]].mean().round(3).to_string())
print("="*80)
