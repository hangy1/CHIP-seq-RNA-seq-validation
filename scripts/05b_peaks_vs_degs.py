#!/usr/bin/env python3
"""Fig 5: do top tumor-specific peaks from plasma find real DEGs?

Precision@100: among the top 100 recovered tumor-specific promoters, what fraction
are true DEGs (on in that line, off in others)?
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfsim import plotstyle  # noqa: E402
from cfsim.simulate import (TumorFractionEstimator, blood_profile, recover_tumor,  # noqa: E402
                            simulate_counts, to_props)

TFS = [0.02, 0.05, 0.1, 0.2, 0.3]
DEPTH_M, PROM_FRAC, REPS, K = 10, 0.25, 20, 100  # precision@K
ON_CUTOFF, OFF_CUTOFF = 10, 1  # TPM thresholds for DEG definition
plotstyle.apply()

chip = pd.read_csv("data/processed/chip_fc.tsv", sep="\t", index_col=0)
rna = pd.read_csv("data/processed/rna_tpm.tsv", sep="\t", index_col=0)
lines = [c for c in rna.columns if c != "gene_name"]
blood_w = {"neutrophil": 0.7, "CD14_monocyte": 0.3}
p_blood = blood_profile({k: chip[k].to_numpy() for k in blood_w}, blood_w)
props = {c: to_props(chip[c].to_numpy()) for c in lines}


def degs_for_line(rna, line, on_cutoff=10, off_cutoff=1):
    """Boolean array: genes on (TPM >= on_cutoff) in line and off (TPM < off_cutoff) in all others."""
    others = [c for c in lines if c != line]
    on_here = rna[line] >= on_cutoff
    off_elsewhere = rna[others].max(axis=1) < off_cutoff
    return (on_here & off_elsewhere).to_numpy()


def precision_at_k(recovered_profile, degs, k=100, eps=1e-5):
    """Precision@k: fraction of top-k tumor-specific peaks that are DEGs."""
    other_max = np.max([props[c] for c in props if c != line], axis=0)
    score = np.log2((recovered_profile + eps) / (other_max + eps))
    top_k_idx = np.argsort(score)[::-1][:k]
    return degs[top_k_idx].mean()


rng = np.random.default_rng(0)
rows = []

for line in lines:
    degs = degs_for_line(rna, line, ON_CUTOFF, OFF_CUTOFF)
    n_degs = degs.sum()
    chance = n_degs / len(degs)  # expected precision if ranking were random

    # Pure profile
    prec = precision_at_k(props[line], degs, K)
    rows.append({"line": line, "tf": "pure", "precision": prec, "chance": chance, "n_degs": n_degs})

    # Simulated plasma
    est = TumorFractionEstimator(p_blood, [props[l] for l in lines if l != line])
    for tf in TFS:
        counts = simulate_counts(props[line], p_blood, tf, DEPTH_M * 1e6 * PROM_FRAC, REPS, rng)
        tf_hat = est(counts)
        recovered = recover_tumor(counts, p_blood, tf_hat)

        prec_reps = []
        for rec in recovered:
            if not np.isnan(rec).any():
                prec_reps.append(precision_at_k(rec, degs, K))

        if prec_reps:
            rows.append({"line": line, "tf": tf, "precision": np.mean(prec_reps),
                        "sd": np.std(prec_reps), "chance": chance, "n_degs": n_degs})

df = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/peaks_vs_degs.tsv", sep="\t", index=False, float_format="%.4g")

# ---- plot ----
xpos = {tf: i for i, tf in enumerate(TFS + ["pure"])}
fig, ax = plt.subplots(figsize=(6, 3.6))

for (line, g), color in zip(df.groupby("line", sort=False), plotstyle.CATEGORICAL):
    s = g.set_index("tf")[["precision", "sd", "chance"]].reindex(TFS + ["pure"])
    x = [xpos[t] for t in s.index]
    ax.errorbar(x, s.precision * 100, yerr=s.sd.fillna(0) * 100,
                color=color, lw=2, capsize=2, label=f"{line} ({g.n_degs.iloc[0]} DEGs)")
    # Chance line for this cell line
    chance_val = g.chance.iloc[0] * 100
    ax.plot(x, [chance_val] * len(x), color=color, ls="--", lw=1, alpha=0.4)

ax.set_xticks(list(xpos.values()), [f"{t:g}" if t != "pure" else "pure" for t in xpos])
ax.set_xlabel("tumor fraction (10M reads)")
ax.set_ylabel(f"precision@{K} (%)")
ax.set_ylim(0, 100)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
ax.set_title(f"Top {K} tumor-specific peaks find real DEGs   (dashed = chance)", loc="left")
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/fig5_peaks_vs_degs.png")

print("\n" + "="*80)
print(f"PRECISION@{K} SUMMARY (DEGs: TPM ≥{ON_CUTOFF} in line, <{OFF_CUTOFF} in others)")
print("="*80)
for line in lines:
    g = df[df.line == line]
    print(f"\n{line} ({g.n_degs.iloc[0]} DEGs, chance = {g.chance.iloc[0]*100:.1f}%):")
    for _, row in g.iterrows():
        tf_str = "pure" if row.tf == "pure" else f"{row.tf:g}"
        sd_str = f" ± {row.sd*100:.1f}" if "sd" in row and not pd.isna(row.sd) else ""
        print(f"  {tf_str:6s}: {row.precision*100:5.1f}%{sd_str}")
print("="*80)
