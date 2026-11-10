#!/usr/bin/env python3
"""Analytical-validation metrics (LoB, LoD, precision) and Figures 2-4."""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfsim import plotstyle  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--figures", default="figures")
ap.add_argument("--call-acc", type=float, default=0.90, help="per-replicate balanced accuracy to count as a pass")
ap.add_argument("--hit-rate", type=float, default=0.95, help="fraction of replicates that must pass")
ap.add_argument("--roc-tfs", default="0.01,0.05,0.2")
args = ap.parse_args()
plotstyle.apply()

res = pd.read_csv(Path(args.results) / "sim_results.tsv", sep="\t")
roc = pd.read_csv(Path(args.results) / "roc_curves.tsv", sep="\t")
figdir = Path(args.figures)
figdir.mkdir(exist_ok=True)
lines = list(dict.fromkeys(res.line))
depths = sorted(res.depth_M.unique())
pos_tfs = sorted(t for t in res.tf.unique() if t > 0)


def lowest_passing(tf_pass: pd.Series):
    """Lowest TF from which every higher TF also passes (NaN if none)."""
    ok = tf_pass.sort_index()
    for tf in ok.index:
        if ok.loc[tf:].all():
            return tf
    return np.nan


summary = []
for (line, depth), g in res.groupby(["line", "depth_M"]):
    lob = g[g.tf == 0].tf_hat.quantile(0.95)
    pos = g[g.tf > 0]
    detect = pos.groupby("tf").tf_hat.apply(lambda s: (s > lob).mean() >= args.hit_rate)
    calls = pos.groupby("tf").bal_acc.apply(lambda s: (s >= args.call_acc).mean() >= args.hit_rate)
    at5 = pos[np.isclose(pos.tf, 0.05)]
    summary.append({
        "line": line, "depth_M": depth, "LoB_tf": lob,
        "LoD_tf_detect": lowest_passing(detect), "LoD_tf_calls": lowest_passing(calls),
        "tf_bias_at_5pct": at5.tf_hat.mean() - 0.05 if len(at5) else np.nan,
        "tf_CV_at_5pct": at5.tf_hat.std() / at5.tf_hat.mean() if len(at5) else np.nan,
        "AUROC_at_5pct": at5.auroc.mean() if len(at5) else np.nan,
    })
summary = pd.DataFrame(summary)
summary.to_csv(Path(args.results) / "validation_summary.tsv", sep="\t", index=False, float_format="%.4g")
print(summary.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

# ---- Figure 2: estimated vs true tumor fraction -----------------------------------
fig, axes = plt.subplots(1, len(lines), figsize=(3 * len(lines), 3.1), sharey=True)
for ax, line in zip(np.atleast_1d(axes), lines):
    g = res[(res.line == line) & (res.tf > 0)]
    ax.plot(pos_tfs, pos_tfs, ls="--", lw=1, color=plotstyle.MUTED, label="identity")
    for depth, color in zip(depths, plotstyle.CATEGORICAL):
        s = g[g.depth_M == depth].groupby("tf").tf_hat.agg(["mean", "std"])
        ax.fill_between(s.index, (s["mean"] - s["std"]).clip(lower=1e-3), s["mean"] + s["std"], color=color, alpha=0.15, lw=0)
        ax.plot(s.index, s["mean"], marker="o", color=color, label=f"{depth:g}M reads")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 0.5)
    ax.set_title(line)
    ax.set_xlabel("true tumor fraction")
np.atleast_1d(axes)[0].set_ylabel("estimated tumor fraction (mean ± SD)")
np.atleast_1d(axes)[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
fig.suptitle("Tumor-fraction estimate from loss of blood-specific promoter signal", x=0.01, y=1.04, ha="left")
fig.savefig(figdir / "fig2_tf_estimate.png")

# ---- Figure 3: call accuracy heatmap (TF x depth) ---------------------------------
fig, axes = plt.subplots(1, len(lines), figsize=(2.6 * len(lines) + 0.8, 3), sharey=True)
for ax, line in zip(np.atleast_1d(axes), lines):
    piv = (res[(res.line == line) & (res.tf > 0)]
           .groupby(["tf", "depth_M"]).bal_acc.mean().unstack("depth_M").reindex(index=pos_tfs, columns=depths))
    im = ax.imshow(piv.to_numpy(), cmap=plotstyle.BLUES, vmin=0.5, vmax=1, aspect="auto", origin="lower")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.8 else plotstyle.TEXT)
    ax.set_xticks(range(len(depths)), [f"{d:g}M" for d in depths])
    ax.set_yticks(range(len(pos_tfs)), [f"{t:g}" for t in pos_tfs])
    ax.grid(False)
    ax.set_title(line)
    ax.set_xlabel("total reads")
np.atleast_1d(axes)[0].set_ylabel("true tumor fraction")
fig.colorbar(im, ax=axes, shrink=0.85, label="marker-panel balanced accuracy")
fig.suptitle("Expression calls on lineage-marker genes", x=0.01, y=1.04, ha="left")
fig.savefig(figdir / "fig3_call_accuracy.png")

# ---- Figure 4: genome-wide ROC at a fixed depth ------------------------------------
roc_tfs = [float(t) for t in args.roc_tfs.split(",")]
fig, axes = plt.subplots(1, len(lines), figsize=(3 * len(lines), 3.1), sharey=True)
for ax, line in zip(np.atleast_1d(axes), lines):
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=plotstyle.MUTED)
    for tf, color in zip(roc_tfs, plotstyle.CATEGORICAL):
        g = roc[(roc.line == line) & np.isclose(roc.tf, tf)]
        if g.empty:
            continue
        auc = np.trapezoid(g.tpr, g.fpr)
        ax.plot(g.fpr, g.tpr, color=color, label=f"TF {tf:g}  (AUC {auc:.2f})")
    ax.set_title(line)
    ax.set_xlabel("false positive rate")
    ax.legend(loc="lower right", fontsize=7)
np.atleast_1d(axes)[0].set_ylabel("true positive rate")
fig.suptitle("Genome-wide expressed-vs-silent calls (held-out line, mean of replicates)", x=0.01, y=1.04, ha="left")
fig.savefig(figdir / "fig4_roc.png")
print(f"figures -> {figdir}/")
