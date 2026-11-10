#!/usr/bin/env python3
"""Simulate plasma at known tumor fractions and depths; score TF estimates and expression calls.

Leave-one-line-out: for each held-out tumor line, the TF estimator and the on/off
classifier are built only from the other lines, then applied to simulated plasma
containing the held-out line. RNA of the held-out line is used only as truth.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfsim.calls import evaluate, feature, fit_classifier, labels, lineage_markers  # noqa: E402
from cfsim.simulate import (TumorFractionEstimator, blood_profile, recover_tumor,  # noqa: E402
                            simulate_counts, to_props)

ap = argparse.ArgumentParser()
ap.add_argument("--processed", default="data/processed")
ap.add_argument("--results", default="results")
ap.add_argument("--tfs", default="0,0.005,0.01,0.02,0.05,0.1,0.2,0.3")
ap.add_argument("--depths", default="1,5,10,20", help="total reads, millions")
ap.add_argument("--promoter-frac", type=float, default=0.25,
                help="fraction of reads landing in promoter windows (FRiP-like)")
ap.add_argument("--blood-weights", default="neutrophil=0.7,CD14_monocyte=0.3")
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--roc-depth", type=float, default=10)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

chip = pd.read_csv(Path(args.processed) / "chip_fc.tsv", sep="\t", index_col=0)
rna = pd.read_csv(Path(args.processed) / "rna_tpm.tsv", sep="\t", index_col=0)
lines = [c for c in rna.columns if c != "gene_name"]
weights = {k: float(v) for k, v in (kv.split("=") for kv in args.blood_weights.split(","))}
weights = {k: v for k, v in weights.items() if k in chip.columns}
if not weights:
    sys.exit("no blood background columns found in chip_fc.tsv")
tfs = [float(x) for x in args.tfs.split(",")]
depths = [float(x) for x in args.depths.split(",")]

p_blood = blood_profile({k: chip[k].to_numpy() for k in weights}, weights)
props = {l: to_props(chip[l].to_numpy()) for l in lines}
tpm = {l: rna[l].to_numpy() for l in lines}
markers = lineage_markers(tpm)
print("lineage markers per line:", {l: len(m) for l, m in markers.items()})

rng = np.random.default_rng(args.seed)
rows, roc_rows = [], []
grid = np.linspace(0, 1, 101)
t0 = time.time()
for held in lines:
    train = [l for l in lines if l != held]
    est = TumorFractionEstimator(p_blood, [props[l] for l in train])
    clf = fit_classifier([props[l] for l in train], [tpm[l] for l in train])
    pos = markers[held]
    neg = np.concatenate([markers[l] for l in train])
    lab = labels(tpm[held])
    keep = lab >= 0
    clean = evaluate(props[held], tpm[held], clf, pos, neg)  # ceiling: undiluted, noiseless
    print(f"{held}: undiluted AUROC={clean['auroc']:.3f} bal_acc={clean['bal_acc']:.3f}")
    for depth in depths:
        n_prom = depth * 1e6 * args.promoter_frac
        for tf in tfs:
            counts = simulate_counts(props[held], p_blood, tf, n_prom, args.reps, rng)
            tf_hat = est(counts)
            rec = recover_tumor(counts, p_blood, tf_hat) if tf > 0 else None
            tprs = []
            for r in range(args.reps):
                m = evaluate(rec[r], tpm[held], clf, pos, neg) if tf > 0 else {}
                rows.append({"line": held, "depth_M": depth, "tf": tf, "rep": r,
                             "tf_hat": tf_hat[r], **m})
                if tf > 0 and depth == args.roc_depth and not np.isnan(rec[r]).any():
                    prob = clf.predict_proba(feature(rec[r])[keep][:, None])[:, 1]
                    fpr, tpr, _ = roc_curve(lab[keep], prob)
                    tprs.append(np.interp(grid, fpr, tpr))
            if tprs:
                roc_rows += [{"line": held, "tf": tf, "fpr": f, "tpr": t}
                             for f, t in zip(grid, np.mean(tprs, axis=0))]
    print(f"  done {held} ({time.time() - t0:.0f}s)")

out = Path(args.results)
out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out / "sim_results.tsv", sep="\t", index=False, float_format="%.5g")
pd.DataFrame(roc_rows).to_csv(out / "roc_curves.tsv", sep="\t", index=False, float_format="%.4g")
print(f"wrote {len(rows)} simulated samples -> {out / 'sim_results.tsv'}")
