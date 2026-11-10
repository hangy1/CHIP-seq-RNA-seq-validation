"""Count-space plasma simulation, tumor-fraction estimation and tumor-signal recovery.

Model: each promoter's share of promoter-mapped reads is proportional to its mean
H3K4me3 fold change. A plasma sample at tumor fraction tf is a multinomial draw
from  tf * p_tumor + (1 - tf) * p_blood.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def to_props(signal: np.ndarray) -> np.ndarray:
    s = np.clip(np.nan_to_num(np.asarray(signal, dtype=float)), 0, None)
    return s / s.sum()


def blood_profile(blood: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = sum(weights.values())
    mix = sum(weights[k] / total * to_props(blood[k]) for k in weights)
    return mix / mix.sum()


def simulate_counts(p_tumor, p_blood, tf, n_reads, reps, rng) -> np.ndarray:
    p = tf * p_tumor + (1 - tf) * p_blood
    return rng.multinomial(int(n_reads), p / p.sum(), size=reps).astype(np.int32)


def blood_specific_genes(p_blood, train_tumor_props, k=200) -> np.ndarray:
    """Promoters strongly marked in blood but not in any *training* tumor line."""
    max_tumor = np.max(np.vstack(train_tumor_props), axis=0)
    score = np.log2((p_blood + EPS) / (max_tumor + EPS))
    eligible = p_blood > np.median(p_blood)
    score = np.where(eligible, score, -np.inf)
    return np.argsort(score)[::-1][:k]


class TumorFractionEstimator:
    """tf = (b_B - q_B) / (b_B - t_B): loss of blood-specific signal, tumor-agnostic.

    b_B: blood share of reads at blood-specific promoters B.
    t_B: expected tumor share at B, taken from training lines only.
    """

    def __init__(self, p_blood, train_tumor_props, k=200):
        self.idx = blood_specific_genes(p_blood, train_tumor_props, k)
        self.b = p_blood[self.idx].sum()
        self.t = float(np.mean([p[self.idx].sum() for p in train_tumor_props]))

    def __call__(self, counts: np.ndarray) -> np.ndarray:
        counts = np.atleast_2d(counts)
        q = counts[:, self.idx].sum(axis=1) / counts.sum(axis=1)
        return np.clip((self.b - q) / (self.b - self.t), 0.0, 1.0)


def recover_tumor(counts, p_blood, tf_hat, min_tf=1e-3) -> np.ndarray:
    """Subtract expected blood contribution; returns per-sample tumor proportions (NaN if tf_hat ~ 0)."""
    counts = np.atleast_2d(counts).astype(float)
    q = counts / counts.sum(axis=1, keepdims=True)
    tf = np.atleast_1d(tf_hat)[:, None]
    t = np.clip((q - (1 - tf) * p_blood[None, :]) / np.maximum(tf, min_tf), 0, None)
    t = t / np.maximum(t.sum(axis=1, keepdims=True), EPS)
    t[np.atleast_1d(tf_hat) < min_tf] = np.nan
    return t
