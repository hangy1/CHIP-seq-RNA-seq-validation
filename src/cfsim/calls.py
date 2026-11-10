"""Expression labels, the on/off classifier, and evaluation metrics."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ON_TPM, OFF_TPM = 10.0, 1.0


def labels(tpm: np.ndarray) -> np.ndarray:
    """1 = expressed (TPM >= 10), 0 = silent (TPM < 1), -1 = ambiguous (excluded)."""
    y = np.full(len(tpm), -1)
    y[tpm >= ON_TPM] = 1
    y[tpm < OFF_TPM] = 0
    return y


def feature(props: np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(props) * 1e6)  # log(1 + CPM)


def fit_classifier(train_props: list[np.ndarray], train_tpm: list[np.ndarray]) -> LogisticRegression:
    X, y = [], []
    for p, tpm in zip(train_props, train_tpm):
        lab = labels(tpm)
        keep = lab >= 0
        X.append(feature(p[keep]))
        y.append(lab[keep])
    clf = LogisticRegression()
    clf.fit(np.concatenate(X)[:, None], np.concatenate(y))
    return clf


def lineage_markers(tpm_by_line: dict[str, np.ndarray], n=50) -> dict[str, np.ndarray]:
    """Genes on in exactly one line and off in all others (top n by TPM)."""
    lines = list(tpm_by_line)
    M = np.vstack([tpm_by_line[l] for l in lines])
    out = {}
    for i, l in enumerate(lines):
        others = np.delete(M, i, axis=0)
        cand = np.where((M[i] >= ON_TPM) & (others < OFF_TPM).all(axis=0))[0]
        out[l] = cand[np.argsort(M[i, cand])[::-1][:n]]
    return out


def evaluate(recovered: np.ndarray, tpm: np.ndarray, clf, pos_idx, neg_idx) -> dict:
    """Metrics for one recovered tumor profile against the held-out line's RNA."""
    if np.isnan(recovered).any():
        return {"spearman": np.nan, "auroc": np.nan, "sens": np.nan, "spec": np.nan, "bal_acc": np.nan}
    x = feature(recovered)
    lab = labels(tpm)
    keep = lab >= 0
    prob = clf.predict_proba(x[:, None])[:, 1]
    sens = float((prob[pos_idx] >= 0.5).mean())
    spec = float((prob[neg_idx] < 0.5).mean())
    return {
        "spearman": float(spearmanr(x, np.log1p(tpm)).statistic),
        "auroc": float(roc_auc_score(lab[keep], prob[keep])),
        "sens": sens, "spec": spec, "bal_acc": (sens + spec) / 2,
    }
