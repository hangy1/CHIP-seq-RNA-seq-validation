import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfsim.calls import labels, lineage_markers  # noqa: E402
from cfsim.features import promoters_from_gtf, strip_version  # noqa: E402
from cfsim.simulate import (TumorFractionEstimator, recover_tumor,  # noqa: E402
                            simulate_counts, to_props)

rng = np.random.default_rng(0)
G = 2000


def profiles():
    blood = to_props(rng.gamma(1, 1, G))
    blood[:100] *= 20                      # blood-specific promoters
    blood = blood / blood.sum()
    tumors = [to_props(np.r_[np.full(100, 0.05), rng.gamma(1, 1, G - 100)]) for _ in range(3)]
    return blood, tumors


def test_mixture_expectation():
    blood, (t, *_) = profiles()
    counts = simulate_counts(t, blood, 0.2, 2e6, 200, rng)
    assert counts.sum(axis=1).tolist() == [2_000_000] * 200
    np.testing.assert_allclose(counts.mean(0) / 2e6, 0.2 * t + 0.8 * blood, rtol=0.05, atol=2e-6)


@pytest.mark.parametrize("tf", [0.01, 0.1, 0.3])
def test_tf_estimator_is_unbiased_at_high_depth(tf):
    blood, tumors = profiles()
    est = TumorFractionEstimator(blood, tumors[1:], k=100)
    counts = simulate_counts(tumors[0], blood, tf, 5e7, 20, rng)
    assert abs(est(counts).mean() - tf) < 0.15 * tf + 0.002


def test_recover_tumor_is_exact_without_noise():
    blood, (t, *_) = profiles()
    q = 0.1 * t + 0.9 * blood
    rec = recover_tumor((q * 1e9)[None, :], blood, np.array([0.1]))[0]
    np.testing.assert_allclose(rec, t, atol=1e-8)


def test_zero_tf_gives_nan_profile():
    blood, (t, *_) = profiles()
    assert np.isnan(recover_tumor(blood[None, :] * 1e6, blood, np.array([0.0]))).all()


def test_labels_and_markers():
    assert labels(np.array([0.5, 5, 50])).tolist() == [0, -1, 1]
    m = lineage_markers({"a": np.array([50, 0, 50]), "b": np.array([0, 50, 50])})
    assert m["a"].tolist() == [0] and m["b"].tolist() == [1]


def test_strip_version_and_gtf(tmp_path):
    assert strip_version(pd.Series(["ENSG01.5", "ENSG02.1_PAR_Y"])).tolist() == ["ENSG01", "ENSG02"]
    gtf = tmp_path / "x.gtf"
    gtf.write_text(
        'chr1\tH\tgene\t5001\t9000\t.\t+\t.\tgene_id "ENSG1.1"; gene_type "protein_coding"; gene_name "A";\n'
        'chr1\tH\tgene\t5001\t9000\t.\t-\t.\tgene_id "ENSG2.1"; gene_type "protein_coding"; gene_name "B";\n'
        'chr1\tH\tgene\t1\t900\t.\t+\t.\tgene_id "ENSG3.1"; gene_type "lncRNA"; gene_name "C";\n')
    p = promoters_from_gtf(gtf, flank=1000)
    assert p.gene_id.tolist() == ["ENSG1", "ENSG2"]
    assert p.start.tolist() == [4000, 7999] and p.end.tolist() == [6001, 10000]
