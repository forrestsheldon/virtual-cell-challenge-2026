from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.linear_response.model1_raw_counts import fit_raw_covariance_columns


def test_streaming_raw_covariance_matches_dense_calculation() -> None:
    counts = np.array(
        [
            [1, 0, 3, 2],
            [2, 1, 0, 4],
            [0, 2, 2, 1],
            [4, 1, 1, 0],
            [3, 3, 0, 2],
            [1, 2, 4, 1],
        ],
        dtype=np.float64,
    )
    data = ad.AnnData(
        sparse.csr_matrix(counts),
        obs=pd.DataFrame(index=[str(index) for index in range(len(counts))]),
    )
    rows = np.arange(len(counts))
    halves = np.array([0, 1, 0, 1, 0, 1])
    targets = np.array([1, 3])
    full, first, second, mean = fit_raw_covariance_columns(data, rows, targets, halves)
    assert np.allclose(full, np.cov(counts, rowvar=False)[:, targets])
    assert np.allclose(first, np.cov(counts[halves == 0], rowvar=False)[:, targets])
    assert np.allclose(second, np.cov(counts[halves == 1], rowvar=False)[:, targets])
    assert np.allclose(mean, counts.mean(0))
