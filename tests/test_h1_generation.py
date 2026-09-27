import numpy as np
from scipy import sparse

from scripts.evaluation.h1_generation import decode_lfc_genewise, expected_counts


def test_genewise_rounding_is_unbiased_and_exact_per_gene() -> None:
    rng = np.random.default_rng(1)
    raw = sparse.csr_matrix(rng.poisson(0.7, size=(400, 60)))
    lfc = rng.normal(0, 0.15, size=60)
    expected = expected_counts(raw, lfc)
    draws = np.stack(
        [decode_lfc_genewise(raw, lfc, np.random.default_rng(s)).toarray() for s in range(300)]
    )
    # Every gene's 400-cell total is within one count of its expectation, on every draw.
    assert np.all(np.abs(draws.sum(axis=1) - expected.sum(axis=0)) < 1 + 1e-9)
    # Entries are unbiased: the mean over draws approaches the expectation.
    se = np.sqrt(0.25 / len(draws))
    assert np.max(np.abs(draws.mean(axis=0) - expected)) < 6 * se
    # Zeros stay zero, and library sizes are preserved on average.
    assert np.all(draws[:, raw.toarray() == 0] == 0)
    assert abs(draws.sum(axis=2).mean() - raw.sum() / 400) < 0.05


def test_zero_change_returns_integer_expectation() -> None:
    raw = sparse.csr_matrix(np.random.default_rng(2).poisson(1.0, size=(50, 20)))
    out = decode_lfc_genewise(raw, np.zeros(20), np.random.default_rng(0))
    np.testing.assert_array_equal(out.toarray(), raw.toarray())
