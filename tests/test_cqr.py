"""Test CQR (Composite Quantile Regression) on simulated dataset."""

import numpy as np

from rehline import CQR_Ridge


def test_CQR_Ridge():
    """CQR_Ridge should fit and return output with the expected shapes."""
    np.random.seed(42)
    X = np.random.randn(1000, 2)
    beta = np.array([1, 2])
    y = X @ beta + np.random.randn(1000)
    sample_weight = np.random.rand(1000)

    quantiles = [0.05, 0.5, 0.95]
    n_qt = len(quantiles)

    cqr = CQR_Ridge(quantiles=quantiles)
    cqr.fit(X, y, sample_weight=sample_weight)

    assert cqr.coef_.shape == (len(beta),), (
        f"coef_ shape should be ({len(beta)},), got {cqr.coef_.shape}"
    )
    assert cqr.intercept_.shape == (n_qt,), (
        f"intercept_ shape should be ({n_qt},), got {cqr.intercept_.shape}"
    )
    assert cqr.quantiles_.shape == (n_qt,), (
        f"quantiles_ shape should be ({n_qt},), got {cqr.quantiles_.shape}"
    )

    pred = cqr.predict(X[:5])
    assert pred.shape == (5, n_qt), (
        f"predict output shape should be (5, {n_qt}), got {pred.shape}"
    )


def test_CQR_Ridge_monotone_quantiles():
    """Quantile predictions should be non-decreasing across quantile levels."""
    np.random.seed(0)
    X = np.random.randn(500, 3)
    y = X[:, 0] + np.random.randn(500)

    quantiles = [0.1, 0.5, 0.9]
    cqr = CQR_Ridge(quantiles=quantiles)
    cqr.fit(X, y)

    pred = cqr.predict(X)
    # Each column corresponds to one quantile level; they should be non-decreasing
    # (on average across samples, not necessarily per sample)
    mean_preds = pred.mean(axis=0)
    assert mean_preds[0] <= mean_preds[1] <= mean_preds[2], (
        f"Mean predictions should be non-decreasing across quantile levels, "
        f"got {mean_preds}"
    )
