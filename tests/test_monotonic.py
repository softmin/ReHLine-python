"""Test monotonic constraints for plq_Ridge_Regressor."""

import numpy as np
from sklearn.datasets import make_regression

from rehline import plq_Ridge_Regressor


def test_monotonic_increasing():
    """Coefficients should be non-decreasing when a monotonic-increasing constraint is set."""
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

    constraint = [{"name": "monotonic", "decreasing": False}]
    clf = plq_Ridge_Regressor(loss={"name": "huber"}, constraint=constraint, C=1.0)
    clf.fit(X, y)

    diffs = np.diff(clf.coef_)
    assert np.all(diffs >= -1e-3), (
        f"Coefficients are not monotonically increasing: {clf.coef_}"
    )


def test_monotonic_decreasing():
    """Coefficients should be non-increasing when a monotonic-decreasing constraint is set."""
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

    constraint = [{"name": "monotonic", "decreasing": True}]
    clf = plq_Ridge_Regressor(loss={"name": "huber"}, constraint=constraint, C=1.0)
    clf.fit(X, y)

    diffs = np.diff(clf.coef_)
    assert np.all(diffs <= 1e-3), (
        f"Coefficients are not monotonically decreasing: {clf.coef_}"
    )
