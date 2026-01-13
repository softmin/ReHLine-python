import numpy as np
from sklearn.datasets import make_regression

from rehline import plq_Ridge_Regressor


def test_monotonic_increasing():
    """Test monotonic increasing constraint."""
    # Generate synthetic data
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

    # Define monotonic increasing constraint
    constraint = [{"name": "monotonic", "decreasing": False}]

    # Fit model
    clf = plq_Ridge_Regressor(loss={"name": "huber"}, constraint=constraint, C=1.0)
    clf.fit(X, y)

    # Check if coefficients are non-decreasing
    coef = clf.coef_
    diffs = np.diff(coef)

    # Allow for small numerical errors
    assert np.all(diffs >= -1e-3), f"Coefficients are not monotonic increasing: {coef}"


def test_monotonic_decreasing():
    """Test monotonic decreasing constraint."""
    # Generate synthetic data
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

    # Define monotonic decreasing constraint
    constraint = [{"name": "monotonic", "decreasing": True}]

    # Fit model
    clf = plq_Ridge_Regressor(loss={"name": "huber"}, constraint=constraint, C=1.0)
    clf.fit(X, y)

    # Check if coefficients are non-increasing
    coef = clf.coef_
    diffs = np.diff(coef)

    # Allow for small numerical errors
    assert np.all(diffs <= 1e-3), f"Coefficients are not monotonic decreasing: {coef}"


if __name__ == "__main__":
    test_monotonic_increasing()
    test_monotonic_decreasing()
    print("All monotonic constraint tests passed!")
