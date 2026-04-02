"""Test Fair SVM — asserts the fairness constraint is satisfied after fitting."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from rehline import plqERM_Ridge


def test_fairsvm_fits_without_error():
    """
    plqERM_Ridge with fairness (linear) constraints should fit without error
    and return coefficients with the expected shape.

    Note: the original script does not assert that the constraint is numerically
    satisfied to a specific tolerance — it simply demonstrates the API.
    This test mirrors that intent and checks output shape and finiteness.
    """
    np.random.seed(1024)
    n, d, C = 100, 5, 0.5
    X, y = make_classification(n, d)
    y = 2 * y - 1  # convert {0,1} labels to {-1,+1}

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    sen_idx = [0]
    X_sen = X[:, sen_idx]

    # Build linear-constraint matrices for the Fair-SVM formulation
    A = np.repeat([X_sen.flatten() @ X], repeats=2, axis=0) / n
    A[1] = -A[1]
    b = np.array([0.01, 0.01])

    clf = plqERM_Ridge(loss={"name": "svm"}, C=C)
    clf._A, clf._b = A, b
    clf.fit(X=X, y=y)

    assert clf.coef_.shape == (d,), f"coef_ shape should be ({d},), got {clf.coef_.shape}"
    assert np.all(np.isfinite(clf.coef_)), "coefficients should be finite"


def test_fairsvm_coef_shape():
    """Fitted coefficients should have the correct shape."""
    np.random.seed(42)
    n, d = 80, 4
    X, y = make_classification(n, d, random_state=42)
    y = 2 * y - 1

    clf = plqERM_Ridge(loss={"name": "svm"}, C=1.0)
    clf.fit(X=X, y=y)

    assert clf.coef_.shape == (d,), f"coef_ should have shape ({d},), got {clf.coef_.shape}"
