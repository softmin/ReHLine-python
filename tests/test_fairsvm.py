"""Test Fair SVM — asserts the fairness constraint is satisfied after fitting."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from rehline import plqERM_Ridge


def test_fairsvm_fits_without_error():
    """Check the requested fairness statistic, independently of generated matrices."""
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

    clf = plqERM_Ridge(
        loss={"name": "svm"},
        C=C,
        constraint=[{"name": "fair", "sen_idx": sen_idx, "tol_sen": [0.01]}],
        tol=1e-8,
        max_iter=50000,
    )
    clf.fit(X=X, y=y)

    assert clf.coef_.shape == (d,), f"coef_ shape should be ({d},), got {clf.coef_.shape}"
    assert np.all(np.isfinite(clf.coef_)), "coefficients should be finite"
    assert clf.converged_
    assert np.min(A @ clf.coef_ + b) >= -clf.tol


def test_fairsvm_coef_shape():
    """Fitted coefficients should have the correct shape."""
    np.random.seed(42)
    n, d = 80, 4
    X, y = make_classification(n, d, random_state=42)
    y = 2 * y - 1

    clf = plqERM_Ridge(loss={"name": "svm"}, C=1.0)
    clf.fit(X=X, y=y)

    assert clf.coef_.shape == (d,), f"coef_ should have shape ({d},), got {clf.coef_.shape}"
