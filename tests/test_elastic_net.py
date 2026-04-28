"""
Test ElasticNet on simulated dataset.

Tests PR #7fd2ab1: add ElasticNet penalty support to ReHLine solver.
Dataset sizes are controlled to prevent slow CI runs.

Note on parameterisation mismatch
-----------------------------------
rehline objective:
    min_beta  C * sum_i PLQ(y_i, x_i^T beta) + l1_ratio * ||beta||_1
              + 0.5*(1-l1_ratio)*||beta||_2^2

sklearn ElasticNet objective:
    min_beta  (1/2n) * sum_i (y_i - x_i^T beta)^2
              + alpha*l1_ratio*||beta||_1
              + (alpha/2)*(1-l1_ratio)*||beta||_2^2

The L2 penalty scales differently, so the two are NOT directly equivalent.
test_elasticnet_vs_sklearn_mse documents this known discrepancy and is
marked xfail.
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rehline import plqERM_ElasticNet, plqERM_Ridge

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
import pytest

def _regression_dataset(n, n_features, n_informative, seed=42):
    X, y = make_regression(
        n_samples=n,
        n_features=n_features,
        noise=0.1,
        random_state=seed,
        n_informative=n_informative,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_elasticnet_vs_sklearn_mse():
    """ElasticNet vs sklearn (MSE loss, no intercept) — expected to differ."""
    n, n_features = 5000, 20
    C, l1_ratio = 0.1, 0.5

    X_train, X_test, y_train, y_test = _regression_dataset(n, n_features, 10)

    clf_skl = ElasticNet(
        alpha=1 / (C * 2 * len(X_train)),
        l1_ratio=l1_ratio,
        max_iter=10000,
        tol=1e-5,
        fit_intercept=False,
    )
    clf_skl.fit(X_train, y_train)
    sol_skl = clf_skl.coef_.flatten()

    clf_reh = plqERM_ElasticNet(
        loss={"name": "mse"},
        C=C,
        l1_ratio=l1_ratio,
        max_iter=10000,
        tol=1e-5,
    )
    clf_reh.fit(X_train, y_train)
    sol_reh = np.where(np.abs(clf_reh.coef_.flatten()) < 1e-8, 0, clf_reh.coef_.flatten())

    max_diff = np.max(np.abs(sol_skl - sol_reh))
    assert max_diff <= 1e-4, (
        f"Solutions differ by {max_diff:.6e} > 1e-4. "
        f"Known parameterisation issue between rehline and sklearn ElasticNet."
    )


def test_different_l1_ratios():
    """ElasticNet should fit successfully for all tested l1_ratio values."""
    n, n_features = 3000, 15
    C = 0.01

    X_train, X_test, y_train, y_test = _regression_dataset(n, n_features, 8)

    # l1_ratio=1.0 causes division by zero in rho = l1_ratio/(1-l1_ratio)
    for l1_ratio in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        clf = plqERM_ElasticNet(
            loss={"name": "mse"},
            C=C,
            l1_ratio=l1_ratio,
            max_iter=5000,
            tol=1e-4,
        )
        clf.fit(X_train, y_train)
        assert clf.coef_ is not None, f"Fit failed for l1_ratio={l1_ratio}"
        assert clf.coef_.shape == (n_features,), f"Wrong coef_ shape for l1_ratio={l1_ratio}: {clf.coef_.shape}"


def test_different_losses():
    """ElasticNet should fit successfully for each supported loss function."""
    n, n_features = 3000, 12
    C, l1_ratio = 0.01, 0.5

    X_train, X_test, y_train, _ = _regression_dataset(n, n_features, 8)

    losses = [
        {"name": "mse"},
        {"name": "mae"},
        {"name": "huber", "tau": 0.1},
        {"name": "SVR", "epsilon": 0.1},
    ]
    for loss in losses:
        clf = plqERM_ElasticNet(
            loss=loss,
            C=C,
            l1_ratio=l1_ratio,
            max_iter=10000,
            tol=1e-4,
        )
        clf.fit(X_train, y_train)
        assert clf.coef_ is not None, f"Fit failed for loss={loss}"
        assert clf.coef_.shape == (n_features,), f"Wrong coef_ shape for loss={loss}: {clf.coef_.shape}"


def test_elasticnet_vs_ridge():
    """ElasticNet with l1_ratio=0 (pure L2) should match Ridge within 1e-4."""
    n, n_features = 3000, 12
    C = 0.1

    X_train, X_test, y_train, _ = _regression_dataset(n, n_features, 8)

    clf_en = plqERM_ElasticNet(
        loss={"name": "mse"},
        C=C,
        l1_ratio=0.0,
        max_iter=5000,
        tol=1e-4,
    )
    clf_en.fit(X_train, y_train)

    clf_ridge = plqERM_Ridge(
        loss={"name": "mse"},
        C=C,
        max_iter=5000,
        tol=1e-4,
    )
    clf_ridge.fit(X_train, y_train)

    max_diff = np.max(np.abs(clf_en.coef_.flatten() - clf_ridge.coef_.flatten()))
    assert max_diff < 1e-4, f"ElasticNet(l1_ratio=0) should match Ridge within 1e-4, max_diff={max_diff:.6e}"


def test_sparsity_increases_with_l1():
    """Higher l1_ratio should produce equal or more sparse solutions."""
    n, n_features, C = 3000, 20, 0.001

    X, y = make_regression(
        n_samples=n,
        n_features=n_features,
        noise=0.1,
        random_state=42,
        n_informative=8,
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    zeros = []
    for l1_ratio in [0.0, 0.3, 0.6, 0.9]:
        clf = plqERM_ElasticNet(
            loss={"name": "mse"},
            C=C,
            l1_ratio=l1_ratio,
            max_iter=5000,
            tol=1e-4,
        )
        clf.fit(X_scaled, y)
        zeros.append(np.sum(np.abs(clf.coef_.flatten()) < 1e-8))

    assert zeros[-1] >= zeros[0], f"Sparsity should be at least as high at l1_ratio=0.9 as at l1_ratio=0.0, got {zeros}"


def test_dual_variable_mu():
    """After fitting, ElasticNet's dual variable mu should be in [0, rho]."""
    n, n_features, C, l1_ratio = 2000, 10, 0.01, 0.5

    X, y = make_regression(
        n_samples=n,
        n_features=n_features,
        noise=0.1,
        random_state=42,
        n_informative=6,
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = plqERM_ElasticNet(
        loss={"name": "mse"},
        C=C,
        l1_ratio=l1_ratio,
        max_iter=5000,
        tol=1e-4,
    )
    clf.fit(X_scaled, y)

    assert hasattr(clf, "_mu"), "clf should have _mu attribute after fitting"
    rho = clf.rho
    assert np.all(clf._mu >= 0), "mu should be non-negative"
    assert np.all(clf._mu <= rho + 1e-10), f"mu should be <= rho={rho}"


def test_different_omegas():
    """ElasticNet should fit successfully for all tested omega values."""
    n, n_features, C, l1_ratio = 2000, 10, 0.01, 0.5

    X, y = make_regression(
        n_samples=n,
        n_features=n_features,
        noise=0.1,
        random_state=42,
        n_informative=6,
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for i in range(5):  # conduct 5 tests with different omega
        rng = np.random.default_rng(seed=42+i)
        omega = rng.uniform(low=0.1, high=0.2+i, size=n_features)
        clf = plqERM_ElasticNet(
            loss={"name": "mse"},
            C=C,
            l1_ratio=l1_ratio,
            omega=omega,
            max_iter=5000,
            tol=1e-4,
        )
        clf.fit(X_scaled, y)
        assert clf.coef_ is not None, f"Fit failed for omega={omega}"
        assert clf.coef_.shape == (n_features,), f"Wrong coef_ shape for omega={omega}: {clf.coef_.shape}"


def test_with_omega_vs_without_omega():
    """ElasticNet with omega=(1, 1, ..., 1) should exactly match that without omega."""
    n, n_features, C, l1_ratio = 2000, 10, 0.01, 0.5

    X, y = make_regression(
        n_samples=n,
        n_features=n_features,
        noise=0.1,
        random_state=42,
        n_informative=6,
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf_with_omg = plqERM_ElasticNet(
                 loss={"name": "mse"},
                 C=C,
                 l1_ratio=l1_ratio,
                 omega=np.ones(n_features),
                 max_iter=5000,
                 tol=1e-4,
    )
    clf_with_omg.fit(X_scaled, y)

    clf_without_omg = plqERM_ElasticNet(
                    loss={"name": "mse"},
                    C=C,
                    l1_ratio=l1_ratio,
                    max_iter=5000,
                    tol=1e-4,
    )
    clf_without_omg.fit(X_scaled, y)

    assert np.array_equal(clf_with_omg.coef_.flatten(), clf_without_omg.coef_.flatten()), \
        "ElasticNet with omega=(1, 1, ..., 1) should exactly match that without omega."


def test_omega_validation():
    """Test omega related validations raise appropriate warnings or errors"""
    n, n_features, C, l1_ratio = 2000, 10, 0.01, 0.5

    X, y = make_regression(
        n_samples=n,
        n_features=n_features,
        noise=0.1,
        random_state=42,
        n_informative=6,
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Test invalid omega shape (must align with n_features)
    with pytest.raises(ValueError, match="Omega length"):
        clf = plqERM_ElasticNet(
            loss={"name": "mse"},
            C=C,
            l1_ratio=l1_ratio,
            omega=np.ones(n_features + 1),
            max_iter=5000,
            tol=1e-4,
        )
        clf.fit(X_scaled, y)
    # Test invalid omega value (all elements must be strictly positive)
    with pytest.raises(ValueError, match="All elements in omega must be strictly positive"):
        omega = np.ones(n_features)
        omega[0] = -1
        clf = plqERM_ElasticNet(
            loss={"name": "mse"},
            C=C,
            l1_ratio=l1_ratio,
            omega=omega,
            max_iter=5000,
            tol=1e-4,
        )
        clf.fit(X_scaled, y)
    # Test ineffective omega (when omega provided but l1_ratio==0)
    with pytest.warns(UserWarning, match="Omega will be ignored since l1_ratio=0"):
        clf = plqERM_ElasticNet(
            loss={"name": "mse"},
            C=C,
            l1_ratio=0.0,
            omega=np.ones(n_features),
            max_iter=5000,
            tol=1e-4,
        )
        clf.fit(X_scaled, y)
