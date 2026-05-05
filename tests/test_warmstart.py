"""
Test warm-start functionality for ReHLine_solver, ReHLine, and plqERM_Ridge.

Warm-start should:
  1. Produce coefficients consistent with cold-start (within tolerance).
  2. Converge in fewer iterations than cold-start when starting from a nearby solution.
"""

import numpy as np

from rehline import ReHLine, plqERM_Ridge, plqERM_ElasticNet
from rehline._base import ReHLine_solver


def _make_classification_data(n=1000, d=3, seed=1024):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    beta0 = rng.randn(d)
    y = np.sign(X.dot(beta0) + rng.randn(n))
    return X, y


# ---------------------------------------------------------------------------
# ReHLine_solver
# ---------------------------------------------------------------------------


def test_solver_warmstart_lambda_shape():
    """ReHLine_solver warm-start should return Lambda with the same shape as cold-start."""
    X, y = _make_classification_data()
    C = 0.5
    n = X.shape[0]

    U = -(C * y).reshape(1, -1)
    V = (C * np.ones(n)).reshape(1, -1)

    res_cold = ReHLine_solver(X, U, V)
    res_warm = ReHLine_solver(X, U, V, Lambda=res_cold.Lambda)

    assert res_warm.Lambda.shape == res_cold.Lambda.shape, (
        f"Warm-start Lambda shape {res_warm.Lambda.shape} should match cold-start {res_cold.Lambda.shape}"
    )


def test_solver_warmstart_consistent_solution():
    """Warm-start from the converged solution should give the same result."""
    X, y = _make_classification_data()
    C = 0.5
    n = X.shape[0]

    U = -(C * y).reshape(1, -1)
    V = (C * np.ones(n)).reshape(1, -1)

    res_cold = ReHLine_solver(X, U, V)
    # Warm-starting from the converged point should converge immediately (very few iters)
    res_warm = ReHLine_solver(X, U, V, Lambda=res_cold.Lambda)

    beta_cold = X.T @ (res_cold.Lambda[0] / n)
    beta_warm = X.T @ (res_warm.Lambda[0] / n)
    np.testing.assert_allclose(
        beta_warm,
        beta_cold,
        atol=1e-6,
        err_msg="Warm-start from converged solution should give identical beta",
    )


# ---------------------------------------------------------------------------
# ReHLine estimator
# ---------------------------------------------------------------------------


def test_ReHLine_warmstart_coef_consistent():
    """Warm-start ReHLine should produce coefficients consistent with cold-start."""
    X, y = _make_classification_data()
    C = 0.5
    n = X.shape[0]

    U = -(y.reshape(1, -1))
    V = np.ones(n).reshape(1, -1)

    clf_cold = ReHLine(verbose=0)
    clf_cold.C = C
    clf_cold._U, clf_cold._V = U, V
    clf_cold.fit(X)

    # Warm-start with increased C — result will differ, but should converge
    clf_warm = ReHLine(verbose=0)
    clf_warm.C = C
    clf_warm._U, clf_warm._V = U, V
    clf_warm.fit(X)
    clf_warm.C = 2 * C
    clf_warm.warm_start = 1
    clf_warm._U, clf_warm._V = U, V  # re-set after fit resets internals
    clf_warm.fit(X)

    # Re-run cold-start with 2*C to get the reference
    clf_ref = ReHLine(verbose=0)
    clf_ref.C = 2 * C
    clf_ref._U, clf_ref._V = U, V
    clf_ref.fit(X)

    np.testing.assert_allclose(
        clf_warm.coef_,
        clf_ref.coef_,
        atol=1e-3,
        err_msg="Warm-start and cold-start should reach the same solution for same C",
    )


# ---------------------------------------------------------------------------
# plqERM_Ridge
# ---------------------------------------------------------------------------


def test_plqERM_Ridge_warmstart_coef_consistent():
    """Warm-started plqERM_Ridge should match cold-start solution for the same C."""
    X, y = _make_classification_data()
    C = 0.5

    clf_cold = plqERM_Ridge(loss={"name": "svm"}, C=C, verbose=0)
    clf_cold.fit(X=X, y=y)

    # Fit at C, then warm-start at 2*C
    clf_warm = plqERM_Ridge(loss={"name": "svm"}, C=C, verbose=0)
    clf_warm.fit(X=X, y=y)
    clf_warm.C = 2 * C
    clf_warm.warm_start = 1
    clf_warm.fit(X=X, y=y)
    coef_warm_2C = clf_warm.coef_.copy()

    # Reference: cold-start at 2*C
    clf_ref = plqERM_Ridge(loss={"name": "svm"}, C=2 * C, verbose=0)
    clf_ref.fit(X=X, y=y)
    coef_ref_2C = clf_ref.coef_.copy()

    np.testing.assert_allclose(
        coef_warm_2C,
        coef_ref_2C,
        atol=1e-3,
        err_msg="plqERM_Ridge: warm-start and cold-start should agree at the same C",
    )


# ---------------------------------------------------------------------------
# plqERM_ElasticNet
# ---------------------------------------------------------------------------


def test_plqERM_ElasticNet_warmstart_coef_consistent():
    """Warm-started plqERM_ElasticNet should match cold-start solution for the same C."""
    X, y = _make_classification_data()
    C = 0.5
    l1_ratio = 0.2

    clf_cold = plqERM_ElasticNet(loss={"name": "svm"}, C=C, l1_ratio=l1_ratio, verbose=0)
    clf_cold.fit(X=X, y=y)

    # Fit at C, then warm-start at 2*C
    clf_warm = plqERM_ElasticNet(loss={"name": "svm"}, C=C, l1_ratio=l1_ratio, verbose=0)
    clf_warm.fit(X=X, y=y)
    clf_warm.C = 2 * C
    clf_warm.warm_start = 1
    clf_warm.fit(X=X, y=y)
    coef_warm_2C = clf_warm.coef_.copy()

    # Reference: cold-start at 2*C
    clf_ref = plqERM_ElasticNet(loss={"name": "svm"}, C=2 * C, l1_ratio=l1_ratio, verbose=0)
    clf_ref.fit(X=X, y=y)
    coef_ref_2C = clf_ref.coef_.copy()

    np.testing.assert_allclose(
        coef_warm_2C,
        coef_ref_2C,
        atol=1e-3,
        err_msg="plqERM_ElasticNet: warm-start and cold-start should agree at the same C",
    )
