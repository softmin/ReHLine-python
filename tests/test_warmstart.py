"""
Test warm-start functionality for ReHLine_solver, ReHLine, and plqERM_Ridge.

Explicit iteration budgets allow coordinate descent to meet the original tolerances.

Warm-start should:
  1. Attain the same independently evaluated objective as cold-start.
  2. Converge in fewer iterations than cold-start when starting from a nearby solution.
"""

import numpy as np

from rehline import ReHLine, plqERM_ElasticNet, plqERM_Ridge
from rehline._base import ReHLine_solver


def _make_classification_data(n=1000, d=3, seed=1024):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    beta0 = rng.randn(d)
    y = np.sign(X.dot(beta0) + rng.randn(n))
    return X, y


def svm_objective(beta, X, y, C, l1_ratio=0):
    return C * np.maximum(1 - y * (X @ beta), 0).sum() + (1 - l1_ratio) * (beta @ beta) / 2 + l1_ratio * abs(beta).sum()


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

    res_cold = ReHLine_solver(X, U, V, max_iter=100_000, tol=1e-8)
    res_warm = ReHLine_solver(X, U, V, max_iter=100_000, tol=1e-8, Lambda=res_cold.Lambda)

    assert res_cold.converged and res_warm.converged
    assert res_warm.Lambda.shape == res_cold.Lambda.shape, (
        f"Warm-start Lambda shape {res_warm.Lambda.shape} should match cold-start {res_cold.Lambda.shape}"
    )


def test_solver_warmstart_consistent_solution(assert_objective_close):
    """Warm-start from the converged solution should give the same result."""
    X, y = _make_classification_data()
    C = 0.5
    n = X.shape[0]

    U = -(C * y).reshape(1, -1)
    V = (C * np.ones(n)).reshape(1, -1)

    res_cold = ReHLine_solver(X, U, V, max_iter=100_000, tol=1e-8)
    # A warm refit should reach the same solution with fewer coordinate sweeps.
    res_warm = ReHLine_solver(X, U, V, max_iter=100_000, tol=1e-8, Lambda=res_cold.Lambda)

    assert res_cold.converged and res_warm.converged
    assert res_warm.niter < res_cold.niter
    values = []
    for result in (res_cold, res_warm):
        reconstructed = -X.T @ (U * result.Lambda).sum(axis=0)
        np.testing.assert_allclose(result.beta, reconstructed, atol=1e-12)
        value = svm_objective(result.beta, X, y, C)
        assert_objective_close(value, result.objective)
        assert_objective_close(value, result.dual_objective)
        values.append(value)
    assert_objective_close(*values)


# ---------------------------------------------------------------------------
# ReHLine estimator
# ---------------------------------------------------------------------------


def test_ReHLine_warmstart_objective_consistent(assert_objective_close):
    """Warm-start ReHLine should attain the cold-start objective."""
    X, y = _make_classification_data()
    C = 0.5
    n = X.shape[0]

    U = -(y.reshape(1, -1))
    V = np.ones(n).reshape(1, -1)

    clf_cold = ReHLine(max_iter=100_000, tol=1e-8, verbose=0)
    clf_cold.C = C
    clf_cold._U, clf_cold._V = U, V
    clf_cold.fit(X)

    # Warm-start with increased C — result will differ, but should converge
    clf_warm = ReHLine(max_iter=100_000, tol=1e-8, verbose=0)
    clf_warm.C = C
    clf_warm._U, clf_warm._V = U, V
    clf_warm.fit(X)
    clf_warm.C = 2 * C
    clf_warm.warm_start = 1
    clf_warm._U, clf_warm._V = U, V  # re-set after fit resets internals
    clf_warm.fit(X)

    # Re-run cold-start with 2*C to get the reference
    clf_ref = ReHLine(max_iter=100_000, tol=1e-8, verbose=0)
    clf_ref.C = 2 * C
    clf_ref._U, clf_ref._V = U, V
    clf_ref.fit(X)

    values = [svm_objective(m.coef_, X, y, 2 * C) for m in (clf_warm, clf_ref)]
    assert_objective_close(*values)
    for model, value in zip((clf_warm, clf_ref), values):
        assert model.converged_
        assert_objective_close(value, model.objective_)
        assert_objective_close(value, model.dual_objective_)


# ---------------------------------------------------------------------------
# plqERM_Ridge
# ---------------------------------------------------------------------------


def test_plqERM_Ridge_warmstart_objective_consistent(assert_objective_close):
    """Warm-started plqERM_Ridge should match cold-start solution for the same C."""
    X, y = _make_classification_data()
    C = 0.5

    clf_cold = plqERM_Ridge(max_iter=100_000, tol=1e-8, loss={"name": "svm"}, C=C, verbose=0)
    clf_cold.fit(X=X, y=y)

    # Fit at C, then warm-start at 2*C
    clf_warm = plqERM_Ridge(max_iter=100_000, tol=1e-8, loss={"name": "svm"}, C=C, verbose=0)
    clf_warm.fit(X=X, y=y)
    clf_warm.C = 2 * C
    clf_warm.warm_start = 1
    clf_warm.fit(X=X, y=y)
    coef_warm_2C = clf_warm.coef_.copy()

    # Reference: cold-start at 2*C
    clf_ref = plqERM_Ridge(max_iter=100_000, tol=1e-8, loss={"name": "svm"}, C=2 * C, verbose=0)
    clf_ref.fit(X=X, y=y)
    coef_ref_2C = clf_ref.coef_.copy()

    values = [svm_objective(beta, X, y, 2 * C) for beta in (coef_warm_2C, coef_ref_2C)]
    assert_objective_close(*values)
    for model, value in zip((clf_warm, clf_ref), values):
        assert model.converged_
        assert_objective_close(value, model.objective_)
        assert_objective_close(value, model.dual_objective_)


# ---------------------------------------------------------------------------
# plqERM_ElasticNet
# ---------------------------------------------------------------------------


def test_plqERM_ElasticNet_warmstart_objective_consistent(assert_objective_close):
    """Warm-started plqERM_ElasticNet should match cold-start solution for the same C."""
    X, y = _make_classification_data()
    C = 0.5
    l1_ratio = 0.2

    clf_cold = plqERM_ElasticNet(max_iter=100_000, tol=1e-8, loss={"name": "svm"}, C=C, l1_ratio=l1_ratio, verbose=0)
    clf_cold.fit(X=X, y=y)

    # Fit at C, then warm-start at 2*C
    clf_warm = plqERM_ElasticNet(max_iter=100_000, tol=1e-8, loss={"name": "svm"}, C=C, l1_ratio=l1_ratio, verbose=0)
    clf_warm.fit(X=X, y=y)
    clf_warm.C = 2 * C
    clf_warm.warm_start = 1
    clf_warm.fit(X=X, y=y)
    coef_warm_2C = clf_warm.coef_.copy()

    # Reference: cold-start at 2*C
    clf_ref = plqERM_ElasticNet(max_iter=100_000, tol=1e-8, loss={"name": "svm"}, C=2 * C, l1_ratio=l1_ratio, verbose=0)
    clf_ref.fit(X=X, y=y)
    coef_ref_2C = clf_ref.coef_.copy()

    values = [svm_objective(beta, X, y, 2 * C, l1_ratio) for beta in (coef_warm_2C, coef_ref_2C)]
    assert_objective_close(*values)
    for model, value in zip((clf_warm, clf_ref), values):
        assert model.converged_
        assert_objective_close(value, model.objective_ * (1 - l1_ratio))
        assert_objective_close(value, model.dual_objective_ * (1 - l1_ratio))
