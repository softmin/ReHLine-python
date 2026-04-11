"""Test SVR on simulated dataset — asserts ReHLine matches sklearn LinearSVR."""

import numpy as np
from sklearn.svm import LinearSVR

from rehline import ReHLine, plqERM_Ridge


def _make_regression_data(n=1000, d=5, seed=1024):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    beta0 = rng.randn(d)
    y = X.dot(beta0) + rng.randn(n)
    return X, y


def test_plqERM_Ridge_svr_matches_sklearn():
    """plqERM_Ridge with SVR loss should produce coefficients close to sklearn LinearSVR."""
    X, y = _make_regression_data()
    C = 0.5
    epsilon = 1e-5

    reg_skl = LinearSVR(
        C=C,
        loss="epsilon_insensitive",
        fit_intercept=False,
        epsilon=epsilon,
        random_state=0,
        tol=1e-6,
        max_iter=1_000_000,
        dual="auto",
    )
    reg_skl.fit(X, y)
    coef_skl = reg_skl.coef_.flatten()

    reg_reh = plqERM_Ridge(loss={"name": "svr", "epsilon": epsilon}, C=C, tol=1e-4, max_iter=100000)
    reg_reh.fit(X=X, y=y)
    coef_reh = reg_reh.coef_.flatten()

    np.testing.assert_allclose(
        coef_reh,
        coef_skl,
        atol=1e-2,
        err_msg="plqERM_Ridge SVR should match sklearn LinearSVR within 1e-2",
    )


def test_ReHLine_manual_svr_params_match_builtin():
    """Manually constructed SVR parameters should produce the same coef as built-in loss."""
    X, y = _make_regression_data()
    C = 0.5
    epsilon = 1e-5
    n = X.shape[0]

    # Built-in loss
    reg_builtin = plqERM_Ridge(loss={"name": "svr", "epsilon": epsilon}, C=C, tol=1e-6, max_iter=1_000_000)
    reg_builtin.fit(X=X, y=y)
    coef_builtin = reg_builtin.coef_.flatten()

    # Manual parameterisation
    U = np.ones((2, n)) * C
    V = np.ones((2, n))
    U[1] = -U[1]
    V[0] = -C * (y + epsilon)
    V[1] = C * (y - epsilon)

    # When U/V are pre-scaled by C, use C=1.0 to avoid double-counting
    reg_manual = ReHLine(C=1.0, tol=1e-6, max_iter=1_000_000)
    reg_manual._U, reg_manual._V = U, V
    reg_manual.fit(X=X)
    coef_manual = reg_manual.coef_.flatten()

    np.testing.assert_allclose(
        coef_manual,
        coef_builtin,
        atol=1e-6,
        err_msg="Manual and built-in SVR params should give identical coefficients",
    )


def test_svr_decision_function_shape():
    """decision_function should return a 1-D array of length n_query."""
    X, y = _make_regression_data(n=100, d=5)
    reg = plqERM_Ridge(loss={"name": "svr", "epsilon": 0.1}, C=0.5)
    reg.fit(X=X, y=y)
    preds = reg.decision_function(X[:5])
    assert preds.shape == (5,), f"Expected shape (5,), got {preds.shape}"
