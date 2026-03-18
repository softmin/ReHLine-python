"""Test SVM on simulated dataset — asserts ReHLine matches sklearn LinearSVC."""

import numpy as np
from sklearn.svm import LinearSVC

from rehline import ReHLine, plqERM_Ridge


def _make_classification_data(n=1000, d=3, seed=1024):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    beta0 = rng.randn(d)
    y = np.sign(X.dot(beta0) + rng.randn(n))
    return X, y


def test_plqERM_Ridge_svm_matches_sklearn():
    """plqERM_Ridge with hinge loss should produce coefficients close to sklearn LinearSVC."""
    X, y = _make_classification_data()
    C = 0.5

    clf_skl = LinearSVC(
        C=C,
        loss="hinge",
        fit_intercept=False,
        random_state=0,
        tol=1e-6,
        max_iter=1_000_000,
    )
    clf_skl.fit(X, y)
    coef_skl = clf_skl.coef_.flatten()

    clf_reh = plqERM_Ridge(loss={"name": "svm"}, C=C)
    clf_reh.fit(X=X, y=y)
    coef_reh = clf_reh.coef_.flatten()

    np.testing.assert_allclose(
        coef_reh,
        coef_skl,
        atol=1e-2,
        err_msg="plqERM_Ridge SVM coef should match sklearn LinearSVC within 1e-2",
    )


def test_ReHLine_manual_svm_params_matches_sklearn():
    """ReHLine with manually constructed SVM parameters should match sklearn LinearSVC."""
    X, y = _make_classification_data()
    C = 0.5
    n = X.shape[0]

    clf_skl = LinearSVC(
        C=C,
        loss="hinge",
        fit_intercept=False,
        random_state=0,
        tol=1e-6,
        max_iter=1_000_000,
    )
    clf_skl.fit(X, y)
    coef_skl = clf_skl.coef_.flatten()

    # Manual SVM parameterisation for ReHLine
    U = -(C * y).reshape(1, -1)
    V = (C * np.ones(n)).reshape(1, -1)

    # When U/V are pre-scaled by C, ReHLine must use C=1.0 to avoid double-counting
    clf_reh = ReHLine(C=1.0)
    clf_reh._U, clf_reh._V = U, V
    clf_reh.fit(X=X)
    coef_reh = clf_reh.coef_.flatten()

    np.testing.assert_allclose(
        coef_reh,
        coef_skl,
        atol=1e-2,
        err_msg="ReHLine (manual params) SVM coef should match sklearn LinearSVC within 1e-2",
    )


def test_plqERM_Ridge_and_ReHLine_agree():
    """plqERM_Ridge and manually-parameterised ReHLine should produce identical results."""
    X, y = _make_classification_data()
    C = 0.5
    n = X.shape[0]

    clf_builtin = plqERM_Ridge(loss={"name": "svm"}, C=C)
    clf_builtin.fit(X=X, y=y)

    U = -(C * y).reshape(1, -1)
    V = (C * np.ones(n)).reshape(1, -1)
    # When U/V are pre-scaled by C, use C=1.0 to avoid double-counting
    clf_manual = ReHLine(C=1.0)
    clf_manual._U, clf_manual._V = U, V
    clf_manual.fit(X=X)

    np.testing.assert_allclose(
        clf_manual.coef_.flatten(),
        clf_builtin.coef_.flatten(),
        atol=1e-6,
        err_msg="Built-in and manual SVM params should give identical ReHLine coefficients",
    )


def test_decision_function_shape():
    """decision_function should return a 1-D array of length n_samples."""
    X, y = _make_classification_data(n=100, d=3)
    clf = plqERM_Ridge(loss={"name": "svm"}, C=0.5)
    clf.fit(X=X, y=y)
    scores = clf.decision_function([[0.1, 0.2, 0.3]])
    assert scores.shape == (1,), f"Expected shape (1,), got {scores.shape}"
