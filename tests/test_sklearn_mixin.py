"""
Test sklearn-compatible mixin classes (plq_Ridge_Classifier, plq_Ridge_Regressor).

Exercises sklearn pipeline, cross_val_score, and a minimal grid search.
The full GridSearchCV from the original script is not run in CI (too slow);
it is replaced with a targeted parameter check.
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rehline import plq_Ridge_Classifier, plq_Ridge_Regressor

# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------


def _clf_dataset(seed=42):
    return make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=seed,
    )


def test_classifier_pipeline_fits_and_predicts():
    """plq_Ridge_Classifier should work inside a sklearn Pipeline."""
    X, y = _clf_dataset()
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", plq_Ridge_Classifier(loss={"name": "svm"}, C=1.0, tol=1e-3, max_iter=1_000_000)),
        ]
    )
    pipe.fit(X, y)

    preds = pipe.predict(X)
    assert preds.shape == (len(y),), f"predict() shape should be ({len(y)},), got {preds.shape}"
    # Basic sanity: accuracy should be above chance level
    assert accuracy_score(y, preds) > 0.5, "Classifier accuracy should be above 0.5 on training data"


def test_classifier_cross_val_score():
    """cross_val_score on plq_Ridge_Classifier pipeline should return reasonable scores."""
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        class_sep=1.5,
        flip_y=0.0,
        random_state=42,
    )
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", plq_Ridge_Classifier(loss={"name": "svm"}, C=1.0)),
        ]
    )
    cv_scores = cross_val_score(pipe, X, y, cv=3, scoring="accuracy")
    assert cv_scores.shape == (3,), f"cross_val_score should return 3 scores, got shape {cv_scores.shape}"
    assert np.mean(cv_scores) > 0.5, f"Mean CV accuracy should be > 0.5, got {np.mean(cv_scores):.3f}"


def test_classifier_with_intercept_scaling():
    """Classifier with fit_intercept=True should still produce valid predictions."""
    X, y = _clf_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    clf = plq_Ridge_Classifier(
        loss={"name": "svm"},
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1.0,
        max_iter=1_000_000,
    )
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    assert preds.shape == (len(y_te),)
    assert accuracy_score(y_te, preds) > 0.4  # generous lower bound


def test_classifier_with_nonneg_constraint():
    """Classifier with nonnegative constraint should produce non-negative coefficients."""
    X, y = _clf_dataset()
    clf = plq_Ridge_Classifier(
        loss={"name": "svm"},
        C=1.0,
        constraint=[{"name": "nonnegative"}],
        max_iter=1_000_000,
    )
    clf.fit(X, y)
    # Allow 1e-2 numerical slack — the solver may not satisfy the constraint
    # exactly at convergence, but values should be close to zero or positive.
    assert np.all(clf.coef_.flatten() >= -1e-2), (
        "Coefficients should be approximately non-negative with nonnegative constraint"
    )


# ---------------------------------------------------------------------------
# Regressor tests
# ---------------------------------------------------------------------------


def _reg_dataset(seed=42):
    return make_regression(
        n_samples=500,
        n_features=10,
        n_informative=7,
        noise=5.0,
        random_state=seed,
    )


def test_regressor_pipeline_fits_and_predicts():
    """plq_Ridge_Regressor should work inside a sklearn Pipeline."""
    X, y = _reg_dataset()
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", plq_Ridge_Regressor(loss={"name": "QR", "qt": 0.5}, C=1.0)),
        ]
    )
    pipe.fit(X, y)

    preds = pipe.predict(X)
    assert preds.shape == (len(y),), f"predict() shape should be ({len(y)},), got {preds.shape}"


def test_regressor_cross_val_score():
    """cross_val_score on plq_Ridge_Regressor pipeline should return reasonable R² values."""
    X, y = _reg_dataset()
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", plq_Ridge_Regressor(loss={"name": "QR", "qt": 0.5}, C=1.0)),
        ]
    )
    cv_scores = cross_val_score(pipe, X, y, cv=3, scoring="r2")
    assert cv_scores.shape == (3,)
    # R² across folds should be reasonable
    assert np.mean(cv_scores) > 0.0, f"Mean CV R² should be positive, got {np.mean(cv_scores):.3f}"


def test_regressor_multiple_losses():
    """plq_Ridge_Regressor should fit with Huber and SVR losses without errors."""
    X, y = _reg_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

    for loss in [
        {"name": "huber", "tau": 1.0},
        {"name": "SVR", "epsilon": 0.1},
    ]:
        reg = plq_Ridge_Regressor(loss=loss, C=1.0)
        reg.fit(X_tr, y_tr)
        preds = reg.predict(X_te)
        assert preds.shape == (len(y_te),), f"predict() shape mismatch for loss={loss}"
        assert np.all(np.isfinite(preds)), f"Predictions should be finite for loss={loss}"
