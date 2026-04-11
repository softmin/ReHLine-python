"""
Test multi-class classification against sklearn baselines.

Compares rehline's plq_Ridge_Classifier (binary, OvR, OvO) with sklearn's
LinearSVC. Solutions must match within tol=1e-3.
Dataset sizes are controlled to ~5000 samples to keep CI runtimes reasonable.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from rehline import plq_Ridge_Classifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scaled_split(X, y, test_size=0.3, seed=42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    return X_tr, X_te, y_tr, y_te


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_binary_vs_sklearn():
    """Binary classification coef should match sklearn LinearSVC within 1e-3."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=3000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        class_sep=1.5,
        random_state=42,
    )
    X_tr, X_te, y_tr, y_te = _scaled_split(X, y)
    C = 1.0

    clf_skl = LinearSVC(
        C=C,
        loss="hinge",
        fit_intercept=True,
        max_iter=1_000_000,
        tol=1e-5,
        random_state=42,
    )
    clf_skl.fit(X_tr, y_tr)
    coef_skl = clf_skl.coef_.flatten()

    clf_reh = plq_Ridge_Classifier(
        loss={"name": "svm"},
        C=C,
        max_iter=1_000_000,
        tol=1e-5,
        verbose=0,
    )
    clf_reh.fit(X_tr, y_tr)
    coef_reh = clf_reh.coef_.flatten()

    max_diff = np.max(np.abs(coef_skl - coef_reh))
    assert max_diff <= 1e-3, f"Binary coef_ max difference {max_diff:.6e} > 1e-3 vs sklearn LinearSVC"


def test_multiclass_ovr_vs_sklearn():
    """OvR multiclass coef should match sklearn LinearSVC(OvR) within 1e-3."""
    np.random.seed(42)
    n_classes, n_features = 4, 10
    X, y = make_classification(
        n_samples=5000,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        n_classes=n_classes,
        class_sep=1.5,
        random_state=42,
    )
    X_tr, X_te, y_tr, y_te = _scaled_split(X, y)
    C = 1.0

    clf_skl = LinearSVC(
        C=C,
        loss="hinge",
        multi_class="ovr",
        fit_intercept=True,
        max_iter=1_000_000,
        tol=1e-5,
        random_state=42,
    )
    clf_skl.fit(X_tr, y_tr)
    coef_skl = clf_skl.coef_

    clf_reh = plq_Ridge_Classifier(
        loss={"name": "svm"},
        C=C,
        multi_class="ovr",
        max_iter=1_000_000,
        tol=1e-5,
        verbose=0,
    )
    clf_reh.fit(X_tr, y_tr)
    coef_reh = clf_reh.coef_

    assert coef_reh.shape == (n_classes, n_features), (
        f"OvR coef_ shape should be ({n_classes}, {n_features}), got {coef_reh.shape}"
    )

    max_diff = np.max(np.abs(coef_skl - coef_reh))
    assert max_diff <= 1e-3, f"OvR coef_ max difference {max_diff:.6e} > 1e-3 vs sklearn LinearSVC(OvR)"


def test_multiclass_ovo_vs_sklearn():
    """OvO multiclass coef should match sklearn OneVsOneClassifier(LinearSVC) within 1e-3."""
    np.random.seed(42)
    n_classes, n_features = 3, 8
    X, y = make_classification(
        n_samples=3000,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        n_classes=n_classes,
        class_sep=1.5,
        random_state=42,
    )
    X_tr, X_te, y_tr, y_te = _scaled_split(X, y)
    C = 1.0
    n_estimators = n_classes * (n_classes - 1) // 2

    base = LinearSVC(
        C=C,
        loss="hinge",
        fit_intercept=True,
        max_iter=1_000_000,
        tol=1e-5,
        random_state=42,
    )
    clf_skl = OneVsOneClassifier(base)
    clf_skl.fit(X_tr, y_tr)
    coef_skl = np.array([e.coef_.flatten() for e in clf_skl.estimators_])

    clf_reh = plq_Ridge_Classifier(
        loss={"name": "svm"},
        C=C,
        multi_class="ovo",
        max_iter=1_000_000,
        tol=1e-5,
        verbose=0,
    )
    clf_reh.fit(X_tr, y_tr)
    coef_reh = clf_reh.coef_

    assert coef_reh.shape == (n_estimators, n_features), (
        f"OvO coef_ shape should be ({n_estimators}, {n_features}), got {coef_reh.shape}"
    )

    # Bug fix (PR #34): sign convention was corrected so coef_ now matches sklearn exactly.
    max_diff = np.max(np.abs(coef_skl - coef_reh))
    assert max_diff <= 1e-3, f"OvO coef_ max difference {max_diff:.6e} > 1e-3 vs sklearn OvO"


def test_decision_function_shapes():
    """decision_function should return arrays with correct shapes for all strategies."""
    np.random.seed(42)
    n_samples, n_features = 1000, 5
    X = np.random.randn(n_samples, n_features)

    # Binary
    y_bin = np.random.randint(0, 2, n_samples)
    clf = plq_Ridge_Classifier(loss={"name": "svm"}, C=1.0, tol=1e-5, max_iter=1_000_000)
    clf.fit(X, y_bin)
    assert clf.decision_function(X).shape == (n_samples,), "Binary decision_function should have shape (n_samples,)"

    # OvR (4 classes)
    y_multi = np.random.randint(0, 4, n_samples)
    clf_ovr = plq_Ridge_Classifier(
        loss={"name": "svm"},
        C=1.0,
        multi_class="ovr",
        tol=1e-5,
    )
    clf_ovr.fit(X, y_multi)
    assert clf_ovr.decision_function(X).shape == (n_samples, 4), (
        "OvR decision_function should have shape (n_samples, 4)"
    )

    # OvO (4 classes → 6 estimators)
    clf_ovo = plq_Ridge_Classifier(
        loss={"name": "svm"},
        C=1.0,
        multi_class="ovo",
        tol=1e-5,
        max_iter=1_000_000,
    )
    clf_ovo.fit(X, y_multi)
    assert clf_ovo.decision_function(X).shape == (n_samples, 6), (
        "OvO decision_function should have shape (n_samples, 6)"
    )


def test_ovo_coef_sign_convention():
    """
    Regression test for the OvO sign-convention bug (PR #34).

    The previous bug assigned cls_i -> +1 and cls_j -> -1, opposite to sklearn's
    convention (cls_j -> +1, cls_i -> -1 for sorted pairs). This caused every
    subproblem's coef_ to be negated. We verify the dot product between sklearn
    and rehline coef_ is positive for every subproblem.
    """
    np.random.seed(0)
    n_samples, n_features, n_classes, C = 2000, 6, 3, 1.0

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_redundant=1,
        n_classes=n_classes,
        class_sep=2.0,
        random_state=0,
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    base_clf = LinearSVC(
        C=C,
        loss="hinge",
        fit_intercept=True,
        max_iter=1_000_000,
        tol=1e-5,
        random_state=0,
    )
    clf_skl = OneVsOneClassifier(base_clf)
    clf_skl.fit(X, y)

    clf_reh = plq_Ridge_Classifier(
        loss={"name": "svm"},
        C=C,
        multi_class="ovo",
        max_iter=1_000_000,
        tol=1e-5,
        verbose=0,
    )
    clf_reh.fit(X, y)

    for k, est in enumerate(clf_skl.estimators_):
        dot = np.dot(est.coef_.flatten(), clf_reh.coef_[k])
        assert dot > 0, f"OvO subproblem {k}: dot product {dot:.4f} <= 0, sign-convention bug has reappeared."


def test_ovo_predict_consistency():
    """
    OvO predict() and decision_function() must be consistent: manually
    reconstructing predictions from decision_function() must match predict().
    """
    np.random.seed(7)
    n_samples, n_features, n_classes, C = 1500, 5, 4, 1.0

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_redundant=0,
        n_classes=n_classes,
        class_sep=1.5,
        random_state=7,
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    clf = plq_Ridge_Classifier(
        loss={"name": "svm"},
        C=C,
        multi_class="ovo",
        max_iter=1_000_000,
        tol=1e-5,
        verbose=0,
    )
    clf.fit(X, y)

    y_pred = clf.predict(X)
    scores = clf.decision_function(X)
    n_cls = len(clf.classes_)
    votes = np.zeros((n_samples, n_cls))
    confidences = np.zeros((n_samples, n_cls))
    for k, (_, _, cls_i, cls_j) in enumerate(clf.estimators_):
        i = np.where(clf.classes_ == cls_i)[0][0]
        j = np.where(clf.classes_ == cls_j)[0][0]
        pred = (scores[:, k] > 0).astype(int)
        votes[:, j] += pred
        votes[:, i] += 1 - pred
        confidences[:, j] += scores[:, k]
        confidences[:, i] -= scores[:, k]
    transformed = confidences / (3 * (np.abs(confidences) + 1))
    y_manual = clf.classes_[np.argmax(votes + transformed, axis=1)]

    n_disagree = np.sum(y_pred != y_manual)
    assert n_disagree == 0, f"predict() and decision_function() are inconsistent: {n_disagree} samples disagree."


def test_ovo_fit_intercept_false():
    """OvO with fit_intercept=False should match sklearn and produce zero intercepts."""
    np.random.seed(13)
    n_samples, n_features, n_classes, C = 2000, 6, 3, 1.0
    n_estimators = n_classes * (n_classes - 1) // 2

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_redundant=1,
        n_classes=n_classes,
        class_sep=2.0,
        random_state=13,
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    base_clf = LinearSVC(
        C=C,
        loss="hinge",
        fit_intercept=False,
        max_iter=1_000_000,
        tol=1e-5,
        random_state=13,
    )
    clf_skl = OneVsOneClassifier(base_clf)
    clf_skl.fit(X, y)

    clf_reh = plq_Ridge_Classifier(
        loss={"name": "svm"},
        C=C,
        multi_class="ovo",
        fit_intercept=False,
        max_iter=1_000_000,
        tol=1e-5,
        verbose=0,
    )
    clf_reh.fit(X, y)

    assert clf_reh.coef_.shape == (n_estimators, n_features)
    assert np.all(clf_reh.intercept_ == 0.0)

    max_diff = max(np.max(np.abs(est.coef_.flatten() - clf_reh.coef_[k])) for k, est in enumerate(clf_skl.estimators_))
    assert max_diff <= 1e-3, f"fit_intercept=False OvO coef_ diff {max_diff:.6e} > 1e-3"


def test_multiclass_invalid_multi_class():
    """Passing an unrecognised multi_class value should raise ValueError."""
    np.random.seed(42)
    X = np.random.randn(200, 4)
    y = np.random.randint(0, 3, 200)

    clf = plq_Ridge_Classifier(
        loss={"name": "svm"},
        C=1.0,
        multi_class="invalid_option",
    )
    try:
        clf.fit(X, y)
        raised = False
    except ValueError:
        raised = True

    assert raised, "Expected ValueError for invalid multi_class parameter."


def test_ovo_more_classes():
    """OvO with 5 classes (10 subproblems) — shape and coef correctness."""
    np.random.seed(99)
    n_samples, n_features, n_classes, C = 3000, 8, 5, 1.0
    n_estimators = n_classes * (n_classes - 1) // 2  # 10

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=1,
        n_classes=n_classes,
        class_sep=1.5,
        random_state=99,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=99,
        stratify=y,
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    base_clf = LinearSVC(
        C=C,
        loss="hinge",
        fit_intercept=True,
        max_iter=1_000_000,
        tol=1e-5,
        random_state=99,
    )
    clf_skl = OneVsOneClassifier(base_clf)
    clf_skl.fit(X_tr, y_tr)

    clf_reh = plq_Ridge_Classifier(
        loss={"name": "svm"},
        C=C,
        multi_class="ovo",
        max_iter=1_000_000,
        tol=1e-5,
        verbose=0,
    )
    clf_reh.fit(X_tr, y_tr)

    assert clf_reh.coef_.shape == (n_estimators, n_features)
    assert clf_reh.intercept_.shape == (n_estimators,)
    assert len(clf_reh.estimators_) == n_estimators

    max_diff = max(np.max(np.abs(est.coef_.flatten() - clf_reh.coef_[k])) for k, est in enumerate(clf_skl.estimators_))
    assert max_diff <= 1e-3, f"5-class OvO coef_ diff {max_diff:.6e} > 1e-3"
