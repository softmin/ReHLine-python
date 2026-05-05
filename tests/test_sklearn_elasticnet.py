"""
Tests for plq_ElasticNet_Classifier and plq_ElasticNet_Regressor.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rehline import plq_ElasticNet_Classifier, plq_ElasticNet_Regressor


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _binary_dataset(seed=42):
    return make_classification(
        n_samples=500, n_features=10, n_informative=5,
        n_redundant=2, n_classes=2, random_state=seed,
    )


def _multiclass_dataset(n_classes=3, seed=42):
    return make_classification(
        n_samples=600, n_features=10, n_informative=6,
        n_redundant=2, n_classes=n_classes,
        n_clusters_per_class=1, random_state=seed,
    )


def _reg_dataset(seed=42):
    return make_regression(
        n_samples=500, n_features=10, n_informative=7,
        noise=5.0, random_state=seed,
    )


# ===========================================================================
# plq_ElasticNet_Classifier — binary
# ===========================================================================

def test_elasticnet_clf_binary_pipeline_fits_and_predicts():
    X, y = _binary_dataset()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", plq_ElasticNet_Classifier(loss={"name": "svm"}, C=1.0, l1_ratio=0.5)),
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (len(y),)
    assert accuracy_score(y, preds) > 0.5


def test_elasticnet_clf_binary_cross_val_score():
    X, y = _binary_dataset()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", plq_ElasticNet_Classifier(loss={"name": "svm"}, C=1.0, l1_ratio=0.5)),
    ])
    scores = cross_val_score(pipe, X, y, cv=3, scoring="accuracy")
    assert scores.shape == (3,)
    assert np.mean(scores) > 0.5


def test_elasticnet_clf_binary_with_intercept():
    X, y = _binary_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = plq_ElasticNet_Classifier(
        loss={"name": "svm"}, C=1.0, l1_ratio=0.5,
        fit_intercept=True, intercept_scaling=1.0,
    )
    clf.fit(X_tr, y_tr)
    assert hasattr(clf, "intercept_")
    assert clf.coef_.shape == (X_tr.shape[1],)
    assert clf.predict(X_te).shape == (len(y_te),)


def test_elasticnet_clf_binary_without_intercept():
    X, y = _binary_dataset()
    clf = plq_ElasticNet_Classifier(
        loss={"name": "svm"}, C=1.0, l1_ratio=0.5, fit_intercept=False,
    )
    clf.fit(X, y)
    assert clf.intercept_ == 0.0
    assert clf.coef_.shape == (X.shape[1],)


def test_elasticnet_clf_l1_ratio_zero():
    """l1_ratio=0 is pure Ridge — should fit without error."""
    X, y = _binary_dataset()
    clf = plq_ElasticNet_Classifier(loss={"name": "svm"}, C=1.0, l1_ratio=0.0)
    clf.fit(X, y)
    assert clf.predict(X).shape == (len(y),)


def test_elasticnet_clf_l1_ratio_invalid_raises():
    with pytest.raises(ValueError, match="l1_ratio"):
        plq_ElasticNet_Classifier(loss={"name": "svm"}, C=1.0, l1_ratio=1.0)

def test_elasticnet_clf_binary_omega_effect():
    """Model coefficient with higher omega weights should be smaller."""
    X, y = _binary_dataset()
    omega_small = np.random.rand(10)
    omega_large = omega_small * 5

    clf1 = plq_ElasticNet_Classifier(loss={"name": "svm"}, C=1.0, l1_ratio=0.5, omega=omega_small)
    clf1.fit(X, y)
    clf2 = plq_ElasticNet_Classifier(loss={"name": "svm"}, C=1.0, l1_ratio=0.5, omega=omega_large)
    clf2.fit(X, y)

    assert np.sum(np.abs(clf2.coef_)) <= np.sum(np.abs(clf1.coef_))


# ===========================================================================
# plq_ElasticNet_Classifier — multiclass OvR
# ===========================================================================

def test_elasticnet_clf_ovr_fits_and_predicts():
    X, y = _multiclass_dataset(n_classes=3)
    clf = plq_ElasticNet_Classifier(
        loss={"name": "svm"}, C=1.0, l1_ratio=0.5, multi_class="ovr"
    )
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (len(y),)
    assert set(np.unique(preds)).issubset(set(np.unique(y)))
    assert accuracy_score(y, preds) > 1 / 3


def test_elasticnet_clf_ovr_estimators_shape():
    X, y = _multiclass_dataset(n_classes=3)
    clf = plq_ElasticNet_Classifier(
        loss={"name": "svm"}, C=1.0, l1_ratio=0.5, multi_class="ovr"
    )
    clf.fit(X, y)
    K = len(clf.classes_)
    assert len(clf.estimators_) == K
    assert clf.coef_.shape == (K, X.shape[1])
    assert clf.intercept_.shape == (K,)


def test_elasticnet_clf_ovr_pipeline():
    X, y = _multiclass_dataset(n_classes=3)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", plq_ElasticNet_Classifier(
            loss={"name": "svm"}, C=1.0, l1_ratio=0.5, multi_class="ovr"
        )),
    ])
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (len(y),)


def test_elasticnet_clf_ovr_omega_effect():
    """Model coefficient with higher omega weights should be smaller."""
    X, y = _multiclass_dataset(n_classes=3)
    omega_small = np.random.rand(10)
    omega_large = omega_small * 5

    clf1 = plq_ElasticNet_Classifier(loss={"name": "svm"}, 
                                     C=1.0, 
                                     l1_ratio=0.5, 
                                     fit_intercept=True,
                                     omega=omega_small,
                                     multi_class="ovr"
    )
    clf1.fit(X, y)
    clf2 = plq_ElasticNet_Classifier(loss={"name": "svm"}, 
                                     C=1.0, 
                                     l1_ratio=0.5, 
                                     fit_intercept=True,
                                     omega=omega_large,
                                     multi_class="ovr"
    )
    clf2.fit(X, y)

    assert np.sum(np.abs(clf2.coef_)) <= np.sum(np.abs(clf1.coef_))


# ===========================================================================
# plq_ElasticNet_Classifier — multiclass OvO
# ===========================================================================

def test_elasticnet_clf_ovo_fits_and_predicts():
    X, y = _multiclass_dataset(n_classes=3)
    clf = plq_ElasticNet_Classifier(
        loss={"name": "svm"}, C=1.0, l1_ratio=0.5, multi_class="ovo"
    )
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (len(y),)
    assert accuracy_score(y, preds) > 1 / 3


def test_elasticnet_clf_ovo_estimators_shape():
    """OvO: K*(K-1)/2 binary classifiers."""
    X, y = _multiclass_dataset(n_classes=3)
    clf = plq_ElasticNet_Classifier(
        loss={"name": "svm"}, C=1.0, l1_ratio=0.5, multi_class="ovo"
    )
    clf.fit(X, y)
    K = len(clf.classes_)
    expected = K * (K - 1) // 2
    assert len(clf.estimators_) == expected
    assert clf.coef_.shape == (expected, X.shape[1])


def test_elasticnet_clf_multiclass_invalid_strategy_raises():
    X, y = _multiclass_dataset(n_classes=3)
    clf = plq_ElasticNet_Classifier(
        loss={"name": "svm"}, C=1.0, l1_ratio=0.5, multi_class="bad"
    )
    with pytest.raises(ValueError, match="multi_class"):
        clf.fit(X, y)


def test_elasticnet_clf_ovo_omega_effect():
    """Model coefficient with higher omega weights should be smaller."""
    X, y = _multiclass_dataset(n_classes=3)
    omega_small = np.random.rand(10)
    omega_large = omega_small * 5

    clf1 = plq_ElasticNet_Classifier(loss={"name": "svm"}, 
                                     C=1.0, 
                                     l1_ratio=0.5, 
                                     fit_intercept=False,
                                     omega=omega_small,
                                     multi_class="ovo"
    )
    clf1.fit(X, y)
    clf2 = plq_ElasticNet_Classifier(loss={"name": "svm"}, 
                                     C=1.0, 
                                     l1_ratio=0.5, 
                                     fit_intercept=False,
                                     omega=omega_large,
                                     multi_class="ovo"
    )
    clf2.fit(X, y)

    assert np.sum(np.abs(clf2.coef_)) <= np.sum(np.abs(clf1.coef_))

# ===========================================================================
# plq_ElasticNet_Regressor
# ===========================================================================

def test_elasticnet_reg_pipeline_fits_and_predicts():
    X, y = _reg_dataset()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", plq_ElasticNet_Regressor(loss={"name": "QR", "qt": 0.5}, C=1.0, l1_ratio=0.5)),
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (len(y),)
    assert np.all(np.isfinite(preds))


def test_elasticnet_reg_cross_val_score():
    X, y = _reg_dataset()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", plq_ElasticNet_Regressor(loss={"name": "QR", "qt": 0.5}, C=1.0, l1_ratio=0.5)),
    ])
    scores = cross_val_score(pipe, X, y, cv=3, scoring="r2")
    assert scores.shape == (3,)
    assert np.mean(scores) > 0.0


def test_elasticnet_reg_multiple_losses():
    X, y = _reg_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    for loss in [{"name": "huber", "tau": 1.0}, {"name": "SVR", "epsilon": 0.1}]:
        reg = plq_ElasticNet_Regressor(loss=loss, C=1.0, l1_ratio=0.3)
        reg.fit(X_tr, y_tr)
        preds = reg.predict(X_te)
        assert preds.shape == (len(y_te),)
        assert np.all(np.isfinite(preds)), f"Non-finite predictions for loss={loss}"


def test_elasticnet_reg_l1_ratio_zero():
    """l1_ratio=0 is pure Ridge — should fit without error."""
    X, y = _reg_dataset()
    reg = plq_ElasticNet_Regressor(loss={"name": "QR", "qt": 0.5}, C=1.0, l1_ratio=0.0)
    reg.fit(X, y)
    assert reg.predict(X).shape == (len(y),)


def test_elasticnet_reg_l1_ratio_invalid_raises():
    with pytest.raises(ValueError, match="l1_ratio"):
        plq_ElasticNet_Regressor(loss={"name": "QR", "qt": 0.5}, C=1.0, l1_ratio=1.0)


def test_elasticnet_reg_intercept_on():
    X, y = _reg_dataset()
    reg = plq_ElasticNet_Regressor(
        loss={"name": "QR", "qt": 0.5}, C=1.0, l1_ratio=0.5, fit_intercept=True,
    )
    reg.fit(X, y)
    assert isinstance(reg.intercept_, float)
    assert reg.coef_.shape == (X.shape[1],)


def test_elasticnet_reg_intercept_off():
    X, y = _reg_dataset()
    reg = plq_ElasticNet_Regressor(
        loss={"name": "QR", "qt": 0.5}, C=1.0, l1_ratio=0.5, fit_intercept=False,
    )
    reg.fit(X, y)
    assert reg.intercept_ == 0.0
    assert reg.coef_.shape == (X.shape[1],)


def test_elasticnet_reg_predict_equals_decision_function():
    X, y = _reg_dataset()
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=0)
    reg = plq_ElasticNet_Regressor(loss={"name": "QR", "qt": 0.5}, C=1.0, l1_ratio=0.5)
    reg.fit(X_tr, y_tr)
    np.testing.assert_array_equal(reg.predict(X_te), reg.decision_function(X_te))

def test_elasticnet_reg_omega_effect():
    """Model coefficient with higher omega weights should be smaller."""
    X, y = _reg_dataset()
    omega_small = np.random.rand(10)
    omega_large = omega_small * 5

    reg1 = plq_ElasticNet_Regressor(loss={"name": "mae"}, C=1.0, l1_ratio=0.5, omega=omega_small)
    reg1.fit(X, y)
    reg2 = plq_ElasticNet_Regressor(loss={"name": "mae"}, C=1.0, l1_ratio=0.5, omega=omega_large)
    reg2.fit(X, y)

    assert np.sum(np.abs(reg2.coef_)) <= np.sum(np.abs(reg1.coef_))
