"""Bounded OvO inference preserves the full pair aggregation and training state."""

import pickle
import tracemalloc
from itertools import combinations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from rehline import plq_ElasticNet_Classifier, plq_Ridge_Classifier


@pytest.mark.parametrize("estimator", [plq_Ridge_Classifier, plq_ElasticNet_Classifier])
@pytest.mark.parametrize("classes", [3, 12, 17])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided"])
def test_blocked_scores_match_full_pair_formula_and_keep_objective(estimator, classes, layout):
    X, y = make_classification(
        n_samples=classes * 5,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=classes,
        n_clusters_per_class=1,
        random_state=14,
    )
    model = estimator(loss={"name": "svm"}, C=0.03, multi_class="ovo", tol=1e-10, max_iter=100000).fit(X, y)
    before = pickle.dumps(vars(model))
    probe = np.asfortranarray(X) if layout == "fortran" else (X[::2] if layout == "strided" else X)
    margins = probe @ model.coef_.T + model.intercept_
    expected = model._class_scores(margins)
    actual = model.decision_function(probe)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(model.predict(probe), model.classes_[expected.argmax(axis=1)])
    model.set_params(decision_function_shape="ovo")
    np.testing.assert_allclose(model.decision_function(probe), -margins, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(model.predict(probe), model.classes_[expected.argmax(axis=1)])
    model.set_params(decision_function_shape="ovr")
    assert pickle.dumps(vars(model)) == before
    # Each pair's full original weighted hinge + L1/L2/intercept objective.
    ratio = getattr(model, "l1_ratio", 0)
    for key, binary in zip(combinations(model.classes_, 2), model._models_):
        selected = np.isin(y, key)
        design = np.column_stack((X[selected], np.ones(selected.sum())))
        target = np.where(y[selected] == key[-1], 1.0, -1.0)
        beta = binary.coef_
        value = 0.03 * np.maximum(1 - target * (design @ beta), 0).sum()
        value += 0.5 * (1 - ratio) * (beta @ beta) + ratio * abs(beta).sum()
        assert value == pytest.approx(binary.objective_ * (1 - ratio), rel=1e-8, abs=1e-9)


def synthetic_model(classes, d):
    model = plq_Ridge_Classifier(loss={"name": "svm"}, multi_class="ovo")
    pairs = list(combinations(range(classes), 2))
    model.classes_, model.multi_class_, model.n_features_in_ = np.arange(classes), "ovo", d
    rng = np.random.default_rng(917)
    model.coef_ = rng.normal(size=(len(pairs), d))
    model.intercept_ = np.zeros(len(pairs))
    model.estimators_ = [(model.coef_[i], 0.0, a, b) for i, (a, b) in enumerate(pairs)]
    return model


def test_zero_margins_and_tied_votes_across_blocks():
    model = synthetic_model(17, 3)
    model.coef_[:] = 0
    model.intercept_[::3] = 1
    model.intercept_[1::3] = -1
    model.intercept_[2::3] = 0  # Exact zeros in each block, including the final partial block.
    X = np.zeros((7, 3))
    expected = model._class_scores(model._decision_function(X))
    np.testing.assert_array_equal(model.decision_function(X), expected)
    np.testing.assert_array_equal(model.predict(X), model.classes_[expected.argmax(axis=1)])


def test_default_scores_do_not_allocate_full_pair_matrix(monkeypatch):
    model = synthetic_model(80, 12)
    X = np.random.default_rng(14).normal(size=(1600, 12))
    expected = model._class_scores(model._decision_function(X))
    monkeypatch.setattr(model, "_decision_function", lambda X: pytest.fail("Full pair matrix allocated"))
    tracemalloc.start()
    try:
        actual = model.decision_function(X)
        predictions = model.predict(X)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 15_000_000  # A single full pair matrix would require 40 MB.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(predictions, expected.argmax(axis=1))
