"""Score formatting follows sklearn without changing the fitted problems."""

import pickle
from itertools import combinations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from rehline import plq_ElasticNet_Classifier, plq_Ridge_Classifier

CLASSIFIERS = (plq_Ridge_Classifier, plq_ElasticNet_Classifier)


def classifier(estimator, **kwargs):
    return estimator(loss={"name": "svm"}, C=0.1, tol=1e-10, max_iter=100000, **kwargs)


def data(classes=4):
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=4, n_redundant=0, n_classes=classes, random_state=371
    )
    return X, np.array([f"label-{31 - 3 * c}" for c in y])


@pytest.mark.parametrize("estimator", CLASSIFIERS)
@pytest.mark.parametrize("classes", [2, 3, 4, 5])
def test_formats_preserve_state_predictions_and_full_objective(estimator, classes, monkeypatch):
    X, y = data(classes)
    weight = np.linspace(0.1, 2, len(y))
    weight[::17] = 0
    model = classifier(estimator, multi_class="ovo", constraint=[{"name": "nonnegative"}], intercept_scaling=3.0).fit(
        X, y, sample_weight=weight
    )
    original = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in vars(model).items()}
    labels, scores = model.predict(X), model.decision_function(X)
    native_margins = X @ model.coef_.T + model.intercept_
    assert np.all(model.coef_ >= -model.tol)
    assert not hasattr(model, "pairwise_decision_function")

    def forbid_solve(*args, **kwargs):
        raise AssertionError("Changing score output must never invoke the solver")

    monkeypatch.setattr("rehline._class.ReHLine_solver", forbid_solve)
    raw = model.set_params(decision_function_shape="ovo").decision_function(X)
    expected_shape = (len(X),) if classes == 2 else (len(X), classes * (classes - 1) // 2)
    assert raw.shape == expected_shape
    np.testing.assert_array_equal(raw, native_margins if classes == 2 else -native_margins)
    np.testing.assert_array_equal(model.predict(X), labels)
    for name in ("coef_", "intercept_", "objective_", "dual_objective_", "n_iter_", "constraint_violation_"):
        np.testing.assert_array_equal(getattr(model, name), original[name])

    # Evaluate the original weighted hinge loss and penalties, including the
    # augmented intercept, independently of reported solver objective values.
    keys = [tuple(model.classes_)] if classes == 2 else list(combinations(model.classes_, 2))
    binaries = [model._model_] if classes == 2 else model._models_
    for key, binary in zip(keys, binaries):
        active = (weight > 0) & np.isin(y, key)
        target = np.where(y[active] == key[1], 1.0, -1.0)
        design = np.column_stack((X[active], np.full(active.sum(), 3.0)))
        beta = binary.coef_
        ratio = getattr(model, "l1_ratio", 0)
        objective = model.C * (weight[active] @ np.maximum(1 - target * (design @ beta), 0))
        objective += 0.5 * (1 - ratio) * (beta @ beta) + ratio * abs(beta).sum()
        assert objective == pytest.approx(binary.objective_ * (1 - ratio), rel=1e-10, abs=1e-9)
        assert objective == pytest.approx(binary.dual_objective_ * (1 - ratio), rel=1e-9, abs=1e-9)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.decision_function(X), raw)
    np.testing.assert_array_equal(restored.predict(X), labels)
    np.testing.assert_array_equal(model.set_params(decision_function_shape="ovr").decision_function(X), scores)


@pytest.mark.parametrize("estimator", CLASSIFIERS)
@pytest.mark.parametrize("strategy", [None, "ovr", "ovo"])
def test_binary_ignores_valid_output_format_and_keeps_second_class_positive(estimator, strategy):
    X, y = data(2)
    model = classifier(estimator, multi_class=strategy, decision_function_shape="ovo").fit(X, y)
    raw = model.decision_function(X)
    assert raw.shape == (len(X),)
    np.testing.assert_array_equal(model.predict(X), model.classes_[(raw > 0).astype(int)])
    np.testing.assert_array_equal(model.set_params(decision_function_shape="ovr").decision_function(X), raw)


@pytest.mark.parametrize("estimator", CLASSIFIERS)
@pytest.mark.parametrize("shape", [None, "bad", "OVO", 1, [], np.array(["ovo", "ovr"])])
def test_invalid_shape_rejected_before_solving(estimator, shape, monkeypatch):
    X, y = data(2)
    model = classifier(estimator, multi_class="ovo", decision_function_shape=shape)
    monkeypatch.setattr(model, "_fit_model", lambda *args: pytest.fail("Invalid format reached solver"))
    with pytest.raises(ValueError, match="decision_function_shape must"):
        model.fit(X, y)


@pytest.mark.parametrize("estimator", CLASSIFIERS)
@pytest.mark.parametrize("strategy", [None, "ovr"])
def test_ovr_cannot_supply_pair_scores_at_fit_or_after_set_params(estimator, strategy):
    X, y = data()
    model = classifier(estimator, multi_class=strategy, decision_function_shape="ovo")
    with pytest.raises(NotFittedError):
        model.decision_function(X)
    with pytest.raises(ValueError, match="requires multi_class='ovo'"):
        model.fit(X, y)
    model.set_params(decision_function_shape="ovr").fit(X, y)
    np.testing.assert_allclose(model.decision_function(X), X @ model.coef_.T + model.intercept_)
    model.set_params(decision_function_shape="ovo")
    with pytest.raises(ValueError, match="requires multi_class='ovo'"):
        model.decision_function(X)
    model.set_params(decision_function_shape="invalid")
    with pytest.raises(ValueError, match="decision_function_shape must"):
        model.decision_function(X)


@pytest.mark.parametrize("classes", [3, 4])
def test_raw_pair_order_and_sign_match_svc(classes):
    # Symmetry makes each pair's optimal intercept zero, so SVC's unpenalized
    # intercept and ReHLine's disabled intercept solve the same hinge objective.
    X = np.repeat(np.eye(classes), 4, axis=0)
    y = np.repeat(np.array([f"class-{10 - i}" for i in range(classes)]), 4)
    probe = np.vstack((X, -X, np.eye(classes) * 0.3))
    model = plq_Ridge_Classifier(
        loss={"name": "svm"},
        C=0.1,
        multi_class="ovo",
        fit_intercept=False,
        decision_function_shape="ovo",
        tol=1e-10,
        max_iter=100000,
    ).fit(X, y)
    reference = SVC(kernel="linear", C=0.1, decision_function_shape="ovo", tol=1e-10).fit(X, y)
    np.testing.assert_array_equal(model.classes_, reference.classes_)
    np.testing.assert_allclose(model.decision_function(probe), reference.decision_function(probe), atol=1e-9)
    for col, (first, second) in enumerate(combinations(model.classes_, 2)):
        assert np.all(model.decision_function(X[y == first])[:, col] > 0)
        assert np.all(model.decision_function(X[y == second])[:, col] < 0)
    model.set_params(decision_function_shape="ovr")
    reference.set_params(decision_function_shape="ovr", break_ties=True)
    np.testing.assert_allclose(model.decision_function(probe), reference.decision_function(probe), atol=1e-9)
    np.testing.assert_array_equal(model.predict(probe), reference.predict(probe))


@pytest.mark.parametrize("estimator", CLASSIFIERS)
def test_default_class_scores_match_sklearn_ovo_wrapper(estimator):
    X, y = data()
    model = classifier(estimator, multi_class="ovo").fit(X, y)
    reference = OneVsOneClassifier(classifier(estimator)).fit(X, y)
    np.testing.assert_allclose(model.decision_function(X), reference.decision_function(X), atol=1e-8)
    np.testing.assert_array_equal(model.predict(X), reference.predict(X))


def test_vote_ties_use_confidence_and_format_does_not_change_prediction():
    X, y = data(3)
    model = classifier(plq_Ridge_Classifier, multi_class="ovo").fit(X, y)
    # A three-way cycle ties votes. Confidence picks the second class; SVC's
    # default break_ties=False would pick the first. All-zero margins pick first.
    model.coef_ = np.zeros((3, 5))
    model.coef_[:, :3] = np.eye(3)
    model.intercept_ = np.zeros(3)
    probe = np.array([[3.0, -1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    expected = model.classes_[[1, 0]]
    for shape in ["ovr", "ovo"]:
        model.set_params(decision_function_shape=shape)
        np.testing.assert_array_equal(model.predict(probe), expected)


@pytest.mark.parametrize("estimator", CLASSIFIERS)
def test_clone_gridsearch_pipeline_and_refit_preserve_format(estimator):
    X, y = data()
    base = classifier(estimator, multi_class="ovo", decision_function_shape="ovo", warm_start=True)
    assert clone(base).get_params()["decision_function_shape"] == "ovo"
    grid = GridSearchCV(
        Pipeline([("scale", StandardScaler()), ("model", base)]),
        {"model__decision_function_shape": ["ovr", "ovo"]},
        cv=2,
        scoring="accuracy",
        error_score="raise",
    ).fit(X, y)
    np.testing.assert_array_equal(grid.cv_results_["split0_test_score"], [grid.cv_results_["split0_test_score"][0]] * 2)
    np.testing.assert_array_equal(grid.cv_results_["split1_test_score"], [grid.cv_results_["split1_test_score"][0]] * 2)
    fitted = base.fit(X, y)
    objective, coef = fitted.objective_.copy(), fitted.coef_.copy()
    for shape in ["ovo", "ovr"]:
        cold = clone(fitted).set_params(warm_start=False, decision_function_shape=shape).fit(X, y)
        np.testing.assert_array_equal(cold.objective_, objective)
        np.testing.assert_array_equal(cold.coef_, coef)
    fitted.set_params(decision_function_shape="ovr").fit(X, y)
    np.testing.assert_allclose(fitted.objective_, objective, atol=1e-9, rtol=1e-9)
    assert np.max(fitted.n_iter_) <= 2
