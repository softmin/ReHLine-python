"""Public constraints, class-aligned scores, warm starts and bounded OvO tasks."""

import pickle
import threading
import tracemalloc
import warnings
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.metrics import top_k_accuracy_score

from rehline import (
    plq_ElasticNet_Classifier,
    plq_ElasticNet_Regressor,
    plq_Ridge_Classifier,
    plq_Ridge_Regressor,
    plqERM_ElasticNet,
    plqERM_Ridge,
)

REGRESSORS = (plqERM_Ridge, plqERM_ElasticNet, plq_Ridge_Regressor, plq_ElasticNet_Regressor)
CLASSIFIERS = (plq_Ridge_Classifier, plq_ElasticNet_Classifier)
MERGED = "combining them so all constraints are enforced"


def regression_options(estimator):
    options = dict(loss={"name": "MSE"}, max_iter=50000, tol=1e-9)
    if estimator in (plq_Ridge_Regressor, plq_ElasticNet_Regressor):
        options["fit_intercept"] = False
    return options


@pytest.mark.parametrize("estimator", REGRESSORS)
@pytest.mark.parametrize("target,expected", [(-2.0, 1.0), (4.0, 2.0)])
def test_both_constraint_sources_hold_and_objective_is_correct(estimator, target, expected):
    X, y, weight = np.ones((4, 1)), np.full(4, target), np.arange(1.0, 5.0)
    A, b = np.ones((1, 1)), np.array([-1.0])
    constraints = [{"name": "custom", "A": np.array([[-1.0]]), "b": np.array([2.0])}]
    model = estimator(A=A, b=b, constraint=constraints, **regression_options(estimator))
    with pytest.warns(UserWarning, match=MERGED) as captured:
        model.fit(X, y, sample_weight=weight)
    assert len(captured) == 1
    assert model.coef_[0] == pytest.approx(expected, abs=1e-8)
    assert model.converged_ and model.constraint_violation_ <= model.tol
    assert (A @ model.coef_ + b).min() >= -model.tol
    assert model.coef_[0] <= 2 + model.tol
    ratio = getattr(model, "l1_ratio", 0)
    objective = weight @ np.square(y - X @ model.coef_)
    objective += ratio * abs(model.coef_).sum() + 0.5 * (1 - ratio) * np.square(model.coef_).sum()
    assert objective == pytest.approx(model.objective_ * (1 - ratio), rel=1e-10, abs=1e-9)
    assert objective == pytest.approx(model.dual_objective_ * (1 - ratio), rel=1e-9, abs=1e-8)
    assert model.get_params()["A"] is A and model.get_params()["constraint"] is constraints
    np.testing.assert_array_equal(A, [[1.0]])
    np.testing.assert_array_equal(b, [-1.0])
    copied = clone(model)
    assert copied.coef_ is None if estimator in (plqERM_Ridge, plqERM_ElasticNet) else not hasattr(copied, "coef_")
    with pytest.warns(UserWarning, match=MERGED):
        copied.fit(X, y, sample_weight=weight)
    np.testing.assert_allclose(copied.coef_, model.coef_, atol=1e-10)
    # Changing the explicit constraint must not retain generated rows from a prior fit.
    model.set_params(A=[[1.0]], b=[-1.5], constraint=None)
    model.fit(X, np.full(4, -2.0))
    assert model.coef_[0] == pytest.approx(1.5, abs=1e-8)
    assert model._A.shape == (1, 1)


@pytest.mark.parametrize("estimator", REGRESSORS)
@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"A": [[1.0]]}, "supplied together"),
        ({"b": [-1.0]}, "supplied together"),
        ({"A": [[1.0]], "b": [0.0, 1.0]}, "one entry per row"),
        ({"A": [[1.0, 0.0]], "b": [0.0]}, "columns"),
        ({"A": [[np.nan]], "b": [0.0]}, "NaN"),
    ],
)
def test_explicit_constraints_validate_at_fit(estimator, kwargs, message):
    model = estimator(**regression_options(estimator), **kwargs)
    with pytest.raises(ValueError, match=message):
        model.fit(np.ones((4, 1)), np.arange(4.0))


@pytest.mark.parametrize("estimator", (plq_Ridge_Regressor, plq_ElasticNet_Regressor))
@pytest.mark.parametrize("scale", [0.2, 3.0])
def test_combined_constraints_use_actual_intercept(estimator, scale):
    A, b = np.array([[0.0, 1.0]]), np.array([-1.5])
    model = estimator(
        loss={"name": "MSE"},
        A=A,
        b=b,
        constraint=[{"name": "nonnegative"}],
        intercept_scaling=scale,
        tol=1e-9,
        max_iter=50000,
    )
    with pytest.warns(UserWarning, match=MERGED):
        model.fit(np.zeros((4, 1)), -np.ones(4))
    assert model.intercept_ == pytest.approx(1.5, abs=1e-8)
    assert model.coef_[0] >= -model.tol
    ratio = getattr(model, "l1_ratio", 0)
    beta = np.r_[model.coef_, model.intercept_ / scale]
    objective = np.square(model.predict(np.zeros((4, 1))) + 1).sum()
    objective += ratio * abs(beta).sum() + 0.5 * (1 - ratio) * (beta @ beta)
    assert objective == pytest.approx(model.objective_ * (1 - ratio), rel=1e-9)
    np.testing.assert_array_equal(A, [[0, 1]])


def classification_data(classes=4):
    return make_classification(
        n_samples=160, n_features=8, n_informative=5, n_redundant=0, n_classes=classes, random_state=42
    )


def classifier(estimator=plq_Ridge_Classifier, **options):
    return estimator(loss={"name": "svm"}, C=0.1, tol=1e-9, max_iter=50000, **options)


@pytest.mark.parametrize("estimator", CLASSIFIERS)
@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
def test_multiclass_combines_constraints_once_per_fit(estimator, strategy):
    X, y = classification_data()
    model = classifier(
        estimator,
        A=np.eye(8),
        b=np.full(8, -0.1),
        constraint=[{"name": "custom", "A": -np.eye(8), "b": np.full(8, 0.3)}],
        multi_class=strategy,
        n_jobs=2,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(X, y)
    assert len(captured) == 1 and MERGED in str(captured[0].message)
    assert np.all(model.converged_)
    assert model.coef_.min() >= 0.1 - model.tol
    assert model.coef_.max() <= 0.3 + model.tol


@pytest.mark.parametrize("estimator", CLASSIFIERS)
@pytest.mark.parametrize("classes", [3, 4, 5])
def test_ovo_scores_align_with_classes_and_existing_prediction_rule(estimator, classes):
    X, y = classification_data(classes)
    y = np.array([f"label-{2 * c + 3}" for c in y])
    model = classifier(estimator, multi_class="ovo").fit(X, y)
    raw = model.set_params(decision_function_shape="ovo").decision_function(X)
    model.set_params(decision_function_shape="ovr")
    scores = model.decision_function(X)
    assert raw.shape == (len(X), classes * (classes - 1) // 2)
    assert scores.shape == (len(X), classes)
    votes, confidence = np.zeros_like(scores), np.zeros_like(scores)
    for col, (i, j) in enumerate(combinations(range(classes), 2)):
        margin = X @ model.coef_[col] + model.intercept_[col]
        np.testing.assert_allclose(raw[:, col], -margin, atol=1e-12)
        votes[:, i] += margin <= 0
        votes[:, j] += margin > 0
        confidence[:, i] -= margin
        confidence[:, j] += margin
    expected = votes + confidence / (3 * (abs(confidence) + 1))
    np.testing.assert_allclose(scores, expected, atol=1e-12)
    np.testing.assert_array_equal(model.predict(X), model.classes_[expected.argmax(axis=1)])
    assert 0 <= top_k_accuracy_score(y, scores, k=2) <= 1
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.decision_function(X), scores)


@pytest.mark.parametrize("estimator", CLASSIFIERS)
def test_calibration_uses_final_ovo_pairs(estimator):
    X, y = classification_data()
    calibrated = CalibratedClassifierCV(classifier(estimator, multi_class="ovo"), cv=3).fit(X, y)
    original = calibrated.predict_proba(X)
    np.testing.assert_allclose(original.sum(axis=1), 1, atol=1e-12)
    for fitted in calibrated.calibrated_classifiers_:
        fitted.estimator.coef_[4:] *= -1000
        fitted.estimator.intercept_[4:] += 1000
    assert np.max(abs(calibrated.predict_proba(X) - original)) > 1e-4


def test_score_interfaces_binary_ovr_and_unfitted():
    with pytest.raises(NotFittedError):
        classifier(decision_function_shape="ovo").decision_function([[1, 2]])
    X, y = classification_data()
    model = classifier().fit(X, y)
    np.testing.assert_allclose(model.decision_function(X), X @ model.coef_.T + model.intercept_)
    with pytest.raises(ValueError, match="multi_class='ovo'"):
        model.set_params(decision_function_shape="ovo").decision_function(X)
    model.fit(X[y < 2], y[y < 2])
    scores = model.decision_function(X)
    np.testing.assert_array_equal(model.set_params(decision_function_shape="ovr").decision_function(X), scores)


@pytest.mark.parametrize("estimator", CLASSIFIERS)
def test_ovo_class_scores_follow_permuted_class_labels(estimator):
    X, y = classification_data()
    mapping = np.array([30, 10, 40, 20])
    original = classifier(estimator, multi_class="ovo").fit(X, y)
    renamed = classifier(estimator, multi_class="ovo").fit(X, mapping[y])
    for column, label in enumerate(renamed.classes_):
        old = int(np.flatnonzero(mapping == label)[0])
        np.testing.assert_allclose(
            renamed.decision_function(X)[:, column], original.decision_function(X)[:, old], atol=1e-7
        )


@pytest.mark.parametrize("estimator", CLASSIFIERS)
@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
@pytest.mark.parametrize("jobs", [1, 2])
def test_warm_refit_preserves_objectives_and_reduces_iterations(estimator, strategy, jobs):
    X, y = classification_data()
    model = classifier(estimator, multi_class=strategy, warm_start=True, n_jobs=jobs).fit(X, y)
    original, iterations, beta = model.objective_.copy(), model.n_iter_.copy(), model.coef_.copy()
    model.fit(X, y)
    assert np.all(model.converged_)
    assert model.n_iter_.max() <= 2
    assert model.n_iter_.sum() < iterations.sum()
    np.testing.assert_allclose(model.objective_, original, atol=1e-8, rtol=1e-9)
    np.testing.assert_allclose(model.coef_, beta, atol=1e-7)
    assert model.kkt_residual_.max() <= model.tol
    # Cold and warm starts must solve the same new objective after changing C.
    model.set_params(C=0.025).fit(X, y)
    cold = clone(model).set_params(warm_start=False).fit(X, y)
    np.testing.assert_allclose(model.objective_, cold.objective_, rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(model.coef_, cold.coef_, atol=1e-7)


@pytest.mark.parametrize("change", ["labels", "strategy", "loss", "samples", "features", "constraints", "penalty"])
def test_incompatible_multiclass_state_is_reset(change, monkeypatch):
    import rehline._class as native_models

    X, y = classification_data()
    model = classifier(plq_ElasticNet_Classifier, multi_class="ovo", warm_start=True).fit(X, y)
    if change == "labels":
        y = y + 10
    elif change == "strategy":
        model.set_params(multi_class="ovr")
    elif change == "loss":
        model.set_params(loss={"name": "MSE"})
    elif change == "samples":
        keep = np.r_[tuple(np.flatnonzero(y == c)[1:] for c in range(4))]
        X, y = X[keep], y[keep]
    elif change == "features":
        X = X[:, :-1]
    elif change == "constraints":
        model.set_params(constraint=[{"name": "nonnegative"}])
    else:
        model.set_params(l1_ratio=0)
    initial = []
    solver = native_models.ReHLine_solver

    def recorded(**kwargs):
        initial.append(sum(np.size(kwargs.get(name)) for name in ("Lambda", "Gamma", "xi", "mu")))
        return solver(**kwargs)

    monkeypatch.setattr(native_models, "ReHLine_solver", recorded)
    model.fit(X, y)
    assert initial and max(initial) == 0
    cold = clone(model).set_params(warm_start=False).fit(X, y)
    np.testing.assert_allclose(model.objective_, cold.objective_, rtol=1e-9, atol=1e-8)


def test_binary_multiclass_transitions_clear_stale_state():
    X, y = classification_data()
    model = classifier(multi_class="ovo", warm_start=True).fit(X[y < 2], y[y < 2])
    assert hasattr(model, "_model_")
    model.fit(X, y)
    assert not hasattr(model, "_model_") and not hasattr(model, "_Lambda")
    model.fit(X[y < 2], y[y < 2])
    for name in ("_models_", "_model_keys_", "_multiclass_signature_"):
        assert not hasattr(model, name)
    assert model.converged_


@pytest.mark.parametrize("jobs", [1, 2])
def test_ovo_task_memory_is_bounded_and_row_order_is_preserved(jobs):
    n, d, classes = 6000, 32, 24
    X = np.zeros((n, d))
    X[:, 0] = np.arange(n)
    y, weight = np.arange(n) % classes, np.arange(1.0, n + 1)
    model = classifier(multi_class="ovo", n_jobs=jobs)
    model.classes_, model.multi_class_ = np.unique(y), "ovo"
    seen, lock = [], threading.Lock()

    def stub(X_sub, target, w_sub, previous=None):
        rows = X_sub[:, 0].astype(int)
        pair = tuple(np.unique(y[rows]))
        expected = np.flatnonzero((y == pair[0]) | (y == pair[1]))
        np.testing.assert_array_equal(rows, expected)
        np.testing.assert_array_equal(target, np.where(y[rows] == pair[1], 1, -1))
        np.testing.assert_array_equal(w_sub, weight[rows])
        with lock:
            seen.append(pair)
        fitted = SimpleNamespace(
            n_iter_=0,
            objective_=0.0,
            dual_objective_=0.0,
            dual_gap_=0.0,
            constraint_violation_=0.0,
            scaled_constraint_violation_=0.0,
            kkt_residual_=0.0,
            converged_=True,
        )
        return fitted, np.zeros(d), 0.0

    model._fit_model = stub
    tracemalloc.start()
    try:
        model._fit_multiclass(X, y, weight)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert sorted(seen) == list(combinations(range(classes), 2))
    # Eager copies alone require 35 MB. The bounded task path stays well below
    # that with headroom for Python/joblib/test overhead, on both worker counts.
    assert peak < 12_000_000
