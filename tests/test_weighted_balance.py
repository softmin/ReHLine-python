"""Balanced class weighting must use the mass of the supplied sample weights."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.svm import LinearSVC

from rehline import plq_ElasticNet_Classifier, plq_Ridge_Classifier
from rehline._validation import balanced_sample_weights


def test_balanced_weights_match_integer_replication_and_analytic_optimum():
    X, y = np.zeros((4, 1)), np.array([0, 0, 1, 1])
    weight = np.array([100, 100, 1, 1])
    options = dict(loss={"name": "svm"}, C=0.1, class_weight="balanced", tol=1e-10, max_iter=100000)
    weighted = plq_Ridge_Classifier(**options).fit(X, y, sample_weight=weight)
    copied = plq_Ridge_Classifier(**options).fit(np.repeat(X, weight, axis=0), np.repeat(y, weight))
    assert weighted.intercept_ == pytest.approx(0.0, abs=1e-9)
    assert weighted.objective_ == pytest.approx(20.2, abs=1e-9)
    assert weighted.dual_objective_ == pytest.approx(20.2, abs=1e-9)
    np.testing.assert_allclose(weighted.coef_, copied.coef_, atol=1e-9, rtol=0)
    assert weighted.intercept_ == pytest.approx(copied.intercept_, abs=1e-9)


def effective_weights(y, weight):
    classes = np.unique(y[weight > 0])
    result = np.zeros(len(y))
    for label in classes:
        rows = y == label
        result[rows] = weight[rows] * weight.sum() / (len(classes) * weight[rows].sum())
    return result


def full_objectives(model, X, y, weight):
    from itertools import combinations

    classes = model.classes_
    weights = effective_weights(y, weight)
    keys = list(combinations(classes, 2)) if len(classes) == 2 or model.multi_class_ == "ovo" else list(classes)
    ratio = getattr(model, "l1_ratio", 0)
    values = []
    for i, key in enumerate(keys):
        pairwise = len(classes) == 2 or model.multi_class_ == "ovo"
        rows = np.isin(y, key) if pairwise else np.ones(len(y), dtype=bool)
        positive = key[1] if pairwise else key
        target = np.where(y[rows] == positive, 1, -1)
        coef = model.coef_ if len(classes) == 2 else model.coef_[i]
        intercept = model.intercept_ if len(classes) == 2 else model.intercept_[i]
        score = X[rows] @ coef + intercept
        beta = np.r_[coef, intercept / model.intercept_scaling] if model.fit_intercept else coef
        omega = np.ones(len(beta))
        value = model.C * (weights[rows] @ np.maximum(1 - target * score, 0))
        value += 0.5 * (1 - ratio) * (beta @ beta) + ratio * (omega @ abs(beta))
        values.append(value)
    return np.array(values)


@pytest.mark.parametrize("estimator", [plq_Ridge_Classifier, plq_ElasticNet_Classifier])
@pytest.mark.parametrize("classes", [2, 3, 4])
@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
@pytest.mark.parametrize("intercept", [False, True])
def test_balanced_binary_and_multiclass_replication_warm_refits_and_full_objective(
    estimator, classes, strategy, intercept
):
    rng = np.random.default_rng(19)
    y = np.array([f"label-{i}" for i in range(classes)] * 6)
    X = rng.normal(size=(len(y), 3))
    weight = rng.integers(1, 5, len(y))
    weight[:classes] = 0
    model = estimator(
        loss={"name": "svm"},
        C=0.05,
        class_weight="balanced",
        multi_class=strategy,
        fit_intercept=intercept,
        intercept_scaling=2.0,
        warm_start=True,
        tol=1e-10,
        max_iter=100000,
    )
    for weights in (weight, np.roll(weight, classes)):
        copied = clone(model).set_params(warm_start=False).fit(np.repeat(X, weights, axis=0), np.repeat(y, weights))
        for _ in range(2):
            model.fit(X, y, sample_weight=weights)
            assert np.all(model.converged_)
            np.testing.assert_allclose(model.coef_, copied.coef_, rtol=0, atol=1e-8)
            np.testing.assert_allclose(model.intercept_, copied.intercept_, rtol=0, atol=1e-8)
            value = full_objectives(model, X, y, weights)
            ratio = getattr(model, "l1_ratio", 0)
            np.testing.assert_allclose(value, model.objective_ * (1 - ratio), rtol=1e-8, atol=1e-9)
            np.testing.assert_allclose(value, model.dual_objective_ * (1 - ratio), rtol=1e-8, atol=1e-9)
            np.testing.assert_allclose(value, np.atleast_1d(copied.objective_) * (1 - ratio), rtol=1e-8, atol=1e-9)
            np.testing.assert_array_equal(model.to_inference().predict(X), model.predict(X))


@pytest.mark.parametrize("classes", [2, 3])
@pytest.mark.parametrize("intercept", [False, True])
def test_weighted_balance_matches_linearsvc_with_independently_explicit_weights(classes, intercept):
    rng = np.random.default_rng(52)
    X, y = rng.normal(size=(30, 3)), np.arange(30) % classes
    weights = rng.uniform(0.1, 5, len(y))
    model = plq_Ridge_Classifier(
        loss={"name": "svm"},
        C=0.1,
        class_weight="balanced",
        multi_class="ovr",
        fit_intercept=intercept,
        intercept_scaling=2.0,
        tol=1e-10,
        max_iter=100000,
    ).fit(X, y, sample_weight=weights)
    # Explicit effective weights work on sklearn 1.6 too; its balanced option
    # predates the fix that incorporated sample_weight into class frequencies.
    reference = LinearSVC(
        loss="hinge",
        C=0.1,
        dual=True,
        fit_intercept=intercept,
        intercept_scaling=2.0,
        tol=1e-10,
        max_iter=100000,
        random_state=42,
    ).fit(X, y, sample_weight=effective_weights(y, weights))
    np.testing.assert_allclose(np.atleast_2d(model.coef_), reference.coef_, rtol=0, atol=1e-8)
    np.testing.assert_allclose(model.intercept_, reference.intercept_, rtol=0, atol=1e-8)


@pytest.mark.parametrize(
    "weights",
    [
        [100.0, 100.0, 1.0, 1.0],
        [1e308] * 4,
        [1e-300, 1e-300, 1e300, 1e300],
        [1e-300, 1e300, 1e300, 1e-300],
        [np.nextafter(0.0, 1.0), np.finfo(float).max],
    ],
)
def test_balanced_weight_arithmetic_against_decimal_reference(weights):
    from decimal import Decimal, localcontext

    weights = np.array(weights)
    y = np.repeat([0, 1], len(weights) // 2)
    with localcontext() as ctx:
        ctx.prec = 80
        exact = [Decimal.from_float(w) for w in weights]
        total = sum(exact)
        class_mass = [sum(w for w, label in zip(exact, y) if label == c) for c in (0, 1)]
        expected = [float(w * total / (2 * class_mass[label])) for w, label in zip(exact, y)]
    with np.errstate(all="raise"):
        actual = balanced_sample_weights(y, weights, 2)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)
    assert np.isfinite(actual).all() and (actual > 0).all()


def test_unrepresentable_balanced_weights_fail_and_preserve_fitted_state():
    import pickle

    y, X = np.array([0, 0, 0, 0, 0, 1]), np.zeros((6, 1))
    model = plq_Ridge_Classifier(loss={"name": "svm"}, class_weight="balanced").fit(X, y)
    before = pickle.dumps(vars(model))
    with pytest.raises(ValueError, match="Balanced sample weights"):
        model.fit(X, y, sample_weight=np.full(6, 1e308))
    assert pickle.dumps(vars(model)) == before


def test_zero_weight_class_is_removed_before_balancing_and_scalar_weights_match():
    X, y = np.zeros((6, 1)), np.array(["a", "a", "b", "b", "ghost", "ghost"])
    model = plq_Ridge_Classifier(loss={"name": "svm"}, class_weight="balanced", tol=1e-10)
    model.fit(X, y, sample_weight=[1.0, 1.0, 100.0, 100.0, 0.0, 0.0])
    np.testing.assert_array_equal(model.classes_, ["a", "b"])
    assert model.intercept_ == pytest.approx(0, abs=1e-9)
    scalar = clone(model).fit(X[:4], y[:4], sample_weight=2.0)
    explicit = clone(model).fit(X[:4], y[:4], sample_weight=np.full(4, 2.0))
    assert scalar.objective_ == pytest.approx(explicit.objective_, abs=1e-10)
