"""Independent covariance regressions and explicit reference-population rules."""

from itertools import combinations

import numpy as np
import pytest

from rehline import (
    _make_constraint_rehline_param,
    plq_ElasticNet_Classifier,
    plq_ElasticNet_Regressor,
    plq_Ridge_Classifier,
    plq_Ridge_Regressor,
    plqERM_ElasticNet,
    plqERM_Ridge,
)


@pytest.mark.parametrize("estimator", [plqERM_Ridge, plqERM_ElasticNet, plq_Ridge_Regressor, plq_ElasticNet_Regressor])
def test_uncentered_counterexample_matches_analytic_optimum(estimator):
    X = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    y = 1 - X[:, 0]
    original = X.copy()
    X.setflags(write=False)
    options = dict(
        loss={"name": "MSE"},
        C=100,
        tol=1e-9,
        max_iter=100000,
        constraint=[{"name": "fair", "sen_idx": [0], "tol_sen": 0.01}],
    )
    if estimator in [plq_Ridge_Regressor, plq_ElasticNet_Regressor]:
        options["fit_intercept"] = False
    model = estimator(**options).fit(X, y)
    ratio = getattr(model, "l1_ratio", 0)
    mean = (200 - ratio) / (401 - ratio)
    expected = np.array([mean - 0.02, mean + 0.02])
    np.testing.assert_allclose(model.coef_, expected, atol=1e-8)
    prediction = X @ model.coef_
    covariance = np.mean((X[:, 0] - 0.5) * (prediction - prediction.mean()))
    assert abs(covariance) <= 0.01 + model.tol
    expected_objective = 100 * np.square(y - X @ expected).sum()
    expected_objective += ratio * abs(expected).sum() + 0.5 * (1 - ratio) * (expected @ expected)
    assert model.objective_ * (1 - ratio) == pytest.approx(expected_objective, rel=1e-9, abs=1e-8)
    assert model.converged_
    np.testing.assert_array_equal(X, original)


def test_covariance_is_translation_invariant_and_constant_columns_vanish():
    X = np.array([[0.0, 2.0, 1.0], [1.0, -1.0, 1.0], [0.0, 3.0, 1.0], [1.0, 4.0, 1.0]])
    constraints = [{"name": "fair", "sen_idx": [0, 2], "tol_sen": [0.01, 0.0]}]
    A, b = _make_constraint_rehline_param(constraints, X)
    shifted_A, shifted_b = _make_constraint_rehline_param(constraints, X + [1e6, -1e8, 1e12])
    np.testing.assert_array_equal(A, shifted_A)
    np.testing.assert_array_equal(b, shifted_b)
    np.testing.assert_array_equal(A[:, -1], 0.0)
    np.testing.assert_array_equal(A[2:], 0.0)
    # Constant predictions have no covariance despite a nonzero product mean.
    assert np.mean(X[:, 0] * 10) == 5
    assert np.max(abs(A @ np.array([0.0, 0.0, 10.0]))) == 0


@pytest.mark.parametrize("estimator", [plqERM_Ridge, plq_Ridge_Regressor])
def test_loss_weights_and_zero_weight_reference_rows(estimator):
    X, y = np.array([[1.0], [2.0], [10.0]]), np.array([10.0, 10.0, -100.0])
    options = dict(
        loss={"name": "MSE"}, tol=1e-9, max_iter=100000, constraint=[{"name": "fair", "sen_idx": [0], "tol_sen": 1.0}]
    )
    if estimator is plq_Ridge_Regressor:
        options["fit_intercept"] = False
    models = [estimator(**options).fit(X, y, sample_weight=w) for w in ([1.0, 1.0, 0.0], [3.0, 20.0, 0.0])]
    reference = X[:2] if estimator is plq_Ridge_Regressor else X
    variance = np.var(reference[:, 0])
    for model in models:
        assert model.coef_[0] == pytest.approx(1 / variance, abs=1e-8)
        assert model.converged_
    np.testing.assert_array_equal(models[0]._A, models[1]._A)


@pytest.mark.parametrize("estimator", [plq_Ridge_Classifier, plq_ElasticNet_Classifier])
@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
def test_each_multiclass_population_is_centered_after_weight_filtering(estimator, strategy):
    rng = np.random.default_rng(42)
    y = np.tile(np.arange(5), 10)
    X = rng.normal(size=(len(y), 3)) + y[:, None] * [1.0, 0.2, -0.3] + 5
    weight = np.ones(len(y))
    weight[::7] = 0
    model = estimator(
        loss={"name": "svm"},
        C=0.1,
        multi_class=strategy,
        n_jobs=2,
        intercept_scaling=3.0,
        class_weight={1: 2.0, 4: 0.0},
        constraint=[{"name": "fair", "sen_idx": [0], "tol_sen": 0.015}],
        tol=1e-9,
        max_iter=100000,
    ).fit(X, y, sample_weight=weight)
    active = (weight > 0) & (y != 4)
    keys = list(combinations(model.classes_, 2)) if strategy == "ovo" else [(c,) for c in model.classes_]
    assert len(model.classes_) == 4
    for k, key in enumerate(keys):
        rows = active & np.isin(y, key) if strategy == "ovo" else active
        s = X[rows, 0]
        score = X[rows] @ model.coef_[k] + model.intercept_[k]
        covariance = (s - s.mean()) @ (score - score.mean()) / len(s)
        assert abs(covariance) <= 0.015 + model.tol
        reference = (s - s.mean()) @ (X[rows] - X[rows].mean(axis=0)) / len(s)
        np.testing.assert_allclose(
            model._models_[k]._A, np.vstack((np.r_[-reference, 0.0], np.r_[reference, 0.0])), atol=1e-12
        )
    assert np.all(model.converged_)


@pytest.mark.parametrize(
    "params",
    [
        {"sen_idx": [], "tol_sen": []},
        {"sen_idx": [0.5], "tol_sen": [0.1]},
        {"sen_idx": [True], "tol_sen": [0.1]},
        {"sen_idx": [3], "tol_sen": [0.1]},
        {"sen_idx": [0], "tol_sen": [-1.0]},
        {"sen_idx": [0], "tol_sen": [np.inf]},
        {"sen_idx": [0], "tol_sen": [np.nan]},
        {"sen_idx": [0, 1], "tol_sen": [0.1]},
    ],
)
def test_invalid_fairness_parameters(params):
    with pytest.raises(ValueError):
        _make_constraint_rehline_param([{"name": "fair", **params}], np.ones((3, 2)))


def test_empty_fairness_population_is_rejected():
    with pytest.raises(ValueError, match="require observations"):
        _make_constraint_rehline_param([{"name": "fair", "sen_idx": [0], "tol_sen": [0.1]}], np.empty((0, 2)))
