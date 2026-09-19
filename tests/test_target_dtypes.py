"""Targets must have the same mathematical loss regardless of numeric container."""

import pickle

import numpy as np
import pytest
from sklearn.base import clone

from rehline import (
    ReHLoss,
    _make_loss_rehline_param,
    plq_ElasticNet_Regressor,
    plq_Ridge_Regressor,
    plqERM_ElasticNet,
    plqERM_Ridge,
    plqERM_Ridge_path_sol,
)

ESTIMATORS = (plqERM_Ridge, plqERM_ElasticNet, plq_Ridge_Regressor, plq_ElasticNet_Regressor)
LOSSES = (
    {"name": "MAE"},
    {"name": "huber", "tau": 0.7},
    {"name": "SVR", "epsilon": 0},
    {"name": "SVR", "epsilon": 2},
    {"name": "MSE"},
    {"name": "QR", "qt": 0.3},
    {"name": "QR_eps", "qt": 0.3, "epsilon": 1},
)


def direct_loss(y, score, loss):
    residual = np.asarray(y, dtype=float) - score
    z = abs(residual)
    if loss["name"] == "MAE":
        return z
    if loss["name"] == "MSE":
        return z**2
    if loss["name"] == "huber":
        tau = loss["tau"]
        return np.where(z <= tau, 0.5 * z**2, tau * (z - 0.5 * tau))
    if loss["name"] == "SVR":
        return np.maximum(z - loss["epsilon"], 0)
    check = np.maximum(loss["qt"] * residual, (loss["qt"] - 1) * residual)
    return np.maximum(check - loss.get("epsilon", 0), 0)


def objective(model, X, y, weights):
    intercept = getattr(model, "intercept_", 0.0)
    beta = (
        np.r_[model.coef_, intercept / model.intercept_scaling]
        if getattr(model, "fit_intercept", False)
        else model.coef_
    )
    ratio = getattr(model, "l1_ratio", 0.0)
    value = model.C * (weights @ direct_loss(y, X @ model.coef_ + intercept, model.loss))
    return value + 0.5 * (1 - ratio) * (beta @ beta) + ratio * abs(beta).sum()


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint64, np.int8, np.int64, np.float32])
def test_numeric_targets_preserve_weighted_objective_across_warm_refits(estimator, loss, dtype):
    rng = np.random.default_rng(65)
    X = rng.normal(size=(12, 3))
    y = np.tile([1, 2, 5], 4).astype(dtype)
    y.setflags(write=False)
    weights = np.linspace(0.1, 2, len(y))
    weights[::5] = 0
    options = dict(loss=loss, C=0.2, warm_start=True, tol=1e-10, max_iter=100000)
    if estimator in (plq_Ridge_Regressor, plq_ElasticNet_Regressor):
        options.update(fit_intercept=True, intercept_scaling=2.0)
    model = estimator(**options)
    for C in (0.2, 0.5):
        reference = clone(model).set_params(C=C, warm_start=False).fit(X, y.astype(float), sample_weight=weights)
        for _ in range(2):
            model.set_params(C=C).fit(X, y, sample_weight=weights)
            assert model.converged_
            actual = objective(model, X, y, weights)
            scale = 1 - getattr(model, "l1_ratio", 0.0)
            np.testing.assert_allclose(
                actual,
                [reference.objective_ * scale, model.objective_ * scale, model.dual_objective_ * scale],
                rtol=1e-8,
                atol=1e-9,
            )
            np.testing.assert_allclose(model.coef_, reference.coef_, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(y, np.tile([1, 2, 5], 4))


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int64])
def test_smallest_signed_integer_is_converted_before_negation(loss, dtype):
    y = np.array([np.iinfo(dtype).min, 0, np.iinfo(dtype).max], dtype=dtype)
    X = np.zeros((3, 1))
    U, V, Tau, S, T = _make_loss_rehline_param(loss, X, y)
    actual = ReHLoss(U, V, S, T, Tau).values(np.zeros(3))
    np.testing.assert_allclose(actual, direct_loss(y, np.zeros(3), loss), rtol=1e-14, atol=1e-12)
    assert np.isfinite(actual).all()


@pytest.mark.parametrize("container", ["list", "pandas"])
def test_list_and_pandas_targets_and_path_preserve_analytic_objective(container):
    y = [1, 2, 3] if container == "list" else pytest.importorskip("pandas").Series([1, 2, 3], dtype="uint8")
    X = np.ones((3, 1))
    U, V, Tau, S, T = _make_loss_rehline_param({"name": "MAE"}, X, y)
    np.testing.assert_array_equal(ReHLoss(U, V, S, T, Tau).values(np.zeros(3)), [1.0, 2.0, 3.0])
    result = plqERM_Ridge_path_sol(
        X, y, loss={"name": "MAE"}, Cs=[0.5, 1.0], warm_start=True, tol=1e-10, max_iter=100000, return_time=False
    )
    np.testing.assert_allclose(result[2], [2.0, 3.5], rtol=0, atol=1e-9)


@pytest.mark.parametrize("target", [[np.nan], [np.inf], [1j], [[1]], ["bad"]])
def test_shared_converter_rejects_invalid_targets(target):
    with pytest.raises(ValueError, match="y"):
        _make_loss_rehline_param({"name": "MAE"}, np.ones((1, 1)), target)


def test_invalid_target_refit_preserves_previous_state():
    X, y = np.ones((3, 1)), np.array([1, 2, 3], dtype=np.uint8)
    model = plqERM_Ridge(loss={"name": "MAE"}, tol=1e-10).fit(X, y)
    before = pickle.dumps(vars(model))
    with pytest.raises(ValueError):
        model.fit(X, [np.nan, 2, 3])
    assert pickle.dumps(vars(model)) == before
