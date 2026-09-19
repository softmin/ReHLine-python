"""Every public estimator keeps a coherent fitted snapshot when fit raises."""

import pickle
from functools import partial

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.utils.validation import check_is_fitted

from rehline import CQR_Ridge, ReHLine, plqERM_ElasticNet, plqERM_Ridge, plqMF_Ridge


def fitted_state(model):
    parameters = model.get_params(deep=False)
    return pickle.dumps({k: v for k, v in vars(model).items() if k not in parameters})


@pytest.mark.parametrize("kind", ["raw", "ridge", "elasticnet", "cqr"])
@pytest.mark.parametrize("failure", ["weights", "solver", "warning"])
def test_failed_convex_refit_preserves_state_and_recovers(kind, failure, monkeypatch):
    import rehline._class as module

    rng = np.random.default_rng(18)
    X, y = rng.normal(size=(24, 3)), rng.normal(size=24)
    options = dict(C=0.1, tol=1e-9, max_iter=100000, warm_start=True)
    if kind == "raw":
        model = ReHLine(U=-np.ones((1, len(y))), V=np.ones((1, len(y))), **options)
    elif kind == "cqr":
        model = CQR_Ridge([0.2, 0.8], **options)
    else:
        model = (plqERM_Ridge if kind == "ridge" else plqERM_ElasticNet)(loss={"name": "MSE"}, **options)

    def fit(candidate, features, **kwargs):
        return candidate.fit(features, **kwargs) if kind == "raw" else candidate.fit(features, y, **kwargs)

    fit(model, X)
    before = fitted_state(model)
    params = model.get_params().copy()
    prediction = model.predict(X) if kind == "cqr" else model.decision_function(X)
    new_X = np.column_stack((X, X[:, 0]))
    if kind == "cqr":
        model.set_params(quantiles=[0.1, 0.5, 0.9])
    kwargs = {}
    if failure == "weights":
        kwargs["sample_weight"] = -np.ones(len(y))
    elif failure == "solver":
        monkeypatch.setattr(
            module, "ReHLine_solver", lambda **kw: (_ for _ in ()).throw(RuntimeError("native failure"))
        )
    else:
        model.set_params(max_iter=1, tol=1e-15)
        new_X *= 100
    with pytest.raises((ValueError, RuntimeError, ConvergenceWarning)):
        fit(model, new_X, **kwargs)
    assert fitted_state(model) == before
    after = model.predict(X) if kind == "cqr" else model.decision_function(X)
    np.testing.assert_array_equal(after, prediction)
    monkeypatch.undo()
    fit(model.set_params(**params), X)
    reference = fit(clone(model).set_params(warm_start=False), X)
    np.testing.assert_allclose(model.objective_, reference.objective_, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("kind", ["raw", "ridge", "elasticnet", "cqr", "mf"])
def test_first_failed_fit_remains_unfitted(kind):
    X, y = np.ones((6, 2)), np.arange(6.0)
    if kind == "raw":
        model = ReHLine(U=np.ones((1, 6)), V=np.full((1, 6), np.nan))
        call = partial(model.fit, X)
    elif kind == "cqr":
        model = CQR_Ridge([0.2, 0.8])
        call = partial(model.fit, X, y, sample_weight=-1)
    elif kind == "mf":
        model = plqMF_Ridge(2, 2, loss={"name": "invalid"}, rank=1, random_state=42)
        call = partial(model.fit, np.tile([[0, 0], [1, 1]], (3, 1)), y)
    else:
        model = (plqERM_Ridge if kind == "ridge" else plqERM_ElasticNet)(loss={"name": "invalid"})
        call = partial(model.fit, X, y)
    with pytest.raises(ValueError):
        call()
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


@pytest.mark.parametrize("failure", ["loss", "constraints", "weights", "solver", "warning"])
@pytest.mark.parametrize("biased", [False, True])
def test_mf_failed_refit_preserves_factors_metadata_and_full_objective(failure, biased, monkeypatch, fit_mf):
    import rehline._mf_class as module

    X = np.array([(u, i) for u in range(3) for i in range(4)])
    y = np.random.default_rng(24).normal(size=len(X))
    weight = np.linspace(0.1, 2, len(X))
    model = plqMF_Ridge(
        3,
        4,
        loss={"name": "MSE"},
        rank=2,
        C=0.3,
        biased=biased,
        random_state=42,
        max_iter=100000,
        tol=1e-9,
        max_iter_CD=5,
    )
    fit_mf(model, X, y, sample_weight=weight)
    params, before = model.get_params().copy(), fitted_state(model)
    prediction = model.decision_function(X).copy()
    value = model.obj(X, y, sample_weight=weight)
    changes = dict(n_users=4, n_items=5, biased=not biased, C=2, rho=0.2)
    kwargs = {}
    if failure == "loss":
        changes["loss"] = {"name": "invalid"}
    elif failure == "constraints":
        changes["constraint_item"] = [{"name": "invalid"}]
    elif failure == "weights":
        kwargs["sample_weight"] = -1
    elif failure == "solver":
        original, calls = module.ReHLine_solver, []

        def fail_late(**kw):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("later MF block failed")
            return original(**kw)

        monkeypatch.setattr(module, "ReHLine_solver", fail_late)
    else:
        changes.update(max_iter=1, tol=1e-15)
    model.set_params(**changes)
    with pytest.raises((ValueError, RuntimeError, ConvergenceWarning)):
        model.fit(X, y, **kwargs)
    assert fitted_state(model) == before
    np.testing.assert_array_equal(model.decision_function(X), prediction)
    np.testing.assert_array_equal(model.obj(X, y, sample_weight=weight), value)
    with pytest.raises(ValueError, match="User IDs"):
        model.decision_function([[3, 0]])  # Old fitted bounds remain effective.
    monkeypatch.undo()
    fit_mf(model.set_params(**params), X, y, sample_weight=weight)
    cold = fit_mf(clone(model), X, y, sample_weight=weight)
    np.testing.assert_allclose(model.objective_, cold.objective_, rtol=1e-10, atol=1e-10)
    prediction = model.decision_function(X)
    penalty = model.rho / 3 * (model.P**2).sum() + (1 - model.rho) / 4 * (model.Q**2).sum()
    if biased:
        penalty += model.rho / 3 * (model.bu**2).sum() + (1 - model.rho) / 4 * (model.bi**2).sum()
    expected = model.C * (weight @ (y - prediction) ** 2) + penalty
    assert model.objective_ == pytest.approx(expected, rel=1e-10, abs=1e-10)
