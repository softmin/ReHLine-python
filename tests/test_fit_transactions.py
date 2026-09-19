"""Failed fits retain coherent state; named losses reject ignored parameters."""

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning, NotFittedError

from rehline import (
    ReHLine,
    plq_ElasticNet_Classifier,
    plq_ElasticNet_Regressor,
    plq_Ridge_Classifier,
    plq_Ridge_Regressor,
    plqERM_ElasticNet,
    plqERM_Ridge,
)

WRAPPERS = (plq_Ridge_Classifier, plq_ElasticNet_Classifier, plq_Ridge_Regressor, plq_ElasticNet_Regressor)
NAMED = (*WRAPPERS, plqERM_Ridge, plqERM_ElasticNet)


def options():
    return dict(loss={"name": "MSE"}, C=0.1, tol=1e-10, max_iter=100000, warm_start=True)


def state(model):
    return pickle.dumps({k: v for k, v in vars(model).items() if k.endswith("_") or k.startswith("_")})


@pytest.mark.parametrize("estimator", WRAPPERS)
@pytest.mark.parametrize("failure", ["constraints", "loss", "weights", "manual", "solver", "warning"])
def test_failed_refit_preserves_state_and_allows_recovery(estimator, failure, monkeypatch):
    rng = np.random.default_rng(721)
    X = rng.normal(size=(40, 3))
    classifier = "Classifier" in estimator.__name__
    y = np.tile(["a", "b", "c", "d"], 10) if classifier else rng.normal(size=40)
    model = estimator(**options(), **(dict(multi_class="ovo", n_jobs=2) if classifier else {})).fit(X, y)
    before, predictions = state(model), model.predict(X)
    new_y = np.char.add(y, "-new") if classifier else y + 3
    new_X = np.column_stack((X, X[:, 0]))
    params, fit_params = {}, {}
    if failure == "constraints":
        params["constraint"] = [{"name": "invalid"}]
    elif failure == "loss":
        params["loss"] = {"name": "invalid"}
    elif failure == "weights":
        fit_params["sample_weight"] = -np.ones(len(y))
    elif failure == "manual":
        params["U"] = [[np.nan]]
    elif failure == "solver":
        import rehline._class as native

        original = native.ReHLine_solver

        def fail_one_task(**kwargs):
            # Some workers finish their newly allocated submodels before one
            # worker fails; old model and warm dual state must remain intact.
            if not classifier or np.any(kwargs["X"][:, 0] == new_X[0, 0]):
                raise RuntimeError("worker failed")
            return original(**kwargs)

        monkeypatch.setattr(native, "ReHLine_solver", fail_one_task)
    else:
        params.update(max_iter=1, tol=1e-15)
    old_params = model.get_params().copy()
    model.set_params(**params)
    with pytest.raises((ValueError, RuntimeError, ConvergenceWarning)):
        model.fit(new_X, new_y, **fit_params)
    assert state(model) == before
    np.testing.assert_array_equal(model.predict(X), predictions)
    assert np.all(model.converged_)
    monkeypatch.undo()
    model.set_params(**old_params).fit(X, y)
    cold = clone(model).set_params(warm_start=False).fit(X, y)
    np.testing.assert_allclose(model.objective_, cold.objective_, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("estimator", WRAPPERS)
def test_first_failed_fit_does_not_claim_fitted(estimator):
    model = estimator(**options(), constraint=[{"name": "invalid"}])
    with pytest.raises(ValueError):
        model.fit(np.ones((4, 2)), np.arange(4))
    with pytest.raises(NotFittedError):
        model.predict(np.ones((4, 2)))
    assert not hasattr(model, "classes_") and not hasattr(model, "n_features_in_")


@pytest.mark.parametrize("estimator", [plq_Ridge_Classifier, plq_ElasticNet_Classifier])
@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
def test_failed_refit_preserves_feature_names_and_fitted_strategy(estimator, strategy):
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame(np.random.default_rng(8).normal(size=(30, 2)), columns=["first", "second"])
    y = np.tile(["a", "b", "c"], 10)
    model = estimator(**options(), multi_class=strategy).fit(X, y)
    before, prediction = state(model), model.predict(X)
    # Fail after new feature names, labels and strategy have been prepared.
    model.set_params(multi_class="ovo" if strategy == "ovr" else "ovr", loss={"name": "invalid"})
    with pytest.raises(ValueError):
        model.fit(X.rename(columns={"first": "new_first"}), np.tile(["new_a", "new_b"], 15))
    assert state(model) == before
    assert model.multi_class_ == strategy
    np.testing.assert_array_equal(model.predict(X), prediction)
    model.set_params(loss={"name": "MSE"}).fit(X, y)
    assert model.multi_class_ != strategy
    cold = clone(model).set_params(warm_start=False).fit(X, y)
    np.testing.assert_allclose(model.objective_, cold.objective_, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("estimator", NAMED)
@pytest.mark.parametrize("name", ["U", "V", "S", "T", "Tau"])
@pytest.mark.parametrize("value", [[[np.nan]], "invalid", 0.0])
def test_manual_loss_parameters_rejected_through_constructor_clone_and_set_params(estimator, name, value):
    model = estimator(**options(), **{name: value})
    for candidate in (model, clone(model), estimator(**options()).set_params(**{name: value})):
        with pytest.raises(ValueError, match=rf"{name}.*ReHLine / ReHLine_solver"):
            candidate.fit(np.ones((4, 1)), np.array([0.0, 0.0, 1.0, 1.0]))


@pytest.mark.parametrize("estimator", NAMED)
def test_legacy_empty_loss_parameters_are_harmless(estimator):
    supplied = dict(U=[], V=np.empty((0, 0)), S=None, T=np.empty((2, 0)), Tau=[])
    model = clone(estimator(**options(), **supplied)).fit(np.ones((4, 1)), np.array([0.0, 0.0, 1.0, 1.0]))
    reference = estimator(**options()).fit(np.ones((4, 1)), np.array([0.0, 0.0, 1.0, 1.0]))
    np.testing.assert_array_equal(model.coef_, reference.coef_)
    np.testing.assert_array_equal(model.objective_, reference.objective_)


def test_raw_manual_loss_api_remains_available():
    model = ReHLine(U=-np.ones((1, 4)), V=np.full((1, 4), 2.0), tol=1e-10).fit(np.ones((4, 1)))
    np.testing.assert_allclose(model.coef_, [2.0], atol=1e-9)
    assert model.objective_ == pytest.approx(2.0, abs=1e-9)
