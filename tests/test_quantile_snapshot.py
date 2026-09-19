"""Fitted quantile labels remain tied to the objective that produced the model."""

import pickle

import numpy as np
import pytest
from sklearn.base import clone

from rehline import CQR_Ridge


def objective(model, X, y, weights, quantiles):
    residual = y[:, None] - model.predict(X)
    loss = np.maximum(quantiles * residual, (quantiles - 1) * residual)
    return model.C * np.sum(weights[:, None] * loss) + 0.5 * (
        model.coef_ @ model.coef_ + model.intercept_ @ model.intercept_
    )


@pytest.mark.parametrize("layout", ["contiguous", "strided", "readonly", "list"])
@pytest.mark.parametrize("warm_start", [False, True])
def test_quantile_snapshot_is_independent_and_refits_use_new_parameters(layout, warm_start):
    rng = np.random.default_rng(71)
    X, y, weights = rng.normal(size=(25, 3)), rng.normal(size=25), rng.uniform(0.1, 2, 25)
    original = np.array([0.8, 0.2, 0.5, 0.5])
    levels = np.repeat(original, 2)[::2] if layout == "strided" else original.copy()
    if layout == "readonly":
        levels.setflags(write=False)
    elif layout == "list":
        levels = levels.tolist()
    model = CQR_Ridge(levels, C=0.1, tol=1e-10, max_iter=100000, warm_start=warm_start).fit(X, y, sample_weight=weights)
    assert model.quantiles is levels  # Constructor semantics are unchanged.
    assert not np.shares_memory(model.quantiles_, np.asarray(levels))
    before = model.predict(X).copy()
    value = objective(model, X, y, weights, original)
    assert value == pytest.approx(model.objective_, rel=1e-8, abs=1e-9)
    replacement = [0.1, 0.9, 0.3, 0.7]
    if layout == "readonly":
        model.set_params(quantiles=np.array(replacement))
    else:
        levels[:] = replacement
    np.testing.assert_array_equal(model.quantiles_, original)
    np.testing.assert_array_equal(model.predict(X), before)
    with pytest.raises(ValueError):
        model.fit(X, y, sample_weight=-1)
    np.testing.assert_array_equal(model.quantiles_, original)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.quantiles_, original)
    np.testing.assert_array_equal(restored.predict(X), before)
    model.fit(X, y, sample_weight=weights)
    np.testing.assert_array_equal(model.quantiles_, replacement)
    assert not np.shares_memory(model.quantiles_, np.asarray(model.quantiles))
    cold = clone(model).set_params(warm_start=False).fit(X, y, sample_weight=weights)
    new_value = objective(model, X, y, weights, np.array(replacement))
    assert new_value == pytest.approx(cold.objective_, rel=1e-8, abs=1e-9)
    assert new_value == pytest.approx(model.dual_objective_, rel=1e-8, abs=1e-9)
