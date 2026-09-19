"""Independent regressions for target dtypes, constant covariance and sparse data."""

import tracemalloc

import numpy as np
import pytest

from rehline import make_mf_dataset, plq_Ridge_Regressor


def test_unsigned_mae_matches_analytic_objective():
    model = plq_Ridge_Regressor(loss={"name": "MAE"}, fit_intercept=False, tol=1e-10, max_iter=100000)
    model.fit(np.ones((3, 1)), np.array([1, 2, 3], dtype=np.uint8))
    assert model.converged_
    assert model.objective_ == pytest.approx(3.5, abs=1e-9)
    np.testing.assert_allclose(model.coef_, [1.0], atol=1e-9)


def test_constant_decimal_sensitive_feature_imposes_no_constraint():
    X, y = np.full((100, 1), 0.1), np.ones(100)
    model = plq_Ridge_Regressor(
        loss={"name": "MSE"},
        fit_intercept=False,
        tol=1e-10,
        max_iter=100000,
        constraint=[{"name": "fair", "sen_idx": [0], "tol_sen": 0.0}],
    ).fit(X, y)
    assert model.converged_
    assert model.objective_ == pytest.approx(100 / 3, abs=1e-8)
    np.testing.assert_array_equal(model._A, 0)


def test_sparse_mf_generator_memory_does_not_track_all_pairs():
    tracemalloc.start()
    try:
        ratings = make_mf_dataset(1000, 10000, n_factors=2, n_interactions=10, seed=42)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(ratings["y"]) == 10
    assert peak < 8 * 1024**2
