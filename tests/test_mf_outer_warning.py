"""Outer-budget warnings report state and preserve failed-fit transactions."""

import pickle
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.utils.validation import check_is_fitted

from rehline import plqMF_Ridge


def model(**kwargs):
    return plqMF_Ridge(
        2, 2, loss={"name": "MSE"}, rank=1, C=0.3, random_state=42, tol=1e-10, max_iter=100000, tol_CD=1e-10, **kwargs
    )


def test_outer_warning_once_and_no_warning_on_convergence():
    X, y = [[0, 0], [1, 1]], [1.0, 2.0]
    short = model(max_iter_CD=1)
    with pytest.warns(ConvergenceWarning, match="max_iter_CD") as caught:
        short.fit(X, y)
    assert len(caught) == 1
    assert short.inner_converged_ and not short.converged_
    assert short.n_iter_ == 1
    np.testing.assert_allclose(short.obj(X, y)[1], short.objective_)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        finished = model(max_iter_CD=2000).fit(X, y)
    assert finished.converged_
    assert not caught


def test_outer_warning_as_error_preserves_fitted_state_and_failed_first_fit():
    X, y = [[0, 0], [1, 1]], [1.0, 2.0]
    fitted = model(max_iter_CD=2000).fit(X, y)
    fitted.set_params(max_iter_CD=1)
    before = pickle.dumps(vars(fitted))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        with pytest.raises(ConvergenceWarning, match="max_iter_CD"):
            fitted.fit(X, y)
        first = model(max_iter_CD=1)
        with pytest.raises(ConvergenceWarning, match="max_iter_CD"):
            first.fit(X, y)
    assert pickle.dumps(vars(fitted)) == before
    with pytest.raises(NotFittedError):
        check_is_fitted(first)
