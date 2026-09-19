"""Helpers for tests that deliberately use a short MF outer budget."""

import warnings

import pytest
from sklearn.exceptions import ConvergenceWarning


@pytest.fixture
def fit_mf():
    def checked(model, X, y, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.filterwarnings("always", message="MF outer iterations failed", category=ConvergenceWarning)
            model.fit(X, y, **kwargs)
        assert model.inner_converged_
        assert model.scaled_constraint_violation_ <= model.tol
        assert len(caught) == int(not model.converged_)
        if caught:
            assert caught[0].category is ConvergenceWarning
            assert "max_iter_CD" in str(caught[0].message)
            assert model.n_iter_ == model.max_iter_CD
        return model

    return checked
