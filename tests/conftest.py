"""Objective acceptance and helpers for deliberately short MF outer budgets."""

import warnings
from functools import partial

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning


def pytest_collection_modifyitems(items):
    # sklearn compares individual predictions across independently optimized
    # fits here. Keep these strict checks in the stress suite; ordinary tests
    # still check weighting semantics against independent objective values.
    output_checks = {
        "check_sample_weight_equivalence_on_dense_data",
        "check_sample_weight_equivalence_on_sparse_data",
        "check_sample_weights_invariance",  # Older supported sklearn releases.
    }
    for item in items:
        if getattr(item, "originalname", None) != "test_sklearn_estimator_contract":
            continue
        check = item.callspec.params["check"]
        while isinstance(check, partial):
            check = check.func
        if check.__name__ in output_checks:
            item.add_marker(pytest.mark.numerical_stress)


@pytest.fixture
def assert_objective_close():
    """Compare optimization values using a scale-aware 1e-8 acceptance bound."""

    def check(actual, reference):
        actual, reference = np.broadcast_arrays(np.asarray(actual), np.asarray(reference))
        assert np.isfinite(actual).all() and np.isfinite(reference).all()
        error = abs(actual - reference) / np.maximum(1, abs(reference))
        assert np.all(error <= 1e-8), f"Normalized objective error {error} exceeds 1e-8"

    return check


@pytest.fixture
def fit_mf_objective(monkeypatch, assert_objective_close, record_property):
    """Certify convex MF blocks by independently recomputed primal/dual bounds."""
    from benchmarks.common.objectives import audit_solver_result

    import rehline._mf_class as mf

    def checked(model, X, y, **kwargs):
        original = mf.ReHLine_solver
        diagnostics = []

        def audited(**problem):
            result = original(**problem)
            report = audit_solver_result(problem, result)
            assert_objective_close(report["feasible_upper_bound"], report["dual_lower_bound"])
            diagnostics.append((result.converged, result.kkt_residual))
            return result

        with monkeypatch.context() as context, warnings.catch_warnings(record=True) as caught:
            context.setattr(mf, "ReHLine_solver", audited)
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, y, **kwargs)
        assert diagnostics
        assert model.scaled_constraint_violation_ <= 1e-8
        assert all(
            str(warning.message).startswith(("ReHLine failed to converge", "MF outer iterations failed"))
            for warning in caught
        )
        record_property("mf_blocks_without_full_kkt_certificate", sum(not converged for converged, _ in diagnostics))
        record_property("mf_max_inner_kkt", max(kkt for _, kkt in diagnostics))
        return model

    return checked


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
