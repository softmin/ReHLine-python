"""Sparse designs retain full objective certificates without dense X storage."""

import pickle

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import get_tags

from rehline import (
    CQR_Ridge,
    CQR_Ridge_path_sol,
    ReHLine,
    ReHLine_solver,
    _make_constraint_rehline_param,
    plq_ElasticNet_Classifier,
    plq_ElasticNet_Regressor,
    plq_Ridge_Classifier,
    plq_Ridge_Regressor,
    plqERM_ElasticNet,
    plqERM_Ridge,
    plqERM_Ridge_path_sol,
    plqMF_Ridge,
)
from tests._helpers.core import make_case, solve_reference
from tests._helpers.numerical_stress import measure
from tests._helpers.problems import routine_problem

OPTIONS = dict(tol=1e-8, max_iter=1000000, verbose=0)
FORMATS = [
    sparse.csr_matrix,
    sparse.csc_matrix,
    sparse.coo_matrix,
    sparse.csr_array,
    sparse.csc_array,
    sparse.coo_array,
]


def data():
    rng = np.random.default_rng(316)
    X = rng.normal(size=(48, 7))
    X[rng.random(X.shape) < 0.7] = 0
    X[0] = 0
    X[:, -1] = 0
    y = X @ np.arange(7) / 3 + rng.normal(size=len(X))
    weight = rng.uniform(0.2, 1.5, len(X))
    weight[::7] = 0
    return X, y, weight


def original_objective(model, X, y, weight, loss):
    scores = X @ model.coef_
    beta = model.coef_
    if getattr(model, "fit_intercept", False):
        scores += model.intercept_
        beta = np.r_[beta, model.intercept_ / model.intercept_scaling]
    residual = y - scores
    losses = np.maximum(1 - y * scores, 0) if loss == "svm" else np.maximum(0.25 * residual, -0.75 * residual)
    ratio = getattr(model, "l1_ratio", 0.0)
    return model.C * (weight @ losses) + (1 - ratio) * (beta @ beta) / 2 + ratio * np.abs(beta).sum()


def check_native(p, result, reference):
    # Reconstruct the dual bound from the original dense problem independently
    # of the native sparse products and reported diagnostics.
    beta = p["A"].T @ result.xi
    beta -= p["X"].T @ (p["U"] * result.Lambda).sum(axis=0)
    beta -= p["X"].T @ (p["S"] * result.Gamma).sum(axis=0)
    if p["rho"].size:
        beta += 2 * result.mu - p["rho"]
    dual = -0.5 * (beta @ beta) - p["b"] @ result.xi
    dual += (p["V"] * result.Lambda).sum() + (p["T"] * result.Gamma).sum() - 0.5 * np.square(result.Gamma).sum()
    np.testing.assert_allclose(dual, result.dual_objective, rtol=1e-12, atol=1e-12)
    record = measure(p, result, reference, tol=OPTIONS["tol"], max_iter=OPTIONS["max_iter"])
    assert result.converged and record["status"] == "met", record


@pytest.mark.parametrize(
    "loss,estimators",
    [
        ("svm", [plq_Ridge_Classifier, plq_ElasticNet_Classifier]),
        ("quantile", [plq_Ridge_Regressor, plq_ElasticNet_Regressor]),
    ],
)
@pytest.mark.parametrize("format_", FORMATS)
@pytest.mark.parametrize("intercept", [False, True])
def test_weighted_svm_qr_fit_predict_and_warm_refit(loss, estimators, format_, intercept, assert_objective_close):
    X, y, weight = data()
    if loss == "svm":
        y = np.where(y > 0, 1, -1)
    for estimator in estimators:
        options = dict(
            **OPTIONS,
            loss={"name": loss, "qt": 0.25},
            C=0.2,
            fit_intercept=intercept,
            intercept_scaling=2.5,
            warm_start=True,
        )
        model = estimator(**options)
        dense = clone(model).fit(X, y, sample_weight=weight)
        for design in [format_(X), X, format_(X)]:
            model.fit(design, y, sample_weight=weight)
            value = original_objective(model, X, y, weight, loss)
            assert_objective_close(value, original_objective(dense, X, y, weight, loss))
            assert model.converged_ and model.scaled_constraint_violation_ <= 1e-8
            scale = 1 - getattr(model, "l1_ratio", 0.0)
            assert_objective_close(value, scale * model.objective_)
            assert_objective_close(value, scale * model.dual_objective_)
            prediction = model.predict(format_(X))
            np.testing.assert_allclose(prediction, model.predict(X), atol=1e-12)
            snapshot = pickle.loads(pickle.dumps(model.to_inference()))
            np.testing.assert_allclose(snapshot.predict(format_(X)), prediction, atol=1e-12)
            assert get_tags(model).input_tags.sparse
            assert get_tags(snapshot).input_tags.sparse


@pytest.mark.parametrize("index", range(12))
@pytest.mark.parametrize("shrink,order", [(0, "cyclic"), (0, "random"), (1, "cyclic"), (1, "random")])
def test_all_loss_families_with_dense_constraints_and_dual_warm_starts(index, shrink, order, assert_objective_close):
    case = make_case(index + 36, max_samples=20, max_dim=5, profile="routine")
    X = case["X"].copy()
    X[np.random.default_rng(index).random(X.shape) < 0.5] = 0
    X[0] = 0
    case["X"] = X
    p = routine_problem(case)
    options = dict(**OPTIONS, shrink=shrink, coordinate_order=order, coordinate_seed=8)
    dense = ReHLine_solver(**p, **options)
    result = ReHLine_solver(**dict(p, X=sparse.csr_matrix(X)), **options)
    for fit in [
        result,
        ReHLine_solver(
            **dict(p, X=sparse.csc_matrix(X)),
            **options,
            **{name: getattr(result, name) for name in ("Lambda", "Gamma", "xi", "mu")},
        ),
    ]:
        check_native(p, fit, dense.objective)


@pytest.mark.parametrize("index", range(12))
def test_sparse_losses_match_independent_cvxpy(index, assert_objective_close):
    pytest.importorskip("cvxpy")
    case = make_case(index + 60, max_samples=20, max_dim=5, profile="routine")
    case["X"][np.random.default_rng(index).random(case["X"].shape) < 0.6] = 0
    reference = solve_reference(case)["objective"] / (1 - case["l1_ratio"])
    p = routine_problem(case)
    result = ReHLine_solver(**dict(p, X=sparse.csr_matrix(case["X"])), **OPTIONS)
    check_native(p, result, reference)


@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
@pytest.mark.parametrize("estimator", [plq_Ridge_Classifier, plq_ElasticNet_Classifier])
def test_multiclass_weighting_threads_and_sparse_prediction(strategy, estimator, assert_objective_close):
    X, _, weight = data()
    y = np.tile(["a", "b", "c"], 16)
    model = estimator(
        **OPTIONS, loss={"name": "svm"}, C=0.1, multi_class=strategy, n_jobs=2, class_weight="balanced", warm_start=True
    )
    dense = clone(model).fit(X, y, sample_weight=weight)
    for _ in range(2):
        model.fit(sparse.csr_matrix(X), y, sample_weight=weight)
        assert np.all(model.converged_)
        assert_objective_close(model.objective_, dense.objective_)
        for format_ in FORMATS:
            np.testing.assert_array_equal(model.predict(format_(X)), model.predict(X))
            np.testing.assert_allclose(model.decision_function(format_(X)), model.decision_function(X), atol=1e-12)
            np.testing.assert_array_equal(model.to_inference().predict(format_(X)), model.predict(X))


@pytest.mark.parametrize("format_", FORMATS)
def test_cqr_implicit_sparse_design_and_path(format_, assert_objective_close):
    X, y, weight = data()
    levels = np.array([0.25, 0.75, 0.25])
    model = CQR_Ridge(**OPTIONS, quantiles=levels, C=0.2, warm_start=True)
    dense = clone(model).fit(X, y, sample_weight=weight)
    for _ in range(2):
        model.fit(format_(X), y, sample_weight=weight)
        residual = y[:, None] - model.predict(format_(X))
        value = model.C * (weight @ np.maximum(levels * residual, (levels - 1) * residual).sum(axis=1))
        value += (model.coef_ @ model.coef_ + model.intercept_ @ model.intercept_) / 2
        assert model.converged_ and get_tags(model).input_tags.sparse
        assert_objective_close(value, dense.objective_)
        assert_objective_close(value, model.dual_objective_)
        np.testing.assert_allclose(model.to_inference().predict(format_(X)), model.predict(X), atol=1e-12)
    _, models, _, _ = CQR_Ridge_path_sol(
        format_(X), y, quantiles=levels, Cs=[0.03, 0.2], **OPTIONS, warm_start=True, compact=True, return_time=False
    )
    for C, fitted in zip([0.03, 0.2], models):
        reference = CQR_Ridge(**OPTIONS, quantiles=levels, C=C).fit(X, y)
        assert_objective_close(fitted.objective_, reference.objective_)
        np.testing.assert_allclose(fitted.predict(format_(X)), fitted.predict(X), atol=1e-12)


@pytest.mark.parametrize("estimator", [ReHLine, plqERM_Ridge, plqERM_ElasticNet])
def test_raw_estimators_predict_sparse(estimator, assert_objective_close):
    X, y, _ = data()
    options = (
        {"U": np.array([[-0.25] * len(y), [0.75] * len(y)]), "V": np.array([0.25 * y, -0.75 * y])}
        if estimator is ReHLine
        else {"loss": {"name": "quantile", "qt": 0.25}}
    )
    model = estimator(**OPTIONS, **options, C=0.2)
    dense = clone(model).fit(X, **({} if estimator is ReHLine else {"y": y}))
    model.fit(sparse.csr_matrix(X), **({} if estimator is ReHLine else {"y": y}))
    assert_objective_close(model.objective_, dense.objective_)
    np.testing.assert_allclose(model.decision_function(sparse.csc_matrix(X)), X @ model.coef_, atol=1e-12)
    assert get_tags(model).input_tags.sparse
    assert not get_tags(plqMF_Ridge(2, 2, loss={"name": "MSE"})).input_tags.sparse


def test_svm_regularization_path_and_sklearn_grid_search(assert_objective_close):
    X, y, _ = data()
    y = np.where(y > 0, 1, -1)
    Cs = [0.01, 0.1, 1.0]
    _, _, values, _, coefs = plqERM_Ridge_path_sol(
        sparse.csr_matrix(X), y, loss={"name": "svm"}, Cs=Cs, **OPTIONS, warm_start=True, return_time=False
    )
    for j, C in enumerate(Cs):
        reference = plqERM_Ridge(**OPTIONS, loss={"name": "svm"}, C=C).fit(X, y)
        value = C * np.maximum(1 - y * (X @ coefs[:, j]), 0).sum() + coefs[:, j] @ coefs[:, j] / 2
        assert_objective_close(value, reference.objective_)
        assert_objective_close(value, values[j])
    pipeline = make_pipeline(StandardScaler(with_mean=False), plq_Ridge_Classifier(**OPTIONS, loss={"name": "svm"}))
    search = GridSearchCV(pipeline, {"plq_ridge_classifier__C": Cs}, cv=3, error_score="raise").fit(
        sparse.csr_matrix(X), y
    )
    assert np.isfinite(search.cv_results_["mean_test_score"]).all()


@pytest.mark.parametrize("format_", [sparse.csr_matrix, sparse.csc_array])
@pytest.mark.parametrize("variant", ["ordinary", "constant", "large_offset", "tiny"])
def test_sparse_fairness_preserves_centering_and_exact_constants(format_, variant):
    X, _, _ = data()
    if variant == "constant":
        X[:, 0], X[:, -1] = 0.1, -0.3
    elif variant == "large_offset":
        X = np.round(X * 4) / 4 + 1e12
    elif variant == "tiny":
        X *= 1e-12
    indices = [0, 1, -1]
    constraint = [{"name": "fair", "sen_idx": indices, "tol_sen": [0, 0, 0]}]
    actual, _ = _make_constraint_rehline_param(constraint, format_(X))
    differences = X[:, None, :] - X[None, :, :]
    covariance = np.einsum("ijk,ijl->kl", differences[:, :, indices], differences) / (2 * len(X) ** 2)
    np.testing.assert_allclose(actual[1::2], covariance, rtol=1e-13, atol=1e-50)
    if variant == "constant":
        np.testing.assert_array_equal(actual[:, [0, -1]], 0)
        np.testing.assert_array_equal(actual[[0, 1, 4, 5]], 0)


@pytest.mark.parametrize(
    "constraint",
    [
        [{"name": "fair", "sen_idx": [0], "tol_sen": 0.03}],
        [{"name": "nonnegative"}],
        [{"name": "monotonic"}],
        [{"name": "custom", "A": np.eye(8), "b": np.ones(8) * 0.2}],
    ],
)
def test_sparse_estimator_constraints_include_intercept_and_original_objective(constraint, assert_objective_close):
    X, y, weight = data()
    model = plq_ElasticNet_Regressor(
        **OPTIONS, loss={"name": "quantile", "qt": 0.25}, C=0.2, constraint=constraint, intercept_scaling=3.0
    )
    dense = clone(model).fit(X, y, sample_weight=weight)
    model.fit(sparse.csr_matrix(X), y, sample_weight=weight)
    assert model.converged_ and model.scaled_constraint_violation_ <= 1e-8
    assert_objective_close(
        original_objective(model, X, y, weight, "quantile"), original_objective(dense, X, y, weight, "quantile")
    )


def test_duplicate_unsorted_int64_and_readonly_input_are_not_mutated(assert_objective_close):
    X = sparse.csr_matrix(
        (np.array([0.25, 1.0, 0.75, 0.0, 2.0, -1.0]), np.array([2, 0, 2, 1, 1, 0]), np.array([0, 4, 6, 6])),
        shape=(3, 4),
    )
    X.indices = X.indices.astype(np.int64)
    X.indptr = X.indptr.astype(np.int64)
    original = [(a.copy(), a.dtype) for a in (X.data, X.indices, X.indptr)]
    dense = X.toarray()
    for a in (X.data, X.indices, X.indptr):
        a.setflags(write=False)
    model = plq_Ridge_Regressor(**OPTIONS, loss={"name": "quantile", "qt": 0.25}, C=0.2)
    reference = clone(model).fit(dense, [1.0, 2.0, -1.0])
    model.fit(X, [1.0, 2.0, -1.0])
    assert_objective_close(model.objective_, reference.objective_)
    for a, (old, dtype) in zip((X.data, X.indices, X.indptr), original):
        np.testing.assert_array_equal(a, old)
        assert a.dtype == dtype and not a.flags.writeable
    assert not X.has_canonical_format


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1j])
def test_invalid_sparse_values_reject_before_replacing_fitted_state(value):
    X, y, _ = data()
    model = plq_Ridge_Regressor(**OPTIONS, loss={"name": "MSE"}, C=0.1).fit(sparse.csr_matrix(X), y)
    previous = model.predict(X)
    broken = sparse.csr_matrix(X.astype(complex if value == 1j else float))
    broken.data[0] = value
    with pytest.raises(ValueError):
        model.fit(broken, y)
    np.testing.assert_array_equal(model.predict(X), previous)


def test_empty_sparse_rows_and_all_zero_design():
    X = sparse.csr_matrix((5, 8))
    result = ReHLine_solver(X, np.ones((1, 5)), np.ones((1, 5)), **OPTIONS)
    np.testing.assert_array_equal(result.beta, 0)
    np.testing.assert_array_equal(result.Lambda, 1)
    assert result.converged and result.objective == result.dual_objective == 5


def test_large_sparse_design_never_materializes_dense_X(monkeypatch):
    # A dense copy would occupy 8 GB; the actual CSR design has only 10k entries.
    n, d = 10000, 100000
    X = sparse.csr_matrix((np.ones(n), (np.arange(n), np.arange(n))), shape=(n, d))
    y = np.where(np.arange(n) % 2, 1.0, -1.0)

    def forbidden(*args, **kwargs):
        raise AssertionError("Sparse X must not be converted to dense")

    for cls in FORMATS:
        monkeypatch.setattr(cls, "toarray", forbidden)
        monkeypatch.setattr(cls, "todense", forbidden)
    model = plq_Ridge_Classifier(**OPTIONS, loss={"name": "svm"}, C=0.1).fit(X, y)
    assert model.converged_
    np.testing.assert_array_equal(model.predict(X), y)
    expected = n * (0.1 - 0.1**2 / 2)
    assert model.objective_ == pytest.approx(expected, rel=1e-8)
    model.to_inference().predict(X)


def test_fairness_and_cqr_never_call_sparse_toarray(monkeypatch):
    X, y, _ = data()
    design = sparse.csr_matrix(X)

    def forbidden(*args, **kwargs):
        raise AssertionError("Do not densify X")

    for cls in (sparse.csr_matrix, sparse.csc_matrix):
        monkeypatch.setattr(cls, "toarray", forbidden)
        monkeypatch.setattr(cls, "todense", forbidden)
    model = plq_Ridge_Regressor(
        **OPTIONS,
        loss={"name": "quantile", "qt": 0.25},
        C=0.1,
        constraint=[{"name": "fair", "sen_idx": [0], "tol_sen": 0.02}],
    ).fit(design, y)
    assert model.converged_ and model.scaled_constraint_violation_ <= 1e-8
    cqr = CQR_Ridge(**OPTIONS, quantiles=[0.25, 0.75], C=0.1).fit(design, y)
    assert cqr.converged_
    assert cqr.to_inference().predict(design).shape == (len(y), 2)


@pytest.mark.parametrize("fault", ["indices", "pointer", "duplicate_overflow", "dimension"])
def test_malformed_or_unrepresentable_csr_is_rejected(fault):
    X = sparse.csr_matrix(np.eye(3))
    if fault == "indices":
        X.indices[0] = 3
    elif fault == "pointer":
        X.indptr[:] = [0, 2, 1, 3]
    elif fault == "duplicate_overflow":
        X = sparse.csr_matrix(([1e308, 1e308], [0, 0], [0, 2, 2, 2]), shape=(3, 3))
    else:
        X = sparse.csr_matrix((1, 2**31))
    with pytest.raises(ValueError):
        ReHLine_solver(X, np.ones((1, X.shape[0])), np.ones((1, X.shape[0])), **OPTIONS)


def test_sparse_A_is_explicitly_rejected():
    with pytest.raises(ValueError, match="A must be dense"):
        ReHLine_solver(sparse.eye(3), np.ones((1, 3)), np.ones((1, 3)), A=sparse.eye(3), b=np.zeros(3), **OPTIONS)


@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.bool_])
def test_sparse_input_is_converted_to_float64(dtype, assert_objective_close):
    X = np.array([[1, 0], [0, 1], [1, 1]], dtype=dtype)
    y = [1.0, -1.0, 1.0]
    model = plq_Ridge_Classifier(**OPTIONS, loss={"name": "svm"}, C=0.2)
    reference = clone(model).fit(X, y)
    model.fit(sparse.csr_array(X), y)
    assert model.coef_.dtype == np.float64
    assert_objective_close(model.objective_, reference.objective_)


@pytest.mark.parametrize("composite", [False, True])
def test_native_sparse_entry_validates_finite_values(composite):
    from rehline import _internal

    n, d, q = 3, 2, 2 if composite else 0
    X = sparse.csr_matrix(np.eye(n, d))
    X.data[0] = np.inf
    count = n * max(1, q)
    args = [
        X,
        np.empty((0, d + q)),
        np.empty(0),
        np.empty(0),
        np.ones((1, count)),
        np.ones((1, count)),
        *(np.empty((0, count)) for _ in range(3)),
    ]
    if composite:
        args.append(q)
    native = _internal.rehline_cqr_sparse_internal if composite else _internal.rehline_sparse_internal
    with pytest.raises(ValueError, match="finite"):
        native(_internal.rehline_result(), *args, 1000, 1e-8)
