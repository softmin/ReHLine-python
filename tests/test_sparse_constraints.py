"""Sparse A: original objective/dual certificates, feasibility, and bounded storage."""

import pickle

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GridSearchCV

from rehline import (
    ReHLine,
    ReHLine_solver,
    _make_constraint_rehline_param,
    plq_ElasticNet_Classifier,
    plq_ElasticNet_Regressor,
    plq_Ridge_Regressor,
    plqERM_Ridge,
    plqERM_Ridge_path_sol,
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


def audit(p, result, reference):
    # Always use original dense arrays: this audit must not reuse the sparse
    # implementation, row normalization, or its reported objective/feasibility.
    beta = p["A"].T @ result.xi
    beta -= p["X"].T @ (p["U"] * result.Lambda).sum(axis=0)
    beta -= p["X"].T @ (p["S"] * result.Gamma).sum(axis=0)
    if p["rho"].size:
        beta += 2 * result.mu - p["rho"]
    dual = -0.5 * (beta @ beta) - p["b"] @ result.xi
    dual += (p["V"] * result.Lambda).sum() + (p["T"] * result.Gamma).sum() - 0.5 * np.square(result.Gamma).sum()
    np.testing.assert_allclose(dual, result.dual_objective, rtol=1e-12, atol=1e-11)
    record = measure(p, result, reference, tol=OPTIONS["tol"], max_iter=OPTIONS["max_iter"])
    assert result.converged and record["status"] == "met", record


@pytest.mark.parametrize("index", range(12, 72))
@pytest.mark.parametrize("shrink,order", [(0, "cyclic"), (0, "random"), (1, "cyclic"), (1, "random")])
def test_all_losses_and_constraint_geometries(index, shrink, order):
    case = make_case(index, max_samples=20, max_dim=5, profile="routine")
    case["X"][np.random.default_rng(index).random(case["X"].shape) < 0.6] = 0
    p = routine_problem(case)
    options = dict(**OPTIONS, shrink=shrink, coordinate_order=order, coordinate_seed=8)
    dense = ReHLine_solver(**p, **options)
    for X in (p["X"], sparse.csr_matrix(p["X"])):
        sparse_problem = dict(p, X=X, A=sparse.csr_matrix(p["A"]))
        result = ReHLine_solver(**sparse_problem, **options)
        audit(p, result, dense.objective)
        warm = ReHLine_solver(
            **sparse_problem,
            **options,
            **{name: getattr(result, name) for name in ("xi", "Lambda", "Gamma", "mu")},
        )
        audit(p, warm, dense.objective)


@pytest.mark.parametrize("index", range(12, 72))
def test_constraints_match_independent_cvxpy(index):
    pytest.importorskip("cvxpy")
    case = make_case(index, seed=20260930, max_samples=20, max_dim=5, profile="routine")
    case["X"][np.random.default_rng(index).random(case["X"].shape) < 0.5] = 0
    p = routine_problem(case)
    reference = solve_reference(case)["objective"] / (1 - case["l1_ratio"])
    for X in (p["X"], sparse.csr_matrix(p["X"])):
        result = ReHLine_solver(**dict(p, X=X, A=sparse.csr_matrix(p["A"])), **OPTIONS)
        audit(p, result, reference)


@pytest.mark.parametrize("format_", FORMATS)
@pytest.mark.parametrize("shrink", [0, 1])
def test_extreme_scales_redundant_rows_and_warm_duals(format_, shrink):
    # min ||beta||^2/2, beta[0]>=1, beta[0]/2+beta[1]>=2;
    # optimum [1, 1.5], original multipliers [0.25, 1.5].
    rows = np.array([1, 0, 1, 2])
    A = np.array([[1.0, 0], [0.5, 1.0], [0.0, 0.0]])[rows]
    b = np.array([-1.0, -2.0, 0.0])[rows]
    scales = np.array([1e-200, 1e200, 1e100, 1.0])
    xi = np.array([0.75, 0.25, 0.75, 0.0]) / scales
    for warm in (None, xi):
        result = ReHLine_solver(
            X=np.zeros((1, 2)),
            U=None,
            V=None,
            A=format_(A * scales[:, None]),
            b=b * scales,
            xi=warm,
            shrink=shrink,
            **OPTIONS,
        )
        assert result.converged and result.scaled_constraint_violation <= 1e-8
        np.testing.assert_allclose(result.objective, 1.625, rtol=1e-8)
        np.testing.assert_allclose(result.dual_objective, 1.625, rtol=1e-8)
        assert np.max(-(A @ result.beta + b)) <= 1e-8
        np.testing.assert_allclose(A.T @ (result.xi * scales), result.beta, atol=1e-9)


@pytest.mark.parametrize("format_", FORMATS)
@pytest.mark.parametrize("empty_rows", [0, 3])
def test_empty_constraints_and_explicit_zero_rows(format_, empty_rows):
    A = format_(np.zeros((empty_rows, 3)))
    result = ReHLine_solver(np.zeros((2, 3)), None, None, A=A, b=np.zeros(empty_rows), **OPTIONS)
    assert result.converged and result.objective == 0
    assert result.xi.shape == (empty_rows,)
    if empty_rows:
        with pytest.raises(ValueError, match="zero constraint"):
            ReHLine_solver(np.zeros((2, 3)), None, None, A=A, b=-np.ones(empty_rows), **OPTIONS)


def test_canonicalization_never_mutates_input():
    # Unsorted duplicates and explicit zeros, with read-only int64 buffers.
    A = sparse.csr_matrix(([0.0, 0.5, 0.5, 1.0, -1.0], [1, 0, 0, 1, 1], [0, 3, 5]), shape=(2, 2))
    A.indices = A.indices.astype(np.int64)
    A.indptr = A.indptr.astype(np.int64)
    assert not A.has_canonical_format
    for value in (A.data, A.indices, A.indptr):
        value.flags.writeable = False
    before = pickle.dumps(A)
    result = ReHLine_solver(np.zeros((1, 2)), None, None, A=A, b=[-1.0, 0.0], **OPTIONS)
    assert result.converged and result.objective == 0.5
    np.testing.assert_array_equal(result.beta, [1.0, 0.0])
    assert pickle.dumps(A) == before
    with pytest.raises(ValueError, match="zero constraint"):
        ReHLine_solver(np.zeros((1, 2)), None, None, A=A, b=[-1.0, -1.0], **OPTIONS)


@pytest.mark.parametrize("fault", ["nan", "inf", "complex", "indices", "pointer", "duplicate_overflow", "columns"])
def test_invalid_sparse_constraints_preserve_fitted_state(fault):
    model = plq_Ridge_Regressor(**OPTIONS, loss={"name": "MSE"}, A=sparse.eye(2), b=[0.0, 0.0], warm_start=True)
    X, y = np.eye(2), np.ones(2)
    model.fit(X, y)
    A = sparse.csr_matrix(np.eye(2).astype(complex if fault == "complex" else float))
    if fault in ("nan", "inf", "complex"):
        A.data[0] = {"nan": np.nan, "inf": np.inf, "complex": 1j}[fault]
    elif fault == "indices":
        A.indices[0] = 2
    elif fault == "pointer":
        A.indptr[:] = [0, 2, 1]
    elif fault == "duplicate_overflow":
        A = sparse.csr_matrix(([1e308, 1e308], [0, 0], [0, 2, 2]), shape=(2, 2))
    else:
        A = sparse.csr_matrix((2, 2**31))
    model.set_params(A=A)
    before = pickle.dumps(vars(model))
    with pytest.raises(ValueError):
        model.fit(X, y)
    assert pickle.dumps(vars(model)) == before


@pytest.mark.parametrize("shrink", [0, 1])
def test_feasibility_or_small_kkt_cannot_hide_bad_objective(shrink):
    result = ReHLine_solver(
        np.zeros((1, 2)),
        None,
        None,
        A=sparse.csr_matrix([[1.0, 0.0], [-1.0, 1e-8]]),
        b=[0.0, -1e-8],
        xi=[1.1e8, 1.1e8],
        max_iter=1,
        tol=1e-8,
        shrink=shrink,
        verbose=0,
    )
    assert result.kkt_residual <= 1e-8 and result.scaled_constraint_violation <= 1e-8
    assert not result.converged and result.dual_gap > 0.1


def test_sparse_scaling_overflow_and_infeasibility_diagnostics():
    for b in (-1.0, -1e-320):
        with pytest.raises(OverflowError, match="floating-point range"):
            ReHLine_solver(np.zeros((1, 1)), None, None, A=sparse.csr_matrix([[1e-320]]), b=[b], **OPTIONS)
    # Inconsistent bounds: keep original-unit violation separate from the
    # normalized certificate, and do not report an infeasible primal bound.
    A = sparse.csr_matrix([[1e8], [-1e-8]])
    b = np.array([-1e8, 0.0])
    result = ReHLine_solver(np.zeros((1, 1)), None, None, A=A, b=b, max_iter=1, tol=1e-8, verbose=0)
    slack = A @ result.beta + b
    assert not result.converged and np.isinf(result.dual_gap)
    assert result.constraint_violation == max(0.0, -slack.min())
    assert result.scaled_constraint_violation == pytest.approx(max(0.0, -(slack / [1e8, 1e-8]).min()))


@pytest.mark.parametrize("estimator", [ReHLine, plqERM_Ridge, plq_Ridge_Regressor])
def test_warm_refit_between_sparse_dense_and_scaled_constraints(estimator):
    kwargs = dict(U=None, V=None) if estimator is ReHLine else dict(loss={"name": "MSE"})
    if estimator is plq_Ridge_Regressor:
        kwargs["fit_intercept"] = False
    model = estimator(**OPTIONS, **kwargs, warm_start=True)
    X, y = np.zeros((2, 2)), np.zeros(2)
    fit_args = () if estimator is ReHLine else (y,)
    A, b = np.array([[1.0, 0.0], [0.5, 1.0]]), np.array([-1.0, -2.0])
    for scale, format_ in [(1.0, sparse.csr_matrix), (1e-200, np.asarray), (1e200, sparse.csc_array)]:
        model.set_params(A=format_(A * scale), b=b * scale).fit(X, *fit_args)
        assert model.converged_ and model.objective_ == pytest.approx(1.625, rel=1e-8)
        inner = getattr(model, "model_", model)
        # Public predictions and fitted snapshots stay usable across storage changes.
        assert np.isfinite(inner.coef_).all()
        np.testing.assert_allclose(
            (model.predict(X) if estimator is plq_Ridge_Regressor else model.decision_function(X)),
            X @ model.coef_,
            atol=1e-12,
        )


@pytest.mark.parametrize("format_", FORMATS)
@pytest.mark.parametrize("loss", ["svm", "QR"])
@pytest.mark.parametrize("intercept", [False, True])
def test_weighted_svm_qr_intercepts_and_sparse_constraints(format_, loss, intercept, assert_objective_close):
    rng = np.random.default_rng(602)
    X = rng.normal(size=(32, 4))
    X[rng.random(X.shape) < 0.6] = 0
    y = np.where(np.arange(len(X)) % 2, 1.0, -1.0) if loss == "svm" else rng.normal(size=len(X))
    weight = rng.uniform(0.1, 2, len(X))
    weight[::7] = 0
    d = X.shape[1] + int(intercept)
    A, b = np.vstack((np.eye(d), -np.eye(d))), np.r_[np.full(d, 0.1), np.full(d, 0.4)]
    cls = plq_ElasticNet_Classifier if loss == "svm" else plq_ElasticNet_Regressor
    model = cls(
        **OPTIONS,
        loss={"name": loss, "qt": 0.25},
        C=0.2,
        l1_ratio=0.3,
        fit_intercept=intercept,
        intercept_scaling=2.5,
        A=A,
        b=b,
    )
    dense = clone(model).fit(X, y, sample_weight=weight)
    original = format_(A)
    before = pickle.dumps(original)
    for design in (X, sparse.csr_matrix(X)):
        model.set_params(A=original).fit(design, y, sample_weight=weight)
        assert model.converged_
        beta = np.r_[model.coef_, model.intercept_] if intercept else model.coef_
        assert np.max(-(A @ beta + b)) <= 1e-8
        score = X @ model.coef_ + model.intercept_
        residual = y - score
        losses = np.maximum(1 - y * score, 0) if loss == "svm" else np.maximum(0.25 * residual, -0.75 * residual)
        if intercept:
            beta[-1] /= model.intercept_scaling
        objective = 0.2 * (weight @ losses) + 0.7 * (beta @ beta) / 2 + 0.3 * abs(beta).sum()
        assert_objective_close(objective, 0.7 * dense.objective_)
        assert_objective_close(objective, 0.7 * model.dual_objective_)
        assert pickle.dumps(original) == before
        snapshot = pickle.loads(pickle.dumps(model.to_inference()))
        np.testing.assert_allclose(snapshot.predict(design), model.predict(design), atol=1e-12)


@pytest.mark.parametrize("strategy", ["ovo", "ovr"])
def test_multiclass_mixed_constraints_and_threaded_fits(strategy, assert_objective_close):
    rng = np.random.default_rng(31)
    X, y = rng.normal(size=(30, 3)), np.arange(30) % 3
    constraints = [{"name": "nonnegative"}, {"name": "monotonic"}, {"name": "fair", "sen_idx": [0], "tol_sen": [0.3]}]
    A, b = -np.eye(3), np.full(3, 0.4)
    model = plq_ElasticNet_Classifier(
        **OPTIONS,
        loss={"name": "svm"},
        C=0.1,
        l1_ratio=0.2,
        constraint=constraints,
        A=A,
        b=b,
        fit_intercept=True,
        multi_class=strategy,
        class_weight="balanced",
        n_jobs=2,
    )
    with pytest.warns(UserWarning, match="combining"):
        dense = clone(model).fit(X, y)
    with pytest.warns(UserWarning, match="combining"):
        model.set_params(A=sparse.csc_matrix(A)).fit(sparse.csr_matrix(X), y)
    assert np.all(model.converged_)
    assert_objective_close(model.objective_, dense.objective_)
    assert np.min(model.coef_) >= -1e-8 and np.max(model.coef_) <= 0.4 + 1e-8
    assert np.min(np.diff(model.coef_, axis=1)) >= -1e-8
    covariance = np.cov(X, rowvar=False, bias=True)[0]
    assert np.max(abs(model.coef_ @ covariance)) <= 0.3 + 1e-8


def test_constraint_paths_and_grid_search(assert_objective_close):
    rng = np.random.default_rng(27)
    X, y = rng.normal(size=(24, 3)), rng.normal(size=24)
    A, b = sparse.eye(3, format="csr"), np.zeros(3)
    constraints = [{"name": "custom", "A": A, "b": b}]
    Cs = [0.02, 0.2, 1.0]
    _, _, values, _, coefs = plqERM_Ridge_path_sol(
        sparse.csr_matrix(X),
        y,
        loss={"name": "QR", "qt": 0.3},
        constraint=constraints,
        Cs=Cs,
        **OPTIONS,
        warm_start=True,
        return_time=False,
    )
    for j, C in enumerate(Cs):
        dense = plqERM_Ridge(**OPTIONS, loss={"name": "QR", "qt": 0.3}, C=C, A=np.eye(3), b=b).fit(X, y)
        residual = y - X @ coefs[:, j]
        value = C * np.maximum(0.3 * residual, -0.7 * residual).sum() + coefs[:, j] @ coefs[:, j] / 2
        assert_objective_close(value, dense.objective_)
        assert_objective_close(value, values[j])
        assert coefs[:, j].min() >= -1e-8
    search = GridSearchCV(plq_Ridge_Regressor(**OPTIONS, A=A, b=b), {"C": Cs}, cv=3, error_score="raise")
    search.fit(sparse.csr_matrix(X), y)
    assert np.isfinite(search.cv_results_["mean_test_score"]).all()


@pytest.mark.parametrize("sparse_x", [False, True])
@pytest.mark.parametrize("composite", [False, True])
def test_direct_native_sparse_A_and_implicit_cqr(sparse_x, composite):
    from rehline import _internal

    q = 2 if composite else 0
    X = np.array([[0.3], [-0.2]])
    A = sparse.eye(1 + q, format="csr")
    b = -np.arange(1.0, 2 + q)
    count = len(X) * max(1, q)
    U, V = np.full((1, count), 0.01), np.ones((1, count))
    empty = np.empty((0, count))
    prefix = "rehline_cqr" if composite else "rehline"
    suffix = "sparse_both" if sparse_x else "sparse_constraints"
    native = getattr(_internal, f"{prefix}_{suffix}_internal")
    args = [sparse.csr_matrix(X) if sparse_x else X, A, b, np.empty(0), U, V, empty, empty, empty]
    if composite:
        args.append(q)
    result = _internal.rehline_result()
    native(result, *args, 100000, 1e-8)
    design = np.hstack((np.tile(X, (q, 1)), np.repeat(np.eye(q), len(X), axis=0))) if q else X
    expected = ReHLine_solver(design, U, V, A=np.eye(1 + q), b=b, **OPTIONS)
    assert result.converged
    np.testing.assert_allclose(result.objective, expected.objective, rtol=1e-8)
    np.testing.assert_allclose(result.beta, -b, atol=1e-8)
    # Public dispatch, including implicit CQR, must choose the same native path.
    public = ReHLine_solver(args[0], U, V, A=A, b=b, _quantile_count=q, **OPTIONS)
    np.testing.assert_allclose(public.objective, result.objective, rtol=1e-12)
    A.data[0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        native(_internal.rehline_result(), *args, 100000, 1e-8)


@pytest.mark.parametrize(
    "constraint",
    [
        {"name": "nonnegative"},
        {"name": ">=0"},
        {"name": "monotonic"},
        {"name": "monotonic", "decreasing": True},
        {"name": "monotonicity"},
    ],
)
@pytest.mark.parametrize("d", [1, 5])
@pytest.mark.parametrize("sparse_x", [False, True])
def test_structured_constraints_always_sparse_with_independent_projection(
    constraint, d, sparse_x, assert_objective_close
):
    # Identity MSE has equal curvature in every coefficient. The constrained
    # optimum is the nonnegative/isotonic projection of y/2, independently of CD.
    X = sparse.eye(d, format="csr") if sparse_x else np.eye(d)
    y = np.resize([3.0, -1.0, 4.0, -2.0, 1.0], d)
    nonnegative = constraint["name"] in ("nonnegative", ">=0")
    if nonnegative:
        optimum = np.maximum(y / 2, 0)
    else:
        optimum = IsotonicRegression(increasing=not constraint.get("decreasing", False)).fit_transform(
            np.arange(d), y / 2
        )
    model = plqERM_Ridge(**OPTIONS, loss={"name": "MSE"}, C=0.5, constraint=[constraint]).fit(X, y)
    assert sparse.isspmatrix_csr(model._A)
    assert model._A.nnz == (d if nonnegative else 2 * (d - 1))
    assert model.converged_ and np.min(model._A @ model.coef_ + model._b, initial=0) >= -1e-8
    expected = 0.5 * (optimum @ optimum + np.square(optimum - y).sum())
    actual = 0.5 * (model.coef_ @ model.coef_ + np.square(model.coef_ - y).sum())
    assert_objective_close(actual, expected)
    assert_objective_close(model.dual_objective_, expected)


@pytest.mark.parametrize("sparse_x", [False, True])
def test_large_sparse_constraints_and_structured_generation_never_densify(monkeypatch, sparse_x):
    # A dense A would require 80 GB. Neither Python preparation nor C++ CD
    # should allocate it. The optimum is analytic: beta=1, objective=d/2.
    d = 100000
    A = sparse.eye(d, format="csr")
    X = sparse.csr_matrix((2, d)) if sparse_x else np.zeros((2, d))

    def forbidden(*args, **kwargs):
        raise AssertionError("Do not densify sparse constraints")

    for cls in FORMATS:
        monkeypatch.setattr(cls, "toarray", forbidden)
        monkeypatch.setattr(cls, "todense", forbidden)
    result = ReHLine_solver(X, None, None, A=A, b=-np.ones(d), **OPTIONS)
    assert result.converged and result.objective == d / 2
    np.testing.assert_array_equal(result.beta, 1.0)
    for name, rows, entries in [("nonnegative", d, d), ("monotonic", d - 1, 2 * (d - 1))]:
        generated, _ = _make_constraint_rehline_param([{"name": name}], X)
        assert sparse.isspmatrix_csr(generated)
        assert generated.shape == (rows, d) and generated.nnz == entries
    model = plq_Ridge_Regressor(
        **OPTIONS,
        constraint=[{"name": "nonnegative"}, {"name": "monotonic", "decreasing": True}],
        fit_intercept=True,
    ).fit(X, [0.0, 0.0])
    assert model.converged_ and model.objective_ == 0
    assert sparse.isspmatrix_csr(model._model_._A)
    assert model._model_._A.shape == (2 * d - 1, d + 1)
    assert model._model_._A[:, -1].nnz == 0  # Feature constraints exclude the intercept.
    # A precision-stalled pair forces the optional polish path while all rows
    # are active. Its dense submatrix would also require 80 GB; skipping that
    # correction must retain the failed objective certificate.
    difficult = sparse.block_diag((np.array([[1.0, 0.0], [-1.0, 1e-8]]), sparse.eye(d - 2)), format="csr")
    result = ReHLine_solver(
        X,
        None,
        None,
        A=difficult,
        b=np.r_[0.0, -1e-8, -np.ones(d - 2)],
        xi=np.r_[1.1e8, 1.1e8, np.ones(d - 2)],
        max_iter=1,
        tol=1e-8,
        shrink=0,
        verbose=0,
    )
    assert result.kkt_residual <= 1e-8 and result.scaled_constraint_violation <= 1e-8
    assert not result.converged and result.dual_gap > 0.1
