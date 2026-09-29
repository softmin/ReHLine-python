"""Update order must be independent of shrinking and preserve solver contracts."""

import numpy as np
import pytest
from benchmarks.correctness.core import _native_problem, make_case, solve_reference
from benchmarks.diagnostics.numerical_stress import ACCURACY, measure
from sklearn.base import clone

import rehline


def problem(index=71):
    return _native_problem(make_case(index, max_samples=20, max_dim=5, profile="routine"))


def solve(p, **options):
    return rehline.ReHLine_solver(**p, **dict(max_iter=53, tol=ACCURACY, verbose=0, **options))


def assert_same(left, right):
    for field in (
        "beta",
        "Lambda",
        "Gamma",
        "xi",
        "mu",
        "objective",
        "dual_objective",
        "niter",
        "converged",
        "kkt_residual",
        "scaled_constraint_violation",
    ):
        np.testing.assert_array_equal(getattr(left, field), getattr(right, field), err_msg=field)


@pytest.mark.parametrize("shrink", [0, 1, 42])
def test_auto_preserves_legacy_order_and_seed(shrink):
    p = problem()
    legacy = solve(p, shrink=shrink)
    explicit = solve(
        p, shrink=shrink, coordinate_order="random" if shrink else "cyclic", coordinate_seed=shrink if shrink else 1
    )
    assert_same(legacy, explicit)


@pytest.mark.parametrize("shrink", [0, 1])
def test_seed_is_reproducible_and_order_is_independent(shrink):
    p = problem()
    first = solve(p, shrink=shrink, coordinate_order="random", coordinate_seed=0)
    other = solve(p, shrink=shrink, coordinate_order="random", coordinate_seed=42)
    assert not np.array_equal(first.beta, other.beta)
    assert_same(first, solve(p, shrink=shrink, coordinate_order="random", coordinate_seed=0))
    assert_same(
        solve(p, shrink=shrink, coordinate_order="cyclic", coordinate_seed=0),
        solve(p, shrink=shrink, coordinate_order="cyclic", coordinate_seed=42),
    )


def test_explicit_seed_overrides_legacy_shrink_seed():
    p = problem()
    assert_same(solve(p, shrink=1, coordinate_order="random", coordinate_seed=42), solve(p, shrink=42))
    # The default seed for random full sweeps is also deterministic.
    assert_same(
        solve(p, shrink=0, coordinate_order="random"), solve(p, shrink=0, coordinate_order="random", coordinate_seed=1)
    )
    assert not np.array_equal(
        solve(p, shrink=0, coordinate_order="random").beta, solve(p, shrink=1, coordinate_order="random").beta
    )


@pytest.mark.parametrize("index", [11, 23, 35, 47, 59, 71])
@pytest.mark.parametrize("shrink", [0, 1])
@pytest.mark.parametrize("order", ["cyclic", "random"])
def test_four_modes_match_independent_mixed_loss_references(index, shrink, order):
    pytest.importorskip("cvxpy")
    case = make_case(index, max_samples=20, max_dim=5, profile="routine")
    p = _native_problem(case)
    reference = solve_reference(case)["objective"] / (1 - case["l1_ratio"])
    result = rehline.ReHLine_solver(
        **p,
        shrink=shrink,
        coordinate_order=order,
        coordinate_seed=7,
        tol=ACCURACY,
        max_iter=1_000_000,
        verbose=1,
        trace_freq=1000,
    )
    audit = measure(p, result, reference, tol=ACCURACY, max_iter=1_000_000)
    assert audit["status"] == "met", audit
    warm = {name: getattr(result, name) for name in ("Lambda", "Gamma", "xi", "mu")}
    refit = rehline.ReHLine_solver(
        **p,
        **warm,
        shrink=shrink,
        coordinate_order=order,
        coordinate_seed=7,
        tol=ACCURACY,
        max_iter=1_000_000,
        verbose=0,
    )
    assert measure(p, refit, reference, tol=ACCURACY, max_iter=1_000_000)["status"] == "met"


@pytest.mark.parametrize(
    "key,value",
    [
        ("coordinate_order", "shuffle"),
        ("coordinate_order", None),
        ("coordinate_order", 1),
        ("coordinate_order", ["random"]),
        ("coordinate_seed", -1),
        ("coordinate_seed", True),
        ("coordinate_seed", 1.5),
        ("coordinate_seed", 2**31),
        ("coordinate_seed", "1"),
    ],
)
def test_invalid_order_options_are_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        solve(problem(), **{key: value})
    with pytest.raises(ValueError, match=key):
        rehline.plqERM_Ridge(loss={"name": "MSE"}, **{key: value}).fit(np.eye(3), np.ones(3))


def test_native_old_positional_call_and_new_options():
    p = problem()
    args = [p[key] for key in ("X", "A", "b", "rho", "U", "V", "S", "T", "Tau")]
    result = rehline.rehline_result()
    rehline.rehline_internal(result, *args, 53, ACCURACY, 1, 0, 100)
    assert_same(result, solve(p, shrink=1))
    rehline.rehline_internal(result := rehline.rehline_result(), *args, 53, ACCURACY, 0, 0, 100, 2, 7)
    assert_same(result, solve(p, shrink=0, coordinate_order="random", coordinate_seed=7))
    for order, seed in [(-1, 1), (3, 1), (2, -2)]:
        with pytest.raises(ValueError, match="solver options"):
            rehline.rehline_internal(rehline.rehline_result(), *args, 53, ACCURACY, 0, 0, 100, order, seed)


MODELS = [
    rehline.ReHLine(),
    rehline.plqERM_Ridge(loss={"name": "MSE"}),
    rehline.plqERM_ElasticNet(loss={"name": "MSE"}),
    rehline.CQR_Ridge([0.2, 0.8]),
    rehline.plq_Ridge_Regressor(),
    rehline.plq_ElasticNet_Regressor(),
    rehline.plq_Ridge_Classifier(loss={"name": "hinge"}, multi_class="ovo"),
    rehline.plq_ElasticNet_Classifier(loss={"name": "hinge"}, multi_class="ovr"),
    rehline.plqMF_Ridge(n_users=3, n_items=3, loss={"name": "MSE"}, random_state=42),
]


@pytest.mark.parametrize("model", MODELS, ids=lambda m: type(m).__name__)
@pytest.mark.parametrize("shrink,order", [(0, "random"), (1, "cyclic")])
def test_estimators_clone_and_forward_order_options(model, shrink, order, monkeypatch):
    import rehline._class as classes
    import rehline._mf_class as mf

    class ReachedSolver(Exception):
        pass

    def inspect(**kwargs):
        assert kwargs["shrink"] == shrink
        assert kwargs["coordinate_order"] == order
        assert kwargs["coordinate_seed"] == 7
        raise ReachedSolver()

    monkeypatch.setattr(classes, "ReHLine_solver", inspect)
    monkeypatch.setattr(mf, "ReHLine_solver", inspect)
    fitted = clone(model.set_params(shrink=shrink, coordinate_order=order, coordinate_seed=7))
    X = np.array([[0, 1], [1, 0], [2, 2], [1, 2], [2, 0], [0, 2]])
    with pytest.raises(ReachedSolver):
        if isinstance(fitted, rehline.ReHLine):
            fitted.fit(X)
        else:
            fitted.fit(X, np.array([0, 1, 2, 0, 1, 2]))


@pytest.mark.parametrize("cqr", [False, True])
def test_paths_forward_order_options(cqr, monkeypatch):
    import rehline._class as classes

    class ReachedSolver(Exception):
        pass

    def inspect(**kwargs):
        assert kwargs["shrink"] == 0
        assert kwargs["coordinate_order"] == "random"
        assert kwargs["coordinate_seed"] == 7
        raise ReachedSolver()

    monkeypatch.setattr(classes, "ReHLine_solver", inspect)
    path = rehline.CQR_Ridge_path_sol if cqr else rehline.plqERM_Ridge_path_sol
    with pytest.raises(ReachedSolver):
        path(
            np.eye(3),
            np.arange(3),
            Cs=[0.1],
            shrink=0,
            coordinate_order="random",
            coordinate_seed=7,
            **({"quantiles": [0.2, 0.8]} if cqr else {"loss": {"name": "MSE"}}),
        )
