"""Compact exports preserve inference and objectives without training storage."""

import pickle
import tracemalloc

import joblib
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rehline import (
    CQR_Ridge,
    CQR_Ridge_path_sol,
    plq_ElasticNet_Classifier,
    plq_ElasticNet_Regressor,
    plq_Ridge_Classifier,
    plq_Ridge_Regressor,
)

DIAGNOSTICS = (
    "objective_",
    "dual_objective_",
    "dual_gap_",
    "kkt_residual_",
    "constraint_violation_",
    "converged_",
    "n_iter_",
)
TRAINING_STATE = {"_model_", "_models_", "_U", "_V", "_S", "_T", "_Tau", "_A", "_b", "_Lambda", "_Gamma", "_xi", "_mu"}


def verify_snapshot(model, snapshot, X):
    assert not TRAINING_STATE.intersection(vars(snapshot))
    for name in DIAGNOSTICS:
        np.testing.assert_array_equal(getattr(snapshot, name), getattr(model, name))
    np.testing.assert_array_equal(snapshot.predict(X), model.predict(X))
    assert not np.shares_memory(snapshot.coef_, model.coef_)
    if isinstance(model.intercept_, np.ndarray):
        assert not np.shares_memory(snapshot.intercept_, model.intercept_)
    before = pickle.dumps(vars(snapshot))
    with pytest.raises(TypeError, match="cannot fit or warm-start"):
        snapshot.fit(X, np.zeros(len(X)))
    assert pickle.dumps(vars(snapshot)) == before
    with pytest.raises(ValueError):
        snapshot.predict(np.zeros((3, X.shape[1] + 1)))


@pytest.mark.parametrize("estimator", [plq_Ridge_Classifier, plq_ElasticNet_Classifier])
@pytest.mark.parametrize("classes,strategy", [(2, "ovr"), (2, "ovo"), (3, "ovr"), (4, "ovo"), (13, "ovo")])
@pytest.mark.parametrize("zero", [False, True])
def test_classifier_snapshot_formats_objectives_and_independence(estimator, classes, strategy, zero, tmp_path):
    rng = np.random.default_rng(314)
    y = np.tile(np.array([f"label-{i}" for i in range(classes)]), 5)
    X = np.zeros((len(y), 3)) if zero else rng.normal(size=(len(y), 3))
    weight = rng.uniform(0.1, 2, len(y))
    model = estimator(
        loss={"name": "svm"},
        multi_class=strategy,
        C=0.1,
        fit_intercept=not zero,
        intercept_scaling=3.0,
        tol=1e-10,
        max_iter=100000,
    ).fit(X, y, sample_weight=weight)
    before = pickle.dumps(vars(model))
    snapshot = model.to_inference()
    assert pickle.dumps(vars(model)) == before
    verify_snapshot(model, snapshot, X)
    original = snapshot.predict(X)
    for output in ("ovr", "ovo") if strategy == "ovo" or classes == 2 else ("ovr",):
        model.set_params(decision_function_shape=output)
        snapshot.set_params(decision_function_shape=output)
        np.testing.assert_array_equal(snapshot.decision_function(X), model.decision_function(X))
        np.testing.assert_array_equal(snapshot.predict(X), original)
        path = tmp_path / "snapshot.joblib"
        joblib.dump(snapshot, path)
        restored = joblib.load(path)
        np.testing.assert_array_equal(restored.decision_function(X), model.decision_function(X))
        np.testing.assert_array_equal(restored.predict(X), original)
    # Compute the original weighted objective, including scaled intercept penalty,
    # for every stored coefficient row, without accessing an inner ERM model.
    keys = [tuple(model.classes_)] if classes == 2 else model._model_keys_
    for i, key in enumerate(keys):
        mask = np.isin(y, key) if len(key) == 2 else np.ones(len(y), dtype=bool)
        target = np.where(y[mask] == key[-1], 1.0, -1.0)
        coef = snapshot.coef_ if classes == 2 else snapshot.coef_[i]
        intercept = snapshot.intercept_ if classes == 2 else snapshot.intercept_[i]
        beta = np.r_[coef, intercept / 3.0] if not zero else coef
        ratio = getattr(model, "l1_ratio", 0)
        value = model.C * (weight[mask] @ np.maximum(1 - target * (X[mask] @ coef + intercept), 0))
        value += 0.5 * (1 - ratio) * (beta @ beta) + ratio * abs(beta).sum()
        recorded = np.atleast_1d(snapshot.objective_)[i] * (1 - ratio)
        lower = np.atleast_1d(snapshot.dual_objective_)[i] * (1 - ratio)
        assert value == pytest.approx(recorded, rel=1e-8, abs=1e-9)
        assert value == pytest.approx(lower, rel=1e-8, abs=1e-9)
    saved = pickle.dumps(vars(snapshot))
    model.fit(X, y, sample_weight=weight * 2)
    model.coef_[:] = 99
    model.classes_[:] = "changed"
    assert pickle.dumps(vars(snapshot)) == saved
    np.testing.assert_array_equal(snapshot.predict(X), original)


@pytest.mark.parametrize("estimator", [plq_Ridge_Regressor, plq_ElasticNet_Regressor])
def test_regression_snapshot_pipeline_feature_names_and_objective(estimator):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(16)
    X, y = pd.DataFrame(rng.normal(size=(30, 4)), columns=list("abcd")), rng.normal(size=30)
    weight = np.linspace(0.1, 2, len(y))
    model = estimator(
        loss={"name": "MSE"}, C=0.1, fit_intercept=True, intercept_scaling=2.0, tol=1e-10, max_iter=100000
    ).fit(X, y, sample_weight=weight)
    snapshot = model.to_inference()
    np.testing.assert_array_equal(snapshot.feature_names_in_, model.feature_names_in_)
    assert not hasattr(snapshot, "decision_function")
    np.testing.assert_array_equal(snapshot.predict(X), model.predict(X))
    with pytest.raises(ValueError, match="feature names"):
        snapshot.predict(X[list("dcba")])
    beta = np.r_[snapshot.coef_, snapshot.intercept_ / 2.0]
    ratio = getattr(model, "l1_ratio", 0)
    value = model.C * (weight @ (y - snapshot.predict(X)) ** 2)
    value += 0.5 * (1 - ratio) * (beta @ beta) + ratio * abs(beta).sum()
    assert value == pytest.approx(snapshot.objective_ * (1 - ratio), rel=1e-8, abs=1e-9)
    pipe = Pipeline(
        [("scale", StandardScaler()), ("model", estimator(loss={"name": "MSE"}, C=0.1, tol=1e-10, max_iter=100000))]
    ).fit(X, y)
    prediction = pipe.predict(X)
    pipe.steps[-1] = ("model", pipe[-1].to_inference())
    np.testing.assert_array_equal(pipe.predict(X), prediction)
    restored = pickle.loads(pickle.dumps(pipe))
    np.testing.assert_array_equal(restored.predict(X), prediction)


@pytest.mark.parametrize("warm_start", [False, True])
@pytest.mark.parametrize("return_time", [False, True])
def test_compact_cqr_path_preserves_outputs_diagnostics_and_joint_objectives(warm_start, return_time):
    rng = np.random.default_rng(71)
    X, y = rng.normal(size=(25, 3)), rng.normal(size=25)
    levels = np.array([0.8, 0.2, 0.5, 0.5])
    options = dict(
        quantiles=levels,
        Cs=[0.03, 0.01, 0.1],
        tol=1e-10,
        max_iter=100000,
        warm_start=warm_start,
        return_time=return_time,
    )
    full = CQR_Ridge_path_sol(X, y, **options)
    compact = CQR_Ridge_path_sol(X, y, **options, compact=True)
    assert len(full) == len(compact) == (5 if return_time else 4)
    for index in (0, 2, 3):
        np.testing.assert_array_equal(full[index], compact[index])
    for C, model, snapshot in zip(full[0], full[1], compact[1]):
        verify_snapshot(model, snapshot, X)
        np.testing.assert_array_equal(snapshot.quantiles_, levels)
        residual = y[:, None] - snapshot.predict(X)
        actual = C * np.maximum(levels * residual, (levels - 1) * residual).sum()
        actual += 0.5 * (snapshot.coef_ @ snapshot.coef_ + snapshot.intercept_ @ snapshot.intercept_)
        assert actual == pytest.approx(snapshot.objective_, rel=1e-8, abs=1e-9)
        assert actual == pytest.approx(snapshot.dual_objective_, rel=1e-8, abs=1e-9)
        restored = pickle.loads(pickle.dumps(snapshot))
        np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    levels[:] = 0.7
    for snapshot in compact[1]:
        np.testing.assert_array_equal(snapshot.quantiles_, [0.8, 0.2, 0.5, 0.5])
    first = compact[1][0].coef_.copy()
    compact[1][-1].coef_[:] = 999
    np.testing.assert_array_equal(compact[1][0].coef_, first)


@pytest.mark.parametrize(
    "estimator",
    [CQR_Ridge, plq_Ridge_Regressor, plq_ElasticNet_Regressor, plq_Ridge_Classifier, plq_ElasticNet_Classifier],
)
def test_unfitted_export_rejected(estimator):
    model = estimator([0.2, 0.8]) if estimator is CQR_Ridge else estimator(loss={"name": "svm"})
    with pytest.raises(NotFittedError):
        model.to_inference()


@pytest.mark.parametrize("bad", [None, 1, "yes", [], np.array([True])])
def test_invalid_compact_mode_rejected_before_fitting(bad, monkeypatch):
    monkeypatch.setattr(CQR_Ridge, "fit", lambda *a, **k: pytest.fail("Invalid options reached fit"))
    with pytest.raises(ValueError, match="compact must be boolean"):
        CQR_Ridge_path_sol(np.ones((3, 2)), np.ones(3), quantiles=[0.5], Cs=[0.1], compact=bad)


def test_compact_storage_does_not_grow_with_training_samples_or_copy_loss_arrays():
    sizes = []
    for n in (100, 1000):
        X, y = np.zeros((n, 4)), np.zeros(n)
        options = dict(quantiles=np.linspace(0.1, 0.9, 9), Cs=[0.001, 0.002, 0.003], return_time=False)
        full = CQR_Ridge_path_sol(X, y, **options)
        compact = CQR_Ridge_path_sol(X, y, **options, compact=True)
        sizes.append(len(pickle.dumps(compact)))
        assert len(pickle.dumps(full)) > 10 * sizes[-1]
    assert sizes[1] <= 1.05 * sizes[0]
    model = CQR_Ridge(np.linspace(0.1, 0.9, 20)).fit(np.zeros((10000, 4)), np.zeros(10000))
    tracemalloc.start()
    try:
        snapshot = model.to_inference()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 100000  # Deep-copying training state would cost > 9 MB.
    assert len(pickle.dumps(snapshot)) < 10000


@pytest.mark.parametrize("classes", [3, 13])
@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
def test_classifier_export_storage_is_independent_of_training_sample_count(classes, strategy):
    sizes = []
    for repeats in (2, 20):
        y = np.tile(np.arange(classes), repeats)
        X = np.zeros((len(y), 3))
        model = plq_Ridge_Classifier(loss={"name": "svm"}, fit_intercept=False, multi_class=strategy).fit(X, y)
        snapshot = model.to_inference()
        sizes.append(len(pickle.dumps(snapshot)))
        assert not TRAINING_STATE.intersection(vars(snapshot))
    assert sizes[1] <= 1.05 * sizes[0]


@pytest.mark.parametrize("family", ["cqr", "regression", "classification"])
def test_benchmark_gate_detects_corrupted_export(family, monkeypatch):
    pytest.importorskip("cvxpy")
    from benchmarks import api_correctness, cqr_correctness

    estimator = {"cqr": CQR_Ridge, "regression": plq_Ridge_Regressor, "classification": plq_Ridge_Classifier}[family]
    original = estimator.to_inference

    def corrupted(model):
        snapshot = original(model)
        snapshot.coef_ += 1
        return snapshot

    monkeypatch.setattr(estimator, "to_inference", corrupted)
    if family == "cqr":
        report = cqr_correctness.run_suite(cases=1)
        assert report["passed"] == 0
    else:
        # Regression case 4 chooses the sklearn Ridge wrapper; classifier case 1
        # chooses Ridge. The reference remains independent of the bad export.
        report = api_correctness.run_suite(cases=1, case_index=4 if family == "regression" else 1)
        assert report["passed"] == 0
