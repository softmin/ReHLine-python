"""Mini GridSearchCV benchmark for ReHLine estimators.

The public helpers in this module are intentionally small so the file can be
copied into a separate benchmark repository and adjusted without touching the
main package.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import statistics
import sys
import time
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import arff
from sklearn.base import clone
from sklearn.datasets import (
    fetch_california_housing,
    fetch_covtype,
    fetch_openml,
    get_data_home,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_linnerud,
    load_wine,
    make_classification,
    make_friedman1,
    make_regression,
    make_sparse_uncorrelated,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from rehline import (
    CQR_Ridge,
    plq_ElasticNet_Classifier,
    plq_ElasticNet_Regressor,
    plq_Ridge_Classifier,
    plq_Ridge_Regressor,
)
from rehline import (
    __version__ as rehline_version,
)

DatasetFactory = Callable[[], tuple[np.ndarray, np.ndarray]]
DEFAULT_C_GRID = (0.1, 1.0, 10.0)
DEFAULT_L1_RATIO_GRID = (0.5,)
DEFAULT_QUANTILE_GRID = (0.25,)
DEFAULT_CQR_QUANTILES_GRID = ((0.1, 0.5, 0.9),)
DEFAULT_CONFIG_PATH = Path(__file__).with_name("mini_config.json")
DEFAULT_RESULTS_DIR = Path(__file__).with_name("results")
OPENML_GUILLERMO_URL = "https://openml.org/data/v1/download/19335532/guillermo.arff"
DEFAULT_MARKDOWN_COLUMNS = (
    "task",
    "dataset",
    "n_samples",
    "n_features",
    "cv",
    "n_candidates",
    "repeats",
    "elapsed_sec_mean",
    "elapsed_sec_std",
    "best_C",
    "best_mse",
    "best_accuracy",
)
CONFIG_DEFAULTS = {
    "task_datasets": {
        "ridge_quantile": [
            "california_housing",
            "make_regression_100k",
            "make_friedman1_5k_100",
        ],
        "ridge_quantile_monotonic": [
            "california_housing",
            "make_regression_100k",
            "make_friedman1_5k_100",
        ],
        "ridge_composite_quantile": [
            "california_housing",
            "make_friedman1_5k_100",
        ],
        "elasticnet_quantile": [
            "california_housing",
            "make_regression_100k",
            "make_friedman1_5k_100",
        ],
        "elasticnet_quantile_monotonic": [
            "california_housing",
            "make_regression_100k",
            "make_friedman1_5k_100",
        ],
        "ridge_svm": [
            "digits_low_high",
            "make_classification_100k",
            "openml_bioresponse",
        ],
        "elasticnet_svm": [
            "digits_low_high",
            "make_classification_100k",
            "openml_bioresponse",
        ],
    },
    "cv": 2,
    "repeats": 1,
    "n_jobs": None,
    "max_iter": 5_000_000,
    "tol": 1e-4,
    "C_grid": list(DEFAULT_C_GRID),
    "l1_ratio_grid": list(DEFAULT_L1_RATIO_GRID),
    "quantile_grid": list(DEFAULT_QUANTILE_GRID),
    "cqr_quantiles_grid": [list(quantiles) for quantiles in DEFAULT_CQR_QUANTILES_GRID],
    "preprocess_X": "standard",
    "output_dir": str(DEFAULT_RESULTS_DIR),
}


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset descriptor used by the benchmark runner."""

    name: str
    problem_type: str
    factory: DatasetFactory


@dataclass(frozen=True)
class BenchmarkTask:
    """GridSearchCV benchmark descriptor."""

    name: str
    problem_type: str
    estimator: Any
    param_grid: dict[str, list[Any]]
    scoring: str | Callable[..., float] | None = None


def _as_float64(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(X, dtype=np.float64), np.asarray(y)


def _as_binary_target(y: Any) -> np.ndarray:
    values = np.asarray(y)
    classes, encoded = np.unique(values, return_inverse=True)
    if classes.shape[0] != 2:
        raise ValueError(f"Expected a binary target, got {classes.shape[0]} classes.")
    return encoded.astype(int)


def _make_toy_regression() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(
        n_samples=500,
        n_features=20,
        n_informative=12,
        noise=10.0,
        random_state=0,
    )
    return _as_float64(X, y)


def _make_toy_classification() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        class_sep=1.5,
        random_state=0,
    )
    return _as_float64(X, y)


def _make_large_regression(n_samples: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(
        n_samples=n_samples,
        n_features=20,
        n_informative=12,
        noise=10.0,
        random_state=random_state,
    )
    return _as_float64(X, y)


def _make_large_classification(n_samples: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        class_sep=1.5,
        random_state=random_state,
    )
    return _as_float64(X, y)


def _fetch_california_housing() -> tuple[np.ndarray, np.ndarray]:
    data = fetch_california_housing()
    return _as_float64(data.data, data.target)


def _fetch_covtype_binary(n_samples: int | None = None, random_state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    data = fetch_covtype()
    mask = data.target <= 2
    X = data.data[mask]
    y = (data.target[mask] == 2).astype(int)
    if n_samples is not None and n_samples < X.shape[0]:
        rng = np.random.default_rng(random_state)
        rows = np.sort(rng.choice(X.shape[0], size=n_samples, replace=False))
        X = X[rows]
        y = y[rows]
    return _as_float64(X, y)


def _fetch_openml_binary(data_id: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = fetch_openml(data_id=data_id, return_X_y=True, as_frame=False, parser="liac-arff")
    return _as_float64(X, _as_binary_target(y))


def _fetch_openml_regression(data_id: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = fetch_openml(data_id=data_id, return_X_y=True, as_frame=False, parser="liac-arff")
    return _as_float64(X, np.asarray(y, dtype=np.float64))


def _fetch_openml_guillermo() -> tuple[np.ndarray, np.ndarray]:
    try:
        return _fetch_openml_binary(41159)
    except ValueError as error:
        if "md5 checksum" not in str(error):
            raise
        return _fetch_guillermo_arff_fallback()


def _fetch_guillermo_arff_fallback() -> tuple[np.ndarray, np.ndarray]:
    cache_path = Path(get_data_home()) / "rehline_openml" / "guillermo_41159.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["X"], cached["y"]

    with urllib.request.urlopen(OPENML_GUILLERMO_URL) as response:
        text = response.read().decode("utf-8")
    data, _ = arff.loadarff(io.StringIO(text))

    names = data.dtype.names
    if names is None or "class" not in names:
        raise ValueError("Could not find target column 'class' in guillermo.arff.")
    feature_names = [name for name in names if name != "class"]
    X = np.column_stack([_numeric_arff_column(data[name]) for name in feature_names])
    y = _as_binary_target(data["class"])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y)
    return X, y


def _numeric_arff_column(column: Any) -> np.ndarray:
    values = np.asarray(column)
    if values.dtype.kind in {"S", "U", "O"}:
        _, encoded = np.unique(values, return_inverse=True)
        return encoded.astype(np.float64)
    return values.astype(np.float64)


def _make_friedman1() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_friedman1(n_samples=1000, n_features=10, noise=1.0, random_state=0)
    return _as_float64(X, y)


def _make_friedman1_stress() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_friedman1(n_samples=5_000, n_features=100, noise=1.0, random_state=20)
    return _as_float64(X, y)


def _make_sparse_uncorrelated() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_sparse_uncorrelated(n_samples=1000, n_features=10, random_state=0)
    return _as_float64(X, y)


def _load_diabetes() -> tuple[np.ndarray, np.ndarray]:
    data = load_diabetes()
    return _as_float64(data.data, data.target)


def _load_linnerud_weight() -> tuple[np.ndarray, np.ndarray]:
    data = load_linnerud()
    return _as_float64(data.data, data.target[:, 0])


def _load_breast_cancer() -> tuple[np.ndarray, np.ndarray]:
    data = load_breast_cancer()
    return _as_float64(data.data, data.target)


def _load_iris_binary() -> tuple[np.ndarray, np.ndarray]:
    data = load_iris()
    mask = data.target != 0
    X = data.data[mask]
    y = (data.target[mask] == 2).astype(int)
    return _as_float64(X, y)


def _load_wine_binary() -> tuple[np.ndarray, np.ndarray]:
    data = load_wine()
    mask = data.target != 2
    X = data.data[mask]
    y = data.target[mask]
    return _as_float64(X, y)


def _load_digits_0_1() -> tuple[np.ndarray, np.ndarray]:
    data = load_digits(n_class=2)
    return _as_float64(data.data, data.target)


def _load_digits_low_high() -> tuple[np.ndarray, np.ndarray]:
    data = load_digits()
    y = (data.target >= 5).astype(int)
    return _as_float64(data.data, y)


def available_datasets() -> dict[str, DatasetSpec]:
    """Return the built-in datasets available for one-line benchmarks."""

    return {
        "toy_regression": DatasetSpec("toy_regression", "regression", _make_toy_regression),
        "make_regression_10k": DatasetSpec(
            "make_regression_10k",
            "regression",
            lambda: _make_large_regression(10_000, 1),
        ),
        "make_regression_100k": DatasetSpec(
            "make_regression_100k",
            "regression",
            lambda: _make_large_regression(100_000, 10),
        ),
        "make_regression_300k": DatasetSpec(
            "make_regression_300k",
            "regression",
            lambda: _make_large_regression(300_000, 30),
        ),
        "california_housing": DatasetSpec("california_housing", "regression", _fetch_california_housing),
        "diabetes": DatasetSpec("diabetes", "regression", _load_diabetes),
        "friedman1": DatasetSpec("friedman1", "regression", _make_friedman1),
        "make_friedman1_5k_100": DatasetSpec(
            "make_friedman1_5k_100",
            "regression",
            _make_friedman1_stress,
        ),
        "openml_buzz_twitter": DatasetSpec(
            "openml_buzz_twitter",
            "regression",
            lambda: _fetch_openml_regression(4549),
        ),
        "sparse_uncorrelated": DatasetSpec("sparse_uncorrelated", "regression", _make_sparse_uncorrelated),
        "linnerud_weight": DatasetSpec("linnerud_weight", "regression", _load_linnerud_weight),
        "toy_classification": DatasetSpec("toy_classification", "classification", _make_toy_classification),
        "make_classification_100k": DatasetSpec(
            "make_classification_100k",
            "classification",
            lambda: _make_large_classification(100_000, 10),
        ),
        "make_classification_300k": DatasetSpec(
            "make_classification_300k",
            "classification",
            lambda: _make_large_classification(300_000, 30),
        ),
        "openml_guillermo": DatasetSpec(
            "openml_guillermo",
            "classification",
            _fetch_openml_guillermo,
        ),
        "openml_bioresponse": DatasetSpec(
            "openml_bioresponse",
            "classification",
            lambda: _fetch_openml_binary(4134),
        ),
        "covtype_binary": DatasetSpec("covtype_binary", "classification", _fetch_covtype_binary),
        "covtype_binary_50k": DatasetSpec(
            "covtype_binary_50k",
            "classification",
            lambda: _fetch_covtype_binary(50_000, 50),
        ),
        "covtype_binary_100k": DatasetSpec(
            "covtype_binary_100k",
            "classification",
            lambda: _fetch_covtype_binary(100_000, 100),
        ),
        "covtype_binary_full": DatasetSpec("covtype_binary_full", "classification", _fetch_covtype_binary),
        "breast_cancer": DatasetSpec("breast_cancer", "classification", _load_breast_cancer),
        "iris_binary": DatasetSpec("iris_binary", "classification", _load_iris_binary),
        "wine_binary": DatasetSpec("wine_binary", "classification", _load_wine_binary),
        "digits_0_1": DatasetSpec("digits_0_1", "classification", _load_digits_0_1),
        "digits_low_high": DatasetSpec("digits_low_high", "classification", _load_digits_low_high),
    }


def make_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one of the built-in benchmark datasets by name."""

    datasets = available_datasets()
    if name not in datasets:
        names = ", ".join(sorted(datasets))
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {names}.")
    return datasets[name].factory()


def available_tasks(
    max_iter: int = 1000,
    tol: float = 1e-4,
    C_grid: Iterable[float] | None = None,
    l1_ratio_grid: Iterable[float] | None = None,
    quantile_grid: Iterable[float] | None = None,
    cqr_quantiles_grid: Iterable[Iterable[float]] | None = None,
) -> dict[str, BenchmarkTask]:
    """Return the default ReHLine GridSearchCV tasks."""

    C_values = _float_list(C_grid, DEFAULT_C_GRID)
    l1_ratio_values = _float_list(l1_ratio_grid, DEFAULT_L1_RATIO_GRID)
    quantile_values = _float_list(quantile_grid, DEFAULT_QUANTILE_GRID)
    quantile_losses = [{"name": "QR", "qt": qt} for qt in quantile_values]
    cqr_quantiles_values = _nested_float_list(cqr_quantiles_grid, DEFAULT_CQR_QUANTILES_GRID)
    monotonic_constraints = [[{"name": "monotonic", "decreasing": False}]]

    return {
        "ridge_quantile": BenchmarkTask(
            name="ridge_quantile",
            problem_type="regression",
            estimator=plq_Ridge_Regressor(loss={"name": "QR", "qt": 0.5}, max_iter=max_iter, tol=tol),
            param_grid={
                "model__C": C_values,
                "model__loss": quantile_losses,
            },
            scoring="neg_mean_squared_error",
        ),
        "ridge_quantile_monotonic": BenchmarkTask(
            name="ridge_quantile_monotonic",
            problem_type="regression",
            estimator=plq_Ridge_Regressor(
                loss={"name": "QR", "qt": 0.5},
                constraint=monotonic_constraints[0],
                max_iter=max_iter,
                tol=tol,
            ),
            param_grid={
                "model__C": C_values,
                "model__loss": quantile_losses,
                "model__constraint": monotonic_constraints,
            },
            scoring="neg_mean_squared_error",
        ),
        "ridge_composite_quantile": BenchmarkTask(
            name="ridge_composite_quantile",
            problem_type="regression",
            estimator=CQR_Ridge(quantiles=cqr_quantiles_values[0], max_iter=max_iter, tol=tol),
            param_grid={
                "model__C": C_values,
                "model__quantiles": cqr_quantiles_values,
            },
            scoring=_cqr_median_neg_mean_squared_error,
        ),
        "ridge_mae": BenchmarkTask(
            name="ridge_mae",
            problem_type="regression",
            estimator=plq_Ridge_Regressor(loss={"name": "mae"}, max_iter=max_iter, tol=tol),
            param_grid={
                "model__C": C_values,
                "model__loss": [{"name": "mae"}],
            },
            scoring="neg_mean_squared_error",
        ),
        "ridge_huber": BenchmarkTask(
            name="ridge_huber",
            problem_type="regression",
            estimator=plq_Ridge_Regressor(loss={"name": "huber", "tau": 1.0}, max_iter=max_iter, tol=tol),
            param_grid={
                "model__C": C_values,
                "model__loss": [{"name": "huber", "tau": 1.0}],
            },
            scoring="neg_mean_squared_error",
        ),
        "ridge_svr": BenchmarkTask(
            name="ridge_svr",
            problem_type="regression",
            estimator=plq_Ridge_Regressor(loss={"name": "svr", "epsilon": 0.1}, max_iter=max_iter, tol=tol),
            param_grid={
                "model__C": C_values,
                "model__loss": [{"name": "svr", "epsilon": 0.1}],
            },
            scoring="neg_mean_squared_error",
        ),
        "ridge_quantile_eps": BenchmarkTask(
            name="ridge_quantile_eps",
            problem_type="regression",
            estimator=plq_Ridge_Regressor(
                loss={"name": "check_eps", "qt": 0.25, "epsilon": 0.1},
                max_iter=max_iter,
                tol=tol,
            ),
            param_grid={
                "model__C": C_values,
                "model__loss": [{"name": "check_eps", "qt": qt, "epsilon": 0.1} for qt in quantile_values],
            },
            scoring="neg_mean_squared_error",
        ),
        "elasticnet_quantile": BenchmarkTask(
            name="elasticnet_quantile",
            problem_type="regression",
            estimator=plq_ElasticNet_Regressor(loss={"name": "QR", "qt": 0.5}, max_iter=max_iter, tol=tol),
            param_grid={
                "model__C": C_values,
                "model__l1_ratio": l1_ratio_values,
                "model__loss": quantile_losses,
            },
            scoring="neg_mean_squared_error",
        ),
        "elasticnet_quantile_monotonic": BenchmarkTask(
            name="elasticnet_quantile_monotonic",
            problem_type="regression",
            estimator=plq_ElasticNet_Regressor(
                loss={"name": "QR", "qt": 0.5},
                constraint=monotonic_constraints[0],
                max_iter=max_iter,
                tol=tol,
            ),
            param_grid={
                "model__C": C_values,
                "model__l1_ratio": l1_ratio_values,
                "model__loss": quantile_losses,
                "model__constraint": monotonic_constraints,
            },
            scoring="neg_mean_squared_error",
        ),
        "elasticnet_mae": BenchmarkTask(
            name="elasticnet_mae",
            problem_type="regression",
            estimator=plq_ElasticNet_Regressor(loss={"name": "mae"}, max_iter=max_iter, tol=tol),
            param_grid={
                "model__C": C_values,
                "model__l1_ratio": l1_ratio_values,
                "model__loss": [{"name": "mae"}],
            },
            scoring="neg_mean_squared_error",
        ),
        "elasticnet_huber": BenchmarkTask(
            name="elasticnet_huber",
            problem_type="regression",
            estimator=plq_ElasticNet_Regressor(
                loss={"name": "huber", "tau": 1.0},
                max_iter=max_iter,
                tol=tol,
            ),
            param_grid={
                "model__C": C_values,
                "model__l1_ratio": l1_ratio_values,
                "model__loss": [{"name": "huber", "tau": 1.0}],
            },
            scoring="neg_mean_squared_error",
        ),
        "elasticnet_svr": BenchmarkTask(
            name="elasticnet_svr",
            problem_type="regression",
            estimator=plq_ElasticNet_Regressor(
                loss={"name": "svr", "epsilon": 0.1},
                max_iter=max_iter,
                tol=tol,
            ),
            param_grid={
                "model__C": C_values,
                "model__l1_ratio": l1_ratio_values,
                "model__loss": [{"name": "svr", "epsilon": 0.1}],
            },
            scoring="neg_mean_squared_error",
        ),
        "elasticnet_quantile_eps": BenchmarkTask(
            name="elasticnet_quantile_eps",
            problem_type="regression",
            estimator=plq_ElasticNet_Regressor(
                loss={"name": "check_eps", "qt": 0.25, "epsilon": 0.1},
                max_iter=max_iter,
                tol=tol,
            ),
            param_grid={
                "model__C": C_values,
                "model__l1_ratio": l1_ratio_values,
                "model__loss": [{"name": "check_eps", "qt": qt, "epsilon": 0.1} for qt in quantile_values],
            },
            scoring="neg_mean_squared_error",
        ),
        "ridge_svm": BenchmarkTask(
            name="ridge_svm",
            problem_type="classification",
            estimator=plq_Ridge_Classifier(loss={"name": "svm"}, max_iter=max_iter, tol=tol),
            param_grid={
                "model__C": C_values,
                "model__loss": [{"name": "svm"}],
            },
            scoring="accuracy",
        ),
        "ridge_smooth_svm": BenchmarkTask(
            name="ridge_smooth_svm",
            problem_type="classification",
            estimator=plq_Ridge_Classifier(loss={"name": "sSVM"}, max_iter=max_iter, tol=tol),
            param_grid={
                "model__C": C_values,
                "model__loss": [{"name": "sSVM"}],
            },
            scoring="accuracy",
        ),
        "ridge_squared_svm": BenchmarkTask(
            name="ridge_squared_svm",
            problem_type="classification",
            estimator=plq_Ridge_Classifier(loss={"name": "squared SVM"}, max_iter=max_iter, tol=tol),
            param_grid={
                "model__C": C_values,
                "model__loss": [{"name": "squared SVM"}],
            },
            scoring="accuracy",
        ),
        "elasticnet_svm": BenchmarkTask(
            name="elasticnet_svm",
            problem_type="classification",
            estimator=plq_ElasticNet_Classifier(loss={"name": "svm"}, max_iter=max_iter, tol=tol),
            param_grid={
                "model__C": C_values,
                "model__l1_ratio": l1_ratio_values,
                "model__loss": [{"name": "svm"}],
            },
            scoring="accuracy",
        ),
        "elasticnet_smooth_svm": BenchmarkTask(
            name="elasticnet_smooth_svm",
            problem_type="classification",
            estimator=plq_ElasticNet_Classifier(loss={"name": "sSVM"}, max_iter=max_iter, tol=tol),
            param_grid={
                "model__C": C_values,
                "model__l1_ratio": l1_ratio_values,
                "model__loss": [{"name": "sSVM"}],
            },
            scoring="accuracy",
        ),
        "elasticnet_squared_svm": BenchmarkTask(
            name="elasticnet_squared_svm",
            problem_type="classification",
            estimator=plq_ElasticNet_Classifier(
                loss={"name": "squared SVM"},
                max_iter=max_iter,
                tol=tol,
            ),
            param_grid={
                "model__C": C_values,
                "model__l1_ratio": l1_ratio_values,
                "model__loss": [{"name": "squared SVM"}],
            },
            scoring="accuracy",
        ),
    }


def run_gridsearch_benchmark(
    tasks: Iterable[BenchmarkTask] | dict[str, BenchmarkTask] | None = None,
    datasets: Iterable[DatasetSpec] | dict[str, DatasetSpec] | None = None,
    *,
    cv: int = 3,
    repeats: int = 1,
    n_jobs: int | None = None,
    preprocess_X: str | None = "standard",
    return_dataframe: bool = True,
    as_markdown: bool = False,
    markdown_columns: Iterable[str] | None = None,
) -> Any:
    """Run GridSearchCV wall-time benchmarks.

    Parameters
    ----------
    tasks
        Benchmark tasks to run. If omitted, all tasks from ``available_tasks()``
        are used. A dict value is accepted so callers can pass
        ``available_tasks()`` directly.
    datasets
        Dataset specs to run. If omitted, all specs from ``available_datasets()``
        are used. Tasks are automatically matched to datasets with the same
        ``problem_type``.
    cv
        Number of cross-validation folds for each GridSearchCV run.
    repeats
        Number of independent timing repeats per task and dataset.
    n_jobs
        Forwarded to GridSearchCV. The default keeps timing single-process.
    preprocess_X
        Feature preprocessing applied inside the cross-validation pipeline.
        Supported values are ``"standard"``, ``"minmax"``, and ``"none"``.
    return_dataframe
        If True and pandas is installed, return a pandas DataFrame. Otherwise a
        list of dictionaries is returned.
    as_markdown
        If True, return a Markdown table string. In this mode
        ``return_dataframe`` is ignored.
    markdown_columns
        Optional column order for Markdown output. The default focuses on
        timing columns.

    Returns
    -------
    pandas.DataFrame, list[dict], or str
        One row per task/dataset pair, including elapsed seconds and best score.
        If ``as_markdown=True``, returns a Markdown table string.
    """

    task_list = _normalize_tasks(tasks)
    dataset_list = _normalize_datasets(datasets)
    rows = []

    for dataset in dataset_list:
        X, y = dataset.factory()
        for task in task_list:
            if task.problem_type != dataset.problem_type:
                continue

            elapsed_values = []
            best_score = None
            best_params = None
            for _ in range(repeats):
                estimator = _make_pipeline(task.estimator, preprocess_X)
                grid = GridSearchCV(
                    estimator,
                    task.param_grid,
                    cv=cv,
                    scoring=task.scoring,
                    n_jobs=n_jobs,
                )

                start = time.perf_counter()
                grid.fit(X, y)
                elapsed_values.append(time.perf_counter() - start)
                best_score = grid.best_score_
                best_params = _json_safe(grid.best_params_)

            rows.append(
                {
                    "task": task.name,
                    "dataset": dataset.name,
                    "problem_type": task.problem_type,
                    "n_samples": int(X.shape[0]),
                    "n_features": int(X.shape[1]),
                    "cv": int(cv),
                    "n_candidates": _count_candidates(task.param_grid),
                    "repeats": int(repeats),
                    "elapsed_sec_mean": float(statistics.mean(elapsed_values)),
                    "elapsed_sec_std": float(statistics.pstdev(elapsed_values)) if repeats > 1 else 0.0,
                    "best_score": float(best_score) if best_score is not None else None,
                    "best_mse": _best_mse(task, best_score),
                    "best_accuracy": _best_accuracy(task, best_score),
                    "best_C": _best_param(best_params, "model__C"),
                    "best_params": json.dumps(best_params, sort_keys=True),
                    "rehline_version": rehline_version,
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                }
            )

    if as_markdown:
        return benchmark_results_to_markdown(rows, columns=markdown_columns)
    return _maybe_dataframe(rows, return_dataframe)


def run_default_benchmark(
    *,
    cv: int = 3,
    repeats: int = 1,
    n_jobs: int | None = None,
    max_iter: int = 1000,
    tol: float = 1e-4,
    C_grid: Iterable[float] | None = None,
    l1_ratio_grid: Iterable[float] | None = None,
    quantile_grid: Iterable[float] | None = None,
    cqr_quantiles_grid: Iterable[Iterable[float]] | None = None,
    preprocess_X: str | None = "standard",
    return_dataframe: bool = True,
    as_markdown: bool = False,
    markdown_columns: Iterable[str] | None = None,
) -> Any:
    """Run the default mini benchmark suite."""

    return run_gridsearch_benchmark(
        available_tasks(
            max_iter=max_iter,
            tol=tol,
            C_grid=C_grid,
            l1_ratio_grid=l1_ratio_grid,
            quantile_grid=quantile_grid,
            cqr_quantiles_grid=cqr_quantiles_grid,
        ),
        available_datasets(),
        cv=cv,
        repeats=repeats,
        n_jobs=n_jobs,
        preprocess_X=preprocess_X,
        return_dataframe=return_dataframe,
        as_markdown=as_markdown,
        markdown_columns=markdown_columns,
    )


def run_configured_benchmark(
    config_path: str | Path | None = None,
    *,
    output: str | Path | None = None,
) -> Path:
    """Run a benchmark from a JSON config and write a versioned Markdown file."""

    config = load_benchmark_config(config_path)
    rows = _run_config_rows(config)

    output_path = Path(output) if output is not None else _versioned_output_path(config["output_dir"])
    _write_markdown(rows, output_path, config=config)
    return output_path


def _run_config_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = available_tasks(
        max_iter=int(config["max_iter"]),
        tol=float(config["tol"]),
        C_grid=config["C_grid"],
        l1_ratio_grid=config["l1_ratio_grid"],
        quantile_grid=config["quantile_grid"],
        cqr_quantiles_grid=config["cqr_quantiles_grid"],
    )
    datasets = available_datasets()
    task_datasets = _task_dataset_mapping(config)
    rows = []
    for task_name, dataset_names in task_datasets.items():
        selected_tasks = _select_by_name(tasks, [task_name])
        selected_datasets = _select_by_name(datasets, dataset_names)
        rows.extend(
            run_gridsearch_benchmark(
                tasks=selected_tasks,
                datasets=selected_datasets,
                cv=int(config["cv"]),
                repeats=int(config["repeats"]),
                n_jobs=config["n_jobs"],
                preprocess_X=config["preprocess_X"],
                return_dataframe=False,
            )
        )
    return rows


def _task_dataset_mapping(config: dict[str, Any]) -> dict[str, list[str]]:
    if "task_datasets" in config:
        return {str(task): list(datasets) for task, datasets in config["task_datasets"].items()}

    tasks = list(config.get("tasks", []))
    datasets = list(config.get("datasets", []))
    return {task: datasets for task in tasks}


def load_benchmark_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load benchmark settings from JSON, filling missing keys with defaults."""

    path = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
    config = CONFIG_DEFAULTS.copy()
    if path.exists():
        config.update(json.loads(path.read_text(encoding="utf-8")))
    return config


def benchmark_results_to_markdown(
    rows: Any,
    *,
    columns: Iterable[str] | None = None,
    float_digits: int = 4,
) -> str:
    """Format benchmark rows as a GitHub-flavored Markdown table."""

    records = _to_records(rows)
    if not records:
        return ""
    if columns is None:
        return _pivot_records_to_markdown(records, float_digits=float_digits)

    selected_columns = list(columns)
    header = "| " + " | ".join(selected_columns) + " |"
    divider = "| " + " | ".join(["---"] * len(selected_columns)) + " |"
    body = [
        "| "
        + " | ".join(_format_markdown_value(record.get(column, ""), float_digits) for column in selected_columns)
        + " |"
        for record in records
    ]
    return "\n".join([header, divider, *body])


def _pivot_records_to_markdown(records: list[dict[str, Any]], float_digits: int) -> str:
    tables = []
    for task in dict.fromkeys(record["task"] for record in records):
        task_records = [record for record in records if record["task"] == task]
        datasets = [record["dataset"] for record in task_records]
        metric_rows = [
            "n_samples",
            "n_features",
            "n_candidates",
            "elapsed_sec_mean",
            "elapsed_sec_std",
            "best_C",
            _score_metric_name(task_records[0]),
        ]

        header = "| metric | " + " | ".join(datasets) + " |"
        divider = "| " + " | ".join(["---"] * (len(datasets) + 1)) + " |"
        body = [
            "| "
            + metric
            + " | "
            + " | ".join(_format_markdown_value(record.get(metric, ""), float_digits) for record in task_records)
            + " |"
            for metric in metric_rows
        ]
        tables.append("\n".join([f"### {task}", "", header, divider, *body]))
    return "\n\n".join(tables)


def _score_metric_name(record: dict[str, Any]) -> str:
    if record.get("problem_type") == "regression":
        return "best_mse"
    return "best_accuracy"


def _best_param(best_params: dict[str, Any] | None, name: str) -> Any:
    if best_params is None:
        return None
    return best_params.get(name)


def _normalize_tasks(tasks: Iterable[BenchmarkTask] | dict[str, BenchmarkTask] | None) -> list[BenchmarkTask]:
    if tasks is None:
        tasks = available_tasks()
    if isinstance(tasks, dict):
        return list(tasks.values())
    return list(tasks)


def _normalize_datasets(datasets: Iterable[DatasetSpec] | dict[str, DatasetSpec] | None) -> list[DatasetSpec]:
    if datasets is None:
        datasets = available_datasets()
    if isinstance(datasets, dict):
        return list(datasets.values())
    return list(datasets)


def _float_list(values: Iterable[float] | None, default: Iterable[float]) -> list[float]:
    source = default if values is None else values
    return [float(value) for value in source]


def _nested_float_list(
    values: Iterable[Iterable[float]] | None,
    default: Iterable[Iterable[float]],
) -> list[list[float]]:
    source = default if values is None else values
    return [[float(value) for value in group] for group in source]


def _cqr_median_neg_mean_squared_error(estimator: Any, X: np.ndarray, y: np.ndarray) -> float:
    predictions = np.asarray(estimator.predict(X), dtype=np.float64)
    if predictions.ndim == 1:
        median_predictions = predictions
    else:
        model = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
        quantiles = np.asarray(getattr(model, "quantiles_", getattr(model, "quantiles", [])), dtype=np.float64)
        if quantiles.size == predictions.shape[1]:
            index = int(np.argmin(np.abs(quantiles - 0.5)))
        else:
            index = predictions.shape[1] // 2
        median_predictions = predictions[:, index]
    errors = np.asarray(y, dtype=np.float64) - median_predictions
    return -float(np.mean(errors * errors))


def _make_pipeline(estimator: Any, preprocess_X: str | None) -> Pipeline:
    preprocessor = _make_preprocessor(preprocess_X)
    steps = []
    if preprocessor is not None:
        steps.append(("preprocess_X", preprocessor))
    steps.append(("model", clone(estimator)))
    return Pipeline(steps)


def _make_preprocessor(preprocess_X: str | None) -> Any:
    name = "none" if preprocess_X is None else preprocess_X.lower()
    if name in {"none", "identity"}:
        return None
    if name in {"standard", "standardize", "standardscaler"}:
        return StandardScaler()
    if name in {"minmax", "minmaxscaler"}:
        return MinMaxScaler()
    raise ValueError("preprocess_X must be one of 'standard', 'minmax', or 'none'.")


def _count_candidates(param_grid: dict[str, list[Any]]) -> int:
    count = 1
    for values in param_grid.values():
        count *= len(values)
    return count


def _best_mse(task: BenchmarkTask, best_score: float | None) -> float | None:
    if best_score is None or task.problem_type != "regression":
        return None
    if task.scoring == "neg_mean_squared_error" or task.scoring is _cqr_median_neg_mean_squared_error:
        return float(-best_score)
    return float(best_score)


def _best_accuracy(task: BenchmarkTask, best_score: float | None) -> float | None:
    if best_score is None or task.problem_type != "classification":
        return None
    return float(best_score)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _maybe_dataframe(rows: list[dict[str, Any]], return_dataframe: bool) -> Any:
    if not return_dataframe:
        return rows
    try:
        import pandas as pd
    except ImportError:
        return rows
    return pd.DataFrame(rows)


def _to_records(rows: Any) -> list[dict[str, Any]]:
    if hasattr(rows, "to_dict"):
        return rows.to_dict(orient="records")
    return list(rows)


def _format_markdown_value(value: Any, float_digits: int) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        value = f"{value:.{float_digits}g}"
    text = str(value)
    return text.replace("\n", "<br>").replace("|", "\\|")


def _select_by_name(items: dict[str, Any], names: list[str] | None) -> dict[str, Any]:
    if names is None:
        return items
    missing = sorted(set(names) - set(items))
    if missing:
        available = ", ".join(sorted(items))
        raise ValueError(f"Unknown names: {', '.join(missing)}. Available names: {available}.")
    return {name: items[name] for name in names}


def _write_csv(rows: Any, path: Path) -> None:
    records = _to_records(rows)
    if not records:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _write_markdown(rows: Any, path: Path, config: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown = benchmark_results_to_markdown(rows)
    if config is not None:
        markdown += "\n\n## Config\n\n```json\n"
        markdown += json.dumps(config, indent=2, sort_keys=True)
        markdown += "\n```\n"
    else:
        markdown += "\n"
    path.write_text(markdown, encoding="utf-8")


def _versioned_output_path(output_dir: str | Path) -> Path:
    version = _safe_filename(rehline_version)
    return Path(output_dir) / f"rehline-{version}.md"


def _safe_filename(value: str) -> str:
    safe_chars = []
    for char in value:
        safe_chars.append(char if char.isalnum() or char in "._-" else "-")
    return "".join(safe_chars)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mini GridSearchCV benchmarks for ReHLine.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"JSON config path. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    parser.add_argument("--task", action="append", choices=sorted(available_tasks()), help="Task to run.")
    parser.add_argument("--dataset", action="append", choices=sorted(available_datasets()), help="Dataset to run.")
    parser.add_argument("--cv", type=int, help="Number of CV folds.")
    parser.add_argument("--repeats", type=int, help="Number of timing repeats.")
    parser.add_argument("--n-jobs", type=int, default=None, help="GridSearchCV n_jobs.")
    parser.add_argument("--max-iter", type=int, help="ReHLine max_iter.")
    parser.add_argument("--tol", type=float, help="ReHLine convergence tolerance.")
    parser.add_argument("--preprocess-X", choices=["standard", "minmax", "none"], help="Feature preprocessing.")
    parser.add_argument(
        "--C",
        dest="C_grid",
        action="append",
        type=float,
        help=f"C values. Repeat the flag to override the default grid {list(DEFAULT_C_GRID)}.",
    )
    parser.add_argument(
        "--l1-ratio",
        dest="l1_ratio_grid",
        action="append",
        type=float,
        help=f"ElasticNet l1_ratio values. Repeat the flag to override {list(DEFAULT_L1_RATIO_GRID)}.",
    )
    parser.add_argument(
        "--quantile",
        dest="quantile_grid",
        action="append",
        type=float,
        help=f"Quantile levels. Repeat the flag to override {list(DEFAULT_QUANTILE_GRID)}.",
    )
    parser.add_argument(
        "--cqr-quantiles",
        dest="cqr_quantiles_grid",
        action="append",
        help=(
            "Composite quantile set, comma-separated. Repeat the flag to search multiple sets. "
            f"Default is {CONFIG_DEFAULTS['cqr_quantiles_grid']}."
        ),
    )
    parser.add_argument("--output", type=Path, help="Optional output path. Defaults to rehline-<version>.md.")
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Write Markdown when --output is set. Without --output, Markdown is printed by default.",
    )
    return parser.parse_args(argv)


def _parse_cqr_quantiles_grid(values: Iterable[str]) -> list[list[float]]:
    quantiles_grid = []
    for value in values:
        quantiles = [float(item.strip()) for item in value.split(",") if item.strip()]
        if not quantiles:
            raise ValueError("--cqr-quantiles must contain at least one quantile.")
        quantiles_grid.append(quantiles)
    return quantiles_grid


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    config = load_benchmark_config(args.config)

    C_grid = args.C_grid if args.C_grid is not None else config["C_grid"]
    l1_ratio_grid = args.l1_ratio_grid if args.l1_ratio_grid is not None else config["l1_ratio_grid"]
    quantile_grid = args.quantile_grid if args.quantile_grid is not None else config["quantile_grid"]
    cqr_quantiles_grid = (
        _parse_cqr_quantiles_grid(args.cqr_quantiles_grid)
        if args.cqr_quantiles_grid is not None
        else config["cqr_quantiles_grid"]
    )
    cv = args.cv if args.cv is not None else int(config["cv"])
    repeats = args.repeats if args.repeats is not None else int(config["repeats"])
    n_jobs = args.n_jobs if args.n_jobs is not None else config["n_jobs"]
    max_iter = args.max_iter if args.max_iter is not None else int(config["max_iter"])
    tol = args.tol if args.tol is not None else float(config["tol"])
    preprocess_X = args.preprocess_X if args.preprocess_X is not None else config["preprocess_X"]
    task_datasets = _cli_task_dataset_mapping(config, args.task, args.dataset)
    rows_config = {
        **config,
        "task_datasets": task_datasets,
        "cv": cv,
        "repeats": repeats,
        "n_jobs": n_jobs,
        "max_iter": max_iter,
        "tol": tol,
        "C_grid": _float_list(C_grid, DEFAULT_C_GRID),
        "l1_ratio_grid": _float_list(l1_ratio_grid, DEFAULT_L1_RATIO_GRID),
        "quantile_grid": _float_list(quantile_grid, DEFAULT_QUANTILE_GRID),
        "cqr_quantiles_grid": _nested_float_list(cqr_quantiles_grid, DEFAULT_CQR_QUANTILES_GRID),
        "preprocess_X": preprocess_X,
    }
    rows = _run_config_rows(rows_config)

    output = args.output if args.output is not None else _versioned_output_path(config["output_dir"])
    if output.suffix.lower() in {".csv"} and not args.markdown:
        _write_csv(rows, output)
    else:
        active_config = {
            **rows_config,
            "cv": cv,
            "repeats": repeats,
            "n_jobs": n_jobs,
            "max_iter": max_iter,
            "tol": tol,
            "C_grid": _float_list(C_grid, DEFAULT_C_GRID),
            "l1_ratio_grid": _float_list(l1_ratio_grid, DEFAULT_L1_RATIO_GRID),
            "quantile_grid": _float_list(quantile_grid, DEFAULT_QUANTILE_GRID),
            "cqr_quantiles_grid": _nested_float_list(cqr_quantiles_grid, DEFAULT_CQR_QUANTILES_GRID),
            "preprocess_X": preprocess_X,
            "output_dir": str(Path(output).parent),
        }
        _write_markdown(rows, output, config=active_config)
    print(f"Wrote {output}")

    return 0


def _cli_task_dataset_mapping(
    config: dict[str, Any],
    task_names: list[str] | None,
    dataset_names: list[str] | None,
) -> dict[str, list[str]]:
    if task_names is None and dataset_names is None:
        return _task_dataset_mapping(config)

    base_mapping = _task_dataset_mapping(config)
    selected_tasks = task_names if task_names is not None else list(base_mapping)
    if dataset_names is None:
        return {task: list(base_mapping.get(task, [])) for task in selected_tasks}
    return {task: list(dataset_names) for task in selected_tasks}


if __name__ == "__main__":
    raise SystemExit(main())
