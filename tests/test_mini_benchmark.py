import json

import numpy as np

from benchmarks import (
    BenchmarkTask,
    DatasetSpec,
    available_datasets,
    available_tasks,
    benchmark_results_to_markdown,
    load_benchmark_config,
    run_configured_benchmark,
    run_gridsearch_benchmark,
)
from rehline import plq_Ridge_Regressor


def test_run_gridsearch_benchmark_returns_timing_rows():
    def tiny_regression():
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 5))
        beta = rng.normal(size=5)
        y = X @ beta + 0.1 * rng.normal(size=40)
        return X, y

    task = BenchmarkTask(
        name="tiny_ridge_quantile",
        problem_type="regression",
        estimator=plq_Ridge_Regressor(loss={"name": "QR", "qt": 0.5}, max_iter=100),
        param_grid={"model__C": [0.1], "model__loss": [{"name": "QR", "qt": 0.5}]},
        scoring="neg_mean_squared_error",
    )
    dataset = DatasetSpec("tiny_regression", "regression", tiny_regression)

    rows = run_gridsearch_benchmark(
        tasks=[task],
        datasets=[dataset],
        cv=2,
        repeats=1,
        preprocess_X="none",
        return_dataframe=False,
    )

    assert len(rows) == 1
    assert rows[0]["task"] == "tiny_ridge_quantile"
    assert rows[0]["dataset"] == "tiny_regression"
    assert rows[0]["n_candidates"] == 1
    assert rows[0]["elapsed_sec_mean"] >= 0.0
    assert rows[0]["best_mse"] >= 0.0
    assert rows[0]["best_accuracy"] is None
    assert rows[0]["best_C"] == 0.1

    markdown = benchmark_results_to_markdown(rows)
    assert markdown.splitlines()[0] == "### tiny_ridge_quantile"
    assert "| metric | tiny_regression |" in markdown
    assert "| n_samples | 40 |" in markdown
    assert "| n_features | 5 |" in markdown
    assert "| n_candidates | 1 |" in markdown
    assert "| best_C | 0.1 |" in markdown
    assert "| best_mse |" in markdown


def test_run_gridsearch_benchmark_can_return_markdown():
    task = BenchmarkTask(
        name="format_only",
        problem_type="regression",
        estimator=plq_Ridge_Regressor(loss={"name": "QR", "qt": 0.5}, max_iter=50),
        param_grid={"model__C": [0.1], "model__loss": [{"name": "QR", "qt": 0.5}]},
        scoring="neg_mean_squared_error",
    )
    dataset = DatasetSpec(
        "format_dataset",
        "regression",
        lambda: (np.arange(20, dtype=float).reshape(10, 2), np.arange(10, dtype=float)),
    )

    markdown = run_gridsearch_benchmark(
        tasks=[task],
        datasets=[dataset],
        cv=2,
        preprocess_X="minmax",
        return_dataframe=False,
        as_markdown=True,
    )

    assert isinstance(markdown, str)
    assert "### format_only" in markdown
    assert "| metric | format_dataset |" in markdown


def test_available_tasks_accepts_custom_grids():
    tasks = available_tasks(
        C_grid=[0.1, 1.0],
        l1_ratio_grid=[0.2, 0.8],
        quantile_grid=[0.25, 0.5],
        cqr_quantiles_grid=[[0.1, 0.5, 0.9], [0.25, 0.5, 0.75]],
    )

    assert tasks["ridge_quantile"].param_grid["model__C"] == [0.1, 1.0]
    assert tasks["elasticnet_svm"].param_grid["model__l1_ratio"] == [0.2, 0.8]
    assert tasks["ridge_quantile"].param_grid["model__loss"] == [
        {"name": "QR", "qt": 0.25},
        {"name": "QR", "qt": 0.5},
    ]
    assert tasks["ridge_composite_quantile"].param_grid["model__quantiles"] == [
        [0.1, 0.5, 0.9],
        [0.25, 0.5, 0.75],
    ]


def test_available_tasks_include_example_losses():
    tasks = available_tasks(C_grid=[0.1], l1_ratio_grid=[0.2], quantile_grid=[0.25])

    expected_tasks = {
        "ridge_quantile",
        "ridge_quantile_monotonic",
        "ridge_composite_quantile",
        "ridge_quantile_eps",
        "ridge_mae",
        "ridge_huber",
        "ridge_svr",
        "elasticnet_quantile",
        "elasticnet_quantile_monotonic",
        "elasticnet_quantile_eps",
        "elasticnet_mae",
        "elasticnet_huber",
        "elasticnet_svr",
        "ridge_svm",
        "ridge_smooth_svm",
        "ridge_squared_svm",
        "elasticnet_svm",
        "elasticnet_smooth_svm",
        "elasticnet_squared_svm",
    }

    assert expected_tasks <= set(tasks)
    assert "ridge_mse" not in tasks
    assert "elasticnet_mse" not in tasks
    assert tasks["ridge_huber"].param_grid["model__loss"] == [{"name": "huber", "tau": 1.0}]
    assert tasks["ridge_svr"].param_grid["model__loss"] == [{"name": "svr", "epsilon": 0.1}]
    assert tasks["ridge_quantile_eps"].param_grid["model__loss"] == [{"name": "check_eps", "qt": 0.25, "epsilon": 0.1}]
    assert tasks["ridge_quantile_monotonic"].param_grid["model__constraint"] == [
        [{"name": "monotonic", "decreasing": False}]
    ]
    assert tasks["ridge_composite_quantile"].param_grid["model__quantiles"] == [[0.1, 0.5, 0.9]]
    assert tasks["elasticnet_quantile_monotonic"].param_grid["model__constraint"] == [
        [{"name": "monotonic", "decreasing": False}]
    ]
    assert tasks["ridge_smooth_svm"].param_grid["model__loss"] == [{"name": "sSVM"}]
    assert tasks["ridge_squared_svm"].param_grid["model__loss"] == [{"name": "squared SVM"}]


def test_builtin_datasets_load():
    datasets = available_datasets()
    assert "california_housing" in datasets
    assert "covtype_binary" in datasets
    assert "covtype_binary_50k" in datasets
    assert "covtype_binary_100k" in datasets
    assert "covtype_binary_full" in datasets
    assert "make_regression_10k" in datasets
    assert "make_regression_100k" in datasets
    assert "make_regression_300k" in datasets
    assert "make_friedman1_5k_100" in datasets
    assert "openml_buzz_twitter" in datasets
    assert "make_classification_100k" in datasets
    assert "make_classification_300k" in datasets
    assert "openml_guillermo" in datasets
    assert "openml_bioresponse" in datasets

    for name in [
        "friedman1",
        "sparse_uncorrelated",
        "linnerud_weight",
        "iris_binary",
        "wine_binary",
        "digits_0_1",
        "digits_low_high",
    ]:
        X, y = datasets[name].factory()
        assert X.shape[0] == y.shape[0]
        assert X.ndim == 2
        assert y.ndim == 1


def test_run_configured_benchmark_writes_markdown(tmp_path):
    config_path = tmp_path / "mini_config.json"
    output_dir = tmp_path / "results"
    config_path.write_text(
        json.dumps(
            {
                "task_datasets": {"ridge_svm": ["toy_classification"]},
                "cv": 2,
                "repeats": 1,
                "max_iter": 50,
                "C_grid": [0.1],
                "l1_ratio_grid": [0.5],
                "quantile_grid": [0.5],
                "preprocess_X": "standard",
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    config = load_benchmark_config(config_path)
    output_path = run_configured_benchmark(config_path)

    assert config["task_datasets"] == {"ridge_svm": ["toy_classification"]}
    assert output_path.parent == output_dir
    assert output_path.name.startswith("rehline-")
    assert output_path.suffix == ".md"
    markdown = output_path.read_text(encoding="utf-8")
    assert "### ridge_svm" in markdown
    assert "| metric | toy_classification |" in markdown
    assert "| n_candidates | 1 |" in markdown
    assert "| best_C |" in markdown
    assert "best_accuracy" in markdown
    assert "## Config" in markdown
    assert '"task_datasets": {' in markdown
    assert '"preprocess_X": "standard"' in markdown


def test_task_dataset_config_does_not_cross_product(tmp_path):
    config_path = tmp_path / "mini_config.json"
    output_dir = tmp_path / "results"
    config_path.write_text(
        json.dumps(
            {
                "task_datasets": {
                    "ridge_quantile": ["toy_regression"],
                    "ridge_svm": ["toy_classification"],
                },
                "cv": 2,
                "repeats": 1,
                "max_iter": 50,
                "C_grid": [0.1],
                "l1_ratio_grid": [0.5],
                "quantile_grid": [0.5],
                "preprocess_X": "none",
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )

    output_path = run_configured_benchmark(config_path)
    markdown = output_path.read_text(encoding="utf-8")

    assert "| metric | toy_regression |" in markdown
    assert "| metric | toy_classification |" in markdown
    assert "| metric | toy_regression | toy_classification |" not in markdown
