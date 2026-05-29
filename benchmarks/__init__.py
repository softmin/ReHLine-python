"""Small GridSearchCV benchmarks for ReHLine."""

from importlib import import_module

__all__ = [
    "BenchmarkTask",
    "DatasetSpec",
    "available_datasets",
    "available_tasks",
    "benchmark_results_to_markdown",
    "load_benchmark_config",
    "make_dataset",
    "run_configured_benchmark",
    "run_default_benchmark",
    "run_gridsearch_benchmark",
]


def __getattr__(name):
    if name in __all__:
        module = import_module("benchmarks.mini_gridsearch")
        return getattr(module, name)
    raise AttributeError(f"module 'benchmarks' has no attribute '{name}'")
