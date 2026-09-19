"""Independently evaluate PLQ objectives and feasible primal/dual bounds.

Supports unconstrained models and the monotonic constraints in the release config.
"""


def main():
    import argparse
    import hashlib
    import json
    import sys
    import time
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Audit training objectives and primal/dual bounds against an installed ReHLine version."
    )
    parser.add_argument("--package", type=Path, help="Optional directory containing the rehline package to measure")
    parser.add_argument("--config", type=Path, default=root / "benchmarks/release_objective_config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if args.package is not None:
        sys.path.insert(0, str(args.package.resolve()))
    import numpy as np

    import rehline
    from rehline import _internal

    sys.path.insert(0, str(root))
    from benchmarks import available_datasets, available_tasks, run_gridsearch_benchmark

    config = json.loads(args.config.read_text())
    config.update(repeats=1, n_jobs=1)
    tasks = available_tasks(
        max_iter=config["max_iter"],
        tol=config["tol"],
        C_grid=config["C_grid"],
        l1_ratio_grid=config["l1_ratio_grid"],
        quantile_grid=config["quantile_grid"],
    )
    datasets = available_datasets()

    from benchmarks.objectives import record_solver_fits

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "label": args.label,
        "package": rehline.__file__,
        "native_sha256": hashlib.sha256(Path(_internal.__file__).read_bytes()).hexdigest(),
        "config": config,
        "rows": [],
    }
    for name, names in config["task_datasets"].items():
        for dataset in names:
            start = time.perf_counter()
            with record_solver_fits(audit=True) as fits:
                row = run_gridsearch_benchmark(
                    tasks=[tasks[name]],
                    datasets=[datasets[dataset]],
                    cv=config["cv"],
                    repeats=1,
                    n_jobs=1,
                    preprocess_X=config["preprocess_X"],
                    return_dataframe=False,
                )[0]
            row["fits"] = list(fits)
            report["rows"].append(row)
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(
                args.label,
                name,
                dataset,
                "max relative optimality gap",
                max(f["certified_gap"] / max(1, abs(f["feasible_upper_bound"])) for f in fits),
                "seconds",
                round(time.perf_counter() - start, 2),
                flush=True,
            )


if __name__ == "__main__":
    main()
