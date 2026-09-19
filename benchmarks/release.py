"""Run release benchmarks with mandatory independent objective validation.

Invoke with the Python environment containing the version being measured.
The benchmark harness comes from this checkout; rehline comes from that environment.
"""

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "benchmarks/release_objective_config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--package", type=Path, help="Optional directory containing the rehline version to measure")
    parser.add_argument("--baseline", type=Path, help="Prior release benchmark JSON; fail if objectives differ")
    parser.add_argument("--objective-rtol", type=float, default=1e-8)
    parser.add_argument("--objective-atol", type=float, default=1e-10)
    args = parser.parse_args(argv)
    if args.package is not None:
        sys.path.insert(0, str(args.package.resolve()))
    import numpy as np
    import scipy
    import sklearn

    import rehline
    from benchmarks import available_datasets, available_tasks, run_gridsearch_benchmark
    from benchmarks.objectives import compare

    if args.baseline is not None and args.baseline.resolve() == args.output.resolve():
        raise ValueError("Baseline and output paths must differ")
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    config = json.loads(args.config.read_text())
    if config["n_jobs"] != 1:
        raise ValueError("Use n_jobs=1 so all fitted models can be audited in this process")
    tasks = available_tasks(
        max_iter=config["max_iter"],
        tol=config["tol"],
        C_grid=config["C_grid"],
        l1_ratio_grid=config["l1_ratio_grid"],
        quantile_grid=config["quantile_grid"],
        cqr_quantiles_grid=config.get("cqr_quantiles_grid"),
    )
    datasets = available_datasets()
    # Import the native module directly to record the binary being measured.
    from rehline import _internal

    native = Path(_internal.__file__)
    report = {
        "label": args.label,
        "package": rehline.__file__,
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "dependencies": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "config": config,
        "objective_validation": "incomplete",
        "baseline_comparison": "pending" if baseline else "not_requested",
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write_report():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    write_report()
    try:
        for task, names in config["task_datasets"].items():
            for name in names:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    row = run_gridsearch_benchmark(
                        tasks=[tasks[task]],
                        datasets=[datasets[name]],
                        cv=config["cv"],
                        repeats=config["repeats"],
                        n_jobs=1,
                        preprocess_X=config["preprocess_X"],
                        verify_objective=True,
                        objective_rtol=args.objective_rtol,
                        objective_atol=args.objective_atol,
                        return_dataframe=False,
                    )[0]
                fits = row["fits"]
                row.update(
                    n_fits=len(fits),
                    n_nonconverged=sum(not fit["converged"] for fit in fits),
                    max_kkt_residual=max(fit["kkt_residual"] for fit in fits),
                    max_constraint_violation=max(fit["constraint_violation"] for fit in fits),
                    mean_iterations=float(np.mean([fit["n_iter"] for fit in fits])),
                    warnings=[str(w.message) for w in captured],
                )
                report["rows"].append(row)
                write_report()
                print(
                    f"{args.label}: {task}/{name}: {row['elapsed_sec_mean']:.3f}s, "
                    f"objective=passed ({row['n_objective_fits']} fits), "
                    f"max relative gap={row['max_relative_certified_gap']:.3g}",
                    flush=True,
                )
        # Self-comparison also rejects an empty or incomplete benchmark config.
        report["objective_summary"] = compare(report, report, rtol=args.objective_rtol, atol=args.objective_atol)
        report["objective_validation"] = "passed"
        if baseline is not None:
            report["baseline_comparison"] = compare(
                baseline, report, rtol=args.objective_rtol, atol=args.objective_atol
            )
            print("Baseline objectives: passed", flush=True)
    except Exception as error:
        report["error"] = str(error)
        if report["objective_validation"] != "passed":
            report["objective_validation"] = "failed"
        elif baseline is not None:
            report["baseline_comparison"] = "failed"
        write_report()
        raise
    write_report()
    return report


if __name__ == "__main__":
    main()
