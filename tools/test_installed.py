"""Run routine tests and optional reference cases against the installed wheel."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correctness-cases", type=int, default=0)
    parser.add_argument("--report-dir", type=Path, default=Path("test-results/wheel-correctness"))
    args = parser.parse_args()
    if args.correctness_cases < 0:
        parser.error("--correctness-cases must be nonnegative")
    report_dir = args.report_dir.resolve()
    project = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="rehline-wheel-test-") as directory:
        target = Path(directory)
        shutil.copytree(project / "tests", target / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["SCIPY_ARRAY_API"] = "1"
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import pathlib, rehline; "
                f"assert not pathlib.Path(rehline.__file__).resolve().is_relative_to({str(project / 'rehline')!r}), "
                "'Tests must use the installed wheel, not the source checkout'; "
                "print('Testing installed package:', rehline.__file__)",
            ],
            cwd=target,
            env=env,
            check=True,
        )
        if args.correctness_cases:
            report_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tests._helpers.core",
                    "--profile",
                    "routine",
                    "--tol",
                    "1e-8",
                    "--max-iter",
                    "1000000",
                    "--max-samples",
                    "20",
                    "--max-dim",
                    "5",
                    "--cases",
                    str(args.correctness_cases),
                    "--output",
                    str(report_dir / "correctness.json"),
                ],
                cwd=target,
                env=env,
                check=True,
            )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "-m",
                "not numerical_stress",
                "-o",
                "markers=numerical_stress: separately reported numerical pressure tests",
                "-q",
                "--tb=short",
                "-W",
                "error::sklearn.exceptions.ConvergenceWarning",
            ],
            cwd=target,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
