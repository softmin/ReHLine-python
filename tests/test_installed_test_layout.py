"""Installed-wheel checks must load external helpers without importing solver source."""

import importlib.util
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools/test_installed.py"
pytestmark = pytest.mark.skipif(not TOOL.is_file(), reason="Source-only test tooling")


@pytest.fixture
def installed_runner(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("installed_runner", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project = tmp_path / "solver"
    (project / "tests").mkdir(parents=True)
    (project / "tests/test_solver.py").write_text("# solver regression\n")
    (project / "rehline").mkdir()
    (project / "rehline/__init__.py").write_text("raise RuntimeError('source must not be copied')\n")
    monkeypatch.setattr(module, "__file__", str(project / "tools/test_installed.py"))
    return module, project


@pytest.mark.parametrize("copy_helpers", [False, True])
def test_wheel_tests_keep_solver_source_outside_isolation(installed_runner, tmp_path, monkeypatch, copy_helpers):
    runner, project = installed_runner
    checkout = tmp_path / "benchmarking"
    package = checkout / "benchmarks"
    (package / "common").mkdir(parents=True)
    (package / "common/objectives.py").write_text("# independent audit\n")
    (package / "quick").mkdir()
    (package / "quick/config.json").write_text("{}\n")
    for name in ("data", "results", "__pycache__"):
        (package / name).mkdir()
        (package / name / "large-file").write_bytes(b"do not copy")
    argv = ["test_installed", "--correctness-cases", "2", "--report-dir", str(tmp_path / "report")]
    if copy_helpers:
        argv += ["--benchmark-source", str(checkout)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("PYTHONPATH", str(project))
    commands = []

    def run(command, *, cwd, env, check):
        assert check
        assert cwd != project
        assert "PYTHONPATH" not in env
        assert (cwd / "tests/test_solver.py").is_file()
        assert not (cwd / "rehline").exists()
        assert (cwd / "benchmarks").exists() is copy_helpers
        if copy_helpers:
            assert (cwd / "benchmarks/quick/config.json").is_file()
            assert not any((cwd / "benchmarks" / name).exists() for name in ("data", "results", "__pycache__"))
        commands.append(command)

    monkeypatch.setattr(runner.subprocess, "run", run)
    runner.main()
    assert "benchmarks.correctness.core" in commands[1]
    assert commands[2][1:4] == ["-m", "pytest", "tests"]


def test_missing_explicit_benchmark_checkout_fails(installed_runner, tmp_path, monkeypatch):
    runner, _ = installed_runner
    monkeypatch.setattr(sys, "argv", ["test_installed", "--benchmark-source", str(tmp_path / "missing")])
    with pytest.raises(ValueError, match="Not a ReHLine-benchmarking checkout"):
        runner.main()
