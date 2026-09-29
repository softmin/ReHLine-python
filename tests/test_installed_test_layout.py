"""Installed-wheel checks carry local references without importing solver source."""

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
    (project / "tests/__init__.py").write_text("")
    helpers = project / "tests/_helpers"
    helpers.mkdir()
    (helpers / "__init__.py").write_text("")
    (helpers / "core.py").write_text("# independent reference\n")
    (helpers / "__pycache__").mkdir()
    (helpers / "__pycache__/stale.pyc").write_bytes(b"do not copy")
    (project / "rehline").mkdir()
    (project / "rehline/__init__.py").write_text("raise RuntimeError('source must not be copied')\n")
    monkeypatch.setattr(module, "__file__", str(project / "tools/test_installed.py"))
    return module, project


@pytest.mark.parametrize("cases", [0, 2])
def test_wheel_tests_keep_solver_source_outside_isolation(installed_runner, tmp_path, monkeypatch, cases):
    runner, project = installed_runner
    argv = ["test_installed", "--correctness-cases", str(cases), "--report-dir", str(tmp_path / "report")]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("PYTHONPATH", str(project))
    commands = []

    def run(command, *, cwd, env, check):
        assert check
        assert cwd != project
        assert "PYTHONPATH" not in env
        assert (cwd / "tests/test_solver.py").is_file()
        assert not (cwd / "rehline").exists()
        assert not (cwd / "benchmarks").exists()
        assert (cwd / "tests/__init__.py").is_file()
        assert (cwd / "tests/_helpers/core.py").is_file()
        assert not (cwd / "tests/_helpers/__pycache__").exists()
        commands.append(command)

    monkeypatch.setattr(runner.subprocess, "run", run)
    runner.main()
    assert len(commands) == (3 if cases else 2)
    if cases:
        assert "tests._helpers.core" in commands[1]
    assert commands[-1][1:4] == ["-m", "pytest", "tests"]
