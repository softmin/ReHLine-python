"""Release validation must select the uploaded binary and fail closed."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.tags import Tag

PROJECT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (PROJECT / "tools" / "test_release_wheel.py").is_file(),
    reason="Release tooling is present only in the source checkout",
)


@pytest.fixture
def gate():
    spec = importlib.util.spec_from_file_location("release_gate", PROJECT / "tools" / "test_release_wheel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_compatible_release_binary_is_selected(gate, tmp_path):
    names = (
        "rehline-1.0-cp312-cp312-manylinux_2_17_x86_64.whl",
        "rehline-1.0-cp313-cp313-win_amd64.whl",
        "other-1.0-cp312-cp312-manylinux_2_17_x86_64.whl",
    )
    for name in names:
        (tmp_path / name).touch()
    tags = [Tag("cp312", "cp312", "manylinux_2_17_x86_64")]
    assert gate.select_wheel(tmp_path, tags).name == names[0]
    with pytest.raises(ValueError, match="exactly one"):
        gate.select_wheel(tmp_path, [Tag("cp312", "cp312", "macosx_11_0_arm64")])
    (tmp_path / names[0].replace("1.0", "1.1")).touch()
    with pytest.raises(ValueError, match="exactly one"):
        gate.select_wheel(tmp_path, tags)


@pytest.mark.parametrize("fail", [False, True])
def test_release_manifest_records_exact_binary_and_test_outcome(gate, tmp_path, monkeypatch, fail):
    import hashlib

    wheel = tmp_path / "rehline-1.0-py3-none-any.whl"
    wheel.write_bytes(b"artifact-under-test")
    report = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["gate", "--wheel-dir", str(tmp_path), "--report-dir", str(report)])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["check"]
        if fail and command[1].endswith("test_installed.py"):
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(gate.subprocess, "run", run)
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            gate.main()
    else:
        gate.main()
    manifest = json.loads((report / "release-wheel.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert manifest["status"] == ("failed" if fail else "passed")
    assert calls[0][-1] == str(wheel)
    assert "--force-reinstall" in calls[0] and "--no-deps" in calls[0]
    assert calls[1][-1] == str(wheel) + "[test,benchmark]"
    assert calls[2][2:4] == ["--correctness-cases", "256"]


def test_release_objective_gate_cannot_be_disabled(gate, tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["gate", "--wheel-dir", str(tmp_path), "--report-dir", str(tmp_path), "--correctness-cases", "0"]
    )
    with pytest.raises(SystemExit) as error:
        gate.main()
    assert error.value.code != 0
