"""Source-only packaging checks; wheel integration is exercised separately."""

import ast
import hashlib
import importlib.util
import io
import json
import os
import runpy
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (PROJECT / "setup.py").is_file(), reason="Build inputs are only present in the source tree"
)


def include_resolver(root=PROJECT):
    source = ast.parse((PROJECT / "setup.py").read_text())
    definition = next(
        node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "get_eigen_include"
    )
    namespace = dict(os=os, Path=Path, SETUP_DIRECTORY=root)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "setup.py", "exec"), namespace)
    return namespace["get_eigen_include"]()


def test_bundled_eigen_uses_absolute_local_path(monkeypatch, tmp_path):
    monkeypatch.delenv("EIGEN3_INCLUDE_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    resolved = Path(str(include_resolver()))
    assert resolved.is_absolute()
    assert (resolved / "Eigen" / "Core").is_file()


def test_eigen_override_is_checked(monkeypatch, tmp_path):
    monkeypatch.setenv("EIGEN3_INCLUDE_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="Eigen/Core"):
        str(include_resolver())
    (tmp_path / "Eigen").mkdir()
    (tmp_path / "Eigen" / "Core").write_text("// test fixture")
    assert Path(str(include_resolver())) == tmp_path


def test_vendor_files_match_recorded_checksums():
    spec = importlib.util.spec_from_file_location("check_sdist", PROJECT / "tools" / "check_sdist.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.verify_vendor(PROJECT) == 414


@pytest.fixture
def preparer():
    spec = importlib.util.spec_from_file_location("prepare_eigen", PROJECT / "tools" / "prepare_eigen.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def eigen_project(tmp_path, preparer):
    root = tmp_path / "checkout"
    (root / "tools").mkdir(parents=True)
    archive = tmp_path / "eigen.zip"
    files = {"Eigen/Core": b"test header", "Eigen/src/fixture.h": b"nested header", "LICENSE": b"test license"}
    with zipfile.ZipFile(archive, "w") as output:
        for name, content in files.items():
            output.writestr(f"eigen-5.0.1/{name}", content)
        # Unlisted content, including paths that escape the archive, is ignored.
        output.writestr("eigen-5.0.1/../../escaped", b"do not extract")
    manifest = {
        "version": "5.0.1",
        "source_url": "https://example.invalid/eigen.zip",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
    }
    (root / "tools" / preparer.MANIFEST).write_text(json.dumps(manifest))
    return root, archive


def test_missing_checkout_headers_explain_preparation(monkeypatch, tmp_path):
    monkeypatch.delenv("EIGEN3_INCLUDE_DIR", raising=False)
    with pytest.raises(RuntimeError, match="python tools/prepare_eigen.py"):
        str(include_resolver(tmp_path))


def test_local_archive_preparation_is_verified_and_reusable(preparer, eigen_project, monkeypatch):
    root, archive = eigen_project

    def no_network(*args, **kwargs):
        pytest.fail("Prepared headers and local ZIPs must not require network access")

    monkeypatch.setattr(preparer.urllib.request, "urlopen", no_network)
    target = preparer.prepare_eigen(root, archive)
    assert preparer.verify_eigen(root) == 3
    assert {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()} == {
        "Eigen/Core", "Eigen/src/fixture.h", "LICENSE"
    }
    assert not (root / "escaped").exists()
    assert preparer.prepare_eigen(root) == target


def test_download_preparation_checks_the_pinned_archive(preparer, eigen_project, monkeypatch):
    root, archive = eigen_project
    calls = []

    def download(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(archive.read_bytes())

    monkeypatch.setattr(preparer.urllib.request, "urlopen", download)
    preparer.prepare_eigen(root)
    assert preparer.verify_eigen(root) == 3
    assert calls == [("https://example.invalid/eigen.zip", 60)]


@pytest.mark.parametrize("failure", ["download", "archive_hash", "member_hash", "missing_member"])
def test_failed_preparation_never_leaves_partial_headers(preparer, eigen_project, monkeypatch, failure):
    root, archive = eigen_project
    path = root / "tools" / preparer.MANIFEST
    manifest = json.loads(path.read_text())
    if failure == "download":
        def download(*args, **kwargs):
            raise OSError("interrupted download")

        monkeypatch.setattr(preparer.urllib.request, "urlopen", download)
        archive = None
        error = OSError
    elif failure == "archive_hash":
        archive.write_bytes(b"corrupted download")
        error = ValueError
    elif failure == "member_hash":
        manifest["files_sha256"]["Eigen/src/fixture.h"] = "0" * 64
        error = ValueError
    else:
        manifest["files_sha256"]["Eigen/missing.h"] = "0" * 64
        error = KeyError
    path.write_text(json.dumps(manifest))
    with pytest.raises(error):
        preparer.prepare_eigen(root, archive)
    assert not list((root / "vendor").iterdir())


@pytest.mark.parametrize("failure", ["missing", "modified"])
def test_prepared_headers_are_rechecked(preparer, eigen_project, failure):
    root, archive = eigen_project
    target = preparer.prepare_eigen(root, archive)
    header = target / "Eigen" / "src" / "fixture.h"
    if failure == "missing":
        header.unlink()
    else:
        header.write_text("modified")
    with pytest.raises((ValueError, RuntimeError), match="Eigen.*(missing|mismatch)"):
        preparer.prepare_eigen(root)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "Eigen/../../outside", "Eigen\\outside", "C:/outside"])
def test_manifest_cannot_escape_destination(preparer, eigen_project, name):
    root, archive = eigen_project
    path = root / "tools" / preparer.MANIFEST
    manifest = json.loads(path.read_text())
    manifest["files_sha256"][name] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Invalid Eigen manifest path"):
        preparer.prepare_eigen(root, archive)
    assert not (root / "vendor").exists()


def test_sdist_prepares_pinned_headers_even_with_override(preparer, eigen_project, monkeypatch):
    root, archive = eigen_project
    (root / "tools" / "prepare_eigen.py").write_bytes((PROJECT / "tools" / "prepare_eigen.py").read_bytes())
    # Exercise the actual command hook without invoking setup() or building C++.
    calls = []

    class FakeSdist:
        distribution = SimpleNamespace(metadata=SimpleNamespace(license_files=["LICENSE"]))

        def run(self):
            calls.append("sdist")

    source = ast.parse((PROJECT / "setup.py").read_text())
    definition = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "checked_sdist")
    namespace = dict(sdist=FakeSdist, runpy=runpy, SETUP_DIRECTORY=root)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "setup.py", "exec"), namespace)
    command = namespace["checked_sdist"]()
    monkeypatch.setenv("EIGEN3_INCLUDE_DIR", str(PROJECT / "vendor" / "eigen-5.0.1"))
    monkeypatch.setattr(preparer.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(archive.read_bytes()))
    command.run()
    assert calls == ["sdist"]
    assert preparer.verify_eigen(root) == 3
    assert command.distribution.metadata.license_files == ["LICENSE", "vendor/eigen-5.0.1/LICENSE"]
    target = root / "vendor" / "eigen-5.0.1"
    (target / "LICENSE").write_text("modified license")
    with pytest.raises(ValueError, match="checksum mismatch: LICENSE"):
        command.run()
    assert calls == ["sdist"]


def test_git_build_automatically_prepares_eigen(preparer, eigen_project, monkeypatch):
    root, archive = eigen_project
    monkeypatch.setattr(preparer.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(archive.read_bytes()))
    target = preparer.ensure_build_eigen(root)
    assert (target / "Eigen" / "Core").is_file()
    assert preparer.verify_eigen(root) == 3


@pytest.mark.parametrize("state", ["valid", "missing_tree", "missing_header", "modified_header"])
def test_sdist_build_never_downloads_eigen(preparer, eigen_project, monkeypatch, state):
    root, archive = eigen_project
    (root / "PKG-INFO").write_text("Metadata-Version: 2.4\nName: rehline\nVersion: 1.0\n")
    if state != "missing_tree":
        target = preparer.prepare_eigen(root, archive)
        header = target / "Eigen" / "Core"
        if state == "missing_header":
            header.unlink()
        elif state == "modified_header":
            header.write_text("corrupted")

    def no_network(*args, **kwargs):
        pytest.fail("Source distributions must not download missing or corrupted Eigen files")

    monkeypatch.setattr(preparer.urllib.request, "urlopen", no_network)
    if state == "valid":
        assert preparer.ensure_build_eigen(root) == target
    else:
        with pytest.raises((ValueError, RuntimeError), match="Eigen.*(missing|mismatch)"):
            preparer.ensure_build_eigen(root)
