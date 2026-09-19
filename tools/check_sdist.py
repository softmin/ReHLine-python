"""Build an exact sdist using local Eigen, with Python network access denied."""

import argparse
import os
import runpy
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

GUARD = """import sys

def deny_network(event, args):
    if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
        raise RuntimeError("Network access denied during the source-build check")

sys.addaudithook(deny_network)
"""


def verify_vendor(root):
    prepare = runpy.run_path(str(Path(__file__).resolve().with_name("prepare_eigen.py")))
    return prepare["verify_eigen"](root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdist", type=Path)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--eigen-include", type=Path)
    args = parser.parse_args()
    output = args.wheel_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rehline-offline-build-") as temporary:
        directory = Path(temporary).resolve()
        # Accept only regular files/directories and reject archive path escapes.
        with tarfile.open(args.sdist) as archive:
            for member in archive.getmembers():
                destination = (directory / member.name).resolve()
                if not destination.is_relative_to(directory):
                    raise ValueError("Archive path escapes the build directory")
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source:
                        destination.write_bytes(source.read())
                else:
                    raise ValueError("Source archive must contain regular files and directories only")
        projects = [p for p in directory.iterdir() if (p / "setup.py").is_file()]
        if len(projects) != 1:
            raise ValueError("Expected exactly one source project in the archive")
        project = projects[0]
        count = verify_vendor(project)
        guard = directory / "network-guard"
        guard.mkdir()
        (guard / "sitecustomize.py").write_text(GUARD)
        env = os.environ.copy()
        env.update(PYTHONPATH=str(guard), PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
        env.pop("EIGEN3_INCLUDE_DIR", None)
        if args.eigen_include is not None:
            env["EIGEN3_INCLUDE_DIR"] = str(args.eigen_include.resolve())
        probe = subprocess.run(
            [sys.executable, "-c", "import socket; socket.getaddrinfo('localhost', 80)"],
            cwd=project,
            env=env,
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0 or "Network access denied" not in probe.stderr:
            raise RuntimeError("The network-denial hook was not active")
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(output)],
            cwd=project,
            env=env,
            check=True,
        )
        print(f"Offline source-build check passed: {count} vendor files verified; wheels in {output}")


if __name__ == "__main__":
    main()
