"""Prepare checksum-pinned Eigen headers before building from a Git checkout."""

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = "eigen-5.0.1.json"


def read_manifest(root):
    manifest = json.loads((Path(root) / "tools" / MANIFEST).read_text(encoding="utf-8"))
    for name in manifest["files_sha256"]:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
            raise ValueError(f"Invalid Eigen manifest path: {name}")
    return manifest


def verify_eigen(root):
    """Check every pinned header and license, without network access."""
    manifest = read_manifest(root)
    vendor = Path(root) / "vendor" / f"eigen-{manifest['version']}"
    for name, expected in manifest["files_sha256"].items():
        file = vendor / name
        if not file.resolve().is_relative_to(vendor.resolve()):
            raise ValueError(f"Eigen path escapes the header directory: {name}")
        if not file.is_file():
            raise RuntimeError(
                f"Eigen file missing: {file}. Run 'python tools/prepare_eigen.py' before building from Git."
            )
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Eigen checksum mismatch: {name}; remove {vendor} and run tools/prepare_eigen.py again")
    return len(manifest["files_sha256"])


def prepare_eigen(root=PROJECT, archive=None):
    """Download (or read a local archive), verify, then install the pinned files."""
    root = Path(root).resolve()
    manifest = read_manifest(root)
    target = root / "vendor" / f"eigen-{manifest['version']}"
    if target.exists():
        verify_eigen(root)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    # Stage next to the destination so an incomplete download/extraction cannot
    # leave a header tree that later builds might mistake for a complete copy.
    with tempfile.TemporaryDirectory(prefix=".eigen-", dir=target.parent) as temporary:
        staging = Path(temporary)
        if archive is None:
            archive = staging / "eigen.zip"
            with urllib.request.urlopen(manifest["source_url"], timeout=60) as source, archive.open("wb") as output:
                shutil.copyfileobj(source, output)
        archive = Path(archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["archive_sha256"]:
            raise ValueError("Eigen archive SHA-256 mismatch; no headers were installed")

        unpacked = staging / target.name
        unpacked.mkdir()
        with zipfile.ZipFile(archive) as source:
            # Only extract pinned files by explicit names; never extract the
            # whole upstream archive or trust its paths/permissions.
            for name, expected in manifest["files_sha256"].items():
                content = source.read(f"{target.name}/{name}")
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError(f"Eigen archive member checksum mismatch: {name}")
                destination = unpacked / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
        unpacked.rename(target)
    return target


def ensure_build_eigen(root=PROJECT):
    """Fetch missing checkout dependencies; published sdists must be complete."""
    root = Path(root).resolve()
    if (root / "PKG-INFO").is_file():
        # A source release already contains Eigen. Missing or changed headers
        # indicate a damaged package, not permission to download replacements.
        verify_eigen(root)
        manifest = read_manifest(root)
        return root / "vendor" / f"eigen-{manifest['version']}"
    return prepare_eigen(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a local release ZIP instead of downloading (same SHA-256 required)")
    args = parser.parse_args()
    target = prepare_eigen(archive=args.archive)
    print(f"Eigen ready: {target} ({verify_eigen(PROJECT)} header/license files verified)")


if __name__ == "__main__":
    main()
