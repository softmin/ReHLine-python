"""Install and validate the compatible wheel from the actual release artifacts."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename


def select_wheel(directory, supported_tags=None):
    supported = set(sys_tags() if supported_tags is None else supported_tags)
    matches = []
    for wheel in sorted(directory.glob("*.whl")):
        name, _, _, tags = parse_wheel_filename(wheel.name)
        if canonicalize_name(name) == "rehline" and supported.intersection(tags):
            matches.append(wheel)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one compatible ReHLine release wheel, found {matches}")
    return matches[0].resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--correctness-cases", type=int, default=256)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.correctness_cases <= 0:
        parser.error("--correctness-cases must be positive for release validation")
    wheel = select_wheel(args.wheel_dir)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = dict(wheel=wheel.name, sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(), status="pending")
    manifest = args.report_dir / "release-wheel.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)], check=True
        )
        subprocess.run([sys.executable, "-m", "pip", "install", str(wheel) + "[test,benchmark]"], check=True)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("test_installed.py")),
                "--correctness-cases",
                str(args.correctness_cases),
                "--report-dir",
                str(args.report_dir),
            ],
            check=True,
        )
    except Exception:
        report["status"] = "failed"
        raise
    else:
        report["status"] = "passed"
    finally:
        manifest.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
