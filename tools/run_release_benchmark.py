"""Run the release benchmark using the selected environment's ReHLine package."""

import sys
from pathlib import Path

# Preserve the existing installed-package entry point. --package is handled by
# benchmarks.release before any ReHLine import when explicitly supplied.
if "--package" not in sys.argv and not any(arg.startswith("--package=") for arg in sys.argv):
    import rehline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.release import main

if __name__ == "__main__":
    main()
