"""Compatibility entry point for the shared benchmark objective comparison."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.objectives import compare, main

if __name__ == "__main__":
    main()
