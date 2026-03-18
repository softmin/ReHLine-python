# Contributing to ReHLine

Thank you for your interest in contributing to ReHLine! This guide will help you get started.

## Getting Started

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/your-username/ReHLine-python.git
   cd ReHLine-python
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or: venv\Scripts\activate  # Windows
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[test]"
   ```

## Development Environment Setup

### Building from Source

ReHLine uses a hybrid Python/C++ architecture. The build process automatically downloads Eigen 3.0.1 during installation.

**Standard build:**
```bash
pip install -e .
```

**Using a custom Eigen installation:**
```bash
export EIGEN3_INCLUDE_DIR=/path/to/eigen
pip install -e . --no-build-isolation
```

**Manual build (for debugging):**
```bash
# On Linux/macOS
c++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) ./src/rehline.cpp -o rehline/_internal$(python3-config --extension-suffix)

# On Windows
cl /LD /EHsc /I%PYTHON_INCLUDE% /I%PYTHON_HOME% /I<path-to-pybind11-include> /I<path-to-eigen> .\src\rehline.cpp /link /OUT:rehline\_internal.pyd
```

### Verifying Installation

Run the test suite to verify the build:
```bash
pytest tests/ -v
```

## Architecture Overview

ReHLine follows a layered architecture:

```
┌─────────────────────────────────────────────────────┐
│  sklearn Mixin Layer                               │  ← rehline/_sklearn_mixin.py
│  (Provides fit/predict compatible with sklearn)    │
├─────────────────────────────────────────────────────┤
│  Python Wrapper Layer                              │  ← rehline/_base.py, _class.py
│  (High-level API, parameter validation)            │
├─────────────────────────────────────────────────────┤
│  pybind11 Binding Layer                            │  ← src/rehline.cpp → _internal*.so
│  (C++ ↔ Python bridge)                             │
├─────────────────────────────────────────────────────┤
│  C++ Core Computation Layer                        │  ← src/rehline.cpp, src/rehline.h
│  (Numerical algorithms, linear algebra)           │
└─────────────────────────────────────────────────────┘
```

### Key Files

| Layer | Files | Description |
|-------|-------|-------------|
| C++ Core | `src/rehline.cpp`, `src/rehline.h` | Core optimization algorithms |
| pybind11 | `src/rehline.cpp` | C++ to Python bindings |
| Python Base | `rehline/_base.py` | Base classes, solver interface |
| Python Classes | `rehline/_class.py` | ReHLine, plqERM_Ridge, etc. |
| sklearn Mixin | `rehline/_sklearn_mixin.py` | sklearn-compatible estimators |

## Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with the following conventions:

- **Line length**: Maximum 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings, single quotes only when containing double quotes
- **Imports**: Use `isort` for sorting imports

### Recommended Tools

Install pre-commit hooks (optional but recommended):
```bash
pip install pre-commit
pre-commit install
```

### Import Order

Organize imports in the following order (use `isort` to enforce):
1. Standard library imports
2. Third-party imports (numpy, scipy, sklearn)
3. Local/rehline imports

Example:
```python
import re
from functools import lru_cache

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, RegressorMixin

from rehline._base import _BaseReHLine
from rehline._internal import rehline_internal
```

### Type Hints

- Use type hints for function signatures and return types
- Prefer `from __future__ import annotations` for forward references

## Testing

All new features should include tests. We use `pytest` as the test framework.

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_rehline.py -v

# Run with coverage
pytest --cov=rehline --cov-report=term-missing tests/
```

### Test File Organization

- Place tests in the `tests/` directory
- Name test files as `test_<module>.py`
- Test classes should be named `Test<Feature>`
- Test functions should be named `test_<description>`

### Test Coverage Requirements

- New code should maintain or improve test coverage
- Aim for >90% coverage on new modules
- Run `pytest --cov=rehline` before submitting PR

## Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and add tests

3. Run the test suite:
   ```bash
   pytest tests/ -v
   ```

4. Commit your changes with a clear commit message:
   ```bash
   git add .
   git commit -m "Add feature: description of changes"
   ```

5. Push to your fork and submit a pull request

## Pull Request Guidelines

- PRs should target the `main` branch
- Include a clear description of changes
- Reference any related issues
- Ensure all tests pass
- Update documentation if needed

## Reporting Issues

### Bug Reports

Include:
- Steps to reproduce
- Expected vs actual behavior
- Minimal code example
- Environment details (OS, Python version, package version)

### Feature Requests

Include:
- Use case description
- Proposed API design
- Any alternative solutions considered

## CI/CD

We use GitHub Actions for continuous integration:

- **CI Tests**: Runs on Ubuntu, macOS, Windows with Python 3.10-3.13
- **Build Wheels**: Builds binary wheels for distribution

All PRs must pass CI checks before merging.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
