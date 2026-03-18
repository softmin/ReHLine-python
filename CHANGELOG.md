# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2025-10-07

### Added

- Support for monotonic constraints (both increasing and decreasing) in solvers
- Monotonic constraint documentation and tutorials

### Fixed

- Various bug fixes and improvements

## [0.1.0] - 2025-06-10

### Added

- Full scikit-learn compatibility with `plq_Ridge_Classifier` and `plq_Ridge_Regressor`
- Integration with scikit-learn `Pipeline`
- Support for `GridSearchCV` and standard evaluation metrics

### Changed

- Improved documentation and tutorials

## [0.0.7] - 2025-06-10

### Added

- Multi-class classification support
- ElasticNet penalty support for `plqERM`
- ElasticNet penalty support for ReHLine solver

### Fixed

- Various bug fixes and improvements

## [0.0.6] - 2025-01-14

### Added

- Matrix factorization support with `plqMF_Ridge` class
- `make_mf_dataset` function for matrix factorization datasets

### Changed

- Updated dependencies

## [0.0.5] - 2024-10-31

### Added

- Additional features and improvements

### Changed

- Documentation improvements and code formatting

## [0.0.4] - 2024-09-03

### Added

- Additional features and improvements

### Changed

- Dependency updates

## [0.0.3] - 2024-04-24

### Added

- Core solver improvements and bug fixes

## [0.0.1] - 2023-10-18

### Added

- Initial release of ReHLine Python package
- ReHLine solver for regularized composite ReLU-ReHU loss minimization
- Support for convex piecewise linear-quadratic loss functions
- Linear equality and inequality constraints support
- Ridge regression and classification estimators

[Unreleased]: https://github.com/softmin/ReHLine-python/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/softmin/ReHLine-python/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/softmin/ReHLine-python/compare/v0.0.7...v0.1.0
[0.0.7]: https://github.com/softmin/ReHLine-python/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/softmin/ReHLine-python/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/softmin/ReHLine-python/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/softmin/ReHLine-python/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/softmin/ReHLine-python/compare/v0.0.1...v0.0.3
[0.0.1]: https://github.com/softmin/ReHLine-python/tree/v0.0.1
