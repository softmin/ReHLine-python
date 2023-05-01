Benchmark repository for linear SVM for binary classification
=============================================================

|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
The linear SVM consists in solving the following program:

.. math::

    \min_w C \sum_{i=1}^{n} max(1 - y_i x_i^\top w, 0) + \frac{1}{2} \sum_{j=1}^p w_j^2

where n (or n_samples) stands for the number of samples, p (or n_features) stands for the number of features and

.. math::

 y \in \mathbb{R}^n, X = [x_1^\top, \dots, x_n^\top]^\top \in \mathbb{R}^{n \times p}

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_linear_svm_binary_classif_no_intercept
   $ benchopt run ./benchmark_linear_svm_binary_classif_no_intercept

Apart from the problem, options can be passed to `benchopt run`, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run ./benchmark_linear_svm_binary_classif_no_intercept -s sklearn -d simulated --max-runs 10 --n-repetitions 10


Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/cli.html.

.. |Build Status| image:: https://github.com/benchopt/benchmark_linear_svm_binary_classif_no_intercept/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/benchmark_linear_svm_binary_classif_no_intercept/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
