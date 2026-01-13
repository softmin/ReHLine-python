.. dnn-inference documentation master file, created by
   sphinx-quickstart on Sun Aug  8 20:28:09 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

🦌 ReHLine
==========

.. image:: figs/logo.png
   :width: 18%
   :align: right

.. -*- mode: rst -*-

|PyPi|_ |MIT|_ |Python3|_ |downloads|_

.. |PyPi| image:: https://badge.fury.io/py/rehline.svg
.. _PyPi: https://pypi.org/project/rehline/

.. |MIT| image:: https://img.shields.io/pypi/l/dnn-inference.svg
.. _MIT: https://opensource.org/licenses/MIT

.. |Python3| image:: https://img.shields.io/badge/python-3-blue.svg
.. _Python3: www.python.org

.. |downloads| image:: https://pepy.tech/badge/rehline
.. _downloads: https://pepy.tech/project/rehline


**ReHLine** is designed to be a computationally efficient and practically useful software package for large-scale ERMs.

- Homepage: `https://rehline.github.io/ <https://rehline.github.io/>`_
- GitHub repo: `https://github.com/softmin/ReHLine-python <https://github.com/softmin/ReHLine-python>`_
- Documentation: `https://rehline-python.readthedocs.io <https://rehline-python.readthedocs.io/en/latest/>`_
- PyPi: `https://pypi.org/project/rehline <https://pypi.org/project/rehline>`_
- Paper: `NeurIPS | 2023 <https://openreview.net/pdf?id=3pEBW2UPAD>`_
.. - Open Source: `MIT license <https://opensource.org/licenses/MIT>`_

The proposed **ReHLine** solver has appealing exhibits appealing properties:

.. list-table::
    :widths: 20 80

    * - **Flexible losses**
      - It applies to ANY convex piecewise linear-quadratic loss function, including the hinge loss, the squared-hinge the check loss, the Huber loss, etc.
    * - **Flexible constraints**
      - It supports linear equality and inequality constraints on the parameter vector.
    * - **Super-Efficient**
      - The optimization algorithm has a provable **LINEAR** convergence rate, and the per-iteration computational complexity is **LINEAR** in the sample size.

📰 News
-------

- **[2026-01] Monotonic Constraints**: We added support for monotonic constraints (both increasing and decreasing) in our solvers. See :doc:`tutorials/constraint` for details.
- **[2025-11] Scikit-Learn Compatibility**: We introduced full scikit-learn compatibility! `ReHLine` now provides `plq_Ridge_Classifier` and `plq_Ridge_Regressor` estimators that integrate seamlessly with the entire scikit-learn ecosystem. This means you can drop `ReHLine` estimators directly into your existing scikit-learn `Pipeline`, perform robust hyperparameter tuning using `GridSearchCV`, and use standard evaluation metrics. See :doc:`tutorials/ReHLine_sklearn` for details.


🔨 Installation
---------------

Install ``rehline`` using ``pip``

.. code:: bash

	pip install rehline

Reference
---------
If you use this code please star 🌟 the repository and cite the following paper:

.. code:: bib

   @article{daiqiu2023rehline,
   title={ReHLine: Regularized Composite ReLU-ReHU Loss Minimization with Linear Computation and Linear Convergence},
   author={Dai, Ben and Yixuan Qiu},
   journal={Advances in Neural Information Processing Systems},
   year={2023},
   }


.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   tutorials
   example
   benchmark
